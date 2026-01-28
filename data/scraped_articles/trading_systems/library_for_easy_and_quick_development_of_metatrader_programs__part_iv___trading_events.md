---
title: Library for easy and quick development of MetaTrader programs (part IV): Trading events
url: https://www.mql5.com/en/articles/5724
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:42:23.033759
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=gyhodltcduvncekjprfjyophignitjlu&ssn=1769186540785024985&ssn_dr=0&ssn_sr=0&fv_date=1769186540&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5724&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20IV)%3A%20Trading%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918654052895367&fz_uniq=5070497321805879019&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Passing trading events to the program](https://www.mql5.com/en/articles/5724#node01)
- [Testing the handling of trading events](https://www.mql5.com/en/articles/5724#node02)
- [What's next?](https://www.mql5.com/en/articles/5724#node03)


[In the first article](https://www.mql5.com/en/articles/5654), we started creating a large cross-platform library simplifying the development of programs for [MetaTrader 5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4. In subsequent articles, we continued the development of the library and completed the base object of the Engine library and the collection of market orders and positions. In this article, we will continue the development of the base object and teach it to identify trading events on the account.

### Passing trading events to the program

If we get back to the test EA created at the very end of the [third article](https://www.mql5.com/en/articles/5687), we can see that the library is able to define trading events happening on the account. However, we should accurately divide all occurring events by their types to send them to a program using the library for work.

To do this, **we should write the method defining events and the one defining event types.**

Let's think about what trading events we need to identify:

- a pending order can be placed,
- a pending order can be removed,
- a pending order can be activated generating a position,
- a pending order can be partially activated generating a position,
- a position can be opened,
- a position can be closed,
- a position can be partially opened,
- a position can be partially closed,
- a position can be closed by an opposite one,
- a position can be partially closed by an opposite one,
- an account can be replenished,
- funds can be withdrawn from an account,
- balance operations can occur on an account

_events that are not tracked yet:_
- a pending order can be modified (changing an activation price, adding/removing/changing StopLoss and TakeProfit levels)

- a position can be modified (adding/removing/changing StopLoss and TakeProfit levels)

Based on the above, you need to decide how to unambiguously identify an event. It is better to immediately divide the solution according to account types:

**Hedging:**

1. increased number of pending orders means adding a pending order (event in the market environment)

2. decreased number of pending orders:

1. increased number of positions means a pending order activation (event in the market and historical environment)

2. number of positions not increasing means removing a pending order (event in the market environment)


4. number of pending orders not decreasing:
1. increased number of positions means opening a new position (event in the market and historical environment)

2. decreased number of positions means closing a position (event in the market and historical environment)

3. number of positions remaining unchanged but accompanied with a decreasing volume means partial closing of a position (event in the historical environment)

**Netting:**

1. increased number of pending orders means adding a pending order
2. decreased number of pending orders:

1. increased number of positions means a pending order activation
2. number of positions remaining unchanged but accompanied by changed position modification time and unchanged volume means a pending order activation and increasing the position volume
3. decreased number of positions means closing a position

4. number of pending orders not decreasing:
1. increased number of positions means opening a new position
2. decreased number of positions means closing a position

3. number of positions remaining unchanged but accompanied by changed position modification time and increased volume means adding a volume to a position
4. number of positions remaining unchanged but accompanied by changed position modification time and decreased volume means partial position closing

* * *

To identify trading events, we need to know the account the program works on. Add the flag of the hedge account type to the private section of the CEngine class, define the account type in the class constructor and write the result to this flag variable:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the first launch flag
   bool                 IsFirstStart(void);
public:
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true)
  {
   ::EventSetMillisecondTimer(TIMER_FREQUENCY);
   this.m_list_counters.Sort();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
  }
//+------------------------------------------------------------------+
```

During the first launch, while constructing the class object, the account type the program is launched at is defined in its constructor. Accordingly, the methods of defining trading events are immediately attributed either to a hedging or a netting account.

After defining an incoming trading event, we need to store its code. It is to remain constant till the next event. Thus the program is always able to define the last event on an account. An event code will consist of a set of flags. Each flag will describe a specific event. For example, a position closing event can be divided into certain subsets characterizing it more accurately:

1. closed in full
2. closed partially
3. closed by an opposite one
4. closed by a stop loss
5. closed by a take profit
6. etc.

All these attributes are inherent to one "position closing" event, which means an event code should contain all these data. In order to construct an event using the flags, let's create two new enumerations in the Defines.mqh file from the library root folder (trading event flags and possible trading events) on the account we are going to track:

```
//+------------------------------------------------------------------+
//| List of trading event flags on the account                       |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT_FLAGS
  {
   TRADE_EVENT_FLAG_NO_EVENT        =  0,                   // No event
   TRADE_EVENT_FLAG_ORDER_PLASED    =  1,                   // Pending order placed
   TRADE_EVENT_FLAG_ORDER_REMOVED   =  2,                   // Pending order removed
   TRADE_EVENT_FLAG_ORDER_ACTIVATED =  4,                   // Pending order activated by price
   TRADE_EVENT_FLAG_POSITION_OPENED =  8,                   // Position opened
   TRADE_EVENT_FLAG_POSITION_CLOSED =  16,                  // Position closed
   TRADE_EVENT_FLAG_ACCOUNT_BALANCE =  32,                  // Balance operation (clarified by a deal type)
   TRADE_EVENT_FLAG_PARTIAL         =  64,                  // Partial execution
   TRADE_EVENT_FLAG_BY_POS          =  128,                 // Executed by opposite position
   TRADE_EVENT_FLAG_SL              =  256,                 // Executed by StopLoss
   TRADE_EVENT_FLAG_TP              =  512                  // Executed by TakeProfit
  };
//+------------------------------------------------------------------+
//| List of possible trading events on the account                   |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT
  {
   TRADE_EVENT_NO_EVENT,                                    // No trading event
   TRADE_EVENT_PENDING_ORDER_PLASED,                        // Pending order placed
   TRADE_EVENT_PENDING_ORDER_REMOVED,                       // Pending order removed
//--- enumeration members matching the ENUM_DEAL_TYPE enumeration members
   TRADE_EVENT_ACCOUNT_CREDIT,                              // Charging credit
   TRADE_EVENT_ACCOUNT_CHARGE,                              // Additional charges
   TRADE_EVENT_ACCOUNT_CORRECTION,                          // Correcting entry
   TRADE_EVENT_ACCOUNT_BONUS,                               // Charging bonuses
   TRADE_EVENT_ACCOUNT_COMISSION,                           // Additional commissions
   TRADE_EVENT_ACCOUNT_COMISSION_DAILY,                     // Commission charged at the end of a day
   TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY,                   // Commission charged at the end of a month
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY,               // Agent commission charged at the end of a trading day
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY,             // Agent commission charged at the end of a month
   TRADE_EVENT_ACCOUNT_INTEREST,                            // Accrual of interest on free funds
   TRADE_EVENT_BUY_CANCELLED,                               // Canceled buy deal
   TRADE_EVENT_SELL_CANCELLED,                              // Canceled sell deal
   TRADE_EVENT_DIVIDENT,                                    // Accrual of dividends
   TRADE_EVENT_DIVIDENT_FRANKED,                            // Accrual of franked dividend
   TRADE_EVENT_TAX,                                         // Tax accrual
//--- members of enumeration related to the DEAL_TYPE_BALANCE deal type from the ENUM_DEAL_TYPE enumeration
   TRADE_EVENT_ACCOUNT_BALANCE_REFILL,                      // Replenishing account balance
   TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL,                  // Withdrawing funds from an account
//---
   TRADE_EVENT_PENDING_ORDER_ACTIVATED,                     // Pending order activated by price
   TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL,             // Pending order partially activated by price
   TRADE_EVENT_POSITION_OPENED,                             // Position opened
   TRADE_EVENT_POSITION_OPENED_PARTIAL,                     // Position opened partially
   TRADE_EVENT_POSITION_CLOSED,                             // Position closed
   TRADE_EVENT_POSITION_CLOSED_PARTIAL,                     // Position closed partially
   TRADE_EVENT_POSITION_CLOSED_BY_POS,                      // Position closed by an opposite one
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS,              // Position partially closed by an opposite one
   TRADE_EVENT_POSITION_CLOSED_BY_SL,                       // Position closed by StopLoss
   TRADE_EVENT_POSITION_CLOSED_BY_TP,                       // Position closed by TakeProfit
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL,               // Position closed partially by StopLoss
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP,               // Position closed partially by TakeProfit
   TRADE_EVENT_POSITION_REVERSED,                           // Position reversal (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD                          // Added volume to position (netting)
  };
//+------------------------------------------------------------------+
```

Here we should make some clarifications concerning the ENUM\_TRADE\_EVENT enumeration.

Since some trading events require no program or human intervention (charging commissions, fees, bonuses, etc.), we will take that data from the deal type (the [ENUM\_DEAL\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_type) enumeration) in MQL5. To simplify tracking the event later, we need to make it so that our events match the value of the ENUM\_DEAL\_TYPE enumeration.

We will divide the balance operation into two events: account balance replenishment and funds withdrawal. Other events from the deal type enumeration, starting from DEAL\_TYPE\_CREDIT, have the same values as in the ENUM\_DEAL\_TYPE enumeration except for buying and selling (DEAL\_TYPE\_BUY and DEAL\_TYPE\_SELL) that are not related to balance operations.

**Let's improve the class of market orders and positions collection.**

We will add a specimen order to the private section of the CMarketCollection class to perform a search by specified order properties, while the public section of the class will receive methods for obtaining the full list of orders and positions, the list of orders and positions selected by a specified time range and lists returning orders and positions selected by a specified criterion from integer, real and string properties of an order or a position. This will allow us to receive the necessary lists of market orders and positions from the collection (as it was done for the collection of historical orders and deals in [part 2](https://www.mql5.com/en/articles/5669#node02) and [part 3](https://www.mql5.com/en/articles/5687#node01) of the library description).

```
//+------------------------------------------------------------------+
//| Collection of market orders and positions                        |
//+------------------------------------------------------------------+
class CMarketCollection
  {
private:
   struct MqlDataCollection
     {
      long           hash_sum_acc;           // Hash sum of all orders and positions on the account
      int            total_pending;          // Number of pending orders on the account
      int            total_positions;        // Number of positions on the account
      double         total_volumes;          // Total volume of orders and positions on the account
     };
   MqlDataCollection m_struct_curr_market;   // Current data on market orders and positions on the account
   MqlDataCollection m_struct_prev_market;   // Previous data on market orders and positions on the account
   CArrayObj         m_list_all_orders;      // List of pending orders and positions on the account
   COrder            m_order_instance;       // Order object for searching by property
   bool              m_is_trade_event;       // Trading event flag
   bool              m_is_change_volume;     // Total volume change flag
   double            m_change_volume_value;  // Total volume change value
   int               m_new_positions;        // Number of new positions
   int               m_new_pendings;         // Number of new pending orders
   //--- Save the current values of the account data status as previous ones
   void              SavePrevValues(void)                                                                { this.m_struct_prev_market=this.m_struct_curr_market;                  }
public:
   //--- Return the list of all pending orders and open positions
   CArrayObj*        GetList(void)                                                                       { return &m_list_all_orders;                                            }
   //--- Return the list of orders and positions with an open time from begin_time to end_time
   CArrayObj*        GetListByTime(const datetime begin_time=0,const datetime end_time=0);
   //--- Return the list of orders and positions by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj*        GetList(ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   //--- Return the number of (1) new pending orders, (2) new positions, (3) occurred trading event flag, (4) changed volume
   int               NewOrders(void)                                                            const    { return this.m_new_pendings;                                           }
   int               NewPosition(void)                                                          const    { return this.m_new_positions;                                          }
   bool              IsTradeEvent(void)                                                         const    { return this.m_is_trade_event;                                         }
   double            ChangedVolumeValue(void)                                                   const    { return this.m_change_volume_value;                                    }
   //--- Constructor
                     CMarketCollection(void);
   //--- Update the list of pending orders and positions
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

Implement the method for selecting orders and positions by time beyond the class body:

```
//+------------------------------------------------------------------------+
//| Select market orders or positions from the collection with the time    |
//| within the range from begin_time to end_time                           |
//+------------------------------------------------------------------------+
CArrayObj* CMarketCollection::GetListByTime(const datetime begin_time=0,const datetime end_time=0)
  {
   CArrayObj* list=new CArrayObj();
   if(list==NULL)
     {
      ::Print(DFUN,TextByLanguage("Ошибка создания временного списка","Error creating temporary list"));
      return NULL;
     }
   datetime begin=begin_time,end=(end_time==0 ? END_TIME : end_time);
   list.FreeMode(false);
   ListStorage.Add(list);
   m_order_instance.SetProperty(ORDER_PROP_TIME_OPEN,begin);
   int index_begin=m_list_all_orders.SearchGreatOrEqual(&m_order_instance);
   if(index_begin==WRONG_VALUE)
      return list;
   m_order_instance.SetProperty(ORDER_PROP_TIME_OPEN,end);
   int index_end=m_list_all_orders.SearchLessOrEqual(&m_order_instance);
   if(index_end==WRONG_VALUE)
      return list;
   for(int i=index_begin; i<=index_end; i++)
      list.Add(m_list_all_orders.At(i));
   return list;
  }
//+------------------------------------------------------------------+
```

The method is almost identical to the one for selecting historical orders and deals by time we described in the [Part 3](https://www.mql5.com/en/articles/5687#node01). Re-read the description in the appropriate section of the third article if necessary. The difference between this method and the one of the historical collection class is that we do not select the time orders are to be selected by. Market orders and positions have only open time.

Also, change the historical orders and deals collection's class constructor. The MQL5 order system features no close time concept — all orders and deals are arranged in lists according to their placement time (or open time according to the MQL4 order system). To do this, change the string that defines the sorting direction in the collection list of historical orders and deals in the CHistoryCollection class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CHistoryCollection::CHistoryCollection(void) : m_index_deal(0),m_delta_deal(0),m_index_order(0),m_delta_order(0),m_is_trade_event(false)
  {
   this.m_list_all_orders.Sort(#ifdef __MQL5__ SORT_BY_ORDER_TIME_OPEN #else SORT_BY_ORDER_TIME_CLOSE #endif );
   this.m_list_all_orders.Clear();
  }
//+------------------------------------------------------------------+
```

Now in MQL5, all orders and deals in the historical orders and deals collection will be sorted by their placement time by default, while in MQL4, they will be sorted by close time defined in the order properties.

Now let's move on to another feature of the MQL5 order system. When closing a position by an opposite one, a special closing order of the ORDER\_TYPE\_CLOSE\_BY type is placed, while when closing a position by a stop order, a market order to close a position is placed instead.

In order to consider market orders, we had to add yet another property to the methods of receiving and returning the integer properties of the base order (using receiving and returning the order magic number as an example):

```
//+------------------------------------------------------------------+
//| Return the magic number                                          |
//+------------------------------------------------------------------+
long COrder::OrderMagicNumber() const
  {
#ifdef __MQL4__
   return ::OrderMagicNumber();
#else
   long res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_POSITION   : res=::PositionGetInteger(POSITION_MAGIC);           break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : res=::OrderGetInteger(ORDER_MAGIC);                 break;
      case ORDER_STATUS_DEAL              : res=::HistoryDealGetInteger(m_ticket,DEAL_MAGIC);   break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=::HistoryOrderGetInteger(m_ticket,ORDER_MAGIC); break;
      default                             : res=0;                                              break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

I made such changes (or the ones logically corresponding to the method) in all methods of receiving and returning the integer properties of the base order where this status really needs to be taken into account. Since such a status exists, create a new class of the CMarketOrder market order in the Objects folder of the library to store such order types. The class is completely identical to the rest previously created market and historical order and deal objects, so here I will provide only the listing:

```
//+------------------------------------------------------------------+
//|                                                  MarketOrder.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Order.mqh"
//+------------------------------------------------------------------+
//| Market order                                                     |
//+------------------------------------------------------------------+
class CMarketOrder : public COrder
  {
public:
   //--- Constructor
                     CMarketOrder(const ulong ticket=0) : COrder(ORDER_STATUS_MARKET_ORDER,ticket) {}
   //--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_ORDER_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_ORDER_PROP_INTEGER property);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CMarketOrder::SupportProperty(ENUM_ORDER_PROP_INTEGER property)
  {
   if(property==ORDER_PROP_TIME_EXP          ||
      property==ORDER_PROP_DEAL_ENTRY        ||
      property==ORDER_PROP_TIME_UPDATE       ||
      property==ORDER_PROP_TIME_UPDATE_MSC   ||
      property==ORDER_PROP_PROFIT_PT         ||
      property==ORDER_PROP_TIME_CLOSE        ||
      property==ORDER_PROP_TIME_CLOSE_MSC    ||
      property==ORDER_PROP_TICKET_FROM       ||
      property==ORDER_PROP_TICKET_TO
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the order supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CMarketOrder::SupportProperty(ENUM_ORDER_PROP_DOUBLE property)
  {
   if(property==ORDER_PROP_PROFIT            ||
      property==ORDER_PROP_PROFIT_FULL       ||
      property==ORDER_PROP_SWAP              ||
      property==ORDER_PROP_COMMISSION        ||
      property==ORDER_PROP_PRICE_CLOSE       ||
      property==ORDER_PROP_SL                ||
      property==ORDER_PROP_TP                ||
      property==ORDER_PROP_PRICE_STOP_LIMIT
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

In the Defines.mqh file of the library, write the new status — market order:

```
//+------------------------------------------------------------------+
//| Abstract order type (status)                                     |
//+------------------------------------------------------------------+
enum ENUM_ORDER_STATUS
  {
   ORDER_STATUS_MARKET_PENDING,                             // Market pending order
   ORDER_STATUS_MARKET_ORDER,                               // Market order
   ORDER_STATUS_MARKET_POSITION,                            // Market position
   ORDER_STATUS_HISTORY_ORDER,                              // Historical market order
   ORDER_STATUS_HISTORY_PENDING,                            // Removed pending order
   ORDER_STATUS_BALANCE,                                    // Balance operation
   ORDER_STATUS_CREDIT,                                     // Credit operation
   ORDER_STATUS_DEAL,                                       // Deal
   ORDER_STATUS_UNKNOWN                                     // Unknown status
  };
//+------------------------------------------------------------------+
```

Now, in the block for adding orders to the list (CMarketCollection class' Refresh() method for updating the list of market orders and positions), implement the check of the order type. Depending on the type, add either a market order object or a pending order object to the collection list:

```
//+------------------------------------------------------------------+
//| Update the order list                                            |
//+------------------------------------------------------------------+
void CMarketCollection::Refresh(void)
  {
   ::ZeroMemory(this.m_struct_curr_market);
   this.m_is_trade_event=false;
   this.m_is_change_volume=false;
   this.m_new_pendings=0;
   this.m_new_positions=0;
   this.m_change_volume_value=0;
   m_list_all_orders.Clear();
#ifdef __MQL4__
   int total=::OrdersTotal();
   for(int i=0; i<total; i++)
     {
      if(!::OrderSelect(i,SELECT_BY_POS)) continue;
      long ticket=::OrderTicket();
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderType();
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)
        {
         CMarketPosition *position=new CMarketPosition(ticket);
         if(position==NULL) continue;
         if(this.m_list_all_orders.InsertSort(position))
           {
            this.m_struct_market.hash_sum_acc+=ticket;
            this.m_struct_market.total_volumes+=::OrderLots();
            this.m_struct_market.total_positions++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить позицию в список","Failed to add position to list"));
            delete position;
           }
        }
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         if(this.m_list_all_orders.InsertSort(order))
           {
            this.m_struct_market.hash_sum_acc+=ticket;
            this.m_struct_market.total_volumes+=::OrderLots();
            this.m_struct_market.total_pending++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
     }
//--- MQ5
#else
//--- Positions
   int total_positions=::PositionsTotal();
   for(int i=0; i<total_positions; i++)
     {
      ulong ticket=::PositionGetTicket(i);
      if(ticket==0) continue;
      CMarketPosition *position=new CMarketPosition(ticket);
      if(position==NULL) continue;
      if(this.m_list_all_orders.InsertSort(position))
        {
         this.m_struct_curr_market.hash_sum_acc+=(long)::PositionGetInteger(POSITION_TIME_UPDATE_MSC);
         this.m_struct_curr_market.total_volumes+=::PositionGetDouble(POSITION_VOLUME);
         this.m_struct_curr_market.total_positions++;
        }
      else
        {
         ::Print(DFUN,TextByLanguage("Не удалось добавить позицию в список","Failed to add position to list"));
         delete position;
        }
     }
//--- Orders
   int total_orders=::OrdersTotal();
   for(int i=0; i<total_orders; i++)
     {
      ulong ticket=::OrderGetTicket(i);
      if(ticket==0) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderGetInteger(ORDER_TYPE);
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)
        {
         CMarketOrder *order=new CMarketOrder(ticket);
         if(order==NULL) continue;
         if(this.m_list_all_orders.InsertSort(order))
           {
            this.m_struct_curr_market.hash_sum_acc+=(long)ticket;
            this.m_struct_curr_market.total_market++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить маркет-ордер в список","Failed to add market order to list"));
            delete order;
           }
        }
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         if(this.m_list_all_orders.InsertSort(order))
           {
            this.m_struct_curr_market.hash_sum_acc+=(long)ticket;
            this.m_struct_curr_market.total_volumes+=::OrderGetDouble(ORDER_VOLUME_INITIAL);
            this.m_struct_curr_market.total_pending++;
           }
         else
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить отложенный ордер в список","Failed to add pending order to list"));
            delete order;
           }
        }
     }
#endif
//--- First launch
   if(this.m_struct_prev_market.hash_sum_acc==WRONG_VALUE)
     {
      this.SavePrevValues();
     }
//--- If the hash sum of all orders and positions changed
   if(this.m_struct_curr_market.hash_sum_acc!=this.m_struct_prev_market.hash_sum_acc)
     {
      this.m_new_market=this.m_struct_curr_market.total_market-this.m_struct_prev_market.total_market;
      this.m_new_pendings=this.m_struct_curr_market.total_pending-this.m_struct_prev_market.total_pending;
      this.m_new_positions=this.m_struct_curr_market.total_positions-this.m_struct_prev_market.total_positions;
      this.m_change_volume_value=::NormalizeDouble(this.m_struct_curr_market.total_volumes-this.m_struct_prev_market.total_volumes,4);
      this.m_is_change_volume=(this.m_change_volume_value!=0 ? true : false);
      this.m_is_trade_event=true;
      this.SavePrevValues();
     }
  }
//+------------------------------------------------------------------+
```

To be able to consider closing orders of the ORDER\_TYPE\_CLOSE\_BY type, add this order type to the order type definition block in the Refresh() historical orders and deals list update method of the CHistoryCollection class so that such orders are included into the collection. Without that, the base object of the CEngine library cannot define that a position was closed by an opposite one:

```
//+------------------------------------------------------------------+
//| Update the list of orders and deals                              |
//+------------------------------------------------------------------+
void CHistoryCollection::Refresh(void)
  {
#ifdef __MQL4__
   int total=::OrdersHistoryTotal(),i=m_index_order;
   for(; i<total; i++)
     {
      if(!::OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)) continue;
      ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)::OrderType();
      //--- Closed positions and balance/credit operations
      if(order_type<ORDER_TYPE_BUY_LIMIT || order_type>ORDER_TYPE_SELL_STOP)
        {
         CHistoryOrder *order=new CHistoryOrder(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
      else
        {
         //--- Removed pending orders
         CHistoryPending *order=new CHistoryPending(::OrderTicket());
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
     }
//---
   int delta_order=i-m_index_order;
   this.m_index_order=i;
   this.m_delta_order=delta_order;
   this.m_is_trade_event=(this.m_delta_order!=0 ? true : false);
//--- __MQL5__
#else
   if(!::HistorySelect(0,END_TIME)) return;
//--- Orders
   int total_orders=::HistoryOrdersTotal(),i=m_index_order;
   for(; i<total_orders; i++)
     {
      ulong order_ticket=::HistoryOrderGetTicket(i);
      if(order_ticket==0) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::HistoryOrderGetInteger(order_ticket,ORDER_TYPE);
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL || type==ORDER_TYPE_CLOSE_BY)
        {
         CHistoryOrder *order=new CHistoryOrder(order_ticket);
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
      else
        {
         CHistoryPending *order=new CHistoryPending(order_ticket);
         if(order==NULL) continue;
         if(!this.m_list_all_orders.InsertSort(order))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить ордер в список","Failed to add order to list"));
            delete order;
           }
        }
     }
//--- save the index of the last added order and the difference as compared to the previous check
   int delta_order=i-this.m_index_order;
   this.m_index_order=i;
   this.m_delta_order=delta_order;

//--- Deals
   int total_deals=::HistoryDealsTotal(),j=m_index_deal;
   for(; j<total_deals; j++)
     {
      ulong deal_ticket=::HistoryDealGetTicket(j);
      if(deal_ticket==0) continue;
      CHistoryDeal *deal=new CHistoryDeal(deal_ticket);
      if(deal==NULL) continue;
      this.m_list_all_orders.InsertSort(deal);
     }
//---save the index of the last added deal and the difference as compared to the previous check
   int delta_deal=j-this.m_index_deal;
   this.m_index_deal=j;
   this.m_delta_deal=delta_deal;
//--- Set the new event flag in history
   this.m_is_trade_event=(this.m_delta_order+this.m_delta_deal);
#endif
  }
//+------------------------------------------------------------------+
```

When testing the CEngine class for defining occurring account events, I have detected and fixed some minor flaws in the service methods. There is no point in describing them here since they do not affect the performance, while their description diverts attention from the development of an important library functionality. All changes have already been made to the class listings. You can see them yourself in the library files attached below.

**Let's continue our work on events definition.**

After debugging the trading events definition, all occurred events will be packed into a single class member variable made as a set of flags. The method reading data from the variable for decomposing its value into components characterizing a specific event will be created afterwards.

Add the class member variable for storing the trading event code, methods of verifying a trading event for hedging and netting accounts and methods returning necessary order objects to the private section of the CEngine class.

In the public section, declare the methods returning the lists of market positions and pending orders, historical market orders and deals, the method returning a trading event code from the m\_trade\_event\_code variable and the method returning the hedge account flag.

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   int                  m_trade_event_code;              // Account trading event status code
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the first launch flag
   bool                 IsFirstStart(void);
//--- Working with (1) hedging and (2) netting collections
   void                 WorkWithHedgeCollections(void);
   void                 WorkWithNettoCollections(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical market order by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(const ulong position_id);
   COrder*              GetLastOrderPosition(const ulong position_id);
   COrder*              GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListHistoryDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Return the (1) trading event code and (2) hedge account flag
   int                  TradeEventCode(void)             const { return this.m_trade_event_code;   }
   bool                 IsHedge(void)                    const { return this.m_is_hedge;           }
//--- Create the timer account
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

Initialize the trading event code in the class constructor's initialization list.

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),m_trade_event_code(TRADE_EVENT_FLAG_NO_EVENT)
  {
   ::EventSetMillisecondTimer(TIMER_FREQUENCY);
   this.m_list_counters.Sort();
   this.CreateCounter(COLLECTION_COUNTER_ID,COLLECTION_COUNTER_STEP,COLLECTION_PAUSE);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
  }
//+------------------------------------------------------------------+
```

Implement declared methods outside the class body:

```
//+------------------------------------------------------------------+
//| Return the list of market positions                              |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketPosition(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of market pending orders                         |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketPendings(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of market orders                                 |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketOrders(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_ORDER,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of historical orders                             |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListHistoryOrders(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_ORDER,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of removed pending orders                        |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListHistoryPendings(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of deals                                         |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListHistoryDeals(void)
  {
   CArrayObj* list=this.m_history.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_DEAL,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//|  Return the list of all position orders                          |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListAllOrdersByPosID(const ulong position_id)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_ID,position_id,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
//| Return the last position                                         |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastPosition(void)
  {
   CArrayObj* list=this.GetListMarketPosition();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return position by ticket                                        |
//+------------------------------------------------------------------+
COrder* CEngine::GetPosition(const ulong ticket)
  {
   CArrayObj* list=this.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TICKET);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last deal                                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastDeal(void)
  {
   CArrayObj* list=this.GetListHistoryDeals();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last market pending order                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastMarketPending(void)
  {
   CArrayObj* list=this.GetListMarketPendings();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical pending order                         |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastHistoryPending(void)
  {
   CArrayObj* list=this.GetListHistoryPendings();
   if(list==NULL) return NULL;
   list.Sort(#ifdef __MQL5__ SORT_BY_ORDER_TIME_OPEN_MSC #else SORT_BY_ORDER_TIME_CLOSE_MSC #endif);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last market order                                     |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastMarketOrder(void)
  {
   CArrayObj* list=this.GetListMarketOrders();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical market order                          |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastHistoryOrder(void)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return historical market order by its ticket                     |
//+------------------------------------------------------------------+
COrder* CEngine::GetHistoryOrder(const ulong ticket)
  {
   CArrayObj* list=this.GetListHistoryOrders();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,(long)ticket,EQUAL);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TICKET);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the first historical market order                         |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetFirstOrderPosition(const ulong position_id)
  {
   CArrayObj* list=this.GetListAllOrdersByPosID(position_id);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN);
   COrder* order=list.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
//| Return the last historical market order                          |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastOrderPosition(const ulong position_id)
  {
   CArrayObj* list=this.GetListAllOrdersByPosID(position_id);
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

**Let's see how the lists are obtained using the following example:**

```
//+------------------------------------------------------------------+
//| Return the list of market positions                              |
//+------------------------------------------------------------------+
CArrayObj* CEngine::GetListMarketPosition(void)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   return list;
  }
//+------------------------------------------------------------------+
```

All is quite simple and convenient: receive the full list of positions from the collection of market orders and positions using the GetList() collection method. After that, select an order having the "position" status from it using the method of selecting orders by a specified property from the CSelect class. The method was described in the [third article](https://www.mql5.com/en/articles/5687#node01) of the library description. Return the obtained list.

_The list may be empty (NULL), therefore the result returned by this method should be checked in the calling program._

**Let's see how to receive a necessary order using the following example:**

```
//+------------------------------------------------------------------+
//| Return the last position                                         |
//+------------------------------------------------------------------+
COrder* CEngine::GetLastPosition(void)
  {
   CArrayObj* list=this.GetListMarketPosition();
   if(list==NULL) return NULL;
   list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   COrder* order=list.At(list.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

First, receive the list of positions using the GetListMarketPosition() method described above. If the list is empty, return NULL. Next, sort the list by open time in milliseconds (since we are going to receive the last open position, the list should be sorted by time) and select the last order in it. As a result, return an order obtained from the list to the calling program.

_The result of searching an order returned by the method may be empty (order is not found) and be equal to NULL. Therefore, check the obtained result for NULL before accessing it._

As you can see, everything is quick and easy. We can construct any method for receiving any data from any existing collection lists, as well as the ones we will create in the future. This provides us with greater flexibility in using such lists.

The methods for receiving the lists are placed in the public class section allowing us to receive any data from the lists in custom programs as it was done with the methods considered above.

The methods for receiving necessary orders are hidden in the private section. They are needed only in the CEngine class for internal needs — in particular, for obtaining data on the last orders, deals and positions. In custom programs, we can make custom functions for receiving specific orders by specified properties (the examples are displayed above).

**To easily obtain any data from the collections, we will eventually make a wide range of functions for end users**.

This will be done in the final articles devoted to working with order collections.

**Now let's implement the method for checking trading events**.

Currently, it works only on hedging accounts for MQL5. After that, I am going to develop the methods for checking trading events for MQL5 netting accounts and for MQL4. To test the method operation, it features display of the check and event results. This ability is to be removed later, as this function is to be fulfilled by another method that will be created after debugging the method of checking trading events on the account.

```
//+------------------------------------------------------------------+
//| Check trading events (hedging)                                   |
//+------------------------------------------------------------------+
void CEngine::WorkWithHedgeCollections(void)
  {
//--- Initialize the trading events code and flag
   this.m_trade_event_code=TRADE_EVENT_FLAG_NO_EVENT;
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- Actions during the first launch
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the market state dynamics and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();
//---
#ifdef __MQL4__

#else // MQL5
//--- If an event is only in market orders and positions
   if(this.m_is_market_trade_event && !this.m_is_history_trade_event)
     {
      Print(DFUN,TextByLanguage("Новое торговое событие на счёте","New trading event on account"));
      //--- If the number of pending orders increased
      if(this.m_market.NewPendingOrders()>0)
        {
         //--- Add the flag for installing a pending order
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_PLASED;
         string text=TextByLanguage("Установлен отложенный ордер: ","Pending order placed: ");
         //--- Take the last market pending order
         COrder* order=this.GetLastMarketPending();
         if(order!=NULL)
           {
            //--- add the order ticket to the message
            text+=order.TypeDescription()+" #"+(string)order.Ticket();
           }
         //--- Add the message to the journal
         Print(DFUN,text);
        }
      //--- If the number of market orders increased
      if(this.m_market.NewMarketOrders()>0)
        {
         //--- do not add the event flag
         //--- ...
         string text=TextByLanguage("Выставлен маркет-ордер: ","Market order placed: ");
         //--- Take the last market order
         COrder* order=this.GetLastMarketOrder();
         if(order!=NULL)
           {
            //--- add the order ticket to the message
            text+=order.TypeDescription()+" #"+(string)order.Ticket();
           }
         //--- Add the message to the journal
         Print(DFUN,text);
        }
     }

//--- If an event is only in historical orders and deals
   else if(this.m_is_history_trade_event && !this.m_is_market_trade_event)
     {
      Print(DFUN,TextByLanguage("Новое торговое событие в истории счёта","New trading event in account history"));
      //--- If a new deal appeared
      if(this.m_history.NewDeals()>0)
        {
         //--- Add the account balance event flag
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ACCOUNT_BALANCE;
         string text=TextByLanguage("Новая сделка: ","New deal: ");
         //--- Take the last deal
         COrder* deal=this.GetLastDeal();
         if(deal!=NULL)
           {
            //--- add its description to the text
            text+=deal.TypeDescription();
            //--- if the deal is a balance operation
            if((ENUM_DEAL_TYPE)deal.GetProperty(ORDER_PROP_TYPE)==DEAL_TYPE_BALANCE)
              {
              //--- check the deal profit and add the event (adding or withdrawing funds) to the message
               text+=(deal.Profit()>0 ? TextByLanguage(": Пополнение счёта: ",": Account Recharge: ") : TextByLanguage(": Вывод средств: ",": Withdrawal: "))+::DoubleToString(deal.Profit(),(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS));
              }
           }
         //--- Display the message in the journal
         Print(DFUN,text);
        }
     }

//--- If the events are in market and historical orders and positions
   else if(this.m_is_market_trade_event && this.m_is_history_trade_event)
     {
      Print(DFUN,TextByLanguage("Новые торговые события на счёте и в истории счёта","New trading events on account and in account history"));

      //--- If the number of pending orders decreased and no new deals appeared
      if(this.m_market.NewPendingOrders()<0 && this.m_history.NewDeals()==0)
        {
         //--- Add the flag for removing a pending order
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_REMOVED;
         string text=TextByLanguage("Удалён отложенный ордер: ","Removed pending order: ");
         //--- Take the last historical pending order
         COrder* order=this.GetLastHistoryPending();
         if(order!=NULL)
           {
            //--- add the ticket to the appropriate message
            text+=order.TypeDescription()+" #"+(string)order.Ticket();
           }
         //--- Display the message in the journal
         Print(DFUN,text);
        }

      //--- If there is a new deal and a new historical order
      if(this.m_history.NewDeals()>0 && this.m_history.NewOrders()>0)
        {
         //--- Take the last deal
         COrder* deal=this.GetLastDeal();
         if(deal!=NULL)
           {
            //--- In case of a market entry deal
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_IN)
              {
               //--- Add the position opening flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_OPENED;
               string text=TextByLanguage("Открыта позиция: ","Position opened: ");
               //--- If the number of pending orders decreased
               if(this.m_market.NewPendingOrders()<0)
                 {
                  //--- Add the pending order activation flag
                  this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
                  text=TextByLanguage("Сработал отложенный ордер: ","Pending order activated: ");
                 }
               //--- Take the order's ticket from the order
               ulong order_ticket=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(order_ticket);
               if(order!=NULL)
                 {
                  //--- If the current order volume exceeds zero
                  if(order.VolumeCurrent()>0)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                     text=TextByLanguage("Частично открыта позиция: ","Position partially open: ");
                    }
                  //--- add the order direction to the message
                  text+=order.DirectionDescription();
                 }
               //--- add the position ticket from the deal to the message and display the message in the journal
               text+=" #"+(string)deal.PositionID();
               Print(DFUN,text);
              }

            //--- In case of a market exit deal
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT)
              {
               //--- Add the position closing flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_CLOSED;
               string text=TextByLanguage("Закрыта позиция: ","Position closed: ");
               //--- Take the deal's order ticket from the deal
               ulong order_ticket=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(order_ticket);
               if(order!=NULL)
                 {
                  //--- If the deal's position is still present in the market
                  COrder* pos=this.GetPosition(deal.PositionID());
                  if(pos!=NULL)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                     text=TextByLanguage("Частично закрыта позиция: ","Partially closed position: ");
                    }
                  //--- Otherwise, if the position is fully closed
                  else
                    {
                     //--- If the order has the flag for closing by StopLoss
                     if(order.IsCloseByStopLoss())
                       {
                        //--- Add the flag for closing by StopLoss
                        this.m_trade_event_code+=TRADE_EVENT_FLAG_SL;
                        text=TextByLanguage("Позиция закрыта по StopLoss: ","Position closed by StopLoss: ");
                       }
                     //--- If the order has the flag for closing by TakeProfit
                     if(order.IsCloseByTakeProfit())
                       {
                        //--- Add the the flag for closing by TakeProfit
                        this.m_trade_event_code+=TRADE_EVENT_FLAG_TP;
                        text=TextByLanguage("Позиция закрыта по TakeProfit: ","Position closed by TakeProfit: ");
                       }
                    }
                  //--- add reverse order direction to the message:
                  //--- closing Buy order for Sell position and closing Sell order for Buy position,
                  //--- therefore, reverse the order direction for correct description of a closed position
                  text+=(order.DirectionDescription()=="Sell" ? "Buy " : "Sell ");
                 }
               //--- add the ticket of the deal's position and display the message in the journal
               text+="#"+(string)deal.PositionID();
               Print(DFUN,text);
              }

            //--- When closing by an opposite position
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT_BY)
              {
               //--- Add the position closing flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_CLOSED;
               //--- Add the flag for closing by an opposite position
               this.m_trade_event_code+=TRADE_EVENT_FLAG_BY_POS;
               string text=TextByLanguage("Позиция закрыта встречной: ","Position closed by opposite position: ");
               //--- Take the deal order
               ulong ticket_from=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(ticket_from);
               if(order!=NULL)
                 {
                  //--- add reverse order direction to the message:
                  //--- closing Buy order for Sell position and closing Sell order for Buy position,
                  //--- therefore, reverse the order direction for correct description of a closed position
                  text+=(order.DirectionDescription()=="Sell" ? "Buy" : "Sell");
                  text+=" #"+(string)order.PositionID()+TextByLanguage(" закрыта позицией #"," closed by position #")+(string)order.PositionByID();
                  //--- If the order position is still present in the market
                  COrder* pos=this.GetPosition(order.PositionID());
                  if(pos!=NULL)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                     text+=TextByLanguage(" частично"," partially");
                    }
                 }
               //--- Display the message in the journal
               Print(DFUN,text);
              }
           //--- end of the last deal processing block
           }
        }
     }
#endif
  }
//+------------------------------------------------------------------+
```

The method is simple but quite large. Therefore, the code features all checks and corresponding actions allowing us to see what happens inside the method more conveniently. For now, the method implements the check for MQL5 hedging account. In fact, everything comes down to checking the number of newly appeared orders and deals either in the account history or in the market. For displaying in the journal, the data is taken from the last position, last deal, last order or last deal's order — all this is done for displaying data in the journal to check the code execution. This functionality will later be removed from the code and replaced with a single method to be used in the programs working with the library.

### Testing the processing of trading events

Let's create a test EA for checking the method defining trading events on the account.

Let's implement the set of buttons in order to manage the new events.

**The set of necessary actions and corresponding buttons is as follows:**

- Open Buy position
- Place a pending BuyLimit order
- Place a pending BuyStop order
- Place a pending BuyStopLimit order
- Close Buy position
- Close half of Buy position
- Close Buy position by opposite Sell position
- Open Sell position
- Place a pending SellLimit order
- Place a pending SellStop order
- Place a pending SellStopLimit order
- Close Sell position
- Close half of Sell position
- Close Sell position by opposite Buy position
- Close all positions
- Withdraw funds from the account

**The set of inputs is as follows:**

- **Magic number** \- magic number
- **Lots**\- volume of opened positions
- **StopLoss in points**
- **TakeProfit in points**
- **Pending orders distance (points)**
- **StopLimit orders distance (points)**


A StopLimit order is placed as a stop order at a distance from the price set by **Pending orders distance** value.


As soon as the price reaches the specified order and activates it, that price level is used to place a limit order at a distance from the price set by **StopLimit orders distance** value.
- **Slippage in points**
- **Withdrawal funds (in tester)** \- funds withdrawn from the account in the tester


We will need the functions for calculating correct values to define order placement prices relative to the StopLevel, stop orders and position volumes. At this stage, let's add the functions to the library of service functions in the DELib.mqh file as long as we do not have trading classes and symbol classes yet:

```
//+------------------------------------------------------------------+
//| Return the minimum symbol lot                                    |
//+------------------------------------------------------------------+
double MinimumLots(const string symbol_name)
  {
   return SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MIN);
  }
//+------------------------------------------------------------------+
//| Return the maximum symbol lot                                    |
//+------------------------------------------------------------------+
double MaximumLots(const string symbol_name)
  {
   return SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MAX);
  }
//+------------------------------------------------------------------+
//| Return the symbol lot change step                                |
//+------------------------------------------------------------------+
double StepLots(const string symbol_name)
  {
   return SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_STEP);
  }
//+------------------------------------------------------------------+
//| Return the normalized lot                                        |
//+------------------------------------------------------------------+
double NormalizeLot(const string symbol_name, double order_lots)
  {
   double ml=SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MIN);
   double mx=SymbolInfoDouble(symbol_name,SYMBOL_VOLUME_MAX);
   double ln=NormalizeDouble(order_lots,int(ceil(fabs(log(ml)/log(10)))));
   return(ln<ml ? ml : ln>mx ? mx : ln);
  }
//+------------------------------------------------------------------+
//| Return correct StopLoss relative to StopLevel                    |
//+------------------------------------------------------------------+
double CorrectStopLoss(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const double stop_loss,const int spread_multiplier=2)
  {
   if(stop_loss==0) return 0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT);
   double price=(order_type==ORDER_TYPE_BUY ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : order_type==ORDER_TYPE_SELL ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price_set);
   return
     (order_type==ORDER_TYPE_BUY       ||
      order_type==ORDER_TYPE_BUY_LIMIT ||
      order_type==ORDER_TYPE_BUY_STOP
      #ifdef __MQL5__                  ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT
      #endif ?
      NormalizeDouble(fmin(price-lv*pt,stop_loss),dg) :
      NormalizeDouble(fmax(price+lv*pt,stop_loss),dg)
     );
  }
//+------------------------------------------------------------------+
//| Return correct StopLoss relative to StopLevel                    |
//+------------------------------------------------------------------+
double CorrectStopLoss(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const int stop_loss,const int spread_multiplier=2)
  {
   if(stop_loss==0) return 0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT);
   double price=(order_type==ORDER_TYPE_BUY ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : order_type==ORDER_TYPE_SELL ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price_set);
   return
     (order_type==ORDER_TYPE_BUY       ||
      order_type==ORDER_TYPE_BUY_LIMIT ||
      order_type==ORDER_TYPE_BUY_STOP
      #ifdef __MQL5__                  ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT
      #endif ?
      NormalizeDouble(fmin(price-lv*pt,price-stop_loss*pt),dg) :
      NormalizeDouble(fmax(price+lv*pt,price+stop_loss*pt),dg)
     );
  }
//+------------------------------------------------------------------+
//| Return correct TakeProfit relative to StopLevel                  |
//+------------------------------------------------------------------+
double CorrectTakeProfit(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const double take_profit,const int spread_multiplier=2)
  {
   if(take_profit==0) return 0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT);
   double price=(order_type==ORDER_TYPE_BUY ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : order_type==ORDER_TYPE_SELL ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price_set);
   return
     (order_type==ORDER_TYPE_BUY       ||
      order_type==ORDER_TYPE_BUY_LIMIT ||
      order_type==ORDER_TYPE_BUY_STOP
      #ifdef __MQL5__                  ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT
      #endif ?
      NormalizeDouble(fmax(price+lv*pt,take_profit),dg) :
      NormalizeDouble(fmin(price-lv*pt,take_profit),dg)
     );
  }
//+------------------------------------------------------------------+
//| Return correct TakeProfit relative to StopLevel                  |
//+------------------------------------------------------------------+
double CorrectTakeProfit(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const int take_profit,const int spread_multiplier=2)
  {
   if(take_profit==0) return 0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT);
   double price=(order_type==ORDER_TYPE_BUY ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : order_type==ORDER_TYPE_SELL ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price_set);
   return
     (order_type==ORDER_TYPE_BUY       ||
      order_type==ORDER_TYPE_BUY_LIMIT ||
      order_type==ORDER_TYPE_BUY_STOP
      #ifdef __MQL5__                  ||
      order_type==ORDER_TYPE_BUY_STOP_LIMIT
      #endif ?
      ::NormalizeDouble(::fmax(price+lv*pt,price+take_profit*pt),dg) :
      ::NormalizeDouble(::fmin(price-lv*pt,price-take_profit*pt),dg)
     );
  }
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const double price_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch(order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,price_set),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,price_set),dg);
      default                          :  Print(DFUN,TextByLanguage("Не правильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CorrectPricePending(const string symbol_name,const ENUM_ORDER_TYPE order_type,const int distance_set,const double price=0,const int spread_multiplier=2)
  {
   double pt=SymbolInfoDouble(symbol_name,SYMBOL_POINT),pp=0;
   int lv=StopLevel(symbol_name,spread_multiplier), dg=(int)SymbolInfoInteger(symbol_name,SYMBOL_DIGITS);
   switch(order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_ASK) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmax(pp+lv*pt,pp+distance_set*pt),dg);
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? SymbolInfoDouble(symbol_name,SYMBOL_BID) : price); return NormalizeDouble(fmin(pp-lv*pt,pp-distance_set*pt),dg);
      default                          :  Print(DFUN,TextByLanguage("Не правильный тип ордера: ","Invalid order type: "),EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
//| Check the stop level in points relative to StopLevel             |
//+------------------------------------------------------------------+
bool CheckStopLevel(const string symbol_name,const int stop_in_points,const int spread_multiplier)
  {
   return(stop_in_points>=StopLevel(symbol_name,spread_multiplier));
  }
//+------------------------------------------------------------------+
//| Return StopLevel in points                                       |
//+------------------------------------------------------------------+
int StopLevel(const string symbol_name,const int spread_multiplier)
  {
   int spread=(int)SymbolInfoInteger(symbol_name,SYMBOL_SPREAD);
   int stop_level=(int)SymbolInfoInteger(symbol_name,SYMBOL_TRADE_STOPS_LEVEL);
   return(stop_level==0 ? spread*spread_multiplier : stop_level);
  }
//+------------------------------------------------------------------+
```

The functions simply calculate correct values so that they do not violate limitations set on the server. Apart from the symbol, the StopLevel calculation function also receives the spread multiplier. This is done because if the StopLevel on the server is set to zero, this means a floating level, and in order to calculate StopLevel, we should use a spread value multiplied by a certain number (usually 2, but 3 is also possible). This multiplier is passed to the function allowing us to either hardcode it in the EA settings or calculate it.

For the sake of saving our time, we will not write custom trading functions. Instead, we will use the ready-made [trading classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) of the standard library, namely, the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class for performing trading operations.

In order to create working chart buttons, we are going to implement an enumeration with its members setting button names, labels and values for checking a certain button pressing.

In MQL5\\Experts\\TestDoEasy\\Part04\\, create a new EA called TestDoEasy04.mqh (check the OnTimer and OnChartEvent event handlers in MQL Wizard when creating the EA):

![](https://c.mql5.com/2/36/Image_1.png)

After creating the EA template by the MQL Wizard, include the custom library and the standard library's trading class into it. Also, add the input parameters:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart04.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <Trade\Trade.mqh>
//--- enums
enum ENUM_BUTTONS
  {
   BUTT_BUY,
   BUTT_BUY_LIMIT,
   BUTT_BUY_STOP,
   BUTT_BUY_STOP_LIMIT,
   BUTT_CLOSE_BUY,
   BUTT_CLOSE_BUY2,
   BUTT_CLOSE_BUY_BY_SELL,
   BUTT_SELL,
   BUTT_SELL_LIMIT,
   BUTT_SELL_STOP,
   BUTT_SELL_STOP_LIMIT,
   BUTT_CLOSE_SELL,
   BUTT_CLOSE_SELL2,
   BUTT_CLOSE_SELL_BY_BUY,
   BUTT_CLOSE_ALL,
   BUTT_PROFIT_WITHDRAWAL
  };
#define TOTAL_BUTT   (16)
//--- structures
struct SDataButt
  {
   string      name;
   string      text;
  };
//--- input variables
input ulong    InpMagic       =  123;  // Magic number
input double   InpLots        =  0.1;  // Lots
input uint     InpStopLoss    =  50;   // StopLoss in points
input uint     InpTakeProfit  =  50;   // TakeProfit in points
input uint     InpDistance    =  50;   // Pending orders distance (points)
input uint     InpDistanceSL  =  50;   // StopLimit orders distance (points)
input uint     InpSlippage    =  0;    // Slippage in points
input double   InpWithdrawal  =  10;   // Withdrawal funds (in tester)
//--- global variables
CEngine        engine;
CTrade         trade;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ulong          magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
//+------------------------------------------------------------------+
```

Here we include the main object of the CEngine library and the CTrade trading class.

Next, create the enumeration specifying all required buttons. _The sequence of enumeration methods is important since it sets the order of buttons creation and their location on a chart._

After that, declare the structure for storing the name of a button graphical object and a text to be inscribed on a button.

In the inputs block, set all EA parameter variables listed above. In the EA's global variables block, declare the library object, trading class object, button structures array and the variables the input values in the **OnInit()** handler are to be assigned to:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Check account type
   if(!engine.IsHedge())
     {
      Alert(TextByLanguage("Ошибка. Счёт должен быть хеджевым","Error. Account must be hedge"));
      return INIT_FAILED;
     }
//--- set global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
//--- create buttons
   if(!CreateButtons())
      return INIT_FAILED;
//---
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_ERRORS);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

In the OnInit() handler, check the account type and, if it is not hedging, inform of that and exit the program with an error.

Next, set the prefix of object names (so that the EA is able to recognize its objects) and fill in the array of structures with the button data in a loop by the number of buttons.

The button object name is set as a prefix+string representation of the ENUM\_BUTTONS enumeration corresponding to the loop index, while the button text is compiled by transforming the string representation of the enumeration corresponding to the loop index using the EnumToButtText() function.

The lot of opened positions and placed orders is calculated further on. Since half of positions is closed, the lot of an opened position should be at least twice the minimum lot. Therefore, the maximum lot is taken out of two:

1) entered in the inputs, 2) minimum lot multiplied by two in the string **fmax(InpLots,MinimumLots(Symbol())\*2.0)**, the value of the obtained lot is normalized and assigned to the **lot** global variable. As a result, if the lot entered by the user in the inputs is less than the double minimum lot, the double minimum lot is used. Otherwise, the lot entered by a user is applied.

The remaining inputs are assigned to the appropriate global variables and the **CreateButtons()** function is called for creating buttons from the structure array already filled with button data during the previous step. If creating the button using the ButtonCreate() function failed, the error message is displayed and the program operation ends with initialization error.

Finally, the CTrade class is initialized:

```
//---
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_ERRORS);
//---
```

- Set the slippage in points,
- set the magic number,

- set the order execution type according to the settings of the current symbol,

- set the margin calculation mode according to the current account settings and
- set the messages logging level to display only error messages in the journal


(the full logging mode is automatically enabled in the tester).

In the **OnDeinit()** handler, implement removing all buttons by the object names prefix:

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- delete objects
   ObjectsDeleteAll(0,prefix);
  }
//+------------------------------------------------------------------+
```

Now let's decide on the library timer and EA events handler launch order.

- If the EA is launched not in the tester, the library timer is to be launched from the EA timer, while the event handler works in normal mode.
- If the EA is launched in the tester, the library timer is to be launched from the EA's OnTick() handler. The button pressing events are tracked in OnTick() as well.

**EA's OnTick(), OnTimer() and OnChartEvent() handlers:**

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
   int total=ObjectsTotal(0);
   for(int i=0;i<total;i++)
     {
      string obj_name=ObjectName(0,i);
      if(StringFind(obj_name,prefix+"BUTT_")<0)
         continue;
      PressButtonEvents(obj_name);
     }
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,"BUTT_")>0)
     {
      PressButtonEvents(sparam);
     }
  }
//+------------------------------------------------------------------+
```

**In OnTick()**

- Check where the EA is launched. If it is launched in the tester,call the OnTimer() handler of the library.
- Next, check the object name by all the current chart objects in the loop. If it matches the name of a button, the button pressing handler is called.


**In OnTimer()**

- Check where the EA is launched. If it is not launched in the tester, call the OnTimer() handler of the library.

**In OnChartEvent()**

- Check where the EA is launched. If it is launched in the tester, exit the handler.
- Next, the event ID is checked, and if this is the event of clicking a graphical object and the object name contains a text of belonging to buttons, the appropriate button pressing handler is called.

**CreateButtons() function:**

```
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(void)
  {
   int h=18,w=84,offset=10;
   int cx=offset,cy=offset+(h+1)*(TOTAL_BUTT/2)+h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-2) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-2 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
```

In this function, it all comes down to calculating the coordinates and colors of a button in a loop by the number of the ENUM\_BUTTONS enumeration members. The coordinates and the color are calculated based on the loop index indicating the number of the ENUM\_BUTTONS enumeration member. After calculating the x and y coordinates, the button creation function with coordinates and color values calculated in the loop is called.

**EnumToButtText() function:**

```
//+------------------------------------------------------------------+
//| Convert enumeration into the button text                         |
//+------------------------------------------------------------------+
string EnumToButtText(const ENUM_BUTTONS member)
  {
   string txt=StringSubstr(EnumToString(member),5);
   StringToLower(txt);
   StringReplace(txt,"buy","Buy");
   StringReplace(txt,"sell","Sell");
   StringReplace(txt,"_limit"," Limit");
   StringReplace(txt,"_stop"," Stop");
   StringReplace(txt,"close_","Close ");
   StringReplace(txt,"2"," 1/2");
   StringReplace(txt,"_by_"," by ");
   StringReplace(txt,"profit_","Profit ");
   return txt;
  }
//+------------------------------------------------------------------+
```

All is simple here: the function receives the enumeration member and converts it into a string removing the unnecessary text. Next, all characters of the obtained string are converted to lowercase and all unsuitable entries are further replaced with necessary ones.

Input enumeration string converted into the text is finally returned.

**The functions for creating the button, placing it and receiving its status:**

```
//+------------------------------------------------------------------+
//| Create the button                                                |
//+------------------------------------------------------------------+
bool ButtonCreate(const string name,const int x,const int y,const int w,const int h,const string text,const color clr,const string font="Calibri",const int font_size=8)
  {
   if(ObjectFind(0,name)<0)
     {
      if(!ObjectCreate(0,name,OBJ_BUTTON,0,0,0))
        {
         Print(DFUN,TextByLanguage("не удалось создать кнопку! Код ошибки=","Could not create button! Error code="),GetLastError());
         return false;
        }
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,font_size);
      ObjectSetString(0,name,OBJPROP_FONT,font);
      ObjectSetString(0,name,OBJPROP_TEXT,text);
      ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
      ObjectSetString(0,name,OBJPROP_TOOLTIP,"\n");
      ObjectSetInteger(0,name,OBJPROP_BORDER_COLOR,clrGray);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//| Return the button status                                         |
//+------------------------------------------------------------------+
bool ButtonState(const string name)
  {
   return (bool)ObjectGetInteger(0,name,OBJPROP_STATE);
  }
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
  }
//+------------------------------------------------------------------+
```

Everything is simple and clear here, so no explanations are needed.

**The function for handling the buttons pressing:**

```
//+------------------------------------------------------------------+
//| Handling buttons pressing                                        |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   //--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         trade.Buy(NormalizeLot(Symbol(),lot),Symbol(),0,sl,tp);
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,takeprofit);
         //--- Set BuyLimit order
         trade.BuyLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set,takeprofit);
         //--- Set BuyStop order
         trade.BuyStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Get the correct BuyStop order placement price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Calculate BuyLimit order price relative to BuyStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,takeprofit);
         //--- Set BuyStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_BUY_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Get correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         trade.Sell(lot,Symbol(),0,sl,tp);
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Get correct order price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,takeprofit);
         //--- Set SellLimit order
         trade.SellLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Get the correct price of placing an order relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set,takeprofit);
         //--- Set SellStop order
         trade.SellStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Get the correct SellStop order price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Calculate SellLimit order price relative to SellStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,takeprofit);
         //--- Set the SellStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_SELL_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the Buy position index with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Buy position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY2 button is closed: Close the half of the Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the Buy position index with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Buy position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.VolumeCurrent()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY_BY_SELL button is pressed: Close Buy with the maximum profit by the opposite Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Select Sell positions only from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         if(index_buy>WRONG_VALUE && index_sell>WRONG_VALUE)
           {
            //--- Select Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            //--- Select Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            if(position_buy!=NULL && position_sell!=NULL)
              {
               //--- Close Buy position by the opposite Sell position
               trade.PositionCloseBy(position_buy.Ticket(),position_sell.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL button is pressed: Close Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL))
        {
         //--- Get the list of all opened positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Sell position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL2 button is pressed: Close the half of the Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Sell position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.VolumeCurrent()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL_BY_BUY button is pressed: Close Sell with the maximum profit by the opposite Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Get only Sell positions from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         if(index_sell>WRONG_VALUE && index_buy>WRONG_VALUE)
           {
            //--- Select the Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            //--- Select the Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            if(position_sell!=NULL && position_buy!=NULL)
              {
               //--- Close the Sell position by the opposite Buy position
               trade.PositionCloseBy(position_sell.Ticket(),position_buy.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_ALL button is pressed: Close all positions starting with the one with the least profit
      else if(button==EnumToString(BUTT_CLOSE_ALL))
        {
         //--- Receive the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         if(list!=NULL)
           {
            //--- Sort the list by profit considering commission and swap
            list.Sort(SORT_BY_ORDER_PROFIT_FULL);
            int total=list.Total();
            //--- In the loop from the position with the least profit
            for(int i=0;i<total;i++)
              {
               COrder* position=list.At(i);
               if(position==NULL)
                  continue;
               //--- close each position by its ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_PROFIT_WITHDRAWAL button is pressed: Withdraw the funds from the account
      if(button==EnumToString(BUTT_PROFIT_WITHDRAWAL))
        {
         //--- If the program is launched in the tester
         if(MQLInfoInteger(MQL_TESTER))
           {
            //--- Emulate funds withdrawal
            TesterWithdrawal(withdrawal);
           }
        }
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button and redraw the chart
      ButtonState(button_name,false);
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

The function is quite large but simple: it receives the name of the button object to be converted into a string ID. The button status is checked next, and if it is pressed, the string ID is checked. The appropriate if-else branch is executed calculating all levels and applying a necessary adjustment in order not to violate the StopLevel limitation. The corresponding method of the trading class is executed.

All explanations are written directly in the string comments of the code.

For the test EA, we did the minimum necessary checks skipping all the other checks important for a real account. Currently, the most important thing for us is to check the library operation, rather than develop an EA to work with it on real accounts.

**The full listing of the test EA:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart04.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#include <Trade\Trade.mqh>
//--- enums
enum ENUM_BUTTONS
  {
   BUTT_BUY,
   BUTT_BUY_LIMIT,
   BUTT_BUY_STOP,
   BUTT_BUY_STOP_LIMIT,
   BUTT_CLOSE_BUY,
   BUTT_CLOSE_BUY2,
   BUTT_CLOSE_BUY_BY_SELL,
   BUTT_SELL,
   BUTT_SELL_LIMIT,
   BUTT_SELL_STOP,
   BUTT_SELL_STOP_LIMIT,
   BUTT_CLOSE_SELL,
   BUTT_CLOSE_SELL2,
   BUTT_CLOSE_SELL_BY_BUY,
   BUTT_CLOSE_ALL,
   BUTT_PROFIT_WITHDRAWAL
  };
#define TOTAL_BUTT   (16)
//--- structures
struct SDataButt
  {
   string      name;
   string      text;
  };
//--- input variables
input ulong    InpMagic       =  123;  // Magic number
input double   InpLots        =  0.1;  // Lots
input uint     InpStopLoss    =  50;   // StopLoss in points
input uint     InpTakeProfit  =  50;   // TakeProfit in points
input uint     InpDistance    =  50;   // Pending orders distance (points)
input uint     InpDistanceSL  =  50;   // StopLimit orders distance (points)
input uint     InpSlippage    =  0;    // Slippage in points
input double   InpWithdrawal  =  10;   // Withdrawal funds (in tester)
//--- global variables
CEngine        engine;
CTrade         trade;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ulong          magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Check account type
   if(!engine.IsHedge())
     {
      Alert(TextByLanguage("Ошибка. Счёт должен быть хеджевым","Error. Account must be hedge"));
      return INIT_FAILED;
     }
//--- set global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      butt_data[i].name=prefix+EnumToString((ENUM_BUTTONS)i);
      butt_data[i].text=EnumToButtText((ENUM_BUTTONS)i);
     }
   lot=NormalizeLot(Symbol(),fmax(InpLots,MinimumLots(Symbol())*2.0));
   magic_number=InpMagic;
   stoploss=InpStopLoss;
   takeprofit=InpTakeProfit;
   distance_pending=InpDistance;
   distance_stoplimit=InpDistanceSL;
   slippage=InpSlippage;
//--- create buttons
   if(!CreateButtons())
      return INIT_FAILED;
//--- set trading parameters
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- delete objects
   ObjectsDeleteAll(0,prefix);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
   int total=ObjectsTotal(0);
   for(int i=0;i<total;i++)
     {
      string obj_name=ObjectName(0,i);
      if(StringFind(obj_name,prefix+"BUTT_")<0)
         continue;
      PressButtonEvents(obj_name);
     }
   if(engine.TradeEventCode()!=TRADE_EVENT_FLAG_NO_EVENT)
     {

      Print(DFUN,EnumToString((ENUM_TRADE_EVENT_FLAGS)engine.TradeEventCode()));
     }
   engine.TradeEventCode();
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   if(MQLInfoInteger(MQL_TESTER))
      return;
   if(id==CHARTEVENT_OBJECT_CLICK && StringFind(sparam,"BUTT_")>0)
     {
      PressButtonEvents(sparam);
     }
  }
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(void)
  {
   int h=18,w=84,offset=10;
   int cx=offset,cy=offset+(h+1)*(TOTAL_BUTT/2)+h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-2) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-2 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
//| Create the button                                                |
//+------------------------------------------------------------------+
bool ButtonCreate(const string name,const int x,const int y,const int w,const int h,const string text,const color clr,const string font="Calibri",const int font_size=8)
  {
   if(ObjectFind(0,name)<0)
     {
      if(!ObjectCreate(0,name,OBJ_BUTTON,0,0,0))
        {
         Print(DFUN,TextByLanguage("не удалось создать кнопку! Код ошибки=","Could not create button! Error code="),GetLastError());
         return false;
        }
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_ANCHOR,ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,font_size);
      ObjectSetString(0,name,OBJPROP_FONT,font);
      ObjectSetString(0,name,OBJPROP_TEXT,text);
      ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
      ObjectSetString(0,name,OBJPROP_TOOLTIP,"\n");
      ObjectSetInteger(0,name,OBJPROP_BORDER_COLOR,clrGray);
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
//| Return the button status                                         |
//+------------------------------------------------------------------+
bool ButtonState(const string name)
  {
   return (bool)ObjectGetInteger(0,name,OBJPROP_STATE);
  }
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
  }
//+------------------------------------------------------------------+
//| Transform enumeration into the button text                       |
//+------------------------------------------------------------------+
string EnumToButtText(const ENUM_BUTTONS member)
  {
   string txt=StringSubstr(EnumToString(member),5);
   StringToLower(txt);
   StringReplace(txt,"buy","Buy");
   StringReplace(txt,"sell","Sell");
   StringReplace(txt,"_limit"," Limit");
   StringReplace(txt,"_stop"," Stop");
   StringReplace(txt,"close_","Close ");
   StringReplace(txt,"2"," 1/2");
   StringReplace(txt,"_by_"," by ");
   StringReplace(txt,"profit_","Profit ");
   return txt;
  }
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   //--- Convert the button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY,0,takeprofit);
         //--- Open Buy position
         trade.Buy(NormalizeLot(Symbol(),lot),Symbol(),0,sl,tp);
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Set BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_LIMIT,price_set,takeprofit);
         //--- Set BuyLimit order
         trade.BuyLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set,takeprofit);
         //--- Set BuyStop order
         trade.BuyStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Get the correct BuyStop price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_STOP,distance_pending);
         //--- Calculate BuyLimit order price relative to BuyStop placement level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_BUY_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get correct StopLoss and TakeProfit prices relative to order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_BUY_STOP,price_set_limit,takeprofit);
         //--- Set BuyStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_BUY_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         trade.Sell(lot,Symbol(),0,sl,tp);
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,takeprofit);
         //--- Set SellLimit order
         trade.SellLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Get the correct order placement price relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set,takeprofit);
         //--- Set SellStop order
         trade.SellStop(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Get the correct SellStop order price relative to StopLevel
         double price_set_stop=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_STOP,distance_pending);
         //--- Calculate SellLimit order price relative to SellStop level considering StopLevel
         double price_set_limit=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_stoplimit,price_set_stop);
         //--- Get the correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_STOP,price_set_limit,takeprofit);
         //--- Set SellStopLimit order
         trade.OrderOpen(Symbol(),ORDER_TYPE_SELL_STOP_LIMIT,lot,price_set_limit,price_set_stop,sl,tp);
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Buy position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY2 button is pressed: Close the half of the Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Buy position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.VolumeCurrent()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY_BY_SELL button is pressed: Close Buy with the maximum profit by the opposite Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         if(index_buy>WRONG_VALUE && index_sell>WRONG_VALUE)
           {
            //--- Select the Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            //--- Select the Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            if(position_buy!=NULL && position_sell!=NULL)
              {
               //--- Close the Buy position by the opposite Sell one
               trade.PositionCloseBy(position_buy.Ticket(),position_sell.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL button is pressed: Close Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Get the Sell position ticket and close the position by the ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL2 button is pressed: Close the half of the Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- Calculate the closed volume and close the half of the Sell position by the ticket
               trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.VolumeCurrent()/2.0));
              }
           }
        }
      //--- If the BUTT_CLOSE_SELL_BY_BUY button is pressed: Close Sell with the maximum profit by the opposite Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list_sell=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list
         list_sell=CSelect::ByOrderProperty(list_sell,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
         //--- Get the list of all open positions
         CArrayObj* list_buy=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list
         list_buy=CSelect::ByOrderProperty(list_buy,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
         if(index_sell>WRONG_VALUE && index_buy>WRONG_VALUE)
           {
            //--- Select the Sell position with the maximum profit
            COrder* position_sell=list_sell.At(index_sell);
            //--- Select the Buy position with the maximum profit
            COrder* position_buy=list_buy.At(index_buy);
            if(position_sell!=NULL && position_buy!=NULL)
              {
               //--- Close the Sell position by the opposite Buy one
               trade.PositionCloseBy(position_sell.Ticket(),position_buy.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_ALL is pressed: Close all positions starting with the one with the least profit
      else if(button==EnumToString(BUTT_CLOSE_ALL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         if(list!=NULL)
           {
            //--- Sort the list by profit considering commission and swap
            list.Sort(SORT_BY_ORDER_PROFIT_FULL);
            int total=list.Total();
            //--- In the loop from the position with the least profit
            for(int i=0;i<total;i++)
              {
               COrder* position=list.At(i);
               if(position==NULL)
                  continue;
               //--- close each position by its ticket
               trade.PositionClose(position.Ticket());
              }
           }
        }
      //--- If the BUTT_PROFIT_WITHDRAWAL button is pressed: Withdraw funds from the account
      if(button==EnumToString(BUTT_PROFIT_WITHDRAWAL))
        {
         //--- If the program is launched in the tester
         if(MQLInfoInteger(MQL_TESTER))
           {
            //--- Emulate funds withdrawal
            TesterWithdrawal(withdrawal);
           }
        }
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button and redraw the chart
      ButtonState(button_name,false);
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

The code looks quite lengthy... However, this is only the beginning as everything is to be simplified for end-users. Most actions we conduct here is to be hidden in the library code, while the more user-friendly mechanism of interacting with the library is to be introduced.

Let's launch the EA in the tester and try the buttons:

![](https://c.mql5.com/2/35/2019-03-14_17-20-07.gif)

All is activated correctly, and the journal receives messages about occurring events.

Currently, the very last event is always fixed. In other words, if we close multiple positions simultaneously, only the last position out of the multiple closed ones will find itself in the event. Mass-closure can be tracked by the number of new deals or orders in history. It is possible then to get the list of all newly closed positions by their number and define their entire set. Let's develop a separate event collection class for that. It will allow us to have constant access to all occurred events in the program.

Currently, all event messages in the tester journal are displayed in the CEngine::WorkWithHedgeCollections() method of the library base object, and we need the custom program to know the event codes in order to "understand" what happened on the account. This will allow us to form the program's respond logic depending on an event. To test the ability to achieve that, we will create two methods in the base object of the library. One method is to store the code of the last event, while another one is to decode this code consisting of a set of event flags.

In the next article, we will create a full-fledged class for working with account events.

In the CEngine class body, define the method decoding an event code and setting an account's trading event code, the method checking the presence of an event flag in the event code, the method for receiving the last trading event from the calling program, as well as the method resetting the value of the last trading event:

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine : public CObject
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CArrayObj            m_list_counters;                 // List of timer counters
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   int                  m_trade_event_code;              // Account trading event status code
   ENUM_TRADE_EVENT     m_acc_trade_event;               // Account trading event
//--- Decode the event code and set the trading event on the account
   void                 SetTradeEvent(void);
//--- Return the counter index by id
   int                  CounterIndex(const int id) const;
//--- Return the (1) first launch flag, (2) presence of the flag in the trading event
   bool                 IsFirstStart(void);
   bool                 IsTradeEventFlag(const int event_code)    const { return (this.m_trade_event_code&event_code)==event_code;  }
//--- Working with (1) hedging, (2) netting collections
   void                 WorkWithHedgeCollections(void);
   void                 WorkWithNettoCollections(void);
//--- Return the last (1) market pending order, (2) market order, (3) last position, (4) position by ticket
   COrder*              GetLastMarketPending(void);
   COrder*              GetLastMarketOrder(void);
   COrder*              GetLastPosition(void);
   COrder*              GetPosition(const ulong ticket);
//--- Return the last (1) removed pending order, (2) historical market order, (3) historical market order by its ticket
   COrder*              GetLastHistoryPending(void);
   COrder*              GetLastHistoryOrder(void);
   COrder*              GetHistoryOrder(const ulong ticket);
//--- Return the (1) first and the (2) last historical market orders from the list of all position orders, (3) the last deal
   COrder*              GetFirstOrderPosition(const ulong position_id);
   COrder*              GetLastOrderPosition(const ulong position_id);
   COrder*              GetLastDeal(void);
public:
   //--- Return the list of market (1) positions, (2) pending orders and (3) market orders
   CArrayObj*           GetListMarketPosition(void);
   CArrayObj*           GetListMarketPendings(void);
   CArrayObj*           GetListMarketOrders(void);
   //--- Return the list of historical (1) orders, (2) removed pending orders, (3) deals, (4) all position market orders by its id
   CArrayObj*           GetListHistoryOrders(void);
   CArrayObj*           GetListHistoryPendings(void);
   CArrayObj*           GetListHistoryDeals(void);
   CArrayObj*           GetListAllOrdersByPosID(const ulong position_id);
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;  }
//--- Return the (1) last trading event, (2) trading event code, (3) hedge account flag
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_acc_trade_event;                }
   int                  TradeEventCode(void)                      const { return this.m_trade_event_code;               }
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                       }
//--- Create the timer counter
   void                 CreateCounter(const int id,const ulong frequency,const ulong pause);
//--- Timer
   void                 OnTimer(void);
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

Beyond the class body, write the **method for decoding a trading event** (write all clarifications directly into the code):

```
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEngine::SetTradeEvent(void)
  {
//--- No trading event. Exit
   if(this.m_trade_event_code==TRADE_EVENT_FLAG_NO_EVENT)
      return;
//--- Pending order is set (check if the event code is matched since there can be only one flag here)
   if(this.m_trade_event_code==TRADE_EVENT_FLAG_ORDER_PLASED)
     {
      this.m_acc_trade_event=TRADE_EVENT_PENDING_ORDER_PLASED;
      Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
      return;
     }
//--- Pending order is removed (check if the event code is matched since there can be only one flag here)
   if(this.m_trade_event_code==TRADE_EVENT_FLAG_ORDER_REMOVED)
     {
      this.m_acc_trade_event=TRADE_EVENT_PENDING_ORDER_REMOVED;
      Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
      return;
     }
//--- Position is opened (Check for multiple flags in the event code)
   if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
      //--- If this pending order is activated by the price
      if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED))
        {
         this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_PENDING_ORDER_ACTIVATED : TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL);
         Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
         return;
        }
      this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_OPENED : TRADE_EVENT_POSITION_OPENED_PARTIAL);
      Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
      return;
     }
//--- Position is closed (Check for multiple flags in the event code)
   if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_POSITION_CLOSED))
     {
      //--- if the position is closed by StopLoss
      if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_SL))
        {
         //--- check the partial closing flag and set the "Position closed by StopLoss" or "Position closed by StopLoss partially" trading event
         this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_SL : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL);
         Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
         return;
        }
      //--- if the position is closed by TakeProfit
      else if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_TP))
        {
         //--- check the partial closing flag and set the "Position closed by TakeProfit" or "Position closed by TakeProfit partially" trading event
         this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
         return;
        }
      //--- if the position is closed by an opposite one
      else if(this.IsTradeEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closing flag and set the "Position closed by opposite one" or "Position closed by opposite one partially" event
         this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
         return;
        }
      //--- If the position is closed
      else
        {
         //--- check the partial closing flag and set the "Position closed" or "Position closed partially" event
         this.m_acc_trade_event=(!this.IsTradeEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED : TRADE_EVENT_POSITION_CLOSED_PARTIAL);
         Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
         return;
        }
     }
//--- Balance operation on the account (clarify the event by the deal type)
   if(this.m_trade_event_code==TRADE_EVENT_FLAG_ACCOUNT_BALANCE)
     {
      //--- Initialize the trading event
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      //--- Take the last deal
      COrder* deal=this.GetLastDeal();
      if(deal!=NULL)
        {
         ENUM_DEAL_TYPE deal_type=(ENUM_DEAL_TYPE)deal.GetProperty(ORDER_PROP_TYPE);
         //--- if the deal is balance operation
         if(deal_type==DEAL_TYPE_BALANCE)
           {
           //--- check the deal profit and set the event (funds deposit or withdrawal)
            this.m_acc_trade_event=(deal.Profit()>0 ? TRADE_EVENT_ACCOUNT_BALANCE_REFILL : TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL);
           }
         //--- Remaining balance operation types match the ENUM_DEAL_TYPE enumeration starting with DEAL_TYPE_CREDIT
         else if(deal_type>DEAL_TYPE_BALANCE)
           {
           //--- set the event
            this.m_acc_trade_event=(ENUM_TRADE_EVENT)deal_type;
           }
        }
      Print(DFUN,"Code=",this.m_trade_event_code,", m_acc_trade_event=",EnumToString(m_acc_trade_event));
      return;
     }
  }
//+------------------------------------------------------------------+
```

Add the call of the trading event encryption method and remove the display of event descriptions in the journal at the very end of the WorkWithHedgeCollections() method checking and creating a trading event code:

```
//+------------------------------------------------------------------+
//| Check trading events (hedging)                                   |
//+------------------------------------------------------------------+
void CEngine::WorkWithHedgeCollections(void)
  {
//--- Initialize trading event code and flags
   this.m_trade_event_code=TRADE_EVENT_FLAG_NO_EVENT;
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- Actions during the first lanch
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the market status and account history changes
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();
//---
#ifdef __MQL4__

#else // MQL5
//--- If an event relates only to market orders and positions
   if(this.m_is_market_trade_event && !this.m_is_history_trade_event)
     {
      //--- If the number of pending orders increased
      if(this.m_market.NewPendingOrders()>0)
        {
         //--- Add the pending order placement flag
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_PLASED;
        }
      //--- If the number of market orders increased
      if(this.m_market.NewMarketOrders()>0)
        {
         //--- do not add the event flag
         //--- ...
        }
     }

//--- If an event relates only to historical orders and deals
   else if(this.m_is_history_trade_event && !this.m_is_market_trade_event)
     {
      //--- If a new deal appears
      if(this.m_history.NewDeals()>0)
        {
         //--- Add the flag of an account balance event
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ACCOUNT_BALANCE;
        }
     }

//--- If events are related to market and historical orders and positions
   else if(this.m_is_market_trade_event && this.m_is_history_trade_event)
     {
      //--- If the number of pending orders decreased and no new deals appeared
      if(this.m_market.NewPendingOrders()<0 && this.m_history.NewDeals()==0)
        {
         //--- Add the flag of removing a pending order
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_REMOVED;
        }

      //--- If there is a new deal and a new historical order
      if(this.m_history.NewDeals()>0 && this.m_history.NewOrders()>0)
        {
         //--- Take the last deal
         COrder* deal=this.GetLastDeal();
         if(deal!=NULL)
           {
            //--- In case of a market entry deal
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_IN)
              {
               //--- Add the position opening flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_OPENED;
               //--- If the number of pending orders decreased
               if(this.m_market.NewPendingOrders()<0)
                 {
                  //--- Add the flag of a pending order activation
                  this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
                 }
               //--- Take the order's ticket
               ulong order_ticket=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(order_ticket);
               if(order!=NULL)
                 {
                  //--- If the current order volume exceeds zero
                  if(order.VolumeCurrent()>0)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                    }
                 }
              }

            //--- In case of a market exit deal
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT)
              {
               //--- Add the position closing flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_CLOSED;
               //--- Take the deal's order ticket
               ulong order_ticket=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(order_ticket);
               if(order!=NULL)
                 {
                  //--- If the deal position is still present on the market
                  COrder* pos=this.GetPosition(deal.PositionID());
                  if(pos!=NULL)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                    }
                  //--- Otherwise, if the position is closed in full
                  else
                    {
                     //--- If the order has the flag of closing by StopLoss
                     if(order.IsCloseByStopLoss())
                       {
                        //--- Add the flag of closing by StopLoss
                        this.m_trade_event_code+=TRADE_EVENT_FLAG_SL;
                       }
                     //--- If the order has the flag of closing by TakeProfit
                     if(order.IsCloseByTakeProfit())
                       {
                        //--- Add the flag of closing by TakeProfit
                        this.m_trade_event_code+=TRADE_EVENT_FLAG_TP;
                       }
                    }
                 }
              }

            //--- If closed by the opposite one
            if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT_BY)
              {
               //--- Add the position closing flag
               this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_CLOSED;
               //--- Add the flag of closing by the opposite one
               this.m_trade_event_code+=TRADE_EVENT_FLAG_BY_POS;
               //--- Take the deal order
               ulong ticket_from=deal.GetProperty(ORDER_PROP_DEAL_ORDER);
               COrder* order=this.GetHistoryOrder(ticket_from);
               if(order!=NULL)
                 {
                  //--- If the order position is still present on the market
                  COrder* pos=this.GetPosition(order.PositionID());
                  if(pos!=NULL)
                    {
                     //--- Add the partial execution flag
                     this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
                    }
                 }
              }
           //--- end of the last deal handling block
           }
        }
     }
#endif
   this.SetTradeEvent();
  }
//+------------------------------------------------------------------+
```

Thus, after creating the event code in the WorkWithHedgeCollections() method, we call the event decoding method. What keeps us from decoding it immediately? The point is that our current decoding method is temporary. It is needed to check the decoding process. A full-fledged trading event class is to be created in the coming articles.

The **SetTradeEvent()** method defines the trading event, writes its value to the **m\_acc\_trade\_event** class member variable, while the **LastTradeEvent()** and **ResetLastTradeEvent()** methods allow reading the value of the variable as the last trading event on the account and reset it (similar to GetLastError()) in the calling program.

In the OnTick() handler of the **TestDoEasyPart04.mqh** test EA, add the strings for reading the last trading event and displaying it as in the chart comment:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
   if(MQLInfoInteger(MQL_TESTER))
      engine.OnTimer();
   int total=ObjectsTotal(0);
   for(int i=0;i<total;i++)
     {
      string obj_name=ObjectName(0,i);
      if(StringFind(obj_name,prefix+"BUTT_")<0)
         continue;
      PressButtonEvents(obj_name);
     }
   if(engine.LastTradeEvent()!=last_event)
     {
      Comment("\nLast trade event: ",EnumToString(engine.LastTradeEvent()));
      last_event=engine.LastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

Now, if you run this EA in the tester and click the buttons, the ongoing trading events from the CEngine::SetTradeEvent() method are displayed in the journal, while the chart comment displays the description of the last event occurred on the account received by the EA from the library using the engine.LastTradeEvent() method:

![](https://c.mql5.com/2/35/Tester2.gif)

### What's next?

In the next article, we will develop classes of event objects and the collection of event objects similar to collections of orders and positions and teach the basic library object to send events to the program.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/5724#node00)

**Previous articles within the series:**

[Part 1](https://www.mql5.com/en/articles/5654)

[Part 2](https://www.mql5.com/en/articles/5669)

[Part 3](https://www.mql5.com/en/articles/5687)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5724](https://www.mql5.com/ru/articles/5724)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5724.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/5724/mql5.zip "Download MQL5.zip")(44.67 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tables in the MVC Paradigm in MQL5: Customizable and sortable table columns](https://www.mql5.com/en/articles/19979)
- [How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)
- [Tables in the MVC Paradigm in MQL5: Integrating the Model Component into the View Component](https://www.mql5.com/en/articles/19288)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Resizable elements](https://www.mql5.com/en/articles/18941)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)
- [The View and Controller components for tables in the MQL5 MVC paradigm: Simple controls](https://www.mql5.com/en/articles/18221)
- [The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/314141)**
(29)


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
18 Dec 2023 at 08:35

**Artyom Trishkin [#](https://www.mql5.com/ru/forum/307247/page2#comment_51189799):**

In the test Expert Advisor, in the button press handler

(and this is just an example of how to process events and get data).

There are code blocks there that are responsible for closing positions. You can see how it is implemented there. For example, a code block for closing a purchase on the current symbol with maximum profit:

About closed positions - I did it a long time ago, now I can't tell you how to get what you need at a glance. I'll take a look later and write - I'm very busy at the moment.

Thanks, but there is no need to catch the event.

![theonementor](https://c.mql5.com/avatar/2024/6/666AE7F9-6516.png)

**[theonementor](https://www.mql5.com/en/users/theonementor)**
\|
5 Jul 2024 at 20:53

ORDER\_STATUS\_MARKET\_ORDER gives error : undeclared identifier. looks like things changed in recent versions for MQL 5, it shows up both in COrder::OrderMagicNumber and CMarketOrder Constructor!


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
6 Jul 2024 at 06:17

**theonementor [#](https://www.mql5.com/en/forum/314141#comment_53896452) :**

ORDER\_STATUS\_MARKET\_ORDER gives error : undeclared identifier. looks like things changed in recent versions for MQL 5, it shows up both in COrder::OrderMagicNumber and CMarketOrder Constructor!

I downloaded the MQL5.zip archive file attached to the article - each file individually and all together (when compiling Engine.mqh or TestDoEasyPart04.mq5) are compiled without errors.

What exactly are you doing that you are getting a compilation error?

![theonementor](https://c.mql5.com/avatar/2024/6/666AE7F9-6516.png)

**[theonementor](https://www.mql5.com/en/users/theonementor)**
\|
6 Jul 2024 at 08:50

**Artyom Trishkin [#](https://www.mql5.com/en/forum/314141#comment_53898681):**

I downloaded the MQL5.zip archive file attached to the article - each file individually and all together (when compiling Engine.mqh or TestDoEasyPart04.mq5) are compiled without errors.

What exactly are you doing that you are getting a compilation error?

Figured it out, there was one entry missing from the define enum. somehow It was missing (although I copy paste the code from the tutorial into the editor)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
6 Jul 2024 at 18:18

**theonementor [#](https://www.mql5.com/en/forum/314141#comment_53899136) :**

Figured it out, there was one entry missing from the define enum. somehow It was missing (although I copy paste the code from the tutorial into the editor)

The codes in the article do not always match the codes in the attached files. Sometimes I may miss something during the description, and sometimes something is added after writing the article. The article is not a step-by-step guide in the “read-copy-use” style, but only a detailed explanation in the form of training material. And small errors and omissions make you think, and that’s good)

![Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://c.mql5.com/2/35/Select_Symbols_Utility_MQL5__2.png)[Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)

In this article, we will continue expanding the functionality of the utility. This time, we will add the ability to display data that simplifies our trading. In particular, we are going to add High and Low prices of the previous day, round levels, High and Low prices of the year, session start time, etc.

![How to visualize multicurrency trading history based on HTML and CSV reports](https://c.mql5.com/2/35/mql5-article-html-csv.png)[How to visualize multicurrency trading history based on HTML and CSV reports](https://www.mql5.com/en/articles/5913)

Since its introduction, MetaTrader 5 provides multicurrency testing options. This possibility is often used by traders. However the function is not universal. The article presents several programs for drawing graphical objects on charts based on HTML and CSV trading history reports. Multicurrency trading can be analyzed in parallel, in several sub-windows, as well as in one window using the dynamic switching command.

![Library for easy and quick development of MetaTrader programs (part V): Classes and collection of trading events, sending events to the program](https://c.mql5.com/2/35/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part V): Classes and collection of trading events, sending events to the program](https://www.mql5.com/en/articles/6211)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the fourth part, we tested tracking trading events on the account. In this article, we will develop trading event classes and place them to the event collections. From there, they will be sent to the base object of the Engine library and the control program chart.

![Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://c.mql5.com/2/35/Logo__3.png)[Studying candlestick analysis techniques (part IV): Updates and additions to Pattern Analyzer](https://www.mql5.com/en/articles/6301)

The article presents a new version of the Pattern Analyzer application. This version provides bug fixes and new features, as well as the revised user interface. Comments and suggestions from previous article were taken into account when developing the new version. The resulting application is described in this article.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=oxttpuzjkwdecxantglxwdpmztutgbgl&ssn=1769186540785024985&ssn_dr=0&ssn_sr=0&fv_date=1769186540&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5724&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20IV)%3A%20Trading%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918654052827415&fz_uniq=5070497321805879019&sv=2552)

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