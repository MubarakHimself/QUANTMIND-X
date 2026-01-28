---
title: Library for easy and quick development of MetaTrader programs (part VII): StopLimit order activation events, preparing the functionality for order and position modification events
url: https://www.mql5.com/en/articles/6482
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:41:46.171318
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=zxclfibvgstehvjdnudxlittyxrsgugx&ssn=1769186504641307706&ssn_dr=0&ssn_sr=0&fv_date=1769186504&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6482&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20VII)%3A%20StopLimit%20order%20activation%20events%2C%20preparing%20the%20functionality%20for%20order%20and%20position%20modification%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918650447175898&fz_uniq=5070489513555334856&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/6482#node01)
- [Implementation](https://www.mql5.com/en/articles/6482#node02)
- [Test](https://www.mql5.com/en/articles/6482#node03)
- [What's next?](https://www.mql5.com/en/articles/6482#node04)


### Concept

In the previous parts devoted to the cross-platform library for [MetaTrader \\
5](https://www.metaquotes.net/en/metatrader5 "https://www.metaquotes.net/en/metatrader5") and MetaTrader 4, we developed tools for creating user-case functions enabling fast access from programs to any data on any orders and
positions on hedging and netting accounts. These are the functions for tracking events occurring to orders and positions — placing,
removing and activating pending orders, as well as opening and closing positions.

However, the functionality for tracking the activation of the already placed StopLimit orders and modification of market orders and
positions is not implemented yet.

In this article, we will implement tracking StopLimit order activation event that leads to placing a Limit order.

The library
will track such events and send the necessary messages to the program, so that the events can be used further.

### Implementation

When testing activation of StopLimit orders, I noticed that this event is not reflected in the account history, which means it cannot be simply
obtained from the account history 'as is'. Therefore, we need to track the status of existing orders up to the moment it changes (in our case,
this means changing the type of a placed order with the same ticket).

I am going to tackle the implementation of StopLimit order activation tracking from a practical perspective. Apart from developing the
required functionality, I am going to let it track other events by changes in existing orders and positions (changing the price of existing
pending orders, their StopLoss and TakeProfit levels, as well as the same levels belonging to open positions).

**The logic of the prepared functionality is to be as follows:**

We have access to the complete list of all active orders and positions on the account. The list also allows us to obtain the current status of
each of the object properties. To track the changes of monitored properties, we need to have an additional list containing the "past" state
of the properties which will initially be equal to the current one.

When comparing object properties from these two lists, a property is considered changed as soon as a difference in any of the monitored
properties is detected. In this case, a "changed" object is immediately created. Both the past and the changed property are written into it,
and the object is placed to the new list — "the list of changed objects".

This list is then to be handled in the class that tracks account events.

Of course, we can send an event immediately
after detecting changes in the object properties, but we may have a situation when several objects are changed in one tick. If we handle the
changes right away, we can handle the change of only the very last object from the pack, which is unacceptable. This means we should create the
list of all changed objects and check the size of the list in the event handler class. Each changed object from the list of changed objects is
handled in it in a loop. This prevents us from losing some of the simultaneously occurred changes in order and position properties.

When [creating the collection of market orders and positions](https://www.mql5.com/en/articles/5687#node04) in
the third part of the library description, we decided to update the list and store the current and previous hash sum calculated as a
ticket+position change time in milliseconds and volume. This allows us to constantly track the current status of orders and positions.
However, in order to track changes in order and position properties, these data are insufficient for the hash sum calculation.

- We need to consider this price to take order price changes into account
- We need to consider these prices as well to take StopLoss and TakeProfit price changes into account.

This means, we add these three prices to the hash sum but each of the prices is converted into a seven-digit ulong number by simply removing a
decimal point and increasing the number capacity by a single order (to consider six-digit quotes). For example, if the price is
1.12345, the hash sum value is 1123450.



**Let's start the implementation.**

Add enumerations with flags of possible position and order change options together with the options themselves that are to be tracked to the **Defines.mqh**
file:

```
//+------------------------------------------------------------------+
//| List of flags of possible order and position change options      |
//+------------------------------------------------------------------+
enum ENUM_CHANGE_TYPE_FLAGS
  {
   CHANGE_TYPE_FLAG_NO_CHANGE    =  0,                      // No changes
   CHANGE_TYPE_FLAG_TYPE         =  1,                      // Order type change
   CHANGE_TYPE_FLAG_PRICE        =  2,                      // Price change
   CHANGE_TYPE_FLAG_STOP         =  4,                      // StopLoss change
   CHANGE_TYPE_FLAG_TAKE         =  8,                      // TakeProfit change
   CHANGE_TYPE_FLAG_ORDER        =  16                      // Order properties change flag
  };
//+------------------------------------------------------------------+
//| Possible order and position change options                       |
//+------------------------------------------------------------------+
enum ENUM_CHANGE_TYPE
  {
   CHANGE_TYPE_NO_CHANGE,                                   // No changes
   CHANGE_TYPE_ORDER_TYPE,                                  // Order type change
   CHANGE_TYPE_ORDER_PRICE,                                 // Order price change
   CHANGE_TYPE_ORDER_PRICE_STOP_LOSS,                       // Order and StopLoss price change
   CHANGE_TYPE_ORDER_PRICE_TAKE_PROFIT,                     // Order and TakeProfit price change
   CHANGE_TYPE_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT,           // Order, StopLoss and TakeProfit price change
   CHANGE_TYPE_ORDER_STOP_LOSS_TAKE_PROFIT,                 // StopLoss and TakeProfit change
   CHANGE_TYPE_ORDER_STOP_LOSS,                             // Order's StopLoss change
   CHANGE_TYPE_ORDER_TAKE_PROFIT,                           // Order's TakeProfit change
   CHANGE_TYPE_POSITION_STOP_LOSS_TAKE_PROFIT,              // Change position's StopLoss and TakeProfit
   CHANGE_TYPE_POSITION_STOP_LOSS,                          // Change position's StopLoss
   CHANGE_TYPE_POSITION_TAKE_PROFIT,                        // Change position's TakeProfit
  };
//+------------------------------------------------------------------+
```

As for the flags of possible order and position property change options:

- order type change flag is set when activating a StopLimit
order,



- price change flag is placed when modifying a pending order
price,



- stop loss and take
profit change flags are self-explanatory,

- order flag is used to identify an order (not position)
property change




I believe, the order flag requires clarification: order type and price may unambiguously change only for pending orders (position type
change (reversal) on a netting account is not considered, since we implemented its tracking in the

[sixth part](https://www.mql5.com/en/articles/6383) of the library description), while StopLoss and TakeProfit prices
can be modified for both orders and positions. This is why we need the order flag. It allows us to accurately define an event and send the
event type to the event tracking class.



The enumeration of all possible order and position modification options
features all options we are to track in the future. In this article, we will implement tracking of only a

StopLimit order activation event (CHANGE\_TYPE\_ORDER\_TYPE).

Add eight new events (to be sent to the program during their
identification) to the ENUM\_TRADE\_EVENT enumeration of possible account trading events list:

```
//+------------------------------------------------------------------+
//| List of possible trading events on the account                   |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT
  {
   TRADE_EVENT_NO_EVENT = 0,                                // No trading event
   TRADE_EVENT_PENDING_ORDER_PLASED,                        // Pending order placed
   TRADE_EVENT_PENDING_ORDER_REMOVED,                       // Pending order removed
//--- enumeration members matching the ENUM_DEAL_TYPE enumeration members
//--- (constant order below should not be changed, no constants should be added/deleted)
   TRADE_EVENT_ACCOUNT_CREDIT = DEAL_TYPE_CREDIT,           // Accruing credit (3)
   TRADE_EVENT_ACCOUNT_CHARGE,                              // Additional charges
   TRADE_EVENT_ACCOUNT_CORRECTION,                          // Correcting entry
   TRADE_EVENT_ACCOUNT_BONUS,                               // Accruing bonuses
   TRADE_EVENT_ACCOUNT_COMISSION,                           // Additional commissions
   TRADE_EVENT_ACCOUNT_COMISSION_DAILY,                     // Commission charged at the end of a trading day
   TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY,                   // Commission charged at the end of a trading month
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY,               // Agent commission charged at the end of a trading day
   TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY,             // Agent commission charged at the end of a month
   TRADE_EVENT_ACCOUNT_INTEREST,                            // Accrued interest on free funds
   TRADE_EVENT_BUY_CANCELLED,                               // Canceled buy deal
   TRADE_EVENT_SELL_CANCELLED,                              // Canceled sell deal
   TRADE_EVENT_DIVIDENT,                                    // Accruing dividends
   TRADE_EVENT_DIVIDENT_FRANKED,                            // Accruing franked dividends
   TRADE_EVENT_TAX                        = DEAL_TAX,       // Tax
//--- constants related to the DEAL_TYPE_BALANCE deal type from the DEAL_TYPE_BALANCE enumeration
   TRADE_EVENT_ACCOUNT_BALANCE_REFILL     = DEAL_TAX+1,     // Replenishing account balance
   TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL = DEAL_TAX+2,     // Withdrawing funds from an account
//--- Remaining possible trading events
//--- (constant order below can be changed, constants can be added/deleted)
   TRADE_EVENT_PENDING_ORDER_ACTIVATED    = DEAL_TAX+3,     // Pending order activated by price
   TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL,             // Pending order partially activated by price
   TRADE_EVENT_POSITION_OPENED,                             // Position opened
   TRADE_EVENT_POSITION_OPENED_PARTIAL,                     // Position opened partially
   TRADE_EVENT_POSITION_CLOSED,                             // Position closed
   TRADE_EVENT_POSITION_CLOSED_BY_POS,                      // Position closed partially
   TRADE_EVENT_POSITION_CLOSED_BY_SL,                       // Position closed by StopLoss
   TRADE_EVENT_POSITION_CLOSED_BY_TP,                       // Position closed by TakeProfit
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET,                 // Position reversal by a new deal (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING,                // Position reversal by activating a pending order (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL,         // Position reversal by partial market order execution (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL,        // Position reversal by partial pending order activation (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET,               // Added volume to a position by a new deal (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL,       // Added volume to a position by partial activation of an order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING,              // Added volume to a position by activating a pending order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL,      // Added volume to a position by partial activation of a pending order (netting)
   TRADE_EVENT_POSITION_CLOSED_PARTIAL,                     // Position closed partially
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS,              // Position closed partially by an opposite one
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL,               // Position closed partially by StopLoss
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP,               // Position closed partially by TakeProfit
   TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER,                  // StopLimit order activation
   TRADE_EVENT_MODIFY_ORDER_PRICE,                          // Changing order price
   TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS,                // Changing order and StopLoss price
   TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT,              // Changing order and TakeProfit price
   TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT,    // Changing order, StopLoss and TakeProfit price
   TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT,          // Changing order's StopLoss and TakeProfit price
   TRADE_EVENT_MODIFY_POSITION_STOP_LOSS,                   // Changing position StopLoss
   TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT,                 // Changing position TakeProfit
  };
```

Finally, add the new constant describing StopLimit order activation to
the ENUM\_EVENT\_REASON list of event reason enumerations:

```
//+------------------------------------------------------------------+
//| Event reason                                                     |
//+------------------------------------------------------------------+
enum ENUM_EVENT_REASON
  {
   EVENT_REASON_REVERSE,                                    // Position reversal (netting)
   EVENT_REASON_REVERSE_PARTIALLY,                          // Position reversal by partial request execution (netting)
   EVENT_REASON_REVERSE_BY_PENDING,                         // Position reversal by pending order activation (netting)
   EVENT_REASON_REVERSE_BY_PENDING_PARTIALLY,               // Position reversal in case of a pending order partial execution (netting)
   //--- All constants related to a position reversal should be located in the above list
   EVENT_REASON_ACTIVATED_PENDING,                          // Pending order activation
   EVENT_REASON_ACTIVATED_PENDING_PARTIALLY,                // Pending order partial activation
   EVENT_REASON_STOPLIMIT_TRIGGERED,                        // StopLimit order activation
   EVENT_REASON_CANCEL,                                     // Cancelation
   EVENT_REASON_EXPIRED,                                    // Order expiration
   EVENT_REASON_DONE,                                       // Request executed in full
   EVENT_REASON_DONE_PARTIALLY,                             // Request executed partially
   EVENT_REASON_VOLUME_ADD,                                 // Add volume to a position (netting)
   EVENT_REASON_VOLUME_ADD_PARTIALLY,                       // Add volume to a position by a partial request execution (netting)
   EVENT_REASON_VOLUME_ADD_BY_PENDING,                      // Add volume to a position when a pending order is activated (netting)
   EVENT_REASON_VOLUME_ADD_BY_PENDING_PARTIALLY,            // Add volume to a position when a pending order is partially executed (netting)
   EVENT_REASON_DONE_SL,                                    // Closing by StopLoss
   EVENT_REASON_DONE_SL_PARTIALLY,                          // Partial closing by StopLoss
   EVENT_REASON_DONE_TP,                                    // Closing by TakeProfit
   EVENT_REASON_DONE_TP_PARTIALLY,                          // Partial closing by TakeProfit
   EVENT_REASON_DONE_BY_POS,                                // Closing by an opposite position
   EVENT_REASON_DONE_PARTIALLY_BY_POS,                      // Partial closing by an opposite position
   EVENT_REASON_DONE_BY_POS_PARTIALLY,                      // Closing an opposite position by a partial volume
   EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY,            // Partial closing of an opposite position by a partial volume
   //--- Constants related to DEAL_TYPE_BALANCE deal type from the ENUM_DEAL_TYPE enumeration
   EVENT_REASON_BALANCE_REFILL,                             // Refilling the balance
   EVENT_REASON_BALANCE_WITHDRAWAL,                         // Withdrawing funds from the account
   //--- List of constants is relevant to TRADE_EVENT_ACCOUNT_CREDIT from the ENUM_TRADE_EVENT enumeration and shifted to +13 relative to ENUM_DEAL_TYPE (EVENT_REASON_ACCOUNT_CREDIT-3)
   EVENT_REASON_ACCOUNT_CREDIT,                             // Accruing credit
   EVENT_REASON_ACCOUNT_CHARGE,                             // Additional charges
   EVENT_REASON_ACCOUNT_CORRECTION,                         // Correcting entry
   EVENT_REASON_ACCOUNT_BONUS,                              // Accruing bonuses
   EVENT_REASON_ACCOUNT_COMISSION,                          // Additional commissions
   EVENT_REASON_ACCOUNT_COMISSION_DAILY,                    // Commission charged at the end of a trading day
   EVENT_REASON_ACCOUNT_COMISSION_MONTHLY,                  // Commission charged at the end of a trading month
   EVENT_REASON_ACCOUNT_COMISSION_AGENT_DAILY,              // Agent commission charged at the end of a trading day
   EVENT_REASON_ACCOUNT_COMISSION_AGENT_MONTHLY,            // Agent commission charged at the end of a month
   EVENT_REASON_ACCOUNT_INTEREST,                           // Accruing interest on free funds
   EVENT_REASON_BUY_CANCELLED,                              // Canceled buy deal
   EVENT_REASON_SELL_CANCELLED,                             // Canceled sell deal
   EVENT_REASON_DIVIDENT,                                   // Accruing dividends
   EVENT_REASON_DIVIDENT_FRANKED,                           // Accruing franked dividends
   EVENT_REASON_TAX                                         // Tax
  };
#define REASON_EVENT_SHIFT    (EVENT_REASON_ACCOUNT_CREDIT-3)
```

We have made all the changes in the Defines.mqh file.

Since we decided to create and store the list of control orders, this list should store objects with a minimally sufficient set of properties to
define the moment one of them changes in market order and position objects.

**Let's create the control order object class.**

Create the new OrderControl.mqh class in the Collections library folder. Set
the

[CObject](https://www.mql5.com/en/docs/standardlibrary/cobject) standard library class as a basic one and include
the files necessary for the class operation:

```
//+------------------------------------------------------------------+
//|                                                 OrderControl.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "..\Objects\Orders\Order.mqh"
#include <Object.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class COrderControl : public CObject
  {
private:

public:
                     COrderControl();
                    ~COrderControl();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrderControl::COrderControl()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
COrderControl::~COrderControl()
  {
  }
//+------------------------------------------------------------------+
```

Declare all necessary variables and methods in the private section of the class right away:

```
private:
   ENUM_CHANGE_TYPE  m_changed_type;                                    // Order change type
   MqlTick           m_tick;                                            // Tick structure
   string            m_symbol;                                          // Symbol
   ulong             m_position_id;                                     // Position ID
   ulong             m_ticket;                                          // Order ticket
   long              m_magic;                                           // Magic number
   ulong             m_type_order;                                      // Order type
   ulong             m_type_order_prev;                                 // Previous order type
   double            m_price;                                           // Order price
   double            m_price_prev;                                      // Previous order price
   double            m_stop;                                            // StopLoss price
   double            m_stop_prev;                                       // Previous StopLoss price
   double            m_take;                                            // TakeProfit price
   double            m_take_prev;                                       // Previous TakeProfit price
   double            m_volume;                                          // Order volume
   datetime          m_time;                                            // Order placement time
   datetime          m_time_prev;                                       // Order previous placement time
   int               m_change_code;                                     // Order change code
//--- return the presence of the property change flag
   bool              IsPresentChangeFlag(const int change_flag)   const { return (this.m_change_code & change_flag)==change_flag;   }
//--- Return the order parameters change type
   void              CalculateChangedType(void);
```

All class member variables have clear descriptions. I should make a clarification concerning the variable
storing the tick structure: when a StopLimit order is activated, we need to save the activation time. The time should be set in
milliseconds, while

[TimeCurrent()](https://www.mql5.com/en/docs/dateandtime/timecurrent) returns the time without milliseconds. In order to
obtain the time of the last tick an order was activated on with milliseconds, we will use the

[SymbolInfoTick()](https://www.mql5.com/en/docs/marketinformation/symbolinfotick) standard function filling the [tick \\
structure](https://www.mql5.com/en/docs/constants/structures/mqltick) with data, including the tick time in milliseconds.

The order change code is composed of flags we described in the
ENUM\_CHANGE\_TYPE\_FLAGS enumeration and depends on occurred order property changes. The CalculateChangedType() private method
described below checks the flags and creates the order modification code.

In the public class section, arrange the methods for receiving and writing data on the previous and current state of the control order
properties,

the method setting the type of the occurred order property modification, the
method setting the new status of a modified order, the method returning
the type of an occurred change and the method checking the change of the
order properties, as well as setting and returning the occurred change type. The method is called from the market orders and
positions collection class for detecting the modification of active orders and positions.

```
//+------------------------------------------------------------------+
//|                                                 OrderControl.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "..\Objects\Orders\Order.mqh"
#include <Object.mqh>
//+------------------------------------------------------------------+
//| Order and position control class                                 |
//+------------------------------------------------------------------+
class COrderControl : public CObject
  {
private:
   ENUM_CHANGE_TYPE  m_changed_type;                                    // Order change type
   MqlTick           m_tick;                                            // Tick structure
   string            m_symbol;                                          // Symbol
   ulong             m_position_id;                                     // Position ID
   ulong             m_ticket;                                          // Order ticket
   long              m_magic;                                           // Magic number
   ulong             m_type_order;                                      // Order type
   ulong             m_type_order_prev;                                 // Previous order type
   double            m_price;                                           // Order price
   double            m_price_prev;                                      // Previous order price
   double            m_stop;                                            // StopLoss price
   double            m_stop_prev;                                       // Previous StopLoss price
   double            m_take;                                            // TakeProfit price
   double            m_take_prev;                                       // Previous TakeProfit price
   double            m_volume;                                          // Order volume
   datetime          m_time;                                            // Order placement time
   datetime          m_time_prev;                                       // Order previous placement time
   int               m_change_code;                                     // Order change code
//--- return the presence of the property change flag
   bool              IsPresentChangeFlag(const int change_flag)   const { return (this.m_change_code & change_flag)==change_flag;   }
//--- Calculate the order parameters change type
   void              CalculateChangedType(void);
public:
//--- Set the (1,2) current and previous type (2,3) current and previous price, (4,5) current and previous StopLoss,
//--- (6,7) current and previous TakeProfit, (8,9) current and previous placement time, (10) volume
   void              SetTypeOrder(const ulong type)                     { this.m_type_order=type;        }
   void              SetTypeOrderPrev(const ulong type)                 { this.m_type_order_prev=type;   }
   void              SetPrice(const double price)                       { this.m_price=price;            }
   void              SetPricePrev(const double price)                   { this.m_price_prev=price;       }
   void              SetStopLoss(const double stop_loss)                { this.m_stop=stop_loss;         }
   void              SetStopLossPrev(const double stop_loss)            { this.m_stop_prev=stop_loss;    }
   void              SetTakeProfit(const double take_profit)            { this.m_take=take_profit;       }
   void              SetTakeProfitPrev(const double take_profit)        { this.m_take_prev=take_profit;  }
   void              SetTime(const datetime time)                       { this.m_time=time;              }
   void              SetTimePrev(const datetime time)                   { this.m_time_prev=time;         }
   void              SetVolume(const double volume)                     { this.m_volume=volume;          }
//--- Set (1) change type, (2) new current status
   void              SetChangedType(const ENUM_CHANGE_TYPE type)        { this.m_changed_type=type;      }
   void              SetNewState(COrder* order);
//--- Check and set order parameters change flags and return the change type
   ENUM_CHANGE_TYPE  ChangeControl(COrder* compared_order);

//--- Return (1,2,3,4) position ID, ticket, magic and symbol, (5,6) current and previous type (7,8) current and previous price,
//--- (9,10) current and previous StopLoss, (11,12) current and previous TakeProfit, (13,14) current and previous placement time, (15) volume
   ulong             PositionID(void)                             const { return this.m_position_id;     }
   ulong             Ticket(void)                                 const { return this.m_ticket;          }
   long              Magic(void)                                  const { return this.m_magic;           }
   string            Symbol(void)                                 const { return this.m_symbol;          }
   ulong             TypeOrder(void)                              const { return this.m_type_order;      }
   ulong             TypeOrderPrev(void)                          const { return this.m_type_order_prev; }
   double            Price(void)                                  const { return this.m_price;           }
   double            PricePrev(void)                              const { return this.m_price_prev;      }
   double            StopLoss(void)                               const { return this.m_stop;            }
   double            StopLossPrev(void)                           const { return this.m_stop_prev;       }
   double            TakeProfit(void)                             const { return this.m_take;            }
   double            TakeProfitPrev(void)                         const { return this.m_take_prev;       }
   ulong             Time(void)                                   const { return this.m_time;            }
   ulong             TimePrev(void)                               const { return this.m_time_prev;       }
   double            Volume(void)                                 const { return this.m_volume;          }
//--- Return the change type
   ENUM_CHANGE_TYPE  GetChangeType(void)                          const { return this.m_changed_type;    }

//--- Constructor
                     COrderControl(const ulong position_id,const ulong ticket,const long magic,const string symbol) :
                                                                        m_change_code(CHANGE_TYPE_FLAG_NO_CHANGE),
                                                                        m_changed_type(CHANGE_TYPE_NO_CHANGE),
                                                                        m_position_id(position_id),m_symbol(symbol),m_ticket(ticket),m_magic(magic) {;}
  };
//+------------------------------------------------------------------+
```

The class constructor receives position
ID, ticket, magic
number and order/position symbol. In its initialization
list, reset

order change flags and the occurred
change type, as well as write order/position data obtained in the passed parameters to the corresponding class member variables
right away.

Implement declared methods outside the class body.

The private **method calculating the type of the order/position parameter**
**change:**

```
//+------------------------------------------------------------------+
//| Calculate order parameters change type                           |
//+------------------------------------------------------------------+
void COrderControl::CalculateChangedType(void)
  {
   this.m_changed_type=
     (
      //--- If the order flag is set
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_ORDER) ?
        (
         //--- If StopLimit order is activated
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TYPE)    ?  CHANGE_TYPE_ORDER_TYPE :
         //--- If an order price is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_PRICE)   ?
           (
            //--- If StopLoss and TakeProfit are modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT :
            //--- If TakeProfit modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_ORDER_PRICE_TAKE_PROFIT :
            //--- If StopLoss modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_PRICE_STOP_LOSS   :
            //--- Only order price is modified
            CHANGE_TYPE_ORDER_PRICE
           ) :
         //--- Price is not modified
         //--- If StopLoss and TakeProfit are modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_STOP_LOSS_TAKE_PROFIT :
         //--- If TakeProfit is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_ORDER_TAKE_PROFIT :
         //--- If StopLoss is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_STOP_LOSS   :
         //--- No changes
         CHANGE_TYPE_NO_CHANGE
        ) :
      //--- Position
      //--- If position's StopLoss and TakeProfit are modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_POSITION_STOP_LOSS_TAKE_PROFIT :
      //--- If position's TakeProfit is modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_POSITION_TAKE_PROFIT :
      //--- If position's StopLoss is modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_POSITION_STOP_LOSS   :
      //--- No changes
      CHANGE_TYPE_NO_CHANGE
     );
  }
//+------------------------------------------------------------------+
```

The method writes the type of the occurred change from the previously declared ENUM\_CHANGE\_TYPE enumeration to the **m\_changed\_type** class member variable depending on the presence of flags within the **m\_change\_code** variable.

All actions
related to checking flags are described in the comments to the method listing strings and should be easy to understand.

The private method checks the presence of the flag within the m\_change\_code variable

```
bool IsPresentChangeFlag(const int change_flag const { return (this.m_change_code & change_flag)==change_flag }
```

The method receives the checked flag. Its presence within m\_change\_code is checked by bit-by-bit [AND \\
operation](https://www.mql5.com/en/docs/basis/operations/bit) and the boolean result of the comparison (bit-by-bit operation between the code and the flag values) with the checked flag
value is returned.

**The method returning a new relevant status of order/position properties:**

```
//+------------------------------------------------------------------+
//| Set the new relevant status                                      |
//+------------------------------------------------------------------+
void COrderControl::SetNewState(COrder* order)
  {
   if(order==NULL || !::SymbolInfoTick(this.Symbol(),this.m_tick))
      return;
   //--- New type
   this.SetTypeOrderPrev(this.TypeOrder());
   this.SetTypeOrder(order.TypeOrder());
   //--- New price
   this.SetPricePrev(this.Price());
   this.SetPrice(order.PriceOpen());
   //--- New StopLoss
   this.SetStopLossPrev(this.StopLoss());
   this.SetStopLoss(order.StopLoss());
   //--- New TakeProfit
   this.SetTakeProfitPrev(this.TakeProfit());
   this.SetTakeProfit(order.TakeProfit());
   //--- New time
   this.SetTimePrev(this.Time());
   this.SetTime(this.m_tick.time_msc);
  }
//+------------------------------------------------------------------+
```

The pointer to the order/position, in which a change of one of the properties
occurred, is passed to the method.

As soon as a change of one of the order/position properties is detected, we need to save
the new status for further checks, the method

first saves its current property status as previous one and writes
the property value from the order passed to the method as its current status.

When saving
the time of an occurred event, use the SymbolInfoTick()
standard function to receive

tick time in milliseconds.

**The main method called from the CMarketCollection class and defining occurred changes:**

```
//+------------------------------------------------------------------+
//| Check and set order parameters change flags                      |
//+------------------------------------------------------------------+
ENUM_CHANGE_TYPE COrderControl::ChangeControl(COrder *compared_order)
  {
   this.m_change_code=CHANGE_TYPE_FLAG_NO_CHANGE;
   if(compared_order==NULL || compared_order.Ticket()!=this.m_ticket)
      return CHANGE_TYPE_NO_CHANGE;
   if(compared_order.Status()==ORDER_STATUS_MARKET_ORDER || compared_order.Status()==ORDER_STATUS_MARKET_PENDING)
      this.m_change_code+=CHANGE_TYPE_FLAG_ORDER;
   if(compared_order.TypeOrder()!=this.m_type_order)
      this.m_change_code+=CHANGE_TYPE_FLAG_TYPE;
   if(compared_order.PriceOpen()!=this.m_price)
      this.m_change_code+=CHANGE_TYPE_FLAG_PRICE;
   if(compared_order.StopLoss()!=this.m_stop)
      this.m_change_code+=CHANGE_TYPE_FLAG_STOP;
   if(compared_order.TakeProfit()!=this.m_take)
      this.m_change_code+=CHANGE_TYPE_FLAG_TAKE;
   this.CalculateChangedType();
   return this.GetChangeType();
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the checked order/position
and initializes the change code. If
an empty object of a compared order is passed or its ticket is not equal to the ticket of the current control order, return the change absence
code.

Then check all tracked properties of control and checked orders. If a mismatch is found, the necessary flag describing this change is added to
the change code.

Next, the change type is calculated by the fully formed change code in the
CalculateChangedType() method and is returned to the calling program
using the GetChangeType() method.

**The full listing of the control order class:**

```
//+------------------------------------------------------------------+
//|                                                 OrderControl.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
#include "..\Objects\Orders\Order.mqh"
#include <Object.mqh>
//+------------------------------------------------------------------+
//| Order and position control class                                 |
//+------------------------------------------------------------------+
class COrderControl : public CObject
  {
private:
   ENUM_CHANGE_TYPE  m_changed_type;                                    // Order change type
   MqlTick           m_tick;                                            // Tick structure
   string            m_symbol;                                          // Symbol
   ulong             m_position_id;                                     // Position ID
   ulong             m_ticket;                                          // Order ticket
   long              m_magic;                                           // Magic number
   ulong             m_type_order;                                      // Order type
   ulong             m_type_order_prev;                                 // Previous order type
   double            m_price;                                           // Order price
   double            m_price_prev;                                      // Previous order price
   double            m_stop;                                            // StopLoss price
   double            m_stop_prev;                                       // Previous StopLoss price
   double            m_take;                                            // TakeProfit price
   double            m_take_prev;                                       // Previous TakeProfit price
   double            m_volume;                                          // Order volume
   datetime          m_time;                                            // Order placement time
   datetime          m_time_prev;                                       // Order previous placement time
   int               m_change_code;                                     // Order change code
//--- return the presence of the property change flag
   bool              IsPresentChangeFlag(const int change_flag)   const { return (this.m_change_code & change_flag)==change_flag;   }
//--- Calculate the order parameters change type
   void              CalculateChangedType(void);
public:
//--- Set the (1,2) current and previous type (2,3) current and previous price, (4,5) current and previous StopLoss,
//--- (6,7) current and previous TakeProfit, (8,9) current and previous placement time, (10) volume
   void              SetTypeOrder(const ulong type)                     { this.m_type_order=type;        }
   void              SetTypeOrderPrev(const ulong type)                 { this.m_type_order_prev=type;   }
   void              SetPrice(const double price)                       { this.m_price=price;            }
   void              SetPricePrev(const double price)                   { this.m_price_prev=price;       }
   void              SetStopLoss(const double stop_loss)                { this.m_stop=stop_loss;         }
   void              SetStopLossPrev(const double stop_loss)            { this.m_stop_prev=stop_loss;    }
   void              SetTakeProfit(const double take_profit)            { this.m_take=take_profit;       }
   void              SetTakeProfitPrev(const double take_profit)        { this.m_take_prev=take_profit;  }
   void              SetTime(const datetime time)                       { this.m_time=time;              }
   void              SetTimePrev(const datetime time)                   { this.m_time_prev=time;         }
   void              SetVolume(const double volume)                     { this.m_volume=volume;          }
//--- Set (1) change type, (2) new current status
   void              SetChangedType(const ENUM_CHANGE_TYPE type)        { this.m_changed_type=type;      }
   void              SetNewState(COrder* order);
//--- Check and set order parameters change flags and return the change type
   ENUM_CHANGE_TYPE  ChangeControl(COrder* compared_order);

//--- Return (1,2,3,4) position ID, ticket, magic and symbol, (5,6) current and previous type (7,8) current and previous price,
//--- (9,10) current and previous StopLoss, (11,12) current and previous TakeProfit, (13,14) current and previous placement time, (15) volume
   ulong             PositionID(void)                             const { return this.m_position_id;     }
   ulong             Ticket(void)                                 const { return this.m_ticket;          }
   long              Magic(void)                                  const { return this.m_magic;           }
   string            Symbol(void)                                 const { return this.m_symbol;          }
   ulong             TypeOrder(void)                              const { return this.m_type_order;      }
   ulong             TypeOrderPrev(void)                          const { return this.m_type_order_prev; }
   double            Price(void)                                  const { return this.m_price;           }
   double            PricePrev(void)                              const { return this.m_price_prev;      }
   double            StopLoss(void)                               const { return this.m_stop;            }
   double            StopLossPrev(void)                           const { return this.m_stop_prev;       }
   double            TakeProfit(void)                             const { return this.m_take;            }
   double            TakeProfitPrev(void)                         const { return this.m_take_prev;       }
   ulong             Time(void)                                   const { return this.m_time;            }
   ulong             TimePrev(void)                               const { return this.m_time_prev;       }
   double            Volume(void)                                 const { return this.m_volume;          }
//--- Return the change type
   ENUM_CHANGE_TYPE  GetChangeType(void)                          const { return this.m_changed_type;    }

//--- Constructor
                     COrderControl(const ulong position_id,const ulong ticket,const long magic,const string symbol) :
                                                                        m_change_code(CHANGE_TYPE_FLAG_NO_CHANGE),
                                                                        m_changed_type(CHANGE_TYPE_NO_CHANGE),
                                                                        m_position_id(position_id),m_symbol(symbol),m_ticket(ticket),m_magic(magic) {;}
  };
//+------------------------------------------------------------------+
//| Check and set the order parameters change flags                  |
//+------------------------------------------------------------------+
ENUM_CHANGE_TYPE COrderControl::ChangeControl(COrder *compared_order)
  {
   this.m_change_code=CHANGE_TYPE_FLAG_NO_CHANGE;
   if(compared_order==NULL || compared_order.Ticket()!=this.m_ticket)
      return CHANGE_TYPE_NO_CHANGE;
   if(compared_order.Status()==ORDER_STATUS_MARKET_ORDER || compared_order.Status()==ORDER_STATUS_MARKET_PENDING)
      this.m_change_code+=CHANGE_TYPE_FLAG_ORDER;
   if(compared_order.TypeOrder()!=this.m_type_order)
      this.m_change_code+=CHANGE_TYPE_FLAG_TYPE;
   if(compared_order.PriceOpen()!=this.m_price)
      this.m_change_code+=CHANGE_TYPE_FLAG_PRICE;
   if(compared_order.StopLoss()!=this.m_stop)
      this.m_change_code+=CHANGE_TYPE_FLAG_STOP;
   if(compared_order.TakeProfit()!=this.m_take)
      this.m_change_code+=CHANGE_TYPE_FLAG_TAKE;
   this.CalculateChangedType();
   return this.GetChangeType();
  }
//+------------------------------------------------------------------+
//| Calculate the order parameters change type                       |
//+------------------------------------------------------------------+
void COrderControl::CalculateChangedType(void)
  {
   this.m_changed_type=
     (
      //--- If the order flag is set
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_ORDER) ?
        (
         //--- If StopLimit order is activated
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TYPE)    ?  CHANGE_TYPE_ORDER_TYPE :
         //--- If an order price is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_PRICE)   ?
           (
            //--- If StopLoss and TakeProfit are modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT :
            //--- If TakeProfit modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_ORDER_PRICE_TAKE_PROFIT :
            //--- If StopLoss modified together with the price
            this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_PRICE_STOP_LOSS   :
            //--- Only order price is modified
            CHANGE_TYPE_ORDER_PRICE
           ) :
         //--- Price is not modified
         //--- If StopLoss and TakeProfit are modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_STOP_LOSS_TAKE_PROFIT :
         //--- If TakeProfit is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_ORDER_TAKE_PROFIT :
         //--- If StopLoss is modified
         this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_ORDER_STOP_LOSS   :
         //--- No changes
         CHANGE_TYPE_NO_CHANGE
        ) :
      //--- Position
      //--- If position's StopLoss and TakeProfit are modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) && this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_POSITION_STOP_LOSS_TAKE_PROFIT :
      //--- If position's TakeProfit is modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_TAKE) ? CHANGE_TYPE_POSITION_TAKE_PROFIT :
      //--- If position's StopLoss is modified
      this.IsPresentChangeFlag(CHANGE_TYPE_FLAG_STOP) ? CHANGE_TYPE_POSITION_STOP_LOSS   :
      //--- No changes
      CHANGE_TYPE_NO_CHANGE
     );
  }
//+------------------------------------------------------------------+
//| Set the new relevant status                                      |
//+------------------------------------------------------------------+
void COrderControl::SetNewState(COrder* order)
  {
   if(order==NULL || !::SymbolInfoTick(this.Symbol(),this.m_tick))
      return;
   //--- New type
   this.SetTypeOrderPrev(this.TypeOrder());
   this.SetTypeOrder(order.TypeOrder());
   //--- New price
   this.SetPricePrev(this.Price());
   this.SetPrice(order.PriceOpen());
   //--- New StopLoss
   this.SetStopLossPrev(this.StopLoss());
   this.SetStopLoss(order.StopLoss());
   //--- New TakeProfit
   this.SetTakeProfitPrev(this.TakeProfit());
   this.SetTakeProfit(order.TakeProfit());
   //--- New time
   this.SetTimePrev(this.Time());
   this.SetTime(this.m_tick.time_msc);
  }
//+------------------------------------------------------------------+
```

**Let's improve the** **CMarketCollection** market orders and positions collection class.

We need to track property changes occurred in active orders and positions. Since we receive all market orders and positions in this class, it
would be reasonable to check their modification in it as well.

Include the control order class file. In the private class
section, declare the

list for storing control orders and positions, the
list for storing changed orders and positions, the class member variable
for storing the order change type and the variable for storing the ratio
for converting the price into the hash sum.

Also, declare the private methods:

the
method for converting order properties into a hash sum, the method adding
an order or a position to the list of pending orders and positions on the account, the
method creating and adding a control order to the list of control orders and the
method creating and adding a changed order to the list of changed orders, the
method for removing an order from the list of control orders by a ticket and a position ID, the
method returning the control order index in the list of control orders by a ticket and a position ID and the
handler of an existing order/position change event.

In the public section of the class, declare the method returning the
created list of changed orders.

```
//+------------------------------------------------------------------+
//|                                             MarketCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\MarketOrder.mqh"
#include "..\Objects\Orders\MarketPending.mqh"
#include "..\Objects\Orders\MarketPosition.mqh"
#include "OrderControl.mqh"
//+------------------------------------------------------------------+
//| Collection of market orders and positions                        |
//+------------------------------------------------------------------+
class CMarketCollection : public CListObj
  {
private:
   struct MqlDataCollection
     {
      ulong          hash_sum_acc;           // Hash sum of all orders and positions on the account
      int            total_market;           // Number of market orders on the account
      int            total_pending;          // Number of pending orders on the account
      int            total_positions;        // Number of positions on the account
      double         total_volumes;          // Total volume of orders and positions on the account
     };
   MqlDataCollection m_struct_curr_market;   // Current data on market orders and positions on the account
   MqlDataCollection m_struct_prev_market;   // Previous data on market orders and positions on the account
   CListObj          m_list_all_orders;      // List of pending orders and positions on the account
   CArrayObj         m_list_control;         // List of control orders
   CArrayObj         m_list_changed;         // List of changed orders
   COrder            m_order_instance;       // Order object for searching by property
   ENUM_CHANGE_TYPE  m_change_type;          // Order change type
   bool              m_is_trade_event;       // Trading event flag
   bool              m_is_change_volume;     // Total volume change flag
   double            m_change_volume_value;  // Total volume change value
   ulong             m_k_pow;                // Ratio for converting the price into a hash sum
   int               m_new_market_orders;    // Number of new market orders
   int               m_new_positions;        // Number of new positions
   int               m_new_pendings;         // Number of new pending orders
//--- Save the current values of the account data status as previous ones
   void              SavePrevValues(void)                                                                { this.m_struct_prev_market=this.m_struct_curr_market;                  }
//--- Convert order data into a hash sum value
   ulong             ConvertToHS(COrder* order) const;
//--- Add an order or a position to the list of pending orders and positions on an account and sets the data on market orders and positions on the account
   bool              AddToListMarket(COrder* order);
//--- (1) Create and add a control order to the list of control orders, (2) a control order to the list of changed control orders
   bool              AddToListControl(COrder* order);
   bool              AddToListChanges(COrderControl* order_control);
//--- Remove an order by a ticket or a position ID from the list of control orders
   bool              DeleteOrderFromListControl(const ulong ticket,const ulong id);
//--- Return the control order index in the list by a position ticket and ID
   int               IndexControlOrder(const ulong ticket,const ulong id);
//--- Handler of an existing order/position change event
   void              OnChangeEvent(COrder* order,const int index);
public:
//--- Return the list of (1) all pending orders and open positions, (2) modified orders and positions
   CArrayObj*        GetList(void)                                                                       { return &this.m_list_all_orders;                                       }
   CArrayObj*        GetListChanges(void)                                                                { return &this.m_list_changed;                                          }
//--- Return the list of orders and positions with an open time from begin_time to end_time
   CArrayObj*        GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the list of orders and positions by selected (1) double, (2) integer and (3) string property fitting a compared condition
   CArrayObj*        GetList(ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
   CArrayObj*        GetList(ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByOrderProperty(this.GetList(),property,value,mode);  }
//--- Return the number of (1) new market order, (2) new pending orders, (3) new positions, (4) occurred trading event flag, (5) changed volume
   int               NewMarketOrders(void)                                                         const { return this.m_new_market_orders;                                      }
   int               NewPendingOrders(void)                                                        const { return this.m_new_pendings;                                           }
   int               NewPositions(void)                                                            const { return this.m_new_positions;                                          }
   bool              IsTradeEvent(void)                                                            const { return this.m_is_trade_event;                                         }
   double            ChangedVolumeValue(void)                                                      const { return this.m_change_volume_value;                                    }
//--- Constructor
                     CMarketCollection(void);
//--- Update the list of pending orders and positions
   void              Refresh(void);
  };
//+------------------------------------------------------------------+
```

Add clearing and sorting the list of control orders and the list
of changed orders, as well as calculation of the ratio defining the hash sum
to the class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CMarketCollection::CMarketCollection(void) : m_is_trade_event(false),m_is_change_volume(false),m_change_volume_value(0)
  {
   this.m_list_all_orders.Sort(SORT_BY_ORDER_TIME_OPEN);
   this.m_list_all_orders.Clear();
   ::ZeroMemory(this.m_struct_prev_market);
   this.m_struct_prev_market.hash_sum_acc=WRONG_VALUE;
   this.m_list_all_orders.Type(COLLECTION_MARKET_ID);
   this.m_list_control.Clear();
   this.m_list_control.Sort();
   this.m_list_changed.Clear();
   this.m_list_changed.Sort();
   this.m_k_pow=(ulong)pow(10,6);
  }
//+------------------------------------------------------------------+
```

**The method of converting order properties into the number for calculating the hash sum:**

```
//+---------------------------------------------------------------------+
//| Convert the order price and its type into a number for the hash sum |
//+---------------------------------------------------------------------+
ulong CMarketCollection::ConvertToHS(COrder *order) const
  {
   if(order==NULL)
      return 0;
   ulong price=ulong(order.PriceOpen()*this.m_k_pow);
   ulong stop=ulong(order.StopLoss()*this.m_k_pow);
   ulong take=ulong(order.TakeProfit()*this.m_k_pow);
   ulong type=order.TypeOrder();
   ulong ticket=order.Ticket();
   return price+stop+take+type+ticket;
  }
//+------------------------------------------------------------------+
```

The method receives a pointer to the order whose data should
be converted into a number. Then the order's double properties are converted into a number for the hash sum by a simple

multiplication by the ratio previously calculated in the class constructor, all property
values are summed and returned as a ulong number.

The method for updating the current data on the **Refresh()**
market environment was improved in the class. We described it in the [third \\
part of the library description](https://www.mql5.com/en/articles/5687#node04).

The changes affected adding objects to the list of orders and positions. Now these same-type strings are located in the single **AddToListMarket()**
method. After declaring an order object in the list of orders and positions, the presence of the same order is checked in the list of control
orders. If such an order is absent, a control order object is created and added to the list of control orders using the

**AddToListControl()** method. If the control order is present, the **OnChangeEvent()** method for comparing the
current order properties with the control order ones is called.

All performed actions are described in string comments and highlighted in the
text of the method listing.

```
//+------------------------------------------------------------------+
//| Update the list of orders                                        |
//+------------------------------------------------------------------+
void CMarketCollection::Refresh(void)
  {
   ::ZeroMemory(this.m_struct_curr_market);
   this.m_is_trade_event=false;
   this.m_is_change_volume=false;
   this.m_new_pendings=0;
   this.m_new_positions=0;
   this.m_change_volume_value=0;
   this.m_list_all_orders.Clear();
#ifdef __MQL4__
   int total=::OrdersTotal();
   for(int i=0; i<total; i++)
     {
      if(!::OrderSelect(i,SELECT_BY_POS)) continue;
      long ticket=::OrderTicket();
      //--- Get the control order index by a position ticket and ID
      int index=this.IndexControlOrder(ticket);
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderType();
      if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL)
        {
         CMarketPosition *position=new CMarketPosition(ticket);
         if(position==NULL) continue;
         //--- Add a position object to the list of market orders and positions
         if(!this.AddToListMarket(position))
            continue;
         //--- If there is no order in the list of control orders and positions, add it
         if(index==WRONG_VALUE)
           {
            if(!this.AddToListControl(order))
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Не удалось добавить контрольный ордер ","Failed to add control order "),order.TypeDescription()," #",order.Ticket());
              }
           }
         //--- If the order is already present in the list of control orders, check it for changed properties
         if(index>WRONG_VALUE)
           {
            this.OnChangeEvent(position,index);
           }
        }
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         //--- Add a pending order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
         //--- If there is no order in the list of control orders and positions, add it
         if(index==WRONG_VALUE)
           {
            if(!this.AddToListControl(order))
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Не удалось добавить контрольный ордер ","Failed to add control order "),order.TypeDescription()," #",order.Ticket());
              }
           }
         //--- If the order is already present in the list of control orders, check it for changed properties
         if(index>WRONG_VALUE)
           {
            this.OnChangeEvent(order,index);
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
      //--- Add a position object to the list of market orders and positions
      if(!this.AddToListMarket(position))
         continue;
      //--- Get the control order index by a position ticket and ID
      int index=this.IndexControlOrder(ticket,position.PositionID());
      //--- If the order is not present in the list of control orders, add it
      if(index==WRONG_VALUE)
        {
         if(!this.AddToListControl(position))
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Не удалось добавить контрольую позицию ","Failed to add control position "),position.TypeDescription()," #",position.Ticket());
           }
        }
      //--- If the order is already present in the list of control orders, check it for changed properties
      else if(index>WRONG_VALUE)
        {
         this.OnChangeEvent(position,index);
        }
     }
//--- Orders
   int total_orders=::OrdersTotal();
   for(int i=0; i<total_orders; i++)
     {
      ulong ticket=::OrderGetTicket(i);
      if(ticket==0) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderGetInteger(ORDER_TYPE);
      //--- Market order
      if(type<ORDER_TYPE_BUY_LIMIT)
        {
         CMarketOrder *order=new CMarketOrder(ticket);
         if(order==NULL) continue;
         //--- Add a market order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
        }
      //--- Pending order
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         //--- Add a pending order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
         //--- Get the control order index by a position ticket and ID
         int index=this.IndexControlOrder(ticket,order.PositionID());
         //--- If the order is not present in the control order list, add it
         if(index==WRONG_VALUE)
           {
            if(!this.AddToListControl(order))
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Не удалось добавить контрольный ордер ","Failed to add control order "),order.TypeDescription()," #",order.Ticket());
              }
           }
         //--- If the order is already in the control order list, check it for changed properties
         else if(index>WRONG_VALUE)
           {
            this.OnChangeEvent(order,index);
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
      this.m_new_market_orders=this.m_struct_curr_market.total_market-this.m_struct_prev_market.total_market;
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

**The method adding orders and positions to the list of the collection's market orders and positions:**

```
//+--------------------------------------------------------------------------------+
//| Add an order or a position to the list of orders and positions on the account  |
//+--------------------------------------------------------------------------------+
bool CMarketCollection::AddToListMarket(COrder *order)
  {
   if(order==NULL)
      return false;
   ENUM_ORDER_STATUS status=order.Status();
   if(this.m_list_all_orders.InsertSort(order))
     {
      if(status==ORDER_STATUS_MARKET_POSITION)
        {
         this.m_struct_curr_market.hash_sum_acc+=order.GetProperty(ORDER_PROP_TIME_UPDATE_MSC)+this.ConvertToHS(order);
         this.m_struct_curr_market.total_volumes+=order.Volume();
         this.m_struct_curr_market.total_positions++;
         return true;
        }
      if(status==ORDER_STATUS_MARKET_PENDING)
        {
         this.m_struct_curr_market.hash_sum_acc+=this.ConvertToHS(order);
         this.m_struct_curr_market.total_volumes+=order.Volume();
         this.m_struct_curr_market.total_pending++;
         return true;
        }
     }
   else
     {
      ::Print(DFUN,order.TypeDescription()," #",order.Ticket()," ",TextByLanguage("не удалось добавить в список","failed to add to list"));
      delete order;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The pointer to the order added to the collection list is passed
to the method. After adding an order to the collection list,
the data of the structure storing the current state of market orders and positions for a subsequent check and defining the changes in the
number of orders and positions is changed

depending on the order status.

- If this is a position, a position
change time and a calculated value for the hash sum
are added to the general hash sum, and

the overall position number is increased.
- If this is a pending order, a calculated value for the hash
sum is added to the general hash sum, and

the overall number of pending orders is increased.

**The method for creating a control order and adding it to the list of control orders:**

```
//+------------------------------------------------------------------+
//| Create and add an order to the list of control orders            |
//+------------------------------------------------------------------+
bool CMarketCollection::AddToListControl(COrder *order)
  {
   if(order==NULL)
      return false;
   COrderControl* order_control=new COrderControl(order.PositionID(),order.Ticket(),order.Magic(),order.Symbol());
   if(order_control==NULL)
      return false;
   order_control.SetTime(order.TimeOpenMSC());
   order_control.SetTimePrev(order.TimeOpenMSC());
   order_control.SetVolume(order.Volume());
   order_control.SetTime(order.TimeOpenMSC());
   order_control.SetTypeOrder(order.TypeOrder());
   order_control.SetTypeOrderPrev(order.TypeOrder());
   order_control.SetPrice(order.PriceOpen());
   order_control.SetPricePrev(order.PriceOpen());
   order_control.SetStopLoss(order.StopLoss());
   order_control.SetStopLossPrev(order.StopLoss());
   order_control.SetTakeProfit(order.TakeProfit());
   order_control.SetTakeProfitPrev(order.TakeProfit());
   if(!this.m_list_control.Add(order_control))
     {
      delete order_control;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The pointer to a market order and position is passed to the
method.

If an invalid object is passed, return false.

A new control order is then created, so that its constructor
immediately receives a position ID, ticket, magic number and symbol of an order object passed to the method.

All data necessary for identifying order/position modifications are then filled.

If adding a new control order to the list of control orders failed, the order
is removed and 'false' is returned.

Since we always add new orders and positions to the list of control orders and positions, it may become quite bulky after a long work. Orders and
positions do not live forever, and their control copies should not be permanently stored in the list occupying memory for no reason. To
remove unnecessary control orders from the list, use the

**DeleteOrderFromListControl() method removing a control order from the list of control orders by a position ticket and ID.**

For now, the method is only declared but not implemented. The implementation will be done after preparing the entire functionality for
tracking order and position modifications.

**The method returning the control order index in the list of control orders by a position ticket and ID:**

```
//+------------------------------------------------------------------+
//| Return an order index by a ticket in the list of control orders  |
//+------------------------------------------------------------------+
int CMarketCollection::IndexControlOrder(const ulong ticket,const ulong id)
  {
   int total=this.m_list_control.Total();
   for(int i=0;i<total;i++)
     {
      COrderControl* order=this.m_list_control.At(i);
      if(order==NULL)
         continue;
      if(order.PositionID()==id && order.Ticket()==ticket)
         return i;
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The method receives order/position ticket and position ID. A
control order having a matching ticket and ID is searched for along all control orders in the loop, and its
index is returned in the list of control orders. If the order is not
found, -1 is returned.

**Event handler method for changing an existing order/position:**

```
//+------------------------------------------------------------------+
//| Handler of changing an existing order/position                   |
//+------------------------------------------------------------------+
void CMarketCollection::OnChangeEvent(COrder* order,const int index)
  {
   COrderControl* order_control=this.m_list_control.At(index);
   if(order_control!=NULL)
     {
      this.m_change_type=order_control.ChangeControl(order);
      ENUM_CHANGE_TYPE change_type=(order.Status()==ORDER_STATUS_MARKET_POSITION ? CHANGE_TYPE_ORDER_TAKE_PROFIT : CHANGE_TYPE_NO_CHANGE);
      if(this.m_change_type>change_type)
        {
         order_control.SetNewState(order);
         if(!this.AddToListChanges(order_control))
           {
            ::Print(DFUN,TextByLanguage("Не удалось добавить модифицированный ордер в список изменённых ордеров","Could not add modified order to list of modified orders"));
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the checked order and the
index of the appropriate control order in the list of control orders.

Get
the control order from the listby its index and check
for the changes in the control order properties corresponding to the properties of the control order checked using the ChangeControl()
method. The method receives the pointer to the control order.
If the difference is found, the method returns the change type that is written to the

**m\_change\_type** class member variable.

Next, check the status of the
checked order and set the value, above which the change is considered to have occurred. For a position,
this value should exceed the CHANGE\_TYPE\_ORDER\_TAKE\_PROFIT constant from the ENUM\_CHANGE\_TYPE enumeration since all values
equal or below this constant are related only to a pending order. For a

pending order, the value should exceed the CHANGE\_TYPE\_NO\_CHANGE
constant.

If the obtained **m\_change\_type** variable exceeds the
specified one, modification is detected. In this case, the current
status of the control order is saved for subsequent check and a copy of a
control order is placed to the list of changed orders for subsequent handling of the list in the CEventsCollection class.

**The method for creating a changed control order and adding it to the list of changed orders:**

```
//+------------------------------------------------------------------+
//|Create and add a control order to the list of changed orders      |
//+------------------------------------------------------------------+
bool CMarketCollection::AddToListChanges(COrderControl* order_control)
  {
   if(order_control==NULL)
      return false;
   COrderControl* order_changed=new COrderControl(order_control.PositionID(),order_control.Ticket(),order_control.Magic(),order_control.Symbol());
   if(order_changed==NULL)
      return false;
   order_changed.SetTime(order_control.Time());
   order_changed.SetTimePrev(order_control.TimePrev());
   order_changed.SetVolume(order_control.Volume());
   order_changed.SetTypeOrder(order_control.TypeOrder());
   order_changed.SetTypeOrderPrev(order_control.TypeOrderPrev());
   order_changed.SetPrice(order_control.Price());
   order_changed.SetPricePrev(order_control.PricePrev());
   order_changed.SetStopLoss(order_control.StopLoss());
   order_changed.SetStopLossPrev(order_control.StopLossPrev());
   order_changed.SetTakeProfit(order_control.TakeProfit());
   order_changed.SetTakeProfitPrev(order_control.TakeProfitPrev());
   order_changed.SetChangedType(order_control.GetChangeType());
   if(!this.m_list_changed.Add(order_changed))
     {
      delete order_changed;
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives the pointer to the modified control order.
The copy of the order should be placed to the list of changed control orders and positions.

Next, a new control order is created. It immediately
receives the position ID, ticket, magic number and symbol matching the ones of the changed control order.

After that, the properties of the changed control order are simply copied to
the properties of the newly created one element by element.

Finally, place
the newly created copy of a changed control order to the list of changed orders.

If
the newly created order could not be placed to the list, the
newly created order object is removed and

false is returned.

**We have finished implementing changes to the CMarketCollection class. Now let's move on to the CEventsCollection class.**

The **CEventsCollection** event collection class should feature handling events, in which the list of changed orders created in
the market order and position collection class is not empty. This means that it contains changed orders and positions that should be handled
(create a new event and send the appropriate message to the calling program).

Let's add the definition of the two methods to the private section of the class in addition to the already existing method: the new overloaded
method of creating a new event and the method handling the change of an
existing order/position, while the Refresh() method receives the ability to pass
the list of changed orders to the method:

```
//+------------------------------------------------------------------+
//|                                             EventsCollection.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "ListObj.mqh"
#include "..\Services\Select.mqh"
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\EventBalanceOperation.mqh"
#include "..\Objects\Events\EventOrderPlaced.mqh"
#include "..\Objects\Events\EventOrderRemoved.mqh"
#include "..\Objects\Events\EventPositionOpen.mqh"
#include "..\Objects\Events\EventPositionClose.mqh"
//+------------------------------------------------------------------+
//| Collection of account events                                     |
//+------------------------------------------------------------------+
class CEventsCollection : public CListObj
  {
private:
   CListObj          m_list_events;                   // List of events
   bool              m_is_hedge;                      // Hedge account flag
   long              m_chart_id;                      // Control program chart ID
   int               m_trade_event_code;              // Trading event code
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property

//--- Create a trading event depending on the (1) order status and (2) change type
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market);
   void              CreateNewEvent(COrderControl* order);
//--- Create an event for a (1) hedging account, (2) netting account
   void              NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
   void              NewDealEventNetto(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
//--- Select and return the list of market pending orders
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
//--- Select from the list and return the list of historical (1) removed pending orders, (2) deals, (3) all closing orders
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListCloseByOrders(CArrayObj* list);
//--- Return the list of (1) all position orders by its ID, (2) all deal positions by its ID,
//--- (3) all market entry deals by position ID, (4) all market exit deals by position ID,
//--- (5) all position reversal deals by position ID
   CArrayObj*        GetListAllOrdersByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsOutByPosID(CArrayObj* list,const ulong position_id);
   CArrayObj*        GetListAllDealsInOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the total volume of all deals (1) IN, (2) OUT of the position by its ID
   double            SummaryVolumeDealsInByPosID(CArrayObj* list,const ulong position_id);
   double            SummaryVolumeDealsOutByPosID(CArrayObj* list,const ulong position_id);
//--- Return the (1) first, (2) last and (3) closing order from the list of all position orders,
//--- (4) an order by ticket, (5) market position by ID,
//--- (6) the last and (7) penultimate InOut deal by position ID
   COrder*           GetFirstOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetLastOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetCloseByOrderFromList(CArrayObj* list,const ulong position_id);
   COrder*           GetHistoryOrderByTicket(CArrayObj* list,const ulong order_ticket);
   COrder*           GetPositionByID(CArrayObj* list,const ulong position_id);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);
//--- Handler of an existing order/position change
   void              OnChangeEvent(CArrayObj* list_changes,CArrayObj* list_history,CArrayObj* list_market,const int index);
public:
//--- Select events from the collection with time within the range from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                       { return &this.m_list_events;                                           }
//--- Return the list by selected (1) integer, (2) real and (3) string properties meeting the compared criterion
   CArrayObj        *GetList(ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode=EQUAL)  { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
   CArrayObj        *GetList(ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode=EQUAL) { return CSelect::ByEventProperty(this.GetList(),property,value,mode);  }
//--- Update the list of events
   void              Refresh(CArrayObj* list_history,
                             CArrayObj* list_market,
                             CArrayObj* list_changes,
                             const bool is_history_event,
                             const bool is_market_event,
                             const int  new_history_orders,
                             const int  new_market_pendings,
                             const int  new_market_positions,
                             const int  new_deals);
//--- Set the control program chart ID
   void              SetChartID(const long id)        { this.m_chart_id=id;         }
//--- Return the last trading event on the account
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)    const { return this.m_trade_event;  }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)        { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
  };
//+------------------------------------------------------------------+
```

Let's implement the new methods outside the class body.

**The overloaded method for creating an order/position modification event:**

```
//+------------------------------------------------------------------+
//| Create a trading event depending on the order change type        |
//+------------------------------------------------------------------+
void CEventsCollection::CreateNewEvent(COrderControl* order)
  {
   CEvent* event=NULL;
//--- Pending StopLimit order placed
   if(order.GetChangeType()==CHANGE_TYPE_ORDER_TYPE)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_PLASED;
      event=new CEventOrderPlased(this.m_trade_event_code,order.Ticket());
     }
//---
   if(event!=NULL)
     {
      event.SetProperty(EVENT_PROP_TIME_EVENT,order.Time());                        // Event time
      event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_STOPLIMIT_TRIGGERED);  // Event reason (from the ENUM_EVENT_REASON enumeration)
      event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrderPrev());          // Type of the order that triggered an event
      event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Ticket of the order that triggered an event
      event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());             // Event order type
      event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());              // Event order ticket
      event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());          // Position first order type
      event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());           // Position first order ticket
      event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
      event.SetProperty(EVENT_PROP_POSITION_BY_ID,0);                               // Opposite position ID
      event.SetProperty(EVENT_PROP_MAGIC_BY_ID,0);                                  // Opposite position magic number

      event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrderPrev());      // Position order type before changing the direction
      event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());           // Position order ticket before changing direction
      event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());         // Current position order type
      event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());          // Current position order ticket

      event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order magic number
      event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimePrev());           // First position order time
      event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PricePrev());                  // Event price
      event.SetProperty(EVENT_PROP_PRICE_OPEN,order.Price());                       // Order open price
      event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.Price());                      // Order close price
      event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                      // StopLoss order price
      event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                    // TakeProfit order price
      event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());            // Requested order volume
      event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,0);                        // Executed order volume
      event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order.Volume());            // Remaining (unexecuted) order volume
      event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,0);                     // Executed position volume
      event.SetProperty(EVENT_PROP_PROFIT,0);                                       // Profit
      event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                          // Order symbol
      event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                    // Opposite position symbol
      //--- Set the control program chart ID, decode the event code and set the event type
      event.SetChartID(this.m_chart_id);
      event.SetTypeEvent();
      //--- Add the event object if it is not in the list
      if(!this.IsPresentEventInList(event))
        {
         this.m_list_events.InsertSort(event);
         //--- Send a message about the event and set the value of the last trading event
         event.SendEvent();
         this.m_trade_event=event.TradeEvent();
        }
      //--- If the event is already present in the list, remove a new event object and display a debugging message
      else
        {
         ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
         delete event;
        }
     }
  }
//+------------------------------------------------------------------+
```

I described the new event creation method in the fifth part of the library description [when \\
creating an event collection](https://www.mql5.com/en/articles/6211#node03).

This method is almost identical. The only difference is the type of the order, the
pointer to which is passed to the method.

The type of an occurred
order change is checked at the very start of the method and the
change code is set in the

**m\_trade\_event\_code** class member variable according to the change type.

Next, the
event matching the change type is created, its properties are filled according to the change type, the
event is placed to the event list and sent to the control program.

**The improved method for updating the event list:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals)
  {
//--- Exit if the lists are empty
   if(list_history==NULL || list_market==NULL)
      return;
//--- If the event is in the market environment
   if(is_market_event)
     {
      //--- if the order properties were changed
      int total_changes=list_changes.Total();
      if(total_changes>0)
        {
         for(int i=total_changes-1;i>=0;i--)
           {
            this.OnChangeEvent(list_changes,i);
           }
        }
      //--- if the number of placed pending orders increased
      if(new_market_pendings>0)
        {
         //--- Receive the list of the newly placed pending orders
         CArrayObj* list=this.GetListMarketPendings(list_market);
         if(list!=NULL)
           {
            //--- Sort the new list by order placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            //--- Take the number of orders equal to the number of newly placed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_market_pendings;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list, if this is a pending order, set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_MARKET_PENDING)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
     }
//--- If the event is in the account history
   if(is_history_event)
     {
      //--- If the number of historical orders increased
      if(new_history_orders>0)
        {
         //--- Receive the list of removed pending orders only
         CArrayObj* list=this.GetListHistoryPendings(list_history);
         if(list!=NULL)
           {
            //--- Sort the new list by order removal time
            list.Sort(SORT_BY_ORDER_TIME_CLOSE_MSC);
            //--- Take the number of orders equal to the number of newly removed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_history_orders;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list. If this is a removed pending order without a position ID,
               //--- this is an order removal - set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_HISTORY_PENDING && order.PositionID()==0)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
      //--- If the number of deals increased
      if(new_deals>0)
        {
         //--- Receive the list of deals only
         CArrayObj* list=this.GetListDeals(list_history);
         if(list!=NULL)
           {
            //--- Sort the new list by deal time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            //--- Take the number of deals equal to the number of new ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_deals;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive a deal from the list and set a trading event
               COrder* order=list.At(i);
               if(order!=NULL)
                  this.CreateNewEvent(order,list_history,list_market);
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

This method was also considered in the fifth part of the library description [when \\
creating an event collection](https://www.mql5.com/en/articles/6211#node03). The difference from that method lies in the added code
block for handling modification events in case the size of the changed
orders list is not zero. Each changed order from the list is handled in the **order change event handler method**
in a loop:

```
//+------------------------------------------------------------------+
//| The handler of an existing order/position change event           |
//+------------------------------------------------------------------+
void CEventsCollection::OnChangeEvent(CArrayObj* list_changes,const int index)
  {
   COrderControl* order_changed=list_changes.Detach(index);
   if(order_changed!=NULL)
     {
      if(order_changed.GetChangeType()==CHANGE_TYPE_ORDER_TYPE)
        {
         this.CreateNewEvent(order_changed);
        }
      delete order_changed;
     }
  }
//+------------------------------------------------------------------+
```

When handling the list of changed orders, we need to obtain a modified order from the list and remove the order object and the appropriate
pointer from the list after handling is complete to avoid handling the same event multiple times.

Fortunately, when working with the [CArrayObj \\
dynamic array of object pointers](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj), the standard library provides the [Detach()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjdetach)
method  that receives the element from the specified position and removes it from the array. In other words, we
receive the pointer to the objectstored in the array by index
and remove this pointer from the array. If the change type is
CHANGE\_TYPE\_ORDER\_TYPE (order type change — triggering a pending StopLimit order and turning it into a Limit
order),

create a new event— StopLimit order activation.
After the object is handled by the pointer obtained using the Detach() method, the pointer (which is no longer needed)
is simply

removed.

**This concludes the improvement of the CEventsCollection class.**

In order for all the changes to take effect, the list of changed orders from the market orders and position collection class should be
received and its size should be written in the library's main object — in the

**CEngine** class (the **TradeEventsControl()** method). When the Refresh() event update method of the event collection class
is called, the size of the changed orders list should be additionally checked, while the list of modified orders should be passed to the
Refresh() method of the event collection for handling:

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize the trading events code and flag
   this.m_is_market_trade_event=false;
   this.m_is_history_trade_event=false;
//--- Update the lists
   this.m_market.Refresh();
   this.m_history.Refresh();
//--- First launch actions
   if(this.IsFirstStart())
     {
      this.m_acc_trade_event=TRADE_EVENT_NO_EVENT;
      return;
     }
//--- Check the changes in the market status and account history
   this.m_is_market_trade_event=this.m_market.IsTradeEvent();
   this.m_is_history_trade_event=this.m_history.IsTradeEvent();

//--- If there is any event, send the lists, the flags and the number of new orders and deals to the event collection, and update it
   int change_total=0;
   CArrayObj* list_changes=this.m_market.GetListChanges();
   if(list_changes!=NULL)
      change_total=list_changes.Total();
   if(this.m_is_history_trade_event || this.m_is_market_trade_event || change_total>0)
     {
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),list_changes,
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewMarketOrders(),this.m_history.NewDeals());
      //--- Receive the last account trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

Since the activation of a StopLimit order leads to placing a Limit order, we will "qualify" this event as placing a pending order, while the event
reason is the activation of a StopLimit order EVENT\_REASON\_STOPLIMIT\_TRIGGERED. We have already set its constant in the
ENUM\_EVENT\_REASON enumeration of the Defines.mqh file.

**Let's improve the EventOrderPlased class to display the event program in the journal and send it to the control program:**

Simply add the EVENT\_REASON\_STOPLIMIT\_TRIGGERED event reason handling.

```
//+------------------------------------------------------------------+
//|                                             EventOrderPlased.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Event.mqh"
//+------------------------------------------------------------------+
//| Placing a pending order event                                    |
//+------------------------------------------------------------------+
class CEventOrderPlased : public CEvent
  {
public:
//--- Constructor
                     CEventOrderPlased(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_PENDING,event_code,ticket) {}
//--- Supported order properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CEventOrderPlased::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   if(property==EVENT_PROP_TYPE_DEAL_EVENT         ||
      property==EVENT_PROP_TICKET_DEAL_EVENT       ||
      property==EVENT_PROP_TYPE_ORDER_POSITION     ||
      property==EVENT_PROP_TICKET_ORDER_POSITION   ||
      property==EVENT_PROP_POSITION_ID             ||
      property==EVENT_PROP_POSITION_BY_ID          ||
      property==EVENT_PROP_TIME_ORDER_POSITION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CEventOrderPlased::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   if(property==EVENT_PROP_PRICE_CLOSE             ||
      property==EVENT_PROP_PROFIT
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventOrderPlased::PrintShort(void)
  {
   int    digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string sl=(this.PriceStopLoss()>0 ? ", sl "+::DoubleToString(this.PriceStopLoss(),digits) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+::DoubleToString(this.PriceTakeProfit(),digits) : "");
   string vol=::DoubleToString(this.VolumeOrderInitial(),DigitsLots(this.Symbol()));
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string type=this.TypeOrderFirstDescription()+" #"+(string)this.TicketOrderEvent();
   string event=TextByLanguage(" Установлен "," Placed ");
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceOpen(),digits);
   string txt=head+this.Symbol()+event+vol+" "+type+price+sl+tp+magic;
   //--- If StopLimit order is activated
   if(this.Reason()==EVENT_REASON_STOPLIMIT_TRIGGERED)
     {
      head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimeEvent())+" -\n";
      event=TextByLanguage(" Сработал "," Triggered ");
      type=
        (
         OrderTypeDescription(this.TypeOrderPosPrevious())+" #"+(string)this.TicketOrderEvent()+
         TextByLanguage(" по цене "," at price ")+DoubleToString(this.PriceEvent(),digits)+" -->\n"+
         vol+" "+OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderEvent()+
         TextByLanguage(" на цену "," on price ")+DoubleToString(this.PriceOpen(),digits)
        );
      txt=head+this.Symbol()+event+"("+TimeMSCtoString(this.TimePosition())+") "+vol+" "+type+sl+tp+magic;
     }
   ::Print(txt);
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventOrderPlased::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.TicketOrderEvent(),this.PriceOpen(),this.Symbol());
  }
//+------------------------------------------------------------------+
```

Here all is quite easy to understand. There is no point in dwelling on simple actions.

This concludes the improvement of the library for tracking a StopLimit order activation.

### Test

To test the implemented improvements, we will use the EA from the [previous \\
article](https://www.mql5.com/en/articles/6383). Simply rename the TestDoEasyPart06.mq5 EA from the \\MQL5\\Experts\\TestDoEasy\\Part06 folder to **TestDoEasyPart07.mq5** and
save it in the new \\MQL5\\Experts\\TestDoEasy\

**Part07** subfolder.

Compile the EA, launch it in the tester, place a StopLimit order and wait for its activation:

![](https://c.mql5.com/2/36/2019-04-23_13-28-23.gif)

### What's next?

The functionality implemented in the article includes the ability to quickly add the tracking of other events: modification of pending
order properties — their price, StopLoss and TakeProfit levels, as well as modification of StopLoss and TakeProfit levels of positions. We
will consider these tasks in the next article.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6482#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program.](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events.](https://www.mql5.com/en/articles/6383)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6482](https://www.mql5.com/ru/articles/6482)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6482.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6482/mql5.zip "Download MQL5.zip")(86.94 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/317669)**
(11)


![Реter Konow](https://c.mql5.com/avatar/avatar_na2.png)

**[Реter Konow](https://www.mql5.com/en/users/peterkonow)**
\|
1 May 2019 at 13:04

**Artyom Trishkin:**

...without finding the required solutions written and published somewhere.

Does that mean the library will become a staff library?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
1 May 2019 at 14:07

**Реter Konow:**

Does that mean the library will become a staff library?

No. It means that one library will be sufficient.


![Реter Konow](https://c.mql5.com/avatar/avatar_na2.png)

**[Реter Konow](https://www.mql5.com/en/users/peterkonow)**
\|
1 May 2019 at 14:29

**Artyom Trishkin:**

No. This means that one library will be quite sufficient.

Good luck.


![BmC](https://c.mql5.com/avatar/2015/8/55D729A0-4144.jpg)

**[BmC](https://www.mql5.com/en/users/bmc)**
\|
27 Apr 2020 at 12:43

Artem, thank you!

In the method:

```
//+------------------------------------------------------------------+
//|| Updates the order list|
//+------------------------------------------------------------------+
void CMarketCollection::Refresh(void)
```

A "market order" is created dynamically

```
#else
//--- Positions
   int total_positions=::PositionsTotal();
   for(int i=0; i<total_positions; i++)
     {
      ulong ticket=::PositionGetTicket(i);
      if(ticket==0) continue;
      CMarketPosition *position=new CMarketPosition(ticket);
      if(position==NULL) continue;
      //--- Add a position object to the list of market orders and positions
      if(!this.AddToListMarket(position))
         continue;
      //--- Get the control order index by ticket and position identifier
      int index=this.IndexControlOrder(ticket,position.PositionID());
      //--- If the order is not in the list of control orders - add it
      if(index==WRONG_VALUE)
        {
         if(!this.AddToListControl(position))
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Failed to add control position ","Failed to add a control position "),position.TypeDescription()," #",position.Ticket());
           }
        }
      //--- If the order already exists in the list of control orders - check it for property changes
      else if(index>WRONG_VALUE)
        {
         this.OnChangeEvent(position,index);
        }
     }
//--- Orders
   int total_orders=::OrdersTotal();
   for(int i=0; i<total_orders; i++)
     {
      ulong ticket=::OrderGetTicket(i);
      if(ticket==0) continue;
      ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)::OrderGetInteger(ORDER_TYPE);
      //--- Market Order
      if(type<ORDER_TYPE_BUY_LIMIT)
        {
         CMarketOrder *order=new CMarketOrder(ticket);
         if(order==NULL) continue;
         //--- Add market order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
        }
      //--- Pending order
      else
        {
         CMarketPending *order=new CMarketPending(ticket);
         if(order==NULL) continue;
         //--- Add a pending order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
         //--- Get the control order index by ticket and position identifier
         int index=this.IndexControlOrder(ticket,order.PositionID());
         //--- If the order is not in the list of control orders - add it
         if(index==WRONG_VALUE)
           {
            if(!this.AddToListControl(order))
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Failed to add control order ","Failed to add a control order "),order.TypeDescription()," #",order.Ticket());
              }
           }
         //--- If the order already exists in the list of control orders - check it for property changes
         else if(index>WRONG_VALUE)
           {
            this.OnChangeEvent(order,index);
           }
        }
     }
#endif
```

Then it is passed by reference to the method :

```
         //--- Add market order object to the list of market orders and positions
         if(!this.AddToListMarket(order))
            continue;
```

In the method"AddToListMarket" the market order is not taken into account in the hash\_sum "hash\_sum" then why do we need to enter and control it?

Please explain why we need it, if we can find out all the information from the position or [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 documentation:")?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
27 Apr 2020 at 15:25

**BmC:**

Artem, thank you!

In the method:

A "market order" is created dynamically

Then it is passed by reference to the method :

In the method"AddToListMarket" the market order is not considered in the hash\_sum "hash\_sum" then why should it be entered and controlled?

Please explain why we need it, if we can find out all the information from the position or pending order?

Market orders are orders to place either a market or [pending order](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties"). They are not needed for the current needs of the library (tracking changes), but they are present in the list of library objects for quick manual retrieval. For example, to determine slippages or delays in execution.

![Evaluating the ability of Fractal index and Hurst exponent to predict financial time series](https://c.mql5.com/2/36/fraktal1.png)[Evaluating the ability of Fractal index and Hurst exponent to predict financial time series](https://www.mql5.com/en/articles/6834)

Studies related to search for the fractal behavior of financial data suggest that behind the seemingly chaotic behavior of economic time series there are hidden stable mechanisms of participants' collective behavior. These mechanisms can lead to the emergence of price dynamics on the exchange, which can define and describe specific properties of price series. When applied to trading, one could benefit from the indicators which can efficiently and reliably estimate the fractal parameters in the scale and time frame, which are relevant in practice.

![Library for easy and quick development of MetaTrader programs (part VI): Netting account events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part VI): Netting account events](https://www.mql5.com/en/articles/6383)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the fifth part of the article series, we created trading event classes and the event collection, from which the events are sent to the base object of the Engine library and the control program chart. In this part, we will let the library to work on netting accounts.

![Price velocity measurement methods](https://c.mql5.com/2/36/Article_Logo__1.png)[Price velocity measurement methods](https://www.mql5.com/en/articles/6947)

There are multiple different approaches to market research and analysis. The main ones are technical and fundamental. In technical analysis, traders collect, process and analyze numerical data and parameters related to the market, including prices, volumes, etc. In fundamental analysis, traders analyze events and news affecting the markets directly or indirectly. The article deals with price velocity measurement methods and studies trading strategies based on that methods.

![Applying OLAP in trading (part 2): Visualizing the interactive multidimensional data analysis results](https://c.mql5.com/2/36/OLAP_02__1.png)[Applying OLAP in trading (part 2): Visualizing the interactive multidimensional data analysis results](https://www.mql5.com/en/articles/6603)

In this article, we consider the creation of an interactive graphical interface for an MQL program, which is designed for the processing of account history and trading reports using OLAP techniques. To obtain a visual result, we will use maximizable and scalable windows, an adaptive layout of rubber controls and a new control for displaying diagrams. To provide the visualization functionality, we will implement a GUI with the selection of variables along coordinate axes, as well as with the selection of aggregate functions, diagram types and sorting options.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=oyoctbsxxsvyuigcglnhnjctecsltnqv&ssn=1769186504641307706&ssn_dr=0&ssn_sr=0&fv_date=1769186504&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6482&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20VII)%3A%20StopLimit%20order%20activation%20events%2C%20preparing%20the%20functionality%20for%20order%20and%20position%20modification%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918650447190641&fz_uniq=5070489513555334856&sv=2552)

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