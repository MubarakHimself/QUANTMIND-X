---
title: Library for easy and quick development of MetaTrader programs (part XI). Compatibility with MQL4 - Position closure events
url: https://www.mql5.com/en/articles/6921
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:40:49.837267
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/6921&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070477826949322382)

MetaTrader 5 / Examples


### Contents

- [Removing unused properties](https://www.mql5.com/en/articles/6921#node01)
- [Implementing position closing events for MQL4](https://www.mql5.com/en/articles/6921#node02)
- [Testing](https://www.mql5.com/en/articles/6921#node03)
- [What's next?](https://www.mql5.com/en/articles/6921#node04)


### Removing unused properties

While working on defining events, I noticed that in MQL5 all time parameters are set in milliseconds. MQL4 features no such order and position
properties, but nothing prevents us from using the time in seconds expressed in milliseconds for MQL4. In other words, any time in seconds is
simply duplicated by time in milliseconds and is not used anywhere. Receiving and displaying time to be set in seconds is quite identical to
receiving it in milliseconds apart from a "tail" of three digits indicating the number of milliseconds in the format of the displayed time.

Therefore, I decided to remove all time properties set in seconds from the order properties in case an order has the same property in milliseconds.


Since we decided to remove something, it would be nice to add something, too. So let's add the new "custom comment" property to each order. It
can be set for any order or position (both open and closed/removed ones) at any time. Why do we need this? For example, this may be necessary for
text labels for orders fitting certain conditions, or for visual display (the library is to feature its own graphical shell later) so that an
order marked with a text label can be easily displayed using various graphical constructions.

Open the **Defines.mqh** file, press **Ctrl+F** to find all order properties containing time in seconds and a similar
property with the "\_MSC" ending (this property is set in milliseconds). Remove the

"millisecond" order properties leaving "second"
properties and replace the number of integer properties
of 24 with **21**:

```
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Order magic number
   ORDER_PROP_TIME_OPEN,                                    // Open time (MQL5 Deal time)
   ORDER_PROP_TIME_CLOSE,                                   // Close time (MQL5 Execution or removal time - ORDER_TIME_DONE)
   ORDER_PROP_TIME_OPEN_MSC,                                // Open time in milliseconds (MQL5 Deal time in msc)
   ORDER_PROP_TIME_CLOSE_MSC,                               // Close time in milliseconds (MQL5 Execution or removal time - ORDER_TIME_DONE_MSC)
   ORDER_PROP_TIME_EXP,                                     // Order expiration date (for pending orders)
   ORDER_PROP_STATUS,                                       // Order status (from the ENUM_ORDER_STATUS enumeration)
   ORDER_PROP_TYPE,                                         // Order/deal type
   ORDER_PROP_REASON,                                       // Deal/order/position reason or source
   ORDER_PROP_STATE,                                        // Order status (from the ENUM_ORDER_STATE enumeration)
   ORDER_PROP_POSITION_ID,                                  // Position ID
   ORDER_PROP_POSITION_BY_ID,                               // Opposite position ID
   ORDER_PROP_DEAL_ORDER_TICKET,                            // Ticket of the order that triggered a deal
   ORDER_PROP_DEAL_ENTRY,                                   // Deal direction – IN, OUT or IN/OUT
   ORDER_PROP_TIME_UPDATE,                                  // Position change time in seconds
   ORDER_PROP_TIME_UPDATE_MSC,                              // Position change time in milliseconds
   ORDER_PROP_TICKET_FROM,                                  // Parent order ticket
   ORDER_PROP_TICKET_TO,                                    // Derived order ticket
   ORDER_PROP_PROFIT_PT,                                    // Profit in points
   ORDER_PROP_CLOSE_BY_SL,                                  // Flag of closing by StopLoss
   ORDER_PROP_CLOSE_BY_TP,                                  // Flag of closing by TakeProfit
   ORDER_PROP_GROUP_ID,                                     // Order/position group ID
   ORDER_PROP_DIRECTION,                                    // Direction type (Buy, Sell)
  };
#define ORDER_PROP_INTEGER_TOTAL    (24)                    // Total number of integer properties
#define ORDER_PROP_INTEGER_SKIP     (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

After the changes, the list of order's integer properties will look as follows:

```
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Order magic number
   ORDER_PROP_TIME_OPEN,                                    // Open time in milliseconds (MQL5 Deal time)
   ORDER_PROP_TIME_CLOSE,                                   // Close time in milliseconds (MQL5 Execution or removal time - ORDER_TIME_DONE)
   ORDER_PROP_TIME_EXP,                                     // Order expiration date (for pending orders)
   ORDER_PROP_STATUS,                                       // Order status (from the ENUM_ORDER_STATUS enumeration)
   ORDER_PROP_TYPE,                                         // Order/deal type
   ORDER_PROP_REASON,                                       // Deal/order/position reason or source
   ORDER_PROP_STATE,                                        // Order status (from the ENUM_ORDER_STATE enumeration)
   ORDER_PROP_POSITION_ID,                                  // Position ID
   ORDER_PROP_POSITION_BY_ID,                               // Opposite position ID
   ORDER_PROP_DEAL_ORDER_TICKET,                            // Ticket of the order that triggered a deal
   ORDER_PROP_DEAL_ENTRY,                                   // Deal direction – IN, OUT or IN/OUT
   ORDER_PROP_TIME_UPDATE,                                  // Position change time in milliseconds
   ORDER_PROP_TICKET_FROM,                                  // Parent order ticket
   ORDER_PROP_TICKET_TO,                                    // Derived order ticket
   ORDER_PROP_PROFIT_PT,                                    // Profit in points
   ORDER_PROP_CLOSE_BY_SL,                                  // Flag of closing by StopLoss
   ORDER_PROP_CLOSE_BY_TP,                                  // Flag of closing by TakeProfit
   ORDER_PROP_GROUP_ID,                                     // Order/position group ID
   ORDER_PROP_DIRECTION,                                    // Direction type (Buy, Sell)
  };
#define ORDER_PROP_INTEGER_TOTAL    (21)                    // Total number of integer properties
#define ORDER_PROP_INTEGER_SKIP     (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

Let's find the enumeration of possible selection options by time and remove selection
constants in milliseconds:

```
//+------------------------------------------------------------------+
//| Possible selection options by time                               |
//+------------------------------------------------------------------+
enum ENUM_SELECT_BY_TIME
  {
   SELECT_BY_TIME_OPEN,                                     // By open time
   SELECT_BY_TIME_CLOSE,                                    // By close time
   SELECT_BY_TIME_OPEN_MSC,                                 // By open time in milliseconds
   SELECT_BY_TIME_CLOSE_MSC,                                // By close time in milliseconds
  };
//+------------------------------------------------------------------+
```

The enumeration will consist of only two constants:

```
//+------------------------------------------------------------------+
//| Possible selection options by time                               |
//+------------------------------------------------------------------+
enum ENUM_SELECT_BY_TIME
  {
   SELECT_BY_TIME_OPEN,                                     // By open time (in milliseconds)
   SELECT_BY_TIME_CLOSE,                                    // By close time (in milliseconds)
  };
//+------------------------------------------------------------------+
```

Now when setting selection by time, selection is performed by time in milliseconds for MQL5 and in seconds for MQL4.

Add the new "custom comment" property to the order string
properties and increase the total number of string properties up to

**4**:

```
//+------------------------------------------------------------------+
//| Order, deal, position string properties                          |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_STRING
  {
   ORDER_PROP_SYMBOL = (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL), // Order symbol
   ORDER_PROP_COMMENT,                                      // Order comment
   ORDER_PROP_COMMENT_EXT,                                  // Order custom comment
   ORDER_PROP_EXT_ID                                        // Order ID in the external trading system
  };
#define ORDER_PROP_STRING_TOTAL     (4)                     // Total number of string properties
//+------------------------------------------------------------------+
```

Let's remove all references to milliseconds in the enumeration of possible sorting criteria (now they are used by default when sorting by time)
and add the

criterion for sorting by a custom comment:

```
//+------------------------------------------------------------------+
//| Possible criteria of orders and deals sorting                    |
//+------------------------------------------------------------------+
#define FIRST_ORD_DBL_PROP          (ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_INTEGER_SKIP)
#define FIRST_ORD_STR_PROP          (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL-ORDER_PROP_INTEGER_SKIP)
enum ENUM_SORT_ORDERS_MODE
  {
   //--- Sort by integer properties
   SORT_BY_ORDER_TICKET          =  0,                      // Sort by order ticket
   SORT_BY_ORDER_MAGIC           =  1,                      // Sort by order magic number
   SORT_BY_ORDER_TIME_OPEN       =  2,                      // Sort by order open time in milliseconds
   SORT_BY_ORDER_TIME_CLOSE      =  3,                      // Sort by order close time in milliseconds
   SORT_BY_ORDER_TIME_EXP        =  4,                      // Sort by order expiration date
   SORT_BY_ORDER_STATUS          =  5,                      // Sort by order status (market order/pending order/deal/balance, credit operation)
   SORT_BY_ORDER_TYPE            =  6,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  7,                      // Sort by order/position reason/source
   SORT_BY_ORDER_STATE           =  8,                     // Sort by order status
   SORT_BY_ORDER_POSITION_ID     =  9,                     // Sort by position ID
   SORT_BY_ORDER_POSITION_BY_ID  =  10,                     // Sort by opposite position ID
   SORT_BY_ORDER_DEAL_ORDER      =  11,                     // Sort by order a deal is based on
   SORT_BY_ORDER_DEAL_ENTRY      =  12,                     // Sort by deal direction – IN, OUT or IN/OUT
   SORT_BY_ORDER_TIME_UPDATE     =  13,                     // Sort by position change time in seconds
   SORT_BY_ORDER_TICKET_FROM     =  14,                     // Sort by parent order ticket
   SORT_BY_ORDER_TICKET_TO       =  15,                     // Sort by derived order ticket
   SORT_BY_ORDER_PROFIT_PT       =  16,                     // Sort by order profit in points
   SORT_BY_ORDER_CLOSE_BY_SL     =  17,                     // Sort by order closing by StopLoss flag
   SORT_BY_ORDER_CLOSE_BY_TP     =  18,                     // Sort by order closing by TakeProfit flag
   SORT_BY_ORDER_GROUP_ID        =  19,                     // Sort by order/position group ID
   SORT_BY_ORDER_DIRECTION       =  20,                     // Sort by direction (Buy, Sell)
   //--- Sort by real properties
   SORT_BY_ORDER_PRICE_OPEN      =  FIRST_ORD_DBL_PROP,     // Sort by open price
   SORT_BY_ORDER_PRICE_CLOSE     =  FIRST_ORD_DBL_PROP+1,   // Sort by close price
   SORT_BY_ORDER_SL              =  FIRST_ORD_DBL_PROP+2,   // Sort by StopLoss price
   SORT_BY_ORDER_TP              =  FIRST_ORD_DBL_PROP+3,   // Sort by TakeProfit price
   SORT_BY_ORDER_PROFIT          =  FIRST_ORD_DBL_PROP+4,   // Sort by profit
   SORT_BY_ORDER_COMMISSION      =  FIRST_ORD_DBL_PROP+5,   // Sort by commission
   SORT_BY_ORDER_SWAP            =  FIRST_ORD_DBL_PROP+6,   // Sort by swap
   SORT_BY_ORDER_VOLUME          =  FIRST_ORD_DBL_PROP+7,   // Sort by volume
   SORT_BY_ORDER_VOLUME_CURRENT  =  FIRST_ORD_DBL_PROP+8,   // Sort by unexecuted volume
   SORT_BY_ORDER_PROFIT_FULL     =  FIRST_ORD_DBL_PROP+9,   // Sort by profit+commission+swap criterion
   SORT_BY_ORDER_PRICE_STOP_LIMIT=  FIRST_ORD_DBL_PROP+10,  // Sort by Limit order when StopLimit order is activated
   //--- Sort by string properties
   SORT_BY_ORDER_SYMBOL          =  FIRST_ORD_STR_PROP,     // Sort by symbol
   SORT_BY_ORDER_COMMENT         =  FIRST_ORD_STR_PROP+1,   // Sort by comment
   SORT_BY_ORDER_COMMENT_EXT     =  FIRST_ORD_STR_PROP+2,   // Sort by custom comment
   SORT_BY_ORDER_EXT_ID          =  FIRST_ORD_STR_PROP+3    // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

This concludes the changes in Defines.mqh. Now we need to remove all references to deleted order properties in the library files:

replace all instances of sorting modes in all the library files

```
SORT_BY_ORDER_TIME_OPEN_MSC
```

and

```
SORT_BY_ORDER_TIME_CLOSE_MSC
```

with

```
SORT_BY_ORDER_TIME_OPEN
```

and

```
SORT_BY_ORDER_TIME_CLOSE
```

In the **HistoryDeal.mqh**, **HistoryOrder.mqh**, **HistoryPending.mqh**, **MarketOrder.mqh**, **MarketPending.mqh**
and

**MarketPosition.mqh** files of the abstract order descendant classes, remove all references to order millisecond properties (they
are now millisecond by default):

```
ORDER_PROP_TIME_CLOSE_MSC
```

and

```
ORDER_PROP_TIME_UPDATE_MSC
```

In the **Order.mqh** file of the COrder abstract order class, remove
the methods returning time in seconds from the private section:

```
   datetime          OrderOpenTime(void)           const;
   datetime          OrderCloseTime(void)          const;
   datetime          OrderExpiration(void)         const;
   datetime          PositionTimeUpdate(void)      const;
   datetime          PositionTimeUpdateMSC(void)   const;
```

Remove simplified access methods returning time in milliseconds
from the class public section. It is to be returned by the methods returning
time in seconds:

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
   //--- Return (1) ticket, (2) parent order ticket, (3) derived order ticket, (4) magic number, (5) order reason,
   //--- (6) position ID, (7) opposite position ID, (8) group ID, (9) type, (10) flag of closing by StopLoss,
   //--- (11) flag of closing by TakeProfit (12) open time, (13) close time, (14) open time in milliseconds,
   //--- (15) close time in milliseconds, (16) expiration date, (17) state, (18) status, (19) order type by direction
   long              Ticket(void)                                       const { return this.GetProperty(ORDER_PROP_TICKET);                     }
   long              TicketFrom(void)                                   const { return this.GetProperty(ORDER_PROP_TICKET_FROM);                }
   long              TicketTo(void)                                     const { return this.GetProperty(ORDER_PROP_TICKET_TO);                  }
   long              Magic(void)                                        const { return this.GetProperty(ORDER_PROP_MAGIC);                      }
   long              Reason(void)                                       const { return this.GetProperty(ORDER_PROP_REASON);                     }
   long              PositionID(void)                                   const { return this.GetProperty(ORDER_PROP_POSITION_ID);                }
   long              PositionByID(void)                                 const { return this.GetProperty(ORDER_PROP_POSITION_BY_ID);             }
   long              GroupID(void)                                      const { return this.GetProperty(ORDER_PROP_GROUP_ID);                   }
   long              TypeOrder(void)                                    const { return this.GetProperty(ORDER_PROP_TYPE);                       }
   bool              IsCloseByStopLoss(void)                            const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_SL);          }
   bool              IsCloseByTakeProfit(void)                          const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_TP);          }
   datetime          TimeOpen(void)                                     const { return (datetime)this.GetProperty(ORDER_PROP_TIME_OPEN);        }
   datetime          TimeClose(void)                                    const { return (datetime)this.GetProperty(ORDER_PROP_TIME_CLOSE);       }
   datetime          TimeOpenMSC(void)                                  const { return (datetime)this.GetProperty(ORDER_PROP_TIME_OPEN_MSC);    }
   datetime          TimeCloseMSC(void)                                 const { return (datetime)this.GetProperty(ORDER_PROP_TIME_CLOSE_MSC);   }
   datetime          TimeExpiration(void)                               const { return (datetime)this.GetProperty(ORDER_PROP_TIME_EXP);         }
   ENUM_ORDER_STATE  State(void)                                        const { return (ENUM_ORDER_STATE)this.GetProperty(ORDER_PROP_STATE);    }
   ENUM_ORDER_STATUS Status(void)                                       const { return (ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS);  }
   ENUM_ORDER_TYPE   TypeByDirection(void)                              const { return (ENUM_ORDER_TYPE)this.GetProperty(ORDER_PROP_DIRECTION); }

   //--- Return (1) open price, (2) close price, (3) profit, (4) commission, (5) swap, (6) volume,
```

Also, add the methods returning and setting
an order custom comment:

```
   //--- Return (1) symbol, (2) comment, (3) ID at an exchange
   string            Symbol(void)                                       const { return this.GetProperty(ORDER_PROP_SYMBOL);                     }
   string            Comment(void)                                      const { return this.GetProperty(ORDER_PROP_COMMENT);                    }
   string            CommentExt(void)                                   const { return this.GetProperty(ORDER_PROP_COMMENT_EXT);                }
   string            ExternalID(void)                                   const { return this.GetProperty(ORDER_PROP_EXT_ID);                     }

   //--- Get the full order profit
   double            ProfitFull(void)                                   const { return this.Profit()+this.Comission()+this.Swap();              }
   //--- Get order profit in points
   int               ProfitInPoints(void) const;
//--- Set (1) group ID and (2) custom comment
   void              SetGroupID(const long group_id)                          { this.SetProperty(ORDER_PROP_GROUP_ID,group_id);                 }
   void              SetCommentExt(const string comment_ext)                  { this.SetProperty(ORDER_PROP_COMMENT_EXT,comment_ext);           }

```

**In the closed constructor of the COrder class**, remove saving the second time properties, replace millisecond
properties with second ones and save time
in milliseconds in them. Add saving a custom comment as an empty string:

```
//+------------------------------------------------------------------+
//| Closed parametric constructor                                    |
//+------------------------------------------------------------------+
COrder::COrder(ENUM_ORDER_STATUS order_status,const ulong ticket)
  {
//--- Save integer properties
   this.m_ticket=ticket;
   this.m_long_prop[ORDER_PROP_STATUS]                               = order_status;
   this.m_long_prop[ORDER_PROP_MAGIC]                                = this.OrderMagicNumber();
   this.m_long_prop[ORDER_PROP_TICKET]                               = this.OrderTicket();
   this.m_long_prop[ORDER_PROP_TIME_EXP]                             = this.OrderExpiration();
   this.m_long_prop[ORDER_PROP_TYPE]                                 = this.OrderType();
   this.m_long_prop[ORDER_PROP_STATE]                                = this.OrderState();
   this.m_long_prop[ORDER_PROP_DIRECTION]                            = this.OrderTypeByDirection();
   this.m_long_prop[ORDER_PROP_POSITION_ID]                          = this.OrderPositionID();
   this.m_long_prop[ORDER_PROP_REASON]                               = this.OrderReason();
   this.m_long_prop[ORDER_PROP_DEAL_ORDER_TICKET]                    = this.DealOrderTicket();
   this.m_long_prop[ORDER_PROP_DEAL_ENTRY]                           = this.DealEntry();
   this.m_long_prop[ORDER_PROP_POSITION_BY_ID]                       = this.OrderPositionByID();
   this.m_long_prop[ORDER_PROP_TIME_OPEN]                            = this.OrderOpenTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE]                           = this.OrderCloseTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE]                          = this.PositionTimeUpdateMSC();


//--- Save real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_OPEN)]         = this.OrderOpenPrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_CLOSE)]        = this.OrderClosePrice();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT)]             = this.OrderProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_COMMISSION)]         = this.OrderCommission();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SWAP)]               = this.OrderSwap();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME)]             = this.OrderVolume();
   this.m_double_prop[this.IndexProp(ORDER_PROP_SL)]                 = this.OrderStopLoss();
   this.m_double_prop[this.IndexProp(ORDER_PROP_TP)]                 = this.OrderTakeProfit();
   this.m_double_prop[this.IndexProp(ORDER_PROP_VOLUME_CURRENT)]     = this.OrderVolumeCurrent();
   this.m_double_prop[this.IndexProp(ORDER_PROP_PRICE_STOP_LIMIT)]   = this.OrderPriceStopLimit();

//--- Save string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_SYMBOL)]             = this.OrderSymbol();
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT)]            = this.OrderComment();
   this.m_string_prop[this.IndexProp(ORDER_PROP_EXT_ID)]             = this.OrderExternalID();

//--- Save additional integer properties
   this.m_long_prop[ORDER_PROP_PROFIT_PT]                            = this.ProfitInPoints();
   this.m_long_prop[ORDER_PROP_TICKET_FROM]                          = this.OrderTicketFrom();
   this.m_long_prop[ORDER_PROP_TICKET_TO]                            = this.OrderTicketTo();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_SL]                          = this.OrderCloseByStopLoss();
   this.m_long_prop[ORDER_PROP_CLOSE_BY_TP]                          = this.OrderCloseByTakeProfit();
   this.m_long_prop[ORDER_PROP_GROUP_ID]                             = 0;

//--- Save additional real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT_FULL)]        = this.ProfitFull();

//--- Save additional string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT_EXT)]        = "";
  }
//+------------------------------------------------------------------+
```

**In the method returning position ID** for MQL4, do the
following: if this is a market position, return its ticket,

otherwise return
zero. A position opening ticket acts as a position ID in MQL5. It remains unchanged during the entire position lifetime.

Thus, in MQL4,
only a position ticket can serve as a position ID. A pending order in MQL4 has no such ID. If an order is deleted, no position was opened for it. If
an order was activated, there is no such order in MQL4 order history but the position receives its ticket, thus the ticket acts as the position
ID.

```
//+------------------------------------------------------------------+
//| Return the position ID                                           |
//+------------------------------------------------------------------+
long COrder::OrderPositionID(void) const
  {
#ifdef __MQL4__
   return(this.Status()==ORDER_STATUS_MARKET_POSITION ? this.Ticket() : 0);
#else
   long id=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_POSITION   : id=::PositionGetInteger(POSITION_IDENTIFIER);             break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : id=::OrderGetInteger(ORDER_POSITION_ID);                  break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : id=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_ID);  break;
      case ORDER_STATUS_DEAL              : id=::HistoryDealGetInteger(m_ticket,DEAL_POSITION_ID);    break;
      default                             : id=0;                                                     break;
     }
   return id;
#endif
  }
//+------------------------------------------------------------------+
```

Supplement the **method returning an opposite position ID** for MQL4:

```
//+------------------------------------------------------------------+
//| Return the opposite position ID                                  |
//+------------------------------------------------------------------+
long COrder::OrderPositionByID(void) const
  {
   long ticket=0;
#ifdef __MQL4__
   string order_comment=::OrderComment();
   if(::StringFind(order_comment,"close hedge by #")>WRONG_VALUE) ticket=::StringToInteger(::StringSubstr(order_comment,16));
#else
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : ticket=::OrderGetInteger(ORDER_POSITION_BY_ID);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : ticket=::HistoryOrderGetInteger(m_ticket,ORDER_POSITION_BY_ID); break;
      default                             : ticket=0;                                                       break;
     }
#endif
   return ticket;
  }
//+------------------------------------------------------------------+
```

Here, if this is MQL4 and if
the order comment features the "close hedge by #" line, calculate the index of the
opposite order ticket number start in the comment string and assign it to the
value returned by the method.

**Remove the implementation of the two methods that are no longer**
**needed** from the class listing since we do not want to get time in seconds anymore:

```
//+------------------------------------------------------------------+
//| Return open time                                                 |
//+------------------------------------------------------------------+
datetime COrder::OrderOpenTime(void) const
  {
#ifdef __MQL4__
   return ::OrderOpenTime();
#else
   datetime res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_MARKET_POSITION   : res=(datetime)::PositionGetInteger(POSITION_TIME);                 break;
      case ORDER_STATUS_MARKET_ORDER      :
      case ORDER_STATUS_MARKET_PENDING    : res=(datetime)::OrderGetInteger(ORDER_TIME_SETUP);                 break;
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=(datetime)::HistoryOrderGetInteger(m_ticket,ORDER_TIME_SETUP); break;
      case ORDER_STATUS_DEAL              : res=(datetime)::HistoryDealGetInteger(m_ticket,DEAL_TIME);         break;
      default                             : res=0;                                                             break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
//| Return close time                                                |
//+------------------------------------------------------------------+
datetime COrder::OrderCloseTime(void) const
  {
#ifdef __MQL4__
   return ::OrderCloseTime();
#else
   datetime res=0;
   switch((ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS))
     {
      case ORDER_STATUS_HISTORY_PENDING   :
      case ORDER_STATUS_HISTORY_ORDER     : res=(datetime)::HistoryOrderGetInteger(m_ticket,ORDER_TIME_DONE);  break;
      case ORDER_STATUS_DEAL              : res=(datetime)::HistoryDealGetInteger(m_ticket,DEAL_TIME);         break;
      default                             : res=0;                                                             break;
     }
   return res;
#endif
  }
//+------------------------------------------------------------------+
```

For more meaningful display of an order status in MQL4, **make**
**small changes in the method returning the order status description**:

```
//+------------------------------------------------------------------+
//| Return the order status name                                     |
//+------------------------------------------------------------------+
string COrder::StatusDescription(void) const
  {
   ENUM_ORDER_STATUS status=this.Status();
   ENUM_ORDER_TYPE   type=(ENUM_ORDER_TYPE)this.TypeOrder();
   return
     (
      status==ORDER_STATUS_BALANCE           ?  TextByLanguage("Балансовая операция","Balance operation") :
      #ifdef __MQL5__
      status==ORDER_STATUS_MARKET_ORDER || status==ORDER_STATUS_HISTORY_ORDER ?
         (
          type==ORDER_TYPE_CLOSE_BY ? TextByLanguage("Закрывающий ордер","Order for closing by")         :
          TextByLanguage("Ордер на ","The order to ")+(type==ORDER_TYPE_BUY ? TextByLanguage("покупку","buy") : TextByLanguage("продажу","sell"))
         ) :
      #else
      status==ORDER_STATUS_HISTORY_ORDER     ?  TextByLanguage("Исторический ордер","History order")     :
      #endif
      status==ORDER_STATUS_DEAL              ?  TextByLanguage("Сделка","Deal")                          :
      status==ORDER_STATUS_MARKET_POSITION   ?  TextByLanguage("Позиция","Active position")              :
      status==ORDER_STATUS_MARKET_PENDING    ?  TextByLanguage("Установленный отложенный ордер","Active pending order") :
      status==ORDER_STATUS_HISTORY_PENDING   ?  TextByLanguage("Отложенный ордер","Pending order") :
      EnumToString(status)
     );
  }
//+------------------------------------------------------------------+
```

Here, for a remote pending order and a closed positionin
MQL4, we will return the status description as "Historical order".

**In the method returning the description of the order integer property, change** the strings containing the
descriptions of ORDER\_PROP\_TIME\_OPEN, ORDER\_PROP\_TIME\_CLOSE and ORDER\_PROP\_TIME\_UPDATE properties so that millisecond properties
are returned for them:

```
//+------------------------------------------------------------------+
//| Return description of an order's integer property                |
//+------------------------------------------------------------------+
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property)
  {
   return
     (
   //--- General properties
      property==ORDER_PROP_MAGIC             ?  TextByLanguage("Магик","Magic number")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET            ?  TextByLanguage("Тикет","Ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_FROM       ?  TextByLanguage("Тикет родительского ордера","Ticket of parent order")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_TO         ?  TextByLanguage("Тикет наследуемого ордера","Inherited order ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_EXP          ?  TextByLanguage("Дата экспирации","Date of expiration")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          (this.GetProperty(property)==0     ?  TextByLanguage(": Не задана",": Not set") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS))
         )  :
      property==ORDER_PROP_TYPE              ?  TextByLanguage("Тип","Type")+": "+this.TypeDescription()                   :
      property==ORDER_PROP_DIRECTION         ?  TextByLanguage("Тип по направлению","Type by direction")+": "+this.DirectionDescription() :

      property==ORDER_PROP_REASON            ?  TextByLanguage("Причина","Reason")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetReasonDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_ID       ?  TextByLanguage("Идентификатор позиции","Position identifier")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ORDER_TICKET ?  TextByLanguage("Сделка на основании ордера с тикетом","Deal by order ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ENTRY        ?  TextByLanguage("Направление сделки","Deal entry")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+this.GetEntryDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_BY_ID    ?  TextByLanguage("Идентификатор встречной позиции","Opposite position identifier")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN         ?  TextByLanguage("Время открытия в милисекундах","Opening time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_CLOSE        ?  TextByLanguage("Время закрытия в милисекундах","Closing time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_UPDATE       ?  TextByLanguage("Время изменения позиции в милисекундах","Time to change the position in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)!=0 ? TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")" : "0")
         )  :
      property==ORDER_PROP_STATE             ?  TextByLanguage("Состояние","Statе")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": \""+this.StateDescription()+"\""
         )  :
   //--- Additional property
      property==ORDER_PROP_STATUS            ?  TextByLanguage("Статус","Status")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": \""+this.StatusDescription()+"\""
         )  :
      property==ORDER_PROP_PROFIT_PT         ?  (
                                                 this.Status()==ORDER_STATUS_MARKET_PENDING ?
                                                 TextByLanguage("Дистанция от цены в пунктах","Distance from price in points") :
                                                 TextByLanguage("Прибыль в пунктах","Profit in points")
                                                )+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_CLOSE_BY_SL       ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_CLOSE_BY_TP       ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_GROUP_ID          ?  TextByLanguage("Идентификатор группы","Group identifier")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

and **add returning a custom comment description to the method returning a string**
**property description:**

```
//+------------------------------------------------------------------+
//| Return description of the order's string property                |
//+------------------------------------------------------------------+
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_STRING property)
  {
   return
     (
      property==ORDER_PROP_SYMBOL         ?  TextByLanguage("Символ","Symbol")+": \""+this.GetProperty(property)+"\""            :
      property==ORDER_PROP_COMMENT        ?  TextByLanguage("Комментарий","Comment")+
         (this.GetProperty(property)==""  ?  TextByLanguage(": Отсутствует",": Not set"):": \""+this.GetProperty(property)+"\"") :
      property==ORDER_PROP_COMMENT_EXT    ?  TextByLanguage("Пользовательский комментарий","Custom comment")+
         (this.GetProperty(property)==""  ?  TextByLanguage(": Не задан",": Not set"):": \""+this.GetProperty(property)+"\"") :
      property==ORDER_PROP_EXT_ID         ?  TextByLanguage("Идентификатор на бирже","Exchange identifier")+
         (!this.SupportProperty(property) ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
         (this.GetProperty(property)==""  ?  TextByLanguage(": Отсутствует",": Not set"):": \""+this.GetProperty(property)+"\"")):
      ""
     );
  }
//+------------------------------------------------------------------+
```

This concludes the changes in the COrder abstract order class.

In the **DELib.mqh** service functions file, make a small **improvement in the function returning an order/position name by order**
**type:**

```
//+------------------------------------------------------------------+
//| Return order name                                                |
//+------------------------------------------------------------------+
string OrderTypeDescription(const ENUM_ORDER_TYPE type,bool as_order=true)
  {
   string pref=(#ifdef __MQL5__ "Market order" #else (as_order ? "Market order" : "Position") #endif );
   return
     (
      type==ORDER_TYPE_BUY_LIMIT       ?  "Buy Limit"                                                 :
      type==ORDER_TYPE_BUY_STOP        ?  "Buy Stop"                                                  :
      type==ORDER_TYPE_SELL_LIMIT      ?  "Sell Limit"                                                :
      type==ORDER_TYPE_SELL_STOP       ?  "Sell Stop"                                                 :
   #ifdef __MQL5__
      type==ORDER_TYPE_BUY_STOP_LIMIT  ?  "Buy Stop Limit"                                            :
      type==ORDER_TYPE_SELL_STOP_LIMIT ?  "Sell Stop Limit"                                           :
      type==ORDER_TYPE_CLOSE_BY        ?  TextByLanguage("Закрывающий ордер","Order for closing by")  :
   #else
      type==ORDER_TYPE_BALANCE         ?  TextByLanguage("Балансовая операция","Balance operation")   :
      type==ORDER_TYPE_CREDIT          ?  TextByLanguage("Кредитная операция","Credit operation")     :
   #endif
      type==ORDER_TYPE_BUY             ?  pref+" Buy"                                                 :
      type==ORDER_TYPE_SELL            ?  pref+" Sell"                                                :
      TextByLanguage("Неизвестный тип ордера","Unknown order type")
     );
  }
//+------------------------------------------------------------------+
```

Here we added the flag managing the display of an order name for
MQL4 either as an order, or as
a position. Display for MQL4 "as an order" is set by default. Why was this done? Suppose that an order that caused a position to be opened
is shown in square brackets when sending the position open event to the journal. In this case, the \[Position Sell #123\] message for the Sell
position opened by a market (not pending) order with the ticket 123 as an order that caused the position opening is replaced with the more
meaningful \[Market order Sell #123\] entry.

**Let's improve the AddToListMarket() method of the market order and position collection class**. Instead of the
ORDER\_PROP\_TIME\_UPDATE\_MSC position update time in milliseconds, we now use the

ORDER\_PROP\_TIME\_UPDATE position update time (it is set in
milliseconds by default):

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
         this.m_struct_curr_market.hash_sum_acc+=order.GetProperty(ORDER_PROP_TIME_UPDATE)+this.ConvertToHS(order);
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
      ::Print(DFUN,order.TypeDescription()," #",order.Ticket()," ",TextByLanguage("не удалось добавить в список","failed to add to the list"));
      delete order;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

**Replace the order time in milliseconds with the order time**
in the method for creating control orders and adding them to the list (for the same reason):

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
   order_control.SetTime(order.TimeOpen());
   order_control.SetTimePrev(order.TimeOpen());
   order_control.SetVolume(order.Volume());
   order_control.SetTime(order.TimeOpen());
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

**In the** **HistoryCollection.mqh file** **of the method for selecting orders by time of the**
**CHistoryCollection collection class of historical orders and deals,**

**improve selection of the compared property**.

Since we
have already had a selection of four properties (open time in milliseconds, close time in milliseconds, open time in seconds and close time
in seconds) and now we have removed two of them, the selection is now simplified:

```
//+------------------------------------------------------------------+
//| Select orders from the collection with time                      |
//| from begin_time to end_time                                      |
//+------------------------------------------------------------------+
CArrayObj *CHistoryCollection::GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                             const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_CLOSE)
  {
   ENUM_ORDER_PROP_INTEGER property=(select_time_mode==SELECT_BY_TIME_CLOSE ? ORDER_PROP_TIME_CLOSE : ORDER_PROP_TIME_OPEN);

   CArrayObj *list=new CArrayObj();
   if(list==NULL)
     {
      ::Print(DFUN+TextByLanguage("Ошибка создания временного списка","Error creating temporary list"));
      return NULL;
     }
   datetime begin=begin_time,end=(end_time==0 ? END_TIME : end_time);
   if(begin_time>end_time) begin=0;
   list.FreeMode(false);
   ListStorage.Add(list);
   //---
   this.m_order_instance.SetProperty(property,begin);
   int index_begin=this.m_list_all_orders.SearchGreatOrEqual(&m_order_instance);
   if(index_begin==WRONG_VALUE)
      return list;
   this.m_order_instance.SetProperty(property,end);
   int index_end=this.m_list_all_orders.SearchLessOrEqual(&m_order_instance);
   if(index_end==WRONG_VALUE)
      return list;
   for(int i=index_begin; i<=index_end; i++)
      list.Add(this.m_list_all_orders.At(i));
   return list;
  }
//+------------------------------------------------------------------+
```

In the **CEngine.mqh** file, replace all instances of the SORT\_BY\_ORDER\_TIME\_OPEN\_MSC time constant in milliseconds with the
SORT\_BY\_ORDER\_TIME\_OPEN time constant. Now the time in milliseconds is used by this constant by default.

This concludes improvements of the library files related to removing time in seconds.

### Implementing position closing events for MQL4

Various experiments and tests conducted while searching for possible options of identifying the occurrence of _position_
_closure and order removal events in MQL4_ have turned out to be discouraging. Unlike MQL5, MQL4 features much less data that can be used
for decisive event identification.

In MQL5, we can easily use data on orders belonging to a certain position and define an event, while in MQL4 both a position and a pending
order are considered as orders. If we had a pending order in MetaTrader 4, then after its removal, we will face with:

- increase in the number of historical orders
- decrease in total volume
- number of market orders remaining unchanged.


To make sure an event belongs to removing a pending order (position in MQL4 is an order as well), check the number of open positions. If it has
not changed, the action has been performed with a pending order. Everything seems to be normal and logical here until we close any of the open
positions (an order) in MetaTrader 4 partially. When closing a position partially, we have the same thing as when removing a pending order:

- increase in the number of historical orders (the partially closed position (its order) has got into history),

- decrease in the account volume since we have partially closed the position,

- the number of open positions has not decreased — it remains the same when closing a position partially.


This is the same state as when removing a pending order. We can observe this when launching a [test \\
EA from the previous article](https://www.mql5.com/en/articles/6767) in the tester. Open a position, close it partially, set a pending order and remove it. During the last
action, two entries appear in the journal simultaneously: partial position closure and pending order removal. I have already mentioned
the conformity of criteria for defining these two events above. The program simply defines two events at once, and one of them is incorrect.

Here we can use the check of the changed number of pending orders — when defining a partial position closure, we will also check changing
the number of market pending orders. If it has not changed, the event is a partial position closure.

Everything seems logical, but this approach imposes restrictions on the allowed order of trading operations in MetaTrader 4. In other words, we
cannot remove a pending order and partially close a position in a single loop as this violates the event-defining logic mentioned
above. We can implement a solution to bypass this limitation but it requires reworking the classes of market and historic collection of
orders and positions. To define a change in the market environment, we need to use temporary lists of orders and positions that were
changed on the account rather than manage the amount of occurred changes. The events should be handled according to the data of these
lists. In that case, each event type has its own list and creating events for sending them to the program is to be carried out according to
the lists of changed orders and positions.



Perhaps, upon completion of this article series, I will arrange such a search and event handling. But for now, let's use the
control of the number of market pending orders. Keep in mind such a limitation for MetaTrader 4 and develop the functions of
closing/removing orders taking it into account. For end users, this limitation remains hidden. They are not supposed to manage it, as
the functions for working with the library considering that limitation for MQL4 are to be introduced.



To define a partial closure, we need to manage the total volume on the account. This means we need to pass its change value to the Refresh()
method of the CEventsCollection class. As usual, all starts from the basic object of the library. Let's make the necessary addition to the
call of the method for updating the event collection class.

**Add an additional transferable parameter to the Refresh**
**method of the CEventsCollection class in the CEngine::TradeEventsControl() class**
**method:**

```
//+------------------------------------------------------------------+
//| Check trading events                                             |
//+------------------------------------------------------------------+
void CEngine::TradeEventsControl(void)
  {
//--- Initialize the trading events code and flags
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
      this.m_events.Refresh(this.m_history.GetList(),this.m_market.GetList(),list_changes,this.m_market.GetListControl(),
                            this.m_is_history_trade_event,this.m_is_market_trade_event,
                            this.m_history.NewOrders(),this.m_market.NewPendingOrders(),
                            this.m_market.NewPositions(),this.m_history.NewDeals(),
                            this.m_market.ChangedVolumeValue());
      //--- Get the account's last trading event
      this.m_acc_trade_event=this.m_events.GetLastTradeEvent();
     }
  }
//+------------------------------------------------------------------+
```

Now we need to change the Refresh() method of the CEventsCollection class itself by implementing yet another parameter (volume change
value) to it.

**Add the new parameter to the definition of the Refresh()**
**method in the EventsCollection.mqh file:**

```
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
                             CArrayObj* list_control,
                             const bool is_history_event,
                             const bool is_market_event,
                             const int  new_history_orders,
                             const int  new_market_pendings,
                             const int  new_market_positions,
                             const int  new_deals,
                             const double changed_volume);
//--- Set the control program chart ID
   void              SetChartID(const long id)        { this.m_chart_id=id;         }
//--- Return the last trading event on the account
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)    const { return this.m_trade_event;  }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)        { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
  };
```

We need to make additions in the implementation of the Refresh()
method:

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals,
                                const double changed_volume)
  {
```

Now we need to create the position closure and partial closure events handler and solve the issue of the incorrect defining of a pending order
removal event during a partial position closure.

In the private class section, change the first form of calling the new
event creation method by adding passing the list of control orders to the
method. We will need it to identify orders taking part in closing one position by an opposite one. Also, add the method
returning the list of historical (closed) positions and the method
returning the pointer to the control order by the position ticket:

```
class CEventsCollection : public CListObj
  {
private:
   CListObj          m_list_events;                   // List of events
   bool              m_is_hedge;                      // Hedge account flag
   long              m_chart_id;                      // Control program chart ID
   int               m_trade_event_code;              // Trading event code
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property
   MqlTick           m_tick;                          // Last tick structure
   ulong             m_position_id;                   // Position ID (MQL4)
   ENUM_ORDER_TYPE   m_type_first;                    // Opening order type (MQL4)

//--- Create a trading event depending on the order (1) status and (2) change type
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market,CArrayObj* list_control);
   void              CreateNewEvent(COrderControl* order);
//--- Create an event for a (1) hedging account, (2) netting account
   void              NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
   void              NewDealEventNetto(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
//--- Select from the list and return the list of (1) market pending orders, (2) open positions
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
   CArrayObj*        GetListPositions(CArrayObj* list);
//--- Select from the list and return the list of historical (1) closed orders,
//--- (2) removed pending orders, (3) deals, (4) all closing orders
   CArrayObj*        GetListHistoryPositions(CArrayObj* list);
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListCloseByOrders(CArrayObj* list);
//--- Return the list of (1) all position orders by its ID, (2) all position deals by its ID
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
//--- Return the (1) control order by ticket, (2) opening order type by position ticket (MQL4)
   COrderControl*    GetOrderControlByTicket(CArrayObj* list,const ulong ticket);
   ENUM_ORDER_TYPE   GetTypeFirst(CArrayObj* list,const ulong ticket);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);
//--- Existing order/position change event handler
   void              OnChangeEvent(CArrayObj* list_changes,const int index);

public:
```

**Implement the method returning the list of closed positions beyond the class body:**

```
//+------------------------------------------------------------------+
//| Select only closed positions from the list                       |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListHistoryPositions(CArrayObj *list)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_ORDER,EQUAL);
   return list_orders;
  }
//+------------------------------------------------------------------+
```

The method contains nothing new for us. I have already described the similar methods in the previous parts of the library description.

**Let's implement the method returning a control order by a ticket and**
**change the**

**method returning the type of a control order by a ticket:**

```
//+------------------------------------------------------------------+
//| Return a control order by a position ticket (MQL4)               |
//+------------------------------------------------------------------+
COrderControl* CEventsCollection::GetOrderControlByTicket(CArrayObj *list,const ulong ticket)
  {
   if(list==NULL)
      return NULL;
   int total=list.Total();
   for(int i=0;i<total;i++)
     {
      COrderControl* ctrl=list.At(i);
      if(ctrl==NULL)
         continue;
      if(ctrl.Ticket()==ticket)
         return ctrl;
     }
   return NULL;
  }
//+------------------------------------------------------------------+
//| Return an opening order type by a position ticket (MQL4)         |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE CEventsCollection::GetTypeFirst(CArrayObj* list,const ulong ticket)
  {
   if(list==NULL)
      return WRONG_VALUE;
   COrderControl* ctrl=this.GetOrderControlByTicket(list,ticket);
   if(ctrl==NULL)
      return WRONG_VALUE;
   return (ENUM_ORDER_TYPE)ctrl.TypeOrder();
  }
//+------------------------------------------------------------------+
```

Like the previously considered similar methods, the method returning a control order by a ticket is simple. It receives the list of control
orders and the ticket of the position whose control order we want to obtain.

Take the order from the list and compare it with the ticket passed to the method in a loop by the list size. If the tickets are equal, return
the pointer to the control order, otherwise return NULL.

The GetTypeFirst() method returning the control order type
previously consisted of a loop by the list of control orders with the search of an order with the ticket equal to the one passed to the method.
When such an order was detected, its type was returned.

Now that we have the method returning a control order by a position ticket, we can delete the search loop from the GetTypeFirst() method.
This is exactly what I have done. Now in the method,

we obtain a control order by a position ticket using the GetOrderControlByTicket()
method. If it is obtained successfully (not NULL), return a type of an obtained
order, otherwise return
-1.

**Now we may add handling position closure for**
**MQL4 to the event collection update method:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals,
                                const double changed_volume)
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
      //--- if the number of placed pending orders increased (MQL5, MQL4)
      if(new_market_pendings>0)
        {
         //--- Receive the list of the newly placed pending orders
         CArrayObj* list=this.GetListMarketPendings(list_market);
         if(list!=NULL)
           {
            //--- Sort the new list by order placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN);
            //--- Take the number of orders equal to the number of newly placed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_market_pendings;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list, if this is a pending order, set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_MARKET_PENDING)
                  this.CreateNewEvent(order,list_history,list_market,list_control);
              }
           }
        }
      #ifdef __MQL4__
         //--- If the number of positions increased (MQL4)
         if(new_market_positions>0)
           {
            //--- Get the list of open positions
            CArrayObj* list=this.GetListPositions(list_market);
            if(list!=NULL)
              {
               //--- Sort the new list by a position open time
               list.Sort(SORT_BY_ORDER_TIME_OPEN);
               //--- Take the number of positions equal to the number of newly placed open positions from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_market_positions;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive a position from the list. If this is a position, search for opening order data and set a trading event
                  COrder* position=list.At(i);
                  if(position!=NULL && position.Status()==ORDER_STATUS_MARKET_POSITION)
                    {
                     //--- Find an order and set (1) a type of an order that led to opening a position and a (2) position ID
                     this.m_type_first=this.GetTypeFirst(list_control,position.Ticket());
                     this.m_position_id=position.Ticket();
                     this.CreateNewEvent(position,list_history,list_market,list_control);
                    }
                 }
              }
           }
         //--- If the number of positions decreased or a position is closed partially (MQL4)
         else if(new_market_positions<0 || (new_market_positions==0 && changed_volume<0 && new_history_orders>0 && new_market_pendings>WRONG_VALUE))
           {
            //--- Get the list of closed positions
            CArrayObj* list=this.GetListHistoryPositions(list_history);
            if(list!=NULL)
              {
               //--- Sort the new list by position close time
               list.Sort(SORT_BY_ORDER_TIME_CLOSE);
               //--- Take the number of positions equal to the number of newly closed positions from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_history_orders;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive an order from the list. If this is a position, look for data of an opening order and set a trading event
                  COrder* position=list.At(i);
                  if(position!=NULL && position.Status()==ORDER_STATUS_HISTORY_ORDER)
                    {
                     //--- If there is a control order of a closed position
                     COrderControl* ctrl=this.GetOrderControlByTicket(list_control,position.Ticket());
                     if(ctrl!=NULL)
                       {
                        //--- Set an (1) order type that led to a position opening, (2) position ID and create a position closure event
                        this.m_type_first=(ENUM_ORDER_TYPE)ctrl.TypeOrder();
                        this.m_position_id=position.Ticket();
                        this.CreateNewEvent(position,list_history,list_market,list_control);
                       }
                    }
                 }
              }
           }
      #endif
     }
//--- If an event in an account history
   if(is_history_event)
     {
      //--- If the number of historical orders increased (MQL5, MQL4)
      if(new_history_orders>0)
        {
         //--- Get the list of newly removed pending orders
         CArrayObj* list=this.GetListHistoryPendings(list_history);
         if(list!=NULL)
           {
            //--- Sort the new list by order removal time
            list.Sort(SORT_BY_ORDER_TIME_CLOSE);
            //--- Take the number of orders equal to the number of newly removed ones from the end of the list in a loop (the last N events)
            int total=list.Total(), n=new_history_orders;
            for(int i=total-1; i>=0 && n>0; i--,n--)
              {
               //--- Receive an order from the list. If this is a removed pending order without a position ID,
               //--- this is an order removal - set a trading event
               COrder* order=list.At(i);
               if(order!=NULL && order.Status()==ORDER_STATUS_HISTORY_PENDING && order.PositionID()==0)
                  this.CreateNewEvent(order,list_history,list_market,list_control);
              }
           }
        }
      //--- If the number of deals increased (MQL5)
      #ifdef __MQL5__
         if(new_deals>0)
           {
            //--- Receive the list of deals only
            CArrayObj* list=this.GetListDeals(list_history);
            if(list!=NULL)
              {
               //--- Sort the new list by deal time
               list.Sort(SORT_BY_ORDER_TIME_OPEN);
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
      #endif
     }
  }
//+------------------------------------------------------------------+
```

Handling position closure is quite clear and transparent. Its description is commented in the listing, and there is no point in dwelling on its
logic as the code comments are detailed thoroughly enough.

Since one more list is now passed to the first form of calling the method for creating a new event, **add**
**passing the list of control orders to the call of the method**

**for MQL5 in the Refresh() method of the CEventsCollection**
**event collection class:**

```
      //--- If the number of deals increased (MQL5)
      #ifdef __MQL5__
         if(new_deals>0)
           {
            //--- Receive the list of deals only
            CArrayObj* list=this.GetListDeals(list_history);
            if(list!=NULL)
              {
               //--- Sort the new list by deal time
               list.Sort(SORT_BY_ORDER_TIME_OPEN);
               //--- Take the number of deals equal to the number of new ones from the end of the list in a loop (the last N events)
               int total=list.Total(), n=new_deals;
               for(int i=total-1; i>=0 && n>0; i--,n--)
                 {
                  //--- Receive a deal from the list and set a trading event
                  COrder* order=list.At(i);
                  if(order!=NULL)
                     this.CreateNewEvent(order,list_history,list_market,list_control);
                 }
              }
           }
      #endif
```

Now let's add the code of creating a position closure event to the first form of an event call in the event creation method:

**CEventsCollection::CreateNewEvent(COrder\* order,CArrayObj\* list\_history,CArrayObj\* list\_market,CArrayObj\* list\_control).**

Since the method is quite bulky, we will have a look only at the code for creating a position closure event for
MQL4:

```
//--- Position closed (__MQL4__)
   if(status==ORDER_STATUS_HISTORY_ORDER)
     {
      //--- Set the "position closed" trading event code
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
      //--- Set the "request executed in full" reason
      ENUM_EVENT_REASON reason=EVENT_REASON_DONE;
      //--- If the closure by StopLoss flag is set for an order, a position is closed by StopLoss
      if(order.IsCloseByStopLoss())
        {
         //--- set the "closure by StopLoss" reason
         reason=EVENT_REASON_DONE_SL;
         //--- add the StopLoss closure flag to the event code
         this.m_trade_event_code+=TRADE_EVENT_FLAG_SL;
        }
      //--- If the closure by TakeProfit flag is set for an order, a position is closed by TakeProfit
      if(order.IsCloseByTakeProfit())
        {
         //--- set the "closure by TakeProfit" reason
         reason=EVENT_REASON_DONE_TP;
         //--- add the TakeProfit closure flag to the event code
         this.m_trade_event_code+=TRADE_EVENT_FLAG_TP;
        }
      //--- If an order has the property with an inherited order filled, a position is closed partially
      if(order.TicketTo()>0)
        {
         //--- set the "partial closure" reason
         reason=EVENT_REASON_DONE_PARTIALLY;
         //--- add the partial closure flag to the event code
         this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
        }
      //--- Check closure by an opposite position
      COrder* order_close_by=this.GetCloseByOrderFromList(list_history,order.Ticket());
      //--- Declare the variables of the opposite order properties and initialize them with the values of the current closed position
      ENUM_ORDER_TYPE close_by_type=this.m_type_first;
      double close_by_volume=order.Volume();
      ulong  close_by_ticket=order.Ticket();
      long   close_by_magic=order.Magic();
      string close_by_symbol=order.Symbol();
      //--- If the list of historical orders features an order with a closed position ID, the position is closed by an opposite one
      if(order_close_by!=NULL)
        {
         //--- Fill in the properties of an opposite closing order using data on the opposite position properties
         close_by_type=(ENUM_ORDER_TYPE)order_close_by.TypeOrder();
         close_by_ticket=order_close_by.Ticket();
         close_by_magic=order_close_by.Magic();
         close_by_symbol=order_close_by.Symbol();
         close_by_volume=order_close_by.Volume();
         //--- set the "close by" reason
         reason=EVENT_REASON_DONE_BY_POS;
         //--- add the close by flag to the event code
         this.m_trade_event_code+=TRADE_EVENT_FLAG_BY_POS;
         //--- Take data on (1) closed and (2) opposite positions from the list of control orders
         //--- (in this list, the properties of two opposite positions remain the same as before the closure)
         COrderControl* ctrl_closed=this.GetOrderControlByTicket(list_control,order.Ticket());
         COrderControl* ctrl_close_by=this.GetOrderControlByTicket(list_control,close_by_ticket);
         double vol_closed=0;
         double vol_close_by=0;
         //--- If no errors detected when receiving these two opposite orders
         if(ctrl_closed!=NULL && ctrl_close_by!=NULL)
           {
            //--- Calculate closed volumes of a (1) closed and (2) an opposite positions
            vol_closed=ctrl_closed.Volume()-order.Volume();
            vol_close_by=vol_closed-close_by_volume;
            //--- If a position is closed partially (the previous volume exceeds the currently closed one)
            if(ctrl_closed.Volume()>order.Volume())
              {
               //--- add the partial closure flag to an event code
               this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
               //--- set the "partial closure" reason
               reason=EVENT_REASON_DONE_PARTIALLY_BY_POS;
              }
           }
        }
      //--- Create the position closure event
      CEvent* event=new CEventPositionClose(this.m_trade_event_code,order.Ticket());
      if(event!=NULL && order.PositionByID()==0)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeClose());                               // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                        // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,close_by_type);                              // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                           // Event deal ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,close_by_type);                             // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,this.m_type_first);                      // Type of an order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,close_by_ticket);                         // Ticket of an order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                       // Ticket of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,this.m_position_id);                             // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,close_by_ticket);                             // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,close_by_magic);                                 // Opposite position magic number

         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());                      // Position order type before direction changed
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());                       // Position order ticket before direction changed
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());                     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());                      // Current position order ticket

         event.SetProperty(EVENT_PROP_PRICE_OPEN_BEFORE,order.PriceOpen());                        // Order price before modification<
         event.SetProperty(EVENT_PROP_PRICE_SL_BEFORE,order.StopLoss());                           // StopLoss before modification
         event.SetProperty(EVENT_PROP_PRICE_TP_BEFORE,order.TakeProfit());                         // TakeProfit before modification
         event.SetProperty(EVENT_PROP_PRICE_EVENT_ASK,this.m_tick.ask);                            // Ask price during an event
         event.SetProperty(EVENT_PROP_PRICE_EVENT_BID,this.m_tick.bid);                            // Bid price during an event

         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                                  // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpen());                       // Time of an order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                              // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                               // Order/deal/position open price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                             // Order/deal/position close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                                  // StopLoss position price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                                // TakeProfit position price
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());                        // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order.VolumeCurrent());                 // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,order.Volume());                    // Executed position volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                                      // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                                      // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,close_by_symbol);                               // Opposite position symbol
         //--- Set control program chart ID, decode the event code and set the event type
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
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event is already in the list."));
            delete event;
           }
        }
     }
#endif
```

Since we have already described the logic of creating events in the previous articles, let's mention it briefly: first, set an event code as
"position closure" and an event reason as "request executed in full". Next view the various properties of a closed order, add the necessary
flags to the event code based on these properties and change the event reason if necessary. When defining an opposite position closure, we
use data of control orders on closed positions. This data provides us with the entire info on opposite orders before their joint closure
allowing us to define order types and their volume, as well as find out which one initiated a close by.

Then all collected data is recorded in the event properties, and a new position closure event is created.

I have **also improved the method returning** **the list of all position's closing orders for MQL4:**

```
//+------------------------------------------------------------------+
//|  Return the list of all closing CloseBy orders from the list     |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListCloseByOrders(CArrayObj *list)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of history collection"));
      return NULL;
     }
#ifdef __MQL5__
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,ORDER_TYPE_CLOSE_BY,EQUAL);
#else
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_BY_ID,0,NO_EQUAL);
#endif
   return list_orders;
  }
//+------------------------------------------------------------------+
```

For MQL5, we select only orders of ORDER\_TYPE\_CLOSE\_BY type from the
list of historical orders. Since there are no such orders

in MQL4, we select only the orders with the opposite
position ID property filled, more specifically, orders that have this property not
equal to zero.

The **method returning the last closing position order for MQL4**
**has also been improved:**

```
//+------------------------------------------------------------------+
//| Return the last closing order                                    |
//| from the list of all position orders                             |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetCloseByOrderFromList(CArrayObj *list,const ulong position_id)
  {
#ifdef __MQL5__
   CArrayObj* list_orders=this.GetListAllOrdersByPosID(list,position_id);
   list_orders=CSelect::ByOrderProperty(list_orders,ORDER_PROP_TYPE,ORDER_TYPE_CLOSE_BY,EQUAL);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   list_orders.Sort(SORT_BY_ORDER_TIME_OPEN);
#else
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_POSITION_BY_ID,position_id,EQUAL);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   list_orders.Sort(SORT_BY_ORDER_TIME_CLOSE);
#endif
   COrder* order=list_orders.At(list_orders.Total()-1);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

In MQL5, we can obtain the last order belonging to a position at any time. However, this is not the case with MQL4. Therefore, in case of MQL4,
I decided to return an

opposite position order if present or NULL
if a position was not closed by an opposite one. This allows us to partially implement obtaining a closing order when a position is closed by an
opposite one.

**These are all the changes and improvements necessary for defining a position closure for MQL4.**

Find the full listings of all classes in the files attached below.

### Testing

To perform the test, we will use the [test EA from the previous article](https://www.mql5.com/en/articles/6767#node01)
TestDoEasyPart10.mq4 located at \\MQL4\\Experts\\TestDoEasy\\Part10 and save it in the new folder

**\\MQL4\\Experts\\TestDoEasy\\Part11** under the name **TestDoEasyPart11.mq4**.

Since we removed time constants in milliseconds from the ENUM\_SORT\_ORDERS\_MODE enumeration of possible order and deal sorting criteria, we
need to replace sorting by placing time where a removed constant is set

```
list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
```

with sorting by time in the EA's handler of the pending order
removal button pressing:

```
      //--- If the BUTT_DELETE_PENDING button is pressed: Remove the first pending order
      else if(button==EnumToString(BUTT_DELETE_PENDING))
        {
         //--- Get the list of all orders
         CArrayObj* list=engine.GetListMarketPendings();
         if(list!=NULL)
           {
            //--- Sort the list by placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN);
            int total=list.Total();
            //--- In the loop from the position with the most amount of time
            for(int i=total-1;i>=0;i--)
              {
               COrder* order=list.At(i);
               if(order==NULL)
                  continue;
               //--- delete the order by its ticket
               #ifdef __MQL5__
                  trade.OrderDelete(order.Ticket());
               #else
                  PendingOrderDelete(order.Ticket());
               #endif
              }
           }
        }
```

Compile the EA and set the **StopLoss in points** and **TakeProfit in points** inputs in the tester to zero so that
positions are closed without stop orders. Then launch the EA in the tester, open a position and close it partially.

Next place and remove a pending order:

![](https://c.mql5.com/2/36/2019-05-19_00-35-54.gif)

Now the events of partial closure and pending order removal are defined as separate events.

Launch the EA once again and click the buttons observing the definition of events:

![](https://c.mql5.com/2/36/2019-05-19_00-42-06.gif)

As we can see, the events are defined correctly. A close by event is defined, modifications of stop levels and pending order prices are also
tracked.

### What's next?

In this article, we finished the conversion of the existing library functionality for compatibility with MQL4. In the coming articles, we
will create new "account" and "symbol" objects, their collection and events.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6921#node00)

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

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events.](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events.](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 - Preparing data.](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending \\
orders.](https://www.mql5.com/en/articles/6767)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6921](https://www.mql5.com/ru/articles/6921)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6921.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6921/mql5.zip "Download MQL5.zip")(100.78 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/6921/mql4.zip "Download MQL4.zip")(100.79 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/320629)**
(2)


![Alvaro Arioni](https://c.mql5.com/avatar/2020/8/5F4C397C-E698.jpg)

**[Alvaro Arioni](https://www.mql5.com/en/users/adarioni)**
\|
4 Sep 2020 at 20:11

I have some remarks related to the effect of the millisecond time variables, and particularly about the **CHistoryCollection::GetListByTime(…)** [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") regarding MQL5 version.

They are ( in decreasing order of relevance):

> **1)** Both datetime variables and long millisecond time variables represent the time elapsed since 01/01/1970. However, there is a factor of 1000 between the values effectively stored. Therefore, both types cannot be directly assigned to each other. As a consequence **GetListByTime(..)** stopped working after the change, since it directly stores datetime parameters in time millisecond property of **COrder m\_order\_instance**. A possible solution could be to include two new conversion functions (probably in **delib.mqh**) to take care of that 1000 factor:
>
> ```
> //+------------------------------------------------------------------+
> //| Converts time in milliseconds to datetime                        |
> //+------------------------------------------------------------------+
> datetime TimeMSCtoDate(const long time_msc)
>   {
>    return datetime(time_msc/1000);
>   }
> //+------------------------------------------------------------------+
> //| Converts datetime to time in milliseconds                        |
> //+------------------------------------------------------------------+
> long DatetoTimeMSC(const datetime time_sec)
>   {
>    return long(time_sec * 1000);
>   }
> ```

> This type of problem wouldn’t happen if MetaQuotes had defined form the beginning datetime variables to be the elapsed time in milliseconds, or even in microseconds (who says that in ten years robots will not be trading in less than millisecons?).

> Anyway, the problem is also present at least in **GetListByTime(…)** functions in **CEventsCollection** and **CMarketCollection** classes, perhaps in some other places…

> **2)** The History collection is an object which knows whether it is sorted or not, and if it is, which is the sorting property (. **SortMode()** function), while **GetListByTime(…)** function assumes that the collection is _already_ sorted by **ORDER\_PROP\_TIME\_OPEN** or **ORDER\_PROP\_TIME\_CLOSE** properties (just **ORDER\_PROP\_TIME\_OPEN** in the **CEventsCollection** and **CMarketCollection** versions). Following OOP philosophy, **GetListByTime(…)** should check if the collection is properly sorted before going on.

> As a matter of fact, the program example **TestDoEasyPart03\_1.mq5** created a history collection ordered by **ORDER\_PROP\_TIME\_OPEN** (the default in MT5) and then called **GetListByTime(…,SELECT\_BY\_TIME\_CLOSE)**. With that check, this would have shown up!

> **3)** Not great deal, but since MT5 is supposed to be the new standard, why don’t let **SELECT\_BY\_TIME\_OPEN** be the default value of **select\_time\_mode** parameter of **CHistoryCollection::GetListByTime(…)** function?

Following the former suggestions, this would be the code of **CHistoryCollection::GetListByTime(…)** function:

```
public:
        .......

   //--- Select orders from the collection with time from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                   const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_OPEN);
        .......

//+------------------------------------------------------------------+
//| Select orders from the collection with time                      |
//| from begin_time to end_time                                      |
//+------------------------------------------------------------------+
CArrayObj *CHistoryCollection::GetListByTime(const datetime begin_time=0,const datetime end_time=0,
                                             const ENUM_SELECT_BY_TIME select_time_mode=SELECT_BY_TIME_OPEN)
  {
   ENUM_ORDER_PROP_INTEGER property=(select_time_mode==SELECT_BY_TIME_CLOSE ? ORDER_PROP_TIME_CLOSE : ORDER_PROP_TIME_OPEN);

   if(property != (ENUM_ORDER_PROP_INTEGER)m_list_all_orders.SortMode())
      {
      ::Print(DFUN+"History list not prpperly sorted");           //Perhaps message to add to ENUM_MESSAGES_LIB
      return NULL;
      }

   CArrayObj *list=new CArrayObj();
   if(list==NULL)
     {
      ::Print(DFUN+CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TEMP_LIST));
      return NULL;
     }
   datetime begin=begin_time,end=(end_time==0 ? END_TIME : end_time);
   if(begin_time>end_time) begin=0;
   list.FreeMode(false);
   ListStorage.Add(list);
   //---
   this.m_order_instance.SetProperty(property,DatetoTimeMSC(begin));
   int index_begin=this.m_list_all_orders.SearchGreatOrEqual(&m_order_instance);
   if(index_begin==WRONG_VALUE)
      return list;
   this.m_order_instance.SetProperty(property,DatetoTimeMSC(end));
   int index_end=this.m_list_all_orders.SearchLessOrEqual(&m_order_instance);
   if(index_end==WRONG_VALUE)
      return list;
   for(int i=index_begin; i<=index_end; i++)
      list.Add(this.m_list_all_orders.At(i));
   return list;
  }
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Sep 2020 at 08:27

**Alvaro Arioni :**

I have some remarks related to the effect of the millisecond time variables, and particularly about the **CHistoryCollection::GetListByTime(…)** [function](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") regarding MQL5 version.

They are ( in decreasing order of relevance):

> **1)** Both datetime  variables and long  millisecond time variables represent the time elapsed since 01/01/1970. However, there is a factor of 1000 between the values effectively stored. Therefore, both types cannot be directly assigned to each other. As a consequence **GetListByTime(..)** stopped working after the change, since it directly stores datetime  parameters in time millisecond property of **COrder m\_order\_instance** . A possible solution could be to include two new conversion functions (probably in **delib.mqh** ) to take care of that 1000 factor:

> This type of problem wouldn’t happen if MetaQuotes had defined form the beginning datetime  variables to be the elapsed time in milliseconds, or even in microseconds (who says that in ten years robots will not be trading in less than millisecons?).

> Anyway, the problem is also present at least in **GetListByTime(…)** functions in **CEventsCollection** and **CMarketCollection** classes, perhaps in some other places…

> **2)** The History collection is an object which knows whether it is sorted or not, and if it is, which is the sorting property (. **SortMode()** function), while **GetListByTime(…)** function assumes that the collection is _already_ sorted by **ORDER\_PROP\_TIME\_OPEN** or **ORDER\_PROP\_TIME\_CLOSE** properties (just **ORDER\_PROP\_TIME\_OPEN** in the **CEventsCollection** and **CMarketCollection** versions). Following OOP philosophy, **GetListByTime(…)** should check if the collection is properly sorted before going on.

> As a matter of fact, the program example **TestDoEasyPart03\_1.mq5** created a history collection ordered by **ORDER\_PROP\_TIME\_OPEN** (the default in MT5) and then called **GetListByTime(…,SELECT\_BY\_TIME\_CLOSE)** . With that check, this would have shown up!

> **3)** Not great deal, but since MT5 is supposed to be the new standard, why don’t let **SELECT\_BY\_TIME\_OPEN** be the default value of **select\_time\_mode** parameter of **CHistoryCollection::GetListByTime(…)** function?

Following the former suggestions, this would be the code of **CHistoryCollection::GetListByTime(…)** function:

Thank. In subsequent versions of the library (in the articles above XI), it seems that time conversion is already used (I generally left datetime - everything is in long).

I did not understand a little about collections - please show an example of wrong sorting. Desirable in the most recent article available (since the articles describe the sequence of creating a library, and everything changes)

![Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__7.png)[Library for easy and quick development of MetaTrader programs (part XII): Account object class and collection of account objects](https://www.mql5.com/en/articles/6952)

In the previous article, we defined position closure events for MQL4 in the library and got rid of the unused order properties. Here we will consider the creation of the Account object, develop the collection of account objects and prepare the functionality for tracking account events.

![Optimization management (Part I): Creating a GUI](https://c.mql5.com/2/36/mql5-avatar-opt_control.png)[Optimization management (Part I): Creating a GUI](https://www.mql5.com/en/articles/7029)

This article describes the process of creating an extension for the MetaTrader terminal. The solution discussed helps to automate the optimization process by running optimizations in other terminals. A few more articles will be written concerning this topic. The extension has been developed using the C# language and design patterns, which additionally demonstrates the ability to expand the terminal capabilities by developing custom modules, as well as the ability to create custom graphical user interfaces using the functionality of a preferred programming language.

![Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__8.png)[Library for easy and quick development of MetaTrader programs (part XIII): Account object events](https://www.mql5.com/en/articles/6995)

The article considers working with account events for tracking important changes in account properties affecting the automated trading. We have already implemented some functionality for tracking account events in the previous article when developing the account object collection.

![Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://c.mql5.com/2/36/mql5_ea_adviser_grid__1.png)[Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)

In this article, we will make an attempt to develop the best possible grid-based EA. As usual, this will be a cross-platform EA capable of working both with MetaTrader 4 and MetaTrader 5. The first EA was good enough, except that it could not make a profit over a long period of time. The second EA could work at intervals of more than several years. Unfortunately, it was unable to yield more than 50% of profit per year with a maximum drawdown of less than 50%.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/6921&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070477826949322382)

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