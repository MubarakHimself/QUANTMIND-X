---
title: Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)
url: https://www.mql5.com/en/articles/7394
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:48:29.220921
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kopyfbyruswctspkcoyjxqhjvwdhugyq&ssn=1769158106630473977&ssn_dr=0&ssn_sr=0&fv_date=1769158106&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7394&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXVI)%3A%20Working%20with%20pending%20trading%20requests%20-%20first%20implementation%20(opening%20positions)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915810644415704&fz_uniq=5062734129763624921&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7394#node01)
- [Using a magic number as a data storage](https://www.mql5.com/en/articles/7394#node02)
- [Pending request class, first implementation of requests](https://www.mql5.com/en/articles/7394#node03)
- [Testing](https://www.mql5.com/en/articles/7394#node04)
- [What's next?](https://www.mql5.com/en/articles/7394#node05)


### Concept

I have already mentioned the concept of a pending request in a number of previous articles.

In this article, we are
going to figure out what it is and why we need it, as well as start implementing pending requests.

When receiving and handling a trade server error, we sometimes need to wait and repeat the request. In the simplest case, waiting is achieved by
performing the [Sleep()](https://www.mql5.com/en/docs/common/sleep) function with the necessary number of milliseconds
and repeating the request. This is sufficient in many programs. However, during the waiting, the program stops and waits for the pause
completion before resuming its logic. All this happens in the trading method, and all other program functions are inaccessible.

To
avoid this drawback, we can create a copy of a trading request causing the error requiring a repeated request after waiting, place that
request to the list of trading requests and exit the trading method. This allows us to free the program from the need to "hang" waiting inside
the trading method and enable it to resume the work according to its built-in logic. The trading class constantly views the list of pending
requests. When it is time to execute a request (its waiting time is over), the library calls the necessary trading method with the
appropriate trading request. Then everything happens in the same sequence - if we get an error again, exit the trading method before the next
trading request. If the request is executed without errors and queued on the server, the pending request is removed from the list of trading
requests.

This is one of the possible ways to apply pending requests allowing us not to interrupt the program execution while waiting, especially if
waiting takes a while.

Another way of applying pending requests is the implementation of StopLimit orders in MQL4. What is a StopLimit
order? It is a double pending order containing prices of Stop and Limit orders. A pending limit order is set after the price reaches the level
set for a Stop order. Thus, a pending request allows us to easily implement a StopLimit order operation logic: we simply create a pending
request to place a Limit order, while the moment of placing a Limit order (the price, at which a request should be made) is written to the
pending request parameters. As soon as the price reaches a specified value, a request for placing a Limit order is sent to the server. Such a
logic completely repeats the operation logic of StopLimit orders.

Another way to apply pending requests is to automatically send trading requests to open market positions when the price reaches a specified level
or a specified time, or both.

Generally, there are numerous options of automating the process of sending trading requests according to
the given conditions.

### Using a magic number as a data storage

When creating a new pending request, we need to mark it somehow so that the program knows that this is the exact order which was placed according
to this exact pending request. In other words, we need to accurately identify and associate the order or position with a specific pending
request. Moreover, this association should remain in emergency situations.

I thought about various possibilities of organizing
such an association for a long time and decided that the pending request ID should be set in the order/position magic number. This is the only
parameter remaining unchanged and being present in the order initially. All other methods are either unreliable (stored in an
order/position comment), or require extensive resources for creation and maintenance (storing in files). I do not consider the terminal
global variables as well, since they may still have not enough time to be written in case of emergency, and therefore also do not provide
complete confidence in the data relevance.

I can see only one solution — storing data in the magic number value. [Previously, we \\
added the group ID to order objects](https://www.mql5.com/en/articles/6383). That ID allows grouping orders/positions into a common group, which can be useful for various EAs
(for example, grid ones). We can add such an ID to the order object only after it physically appears on the server. It will be lost in emergency
situations. Of course, all collections are to be saved to HDD together with their objects but that will be done much later. Right now, we can
add group IDs to the order magic number along with the pending request ID.

We are able to store several IDs in the magic number value:

- magic number ID (set in the EA inputs)

- first group ID (with subgroup numbers from 0 to 15, zero — does not belong to the group)

- second group ID (with subgroup numbers from 0 to 15, zero — does not belong to the group)

- pending request ID (with possible numbers from 0 to 255, zero — no ID)

Thus, we will be able to set the number of the first and second order groups. Each order group may contain up to fifteen subgroups. How can that
be of use for us? Suppose that we have 20 orders/positions that we want to group into a single subgroup by a certain criterion. We set the
subgroup number of 1 for them in the first group. Besides, we have another 20 orders/positions we want to group to another subgroup of the
same first group according to some other criterion. We set the subgroup number of 2 for them. The subgroup 3 is assigned to another twenty
orders/positions. Now we have three groups of orders/positions we can easily handle together using the handler for each of the
subgroups of the first group. If we need to handle/analyze two groups out of three in some other way using another handler without losing
their affiliation to the already established groups, we can assign them to the second group (up to fifteen subgroups). This provides us
with more flexibility in combining orders/positions into different groups as compared to a single group, albeit having a g _r_ eater
quantity of subgroup numbers.

As we can see, we have planned a lot, but what is the catch here? The catch is in the fact that the size of the integer value the magic number
for MQL4 is stored in is only 4 bytes (int). Therefore, we have to sacrifice the magic number value we are able to set in custom programs.
For MQL5, the magic number size is set by the ulong type. We are able to set greater values and store much more data there. But there is also
an issue of compatibility, which means we have to sacrifice something, namely, the magic number size — it will be equal to only two bytes
(ushort), while the released two bytes are to be allocated for the IDs of two groups (uchar) and a pending order ID (uchar).

**The table displays the magic number structure and data location in the [uint \**\
**value](https://www.mql5.com/en/docs/basis/types/integer/integertypes#uint) of the magic number:**

```
-------------------------------------------------------------------------
| bit   |31           24|23   20|19   16|15            8|7             0|
-------------------------------------------------------------------------
| byte  |       3       |       2       |       1       |       0       |
-------------------------------------------------------------------------
| type  |     uchar     |     uchar     |            ushort             |
-------------------------------------------------------------------------
| descr |  request id   |  id2  |  id1  |             magic             |
-------------------------------------------------------------------------
```

As we can see from the table,

- the magic number value has the size of two bytes and is stored in the two
lower bytes of 0 and 1 of the [uint number](https://www.mql5.com/en/docs/basis/types/integer/integertypes#uint)
(bits 0 — 15) corresponding to the [ushort](https://www.mql5.com/en/docs/basis/types/integer/integertypes#ushort)
type. We will have to use this magic number type in our programs with the possible values from 0 to 65535.
- The next one in the hierarchy is the byte 2 of the uint number
used for storing two group IDs and having the [uchar](https://www.mql5.com/en/docs/basis/types/integer/integertypes#uchar)
size (bits 16 — 23).

The
first group ID is stored in the lower four bits of the uchar number (bits 16 —
19), while the second group ID is stored in the upper four bits of
this uchar number (bits 20 — 23).

Thus, we can
save two groups in a one-byte uchar value where the number of each of them may vary from zero (no group) to 15 (maximum value we can
store in four bits).
- In the third and the last uint number byte, we will store the uchar
value of a pending request ID (bits 24 — 31), which may
have the values varying from zero (no ID) to 255. This means that we may simultaneously have up to 255 active pending requests.

Let's improve the abstract order class and the event classes to store data in the "magic number" property value.


But first, **in Defines.mqh, add new integer properties of the abstract**
**order object:**

```
//+------------------------------------------------------------------+
//| Order, deal, position integer properties                         |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_INTEGER
  {
   ORDER_PROP_TICKET = 0,                                   // Order ticket
   ORDER_PROP_MAGIC,                                        // Real order magic number
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
   ORDER_PROP_MAGIC_ID,                                     // Order's "magic number" ID
   ORDER_PROP_GROUP_ID1,                                    // First order/position group ID
   ORDER_PROP_GROUP_ID2,                                    // Second order/position group ID
   ORDER_PROP_PEND_REQ_ID,                                  // Pending request ID
   ORDER_PROP_DIRECTION,                                    // Type by direction (Buy, Sell)
  };
#define ORDER_PROP_INTEGER_TOTAL    (24)                    // Total number of integer properties
#define ORDER_PROP_INTEGER_SKIP     (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

These properties are to store previously described IDs. The IDs are to be stored in the magic number value. Since we have added three new
properties and changed one, change the total number of order integer properties
from 21 to **24** in the appropriate macro substitution.

Also, **add sorting by these properties to the enumeration of possible**
**order and deal sorting criteria:**

```
//+------------------------------------------------------------------+
//| Possible criteria of sorting orders and deals                    |
//+------------------------------------------------------------------+
#define FIRST_ORD_DBL_PROP          (ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_INTEGER_SKIP)
#define FIRST_ORD_STR_PROP          (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL-ORDER_PROP_INTEGER_SKIP)
enum ENUM_SORT_ORDERS_MODE
  {
//--- Sort by integer properties
   SORT_BY_ORDER_TICKET = 0,                                // Sort by an order ticket
   SORT_BY_ORDER_MAGIC,                                     // Sort by an order magic number
   SORT_BY_ORDER_TIME_OPEN,                                 // Sort by an order open time in milliseconds
   SORT_BY_ORDER_TIME_CLOSE,                                // Sort by an order close time in milliseconds
   SORT_BY_ORDER_TIME_EXP,                                  // Sort by an order expiration date
   SORT_BY_ORDER_STATUS,                                    // Sort by an order status (market order/pending order/deal/balance and credit operation)
   SORT_BY_ORDER_TYPE,                                      // Sort by an order type
   SORT_BY_ORDER_REASON,                                    // Sort by a deal/order/position reason/source
   SORT_BY_ORDER_STATE,                                     // Sort by an order status
   SORT_BY_ORDER_POSITION_ID,                               // Sort by a position ID
   SORT_BY_ORDER_POSITION_BY_ID,                            // Sort by an opposite position ID
   SORT_BY_ORDER_DEAL_ORDER,                                // Sort by the order a deal is based on
   SORT_BY_ORDER_DEAL_ENTRY,                                // Sort by a deal direction – IN, OUT or IN/OUT
   SORT_BY_ORDER_TIME_UPDATE,                               // Sort by position change time in seconds
   SORT_BY_ORDER_TICKET_FROM,                               // Sort by a parent order ticket
   SORT_BY_ORDER_TICKET_TO,                                 // Sort by a derived order ticket
   SORT_BY_ORDER_PROFIT_PT,                                 // Sort by order profit in points
   SORT_BY_ORDER_CLOSE_BY_SL,                               // Sort by the flag of closing an order by StopLoss
   SORT_BY_ORDER_CLOSE_BY_TP,                               // Sort by the flag of closing an order by TakeProfit
   SORT_BY_ORDER_MAGIC_ID,                                  // Sort by an order/position "magic number" ID
   SORT_BY_ORDER_GROUP_ID1,                                 // Sort by the first order/position group ID
   SORT_BY_ORDER_GROUP_ID2,                                 // Sort by the second order/position group ID
   SORT_BY_ORDER_PEND_REQ_ID,                               // Sort by a pending request ID
   SORT_BY_ORDER_DIRECTION,                                 // Sort by direction (Buy, Sell)
//--- Sort by real properties
   SORT_BY_ORDER_PRICE_OPEN = FIRST_ORD_DBL_PROP,           // Sort by open price
   SORT_BY_ORDER_PRICE_CLOSE,                               // Sort by close price
   SORT_BY_ORDER_SL,                                        // Sort by StopLoss price
   SORT_BY_ORDER_TP,                                        // Sort by TakeProfit price
   SORT_BY_ORDER_PROFIT,                                    // Sort by profit
   SORT_BY_ORDER_COMMISSION,                                // Sort by commission
   SORT_BY_ORDER_SWAP,                                      // Sort by swap
   SORT_BY_ORDER_VOLUME,                                    // Sort by volume
   SORT_BY_ORDER_VOLUME_CURRENT,                            // Sort by unexecuted volume
   SORT_BY_ORDER_PROFIT_FULL,                               // Sort by profit+commission+swap
   SORT_BY_ORDER_PRICE_STOP_LIMIT,                          // Sort by Limit order when StopLimit order is activated
//--- Sort by string properties
   SORT_BY_ORDER_SYMBOL = FIRST_ORD_STR_PROP,               // Sort by symbol
   SORT_BY_ORDER_COMMENT,                                   // Sort by comment
   SORT_BY_ORDER_COMMENT_EXT,                               // Sort by custom comment
   SORT_BY_ORDER_EXT_ID                                     // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

To correctly display the order properties in the journal, **add the**
**indices of the new properties and the appropriate messages**
**in the Datas.mqh file:**

```
   MSG_ORD_PROFIT_PT,                                 // Profit in points
   MSG_ORD_MAGIC_ID,                                  // Magic number ID
   MSG_ORD_GROUP_ID1,                                 // First group ID
   MSG_ORD_GROUP_ID2,                                 // Second group ID
   MSG_ORD_PEND_REQ_ID,                               // Pending request ID
   MSG_ORD_PRICE_OPEN,                                // Open price

   {"Прибыль в пунктах","Profit in points"},
   {"Идентификатор магического номера","Magic number's identifier"},
   {"Идентификатор первой группы","First group's identifier"},
   {"Идентификатор второй группы","Second group's identifier"},
   {"Идентификатор отложенного запроса","Pending request's identifier"},
   {"Цена открытия","Price open"},
```

Add the necessary changes to the abstract order class in the **Order.mqh** file.

**In the private section of the class, add four methods retrieving the "magic number" from the order property value and returning the**
**ID of the magic number (the magic number set in the program settings), group**
**1 and2 IDs and the pending**
**request ID:**

```
//+------------------------------------------------------------------+
//| Abstract order class                                             |
//+------------------------------------------------------------------+
class COrder : public CObject
  {
private:
   ulong             m_ticket;                                    // Selected order/deal ticket (MQL5)
   long              m_long_prop[ORDER_PROP_INTEGER_TOTAL];       // Integer properties
   double            m_double_prop[ORDER_PROP_DOUBLE_TOTAL];      // Real properties
   string            m_string_prop[ORDER_PROP_STRING_TOTAL];      // String properties

//--- Return the index of the array the order's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_ORDER_PROP_DOUBLE property)      const { return(int)property-ORDER_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_ORDER_PROP_STRING property)      const { return(int)property-ORDER_PROP_INTEGER_TOTAL-ORDER_PROP_DOUBLE_TOTAL; }

//--- Data location in the magic number int value
      //-----------------------------------------------------------
      //  bit   32|31       24|23       16|15        8|7         0|
      //-----------------------------------------------------------
      //  byte    |     3     |     2     |     1     |     0     |
      //-----------------------------------------------------------
      //  data    |   uchar   |   uchar   |         ushort        |
      //-----------------------------------------------------------
      //  descr   |pend req id| id2 | id1 |          magic        |
      //-----------------------------------------------------------
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(void)                                const { return ushort(this.Magic() & 0xFFFF);                                 }
   uchar             GetGroupID1(void)                               const { return uchar(this.Magic()>>16) & 0x0F;                                }
   uchar             GetGroupID2(void)                               const { return uchar((this.Magic()>>16) & 0xF0)>>4;                           }
   uchar             GetPendReqID(void)                              const { return uchar(this.Magic()>>24) & 0xFF;                                }

public:
//--- Default constructor
                     COrder(void){;}
```

To return the ushort magic number value from the uint value of
the order magic number, apply the mask (0xFFFF) leaving only two lower bytes unchanged in the uint number, while two higher bytes of the uint
number are filled with zeros. When converting uint to ushort, the upper two bytes are discarded automatically.

To retrieve the first group ID, we first need to shift the magic
number property 16 bits right (so that the uchar value of group IDs makes it onto the uint number zero byte). The 0x0F mask is then applied to the
obtained number. The mask leaves only the lower four bits of the value obtained during the shift. Conversion of uint into uchar discards all
upper bytes of the number leaving a lower one the mask is applied to. Thus, we obtain a four-bit value from 0 to 15.

Retrieving the second group ID is different as the necessary
value is located in the upper four bits of the uchar value. Therefore, first we do the same as when retrieving the first group ID — shift the
value of the magic number property 16 bits right (so that the uchar value of the group IDs makes it onto the zero byte of the uint number) and
apply the mask of 0xF0 to the obtained number. The mask leaves only four upper bits of the value obtained during the shift. Next, the obtained
value is additionally shifted four bits right, so that the upper bits storing the ID number are corrected to 0 and 15.

To retrieve the pending request ID, shift the oldest byte of the
uint number 24 bits right, so that this one-byte uchar value makes it onto the uint number zero bite, and apply the 0xFF mask to it (which is, in
fact, not necessary since a single lower byte remains anyway when converting a uint number into uchar type).

**Add the methods returning the four new properties to the**
**block of methods for a simplified access to the abstract order object properties:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Return (1) ticket, (2) parent order ticket, (3) derived order ticket, (4) magic number, (5) order reason,
//--- (6) position ID, (7) opposite position ID, (8) first group ID, (9) second group ID,
//--- (10) pending request ID, (11) magic number ID, (12) type, (13) flag of closing by StopLoss,
//--- (14) flag of closing by TakeProfit (15) open time, (16) close time,
//--- (17) order expiration date, (18) state, (19) status, (20) type by direction
   long              Ticket(void)                                       const { return this.GetProperty(ORDER_PROP_TICKET);                     }
   long              TicketFrom(void)                                   const { return this.GetProperty(ORDER_PROP_TICKET_FROM);                }
   long              TicketTo(void)                                     const { return this.GetProperty(ORDER_PROP_TICKET_TO);                  }
   long              Magic(void)                                        const { return this.GetProperty(ORDER_PROP_MAGIC);                      }
   long              Reason(void)                                       const { return this.GetProperty(ORDER_PROP_REASON);                     }
   long              PositionID(void)                                   const { return this.GetProperty(ORDER_PROP_POSITION_ID);                }
   long              PositionByID(void)                                 const { return this.GetProperty(ORDER_PROP_POSITION_BY_ID);             }
   long              MagicID(void)                                      const { return this.GetProperty(ORDER_PROP_MAGIC_ID);                   }
   long              GroupID1(void)                                     const { return this.GetProperty(ORDER_PROP_GROUP_ID1);                  }
   long              GroupID2(void)                                     const { return this.GetProperty(ORDER_PROP_GROUP_ID2);                  }
   long              PendReqID(void)                                    const { return this.GetProperty(ORDER_PROP_PEND_REQ_ID);                }
   long              TypeOrder(void)                                    const { return this.GetProperty(ORDER_PROP_TYPE);                       }
   bool              IsCloseByStopLoss(void)                            const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_SL);          }
   bool              IsCloseByTakeProfit(void)                          const { return (bool)this.GetProperty(ORDER_PROP_CLOSE_BY_TP);          }
   long              TimeOpen(void)                                     const { return this.GetProperty(ORDER_PROP_TIME_OPEN);                  }
   long              TimeClose(void)                                    const { return this.GetProperty(ORDER_PROP_TIME_CLOSE);                 }
   datetime          TimeExpiration(void)                               const { return (datetime)this.GetProperty(ORDER_PROP_TIME_EXP);         }
   ENUM_ORDER_STATE  State(void)                                        const { return (ENUM_ORDER_STATE)this.GetProperty(ORDER_PROP_STATE);    }
   ENUM_ORDER_STATUS Status(void)                                       const { return (ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS);  }
   ENUM_ORDER_TYPE   TypeByDirection(void)                              const { return (ENUM_ORDER_TYPE)this.GetProperty(ORDER_PROP_DIRECTION); }
```

Also, **add the three methods for setting the new properties for the abstract**
**order:**

```
//--- Get the full order profit
   double            ProfitFull(void)                                   const { return this.Profit()+this.Comission()+this.Swap();              }
//--- Get order profit in points
   int               ProfitInPoints(void) const;
//--- Set (1) the first group ID, (2) the second group ID, (3) the pending request ID, (4) custom comment
   void              SetGroupID1(const long group_id)                         { this.SetProperty(ORDER_PROP_GROUP_ID1,group_id);                }
   void              SetGroupID2(const long group_id)                         { this.SetProperty(ORDER_PROP_GROUP_ID2,group_id);                }
   void              SetPendReqID(const long req_id)                          { this.SetProperty(ORDER_PROP_PEND_REQ_ID,req_id);                }
   void              SetCommentExt(const string comment_ext)                  { this.SetProperty(ORDER_PROP_COMMENT_EXT,comment_ext);           }

//+------------------------------------------------------------------+
```

**In the closed class constructor, fill in the new properties fields by ID**
**values using the above methods:**

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
   this.m_long_prop[ORDER_PROP_MAGIC_ID]                             = this.GetMagicID();
   this.m_long_prop[ORDER_PROP_GROUP_ID1]                            = this.GetGroupID1();
   this.m_long_prop[ORDER_PROP_GROUP_ID2]                            = this.GetGroupID2();
   this.m_long_prop[ORDER_PROP_PEND_REQ_ID]                          = this.GetPendReqID();

//--- Save additional real properties
   this.m_double_prop[this.IndexProp(ORDER_PROP_PROFIT_FULL)]        = this.ProfitFull();

//--- Save additional string properties
   this.m_string_prop[this.IndexProp(ORDER_PROP_COMMENT_EXT)]        = "";
  }
//+------------------------------------------------------------------+
```

**Add the display of the description of all new properties of the abstract order**
**to the method returning descriptions of integer properties:**

```
//+------------------------------------------------------------------+
//| Return description of an order's integer property                |
//+------------------------------------------------------------------+
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property)
  {
   return
     (
   //--- General properties
      property==ORDER_PROP_MAGIC             ?  CMessage::Text(MSG_ORD_MAGIC)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET            ?  CMessage::Text(MSG_ORD_TICKET)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_FROM       ?  CMessage::Text(MSG_ORD_TICKET_FROM)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_TO         ?  CMessage::Text(MSG_ORD_TICKET_TO)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_EXP          ?  CMessage::Text(MSG_ORD_TIME_EXP)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          (this.GetProperty(property)==0     ?  CMessage::Text(MSG_LIB_PROP_NOT_SET)+": "+CMessage::Text(MSG_LIB_PROP_NOT_SET)  :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS))
         )  :
      property==ORDER_PROP_TYPE              ?  CMessage::Text(MSG_ORD_TYPE)+": "+this.TypeDescription()   :
      property==ORDER_PROP_DIRECTION         ?  CMessage::Text(MSG_ORD_TYPE_BY_DIRECTION)+": "+this.DirectionDescription() :

      property==ORDER_PROP_REASON            ?  CMessage::Text(MSG_ORD_REASON)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetReasonDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_ID       ?  CMessage::Text(MSG_ORD_POSITION_ID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ORDER_TICKET ?  CMessage::Text(MSG_ORD_DEAL_ORDER_TICKET)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_DEAL_ENTRY        ?  CMessage::Text(MSG_ORD_DEAL_ENTRY)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetEntryDescription(this.GetProperty(property))
         )  :
      property==ORDER_PROP_POSITION_BY_ID    ?  CMessage::Text(MSG_ORD_POSITION_BY_ID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN         ?  CMessage::Text(MSG_ORD_TIME_OPEN)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_CLOSE        ?  CMessage::Text(MSG_ORD_TIME_CLOSE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_UPDATE       ?  CMessage::Text(MSG_ORD_TIME_UPDATE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)!=0 ? TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")" : "0")
         )  :
      property==ORDER_PROP_STATE             ?  CMessage::Text(MSG_ORD_STATE)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": \""+this.StateDescription()+"\""
         )  :
   //--- Additional property
      property==ORDER_PROP_STATUS            ?  CMessage::Text(MSG_ORD_STATUS)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": \""+this.StatusDescription()+"\""
         )  :
      property==ORDER_PROP_PROFIT_PT         ?  (
                                                 this.Status()==ORDER_STATUS_MARKET_PENDING ?
                                                 CMessage::Text(MSG_ORD_DISTANCE_PT)  :
                                                 CMessage::Text(MSG_ORD_PROFIT_PT)
                                                )+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_CLOSE_BY_SL       ?  CMessage::Text(MSG_LIB_PROP_CLOSE_BY_SL)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==ORDER_PROP_CLOSE_BY_TP       ?  CMessage::Text(MSG_LIB_PROP_CLOSE_BY_TP)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(this.GetProperty(property)   ?  CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))
         )  :
      property==ORDER_PROP_MAGIC_ID          ?  CMessage::Text(MSG_ORD_MAGIC_ID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_GROUP_ID1         ?  CMessage::Text(MSG_ORD_GROUP_ID1)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_GROUP_ID2         ?  CMessage::Text(MSG_ORD_GROUP_ID2)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_PEND_REQ_ID       ?  CMessage::Text(MSG_ORD_PEND_REQ_ID)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

The abstract order class is ready. Now we need to make changes to the event classes.

**In the abstract event class in the Event.mqh file, add the methods**
**returning new IDs to the protected class section:**

```
protected:
   ENUM_TRADE_EVENT  m_trade_event;                                  // Trading event
   bool              m_is_hedge;                                     // Hedge account flag
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits;                                       // Symbol's Digits()
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(void)                          const { return ushort(this.Magic() & 0xFFFF);                           }
   uchar             GetGroupID1(void)                         const { return uchar(this.Magic()>>16) & 0x0F;                          }
   uchar             GetGroupID2(void)                         const { return uchar((this.Magic()>>16) & 0xF0)>>4;                     }
   uchar             GetPendReqID(void)                        const { return uchar(this.Magic()>>24) & 0xFF;                          }
//--- Protected parametric constructor
                     CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket);
```

The methods are similar to the abstract order class methods described above.

Now in the five classes derived from the abstract event class (in the files **EventModify.mqh**, **EventOrderPlaced.mqh**,
**EventOrderRemoved.mqh**, **EventPositionClose.mqh** and **EventPositionOpen.mqh**), namely in the methods of the
event brief description, instead of the single string

```
//+------------------------------------------------------------------+
//| Create and return a short event message                          |
//+------------------------------------------------------------------+
string CEventModify::EventsMessage(void)
  {

//--- (1) header, (2) magic number
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string magic=(this.Magic()!=0 ? ", "+CMessage::Text(MSG_ORD_MAGIC)+" "+(string)this.Magic() : "");
   string text="";
```

add the following strings for each class:

```
//+------------------------------------------------------------------+
//| Create and return a short event message                          |
//+------------------------------------------------------------------+
string CEventModify::EventsMessage(void)
  {
//--- (1) header, (2) magic number
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string magic_id=((this.GetPendReqID()>0 || this.GetGroupID1()>0 || this.GetGroupID2()>0) ? " ("+(string)this.GetMagicID()+")" : "");
   string group_id1=(this.GetGroupID1()>0 ? ", G1: "+(string)this.GetGroupID1() : "");
   string group_id2=(this.GetGroupID2()>0 ? ", G2: "+(string)this.GetGroupID2() : "");
   string magic=(this.Magic()!=0 ? ", "+CMessage::Text(MSG_ORD_MAGIC)+" "+(string)this.Magic()+magic_id+group_id1+group_id2 : "");
   string text="";
```

When storing multiple data in a single magic number value, a completely different value (not the one set in the program settings) is displayed
when showing the number in the journal because the magic number is stored only in two lower bytes while groups and pending request IDs are
stored in two upper bytes. Therefore, if IDs (or at least one of them) are added in the magic number value, descriptions of each separate ID are
added to the values displayed in the journal.

We have implemented all the necessary changes for storing data in the magic number value. Now let's move on to the pending request class and
the first implementation of generating pending requests in case of errors when opening a position.

### Pending request class, first implementation of requests

In the **Trading.mqh** trading class file before the body of the **CTrading** trading class, add the new class
describing the pending request object:

```
//+------------------------------------------------------------------+
//| Pending request object class                                     |
//+------------------------------------------------------------------+
class CPendingReq : public CObject
  {
private:
   MqlTradeRequest      m_request;                       // Trade request structure
   uchar                m_id;                            // Trading request ID
   int                  m_retcode;                       // Result a request is based on
   double               m_price_create;                  // Price at the moment of a request generation
   ulong                m_time_create;                   // Request generation time
   ulong                m_time_activate;                 // Next attempt activation time
   ulong                m_waiting_msc;                   // Waiting time between requests
   uchar                m_current_attempt;               // Current attempt index
   uchar                m_total_attempts;                // Number of attempts
//--- Copy trading request data
   void                 CopyRequest(const MqlTradeRequest &request)  { this.m_request=request;        }
//--- Compare CPendingReq objects by IDs
   virtual int          Compare(const CObject *node,const int mode=0) const;

public:
//--- Return (1) the request structure, (2) the price at the moemnt of the request generation,
//--- (3) request generation time, (4) current attempt time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
   MqlTradeRequest      MqlRequest(void)                    const { return this.m_request;            }
   double               PriceCreate(void)                   const { return this.m_price_create;       }
   ulong                TimeCreate(void)                    const { return this.m_time_create;        }
   ulong                TimeActivate(void)                  const { return this.m_time_activate;      }
   ulong                WaitingMSC(void)                    const { return this.m_waiting_msc;        }
   uchar                CurrentAttempt(void)                const { return this.m_current_attempt;    }
   uchar                TotalAttempts(void)                 const { return this.m_total_attempts;     }
   uchar                ID(void)                            const { return this.m_id;                 }
//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) request ID
   void                 SetPriceCreate(const double price)        { this.m_price_create=price;        }
   void                 SetTimeCreate(const ulong time)           { this.m_time_create=time;          }
   void                 SetTimeActivate(const ulong time)         { this.m_time_activate=time;        }
   void                 SetWaitingMSC(const ulong miliseconds)    { this.m_waiting_msc=miliseconds;   }
   void                 SetCurrentAttempt(const uchar number)     { this.m_current_attempt=number;    }
   void                 SetTotalAttempts(const uchar number)      { this.m_total_attempts=number;     }
   void                 SetID(const uchar id)                     { this.m_id=id;                     }
//--- Constructors
                        CPendingReq(void){;}
                        CPendingReq(const uchar id,const double price,const ulong time,const MqlTradeRequest &request,const int retcode);
  };
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CPendingReq::CPendingReq(const uchar id,const double price,const ulong time,const MqlTradeRequest &request,const int retcode) : m_price_create(price),
                                                                                                                                m_time_create(time),
                                                                                                                                m_id(id),
                                                                                                                                m_retcode(retcode)
  {
   this.CopyRequest(request);
  }
//+------------------------------------------------------------------+
//| Compare CPendingReq objects by IDs                               |
//+------------------------------------------------------------------+
int CPendingReq::Compare(const CObject *node,const int mode=0) const
  {
   const CPendingReq *compared_req=node;
   return(this.ID()>compared_req.ID() ? 1 : this.ID()<compared_req.ID() ? -1 : 0);
   return 0;
  }
//+------------------------------------------------------------------+
```

I believe, this class is quite simple. There is no need to describe the strings commented out in it. All is clear by method names and class
member variables. However, I think I should explain how this object, related methods and the trading class functionality should work.

When receiving an error from the server, we want to create another server request and exit the trading method. When the waiting time of the newly
created request is up, it is sent to the server again. If an error is received again, we seemingly need to create a pending request. But it has
already been generated when receiving the server error for the first time. Therefore, the presence of the pending request ID is checked in
the magic number of the received trading request. If the request is present, the request has already been created and another attempt is
being sent to the server at the moment, which means no new request is required. If the trading request magic number has no ID, a new pending
request featuring the first lowest ID is generated and the exit from the trading method is performed to release the program for other
actions.

The list of trading requests can always be seen in the trading class timer. If the waiting time of the next request is up,
the appropriate trading method is called from the timer. When checking each request from the list of pending requests, the presence of an
appropriate position or an order in the list of market orders and positions is checked. If an order or a position with the current ID is
present, the pending order has fulfilled its function and should be removed from the request list.

Let's start implementation.

We have already added the class to the Trading.mqh file.

Now, **declare**
**the method for searching and returning the first lowest unused pending request ID in its private section:**

```
//--- Look for the first free pending request ID
   int                  GetFreeID(void);

public:
//--- Constructor
                        CTrading();
```

**Let's write its implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Look for the first free pending request ID                       |
//+------------------------------------------------------------------+
int CTrading::GetFreeID(void)
  {
   int id=WRONG_VALUE;
   CPendingReq *element=new CPendingReq();
   if(element==NULL)
      return 0;
   for(int i=1;i<256;i++)
     {
      element.SetID((uchar)i);
      this.m_list_request.Sort();
      if(this.m_list_request.Search(element)==WRONG_VALUE)
        {
         id=i;
         break;
        }
     }
   delete element;
   return id;
  }
//+------------------------------------------------------------------+
```

We may have 255 independent pending requests in total. Each of the requests has its own properties, its own waiting time between trading
attempts and, consequently, its own pending request object lifetime. In this regard, there may be a situation when the number of, say, 255 is
already used by the request ID, while the ID with the number of 0 or 1 or any of the lower ones are already released and can be used for new trading
requests. The method is used to find the lowest released ID number.

A temporary class object of a pending request and the
ID with the value of -1 are created first. This value informs there are no free IDs — all 255 of them are occupied, while the
value of 0 means a temporary object creation error. Further on, in a
loop by possible ID index values from 1 to 255, check the presence of a request object with its ID equal to the current loop index in the
list of pending requests. To do this, we first set the ID equal to the loop index
number for the temporary object, set the sorted list flag for the list
and simply [look \\
for the request object with the same ID in the list](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch). In other words, we look for the request object equal to the temporary object, to
which a loop index is set as an ID. If such an object is not found in the list, set
the loop index for the value returned from the method and break the loop.

After completing the loop, remove
the temporary request object and return the ID value, which
may be either -1, or from 1 to 255.

**In the public class section, declare the method for creating a pending**
**request and add the methods for setting/returning**
**ID values to/from the "magic number" order/position property value:**

```
//--- Create a pending request
   bool                 CreatePendingRequest(const uchar id,const uchar attempts,const ulong wait,const MqlTradeRequest &request,const int retcode,CSymbol *symbol_obj);

//--- Data location in the magic number int value
      //-----------------------------------------------------------
      //  bit   32|31       24|23       16|15        8|7         0|
      //-----------------------------------------------------------
      //  byte    |     3     |     2     |     1     |     0     |
      //-----------------------------------------------------------
      //  data    |   uchar   |   uchar   |         ushort        |
      //-----------------------------------------------------------
      //  descr   |pend req id| id2 | id1 |          magic        |
      //-----------------------------------------------------------

//--- Set the ID of the (1) first group, (2) second group, (3) pending request to the magic number value
   void              SetGroupID1(const uchar group,uint &magic)            { magic &=0xFFF0FFFF; magic |= uint(ConvToXX(group,0)<<16);    }
   void              SetGroupID2(const uchar group,uint &magic)            { magic &=0xFF0FFFFF; magic |= uint(ConvToXX(group,1)<<16);    }
   void              SetPendReqID(const uchar id,uint &magic)              { magic &=0x00FFFFFF; magic |= (uint)id<<24;                   }
//--- Convert the value of 0 - 15 into the necessary uchar number bits (0 - lower, 1 - upper ones)
   uchar             ConvToXX(const uchar number,const uchar index)  const { return((number>15 ? 15 : number)<<(4*(index>1 ? 1 : index)));}
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(const uint magic)                    const { return ushort(magic & 0xFFFF);                               }
   uchar             GetGroupID1(const uint magic)                   const { return uchar(magic>>16) & 0x0F;                              }
   uchar             GetGroupID2(const uint magic)                   const { return uchar((magic>>16) & 0xF0)>>4;                         }
   uchar             GetPendReqID(const uint magic)                  const { return uchar(magic>>24) & 0xFF;                              }
```

We have already considered the methods of returning values.
The identical ones are used here. Let's consider the methods for setting
different IDs.

Since two group IDs are stored in a single byte and the numerical ID value may only take values from 0 to 15 (4 bytes), we need to shift its values 4
bits left to set the second group ID. This will allow us to store it in the four upper bits of a one-byte number. This is done by the **ConvToXX()**
method. Depending on the group index (0 or 1), it either shifts a number passed to it (0-15) 4 bits left (the second group, index 1) or does not shift it
(the first group, index 0)

To set the first group ID value, we first need to reset the four lower bits of the byte we are going to save the ID value to. This can be done by
applying the mask to the magic value. The F value is used for each half-byte (4 bits) in the mask.

In other words, the hexadecimal value of
the decimal number 15 (F) is applied to the bits, in which the values should be left unchanged, while zero is applied to the bits that should be
deleted. Thus, the mask applied to the magic number value is to be as follows: 0x FF **F0** FFFF.

where:

- FFFF — leave the magic number ID (the magic number set in the EA
settings),
- **F0** — delete (0) the lower four bits in the byte storing
the group IDs, while the upper ones remain set to (F) — the second group ID is stored there,
- FF — leave the pending request ID value


Next, place the group number obtained from the ConvToXX() method with the index of
0 and shifted 16 bits left to the byte prepared for storing group IDs, so that the obtained number makes it onto the required byte the
group IDs are stored in.

To set the second group ID value, reset the four upper bits of the byte we are going to save the ID value to. We do that by applying the 0x FF **0F** FFFF
mask to the magic number.

where:

- FFFF — leave the magic number ID (the magic number set in the EA
settings),
- **0F** — delete (0) the upper four bits in the byte storing
the group IDs, while the lower ones remain set to (F) — the first group ID is stored there,
- FF — leave the pending request ID value


Next, place the group number obtained from the ConvToXX() method with the index of
1 and shifted 16 bits left to the byte prepared for storing group IDs, so that the obtained number makes it onto the required byte the
group IDs are stored in.

To set the pending request ID value, reset the byte value we are going to save the ID value to. We do that by applying the 0x **00** FFFFFF
mask to the magic number.

where:

- FFFF — leave the magic number ID (the magic number set in the EA
settings),
- FF — leave group ID values,
- **00**— delete the pending request ID value


Next, place the ID uchar value shifted 24 bits left to the byte prepared for storing
a pending request ID, so that the obtained number makes it onto the required byte the pending request ID is stored at.

**Implement the method for creating a pending request object beyond the class body:**

```
//+------------------------------------------------------------------+
//| Create a pending request                                         |
//+------------------------------------------------------------------+
bool CTrading::CreatePendingRequest(const uchar id,const uchar attempts,const ulong wait,const MqlTradeRequest &request,const int retcode,CSymbol *symbol_obj)
  {
   //--- Create a new pending request object
   CPendingReq *req_obj=new CPendingReq(id,symbol_obj.Bid(),symbol_obj.Time(),request,retcode);
   if(req_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ));
      return false;
     }
   //--- If failed to add the request to the list, display the appropriate message,
   //--- remove the created object and return 'false'
   if(!this.m_list_request.Add(req_obj))
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ));
      delete req_obj;
      return false;
     }
   //--- Filled in the fields of a successfully created object by the values passed to the method
   req_obj.SetTimeActivate(symbol_obj.Time()+wait);
   req_obj.SetWaitingMSC(wait);
   req_obj.SetCurrentAttempt(0);
   req_obj.SetTotalAttempts(attempts);
   return true;
  }
//+------------------------------------------------------------------+
```

The method is simple — a new request object is created and added to the list of pending requests. The values passed to the method are added to the
object fields (request activation time is calculated as a request creation time + waiting time) and true
is returned. Otherwise, the method returns false.

**In the trading class timer we have developed [in the previous article](https://www.mql5.com/en/articles/7365),**
**add the logic of working with pending requests:**

```
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CTrading::OnTimer(void)
  {
   //--- In a loop by the list of pending requests
   int total=this.m_list_request.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      //--- receive the next request object
      CPendingReq *req_obj=this.m_list_request.At(i);
      if(req_obj==NULL)
         continue;
      //--- if the current attempt exceeds the defined number of trading attempts,
      //--- remove the current request object and move on to the next one
      if(req_obj.CurrentAttempt()>req_obj.TotalAttempts() || req_obj.CurrentAttempt()>=UCHAR_MAX)
        {
         this.m_list_request.Delete(i);
         continue;
        }
      //--- get the request structure and the symbol object a trading operation should be performed for
      MqlTradeRequest request=req_obj.MqlRequest();
      CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(request.symbol);
      if(symbol_obj==NULL || !symbol_obj.RefreshRates())
         continue;

      //--- Set the request activation time in the request object
      req_obj.SetTimeActivate(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.CurrentAttempt()+1));

      //--- If the current time is less than the request activation time,
      //--- this is not the request time - move on to the next request in the list
      if(symbol_obj.Time()<req_obj.TimeActivate())
         continue;

      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

      //--- Get the pending request ID
      uchar id=this.GetPendReqID((uint)request.magic);

      //--- Get the list of orders/positions containing the order/position with the pending request ID
      CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
      if(::CheckPointer(list)==POINTER_INVALID)
         continue;
      //--- Depending on the type of action performed in the trading request
      switch(request.action)
        {
         //--- Open a position
         case TRADE_ACTION_DEAL :
            //--- if there is no position/order with the obtained pending request ID (the list is empty), send a trading request
            if(list.Total()==0)
              {
               this.OpenPosition((ENUM_POSITION_TYPE)request.type,request.volume,request.symbol,request.magic,request.sl,request.tp,request.comment,request.deviation);
              }
            //--- if a position/order with the obtained pending request ID is already present (the list is empty), the request has been handled and should be removed
            else
               this.m_list_request.Delete(i);
            break;
         //---
         default:
            break;
        }
     }
  }
//+------------------------------------------------------------------+
```

The operation logic is described in details in the code comments and requires no explanations. The only thing worth noting is the next trading
request activation time calculation. The time is calculated as "request object creation time" + waiting time in milliseconds \* next
attempt index. Thus, the request time is bound to the first request creation time and attempt index. The higher the attempt number, the more
time should pass from the object generation till its activation. The time is increased discretely: if we wait for 10 seconds, the first
attempt should occur in 10 seconds, the second one — in 20 seconds, the third one — in 30 seconds etc. Thus, the interval between trading
attempts will always be no less than the specified waiting time between them.

**In the method returning the way of handling errors, shift the code of**
**the trade server connection absence error to the block returning the "waiting" handling type. Previously, the code was handled as**
**"create a pending trading request":**

```
//+------------------------------------------------------------------+
//| Return the error handling method                                 |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::ResultProccessingMethod(const uint result_code)
  {
   switch(result_code)
     {
   #ifdef __MQL4__
      //--- Malfunctional trade operation
      case 9   :
      //--- Account disabled
      case 64  :
      //--- Invalid account number
      case 65  :  return ERROR_CODE_PROCESSING_METHOD_DISABLE;

      //--- No error but result is unknown
      case 1   :
      //--- General error
      case 2   :
      //--- Old client terminal version
      case 5   :
      //--- Not enough rights
      case 7   :
      //--- Market closed
      case 132 :
      //--- Trading disabled
      case 133 :
      //--- Order is locked and being processed
      case 139 :
      //--- Buy only
      case 140 :
      //--- The number of open and pending orders has reached the limit set by the broker
      case 148 :
      //--- Attempt to open an opposite order if hedging is disabled
      case 149 :
      //--- Attempt to close a position on a symbol contradicts the FIFO rule
      case 150 :  return ERROR_CODE_PROCESSING_METHOD_EXIT;

      //--- Invalid trading request parameters
      case 3   :
      //--- Invalid price
      case 129 :
      //--- Invalid stop levels
      case 130 :
      //--- Invalid volume
      case 131 :
      //--- Not enough money to perform the operation
      case 134 :
      //--- Expirations are denied by broker
      case 147 :  return ERROR_CODE_PROCESSING_METHOD_CORRECT;

      //--- Trade server is busy
      case 4   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- No connection to the trade server
      case 6   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- Too frequent requests
      case 8   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- No price
      case 136 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- Broker is busy
      case 137 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- Too many requests
      case 141 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- Modification denied because the order is too close to market
      case 145 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- Trade context is busy
      case 146 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)1000;    // ERROR_CODE_PROCESSING_METHOD_WAIT

      //--- Trade timeout
      case 128 :
      //--- Price has changed
      case 135 :
      //--- New prices
      case 138 :  return ERROR_CODE_PROCESSING_METHOD_REFRESH;

   //--- MQL5
   #else
      //--- Auto trading disabled by the server
      case 10026  :  return ERROR_CODE_PROCESSING_METHOD_DISABLE;

      //--- Request canceled by a trader
      case 10007  :
      //--- Request expired
      case 10012  :
      //--- Trading disabled
      case 10017 :
      //--- Market closed
      case 10018  :
      //--- Order status changed
      case 10023  :
      //--- Request unchanged
      case 10025  :
      //--- Request blocked for handling
      case 10028  :
      //--- Transaction is allowed for live accounts only
      case 10032  :
      //--- The maximum number of pending orders is reached
      case 10033  :
      //--- Reached the maximum order and position volume for this symbol
      case 10034  :
      //--- Invalid or prohibited order type
      case 10035  :
      //--- Position with the specified ID already closed
      case 10036  :
      //--- A close order is already present for a specified position
      case 10039  :
      //--- The maximum number of open positions is reached
      case 10040  :
      //--- Request to activate a pending order is rejected, the order is canceled
      case 10041  :
      //--- Request is rejected, because the rule "Only long positions are allowed" is set for the symbol
      case 10042  :
      //--- Request is rejected, because the rule "Only short positions are allowed" is set for the symbol
      case 10043  :
      //--- Request is rejected, because the rule "Only closing of existing positions is allowed" is set for the symbol
      case 10044  :
      //--- Request is rejected, because the rule "Only closing of existing positions by FIFO rule is allowed" is set for the symbol
      case 10045  :  return ERROR_CODE_PROCESSING_METHOD_EXIT;

      //--- Requote
      case 10004  :
      //--- Request rejected
      case 10006  :
      //--- Prices changed
      case 10020  :  return ERROR_CODE_PROCESSING_METHOD_REFRESH;

      //--- Invalid request
      case 10013  :
      //--- Invalid request volume
      case 10014  :
      //--- Invalid request price
      case 10015  :
      //--- Invalid request stop levels
      case 10016  :
      //--- Insufficient funds for request execution
      case 10019  :
      //--- Invalid order expiration in a request
      case 10022  :
      //--- The specified type of order execution by balance is not supported
      case 10030  :
      //--- Closed volume exceeds the current position volume
      case 10038  :  return ERROR_CODE_PROCESSING_METHOD_CORRECT;

      //--- No quotes to handle the request
      case 10021  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT;
      //--- Too frequent requests
      case 10024  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- An order or a position is frozen
      case 10029  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT;
      //--- No connection to the trade server
      case 10031  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT;

      //--- Request handling error
      case 10011  :
      //--- Auto trading disabled by the client terminal
      case 10027  :  return ERROR_CODE_PROCESSING_METHOD_PENDING;

      //--- Order placed
      case 10008  :
      //--- Request executed
      case 10009  :
      //--- Request executed partially
      case 10010  :
   #endif
      //--- "OK"
      default:
        break;
     }
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

Why? First, in order to test generation of pending requests with waiting, return
20 seconds waiting between requests. Besides, this makes performing multiple trading attempts while waiting for the connection
with the trade server to be restored more convenient. Anyway, this is the first test version of handling pending requests and it will be
improved and changed.

Since we are testing the concept here, we are going to create a pending request only to open a position and only to obtain a trade server error. When
checking the validity of trading requests, we are not going to create pending requests leaving waiting inside the position opening method.

**Add the pending request creation block to the position**
**opening method:**

```
//+------------------------------------------------------------------+
//| Open a position                                                  |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CTrading::OpenPosition(const ENUM_POSITION_TYPE type,
                            const double volume,
                            const string symbol,
                            const ulong magic=ULONG_MAX,
                            const SL sl=0,
                            const TP tp=0,
                            const string comment=NULL,
                            const ulong deviation=ULONG_MAX)
  {
//--- Set the trading request result as 'true' and the error flag as "no errors"
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)type;
   ENUM_ACTION_TYPE action=(ENUM_ACTION_TYPE)order_type;
//--- Get a symbol object by a symbol name. If failed to get
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
//--- If failed to get - write the "internal error" flag, display the message in the journal and return 'false'
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
//--- If failed to get - write the "internal error" flag, display the message in the journal and return 'false'
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- Set the prices
//--- If failed to set - write the "internal error" flag, set the error code in the return structure,
//--- display the message in the journal and return 'false'
   if(!this.SetPrices(order_type,0,sl,tp,0,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
      return false;
     }

//--- Write the volume to the request structure
   this.m_request.volume=volume;
//--- Get the method of handling errors from the CheckErrors() method while checking for errors in the request parameters
   double pr=(type==POSITION_TYPE_BUY ? symbol_obj.Ask() : symbol_obj.Bid());
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(this.m_request.volume,pr,action,order_type,symbol_obj,trade_obj,DFUN,0,this.m_request.sl,this.m_request.tp);
//--- In case of trading limitations, funds insufficiency,
//--- if there are limitations by StopLevel or FreezeLevel ...
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled, set the error code to the return structure,
      //--- display a journal message, play the error sound and exit
      if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
        {
         trade_obj.SetResultRetcode(MSG_LIB_TEXT_TRADING_DISABLE);
         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "abort trading operation" - set the last error code to the return structure,
      //--- display a journal message, play the error sound and exit
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
        {
         int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
         if(code!=NULL)
           {
            trade_obj.SetResultRetcode(code);
            trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
           }
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_OPERATION_ABORTED));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal
      if(method==ERROR_CODE_PROCESSING_METHOD_WAIT)
        {
         int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
         if(code!=NULL)
           {
            trade_obj.SetResultRetcode(code);
            trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
           }
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_CREATE_PENDING_REQUEST));
         //--- Instead of creating a pending request, we temporarily wait the required time period (the CheckErrors() method result is returned)
         ::Sleep(method);
         //--- after waiting, update all data
         symbol_obj.Refresh();
        }
      //--- If the check result is "create a pending request", do nothing temporarily
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_PENDING_REQUEST)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_CREATE_PENDING_REQUEST));
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.OpenPosition(type,this.m_request.volume,this.m_request.sl,this.m_request.tp,magic,comment,deviation);
      //--- If the request is executed successfully or the asynchronous order sending mode is set, play the success sound
      //--- set for a symbol trading object for this type of trading operation and return 'true'
      if(res || trade_obj.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj.PlaySoundSuccess(action,order_type);
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRY_N),string(i+1),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type);

         //--- Get the error handling method
         method=this.ResultProccessingMethod(trade_obj.GetResultRetcode());
         //--- If "Disable trading for the EA" is received as a result of sending a request, enable the disabling flag and end the attempt loop
         if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
           {
            this.SetTradingDisableFlag(true);
            break;
           }
         //--- If "Exit the trading method" is received as a result of sending a request, end the attempt loop
         if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
           {
            break;
           }
         //--- If "Correct the parameters and repeat" is received as a result of sending a request -
         //--- correct the parameters and start the next iteration
         if(method==ERROR_CODE_PROCESSING_METHOD_CORRECT)
           {
            this.RequestErrorsCorrecting(this.m_request,order_type,trade_obj.SpreadMultiplier(),symbol_obj,trade_obj);
            continue;
           }
         //--- If "Update data and repeat" is received as a result of sending a request -
         //--- update data and start the next iteration
         if(method==ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            symbol_obj.Refresh();
            continue;
           }
         //--- If "Wait and repeat" or "Create a pending request" is received as a result of sending a request,
         //--- create a pending request and end the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the trading request magic number, has no pending request ID
            if(this.GetPendReqID((uint)magic)==0)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=(method>ERROR_CODE_PROCESSING_METHOD_PENDING ? method : 0);
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write the request ID to the magic number, while a symbol name is set in the request structure
               //--- Set position and trading operation types (the remaining structure fields are already filled in)
               uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
               this.SetPendReqID((uchar)id,mn);
               this.m_request.magic=mn;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.action=TRADE_ACTION_DEAL;
               this.m_request.type=order_type;
               //--- Pass the number of trading attempts minus one to the pending request,
               //--- since there already has been one failed attempt
               uchar attempts=(this.m_total_try-1 < 1 ? 1 : this.m_total_try-1);
               this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
               break;
              }
           }
        }
     }
//--- Return the result of sending a trading request in a symbol trading object
   return res;
  }
//+------------------------------------------------------------------+
```

The code comments contain all the details. However, you are welcome to use the comments section if you have any questions.

Let's make slight improvements in the **Engine.mqh** file of the library's **CEngine** base object class.

**Add the method setting the number of trading attempts in**
**the block of methods for placing trading object properties:**

```
//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level, (12) number of trading attempts
   void                 TradingSetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetMagic(const uint magic,const string symbol_name=NULL);
   void                 TradingSetComment(const string comment,const string symbol_name=NULL);
   void                 TradingSetDeviation(const ulong deviation,const string symbol_name=NULL);
   void                 TradingSetVolume(const double volume=0,const string symbol_name=NULL);
   void                 TradingSetExpiration(const datetime expiration=0,const string symbol_name=NULL);
   void                 TradingSetAsyncMode(const bool async_mode=false,const string symbol_name=NULL);
   void                 TradingSetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL);
   void                 TradingSetTotalTry(const uchar attempts)                          { this.m_trading.SetTotalTry(attempts);               }

//--- Set standard sounds (symbol==NULL) for a symbol trading object, (symbol!=NULL) for trading objects of all symbols
```

The method simply calls the trading class method of the same name.

**In the private class section, add the method of converting group ID**
**values into a uchar value, while in the public section, declare the**
**method for creating and returning the composite magic number value:**

```
//--- Constructor/destructor
                        CEngine();
                       ~CEngine();

private:
//--- Convert the value of 0 - 15 into the necessary uchar number bits (0 - lower, 1 - upper ones)
   uchar                ConvToXX(const uchar number,const uchar index)  const { return((number>15 ? 15 : number)<<(4*(index>1 ? 1 : index)));   }

public:
//--- Create and return the composite magic number from the specified magic number value, the first and second group IDs and the pending request ID
   uint                 SetCompositeMagicNumber(ushort magic_id,const uchar group_id1=0,const uchar group_id2=0,const uchar pending_req_id=0);

  };
//+------------------------------------------------------------------+
```

We have already considered the conversion method above. The method of creating the composite magic number is used to combine the values of
the magic number, the first and second groups and the pending order ID into a single magic number set for an order when sending it to the server.

**Let's write its implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Create and return the composite magic number                     |
//| from the specified magic number value,                           |
//| first and seconf group IDs and                                   |
//| the pending request ID                                           |
//+------------------------------------------------------------------+
uint CEngine::SetCompositeMagicNumber(ushort magic_id,const uchar group_id1=0,const uchar group_id2=0,const uchar pending_req_id=0)
  {
   uint magic=magic_id;
   this.m_trading.SetGroupID1(group_id1,magic);
   this.m_trading.SetGroupID2(group_id2,magic);
   this.m_trading.SetPendReqID(pending_req_id,magic);
   return magic;
  }
//+------------------------------------------------------------------+
```

The method receives all IDs which are all added to the magic number value (to be returned from the method) using the previously considered
trading class methods for setting IDs.

**This is all we needed to do to check the offered idea.**

To test the generation and handling a pending request, we need to simulate an error requiring a second request after waiting. As you may
remember, we have made the handling of the "no connection to trade server" error as 20 seconds waiting. We have five trading attempts by
default. This means we simply need to launch the EA, disable the Internet (disconnect from the trade server) and try to open a position
(Buy/Sell button on the test EA trading panel). After getting an error, we will have 20 \* 5 = 100 seconds to enable Internet again and observe
how the EA handles the created pending request. After 100 seconds (necessary to complete five repeated attempts), the pending request
should be automatically removed from the request list (after connection to the server is restored since the time can be obtained only if the
connection is active). This feature is not implemented yet, since we are currently testing the pending request operation. Besides, the
functionality is still under development and requires changes so that all remaining features are implemented. This means that after the
connection to the trade server is restored, the EA will start sending trading requests set in the request object in any case. After the first
of the repeated attempts, a position should be opened and the pending request object removed from the request list.

We have implemented storing several IDs in the magic number value along with the pending request. In order to test adding these IDs to the sent
request magic number, let's make a random selection of the first and second subgroup numbers in the groups 1 and 2, and write them to the "magic
number" order property. When opening a position, the journal displays both a real open position/placed order magic number and a magic
number ID set in the settings (in brackets after the true value of the magic number), as well as subgroup IDs in the first and second groups
(indicated as G1 and G2).

### Testing

To test pending requests, let's use the EA from the previous article and save it to \\MQL5\\Experts\\TestDoEasy\ **Part26\**
under the name **TestDoEasyPart26.mq5**.

**In the EA inputs, change the magic number type from ulong**
**to ushort** **—**
**now the maximum size of the magic number cannot exceed two bytes (65535). Also, add yet another variable — the**
**number of trading attempts:**

```
//--- input variables
input    ushort            InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  150;  // StopLoss in points
input    uint              InpTakeProfit        =  150;  // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)
sinput   uint              InpButtShiftX        =  40;   // Buttons X shift
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift
input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;   // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
//--- global variables
```

**In the global variables, change the magic\_number variable type from ulong**
**to ushort and add the two**
**variables for storing the group values:**

```
//--- global variables
CEngine        engine;
SDataButt      butt_data[TOTAL_BUTT];
string         prefix;
double         lot;
double         withdrawal=(InpWithdrawal<0.1 ? 0.1 : InpWithdrawal);
ushort         magic_number;
uint           stoploss;
uint           takeprofit;
uint           distance_pending;
uint           distance_stoplimit;
uint           slippage;
bool           trailing_on;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
int            used_symbols_mode;
string         used_symbols;
string         array_used_symbols[];
bool           testing;
uchar          group1;
uchar          group2;
//+------------------------------------------------------------------+
```

**In the OnInit() handler, initialize the group variables**
**and [set the initial \**\
**status](https://www.mql5.com/en/docs/math/mathsrand) to generate pseudo-random numbers:**

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();

//--- Set EA global variables
   prefix=MQLInfoString(MQL_PROGRAM_NAME)+"_";
   testing=engine.IsTester();
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
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;

//--- Initialize random group numbers
   group1=0;
   group2=0;
   srand(GetTickCount());

//--- Initialize DoEasy library
   OnInitDoEasy();
```

**In the library initialization function, set the default magic number**
**for all trading objects and the number of trading attempts:**

```
//+------------------------------------------------------------------+
//| Initializing DoEasy library                                      |
//+------------------------------------------------------------------+
void OnInitDoEasy()
  {
//--- Check if working with the full list is selected
   used_symbols_mode=InpModeUsedSymbols;
   if((ENUM_SYMBOLS_MODE)used_symbols_mode==SYMBOLS_MODE_ALL)
     {
      int total=SymbolsTotal(false);
      string ru_n="\nКоличество символов на сервере "+(string)total+".\nМаксимальное количество: "+(string)SYMBOLS_COMMON_TOTAL+" символов.";
      string en_n="\nNumber of symbols on server "+(string)total+".\nMaximum number: "+(string)SYMBOLS_COMMON_TOTAL+" symbols.";
      string caption=TextByLanguage("Внимание!","Attention!");
      string ru="Выбран режим работы с полным списком.\nВ этом режиме первичная подготовка списка коллекции символов может занять длительное время."+ru_n+"\nПродолжить?\n\"Нет\" - работа с текущим символом \""+Symbol()+"\"";
      string en="Full list mode selected.\nIn this mode, the initial preparation of the collection symbols list may take a long time."+en_n+"\nContinue?\n\"No\" - working with the current symbol \""+Symbol()+"\"";
      string message=TextByLanguage(ru,en);
      int flags=(MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
      int mb_res=MessageBox(message,caption,flags);
      switch(mb_res)
        {
         case IDNO :
           used_symbols_mode=SYMBOLS_MODE_CURRENT;
           break;
         default:
           break;
        }
     }
//--- Fill in the array of used symbols
   used_symbols=InpUsedSymbols;
   CreateUsedSymbolsArray((ENUM_SYMBOLS_MODE)used_symbols_mode,used_symbols,array_used_symbols);

//--- Set the type of the used symbol list in the symbol collection
   engine.SetUsedSymbols(array_used_symbols);
//--- Displaying the selected mode of working with the symbol object collection
   Print(engine.ModeSymbolsListDescription(),TextByLanguage(". Number of used symbols: ",". Number of symbols used: "),engine.GetSymbolsCollectionTotal());

//--- Create resource text files
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Falling coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Button click 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Button click 2"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Button click 3"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the trading class
   engine.TradingOnInit();

//--- Set the default magic number for all used symbols
   engine.TradingSetMagic(engine.SetCompositeMagicNumber(magic_number));
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode(false);
//--- Set the number of trading attempts in case of an error
   engine.TradingSetTotalTry(InpTotalAttempts);
//--- Set standard sounds for trading objects of all used symbols
   engine.SetSoundsStandart();
//--- Set the general flag of using sounds
   engine.SetUseSounds(InpUseSounds);
//--- Set the spread multiplier for symbol trading objects in the symbol collection
   engine.SetSpreadMultiplier(InpSpreadMultiplier);

//--- Set controlled values for symbols
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      /*
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- Set control of the symbol price increase by 100 points
         symbol.SetControlBidInc(100000*symbol.Point());
         //--- Set control of the symbol price decrease by 100 points
         symbol.SetControlBidDec(100000*symbol.Point());
         //--- Set control of the symbol spread increase by 40 points
         symbol.SetControlSpreadInc(400);
         //--- Set control of the symbol spread decrease by 40 points
         symbol.SetControlSpreadDec(400);
         //--- Set control of the current spread by the value of 40 points
         symbol.SetControlSpreadLevel(400);
        }
      */
     }
//--- Set controlled values for the current account
   CAccount* account=engine.GetAccountCurrent();
   if(account!=NULL)
     {
      //--- Set control of the profit increase to 10
      account.SetControlledValueINC(ACCOUNT_PROP_PROFIT,10.0);
      //--- Set control of the funds increase to 15
      account.SetControlledValueINC(ACCOUNT_PROP_EQUITY,15.0);
      //--- Set profit control level to 20
      account.SetControlledValueLEVEL(ACCOUNT_PROP_PROFIT,20.0);
     }
  }
//+------------------------------------------------------------------+
```

To test the magic number with random group ID values, we are going to introduce the **comp\_magic**
boolean variable equal to true and indicating the usage of the composite
magic number in the functions of opening positions/placing pending orders. Instead of using the **magic\_number** variable,
introduce the new **magic** variable storing the magic value
depending on the **comp\_magic** variable value.

When setting the
value for **magic**(the permanent magic number defined in the settings or the composite magic number consisting of a specified magic
number + random group 1 and 2 ID values), we are going to check the **comp\_magic** value. If true,
use the composite magic number. If false,
use the one defined in the settings.

**Make changes in the PressButtonEvents() function handling pressing the EA trading panel buttons:**

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   bool comp_magic=true;   // Temporary variable selecting the composite magic number with random group IDs
   string comment="";
   //--- Convert button name into its string ID
   string button=StringSubstr(button_name,StringLen(prefix));
   //--- Random group 1 and 2 numbers within the range of 0 - 15
   group1=(uchar)Rand();
   group2=(uchar)Rand();
   uint magic=(comp_magic ? engine.SetCompositeMagicNumber(magic_number,group1,group2) : magic_number);
   //--- If the button is pressed
   if(ButtonState(button_name))
     {
      //--- If the BUTT_BUY button is pressed: Open Buy position
      if(button==EnumToString(BUTT_BUY))
        {
         //--- Open Buy position
         engine.OpenBuy(lot,Symbol(),magic,stoploss,takeprofit);   // No comment - the default comment is to be set
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- Set BuyLimit order
         engine.PlaceBuyLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- Set BuyStop order
         engine.PlaceBuyStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- Set BuyStopLimit order
         engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Open Sell position
         engine.OpenSell(lot,Symbol(),magic,stoploss,takeprofit);  // No comment - the default comment is to be set
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Set SellLimit order
         engine.PlaceSellLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellLimit","Pending SellLimit order"));
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Set SellStop order
         engine.PlaceSellStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- Set SellStopLimit order
         engine.PlaceSellStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
```

Replace the **magic\_number** variable with magic in all
the strings calling the library trading methods.

**To set random values for group IDs, add the value returned by the Rand()**
**function already featuring minimum and maximum range values, within which the function returns a pseudo-random value:**

```
//+------------------------------------------------------------------+
//| A random value within the range                                  |
//+------------------------------------------------------------------+
uint Rand(const uint min=0, const uint max=15)
  {
   return (rand() % (max+1-min))+min;
  }
//+------------------------------------------------------------------+
```

Compile and launch the EA. Turn off the Internet and wait till the following image appears in the lower right corner of the terminal:

![](https://c.mql5.com/2/37/WkjYQlsw9Y.png)

![](https://c.mql5.com/2/37/bmfIngysXo.gif)

After disabling the Internet and clicking Sell, the trade server returns the error and the following entries are displayed in the journal:

```
2019.11.26 15:34:48.661 CTrading::OpenPosition<uint,uint>: Invalid request:
2019.11.26 15:34:48.661 No connection with the trade server
2019.11.26 15:34:48.661 Correction of trade request parameters ...
2019.11.26 15:34:48.661 Trading attempt #1. Error: No connection with the trade server
```

After receiving the error, the library creates a pending request with the parameters set during the unsuccessful attempt to open a short
position.

The pending request also features the number of attempts and the waiting time of 20 seconds.

Then enable the Internet to restore the connection to the trade server:

![](https://c.mql5.com/2/37/terminal64_SwjtIhMXo0.png)

As soon as the connection is restored, the library starts handling a pending request sending it to the server. As a result, we have an open
position with journal entries:

```
2019.11.26 15:35:00.853 CTrading::OpenPosition<double,double>: Invalid request:
2019.11.26 15:35:00.853 Trading is prohibited for the current account
2019.11.26 15:35:00.853 Correction of trade request parameters ...
2019.11.26 15:35:00.853 Trading operation aborted
2019.11.26 15:35:01.192 CTrading::OpenPosition<double,double>: Invalid request:
2019.11.26 15:35:01.192 Trading is prohibited for the current account
2019.11.26 15:35:01.192 Correction of trade request parameters ...
2019.11.26 15:35:01.192 Trading operation aborted
2019.11.26 15:35:01.942 - Position is open: 2019.11.26 10:35:01.660 -
2019.11.26 15:35:01.942 EURUSD Opened 0.10 Sell #486405595 [0.10 Market-order Sell #486405595] at price 1.10126, sl 1.10285, tp 1.09985, Magic number 17629307 (123), G1: 13
2019.11.26 15:35:01.942 OnDoEasyEvent: Position is open
```

As we can see, after restoring the connection to the trade server, trading for
the current account was enabled with a delay.

But the pending
request did the trick anyway...

Also, we can see the real magic number 17629307 in the journal followed by the magic number defined in the EA settings in brackets (123) plus
another entry G1: 13 informing that the first group ID is equal to 13, while the second group ID is absent — its value turned out to be equal to
zero, therefore there was no second entry with the G2: XX second group ID.

**Please note:**

Do not use
the results of the trading class with the pending requests, described in the article, and the attached test EA in real trading!

The
article, its accompanying materials and results are meant only as a test of the pending requests concept. In its current state, it is not
a finished product and is in no way intended for real trading. Instead, it is only meant for
the demo mode or the tester.

### What's next?

In the following articles, we will continue the development of the pending request class.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7394#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part \\
6\. Netting account events](https://www.mql5.com/en/articles/6383)

[Part 7. StopLimit order activation events, preparing \\
the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part 8. Order and \\
position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 — Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending \\
orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part \\
13\. Account object events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part \\
15\. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part 16. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part \\
18\. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part 19. Class of \\
library messages](https://www.mql5.com/en/articles/7176)

[Part 20. Creating and storing program resources](https://www.mql5.com/en/articles/7195)

[Part 21. Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

[Part \\
22\. Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

[Part 23. \\
Trading classes - Base trading class, verification of valid parameters](https://www.mql5.com/en/articles/7286)

[Part 24. \\
Trading classes - Base trading class, auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

[Part \\
25\. Trading classes - Base trading class, handling errors returned by the trade server](https://www.mql5.com/en/articles/7365)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7394](https://www.mql5.com/ru/articles/7394)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7394.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7394/mql5.zip "Download MQL5.zip")(3609.19 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7394/mql4.zip "Download MQL4.zip")(3609.19 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/332203)**
(10)


![iabbott](https://c.mql5.com/avatar/avatar_na2.png)

**[iabbott](https://www.mql5.com/en/users/iabbott)**
\|
23 Oct 2020 at 12:16

Is there an implementation in the library to open an immediate [market execution](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution "MQL5 documentation: Symbol properties")?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
23 Oct 2020 at 13:17

**iabbott:**

Is there an implementation in the library to open an immediate [market execution](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution "MQL5 documentation: Symbol properties")?

Yes


![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
14 Jul 2024 at 10:53

Could you tell me, please, where exactly, in what method, do you decode the encoded magic in the pending request?

After an error when opening a position [(insufficient funds](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: No sufficient funds to execute the request")) and then triggering a pending request, I have a position created with a completely different magic (apparently, the one encoded).

Where do you decode it? I want a native majik, not a new one.

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
14 Jul 2024 at 11:14

```
void CTradingControl::OnPReqByErrCodeHandler()
```

This is where you're supposed to call the method that returns the original magick. And what method is that? GetMagicID()?

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
28 Jul 2024 at 10:34

```
//+------------------------------------------------------------------+
//| Returns the order's profit in pips|
//+------------------------------------------------------------------+
int COrder::ProfitInPoints(void) const
  {
   MqlTick tick={0};
   string symbol=this.Symbol();
   if(!::SymbolInfoTick(symbol,tick))
      return 0;
   ENUM_ORDER_TYPE type=(ENUM_ORDER_TYPE)this.TypeOrder();
   double point=::SymbolInfoDouble(symbol,SYMBOL_POINT);
   if(type==ORDER_TYPE_CLOSE_BY || point==0) return 0;
   if(this.Status()==ORDER_STATUS_HISTORY_ORDER)
      return int(type==ORDER_TYPE_BUY ? (this.PriceClose()-this.PriceOpen())/point : type==ORDER_TYPE_SELL ? (this.PriceOpen()-this.PriceClose())/point : 0);
   else if(this.Status()==ORDER_STATUS_MARKET_POSITION)
     {
      if(type==ORDER_TYPE_BUY)
         return int((tick.bid-this.PriceOpen())/point);
      else if(type==ORDER_TYPE_SELL)
         return int((this.PriceOpen()-tick.ask)/point);
     }
   else if(this.Status()==ORDER_STATUS_MARKET_PENDING)
     {
      if(type==ORDER_TYPE_BUY_LIMIT || type==ORDER_TYPE_BUY_STOP || type==ORDER_TYPE_BUY_STOP_LIMIT)
         return (int)fabs((tick.bid-this.PriceOpen())/point);
      else if(type==ORDER_TYPE_SELL_LIMIT || type==ORDER_TYPE_SELL_STOP || type==ORDER_TYPE_SELL_STOP_LIMIT)
         return (int)fabs((this.PriceOpen()-tick.ask)/point);
     }
   return 0;
  }
```

Why is there no calculation here for ORDER\_STATUS\_DEAL type ?

In general, it is not clear how to get the profit in points of a closed trade or position...

And it is always 0:

```
deal_profit_pts=(int)deal.GetProperty(ORDER_PROP_PROFIT_PT)
```

![Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://c.mql5.com/2/37/MQL5-avatar-doeasy__15.png)[Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders](https://www.mql5.com/en/articles/7418)

In this article, we will continue the development of trading requests, implement placing pending orders and eliminate detected shortcomings of the trading class operation.

![Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://c.mql5.com/2/37/MQL5-avatar-season_research.png)[Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://www.mql5.com/en/articles/7038)

In this article we will view seasonal characteristics of financial time series using Boxplot diagrams. Each separate boxplot (or box-and-whiskey diagram) provides a good visualization of how values are distributed along the dataset. Boxplots should not be confused with the candlestick charts, although they can be visually similar.

![Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

The first article within the Walk-Through Optimization series described the creation of a DLL to be used in our auto optimizer. This continuation is entirely devoted to the MQL5 language.

![Library for easy and quick development of MetaTrader programs (part XXV): Handling errors returned by the trade server](https://c.mql5.com/2/37/MQL5-avatar-doeasy__12.png)[Library for easy and quick development of MetaTrader programs (part XXV): Handling errors returned by the trade server](https://www.mql5.com/en/articles/7365)

After we send a trading order to the server, we need to check the error codes or the absence of errors. In this article, we will consider handling errors returned by the trade server and prepare for creating pending trading requests.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/7394&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062734129763624921)

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