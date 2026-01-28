---
title: Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests
url: https://www.mql5.com/en/articles/7438
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:46.962730
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=mtrwnzcyzrjpveinpixompfbhzqxeybe&ssn=1769186263727098697&ssn_dr=0&ssn_sr=0&fv_date=1769186263&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7438&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXVIII)%3A%20Closure%2C%20removal%20and%20modification%20of%20pending%20trading%20requests%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918626329264988&fz_uniq=5070438188696147414&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Preparing data](https://www.mql5.com/en/articles/7438#node01)
- [Pending requests to close/remove/modify positions and pending orders](https://www.mql5.com/en/articles/7438#node02)
- [Testing](https://www.mql5.com/en/articles/7438#node03)
- [What's next?](https://www.mql5.com/en/articles/7438#node04)


This is the third article about the concept of [pending requests](https://www.mql5.com/en/articles/7394). In
this article, we are going to complete the tests of the concept by creating the methods for closing positions, removing orders and modifying
position stop orders and pending order parameters that can be modified.

Besides, we will slightly improve the [abstract order](https://www.mql5.com/en/articles/5654#node04) class by
adding the return of the two order and position property values — order filling and expiration types. The code of all trading methods of the
basic cross-platform trading object has been slightly optimized. There is no point in dwelling on the changes. Instead, I will display a
single example of one of the changed methods.

### Preparing data

We will need text messages to describe added properties of the abstract order and the message for the pending request class.

**Add indices of these messages to the Datas.mqh file:**

```
   MSG_ORD_TIME_EXP,                                  // Expiration date
   MSG_ORD_TYPE_FILLING,                              // Type of execution by remainder
   MSG_ORD_TYPE_TIME,                                 // Order lifetime
   MSG_ORD_TYPE,                                      // Type
```

...

```
   MSG_LIB_TEXT_PEND_REQUEST_CREATED,                 // Pending request created
   MSG_LIB_TEXT_PEND_REQUEST_DELETED,                 // Pending request is removed due to its expiration
   MSG_LIB_TEXT_PEND_REQUEST_EXECUTED,                // Pending request is removed due to its execution
```

**and the text messages corresponding to the declared indices:**

```
   {"Дата экспирации","Date of expiration"},
   {"Тип исполнения по остатку","Order filling type"},
   {"Время жизни ордера","Order lifetime"},
   {"Тип","Type"},
```

...

```
   {"Создан отложенный запрос","Pending request created"},
   {"Отложенный запрос удалён в связи с окончанием времени его действия","Pending request deleted due to expiration"},
   {"Отложенный запрос удалён в связи с его исполнением","Pending request deleted due to execution"},
```

Since we add two more properties to the abstract order object, make changes in its properties in the **Defines.mqh**
file.

**Add the two new properties to the block of integer order**
**properties and increase the total number of integer properties from 24 to 26:**

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
   ORDER_PROP_TYPE_FILLING,                                 // Type of execution by remainder
   ORDER_PROP_TYPE_TIME,                                    // Order lifetime
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
#define ORDER_PROP_INTEGER_TOTAL    (26)                    // Total number of integer properties
#define ORDER_PROP_INTEGER_SKIP     (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

**Add the ability to sort orders by these two properties:**

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
   SORT_BY_ORDER_TYPE_FILLING,                              // Sort by execution type by remainder
   SORT_BY_ORDER_TYPE_TIME,                                 // Sort by order lifetime
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

Since we now completely switch to pending requests when receiving server errors requiring some waiting (instead of [Sleep()](https://www.mql5.com/en/docs/common/sleep)
delaying the execution of the remaining program logic while waiting), then **from the enumeration describing EA behavior when**
**handling errors, remove the pending request creation option** **—**
**it will be created by default:**

```
//+------------------------------------------------------------------+
//| EA behavior when handling errors                                 |
//+------------------------------------------------------------------+
enum ENUM_ERROR_HANDLING_BEHAVIOR
  {
   ERROR_HANDLING_BEHAVIOR_BREAK,                           // Abort trading attempt
   ERROR_HANDLING_BEHAVIOR_CORRECT,                         // Correct invalid parameters
   ERROR_HANDLING_BEHAVIOR_PENDING_REQUEST,                 // Create a pending request
  };
//+------------------------------------------------------------------+
```

**Now this enumeration looks as follows:**

```
//+------------------------------------------------------------------+
//| EA behavior when handling errors                                 |
//+------------------------------------------------------------------+
enum ENUM_ERROR_HANDLING_BEHAVIOR
  {
   ERROR_HANDLING_BEHAVIOR_BREAK,                           // Abort trading attempt
   ERROR_HANDLING_BEHAVIOR_CORRECT,                         // Correct invalid parameters
  };
//+------------------------------------------------------------------+
```

We also do not need this option in the enumeration describing the methods of handling server return codes. Let's **remove**
**it:**

```
//+------------------------------------------------------------------+
//| The methods of handling errors and server return codes           |
//+------------------------------------------------------------------+
enum ENUM_ERROR_CODE_PROCESSING_METHOD
  {
   ERROR_CODE_PROCESSING_METHOD_OK,                         // No errors
   ERROR_CODE_PROCESSING_METHOD_DISABLE,                    // Disable trading for the EA
   ERROR_CODE_PROCESSING_METHOD_EXIT,                       // Exit the trading method
   ERROR_CODE_PROCESSING_METHOD_CORRECT,                    // Correct trading request parameters and repeat
   ERROR_CODE_PROCESSING_METHOD_REFRESH,                    // Update data and repeat
   ERROR_CODE_PROCESSING_METHOD_PENDING,                    // Create a pending request
   ERROR_CODE_PROCESSING_METHOD_WAIT,                       // Wait and repeat
  };
//+------------------------------------------------------------------+
```

**Now this enumeration looks as follows:**

```
//+------------------------------------------------------------------+
//| The methods of handling errors and server return codes           |
//+------------------------------------------------------------------+
enum ENUM_ERROR_CODE_PROCESSING_METHOD
  {
   ERROR_CODE_PROCESSING_METHOD_OK,                         // No errors
   ERROR_CODE_PROCESSING_METHOD_DISABLE,                    // Disable trading for the EA
   ERROR_CODE_PROCESSING_METHOD_EXIT,                       // Exit the trading method
   ERROR_CODE_PROCESSING_METHOD_CORRECT,                    // Correct trading request parameters and repeat
   ERROR_CODE_PROCESSING_METHOD_REFRESH,                    // Update data and repeat
   ERROR_CODE_PROCESSING_METHOD_WAIT,                       // Wait and repeat
  };
//+------------------------------------------------------------------+
```

Add some new properties to the pending request parameters. To avoid resorting to the **Defines.mqh** any more, **add**
**the enumeration with possible pending request sorting criteria to its very end** **. This is necessary in order to find such requests:**

```
//+------------------------------------------------------------------+
//| Possible pending request sorting criteria                        |
//+------------------------------------------------------------------+
enum ENUM_SORT_PEND_REQ_MODE
  {
   SORT_BY_PEND_REQ_ID = 0,                                 // Sort by ID
   SORT_BY_PEND_REQ_TYPE,                                   // Sort by type
   SORT_BY_PEND_REQ_TICKET,                                 // Sort by ticket
  };
//+------------------------------------------------------------------+
```

Add the necessary changes to the **COrder** abstract order class in the **Order.mqh** file.

**Add the methods returning the new properties to the block**
**of methods for a simplified access to the order object integer properties:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Return (1) ticket, (2) parent order ticket, (3) derived order ticket, (4) magic number, (5) order reason,
//--- (6) position ID, (7) opposite position ID, (8) first group ID, (9) second group ID,
//--- (10) pending request ID, (11) magic number ID, (12) type, (13) flag of closing by StopLoss,
//--- (14) flag of closing by TakeProfit (15) open time, (16) close time,
//--- (17) order expiration date, (18) state, (19) status, (20) type by direction, (21) execution type by remainder, (22) order lifetime
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
   ENUM_ORDER_TYPE_FILLING TypeFilling(void)                            const { return (ENUM_ORDER_TYPE_FILLING)this.GetProperty(ORDER_PROP_TYPE_FILLING);  }
   ENUM_ORDER_TYPE_TIME TypeTime(void)                                  const { return (ENUM_ORDER_TYPE_TIME)this.GetProperty(ORDER_PROP_TYPE_TIME);        }
```

The methods simply return values set in the integer parameters corresponding to these properties.

**Add filling the two new order object properties in the**
**class closed constructor:**

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
   this.m_long_prop[ORDER_PROP_TYPE_FILLING]                         = this.OrderTypeFilling();
   this.m_long_prop[ORDER_PROP_TYPE_TIME]                            = this.OrderTypeTime();
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

The values returned by the **OrderTypeFilling()** and **OrderTypeTime()** methods for obtaining the properties are
added to the appropriate values of the order property array.

**Add the description of the two new properties to the**
**method returning the descriptions of the order object integer properties:**

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
          (this.GetProperty(property)==0     ?  CMessage::Text(MSG_LIB_PROP_NOT_SET)            :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS))
         )  :
      property==ORDER_PROP_TYPE_FILLING      ?  CMessage::Text(MSG_ORD_TYPE_FILLING)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+OrderTypeFillingDescription((ENUM_ORDER_TYPE_FILLING)this.GetProperty(property))
         )  :
      property==ORDER_PROP_TYPE_TIME         ?  CMessage::Text(MSG_ORD_TYPE_TIME)+
         (!this.SupportProperty(property)    ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+OrderTypeTimeDescription((ENUM_ORDER_TYPE_TIME)this.GetProperty(property))
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

We have considered the operation of the method [in the first article](https://www.mql5.com/en/articles/5654#node04)
devoted to the creation of the abstract order object, so there is no point in dwelling on it here.

In the previous articles, we implemented the methods of [opening positions](https://www.mql5.com/en/articles/7394)
and [placing pending orders](https://www.mql5.com/en/articles/7418) using pending request objects. In order to identify
each pending request, we used its unique ID set to a position or order magic number when opening a position or placing an order. Therefore, it
was easy to identify the presence of a previously created pending request and find out how a position was opened (via a common trading request
or as a result of the pending request object operation). This is necessary in order to avoid creating multiple pending request objects for
the same trading action.

However, when closing a position, removing or modifying an order, the ID is not helpful since we can conduct trading operations with any orders or
positions. They are performed by a ticket. Therefore, we need the ticket number to understand what order/position the pending request was
created for. The ticket is present in the trading request structure of the pending request object. Its value allows us to accurately define
the order or position ticket the pending request has been created for. We will also need to define successful handling of a pending request.
In particular, we should define when a trading order was sent to the server using the request before being successfully executed. Next, we
should determine whether the pending trading request is successful and remove the request object to avoid unnecessary trading attempt
after the previous one completes successfully.

We have already prepared everything for that — [class of trading events in an \\
account and the collection class of these events](https://www.mql5.com/en/articles/6211). The trading event collection class already provides us complete access to all events
occurred since the program launch, as well as to the current events occurring "here and now". Therefore, in order to access the trading event
collection class, we need to pass the pointer to the trading event collection class into the trading class. To achieve this, we need the class
to be able to return itself. Let's add this feature to it.

**Add the GetObject() public method to the file of the**
**EventsCollection.mqh trading event class:**

```
//+------------------------------------------------------------------+
//| Collection of account trading events                             |
//+------------------------------------------------------------------+
class CEventsCollection : public CBaseObj
  {
private:
   CListObj          m_list_trade_events;             // List of events
   bool              m_is_hedge;                      // Hedge account flag
   int               m_trade_event_code;              // Trading event code
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property
   ulong             m_position_id;                   // Position ID (MQL4)
   ENUM_ORDER_TYPE   m_type_first;                    // Opening order type (MQL4)

//--- Create a trading event depending on the (1) order status and (2) change type
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market,CArrayObj* list_control);
   void              CreateNewEvent(COrderControl* order);
//--- Create an event for a (1) hedging account, (2) netting account
   void              NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
   void              NewDealEventNetto(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
//--- Select and return the list of (1) market pending orders and (2) open positions
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
   CArrayObj*        GetListPositions(CArrayObj* list);
//--- Select and return the list of historical (1) closed positions,
//--- (2) removed pending orders, (3) deals, (4) all closing orders
   CArrayObj*        GetListHistoryPositions(CArrayObj* list);
   CArrayObj*        GetListHistoryPendings(CArrayObj* list);
   CArrayObj*        GetListDeals(CArrayObj* list);
   CArrayObj*        GetListCloseByOrders(CArrayObj* list);
//--- Return the list of (1) all position orders by its ID, (2) all deal positions by its ID
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
//--- Return (1) the control order by a ticket, (2) the type of the opening order by the position ticket (MQL4)
   COrderControl*    GetOrderControlByTicket(CArrayObj* list,const ulong ticket);
   ENUM_ORDER_TYPE   GetTypeFirst(CArrayObj* list,const ulong ticket);
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);
//--- The handler of an existing order/position change event
   void              OnChangeEvent(CArrayObj* list_changes,const int index);

public:
//--- Return itself
   CEventsCollection*GetObject(void)                                                                     { return &this;                                                         }
//--- Select events from the collection with time within the range from begin_time to end_time
   CArrayObj        *GetListByTime(const datetime begin_time=0,const datetime end_time=0);
//--- Return the full event collection list "as is"
   CArrayObj        *GetList(void)                                                                       { return &this.m_list_trade_events;                                     }
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
//--- Return (1) the last trading event on an account, (2) base event object by index and (3) number of new events
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)                const { return this.m_trade_event;                 }
   CEventBaseObj    *GetTradeEventByIndex(const int index)        { return this.GetEvent(index,false);         }
   int               GetTradeEventsTotal(void)              const { return this.m_list_events.Total();         }
//--- Reset the last trading event
   void              ResetLastTradeEvent(void)                    { this.m_trade_event=TRADE_EVENT_NO_EVENT;   }
//--- Constructor
                     CEventsCollection(void);
  };
//+------------------------------------------------------------------+
```

Now the **CEventsCollection** class can return the pointer to itself. This is exactly what we will need later.

I have already mentioned that the base trading class has undergone some optimization regarding the trading method code. Let's use the
position opening method as an example. The changes mainly affected the block of
sending a trading order to the server.

**Previously, the code looked as follows:**

```
//+------------------------------------------------------------------+
//| Open a position                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::OpenPosition(const ENUM_POSITION_TYPE type,
                             const double volume,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=ULONG_MAX,
                             const string comment=NULL,
                             const ulong deviation=ULONG_MAX,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   ::ResetLastError();
   //--- If failed to get the current prices, write the error code and description, send the message to the journal and return 'false'
   if(!::SymbolInfoTick(this.m_symbol,this.m_tick))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_PRICE),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action   =  TRADE_ACTION_DEAL;
   this.m_request.symbol   =  this.m_symbol;
   this.m_request.magic    =  (magic==ULONG_MAX ? this.m_magic : magic);
   this.m_request.type     =  (ENUM_ORDER_TYPE)type;
   this.m_request.price    =  (type==POSITION_TYPE_BUY ? this.m_tick.ask : this.m_tick.bid);
   this.m_request.volume   =  volume;
   this.m_request.sl       =  sl;
   this.m_request.tp       =  tp;
   this.m_request.deviation=  (deviation==ULONG_MAX ? this.m_deviation : deviation);
   this.m_request.comment  =  (comment==NULL ? this.m_comment : comment);
   //--- Return the result of sending a request to the server
#ifdef __MQL5__
   return(!this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result));
#else
   ::ResetLastError();
   int ticket=::OrderSend(m_request.symbol,m_request.type,m_request.volume,m_request.price,(int)m_request.deviation,m_request.sl,m_request.tp,m_request.comment,(int)m_request.magic,m_request.expiration,clrNONE);
   ::SymbolInfoTick(this.m_symbol,this.m_tick);
   if(ticket!=WRONG_VALUE)
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.ask=this.m_tick.ask;
      this.m_result.bid=this.m_tick.bid;
      this.m_result.deal=ticket;
      this.m_result.price=(::OrderSelect(ticket,SELECT_BY_TICKET) ? ::OrderOpenPrice() : this.m_request.price);
      this.m_result.volume=(::OrderSelect(ticket,SELECT_BY_TICKET) ? ::OrderLots() : this.m_request.volume);
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      return true;
     }
   else
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.ask=this.m_tick.ask;
      this.m_result.bid=this.m_tick.bid;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      return false;
     }
#endif
  }
//+------------------------------------------------------------------+
```

If we look closely, we can see a lot of repetitive code constructions. All of them have been placed outside the ticket validation block.
Besides, there has been a logical error when resetting the last error code in MQL4 — it has been reset before sending a trading request to the
server, while the price receiving function (which may also return the error) is set before receiving the error code. In this case, the last
error code no longer belongs to the trade server return code in MQL4.

Considering all of the above, **the block of sending a trading order to the server in the position opening method looks as follows:**

```
//+------------------------------------------------------------------+
//| Open a position                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::OpenPosition(const ENUM_POSITION_TYPE type,
                             const double volume,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=ULONG_MAX,
                             const string comment=NULL,
                             const ulong deviation=ULONG_MAX,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   ::ResetLastError();
   //--- If failed to get the current prices, write the error code and description, send the message to the journal and return 'false'
   if(!::SymbolInfoTick(this.m_symbol,this.m_tick))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_PRICE),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action   =  TRADE_ACTION_DEAL;
   this.m_request.symbol   =  this.m_symbol;
   this.m_request.magic    =  (magic==ULONG_MAX ? this.m_magic : magic);
   this.m_request.type     =  (ENUM_ORDER_TYPE)type;
   this.m_request.price    =  (type==POSITION_TYPE_BUY ? this.m_tick.ask : this.m_tick.bid);
   this.m_request.volume   =  volume;
   this.m_request.sl       =  sl;
   this.m_request.tp       =  tp;
   this.m_request.deviation=  (deviation==ULONG_MAX ? this.m_deviation : deviation);
   this.m_request.comment  =  (comment==NULL ? this.m_comment : comment);
   //--- Return the result of sending a request to the server
#ifdef __MQL5__
   return(!this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result));
#else
   ::ResetLastError();
   int ticket=::OrderSend(m_request.symbol,m_request.type,m_request.volume,m_request.price,(int)m_request.deviation,m_request.sl,m_request.tp,m_request.comment,(int)m_request.magic,m_request.expiration,clrNONE);
   this.m_result.retcode=::GetLastError();
   ::SymbolInfoTick(this.m_symbol,this.m_tick);
   this.m_result.ask=this.m_tick.ask;
   this.m_result.bid=this.m_tick.bid;
   this.m_result.comment=CMessage::Text(this.m_result.retcode);
   if(ticket!=WRONG_VALUE)
     {
      this.m_result.deal=ticket;
      this.m_result.price=(::OrderSelect(ticket,SELECT_BY_TICKET) ? ::OrderOpenPrice() : this.m_request.price);
      this.m_result.volume=(::OrderSelect(ticket,SELECT_BY_TICKET) ? ::OrderLots() : this.m_request.volume);
      return true;
     }
   else
     {
      return false;
     }
#endif
  }
//+------------------------------------------------------------------+
```

All the remaining class trading methods have been improved in the same way with slight differences considering specifics of each method.


You can see all the changes in the codes attached below.

Now let's start developing the methods for working with pending requests to close positions/remove orders and modify position and order
parameters.

### Pending requests to close/remove/modify positions and pending orders

**In the Trading.mqh base trading class file, include the class of the**
**account trading event collection:**

```
//+------------------------------------------------------------------+
//|                                                      Trading.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayInt.mqh>
#include "Objects\Trade\TradeObj.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\EventsCollection.mqh"
//+------------------------------------------------------------------+
```

In the class of the **CPendingReq** pending request whose listing is located in the same file, **add the**
**methods of returning the order ticket, position ticket and trading**
**operation type:**

```
//+------------------------------------------------------------------+
//| Pending request object class                                     |
//+------------------------------------------------------------------+
class CPendingReq : public CObject
  {
private:
   MqlTradeRequest      m_request;                       // Trade request structure
   uchar                m_id;                            // Trading request ID
   int                  m_type;                          // Pending request type
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
//--- Return (1) the request structure, (2) the price at the moment of the request generation,
//--- (3) request generation time, (4) next attempt activation time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
//--- (9) result a request is based on,
//--- (10) order ticket, (11) position ticket, (12) trading operation type
   MqlTradeRequest      MqlRequest(void)                       const { return this.m_request;         }
   double               PriceCreate(void)                      const { return this.m_price_create;    }
   ulong                TimeCreate(void)                       const { return this.m_time_create;     }
   ulong                TimeActivate(void)                     const { return this.m_time_activate;   }
   ulong                WaitingMSC(void)                       const { return this.m_waiting_msc;     }
   uchar                CurrentAttempt(void)                   const { return this.m_current_attempt; }
   uchar                TotalAttempts(void)                    const { return this.m_total_attempts;  }
   uchar                ID(void)                               const { return this.m_id;              }
   int                  Retcode(void)                          const { return this.m_retcode;         }
   ulong                Order(void)                            const { return this.m_request.order;   }
   ulong                Position(void)                         const { return this.m_request.position;}
   ENUM_TRADE_REQUEST_ACTIONS Action(void)                     const { return this.m_request.action;  }
```

**Also, add the methods of placing order and position**
**tickets:**

```
//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) ID,
//--- (8) order ticket, (9) position ticket
   void                 SetPriceCreate(const double price)           { this.m_price_create=price;     }
   void                 SetTimeCreate(const ulong time)              { this.m_time_create=time;       }
   void                 SetTimeActivate(const ulong time)            { this.m_time_activate=time;     }
   void                 SetWaitingMSC(const ulong miliseconds)       { this.m_waiting_msc=miliseconds;}
   void                 SetCurrentAttempt(const uchar number)        { this.m_current_attempt=number; }
   void                 SetTotalAttempts(const uchar number)         { this.m_total_attempts=number;  }
   void                 SetID(const uchar id)                        { this.m_id=id;                  }
   void                 SetOrder(const ulong ticket)                 { this.m_request.order=ticket;   }
   void                 SetPosition(const ulong ticket)              { this.m_request.position=ticket;}

```

The Compare() method for comparing CPendingReq class objects by specified properties has been changed. **Previously, it looked as follows:**

```
//+------------------------------------------------------------------+
//| Compare CPendingReq objects by properties                        |
//+------------------------------------------------------------------+
int CPendingReq::Compare(const CObject *node,const int mode=0) const
  {
   const CPendingReq *compared_req=node;
   return
     (
      //--- Compare by ID
      mode==0  ?
      (this.ID()>compared_req.ID() ? 1 : this.ID()<compared_req.ID() ? -1 : 0)   :
      //--- Compare by type
      (this.Type()>compared_req.Type() ? 1 : this.Type()<compared_req.Type() ? -1 : 0)
     );
  }
//+------------------------------------------------------------------+
```

In other words, the comparison has been performed only by two pending request object properties.

Now that we have
the third one, **the method looks as follows:**

```
//+------------------------------------------------------------------+
//| Compare CPendingReq objects by properties                        |
//+------------------------------------------------------------------+
int CPendingReq::Compare(const CObject *node,const int mode=0) const
  {
   const CPendingReq *compared_req=node;
   return
     (
      //--- Compare by ID
      mode==SORT_BY_PEND_REQ_ID        ?
      (this.ID()>compared_req.ID()     ? 1 : this.ID()<compared_req.ID() ? -1 : 0)     :
      //--- Compare by type
      mode==SORT_BY_PEND_REQ_TYPE      ?
      (this.Type()>compared_req.Type() ? 1 : this.Type()<compared_req.Type() ? -1 : 0) :
      //--- Compare by ticket
      (
       //--- modifying position sl, tp, opening/closing a position or closing by an opposite one
       this.m_request.action==TRADE_ACTION_SLTP    || this.m_request.action==TRADE_ACTION_DEAL     || this.m_request.action==TRADE_ACTION_CLOSE_BY ?
       (this.m_request.position>compared_req.m_request.position ? 1 : this.m_request.position<compared_req.m_request.position ? -1 : 0)            :
       //--- modifying parameters, placing/removing a pending order
       this.m_request.action==TRADE_ACTION_MODIFY  || this.m_request.action==TRADE_ACTION_PENDING  || this.m_request.action==TRADE_ACTION_REMOVE   ?
       (this.m_request.order>compared_req.m_request.order ? 1 : this.m_request.order<compared_req.m_request.order ? -1 : 0)                        :
       //--- otherwise
       0
      )
     );
  }
//+------------------------------------------------------------------+
```

- If the comparison mode passed to the method is equal to the **SORT\_BY\_PEND\_REQ\_ID** constant value from the
ENUM\_SORT\_PEND\_REQ\_MODE enumeration, the comparison is performed by a pending request object ID;
- If the comparison mode is **SORT\_BY\_PEND\_REQ\_TYPE**, comparing is performed by a pending request object type;
- The third comparison mode (SORT\_BY\_PEND\_REQ\_TICKET) works as follows:

\- check the trading operation type, and
  - if this is a modification of a position stop order, opening/closing a position or closing by an opposite one, the comparison is
     performed by the values of the **position** fields of the trading request structure storing position tickets;
  - if this is a modification of parameters, placing or removing a pending order, the comparison is performed by the values of the **order** fields
     of the trading request structure storing order tickets;
  - otherwise, zero is returned.

**In the private section of the CTrading trading class, declare the**
**pointer to the collection class of the account trading events:**

```
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading
  {
private:
   CAccount            *m_account;                       // Pointer to the current account object
   CSymbolsCollection  *m_symbols;                       // Pointer to the symbol collection list
   CMarketCollection   *m_market;                        // Pointer to the list of the collection of market orders and positions
   CHistoryCollection  *m_history;                       // Pointer to the list of the collection of historical orders and deals
   CEventsCollection   *m_events;                        // Pointer to the event collection list
   CArrayObj            m_list_request;                  // List of pending requests
   CArrayInt            m_list_errors;                   // Error list
   bool                 m_is_trade_disable;              // Flag disabling trading
   bool                 m_use_sound;                     // The flag of using sounds of the object trading events
   uchar                m_total_try;                     // Number of trading attempts
   ENUM_LOG_LEVEL       m_log_level;                     // Logging level
   MqlTradeRequest      m_request;                       // Trading request prices
   ENUM_TRADE_REQUEST_ERR_FLAGS m_error_reason_flags;    // Flags of error source in a trading method
   ENUM_ERROR_HANDLING_BEHAVIOR m_err_handling_behavior; // Behavior when handling error

```

**In the same section, declare the methods returning the pending request**
**object index by the order and position tickets:**

```
//--- Return the request object index in the list by (1) ID,
//--- (2) order ticket, (3) position ticket in the request
   int                  GetIndexPendingRequestByID(const uchar id);
   int                  GetIndexPendingRequestByOrder(const ulong ticket);
   int                  GetIndexPendingRequestByPosition(const ulong ticket);

public:
```

**Let's write their implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Return the request object index in the list by the order ticket  |
//+------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByOrder(const ulong ticket)
  {
   CPendingReq *req=new CPendingReq();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetOrder(ticket);
   this.m_list_request.Sort(SORT_BY_PEND_REQ_TICKET);
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+-------------------------------------------------------------------+
//| Return the request object index in the list by the position ticket|
//+-------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByPosition(const ulong ticket)
  {
   CPendingReq *req=new CPendingReq();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetPosition(ticket);
   this.m_list_request.Sort(SORT_BY_PEND_REQ_TICKET);
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+------------------------------------------------------------------+
```

We have already considered a similar method for returning the request object index by its ID [in \\
the previous article](https://www.mql5.com/en/articles/7418). You can always re-read its description to understand the principles of its operation. The current method is
exactly the same.

In the public section of the class, supplement the class initialization method allowing to pass the pointers to all the necessary
collection lists to the trading class.

**Add passing the pointer to the trading event collection**
**and writing the pointer to the m\_events variable value:**

```
public:
//--- Constructor
                        CTrading();
//--- Timer
   void                 OnTimer(void);
//--- Get the pointers to the lists (make sure to call the method in program's OnInit() since the symbol collection list is created there)
   void                 OnInit(CAccount *account,CSymbolsCollection *symbols,CMarketCollection *market,CHistoryCollection *history,CEventsCollection *events)
                          {
                           this.m_account=account;
                           this.m_symbols=symbols;
                           this.m_market=market;
                           this.m_history=history;
                           this.m_events=events;
                          }
```

**Add passing** **an**
**order or position ticket to the method of checking limitations and errors:**

```
//--- Check limitations and errors
   ENUM_ERROR_CODE_PROCESSING_METHOD CheckErrors(const double volume,
                                                 const double price,
                                                 const ENUM_ACTION_TYPE action,
                                                 const ENUM_ORDER_TYPE order_type,
                                                 CSymbol *symbol_obj,
                                                 CTradeObj *trade_obj,
                                                 const string source_method,
                                                 const double limit=0,
                                                 double sl=0,
                                                 double tp=0,
                                                 ulong ticket=0);
```

The ticket allows us to define whether the method contains a previously created pending request object for an order or a position having that
ticket. This is done to avoid displaying the same message about the same error in the journal. In other words, when the server returns the
error for the first time, we display the error message in the journal, create a pending request and exit the trading method. If the error
repeats, we check if a previously created pending request object for an order or a position having that ticket is present. If yes, the error
message has already been displayed. There is no point in repeating it.

**Let's see the implementation of the improved method:**

```
//+------------------------------------------------------------------+
//| Check limitations and errors                                     |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::CheckErrors(const double volume,
                                                        const double price,
                                                        const ENUM_ACTION_TYPE action,
                                                        const ENUM_ORDER_TYPE order_type,
                                                        CSymbol *symbol_obj,
                                                        CTradeObj *trade_obj,
                                                        const string source_method,
                                                        const double limit=0,
                                                        double sl=0,
                                                        double tp=0,
                                                        ulong ticket=0)
  {
//--- Check the previously set flag disabling trading for an EA
   if(this.IsTradingDisable())
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_FATAL_ERROR;
      this.AddErrorCodeToList(MSG_LIB_TEXT_TRADING_DISABLE);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return ERROR_CODE_PROCESSING_METHOD_DISABLE;
     }
//--- result of all checks and error flags
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   bool res=true;
//--- Clear the error list
   this.m_list_errors.Clear();
   this.m_list_errors.Sort();
//--- Check trading limitations
   res &=this.CheckTradeConstraints(volume,action,symbol_obj,source_method,sl,tp);
//--- Check the funds sufficiency for opening positions/placing orders
   if(action<ACTION_TYPE_CLOSE_BY)
      res &=this.CheckMoneyFree(volume,price,order_type,symbol_obj,source_method);
//--- Check parameter values by StopLevel and FreezeLevel
   res &=this.CheckLevels(action,order_type,price,limit,sl,tp,symbol_obj,source_method);

//--- If there are limitations and errors, display the header and the error list
   if(!res)
     {
      //--- Declare the pending request index
      int index=WRONG_VALUE;
      //--- Request was rejected before sending to the server due to:
      int total=this.m_list_errors.Total();
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         //--- in case of closing by an opposite position or opening/closing a position, the index is received by the position ticket
         if(action==ACTION_TYPE_CLOSE_BY || action<ACTION_TYPE_BUY_LIMIT)
            index=this.GetIndexPendingRequestByPosition(ticket);
         //--- if a trading operation is closing a position, removing an order or modifying it...
         else if(action>ACTION_TYPE_CLOSE_BY)
           {
            //--- if this is a position, receive the index by a position ticket
            if(order_type<ORDER_TYPE_BUY_LIMIT)
               index=this.GetIndexPendingRequestByPosition(ticket);
            //--- otherwise, receive the index by an order ticket
            else
               index=this.GetIndexPendingRequestByOrder(ticket);
           }
         //--- otherwise, if a trading operation is placing a pending order, receive the index by an order ticket
         else index=this.GetIndexPendingRequestByOrder(ticket);

         //--- if no index is received, there is no such pending request object, display the error message
         if(index==WRONG_VALUE)
           {
            //--- For MQL5, first display the list header followed by the error list
            #ifdef __MQL5__
            ::Print(source_method,CMessage::Text(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK ? MSG_LIB_TEXT_REQUEST_REJECTED_DUE : MSG_LIB_TEXT_INVALID_REQUEST));
            for(int i=0;i<total;i++)
               ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
            //--- For MQL4, the journal messages are displayed in the reverse order: the error list in the reverse loop is followed by the list header
            #else
            for(int i=total-1;i>WRONG_VALUE;i--)
               ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
            ::Print(source_method,CMessage::Text(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK ? MSG_LIB_TEXT_REQUEST_REJECTED_DUE : MSG_LIB_TEXT_INVALID_REQUEST));
            #endif
           }
        }
      //--- If the action is performed at the "abort trading operation" error
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
         return ERROR_CODE_PROCESSING_METHOD_EXIT;
      //--- If the action is performed at the "correct parameters" error
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_CORRECT)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG && index==WRONG_VALUE)
            ::Print(CMessage::Text(MSG_LIB_TEXT_CORRECTED_TRADE_REQUEST));
         //--- Return the result of an attempt to correct the request parameters
         return this.RequestErrorsCorrecting(this.m_request,order_type,trade_obj.SpreadMultiplier(),symbol_obj,trade_obj);
        }
     }
//--- No limitations and errors
   trade_obj.SetResultRetcode(0);
   trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

The entire logic is described in the code comments. I believe, everything is clear there.

**Handling pending trading requests has been implemented in the class timer.**

**I have made quite a lot of changes and done**
**my best to describe the entire operation logic in the code comments:**

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

      //--- get the request structure and the symbol object a trading operation should be performed for
      MqlTradeRequest request=req_obj.MqlRequest();
      CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(request.symbol);
      if(symbol_obj==NULL || !symbol_obj.RefreshRates())
         continue;

      //--- if the current attempt exceeds the defined number of trading attempts,
      //--- or the current time exceeds the waiting time of all attempts
      //--- remove the current request object and move on to the next one
      if(req_obj.CurrentAttempt()>req_obj.TotalAttempts() || req_obj.CurrentAttempt()>=UCHAR_MAX ||
         (long)symbol_obj.Time()>long(req_obj.TimeCreate()+req_obj.WaitingMSC()*req_obj.TotalAttempts()))
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
         this.m_list_request.Delete(i);
         continue;
        }

      //--- If this is a position opening or placing a pending order
      if((req_obj.Action()==TRADE_ACTION_DEAL && req_obj.Position()==0) || req_obj.Action()==TRADE_ACTION_PENDING)
        {
         //--- Get the pending request ID
         uchar id=this.GetPendReqID((uint)request.magic);
         //--- Get the list of orders/positions containing the order/position with the pending request ID
         CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
         if(::CheckPointer(list)==POINTER_INVALID)
            continue;
         //--- If the order/position is present, the request is handled: remove it and proceed to the next
         if(list.Total()>0)
           {
            if(this.m_log_level>LOG_LEVEL_NO_MSG)
               ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
            this.m_list_request.Delete(i);
            continue;
           }
        }
      //--- Otherwise: full and partial position closure, removing an order, modifying order parameters and position stop orders
      else
        {
         CArrayObj *list=NULL;
         //--- if this is a position closure, including a closure by an opposite one
         if((req_obj.Action()==TRADE_ACTION_DEAL && req_obj.Position()>0) || req_obj.Action()==TRADE_ACTION_CLOSE_BY)
           {
            //--- Get a position with the necessary ticket from the list of open positions
            list=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Position(),EQUAL);
            if(::CheckPointer(list)==POINTER_INVALID)
               continue;
            //--- If the market has no such position - the request is handled: remove it and proceed to the next one
            if(list.Total()==0)
              {
               if(this.m_log_level>LOG_LEVEL_NO_MSG)
                  ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
               this.m_list_request.Delete(i);
               continue;
              }
            //--- Otherwise, if the position still exists, this is a partial closure
            else
              {
               //--- Get the list of all account trading events
               list=this.m_events.GetList();
               if(list==NULL)
                  continue;
               //--- In the loop from the end of the account trading event list
               int events_total=list.Total();
               for(int j=events_total-1; j>WRONG_VALUE; j--)
                 {
                  //--- get the next trading event
                  CEvent *event=list.At(j);
                  if(event==NULL)
                     continue;
                  //--- If this event is a partial closure or there was a partial closure when closing by an opposite one
                  if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL || event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS)
                    {
                     //--- If a position ticket in a trading event coincides with the ticket in a pending trading request -
                     //--- the request is handled: remove it and break the loop by the list of account trading events
                     if(event.TicketFirstOrderPosition()==req_obj.Position())
                       {
                        if(this.m_log_level>LOG_LEVEL_NO_MSG)
                           ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                        this.m_list_request.Delete(i);
                        break;
                       }
                    }
                 }
               //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one
               if(::CheckPointer(req_obj)==POINTER_INVALID)
                  continue;
              }
           }
         //--- If this is a modification of position stop orders
         if(req_obj.Action()==TRADE_ACTION_SLTP)
           {
            //--- Get the list of all account trading events
            list=this.m_events.GetList();
            if(list==NULL)
               continue;
            //--- In the loop from the end of the account trading event list
            int events_total=list.Total();
            for(int j=events_total-1; j>WRONG_VALUE; j--)
              {
               //--- get the next trading event
               CEvent *event=list.At(j);
               if(event==NULL)
                  continue;
               //--- If this is a change of the position's stop orders
               if(event.TypeEvent()>TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT)
                 {
                  //--- If a position ticket in a trading event coincides with the ticket in a pending trading request -
                  //--- the request is handled: remove it and break the loop by the list of account trading events
                  if(event.TicketFirstOrderPosition()==req_obj.Position())
                    {
                     if(this.m_log_level>LOG_LEVEL_NO_MSG)
                        ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                     this.m_list_request.Delete(i);
                     break;
                    }
                 }
              }
            //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one
            if(::CheckPointer(req_obj)==POINTER_INVALID)
               continue;
           }
         //--- If this is a pending order removal
         if(req_obj.Action()==TRADE_ACTION_REMOVE)
           {
            //--- Get the list of removed pending orders
            list=this.m_history.GetList(ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
            if(::CheckPointer(list)==POINTER_INVALID)
               continue;
            //--- Leave a single order with the necessary ticket in the list
            list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,req_obj.Order(),EQUAL);
            //--- If the order is present, the request is handled: remove it and proceed to the next
            if(list.Total()>0)
              {
               if(this.m_log_level>LOG_LEVEL_NO_MSG)
                  ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
               this.m_list_request.Delete(i);
               continue;
              }
           }
         //--- If this is a pending order modification
         if(req_obj.Action()==TRADE_ACTION_MODIFY)
           {
            //--- Get the list of all account trading events
            list=this.m_events.GetList();
            if(list==NULL)
               continue;
            //--- In the loop from the end of the account trading event list
            int events_total=list.Total();
            for(int j=events_total-1; j>WRONG_VALUE; j--)
              {
               //--- get the next trading event
               CEvent *event=list.At(j);
               if(event==NULL)
                  continue;
               //--- If this event involves any change of modified pending order parameters
               if(event.TypeEvent()>TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER && event.TypeEvent()<TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT)
                 {
                  //--- If an order ticket in a trading event coincides with the ticket in a pending trading request -
                  //--- the request is handled: remove it and break the loop by the list of account trading events
                  if(event.TicketOrderEvent()==req_obj.Order())
                    {
                     if(this.m_log_level>LOG_LEVEL_NO_MSG)
                        ::Print(req_obj.IdDescription(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                     this.m_list_request.Delete(i);
                     break;
                    }
                 }
              }
           }
        }
      //--- Exit if the pending request object has been removed after checking its operation
      if(::CheckPointer(req_obj)==POINTER_INVALID)
         return;
      //--- Set the request activation time in the request object
      req_obj.SetTimeActivate(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.CurrentAttempt()+1));

      //--- If the current time is less than the request activation time,
      //--- this is not the request time - move on to the next request in the list
      if((long)symbol_obj.Time()<(long)req_obj.TimeActivate())
         continue;

      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

      //--- Display the number of a trading attempt in the journal

      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_TEXT_RE_TRY_N)+(string)req_obj.CurrentAttempt());

      //--- Depending on the type of action performed in the trading request
      switch(request.action)
        {
         //--- Opening/closing a position
         case TRADE_ACTION_DEAL :
            //--- If no ticket is present in the request structure - this is opening a position
            if(request.position==0)
               this.OpenPosition((ENUM_POSITION_TYPE)request.type,request.volume,request.symbol,request.magic,request.sl,request.tp,request.comment,request.deviation,request.type_filling);
            //--- If the ticket is present in the request structure - this is a position closure
            else
               this.ClosePosition(request.position,request.volume,request.comment,request.deviation);
            break;
         //--- Modify StopLoss/TakeProfit position
         case TRADE_ACTION_SLTP :
            this.ModifyPosition(request.position,request.sl,request.tp);
            break;
         //--- Close by an opposite one
         case TRADE_ACTION_CLOSE_BY :
            this.ClosePositionBy(request.position,request.position_by);
            break;
         //---
         //--- Place a pending order
         case TRADE_ACTION_PENDING :
            this.PlaceOrder(request.type,request.volume,request.symbol,request.price,request.stoplimit,request.sl,request.tp,request.magic,request.comment,request.expiration,request.type_time,request.type_filling);
            break;
         //--- Modify a pending order
         case TRADE_ACTION_MODIFY :
            this.ModifyOrder(request.order,request.price,request.sl,request.tp,request.stoplimit,request.expiration,request.type_time,request.type_filling);
            break;
         //--- Remove a pending order
         case TRADE_ACTION_REMOVE :
            Print(DFUN,RequestActionDescription(req_obj.MqlRequest()));
            this.DeleteOrder(request.order);
            break;
         //---
         default:
            break;
        }
     }
  }
//+------------------------------------------------------------------+
```

This is where we need the list of all trading events. It allows us to define a pending request activation and remove it. Otherwise, the pending
request would have been removed only after using all trading attempts allocated for it. For example, in case of a partial position closure,
such a pending request would have gradually closed part of a position time after time until the entire position had been closed completely or
all trading attempts had been used. This is wrong and should be amended. Therefore, we define the occurred trading event and compare it with
the pending trading request object. In case of a match, remove this object as executed and inform of that in the journal.

**In the method returning the way of handling errors, move the error of**
**the absence of the permission to trade from the terminal sideto the block**
**returning waiting:**

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
      case 4   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 5 seconds
      //--- No connection to the trade server
      case 6   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 20 seconds
      //--- Too frequent requests
      case 8   :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 10 seconds
      //--- No price
      case 136 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 5 seconds
      //--- Broker is busy
      case 137 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)15000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 15 seconds
      //--- Too many requests
      case 141 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 10 seconds
      //--- Modification denied because the order is too close to market
      case 145 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 10 seconds
      //--- Trade context is busy
      case 146 :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 5 seconds

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
      //--- Request handling error
      case 10011  :
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

      //--- Request rejected
      case 10006  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)15000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 15 seconds
      //--- No quotes to handle the request
      case 10021  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 5 seconds
      //--- Too frequent requests
      case 10024  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 10 seconds
      //--- Auto trading disabled by the client terminal
      case 10027  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 20 seconds
      //--- An order or a position is frozen
      case 10029  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 10 seconds
      //--- No connection to the trade server
      case 10031  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT; Wait 20 seconds

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

Thus, if we disable the AutoTrading button in the terminal and try to send a trading request, a pending request is generated. After enabling the
AutoTrading button, the pending request should be activated. In fact, such an error should be handled by waiting for auto trading
activation, but only in case the total waiting time allocated for all trading attempts is not up yet.

**In the error correction method, remove the code block returning from the**
**method using the "exit from trading method" code:**

```
//+------------------------------------------------------------------+
//| Correct errors                                                   |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::RequestErrorsCorrecting(MqlTradeRequest &request,
                                                                    const ENUM_ORDER_TYPE order_type,
                                                                    const uint spread_multiplier,
                                                                    CSymbol *symbol_obj,
                                                                    CTradeObj *trade_obj)
  {
//--- The empty error list means no errors are detected, return success
   int total=this.m_list_errors.Total();
   if(total==0)
      return ERROR_CODE_PROCESSING_METHOD_OK;

//--- Trading is disabled for the current account
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Trading on the trading server side is disabled for EAs on the current account
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Trading operations are disabled in the terminal
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Trading operations are disabled for the EA
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Disable trading on a symbol
```

**In the block handling the list of detected errors, add the return of**
**the 20 sec waiting code from the method if trading is disabled in the terminal or for the EA:**

```
//--- View the list of remaining errors and correct trading request parameters
   for(int i=0;i<total;i++)
     {
      int err=this.m_list_errors.At(i);
      if(err==NULL)
         continue;
      switch(err)
        {
         //--- Correct an invalid volume and disabling stop levels in a trading request
         case MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME    :
         case MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME    :
         case MSG_LIB_TEXT_INVALID_VOLUME_STEP        :  request.volume=symbol_obj.NormalizedLot(request.volume);                      break;
         case MSG_SYM_SL_ORDER_DISABLED               :  request.sl=0;                                                                 break;
         case MSG_SYM_TP_ORDER_DISABLED               :  request.tp=0;                                                                 break;

         //--- If unable to select the position lot, return "abort trading attempt" since the funds are insufficient even for the minimum lot
         case MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR       :  request.volume=this.CorrectVolume(request.price,order_type,symbol_obj,DFUN);
                                                         if(request.volume==0)
                                                           {
                                                            trade_obj.SetResultRetcode(MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT);
                                                            trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                            return ERROR_CODE_PROCESSING_METHOD_EXIT;                                                                                      break;
                                                           }
         //--- No quotes to handle the request
         case 10021                                   :  trade_obj.SetResultRetcode(10021);
                                                         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                         return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds
         //--- No connection to the trade server
         case 10031                                   :  trade_obj.SetResultRetcode(10031);
                                                         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                         return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 20 seconds

         //--- Proximity to the order activation level is handled by five-second waiting - during this time, the price may go beyond the freeze level
         case MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL       :
         case MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL       :
         case MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL       :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds

         //--- Trading operations are disabled in the terminal
         case MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED :
         //--- Trading operations are disabled for the EA
         case MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED       :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)20000;   // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 20 seconds

         default:
           break;
        }
     }
//--- No errors - return ОК
   trade_obj.SetResultRetcode(0);
   trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

In the first two articles describing the creation of [pending requests to open \\
positions](https://www.mql5.com/en/articles/7394) and [place pending orders](https://www.mql5.com/en/articles/7418), we handled only the errors received
from the trade server — upon getting an error after the first request, we created a pending request object, and the remaining trading
attempts were made from the pending request object. The trading methods also feature the preliminary check for the permission to trade and
the validity of the parameters. They are checked before sending a trading order to the server. Errors detected during these checks were not
handled by pending requests.

In this article, we add the creation of pending requests to the trading methods after preliminarily defining the errors before sending a
trading order. Also, add the creation of pending requests to all trading methods that do not have it yet — the methods of closing
positions/removing pending orders, as well as to the methods of modifying positions and pending orders.

**Let's consider the trading method for opening a position:**

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
                            const ulong deviation=ULONG_MAX,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
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

//--- Write the volume, deviation, comment and filling type to the request structure
   this.m_request.volume=volume;
   this.m_request.deviation=(deviation==ULONG_MAX ? trade_obj.GetDeviation() : deviation);
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : trade_obj.GetTypeFilling());
//--- Get the method of handling errors from the CheckErrors() method while checking for errors in the request parameters
   double pr=(type==POSITION_TYPE_BUY ? symbol_obj.AskLast() : symbol_obj.BidLast());
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
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the trading request magic number, has no pending request ID
         if(this.GetPendReqID((uint)magic)==0)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the pending request object ID to the magic number and fill in the remaining unfilled fields of the trading request structure
            uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
            this.SetPendReqID((uchar)id,mn);
            this.m_request.magic=mn;
            this.m_request.action=TRADE_ACTION_DEAL;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.type=order_type;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.OpenPosition(type,this.m_request.volume,this.m_request.sl,this.m_request.tp,magic,comment,deviation,type_filling);
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
         //--- If "Wait and repeat" is received as a result of sending a request - create a pending request and end the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the trading request magic number, has no pending request ID
            if(this.GetPendReqID((uint)magic)==0)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write the pending request object ID to the magic number and fill in the remaining unfilled fields of the trading request structure
               uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
               this.SetPendReqID((uchar)id,mn);
               this.m_request.magic=mn;
               this.m_request.action=TRADE_ACTION_DEAL;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.type=order_type;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

As we can see in the method listing, after the preliminary checks for permissions and errors in the trading order, as well as upon
the return of the "wait and repeat" code from the error verification method, we create a pending request here with the parameters currently
present in the trading request structure and exit the trading method. The pending request object takes over after that. It resends
trading requests specified in its parameters for as many times as set in the number of attempts.

If the initial check is passed, the trading order is sent to the server. The trade server return code is checked next. If
the result is "wait and repeat", a pending order is generated and an exit from the trading method is performed. Further trading
attempts are performed from the newly created pending request object.

The entire operation logic of generating pending request objects is specified in all trading methods. We will only have a look at the listings
of these methods where the block for creating a pending request after the initial
check and the block for creating a pending request after receiving a
trade server error are highlighted in colors. The following elements are provided for self-study:

**The method of modifying stop orders of an open position:**

```
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CTrading::ModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_MODIFY;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
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
   if(!this.SetPrices(order_type,0,(sl==WRONG_VALUE ? order.StopLoss() : sl),(tp==WRONG_VALUE ? order.TakeProfit() : tp),0,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
      return false;
     }

//--- If there are trading limitations, there are StopLevel/FreezeLevel limitations - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,0,action,order_type,symbol_obj,trade_obj,DFUN,0,this.m_request.sl,this.m_request.tp,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
      if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
        {
         trade_obj.SetResultRetcode(MSG_LIB_TEXT_TRADING_DISABLE);
         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true));
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
         if(this.IsUseSounds() && trade_obj.GetResultRetcode()!=10025)
            trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true));
         return false;
        }
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true));
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write a type of a conducted operation, as well as a symbol and a ticket of a modified position to the request structure
            this.m_request.action=TRADE_ACTION_SLTP;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.ModifyPosition(ticket,this.m_request.sl,this.m_request.tp);
      //--- If the request is executed successfully or the asynchronous order sending mode is set, play the success sound
      //--- set for a symbol trading object for this type of trading operation and return 'true'
      if(res || trade_obj.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj.PlaySoundSuccess(action,(int)order.TypeOrder(),(sl<0 ? false : true),(tp<0 ? false : true));
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRY_N),string(i+1),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.IsUseSounds() && trade_obj.GetResultRetcode()!=10025)
            trade_obj.PlaySoundError(action,(int)order.TypeOrder(),(sl<0 ? false : true),(tp<0 ? false : true));

         method=this.ResultProccessingMethod(trade_obj.GetResultRetcode());
         //--- If "Disable trading for the EA" is received as a result of sending a request, enable the disabling flag and end the attempt loop
         if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
           {
            this.SetTradingDisableFlag(true);
            break;
           }
         //--- If "Exit the trading method" is received as a result of sending a request -
         //--- set the last error code to the return structure and end the attempt loop
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
         //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
         //--- create a pending request and break the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the pending request object with the order ticket is not present in the list
            if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write a type of a conducted operation, as well as a symbol and a ticket of a modified position to the request structure
               this.m_request.action=TRADE_ACTION_SLTP;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.position=ticket;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

**Position closing method:**

```
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CTrading::ClosePosition(const ulong ticket,const double volume=WRONG_VALUE,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_CLOSE;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- get a trading object from a symbol object

   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
     }

//--- Write a deviation and a comment to the request structure
   this.m_request.deviation=(deviation==ULONG_MAX ? trade_obj.GetDeviation() : deviation);
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
//--- If there are trading limitations,
//--- there are limitations on FreezeLevel - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,0,action,order_type,symbol_obj,trade_obj,DFUN,0,0,0,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
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
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the trading operation type, symbol, ticket and volume to the request structure
            this.m_request.action=TRADE_ACTION_DEAL;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.volume=volume;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=(volume==WRONG_VALUE ? trade_obj.ClosePosition(ticket,comment,deviation) : trade_obj.ClosePositionPartially(ticket,volume,comment,deviation));
      //--- If the request is executed successfully or the asynchronous order sending mode is set, play the success sound
      //--- set for a symbol trading object for this type of trading operation and return 'true'
      if(res || trade_obj.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj.PlaySoundSuccess(action,(int)order.TypeOrder());
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRY_N),string(i+1),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,(int)order.TypeOrder());

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
         //--- If "Wait and repeat" is received as a result of sending a request -
         //--- create a pending request and break the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the pending request object with the position ticket is not present in the list
            if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write the trading operation type, symbol, ticket and volume to the request structure
               this.m_request.action=TRADE_ACTION_DEAL;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.position=ticket;
               this.m_request.volume=volume;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

**The method of closing a position by an opposite order:**

```
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CTrading::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_CLOSE_BY;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- trading object of a closed position
   CTradeObj *trade_obj_pos=this.GetTradeObjByPosition(ticket,DFUN);
   if(trade_obj_pos==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   if(!this.m_account.IsHedge())
     {
      trade_obj_pos.SetResultRetcode(MSG_ACC_UNABLE_CLOSE_BY);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      return false;
     }
//--- check the presence of an opposite position
   if(!this.CheckPositionAvailablity(ticket_by,DFUN))
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_SYS_ERROR_POSITION_BY_ALREADY_CLOSED);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      return false;
     }
//--- trading object of an opposite position
   CTradeObj *trade_obj_pos_by=this.GetTradeObjByPosition(ticket_by,DFUN);
   if(trade_obj_pos_by==NULL)
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- If a symbol of a closed position is not equal to an opposite position's one, inform of that and exit
   if(symbol_obj.Name()!=trade_obj_pos_by.GetSymbol())
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL));
      return false;
     }
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj_pos.SetResultRetcode(10021);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
     }

//--- If there are trading limitations, there are FreezeLevel limitations - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,0,action,order_type,symbol_obj,trade_obj_pos,DFUN,0,0,0,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
      if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
        {
         trade_obj_pos.SetResultRetcode(MSG_LIB_TEXT_TRADING_DISABLE);
         trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
         if(this.IsUseSounds())
            trade_obj_pos.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "abort trading operation" - set the last error code to the return structure,
      //--- display a journal message, play the error sound and exit
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
        {
         int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
         if(code!=NULL)
           {
            trade_obj_pos.SetResultRetcode(code);
            trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
           }
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_OPERATION_ABORTED));
         if(this.IsUseSounds())
            trade_obj_pos.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj_pos.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj_pos.SetResultRetcode(code);
               trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the trading operation type, symbol and tickets of two positions to the request structure
            this.m_request.action=TRADE_ACTION_CLOSE_BY;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.position_by=ticket_by;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj_pos.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj_pos.ClosePositionBy(ticket,ticket_by);
      //--- If the request is successful, play the success sound set for a symbol trading object for this type of trading operation
      if(res || trade_obj_pos.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj_pos.PlaySoundSuccess(action,(int)order.TypeOrder());
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj_pos.GetResultRetcode()));
         if(this.IsUseSounds())
            trade_obj_pos.PlaySoundError(action,(int)order.TypeOrder());

         method=this.ResultProccessingMethod(trade_obj_pos.GetResultRetcode());
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
            this.RequestErrorsCorrecting(this.m_request,order_type,trade_obj_pos.SpreadMultiplier(),symbol_obj,trade_obj_pos);
            continue;
           }
         //--- If "Update data and repeat" is received as a result of sending a request -
         //--- update data and start the next iteration
         if(method==ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            symbol_obj.Refresh();
            continue;
           }
         //--- If "Wait and repeat" is received as a result of sending a request -
         //--- create a pending request and break the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the pending request object with the position ticket is not present in the list
            if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write the trading operation type, symbol and tickets of two positions to the request structure
               this.m_request.action=TRADE_ACTION_CLOSE_BY;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.position=ticket;
               this.m_request.position_by=ticket_by;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
               this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj_pos.GetResultRetcode(),symbol_obj);
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

**The method for placing a pending order:**

```
//+------------------------------------------------------------------+
//| Place a pending order                                            |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
bool CTrading::PlaceOrder(const ENUM_ORDER_TYPE order_type,
                          const double volume,
                          const string symbol,
                          const PS price_stop,
                          const PL price_limit=0,
                          const SL sl=0,
                          const TP tp=0,
                          const ulong magic=ULONG_MAX,
                          const string comment=NULL,
                          const datetime expiration=0,
                          const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                          const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=(ENUM_ACTION_TYPE)order_type;
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
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
   if(!this.SetPrices(order_type,price_stop,sl,tp,price_limit,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
      return false;
     }

//--- Write the volume, comment, as well as expiration and filling types to the request structure
   this.m_request.volume=volume;
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : trade_obj.GetTypeExpiration());
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : trade_obj.GetTypeFilling());
//--- In case of trading limitations, funds insufficiency,
//--- there are limitations on StopLevel - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(this.m_request.volume,
                                                             this.m_request.price,
                                                             action,
                                                             order_type,
                                                             symbol_obj,
                                                             trade_obj,
                                                             DFUN,
                                                             this.m_request.stoplimit,
                                                             this.m_request.sl,
                                                             this.m_request.tp);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
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
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the trading request magic number, has no pending request ID
         if(this.GetPendReqID((uint)magic)==0)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the request ID to the magic number, while a symbol name is set in the request structure,
            //--- trading operation and order types
            uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
            this.SetPendReqID((uchar)id,mn);
            this.m_request.magic=mn;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.action=TRADE_ACTION_PENDING;
            this.m_request.type=order_type;
               //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.SetOrder(order_type,
                             this.m_request.volume,
                             this.m_request.price,
                             this.m_request.sl,
                             this.m_request.tp,
                             this.m_request.stoplimit,
                             magic,
                             comment,
                             this.m_request.expiration,
                             this.m_request.type_time,
                             this.m_request.type_filling);
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
         //--- If "Wait and repeat" is received as a result of sending a request - create a pending request and end the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the trading request magic number, has no pending request ID
            if(this.GetPendReqID((uint)magic)==0)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Write the request ID to the magic number, while a symbol name is set in the request structure,
               //--- trading operation and order types
               uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
               this.SetPendReqID((uchar)id,mn);
               this.m_request.magic=mn;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.action=TRADE_ACTION_PENDING;
               this.m_request.type=order_type;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

**The method for modifying a pending order:**

```
//+------------------------------------------------------------------+
//| Modify a pending order                                           |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
bool CTrading::ModifyOrder(const ulong ticket,
                           const PS price=WRONG_VALUE,
                           const SL sl=WRONG_VALUE,
                           const TP tp=WRONG_VALUE,
                           const PL limit=WRONG_VALUE,
                           datetime expiration=WRONG_VALUE,
                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_MODIFY;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by an order ticket
   CSymbol *symbol_obj=this.GetSymbolObjByOrder(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
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
   if(!this.SetPrices(order_type,
                      (price>0 ? price : order.PriceOpen()),
                      (sl>0 ? sl : sl<0 ? order.StopLoss() : 0),
                      (tp>0 ? tp : tp<0 ? order.TakeProfit() : 0),
                      (limit>0 ? limit : order.PriceStopLimit()),
                      DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
      return false;
     }

//--- Write the filling type, as well as expiration date and type to the request structure
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : order.TypeFilling());
   this.m_request.expiration=(expiration>WRONG_VALUE ? expiration : order.TimeExpiration());
   this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : order.TypeTime());
//--- If there are trading limitations,
//--- StopLevel or FreezeLevel limitations, play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,this.m_request.price,action,order_type,symbol_obj,trade_obj,DFUN,this.m_request.stoplimit,this.m_request.sl,this.m_request.tp,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
      if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
        {
         trade_obj.SetResultRetcode(MSG_LIB_TEXT_TRADING_DISABLE);
         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));
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
         if(this.IsUseSounds() && trade_obj.GetResultRetcode()!=10025)
            trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));
         return false;
        }
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the order ticket is not present in the list
         if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Set the trading operation type, as well as modified order's symbol and ticket in the request structure
            this.m_request.action=TRADE_ACTION_MODIFY;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.order=ticket;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.ModifyOrder(ticket,price,sl,tp,limit,expiration,type_time,type_filling);
      //--- If the request is executed successfully or the asynchronous order sending mode is set, play the success sound
      //--- set for a symbol trading object for this type of trading operation and return 'true'
      if(res || trade_obj.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj.PlaySoundSuccess(action,(int)order.TypeOrder(),(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRY_N),string(i+1),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,(int)order.TypeOrder(),(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));

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
         //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
         //--- create a pending request and break the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the pending request object with the order ticket is not present in the list
            if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Set the trading operation type, as well as modified order's symbol and ticket in the request structure
               this.m_request.action=TRADE_ACTION_MODIFY;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.order=ticket;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

**The method for removing a pending order:**

```
//+------------------------------------------------------------------+
//| Remove a pending order                                           |
//+------------------------------------------------------------------+
bool CTrading::DeleteOrder(const ulong ticket)
  {
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_CLOSE;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by an order ticket
   CSymbol *symbol_obj=this.GetSymbolObjByOrder(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
     }

//--- If there are trading limitations,
//--- there are limitations on FreezeLevel - play the error sound and exit
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(0,0,action,order_type,symbol_obj,trade_obj,DFUN,0,0,0,ticket);
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is completely disabled
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
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the order ticket is not present in the list
         if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Set the trading operation type, as well as deleted order's symbol and ticket in the request structure
            this.m_request.action=TRADE_ACTION_REMOVE;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.order=ticket;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest((uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
     }

//--- In the loop by the number of attempts
   for(int i=0;i<this.m_total_try;i++)
     {
      //--- Send the request
      res=trade_obj.DeleteOrder(ticket);
      //--- If the request is executed successfully or the asynchronous order sending mode is set, play the success sound
      //--- set for a symbol trading object for this type of trading operation and return 'true'
      if(res || trade_obj.IsAsyncMode())
        {
         if(this.IsUseSounds())
            trade_obj.PlaySoundSuccess(action,(int)order.TypeOrder());
         return true;
        }
      //--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRY_N),string(i+1),". ",CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,(int)order.TypeOrder());

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
         //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
         //--- create a pending request and break the loop
         if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
           {
            //--- If the pending request object with the order ticket is not present in the list
            if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
              {
               //--- Waiting time in milliseconds:
               //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
               //--- for the "Create a pending request" handling method - till there is a zero waiting time
               ulong wait=method;
               //--- Look for the least of the possible IDs. If failed to find
               //--- or in case of an error while updating the current symbol data, return 'false'
               int id=this.GetFreeID();
               if(id<1 || !symbol_obj.RefreshRates())
                  return false;
               //--- Set the trading operation type, as well as deleted order's symbol and ticket in the request structure
               this.m_request.action=TRADE_ACTION_REMOVE;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.order=ticket;
               //--- Set the number of trading attempts, create a pending request and break the loop
               uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
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

As we can see from the method listing, almost all blocks of creating pending requests are similar and involve filling in the trading request
structure, setting the number of trading attempts, creating a pending request and exiting the trading method. Pending request objects
from the class timer considered above take over afterwards.

**This are all the changes in the trading class necessary for implementing the work with pending requests.**

Since the **OnInit()** method now accepts the additional parameter in the trading class (the pointer to the trading event
collection class object), that parameter should be passed there.

This is done in the **CEngine** library base object class.


Open the **Engine.mqh** file and **supplement its**
**TradingOnInit() method:**

```
//--- Pass the pointers to all the necessary collections to the trading class
   void                 TradingOnInit(void)
                          {
                           this.m_trading.OnInit(this.GetAccountCurrent(),m_symbols.GetObject(),m_market.GetObject(),m_history.GetObject(),m_events.GetObject());
                          }
```

Here we call the **OnInit()** method of the **CTrading** trading class where the
pointer to the account trading event collection class is passed to the trading class as the last parameter.

### Testing

To test the pending request objects, we simply need to use [the EA from \\
the previous article](https://www.mql5.com/en/articles/7418#node03) without any changes.

But we are still going to save it in the new folder \\MQL5\\Experts\\TestDoEasy\ **Part28\**
under the name **TestDoEasyPart28.mq5**.

**Here is what we are going to do to check pending requests:**

Disable
the AutoTrading button in the terminal and click the position opening button in the test EA trading panel. Get the error of sending a
trading request and the appropriate journal entries, including the one about creating a
pending request.

Enable the AutoTrading button back
and wait till the time of the repeated trading attempt (the time of 20 seconds is set in case trading is disabled by the terminal). When the time
comes for the next attempt, a position is opened, the appropriate journal entry appears together with the one notifying that the
pending request has completed its work and has been removed:

```
2019.12.12 15:36:17.878 automated trading is disabled
2019.12.12 15:36:23.206 CTrading::OpenPosition<uint,uint>: Invalid request:
2019.12.12 15:36:23.206 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.12 15:36:23.206 Correction of trade request parameters ...
2019.12.12 15:36:23.207 Pending request created #1:
2019.12.12 15:36:23.207 ================== Pending trade request's parameters ==================
2019.12.12 15:36:23.207  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 15:36:23.207  - Trade request ID: #1
2019.12.12 15:36:23.207  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 15:36:23.207  - Request creation time: 2019.12.12 10:36:23.006
2019.12.12 15:36:23.207  - Price at time of request create: : 1.11353
2019.12.12 15:36:23.207  - Request activation time: 2019.12.12 10:36:43.006
2019.12.12 15:36:23.207  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 15:36:23.207  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 15:36:23.207  - Total trade attempts: 5
2019.12.12 15:36:23.207 ================== Trade request's parameters ==================
2019.12.12 15:36:23.207  - Trade operation type: Place market order
2019.12.12 15:36:23.207  - Trade symbol: EURUSD
2019.12.12 15:36:23.207  - Requested volume for a deal in lots: 0.10
2019.12.12 15:36:23.207  - Price: 1.11356
2019.12.12 15:36:23.207  - Stop Loss level of the order: 1.11206
2019.12.12 15:36:23.207  - Take Profit level of the order: 1.11506
2019.12.12 15:36:23.207  - Maximal deviation from the price: 5
2019.12.12 15:36:23.207  - Order type: Market-order Buy
2019.12.12 15:36:23.207  - Order execution type: The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)
2019.12.12 15:36:23.207  - Magic number: 25821307
2019.12.12 15:36:23.207  - Order comment: "TestDoEasyPart28 by DoEasy"
2019.12.12 15:36:23.207 ==================
2019.12.12 15:36:23.207
2019.12.12 15:36:26.845 automated trading is enabled
2019.12.12 15:36:43.536 Retry trading attempt #1
2019.12.12 15:36:43.883 - Position is open: 2019.12.12 10:36:43.797 -
2019.12.12 15:36:43.883 EURUSD Opened 0.10 Buy #494348590 [0.10 Market-order Buy #494348590] at price 1.11354, sl 1.11206, tp 1.11506, Magic number 25821307 (123), G1: 10, G2: 8, ID: 1
2019.12.12 15:36:43.883 OnDoEasyEvent: Position is open
2019.12.12 15:36:44.101 Trade request ID: #1: Pending request deleted due to execution
```

After the position is opened, disable the terminal's AutoTrading button
again and click the position closing button in the test EA trading panel. Get the error and the appropriate journal entries, including the one about creating
a pending request.

Then enable the AutoTrading button back
and wait till the time of the repeated trading attempt. When the time comes for the next attempt, the previously opened position is closed, the
appropriate journal entry appears together with the one notifying that the
pending request has completed its work and has been removed:

```
2019.12.12 15:37:02.347 automated trading is disabled
2019.12.12 15:37:06.566 CTrading::ClosePosition: Invalid request:
2019.12.12 15:37:06.566 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.12 15:37:06.566 Correction of trade request parameters ...
2019.12.12 15:37:06.566 Pending request created #1:
2019.12.12 15:37:06.566 ================== Pending trade request's parameters ==================
2019.12.12 15:37:06.566  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 15:37:06.566  - Trade request ID: #1
2019.12.12 15:37:06.566  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 15:37:06.566  - Request creation time: 2019.12.12 10:37:03.586
2019.12.12 15:37:06.566  - Price at time of request create: : 1.11351
2019.12.12 15:37:06.566  - Request activation time: 2019.12.12 10:37:23.586
2019.12.12 15:37:06.566  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 15:37:06.566  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 15:37:06.566  - Total trade attempts: 5
2019.12.12 15:37:06.566 ================== Trade request's parameters ==================
2019.12.12 15:37:06.566  - Trade operation type: Place market order
2019.12.12 15:37:06.566  - Trade symbol: EURUSD
2019.12.12 15:37:06.566  - Requested volume for a deal in lots: Value not set
2019.12.12 15:37:06.566  - Price: 1.11354
2019.12.12 15:37:06.566  - Stop Loss level of the order: 1.11206
2019.12.12 15:37:06.566  - Take Profit level of the order: 1.11506
2019.12.12 15:37:06.566  - Maximal deviation from the price: 5
2019.12.12 15:37:06.566  - Order type: Market-order Buy
2019.12.12 15:37:06.566  - Order execution type: The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)
2019.12.12 15:37:06.566  - Magic number: 0
2019.12.12 15:37:06.566  - Order comment: "TestDoEasyPart28 by DoEasy"
2019.12.12 15:37:06.566 ==================
2019.12.12 15:37:06.566
2019.12.12 15:37:11.611 automated trading is enabled
2019.12.12 15:37:25.196 Retry trading attempt #1
2019.12.12 15:37:25.679 - Position closed: 2019.12.12 10:36:43.797 -
2019.12.12 15:37:25.679 EURUSD Closed Buy #494348590 [0.10 Market-order Sell #494348941] at price 1.11349, sl 1.11206, tp 1.11506, Magic number 25821307 (123), G1: 10, G2: 8, ID: 1, Profit -0.50 USD
2019.12.12 15:37:25.679 OnDoEasyEvent: Position closed
2019.12.12 15:37:25.838 Trade request ID: #1: Pending request deleted due to execution
```

**Now let's check the work of several trading orders at once.**

Disable
auto trading and click the buttons for placing pending orders one by one. After each order placement attempt, get the error message, pending
request creation entry and its parameters.

Next, enable auto
trading in the terminal. As soon as the time of the next attempt of each of pending trading requests comes, the message about an attempt
number from each request is displayed together with the one about successful placement of each order.


Next, each pending trading request is removed, and the appropriate messages are displayed in the journal:

```
2019.12.12 17:47:16.774 automated trading is disabled
2019.12.12 17:47:20.297 CTrading::PlaceOrder<uint,int,uint,uint>: Invalid request:
2019.12.12 17:47:20.297 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.12 17:47:20.298 Correction of trade request parameters ...
2019.12.12 17:47:20.298 Pending request created #1:
2019.12.12 17:47:20.298 ================== Pending trade request's parameters ==================
2019.12.12 17:47:20.298  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 17:47:20.298  - Trade request ID: #1
2019.12.12 17:47:20.298  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 17:47:20.298  - Request creation time: 2019.12.12 12:47:18.719
2019.12.12 17:47:20.298  - Price at time of request create: : 1.11293
2019.12.12 17:47:20.298  - Request activation time: 2019.12.12 12:47:38.719
2019.12.12 17:47:20.298  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 17:47:20.298  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 17:47:20.298  - Total trade attempts: 5
2019.12.12 17:47:20.298 ================== Trade request's parameters ==================
2019.12.12 17:47:20.298  - Trade operation type: Place pending order
2019.12.12 17:47:20.298  - Trade symbol: EURUSD
2019.12.12 17:47:20.298  - Requested volume for a deal in lots: 0.10
2019.12.12 17:47:20.298  - Price: 1.11343
2019.12.12 17:47:20.298  - StopLimit level of the order: Value not set
2019.12.12 17:47:20.298  - Stop Loss level of the order: 1.11493
2019.12.12 17:47:20.298  - Take Profit level of the order: 1.11193
2019.12.12 17:47:20.298  - Order type: Pending order Sell Limit
2019.12.12 17:47:20.298  - Order execution type: The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)
2019.12.12 17:47:20.298  - Order expiration type: Good till cancel order
2019.12.12 17:47:20.298  - Order expiration time: Value not set
2019.12.12 17:47:20.298  - Magic number: 27525243
2019.12.12 17:47:20.298  - Order comment: "Pending order SellLimit"
2019.12.12 17:47:20.298  ==================
2019.12.12 17:47:20.298
2019.12.12 17:47:24.514 Pending request created #2:
2019.12.12 17:47:24.514 ================== Pending trade request's parameters ==================
2019.12.12 17:47:24.514  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 17:47:24.514  - Trade request ID: #2
2019.12.12 17:47:24.514  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 17:47:24.514  - Request creation time: 2019.12.12 12:47:23.638
2019.12.12 17:47:24.514  - Price at time of request create: : 1.11296
2019.12.12 17:47:24.514  - Request activation time: 2019.12.12 12:47:43.638
2019.12.12 17:47:24.514  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 17:47:24.514  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 17:47:24.514  - Total trade attempts: 5
2019.12.12 17:47:24.514 ================== Trade request's parameters ==================
2019.12.12 17:47:24.514  - Trade operation type: Place pending order
2019.12.12 17:47:24.514  - Trade symbol: EURUSD
2019.12.12 17:47:24.514  - Requested volume for a deal in lots: 0.10
2019.12.12 17:47:24.514  - Price: 1.11251
2019.12.12 17:47:24.514  - StopLimit level of the order: Value not set
2019.12.12 17:47:24.514  - Stop Loss level of the order: 1.11101
2019.12.12 17:47:24.514  - Take Profit level of the order: 1.11401
2019.12.12 17:47:24.514  - Order type: Pending order Buy Limit
2019.12.12 17:47:24.514  - Order execution type: The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)
2019.12.12 17:47:24.514  - Order expiration type: Good till cancel order
2019.12.12 17:47:24.514  - Order expiration time: Value not set
2019.12.12 17:47:24.514  - Magic number: 39387259
2019.12.12 17:47:24.514  - Order comment: "Pending order BuyLimit"
2019.12.12 17:47:24.514  ==================
2019.12.12 17:47:24.514
2019.12.12 17:47:29.167 automated trading is enabled
2019.12.12 17:47:45.796 Retry trading attempt #1
2019.12.12 17:47:46.045 Retry trading attempt #1
2019.12.12 17:47:46.256 - Pending order placed: 2019.12.12 12:47:46.319 -
2019.12.12 17:47:46.256 EURUSD Placed 0.10 Pending order Sell Limit #494419747 at price 1.11343, sl 1.11493, tp 1.11193, Magic number 27525243 (123), G1: 4, G2: 10, ID: 1
2019.12.12 17:47:46.256 - Pending order placed: 2019.12.12 12:47:46.162 -
2019.12.12 17:47:46.256 EURUSD Placed 0.10 Pending order Buy Limit #494419742 at price 1.11251, sl 1.11101, tp 1.11401, Magic number 39387259 (123), G1: 9, G2: 5, ID: 2
2019.12.12 17:47:46.256 OnDoEasyEvent: Pending order placed
2019.12.12 17:47:46.256 OnDoEasyEvent: Pending order placed
2019.12.12 17:47:46.505 Trade request ID: #2: Pending request deleted due to execution
2019.12.12 17:47:46.505 Trade request ID: #1: Pending request deleted due to execution
```

**Now let's try to remove both pending orders at once.**

Disable
auto trading in the terminal and click the button removing all pending orders on the test EA trading panel (Delete pending). Get the
error message followed by the entries about the pending request creation,
as well as its parameters for each of the removed orders.

Next, enable auto
trading in the terminal. As soon as the time of the next attempt of each of pending trading requests comes, the message about an attempt
number from each request is displayed together with the one about successful removal of each order.

Next,
each pending trading request is removed, and the appropriate messages are displayed in the journal:

```
2019.12.12 17:47:57.310 automated trading is disabled
2019.12.12 17:48:03.105 CTrading::DeleteOrder: Invalid request:
2019.12.12 17:48:03.105 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.12 17:48:03.105 Correction of trade request parameters ...
2019.12.12 17:48:03.105 Pending request created #1:
2019.12.12 17:48:03.106 ================== Pending trade request's parameters ==================
2019.12.12 17:48:03.106  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 17:48:03.106  - Trade request ID: #1
2019.12.12 17:48:03.106  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 17:48:03.106  - Request creation time: 2019.12.12 12:48:00.305
2019.12.12 17:48:03.106  - Price at time of request create: : 1.11300
2019.12.12 17:48:03.106  - Request activation time: 2019.12.12 12:48:20.305
2019.12.12 17:48:03.106  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 17:48:03.106  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 17:48:03.106  - Total trade attempts: 5
2019.12.12 17:48:03.106 ================== Trade request's parameters ==================
2019.12.12 17:48:03.106  - Trade operation type: Delete the pending order placed previously
2019.12.12 17:48:03.106  - Order ticket: 494419747
2019.12.12 17:48:03.106  ==================
2019.12.12 17:48:03.106
2019.12.12 17:48:03.106 CTrading::DeleteOrder: Invalid request:
2019.12.12 17:48:03.106 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.12 17:48:03.106 Correction of trade request parameters ...
2019.12.12 17:48:03.106 Pending request created #2:
2019.12.12 17:48:03.106 ================== Pending trade request's parameters ==================
2019.12.12 17:48:03.106  - Pending request type: Pending request that was created as a result of the server code
2019.12.12 17:48:03.106  - Trade request ID: #2
2019.12.12 17:48:03.106  - Return code based on which the request was created: 10027 "Autotrading disabled by client terminal"
2019.12.12 17:48:03.106  - Request creation time: 2019.12.12 12:48:00.305
2019.12.12 17:48:03.106  - Price at time of request create: : 1.11300
2019.12.12 17:48:03.106  - Request activation time: 2019.12.12 12:48:20.305
2019.12.12 17:48:03.106  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.12 17:48:03.106  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.12 17:48:03.106  - Total trade attempts: 5
2019.12.12 17:48:03.106 ================== Trade request's parameters ==================
2019.12.12 17:48:03.106  - Trade operation type: Delete the pending order placed previously
2019.12.12 17:48:03.106  - Order ticket: 494419742
2019.12.12 17:48:03.106  ==================
2019.12.12 17:48:03.106
2019.12.12 17:48:09.073 automated trading is enabled
2019.12.12 17:48:24.428 Retry trading attempt #1
2019.12.12 17:48:24.428 CTrading::OnTimer: Trade operation type: Delete the pending order placed previously
2019.12.12 17:48:24.593 Retry trading attempt #1
2019.12.12 17:48:24.593 CTrading::OnTimer: Trade operation type: Delete the pending order placed previously
2019.12.12 17:48:24.771 - Pending order removed: 2019.12.12 12:47:46.319 -
2019.12.12 17:48:24.771 EURUSD Deleted 0.10 Pending order Sell Limit #494419747 at price 1.11343, sl 1.11493, tp 1.11193, Magic number 27525243 (123), G1: 4, G2: 10, ID: 1
2019.12.12 17:48:24.771 - Pending order removed: 2019.12.12 12:47:46.162 -
2019.12.12 17:48:24.771 EURUSD Deleted 0.10 Pending order Buy Limit #494419742 at price 1.11251, sl 1.11101, tp 1.11401, Magic number 39387259 (123), G1: 9, G2: 5, ID: 2
2019.12.12 17:48:24.771 OnDoEasyEvent: Pending order removed
2019.12.12 17:48:24.771 OnDoEasyEvent: Pending order removed
2019.12.12 17:48:25.024 Trade request ID: #2: Pending request deleted due to execution
2019.12.12 17:48:25.024 Trade request ID: #1: Pending request deleted due to execution
```

According to the debugging messages in the journal, pending trading requests are activated as planned and removed after the work is complete.

This is not the final version of pending request objects.

Instead, this is only a check of the offered concept for MQL5 on a
hedging account.

**Please note:**

Do not use
the results of the trading class with the pending requests, described in the article, and the attached test EA in real trading!

All
tests have been conducted on a hedging account only.

The article, its accompanying materials and results are meant only as a test of
the pending requests concept. In its current state, it is not a finished product and is in no way intended for real trading. Instead,
it is only meant for the demo mode or the tester.

### What's next?

In the next article, we will summarize the test concept of working with pending trading requests. We will arrange a pending request object in
separate classes making it possible to conveniently expand the functionality for working with pending trading requests.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7438#node00)

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

[Part \\
26\. Working with pending trading requests - first implementation (opening positions)](https://www.mql5.com/en/articles/7394)

[Part \\
27\. Working with pending trading requests - placing pending orders](https://www.mql5.com/en/articles/7418)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7438](https://www.mql5.com/ru/articles/7438)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7438.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7438/mql5.zip "Download MQL5.zip")(3620.21 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7438/mql4.zip "Download MQL4.zip")(3620.22 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/333795)**
(7)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
15 Aug 2020 at 17:03

**Artyom Trishkin:**

[The Search() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch) works with a pointer to an object:

Ah right... but couldn't you invoke the method and pass the pointer of the local variable instead?

```
int index=this.m_list_request.Search(GetPointer(req));
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
15 Aug 2020 at 20:03

**ddiall :**

Ah right... but couldn't you invoke the method and pass the pointer of the local variable instead?

You can do it in different ways ... I've done this ... :)

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
15 Aug 2020 at 20:33

**Artyom Trishkin:**

You can do it in different ways ... I've done this ... :)

Cool, thanks - it was just to confirm my understanding as I am still reading the articles and studying the code ;-)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
15 Aug 2020 at 21:10

**ddiall:**

Cool, thanks - it was just to confirm my understanding as I am still reading the articles and studying the code ;-)

OK. Are you welcome :)


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
9 Jan 2021 at 19:20

```
   CEventsCollection*GetObject(void)                                                                     { return &this;
```

А чё, так можно было что ли ! YouTube - YouTube

[Photo image of Cornei Gor](https://www.youtube.com/channel/UC96JAp9a7RXaCH8N6MNLC_w?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7438)

Cornei Gor

1.82K subscribers

[А чё, так можно было что ли ! YouTube](https://www.youtube.com/watch?v=Iy7-GGuxaiI)

Cornei Gor

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=Iy7-GGuxaiI&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7438)

0:00

0:00 / 0:05

•Live

•

![Neural Networks Made Easy](https://c.mql5.com/2/48/Neural_networks_made_easy_001.png)[Neural Networks Made Easy](https://www.mql5.com/en/articles/7447)

Artificial intelligence is often associated with something fantastically complex and incomprehensible. At the same time, artificial intelligence is increasingly mentioned in everyday life. News about achievements related to the use of neural networks often appear in different media. The purpose of this article is to show that anyone can easily create a neural network and use the AI achievements in trading.

![SQLite: Native handling of SQL databases in MQL5](https://c.mql5.com/2/37/database-mql5.png)[SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463)

The development of trading strategies is associated with handling large amounts of data. Now, you are able to work with databases using SQL queries based on SQLite directly in MQL5. An important feature of this engine is that the entire database is placed in a single file located on a user's PC.

![Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://c.mql5.com/2/37/MQL5-avatar-doeasy__17.png)[Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

In the previous articles, we checked the concept of pending trading requests. A pending request is, in fact, a common trading order executed by a certain condition. In this article, we are going to create full-fledged classes of pending request objects — a base request object and its descendants.

![Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

The first article within the Walk-Through Optimization series described the creation of a DLL to be used in our auto optimizer. This continuation is entirely devoted to the MQL5 language.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/7438&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070438188696147414)

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