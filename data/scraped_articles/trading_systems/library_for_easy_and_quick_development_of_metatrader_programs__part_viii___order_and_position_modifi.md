---
title: Library for easy and quick development of MetaTrader programs (part VIII): Order and position modification events
url: https://www.mql5.com/en/articles/6595
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:41:34.505265
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=syormrhqulmqeuyddpcxiijgphymkvci&ssn=1769186491876431340&ssn_dr=0&ssn_sr=0&fv_date=1769186491&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6595&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20VIII)%3A%20Order%20and%20position%20modification%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918649185646722&fz_uniq=5070487134143452859&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Modification event class](https://www.mql5.com/en/articles/6595#node01)
- [Testing order and position modification events](https://www.mql5.com/en/articles/6595#node02)
- [What's next?](https://www.mql5.com/en/articles/6595#node03)


In the previous article, we implemented [tracking StopLimit order activation \\
events](https://www.mql5.com/en/articles/6482#node02). The entire functionality for defining the events was made extensible allowing us to easily add the search for other necessary
events.

We "qualified" a StopLimit order activation as placing a new pending order, which is reasonable since a new order type requires a new
placement event. In this article, we will track events of a different kind — modification of the already existing orders and positions (we
have already found out about their placement and opening thanks to obtaining the appropriate events in the program). This means we need yet
another (modification event) class derived from the CEvent abstract event class.

### Modification event class

As usual, we start with preparing the required enumeration constants.

Open the **Defines.mqh** library file and
set

the number of skipped properties to zeroin
the macro substitution containing the number of unused order integer properties when sorting by them. We will need all integer order
properties later. Previously, we skipped one property when searching and sorting — the last one from the list of enumeration constants:

ORDER\_PROP\_DIRECTION — order type by its direction (remember that we
place all unused properties at the end of the enumeration constant list). In the current and following articles, we will need this property
for searching all unidirectional pending orders in the market collection list.

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

Since we removed skipping the property, we also need to add the ability to sort by it. Let's add the criterion
of sorting by an order direction to the enumeration of orders' and deals' possible sorting criteria:

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
   SORT_BY_ORDER_TIME_OPEN       =  2,                      // Sort by order open time
   SORT_BY_ORDER_TIME_CLOSE      =  3,                      // Sort by order close time
   SORT_BY_ORDER_TIME_OPEN_MSC   =  4,                      // Sort by order open time in milliseconds
   SORT_BY_ORDER_TIME_CLOSE_MSC  =  5,                      // Sort by order close time in milliseconds
   SORT_BY_ORDER_TIME_EXP        =  6,                      // Sort by order expiration date
   SORT_BY_ORDER_STATUS          =  7,                      // Sort by order status (market order/pending order/deal/balance, credit operation)
   SORT_BY_ORDER_TYPE            =  8,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  9,                      // Sort by order/position reason/source
   SORT_BY_ORDER_STATE           =  10,                     // Sort by order status
   SORT_BY_ORDER_POSITION_ID     =  11,                     // Sort by position ID
   SORT_BY_ORDER_POSITION_BY_ID  =  12,                     // Sort by opposite position ID
   SORT_BY_ORDER_DEAL_ORDER      =  13,                     // Sort by order a deal is based on
   SORT_BY_ORDER_DEAL_ENTRY      =  14,                     // Sort by deal direction – IN, OUT or IN/OUT
   SORT_BY_ORDER_TIME_UPDATE     =  15,                     // Sort by position change time in seconds
   SORT_BY_ORDER_TIME_UPDATE_MSC =  16,                     // Sort by position change time in milliseconds
   SORT_BY_ORDER_TICKET_FROM     =  17,                     // Sort by parent order ticket
   SORT_BY_ORDER_TICKET_TO       =  18,                     // Sort by derived order ticket
   SORT_BY_ORDER_PROFIT_PT       =  19,                     // Sort by order profit in points
   SORT_BY_ORDER_CLOSE_BY_SL     =  20,                     // Sort by order closing by StopLoss flag
   SORT_BY_ORDER_CLOSE_BY_TP     =  21,                     // Sort by order closing by TakeProfit flag
   SORT_BY_ORDER_GROUP_ID        =  22,                     // Sort by order/position group ID
   SORT_BY_ORDER_DIRECTION       =  23,                     // Sort by direction (Buy, Sell)
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
   SORT_BY_ORDER_EXT_ID          =  FIRST_ORD_STR_PROP+2    // Sort by order ID in an external trading system
  };
//+------------------------------------------------------------------+
```

As you may remember, to create an event ID, we use the event code consisting of a set of flags that together indicate the type of an occurred
event. Since we are going to track modification events, we need to add necessary flags to the trading event flags enumeration:

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
   TRADE_EVENT_FLAG_POSITION_CHANGED=  16,                  // Position changed
   TRADE_EVENT_FLAG_POSITION_REVERSE=  32,                  // Position reversed
   TRADE_EVENT_FLAG_POSITION_CLOSED =  64,                  // Position closed
   TRADE_EVENT_FLAG_ACCOUNT_BALANCE =  128,                 // Balance operation (clarified by a deal type)
   TRADE_EVENT_FLAG_PARTIAL         =  256,                 // Partial execution
   TRADE_EVENT_FLAG_BY_POS          =  512,                 // Executed by opposite position
   TRADE_EVENT_FLAG_PRICE           =  1024,                // Placement price modification
   TRADE_EVENT_FLAG_SL              =  2048,                // Executed by StopLoss
   TRADE_EVENT_FLAG_TP              =  4096,                // Executed by TakeProfit
   TRADE_EVENT_FLAG_ORDER_MODIFY    =  8192,                // Order modification
   TRADE_EVENT_FLAG_POSITION_MODIFY =  16384,               // Position modification
  };
//+------------------------------------------------------------------+
```

Let's add order and position
modification events to the list of possible account trading events (previously, we already added some events to the list, but this
was a preliminary declaration of constants):

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
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL,       // Added volume to a position by partial activation of a market order (netting)
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
   TRADE_EVENT_MODIFY_ORDER_STOP_LOSS,                      // Changing order's StopLoss
   TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT,                    // Changing order's TakeProfit
   TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT,       // Changing position's StopLoss and TakeProfit
   TRADE_EVENT_MODIFY_POSITION_STOP_LOSS,                   // Changing position's StopLoss
   TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT,                 // Changing position's TakeProfit
  };
//+------------------------------------------------------------------+
```

Since we are going to develop the new class derived from the CEvent abstract event class, we need to set yet another event
status — "modification" for the newly derived class. Let's add it to the event states enumeration list:

```
//+------------------------------------------------------------------+
//| Event status                                                     |
//+------------------------------------------------------------------+
enum ENUM_EVENT_STATUS
  {
   EVENT_STATUS_MARKET_POSITION,                            // Market position event (opening, partial opening, partial closing, adding volume, reversal)
   EVENT_STATUS_MARKET_PENDING,                             // Market pending order event (placing)
   EVENT_STATUS_HISTORY_PENDING,                            // Historical pending order event (removal)
   EVENT_STATUS_HISTORY_POSITION,                           // Historical position event (closing
   EVENT_STATUS_BALANCE,                                    // Balance operation event (accruing balance, withdrawing funds and events from the ENUM_DEAL_TYPE enumeration)
   EVENT_STATUS_MODIFY                                      // Order/position modification event
  };
//+------------------------------------------------------------------+
```

Let's add the "modification" event reason to the event reason
enumeration list:

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
   EVENT_REASON_MODIFY,                                     // Modification
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
   EVENT_REASON_DONE_BY_POS_PARTIALLY,                      // Partial closing by an opposite position
   EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY,            // Closing an opposite position by a partial volume
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
//+------------------------------------------------------------------+
```

If we want to always be aware of what changed in order/position properties, we should add order/position
property prices before modification (post-modification prices are taken from the already existing properties) and properties
for writing the current prices during an event to the event's integer properties, as well as change the number
of event's real properties from

**10 to 15**and add the number of unused properties during
the search and sorting (modification data and prices during a modification event are not used for search and sorting):

```
//+------------------------------------------------------------------+
//| Event's real properties                                          |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_DOUBLE
  {
   EVENT_PROP_PRICE_EVENT = EVENT_PROP_INTEGER_TOTAL,       // Price an event occurred at
   EVENT_PROP_PRICE_OPEN,                                   // Order/deal/position open price
   EVENT_PROP_PRICE_CLOSE,                                  // Order/deal/position close price
   EVENT_PROP_PRICE_SL,                                     // StopLoss order/deal/position price
   EVENT_PROP_PRICE_TP,                                     // TakeProfit Order/deal/position
   EVENT_PROP_VOLUME_ORDER_INITIAL,                         // Requested order volume
   EVENT_PROP_VOLUME_ORDER_EXECUTED,                        // Executed order volume
   EVENT_PROP_VOLUME_ORDER_CURRENT,                         // Remaining order volume
   EVENT_PROP_VOLUME_POSITION_EXECUTED,                     // Current executed position volume after a deal
   EVENT_PROP_PROFIT,                                       // Profit
   //---
   EVENT_PROP_PRICE_OPEN_BEFORE,                            // Order price before modification
   EVENT_PROP_PRICE_SL_BEFORE,                              // StopLoss price before modification
   EVENT_PROP_PRICE_TP_BEFORE,                              // TakeProfit price before modification
   EVENT_PROP_PRICE_EVENT_ASK,                              // Ask price during an event
   EVENT_PROP_PRICE_EVENT_BID,                              // Bid price during an event
  };
#define EVENT_PROP_DOUBLE_TOTAL  (15)                       // Total number of event's real properties
#define EVENT_PROP_DOUBLE_SKIP   (5)                        // Number of order properties not used in sorting
//+------------------------------------------------------------------+
```

Let's change the calculation of the appropriate macro substitution
to correctly find the index of the event's first string property in the enumeration of events sorting criteria:

```
//+------------------------------------------------------------------+
//| Possible event sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_EVN_DBL_PROP       (EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_INTEGER_SKIP)
#define FIRST_EVN_STR_PROP       (EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_INTEGER_SKIP+EVENT_PROP_DOUBLE_TOTAL-EVENT_PROP_DOUBLE_SKIP)
enum ENUM_SORT_EVENTS_MODE
  {
   //--- Sort by integer properties
   SORT_BY_EVENT_TYPE_EVENT               = 0,                       // Sort by event type
   SORT_BY_EVENT_TIME_EVENT               = 1,                       // Sort by event time
   SORT_BY_EVENT_STATUS_EVENT             = 2,                       // Sort by event status (from the ENUM_EVENT_STATUS enumeration)
   SORT_BY_EVENT_REASON_EVENT             = 3,                       // Sort by event reason (from the ENUM_EVENT_REASON enumeration)
   SORT_BY_EVENT_TYPE_DEAL_EVENT          = 4,                       // Sort by deal event type
   SORT_BY_EVENT_TICKET_DEAL_EVENT        = 5,                       // Sort by deal event ticket
   SORT_BY_EVENT_TYPE_ORDER_EVENT         = 6,                       // Sort by type of an order, based on which a deal event is opened (the last position order)
   SORT_BY_EVENT_TICKET_ORDER_EVENT       = 7,                       // Sort by type of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TIME_ORDER_POSITION      = 8,                       // Sort by time of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TYPE_ORDER_POSITION      = 9,                       // Sort by type of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TICKET_ORDER_POSITION    = 10,                      // Sort by a ticket of an order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_POSITION_ID              = 11,                      // Sort by position ID
   SORT_BY_EVENT_POSITION_BY_ID           = 12,                      // Sort by opposite position ID
   SORT_BY_EVENT_MAGIC_ORDER              = 13,                      // Sort by order/deal/position magic number
   SORT_BY_EVENT_MAGIC_BY_ID              = 14,                      // Sort by opposite position magic number
   //--- Sort by real properties
   SORT_BY_EVENT_PRICE_EVENT              =  FIRST_EVN_DBL_PROP,     // Sort by a price an event occurred at
   SORT_BY_EVENT_PRICE_OPEN               =  FIRST_EVN_DBL_PROP+1,   // Sort by position open price
   SORT_BY_EVENT_PRICE_CLOSE              =  FIRST_EVN_DBL_PROP+2,   // Sort by position close price
   SORT_BY_EVENT_PRICE_SL                 =  FIRST_EVN_DBL_PROP+3,   // Sort by position's StopLoss price
   SORT_BY_EVENT_PRICE_TP                 =  FIRST_EVN_DBL_PROP+4,   // Sort by position's TakeProfit price
   SORT_BY_EVENT_VOLUME_ORDER_INITIAL     =  FIRST_EVN_DBL_PROP+5,   // Sort by initial volume
   SORT_BY_EVENT_VOLUME_ORDER_EXECUTED    =  FIRST_EVN_DBL_PROP+6,   // Sort by the current volume
   SORT_BY_EVENT_VOLUME_ORDER_CURRENT     =  FIRST_EVN_DBL_PROP+7,   // Sort by remaining volume
   SORT_BY_EVENT_VOLUME_POSITION_EXECUTED =  FIRST_EVN_DBL_PROP+8,   // Sort by remaining volume
   SORT_BY_EVENT_PROFIT                   =  FIRST_EVN_DBL_PROP+9,   // Sort by profit
   //--- Sort by string properties
   SORT_BY_EVENT_SYMBOL                   =  FIRST_EVN_STR_PROP,     // Sort by order/position/deal symbol
   SORT_BY_EVENT_SYMBOL_BY_ID                                        // Sort by an opposite position symbol
  };
//+------------------------------------------------------------------+
```

**Let's improve the CEvent abstract event class.**

Since we display data in the journal in the classes derived from the CEvent abstract event, we need to know the number of decimal places in the
symbol quote the event occurred at — symbol's Digits(). In order not to receive it each time in each of the derived classes, we simply get it
once in the parent class.

In the private class section, declare the class member variable for storing the
Digits() value of the event symbol and initialize that variable in the
initialization list of the class constructor:

```
//+------------------------------------------------------------------+
//| Abstract event class                                             |
//+------------------------------------------------------------------+
class CEvent : public CObject
  {
private:
   int               m_event_code;                                   // Event code
//--- Return the index of the array the event's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_EVENT_PROP_DOUBLE property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL;                         }
   int               IndexProp(ENUM_EVENT_PROP_STRING property)const { return(int)property-EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_DOUBLE_TOTAL; }
protected:
   ENUM_TRADE_EVENT  m_trade_event;                                  // Trading event
   bool              m_is_hedge;                                     // Hedging account flag
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits;                                       // Symbol Digits()
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- Return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }

//--- Protected parametric constructor
                     CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket);
public:
//--- Default constructor
                     CEvent(void){;}
```

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code),m_digits(0)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   this.m_digits_acc=(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
```

**Add descriptions of the new event properties to the methods returning descriptions of integer**
**and real properties:**

```
//+------------------------------------------------------------------+
//| Return the description of the event's integer property           |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property)
  {
   return
     (
      property==EVENT_PROP_TYPE_EVENT              ?  TextByLanguage("Тип события","Event's type")+": "+this.TypeEventDescription()                                                       :
      property==EVENT_PROP_TIME_EVENT              ?  TextByLanguage("Время события","Time of event")+": "+TimeMSCtoString(this.GetProperty(property))                                    :
      property==EVENT_PROP_STATUS_EVENT            ?  TextByLanguage("Статус события","Status of event")+": \""+this.StatusDescription()+"\""                                             :
      property==EVENT_PROP_REASON_EVENT            ?  TextByLanguage("Причина события","Reason of event")+": "+this.ReasonDescription()                                                   :
      property==EVENT_PROP_TYPE_DEAL_EVENT         ?  TextByLanguage("Тип сделки","Deal's type")+": "+DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(property))                     :
      property==EVENT_PROP_TICKET_DEAL_EVENT       ?  TextByLanguage("Тикет сделки","Deal's ticket")+": #"+(string)this.GetProperty(property)                                              :
      property==EVENT_PROP_TYPE_ORDER_EVENT        ?  TextByLanguage("Тип ордера события","Event's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property))    :
      property==EVENT_PROP_TYPE_ORDER_POSITION     ?  TextByLanguage("Тип ордера позиции","Position's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property)) :
      property==EVENT_PROP_TICKET_ORDER_POSITION   ?  TextByLanguage("Тикет первого ордера позиции","Position's first order ticket")+": #"+(string)this.GetProperty(property)              :
      property==EVENT_PROP_TICKET_ORDER_EVENT      ?  TextByLanguage("Тикет ордера события","Event's order ticket")+": #"+(string)this.GetProperty(property)                               :
      property==EVENT_PROP_POSITION_ID             ?  TextByLanguage("Идентификатор позиции","Position ID")+": #"+(string)this.GetProperty(property)                                       :
      property==EVENT_PROP_POSITION_BY_ID          ?  TextByLanguage("Идентификатор встречной позиции","Opposite position's ID")+": #"+(string)this.GetProperty(property)                  :
      property==EVENT_PROP_MAGIC_ORDER             ?  TextByLanguage("Магический номер","Magic number")+": "+(string)this.GetProperty(property)                                           :
      property==EVENT_PROP_MAGIC_BY_ID             ?  TextByLanguage("Магический номер встречной позиции","Magic number of opposite position")+": "+(string)this.GetProperty(property)    :
      property==EVENT_PROP_TIME_ORDER_POSITION     ?  TextByLanguage("Время открытия позиции","Position's opened time")+": "+TimeMSCtoString(this.GetProperty(property))                  :
      property==EVENT_PROP_TYPE_ORD_POS_BEFORE     ?  TextByLanguage("Тип ордера позиции до смены направления","Type order of position before changing direction")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property)) :
      property==EVENT_PROP_TICKET_ORD_POS_BEFORE   ?  TextByLanguage("Тикет ордера позиции до смены направления","Ticket order of position before changing direction")+": #"+(string)this.GetProperty(property)                            :
      property==EVENT_PROP_TYPE_ORD_POS_CURRENT    ?  TextByLanguage("Тип ордера текущей позиции","Type order of current position")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property))                                :
      property==EVENT_PROP_TICKET_ORD_POS_CURRENT  ?  TextByLanguage("Тикет ордера текущей позиции","Ticket order of current position")+": #"+(string)this.GetProperty(property)                                                           :
      EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's real property              |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property)
  {
   int dg=(int)::SymbolInfoInteger(this.GetProperty(EVENT_PROP_SYMBOL),SYMBOL_DIGITS);
   int dgl=(int)DigitsLots(this.GetProperty(EVENT_PROP_SYMBOL));
   return
     (
      property==EVENT_PROP_PRICE_EVENT             ?  TextByLanguage("Цена на момент события","Price at the time of event")+": "+::DoubleToString(this.GetProperty(property),dg)               :
      property==EVENT_PROP_PRICE_OPEN              ?  TextByLanguage("Цена открытия","Open price")+": "+::DoubleToString(this.GetProperty(property),dg)                                            :
      property==EVENT_PROP_PRICE_CLOSE             ?  TextByLanguage("Цена закрытия","Close price")+": "+::DoubleToString(this.GetProperty(property),dg)                                           :
      property==EVENT_PROP_PRICE_SL                ?  TextByLanguage("Цена StopLoss","StopLoss price")+": "+::DoubleToString(this.GetProperty(property),dg)                                        :
      property==EVENT_PROP_PRICE_TP                ?  TextByLanguage("Цена TakeProfit","TakeProfit price")+": "+::DoubleToString(this.GetProperty(property),dg)                                    :
      property==EVENT_PROP_VOLUME_ORDER_INITIAL    ?  TextByLanguage("Начальный объём ордера","Order initial volume")+": "+::DoubleToString(this.GetProperty(property),dgl)                        :
      property==EVENT_PROP_VOLUME_ORDER_EXECUTED   ?  TextByLanguage("Исполненный объём ордера","Order executed volume")+": "+::DoubleToString(this.GetProperty(property),dgl)                     :
      property==EVENT_PROP_VOLUME_ORDER_CURRENT    ?  TextByLanguage("Оставшийся объём ордера","Order remaining volume")+": "+::DoubleToString(this.GetProperty(property),dgl)                     :
      property==EVENT_PROP_VOLUME_POSITION_EXECUTED ? TextByLanguage("Текущий объём позиции","Position current volume")+": "+::DoubleToString(this.GetProperty(property),dgl)                      :
      property==EVENT_PROP_PROFIT                  ?  TextByLanguage("Профит","Profit")+": "+::DoubleToString(this.GetProperty(property),this.m_digits_acc)                                        :
      property==EVENT_PROP_PRICE_OPEN_BEFORE       ?  TextByLanguage("Цена открытия до модификации","Price open before modification")+": "+::DoubleToString(this.GetProperty(property),dg)         :
      property==EVENT_PROP_PRICE_SL_BEFORE         ?  TextByLanguage("Цена StopLoss до модификации","StopLoss price before modification")+": "+::DoubleToString(this.GetProperty(property),dg)     :
      property==EVENT_PROP_PRICE_TP_BEFORE         ?  TextByLanguage("Цена TakeProfit до модификации","TakeProfit price before modification")+": "+::DoubleToString(this.GetProperty(property),dg) :
      property==EVENT_PROP_PRICE_EVENT_ASK         ?  TextByLanguage("Цена Ask в момент события","Ask price at the time of event")+": "+::DoubleToString(this.GetProperty(property),dg)        :
      property==EVENT_PROP_PRICE_EVENT_BID         ?  TextByLanguage("Цена Bid в момент события","Bid price at the time of event")+": "+::DoubleToString(this.GetProperty(property),dg)        :
      EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
```

Let's add event absence description, descriptions
of newly added events and description of an unknown event
to the method returning trading event names:

```
//+------------------------------------------------------------------+
//| Return the trading event name                                    |
//+------------------------------------------------------------------+
string CEvent::TypeEventDescription(void) const
  {
   ENUM_TRADE_EVENT event=this.TypeEvent();
   return
     (
      event==TRADE_EVENT_NO_EVENT                              ?  TextByLanguage("Нет торгового события","No trading event")                                              :
      event==TRADE_EVENT_PENDING_ORDER_PLASED                  ?  TextByLanguage("Отложенный ордер установлен","Pending order placed")                                  :
      event==TRADE_EVENT_PENDING_ORDER_REMOVED                 ?  TextByLanguage("Отложенный ордер удалён","Pending order removed")                                     :
      event==TRADE_EVENT_ACCOUNT_CREDIT                        ?  TextByLanguage("Начисление кредита","Credit")                                                         :
      event==TRADE_EVENT_ACCOUNT_CHARGE                        ?  TextByLanguage("Дополнительные сборы","Additional charge")                                            :
      event==TRADE_EVENT_ACCOUNT_CORRECTION                    ?  TextByLanguage("Корректирующая запись","Correction")                                                  :
      event==TRADE_EVENT_ACCOUNT_BONUS                         ?  TextByLanguage("Перечисление бонусов","Bonus")                                                        :
      event==TRADE_EVENT_ACCOUNT_COMISSION                     ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                     :
      event==TRADE_EVENT_ACCOUNT_COMISSION_DAILY               ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY             ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                           :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY         ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")      :
      event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY       ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")           :
      event==TRADE_EVENT_ACCOUNT_INTEREST                      ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                          :
      event==TRADE_EVENT_BUY_CANCELLED                         ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                       :
      event==TRADE_EVENT_SELL_CANCELLED                        ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                      :
      event==TRADE_EVENT_DIVIDENT                              ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                          :
      event==TRADE_EVENT_DIVIDENT_FRANKED                      ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations")    :
      event==TRADE_EVENT_TAX                                   ?  TextByLanguage("Начисление налога","Tax charges")                                                     :
      event==TRADE_EVENT_ACCOUNT_BALANCE_REFILL                ?  TextByLanguage("Пополнение средств на балансе","Balance refill")                                      :
      event==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL            ?  TextByLanguage("Снятие средств с баланса","Withdrawals")                                              :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED               ?  TextByLanguage("Отложенный ордер активирован ценой","Pending order activated")                        :
      event==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL       ?  TextByLanguage("Отложенный ордер активирован ценой частично","Pending order activated partially")     :
      event==TRADE_EVENT_POSITION_OPENED                       ?  TextByLanguage("Позиция открыта","Position open")                                                  :
      event==TRADE_EVENT_POSITION_OPENED_PARTIAL               ?  TextByLanguage("Позиция открыта частично","Position open partially")                               :
      event==TRADE_EVENT_POSITION_CLOSED                       ?  TextByLanguage("Позиция закрыта","Position closed")                                                   :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL               ?  TextByLanguage("Позиция закрыта частично","Position closed partially")                                :
      event==TRADE_EVENT_POSITION_CLOSED_BY_POS                ?  TextByLanguage("Позиция закрыта встречной","Position closed by opposite position")                    :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS        ?  TextByLanguage("Позиция закрыта встречной частично","Position closed partially by opposite position") :
      event==TRADE_EVENT_POSITION_CLOSED_BY_SL                 ?  TextByLanguage("Позиция закрыта по StopLoss","Position closed by StopLoss")                           :
      event==TRADE_EVENT_POSITION_CLOSED_BY_TP                 ?  TextByLanguage("Позиция закрыта по TakeProfit","Position closed by TakeProfit")                       :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL         ?  TextByLanguage("Позиция закрыта частично по StopLoss","Position closed partially by StopLoss")        :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP         ?  TextByLanguage("Позиция закрыта частично по TakeProfit","Position closed partially by TakeProfit")    :
      event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET           ?  TextByLanguage("Разворот позиции по рыночному запросу","Position reversal by market request")         :
      event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING          ?  TextByLanguage("Разворот позиции срабатыванием отложенного ордера","Position reversal by triggering pending order")                            :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET         ?  TextByLanguage("Добавлен объём к позиции по рыночному запросу","Added volume to position by market request")                                    :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING        ?  TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by activation of pending order")        :

      event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL   ?  TextByLanguage("Разворот позиции частичным исполнением запроса","Position reversal by partial completion of market request")                   :
      event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL  ?  TextByLanguage("Разворот позиции частичным срабатыванием отложенного ордера","Position reversal by partial activation of pending order")        :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL ?  TextByLanguage("Добавлен объём к позиции частичным исполнением запроса","Added volume to position by partial completion of market request")    :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL ? TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by partial activation of pending order")  :

      event==TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER            ?  TextByLanguage("Сработал StopLimit-ордер","StopLimit order triggered.")                                                                         :
      event==TRADE_EVENT_MODIFY_ORDER_PRICE                    ?  TextByLanguage("Модифицирована цена установки ордера ","Order price modified")                                                                  :
      event==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS          ?  TextByLanguage("Модифицированы цена установки и StopLoss ордера","Order price and StopLoss modified")                                        :
      event==TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT        ?  TextByLanguage("Модифицированы цена установки и TakeProfit ордера","Order price and TakeProfit modified")                                    :
      event==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT ?  TextByLanguage("Модифицированы цена установки, StopLoss и TakeProfit ордера","Order price, StopLoss and TakeProfit modified")             :
      event==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT    ?  TextByLanguage("Модифицированы цены StopLoss и TakeProfit ордера","Order StopLoss and TakeProfit modified")                                  :
      event==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS                ?  TextByLanguage("Модифицирован StopLoss ордера","Order StopLoss modified")                                                                       :
      event==TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT              ?  TextByLanguage("Модифицирован TakeProfit ордера","Order TakeProfit modified")                                                                   :
      event==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT ?  TextByLanguage("Модифицированы цены StopLoss и TakeProfit позиции","Position StopLoss and TakeProfit modified")                              :
      event==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS             ?  TextByLanguage("Модифицирован StopLoss позиции","Position StopLoss modified")                                                                   :
      event==TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT           ?  TextByLanguage("Модифицирован TakeProfit позиции","Position TakeProfit modified")                                                               :
      EnumToString(event)
     );
  }
//+------------------------------------------------------------------+
```

Add two new reasons to the method returning the event
reason description:

```
//+------------------------------------------------------------------+
//| Return the name of the deal/order/position reason                |
//+------------------------------------------------------------------+
string CEvent::ReasonDescription(void) const
  {
   ENUM_EVENT_REASON reason=this.Reason();
   return
     (
      reason==EVENT_REASON_ACTIVATED_PENDING                ?  TextByLanguage("Активирован отложенный ордер","Pending order activated")                           :
      reason==EVENT_REASON_ACTIVATED_PENDING_PARTIALLY      ?  TextByLanguage("Частичное срабатывание отложенного ордера","Pending order partially triggered")    :
      reason==EVENT_REASON_STOPLIMIT_TRIGGERED              ?  TextByLanguage("Срабатывание StopLimit-ордера","StopLimit order triggered")                        :
      reason==EVENT_REASON_MODIFY                           ?  TextByLanguage("Модификация","Modified")                                                           :
      reason==EVENT_REASON_CANCEL                           ?  TextByLanguage("Отмена","Canceled")                                                                :
      reason==EVENT_REASON_EXPIRED                          ?  TextByLanguage("Истёк срок действия","Expired")                                                    :
      reason==EVENT_REASON_DONE                             ?  TextByLanguage("Рыночный запрос, выполненный в полном объёме","Fully completed market request")    :
      reason==EVENT_REASON_DONE_PARTIALLY                   ?  TextByLanguage("Выполненный частично рыночный запрос","Partially completed market request")        :
      reason==EVENT_REASON_VOLUME_ADD                       ?  TextByLanguage("Добавлен объём к позиции","Added volume to position")                              :
      reason==EVENT_REASON_VOLUME_ADD_PARTIALLY             ?  TextByLanguage("Добавлен объём к позиции частичным исполнением заявки","Volume added to the position by request partial completion")                 :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING            ?  TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by activating pending order")                      :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING_PARTIALLY  ?  TextByLanguage("Добавлен объём к позиции частичной активацией отложенного ордера","Added volume to position by partial activation of pending order")  :
      reason==EVENT_REASON_REVERSE                          ?  TextByLanguage("Разворот позиции","Position reversal")  :
      reason==EVENT_REASON_REVERSE_PARTIALLY                ?  TextByLanguage("Разворот позиции частичным исполнением заявки","Position reversal by partial completion of request")                             :
      reason==EVENT_REASON_REVERSE_BY_PENDING               ?  TextByLanguage("Разворот позиции при срабатывании отложенного ордера","Position reversal on triggered pending order")                               :
      reason==EVENT_REASON_REVERSE_BY_PENDING_PARTIALLY     ?  TextByLanguage("Разворот позиции при при частичном срабатывании отложенного ордера","Position reversal on partially triggered pending order")       :
      reason==EVENT_REASON_DONE_SL                          ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss triggered")                               :
      reason==EVENT_REASON_DONE_SL_PARTIALLY                ?  TextByLanguage("Частичное закрытие по StopLoss","Partial close by StopLoss triggered")           :
      reason==EVENT_REASON_DONE_TP                          ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit triggered")                           :
      reason==EVENT_REASON_DONE_TP_PARTIALLY                ?  TextByLanguage("Частичное закрытие по TakeProfit","Partial close by TakeProfit triggered")       :
      reason==EVENT_REASON_DONE_BY_POS                      ?  TextByLanguage("Закрытие встречной позицией","Closed by opposite position")                        :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS            ?  TextByLanguage("Частичное закрытие встречной позицией","Closed partially by opposite position")    :
      reason==EVENT_REASON_DONE_BY_POS_PARTIALLY            ?  TextByLanguage("Закрытие частью объёма встречной позиции","Closed by incomplete volume of opposite position") :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY  ?  TextByLanguage("Частичное закрытие частью объёма встречной позиции","Closed partially by incomplete volume of opposite position")  :
      reason==EVENT_REASON_BALANCE_REFILL                   ?  TextByLanguage("Пополнение баланса","Balance refill")                                              :
      reason==EVENT_REASON_BALANCE_WITHDRAWAL               ?  TextByLanguage("Снятие средств с баланса","Withdrawal from the balance")                          :
      reason==EVENT_REASON_ACCOUNT_CREDIT                   ?  TextByLanguage("Начисление кредита","Credit")                                                      :
      reason==EVENT_REASON_ACCOUNT_CHARGE                   ?  TextByLanguage("Дополнительные сборы","Additional charge")                                         :
      reason==EVENT_REASON_ACCOUNT_CORRECTION               ?  TextByLanguage("Корректирующая запись","Correction")                                               :
      reason==EVENT_REASON_ACCOUNT_BONUS                    ?  TextByLanguage("Перечисление бонусов","Bonus")                                                     :
      reason==EVENT_REASON_ACCOUNT_COMISSION                ?  TextByLanguage("Дополнительные комиссии","Additional commission")                                  :
      reason==EVENT_REASON_ACCOUNT_COMISSION_DAILY          ?  TextByLanguage("Комиссия, начисляемая в конце торгового дня","Daily commission")                   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_MONTHLY        ?  TextByLanguage("Комиссия, начисляемая в конце месяца","Monthly commission")                        :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_DAILY    ?  TextByLanguage("Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission")   :
      reason==EVENT_REASON_ACCOUNT_COMISSION_AGENT_MONTHLY  ?  TextByLanguage("Агентская комиссия, начисляемая в конце месяца","Monthly agent commission")        :
      reason==EVENT_REASON_ACCOUNT_INTEREST                 ?  TextByLanguage("Начисления процентов на свободные средства","Interest rate")                       :
      reason==EVENT_REASON_BUY_CANCELLED                    ?  TextByLanguage("Отмененная сделка покупки","Canceled buy deal")                                    :
      reason==EVENT_REASON_SELL_CANCELLED                   ?  TextByLanguage("Отмененная сделка продажи","Canceled sell deal")                                   :
      reason==EVENT_REASON_DIVIDENT                         ?  TextByLanguage("Начисление дивиденда","Dividend operations")                                       :
      reason==EVENT_REASON_DIVIDENT_FRANKED                 ?  TextByLanguage("Начисление франкированного дивиденда","Franked (non-taxable) dividend operations") :
      reason==EVENT_REASON_TAX                              ?  TextByLanguage("Начисление налога","Tax charges")                                                  :
      EnumToString(reason)
     );
  }
//+------------------------------------------------------------------+
```

**Add the methods returning added new properties to the**
**section of the simplified access to event properties in the public section of the class:**

```
//+------------------------------------------------------------------+
//| Methods of simplified access to event object properties          |
//+------------------------------------------------------------------+
//--- Return (1) event type, (2) event time in milliseconds, (3) event status, (4) event reason, (5) deal type, (6) deal ticket,
//--- (7) order type, based on which a deal was executed, (8) position opening order type, (9) position last order ticket,
//--- (10) position first order ticket, (11) position ID, (12) opposite position ID, (13) magic number, (14) opposite position magic number, (15) position open time
   ENUM_TRADE_EVENT  TypeEvent(void)                                    const { return (ENUM_TRADE_EVENT)this.GetProperty(EVENT_PROP_TYPE_EVENT);           }
   long              TimeEvent(void)                                    const { return this.GetProperty(EVENT_PROP_TIME_EVENT);                             }
   ENUM_EVENT_STATUS Status(void)                                       const { return (ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);        }
   ENUM_EVENT_REASON Reason(void)                                       const { return (ENUM_EVENT_REASON)this.GetProperty(EVENT_PROP_REASON_EVENT);        }
   ENUM_DEAL_TYPE    TypeDeal(void)                                     const { return (ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);        }
   long              TicketDeal(void)                                   const { return this.GetProperty(EVENT_PROP_TICKET_DEAL_EVENT);                      }
   ENUM_ORDER_TYPE   TypeOrderEvent(void)                               const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT);      }
   ENUM_ORDER_TYPE   TypeFirstOrderPosition(void)                       const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION);   }
   long              TicketOrderEvent(void)                             const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_EVENT);                     }
   long              TicketFirstOrderPosition(void)                     const { return this.GetProperty(EVENT_PROP_TICKET_ORDER_POSITION);                  }
   long              PositionID(void)                                   const { return this.GetProperty(EVENT_PROP_POSITION_ID);                            }
   long              PositionByID(void)                                 const { return this.GetProperty(EVENT_PROP_POSITION_BY_ID);                         }
   long              Magic(void)                                        const { return this.GetProperty(EVENT_PROP_MAGIC_ORDER);                            }
   long              MagicCloseBy(void)                                 const { return this.GetProperty(EVENT_PROP_MAGIC_BY_ID);                            }
   long              TimePosition(void)                                 const { return this.GetProperty(EVENT_PROP_TIME_ORDER_POSITION);                    }

//--- When changing position direction, return (1) previous position order type, (2) previous position order ticket
//--- (3) current position order type, (4) current position order ticket
//--- (5) position type and (6) ticket before changing direction, (7) position type and (8) ticket after changing direction
   ENUM_ORDER_TYPE   TypeOrderPosPrevious(void)                         const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE);   }
   long              TicketOrderPosPrevious(void)                       const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE); }
   ENUM_ORDER_TYPE   TypeOrderPosCurrent(void)                          const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT);  }
   long              TicketOrderPosCurrent(void)                        const { return (ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT);}
   ENUM_POSITION_TYPE TypePositionPrevious(void)                        const { return PositionTypeByOrderType(this.TypeOrderPosPrevious());                }
   ulong             TicketPositionPrevious(void)                       const { return this.TicketOrderPosPrevious();                                       }
   ENUM_POSITION_TYPE TypePositionCurrent(void)                         const { return PositionTypeByOrderType(this.TypeOrderPosCurrent());                 }
   ulong             TicketPositionCurrent(void)                        const { return this.TicketOrderPosCurrent();                                        }

//--- Return (1) the price the event occurred at, (2) open price, (3) close price,
//--- (4) StopLoss price, (5) TakeProfit price, (6) profit, (7) requested order volume,
//--- (8) executed order volume, (9) remaining order volume, (10) executed position volume
   double            PriceEvent(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_EVENT);                            }
   double            PriceOpen(void)                                    const { return this.GetProperty(EVENT_PROP_PRICE_OPEN);                             }
   double            PriceClose(void)                                   const { return this.GetProperty(EVENT_PROP_PRICE_CLOSE);                            }
   double            PriceStopLoss(void)                                const { return this.GetProperty(EVENT_PROP_PRICE_SL);                               }
   double            PriceTakeProfit(void)                              const { return this.GetProperty(EVENT_PROP_PRICE_TP);                               }
   double            Profit(void)                                       const { return this.GetProperty(EVENT_PROP_PROFIT);                                 }
   double            VolumeOrderInitial(void)                           const { return this.GetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL);                   }
   double            VolumeOrderExecuted(void)                          const { return this.GetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED);                  }
   double            VolumeOrderCurrent(void)                           const { return this.GetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT);                   }
   double            VolumePositionExecuted(void)                       const { return this.GetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED);               }

//--- When modifying prices, return (1) order price, (2) StopLoss and (3) TakeProfit before modification
   double            PriceOpenBefore(void)                              const { return this.GetProperty(EVENT_PROP_PRICE_OPEN_BEFORE);                      }
   double            PriceStopLossBefore(void)                          const { return this.GetProperty(EVENT_PROP_PRICE_SL_BEFORE);                        }
   double            PriceTakeProfitBefore(void)                        const { return this.GetProperty(EVENT_PROP_PRICE_TP_BEFORE);                        }
   double            PriceEventAsk(void)                                const { return this.GetProperty(EVENT_PROP_PRICE_EVENT_ASK);                        }
   double            PriceEventBid(void)                                const { return this.GetProperty(EVENT_PROP_PRICE_EVENT_BID);                        }

//--- Return the (1) symbol and (2) opposite position symbol
   string            Symbol(void)                                       const { return this.GetProperty(EVENT_PROP_SYMBOL);                                 }
   string            SymbolCloseBy(void)                                const { return this.GetProperty(EVENT_PROP_SYMBOL_BY_ID);                           }

//+------------------------------------------------------------------+
```

Since most class properties are filled in the event collection class of the CreateNewEvent() method and an event type is then set by calling the
SetTypeEvent() method of the CEvent class,

set Digits() of the event symbol in the **SetTypeEvent()**
method of the

**CEvent** class together with defining modification events:

```
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEvent::SetTypeEvent(void)
  {
//--- Set event symbol Digits()
   this.m_digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
//--- Pending order placed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_PLASED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_PLASED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Pending order removed (check for matching the event code since there can only be one flag here)
   if(this.m_event_code==TRADE_EVENT_FLAG_ORDER_REMOVED)
     {
      this.m_trade_event=TRADE_EVENT_PENDING_ORDER_REMOVED;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Pending order is modified
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_MODIFY))
     {
      //--- If the placement price is modified
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_PRICE))
        {
         this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_PRICE;
         //--- If StopLoss and TakeProfit are modified
         if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL) && this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT;
         //--- If StopLoss is modified
         else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS;
         //--- If TakeProfit is modified
         else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT;
        }
      //--- If the placement price is not modified
      else
        {
         //--- If StopLoss and TakeProfit are modified
         if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL) && this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT;
         //--- If StopLoss is modified
         else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_STOP_LOSS;
         //--- If TakeProfit is modified
         else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
            this.m_trade_event=TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT;
        }
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- If a position is modified
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_MODIFY))
     {
      //--- If StopLoss and TakeProfit are modified
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL) && this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
         this.m_trade_event=TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT;
      //--- If StopLoss is modified
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
         this.m_trade_event=TRADE_EVENT_MODIFY_POSITION_STOP_LOSS;
      //--- If TakeProfit is modified
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
         this.m_trade_event=TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT;
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
//--- Position opened (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
   //--- If an existing position is changed
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CHANGED))
        {
         //--- If a pending order is activated by a price
         if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED))
           {
            //--- If this is a position reversal
            if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_REVERSE))
              {
               //--- check the partial closure flag and set the
               //--- "position reversal by activation of a pending order" or "position reversal by partial activation of a pending order" trading event
               this.m_trade_event=
                 (
                  !this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ?
                  TRADE_EVENT_POSITION_REVERSED_BY_PENDING :
                  TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL
                 );
               this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
               return;
              }
            //--- If this is adding a volume to a position
            else
              {
               //--- check the partial opening flag and set the
               //--- "added volume to a position by activating a pending order" or "added volume to a position by partially activating a pending order" trading event
               this.m_trade_event=
                 (
                  !this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ?
                  TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING :
                  TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL
                 );
               this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
               return;
              }
           }
         //--- If a position was changed by a market deal
         else
           {
            //--- If this is a position reversal
            if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_REVERSE))
              {
               //--- check the partial opening flag and set the "position reversal" or "position reversal by partial execution" trading event
               this.m_trade_event=
                 (
                  !this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ?
                  TRADE_EVENT_POSITION_REVERSED_BY_MARKET :
                  TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL
                 );
               this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
               return;
              }
            //--- If this is adding a volume to a position
            else
              {
               //--- check the partial opening flag and set "added volume to a position" or "added volume to a position by partial execution" trading event
               this.m_trade_event=
                 (
                  !this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ?
                  TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET :
                  TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL
                 );
               this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
               return;
              }
           }
        }
   //--- If a new position is opened
      else
        {
         //--- If a pending order is activated by a price
         if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_ORDER_ACTIVATED))
           {
            //--- check the partial opening flag and set "pending order activated" or "pending order partially activated" trading event
            this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_PENDING_ORDER_ACTIVATED : TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL);
            this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
            return;
           }
         //--- check the partial opening flag and set the "Position opened" or "Position partially opened" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_OPENED : TRADE_EVENT_POSITION_OPENED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
     }

//--- Position closed (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CLOSED))
     {
      //--- if a position is closed by StopLoss
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_SL))
        {
         //--- check the partial closing flag and set the "Position closed by StopLoss" or "Position partially closed by StopLoss" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_SL : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by TakeProfit
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
        {
         //--- check the partial closure flag and set the "Position closed by TakeProfit" or "Position partially closed by TakeProfit" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by an opposite one
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closure flag and set the "Position closed by opposite one" or "Position partially closed by opposite one" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- If a position is closed
      else
        {
         //--- check the partial closure flag and set the "Position closed" or "Position partially closed" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED : TRADE_EVENT_POSITION_CLOSED_PARTIAL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
     }
//--- Balance operation on the account (clarify the event by deal type)
   if(this.m_event_code==TRADE_EVENT_FLAG_ACCOUNT_BALANCE)
     {
      //--- Initialize a trading event
      this.m_trade_event=TRADE_EVENT_NO_EVENT;
      //--- Take a deal type
      ENUM_DEAL_TYPE deal_type=(ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT);
      //--- if a deal is a balance operation
      if(deal_type==DEAL_TYPE_BALANCE)
        {
        //--- check the deal profit and set an event (funds deposit or withdrawal)
         this.m_trade_event=(this.GetProperty(EVENT_PROP_PROFIT)>0 ? TRADE_EVENT_ACCOUNT_BALANCE_REFILL : TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL);
        }
      //--- The remaining balance operation types match the ENUM_DEAL_TYPE enumeration starting from DEAL_TYPE_CREDIT
      else if(deal_type>DEAL_TYPE_BALANCE)
        {
        //--- set the event
         this.m_trade_event=(ENUM_TRADE_EVENT)deal_type;
        }
      this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
      return;
     }
  }
//+------------------------------------------------------------------+
```

The code comments in the method listing describe all the necessary checks and actions, so there is no point in dwelling on the already
commented actions. I believe, everything is quite simple and easy to understand here.

This completes the improvement of the abstract event class.

Looking a bit ahead, it should be noted that when checking the tracking of price modification for placing pending orders in a test EA, it became
necessary to find an order furthest from the price. Looking through the properties of orders, I realized that the library has no quick
and versatile solution for this. Therefore, we will use one of the additional order integer properties — profit in points. For pending
orders, this is a distance of an order from the price in points. Thus, to find the order that is furthest from the price, we simply need to
look for an order with the highest “profit” (distance) in points.



This case is similar to searching all pending orders by their direction. To find a pending order, which is the furthest one from the
price, we select all orders in one direction and sort the obtained list by the greatest distance. As a result, we obtain one order from all
orders of various types, although in one direction (BuyLimit, BuyStop and BuyStopLimit are all Buy ones. The opposite is true for
Sell).

**Let's change the method of obtaining the type of order by its direction in the listing of the Order.mqh abstract order class:**

```
//+------------------------------------------------------------------+
//| Return order profit in points                                    |
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
//+------------------------------------------------------------------+
```

Here we essentially add a check for pending orders and
return

a distance from the order price to the current price in points.

**Let's add the display of the distance from the price to the pending order in**
**points to the method of describing integer properties of the abstract order class:**

```
//+------------------------------------------------------------------+
//| Return description of an order's integer property                |
//+------------------------------------------------------------------+
string COrder::GetPropertyDescription(ENUM_ORDER_PROP_INTEGER property)
  {
   return
     (
   //--- General properties
      property==ORDER_PROP_MAGIC             ?  TextByLanguage("Магик","Magic")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET            ?  TextByLanguage("Тикет","Ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_FROM       ?  TextByLanguage("Тикет родительского ордера","Parent order ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TICKET_TO         ?  TextByLanguage("Тикет наследуемого ордера","Inherited order ticket")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          " #"+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN         ?  TextByLanguage("Время открытия","Time open")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==ORDER_PROP_TIME_CLOSE        ?  TextByLanguage("Время закрытия","Close time")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS)
         )  :
      property==ORDER_PROP_TIME_EXP          ?  TextByLanguage("Дата экспирации","Expiration date")+
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
      property==ORDER_PROP_POSITION_ID       ?  TextByLanguage("Идентификатор позиции","Position ID")+
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
      property==ORDER_PROP_POSITION_BY_ID    ?  TextByLanguage("Идентификатор встречной позиции","Opposite position ID")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_TIME_OPEN_MSC     ?  TextByLanguage("Время открытия в милисекундах","Open time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_CLOSE_MSC    ?  TextByLanguage("Время закрытия в милисекундах","Close time in milliseconds")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+TimeMSCtoString(this.GetProperty(property))+" ("+(string)this.GetProperty(property)+")"
         )  :
      property==ORDER_PROP_TIME_UPDATE       ?  TextByLanguage("Время изменения позиции","Position change time")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)!=0 ? ::TimeToString(this.GetProperty(property),TIME_DATE|TIME_MINUTES|TIME_SECONDS) : "0")
         )  :
      property==ORDER_PROP_TIME_UPDATE_MSC   ?  TextByLanguage("Время изменения позиции в милисекундах","Time to change the position in milliseconds")+
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
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_CLOSE_BY_TP       ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ?  TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_GROUP_ID          ?  TextByLanguage("Идентификатор группы","Group ID")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Here we check the order status and if this is an existing pending order, a
message about the distance is displayed, otherwise a message about
the profit is displayed in points.

This completes the changes in the abstract order class.

**Now we need to create another class that inherits the CEvent abstract event class. This is a modification event class.**

[When implementing working on netting accounts in the sixth article](https://www.mql5.com/en/articles/6383#node02),
we improved the position opening class event: the CEventPositionOpen class now features the method creating a short message text
depending on the event status and the presence of some event object properties.

When creating a new modification event, we do the same — checking the modification event type and creating an event text depending on the
obtained type. Also, when sending an event to the control program chart, we need to define the price to be passed in the

**dparam** parameter of the [EventChartCustom()](https://www.mql5.com/en/docs/eventfunctions/eventchartcustom)
function. In the position opening event class, we used this parameter to pass the open price, while in the modification event class, several
price change options are possible, and we need to decide on the

prices we are to send in the user event's dparam parameter:

- Only an order price can be changed — send a new pending order price,
- an order and StopLoss prices can be changed — send a new pending order price,
- an order and TakeProfit prices can be changed — send a new pending order price,
- an order, StopLoss and TakeProfit prices can be changed — send a new pending order price,
- order StopLoss can be changed — send a new StopLoss price,
- order TakeProfit can be changed — send a new TakeProfit price.
- position StopLoss can be changed — send position StopLoss,
- position TakeProfit can be changed — send position TakeProfit,
- position StopLoss and TakeProfit can be changed — send position open price.

As we can see, when changing a single price, we pass the changed price to the event. When changing several prices simultaneously, we send
position open or order price only (which in turn can also be changed). In a custom program, you can clarify the change of each of the prices
(during their simultaneous modification) by the type of the occurred modification event.

In the new **EventModify.mqh** file of the library's \\MQL5\\Include\\DoEasy\\Objects\\Events folder, create the new **CEventModify**
class.



Set the CEvent abstract event class as a base class for it.




Do not forget to include the file of the abstract event class to the
modification class file.

Since the class is relatively small, I will provide its full listing here for you to study. I have
already described a similar class

[in the sixth part of the library description](https://www.mql5.com/en/articles/6383#node02) when implementing
changes in the

**CEventPositionOpen** class.

```
//+------------------------------------------------------------------+
//|                                                  EventModify.mqh |
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
class CEventModify : public CEvent
  {
private:
   double            m_price;                               // Price passed to an event
//--- Create and return a brief event description
   string            EventsMessage(void);
public:
//--- Constructor
                     CEventModify(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MODIFY,event_code,ticket),m_price(0) {}
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
bool CEventModify::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
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
bool CEventModify::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   if(property==EVENT_PROP_PRICE_CLOSE             ||
      property==EVENT_PROP_PROFIT
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventModify::PrintShort(void)
  {
   ::Print(this.EventsMessage());
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventModify::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.TicketOrderEvent(),this.m_price,this.Symbol());
  }
//+------------------------------------------------------------------+
//| Create and return a brief event message                          |
//+------------------------------------------------------------------+
string CEventModify::EventsMessage(void)
  {
//--- (1) header, (2) magic number
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string text="";

//--- Pending order price is modified
   if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string price="["+::DoubleToString(this.PriceOpenBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceOpen(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирована цена: ",": modified price: ")+price+magic;
      this.m_price=this.PriceOpen();
     }
//--- Pending order price and StopLoss are modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string price="["+::DoubleToString(this.PriceOpenBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceOpen(),this.m_digits)+"]";
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирована цена: ",": modified price: ")+price+TextByLanguage(" и"," and")+" StopLoss: "+sl+magic;
      this.m_price=this.PriceOpen();
     }
//--- Pending order price and TakeProfit are modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string price="["+::DoubleToString(this.PriceOpenBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceOpen(),this.m_digits)+"]";
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирована цена: ",": modified price: ")+price+TextByLanguage(" и"," and")+" TakeProfit: "+tp+magic;
      this.m_price=this.PriceOpen();
     }
//--- Pending order price, as well as its StopLoss and TakeProfit are modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string price="["+::DoubleToString(this.PriceOpenBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceOpen(),this.m_digits)+"]";
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирована цена: ",": modified price: ")+price+", StopLoss: "+sl+TextByLanguage(" и"," and")+" TakeProfit: "+tp+magic;
      this.m_price=this.PriceOpen();
     }
//--- Pending order StopLoss is modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован StopLoss: ",": modified StopLoss: ")+sl+magic;
      this.m_price=this.PriceStopLoss();
     }
//--- Pending order TakeProfit is modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован TakeProfit: ",": modified TakeProfit: ")+tp+magic;
      this.m_price=this.PriceTakeProfit();
     }
//--- Pending order StopLoss and TakeProfit are modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT)
     {
      string order=OrderTypeDescription(this.TypeOrderPosCurrent())+" #"+(string)this.TicketOrderPosCurrent();
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован StopLoss: ",": modified StopLoss: ")+sl+TextByLanguage(" и"," and")+" TakeProfit: "+tp+magic;
      this.m_price=this.PriceOpen();
     }
//--- Position StopLoss is modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS)
     {
      string order=PositionTypeDescription(this.TypePositionCurrent())+" #"+(string)this.TicketPositionCurrent();
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован StopLoss: ",": modified StopLoss: ")+sl+magic;
      this.m_price=this.PriceStopLoss();
     }
//--- Position TakeProfit is modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT)
     {
      string order=PositionTypeDescription(this.TypePositionCurrent())+" #"+(string)this.TicketPositionCurrent();
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован TakeProfit: ",": modified TakeProfit: ")+tp+magic;
      this.m_price=this.PriceTakeProfit();
     }
//--- Position StopLoss and TakeProfit are modified
   else if(this.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT)
     {
      string order=PositionTypeDescription(this.TypePositionCurrent())+" #"+(string)this.TicketPositionCurrent();
      string sl="["+::DoubleToString(this.PriceStopLossBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceStopLoss(),this.m_digits)+"]";
      string tp="["+::DoubleToString(this.PriceTakeProfitBefore(),this.m_digits)+" --> "+::DoubleToString(this.PriceTakeProfit(),this.m_digits)+"]";
      text=order+TextByLanguage(": модифицирован StopLoss: ",": modified StopLoss: ")+sl+TextByLanguage(" и"," and")+" TakeProfit: "+tp+magic;
      this.m_price=this.PriceOpen();
     }
   return head+this.Symbol()+" "+text;
  }
//+------------------------------------------------------------------+
```

Now we need to define events of modifying the already existing orders and positions, create a new event and add it to the event collection list
in the event collection class.

**Let's implement the necessary improvements to the CEventsCollection class** in the **EventsCollection.mqh**
file from \\MQL5\\Include\\DoEasy\\Collections of the library folder.

Include the file of the new modification event class.


In the private class section, declare the

class member variable —
the structure for storing the tick data. It is to be used to obtain data on the last modification event prices.

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
#include "..\Objects\Events\EventModify.mqh"
//+------------------------------------------------------------------+
//| Account event collection                                         |
//+------------------------------------------------------------------+
class CEventsCollection : public CListObj
  {
private:
   CListObj          m_list_events;                   // Event list
   bool              m_is_hedge;                      // Hedging account flag
   long              m_chart_id;                      // Control program chart ID
   int               m_trade_event_code;              // Trading event code
   ENUM_TRADE_EVENT  m_trade_event;                   // Account trading event
   CEvent            m_event_instance;                // Event object for searching by property
   MqlTick           m_tick;                          // Last tick structure

```

Initialize the tick structure in the class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEventsCollection::CEventsCollection(void) : m_trade_event(TRADE_EVENT_NO_EVENT),m_trade_event_code(TRADE_EVENT_FLAG_NO_EVENT)
  {
   this.m_list_events.Clear();
   this.m_list_events.Sort(SORT_BY_EVENT_TIME_EVENT);
   this.m_list_events.Type(COLLECTION_EVENTS_ID);
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   this.m_chart_id=::ChartID();
   ::ZeroMemory(this.m_tick);
  }
//+------------------------------------------------------------------+
```

[In the seventh part of the library description](https://www.mql5.com/en/articles/6482#node02), we have developed
an overloaded method for creating a new event. Now we have two of them — the method for creating events when changing the number of orders and
positions on the account and the method creating a new event when changing (modifying) an already existing order or position.

**The second method should be improved, so that it is able to track order and position modification events** (in
the seventh part, the method processed only a StopLimit order activation event).

Let's add the code strings handling an order/position modification event
and saving order/position properties before modification:

```
//+------------------------------------------------------------------+
//| Create a trading event depending on the order change type        |
//+------------------------------------------------------------------+
void CEventsCollection::CreateNewEvent(COrderControl* order)
  {
   if(!::SymbolInfoTick(order.Symbol(),this.m_tick))
     {
      Print(DFUN,TextByLanguage("Не удалось получить текущие цены по символу события ","Failed to get current prices by event symbol "),order.Symbol());
      return;
     }
   CEvent* event=NULL;
//--- Pending StopLimit order activated
   if(order.GetChangeType()==CHANGE_TYPE_ORDER_TYPE)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_PLASED;
      event=new CEventOrderPlased(this.m_trade_event_code,order.Ticket());
     }
//--- Modification
   else
     {
      //--- Pending order price is modified
      if(order.GetChangeType()==CHANGE_TYPE_ORDER_PRICE)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_PRICE;
      //--- Pending order price and StopLoss are modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_PRICE_STOP_LOSS)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_PRICE+TRADE_EVENT_FLAG_SL;
      //--- Pending order price and TakeProfit are modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_PRICE_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_PRICE+TRADE_EVENT_FLAG_TP;
      //--- Pending order price, as well as its StopLoss and TakeProfit are modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_PRICE+TRADE_EVENT_FLAG_SL+TRADE_EVENT_FLAG_TP;
      //--- Pending order StopLoss is modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_STOP_LOSS)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_SL;
      //--- Pending order TakeProfit is modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_TP;
      //--- Pending order StopLoss and TakeProfit are modified
      else if(order.GetChangeType()==CHANGE_TYPE_ORDER_STOP_LOSS_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_MODIFY+TRADE_EVENT_FLAG_SL+TRADE_EVENT_FLAG_TP;

      //--- Position StopLoss is modified
      else if(order.GetChangeType()==CHANGE_TYPE_POSITION_STOP_LOSS)
         this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_MODIFY+TRADE_EVENT_FLAG_SL;
      //--- Position TakeProfit is modified
      else if(order.GetChangeType()==CHANGE_TYPE_POSITION_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_MODIFY+TRADE_EVENT_FLAG_TP;
      //--- Position StopLoss and TakeProfit are modified
      else if(order.GetChangeType()==CHANGE_TYPE_POSITION_STOP_LOSS_TAKE_PROFIT)
         this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_MODIFY+TRADE_EVENT_FLAG_SL+TRADE_EVENT_FLAG_TP;

      //--- Create a modification event
      event=new CEventModify(this.m_trade_event_code,order.Ticket());
     }
//--- Create an event
   if(event!=NULL)
     {
      event.SetProperty(EVENT_PROP_TIME_EVENT,order.Time());                        // Event time
      event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_STOPLIMIT_TRIGGERED);  // Event reason (from the ENUM_EVENT_REASON enumeration)
      event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,PositionTypeByOrderType((ENUM_ORDER_TYPE)order.TypeOrderPrev())); // Type of the order that triggered an event
      event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Ticket of the order that triggered an event
      event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());             // Event order type
      event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());              // Event order ticket
      event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());          // First position order type
      event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());           // First position order ticket
      event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
      event.SetProperty(EVENT_PROP_POSITION_BY_ID,0);                               // Opposite position ID
      event.SetProperty(EVENT_PROP_MAGIC_BY_ID,0);                                  // Opposite position magic number

      event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrderPrev());      // Position order type before changing the direction
      event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());           // Position order ticket before changing direction
      event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());         // Current position order type
      event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());          // Current position order ticket

      event.SetProperty(EVENT_PROP_PRICE_OPEN_BEFORE,order.PricePrev());            // Order price before modification
      event.SetProperty(EVENT_PROP_PRICE_SL_BEFORE,order.StopLossPrev());           // StopLoss price before modification
      event.SetProperty(EVENT_PROP_PRICE_TP_BEFORE,order.TakeProfitPrev());         // TakeProfit price before modification
      event.SetProperty(EVENT_PROP_PRICE_EVENT_ASK,this.m_tick.ask);                // Ask price during an event
      event.SetProperty(EVENT_PROP_PRICE_EVENT_BID,this.m_tick.bid);                // Bid price during an event

      event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order magic number
      event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimePrev());           // Position first order time
      event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PricePrev());                  // Price the event occurred at
      event.SetProperty(EVENT_PROP_PRICE_OPEN,order.Price());                       // Order placement price
      event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.Price());                      // Order close price
      event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                      // Order StopLoss price
      event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                    // Order TakeProfit price
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

Handling conditions of various modification types is
relatively easy and described in the code comments. Depending on the order/position change type, the event code is created using a set of
flags. The code is sent to the CEventModify class constructor when creating a new modification event.

The color-marked code blocks for saving new order/position properties are added
to all methods for saving the class position/order properties. We will not dwell on them here since their code strings are identical. They
can be found in the attached files below.

Now all is ready to test events of modifying existing orders and positions.

### Testing order and position modification events

To perform the test, we will need to supplement the existing set of the test EA buttons from the [seventh \\
article](https://www.mql5.com/en/articles/6482).

Let's add three more buttons to it together with their press handlers: **Set StopLoss**, **Set**
**TakeProfit** and **Trailing All**.

The first two buttons set stop loss and take profit to all orders and positions that do not
have them, while the third button will have two states — Enabled/Disabled, i.e. when pressing it, the button remains pressed and the two
trailing functions start working. As a result, the EA starts trailing stop levels of all positions and move all active pending orders while
following the price. Pressing the button yet again disables both trailings.

Let's take TestDoEasyPart07.mq5 EA from \\MQL5\\Experts\\TestDoEasy\\Part07 and save it in the new \\MQL5\\Experts\\TestDoEasy\ **Part08**
folder under the name

**TestDoEasyPart08.mq5**.

Add three new constants to the buttons enumeration and change the
total number of buttons

from 17 to **20**in the appropriate macro
substitution:

```
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
   BUTT_DELETE_PENDING,
   BUTT_CLOSE_ALL,
   BUTT_PROFIT_WITHDRAWAL,
   BUTT_SET_STOP_LOSS,
   BUTT_SET_TAKE_PROFIT,
   BUTT_TRAILING_ALL
  };
#define TOTAL_BUTT   (20)
```

Add the variables for specifying the StopLoss level distance from the price,

trailing step, number
of profit points to start trailing, as well as StopLoss and TakeProfit
in points to be set by clicking the appropriate buttons (the InpStopLoss and InpTakeProfit parameters are used to set stop levels immediately
after opening/placing a pending order) to the inputs.

Add the necessary variables for storing the values of newly added inputs
and the

flag variable indicating the activity of trailing functions to the
list of global variables:

```
//--- input variables
input ulong    InpMagic             =  123;  // Magic number
input double   InpLots              =  0.1;  // Lots
input uint     InpStopLoss          =  50;   // StopLoss in points
input uint     InpTakeProfit        =  50;   // TakeProfit in points
input uint     InpDistance          =  50;   // Pending orders distance (points)
input uint     InpDistanceSL        =  50;   // StopLimit orders distance (points)
input uint     InpSlippage          =  0;    // Slippage in points
input double   InpWithdrawal        =  10;   // Withdrawal funds (in tester)
input uint     InpButtShiftX        =  40;   // Buttons X shift
input uint     InpButtShiftY        =  10;   // Buttons Y shift
input uint     InpTrailingStop      =  50;   // Trailing Stop (points)
input uint     InpTrailingStep      =  20;   // Trailing Step (points)
input uint     InpTrailingStart     =  0;    // Trailing Start (points)
input uint     InpStopLossModify    =  20;   // StopLoss for modification (points)
input uint     InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
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
bool           trailing_on;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
//+------------------------------------------------------------------+
```

Since this is a test EA, the program operation is often completed in a critical error when debugging the library. In these cases, all plotted
graphical objects (buttons) remain on the chart. After the error is fixed and the EA is relaunched, it is unable to re-draw the buttons. You
have to launch it once again to let it first remove the existing buttons from the chart in its OnDeinit() handler, so that it is able to re-draw
all the buttons on a clean chart during the next launch.

Add the check for the presence of the buttons on the chart
to the OnInit() handler,

set the values for the trailing functions variables and stop
levels, check the trailing button activity flag and enable the button if the
flag is set after plotting all the buttons.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal,
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();
//--- check for undeleted objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);
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
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;
//--- create buttons
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- set button trailing
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- setting trade parameters
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

Let's write the function for defining the presence of a graphical object with the
specified prefix on the chart and the function for tracking the buttons'
status. For more convenience in reading the code, we will move the tracking from the EA's OnTick() handler into a separate function:

```
//+------------------------------------------------------------------+
//| Return the flag of a prefixed object presence                    |
//+------------------------------------------------------------------+
bool IsPresentObects(const string object_prefix)
  {
   for(int i=ObjectsTotal(0)-1;i>=0;i--)
      if(StringFind(ObjectName(0,i,0),object_prefix)>WRONG_VALUE)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Tracking the buttons' status                                     |
//+------------------------------------------------------------------+
void PressButtonsControl(void)
  {
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
```

**Let's change the function for setting the button object status:**

```
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
   if(name==butt_data[TOTAL_BUTT-1].name)
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'220,255,240');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
  }
//+------------------------------------------------------------------+
```

Here:

set the button status (enabled/disabled),

if
this is the very last button and

if it is "enabled",
change the button object background color,

otherwise, return the background color to the "disabled" status.

Since we have three new buttons, addconverting the names of new button objects to their text
to the function of creating a button text from its name:

```
//+------------------------------------------------------------------+
//| Convert enumeration into the button text                         |
//+------------------------------------------------------------------+
string EnumToButtText(const ENUM_BUTTONS member)
  {
   string txt=StringSubstr(EnumToString(member),5);
   StringToLower(txt);
   StringReplace(txt,"set_take_profit","Set TakeProfit");
   StringReplace(txt,"set_stop_loss","Set StopLoss");
   StringReplace(txt,"trailing_all","Trailing All");
   StringReplace(txt,"buy","Buy");
   StringReplace(txt,"sell","Sell");
   StringReplace(txt,"_limit"," Limit");
   StringReplace(txt,"_stop"," Stop");
   StringReplace(txt,"close_","Close ");
   StringReplace(txt,"2"," 1/2");
   StringReplace(txt,"_by_"," by ");
   StringReplace(txt,"profit_","Profit ");
   StringReplace(txt,"delete_","Delete ");
   return txt;
  }
//+------------------------------------------------------------------+
```

Now we need to handle pressing of the three new buttons. To achieve this, add the following code strings at the very end of the **PressButtonEvents()** button
pressing handling function (

after the code block handling the pressing of the funds withdrawal button):

```
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
      //--- If the BUTT_SET_STOP_LOSS button is pressed: Place StopLoss to all orders and positions where it is not present
      if(button==EnumToString(BUTT_SET_STOP_LOSS))
        {
         SetStopLoss();
        }
      //--- If the BUTT_SET_TAKE_PROFIT button is pressed: Place TakeProfit to all orders and positions where it is not present
      if(button==EnumToString(BUTT_SET_TAKE_PROFIT))
        {
         SetTakeProfit();
        }
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button (if this is not a trailing button)
      if(button!=EnumToString(BUTT_TRAILING_ALL))
         ButtonState(button_name,false);
      //--- If the BUTT_TRAILING_ALL button is pressed
      else
        {
         //--- Set the color of the active button
         ButtonState(button_name,true);
         trailing_on=true;
        }
      //--- re-draw the chart
      ChartRedraw();
     }
   //--- Return the inactive button color (if this is a trailing button)
   else if(button==EnumToString(BUTT_TRAILING_ALL))
     {
      ButtonState(button_name,false);
      trailing_on=false;
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

As we can see, the two new functions are called here: SetStopLoss() and SetTakeProfit(). They allow you to set the appropriate order and
position levels:

```
//+------------------------------------------------------------------+
//| Set StopLoss to all orders and positions                         |
//+------------------------------------------------------------------+
void SetStopLoss(void)
  {
   if(stoploss_to_modify==0)
      return;
//--- Set StopLoss to all positions where it is absent
   CArrayObj* list=engine.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SL,0,EQUAL);
   if(list==NULL)
      return;
   int total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* position=list.At(i);
      if(position==NULL)
         continue;
      double sl=CorrectStopLoss(position.Symbol(),position.TypeByDirection(),0,stoploss_to_modify);
      trade.PositionModify(position.Ticket(),sl,position.TakeProfit());
     }
//--- Set StopLoss to all pending orders where it is absent
   list=engine.GetListMarketPendings();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SL,0,EQUAL);
   if(list==NULL)
      return;
   total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* order=list.At(i);
      if(order==NULL)
         continue;
      double sl=CorrectStopLoss(order.Symbol(),(ENUM_ORDER_TYPE)order.TypeOrder(),order.PriceOpen(),stoploss_to_modify);
      trade.OrderModify(order.Ticket(),order.PriceOpen(),sl,order.TakeProfit(),trade.RequestTypeTime(),trade.RequestExpiration(),order.PriceStopLimit());
     }
  }
//+------------------------------------------------------------------+
//| Set TakeProfit to all orders and positions                       |
//+------------------------------------------------------------------+
void SetTakeProfit(void)
  {
   if(takeprofit_to_modify==0)
      return;
//--- Set TakeProfit to all positions where it is absent
   CArrayObj* list=engine.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TP,0,EQUAL);
   if(list==NULL)
      return;
   int total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* position=list.At(i);
      if(position==NULL)
         continue;
      double tp=CorrectTakeProfit(position.Symbol(),position.TypeByDirection(),0,takeprofit_to_modify);
      trade.PositionModify(position.Ticket(),position.StopLoss(),tp);
     }
//--- Set TakeProfit to all pending orders where it is absent
   list=engine.GetListMarketPendings();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TP,0,EQUAL);
   if(list==NULL)
      return;
   total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* order=list.At(i);
      if(order==NULL)
         continue;
      double tp=CorrectTakeProfit(order.Symbol(),(ENUM_ORDER_TYPE)order.TypeOrder(),order.PriceOpen(),takeprofit_to_modify);
      trade.OrderModify(order.Ticket(),order.PriceOpen(),order.StopLoss(),tp,trade.RequestTypeTime(),trade.RequestExpiration(),order.PriceStopLimit());
     }
  }
//+------------------------------------------------------------------+
```

The functions are quite simple. Let's have a look at placing TakeProfit to all orders and positions it is not present at:

First, check stop losses to be set in points. If the value is zero, exit at once,
as there is nothing to change here.

Next, receive the list of active market positions only
and

sort it by TakeProfit equal to zero, by the absence of position's
TakeProfit.

Next, use a loop by the final list to obtain
positions from it, calculate correct TakeProfit for each of them using
the service function we have described in the

[fourth part](https://www.mql5.com/en/articles/5724#node02) of the library description and send
it to the position modification method of the standard library's [CTrade \\
class](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionmodify).

To set TakeProfit to orders, we obtain the list of active pending orders and perform the actions described above.

**Now we only have to write the functions for trailing position stops**
**and order placement prices:**

```
//+------------------------------------------------------------------+
//| Trailing stop of a position with the maximum profit              |
//+------------------------------------------------------------------+
void TrailingPositions(void)
  {
   MqlTick tick;
   if(!SymbolInfoTick(Symbol(),tick))
      return;
   double stop_level=StopLevel(Symbol(),2)*Point();
   //--- Get the list of all open positions
   CArrayObj* list=engine.GetListMarketPosition();
   //--- Select only Buy positions from the list
   CArrayObj* list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
   //--- Sort the list by profit considering commission and swap
   list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
   //--- Get the index of the Buy position with the maximum profit
   int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
   if(index_buy>WRONG_VALUE)
     {
      COrder* buy=list_buy.At(index_buy);
      if(buy!=NULL)
        {
         //--- Calculate the new StopLoss
         double sl=NormalizeDouble(tick.bid-trailing_stop,Digits());
         //--- If the price and the StopLevel based on it are higher than the new StopLoss (the distance by StopLevel is maintained)
         if(tick.bid-stop_level>sl)
           {
            //--- If the new StopLoss level exceeds the trailing step based on the current StopLoss
            if(buy.StopLoss()+trailing_step<sl)
              {
               //--- If we trail at any profit or position profit in points exceeds the trailing start, modify StopLoss
               if(trailing_start==0 || buy.ProfitInPoints()>(int)trailing_start)
                  trade.PositionModify(buy.Ticket(),sl,buy.TakeProfit());
              }
           }
        }
     }
   //--- Select only Sell positions from the list
   CArrayObj* list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
   //--- Sort the list by profit considering commission and swap
   list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
   //--- Get Sell position index with the maximum profit
   int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
   if(index_sell>WRONG_VALUE)
     {
      COrder* sell=list_sell.At(index_sell);
      if(sell!=NULL)
        {
         //--- Calculate the new StopLoss
         double sl=NormalizeDouble(tick.ask+trailing_stop,Digits());
         //--- If the price and StopLevel based on it are below the new StopLoss (the distance by StopLevel is maintained)
         if(tick.ask+stop_level<sl)
           {
            //--- If the new StopLoss level is below the trailing step based on the current StopLoss or a position has no StopLoss
            if(sell.StopLoss()-trailing_step>sl || sell.StopLoss()==0)
              {
               //--- If we trail at any profit or position profit in points exceeds the trailing start value, modify StopLoss
               if(trailing_start==0 || sell.ProfitInPoints()>(int)trailing_start)
                  trade.PositionModify(sell.Ticket(),sl,sell.TakeProfit());
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Trailing the farthest pending orders                             |
//+------------------------------------------------------------------+
void TrailingOrders(void)
  {
   MqlTick tick;
   if(!SymbolInfoTick(Symbol(),tick))
      return;
   double stop_level=StopLevel(Symbol(),2)*Point();
//--- Get the list of all placed orders
   CArrayObj* list=engine.GetListMarketPendings();
//--- Select only Buy orders from the list
   CArrayObj* list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_BUY,EQUAL);
   //--- Sort the list by distance from the price in points (by profit in points)
   list_buy.Sort(SORT_BY_ORDER_PROFIT_PT);
   //--- Get the index of the Buy order with the greatest distance
   int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_PT);
   if(index_buy>WRONG_VALUE)
     {
      COrder* buy=list_buy.At(index_buy);
      if(buy!=NULL)
        {
         //--- If the order is below the price (BuyLimit) and it should be "elevated" following the price
         if(buy.TypeOrder()==ORDER_TYPE_BUY_LIMIT)
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.ask-trailing_stop,Digits());
            double sl=(buy.StopLoss()>0 ? NormalizeDouble(price-(buy.PriceOpen()-buy.StopLoss()),Digits()) : 0);
            double tp=(buy.TakeProfit()>0 ? NormalizeDouble(price+(buy.TakeProfit()-buy.PriceOpen()),Digits()) : 0);
            //--- If the calculated price is below the StopLevel distance based on Ask order price (the distance by StopLevel is maintained)
            if(price<tick.ask-stop_level)
              {
               //--- If the calculated price exceeds the trailing step based on the order placement price, modify the order price
               if(price>buy.PriceOpen()+trailing_step)
                 {
                  trade.OrderModify(buy.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),buy.PriceStopLimit());
                 }
              }
           }
         //--- If the order exceeds the price (BuyStop and BuyStopLimit), and it should be "decreased" following the price
         else
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.ask+trailing_stop,Digits());
            double sl=(buy.StopLoss()>0 ? NormalizeDouble(price-(buy.PriceOpen()-buy.StopLoss()),Digits()) : 0);
            double tp=(buy.TakeProfit()>0 ? NormalizeDouble(price+(buy.TakeProfit()-buy.PriceOpen()),Digits()) : 0);
            //--- If the calculated price exceeds the StopLevel based on Ask order price (the distance by StopLevel is maintained)
            if(price>tick.ask+stop_level)
              {
               //--- If the calculated price is lower than the trailing step based on order price, modify the order price
               if(price<buy.PriceOpen()-trailing_step)
                 {
                  trade.OrderModify(buy.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),(buy.PriceStopLimit()>0 ? price-distance_stoplimit*Point() : 0));
                 }
              }
           }
        }
     }
//--- Select only Sell order from the list
   CArrayObj* list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_SELL,EQUAL);
   //--- Sort the list by the distance from the price in points (by profit in points)
   list_sell.Sort(SORT_BY_ORDER_PROFIT_PT);
   //--- Get the index of the Sell order having the greatest distance
   int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_PT);
   if(index_sell>WRONG_VALUE)
     {
      COrder* sell=list_sell.At(index_sell);
      if(sell!=NULL)
        {
         //--- If the order exceeds the price (SellLimit), and it needs to be "decreased" following the price
         if(sell.TypeOrder()==ORDER_TYPE_SELL_LIMIT)
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.bid+trailing_stop,Digits());
            double sl=(sell.StopLoss()>0 ? NormalizeDouble(price+(sell.StopLoss()-sell.PriceOpen()),Digits()) : 0);
            double tp=(sell.TakeProfit()>0 ? NormalizeDouble(price-(sell.PriceOpen()-sell.TakeProfit()),Digits()) : 0);
            //--- If the calculated price exceeds the StopLevel distance based on the Bid order price (the distance by StopLevel is maintained)
            if(price>tick.bid+stop_level)
              {
               //--- If the calculated price is below the trailing step based on the order price, modify the order price
               if(price<sell.PriceOpen()-trailing_step)
                 {
                  trade.OrderModify(sell.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),sell.PriceStopLimit());
                 }
              }
           }
         //--- If the order is below the price (SellStop and SellStopLimit), and it should be "elevated" following the price
         else
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.bid-trailing_stop,Digits());
            double sl=(sell.StopLoss()>0 ? NormalizeDouble(price+(sell.StopLoss()-sell.PriceOpen()),Digits()) : 0);
            double tp=(sell.TakeProfit()>0 ? NormalizeDouble(price-(sell.PriceOpen()-sell.TakeProfit()),Digits()) : 0);
            //--- If the calculated price is below the StopLevel distance based on the Bid order price (the distance by StopLevel is maintained)
            if(price<tick.bid-stop_level)
              {
               //--- If the calculated price exceeds the trailing step based on the order price, modify the order price
               if(price>sell.PriceOpen()+trailing_step)
                 {
                  trade.OrderModify(sell.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),(sell.PriceStopLimit()>0 ? price+distance_stoplimit*Point() : 0));
                 }
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The functions do not contain nothing new. All the necessary actions are described directly in the code comments. I believe, you will be able to
study the code on your own without much difficulty.

Since we now have three more buttons, the calculation of buttons' coordinates has been adjusted in the button panel creation function (see the
final listing).

Call all the trailing functions in the OnTick() handler:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Initialize the last trading event
   static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();
      PressButtonsControl();
     }
//--- If the last trading event changed
   if(engine.LastTradeEvent()!=last_event)
     {
      last_event=engine.LastTradeEvent();
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();
      TrailingOrders();
     }
  }
//+------------------------------------------------------------------+
```

**The full listing of the test EA:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart08.mq5 |
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
   BUTT_DELETE_PENDING,
   BUTT_CLOSE_ALL,
   BUTT_PROFIT_WITHDRAWAL,
   BUTT_SET_STOP_LOSS,
   BUTT_SET_TAKE_PROFIT,
   BUTT_TRAILING_ALL
  };
#define TOTAL_BUTT   (20)
//--- structures
struct SDataButt
  {
   string      name;
   string      text;
  };
//--- input variables
input ulong    InpMagic             =  123;  // Magic number
input double   InpLots              =  0.1;  // Lots
input uint     InpStopLoss          =  50;   // StopLoss in points
input uint     InpTakeProfit        =  50;   // TakeProfit in points
input uint     InpDistance          =  50;   // Pending orders distance (points)
input uint     InpDistanceSL        =  50;   // StopLimit orders distance (points)
input uint     InpSlippage          =  0;    // Slippage in points
input double   InpWithdrawal        =  10;   // Withdrawal funds (in tester)
input uint     InpButtShiftX        =  40;   // Buttons X shift
input uint     InpButtShiftY        =  10;   // Buttons Y shift
input uint     InpTrailingStop      =  50;   // Trailing Stop (points)
input uint     InpTrailingStep      =  20;   // Trailing Step (points)
input uint     InpTrailingStart     =  0;    // Trailing Start (points)
input uint     InpStopLossModify    =  20;   // StopLoss for modification (points)
input uint     InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
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
bool           trailing_on;
double         trailing_stop;
double         trailing_step;
uint           trailing_start;
uint           stoploss_to_modify;
uint           takeprofit_to_modify;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal,
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();
//--- check for undeleted objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);
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
   trailing_stop=InpTrailingStop*Point();
   trailing_step=InpTrailingStep*Point();
   trailing_start=InpTrailingStart;
   stoploss_to_modify=InpStopLossModify;
   takeprofit_to_modify=InpTakeProfitModify;
//--- create buttons
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- set button trailing
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- setting trade parameters
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
   Comment("");
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Initialize the last trading event
   static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
//--- If working in the tester
   if(MQLInfoInteger(MQL_TESTER))
     {
      engine.OnTimer();
      PressButtonsControl();
     }
//--- If the last trading event changed
   if(engine.LastTradeEvent()!=last_event)
     {
      last_event=engine.LastTradeEvent();
     }
//--- If the trailing flag is set
   if(trailing_on)
     {
      TrailingPositions();
      TrailingOrders();
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
   if(id>=CHARTEVENT_CUSTOM)
     {
      ushort event=ushort(id-CHARTEVENT_CUSTOM);
      Print(DFUN,"id=",id,", event=",EnumToString((ENUM_TRADE_EVENT)event),", lparam=",lparam,", dparam=",DoubleToString(dparam,Digits()),", sparam=",sparam);
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of a prefixed object presence                    |
//+------------------------------------------------------------------+
bool IsPresentObects(const string object_prefix)
  {
   for(int i=ObjectsTotal(0)-1;i>=0;i--)
      if(StringFind(ObjectName(0,i,0),object_prefix)>WRONG_VALUE)
         return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Tracking the buttons' status                                     |
//+------------------------------------------------------------------+
void PressButtonsControl(void)
  {
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
//| Create the button panel                                          |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=30,const int shift_y=0)
  {
   int h=18,w=84,offset=2;
   int cx=offset+shift_x,cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+3*h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-6) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-6 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
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
   if(name==butt_data[TOTAL_BUTT-1].name)
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'220,255,240');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
  }
//+------------------------------------------------------------------+
//| Transform enumeration into the button text                       |
//+------------------------------------------------------------------+
string EnumToButtText(const ENUM_BUTTONS member)
  {
   string txt=StringSubstr(EnumToString(member),5);
   StringToLower(txt);
   StringReplace(txt,"set_take_profit","Set TakeProfit");
   StringReplace(txt,"set_stop_loss","Set StopLoss");
   StringReplace(txt,"trailing_all","Trailing All");
   StringReplace(txt,"buy","Buy");
   StringReplace(txt,"sell","Sell");
   StringReplace(txt,"_limit"," Limit");
   StringReplace(txt,"_stop"," Stop");
   StringReplace(txt,"close_","Close ");
   StringReplace(txt,"2"," 1/2");
   StringReplace(txt,"_by_"," by ");
   StringReplace(txt,"profit_","Profit ");
   StringReplace(txt,"delete_","Delete ");
   return txt;
  }
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
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
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
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
         //--- Get correct order placement relative to StopLevel
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
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         trade.Sell(lot,Symbol(),0,sl,tp);
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- Get correct order placement relative to StopLevel
         double price_set=CorrectPricePending(Symbol(),ORDER_TYPE_SELL_LIMIT,distance_pending);
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL_LIMIT,price_set,takeprofit);
         //--- Set SellLimit order
         trade.SellLimit(lot,price_set,Symbol(),sl,tp);
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- Get correct order placement relative to StopLevel
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
         //--- Get correct StopLoss and TakeProfit prices relative to the order placement level considering StopLevel
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
               if(engine.IsHedge())
                  trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
               else
                  trade.Sell(NormalizeLot(position.Symbol(),position.Volume()/2.0));
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
               if(engine.IsHedge())
                  trade.PositionClosePartial(position.Ticket(),NormalizeLot(position.Symbol(),position.Volume()/2.0));
               else
                  trade.Buy(NormalizeLot(position.Symbol(),position.Volume()/2.0));
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
      //--- If the BUTT_DELETE_PENDING button is pressed: Remove the first pending order
      else if(button==EnumToString(BUTT_DELETE_PENDING))
        {
         //--- Get the list of all orders
         CArrayObj* list=engine.GetListMarketPendings();
         if(list!=NULL)
           {
            //--- Sort the list by placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
            int total=list.Total();
            //--- In the loop from the position with the most amount of time
            for(int i=total-1;i>=0;i--)
              {
               COrder* order=list.At(i);
               if(order==NULL)
                  continue;
               //--- delete the order by its ticket
               trade.OrderDelete(order.Ticket());
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
      //--- If the BUTT_SET_STOP_LOSS button is pressed: Place StopLoss to all orders and positions where it is not present
      if(button==EnumToString(BUTT_SET_STOP_LOSS))
        {
         SetStopLoss();
        }
      //--- If the BUTT_SET_TAKE_PROFIT button is pressed: Place TakeProfit to all orders and positions where it is not present
      if(button==EnumToString(BUTT_SET_TAKE_PROFIT))
        {
         SetTakeProfit();
        }
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button (if this is not a trailing button)
      if(button!=EnumToString(BUTT_TRAILING_ALL))
         ButtonState(button_name,false);
      //--- If the BUTT_TRAILING_ALL button is pressed
      else
        {
         //--- Set the color of the active button
         ButtonState(button_name,true);
         trailing_on=true;
        }
      //--- re-draw the chart
      ChartRedraw();
     }
   //--- Return the inactive button color (if this is a trailing button)
   else if(button==EnumToString(BUTT_TRAILING_ALL))
     {
      ButtonState(button_name,false);
      trailing_on=false;
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
//| Set StopLoss to all orders and positions                         |
//+------------------------------------------------------------------+
void SetStopLoss(void)
  {
   if(stoploss_to_modify==0)
      return;
//--- Set StopLoss to all positions where it is absent
   CArrayObj* list=engine.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SL,0,EQUAL);
   if(list==NULL)
      return;
   int total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* position=list.At(i);
      if(position==NULL)
         continue;
      double sl=CorrectStopLoss(position.Symbol(),position.TypeByDirection(),0,stoploss_to_modify);
      trade.PositionModify(position.Ticket(),sl,position.TakeProfit());
     }
//--- Set StopLoss to all pending orders where it is absent
   list=engine.GetListMarketPendings();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SL,0,EQUAL);
   if(list==NULL)
      return;
   total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* order=list.At(i);
      if(order==NULL)
         continue;
      double sl=CorrectStopLoss(order.Symbol(),(ENUM_ORDER_TYPE)order.TypeOrder(),order.PriceOpen(),stoploss_to_modify);
      trade.OrderModify(order.Ticket(),order.PriceOpen(),sl,order.TakeProfit(),trade.RequestTypeTime(),trade.RequestExpiration(),order.PriceStopLimit());
     }
  }
//+------------------------------------------------------------------+
//| Set TakeProfit to all orders and positions                       |
//+------------------------------------------------------------------+
void SetTakeProfit(void)
  {
   if(takeprofit_to_modify==0)
      return;
//--- Set TakeProfit to all positions where it is absent
   CArrayObj* list=engine.GetListMarketPosition();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TP,0,EQUAL);
   if(list==NULL)
      return;
   int total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* position=list.At(i);
      if(position==NULL)
         continue;
      double tp=CorrectTakeProfit(position.Symbol(),position.TypeByDirection(),0,takeprofit_to_modify);
      trade.PositionModify(position.Ticket(),position.StopLoss(),tp);
     }
//--- Set TakeProfit to all pending orders where it is absent
   list=engine.GetListMarketPendings();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TP,0,EQUAL);
   if(list==NULL)
      return;
   total=list.Total();
   for(int i=total-1;i>=0;i--)
     {
      COrder* order=list.At(i);
      if(order==NULL)
         continue;
      double tp=CorrectTakeProfit(order.Symbol(),(ENUM_ORDER_TYPE)order.TypeOrder(),order.PriceOpen(),takeprofit_to_modify);
      trade.OrderModify(order.Ticket(),order.PriceOpen(),order.StopLoss(),tp,trade.RequestTypeTime(),trade.RequestExpiration(),order.PriceStopLimit());
     }
  }
//+------------------------------------------------------------------+
//| Trailing stop of a position with the maximum profit              |
//+------------------------------------------------------------------+
void TrailingPositions(void)
  {
   MqlTick tick;
   if(!SymbolInfoTick(Symbol(),tick))
      return;
   double stop_level=StopLevel(Symbol(),2)*Point();
   //--- Get the list of all open positions
   CArrayObj* list=engine.GetListMarketPosition();
   //--- Select only Buy positions from the list
   CArrayObj* list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
   //--- Sort the list by profit considering commission and swap
   list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
   //--- Get the index of the Buy position with the maximum profit
   int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);
   if(index_buy>WRONG_VALUE)
     {
      COrder* buy=list_buy.At(index_buy);
      if(buy!=NULL)
        {
         //--- Calculate the new StopLoss
         double sl=NormalizeDouble(tick.bid-trailing_stop,Digits());
         //--- If the price and the StopLevel based on it are higher than the new StopLoss (the distance by StopLevel is maintained)
         if(tick.bid-stop_level>sl)
           {
            //--- If the new StopLoss level exceeds the trailing step based on the current StopLoss
            if(buy.StopLoss()+trailing_step<sl)
              {
               //--- If we trail at any profit or position profit in points exceeds the trailing start, modify StopLoss
               if(trailing_start==0 || buy.ProfitInPoints()>(int)trailing_start)
                  trade.PositionModify(buy.Ticket(),sl,buy.TakeProfit());
              }
           }
        }
     }
   //--- Select only Sell positions from the list
   CArrayObj* list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
   //--- Sort the list by profit considering commission and swap
   list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
   //--- Get Sell position index with the maximum profit
   int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);
   if(index_sell>WRONG_VALUE)
     {
      COrder* sell=list_sell.At(index_sell);
      if(sell!=NULL)
        {
         //--- Calculate the new StopLoss
         double sl=NormalizeDouble(tick.ask+trailing_stop,Digits());
         //--- If the price and StopLevel based on it are below the new (the distance by StopLevel is maintained)
         if(tick.ask+stop_level<sl)
           {
            //--- If the new StopLoss level is below the trailing step based on the current StopLoss or a position has no StopLoss
            if(sell.StopLoss()-trailing_step>sl || sell.StopLoss()==0)
              {
               //--- If we trail at any profit or position profit in points exceeds the trailing start value, modify StopLoss
               if(trailing_start==0 || sell.ProfitInPoints()>(int)trailing_start)
                  trade.PositionModify(sell.Ticket(),sl,sell.TakeProfit());
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Trailing the farthest pending orders                             |
//+------------------------------------------------------------------+
void TrailingOrders(void)
  {
   MqlTick tick;
   if(!SymbolInfoTick(Symbol(),tick))
      return;
   double stop_level=StopLevel(Symbol(),2)*Point();
//--- Get the list of all placed orders
   CArrayObj* list=engine.GetListMarketPendings();
//--- Select only Buy orders from the list
   CArrayObj* list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_BUY,EQUAL);
   //--- Sort the list by distance from the price in points (by profit in points)
   list_buy.Sort(SORT_BY_ORDER_PROFIT_PT);
   //--- Get the index of the Buy order with the greatest distance
   int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_PT);
   if(index_buy>WRONG_VALUE)
     {
      COrder* buy=list_buy.At(index_buy);
      if(buy!=NULL)
        {
         //--- If the order is below the price (BuyLimit) and it should be "elevated" following the price
         if(buy.TypeOrder()==ORDER_TYPE_BUY_LIMIT)
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.ask-trailing_stop,Digits());
            double sl=(buy.StopLoss()>0 ? NormalizeDouble(price-(buy.PriceOpen()-buy.StopLoss()),Digits()) : 0);
            double tp=(buy.TakeProfit()>0 ? NormalizeDouble(price+(buy.TakeProfit()-buy.PriceOpen()),Digits()) : 0);
            //--- If the calculated price is below the StopLevel distance based on Ask order price (the distance by StopLevel is maintained)
            if(price<tick.ask-stop_level)
              {
               //--- If the calculated price exceeds the trailing step based on the order placement price, modify the order price
               if(price>buy.PriceOpen()+trailing_step)
                 {
                  trade.OrderModify(buy.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),buy.PriceStopLimit());
                 }
              }
           }
         //--- If the order exceeds the price (BuyStop and BuyStopLimit), and it should be "decreased" following the price
         else
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.ask+trailing_stop,Digits());
            double sl=(buy.StopLoss()>0 ? NormalizeDouble(price-(buy.PriceOpen()-buy.StopLoss()),Digits()) : 0);
            double tp=(buy.TakeProfit()>0 ? NormalizeDouble(price+(buy.TakeProfit()-buy.PriceOpen()),Digits()) : 0);
            //--- If the calculated price exceeds the StopLevel based on Ask order price (the distance by StopLevel is maintained)
            if(price>tick.ask+stop_level)
              {
               //--- If the calculated price is lower than the trailing step based on order price, modify the order price
               if(price<buy.PriceOpen()-trailing_step)
                 {
                  trade.OrderModify(buy.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),(buy.PriceStopLimit()>0 ? price-distance_stoplimit*Point() : 0));
                 }
              }
           }
        }
     }
//--- Select only Sell order from the list
   CArrayObj* list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_SELL,EQUAL);
   //--- Sort the list by the distance from the price in points (by profit in points)
   list_sell.Sort(SORT_BY_ORDER_PROFIT_PT);
   //--- Get the index of the Sell order having the greatest distance
   int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_PT);
   if(index_sell>WRONG_VALUE)
     {
      COrder* sell=list_sell.At(index_sell);
      if(sell!=NULL)
        {
         //--- If the order exceeds the price (SellLimit), and it needs to be "decreased" following the price
         if(sell.TypeOrder()==ORDER_TYPE_SELL_LIMIT)
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.bid+trailing_stop,Digits());
            double sl=(sell.StopLoss()>0 ? NormalizeDouble(price+(sell.StopLoss()-sell.PriceOpen()),Digits()) : 0);
            double tp=(sell.TakeProfit()>0 ? NormalizeDouble(price-(sell.PriceOpen()-sell.TakeProfit()),Digits()) : 0);
            //--- If the calculated price exceeds the StopLevel distance based on the Bid order price (the distance by StopLevel is maintained)
            if(price>tick.bid+stop_level)
              {
               //--- If the calculated price is below the trailing step based on the order price, modify the order price
               if(price<sell.PriceOpen()-trailing_step)
                 {
                  trade.OrderModify(sell.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),sell.PriceStopLimit());
                 }
              }
           }
         //--- If the order is below the price (SellStop and SellStopLimit), and it should be "elevated" following the price
         else
           {
            //--- Calculate the new order price and stop levels based on it
            double price=NormalizeDouble(tick.bid-trailing_stop,Digits());
            double sl=(sell.StopLoss()>0 ? NormalizeDouble(price+(sell.StopLoss()-sell.PriceOpen()),Digits()) : 0);
            double tp=(sell.TakeProfit()>0 ? NormalizeDouble(price-(sell.PriceOpen()-sell.TakeProfit()),Digits()) : 0);
            //--- If the calculated price is below the StopLevel distance based on the Bid order price (the distance by StopLevel is maintained)
            if(price<tick.bid-stop_level)
              {
               //--- If the calculated price exceeds the trailing step based on the order price, modify the order price
               if(price>sell.PriceOpen()+trailing_step)
                 {
                  trade.OrderModify(sell.Ticket(),price,sl,tp,trade.RequestTypeTime(),trade.RequestExpiration(),(sell.PriceStopLimit()>0 ? price+distance_stoplimit*Point() : 0));
                 }
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

Let's compile the EA.

Set **StopLoss in points** and **TaleProfit in points** values to zero to open positions
and place pending orders without stop levels. Set

**StopLoss for modification (points)** and **TakeProfit for modification (points)** to **20** and **60** accordingly
(the default values) — these StopLoss and TakeProfit levels are to be set by pressing the buttons.

Launch the EA in the tester and set pending orders. Then press the buttons for setting StopLoss and TakeProfit one after another. The
levels are set and the appropriate entries appear in the journal. Next, enable trailing and observe the orders as they follow the price and
the appropriate entries appear in the journal. Positions triggered by orders have their StopLoss levels trailed, and the appropriate
entries appear in the journal.

Netting:

![](https://c.mql5.com/2/36/2019-04-29_22-44-41.gif)

Hedging:

![](https://c.mql5.com/2/36/2019-04-29_22-51-12.gif)

### What's next?

In the coming articles, we will expand the library and implement its compatibility with MQL4. More exciting things are yet to come.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6595#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6595](https://www.mql5.com/ru/articles/6595)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6595.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6595/mql5.zip "Download MQL5.zip")(93.77 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/318528)**
(34)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
3 Jul 2022 at 23:41

**MQL\_User [#](https://www.mql5.com/ru/forum/313062/page3#comment_40563070):**

The spread in the tester is 1. Both on PC and laptop. Symbol @Si.

Can this have any effect?

I thought maybe the slippage (slippage) was affecting it somehow and tried changing it. But it doesn't work.

In one of the cases, the price does not reach the stop level.

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
4 Jul 2022 at 20:24

Artyom.

Screen attached.

I.e. it passes through sl and tp levels as if they are not there at all.

Today I tried it on another computer - everything is normal, sl and tp trigger without problems. It turns out that only my PC has these problems. I also tried it on my PC, but on a [virtual machine](https://www.mql5.com/en/vps "Forex VPS for MetaTrader 4/5") \- everything is fine too (same code everywhere).

In general, the situation is unclear....

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
4 Jul 2022 at 20:30

**MQL\_User virtual machine - everything is fine too (same code everywhere).**

**In general, the situation is unclear....**

Good evening, "confusion" can still occur when a 32-bit version of Windows is installed on a personal computer.

Regards, Vladimir.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
4 Jul 2022 at 21:23

**MQL\_User virtual machine - everything is fine too (same code everywhere).**

**In general, the situation is unclear....**

And what is written in the "Experts" log?

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
5 Jul 2022 at 19:55

Vladimir, Artem.

My PC has a 64-bit version of Windows10 installed, as well as others on which sl and tp trigger normally. Nothing is written in the "Experts" log. But if I do it in the tester, I understand that nothing should be written there.

But what is interesting is that if I switch on trailing all then tp starts triggering. I assumed that maybe trade.PositionModify(...) has some effect? But I tried in my Expert Advisor to modify it immediately after [opening a position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "MetaTrader 5 Help: Opening and Closing Positions in the MetaTrader 5 Trading Terminal") (slightly change sl and tp) - it didn't help.

Maybe Windows settings somehow affect it (it works fine on other computers)?

![Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://c.mql5.com/2/36/mql5_ea_adviser_grid.png)[Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)

In this article, we will develop a grider EA for trading in a trend direction within a range. Thus, the EA is to be suited mostly for Forex and commodity markets. According to the tests, our grider showed profit since 2018. Unfortunately, this is not true for the period of 2014-2018.

![Grokking market "memory" through differentiation and entropy analysis](https://c.mql5.com/2/36/snip_20190614154924__2.png)[Grokking market "memory" through differentiation and entropy analysis](https://www.mql5.com/en/articles/6351)

The scope of use of fractional differentiation is wide enough. For example, a differentiated series is usually input into machine learning algorithms. The problem is that it is necessary to display new data in accordance with the available history, which the machine learning model can recognize. In this article we will consider an original approach to time series differentiation. The article additionally contains an example of a self optimizing trading system based on a received differentiated series.

![Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://c.mql5.com/2/36/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the eighth part, we implemented the class for tracking order and position modification events. Here, we will improve the library by making it fully compatible with MQL4.

![Price velocity measurement methods](https://c.mql5.com/2/36/Article_Logo__1.png)[Price velocity measurement methods](https://www.mql5.com/en/articles/6947)

There are multiple different approaches to market research and analysis. The main ones are technical and fundamental. In technical analysis, traders collect, process and analyze numerical data and parameters related to the market, including prices, volumes, etc. In fundamental analysis, traders analyze events and news affecting the markets directly or indirectly. The article deals with price velocity measurement methods and studies trading strategies based on that methods.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/6595&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070487134143452859)

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