---
title: Library for easy and quick development of MetaTrader programs (part VI): Netting account events
url: https://www.mql5.com/en/articles/6383
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:41:59.579270
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/6383&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070491768413165266)

MetaTrader 5 / Examples


### Contents

- [Similarities and differences of the account types](https://www.mql5.com/en/articles/6383#node01)
- [Implementing event handling on a netting account](https://www.mql5.com/en/articles/6383#node02)
- [Testing the performance on hedging and netting accounts](https://www.mql5.com/en/articles/6383#node03)
- [What's next?](https://www.mql5.com/en/articles/6383#node04)


### Similarities and differences of the account types

To track netting account events, we need to understand the differences between hedging and netting accounts.

The differences relate to representation of positions. A hedging account allows us to open any number of positions on a single symbol, while a
netting account allows for only one. A hedging account allows closing a position by a volume of an oppositely directed position.

In this case:

- If the volume of the opposite position is smaller than the volume of the closed position, then the opposite position is closed
completely, while the closed position is eliminated partially,
- If the volume of the opposite position is higher than the volume of the closed position, then the opposite position is closed
partially, while the closed position is eliminated in full,
- if the two positions have equal volumes, both are closed;
- Each position has an ID equal to the opening order's ticket. This ID does not change during the entire position lifetime;
- Each position has its own ticket equal to the ticket of the order which led to the position opening;
- If we send a request to open a new position in the current position direction, a new position with a new ID and ticket is opened.

On a netting account, working with one position on one symbol excludes the possibility of closing a position by an opposite one. However, in a
remotely similar situation (when an oppositely directed order is activated), this position can be closed either partially or in full, or
change its direction:

- If the volume of an activated opposite order is less than the current position one, the position is closed partially,
- If the volume of an activated opposite order is equal to the current position one, the position is closed in full,
- If the volume of an activated opposite order exceeds the current position one, the position changes its direction (reversal),
- Each position has an ID equal to the opening order's ticket. This ID does not change during the entire position lifetime;
- Each position has a ticket equal to the ticket of the order which led to the position reversal. The ticket may differ from the ID. To some
extent, it repeats the tickets of multiple positions on a hedge account;
- If we send a request to open a new position in the current position direction, a volume of an activated order is added to the volume of the
current position. The position ticket is not changed.




### Implementing event handling on a netting account

To track netting account events, we will simply divide handling position events by account types. This increases the amount of code but
clarifies the logic due to separating the functionality. We will optimize the code and get rid of all redundancies later — after debugging
and confirming steady operation.

When adding new constants to the event type enumerations, I noticed that sorting sometimes works incorrectly. Checking the reasons for such a
behavior revealed that the two main factors are event/order properties matching the sorting by this type and their location in the
enumeration regardless of the fact that each constant is numbered. For example, if a property is not used for searching, it should be skipped
and correct numbers should be assigned to search method enumeration constants. Additionally, event/order properties not used in sorting
should also be placed at the end of the property type list. To calculate the initial number of the following property types, the number of
unused properties within the amount of previous type properties should be subtracted from the initial property types index.

**To verify the creation of sorting method enumerations, a small function was added to the DELib.mqh service functions file:**

```
//+------------------------------------------------------------------+
//| Display all sorting enumeration constants in the journal         |
//+------------------------------------------------------------------+
void EnumNumbersTest()
  {
   string enm="ENUM_SORT_ORDERS_MODE";
   string t=StringSubstr(enm,5,5)+"BY";
   Print("Search of the values of the enumaration ",enm,":");
   ENUM_SORT_ORDERS_MODE type=0;
   while(StringFind(EnumToString(type),t)==0)
     {
      Print(enm,"[",type,"]=",EnumToString(type));
      if(type>500) break;
      type++;
     }
   Print("\nNumber of members of the ",enm,"=",type);
  }
//+------------------------------------------------------------------+
```

To check the contents of a specific sorting types enumeration, it should be entered manually in
two function strings (I did not find a way to automatically
set a certain enumeration in the

ENUM\_SORT\_ORDERS\_MODE type=0; string).

Now, if we call this function in the test EA's OnInit() handler, all constant names of a specified enumeration and the corresponding indices
are displayed in the journal.

When checking the enumerations, I detected that they were created incorrectly. To fix this, I slightly modified the enumerations in the **Defines.mqh**
file.

I set a different order of constants in the enumerations — properties
not used in sorting are placed at the end of the constant list for object properties enumeration. Also, macro
substitutions for specifying the number of unused properties for searching and sorting should be added. These macro
substitutions are to be

used when calculating initial property indices in sorting enumerations
leading to calculation of correct indices of initial constants in enumerations.

Also, new constant types for events on netting accounts
and

constants for storing the magic number and an opposite position symbol for hedge accounts
should be added.

Orders and positions are often required to be grouped, so that a group of selected orders and positions can be handled simultaneously. The
library allows implementing this by simply adding a group ID to the abstract order property. This makes it possible to arrange any orders and
positions having a similar ID into a single list and work with a selected group.

Such an ID was added to order properties and order sorting lists.

Below is a **full listing of the modified Defines.mqh:**

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- "Description of a function with the error string number"
#define DFUN_ERR_LINE            (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                     (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG             ("Russian")                // Country language
#define END_TIME                 (D'31.12.3000 23:59:59')   // End date for requesting account history data
#define TIMER_FREQUENCY          (16)                       // Minimal frequency of the library timer in milliseconds
#define COLLECTION_PAUSE         (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_COUNTER_STEP  (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_COUNTER_ID    (1)                        // Orders and deals collection timer counter ID
#define COLLECTION_HISTORY_ID    (0x7778+1)                 // Historical collection list ID
#define COLLECTION_MARKET_ID     (0x7778+2)                 // Market collection list ID
#define COLLECTION_EVENTS_ID     (0x7778+3)                 // Events collection list ID
//+------------------------------------------------------------------+
//| Structures                                                       |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Search and sorting data                                          |
//+------------------------------------------------------------------+
enum ENUM_COMPARER_TYPE
  {
   EQUAL,                                                   // Equal
   MORE,                                                    // More
   LESS,                                                    // Less
   NO_EQUAL,                                                // Not equal
   EQUAL_OR_MORE,                                           // Equal or more
   EQUAL_OR_LESS                                            // Equal or less
  };
//+------------------------------------------------------------------+
//| Possible options of sorting by time                              |
//+------------------------------------------------------------------+
enum ENUM_SELECT_BY_TIME
  {
   SELECT_BY_TIME_OPEN,                                     // By open time
   SELECT_BY_TIME_CLOSE,                                    // By close time
   SELECT_BY_TIME_OPEN_MSC,                                 // By open time in milliseconds
   SELECT_BY_TIME_CLOSE_MSC,                                // By close time in milliseconds
  };
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Data for working with orders                                     |
//+------------------------------------------------------------------+
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
   ORDER_PROP_STATE,                                        // Order state (from the ENUM_ORDER_STATE enumeration)
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
   ORDER_PROP_DIRECTION,                                    // Direction-based type (Buy, Sell)
  };
#define ORDER_PROP_INTEGER_TOTAL    (24)                    // Total number of integer properties
#define ORDER_PROP_INTEGER_SKIP     (1)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
//| Order, deal, position real properties                            |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_DOUBLE
  {
   ORDER_PROP_PRICE_OPEN = ORDER_PROP_INTEGER_TOTAL,        // Open price (MQL5 deal price)
   ORDER_PROP_PRICE_CLOSE,                                  // Close price
   ORDER_PROP_SL,                                           // StopLoss price
   ORDER_PROP_TP,                                           // TakeProfit price
   ORDER_PROP_PROFIT,                                       // Profit
   ORDER_PROP_COMMISSION,                                   // Commission
   ORDER_PROP_SWAP,                                         // Swap
   ORDER_PROP_VOLUME,                                       // Volume
   ORDER_PROP_VOLUME_CURRENT,                               // Unexecuted volume
   ORDER_PROP_PROFIT_FULL,                                  // Profit+commission+swap
   ORDER_PROP_PRICE_STOP_LIMIT,                             // Limit order price when StopLimit order is activated
  };
#define ORDER_PROP_DOUBLE_TOTAL     (11)                    // Total number of real properties
//+------------------------------------------------------------------+
//| Order, deal, position string properties                          |
//+------------------------------------------------------------------+
enum ENUM_ORDER_PROP_STRING
  {
   ORDER_PROP_SYMBOL = (ORDER_PROP_INTEGER_TOTAL+ORDER_PROP_DOUBLE_TOTAL), // Order symbol
   ORDER_PROP_COMMENT,                                      // Order comment
   ORDER_PROP_EXT_ID                                        // Order ID in an external trading system
  };
#define ORDER_PROP_STRING_TOTAL     (3)                     // Total number of string properties
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
   SORT_BY_ORDER_STATUS          =  7,                      // Sort by order status (market order/pending order/deal/balance credit operation)
   SORT_BY_ORDER_TYPE            =  8,                      // Sort by order type
   SORT_BY_ORDER_REASON          =  9,                      // Sort by deal/order/position reason/source
   SORT_BY_ORDER_STATE           =  10,                     // Sort by order state
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
//| Data for working with account events                             |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| List of account trading event flags                              |
//+------------------------------------------------------------------+
enum ENUM_TRADE_EVENT_FLAGS
  {
   TRADE_EVENT_FLAG_NO_EVENT        =  0,                   // No event
   TRADE_EVENT_FLAG_ORDER_PLASED    =  1,                   // Pending order placed
   TRADE_EVENT_FLAG_ORDER_REMOVED   =  2,                   // Pending order removed
   TRADE_EVENT_FLAG_ORDER_ACTIVATED =  4,                   // Pending order activated by price
   TRADE_EVENT_FLAG_POSITION_OPENED =  8,                   // Position opened
   TRADE_EVENT_FLAG_POSITION_CHANGED=  16,                  // Position changed
   TRADE_EVENT_FLAG_POSITION_REVERSE=  32,                  // Position reversal
   TRADE_EVENT_FLAG_POSITION_CLOSED =  64,                  // Position closed
   TRADE_EVENT_FLAG_ACCOUNT_BALANCE =  128,                 // Balance operation (clarified by a deal type)
   TRADE_EVENT_FLAG_PARTIAL         =  256,                 // Partial execution
   TRADE_EVENT_FLAG_BY_POS          =  512,                 // Executed by opposite position
   TRADE_EVENT_FLAG_SL              =  1024,                // Executed by StopLoss
   TRADE_EVENT_FLAG_TP              =  2048                 // Executed by TakeProfit
  };
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
   TRADE_EVENT_ACCOUNT_CREDIT = DEAL_TYPE_CREDIT,           // Charging credit (3)
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
   TRADE_EVENT_TAX                        = DEAL_TAX,       // Tax accrual
//--- constants related to the DEAL_TYPE_BALANCE deal type from the ENUM_DEAL_TYPE enumeration
   TRADE_EVENT_ACCOUNT_BALANCE_REFILL     = DEAL_TAX+1,     // Replenishing account balance
   TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL = DEAL_TAX+2,     // Withdrawing funds from an account
//--- Remaining possible trading events
//--- (constant order below can be changed, constants can be added/deleted)
   TRADE_EVENT_PENDING_ORDER_ACTIVATED    = DEAL_TAX+3,     // Pending order activated by price
   TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL,             // Pending order partially activated by price
   TRADE_EVENT_POSITION_OPENED,                             // Position opened
   TRADE_EVENT_POSITION_OPENED_PARTIAL,                     // Position opened partially
   TRADE_EVENT_POSITION_CLOSED,                             // Position closed
   TRADE_EVENT_POSITION_CLOSED_BY_POS,                      // Position closed by an opposite one
   TRADE_EVENT_POSITION_CLOSED_BY_SL,                       // Position closed by StopLoss
   TRADE_EVENT_POSITION_CLOSED_BY_TP,                       // Position closed by TakeProfit
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET,                 // Position reversal by a new deal (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING,                // Position reversal by activating a pending order (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL,         // Position reversal by partial market order execution (netting)
   TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL,        // Position reversal by partial pending order activation (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET,               // Added volume to a position by a new deal (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL,       // Added volume to a position by partial execution of a market order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING,              // Added volume to a position by activating a pending order (netting)
   TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL,      // Added volume to a position by partial activation of a pending order (netting)
   TRADE_EVENT_POSITION_CLOSED_PARTIAL,                     // Position closed partially
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS,              // Position closed partially by an opposite one
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL,               // Position closed partially by StopLoss
   TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP                // Position closed partially by TakeProfit
  };
//+------------------------------------------------------------------+
//| Event status                                                     |
//+------------------------------------------------------------------+
enum ENUM_EVENT_STATUS
  {
   EVENT_STATUS_MARKET_POSITION,                            // Market position event (opening, partial opening, partial closing, adding volume, reversal
   EVENT_STATUS_MARKET_PENDING,                             // Market pending order event (placing)
   EVENT_STATUS_HISTORY_PENDING,                            // Historical pending order event (removal)
   EVENT_STATUS_HISTORY_POSITION,                           // Historical position event (closing)
   EVENT_STATUS_BALANCE,                                    // Balance operation event (accruing balance, withdrawing funds and events from the ENUM_DEAL_TYPE enumeration)
  };
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
//+------------------------------------------------------------------+
//| Event's integer properties                                       |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_INTEGER
  {
   EVENT_PROP_TYPE_EVENT = 0,                               // Account trading event type (from the ENUM_TRADE_EVENT enumeration)
   EVENT_PROP_TIME_EVENT,                                   // Event time in milliseconds
   EVENT_PROP_STATUS_EVENT,                                 // Event status (from the ENUM_EVENT_STATUS enumeration)
   EVENT_PROP_REASON_EVENT,                                 // Event reason (from the ENUM_EVENT_REASON enumeration)
   //---
   EVENT_PROP_TYPE_DEAL_EVENT,                              // Deal event type
   EVENT_PROP_TICKET_DEAL_EVENT,                            // Deal event ticket
   EVENT_PROP_TYPE_ORDER_EVENT,                             // Type of the order, based on which a deal event is opened (the last position order)
   EVENT_PROP_TICKET_ORDER_EVENT,                           // Ticket of the order, based on which a deal event is opened (the last position order)
   //---
   EVENT_PROP_TIME_ORDER_POSITION,                          // Time of the order, based on which the first position deal is opened (the first position order on a hedge account)
   EVENT_PROP_TYPE_ORDER_POSITION,                          // Type of the order, based on which the first position deal is opened (the first position order on a hedge account)
   EVENT_PROP_TICKET_ORDER_POSITION,                        // Ticket of the order, based on which the first position deal is opened (the first position order on a hedge account)
   EVENT_PROP_POSITION_ID,                                  // Position ID
   //---
   EVENT_PROP_POSITION_BY_ID,                               // Opposite position ID
   EVENT_PROP_MAGIC_ORDER,                                  // Order/deal/position magic number
   EVENT_PROP_MAGIC_BY_ID,                                  // Opposite position magic number
   //---
   EVENT_PROP_TYPE_ORD_POS_BEFORE,                          // Position type before direction changed
   EVENT_PROP_TICKET_ORD_POS_BEFORE,                        // Position order ticket before direction changed
   EVENT_PROP_TYPE_ORD_POS_CURRENT,                         // Current position type
   EVENT_PROP_TICKET_ORD_POS_CURRENT                        // Current position order ticket
  };
#define EVENT_PROP_INTEGER_TOTAL (19)                       // Total number of integer event properties
#define EVENT_PROP_INTEGER_SKIP  (4)                        // Number of event properties not used in sorting event properties
//+------------------------------------------------------------------+
//| Event's real properties                                          |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_DOUBLE
  {
   EVENT_PROP_PRICE_EVENT = EVENT_PROP_INTEGER_TOTAL,       // Price an event occurred at
   EVENT_PROP_PRICE_OPEN,                                   // Order/deal/position open price
   EVENT_PROP_PRICE_CLOSE,                                  // Order/deal/position close price
   EVENT_PROP_PRICE_SL,                                     // StopLoss order/deal/position price
   EVENT_PROP_PRICE_TP,                                     // TakeProfit order/deal/position price
   EVENT_PROP_VOLUME_ORDER_INITIAL,                         // Requested order volume
   EVENT_PROP_VOLUME_ORDER_EXECUTED,                        // Executed order volume
   EVENT_PROP_VOLUME_ORDER_CURRENT,                         // Remaining order volume
   EVENT_PROP_VOLUME_POSITION_EXECUTED,                     // Current executed position volume after a deal
   EVENT_PROP_PROFIT                                        // Profit
  };
#define EVENT_PROP_DOUBLE_TOTAL  (10)                       // Total number of event's real properties
//+------------------------------------------------------------------+
//| Event's string properties                                        |
//+------------------------------------------------------------------+
enum ENUM_EVENT_PROP_STRING
  {
   EVENT_PROP_SYMBOL = (EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_DOUBLE_TOTAL), // Order symbol
   EVENT_PROP_SYMBOL_BY_ID                                  // Opposite position symbol
  };
#define EVENT_PROP_STRING_TOTAL     (2)                     // Total number of event's string properties
//+------------------------------------------------------------------+
//| Possible event sorting criteria                                  |
//+------------------------------------------------------------------+
#define FIRST_EVN_DBL_PROP       (EVENT_PROP_INTEGER_TOTAL-EVENT_PROP_INTEGER_SKIP)
#define FIRST_EVN_STR_PROP       (EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_DOUBLE_TOTAL-EVENT_PROP_INTEGER_SKIP)
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
   SORT_BY_EVENT_TICKET_ORDER_EVENT       = 7,                       // Sort by a ticket of the order, based on which a deal event is opened (the last position order)
   SORT_BY_EVENT_TIME_ORDER_POSITION      = 8,                       // Sort by time of the order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TYPE_ORDER_POSITION      = 9,                       // Sort by type of the order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_TICKET_ORDER_POSITION    = 10,                      // Sort by a ticket of the order, based on which a position deal is opened (the first position order)
   SORT_BY_EVENT_POSITION_ID              = 11,                      // Sort by position ID
   SORT_BY_EVENT_POSITION_BY_ID           = 12,                      // Sort by opposite position ID
   SORT_BY_EVENT_MAGIC_ORDER              = 13,                      // Sort by order/deal/position magic number
   SORT_BY_EVENT_MAGIC_BY_ID              = 14,                      // Sort by an opposite position magic number
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

Since we have added the order group ID to the article subject, the abstract order object should be changed as well. Let's add the method
returning a group ID assigned to an orders and the method setting a
group ID's value:

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

```

```
   //--- Return (1) open price, (2) close price, (3) profit, (4) commission, (5) swap, (6) volume,
   //--- (7) unexecuted volume (8) StopLoss and (9) TakeProfit (10) StopLimit order price
   double            PriceOpen(void)                                    const { return this.GetProperty(ORDER_PROP_PRICE_OPEN);                 }
   double            PriceClose(void)                                   const { return this.GetProperty(ORDER_PROP_PRICE_CLOSE);                }
   double            Profit(void)                                       const { return this.GetProperty(ORDER_PROP_PROFIT);                     }
   double            Comission(void)                                    const { return this.GetProperty(ORDER_PROP_COMMISSION);                 }
   double            Swap(void)                                         const { return this.GetProperty(ORDER_PROP_SWAP);                       }
   double            Volume(void)                                       const { return this.GetProperty(ORDER_PROP_VOLUME);                     }
   double            VolumeCurrent(void)                                const { return this.GetProperty(ORDER_PROP_VOLUME_CURRENT);             }
   double            StopLoss(void)                                     const { return this.GetProperty(ORDER_PROP_SL);                         }
   double            TakeProfit(void)                                   const { return this.GetProperty(ORDER_PROP_TP);                         }
   double            PriceStopLimit(void)                               const { return this.GetProperty(ORDER_PROP_PRICE_STOP_LIMIT);           }

   //--- Return (1) symbol, (2) comment, (3) ID at an exchange
   string            Symbol(void)                                       const { return this.GetProperty(ORDER_PROP_SYMBOL);                     }
   string            Comment(void)                                      const { return this.GetProperty(ORDER_PROP_COMMENT);                    }
   string            ExternalID(void)                                   const { return this.GetProperty(ORDER_PROP_EXT_ID);                     }

   //--- Get the full order profit
   double            ProfitFull(void)                                   const { return this.Profit()+this.Comission()+this.Swap();              }
   //--- Get order profit in points
   int               ProfitInPoints(void) const;
//--- Set group ID
   void              SetGroupID(long group_id)                                { this.SetProperty(ORDER_PROP_GROUP_ID,group_id);                 }

```

The group ID is to be set to zero by default. To achieve this, set
the order property value to zero in the closed constructor of the COrder class:

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
   this.m_long_prop[ORDER_PROP_TIME_OPEN]                            = (long)(ulong)this.OrderOpenTime();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE]                           = (long)(ulong)this.OrderCloseTime();
   this.m_long_prop[ORDER_PROP_TIME_EXP]                             = (long)(ulong)this.OrderExpiration();
   this.m_long_prop[ORDER_PROP_TYPE]                                 = this.OrderType();
   this.m_long_prop[ORDER_PROP_STATE]                                = this.OrderState();
   this.m_long_prop[ORDER_PROP_DIRECTION]                            = this.OrderTypeByDirection();
   this.m_long_prop[ORDER_PROP_POSITION_ID]                          = this.OrderPositionID();
   this.m_long_prop[ORDER_PROP_REASON]                               = this.OrderReason();
   this.m_long_prop[ORDER_PROP_DEAL_ORDER_TICKET]                    = this.DealOrderTicket();
   this.m_long_prop[ORDER_PROP_DEAL_ENTRY]                           = this.DealEntry();
   this.m_long_prop[ORDER_PROP_POSITION_BY_ID]                       = this.OrderPositionByID();
   this.m_long_prop[ORDER_PROP_TIME_OPEN_MSC]                        = this.OrderOpenTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_CLOSE_MSC]                       = this.OrderCloseTimeMSC();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE]                          = (long)(ulong)this.PositionTimeUpdate();
   this.m_long_prop[ORDER_PROP_TIME_UPDATE_MSC]                      = (long)(ulong)this.PositionTimeUpdateMSC();

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
  }
//+------------------------------------------------------------------+
```

Add description of the group ID to the method returning the
property description:

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
      property==ORDER_PROP_PROFIT_PT         ?  TextByLanguage("Прибыль в пунктах","Profit in points")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(string)this.GetProperty(property)
         )  :
      property==ORDER_PROP_CLOSE_BY_SL       ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property) ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
         )  :
      property==ORDER_PROP_CLOSE_BY_TP       ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit")+
         (!this.SupportProperty(property)    ?  TextByLanguage(": Свойство не поддерживается",": Property not supported") :
          ": "+(this.GetProperty(property)   ? TextByLanguage("Да","Yes") : TextByLanguage("Нет","No"))
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

After these changes, you can assign a group ID to any order/position, thus arranging orders and positions into certain groups to work with a
specific group only. 0 is assigned by default to all newly opened positions and set orders. However, you are able to assign another group
to any order/position using the

**SetGroupID(group\_index)** method. Besides, you can find out any order's group using the **GroupID()**
method.

**Let's return to implementing event tracking on a netting account.**

In order to divide the functionality by account
types, we are going to add a class member variable to the private section of the CEvent abstract event class in the

**Event.mqh** file:

```
protected:
   ENUM_TRADE_EVENT  m_trade_event;                                  // Trading event
   bool              m_is_hedge;                                     // Hedge account flag
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }
```

Add declaration of methods returning

(for a hedge account) magic
number and a symbol of an opposite
position,

(for considering a position reversal on a netting account) previous
position's order type and ticket, current position's order ticket,

position type and ticket before changing direction, position
type and ticket after changing direction to the list of methods with simplified access of the class' public section:

```
//+------------------------------------------------------------------+
//| Methods of simplified access to event object properties          |
//+------------------------------------------------------------------+
//--- Return (1) event type, (2) event time in milliseconds, (3) event status, (4) event reason, (5) deal type, (6) deal ticket,
//---(7) order type, based on which a deal was executed, (8) position opening order type, (9) position last order ticket,
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

//--- When changing position direction, return (1) previous position order type, (2) previous position order ticket,
//--- (3) current position order type, (4) current position order ticket,
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
//--- Return the (1) symbol and (2) opposite position symbol
   string            Symbol(void)                                       const { return this.GetProperty(EVENT_PROP_SYMBOL);                                 }
   string            SymbolCloseBy(void)                                const { return this.GetProperty(EVENT_PROP_SYMBOL_BY_ID);                           }

//+------------------------------------------------------------------+
```

The methods are simple: the appropriate event property is returned for orders, a ticket of the order which opened or changed a position is
returned for tickets, while a position type (by type of the order, which triggered it) is returned for type name using the previously
described

**PositionTypeByOrderType()** function from the **DELib.mqh** service function file.

Set saving data on the account type in the class constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   this.m_digits_acc=(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
```

Add the definition of methods returning a name of the order a deal event has
happened for, the very first (opening) position order, an
order, which led to opening (netting, hedging) or changing (netting) the current position, name
of the current position type, type of the order which led to opening the
previous position and the previous position type name
to the event property description methods:

```
//+------------------------------------------------------------------+
//| Descriptions of order object properties                          |
//+------------------------------------------------------------------+
//--- Get description of an order's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_STRING property);
//--- Return the event's (1) status and (2) type
   string            StatusDescription(void)                const;
   string            TypeEventDescription(void)             const;
//--- Return the name of an (1) event deal order, (2) position's parent order, (3) current position order and the (4) current position
//--- Return the name of an (5) order and (6) position before the direction was changed
   string            TypeOrderDealDescription(void)         const;
   string            TypeOrderFirstDescription(void)        const;
   string            TypeOrderEventDescription(void)        const;
   string            TypePositionCurrentDescription(void)   const;
   string            TypeOrderPreviousDescription(void)     const;
   string            TypePositionPreviousDescription(void)  const;
//--- Return the name of the deal/order/position reason
   string            ReasonDescription(void)                const;
```

and **their implementation** beyond the class body as well:

```
//+------------------------------------------------------------------+
//| Return the name of the order/position/deal                       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderDealDescription(void) const
  {
   ENUM_EVENT_STATUS status=this.Status();
   return
     (
      status==EVENT_STATUS_MARKET_PENDING  || status==EVENT_STATUS_HISTORY_PENDING  ?  OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT))      :
      status==EVENT_STATUS_MARKET_POSITION || status==EVENT_STATUS_HISTORY_POSITION ?  PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT)) :
      status==EVENT_STATUS_BALANCE  ?  DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT))  :
      TextByLanguage("Неизвестный тип ордера","Unknown order type")
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the position's first order                    |
//+------------------------------------------------------------------+
string CEvent::TypeOrderFirstDescription(void) const
  {
   return OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
  }
//+------------------------------------------------------------------+
//| Return the name of the order that changed the position           |
//+------------------------------------------------------------------+
string CEvent::TypeOrderEventDescription(void) const
  {
   return OrderTypeDescription(this.TypeOrderEvent());
  }
//+------------------------------------------------------------------+
//| Return the name of the current position                          |
//+------------------------------------------------------------------+
string CEvent::TypePositionCurrentDescription(void) const
  {
   return PositionTypeDescription(this.TypePositionCurrent());
  }
//+------------------------------------------------------------------+
//| Return the name of the order before changing the direction       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderPreviousDescription(void) const
  {
   return OrderTypeDescription(this.TypeOrderPosPrevious());
  }
//+------------------------------------------------------------------+
//| Return the name of the position before changing the direction    |
//+------------------------------------------------------------------+
string CEvent::TypePositionPreviousDescription(void) const
  {
   return PositionTypeDescription(this.TypePositionPrevious());
  }
//+------------------------------------------------------------------+
```

These methods are simple, just like the ones returning order and position types. The only difference is in the functions from the DELib.mqh file
returning the type of orders and positions as their type description:

**PositionTypeDescription()** and **OrderTypeDescription()**.

Now, the **ReasonDescription()** method should be improved
for considering and returning descriptions of newly added enumerations related to event reasons for a netting account:

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
      reason==EVENT_REASON_CANCEL                           ?  TextByLanguage("Отмена","Canceled")                                                                :
      reason==EVENT_REASON_EXPIRED                          ?  TextByLanguage("Истёк срок действия","Expired")                                                    :
      reason==EVENT_REASON_DONE                             ?  TextByLanguage("Рыночный запрос, выполненный в полном объёме","Fully completed market request")    :
      reason==EVENT_REASON_DONE_PARTIALLY                   ?  TextByLanguage("Выполненный частично рыночный запрос","Partially completed market request")        :
      reason==EVENT_REASON_VOLUME_ADD                       ?  TextByLanguage("Добавлен объём к позиции","Added volume to position")                              :
      reason==EVENT_REASON_VOLUME_ADD_PARTIALLY             ?  TextByLanguage("Добавлен объём к позиции частичным исполнением заявки","Volume added to position by partially completed request")                 :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING            ?  TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by triggered pending order")                      :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING_PARTIALLY  ?  TextByLanguage("Добавлен объём к позиции частичной активацией отложенного ордера","Added volume to position by partially triggered pending order")  :
      reason==EVENT_REASON_REVERSE                          ?  TextByLanguage("Разворот позиции","Position reversal")  :
      reason==EVENT_REASON_REVERSE_PARTIALLY                ?  TextByLanguage("Разворот позиции частичным исполнением заявки","Position reversal by partially completing request")                             :
      reason==EVENT_REASON_REVERSE_BY_PENDING               ?  TextByLanguage("Разворот позиции при срабатывании отложенного ордера","Position reversal on a triggered pending order")                               :
      reason==EVENT_REASON_REVERSE_BY_PENDING_PARTIALLY     ?  TextByLanguage("Разворот позиции при при частичном срабатывании отложенного ордера","Position reversal on a partially triggered pending order")       :
      reason==EVENT_REASON_DONE_SL                          ?  TextByLanguage("Закрытие по StopLoss","Close by StopLoss triggered")                               :
      reason==EVENT_REASON_DONE_SL_PARTIALLY                ?  TextByLanguage("Частичное закрытие по StopLoss","Partial close by StopLoss triggered")           :
      reason==EVENT_REASON_DONE_TP                          ?  TextByLanguage("Закрытие по TakeProfit","Close by TakeProfit triggered")                           :
      reason==EVENT_REASON_DONE_TP_PARTIALLY                ?  TextByLanguage("Частичное закрытие по TakeProfit","Partial close by TakeProfit triggered")       :
      reason==EVENT_REASON_DONE_BY_POS                      ?  TextByLanguage("Закрытие встречной позицией","Closed by opposite position")                        :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS            ?  TextByLanguage("Частичное закрытие встречной позицией","Closed partially by opposite position")    :
      reason==EVENT_REASON_DONE_BY_POS_PARTIALLY            ?  TextByLanguage("Закрытие частью объёма встречной позиции","Closed by incomplete volume of opposite position") :
      reason==EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY  ?  TextByLanguage("Частичное закрытие частью объёма встречной позиции","Closed partially by incomplete volume of opposite position")  :
      reason==EVENT_REASON_BALANCE_REFILL                   ?  TextByLanguage("Пополнение баланса","Balance refill")                                              :
      reason==EVENT_REASON_BALANCE_WITHDRAWAL               ?  TextByLanguage("Снятие средств с баланса","Withdrawal from balance")                          :
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

[In the fifth part of the library description](https://www.mql5.com/en/articles/6211), we have already developed
the method decoding a trading event code. Let's recall its logic:

An event code is passed to the method and the event code flags are then checked. If the code has the checked flag, the appropriate trading
event is set. Since the event code may have multiple flags, all possible flags for the event are checked and the event type is defined from
their combination. Next, the event type is added to the appropriate class variable and is entered in the property of the event object
(EVENT\_PROP\_TYPE\_EVENT).

Now we simply need to add tracking new flags matching possible netting
account events in the trading event code:

```
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEvent::SetTypeEvent(void)
  {
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
//--- Position opened (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
   //--- If an existing position is changed
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CHANGED))
        {
         //--- If the pending order is activated by a price
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
               //--- check the partial opening flag and set a trading event
               //--- "added volume to a position by activating a pending order" or "added volume to a position by partially activating a pending order"
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
         //---  check the partial closing flag and set the "Position closed by StopLoss" or "Position partially closed by StopLoss" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_SL : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by TakeProfit
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_TP))
        {
         //--- check the partial closing flag and set the "Position closed by TakeProfit" or "Position partially closed by TakeProfit" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by an opposite one
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closing flag and set the "Position closed by opposite one" or "Position partially closed by opposite one" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- If a position is closed
      else
        {
         //--- check the partial closing flag and set the "Position closed" or "Position partially closed" trading event
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

The entire logic is quite simple and commented in the code. Therefore, I am not going to dwell on the <if-else> method.

**We have made the changes in the abstract event class. Let's provide its full listing:**

```
//+------------------------------------------------------------------+
//|                                                        Event.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Needed for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "\..\..\Services\DELib.mqh"
#include "..\..\Collections\HistoryCollection.mqh"
#include "..\..\Collections\MarketCollection.mqh"
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
   bool              m_is_hedge;                                     // Hedge account flag
   long              m_chart_id;                                     // Control program chart ID
   int               m_digits_acc;                                   // Number of decimal places for the account currency
   long              m_long_prop[EVENT_PROP_INTEGER_TOTAL];          // Event integer properties
   double            m_double_prop[EVENT_PROP_DOUBLE_TOTAL];         // Event real properties
   string            m_string_prop[EVENT_PROP_STRING_TOTAL];         // Event string properties
//--- return the flag presence in the trading event
   bool              IsPresentEventFlag(const int event_code)  const { return (this.m_event_code & event_code)==event_code;            }

//--- Protected parametric constructor
                     CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket);
public:
//--- Default constructor
                     CEvent(void){;}

//--- Set event's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_EVENT_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                      }
   void              SetProperty(ENUM_EVENT_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;    }
   void              SetProperty(ENUM_EVENT_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;    }
//--- Return the event's (1) integer, (2) real and (3) string properties from the property array
   long              GetProperty(ENUM_EVENT_PROP_INTEGER property)      const { return this.m_long_prop[property];                     }
   double            GetProperty(ENUM_EVENT_PROP_DOUBLE property)       const { return this.m_double_prop[this.IndexProp(property)];   }
   string            GetProperty(ENUM_EVENT_PROP_STRING property)       const { return this.m_string_prop[this.IndexProp(property)];   }

//--- Return the flag of the event supporting the property
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_EVENT_PROP_STRING property)         { return true; }

//--- Set the control program chart ID
   void              SetChartID(const long id)                                { this.m_chart_id=id;                                    }
//--- Decode the event code and set the trading event, (2) return the trading event
   void              SetTypeEvent(void);
   ENUM_TRADE_EVENT  TradeEvent(void)                                   const { return this.m_trade_event;                             }
//--- Send the event to the chart (implementation in the class descendants)
   virtual void      SendEvent(void) {;}

//--- Compare CEvent objects by a specified property (to sort the lists by a specified event object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CEvent objects by all properties (to search for equal event objects)
   bool              IsEqual(CEvent* compared_event);
//+------------------------------------------------------------------+
//| Methods of simplified access to event object properties          |
//+------------------------------------------------------------------+
//--- Return (1) event type, (2) event time in milliseconds, (3) event status, (4) event reason, (5) deal type, (6) deal ticket,
//--- (7) order type, based on which a deal was executed, (8) position opening order type, (9) position last order ticket,
//--- (10) position first order ticket, (11) position ID, (12) opposite position ID, (13) magic number, (14) opposite position'a magic number, (15) position open time
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

//--- When changing position direction, return (1) previous position order type, (2) previous position order ticket,
//--- (3) current position order type, (4) current position order ticket,
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
//--- Return a (1) symbol and (2) opposite position symbol
   string            Symbol(void)                                       const { return this.GetProperty(EVENT_PROP_SYMBOL);                                 }
   string            SymbolCloseBy(void)                                const { return this.GetProperty(EVENT_PROP_SYMBOL_BY_ID);                           }

//+------------------------------------------------------------------+
//| Descriptions of order object properties                          |
//+------------------------------------------------------------------+
//--- Return description of the order's (1) integer, (2) real and (3) string properties
   string            GetPropertyDescription(ENUM_EVENT_PROP_INTEGER property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_DOUBLE property);
   string            GetPropertyDescription(ENUM_EVENT_PROP_STRING property);
//--- Return the event's (1) status and (2) type
   string            StatusDescription(void)                const;
   string            TypeEventDescription(void)             const;
//--- Return the name of an (1) event deal order, (2) position's parent order, (3) current position's order, (4) current position
//--- Return the name of an (5) order and (6) position before changing direction
   string            TypeOrderDealDescription(void)         const;
   string            TypeOrderFirstDescription(void)        const;
   string            TypeOrderEventDescription(void)        const;
   string            TypePositionCurrentDescription(void)   const;
   string            TypeOrderPreviousDescription(void)     const;
   string            TypePositionPreviousDescription(void)  const;
//--- Return the name of the deal/order/position reason
   string            ReasonDescription(void)                const;

//---  Display (1) description of order properties (full_prop=true - all properties, false - only supported ones),
//--- (2) short event message (implementation in the class descendants)
   void              Print(const bool full_prop=false);
   virtual void      PrintShort(void) {;}
  };
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CEvent::CEvent(const ENUM_EVENT_STATUS event_status,const int event_code,const ulong ticket) : m_event_code(event_code)
  {
   this.m_long_prop[EVENT_PROP_STATUS_EVENT]       =  event_status;
   this.m_long_prop[EVENT_PROP_TICKET_ORDER_EVENT] =  (long)ticket;
   this.m_is_hedge=bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   this.m_digits_acc=(int)::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS);
   this.m_chart_id=::ChartID();
  }
//+------------------------------------------------------------------+
//| Compare CEvent objects by a specified property                   |
//+------------------------------------------------------------------+
int CEvent::Compare(const CObject *node,const int mode=0) const
  {
   const CEvent *event_compared=node;
//--- compare integer properties of two events
   if(mode<EVENT_PROP_INTEGER_TOTAL)
     {
      long value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_EVENT_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare integer properties of two objects
   if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL)
     {
      double value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_EVENT_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<EVENT_PROP_DOUBLE_TOTAL+EVENT_PROP_INTEGER_TOTAL+EVENT_PROP_STRING_TOTAL)
     {
      string value_compared=event_compared.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_EVENT_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
//| Compare CEvent objects by all properties                         |
//+------------------------------------------------------------------+
bool CEvent::IsEqual(CEvent *compared_event)
  {
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_event.GetProperty(prop)) return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Decode the event code and set a trading event                    |
//+------------------------------------------------------------------+
void CEvent::SetTypeEvent(void)
  {
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
//--- Position opened (Check the presence of multiple flags in the event code)
   if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_OPENED))
     {
   //--- If an existing position is modified
      if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_POSITION_CHANGED))
        {
         //--- If the pending order is activated by a price
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
               //--- check the partial opening flag and set a trading event
               //--- "added volume to a position by activating a pending order" or "added volume to a position by partially activating a pending order"
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
         //--- check the partial closing flag and set the "Position closed by TakeProfit" or "Position partially closed by TakeProfit" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_TP : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- if a position is closed by an opposite one
      else if(this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS))
        {
         //--- check the partial closing flag and set the "Position closed by opposite one" or "Position partially closed by opposite one" trading event
         this.m_trade_event=(!this.IsPresentEventFlag(TRADE_EVENT_FLAG_PARTIAL) ? TRADE_EVENT_POSITION_CLOSED_BY_POS : TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS);
         this.SetProperty(EVENT_PROP_TYPE_EVENT,this.m_trade_event);
         return;
        }
      //--- If a position is closed
      else
        {
         //--- check the partial closing flag and set the "Position closed" or "Position partially closed" trading event
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
      //--- Remaining balance operation types match the ENUM_DEAL_TYPE enumeration starting with DEAL_TYPE_CREDIT
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
      property==EVENT_PROP_TICKET_DEAL_EVENT       ?  TextByLanguage("Тикет сделки","Deal's ticket")+" #"+(string)this.GetProperty(property)                                              :
      property==EVENT_PROP_TYPE_ORDER_EVENT        ?  TextByLanguage("Тип ордера события","Event's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property))    :
      property==EVENT_PROP_TYPE_ORDER_POSITION     ?  TextByLanguage("Тип ордера позиции","Position's order type")+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(property)) :
      property==EVENT_PROP_TICKET_ORDER_POSITION   ?  TextByLanguage("Тикет первого ордера позиции","Position's first order ticket")+" #"+(string)this.GetProperty(property)              :
      property==EVENT_PROP_TICKET_ORDER_EVENT      ?  TextByLanguage("Тикет ордера события","Event's order ticket")+" #"+(string)this.GetProperty(property)                               :
      property==EVENT_PROP_POSITION_ID             ?  TextByLanguage("Идентификатор позиции","Position ID")+" #"+(string)this.GetProperty(property)                                       :
      property==EVENT_PROP_POSITION_BY_ID          ?  TextByLanguage("Идентификатор встречной позиции","Opposite position's ID")+" #"+(string)this.GetProperty(property)                  :
      property==EVENT_PROP_MAGIC_ORDER             ?  TextByLanguage("Магический номер","Magic number")+": "+(string)this.GetProperty(property)                                           :
      property==EVENT_PROP_MAGIC_BY_ID             ?  TextByLanguage("Магический номер встречной позиции","Magic number of opposite position")+": "+(string)this.GetProperty(property)    :
      property==EVENT_PROP_TIME_ORDER_POSITION     ?  TextByLanguage("Время открытия позиции","Position's opened time")+": "+TimeMSCtoString(this.GetProperty(property))                  :
      property==EVENT_PROP_TYPE_ORD_POS_BEFORE     ?  TextByLanguage("Тип ордера позиции до смены направления","Type order of position before changing direction")                        :
      property==EVENT_PROP_TICKET_ORD_POS_BEFORE   ?  TextByLanguage("Тикет ордера позиции до смены направления","Ticket order of position before changing direction")                    :
      property==EVENT_PROP_TYPE_ORD_POS_CURRENT    ?  TextByLanguage("Тип ордера текущей позиции","Type order of current position")                                                       :
      property==EVENT_PROP_TICKET_ORD_POS_CURRENT  ?  TextByLanguage("Тикет ордера текущей позиции","Ticket order of current position")                                                   :
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
      property==EVENT_PROP_PRICE_EVENT             ?  TextByLanguage("Цена события","Price at the time of the event")+": "+::DoubleToString(this.GetProperty(property),dg)          :
      property==EVENT_PROP_PRICE_OPEN              ?  TextByLanguage("Цена открытия","Open price")+": "+::DoubleToString(this.GetProperty(property),dg)                             :
      property==EVENT_PROP_PRICE_CLOSE             ?  TextByLanguage("Цена закрытия","Close price")+": "+::DoubleToString(this.GetProperty(property),dg)                            :
      property==EVENT_PROP_PRICE_SL                ?  TextByLanguage("Цена StopLoss","StopLoss price")+": "+::DoubleToString(this.GetProperty(property),dg)                         :
      property==EVENT_PROP_PRICE_TP                ?  TextByLanguage("Цена TakeProfit","TakeProfit price")+": "+::DoubleToString(this.GetProperty(property),dg)                     :
      property==EVENT_PROP_VOLUME_ORDER_INITIAL    ?  TextByLanguage("Начальный объём ордера","Initial order volume")+": "+::DoubleToString(this.GetProperty(property),dgl)         :
      property==EVENT_PROP_VOLUME_ORDER_EXECUTED   ?  TextByLanguage("Исполненный объём ордера","Executed order volume")+": "+::DoubleToString(this.GetProperty(property),dgl)      :
      property==EVENT_PROP_VOLUME_ORDER_CURRENT    ?  TextByLanguage("Оставшийся объём ордера","Remaining order volume")+": "+::DoubleToString(this.GetProperty(property),dgl)      :
      property==EVENT_PROP_VOLUME_POSITION_EXECUTED ? TextByLanguage("Текущий объём позиции","Current position volume")+": "+::DoubleToString(this.GetProperty(property),dgl)       :
      property==EVENT_PROP_PROFIT                  ?  TextByLanguage("Профит","Profit")+": "+::DoubleToString(this.GetProperty(property),this.m_digits_acc)                         :
      EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the event's string property            |
//+------------------------------------------------------------------+
string CEvent::GetPropertyDescription(ENUM_EVENT_PROP_STRING property)
  {
   return
     (
      property==EVENT_PROP_SYMBOL ? TextByLanguage("Символ","Symbol")+": \""+this.GetProperty(property)+"\""   :
      TextByLanguage("Символ встречной позиции","Symbol of opposite position")+": \""+this.GetProperty(property)+"\""
     );
  }
//+------------------------------------------------------------------+
//| Return the event status name                                     |
//+------------------------------------------------------------------+
string CEvent::StatusDescription(void) const

  {
   ENUM_EVENT_STATUS status=(ENUM_EVENT_STATUS)this.GetProperty(EVENT_PROP_STATUS_EVENT);
   return
     (
      status==EVENT_STATUS_MARKET_PENDING    ?  TextByLanguage("Установлен отложенный ордер","Pending order placed") :
      status==EVENT_STATUS_MARKET_POSITION   ?  TextByLanguage("Открыта позиция","Position opened")                 :
      status==EVENT_STATUS_HISTORY_PENDING   ?  TextByLanguage("Удален отложенный ордер","Pending order removed")    :
      status==EVENT_STATUS_HISTORY_POSITION  ?  TextByLanguage("Закрыта позиция","Position closed")                  :
      status==EVENT_STATUS_BALANCE           ?  TextByLanguage("Балансная операция","Balance operation")             :
      TextByLanguage("Неизвестный статус","Unknown status")
     );
  }
//+------------------------------------------------------------------+
//| Return the trading event name                                    |
//+------------------------------------------------------------------+
string CEvent::TypeEventDescription(void) const
  {
   ENUM_TRADE_EVENT event=this.TypeEvent();
   return
     (
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
      event==TRADE_EVENT_POSITION_OPENED                       ?  TextByLanguage("Позиция открыта","Position opened")                                                  :
      event==TRADE_EVENT_POSITION_OPENED_PARTIAL               ?  TextByLanguage("Позиция открыта частично","Position opened partially")                               :
      event==TRADE_EVENT_POSITION_CLOSED                       ?  TextByLanguage("Позиция закрыта","Position closed")                                                   :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL               ?  TextByLanguage("Позиция закрыта частично","Position closed partially")                                :
      event==TRADE_EVENT_POSITION_CLOSED_BY_POS                ?  TextByLanguage("Позиция закрыта встречной","Position closed by opposite position")                    :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS        ?  TextByLanguage("Позиция закрыта встречной частично","Position closed partially by opposite position") :
      event==TRADE_EVENT_POSITION_CLOSED_BY_SL                 ?  TextByLanguage("Позиция закрыта по StopLoss","Position closed by StopLoss")                           :
      event==TRADE_EVENT_POSITION_CLOSED_BY_TP                 ?  TextByLanguage("Позиция закрыта по TakeProfit","Position closed by TakeProfit")                       :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL         ?  TextByLanguage("Позиция закрыта частично по StopLoss","Position closed partially by StopLoss")        :
      event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP         ?  TextByLanguage("Позиция закрыта частично по TakeProfit","Position closed partially by TakeProfit")    :
      event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET           ?  TextByLanguage("Разворот позиции по рыночному запросу","Position reversal by market request")         :
      event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING          ?  TextByLanguage("Разворот позиции срабатыванием отложенного ордера","Position reversal by a triggered pending order")                            :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET         ?  TextByLanguage("Добавлен объём к позиции по рыночному запросу","Added volume to position by market request")                                    :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING        ?  TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by activation of pending order")        :

      event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL   ?  TextByLanguage("Разворот позиции частичным исполнением запроса","Position reversal by partial completion of market request")                   :
      event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL  ?  TextByLanguage("Разворот позиции частичным срабатыванием отложенного ордера","Position reversal by partially triggered pending order")        :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL ?  TextByLanguage("Добавлен объём к позиции частичным исполнением запроса","Added volume to position by partial completion of market request")    :
      event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL ? TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by partially triggering a pending order")  :
      TextByLanguage("Нет торгового события","No trade event")
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the order/position/deal                       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderDealDescription(void) const
  {
   ENUM_EVENT_STATUS status=this.Status();
   return
     (
      status==EVENT_STATUS_MARKET_PENDING  || status==EVENT_STATUS_HISTORY_PENDING  ?  OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_EVENT))      :
      status==EVENT_STATUS_MARKET_POSITION || status==EVENT_STATUS_HISTORY_POSITION ?  PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT)) :
      status==EVENT_STATUS_BALANCE  ?  DealTypeDescription((ENUM_DEAL_TYPE)this.GetProperty(EVENT_PROP_TYPE_DEAL_EVENT))  :
      TextByLanguage("Неизвестный тип ордера","Unknown order type")
     );
  }
//+------------------------------------------------------------------+
//| Return the name of the position's first order                    |
//+------------------------------------------------------------------+
string CEvent::TypeOrderFirstDescription(void) const
  {
   return OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(EVENT_PROP_TYPE_ORDER_POSITION));
  }
//+------------------------------------------------------------------+
//| Return the name of the order that changed the position           |
//+------------------------------------------------------------------+
string CEvent::TypeOrderEventDescription(void) const
  {
   return OrderTypeDescription(this.TypeOrderEvent());
  }
//+------------------------------------------------------------------+
//| Return the name of the current position                          |
//+------------------------------------------------------------------+
string CEvent::TypePositionCurrentDescription(void) const
  {
   return PositionTypeDescription(this.TypePositionCurrent());
  }
//+------------------------------------------------------------------+
//| Return the name of the order before changing the direction       |
//+------------------------------------------------------------------+
string CEvent::TypeOrderPreviousDescription(void) const
  {
   return OrderTypeDescription(this.TypeOrderPosPrevious());
  }
//+------------------------------------------------------------------+
//| Return the name of the position before changing the direction    |
//+------------------------------------------------------------------+
string CEvent::TypePositionPreviousDescription(void) const
  {
   return PositionTypeDescription(this.TypePositionPrevious());
  }
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
      reason==EVENT_REASON_CANCEL                           ?  TextByLanguage("Отмена","Canceled")                                                                :
      reason==EVENT_REASON_EXPIRED                          ?  TextByLanguage("Истёк срок действия","Expired")                                                    :
      reason==EVENT_REASON_DONE                             ?  TextByLanguage("Рыночный запрос, выполненный в полном объёме","Fully completed market request")    :
      reason==EVENT_REASON_DONE_PARTIALLY                   ?  TextByLanguage("Выполненный частично рыночный запрос","Partially completed market request")        :
      reason==EVENT_REASON_VOLUME_ADD                       ?  TextByLanguage("Добавлен объём к позиции","Added volume to position")                              :
      reason==EVENT_REASON_VOLUME_ADD_PARTIALLY             ?  TextByLanguage("Добавлен объём к позиции частичным исполнением заявки","Volume added to the position by partially completed request")                 :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING            ?  TextByLanguage("Добавлен объём к позиции активацией отложенного ордера","Added volume to position by triggering pending order")                      :
      reason==EVENT_REASON_VOLUME_ADD_BY_PENDING_PARTIALLY  ?  TextByLanguage("Добавлен объём к позиции частичной активацией отложенного ордера","Added volume to position by triggering pending order partially")  :
      reason==EVENT_REASON_REVERSE                          ?  TextByLanguage("Разворот позиции","Position reversal")  :
      reason==EVENT_REASON_REVERSE_PARTIALLY                ?  TextByLanguage("Разворот позиции частичным исполнением заявки","Position reversal by partial completion of request")                             :
      reason==EVENT_REASON_REVERSE_BY_PENDING               ?  TextByLanguage("Разворот позиции при срабатывании отложенного ордера","Position reversal when triggering pending order")                               :
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
      reason==EVENT_REASON_BALANCE_WITHDRAWAL               ?  TextByLanguage("Снятие средств с баланса","Withdrawal from balance")                          :
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
//| Display the event properties in the journal                      |
//+------------------------------------------------------------------+
void CEvent::Print(const bool full_prop=false)
  {
   ::Print("============= ",TextByLanguage("Начало списка параметров события: \"","Beginning of event parameter list: \""),this.StatusDescription(),"\" =============");
   int beg=0, end=EVENT_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_INTEGER prop=(ENUM_EVENT_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_DOUBLE prop=(ENUM_EVENT_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=EVENT_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_EVENT_PROP_STRING prop=(ENUM_EVENT_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("================== ",TextByLanguage("Конец списка параметров: \"","End of parameter list: \""),this.StatusDescription(),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

Since the differences of hedging and netting accounts are evident only when working with positions, the **CEventPositionOpen** and **CEventPositionClose** derived classes of the **CEvent** abstract class require fine-tuning — only methods of displaying event messages to the journal are
refined. The remaining methods of the classes remain unchanged.

Open the **EventPositionOpen.mqh** file and add the private method creating
and returning a short event description:

```
//+------------------------------------------------------------------+
//| Position open event                                              |
//+------------------------------------------------------------------+
class CEventPositionOpen : public CEvent
  {
private:
//--- Create and return a short event message
   string            EventsMessage(void);
public:
//--- Constructor
                     CEventPositionOpen(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_POSITION,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a short event message in the journal, (2) Send an event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
```

**Let's write its implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Create and return a short event message                          |
//+------------------------------------------------------------------+
string CEventPositionOpen::EventsMessage(void)
  {
//--- number of decimal places in an event symbol quote
   int digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
//--- (1) header, (2) executed order volume, (3) executed position volume, (4) event price,
//--- (5) StopLoss price, (6) TakeProfit price, (7) magic number, (6) profit in account currency
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string vol_ord=::DoubleToString(this.VolumeOrderExecuted(),DigitsLots(this.Symbol()));
   string vol_pos=::DoubleToString(this.VolumePositionExecuted(),DigitsLots(this.Symbol()));
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceEvent(),digits);
   string sl=(this.PriceStopLoss()>0 ? ", sl "+ ::DoubleToString(this.PriceStopLoss(),digits) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+ ::DoubleToString(this.PriceTakeProfit(),digits) : "");
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string profit=TextByLanguage(", профит ",", profit ")+::DoubleToString(this.Profit(),this.m_digits_acc)+" "+::AccountInfoString(ACCOUNT_CURRENCY);
   //---
   string text="";
   //--- Position reversal
   if(this.GetProperty(EVENT_PROP_REASON_EVENT)<EVENT_REASON_ACTIVATED_PENDING)
     {
      //--- EURUSD: Buy #xx changed to 0.1 Sell #xx (0.2 SellLimit order #XX) at х.ххххх, sl х.ххххх, tp x.xxxxx, magic, profit xxxx
      text=
        (
         this.Symbol()+" "+
         this.TypePositionPreviousDescription()+" #"+(string)this.TicketPositionPrevious()+
         TextByLanguage(" изменен на "," turned to ")+vol_pos+" "+this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
         " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+sl+tp+magic+profit
        );
     }
   else
     {
      //--- Add volume
      if(this.GetProperty(EVENT_PROP_TICKET_ORDER_EVENT)!=this.GetProperty(EVENT_PROP_POSITION_ID))
        {
         //--- EURUSD: Added 0.1 to Buy #xx (BuyLimit order #XX) at х.ххххх, magic
         text=
           (
            this.Symbol()+" "+
            TextByLanguage("Добавлено ","Added ")+vol_ord+TextByLanguage(" к "," to ")+
            this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
            " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+magic
           );
        }
      //--- Open a position
      else
        {
         //--- EURUSD: Opened 0.1 Buy #xx (BuyLimit order #XX) at х.ххххх, sl х.ххххх, tp x.xxxxx, magic
         text=
           (
            this.Symbol()+" "+
            TextByLanguage("Открыт ","Open ")+vol_pos+" "+
            this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
            " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+sl+tp+magic
           );
        }
     }
   return head+text;
  }
//+------------------------------------------------------------------+
```

The method creates message variants depending on the event status and the presence of certain event object properties.

For example, if
StopLoss is set, "sl" header and its price are added to the text. Otherwise, an empty string is inserted instead of a StopLoss entry. The same
is done to some other event properties. The comments of the method listing contain the

conditions of creating an event text, as well as examples
of a text returned by the method.

The text created in the method is displayed in the journal from the PrintShort()
method, which in turn is called from the Refresh() method in the event collection class by calling the virtual method SendEvent()
of the CEvent class redefined here in the CEventPositionOpen class.

Below is the **full listing of the CEventPositionOpen class:**

```
//+------------------------------------------------------------------+
//|                                            EventPositionOpen.mqh |
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
//| Position opening event                                           |
//+------------------------------------------------------------------+
class CEventPositionOpen : public CEvent
  {
private:
//--- Create and return a short event message
   string            EventsMessage(void);
public:
//--- Constructor
                     CEventPositionOpen(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_MARKET_POSITION,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventPositionOpen::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   return(property==EVENT_PROP_POSITION_BY_ID ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CEventPositionOpen::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   if(property==EVENT_PROP_PRICE_CLOSE ||
      property==EVENT_PROP_PROFIT
     ) return false;
   return true;

  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventPositionOpen::PrintShort(void)
  {
   ::Print(this.EventsMessage());
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventPositionOpen::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.PositionID(),this.PriceOpen(),this.Symbol());
  }
//+------------------------------------------------------------------+
//| Create and return a short event message                          |
//+------------------------------------------------------------------+
string CEventPositionOpen::EventsMessage(void)
  {
//--- number of decimal places in an event symbol quote
   int digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
//--- (1) header, (2) executed order volume, (3) executed position volume, (4) event price,
//--- (5) StopLoss price, (6) TakeProfit price, (7) magic number, (6) profit in account currency
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string vol_ord=::DoubleToString(this.VolumeOrderExecuted(),DigitsLots(this.Symbol()));
   string vol_pos=::DoubleToString(this.VolumePositionExecuted(),DigitsLots(this.Symbol()));
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceEvent(),digits);
   string sl=(this.PriceStopLoss()>0 ? ", sl "+ ::DoubleToString(this.PriceStopLoss(),digits) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+ ::DoubleToString(this.PriceTakeProfit(),digits) : "");
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string profit=TextByLanguage(", профит ",", profit ")+::DoubleToString(this.Profit(),this.m_digits_acc)+" "+::AccountInfoString(ACCOUNT_CURRENCY);
   //---
   string text="";
   //--- Position reversal
   if(this.GetProperty(EVENT_PROP_REASON_EVENT)<EVENT_REASON_ACTIVATED_PENDING)
     {
      //--- EURUSD: Buy #xx changed to 0.1 Sell #xx [0.2 SellLimit order #XX] at х.ххххх, sl х.ххххх, tp x.xxxxx, magic, profit xxxx
      text=
        (
         this.Symbol()+" "+
         this.TypePositionPreviousDescription()+" #"+(string)this.TicketPositionPrevious()+
         TextByLanguage(" изменен на "," turned to ")+vol_pos+" "+this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
         " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+sl+tp+magic+profit
        );
     }
   else
     {
      //--- Add volume
      if(this.GetProperty(EVENT_PROP_TICKET_ORDER_EVENT)!=this.GetProperty(EVENT_PROP_POSITION_ID))
        {
         //--- EURUSD: Added 0.1 to Buy #xx [BuyLimit order #XX] at х.ххххх, magic
         text=
           (
            this.Symbol()+" "+
            TextByLanguage("Добавлено ","Added ")+vol_ord+TextByLanguage(" к "," to ")+
            this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
            " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+magic
           );
        }
      //--- Open a position
      else
        {
         //--- EURUSD: Opened 0.1 Buy #xx [BuyLimit order #XX] at х.ххххх, sl х.ххххх, tp x.xxxxx, magic
         text=
           (
            this.Symbol()+" "+
            TextByLanguage("Открыт ","Open ")+vol_pos+" "+
            this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
            " ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+sl+tp+magic
           );
        }
     }
   return head+text;
  }
//+------------------------------------------------------------------+
```

Similarly, **change the CEventPositionClose class:**

```
//+------------------------------------------------------------------+
//|                                           EventPositionClose.mqh |
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
//| Position opening event                                           |
//+------------------------------------------------------------------+
class CEventPositionClose : public CEvent
  {
private:
//--- Create and return a short event message
   string            EventsMessage(void);
public:
//--- Constructor
                     CEventPositionClose(const int event_code,const ulong ticket=0) : CEvent(EVENT_STATUS_HISTORY_POSITION,event_code,ticket) {}
//--- Supported (1) real and (2) integer order properties
   virtual bool      SupportProperty(ENUM_EVENT_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_EVENT_PROP_DOUBLE property);
//--- (1) Display a brief message about the event in the journal, (2) Send the event to the chart
   virtual void      PrintShort(void);
   virtual void      SendEvent(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| integer property, otherwise, return 'false'                      |
//+------------------------------------------------------------------+
bool CEventPositionClose::SupportProperty(ENUM_EVENT_PROP_INTEGER property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if the event supports the passed                   |
//| real property, otherwise, return 'false'                         |
//+------------------------------------------------------------------+
bool CEventPositionClose::SupportProperty(ENUM_EVENT_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message about the event in the journal           |
//+------------------------------------------------------------------+
void CEventPositionClose::PrintShort(void)
  {
   ::Print(this.EventsMessage());
  }
//+------------------------------------------------------------------+
//| Send the event to the chart                                      |
//+------------------------------------------------------------------+
void CEventPositionClose::SendEvent(void)
  {
   this.PrintShort();
   ::EventChartCustom(this.m_chart_id,(ushort)this.m_trade_event,this.PositionID(),this.PriceClose(),this.Symbol());
  }
//+------------------------------------------------------------------+
//| Create and return a short event message                          |
//+------------------------------------------------------------------+
string CEventPositionClose::EventsMessage(void)
  {
//--- number of decimal places in an event symbol quote
   int digits=(int)::SymbolInfoInteger(this.Symbol(),SYMBOL_DIGITS);
//--- (1) header, (2) executed order volume, (3) executed position volume, (4) event price,
//--- (5) StopLoss price, (6) TakeProfit price, (7) magic number, (6) profit in account currency, (7,8) closure message options
   string head="- "+this.TypeEventDescription()+": "+TimeMSCtoString(this.TimePosition())+" -\n";
   string vol_ord=::DoubleToString(this.VolumeOrderExecuted(),DigitsLots(this.Symbol()));
   string vol_pos=::DoubleToString(this.VolumePositionExecuted(),DigitsLots(this.Symbol()));
   string price=TextByLanguage(" по цене "," at price ")+::DoubleToString(this.PriceEvent(),digits);
   string sl=(this.PriceStopLoss()>0 ? ", sl "+ ::DoubleToString(this.PriceStopLoss(),digits) : "");
   string tp=(this.PriceTakeProfit()>0 ? ", tp "+ ::DoubleToString(this.PriceTakeProfit(),digits) : "");
   string magic=(this.Magic()!=0 ? TextByLanguage(", магик ",", magic ")+(string)this.Magic() : "");
   string profit=TextByLanguage(", профит ",", profit ")+::DoubleToString(this.Profit(),this.m_digits_acc)+" "+::AccountInfoString(ACCOUNT_CURRENCY);
   string close=TextByLanguage("Закрыт ","Close ");
   string in_pos="";
   //---
   if(this.GetProperty(EVENT_PROP_TYPE_EVENT)>TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL)
     {
      close=TextByLanguage("Закрыт объём ","Closed volume ")+vol_ord;
      in_pos=TextByLanguage(" в "," in ");
     }
   string opposite=
     (
      this.IsPresentEventFlag(TRADE_EVENT_FLAG_BY_POS)   ?
      TextByLanguage(" встречным "," by opposite ")+this.SymbolCloseBy()+" "+
      this.TypeOrderDealDescription()+" #"+(string)this.PositionByID()+(this.MagicCloseBy()> 0 ? "("+(string)this.MagicCloseBy()+" ]" : "")
                                                         : ""
     );
   //--- EURUSD: Closed 0.1 Sell #xx [0.2 SellLimit order #XX] at х.ххххх, sl х.ххххх, tp x.xxxxx, magic, profit xxxx
   string text=
     (
      this.Symbol()+" "+close+in_pos+this.TypePositionCurrentDescription()+" #"+(string)this.TicketPositionCurrent()+
      opposite+" ["+vol_ord+" "+this.TypeOrderEventDescription()+" #"+(string)this.TicketOrderEvent()+" ]"+price+sl+tp+magic+profit
     );
   return head+text;
  }
//+------------------------------------------------------------------+
```

**All event object classes were changed for new tasks of working on netting accounts.**

**Now let's deal with the**
**CEventCollection event collection class.**

Previously, the CreateNewEvent() method (described [in the fifth part](https://www.mql5.com/en/articles/6211#node03))
featured a local variable for storing a trading event code.

Let's make it a private class member by removing it from the new event creation method and declaring in
the private class section. Also, add declarations of the necessary methods of creating a new event for hedging
and netting account types, the method for returning the list of
all InOut deals by position ID and the method for obtaining a market position
object by its ID.

```
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

//--- Create a trading event depending on the order status
   void              CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market);
//--- Create an event for a (1) hedging account, (2) netting account
   void              NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
   void              NewDealEventNetto(COrder* deal,CArrayObj* list_history,CArrayObj* list_market);
//--- Select and return the list of market pending orders
   CArrayObj*        GetListMarketPendings(CArrayObj* list);
//--- Select from the list and return the list of historical (1) removed pending orders, (2) deals, (3) all closing orders
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
//--- Return the flag of the event object presence in the event list
   bool              IsPresentEventInList(CEvent* compared_event);

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

Reset the trading event code in the class constructor's
initialization list:

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
  }
//+------------------------------------------------------------------+
```

A netting account may experience multiple trading actions. It may have only one position undergoing changes on a single symbol. They may
include changing the volume in case of partial closing triggered by activation of an opposite order having a lesser volume, as well as adding
a volume to the position when orders in the same direction are triggered.

The most interesting changes, however, happen to a position when opposite orders having greater volume are
triggered. In this case, a new ticket is assigned to a position. The ticket corresponds to a triggered order, and the position type is changed
to the opposite one (position reversal). The position ID remains unchanged and is equal to the ticket of the very first order that triggered
the position on the account.

We need to track all position direction changes throughout its lifetime in order to (1) correctly display position reversal entries
in the journal and (2) have the ability to obtain data on position reversal events in our programs. To achieve this, we need to have access to
all its deals having the DEAL\_ENTRY\_INOUT position change method from the

[ENUM\_DEAL\_ENTRY](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) enumeration.

In this case, we only need to arrange such deals sequentially by time of their occurrence and take a necessary deal. The deal itself features
all order properties that triggered it.

Thus, if we have a deal order, we can receive a ticket for a position with a changed direction, as well as
type of an order, which triggered a position reversal, along with new StopLoss and TakeProfit levels, etc. All we need to get such
functionality is to create a list of all InOut deals by its ID, which is very easy to do using the library we develop.

**Let's consider the method for receiving all InOut deals of a position by its ID:**

```
//+------------------------------------------------------------------+
//| Return the list of all reversal deals (IN_OUT)                   |
//| by a position ID                                                 |
//+------------------------------------------------------------------+
CArrayObj* CEventsCollection::GetListAllDealsInOutByPosID(CArrayObj *list,const ulong position_id)
  {
   if(list.Type()!=COLLECTION_HISTORY_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of the history collection"));
      return NULL;
     }
   CArrayObj* list_deals=this.GetListAllDealsByPosID(list,position_id);
   list_deals=CSelect::ByOrderProperty(list_deals,ORDER_PROP_DEAL_ENTRY,DEAL_ENTRY_INOUT,EQUAL);
   return list_deals;
  }
//+------------------------------------------------------------------+
```

Check the type of the list passed to the method. If it is not a
historical orders and deals collection one, warn about the error and return

NULL.

We need all these checks of lists in the classes to detect our own errors. They are to be removed after debugging not to burden the
calculations with unnecessary checks.

Next, we receive the list of deals by position ID (the
method was considered

[in the previous article](https://www.mql5.com/en/articles/6211#node03)), sort
the obtained list by InOut position change method and return the final list.

To receive data on an open position or define its absence, create a **method receiving a market position object by its ID:**

```
//+------------------------------------------------------------------+
//| Return a position by ID                                          |
//+------------------------------------------------------------------+
COrder* CEventsCollection::GetPositionByID(CArrayObj *list,const ulong position_id)
  {
   if(list.Type()!=COLLECTION_MARKET_ID)
     {
      Print(DFUN,TextByLanguage("Ошибка. Список не является списком рыночной коллекции","Error. The list is not a list of the market collection"));
      return NULL;
     }
   CArrayObj* list_orders=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list_orders=CSelect::ByOrderProperty(list_orders,ORDER_PROP_POSITION_ID,position_id,EQUAL);
   if(list_orders==NULL || list_orders.Total()==0) return NULL;
   COrder* order=list_orders.At(0);
   return(order!=NULL ? order : NULL);
  }
//+------------------------------------------------------------------+
```

The method is simple, just like other similar methods from the library. Check
the type of a selected list. If it is not a list of market orders and positions collection, warn about the error and return NULL.

Next, take only active position objects from the list passed to the method
and

sort it by position ID passed to the method.

If
failed to obtain the list or it has no objects, return NULL—
there is no requested position.

Next, receive a single market position object from the list
(there can only be one position with a specified ID in the market) and return
either the object itself or

NULL in case receiving it ended in an error.

**The method of creating a new CreateNewEvent() event object** was described in the [previous \\
article](https://www.mql5.com/en/articles/6211#node03).

Here I will only show implemented changes.

The following local variable has been removed from the method

```
int trade_event_code
```

It has become a member of the class we created in the private
section.

The method logic remains the same but now it also features calling the necessary methods for handling the type of the account we are working on.
If it is hedging,

the method of creating a new event for a hedging account is called.
Otherwise,

the method of creating a new event for a netting account is used:

```
//+------------------------------------------------------------------+
//| Create a trading event depending on the order status             |
//+------------------------------------------------------------------+
void CEventsCollection::CreateNewEvent(COrder* order,CArrayObj* list_history,CArrayObj* list_market)
  {
   this.m_trade_event_code=TRADE_EVENT_FLAG_NO_EVENT;
   ENUM_ORDER_STATUS status=order.Status();
//--- Pending order placed
   if(status==ORDER_STATUS_MARKET_PENDING)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_PLASED;
      CEvent* event=new CEventOrderPlased(this.m_trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                             // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_DONE);                             // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());                          // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                           // Event order ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());                         // Event order type
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());                      // Event order type
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());                          // Event order ticket
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                       // Order ticket
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                             // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());                        // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order.Magic());                                  // Opposite position magic number

         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());                      // Position order type before changing direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());                       // Position order ticket before changing direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());                     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());                      // Current position order ticket

         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                                  // Order magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());                    // Order time
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                              // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                               // Order price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                             // Order close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                                  // StopLoss order price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                                // TakeProfit order price
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());                        // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order.VolumeCurrent());                 // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,0);                                 // Executed position volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                                      // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                                      // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                                // Opposite position symbol
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
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Pending order removed
   if(status==ORDER_STATUS_HISTORY_PENDING)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_ORDER_REMOVED;
      CEvent* event=new CEventOrderRemoved(this.m_trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         ENUM_EVENT_REASON reason=
           (
            order.State()==ORDER_STATE_CANCELED ? EVENT_REASON_CANCEL :
            order.State()==ORDER_STATE_EXPIRED  ? EVENT_REASON_EXPIRED : EVENT_REASON_DONE
           );
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeCloseMSC());                            // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                        // Event reason (from the ENUM_EVENT_REASON reason)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());                          // Event order type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                           // Event order ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());                         // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());                      // Type of the order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());                          // Ticket of the order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                       // Ticket of the order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                             // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());                        // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order.Magic());                                  // Opposite position magic number

         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());                      // Position order type before changing the direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());                       // Position order ticket before changing the direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());                     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());                      // Current position order ticket

         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                                  // Order magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());                    // Time of the order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                              // Event price
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                               // Order open price
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceClose());                             // Order close price
         event.SetProperty(EVENT_PROP_PRICE_SL,order.StopLoss());                                  // StopLoss order price
         event.SetProperty(EVENT_PROP_PRICE_TP,order.TakeProfit());                                // TakeProfit order price
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());                        // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order.Volume()-order.VolumeCurrent()); // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order.VolumeCurrent());                 // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,0);                                 // Executed position volume
         event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                                      // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                                      // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                                // Opposite position symbol
         //--- Set control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not present on the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send a message about the event and set the last trading event value
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //---  If the event is already in the list, remove the new event object and display the debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Position opened (__MQL4__)
   if(status==ORDER_STATUS_MARKET_POSITION)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
      CEvent* event=new CEventPositionOpen(this.m_trade_event_code,order.Ticket());
      if(event!=NULL)
        {
         event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpen());                                // Event time
         event.SetProperty(EVENT_PROP_REASON_EVENT,EVENT_REASON_DONE);                             // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());                          // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());                           // Event deal ticket
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());                         // Type of the order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());                      // Type of the order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());                          // Ticket of the order, based on which an event deal is opened (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());                       // Ticket of the order, based on which a position deal is opened (the first position order)
         event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                             // Position ID
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());                        // Opposite position ID
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order.Magic());                                  // Opposite position magic number

         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());                      // Position order type before changing the direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());                       // Position order ticket before changing the direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());                     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());                      // Current position order ticket

         event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                                  // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpen());                       // Time of the order that triggered a position deal (the first position order)
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
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                                // Opposite position symbol
         //--- Set the control program chart ID, decode the event code and set the event type
         event.SetChartID(this.m_chart_id);
         event.SetTypeEvent();
         //--- Add the event object if it is not present in the list
         if(!this.IsPresentEventInList(event))
           {
            this.m_list_events.InsertSort(event);
            //--- Send the message about the event and set the value of the last trading event
            event.SendEvent();
            this.m_trade_event=event.TradeEvent();
           }
         //--- If the event is already present in the list, remove the new event object and display the debugging message
         else
           {
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- New deal (__MQL5__)
   if(status==ORDER_STATUS_DEAL)
     {
      //--- New balance operation
      if((ENUM_DEAL_TYPE)order.TypeOrder()>DEAL_TYPE_SELL)
        {
         this.m_trade_event_code=TRADE_EVENT_FLAG_ACCOUNT_BALANCE;
         CEvent* event=new CEventBalanceOperation(this.m_trade_event_code,order.Ticket());
         if(event!=NULL)
           {
            ENUM_EVENT_REASON reason=
              (
               (ENUM_DEAL_TYPE)order.TypeOrder()==DEAL_TYPE_BALANCE ? (order.Profit()>0 ? EVENT_REASON_BALANCE_REFILL : EVENT_REASON_BALANCE_WITHDRAWAL) :
               (ENUM_EVENT_REASON)(order.TypeOrder()+REASON_EVENT_SHIFT)
              );
            event.SetProperty(EVENT_PROP_TIME_EVENT,order.TimeOpenMSC());                 // Event time
            event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
            event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,order.TypeOrder());              // Event deal type
            event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,order.Ticket());               // Event order ticket
            event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order.TypeOrder());             // Type of the order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order.TypeOrder());          // Type of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order.Ticket());              // Ticket of the order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order.Ticket());           // Ticket of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_POSITION_ID,order.PositionID());                 // Position ID
            event.SetProperty(EVENT_PROP_POSITION_BY_ID,order.PositionByID());            // Opposite position ID
            event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order.Magic());                      // Opposite position magic number

            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order.TypeOrder());          // Position order type before changing the direction
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order.Ticket());           // Position order ticket before changing the direction
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order.TypeOrder());         // Current position order type
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order.Ticket());          // Current position order ticket

            event.SetProperty(EVENT_PROP_MAGIC_ORDER,order.Magic());                      // Order/deal/position magic number
            event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order.TimeOpenMSC());        // Time of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_PRICE_EVENT,order.PriceOpen());                  // Event price
            event.SetProperty(EVENT_PROP_PRICE_OPEN,order.PriceOpen());                   // Order/deal/position open price
            event.SetProperty(EVENT_PROP_PRICE_CLOSE,order.PriceOpen());                  // Order/deal/position close price
            event.SetProperty(EVENT_PROP_PRICE_SL,0);                                     // StopLoss deal price
            event.SetProperty(EVENT_PROP_PRICE_TP,0);                                     // TakeProfit deal price
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order.Volume());            // Requested deal volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order.Volume());           // Executed deal volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,0);                         // Remaining (unexecuted) deal volume
            event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,order.Volume());        // Executed position volume
            event.SetProperty(EVENT_PROP_PROFIT,order.Profit());                          // Profit
            event.SetProperty(EVENT_PROP_SYMBOL,order.Symbol());                          // Order symbol
            event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order.Symbol());                    // Opposite position symbol
            //--- Set control program chart ID, decode the event code and set the event type
            event.SetChartID(this.m_chart_id);
            event.SetTypeEvent();
            //--- Add the event object if it is not in the list
            if(!this.IsPresentEventInList(event))
              {
               //--- Send a message about the event and set the value of the last trading event
               this.m_list_events.InsertSort(event);
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
      //--- If this is not a balance operation
      else
        {
         if(this.m_is_hedge)
            this.NewDealEventHedge(order,list_history,list_market);
         else
            this.NewDealEventNetto(order,list_history,list_market);
        }
     }
  }
//+------------------------------------------------------------------+
```

**The method of creating a new event for a hedge account:**

```
//+------------------------------------------------------------------+
//| Create a hedging account event                                   |
//+------------------------------------------------------------------+
void CEventsCollection::NewDealEventHedge(COrder* deal,CArrayObj* list_history,CArrayObj* list_market)
  {
   //--- Market entry
   if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_IN)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
      int reason=EVENT_REASON_DONE;
      //--- Search for all position deals in the direction of its opening and calculate their overall volume
      double volume_in=this.SummaryVolumeDealsInByPosID(list_history,deal.PositionID());
      //--- Take the deal order and the last position order from the list of all position orders
      ulong order_ticket=deal.GetProperty(ORDER_PROP_DEAL_ORDER_TICKET);
      COrder* order_first=this.GetHistoryOrderByTicket(list_history,order_ticket);
      COrder* order_last=this.GetLastOrderFromList(list_history,deal.PositionID());
      //--- Get an open position by ticket
      COrder* position=this.GetPositionByID(list_market,deal.PositionID());
      double vol_position=(position!=NULL ? position.Volume() : 0);
      //--- If there is no last order, the first and last position orders coincide
      if(order_last==NULL)
         order_last=order_first;
      if(order_first!=NULL)
        {
         //--- If the order volume is opened partially, this is a partial execution
         if(this.SummaryVolumeDealsInByPosID(list_history,deal.PositionID())<order_first.Volume())
           {
            this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
            reason=EVENT_REASON_DONE_PARTIALLY;
           }
         //--- If an opening order is a pending one, the pending order is activated
         if(order_first.TypeOrder()>ORDER_TYPE_SELL && order_first.TypeOrder()<ORDER_TYPE_CLOSE_BY)
           {
            this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
            //--- If an order is executed partially, set the partial order execution as an event reason
            reason=
              (this.SummaryVolumeDealsInByPosID(list_history,deal.PositionID())<order_first.Volume() ?
               EVENT_REASON_ACTIVATED_PENDING_PARTIALLY :
               EVENT_REASON_ACTIVATED_PENDING
              );
           }
         CEvent* event=new CEventPositionOpen(this.m_trade_event_code,deal.PositionID());
         if(event!=NULL)
           {
            event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                        // Event time (position open time)
            event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                  // Event reason (from the ENUM_EVENT_REASON enumeration)
            event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());                     // Event deal type
            event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                      // Event deal ticket
            event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());          // Type of the order that triggered an event deal (the first position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());           // Ticket of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_last.TypeOrder());              // Type of the order that triggered a position deal (the last position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_last.Ticket());               // Ticket of the order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                        // Position ID
            event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last.PositionByID());             // Opposite position ID
            //---
            event.SetProperty(EVENT_PROP_MAGIC_BY_ID,deal.Magic());                             // Opposite position magic number
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order_first.TypeOrder());          // Position order type before changing the direction
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order_first.Ticket());           // Position order ticket before changing the direction
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order_first.TypeOrder());         // Current position order type
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order_first.Ticket());          // Current position order ticket
            event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,deal.Symbol());                           // Opposite position symbol
            //---
            event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                             // Order/deal/podition magic number
            event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());        // Time of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                         // Event price (position open price)
            event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());                   // Order open price (position's opening order price)
            event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last.PriceClose());                  // Order close price (position's last order close price)
            event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                      // StopLoss price (position's StopLoss order price)
            event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());                    // TakeProfit price (position's TakeProfit order price)
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order_first.Volume());                                 // Requested order volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,(order_first.Volume()-order_first.VolumeCurrent()));  // Executed order volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order_first.VolumeCurrent());                          // Remaining (unexecuted) order volume
            event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);                                     // Executed position volume
            event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                             // Profit
            event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                                 // Order symbol
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
               ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
               delete event;
              }
           }
        }
     }
   //--- Exit the market
   else if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
      int reason=EVENT_REASON_DONE;
      //--- Take the first and last position orders from the list of all position orders
      COrder* order_first=this.GetFirstOrderFromList(list_history,deal.PositionID());
      COrder* order_last=this.GetLastOrderFromList(list_history,deal.PositionID());
      //--- Get an open position by ticket
      COrder* position=this.GetPositionByID(list_market,deal.PositionID());
      double vol_position=(position!=NULL ? position.Volume() : 0);
      if(order_first!=NULL && order_last!=NULL)
        {
         //--- Search for all position deals in the directions of its opening and closing, and count their total volume
         double volume_in=this.SummaryVolumeDealsInByPosID(list_history,deal.PositionID());
         double volume_out=this.SummaryVolumeDealsOutByPosID(list_history,deal.PositionID());
         //--- Calculate the current volume of the closed position
         int dgl=(int)DigitsLots(deal.Symbol());
         double volume_current=::NormalizeDouble(volume_in-volume_out,dgl);
         //--- If the order volume is closed partially, this is a partial execution
         if(volume_current>0)
           {
            this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
           }
         //--- If the closing order is executed partially, set the closing order partial execution as an event reason
         if(order_last.VolumeCurrent()>0)
           {
            reason=EVENT_REASON_DONE_PARTIALLY;
           }
         //--- If the closing flag is set to StopLoss for a position's closing order, then closing is performed by StopLoss
         //--- If a StopLoss order is executed partially, set partial StopLoss order execution as the event reason
         if(order_last.IsCloseByStopLoss())
           {
            this.m_trade_event_code+=TRADE_EVENT_FLAG_SL;
            reason=(order_last.VolumeCurrent()>0 ? EVENT_REASON_DONE_SL_PARTIALLY : EVENT_REASON_DONE_SL);
           }
         //--- If the closing flag is set to TakeProfit for a position's closing order, then closing is performed by TakeProfit
         //--- If a TakeProfit order is executed partially, set partial TakeProfit order execution as the event reason
         else if(order_last.IsCloseByTakeProfit())
           {
            this.m_trade_event_code+=TRADE_EVENT_FLAG_TP;
            reason=(order_last.VolumeCurrent()>0 ? EVENT_REASON_DONE_TP_PARTIALLY : EVENT_REASON_DONE_TP);
           }
         //---
         CEvent* event=new CEventPositionClose(this.m_trade_event_code,deal.PositionID());
         if(event!=NULL)
           {
            event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                        // Event time (position open time)
            event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                  // Event reason (from the ENUM_EVENT_REASON enumeration)
            event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());                     // Event deal type
            event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                      // Event deal ticket
            event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());          // Type of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_last.TypeOrder());              // Type of the order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());           // Ticket of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_last.Ticket());               // Ticket of the order that triggered an event deal (the last position order)
            event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                        // Position ID
            event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last.PositionByID());             // Opposite position ID
            //---
            event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order_last.Magic());                       // Opposite position magic number
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order_first.TypeOrder());          // Position order type before changing the direction
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order_first.Ticket());           // Position order ticket before changing the direction
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order_first.TypeOrder());         // Current position order type
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order_first.Ticket());          // Current position order ticket
            event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order_last.Symbol());                     // Opposite position symbol
            //---
            event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                             // Order/deal/podition magic number
            event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());        // Time of the order that triggered a position deal (the first position order)
            event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                         // Event price (position open price)
            event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());                   // Order open price (position's opening order price)
            event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last.PriceClose());                  // Order close price  (position's last order close price)
            event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                      // StopLoss price (position's StopLoss order price)
            event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());                    // TakeProfit price (position's TakeProfit order price)
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order_last.Volume());                               // Requested order volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,order_last.Volume()-order_last.VolumeCurrent());   // Executed order volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,order_last.VolumeCurrent());                        // Remaining (unexecuted) order volume
            event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);                                  // Remaining (current) position volume
            //---
            event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                             // Profit
            event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                                 // Order symbol
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
               ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
               delete event;
              }
           }
        }
     }
   //--- Opposite position
   else if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT_BY)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
      int reason=EVENT_REASON_DONE_BY_POS;
      //--- Take the first and closing position orders from the list of all position orders
      COrder* order_first=this.GetFirstOrderFromList(list_history,deal.PositionID());
      COrder* order_close=this.GetCloseByOrderFromList(list_history,deal.PositionID());
      //--- Get an open position by ID
      COrder* position=this.GetPositionByID(list_market,order_first.PositionID());
      double vol_position=(position!=NULL ? position.Volume() : 0);
      if(order_first!=NULL && order_close!=NULL)
        {
         //--- Add the flag of closing by an opposite position
         this.m_trade_event_code+=TRADE_EVENT_FLAG_BY_POS;

         //--- Take the first order of the closing position
         Print(DFUN,"PositionByID=",order_close.PositionByID());
         CArrayObj* list_close_by=this.GetListAllOrdersByPosID(list_history,order_close.PositionByID());
         COrder* order_close_by=list_close_by.At(0);
         if(order_close_by==NULL)
            return;
         //--- Search for all closed position deals in the direction of its opening and closing and count their total volume
         double volume_in=this.SummaryVolumeDealsInByPosID(list_history,deal.PositionID());
         double volume_out=this.SummaryVolumeDealsOutByPosID(list_history,deal.PositionID());//+order_close.Volume();
         //--- Calculate the current volume of the closed position
         int dgl=(int)DigitsLots(deal.Symbol());
         double volume_current=::NormalizeDouble(volume_in-volume_out,dgl);
         //--- Search for all opposite position deals in the directions of its opening and closing and calculate their total volume
         double volume_opp_in=this.SummaryVolumeDealsInByPosID(list_history,order_close.PositionByID());
         double volume_opp_out=this.SummaryVolumeDealsOutByPosID(list_history,order_close.PositionByID());
         //--- Calculate the current volume of the opposite position
         double volume_opp_current=::NormalizeDouble(volume_opp_in-volume_opp_out,dgl);
         //--- If the closed position volume is closed partially, this is a partial closing
         if(volume_current>0 || order_close.VolumeCurrent()>0)
           {
            //--- Add the partial closing flag
            this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
            //--- If the opposite position is closed partially, there is a partial closing by the part of the opposite position volume
            reason=(volume_opp_current>0 ? EVENT_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY : EVENT_REASON_DONE_PARTIALLY_BY_POS);
           }
         //--- If the position volume is closed in full and there is a partial execution by the opposite one, there is a closing by the part of the opposite position volume
         else
           {
            if(volume_opp_current>0)
              {
               reason=EVENT_REASON_DONE_BY_POS_PARTIALLY;
              }
           }
         CEvent* event=new CEventPositionClose(this.m_trade_event_code,deal.PositionID());
         if(event!=NULL)
           {
            event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                        // Event time
            event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                                  // Event reason (from the ENUM_EVENT_REASON enumeration)
            event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());                     // Event deal type
            event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                      // Event deal ticket
            event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,order_close.TypeOrder());             // Type of the order, based on which an event deal is opened (the last position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,order_close.Ticket());              // Ticket of the order, based on which an event deal is opened (the last position order)
            event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first.TimeOpenMSC());        // Time of the order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,order_first.TypeOrder());          // Type of the order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,order_first.Ticket());           // Ticket of the order, based on which a position deal is opened (the first position order)
            event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                        // Position ID
            event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_close.PositionByID());            // Opposite position ID
            //---
            event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order_close_by.Magic());                   // Opposite position magic number
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,order_first.TypeOrder());          // Position order type before changing the direction
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,order_first.Ticket());           // Position order ticket before changing the direction
            event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,order_first.TypeOrder());         // Current position order type
            event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,order_first.Ticket());          // Current position order ticket
            event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,order_close_by.Symbol());                 // Opposite position symbol
            //---
            event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                             // Order/deal/position magic number
            event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                         // Event price
            event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first.PriceOpen());                   // Order/deal/position open price
            event.SetProperty(EVENT_PROP_PRICE_CLOSE,deal.PriceClose());                        // Order/deal/position close price
            event.SetProperty(EVENT_PROP_PRICE_SL,order_first.StopLoss());                      // StopLoss price (Position order StopLoss price)
            event.SetProperty(EVENT_PROP_PRICE_TP,order_first.TakeProfit());                    // TakeProfit price (Position order TakeProfit price)
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,::NormalizeDouble(volume_in,dgl));// Initial volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,deal.Volume());                  // Closed volume
            event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,volume_current);                  // Remaining (current) volume
            event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);                // Remaining (current) volume
            event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                             // Profit
            event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                                 // Order symbol
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
            //--- If the event is already present in the list, remove the new event object and display the debugging message
            else
              {
               ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
               delete event;
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The method is quite large, although all actions are similar and described in the method listing comments. I believe, the method code should
not cause any issues.

The **method of creating a new event for a netting account** has the same logic:

```
//+------------------------------------------------------------------+
//| Create an event for a netting account                            |
//+------------------------------------------------------------------+
void CEventsCollection::NewDealEventNetto(COrder *deal,CArrayObj *list_history,CArrayObj *list_market)
  {
//--- Prepare position history data
//--- Lists of all deals and position direction changes
   CArrayObj* list_deals=this.GetListAllDealsByPosID(list_history,deal.PositionID());
   CArrayObj* list_changes=this.GetListAllDealsInOutByPosID(list_history,deal.PositionID());
   if(list_deals==NULL || list_changes==NULL)
      return;
   list_deals.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   list_changes.Sort(SORT_BY_ORDER_TIME_OPEN_MSC);
   if(!list_changes.InsertSort(list_deals.At(0)))
      return;

//--- Orders of the first and last position deals
   CArrayObj* list_tmp=this.GetListAllOrdersByPosID(list_history,deal.PositionID());
   COrder* order_first_deal=list_tmp.At(0);
   list_tmp=CSelect::ByOrderProperty(list_tmp,ORDER_PROP_TICKET,deal.GetProperty(ORDER_PROP_DEAL_ORDER_TICKET),EQUAL);
   COrder* order_last_deal=list_tmp.At(list_tmp.Total()-1);
   if(order_first_deal==NULL || order_last_deal==NULL)
      return;
//--- Type and tickets of the first and last position deals' orders
   ENUM_ORDER_TYPE type_order_first_deal=(ENUM_ORDER_TYPE)order_first_deal.TypeOrder();
   ENUM_ORDER_TYPE type_order_last_deal=(ENUM_ORDER_TYPE)order_last_deal.TypeOrder();
   ulong ticket_order_first_deal=order_first_deal.Ticket();
   ulong ticket_order_last_deal=order_last_deal.Ticket();

//--- Current and previous positions
   COrder* position_current=list_changes.At(list_changes.Total()-1);
   COrder* position_previous=(list_changes.Total()>1 ? list_changes.At(list_changes.Total()-2) : position_current);
   if(position_current==NULL || position_previous==NULL)
      return;
   ENUM_ORDER_TYPE type_position_current=(ENUM_ORDER_TYPE)position_current.TypeOrder();
   ulong ticket_position_current=position_current.GetProperty(ORDER_PROP_DEAL_ORDER_TICKET);
   ENUM_ORDER_TYPE type_position_previous=(ENUM_ORDER_TYPE)position_previous.TypeOrder();
   ulong ticket_position_previous=position_previous.GetProperty(ORDER_PROP_DEAL_ORDER_TICKET);

//--- Get an open position by the ticket and write its volume
   COrder* position=this.GetPositionByID(list_market,deal.PositionID());
   double vol_position=(position!=NULL ? position.Volume() : 0);
//--- Executed order volume
   double vol_order_done=order_last_deal.Volume()-order_last_deal.VolumeCurrent();
//--- Remaining (unexecuted) order volume
   double vol_order_current=order_last_deal.VolumeCurrent();

//--- Enter the market
   if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_IN)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED;
      int num_deals=list_deals.Total();
      int reason=(num_deals>1 ? EVENT_REASON_VOLUME_ADD : EVENT_REASON_DONE);
      //--- If this is not the first deal in the position, add the position change flag
      if(num_deals>1)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_POSITION_CHANGED;
        }
      //--- If the order volume is opened partially, this means a partial execution
      if(order_last_deal.VolumeCurrent()>0)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
         //--- If this is not the first position deal, the volume is added by partial execution, otherwise - partial opening
         reason=(num_deals>1 ? EVENT_REASON_VOLUME_ADD_PARTIALLY : EVENT_REASON_DONE_PARTIALLY);
        }
      //--- If an opening order is a pending one, the pending order is activated
      if(order_last_deal.TypeOrder()>ORDER_TYPE_SELL && order_last_deal.TypeOrder()<ORDER_TYPE_CLOSE_BY)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
         //--- If this is not the first position deal
         if(num_deals>1)
           {
            //--- If the order is executed partially, set adding the volume to the position by pending order partial execution as an event reason,
            //--- otherwise, the volume is added to the position by executing a pending order
            reason=
              (order_last_deal.VolumeCurrent()>0 ?
               EVENT_REASON_VOLUME_ADD_BY_PENDING_PARTIALLY  :
               EVENT_REASON_VOLUME_ADD_BY_PENDING
              );
           }
         //--- If this is a new position
         else
           {
            //--- If the order is executed partially, set pending order partial execution as an event reason,
            //--- otherwise, the position is opened by activating a pending order
            reason=
              (order_last_deal.VolumeCurrent()>0 ?
               EVENT_REASON_ACTIVATED_PENDING_PARTIALLY  :
               EVENT_REASON_ACTIVATED_PENDING
              );
           }
        }
      CEvent* event=new CEventPositionOpen(this.m_trade_event_code,deal.PositionID());
      if(event!=NULL)
        {
         //--- Event deal parameters
         event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                  // Event time (position open time)
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());               // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                // Event deal ticket
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                       // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                   // Event price (position open price)
         event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                       // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                           // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,deal.Symbol());                     // Opposite position symbol
         event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                  // Position ID

         //--- Event order parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,type_order_last_deal);          // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,ticket_order_last_deal);      // Ticket of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last_deal.PositionByID());  // Opposite position ID
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last_deal.PriceClose());       // Order close price (position's last order close price)
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order_last_deal.Volume());  // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,vol_order_done);           // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,vol_order_current);         // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,deal.Magic());                       // Opposite position magic number

         //--- Position parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,type_order_first_deal);      // Type of an order that triggered the first position deal
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,ticket_order_first_deal);  // Ticket of the order that triggered the first position deal (the first position order)
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first_deal.TimeOpenMSC());  // Time of the order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first_deal.PriceOpen());        // Position first order open price
         event.SetProperty(EVENT_PROP_PRICE_SL,order_first_deal.StopLoss());           // StopLoss price (position order StopLoss price)
         event.SetProperty(EVENT_PROP_PRICE_TP,order_first_deal.TakeProfit());         // TakeProfit price (position order TakeProfit price)
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,type_position_previous);     // Position type before changing the direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,ticket_position_previous); // Position order ticket before changing the direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,type_position_current);     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,ticket_position_current); // Current position order ticket
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);          // Executed position volume

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
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Position reversal
   else if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_INOUT)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_OPENED+TRADE_EVENT_FLAG_POSITION_CHANGED+TRADE_EVENT_FLAG_POSITION_REVERSE;
      int reason=EVENT_REASON_REVERSE;
      //--- If not the entire order volume is opened, this is a partial execution
      if(order_last_deal.VolumeCurrent()>0)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
         reason=EVENT_REASON_REVERSE_PARTIALLY;
        }
      //--- If an opening order is a pending one, the pending order is activated
      if(order_last_deal.TypeOrder()>ORDER_TYPE_SELL && order_last_deal.TypeOrder()<ORDER_TYPE_CLOSE_BY)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_ORDER_ACTIVATED;
         //--- If the order is executed partially, set the position reversal by a pending order partial execution as the event reason
         reason=
           (order_last_deal.VolumeCurrent()>0 ?
            EVENT_REASON_REVERSE_BY_PENDING_PARTIALLY  :
            EVENT_REASON_REVERSE_BY_PENDING
           );
        }
      CEvent* event=new CEventPositionOpen(this.m_trade_event_code,deal.PositionID());
      if(event!=NULL)
        {
         //--- Event deal parameters
         event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                  // Event time (position open time)
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());               // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                // Event deal ticket
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                       // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                   // Event price (position open price)
         event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                       // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                           // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,deal.Symbol());                     // Opposite position symbol
         event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                  // Position ID

         //--- Event order parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,type_order_last_deal);          // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,ticket_order_last_deal);      // Ticket of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last_deal.PositionByID());  // Opposite position ID
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last_deal.PriceClose());       // Order close price (position's last order close price)
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order_last_deal.Volume());  // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,vol_order_done);           // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,vol_order_current);         // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,deal.Magic());                       // Opposite position magic number

         //--- Position parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,type_order_first_deal);      // Type of the order that triggered the first position deal
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,ticket_order_first_deal);  // Ticket of the order that triggered the first position deal (the first position order)
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first_deal.TimeOpenMSC());  // Time of the order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first_deal.PriceOpen());        // Position first order open price
         event.SetProperty(EVENT_PROP_PRICE_SL,order_first_deal.StopLoss());           // StopLoss price (position order StopLoss price)
         event.SetProperty(EVENT_PROP_PRICE_TP,order_first_deal.TakeProfit());         // TakeProfit price (position order TakeProfit price)
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,type_position_previous);     // Position order type before changing the direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,ticket_position_previous); // Position order ticket before changing the direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,type_position_current);     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,ticket_position_current); // Current position order ticket
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);          // Executed position volume

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
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
//--- Exit the market
   else if(deal.GetProperty(ORDER_PROP_DEAL_ENTRY)==DEAL_ENTRY_OUT)
     {
      this.m_trade_event_code=TRADE_EVENT_FLAG_POSITION_CLOSED;
      int reason=EVENT_REASON_DONE;
      //--- If the position with the ID is still in the market, this means partial execution
      if(this.GetPositionByID(list_market,deal.PositionID())!=NULL)
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_PARTIAL;
        }
      //--- If a closing order is executed partially, set partial execution of a closing order as an event reason
      if(order_last_deal.VolumeCurrent()>0)
        {
         reason=EVENT_REASON_DONE_PARTIALLY;
        }
      //--- If the closing flag is set to StopLoss for a position's closing order, then closing is performed by StopLoss
      //--- If a StopLoss order is executed partially, set partial StopLoss order execution as the event reason
      if(order_last_deal.IsCloseByStopLoss())
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_SL;
         reason=(order_last_deal.VolumeCurrent()>0 ? EVENT_REASON_DONE_SL_PARTIALLY : EVENT_REASON_DONE_SL);
        }
      //--- If the closing flag is set to TakeProfit for a position's closing order, then closing is performed by TakeProfit
      //--- If a TakeProfit order is executed partially, set partial TakeProfit order execution as the event reason
      else if(order_last_deal.IsCloseByTakeProfit())
        {
         this.m_trade_event_code+=TRADE_EVENT_FLAG_TP;
         reason=(order_last_deal.VolumeCurrent()>0 ? EVENT_REASON_DONE_TP_PARTIALLY : EVENT_REASON_DONE_TP);
        }
      //---
      CEvent* event=new CEventPositionClose(this.m_trade_event_code,deal.PositionID());
      if(event!=NULL)
        {
         //--- Event deal parameters
         event.SetProperty(EVENT_PROP_TIME_EVENT,deal.TimeOpenMSC());                  // Event time (position open time)
         event.SetProperty(EVENT_PROP_REASON_EVENT,reason);                            // Event reason (from the ENUM_EVENT_REASON enumeration)
         event.SetProperty(EVENT_PROP_TYPE_DEAL_EVENT,deal.TypeOrder());               // Event deal type
         event.SetProperty(EVENT_PROP_TICKET_DEAL_EVENT,deal.Ticket());                // Event deal ticket
         event.SetProperty(EVENT_PROP_MAGIC_ORDER,deal.Magic());                       // Order/deal/position magic number
         event.SetProperty(EVENT_PROP_PRICE_EVENT,deal.PriceOpen());                   // Event price (position open price)
         event.SetProperty(EVENT_PROP_PROFIT,deal.ProfitFull());                       // Profit
         event.SetProperty(EVENT_PROP_SYMBOL,deal.Symbol());                           // Order symbol
         event.SetProperty(EVENT_PROP_SYMBOL_BY_ID,deal.Symbol());                     // Opposite position symbol
         event.SetProperty(EVENT_PROP_POSITION_ID,deal.PositionID());                  // Position ID

         //--- Event order parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_EVENT,type_order_last_deal);          // Type of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_TICKET_ORDER_EVENT,ticket_order_last_deal);      // Ticket of the order that triggered an event deal (the last position order)
         event.SetProperty(EVENT_PROP_POSITION_BY_ID,order_last_deal.PositionByID());  // Opposite position ID
         event.SetProperty(EVENT_PROP_PRICE_CLOSE,order_last_deal.PriceClose());       // Order close price (position's last order close price)
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_INITIAL,order_last_deal.Volume());  // Requested order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_EXECUTED,vol_order_done);           // Executed order volume
         event.SetProperty(EVENT_PROP_VOLUME_ORDER_CURRENT,vol_order_current);         // Remaining (unexecuted) order volume
         event.SetProperty(EVENT_PROP_MAGIC_BY_ID,order_last_deal.Magic());            // Opposite position magic number

         //--- Position parameters
         event.SetProperty(EVENT_PROP_TYPE_ORDER_POSITION,type_order_first_deal);      // Type of the order that triggered the first position deal
         event.SetProperty(EVENT_PROP_TICKET_ORDER_POSITION,ticket_order_first_deal);  // Ticket of the order that triggered the first position deal (the first position order)
         event.SetProperty(EVENT_PROP_TIME_ORDER_POSITION,order_first_deal.TimeOpenMSC());  // Time of the order that triggered a position deal (the first position order)
         event.SetProperty(EVENT_PROP_PRICE_OPEN,order_first_deal.PriceOpen());        // Position first order open price
         event.SetProperty(EVENT_PROP_PRICE_SL,order_first_deal.StopLoss());           // StopLoss price (position order StopLoss price)
         event.SetProperty(EVENT_PROP_PRICE_TP,order_first_deal.TakeProfit());         // TakeProfit price (position order TakeProfit price)
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_BEFORE,type_position_previous);     // Position type before changing the direction
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_BEFORE,ticket_position_previous); // Position order ticket before changing the direction
         event.SetProperty(EVENT_PROP_TYPE_ORD_POS_CURRENT,type_position_current);     // Current position order type
         event.SetProperty(EVENT_PROP_TICKET_ORD_POS_CURRENT,ticket_position_current); // Current position order ticket
         event.SetProperty(EVENT_PROP_VOLUME_POSITION_EXECUTED,vol_position);          // Executed position volume

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
            ::Print(DFUN_ERR_LINE,TextByLanguage("Такое событие уже есть в списке","This event already in the list."));
            delete event;
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The CreateNewEvent(), NewDealEventHedge() and NewDealEventNetto() methods have identical logic and actions. So, it would be reasonable
to combine them. But so far we have done as displayed above (on a "simple-to-complex" basis). As I have already mentioned, the codes of the
classes and their methods are to be optimized later.

We have implemented the changes in the events collection class for working on hedging and netting account types. The full listing of the
class is provided in the library files attached below. The code is quite bulky.

### Testing the performance on hedging and netting accounts

To check the implemented changes, create a test EA based on the one from the [previous \\
article](https://www.mql5.com/en/articles/6211#node04).

Save it in the new \\MQL5\\Experts\\TestDoEasy\ **Part06** directory under the name **TestDoEasyPart06.mq5**.

Remove the strings checking the account type from the EA's OnInit() handler:

```
int OnInit()
  {
//--- Check account type
   if(!engine.IsHedge())
     {
      Alert(TextByLanguage("Ошибка. Счёт должен быть хеджевым","Error. Account must be hedge"));
      return INIT_FAILED;
     }
//--- set global variables
```

Instead, add calling the function of checking the validity of creating enumerations for searching and sorting by object properties:

```
int OnInit()
  {
//--- Calling the function displays the list of enumeration constants in the journal
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();
//--- set global variables
```

In order to close a position partially, a netting account requires placing a position that is opposite to the direction of the existing one
and has the volume sufficient for a partial closing. Therefore, we need to make slight corrections in the PressButtonEvents() button
pressing event handler function.

**To partially close a Buy position:**

```
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
```

Check the account. If hedging,close
part of the position, otherwise (if netting) — send a Sell position
open order with a volume equal to the half of the current Buy position
volume.

**To close part of a Sell position:**

```
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
```

Check the account. If hedging,close
part of the position, otherwise (if netting) — send a Buy position open
order with a volume equal to the half of the current Sell position
volume.

These are the necessary changes that should be implemented to make the EA work on a netting account.

**The full listing of the test EA:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart06.mq5 |
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
   BUTT_PROFIT_WITHDRAWAL
  };
#define TOTAL_BUTT   (17)
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
input uint     InpButtShiftX  =  40;   // Buttons X shift
input uint     InpButtShiftY  =  10;   // Buttons Y shift
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
//--- Calling the function displays the list of enumeration constants in the journal,
//--- (the list is set in the strings 22 and 25 of the DELib.mqh file) for checking the constants validity
   //EnumNumbersTest();
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
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
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
//---
   static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
   if(MQLInfoInteger(MQL_TESTER))
     {
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
   if(engine.LastTradeEvent()!=last_event)
     {
      Comment("\nLast trade event: ",EnumToString(engine.LastTradeEvent()));
      last_event=engine.LastTradeEvent();
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
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=30,const int shift_y=0)
  {
   int h=18,w=84,offset=2;
   int cx=offset+shift_x,cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+2*h+1;
   int x=cx,y=cy;
   int shift=0;
   for(int i=0;i<TOTAL_BUTT;i++)
     {
      x=x+(i==7 ? w+2 : 0);
      if(i==TOTAL_BUTT-3) x=cx;
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name,x,y,(i<TOTAL_BUTT-3 ? w : w*2+2),h,butt_data[i].text,(i<4 ? clrGreen : i>6 && i<11 ? clrRed : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text);
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
//| Create the button                                                |
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
      //--- Wait for 1/10 of a second
      Sleep(100);
      //--- "Unpress" the button and redraw the chart
      ButtonState(button_name,false);
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

**Compile the EA, launch it on a hedging account and try the buttons:**

![](https://c.mql5.com/2/35/2019-04-16_17-04-51.gif)

Short messages about account events are displayed in the journal, while the chart comment describes the last event that occurred on the
account.

**Now let's switch to the netting account and launch the test:**

![](https://c.mql5.com/2/35/2019-04-16_17-26-45.gif)

In this case, the journal contains entries related to position events that are possible on a netting account only — now new positions are
opened, the EA works with a single position. However, the tickets assigned to it are different. This can be seen in the beginning — after the
position reversal from Sell #2 to Buy #3.

### What's next?

Next, we will implement tracking StopLimit orders activation and prepare the functionality to track modifying orders and positions.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/6383#node00)

**Previous articles within the series:**

[Part 1. Concept, data management.](https://www.mql5.com/en/articles/5654)

[Part \\
2\. Collection of historical orders and deals.](https://www.mql5.com/en/articles/5669)

[Part 3. Collection of market orders \\
and positions, arranging the search.](https://www.mql5.com/en/articles/5687)

[Part 4. Trading events. Concept.](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. Sending events to the program.](https://www.mql5.com/en/articles/6211)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6383](https://www.mql5.com/ru/articles/6383)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6383.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/6383/mql5.zip "Download MQL5.zip")(81.21 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/317064)**
(2)


![Ugochinyere Diala](https://c.mql5.com/avatar/avatar_na2.png)

**[Ugochinyere Diala](https://www.mql5.com/en/users/ugo.chinyere)**
\|
3 Jul 2019 at 21:29

Please, can I edit my login?


![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
3 Jul 2019 at 21:45

**Ugochinyere Diala:**

Please, can I edit my login?

What are you talking about and what does it have to do with this topic??

![Library for easy and quick development of MetaTrader programs (part VII): StopLimit order activation events, preparing the functionality for order and position modification events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part VII): StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the sixth part, we trained the library to work with positions on netting accounts. Here we will implement tracking StopLimit orders activation and prepare the functionality to track order and position modification events.

![Applying OLAP in trading (part 2): Visualizing the interactive multidimensional data analysis results](https://c.mql5.com/2/36/OLAP_02__1.png)[Applying OLAP in trading (part 2): Visualizing the interactive multidimensional data analysis results](https://www.mql5.com/en/articles/6603)

In this article, we consider the creation of an interactive graphical interface for an MQL program, which is designed for the processing of account history and trading reports using OLAP techniques. To obtain a visual result, we will use maximizable and scalable windows, an adaptive layout of rubber controls and a new control for displaying diagrams. To provide the visualization functionality, we will implement a GUI with the selection of variables along coordinate axes, as well as with the selection of aggregate functions, diagram types and sorting options.

![Evaluating the ability of Fractal index and Hurst exponent to predict financial time series](https://c.mql5.com/2/36/fraktal1.png)[Evaluating the ability of Fractal index and Hurst exponent to predict financial time series](https://www.mql5.com/en/articles/6834)

Studies related to search for the fractal behavior of financial data suggest that behind the seemingly chaotic behavior of economic time series there are hidden stable mechanisms of participants' collective behavior. These mechanisms can lead to the emergence of price dynamics on the exchange, which can define and describe specific properties of price series. When applied to trading, one could benefit from the indicators which can efficiently and reliably estimate the fractal parameters in the scale and time frame, which are relevant in practice.

![Applying OLAP in trading (part 1): Online analysis of multidimensional data](https://c.mql5.com/2/36/OLAP_02.png)[Applying OLAP in trading (part 1): Online analysis of multidimensional data](https://www.mql5.com/en/articles/6602)

The article describes how to create a framework for the online analysis of multidimensional data (OLAP), as well as how to implement this in MQL and to apply such analysis in the MetaTrader environment using the example of trading account history processing.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hyofdeiqnokzmcpewalzvwxewnmdqbxw&ssn=1769186515922455167&ssn_dr=0&ssn_sr=0&fv_date=1769186515&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6383&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20VI)%3A%20Netting%20account%20events%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918651596193515&fz_uniq=5070491768413165266&sv=2552)

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