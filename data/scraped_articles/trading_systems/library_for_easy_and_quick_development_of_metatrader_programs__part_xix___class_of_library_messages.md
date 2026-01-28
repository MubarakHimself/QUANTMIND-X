---
title: Library for easy and quick development of MetaTrader programs (part XIX): Class of library messages
url: https://www.mql5.com/en/articles/7176
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:39:06.325001
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=eljlrzrgrlgrderqohkfpiivtrmasozz&ssn=1769186343176836879&ssn_dr=0&ssn_sr=0&fv_date=1769186343&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7176&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XIX)%3A%20Class%20of%20library%20messages%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918634329069910&fz_uniq=5070455510299252262&sv=2552)

MetaTrader 5 / Examples


### Contents

- [The concept of storing and displaying messages](https://www.mql5.com/en/articles/7176#node01)
- [Database of text messages](https://www.mql5.com/en/articles/7176#node02)
- [Class for displaying messages](https://www.mql5.com/en/articles/7176#node03)
- [Improving library classes](https://www.mql5.com/en/articles/7176#node04)
- [Testing the display of messages](https://www.mql5.com/en/articles/7176#node05)
- [What's next?](https://www.mql5.com/en/articles/7176#node06)


Text messages in any program may seem auxiliary and completely insignificant. However, this is not the case. Any program messages are an
integral part of communication between a user and an application regardless of the program's size and complexity.

Our library currently features the simplest method of arranging and displaying messages. The code of displaying a message in two languages
(English or Russian — depending on the terminal language) in the Experts journal is implemented directly in the appropriate location of the
library class text. This method is sufficient for the current needs. However, it becomes very inconvenient in case we need to translate
Russian texts into a language other than English or add a new language to the two existing ones since we have to search and replace all messages
scattered along multiple files.

Such a method has yet another issue: when creating multiple same-type objects containing texts, they contain identical text message,
which ultimately affects the size of the compiled program. Instead, texts repeating from object to object can be written only in a single
copy and accessed to at their location.

At the current stage of the library development, we have a sufficient number of different text messages. It is time to re-arrange the methods
of their storage, display and translation of Russian or English messages to other languages. Besides, it would be good to introduce
convenient ways of adding new languages to the library and quickly switching between them.

### The concept of storing and displaying messages

I believe, the most convenient way to store and use text messages is to create a multidimensional array of these messages, in which the first
dimension stores the index pointing at a specific message, while the second one stores the messages themselves in different languages. The
zero index of the second dimension is to store messages in a user's country language (in the current library edition, it is Russian), the
first index of the second dimension is to store messages in the international language (which is English), while all subsequent indices
contain any languages the user adds.

Thus, all library messages are to be gathered in the array, while the indices of necessary message are to be collected in the enumeration. The
second array (and, as it has turned out, a few more ones) is to store messages about errors returned directly by the terminal — trade server
return codes and program runtime error codes. The code returned by the

[GetLastError()](https://www.mql5.com/en/docs/check/getlasterror) function is used as a necessary message index for these
arrays.

The class for working with messages we are to develop here should be able to:

- display a necessary message in a selected language by a predefined message index (all library messages),

- display a message in a selected language by a trade server return code or error code from the GetLastError() function,

- display a message sent to it "as is",
- send an e-mail,
- send a push notification to a mobile device,
- send a specified file to an FTP address,
- play a specified sound,

- conveniently add messages in additional languages to each of the existing pre-defined messages by simply adding a translation in a desired
language to the database (which is a two-dimensional array of messages),
- easily switch between program languages — a user simply has to specify a necessary index of the second dimension of the array the messages in
a necessary language are located in.




### Database of text messages

Since we decided to store all pre-defined messages in the arrays and have enumeration lists with indices indicating the location of a required
message in the array, create the required enumeration for them in the \\MQL5\\Include\\DoEasy\

**Datas.mqh** library file (the first message index is to be equal to the
index the list of user errors starts from — this will allow the library message indices to avoid overlapping with the codes of the
terminal's standard messages):

```
//+------------------------------------------------------------------+
//| List of the library's text message indices                       |
//+------------------------------------------------------------------+
enum ENUM_MESSAGES_LIB
  {
   MSG_LIB_PARAMS_LIST_BEG=ERR_USER_ERROR_FIRST,      // Beginning of the parameter list
   MSG_LIB_PARAMS_LIST_END,                           // End of the parameter list
   MSG_LIB_PROP_NOT_SUPPORTED,                        // Property not supported
   MSG_LIB_PROP_NOT_SUPPORTED_MQL4,                   // Property not supported in MQL4
   MSG_LIB_PROP_NOT_SUPPORTED_POSITION,               // Property not supported for position
   MSG_LIB_PROP_NOT_SUPPORTED_PENDING,                // Property not supported for pending order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET,                 // Property not supported for market order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET_HIST,            // Property not supported for historical market order
   MSG_LIB_PROP_NOT_SET,                              // Value not set
   MSG_LIB_PROP_EMPTY,                                // Not set

   MSG_LIB_SYS_ERROR,                                 // Error
   MSG_LIB_SYS_NOT_SYMBOL_ON_SERVER,                  // Error. No such symbol on server
   MSG_LIB_SYS_FAILED_PUT_SYMBOL,                     // Failed to place to market watch. Error:
   MSG_LIB_SYS_NOT_GET_PRICE,                         // Failed to get current prices. Error:
   MSG_LIB_SYS_NOT_GET_MARGIN_RATES,                  // Failed to get margin ratios. Error:
   MSG_LIB_SYS_NOT_GET_DATAS,                         // Failed to get data

   MSG_LIB_SYS_FAILED_CREATE_STORAGE_FOLDER,          // Failed to create folder for storing files. Error:
   MSG_LIB_SYS_FAILED_ADD_ACC_OBJ_TO_LIST,            // Error. Failed to add current account object to collection list
   MSG_LIB_SYS_FAILED_CREATE_CURR_ACC_OBJ,            // Error. Failed to create account object with current account data
   MSG_LIB_SYS_FAILED_OPEN_FILE_FOR_WRITE,            // Could not open file for writing
   MSG_LIB_SYS_INPUT_ERROR_NO_SYMBOL,                 // Input error: no symbol
   MSG_LIB_SYS_FAILED_CREATE_SYM_OBJ,                 // Failed to create symbol object
   MSG_LIB_SYS_FAILED_ADD_SYM_OBJ,                    // Failed to add symbol

   MSG_LIB_SYS_NOT_GET_CURR_PRICES,                   // Failed to get current prices by event symbol
   MSG_LIB_SYS_EVENT_ALREADY_IN_LIST,                 // This event is already in the list
   MSG_LIB_SYS_ERROR_ALREADY_CREATED_COUNTER,         // Error. Counter with ID already created
   MSG_LIB_SYS_FAILED_CREATE_COUNTER,                 // Failed to create timer counter
   MSG_LIB_SYS_FAILED_CREATE_TEMP_LIST,               // Error creating temporary list
   MSG_LIB_SYS_ERROR_NOT_MARKET_LIST,                 // Error. This is not a market collection list
   MSG_LIB_SYS_ERROR_NOT_HISTORY_LIST,                // Error. This is not a history collection list
   MSG_LIB_SYS_FAILED_ADD_ORDER_TO_LIST,              // Could not add order to the list
   MSG_LIB_SYS_FAILED_ADD_DEAL_TO_LIST,               // Could not add deal to the list
   MSG_LIB_SYS_FAILED_ADD_CTRL_ORDER_TO_LIST,         // Failed to add control order
   MSG_LIB_SYS_FAILED_ADD_CTRL_POSITION_TO_LIST,      // Failed to add control position
   MSG_LIB_SYS_FAILED_ADD_MODIFIED_ORD_TO_LIST,       // Could not add modified order to the list of modified orders

   MSG_LIB_SYS_NO_TICKS_YET,                          // No ticks yet
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT,              // Could not create object structure
   MSG_LIB_SYS_FAILED_WRITE_UARRAY_TO_FILE,           // Could not write uchar array to file
   MSG_LIB_SYS_FAILED_LOAD_UARRAY_FROM_FILE,          // Could not load uchar array from file
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT_FROM_UARRAY,  // Could not create object structure from uchar array
   MSG_LIB_SYS_FAILED_SAVE_OBJ_STRUCT_TO_UARRAY,      // Failed to save object structure to uchar array, error
   MSG_LIB_SYS_ERROR_INDEX,                           // Error. "index" value should be within 0 - 3

   MSG_LIB_SYS_ERROR_EMPTY_STRING,                    // Error. Predefined symbols string empty, to be used
   MSG_LIB_SYS_FAILED_PREPARING_SYMBOLS_ARRAY,        // Failed to prepare array of used symbols. Error
   MSG_LIB_SYS_INVALID_ORDER_TYPE,                    // Invalid order type:

   MSG_LIB_SYS_ERROR_FAILED_GET_PRICE_ASK,            // Failed to get Ask price. Error
   MSG_LIB_SYS_ERROR_FAILED_GET_PRICE_BID,            // Failed to get Bid price. Error
   MSG_LIB_SYS_ERROR_FAILED_OPEN_BUY,                 // Failed to open Buy position. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_BUYLIMIT,           // Failed to set BuyLimit order. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_BUYSTOP,            // Failed to set BuyStop order. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_BUYSTOPLIMIT,       // Failed to set BuyStopLimit order. Error
   MSG_LIB_SYS_ERROR_FAILED_OPEN_SELL,                // Failed to open Sell position. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_SELLLIMIT,          // Failed to set SellLimit order. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_SELLSTOP,           // Failed to set SellStop order. Error
   MSG_LIB_SYS_ERROR_FAILED_PLACE_SELLSTOPLIMIT,      // Failed to set SellStopLimit order. Error
   MSG_LIB_SYS_ERROR_FAILED_SELECT_POS,               // Failed to select position. Error
   MSG_LIB_SYS_ERROR_POSITION_ALREADY_CLOSED,         // Position already closed
   MSG_LIB_SYS_ERROR_NOT_POSITION,                    // Error. Not a position:
   MSG_LIB_SYS_ERROR_FAILED_CLOSE_POS,                // Failed to closed position. Error
   MSG_LIB_SYS_ERROR_FAILED_SELECT_POS_BY,            // Failed to select opposite position. Error
   MSG_LIB_SYS_ERROR_POSITION_BY_ALREADY_CLOSED,      // Opposite position already closed
   MSG_LIB_SYS_ERROR_NOT_POSITION_BY,                 // Error. Opposite position is not a position:
   MSG_LIB_SYS_ERROR_FAILED_CLOSE_POS_BY,             // Failed to close position by opposite one. Error
   MSG_LIB_SYS_ERROR_FAILED_SELECT_ORD,               // Failed to select order. Error
   MSG_LIB_SYS_ERROR_ORDER_ALREADY_DELETED,           // Order already deleted
   MSG_LIB_SYS_ERROR_NOT_ORDER,                       // Error. Not an order:
   MSG_LIB_SYS_ERROR_FAILED_DELETE_ORD,               // Failed to delete order. Error
   MSG_LIB_SYS_ERROR_SELECT_CLOSED_POS_TO_MODIFY,     // Error. Closed position selected for modification:
   MSG_LIB_SYS_ERROR_FAILED_MODIFY_POS,               // Failed to modify position. Error
   MSG_LIB_SYS_ERROR_SELECT_DELETED_ORD_TO_MODIFY,    // Error. Removed order selected for modification:
   MSG_LIB_SYS_ERROR_FAILED_MODIFY_ORD,               // Failed to modify order. Error
   MSG_LIB_SYS_ERROR_CODE_OUT_OF_RANGE,               // Return code out of range of error codes

   MSG_LIB_TEXT_YES,                                  // Yes
   MSG_LIB_TEXT_NO,                                   // No
   MSG_LIB_TEXT_AND,                                  // and
   MSG_LIB_TEXT_IN,                                   // in
   MSG_LIB_TEXT_TO,                                   // to
   MSG_LIB_TEXT_OPENED,                               // Opened
   MSG_LIB_TEXT_PLACED,                               // Placed
   MSG_LIB_TEXT_DELETED,                              // Deleted
   MSG_LIB_TEXT_CLOSED,                               // Closed
   MSG_LIB_TEXT_CLOSED_BY,                            // close by
   MSG_LIB_TEXT_CLOSED_VOL,                           // Closed volume
   MSG_LIB_TEXT_AT_PRICE,                             // at price
   MSG_LIB_TEXT_ON_PRICE,                             // on price
   MSG_LIB_TEXT_TRIGGERED,                            // Triggered
   MSG_LIB_TEXT_TURNED_TO,                            // turned to
   MSG_LIB_TEXT_ADDED,                                // Added
   MSG_LIB_TEXT_SYMBOL_ON_SERVER,                     // on server
   MSG_LIB_TEXT_SYMBOL_TO_LIST,                       // to list
   MSG_LIB_TEXT_FAILED_ADD_TO_LIST,                   // failed to add to list
   MSG_LIB_TEXT_SUNDAY,                               // Sunday
   MSG_LIB_TEXT_MONDAY,                               // Monday
   MSG_LIB_TEXT_TUESDAY,                              // Tuesday
   MSG_LIB_TEXT_WEDNESDAY,                            // Wednesday
   MSG_LIB_TEXT_THURSDAY,                             // Thursday
   MSG_LIB_TEXT_FRIDAY,                               // Friday
   MSG_LIB_TEXT_SATURDAY,                             // Saturday
   MSG_LIB_TEXT_SYMBOL,                               // symbol:
   MSG_LIB_TEXT_ACCOUNT,                              // account:

   MSG_LIB_TEXT_PROP_VALUE,                           // Property value
   MSG_LIB_TEXT_INC_BY,                               // increased by
   MSG_LIB_TEXT_DEC_BY,                               // decreased by
   MSG_LIB_TEXT_MORE_THEN,                            // more than
   MSG_LIB_TEXT_LESS_THEN,                            // less than
   MSG_LIB_TEXT_EQUAL,                                // equal

   MSG_LIB_TEXT_ERROR_COUNTER_WITN_ID,                // Error. Counter with ID
   MSG_LIB_TEXT_STEP,                                 // , step
   MSG_LIB_TEXT_AND_PAUSE,                            //  and pause
   MSG_LIB_TEXT_ALREADY_EXISTS,                       // already exists

   MSG_LIB_TEXT_BASE_OBJ_UNKNOWN_EVENT,               // Base object unknown event

   MSG_LIB_TEXT_NOT_MAIL_ENABLED,                     // Sending emails disabled in terminal
   MSG_LIB_TEXT_NOT_PUSH_ENABLED,                     // Sending push notifications disabled in terminal
   MSG_LIB_TEXT_NOT_FTP_ENABLED,                      // Sending files to FTP address disabled in terminal

   MSG_LIB_TEXT_ARRAY_DATA_INTEGER_NULL,              // Controlled integer properties data array has zero size
   MSG_LIB_TEXT_NEED_SET_INTEGER_VALUE,               // You should first set the size of the array equal to the number of object integer properties
   MSG_LIB_TEXT_TODO_USE_INTEGER_METHOD,              // To do this, use the method
   MSG_LIB_TEXT_WITH_NUMBER_INTEGER_VALUE,            // with number value of integer properties of object in the parameter

   MSG_LIB_TEXT_ARRAY_DATA_DOUBLE_NULL,               // Controlled double properties data array has zero size
   MSG_LIB_TEXT_NEED_SET_DOUBLE_VALUE,                // You should first set the size of the array equal to the number of object double properties
   MSG_LIB_TEXT_TODO_USE_DOUBLE_METHOD,               // To do this, use the method
   MSG_LIB_TEXT_WITH_NUMBER_DOUBLE_VALUE,             // with number value of double properties of object in the parameter

   MSG_LIB_PROP_BID,                                  // Bid price
   MSG_LIB_PROP_ASK,                                  // Ask price
   MSG_LIB_PROP_LAST,                                 // Last deal price
   MSG_LIB_PROP_PRICE_SL,                             // StopLoss price
   MSG_LIB_PROP_PRICE_TP,                             // TakeProfit price
   MSG_LIB_PROP_PROFIT,                               // Profit
   MSG_LIB_PROP_SYMBOL,                               // Symbol
   MSG_LIB_PROP_BALANCE,                              // Balance operation
   MSG_LIB_PROP_CREDIT,                               // Credit operation
   MSG_LIB_PROP_CLOSE_BY_SL,                          // Closing by StopLoss
   MSG_LIB_PROP_CLOSE_BY_TP,                          // Closing by TakeProfit
   MSG_LIB_PROP_ACCOUNT,                              // Account

//--- COrder
   MSG_ORD_BUY,                                       // Buy
   MSG_ORD_SELL,                                      // Sell
   MSG_ORD_TO_BUY,                                    // Buy order
   MSG_ORD_TO_SELL,                                   // Sell order
   MSG_DEAL_TO_BUY,                                   // Buy deal
   MSG_DEAL_TO_SELL,                                  // Sell deal
   MSG_ORD_HISTORY,                                   // Historical order
   MSG_ORD_DEAL,                                      // Deal
   MSG_ORD_POSITION,                                  // Position
   MSG_ORD_PENDING_ACTIVE,                            // Active pending order
   MSG_ORD_PENDING,                                   // Pending order
   MSG_ORD_UNKNOWN_TYPE,                              // Unknown order type
   MSG_POS_UNKNOWN_TYPE,                              // Unknown position type
   MSG_POS_UNKNOWN_DEAL,                              // Unknown deal type
   //---
   MSG_ORD_SL_ACTIVATED,                              // Due to StopLoss
   MSG_ORD_TP_ACTIVATED,                              // Due to TakeProfit
   MSG_ORD_PLACED_FROM_MQL4,                          // Placed from mql4 program
   MSG_ORD_STATE_CANCELLED,                           // Order cancelled
   MSG_ORD_STATE_CANCELLED_CLIENT,                    // Order withdrawn by client
   MSG_ORD_STATE_STARTED,                             // Order verified but not yet accepted by broker
   MSG_ORD_STATE_PLACED,                              // Order accepted
   MSG_ORD_STATE_PARTIAL,                             // Order filled partially
   MSG_ORD_STATE_FILLED,                              // Order filled
   MSG_ORD_STATE_REJECTED,                            // Order rejected
   MSG_ORD_STATE_EXPIRED,                             // Order withdrawn upon expiration
   MSG_ORD_STATE_REQUEST_ADD,                         // Order in the state of registration (placing in the trading system)
   MSG_ORD_STATE_REQUEST_MODIFY,                      // Order in the state of modification
   MSG_ORD_STATE_REQUEST_CANCEL,                      // Order in deletion state
   MSG_ORD_STATE_UNKNOWN,                             // Unknown state
   //---
   MSG_ORD_REASON_CLIENT,                             // Order set from desktop terminal
   MSG_ORD_REASON_MOBILE,                             // Order set from mobile app
   MSG_ORD_REASON_WEB,                                // Order set from web platform
   MSG_ORD_REASON_EXPERT,                             // Order set from EA or script
   MSG_ORD_REASON_SO,                                 // Due to Stop Out
   MSG_ORD_REASON_DEAL_CLIENT,                        // Deal carried out from desktop terminal
   MSG_ORD_REASON_DEAL_MOBILE,                        // Deal carried out from mobile app
   MSG_ORD_REASON_DEAL_WEB,                           // Deal carried out from web platform
   MSG_ORD_REASON_DEAL_EXPERT,                        // Deal carried out from EA or script
   MSG_ORD_REASON_DEAL_STOPOUT,                       // Due to Stop Out
   MSG_ORD_REASON_DEAL_ROLLOVER,                      // Due to position rollover
   MSG_ORD_REASON_DEAL_VMARGIN,                       // Due to variation margin
   MSG_ORD_REASON_DEAL_SPLIT,                         // Due to split
   MSG_ORD_REASON_POS_CLIENT,                         // Position opened from desktop terminal
   MSG_ORD_REASON_POS_MOBILE,                         // Position opened from mobile app
   MSG_ORD_REASON_POS_WEB,                            // Position opened from web platform
   MSG_ORD_REASON_POS_EXPERT,                         // Position opened from EA or script
   //---
   MSG_ORD_MAGIC,                                     // Magic number
   MSG_ORD_TICKET,                                    // Ticket
   MSG_ORD_TICKET_FROM,                               // Parent order ticket
   MSG_ORD_TICKET_TO,                                 // Inherited order ticket
   MSG_ORD_TIME_EXP,                                  // Expiration date
   MSG_ORD_TYPE,                                      // Type
   MSG_ORD_TYPE_BY_DIRECTION,                         // Direction
   MSG_ORD_REASON,                                    // Reason
   MSG_ORD_POSITION_ID,                               // Position ID
   MSG_ORD_DEAL_ORDER_TICKET,                         // Deal by order ticket
   MSG_ORD_DEAL_ENTRY,                                // Deal direction
   MSG_ORD_DEAL_IN,                                   // Entry to market
   MSG_ORD_DEAL_OUT,                                  // Out from market
   MSG_ORD_DEAL_INOUT,                                // Reversal
   MSG_ORD_DEAL_OUT_BY,                               // Close by
   MSG_ORD_POSITION_BY_ID,                            // Opposite position ID
   MSG_ORD_TIME_OPEN,                                 // Open time in milliseconds
   MSG_ORD_TIME_CLOSE,                                // Close time in milliseconds
   MSG_ORD_TIME_UPDATE,                               // Position change time in milliseconds
   MSG_ORD_STATE,                                     // State
   MSG_ORD_STATUS,                                    // Status
   MSG_ORD_DISTANCE_PT,                               // Distance from price in points
   MSG_ORD_PROFIT_PT,                                 // Profit in points
   MSG_ORD_GROUP_ID,                                  // Group ID
   MSG_ORD_PRICE_OPEN,                                // Open price
   MSG_ORD_PRICE_CLOSE,                               // Close price
   MSG_ORD_PRICE_STOP_LIMIT,                          // Limit order price when StopLimit order is activated
   MSG_ORD_COMMISSION,                                // Commission
   MSG_ORD_SWAP,                                      // Swap
   MSG_ORD_VOLUME,                                    // Volume
   MSG_ORD_VOLUME_CURRENT,                            // Unfulfilled volume
   MSG_ORD_PROFIT_FULL,                               // Profit+commission+swap
   MSG_ORD_COMMENT,                                   // Comment
   MSG_ORD_COMMENT_EXT,                               // Custom comment
   MSG_ORD_EXT_ID,                                    // Exchange ID
   MSG_ORD_CLOSE_BY,                                  // Closing order

//--- CEvent
   MSG_EVN_TYPE,                                      // Event type
   MSG_EVN_TIME,                                      // Event time
   MSG_EVN_STATUS,                                    // Event status
   MSG_EVN_REASON,                                    // Event reason
   MSG_EVN_TYPE_DEAL,                                 // Deal type
   MSG_EVN_TICKET_DEAL,                               // Deal ticket
   MSG_EVN_TYPE_ORDER,                                // Event order type
   MSG_EVN_TYPE_ORDER_POSITION,                       // Position order type
   MSG_EVN_TICKET_ORDER_POSITION,                     // Position first order ticket
   MSG_EVN_TICKET_ORDER_EVENT,                        // Event order ticket
   MSG_EVN_POSITION_ID,                               // Position ID
   MSG_EVN_POSITION_BY_ID,                            // Opposite position ID
   MSG_EVN_MAGIC_BY_ID,                               // Opposite position magic number
   MSG_EVN_TIME_ORDER_POSITION,                       // Position open time
   MSG_EVN_TYPE_ORD_POS_BEFORE,                       // Position order type before changing direction
   MSG_EVN_TICKET_ORD_POS_BEFORE,                     // Position order ticket before changing direction
   MSG_EVN_TYPE_ORD_POS_CURRENT,                      // Current position order type
   MSG_EVN_TICKET_ORD_POS_CURRENT,                    // Current position order ticket
   MSG_EVN_PRICE_EVENT,                               // Price during an event
   MSG_EVN_VOLUME_ORDER_INITIAL,                      // Order initial volume
   MSG_EVN_VOLUME_ORDER_EXECUTED,                     // Executed order volume
   MSG_EVN_VOLUME_ORDER_CURRENT,                      // Remaining order volume
   MSG_EVN_VOLUME_POSITION_EXECUTED,                  // Current position volume
   MSG_EVN_PRICE_OPEN_BEFORE,                         // Price open before modification
   MSG_EVN_PRICE_SL_BEFORE,                           // StopLoss price before modification
   MSG_EVN_PRICE_TP_BEFORE,                           // TakeProfit price before modification
   MSG_EVN_PRICE_EVENT_ASK,                           // Ask price during an event
   MSG_EVN_PRICE_EVENT_BID,                           // Bid price during an event
   MSG_EVN_SYMBOL_BY_POS,                             // Opposite position symbol
   //---
   MSG_EVN_STATUS_MARKET_PENDING,                     // Pending order placed
   MSG_EVN_STATUS_MARKET_POSITION,                    // Position opened
   MSG_EVN_STATUS_HISTORY_PENDING,                    // Pending order removed
   MSG_EVN_STATUS_HISTORY_POSITION,                   // Position closed
   MSG_EVN_STATUS_UNKNOWN,                            // Unknown status
   //---
   MSG_EVN_NO_EVENT,                                  // No trading event
   MSG_EVN_PENDING_ORDER_PLASED,                      // Pending order placed
   MSG_EVN_PENDING_ORDER_REMOVED,                     // Pending order removed
   MSG_EVN_ACCOUNT_CREDIT,                            // Accruing credit

   MSG_EVN_ACCOUNT_CREDIT_WITHDRAWAL,                 // Withdrawal of credit
   MSG_EVN_ACCOUNT_CHARGE,                            // Additional charges
   MSG_EVN_ACCOUNT_CORRECTION,                        // Correcting entry
   MSG_EVN_ACCOUNT_BONUS,                             // Charging bonuses
   MSG_EVN_ACCOUNT_COMISSION,                         // Additional commissions
   MSG_EVN_ACCOUNT_COMISSION_DAILY,                   // Commission charged at the end of a day
   MSG_EVN_ACCOUNT_COMISSION_MONTHLY,                 // Commission charged at the end of a trading month
   MSG_EVN_ACCOUNT_COMISSION_AGENT_DAILY,             // Agent commission charged at the end of a trading day
   MSG_EVN_ACCOUNT_COMISSION_AGENT_MONTHLY,           // Agent commission charged at the end of a month
   MSG_EVN_ACCOUNT_INTEREST,                          // Accruing interest on free funds
   MSG_EVN_BUY_CANCELLED,                             // Canceled buy deal
   MSG_EVN_SELL_CANCELLED,                            // Canceled sell deal
   MSG_EVN_DIVIDENT,                                  // Accruing dividends
   MSG_EVN_DIVIDENT_FRANKED,                          // Accrual of franked dividend
   MSG_EVN_TAX,                                       // Tax accrual
   MSG_EVN_BALANCE_REFILL,                            // Balance refill
   MSG_EVN_BALANCE_WITHDRAWAL,                        // Withdrawing funds from an account
   MSG_EVN_ACTIVATED_PENDING,                         // Pending order activated
   MSG_EVN_ACTIVATED_PENDING_PARTIALLY,               // Pending order partial activation
   MSG_EVN_POSITION_OPENED_PARTIALLY,                 // Position opened partially
   MSG_EVN_POSITION_CLOSED_PARTIALLY,                 // Position closed partially
   MSG_EVN_POSITION_CLOSED_BY_POS,                    // Position closed by an opposite one
   MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_POS,          // Position closed partially by opposite position
   MSG_EVN_POSITION_CLOSED_BY_SL,                     // Position closed by StopLoss
   MSG_EVN_POSITION_CLOSED_BY_TP,                     // Position closed by TakeProfit
   MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_SL,           // Position closed partially by StopLoss
   MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_TP,           // Position closed partially by TakeProfit
   MSG_EVN_POSITION_REVERSED_BY_MARKET,               // Position reversal by market request
   MSG_EVN_POSITION_REVERSED_BY_PENDING,              // Position reversal by triggered a pending order
   MSG_EVN_POSITION_REVERSE_PARTIALLY,                // Position reversal by partial request execution
   MSG_EVN_POSITION_VOLUME_ADD_BY_MARKET,             // Added volume to position by market request
   MSG_EVN_POSITION_VOLUME_ADD_BY_PENDING,            // Added volume to a position by activating a pending order
   MSG_EVN_MODIFY_ORDER_PRICE,                        // Modified order price
   MSG_EVN_MODIFY_ORDER_PRICE_SL,                     // Modified order price and StopLoss
   MSG_EVN_MODIFY_ORDER_PRICE_TP,                     // Modified order price and TakeProfit
   MSG_EVN_MODIFY_ORDER_PRICE_SL_TP,                  // Modified order price, StopLoss and TakeProfit
   MSG_EVN_MODIFY_ORDER_SL_TP,                        // Modified order's StopLoss and TakeProfit
   MSG_EVN_MODIFY_ORDER_SL,                           // Modified order StopLoss
   MSG_EVN_MODIFY_ORDER_TP,                           // Modified order TakeProfit
   MSG_EVN_MODIFY_POSITION_SL_TP,                     // Modified of position StopLoss and TakeProfit
   MSG_EVN_MODIFY_POSITION_SL,                        // Modified position's StopLoss
   MSG_EVN_MODIFY_POSITION_TP,                        // Modified position's TakeProfit
   //---
   MSG_EVN_REASON_ADD,                                // Added volume to position
   MSG_EVN_REASON_ADD_PARTIALLY,                      // Volume added to the position by partially completed request
   MSG_EVN_REASON_ADD_BY_PENDING_PARTIALLY,           // Added volume to a position by partial activation of a pending order
   MSG_EVN_REASON_STOPLIMIT_TRIGGERED,                // StopLimit order triggered
   MSG_EVN_REASON_MODIFY,                             // Modification
   MSG_EVN_REASON_CANCEL,                             // Cancelation
   MSG_EVN_REASON_EXPIRED,                            // Expired
   MSG_EVN_REASON_DONE,                               // Fully completed market request
   MSG_EVN_REASON_DONE_PARTIALLY,                     // Partially completed market request
   MSG_EVN_REASON_REVERSE,                            // Position reversal
   MSG_EVN_REASON_REVERSE_BY_PENDING_PARTIALLY,       // Position reversal in case of a pending order partial execution
   MSG_EVN_REASON_DONE_SL_PARTIALLY,                  // Partial closing by StopLoss
   MSG_EVN_REASON_DONE_TP_PARTIALLY,                  // Partial closing by TakeProfit
   MSG_EVN_REASON_DONE_BY_POS,                        // Close by
   MSG_EVN_REASON_DONE_PARTIALLY_BY_POS,              // Partial closing by an opposite position
   MSG_EVN_REASON_DONE_BY_POS_PARTIALLY,              // Closed by incomplete volume of opposite position
   MSG_EVN_REASON_DONE_PARTIALLY_BY_POS_PARTIALLY,    // Closed partially by incomplete volume of opposite position

//--- CSymbol
   MSG_SYM_PROP_INDEX,                                // Index in \"Market Watch window\"
   MSG_SYM_PROP_CUSTOM,                               // Custom symbol
   MSG_SYM_PROP_CHART_MODE,                           // Price type used for generating symbols bars
   MSG_SYM_PROP_EXIST,                                // Symbol with this name exists
   MSG_SYM_PROP_SELECT,                               // Symbol selected in Market Watch
   MSG_SYM_PROP_VISIBLE,                              // Symbol visible in Market Watch
   MSG_SYM_PROP_SESSION_DEALS,                        // Number of deals in the current session
   MSG_SYM_PROP_SESSION_BUY_ORDERS,                   // Number of Buy orders at the moment
   MSG_SYM_PROP_SESSION_SELL_ORDERS,                  // Number of Sell orders at the moment
   MSG_SYM_PROP_VOLUME,                               // Volume of the last deal
   MSG_SYM_PROP_VOLUMEHIGH,                           // Maximal day volume
   MSG_SYM_PROP_VOLUMELOW,                            // Minimal day volume
   MSG_SYM_PROP_TIME,                                 // Latest quote time
   MSG_SYM_PROP_DIGITS,                               // Number of decimal places
   MSG_SYM_PROP_DIGITS_LOTS,                          // Digits after a decimal point in the value of the lot
   MSG_SYM_PROP_SPREAD,                               // Spread in points
   MSG_SYM_PROP_SPREAD_FLOAT,                         // Floating spread
   MSG_SYM_PROP_TICKS_BOOKDEPTH,                      // Maximum number of orders displayed in the Depth of Market
   MSG_SYM_PROP_TRADE_CALC_MODE,                      // Contract price calculation mode
   MSG_SYM_PROP_TRADE_MODE,                           // Order filling type
   MSG_SYM_PROP_START_TIME,                           // Symbol trading start date
   MSG_SYM_PROP_EXPIRATION_TIME,                      // Symbol trading end date
   MSG_SYM_PROP_TRADE_STOPS_LEVEL,                    // Minimal indention from the close price to place Stop orders
   MSG_SYM_PROP_TRADE_FREEZE_LEVEL,                   // Freeze distance for trading operations
   MSG_SYM_PROP_TRADE_EXEMODE,                        // Trade execution mode
   MSG_SYM_PROP_SWAP_MODE,                            // Swap calculation model
   MSG_SYM_PROP_SWAP_ROLLOVER3DAYS,                   // Triple-day swap day
   MSG_SYM_PROP_MARGIN_HEDGED_USE_LEG,                // Calculating hedging margin using the larger leg
   MSG_SYM_PROP_EXPIRATION_MODE,                      // Flags of allowed order expiration modes
   MSG_SYM_PROP_FILLING_MODE,                         // Flags of allowed order filling modes
   MSG_SYM_PROP_ORDER_MODE,                           // Flags of allowed order types
   MSG_SYM_PROP_ORDER_GTC_MODE,                       // Expiration of Stop Loss and Take Profit orders
   MSG_SYM_PROP_OPTION_MODE,                          // Option type
   MSG_SYM_PROP_OPTION_RIGHT,                         // Option right
   MSG_SYM_PROP_BACKGROUND_COLOR,                     // Background color of the symbol in Market Watch
   //---
   MSG_SYM_PROP_BIDHIGH,                              // Maximal Bid of the day
   MSG_SYM_PROP_BIDLOW,                               // Minimal Bid of the day
   MSG_SYM_PROP_ASKHIGH,                              // Maximum Ask of the day
   MSG_SYM_PROP_ASKLOW,                               // Minimal Ask of the day
   MSG_SYM_PROP_LASTHIGH,                             // Maximal Last of the day
   MSG_SYM_PROP_LASTLOW,                              // Minimal Last of the day
   MSG_SYM_PROP_VOLUME_REAL,                          // Real volume of the last deal
   MSG_SYM_PROP_VOLUMEHIGH_REAL,                      // Maximum real volume of the day
   MSG_SYM_PROP_VOLUMELOW_REAL,                       // Minimum real volume of the day
   MSG_SYM_PROP_OPTION_STRIKE,                        // Option execution price
   MSG_SYM_PROP_POINT,                                // One point value
   MSG_SYM_PROP_TRADE_TICK_VALUE,                     // Calculated tick price for a position
   MSG_SYM_PROP_TRADE_TICK_VALUE_PROFIT,              // Calculated tick value for a winning position
   MSG_SYM_PROP_TRADE_TICK_VALUE_LOSS,                // Calculated tick value for a losing position
   MSG_SYM_PROP_TRADE_TICK_SIZE,                      // Minimum price change
   MSG_SYM_PROP_TRADE_CONTRACT_SIZE,                  // Trade contract size
   MSG_SYM_PROP_TRADE_ACCRUED_INTEREST,               // Accrued interest
   MSG_SYM_PROP_TRADE_FACE_VALUE,                     // Initial bond value set by the issuer
   MSG_SYM_PROP_TRADE_LIQUIDITY_RATE,                 // Liquidity rate
   MSG_SYM_PROP_VOLUME_MIN,                           // Minimum volume for a deal
   MSG_SYM_PROP_VOLUME_MAX,                           // Maximum volume for a deal
   MSG_SYM_PROP_VOLUME_STEP,                          // Minimum volume change step for deal execution
   MSG_SYM_PROP_VOLUME_LIMIT,                         // Maximum allowed aggregate volume of an open position and pending orders in one direction
   MSG_SYM_PROP_SWAP_LONG,                            // Long swap value
   MSG_SYM_PROP_SWAP_SHORT,                           // Short swap value
   MSG_SYM_PROP_MARGIN_INITIAL,                       // Initial margin
   MSG_SYM_PROP_MARGIN_MAINTENANCE,                   // Maintenance margin for an instrument
   MSG_SYM_PROP_MARGIN_LONG_INITIAL,                  // Initial margin requirement applicable to long positions
   MSG_SYM_PROP_MARGIN_SHORT_INITIAL,                 // Initial margin requirement applicable to short positions
   MSG_SYM_PROP_MARGIN_LONG_MAINTENANCE,              // Maintenance margin requirement applicable to long positions
   MSG_SYM_PROP_MARGIN_SHORT_MAINTENANCE,             // Maintenance margin requirement applicable to short positions
   MSG_SYM_PROP_MARGIN_BUY_STOP_INITIAL,              // Initial margin requirement applicable to BuyStop orders
   MSG_SYM_PROP_MARGIN_BUY_LIMIT_INITIAL,             // Initial margin requirement applicable to BuyLimit orders
   MSG_SYM_PROP_MARGIN_BUY_STOPLIMIT_INITIAL,         // Initial margin requirement applicable to BuyStopLimit orders
   MSG_SYM_PROP_MARGIN_SELL_STOP_INITIAL,             // Initial margin requirement applicable to SellStop orders
   MSG_SYM_PROP_MARGIN_SELL_LIMIT_INITIAL,            // Initial margin requirement applicable to SellLimit orders
   MSG_SYM_PROP_MARGIN_SELL_STOPLIMIT_INITIAL,        // Initial margin requirement applicable to SellStopLimit orders
   MSG_SYM_PROP_MARGIN_BUY_STOP_MAINTENANCE,          // Maintenance margin requirement applicable to BuyStop orders
   MSG_SYM_PROP_MARGIN_BUY_LIMIT_MAINTENANCE,         // Maintenance margin requirement applicable to BuyLimit orders
   MSG_SYM_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE,     // Maintenance margin requirement applicable to BuyStopLimit orders
   MSG_SYM_PROP_MARGIN_SELL_STOP_MAINTENANCE,         // Maintenance margin requirement applicable to SellStop orders
   MSG_SYM_PROP_MARGIN_SELL_LIMIT_MAINTENANCE,        // Maintenance margin requirement applicable to SellLimit orders
   MSG_SYM_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE,    // Maintenance margin requirement applicable to SellStopLimit orders
   MSG_SYM_PROP_SESSION_VOLUME,                       // Summary volume of the current session deals
   MSG_SYM_PROP_SESSION_TURNOVER,                     // Summary turnover of the current session
   MSG_SYM_PROP_SESSION_INTEREST,                     // Summary open interest
   MSG_SYM_PROP_SESSION_BUY_ORDERS_VOLUME,            // Current volume of Buy orders
   MSG_SYM_PROP_SESSION_SELL_ORDERS_VOLUME,           // Current volume of Sell orders
   MSG_SYM_PROP_SESSION_OPEN,                         // Open price of the current session
   MSG_SYM_PROP_SESSION_CLOSE,                        // Close price of the current session
   MSG_SYM_PROP_SESSION_AW,                           // Average weighted session price
   MSG_SYM_PROP_SESSION_PRICE_SETTLEMENT,             // Settlement price of the current session
   MSG_SYM_PROP_SESSION_PRICE_LIMIT_MIN,              // Minimum session price
   MSG_SYM_PROP_SESSION_PRICE_LIMIT_MAX,              // Maximum session price
   MSG_SYM_PROP_MARGIN_HEDGED,                        // Size of a contract or margin for one lot of hedged positions
   //---
   MSG_SYM_PROP_NAME,                                 // Symbol name
   MSG_SYM_PROP_BASIS,                                // Underlying asset of derivative
   MSG_SYM_PROP_CURRENCY_BASE,                        // Basic currency of symbol
   MSG_SYM_PROP_CURRENCY_PROFIT,                      // Profit currency
   MSG_SYM_PROP_CURRENCY_MARGIN,                      // Margin currency
   MSG_SYM_PROP_BANK,                                 // Feeder of the current quote
   MSG_SYM_PROP_DESCRIPTION,                          // Symbol description
   MSG_SYM_PROP_FORMULA,                              // Formula used for custom symbol pricing
   MSG_SYM_PROP_ISIN,                                 // Symbol name in ISIN system
   MSG_SYM_PROP_PAGE,                                 // Address of web page containing symbol information
   MSG_SYM_PROP_PATH,                                 // Location in symbol tree
   //---
   MSG_SYM_STATUS_FX,                                 // Forex symbol
   MSG_SYM_STATUS_FX_MAJOR,                           // Major Forex symbol
   MSG_SYM_STATUS_FX_MINOR,                           // Minor Forex symbol
   MSG_SYM_STATUS_FX_EXOTIC,                          // Exotic Forex symbol
   MSG_SYM_STATUS_FX_RUB,                             // Forex symbol/RUB
   MSG_SYM_STATUS_METAL,                              // Metal
   MSG_SYM_STATUS_INDEX,                              // Index
   MSG_SYM_STATUS_INDICATIVE,                         // Indicative
   MSG_SYM_STATUS_CRYPTO,                             // Cryptocurrency symbol
   MSG_SYM_STATUS_COMMODITY,                          // Commodity symbol
   MSG_SYM_STATUS_EXCHANGE,                           // Exchange symbol
   MSG_SYM_STATUS_FUTURES,                            // Futures
   MSG_SYM_STATUS_CFD,                                // CFD
   MSG_SYM_STATUS__STOCKS,                            // Security
   MSG_SYM_STATUS_BONDS,                              // Bond
   MSG_SYM_STATUS_OPTION,                             // Option
   MSG_SYM_STATUS_COLLATERAL,                         // Non-tradable asset
   MSG_SYM_STATUS_CUSTOM,                             // Custom symbol
   MSG_SYM_STATUS_COMMON,                             // General group symbol
   //---
   MSG_SYM_CHART_MODE_BID,                            // Bars are based on Bid prices
   MSG_SYM_CHART_MODE_LAST,                           // Bars are based on Last prices
   MSG_SYM_CALC_MODE_FOREX,                           // Forex mode
   MSG_SYM_CALC_MODE_FOREX_NO_LEVERAGE,               // Forex No Leverage mode
   MSG_SYM_CALC_MODE_FUTURES,                         // Futures mode
   MSG_SYM_CALC_MODE_CFD,                             // CFD mode
   MSG_SYM_CALC_MODE_CFDINDEX,                        // CFD index mode
   MSG_SYM_CALC_MODE_CFDLEVERAGE,                     // CFD Leverage mode
   MSG_SYM_CALC_MODE_EXCH_STOCKS,                     // Exchange mode
   MSG_SYM_CALC_MODE_EXCH_FUTURES,                    // Futures mode
   MSG_SYM_CALC_MODE_EXCH_FUTURES_FORTS,              // FORTS Futures mode
   MSG_SYM_CALC_MODE_EXCH_BONDS,                      // Exchange Bonds mode
   MSG_SYM_CALC_MODE_EXCH_STOCKS_MOEX,                // Exchange MOEX Stocks mode
   MSG_SYM_CALC_MODE_EXCH_BONDS_MOEX,                 // Exchange MOEX Bonds mode
   MSG_SYM_CALC_MODE_SERV_COLLATERAL,                 // Collateral mode
   MSG_SYM_MODE_UNKNOWN,                              // Unknown mode
   //---
   MSG_SYM_TRADE_MODE_DISABLED,                       // Trade disabled for symbol
   MSG_SYM_TRADE_MODE_LONGONLY,                       // Only long positions allowed
   MSG_SYM_TRADE_MODE_SHORTONLY,                      // Only short positions allowed
   MSG_SYM_TRADE_MODE_CLOSEONLY,                      // Enable close only
   MSG_SYM_TRADE_MODE_FULL,                           // No trading limitations
   //---
   MSG_SYM_TRADE_EXECUTION_REQUEST,                   // Execution by request
   MSG_SYM_TRADE_EXECUTION_INSTANT,                   // Instant execution
   MSG_SYM_TRADE_EXECUTION_MARKET,                    // Market execution
   MSG_SYM_TRADE_EXECUTION_EXCHANGE,                  // Exchange execution
   //---
   MSG_SYM_SWAP_MODE_DISABLED,                        // No swaps
   MSG_SYM_SWAP_MODE_POINTS,                          // Swaps charged in points
   MSG_SYM_SWAP_MODE_CURRENCY_SYMBOL,                 // Swaps charged in money in symbol base currency
   MSG_SYM_SWAP_MODE_CURRENCY_MARGIN,                 // Swaps charged in money in symbol margin currency
   MSG_SYM_SWAP_MODE_CURRENCY_DEPOSIT,                // Swaps charged in money in client deposit currency
   MSG_SYM_SWAP_MODE_INTEREST_CURRENT,                // Swaps charged as specified annual interest from symbol price at calculation of swap
   MSG_SYM_SWAP_MODE_INTEREST_OPEN,                   // Swaps charged as specified annual interest from position open price
   MSG_SYM_SWAP_MODE_REOPEN_CURRENT,                  // Swaps charged by reopening positions by close price
   MSG_SYM_SWAP_MODE_REOPEN_BID,                      // Swaps charged by reopening positions by the current Bid price
   //---
   MSG_SYM_ORDERS_GTC,                                // Pending orders and Stop Loss/Take Profit levels valid for unlimited period until their explicit cancellation
   MSG_SYM_ORDERS_DAILY,                              // At the end of the day, all Stop Loss and Take Profit levels, as well as pending orders deleted
   MSG_SYM_ORDERS_DAILY_EXCLUDING_STOPS,              // At the end of the day, only pending orders deleted, while Stop Loss and Take Profit levels preserved
   //---
   MSG_SYM_OPTION_MODE_EUROPEAN,                      // European option may only be exercised on specified date
   MSG_SYM_OPTION_MODE_AMERICAN,                      // American option may be exercised on any trading day or before expiry
   MSG_SYM_OPTION_MODE_UNKNOWN,                       // Unknown option type
   MSG_SYM_OPTION_RIGHT_CALL,                         // Call option gives you right to buy asset at specified price
   MSG_SYM_OPTION_RIGHT_PUT,                          // Put option gives you right to sell asset at specified price
   //---
   MSG_SYM_MARKET_ORDER_ALLOWED_YES,                  // Market order (Yes)
   MSG_SYM_MARKET_ORDER_ALLOWED_NO,                   // Market order (No)
   MSG_SYM_LIMIT_ORDER_ALLOWED_YES,                   // Limit order (Yes)
   MSG_SYM_LIMIT_ORDER_ALLOWED_NO,                    // Limit order (No)
   MSG_SYM_STOP_ORDER_ALLOWED_YES,                    // Stop order (Yes)
   MSG_SYM_STOP_ORDER_ALLOWED_NO,                     // Stop order (No)
   MSG_SYM_STOPLIMIT_ORDER_ALLOWED_YES,               // Stop limit order (Yes)
   MSG_SYM_STOPLIMIT_ORDER_ALLOWED_NO,                // Stop limit order (No)
   MSG_SYM_STOPLOSS_ORDER_ALLOWED_YES,                // StopLoss (Yes)
   MSG_SYM_STOPLOSS_ORDER_ALLOWED_NO,                 // StopLoss (No)
   MSG_SYM_TAKEPROFIT_ORDER_ALLOWED_YES,              // TakeProfit (Yes)
   MSG_SYM_TAKEPROFIT_ORDER_ALLOWED_NO,               // TakeProfit (No)
   MSG_SYM_CLOSEBY_ORDER_ALLOWED_YES,                 // Close by (Yes)
   MSG_SYM_CLOSEBY_ORDER_ALLOWED_NO,                  // Close by (No)
   MSG_SYM_FILLING_MODE_RETURN_YES,                   // Return (Yes)
   MSG_SYM_FILLING_MODE_FOK_YES,                      // Fill or Kill (Yes)
   MSG_SYM_FILLING_MODE_FOK_NO,                       // Fill or Kill (No)
   MSG_SYM_FILLING_MODE_IOK_YES,                      // Immediate or Cancel order (Yes)
   MSG_SYM_FILLING_MODE_IOK_NO,                       // Immediate or Cancel order (No)
   MSG_SYM_EXPIRATION_MODE_GTC_YES,                   // Unlimited (Yes)
   MSG_SYM_EXPIRATION_MODE_GTC_NO,                    // Unlimited (No)
   MSG_SYM_EXPIRATION_MODE_DAY_YES,                   // Valid till the end of the day (Yes)
   MSG_SYM_EXPIRATION_MODE_DAY_NO,                    // Valid till the end of the day (No)
   MSG_SYM_EXPIRATION_MODE_SPECIFIED_YES,             // Time is specified in the order (Yes)
   MSG_SYM_EXPIRATION_MODE_SPECIFIED_NO,              // Time is specified in the order (No)
   MSG_SYM_EXPIRATION_MODE_SPECIFIED_DAY_YES,         // Date specified in order (Yes)
   MSG_SYM_EXPIRATION_MODE_SPECIFIED_DAY_NO,          // Date specified in order (No)

   MSG_SYM_EVENT_SYMBOL_ADD,                          // Added symbol to Market Watch window
   MSG_SYM_EVENT_SYMBOL_DEL,                          // Symbol removed from Market Watch window
   MSG_SYM_EVENT_SYMBOL_SORT,                         // Changed location of symbols in Market Watch window
   MSG_SYM_SYMBOLS_MODE_CURRENT,                      // Work with current symbol only
   MSG_SYM_SYMBOLS_MODE_DEFINES,                      // Work with predefined symbol list
   MSG_SYM_SYMBOLS_MODE_MARKET_WATCH,                 // Work with Market Watch window symbols
   MSG_SYM_SYMBOLS_MODE_ALL,                          // Work with full list of all available symbols

//--- CAccount
   MSG_ACC_PROP_LOGIN,                                // Account number
   MSG_ACC_PROP_TRADE_MODE,                           // Trading account type
   MSG_ACC_PROP_LEVERAGE,                             // Leverage
   MSG_ACC_PROP_LIMIT_ORDERS,                         // Maximum allowed number of active pending orders
   MSG_ACC_PROP_MARGIN_SO_MODE,                       // Mode of setting the minimum available margin level
   MSG_ACC_PROP_TRADE_ALLOWED,                        // Trading permission of the current account
   MSG_ACC_PROP_TRADE_EXPERT,                         // Trading permission of an EA
   MSG_ACC_PROP_MARGIN_MODE,                          // Margin calculation mode
   MSG_ACC_PROP_CURRENCY_DIGITS,                      // Number of decimal places for the account currency
   MSG_ACC_PROP_SERVER_TYPE,                          // Trade server type
   //---
   MSG_ACC_PROP_BALANCE,                              // Account balance
   MSG_ACC_PROP_CREDIT,                               // Account credit
   MSG_ACC_PROP_PROFIT,                               // Current profit of an account
   MSG_ACC_PROP_EQUITY,                               // Account equity
   MSG_ACC_PROP_MARGIN,                               // Account margin used in deposit currency
   MSG_ACC_PROP_MARGIN_FREE,                          // Free margin of account
   MSG_ACC_PROP_MARGIN_LEVEL,                         // Margin level on an account in %
   MSG_ACC_PROP_MARGIN_SO_CALL,                       // Margin call level
   MSG_ACC_PROP_MARGIN_SO_SO,                         // Margin stop out level
   MSG_ACC_PROP_MARGIN_INITIAL,                       // Amount reserved on account to cover margin of all pending orders
   MSG_ACC_PROP_MARGIN_MAINTENANCE,                   // Min equity reserved on account to cover min amount of all open positions
   MSG_ACC_PROP_ASSETS,                               // Current assets on an account
   MSG_ACC_PROP_LIABILITIES,                          // Current liabilities on an account
   MSG_ACC_PROP_COMMISSION_BLOCKED,                   // Sum of blocked commissions on an account
   //---
   MSG_ACC_PROP_NAME,                                 // Client name
   MSG_ACC_PROP_SERVER,                               // Trade server name
   MSG_ACC_PROP_CURRENCY,                             // Deposit currency
   MSG_ACC_PROP_COMPANY,                              // Name of a company serving an account
   //---
   MSG_ACC_TRADE_MODE_DEMO,                           // Demo account
   MSG_ACC_TRADE_MODE_CONTEST,                        // Contest account
   MSG_ACC_TRADE_MODE_REAL,                           // Real account
   MSG_ACC_TRADE_MODE_UNKNOWN,                        // Unknown account type
   //---
   MSG_ACC_STOPOUT_MODE_PERCENT,                      // Account stop out mode in %
   MSG_ACC_STOPOUT_MODE_MONEY,                        // Account stop out mode in money
   MSG_ACC_MARGIN_MODE_RETAIL_NETTING,                // Netting mode
   MSG_ACC_MARGIN_MODE_RETAIL_HEDGING,                // Hedging mode
   MSG_ACC_MARGIN_MODE_RETAIL_EXCHANGE,               // Exchange markets mode

//--- CEngine
   MSG_ENG_NO_TRADE_EVENTS,                           // There have been no trade events since the last launch of EA
   MSG_ENG_FAILED_GET_LAST_TRADE_EVENT_DESCR,         // Failed to get description of the last trading event

  };
//---
```

As we can see, the enumeration features all existing library messages divided into groups by words the texts are created from, as well as
messages from different library classes. As we create other library classes, we will add new constants describing library messages to the
enumeration list. A message to be displayed in the journal is now to be set by the name of the constant corresponding to the message whose text
should be received from the array.

Now we need to create two-dimensional arrays so that each message from the above enumeration corresponds to the location of the
corresponding text in the array (first dimension), while the messages themselves are to be located in the second dimension — the zero index
contains the message in user's country language (in the current version, it is Russian), while the first index contains the message in
English.

**Let's write the array of all library messages and add the macro**
**substitution specifying the number of used languages for more convenience:**

```
//---
#define TOTAL_LANG      (2)                           // Number of used languages
//+------------------------------------------------------------------+
//| Array of predefined library messages                             |
//| (1) in user's country language                                   |
//| (2) in the international language (English)                      |
//| (3) any additional language.                                     |
//|  The default languages are English and Russian.                  |
//|  To add the necessary number of other languages, simply          |
//|  set the total number of used languages in TOTAL_LANG            |
//|  and add the necessary translation after the English text        |
//+------------------------------------------------------------------+
string messages_library[][TOTAL_LANG]=
  {
   {"Начало списка параметров","Beginning of the event parameter list"},
   {"Конец списка параметров","End of the parameter list"},
   {"Свойство не поддерживается","Property not supported"},
   {"Свойство не поддерживается в MQL4","Property not supported in MQL4"},
   {"Свойство не поддерживается у позиции","Property not supported for position"},
   {"Свойство не поддерживается у отложенного ордера","Property not supported for pending order"},
   {"Свойство не поддерживается у маркет-ордера","Property not supported for market order"},
   {"Свойство не поддерживается у исторического маркет-ордера","Property not supported for historical market order"},
   {"Значение не задано","Value not set"},
   {"Отсутствует","Not set"},

   {"Ошибка ","Error "},
   {"Ошибка. Такого символа нет на сервере","Error. No such symbol on server"},
   {"Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in market watch. Error: "},
   {"Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "},
   {"Не удалось получить коэффициенты взимания маржи. Ошибка: ","Failed to get margin rates. Error: "},
   {"Не удалось получить данные ","Failed to get data of "},

   {"Не удалось создать папку хранения файлов. Ошибка: ","Could not create file storage folder. Error: "},
   {"Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию","Error. Failed to add current account object to collection list"},
   {"Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта","Error. Failed to create an account object with current account data"},
   {"Не удалось открыть для записи файл ","Could not open file for writing: "},
   {"Ошибка входных данных: нет символа ","Input error: no "},
   {"Не удалось создать объект-символ ","Failed to create symbol object "},
   {"Не удалось добавить символ ","Failed to add "},

   {"Не удалось получить текущие цены по символу события ","Failed to get current prices by event symbol "},
   {"Такое событие уже есть в списке","This event already in the list"},

   {"Ошибка. Уже создан счётчик с идентификатором ","Error. Already created a counter with id "},
   {"Не удалось создать счётчик таймера ","Failed to create timer counter "},

   {"Ошибка создания временного списка","Error creating temporary list"},
   {"Ошибка. Список не является списком рыночной коллекции","Error. The list is not a list of market collection"},
   {"Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of history collection"},
   {"Не удалось добавить ордер в список","Could not add order to list"},
   {"Не удалось добавить сделку в список","Could not add deal to list"},
   {"Не удалось добавить контрольный ордер ","Failed to add control order "},
   {"Не удалось добавить контрольую позицию ","Failed to add control position "},
   {"Не удалось добавить модифицированный ордер в список изменённых ордеров","Could not add modified order to list of modified orders"},

   {"Ещё не было тиков","No ticks yet"},
   {"Не удалось создать структуру объекта","Could not create object structure"},
   {"Не удалось записать uchar-массив в файл","Could not write uchar array to file"},
   {"Не удалось загрузить uchar-массив из файла","Could not load uchar array from file"},
   {"Не удалось создать структуру объекта из uchar-массива","Could not create object structure from uchar array"},
   {"Не удалось сохранить структуру объекта в uchar-массив, ошибка ","Failed to save object structure to uchar array, error "},
   {"Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"},

   {"Ошибка. Строка предопределённых символов пустая, будет использоваться ","Error. String of predefined symbols is empty, symbol will be used: "},
   {"Не удалось подготовить массив используемых символов. Ошибка ","Failed to create array of used characters. Error "},
   {"Неправильный тип ордера: ","Invalid order type: "},

   {"Не удалось получить цену Ask. Ошибка ","Could not get Ask price. Error "},
   {"Не удалось получить цену Bid. Ошибка ","Could not get Bid price. Error "},
   {"Не удалось открыть позицию Buy. Ошибка ","Failed to open Buy position. Error "},
   {"Не удалось установить ордер BuyLimit. Ошибка ","Could not place BuyLimit order. Error "},
   {"Не удалось установить ордер BuyStop. Ошибка ","Could not place BuyStop order. Error "},
   {"Не удалось установить ордер BuyStopLimit. Ошибка ","Could not place BuyStopLimit order. Error "},

   {"Не удалось открыть позицию Sell. Ошибка ","Failed to open Sell position. Error "},
   {"Не удалось установить ордер SellLimit. Ошибка ","Could not place SellLimit order. Error "},
   {"Не удалось установить ордер SellStop. Ошибка ","Could not place SellStop order. Error "},
   {"Не удалось установить ордер SellStopLimit. Ошибка ","Could not place SellStopLimit order. Error "},

   {"Не удалось выбрать позицию. Ошибка ","Could not select position. Error "},
   {"Позиция уже закрыта","Position already closed"},
   {"Ошибка. Не позиция: ","Error. Not position: "},
   {"Не удалось закрыть позицию. Ошибка ","Could not close position. Error "},
   {"Не удалось выбрать встречную позицию. Ошибка ","Could not select opposite position. Error "},
   {"Встречная позиция уже закрыта","Opposite position already closed"},
   {"Ошибка. Встречная позиция не является позицией: ","Error. Opposite position is not a position: "},
   {"Не удалось закрыть позицию встречной. Ошибка ","Could not close position by opposite position. Error "},
   {"Не удалось выбрать ордер. Ошибка ","Could not select order. Error "},
   {"Ордер уже удалён","Order already deleted"},
   {"Ошибка. Не ордер: ","Error. Not order: "},
   {"Не удалось удалить ордер. Ошибка ","Could not delete order. Error "},
   {"Ошибка. Для модификации выбрана закрытая позиция: ","Error. Closed position selected for modification: "},
   {"Не удалось модифицировать позицию. Ошибка ","Failed to modify position. Error "},
   {"Ошибка. Для модификации выбран удалённый ордер: ","Error. Deleted order selected for modification: "},
   {"Не удалось модифицировать ордер. Ошибка ","Failed to order modify. Error "},
   {"Код возврата вне заданного диапазона кодов ошибок","Return code out of range of error codes"},

   {"Да","Yes"},
   {"Нет","No"},
   {"и","and"},
   {"в","in"},
   {"к","to"},

   {"Открыт","Opened"},
   {"Установлен","Placed"},
   {"Удалён","Deleted"},
   {"Закрыт","Closed"},
   {"встречным","by opposite"},
   {"Закрыт объём","Closed volume"},
   {"по цене","at price"},
   {"на цену","on price"},
   {"Сработал","Triggered"},
   {"изменён на","turned to"},
   {"Добавлено","Added"},
   {" на сервере"," symbol on server"},
   {" в список"," symbol to list"},
   {"не удалось добавить в список","failed to add to list"},

   {"Воскресение","Sunday"},
   {"Понедельник","Monday"},
   {"Вторник","Tuesday"},
   {"Среда","Wednesday"},
   {"Четверг","Thursday"},
   {"Пятница","Friday"},
   {"Суббота","Saturday"},
   {"символа: ","symbol property: "},
   {"аккаунта: ","account property: "},

   {"Значение свойства ","Value of the "},
   {" увеличено на "," increased by "},
   {" уменьшено на "," decreased by "},
   {" стало больше "," became more than "},
   {" стало меньше "," became less than "},
   {" равно "," equal to "},

   {"Ошибка. Счётчик с идентификатором ","Error. Counter with ID "},
   {" шагом ",", step "},
   {" и паузой "," and pause "},
   {" уже существует"," already exists"},

   {"Неизвестное событие базового объекта ","Unknown event of base object "},

   {"В терминале нет разрешения на отправку e-mail","Terminal does not have permission to send e-mails"},
   {"В терминале нет разрешения на отправку Push-уведомлений","Terminal does not have permission to send push notifications"},
   {"В терминале нет разрешения на отправку файлов на FTP-адрес","Terminal does not have permission to send files to FTP address"},

   {"Массив данных контролируемых integer-свойств имеет нулевой размер","Controlled integer properties data array has zero size"},
   {"Необходимо сначала установить размер массива равным количеству integer-свойств объекта","You should first set size of array equal to number of object integer properties"},
   {"Для этого используйте метод CBaseObj::SetControlDataArraySizeLong()","To do this, use CBaseObj::SetControlDataArraySizeLong() method"},
   {"со значением количества integer-свойств объекта в параметре \"size\"","with value of number of integer properties of object in \"size\" parameter"},

   {"Массив данных контролируемых double-свойств имеет нулевой размер","Controlled double properties data array has zero size"},
   {"Необходимо сначала установить размер массива равным количеству double-свойств объекта","You should first set size of array equal to number of object double properties"},
   {"Для этого используйте метод CBaseObj::SetControlDataArraySizeDouble()","To do this, use CBaseObj::SetControlDataArraySizeDouble() method"},
   {"со значением количества double-свойств объекта в параметре \"size\"","with value of number of double properties of object in \"size\" parameter"},

   {"Цена Bid","Bid price"},
   {"Цена Ask","Ask price"},
   {"Цена Last","Last price"},
   {"Цена StopLoss","StopLoss price"},
   {"Цена TakeProfit","TakeProfit price"},
   {"Прибыль","Profit"},
   {"Символ","Symbol"},
   {"Балансовая операция","Balance operation"},
   {"Кредитная операция","Credit operation"},
   {"Закрытие по StopLoss","Close by StopLoss"},
   {"Закрытие по TakeProfit","Close by TakeProfit"},
   {"Счёт","Account"},

//--- COrder
   {"Buy","Buy"},
   {"Sell","Sell"},
   {"Ордер на покупку","Buy order"},
   {"Ордер на продажу","Sell order"},

   {"Сделка на покупку","Buy deal"},
   {"Сделка на продажу","Sell deal"},

   {"Исторический ордер","History order"},
   {"Сделка","Deal"},
   {"Позиция","Active position"},
   {"Установленный отложенный ордер","Active pending order"},
   {"Отложенный ордер","Pending order"},
   {"Неизвестный тип ордера","Unknown order type"},
   {"Неизвестный тип позиции","Unknown position type"},
   {"Неизвестный тип сделки","Unknown deal type"},
   //---
   {"Срабатывание StopLoss","Due to StopLoss"},
   {"Срабатывание TakeProfit","Due to TakeProfit"},
   {"Выставлен из mql4-программы","Placed from mql4 program"},
   {"Ордер отменён","Order cancelled"},
   {"Ордер снят клиентом","Order withdrawn by client"},
   {"Ордер проверен на корректность, но еще не принят брокером","Order verified but not yet accepted by broker"},
   {"Ордер принят","Order accepted"},
   {"Ордер выполнен частично","Order filled partially"},
   {"Ордер выполнен полностью","Order filled in full"},
   {"Ордер отклонен","Order rejected"},
   {"Ордер снят по истечении срока его действия","Order withdrawn upon expiration"},
   {"Ордер в состоянии регистрации (выставление в торговую систему)","Order in the state of registration (placing in trading system)"},
   {"Ордер в состоянии модификации","Order in the state of modification"},
   {"Ордер в состоянии удаления","Order in deletion state"},
   {"Неизвестное состояние","Unknown state"},
   //---
   {"Ордер выставлен из десктопного терминала","Order set from desktop terminal"},
   {"Ордер выставлен из мобильного приложения","Order set from mobile app"},
   {"Ордер выставлен из веб-платформы","Order set from web platform"},
   {"Ордер выставлен советником или скриптом","Order set from EA or script"},
   {"Ордер выставлен в результате наступления Stop Out","Due to Stop Out"},
   {"Сделка проведена из десктопного терминала","Deal carried out from desktop terminal"},
   {"Сделка проведена из мобильного приложения","Deal carried out from mobile app"},
   {"Сделка проведена из веб-платформы","Deal carried out from web platform"},
   {"Сделка проведена из советника или скрипта","Deal carried out from EA or script"},
   {"Сделка проведена в результате наступления Stop Out","Due to Stop Out"},
   {"Сделка проведена по причине переноса позиции","Due to position rollover"},
   {"Сделка проведена по причине начисления/списания вариационной маржи","Due to variation margin"},
   {"Сделка проведена по причине сплита (понижения цены) инструмента","Due to split"},
   {"Позиция открыта из десктопного терминала","Position open from desktop terminal"},
   {"Позиция открыта из мобильного приложения","Position open from mobile app"},
   {"Позиция открыта из веб-платформы","Position open from web platform"},
   {"Позиция открыта из советника или скрипта","Position opened from EA or script"},
   //---
   {"Магический номер","Magic number"},
   {"Тикет","Ticket"},
   {"Тикет родительского ордера","Ticket of parent order"},
   {"Тикет наследуемого ордера","Inherited order ticket"},
   {"Дата экспирации","Date of expiration"},
   {"Тип","Type"},
   {"Тип по направлению","Type by direction"},
   {"Причина","Reason"},
   {"Идентификатор позиции","Position identifier"},
   {"Сделка на основании ордера с тикетом","Deal by order ticket"},
   {"Направление сделки","Deal entry"},
   {"Вход в рынок","Entry to market"},
   {"Выход из рынка","Out from market"},
   {"Разворот","Reversal"},
   {"Закрытие встречной позицией","Closing by opposite position"},
   {"Идентификатор встречной позиции","Opposite position identifier"},
   {"Время открытия в милисекундах","Opening time in milliseconds"},
   {"Время закрытия в милисекундах","Closing time in milliseconds"},
   {"Время изменения позиции в милисекундах","Position change time in milliseconds"},
   {"Состояние","Statе"},
   {"Статус","Status"},
   {"Дистанция от цены в пунктах","Distance from price in points"},
   {"Прибыль в пунктах","Profit in points"},
   {"Идентификатор группы","Group's identifier"},
   {"Цена открытия","Price open"},
   {"Цена закрытия","Price close"},
   {"Цена постановки Limit ордера при активации StopLimit ордера","Price of placing Limit order when StopLimit order activated"},
   {"Комиссия","Comission"},
   {"Своп","Swap"},
   {"Объём","Volume"},
   {"Невыполненный объём","Unfulfilled volume"},
   {"Прибыль+комиссия+своп","Profit+Comission+Swap"},
   {"Комментарий","Comment"},
   {"Пользовательский комментарий","Custom comment"},
   {"Идентификатор на бирже","Exchange identifier"},
   {"Закрывающий ордер","Order for closing by"},

//--- CEvent
   {"Тип события","Event's type"},
   {"Время события","Time of event"},
   {"Статус события","Status of event"},
   {"Причина события","Reason of event"},
   {"Тип сделки","Deal's type"},
   {"Тикет сделки","Deal's ticket"},
   {"Тип ордера события","Event's order type"},
   {"Тип ордера позиции","Position's order type"},
   {"Тикет первого ордера позиции","Position's first order ticket"},
   {"Тикет ордера события","Event's order ticket"},
   {"Идентификатор позиции","Position ID"},
   {"Идентификатор встречной позиции","Opposite position's ID"},
   {"Магический номер встречной позиции","Magic number of opposite position"},
   {"Время открытия позиции","Position's opened time"},
   {"Тип ордера позиции до смены направления","Type order of position before changing direction"},
   {"Тикет ордера позиции до смены направления","Ticket order of position before changing direction"},
   {"Тип ордера текущей позиции","Type order of current position"},
   {"Тикет ордера текущей позиции","Ticket order of current position"},
   {"Цена на момент события","Price at the time of event"},
   {"Начальный объём ордера","Order initial volume"},
   {"Исполненный объём ордера","Order executed volume"},
   {"Оставшийся объём ордера","Order remaining volume"},
   {"Текущий объём позиции","Position current volume"},
   {"Цена открытия до модификации","Price open before modification"},
   {"Цена StopLoss до модификации","Price StopLoss before modification"},
   {"Цена TakeProfit до модификации","Price TakeProfit before modification"},
   {"Цена Ask в момент события","Price Ask at the time of event"},
   {"Цена Bid в момент события","Price Bid at the time of event"},
   {"Символ встречной позиции","Symbol of opposite position"},
   //---
   {"Установлен отложенный ордер","Pending order placed"},
   {"Открыта позиция","Position opened"},
   {"Удален отложенный ордер","Pending order removed"},
   {"Закрыта позиция","Position closed"},
   {"Неизвестный статус","Unknown status"},
   //---
   {"Нет торгового события","No trade event"},
   {"Отложенный ордер установлен","Pending order placed"},
   {"Отложенный ордер удалён","Pending order removed"},
   {"Начисление кредита","Credit"},
   {"Изъятие кредитных средств","Withdrawal of credit"},
   {"Дополнительные сборы","Additional charge"},
   {"Корректирующая запись","Correction"},
   {"Перечисление бонусов","Bonus"},
   {"Дополнительные комиссии","Additional commission"},
   {"Комиссия, начисляемая в конце торгового дня","Daily commission"},
   {"Комиссия, начисляемая в конце месяца","Monthly commission"},
   {"Агентская комиссия, начисляемая в конце торгового дня","Daily agent commission"},
   {"Агентская комиссия, начисляемая в конце месяца","Monthly agent commission"},
   {"Начисления процентов на свободные средства","Interest rate"},
   {"Отмененная сделка покупки","Canceled buy deal"},
   {"Отмененная сделка продажи","Canceled sell deal"},
   {"Начисление дивиденда","Dividend operations"},
   {"Начисление франкированного дивиденда","Franked (non-taxable) dividend operations"},
   {"Начисление налога","Tax charges"},
   {"Пополнение баланса","Balance refill"},
   {"Снятие средств с баланса","Withdrawal from balance"},
   //---
   {"Активирован отложенный ордер","Pending order activated"},
   {"Частичное срабатывание отложенного ордера","Pending order partially triggered"},
   {"Позиция открыта частично","Position opened partially"},
   {"Позиция закрыта частично","Position closed partially"},
   {"Позиция закрыта встречной","Position closed by opposite position"},
   {"Позиция закрыта встречной частично","Position closed partially by opposite position"},
   {"Позиция закрыта по StopLoss","Position closed by StopLoss"},
   {"Позиция закрыта по TakeProfit","Position closed by TakeProfit"},
   {"Позиция закрыта частично по StopLoss","Position closed partially by StopLoss"},
   {"Позиция закрыта частично по TakeProfit","Position closed partially by TakeProfit"},
   {"Разворот позиции по рыночному запросу","Position reversal by market request"},
   {"Разворот позиции срабатыванием отложенного ордера","Position reversal by triggered pending order"},
   {"Разворот позиции частичным исполнением заявки","Position reversal by partial request execution"},
   {"Добавлен объём к позиции по рыночному запросу","Added volume to position by market request"},
   {"Добавлен объём к позиции активацией отложенного ордера","Added volume to position by activation of pending order"},
   {"Модифицирована цена установки ордера","Modified order price"},
   {"Модифицированы цена установки и StopLoss ордера","Modified order price and StopLoss"},
   {"Модифицированы цена установки и TakeProfit ордера","Modified order price and TakeProfit"},
   {"Модифицированы цена установки, StopLoss и TakeProfit ордера","Modified order price, StopLoss and TakeProfit"},
   {"Модифицированы цены StopLoss и TakeProfit ордера","Modified order StopLoss and TakeProfit"},
   {"Модифицирован StopLoss ордера","Modified order's StopLoss"},
   {"Модифицирован TakeProfit ордера","Modified order's TakeProfit"},
   {"Модифицированы цены StopLoss и TakeProfit позиции","Modified position's StopLoss and TakeProfit"},
   {"Модифицирован StopLoss позиции","Modified position's StopLoss"},
   {"Модифицирован TakeProfit позиции","Modified position's TakeProfit"},
   //---
   {"Добавлен объём к позиции","Added volume to position"},
   {"Добавлен объём к позиции частичным исполнением заявки","Volume added to position by request partial completion"},
   {"Добавлен объём к позиции частичной активацией отложенного ордера","Added volume to position by a partially triggered pending order"},
   {"Сработал StopLimit-ордер","StopLimit order triggered"},
   {"Модификация","Modified"},
   {"Отмена","Canceled"},
   {"Истёк срок действия","Expired"},
   {"Рыночный запрос, выполненный в полном объёме","Fully completed market request"},
   {"Выполненный частично рыночный запрос","Partially completed market request"},
   {"Разворот позиции","Position reversal"},
   {"Разворот позиции при при частичном срабатывании отложенного ордера","Position reversal on partially triggered pending order"},
   {"Частичное закрытие по StopLoss","Partial close by StopLoss triggered"},
   {"Частичное закрытие по TakeProfit","Partial close by TakeProfit triggered"},
   {"Закрытие встречной позицией","Closed by opposite position"},
   {"Частичное закрытие встречной позицией","Closed partially by opposite position"},
   {"Закрытие частью объёма встречной позиции","Closed by incomplete volume of opposite position"},
   {"Частичное закрытие частью объёма встречной позиции","Closed partially by incomplete volume of opposite position"},

//--- CSymbol
   {"Индекс в окне \"Обзор рынка\"","Index in \"Market Watch window\""},
   {"Пользовательский символ","Custom symbol"},
   {"Тип цены для построения баров","Price type used for generating symbols bars"},
   {"Символ с таким именем существует","Symbol with this name exists"},
   {"Символ выбран в Market Watch","Symbol selected in Market Watch"},
   {"Символ отображается в Market Watch","Symbol visible in Market Watch"},
   {"Количество сделок в текущей сессии","Number of deals in the current session"},
   {"Общее число ордеров на покупку в текущий момент","Number of Buy orders at the moment"},
   {"Общее число ордеров на продажу в текущий момент","Number of Sell orders at the moment"},
   {"Объем в последней сделке","Volume of the last deal"},
   {"Максимальный объём за день","Maximal day volume"},
   {"Минимальный объём за день","Minimal day volume"},
   {"Время последней котировки","Time of last quote"},
   {"Количество знаков после запятой","Digits after decimal point"},
   {"Количество знаков после запятой в значении лота","Digits after decimal point in value of the lot"},
   {"Размер спреда в пунктах","Spread value in points"},
   {"Плавающий спред","Floating spread"},
   {"Максимальное количество показываемых заявок в стакане","Maximal number of requests shown in Depth of Market"},
   {"Способ вычисления стоимости контракта","Contract price calculation mode"},
   {"Тип исполнения ордеров","Order execution type"},
   {"Дата начала торгов по инструменту","Date of symbol trade beginning"},
   {"Дата окончания торгов по инструменту","Date of symbol trade end"},
   {"Минимальный отступ от цены закрытия для установки Stop ордеров","Minimal indention from close price to place Stop orders"},
   {"Дистанция заморозки торговых операций","Distance to freeze trade operations in points"},
   {"Режим заключения сделок","Deal execution mode"},
   {"Модель расчета свопа","Swap calculation model"},
   {"День недели для начисления тройного свопа","Day of week to charge 3 days swap rollover"},
   {"Расчет хеджированной маржи по наибольшей стороне","Calculating hedging margin using larger leg"},
   {"Флаги разрешенных режимов истечения ордера","Flags of allowed order expiration modes"},
   {"Флаги разрешенных режимов заливки ордера","Flags of allowed order filling modes"},
   {"Флаги разрешённых типов ордеров","Flags of allowed order types"},
   {"Срок действия StopLoss и TakeProfit ордеров","Expiration of Stop Loss and Take Profit orders"},
   {"Тип опциона","Option type"},
   {"Право опциона","Option right"},
   {"Цвет фона символа в Market Watch","Background color of symbol in Market Watch"},
   {"Максимальный Bid за день","Maximal Bid of the day"},
   {"Минимальный Bid за день","Minimal Bid of the day"},
   {"Максимальный Ask за день","Maximal Ask of the day"},
   {"Минимальный Ask за день","Minimal Ask of the day"},
   {"Максимальный Last за день","Maximal Last of the day"},
   {"Минимальный Last за день","Minimal Last of the day"},
   {"Реальный объём за день","Real volume of last deal"},
   {"Максимальный реальный объём за день","Maximal real volume of the day"},
   {"Минимальный реальный объём за день","Minimal real volume of the day"},
   {"Цена исполнения опциона","Strike price"},
   {"Значение одного пункта","Symbol point value"},
   {"Рассчитанная стоимость тика для позиции","Calculated tick price for position"},
   {"Рассчитанная стоимость тика для прибыльной позиции","Calculated tick price for profitable position"},
   {"Рассчитанная стоимость тика для убыточной позиции","Calculated tick price for losing position"},
   {"Минимальное изменение цены","Minimal price change"},
   {"Размер торгового контракта","Trade contract size"},
   {"Накопленный купонный доход","Accumulated coupon interest"},
   {"Начальная стоимость облигации, установленная эмитентом","Initial bond value set by issuer"},
   {"Коэффициент ликвидности","Liquidity rate"},
   {"Минимальный объем для заключения сделки","Minimal volume for deal"},
   {"Максимальный объем для заключения сделки","Maximal volume for deal"},
   {"Минимальный шаг изменения объема для заключения сделки","Minimal volume change step for deal execution"},
   {
    "Максимально допустимый общий объем позиции и отложенных ордеров в одном направлении",
    "Maximum allowed aggregate volume of open position and pending orders in one direction"
   },
   {"Значение свопа на покупку","Long swap value"},
   {"Значение свопа на продажу","Short swap value"},
   {"Начальная (инициирующая) маржа","Initial margin"},
   {"Поддерживающая маржа по инструменту","Maintenance margin"},
   {"Коэффициент взимания начальной маржи по длинным позициям","Coefficient of margin initial charging for long positions"},
   {"Коэффициент взимания начальной маржи по коротким позициям","Coefficient of margin initial charging for short positions"},
   {"Коэффициент взимания поддерживающей маржи по длинным позициям","Coefficient of margin maintenance charging for long positions"},
   {"Коэффициент взимания поддерживающей маржи по коротким позициям","Coefficient of margin maintenance charging for short positions"},
   {"Коэффициент взимания начальной маржи по BuyStop ордерам","Coefficient of margin initial charging for BuyStop orders"},
   {"Коэффициент взимания начальной маржи по BuyLimit ордерам","Coefficient of margin initial charging for BuyLimit orders"},
   {"Коэффициент взимания начальной маржи по BuyStopLimit ордерам","Coefficient of margin initial charging for BuyStopLimit orders"},
   {"Коэффициент взимания начальной маржи по SellStop ордерам","Coefficient of margin initial charging for SellStop orders"},
   {"Коэффициент взимания начальной маржи по SellLimit ордерам","Coefficient of margin initial charging for SellLimit orders"},
   {"Коэффициент взимания начальной маржи по SellStopLimit ордерам","Coefficient of margin initial charging for SellStopLimit orders"},
   {"Коэффициент взимания поддерживающей маржи по BuyStop ордерам","Coefficient of margin maintenance charging for BuyStop orders"},
   {"Коэффициент взимания поддерживающей маржи по BuyLimit ордерам","Coefficient of margin maintenance charging for BuyLimit orders"},
   {"Коэффициент взимания поддерживающей маржи по BuyStopLimit ордерам","Coefficient of margin maintenance charging for BuyStopLimit orders"},
   {"Коэффициент взимания поддерживающей маржи по SellStop ордерам","Coefficient of margin maintenance charging for SellStop orders"},
   {"Коэффициент взимания поддерживающей маржи по SellLimit ордерам","Coefficient of margin maintenance charging for SellLimit orders"},
   {"Коэффициент взимания поддерживающей маржи по SellStopLimit ордерам","Coefficient of margin maintenance charging for SellStopLimit orders"},
   {"Cуммарный объём сделок в текущую сессию","Summary volume of the current session deals"},
   {"Cуммарный оборот в текущую сессию","Summary turnover of the current session"},
   {"Cуммарный объём открытых позиций","Summary open interest"},
   {"Общий объём ордеров на покупку в текущий момент","Current volume of Buy orders"},
   {"Общий объём ордеров на продажу в текущий момент","Current volume of Sell orders"},
   {"Цена открытия сессии","Open price of the current session"},
   {"Цена закрытия сессии","Close price of the current session"},
   {"Средневзвешенная цена сессии","Average weighted price of the current session"},
   {"Цена поставки на текущую сессию","Settlement price of the current session"},
   {"Минимально допустимое значение цены на сессию","Minimal price of the current session"},
   {"Максимально допустимое значение цены на сессию","Maximal price of the current session"},
   {"Размер контракта или маржи для одного лота перекрытых позиций","Contract size or margin value per one lot of hedged positions"},
   {"Имя символа","Symbol name"},
   {"Имя базового актива для производного инструмента","Underlying asset of derivative"},
   {"Базовая валюта инструмента","Basic currency of symbol"},
   {"Валюта прибыли","Profit currency"},
   {"Валюта залоговых средств","Margin currency"},
   {"Источник текущей котировки","Feeder of the current quote"},
   {"Описание символа","Symbol description"},
   {"Формула для построения цены пользовательского символа","Formula used for custom symbol pricing"},
   {"Имя торгового символа в системе международных идентификационных кодов","Symbol name in ISIN system"},
   {"Адрес интернет страницы с информацией по символу","Address of web page containing symbol information"},
   {"Путь в дереве символов","Path in symbol tree"},
   //---
   {"Форекс символ","Forex symbol"},
   {"Форекс символ-мажор","Forex major symbol"},
   {"Форекс символ-минор","Forex minor symbol"},
   {"Форекс символ-экзотик","Forex Exotic Symbol"},
   {"Форекс символ/рубль","Forex symbol RUB"},
   {"Металл","Metal"},
   {"Индекс","Index"},
   {"Индикатив","Indicative"},
   {"Криптовалютный символ","Crypto symbol"},
   {"Товарный символ","Commodity symbol"},
   {"Биржевой символ","Exchange symbol"},
   {"Фьючерс","Furures"},
   {"Контракт на разницу","Contract For Difference"},
   {"Ценная бумага","Stocks"},
   {"Облигация","Bonds"},
   {"Опцион","Option"},
   {"Неторгуемый актив","Collateral"},
   {"Пользовательский символ","Custom symbol"},
   {"Символ общей группы","Common group symbol"},
   //---
   {"Бары строятся по ценам Bid","Bars based on Bid prices"},
   {"Бары строятся по ценам Last","Bars based on Last prices"},
   {"Расчет прибыли и маржи для Форекс","Forex mode"},
   {"Расчет прибыли и маржи для Форекс без учета плеча","Forex No Leverage mode"},
   {"Расчет залога и прибыли для фьючерсов","Futures mode"},
   {"Расчет залога и прибыли для CFD","CFD mode"},
   {"Расчет залога и прибыли для CFD на индексы","CFD index mode"},
   {"Расчет залога и прибыли для CFD при торговле с плечом","CFD Leverage mode"},
   {"Расчет залога и прибыли для торговли ценными бумагами на бирже","Exchange mode"},
   {"Расчет залога и прибыли для торговли фьючерсными контрактами на бирже","Futures mode"},
   {"Расчет залога и прибыли для торговли фьючерсными контрактами на FORTS","FORTS Futures mode"},
   {"Расчет прибыли и маржи по торговым облигациям на бирже","Exchange Bonds mode"},
   {"Расчет прибыли и маржи при торговле ценными бумагами на MOEX","Exchange MOEX Stocks mode"},
   {"Расчет прибыли и маржи по торговым облигациям на MOEX","Exchange MOEX Bonds mode"},
   {"Используется в качестве неторгуемого актива на счете","Collateral mode"},
   {"Неизвестный режим","Unknown mode"},
   {"Торговля по символу запрещена","Trade is disabled for symbol"},
   {"Разрешены только покупки","Allowed only long positions"},
   {"Разрешены только продажи","Allowed only short positions"},
   {"Разрешены только операции закрытия позиций","Allowed only position close operations"},
   {"Нет ограничений на торговые операции","No trade restrictions"},
   //---
   {"Торговля по запросу","Execution by request"},
   {"Торговля по потоковым ценам","Instant execution"},
   {"Исполнение ордеров по рынку","Market execution"},
   {"Биржевое исполнение","Exchange execution"},
   //---
   {"Нет свопов","Swaps disabled (no swaps)"},
   {"Свопы начисляются в пунктах","Swaps charged in points"},
   {"Свопы начисляются в деньгах в базовой валюте символа","Swaps charged in money in symbol base currency"},
   {"Свопы начисляются в деньгах в маржинальной валюте символа","Swaps charged in money in symbol margin currency"},
   {"Свопы начисляются в деньгах в валюте депозита клиента","Swaps charged in money in client deposit currency"},
   {
    "Свопы начисляются в годовых процентах от цены инструмента на момент расчета свопа",
    "Swaps charged as specified annual interest from the instrument price at calculation of swap"
   },
   {"Свопы начисляются в годовых процентах от цены открытия позиции по символу","Swaps charged as specified annual interest from open price of position"},
   {"Свопы начисляются переоткрытием позиции по цене закрытия","Swaps charged by reopening positions by close price"},
   {"Свопы начисляются переоткрытием позиции по текущей цене Bid","Swaps charged by reopening positions by the current Bid price"},
   //---
   {
    "Отложенные ордеры и уровни Stop Loss/Take Profit действительны неограниченно по времени до явной отмены",
    "Pending orders and Stop Loss/Take Profit levels valid for unlimited period until their explicit cancellation"
   },
   {
    "При смене торгового дня отложенные ордеры и все уровни StopLoss и TakeProfit удаляются",
    "At the end of the day, all Stop Loss and Take Profit levels, as well as pending orders, deleted"
   },
   {
    "При смене торгового дня удаляются только отложенные ордеры, уровни StopLoss и TakeProfit сохраняются",
    "At the end of the day, only pending orders deleted, while Stop Loss and Take Profit levels preserved"
   },
   //---
   {"Европейский тип опциона – может быть погашен только в указанную дату","European option may only be exercised on specified date"},
   {"Американский тип опциона – может быть погашен в любой день до истечения срока опциона","American option may be exercised on any trading day or before expiry"},
   {"Неизвестный тип опциона","Unknown option type"},
   {"Опцион, дающий право купить актив по фиксированной цене","Call option gives you right to buy asset at specified price"},
   {"Опцион, дающий право продать актив по фиксированной цене","Put option gives you right to sell asset at specified price"},
   //---
   {"Рыночный ордер (Да)","Market order (Yes)"},
   {"Рыночный ордер (Нет)","Market order (No)"},
   {"Лимит ордер (Да)","Limit order (Yes)"},
   {"Лимит ордер (Нет)","Limit order (No)"},
   {"Стоп ордер (Да)","Stop order (Yes)"},
   {"Стоп ордер (Нет)","Stop order (No)"},
   {"Стоп-лимит ордер (Да)","StopLimit order (Yes)"},
   {"Стоп-лимит ордер (Нет)","StopLimit order (No)"},
   {"StopLoss (Да)","StopLoss (Yes)"},
   {"StopLoss (Нет)","StopLoss (No)"},
   {"TakeProfit (Да)","TakeProfit (Yes)"},
   {"TakeProfit (Нет)","TakeProfit (No)"},
   {"Закрытие встречным (Да)","CloseBy order (Yes)"},
   {"Закрытие встречным (Нет)","CloseBy order (No)"},
   {"Вернуть (Да)","Return (Yes)"},
   {"Всё/Ничего (Да)","Fill or Kill (Yes)"},
   {"Всё/Ничего (Нет)","Fill or Kill (No)"},
   {"Всё/Частично (Да)","Immediate or Cancel order (Yes)"},
   {"Всё/Частично (Нет)","Immediate or Cancel order (No)"},
   {"Неограниченно (Да)","Unlimited (Yes)"},
   {"Неограниченно (Нет)","Unlimited (No)"},
   {"До конца дня (Да)","Valid till the end of the day (Yes)"},
   {"До конца дня (Нет)","Valid till the end of the day (No)"},
   {"Срок указывается в ордере (Да)","Time specified in order (Yes)"},
   {"Срок указывается в ордере (Нет)","Time specified in order (No)"},
   {"День указывается в ордере (Да)","Date specified in order (Yes)"},
   {"День указывается в ордере (Нет)","Date specified in order (No)"},

   {"В окно \"Обзор рынка\" добавлен символ","Added symbol to \"Market Watch\" window"},
   {"Из окна \"Обзор рынка\" удалён символ","Removed from \"Market Watch\" window"},
   {"Изменено расположение символов в окне \"Обзор рынка\"","Changed arrangement of symbols in \"Market Watch\" window"},
   {"Работа только с текущим символом","Work only with the current symbol"},
   {"Работа с предопределённым списком символов","Work with predefined list of symbols"},
   {"Работа с символами из окна \"Обзор рынка\"","Working with symbols from \"Market Watch\" window"},
   {"Работа с полным списком всех доступных символов","Work with full list of all available symbols"},


//--- CAccount
   {"Номер счёта","Account number"},
   {"Тип торгового счета","Account trade mode"},
   {"Размер предоставленного плеча","Account leverage"},
   {"Максимально допустимое количество действующих отложенных ордеров","Maximum allowed number of active pending orders"},
   {"Режим задания минимально допустимого уровня залоговых средств","Mode for setting minimal allowed margin"},
   {"Разрешенность торговли для текущего счета","Allowed trade for the current account"},
   {"Разрешенность торговли для эксперта","Allowed trade for Expert Advisor"},
   {"Режим расчета маржи","Margin calculation mode"},
   {"Количество знаков после запятой для валюты счета","Number of decimal places in account currency"},
   {"Тип торгового сервера","Type of trading server"},
   //---
   {"Баланс счета","Account balance"},
   {"Предоставленный кредит","Account credit"},
   {"Текущая прибыль на счете","Current profit on account"},
   {"Собственные средства на счете","Account equity"},
   {"Зарезервированные залоговые средства на счете","Account margin used in deposit currency"},
   {"Свободные средства на счете, доступные для открытия позиции","Free margin on account"},
   {"Уровень залоговых средств на счете в процентах","Account margin level in %"},
   {"Уровень залоговых средств для наступления Margin Call","Margin call level"},
   {"Уровень залоговых средств для наступления Stop Out","Margin stop out level"},
   {
    "Зарезервированные средства для обеспечения гарантийной суммы по всем отложенным ордерам",
    "Amount reserved on account to cover margin of all pending orders"
   },
   {
    "Зарезервированные средства для обеспечения минимальной суммы по всем открытым позициям",
    "Min equity reserved on account to cover the min amount of all open positions"
   },
   {"Текущий размер активов на счёте","Current assets of account"},
   {"Текущий размер обязательств на счёте","Current liabilities on account"},
   {"Сумма заблокированных комиссий по счёту","Current blocked commission amount on account"},
   //---
   {"Имя клиента","Client name"},
   {"Имя торгового сервера","Trade server name"},
   {"Валюта депозита","Account currency"},
   {"Имя компании, обслуживающей счет","Name of company that serves the account"},
   //---
   {"Демонстрационный счёт","Demo account"},
   {"Конкурсный счёт","Contest account"},
   {"Реальный счёт","Real account"},
   {"Неизвестный тип счёта","Unknown account type"},
   //---
   {"Уровень задается в процентах","Account StopOut mode in %"},
   {"Уровень задается в деньгах","Account StopOut mode in money"},
   {"Внебиржевой рынок в режиме \"Неттинг\"","Netting mode"},
   {"Внебиржевой рынок в режиме \"Хеджинг\"","Hedging mode"},
   {"Биржевой рынок","Exchange market mode"},

//--- CEngine
   {"С момента последнего запуска ЕА торговых событий не было","There have been no trade events since the last launch of EA"},
   {"Не удалось получить описание последнего торгового события","Failed to get description of the last trading event"},

  };
//+---------------------------------------------------------------------+
```

When looking at the enumeration of library messages and the array of these messages, we can see that each message accurately corresponds to
declaring its constant in the enumeration.

All messages located in the arrays should
strictly correspond to their constants located in the enumeration or accurately correspond to an error message code returned from the
GetLastError() function.

To add a translation into another language, simply add the text in that language to each message (after an English text). If you want to
replace any of the two predefined languages with another one, simply edit the existing messages.

For example (for the highlighted code above), in order to add a
translation of the text "Unknown account type" into German,

add the translation after the appropriate
English message:

```
{"Неизвестный тип счёта","Unknown account type","Unbekannter Kontotyp"},
```

When adding another language,
it should be added to all messages in all message text arrays, otherwise message indices will be broken causing
unpredictable results when trying to display messages.

**Let's write the array of messages for trade server return codes:**

```
//+---------------------------------------------------------------------+
//| Array of messages for trade server return codes (10004 - 10044)     |
//| (1) in user's country language                                      |
//| (2) in the international language                                   |
//+---------------------------------------------------------------------+
string messages_ts_ret_code[][TOTAL_LANG]=
  {
   {"Реквота","Requote"},                                                                                                                          // 10004
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},                                                            // 10005
   {"Запрос отклонен","Request rejected"},                                                                                                         // 10006
   {"Запрос отменен трейдером","Request canceled by trader"},                                                                                      // 10007
   {"Ордер размещен","Order placed"},                                                                                                              // 10008
   {"Заявка выполнена","Request completed"},                                                                                                       // 10009
   {"Заявка выполнена частично","Only part of the request was completed"},                                                                         // 10010
   {"Ошибка обработки запроса","Request processing error"},                                                                                        // 10011
   {"Запрос отменен по истечению времени","Request canceled by timeout"},                                                                          // 10012
   {"Неправильный запрос","Invalid request"},                                                                                                      // 10013
   {"Неправильный объем в запросе","Invalid volume in request"},                                                                                   // 10014
   {"Неправильная цена в запросе","Invalid price in request"},                                                                                     // 10015
   {"Неправильные стопы в запросе","Invalid stops in request"},                                                                                    // 10016
   {"Торговля запрещена","Trading disabled"},                                                                                                      // 10017
   {"Рынок закрыт","Market closed"},                                                                                                               // 10018
   {"Нет достаточных денежных средств для выполнения запроса","Not enough money to complete request"},                                             // 10019
   {"Цены изменились","Prices changed"},                                                                                                           // 10020
   {"Отсутствуют котировки для обработки запроса","No quotes to process request"},                                                                 // 10021
   {"Неверная дата истечения ордера в запросе","Invalid order expiration date in request"},                                                        // 10022
   {"Состояние ордера изменилось","Order state changed"},                                                                                          // 10023
   {"Слишком частые запросы","Too frequent requests"},                                                                                             // 10024
   {"В запросе нет изменений","No changes in request"},                                                                                            // 10025
   {"Автотрейдинг запрещен сервером","Autotrading disabled by server"},                                                                            // 10026
   {"Автотрейдинг запрещен клиентским терминалом","Autotrading disabled by client terminal"},                                                      // 10027
   {"Запрос заблокирован для обработки","Request locked for processing"},                                                                          // 10028
   {"Ордер или позиция заморожены","Order or position frozen"},                                                                                    // 10029
   {"Указан неподдерживаемый тип исполнения ордера по остатку","Invalid order filling type"},                                                      // 10030
   {"Нет соединения с торговым сервером","No connection with trade server"},                                                                       // 10031
   {"Операция разрешена только для реальных счетов","Operation allowed only for live accounts"},                                                   // 10032
   {"Достигнут лимит на количество отложенных ордеров","Number of pending orders reached limit"},                                                  // 10033
   {"Достигнут лимит на объем ордеров и позиций для данного символа","Volume of orders and positions for symbol reached limit"},                   // 10034
   {"Неверный или запрещённый тип ордера","Incorrect or prohibited order type"},                                                                   // 10035
   {"Позиция с указанным идентификатором уже закрыта","Position with specified identifier already closed"},                                        // 10036
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},                                                            // 10037
   {"Закрываемый объем превышает текущий объем позиции","Close volume exceeds the current position volume"},                                       // 10038
   {"Для указанной позиции уже есть ордер на закрытие","Close order already exists for specified position"},                                       // 10039
   {"Достигнут лимит на количество открытых позиций","Number of positions reached limit"},                                                         // 10040
   {
    "Запрос на активацию отложенного ордера отклонен, а сам ордер отменен",                                                                        // 10041
    "Pending order activation request rejected, order canceled"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешены только длинные позиции\"",                                                // 10042
    "Request rejected, because \"Only long positions are allowed\" rule set for symbol"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешены только короткие позиции\"",                                               // 10043
    "Request rejected, because \"Only short positions are allowed\" rule set for symbol"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешено только закрывать существующие позиции\"",                                 // 10044
    "Request rejected, because \"Only position closing is allowed\" rule set for symbol "
   },
  };
//+------------------------------------------------------------------+
```

All codes returned by the trade server start with 10004. Some codes are skipped and have no description, for example 10005 and 10037. But they
still should be present in the array as all codes are located in the array strictly in ascending order. Therefore, the message "Unknown trade
server return code" has been added for unused codes.

More importantly, trade server return codes start from 10004, while in the array, they are located beginning from index 0. This is why, when
accessing the array, we are going to subtract the value of the very first return code of the trade server from the obtained code. In this case,
we will definitely get the index of the necessary code whose description is located in the array.

For example, for 10004, we get index 0 (10004 - 10004 = 0), while for 10007, we get index 3 (10007 - 10004 = 3).

**Array of execution time error messages:**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (0, 4001 - 4019)          |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime[][TOTAL_LANG]=
  {
   {"Операция выполнена успешно","Operation successful"},                                                                                          // 0
   {"Неожиданная внутренняя ошибка","Unexpected internal error"},                                                                                  // 4001
   {"Ошибочный параметр при внутреннем вызове функции клиентского терминала","Wrong parameter in inner call of client terminal function"},         // 4002
   {"Ошибочный параметр при вызове системной функции","Wrong parameter when calling system function"},                                             // 4003
   {"Недостаточно памяти для выполнения системной функции","Not enough memory to perform system function"},                                        // 4004
   {
    "Структура содержит объекты строк и/или динамических массивов и/или структуры с такими объектами и/или классы",                                // 4005
    "Structure contains objects of strings and/or dynamic arrays and/or structure of such objects and/or classes"
   },
   {
    "Массив неподходящего типа, неподходящего размера или испорченный объект динамического массива",                                               // 4006
    "Array of wrong type, wrong size, or damaged object of dynamic array"
   },
   {
    "Недостаточно памяти для перераспределения массива либо попытка изменения размера статического массива",                                       // 4007
    "Not enough memory for relocation of array, or attempt to change size of static array"
   },
   {"Недостаточно памяти для перераспределения строки","Not enough memory for relocation of string"},                                              // 4008
   {"Неинициализированная строка","Not initialized string"},                                                                                       // 4009
   {"Неправильное значение даты и/или времени","Invalid date and/or time"},                                                                        // 4010
   {"Общее число элементов в массиве не может превышать 2147483647","Total amount of elements in array cannot exceed 2147483647"},                 // 4011
   {"Ошибочный указатель","Wrong pointer"},                                                                                                        // 4012
   {"Ошибочный тип указателя","Wrong type of pointer"},                                                                                            // 4013
   {"Системная функция не разрешена для вызова","Function not allowed for call"},                                                                  // 4014
   {"Совпадение имени динамического и статического ресурсов","Names of dynamic and static resource match"},                                        // 4015
   {"Ресурс с таким именем в EX5 не найден","Resource with this name not found in EX5"},                                                           // 4016
   {"Неподдерживаемый тип ресурса или размер более 16 MB","Unsupported resource type or its size exceeds 16 Mb"},                                  // 4017
   {"Имя ресурса превышает 63 символа","Resource name exceeds 63 characters"},                                                                     // 4018
   {"При вычислении математической функции произошло переполнение ","Overflow occurred when calculating math function "},                          // 4019
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Charts error section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4101 - 4116)             |
//| (Charts)                                                         |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_charts[][TOTAL_LANG]=
  {
   {"Ошибочный идентификатор графика","Wrong chart ID"},                                                                                           // 4101
   {"График не отвечает","Chart does not respond"},                                                                                                // 4102
   {"График не найден","Chart not found"},                                                                                                         // 4103
   {"У графика нет эксперта, который мог бы обработать событие","No Expert Advisor in chart that could handle event"},                             // 4104
   {"Ошибка открытия графика","Chart opening error"},                                                                                              // 4105
   {"Ошибка при изменении для графика символа и периода","Failed to change chart symbol and period"},                                              // 4106
   {"Ошибочное значение параметра для функции по работе с графиком","Error value of parameter for function of working with charts"},               // 4107
   {"Ошибка при создании таймера","Failed to create timer"},                                                                                       // 4108
   {"Ошибочный идентификатор свойства графика","Wrong chart property ID"},                                                                         // 4109
   {"Ошибка при создании скриншота","Error creating screenshots"},                                                                                 // 4110
   {"Ошибка навигации по графику","Error navigating through chart"},                                                                               // 4111
   {"Ошибка при применении шаблона","Error applying template"},                                                                                    // 4112
   {"Подокно, содержащее указанный индикатор, не найдено","Subwindow containing indicator not found"},                                             // 4113
   {"Ошибка при добавлении индикатора на график","Error adding indicator to chart"},                                                               // 4114
   {"Ошибка при удалении индикатора с графика","Error deleting indicator from chart"},                                                             // 4115
   {"Индикатор не найден на указанном графике","Indicator not found on specified chart"},                                                          // 4116
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Graphical objects errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4201 - 4205)             |
//| (Graphical objects)                                              |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_graph_obj[][TOTAL_LANG]=
  {
   {"Ошибка при работе с графическим объектом","Error working with graphical object"},                                                           // 4201
   {"Графический объект не найден","Graphical object not found"},                                                                                // 4202
   {"Ошибочный идентификатор свойства графического объекта","Wrong ID of graphical object property"},                                            // 4203
   {"Невозможно получить дату, соответствующую значению","Unable to get date corresponding to value"},                                           // 4204
   {"Невозможно получить значение, соответствующее дате","Unable to get value corresponding to date"},                                           // 4205
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **MarketInfo errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4301 - 4305)             |
//| (MarketInfo)                                                     |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_market[][TOTAL_LANG]=
  {
   {"Неизвестный символ","Unknown symbol"},                                                                                                        // 4301
   {"Символ не выбран в MarketWatch","Symbol not selected in MarketWatch"},                                                                        // 4302
   {"Ошибочный идентификатор свойства символа","Wrong identifier of symbol property"},                                                             // 4303
   {"Время последнего тика неизвестно (тиков не было)","Time of the last tick not known (no ticks)"},                                              // 4304
   {"Ошибка добавления или удаления символа в MarketWatch","Error adding or deleting a symbol in MarketWatch"},                                    // 4305
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Access to history errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4401 - 4407)             |
//| (Access to history)                                              |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_history[][TOTAL_LANG]=
  {
   {"Запрашиваемая история не найдена","Requested history not found"},                                                                             // 4401
   {"Ошибочный идентификатор свойства истории","Wrong ID of history property"},                                                                    // 4402
   {"Превышен таймаут при запросе истории","Exceeded history request timeout"},                                                                    // 4403
   {"Количество запрашиваемых баров ограничено настройками терминала","Number of requested bars limited by terminal settings"},                    // 4404
   {"Множество ошибок при загрузке истории","Multiple errors when loading history"},                                                               // 4405
   {"Неизвестный код ошибки","Unknown error code"},                                                                                                // 4406
   {"Принимающий массив слишком мал чтобы вместить все запрошенные данные","Receiving array too small to store all requested data"},               // 4407
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Global Variables errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4501 - 4524)             |
//| (Global Variables)                                               |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_global[][TOTAL_LANG]=
  {
   {"Глобальная переменная клиентского терминала не найдена","Global variable of client terminal not found"},                                      // 4501
   {
    "Глобальная переменная клиентского терминала с таким именем уже существует",                                                                   // 4502
    "Global variable of client terminal with the same name already exists"
   },
   {"Не было модификаций глобальных переменных","Global variables were not modified"},                                                             // 4503
   {"Не удалось открыть и прочитать файл со значениями глобальных переменных","Cannot read file with global variable values"},                     // 4504
   {"Не удалось записать файл со значениями глобальных переменных","Cannot write file with global variable values"},                               // 4505

   {"Неизвестный код ошибки","Unknown error code"},                                                                                                // 4506
   {"Неизвестный код ошибки","Unknown error code"},                                                                                                // 4507
   {"Неизвестный код ошибки","Unknown error code"},                                                                                                // 4508
   {"Неизвестный код ошибки","Unknown error code"},                                                                                                // 4509

   {"Не удалось отправить письмо","Email sending failed"},                                                                                         // 4510
   {"Не удалось воспроизвести звук","Sound playing failed"},                                                                                       // 4511
   {"Ошибочный идентификатор свойства программы","Wrong identifier of program property"},                                                          // 4512
   {"Ошибочный идентификатор свойства терминала","Wrong identifier of terminal property"},                                                         // 4513
   {"Не удалось отправить файл по ftp","File sending via ftp failed"},                                                                             // 4514
   {"Не удалось отправить уведомление","Failed to send notification"},                                                                             // 4515
   {
    "Неверный параметр для отправки уведомления – в функцию SendNotification() передали пустую строку или NULL",                                   // 4516
    "Invalid parameter for sending notification – empty string or NULL passed to SendNotification() function"
   },
   {
    "Неверные настройки уведомлений в терминале (не указан ID или не выставлено разрешение)",                                                      // 4517
    "Wrong settings of notifications in terminal (ID not specified or permission not set)"
   },
   {"Слишком частая отправка уведомлений","Too frequent sending of notifications"},                                                                // 4518
   {"Не указан FTP сервер","FTP server not specified"},                                                                                            // 4519
   {"Не указан FTP логин","FTP login not specified"},                                                                                              // 4520
   {"Не найден файл в директории MQL5\\Files для отправки на FTP сервер","File not found in MQL5\\Files directory to send on FTP server"},         // 4521
   {"Ошибка при подключении к FTP серверу","FTP connection failed"},                                                                               // 4522
   {"На FTP сервере не найдена директория для выгрузки файла ","FTP path not found on server"},                                                    // 4523
   {"Подключение к FTP серверу закрыто","FTP connection closed"},                                                                                  // 4524
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Custom indicator buffers and properties errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4601 - 4603)             |
//| (Custom indicator buffers and properties)                        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_custom_indicator[][TOTAL_LANG]=
  {
   {"Недостаточно памяти для распределения индикаторных буферов","Not enough memory for distribution of indicator buffers"},                       // 4601
   {"Ошибочный индекс своего индикаторного буфера","Wrong indicator buffer index"},                                                                // 4602
   {"Ошибочный идентификатор свойства пользовательского индикатора","Wrong ID of custom indicator property"},                                      // 4603
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Account errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4701 - 4758)             |
//| (Account)                                                        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_account[][TOTAL_LANG]=
  {
   {"Ошибочный идентификатор свойства счета","Wrong account property ID"},                                                                         // 4701

   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4702
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4703
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4704
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4705
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4706
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4707
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4708
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4709
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4710
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4711
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4712
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4713
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4714
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4715
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4716
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4717
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4718
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4719
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4720
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4721
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4722
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4723
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4724
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4725
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4726
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4727
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4728
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4729
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4730
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4731
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4732
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4733
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4734
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4735
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4736
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4737
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4738
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4739
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4740
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4741
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4742
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4743
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4744
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4745
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4746
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4747
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4748
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4749
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4750

   {"Ошибочный идентификатор свойства торговли","Wrong trading property ID"},                                                                      // 4751
   {"Торговля для эксперта запрещена","Trading by Expert Advisors prohibited"},                                                                    // 4752
   {"Позиция не найдена","Position not found"},                                                                                                    // 4753
   {"Ордер не найден","Order not found"},                                                                                                          // 4754
   {"Сделка не найдена","Deal not found"},                                                                                                         // 4755
   {"Не удалось отправить торговый запрос","Trade request sending failed"},                                                                        // 4756
   {"Неизвестный код ошибки","Unknown error code"},                                                                                       // 4757
   {"Не удалось вычислить значение прибыли или маржи","Failed to calculate profit or margin"},                                                     // 4758
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Indicators errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4801 - 4812)             |
//| (Indicators)                                                     |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_indicator[][TOTAL_LANG]=
  {
   {"Неизвестный символ","Unknown symbol"},                                                                                                        // 4801
   {"Индикатор не может быть создан","Indicator cannot be created"},                                                                               // 4802
   {"Недостаточно памяти для добавления индикатора","Not enough memory to add indicator"},                                                         // 4803
   {"Индикатор не может быть применен к другому индикатору","Indicator cannot be applied to another indicator"},                                   // 4804
   {"Ошибка при добавлении индикатора","Error applying indicator to chart"},                                                                       // 4805
   {"Запрошенные данные не найдены","Requested data not found"},                                                                                   // 4806
   {"Ошибочный хэндл индикатора","Wrong indicator handle"},                                                                                        // 4807
   {"Неправильное количество параметров при создании индикатора","Wrong number of parameters when creating indicator"},                            // 4808
   {"Отсутствуют параметры при создании индикатора","No parameters when creating indicator"},                                                      // 4809
   {
    "Первым параметром в массиве должно быть имя пользовательского индикатора",                                                                    // 4810
    "First parameter in array should be name of custom indicator"
   },
   {"Неправильный тип параметра в массиве при создании индикатора","Invalid parameter type in array when creating indicator"},                 // 4811
   {"Ошибочный индекс запрашиваемого индикаторного буфера","Wrong index of requested indicator buffer"},                                       // 4812
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Market depth errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (4901 - 4904)             |
//| (Market depth)                                                   |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_books[][TOTAL_LANG]=
  {
   {"Стакан цен не может быть добавлен","Depth Of Market cannot be added"},                                                                       // 4901
   {"Стакан цен не может быть удален","Depth Of Market cannot be removed"},                                                                       // 4902
   {"Данные стакана цен не могут быть получены","Data from Depth Of Market cannot be obtained"},                                                  // 4903
   {"Ошибка при подписке на получение новых данных стакана цен","Error in subscribing to receive new data from Depth Of Market"},                 // 4904
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **File operations errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5001 - 5027)             |
//| (File operations)                                                |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_files[][TOTAL_LANG]=
  {
   {"Не может быть открыто одновременно более 64 файлов","More than 64 files cannot be opened at the same time"},                                  // 5001
   {"Недопустимое имя файла","Invalid file name"},                                                                                                 // 5002
   {"Слишком длинное имя файла","Too long file name"},                                                                                             // 5003
   {"Ошибка открытия файла","File opening error"},                                                                                                 // 5004
   {"Недостаточно памяти для кеша чтения","Not enough memory for cache to read"},                                                                  // 5005
   {"Ошибка удаления файла","File deleting error"},                                                                                                // 5006
   {"Файл с таким хэндлом уже был закрыт, либо не открывался вообще","File with this handle was closed or was not opening at all"},                // 5007
   {"Ошибочный хэндл файла","Wrong file handle"},                                                                                                  // 5008
   {"Файл должен быть открыт для записи","File must be opened for writing"},                                                                       // 5009
   {"Файл должен быть открыт для чтения","File must be opened for reading"},                                                                       // 5010
   {"Файл должен быть открыт как бинарный","File must be opened as binary one"},                                                                   // 5011
   {"Файл должен быть открыт как текстовый","File must be opened as text"},                                                                        // 5012
   {"Файл должен быть открыт как текстовый или CSV","File must be opened as text or CSV"},                                                         // 5013
   {"Файл должен быть открыт как CSV","File must be opened as CSV"},                                                                               // 5014
   {"Ошибка чтения файла","File reading error"},                                                                                                   // 5015
   {"Должен быть указан размер строки, так как файл открыт как бинарный","String size must be specified, because the file opened as binary"},      // 5016
   {
    "Для строковых массивов должен быть текстовый файл, для остальных – бинарный",                                                                 // 5017
    "Text file must be for string arrays, for other arrays - binary"
   },
   {"Это не файл, а директория","This is not file, this is directory"},                                                                            // 5018
   {"Файл не существует","File does not exist"},                                                                                                   // 5019
   {"Файл не может быть переписан","File cannot be rewritten"},                                                                                    // 5020
   {"Ошибочное имя директории","Wrong directory name"},                                                                                            // 5021
   {"Директория не существует","Directory does not exist"},                                                                                        // 5022
   {"Это файл, а не директория","This is file, not directory"},                                                                                    // 5023
   {"Директория не может быть удалена","Directory cannot be removed"},                                                                             // 5024
   {
    "Не удалось очистить директорию (возможно, один или несколько файлов заблокированы)",                                                          // 5025
    "Failed to clear directory (probably one or more files blocked)"
   },
   {"Не удалось записать ресурс в файл","Failed to write resource to file"},                                                                       // 5026
   {
    "Не удалось прочитать следующую порцию данных из CSV-файла, так как достигнут конец файла",                                                    // 5027
    "Unable to read next piece of data from CSV file, since end of file reached"
   },
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **String conversion errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5030 - 5044)             |
//| (String conversion)                                              |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_string[][TOTAL_LANG]=
  {
   {"В строке нет даты","No date in string"},                                                                                                  // 5030
   {"В строке ошибочная дата","Wrong date in string"},                                                                                         // 5031
   {"В строке ошибочное время","Wrong time in string"},                                                                                        // 5032
   {"Ошибка преобразования строки в дату","Error converting string to date"},                                                                  // 5033
   {"Недостаточно памяти для строки","Not enough memory for string"},                                                                          // 5034
   {"Длина строки меньше, чем ожидалось","String length less than expected"},                                                                  // 5035
   {"Слишком большое число, больше, чем ULONG_MAX","Too large number, more than ULONG_MAX"},                                                   // 5036
   {"Ошибочная форматная строка","Invalid format string"},                                                                                     // 5037
   {"Форматных спецификаторов больше, чем параметров","Amount of format specifiers more than parameters"},                                     // 5038
   {"Параметров больше, чем форматных спецификаторов","Amount of parameters more than format specifiers"},                                     // 5039
   {"Испорченный параметр типа string","Damaged parameter of string type"},                                                                    // 5040
   {"Позиция за пределами строки","Position outside string"},                                                                                  // 5041
   {"К концу строки добавлен 0, бесполезная операция","0 added to string end, useless operation"},                                             // 5042
   {"Неизвестный тип данных при конвертации в строку","Unknown data type when converting to string"},                                          // 5043
   {"Испорченный объект строки","Damaged string object"},                                                                                      // 5044
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Working with arrays errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5050 - 5063)             |
//| (Working with arrays)                                            |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_array[][TOTAL_LANG]=
  {
   {"Копирование несовместимых массивов","Copying incompatible arrays"},                                                                           // 5050
   {
    "Приемный массив объявлен как AS_SERIES, и он недостаточного размера",                                                                         // 5051
    "Receiving array declared as AS_SERIES, and it is of insufficient size"
   },
   {"Слишком маленький массив, стартовая позиция за пределами массива","Too small array, starting position outside the array"},                    // 5052
   {"Массив нулевой длины","Array of zero length"},                                                                                                // 5053
   {"Должен быть числовой массив","Must be numeric array"},                                                                                        // 5054
   {"Должен быть одномерный массив","Must be one-dimensional array"},                                                                              // 5055
   {"Таймсерия не может быть использована","Timeseries cannot be used"},                                                                           // 5056
   {"Должен быть массив типа double","Must be array of double type"},                                                                              // 5057
   {"Должен быть массив типа float","Must be array of float type"},                                                                                // 5058
   {"Должен быть массив типа long","Must be array of long type"},                                                                                  // 5059
   {"Должен быть массив типа int","Must be array of int type"},                                                                                    // 5060
   {"Должен быть массив типа short","Must be array of short type"},                                                                                // 5061
   {"Должен быть массив типа char","Must be array of char type"},                                                                                  // 5062
   {"Должен быть массив типа string","String array only"},                                                                                         // 5063
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Working with OpenCL errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5100 - 5114)             |
//| (Working with OpenCL)                                            |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_opencl[][TOTAL_LANG]=
  {
   {"Функции OpenCL на данном компьютере не поддерживаются","OpenCL functions not supported on this computer"},                                    // 5100
   {"Внутренняя ошибка при выполнении OpenCL","Internal error occurred when running OpenCL"},                                                      // 5101
   {"Неправильный хэндл OpenCL","Invalid OpenCL handle"},                                                                                          // 5102
   {"Ошибка при создании контекста OpenCL","Error creating the OpenCL context"},                                                                   // 5103
   {"Ошибка создания очереди выполнения в OpenCL","Failed to create run queue in OpenCL"},                                                         // 5104
   {"Ошибка при компиляции программы OpenCL","Error occurred when compiling OpenCL program"},                                                      // 5105
   {"Слишком длинное имя точки входа (кернел OpenCL)","Too long kernel name (OpenCL kernel)"},                                                     // 5106
   {"Ошибка создания кернел - точки входа OpenCL","Error creating OpenCL kernel"},                                                                 // 5107
   {
    "Ошибка при установке параметров для кернел OpenCL (точки входа в программу OpenCL)",                                                          // 5108
    "Error occurred when setting parameters for OpenCL kernel"
   },
   {"Ошибка выполнения программы OpenCL","OpenCL program runtime error"},                                                                          // 5109
   {"Неверный размер буфера OpenCL","Invalid size of OpenCL buffer"},                                                                              // 5110
   {"Неверное смещение в буфере OpenCL","Invalid offset in OpenCL buffer"},                                                                        // 5111
   {"Ошибка создания буфера OpenCL","Failed to create OpenCL buffer"},                                                                             // 5112
   {"Превышено максимальное число OpenCL объектов","Too many OpenCL objects"},                                                                     // 5113
   {"Ошибка выбора OpenCL устройства","OpenCL device selection error"},                                                                            // 5114
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Working with WebRequest() errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5200 - 5203)             |
//| (Working with WebRequest())                                      |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_webrequest[][TOTAL_LANG]=
  {
   {"URL не прошел проверку","Invalid URL"},                                                                                                       // 5200
   {"Не удалось подключиться к указанному URL","Failed to connect to specified URL"},                                                              // 5201
   {"Превышен таймаут получения данных","Timeout exceeded"},                                                                                       // 5202
   {"Ошибка в результате выполнения HTTP запроса","HTTP request failed"},                                                                          // 5203
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Working with network (sockets) errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5270 - 5275)             |
//| (Working with network (sockets))                                 |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_netsocket[][TOTAL_LANG]=
  {
   {"В функцию передан неверный хэндл сокета","Invalid socket handle passed to function"},                                                         // 5270
   {"Открыто слишком много сокетов (максимум 128)","Too many open sockets (max 128)"},                                                             // 5271
   {"Ошибка соединения с удаленным хостом","Failed to connect to remote host"},                                                                    // 5272
   {"Ошибка отправки/получения данных из сокета","Failed to send/receive data from socket"},                                                       // 5273
   {"Ошибка установления защищенного соединения (TLS Handshake)","Failed to establish secure connection (TLS Handshake)"},                         // 5274
   {"Отсутствуют данные о сертификате, которым защищено подключение","No data on certificate protecting connection"},                              // 5275
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (** **Custom symbols errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5300 - 5310)             |
//| (Custom symbols)                                                 |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_custom_symbol[][TOTAL_LANG]=
  {
   {"Должен быть указан пользовательский символ","Custom symbol must be specified"},                                                             // 5300
   {"Некорректное имя пользовательского символа","Name of custom symbol invalid"},                                                      // 5301
   {"Слишком длинное имя для пользовательского символа","Name of custom symbol too long"},                                              // 5302
   {"Слишком длинный путь для пользовательского символа","Path of custom symbol too long"},                                             // 5303
   {"Пользовательский символ с таким именем уже существует","Custom symbol with the same name already exists"},                                  // 5304
   {
    "Ошибка при создании, удалении или изменении пользовательского символа",                                                                     // 5305
    "Error occurred while creating, deleting or changing custom symbol"
   },
   {"Попытка удалить пользовательский символ, выбранный в обзоре рынка","You are trying to delete custom symbol selected in Market Watch"},      // 5306
   {"Неправильное свойство пользовательского символа","Invalid custom symbol property"},                                                         // 5307
   {"Ошибочный параметр при установке свойства пользовательского символа","Wrong parameter while setting property of custom symbol"},            // 5308
   {
    "Слишком длинный строковый параметр при установке свойства пользовательского символа",                                                       // 5309
    "Too long string parameter while setting property of custom symbol"
   },
   {"Не упорядоченный по времени массив тиков","Ticks in array not arranged in order of time"},                                                  // 5310
  };
//+------------------------------------------------------------------+
```

**Array of execution time error messages (Economic calendar errors section):**

```
//+------------------------------------------------------------------+
//| Array of execution time error messages (5400 - 5402)             |
//| (Economic calendar)                                              |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_calendar[][TOTAL_LANG]=
  {
   {"Размер массива недостаточен для получения описаний всех значений","Array size insufficient for receiving descriptions of all values"},     // 5400
   {"Превышен лимит запроса по времени","Request time limit exceeded"},                                                                         // 5401
   {"Страна не найдена","Country not found"},                                                                                                   // 5402
  };
//+------------------------------------------------------------------+
```

Since error codes from different subsections start with a number quite different from the error codes of the previous section, I have created
several arrays (each one corresponding to a separate error section) in order not to fill a single array with multiple empty cells.

For MQL4, write separate arrays since almost all return codes are different from codes in MQL5. We will not name the arrays according to their
codes belonging to a certain section. Instead, we will simply specify the range of codes stored in the array in the array's name.

**Array of messages for MQL4 trade server return codes:**

```
//+------------------------------------------------------------------+
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| Array of messages for MQL4 trade server return codes (0 - 150)   |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_ts_ret_code_mql4[][TOTAL_LANG]=
  {
   {"Нет ошибки","No error returned"},                                                             // 0
   {"Нет ошибки, но результат неизвестен","No error returned, but result unknown"},         // 1
   {"Общая ошибка","Common error"},                                                                // 2
   {"Неправильные параметры","Invalid trade parameters"},                                          // 3
   {"Торговый сервер занят","Trade server busy"},                                                  // 4
   {"Старая версия клиентского терминала","Old version of client terminal"},                       // 5
   {"Нет связи с торговым сервером","No connection with trade server"},                            // 6
   {"Недостаточно прав","Not enough rights"},                                                      // 7
   {"Слишком частые запросы","Too frequent requests"},                                             // 8
   {"Недопустимая операция, нарушающая функционирование сервера","Malfunctional trade operation"}, // 9

   {"Счет заблокирован","Account disabled"},                                                       // 64
   {"Неправильный номер счета","Invalid account"},                                                 // 65

   {"Истек срок ожидания совершения сделки","Trade timeout"},                                      // 128
   {"Неправильная цена","Invalid price"},                                                          // 129
   {"Неправильные стопы","Invalid stops"},                                                         // 130
   {"Неправильный объем","Invalid trade volume"},                                                  // 131
   {"Рынок закрыт","Market closed"},                                                               // 132
   {"Торговля запрещена","Trade disabled"},                                                        // 133
   {"Недостаточно денег для совершения операции","Not enough money"},                              // 134
   {"Цена изменилась","Price changed"},                                                            // 135
   {"Нет цен","Off quotes"},                                                                       // 136
   {"Брокер занят","Broker busy"},                                                                 // 137
   {"Новые цены","Requote"},                                                                       // 138
   {"Ордер заблокирован и уже обрабатывается","Order locked"},                                     // 139
   {"Разрешена только покупка","Buy orders only allowed"},                                         // 140
   {"Слишком много запросов","Too many requests"},                                                 // 141
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},     // 142
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},     // 143
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},     // 144
   {
    "Модификация запрещена, так как ордер слишком близок к рынку",                                 // 145
    "Modification denied because order too close to market"
   },
   {"Подсистема торговли занята","Trade context busy"},     // 146
   {"Использование даты истечения ордера запрещено брокером","Expirations denied by broker"},      // 147
   {
    "Количество открытых и отложенных ордеров достигло предела, установленного брокером",          // 148
    "Amount of open and pending orders reached limit set by broker"
   },
   {
    "Попытка открыть противоположный ордер в случае, если хеджирование запрещено",                 // 149
    "Attempt to open order opposite to existing one when hedging disabled"
   },
   {
    "Попытка закрыть позицию по инструменту в противоречии с правилом FIFO",                       // 150
    "Attempt to close order contravening FIFO rule"
   },
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (code range 4000** **— 4030):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (4000 - 4030)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_4000_4030[][TOTAL_LANG]=
  {
   {"Нет ошибки","No error returned"},                                                             // 4000
   {"Неправильный указатель функции","Wrong function pointer"},                                    // 4001
   {"Индекс массива - вне диапазона","Array index out of range"},                                  // 4002
   {"Нет памяти для стека функций","No memory for function call stack"},                           // 4003
   {"Переполнение стека после рекурсивного вызова","Recursive stack overflow"},                    // 4004
   {"На стеке нет памяти для передачи параметров","Not enough stack for parameter"},               // 4005
   {"Нет памяти для строкового параметра","No memory for parameter string"},                       // 4006
   {"Нет памяти для временной строки","No memory for temp string"},                                // 4007
   {"Неинициализированная строка","Not initialized string"},                                       // 4008
   {"Неинициализированная строка в массиве","Not initialized string in array"},                    // 4009
   {"Нет памяти для строкового массива","No memory for array string"},                             // 4010
   {"Слишком длинная строка","Too long string"},                                                   // 4011
   {"Остаток от деления на ноль","Remainder from zero divide"},                                    // 4012
   {"Деление на ноль","Zero divide"},                                                              // 4013
   {"Неизвестная команда","Unknown command"},                                                      // 4014
   {"Неправильный переход","Wrong jump (never generated error)"},                                  // 4015
   {"Неинициализированный массив","Not initialized array"},                                        // 4016
   {"Вызовы DLL не разрешены","DLL calls not allowed"},                                            // 4017
   {"Невозможно загрузить библиотеку","Cannot load library"},                                      // 4018
   {"Невозможно вызвать функцию","Cannot call function"},                                          // 4019
   {"Вызовы внешних библиотечных функций не разрешены","Expert function calls not allowed"},       // 4020
   {
    "Недостаточно памяти для строки, возвращаемой из функции",                                     // 4021
    "Not enough memory for temp string returned from function"
   },
   {"Система занята","System busy (never generated error)"},                                       // 4022
   {"Критическая ошибка вызова DLL-функции","DLL-function call critical error"},                   // 4023
   {"Внутренняя ошибка","Internal error"},                                                         // 4024
   {"Нет памяти","Out of memory"},                                                                 // 4025
   {"Неверный указатель","Invalid pointer"},                                                       // 4026
   {
    "Слишком много параметров форматирования строки",                                              // 4027
    "Too many formatters in format function"
   },
   {
    "Число параметров превышает число параметров форматирования строки",                           // 4028
    "Parameters count exceeds formatters count"
   },
   {"Неверный массив","Invalid array"},                                                            // 4029
   {"График не отвечает","No reply from chart"},                                                   // 4030
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range** **4050** **— 4075):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (4050 - 4075)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_4050_4075[][TOTAL_LANG]=
  {
   {"Неправильное количество параметров функции","Invalid function parameters count"},             // 4050
   {"Недопустимое значение параметра функции","Invalid function parameter value"},                 // 4051
   {"Внутренняя ошибка строковой функции","String function internal error"},                       // 4052
   {"Ошибка массива","Some array error"},                                                          // 4053
   {"Неправильное использование массива-таймсерии","Incorrect series array using"},                // 4054
   {"Ошибка пользовательского индикатора","Custom indicator error"},                               // 4055
   {"Массивы несовместимы","Arrays incompatible"},                                                 // 4056
   {"Ошибка обработки глобальных переменных","Global variables processing error"},                 // 4057
   {"Глобальная переменная не обнаружена","Global variable not found"},                            // 4058
   {"Функция не разрешена в тестовом режиме","Function not allowed in testing mode"},              // 4059
   {"Функция не разрешена","Function not allowed for call"},                                       // 4060
   {"Ошибка отправки почты","Send mail error"},                                                    // 4061
   {"Ожидается параметр типа string","String parameter expected"},                                 // 4062
   {"Ожидается параметр типа integer","Integer parameter expected"},                               // 4063
   {"Ожидается параметр типа double","Double parameter expected"},                                 // 4064
   {"В качестве параметра ожидается массив","Array as parameter expected"},                        // 4065
   {
    "Запрошенные исторические данные в состоянии обновления",                                      // 4066
    "Requested history data in updating state"
   },
   {"Ошибка при выполнении торговой операции","Internal trade error"},                             // 4067
   {"Ресурс не найден","Resource not found"},                                                      // 4068
   {"Ресурс не поддерживается","Resource not supported"},                                          // 4069
   {"Дубликат ресурса","Duplicate resource"},                                                      // 4070
   {"Ошибка инициализации пользовательского индикатора","Custom indicator cannot initialize"},     // 4071
   {"Ошибка загрузки пользовательского индикатора","Cannot load custom indicator"},                // 4072
   {"Нет исторических данных","No history data"},                                                  // 4073
   {"Не хватает памяти для исторических данных","No memory for history data"},                     // 4074
   {"Не хватает памяти для расчёта индикатора","Not enough memory for indicator calculation"},     // 4075
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range 4099** **— 4112):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (4099 - 4112)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_4099_4112[][TOTAL_LANG]=
  {
   {"Конец файла","End of file"},                                                                  // 4099
   {"Ошибка при работе с файлом","Some file error"},                                               // 4100
   {"Неправильное имя файла","Wrong file name"},                                                   // 4101
   {"Слишком много открытых файлов","Too many opened files"},                                      // 4102
   {"Невозможно открыть файл","Cannot open file"},                                                 // 4103
   {"Несовместимый режим доступа к файлу","Incompatible access to file"},                          // 4104
   {"Ни один ордер не выбран","No order selected"},                                                // 4105
   {"Неизвестный символ","Unknown symbol"},                                                        // 4106
   {"Неправильный параметр цены для торговой функции","Invalid price"},                            // 4107
   {"Неверный номер тикета","Invalid ticket"},                                                     // 4108
   {
    "Торговля не разрешена. Необходимо включить опцию \"Разрешить советнику торговать\" в свойствах эксперта", // 4109
    "Trading not allowed. Enable \"Allow live trading\" checkbox in Expert Advisor properties"
   },
   {
    "Ордера на покупку не разрешены. Необходимо проверить свойства эксперта",                      // 4110
    "Longs not allowed. Check Expert Advisor properties"
   },
   {
    "Ордера на продажу не разрешены. Необходимо проверить свойства эксперта",                      // 4111
    "Shorts not allowed. Check Expert Advisor properties"
   },
   {
    "Автоматическая торговля с помощью экспертов/скриптов запрещена на стороне сервера",           // 4112
    "Automated trading by Expert Advisors/Scripts disabled by trade server"
   },
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range 4200** **— 4220):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (4200 - 4220)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_4200_4220[][TOTAL_LANG]=
  {
   {"Объект уже существует","Object already exists"},                                              // 4200
   {"Запрошено неизвестное свойство объекта","Unknown object property"},                           // 4201
   {"Объект не существует","Object does not exist"},                                               // 4202
   {"Неизвестный тип объекта","Unknown object type"},                                              // 4203
   {"Нет имени объекта","No object name"},                                                         // 4204
   {"Ошибка координат объекта","Object coordinates error"},                                        // 4205
   {"Не найдено указанное подокно","No specified subwindow"},                                      // 4206
   {"Ошибка при работе с объектом","Graphical object error"},                                      // 4207
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4208
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4209
   {"Неизвестное свойство графика","Unknown chart property"},                                      // 4210
   {"График не найден","Chart not found"},                                                         // 4211
   {"Не найдено подокно графика","Chart subwindow not found"},                                     // 4212
   {"Индикатор не найден","Chart indicator not found"},                                            // 4213
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4214
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4215
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4216
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4217
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4218
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4219
   {"Ошибка выбора инструмента","Symbol select error"},                                            // 4220
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range 4250** **— 4266):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (4250 - 4266)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_4250_4266[][TOTAL_LANG]=
  {
   {"Ошибка отправки push-уведомления","Notification error"},                                      // 4250
   {"Ошибка параметров push-уведомления","Notification parameter error"},                          // 4251
   {"Уведомления запрещены","Notifications disabled"},                                             // 4252
   {"Слишком частые запросы отсылки push-уведомлений","Notification send requests too frequent"},           // 4253
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4254
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4255
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4256
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4257
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4258
   {"Неизвестный код ошибки","Unknown error code"},                                       // 4259
   {"Не указан FTP сервер","FTP server not specified"},                                         // 4260
   {"Не указан FTP логин","FTP login not specified"},                                           // 4261
   {"Ошибка при подключении к FTP серверу","FTP connection failed"},                               // 4262
   {"Подключение к FTP серверу закрыто","FTP connection closed"},                                  // 4263
   {"На FTP сервере не найдена директория для выгрузки файла ","FTP path not found on server"},    // 4264
   {
    "Не найден файл в директории MQL4\\Files для отправки на FTP сервер",                          // 4265
    "File not found in MQL4\\Files directory to send to FTP server"
   },
   {"Ошибка при передаче файла на FTP сервер","Common error during FTP data transmission"},        // 4266
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range 5001** **— 5029):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (5001 - 5029)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_5001_5029[][TOTAL_LANG]=
  {
   {"Слишком много открытых файлов","Too many opened files"},                                      // 5001
   {"Неверное имя файла","Wrong file name"},                                                       // 5002
   {"Слишком длинное имя файла","Too long file name"},                                             // 5003
   {"Ошибка открытия файла","Cannot open file"},                                                   // 5004
   {"Ошибка размещения буфера текстового файла","Text file buffer allocation error"},              // 5005
   {"Ошибка удаления файла","Cannot delete file"},                                                 // 5006
   {
    "Неверный хендл файла (файл закрыт или не был открыт)",                                        // 5007
    "Invalid file handle (file closed or was not opened)"
   },
   {
    "Неверный хендл файла (индекс хендла отсутствует в таблице)",                                  // 5008
    "Wrong file handle (handle index out of handle table)"
   },
   {"Файл должен быть открыт с флагом FILE_WRITE","File must be opened with FILE_WRITE flag"},     // 5009
   {"Файл должен быть открыт с флагом FILE_READ","File must be opened with FILE_READ flag"},       // 5010
   {"Файл должен быть открыт с флагом FILE_BIN","File must be opened with FILE_BIN flag"},         // 5011
   {"Файл должен быть открыт с флагом FILE_TXT","File must be opened with FILE_TXT flag"},         // 5012
   {
    "Файл должен быть открыт с флагом FILE_TXT или FILE_CSV",                                      // 5013
    "File must be opened with FILE_TXT or FILE_CSV flag"
   },
   {"Файл должен быть открыт с флагом FILE_CSV","File must be opened with FILE_CSV flag"},         // 5014
   {"Ошибка чтения файла","File read error"},                                                      // 5015
   {"Ошибка записи файла","File write error"},                                                     // 5016
   {
    "Размер строки должен быть указан для двоичных файлов",                                        // 5017
    "String size must be specified for binary file"
   },
   {
    "Неверный тип файла (для строковых массивов-TXT, для всех других-BIN)",                        // 5018
    "Incompatible file (for string arrays-TXT, for others-BIN)"
   },
   {"Файл является директорией","File is directory not file"},                                     // 5019
   {"Файл не существует","File does not exist"},                                                   // 5020
   {"Файл не может быть перезаписан","File cannot be rewritten"},                                  // 5021
   {"Неверное имя директории","Wrong directory name"},                                             // 5022
   {"Директория не существует","Directory does not exist"},                                        // 5023
   {"Указанный файл не является директорией","Specified file is not directory"},                   // 5024
   {"Ошибка удаления директории","Cannot delete directory"},                                       // 5025
   {"Ошибка очистки директории","Cannot clean directory"},                                         // 5026
   {"Ошибка изменения размера массива","Array resize error"},                                      // 5027
   {"Ошибка изменения размера строки","String resize error"},                                      // 5028
   {
    "Структура содержит строки или динамические массивы",                                          // 5029
    "Structure contains strings or dynamic arrays"
   },
  };
//+------------------------------------------------------------------+
```

**Array of MQL4 execution time error messages (** **code range 5200** **— 5203):**

```
//+------------------------------------------------------------------+
//| Array of MQL4 execution time error messages (5200 - 5203)        |
//| (1) in user's country language                                   |
//| (2) in the international language                                |
//+------------------------------------------------------------------+
string messages_runtime_5200_5203[][TOTAL_LANG]=
  {
   {"URL не прошел проверку","Invalid URL"},                                                       // 5200
   {"Не удалось подключиться к указанному URL","Failed to connect to specified URL"},              // 5201
   {"Превышен таймаут получения данных","Timeout exceeded"},                                       // 5202
   {"Ошибка в результате выполнения HTTP запроса","HTTP request failed"},                          // 5203
  };
#endif
//+------------------------------------------------------------------+
```

We need to include this file to the library. Since the **Defines.mqh** file with macro substitutions and enumerations is
already connected to the library and is visible anywhere in it,

**include the new file Datas.mqh**
**to Defines.mqh** **— then it will also become visible anywhere in the library:**

```
//+------------------------------------------------------------------+
//|                                                      Defines.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Datas.mqh"
#ifdef __MQL4__
#include "ToMQL4.mqh"
#endif
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
```

This completes preparation of text message data. If new classes are added to the library and you need messages from them to be displayed, all new
messages whose texts are not yet present in the created message text are added to the data table.

**Now we need to create the class for displaying messages from the data we have prepared**.

The class should be the same for the entire library. We can use the class with static variables and methods accessible from anywhere in the
library. There is no need to create separate message class instances for each library class. Since the class is an auxiliary program method,
it is to be located in the library folder meant for service classes (

**Services**).

### Class for displaying messages

Create the new file of the **CMessage** **class** named
\\MQL5\\Include\\DoEasy\\Services\

**Message.mqh**.

**Include the Defines.mqh file to it right away:**

```
//+------------------------------------------------------------------+
//|                                                      Message.mqh |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\Defines.mqh"
//+------------------------------------------------------------------+
//| Message class                                                    |
//+------------------------------------------------------------------+
class CMessage
  {
private:

public:

  };
//+------------------------------------------------------------------+
```

Now the message display class can see the previously prepared data arrays of all library messages and standard return codes.

**Add static class member variables and statistical method of receiving data from arrays (the message data table) by message ID to the**
**private section of the class:**

```
//+------------------------------------------------------------------+
//| Message class                                                    |
//+------------------------------------------------------------------+
class CMessage
  {
private:
   static int        m_global_error;
   static uchar      m_lang_num;
   static string     m_subject;
   static string     m_text;
   static bool       m_log;
   static bool       m_push;
   static bool       m_mail;
//--- Get a message from the necessary array by message ID
   static void       GetTextByID(const int msg_id);
public:
```

The **m\_global\_error variable** is meant for storing the error
code during the class methods operation.

The **m\_lang\_num** variable is to feature the number of the
necessary message language (0 — Russian, 1 — English, 2, 3, ..., N — any other necessary languages added by a user).

The **m\_subject variable** is used to set a header of an email
message sent to an email address by the

[SendMail()](https://www.mql5.com/en/docs/network/sendmail) function.

The **m\_text**
variable receives a message from the arrays by a specified ID and language index set in the m\_lang\_num variable.


The

**m\_log** variable stores the global flag of enabling/disabling
sending messages to the journal.

The **m\_push** variable stores the global flag of
enabling/disabling sending push notifications to a mobile device.

The **m\_mail** variable stores the global flag of
enabling/disabling sending messages to an email address.

The method of receiving an array by a message ID writes the
necessary message indicated by its ID (either from the library message enumeration or the error code returned by the GetLastError()
function) to the

**m\_text** variable.

**Add the necessary methods to the public section of the class:**

```
//+------------------------------------------------------------------+
//| Message class                                                    |
//+------------------------------------------------------------------+
class CMessage
  {
private:
   static int        m_global_error;
   static uchar      m_lang_num;
   static string     m_subject;
   static string     m_text;
   static bool       m_log;
   static bool       m_push;
   static bool       m_mail;
//--- Get a message from the necessary array by message ID
   static void       GetTextByID(const int msg_id);
public:
//--- (1) Display a text message in the journal, send a push notification and e-mail,
//--- (2) Display a message by ID, send a push notification and e-mail,
//--- (3) play an audio file
   static bool       Out(const string text,const bool push=false,const bool mail=false,const string subject=NULL);
   static bool       OutByID(const int msg_id,const bool code=true);
   static bool       PlaySound(const string file_name);
   //--- Return (1) a message, (2) a code in the "(code)" format
   static string     Text(const int msg_id);
   static string     Retcode(const int msg_id)     { return "("+(string)msg_id+")";    }
//--- Set (1) the text message language index (0 - user's country language, 1 - English, 2 ... N - added by a user),
//--- (2) email header,
//--- the flag of sending messages (3) to the journal, (4) a mobile device, (5) a mailbox
   static void       SetLangNum(const uchar num=0) { CMessage::m_lang_num=num;         }
   static void       SetSubject(const string subj) { CMessage::m_subject=subj;         }
   static void       SetLog(const bool flag)       { CMessage::m_log=flag;             }
   static void       SetPush(const bool flag)      { CMessage::m_push=flag;            }
   static void       SetMail(const bool flag)      { CMessage::m_mail=flag;            }
//--- (1) display a message in the journal by ID, (2) e-mail, (3) mobile device
   static void       ToLog(const int msg_id,const bool code=false);
   static bool       ToMail(const string message,const string subject=NULL);
   static bool       Push(const string message);
//--- (1) send a file to FTP, (2) return an error code
   static bool       ToFTP(const string filename,const string ftp_path=NULL);
   static int        GetError(void)                { return CMessage::m_global_error;  }
  };
//+------------------------------------------------------------------+
```

Each method features the description of its belonging to the class listing. We will consider this a bit later.

As we know, " [the static member of a class should be explicitly \\
initialized with a desired value. To do this, it should be declared and initialized at the global level](https://www.mql5.com/en/docs/basis/oop/staticmembers)". Therefore, write
initialization of all its static variables in the code beyond the class body:

```
//+------------------------------------------------------------------+
//| Initialization of static variables                               |
//+------------------------------------------------------------------+
uchar  CMessage::m_lang_num=(::TerminalInfoString(TERMINAL_LANGUAGE)==COUNTRY_LANG ? 0 : 1);
int    CMessage::m_global_error=ERR_SUCCESS;
string CMessage::m_subject=::MQLInfoString(MQL_PROGRAM_NAME);
string CMessage::m_text=NULL;
bool   CMessage::m_log=true;
bool   CMessage::m_push=true;
bool   CMessage::m_mail=false;
//+------------------------------------------------------------------+
```

Here we initialize all variables by their default values:

Text
messages language index is set depending on the language set in the terminal. If the terminal language returned using
TerminalInfoString(

[TERMINAL\_LANGUAGE](https://www.mql5.com/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_string)) matches the
user's country language (in the current version, it is Russian), the message language index is equal to zero, otherwise the message
language is English corresponding to the language index 1.

The error code is set to "No error".

A default email
header for sending to a mailbox is a program name returned using MQLInfoString( [MQL\_PROGRAM\_NAME](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_mql_info_string)).

The message text is absent by default, sending
to the journal is allowed, push notifications are
enabled,

while sending to email is disabled.

Let's have a look at the implementation of the class methods.

**The method for displaying a text message in the journal, as well as sending a push notification and an email message:**

```
//+--------------------------------------------------------------------------+
//| Display a message in the journal, send a push notification and an e-mail |
//+--------------------------------------------------------------------------+
bool CMessage::Out(const string text,const bool push=false,const bool mail=false,const string subject=NULL)
  {
   bool res=true;
   if(CMessage::m_log)
      ::Print(text);
   if(push)
      res &=CMessage::Push(text);
   if(mail)
      res &=CMessage::ToMail(text,subject);
   return res;
  }
//+------------------------------------------------------------------+
```

The method passes a message text and the flags
indicating the necessity to send a push notification and an email message,
as well as the

email header.

First, check the global flag allowing the display of messages in the journal.
If it is enabled,

display the message.

Next, check
the appropriate flags for sending a push notification
and an

email passed to the method using the parameters. If the flags are set,
call the appropriate methods for sending messages. The

**res** local variable receivesthe operation result of called methods for sending messages. Initially, it is set to true.
If at least one of the methods returns

false, the value is set for the variable. The

**res** variable final value is returned from the method.

**The method for displaying a message in the journal by the message ID, as well as sending a push notification and an email message:**

```
//+------------------------------------------------------------------+
//| Display a message in the journal by ID,                          |
//| send a push notification and an e-mail                           |
//+------------------------------------------------------------------+
bool CMessage::OutByID(const int msg_id,const bool code=true)
  {
   bool res=true;
   if(CMessage::m_log)
      CMessage::ToLog(msg_id,code);
   else
      CMessage::GetTextByID(msg_id);
   if(CMessage::m_push)
      res &=CMessage::Push(CMessage::m_text);
   if(CMessage::m_mail)
      res &=CMessage::ToMail(CMessage::m_text,CMessage::m_subject);
   return res;
  }
//+------------------------------------------------------------------+
```

The method receives the message ID (index of its location in the
message array or the error code) and the

display flag after the error code message.

If
the flag for displaying in the journal is enabled, the method for
displaying a message in the journal by ID is called.

If the flag for displaying in the journal is not set, receive
a text message by ID and try to send the text message to a mobile
device and email depending on whether
the flags of these actions are set.

Like in the previous method, the results of calling the methods for sending a message to a
mobile device and a mailbox are written to the

**res** local variable whose value is ultimately returned from the method.

**The method displaying a message to the journal by a message ID:**

```
//+------------------------------------------------------------------+
//| Display a message in the journal by a message ID                 |
//+------------------------------------------------------------------+
void CMessage::ToLog(const int msg_id,const bool code=false)
  {
   CMessage::GetTextByID(msg_id);
   ::Print(m_text,(!code || msg_id>ERR_USER_ERROR_FIRST-1 ? "" : " "+CMessage::Retcode(msg_id)));
  }
//+------------------------------------------------------------------+
```

The method receives the message ID (index of its location in the
message array or the error code) and the

display flag after the error code message.

Next, use a message
ID to receive a message text in a specified language from
the library message arrays and

display the obtained text in the journal.

If
the error code display flag is not set or a message ID corresponds to the start
of the library message range, there is no need to display the error code.
Otherwise,

the error code text is added to a message text.

**The methods for sending emails and push notifications to a mobile device, as well as sending a file to an FTP server:**

```
//+------------------------------------------------------------------+
//| Send an email                                                    |
//+------------------------------------------------------------------+
bool CMessage::ToMail(const string message,const string subject=NULL)
  {
   //--- If sending emails is disabled in the terminal
   if(!::TerminalInfoInteger(TERMINAL_EMAIL_ENABLED))
     {
      /--- display the appropriate message in the journal, write the error code and return 'false'
      CMessage::ToLog(MSG_LIB_TEXT_NOT_MAIL_ENABLED,false);
      CMessage::m_global_error=ERR_MAIL_SEND_FAILED;
      return false;
     }
   //--- If failed to send a message
   if(!::SendMail(subject==NULL ? CMessage::m_subject : subject,message))
     {
      //--- write an error code, create an error description text, display it in the journal and return 'false'
      CMessage::m_global_error=::GetLastError();
      string txt=CMessage::Text(MSG_LIB_SYS_ERROR)+CMessage::Text(CMessage::m_global_error);
      string code=CMessage::Retcode(CMessage::m_global_error);
      ::Print(txt+" "+code);
      return false;
     }
   //--- Successful - return 'true'
   return true;
  }
//+------------------------------------------------------------------+
//| Send push notifications to a mobile device                       |
//+------------------------------------------------------------------+
bool CMessage::Push(const string message)
  {
   //--- If sending push notifications is not allowed in the terminal
   if(!::TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED))
     {
      /--- display the appropriate message in the journal, write the error code and return 'false'
      CMessage::ToLog(MSG_LIB_TEXT_NOT_PUSH_ENABLED,false);
      CMessage::m_global_error=ERR_NOTIFICATION_SEND_FAILED;
      return false;
     }
   //--- If failed to send a message
   if(!::SendNotification(message))
     {
      //--- write an error code, create an error description text, display it in the journal and return 'false'
      CMessage::m_global_error=::GetLastError();
      string txt=CMessage::Text(MSG_LIB_SYS_ERROR)+CMessage::Text(CMessage::m_global_error);
      string code=CMessage::Retcode(CMessage::m_global_error);
      ::Print(txt+" "+code);
      return false;
     }
   //--- Successful - return 'true'
   return true;
  }
//+------------------------------------------------------------------+
//| Send a file to a specified address                               |
//+------------------------------------------------------------------+
bool CMessage::ToFTP(const string filename,const string ftp_path=NULL)
  {
   //--- If sending files to an FTP server is not allowed in the terminal
   if(!::TerminalInfoInteger(TERMINAL_FTP_ENABLED))
     {
      /--- display the appropriate message in the journal, write the error code and return 'false'
      CMessage::ToLog(MSG_LIB_TEXT_NOT_FTP_ENABLED,false);
      CMessage::m_global_error=ERR_FTP_SEND_FAILED;
      return false;
     }
   //--- If failed to send a file
   if(!::SendFTP(filename,ftp_path))
     {
      //--- write an error code, create an error description text, display it in the journal and return 'false'
      CMessage::m_global_error=::GetLastError();
      string txt=CMessage::Text(MSG_LIB_SYS_ERROR)+CMessage::Text(CMessage::m_global_error);
      string code=CMessage::Retcode(CMessage::m_global_error);
      ::Print(txt+" "+code);
      return false;
     }
   return true;
  }
//+------------------------------------------------------------------+
```

All three methods are logically identical and simple. The entire logic is commented in the method listing making it easy to understand.

**The method for playing an audio file:**

```
//+------------------------------------------------------------------+
//| Play an audio file                                               |
//+------------------------------------------------------------------+
bool CMessage::PlaySound(const string file_name)
  {
   bool res=::PlaySound(file_name);
   CMessage::m_global_error=(res ? ERR_SUCCESS : ::GetLastError());
   return res;
  }
//+------------------------------------------------------------------+
```

The **res** variable receives the results of the [PlaySound()](https://www.mql5.com/en/docs/common/playsound)
function operation. If the function works correctly, write the error absence code to the **m\_global\_error** variable.
Otherwise, write the last error code to it.

As a result, return the **res** variable value where the result of the PlaySound() function operation is stored.

**The method returning the message text by ID:**

```
//+------------------------------------------------------------------+
//| Return the message text by a message ID                          |
//+------------------------------------------------------------------+
string CMessage::Text(const int msg_id)
  {
   CMessage::GetTextByID(msg_id);
   return m_text;
  }
//+------------------------------------------------------------------+
```

The method receives the message ID. Next, the
GetTextByID() method is used to retrieve the necessary message from the appropriate text array and the message
text is returned to the calling program.

**The method retrieving a message text by a message ID from the appropriate text array:**

```
//+------------------------------------------------------------------+
//| Get messages from the text array by an ID                        |
//+------------------------------------------------------------------+
void CMessage::GetTextByID(const int msg_id)
  {
   CMessage::m_text=
     (
      //--- Runtime errors (0, 4001 - 4019)
      msg_id==0                     ?  messages_runtime[msg_id][m_lang_num]                       :
     #ifdef __MQL5__
      msg_id>4000 && msg_id<4020    ?  messages_runtime[msg_id-4000][m_lang_num]                  :
      //--- Runtime errors (Charts 4101 - 4116)
      msg_id>4100 && msg_id<4117    ?  messages_runtime_charts[msg_id-4101][m_lang_num]           :
      //--- Runtime errors (Charts 4201 - 4205)
      msg_id>4200 && msg_id<4206    ?  messages_runtime_graph_obj[msg_id-4201][m_lang_num]        :
      //--- Runtime errors (MarketInfo 4301 - 4305)
      msg_id>4300 && msg_id<4306    ?  messages_runtime_market[msg_id-4301][m_lang_num]           :
      //--- Runtime errors (Access to history 4401 - 4407)
      msg_id>4400 && msg_id<4408    ?  messages_runtime_history[msg_id-4401][m_lang_num]          :
      //--- Runtime errors (Global Variables 4501 - 4524)
      msg_id>4500 && msg_id<4525    ?  messages_runtime_global[msg_id-4501][m_lang_num]           :
      //--- Runtime errors (Custom indicators 4601 - 4603)
      msg_id>4600 && msg_id<4604    ?  messages_runtime_custom_indicator[msg_id-4601][m_lang_num] :
      //--- Runtime errors (Account 4701 - 4758)
      msg_id>4700 && msg_id<4759    ?  messages_runtime_account[msg_id-4701][m_lang_num]          :
      //--- Runtime errors (Indicators 4801 - 4812)
      msg_id>4800 && msg_id<4813    ?  messages_runtime_indicator[msg_id-4801][m_lang_num]        :
      //--- Runtime errors (Market depth 4901 - 4904)
      msg_id>4900 && msg_id<4905    ?  messages_runtime_books[msg_id-4901][m_lang_num]            :
      //--- Runtime errors (File operations 5001 - 5027)
      msg_id>5000 && msg_id<5028    ?  messages_runtime_files[msg_id-5001][m_lang_num]            :
      //--- Runtime errors (Converting strings 5030 - 5044)
      msg_id>5029 && msg_id<5045    ?  messages_runtime_string[msg_id-5030][m_lang_num]           :
      //--- Runtime errors (Working with arrays 5050 - 5063)
      msg_id>5049 && msg_id<5064    ?  messages_runtime_array[msg_id-5050][m_lang_num]            :
      //--- Runtime errors (Working with OpenCL 5100 - 5114)
      msg_id>5099 && msg_id<5115    ?  messages_runtime_opencl[msg_id-5100][m_lang_num]           :
      //--- Runtime errors (Working with WebRequest() 5200 - 5203)
      msg_id>5199 && msg_id<5204    ?  messages_runtime_webrequest[msg_id-5200][m_lang_num]       :
      //--- Runtime errors (Working with network (sockets) 5270 - 5275)
      msg_id>5269 && msg_id<5276    ?  messages_runtime_netsocket[msg_id-5270][m_lang_num]        :
      //--- Runtime errors (Custom symbols 5300 - 5310)
      msg_id>5299 && msg_id<5311    ?  messages_runtime_custom_symbol[msg_id-5300][m_lang_num]    :
      //--- Runtime errors (Economic calendar 5400 - 5402)
      msg_id>5399 && msg_id<5403    ?  messages_runtime_calendar[msg_id-5400][m_lang_num]         :
      //--- Trade server return codes (10004 - 10044)
      msg_id>10003 && msg_id<10045  ?  messages_ts_ret_code[msg_id-10004][m_lang_num]             :
     #else // MQL4
      msg_id>0 && msg_id<10         ?  messages_ts_ret_code_mql4[msg_id][m_lang_num]              :
      msg_id>63 && msg_id<66        ?  messages_ts_ret_code_mql4[msg_id-54][m_lang_num]           :
      msg_id>127 && msg_id<151      ?  messages_ts_ret_code_mql4[msg_id-116][m_lang_num]          :
      msg_id<4000                   ?  messages_ts_ret_code_mql4[26][m_lang_num]                  :
      //--- MQL4 runtime errors (4000 - 4030)
      msg_id<4031                   ?  messages_runtime_4000_4030[msg_id-4000][m_lang_num]        :
      //--- MQL4 runtime errors (4050 - 4075)
      msg_id>4049 && msg_id<4076    ?  messages_runtime_4050_4075[msg_id-4050][m_lang_num]        :
      //--- MQL4 runtime errors (4099 - 4112)
      msg_id>4098 && msg_id<4113    ?  messages_runtime_4099_4112[msg_id-4099][m_lang_num]        :
      //--- MQL4 runtime errors (4200 - 4220)
      msg_id>4199 && msg_id<4221    ?  messages_runtime_4200_4220[msg_id-4200][m_lang_num]        :
      //--- MQL4 runtime errors (4250 - 4266)
      msg_id>4249 && msg_id<4267    ?  messages_runtime_4250_4266[msg_id-4250][m_lang_num]        :
      //--- MQL4 runtime errors (5001 - 5029)
      msg_id>5000 && msg_id<5030    ?  messages_runtime_5001_5029[msg_id-5001][m_lang_num]        :
      //--- MQL4 runtime errors (5200 - 5203)
      msg_id>5199 && msg_id<5204    ?  messages_runtime_5200_5203[msg_id-5200][m_lang_num]        :
     #endif
      //--- Library messages (ERR_USER_ERROR_FIRST)
      msg_id>ERR_USER_ERROR_FIRST-1 ?  messages_library[msg_id-ERR_USER_ERROR_FIRST][m_lang_num]  :
      messages_library[MSG_LIB_SYS_ERROR_CODE_OUT_OF_RANGE-ERR_USER_ERROR_FIRST][m_lang_num]
     );
  }
//+------------------------------------------------------------------+
```

Here all is simple. Depending on the numerical value of an ID passed to the method, select an array the texts within the current range are located
in. To obtain a text index in the array, we need to subtract the initial value of the very first ID (whose description is stored in the zero index
of the array's first dimension) from the ID value. From the array's second dimension (namely, from the cell the value set in the

**m\_lang\_num** variable points at), retrieve an ID's text description and write it to the **m\_text** variable.


All is set. Now the m\_text variable stores the text description of a message whose ID was passed to the method.

**This completes the development of the message class**.

Further on, I will supplement the class with new
functionality or adjust the existing one if necessary. At this stage, all that has been done is sufficient to store all text messages in the
message database in a separate file and gain access to the necessary message by a single value — a message ID, which is actually a pointer to the
array cell index where the necessary message is stored.

The only thing that needs to be done additionally is to consider the shift of the real ID value relative to its location in the array. This
has been done in the method we just examined.

The only thing left is to find all entries of the "TextByLanguage(" strings, replace them with "CMessage::Text(" and add the message ID.

### Improving library classes

To search in all files, use the "Find In Files" window ( **Shift+Ctrl+F**):

![](https://c.mql5.com/2/37/metaeditor64_OXRgddD4HR.png)

In the "Find what" field, enter "TextByLanguage(", while in the "In folder" folder, select the path to the library directory (in my case, it
is E:\\MetaQuotes\\MetaTrader 5\\MQL5\\Include\\DoEasy). Make sure to check the "Look in subfolders" box and click Find.

The paths to the files containing the necessary string are displayed in the editor's Search tab upon the search completion.

It only
remains to open each of the files (this can be done by double-clicking the file path line in the search results tab) and either use a simple
search (

**Ctrl+F**) of the same line but in a separate file or use a search with replacement ( **Ctrl+H**) and replace each
detected "TextByLanguage(" string with "CMessage::Text(".

In any case, after the replacement, we need to replace the text in the parameters
TextByLanguage("Этот\_текст\_заменить","Replace\_this\_text")
with the message ID constant. Finding a constant is simple: highlight the Russian text, press

**Ctrl+F** (the search window is opened), move to the tab with the **Datas.mqh** file and click "Find Next" in the search
window. Since we have a comment to each constant, the comment related to the

necessary constant is found. Copy the constant and enter it to
CMessage::Text(

FOUND\_CONSTANT).

For example, in the **BaseObj.mqh** file, the following string
has been found:

```
//+------------------------------------------------------------------+
//| Pack a 'ushort' number to a passed 'long' number                 |
//+------------------------------------------------------------------+
long CBaseObj::UshortToLong(const ushort ushort_value,const uchar index,long &long_value)
  {
   if(index>3)
     {
      ::Print(DFUN,TextByLanguage("Ошибка. Значение \"index\" должно быть в пределах 0 - 3","Error. \"index\" value should be between 0 - 3"));
      return 0;
     }
   return(long_value |= UshortToByte(ushort_value,index));
  }
//+------------------------------------------------------------------+
```

After the replacement, it will look as follows:

```
//+------------------------------------------------------------------+
//| Pack a 'ushort' number to a passed 'long' number                 |
//+------------------------------------------------------------------+
long CBaseObj::UshortToLong(const ushort ushort_value,const uchar index,long &long_value)
  {
   if(index>3)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_INDEX));
      return 0;
     }
   return(long_value |= UshortToByte(ushort_value,index));
  }
//+------------------------------------------------------------------+
```

All the necessary replacements in all library files containing the necessary string have already been done. There is no point in describing
them here as this would take too much article space for describing similar actions. All the necessary data can be found in the files attached
below.

When attempting to compile updated library files in MQL4, we get several unknown constant errors. Let's fix this.


Open \\MQL5\\Include\\DoEasy\

**ToMQL4.mqh** and **add the definitions of unknown MQL5 error codes in it:**

```
//+------------------------------------------------------------------+
//|                                                       ToMQL4.mqh |
//|              Copyright 2017, Artem A. Trishkin, Skype artmedia70 |
//|                         https://www.mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Artem A. Trishkin, Skype artmedia70"
#property link      "https://www.mql5.com/en/users/artmedia70"
#property strict
#ifdef __MQL4__
//+------------------------------------------------------------------+
//| Error codes                                                      |
//+------------------------------------------------------------------+
#define ERR_SUCCESS                       (ERR_NO_ERROR)
#define ERR_MARKET_UNKNOWN_SYMBOL         (ERR_UNKNOWN_SYMBOL)
#define ERR_ZEROSIZE_ARRAY                (ERR_ARRAY_INVALID)
#define ERR_MAIL_SEND_FAILED              (ERR_SEND_MAIL_ERROR)
#define ERR_NOTIFICATION_SEND_FAILED      (ERR_NOTIFICATION_ERROR)
#define ERR_FTP_SEND_FAILED               (ERR_FTP_ERROR)
//+------------------------------------------------------------------+
```

Now everything should compile in MQL4.

### Testing the display of messages

It took me a while to decide on what to display for visual testing. Finally, I decided that all has already been done. Messages already appear
in the journal. Therefore, in order to verify them in a new version of storing and displaying data, we should simply launch the EA from the
previous article: \\MQL5\\Experts\\TestDoEasy\\Part18\\TestDoEasyPart18.mq5.

Let's save it in the new directory under the name \\MQL5\\Experts\\TestDoEasy\ **Part19**\ **TestDoEasyPart19.mq5**.

Nevertheless, I have made some minor changes in it. They are related to the size of tracked changes and the look of displayed texts,
therefore there is no point in describing them here.

Compile and launch the EA in the tester in visual mode. The EA tracks the values of a spread, a profit and funds. So, let's have a look at the messages
related to a spread change and its crossing of the value of 2 points. Upon an increase of funds, positions are closed. The appropriate entries
about all the actions, including opening/closing positions, should be displayed in the journal. We are already familiar with them from the
previous articles' tests. Here, we are mostly interested in their correct display since they are now stored and displayed in a completely
different way. All messages are stored in one place and displayed upon specifying their location in the message database.

Let's have a look:

![](https://c.mql5.com/2/37/IpWQfEEDVL.gif)

We can see that all event entries are displayed correctly, which means that everything works as intended.

Now we have the message class with the common database of all library texts. Besides, we have all the descriptions to all trade server return
codes and runtime errors. The access to all messages is performed by an error code or a message ID.

### What's next?

Just a few finishing touches remain before we can finally proceed to trading classes...

In the next article, we will have a look at storing data in the form of a source code and automatic creation of audio and image files from this
data.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7176#node00)

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

[Part 9. Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure events](https://www.mql5.com/en/articles/6921)

[Part \\
12\. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object \\
events](https://www.mql5.com/en/articles/6995)

[Part 14. Symbol object](https://www.mql5.com/en/articles/7014)

[Part \\
15\. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part 16. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part \\
18\. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7176](https://www.mql5.com/ru/articles/7176)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7176.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7176/mql5.zip "Download MQL5.zip")(237.31 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7176/mql4.zip "Download MQL4.zip")(237.33 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/327300)**
(1)


![chleowhite.25](https://c.mql5.com/avatar/avatar_na2.png)

**[chleowhite.25](https://www.mql5.com/en/users/chleowhite.25)**
\|
4 Dec 2019 at 00:10

We are always amazed by the technology and new methods that exist. Wonderful


![Building an Expert Advisor using separate modules](https://c.mql5.com/2/37/mql5_avatar_adviser_modules.png)[Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)

When developing indicators, Expert Advisors and scripts, developers often need to create various pieces of code, which are not directly related to the trading strategy. In this article, we consider a way to create Expert Advisors using earlier created blocks, such as trailing, filtering and scheduling code, among others. We will see the benefits of this programming approach.

![Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://c.mql5.com/2/37/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XVIII): Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

The article arranges the work of an account object on a new base object of all library objects, improves the CBaseObj base object and tests setting tracked parameters, as well as receiving events for any library objects.

![Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://c.mql5.com/2/37/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://www.mql5.com/en/articles/7195)

The article deals with storing data in the program's source code and creating audio and graphical files out of them. When developing an application, we often need audio and images. The MQL language features several methods of using such data.

![Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://c.mql5.com/2/36/MQL5-avatar-doeasy__12.png)[Library for easy and quick development of MetaTrader programs (part XVII): Interactivity of library objects](https://www.mql5.com/en/articles/7124)

In this article, we are going to finish the development of the base object of all library objects, so that any library object based on it is able to interact with a user. For example, users will be able to set the maximum acceptable size of a spread for opening a position and a price level, upon reaching which an event from a symbol object is sent to the program with the spread or price level-based signal.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/7176&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070455510299252262)

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