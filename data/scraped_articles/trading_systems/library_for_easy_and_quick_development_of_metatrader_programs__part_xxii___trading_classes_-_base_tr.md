---
title: Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations
url: https://www.mql5.com/en/articles/7258
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:38:43.988919
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=balafwtkyktmvmwvgnorusfrpzqcyxub&ssn=1769186319598325657&ssn_dr=0&ssn_sr=0&fv_date=1769186319&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7258&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXII)%3A%20Trading%20classes%20-%20Base%20trading%20class%2C%20verification%20of%20limitations%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918631988754661&fz_uniq=5070450730000651791&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7258#node01)
- [Extending the base trading object functionality](https://www.mql5.com/en/articles/7258#node02)
- [Trading class](https://www.mql5.com/en/articles/7258#node03)
- [Testing](https://www.mql5.com/en/articles/7258#node04)
- [What's next?](https://www.mql5.com/en/articles/7258#node05)


In [the previous article](https://www.mql5.com/en/articles/7229), we opened an extensive library section devoted to
trading functions and created the symbol base trading object. The trading object receives all the properties of a trading request sent to
the server in its parameters, fills in the trading request structure according to the type of a called class method (opening a
position/placing a pending order/modification/closing/removal) and sends a trading order to the server. Valid trading request
property values are passed to the base trading object for sending the trading order. However, in order to use the trading object to the
fullest extent, we first need to check the existing limitations for conducting trading operations in the terminal, program, account and
symbol. After passing these initial checks, we are able to verify trading request properties.

In this article, we will start the development of a full-fledged trading class. The first thing we are going to implement is the verification
of the trading operation limitations.

### Concept

We already have the base trading object, which is a part of the symbol object. It fills in the trading request structure according to the
parameters passed to one of the class methods and sends a trading order to the server. Before implementing a full-fledged trading class, we
will add some base trading object functionality — namely, the ability to voice the results of sending trading orders. This will allow us to
set any sound for voicing trading events. We can set custom sounds for each trading event in each symbol. Of course, we can specify a set of
sounds for each of the events common to all symbols. The base trading object is to provide ample opportunities to set sounds for trading
events regardless of whether the sounds should be the same for all symbols and events or whether they should be different for each individual
symbol and event.

Next, we will create a trading class all trading operations are to be conducted from. In this article, we are going to implement the minimum
functionality for the class — verifying trading operation permissions and calling the necessary methods of base trading objects of
required symbols.

### Extending the base trading object functionality

To assign sounds to trading events, we need macro substitutions and enumerations. Open the **Defines.mqh** file and **add**

**macro substitutions replacing the names of standard audio files**:

```
//+------------------------------------------------------------------+
//| Macro substitutions                                              |
//+------------------------------------------------------------------+
//--- Describe the function with the error line number
#define DFUN_ERR_LINE                  (__FUNCTION__+(TerminalInfoString(TERMINAL_LANGUAGE)=="Russian" ? ", Page " : ", Line ")+(string)__LINE__+": ")
#define DFUN                           (__FUNCTION__+": ")        // "Function description"
#define COUNTRY_LANG                   ("Russian")                // Country language
#define END_TIME                       (D'31.12.3000 23:59:59')   // End date for account history data requests
#define TIMER_FREQUENCY                (16)                       // Minimal frequency of the library timer in milliseconds
//--- Standard sounds
#define SND_ALERT                      "alert.wav"
#define SND_ALERT2                     "alert2.wav"
#define SND_CONNECT                    "connect.wav"
#define SND_DISCONNECT                 "disconnect.wav"
#define SND_EMAIL                      "email.wav"
#define SND_EXPERT                     "expert.wav"
#define SND_NEWS                       "news.wav"
#define SND_OK                         "ok.wav"
#define SND_REQUEST                    "request.wav"
#define SND_STOPS                      "stops.wav"
#define SND_TICK                       "tick.wav"
#define SND_TIMEOUT                    "timeout.wav"
#define SND_WAIT                       "wait.wav"
//--- Parameters of the orders and deals collection timer
```

This makes it more convenient to specify names for setting standard audio files if we want to use them as audio files for voicing trading events.

After the listing within the data block for working with trading classes, **add the enumeration**
**of trading operation types and the enumeration of sound setting modes:**

```
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Data for working with trading classes                            |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|  Logging level                                                   |
//+------------------------------------------------------------------+
enum ENUM_LOG_LEVEL
  {
   LOG_LEVEL_NO_MSG,                                        // Trading logging disabled
   LOG_LEVEL_ERROR_MSG,                                     // Only trading errors
   LOG_LEVEL_ALL_MSG                                        // Full logging
  };
//+------------------------------------------------------------------+
//| Types of performed operations                                    |
//+------------------------------------------------------------------+
enum ENUM_ACTION_TYPE
  {
   ACTION_TYPE_BUY               =  ORDER_TYPE_BUY,               // Open Buy
   ACTION_TYPE_SELL              =  ORDER_TYPE_SELL,              // Open Sell
   ACTION_TYPE_BUY_LIMIT         =  ORDER_TYPE_BUY_LIMIT,         // Place BuyLimit
   ACTION_TYPE_SELL_LIMIT        =  ORDER_TYPE_SELL_LIMIT,        // Place SellLimit
   ACTION_TYPE_BUY_STOP          =  ORDER_TYPE_BUY_STOP,          // Place BuyStop
   ACTION_TYPE_SELL_STOP         =  ORDER_TYPE_SELL_STOP,         // Place SellStop
   ACTION_TYPE_BUY_STOP_LIMIT    =  ORDER_TYPE_BUY_STOP_LIMIT,    // Place BuyStopLimit
   ACTION_TYPE_SELL_STOP_LIMIT   =  ORDER_TYPE_SELL_STOP_LIMIT,   // Place SellStopLimit
   ACTION_TYPE_CLOSE_BY          =  ORDER_TYPE_CLOSE_BY,          // Close a position by an opposite one
   ACTION_TYPE_MODIFY            =  ACTION_TYPE_CLOSE_BY+1,       // Modification
  };
//+------------------------------------------------------------------+
//| Sound setting mode                                               |
//+------------------------------------------------------------------+
enum ENUM_MODE_SET_SOUND
  {
   MODE_SET_SOUND_OPEN,                                     // Opening/placing sound setting mode
   MODE_SET_SOUND_CLOSE,                                    // Closing/removal sound setting mode
   MODE_SET_SOUND_MODIFY_SL,                                // StopLoss modification sound setting mode
   MODE_SET_SOUND_MODIFY_TP,                                // TakeProfit modification sound setting mode
   MODE_SET_SOUND_MODIFY_PRICE,                             // Placing price modification sound setting mode
   MODE_SET_SOUND_ERROR_OPEN,                               // Opening/placing error sound setting mode
   MODE_SET_SOUND_ERROR_CLOSE,                              // Closing/removal error sound setting mode
   MODE_SET_SOUND_ERROR_MODIFY_SL,                          // StopLoss modification error sound setting mode
   MODE_SET_SOUND_ERROR_MODIFY_TP,                          // TakeProfit modification error sound setting mode
   MODE_SET_SOUND_ERROR_MODIFY_PRICE,                       // Placing price modification error sound setting mode
  };
//+------------------------------------------------------------------+
```

The terminal version 2155 introduced new symbol and account properties in MQL5:

1. MQL5: The following values have been added to the [ENUM\_SYMBOL\_INFO\_STRING](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_string)
    enumeration:





   - SYMBOL\_CATEGORY — symbol category. It is used for additional marking of financial instruments. For example, this can be the market sector to
      which the symbol belongs: Agriculture, Oil & Gas and others.
   - SYMBOL\_EXCHANGE — the name of the exchange in which the symbol is traded.

2. MQL5: Added support for position closure by FIFO rule.





   - ACCOUNT\_FIFO\_CLOSE value has been added to [ENUM\_ACCOUNT\_INFO\_INTEGER](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer).
      It indicates that positions can be closed only by the FIFO rule. If the property value is true, then positions for each instrument
      can only be closed in the same order in which they were opened: the oldest one should be closed first, then the next one, etc. In case of
      an attempt to close positions in a different order, an error will be returned. The property value is always 'false' for accounts
      without hedging position management (ACCOUNT\_MARGIN\_MODE!=ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING).
   - New return code: [MT\_RET\_REQUEST\_CLOSE\_ONLY](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes) —
      the request is rejected, because the rule "Only closing of existing positions by FIFO rule is allowed" is set for the symbol




There are three main methods to close a position:

   - Closing from the client terminal: the trader closes the position manually, using a trading robot, based on the Signals service
      subscription, etc. In case of an attempt to close a position, which does not meet the FIFO rule, the trader will receive an
      appropriate error.
   - Closing upon Stop Loss or Take Profit activation: these orders are processed on the server side, so the position closure is not
      requested on the trader (terminal) side, but is initiated by the server. If Stop Loss or Take Profit triggers for a position,
      and this position does not comply with the FIFO rule (there is an older position for the same symbol), the position will not be
      closed.
   - Closing upon Stop Out triggering: such operations are also processed on the server side. In a normal mode, in which FIFO-based
      closing is disabled, in case of Stop Out positions are closed starting with the one having the largest loss. If this option is
      enabled, the open time will be additionally checked for losing positions. The server determines losing positions for each
      symbol, finds the oldest position for each symbol, and then closes the one which has the greatest loss among the found
      positions.

Considering all this, the new properties have been added to the symbol and account objects.

**The new property has been added to the block of account integer**
**properties, while the number of integer properties has been increased to**

**11:**

```
//+------------------------------------------------------------------+
//| Account integer properties                                       |
//+------------------------------------------------------------------+
enum ENUM_ACCOUNT_PROP_INTEGER
  {
   ACCOUNT_PROP_LOGIN,                                      // Account number
   ACCOUNT_PROP_TRADE_MODE,                                 // Trading account type
   ACCOUNT_PROP_LEVERAGE,                                   // Leverage
   ACCOUNT_PROP_LIMIT_ORDERS,                               // Maximum allowed number of active pending orders
   ACCOUNT_PROP_MARGIN_SO_MODE,                             // Mode of setting the minimum available margin level
   ACCOUNT_PROP_TRADE_ALLOWED,                              // Permission to trade for the current account from the server side
   ACCOUNT_PROP_TRADE_EXPERT,                               // Permission to trade for an EA from the server side
   ACCOUNT_PROP_MARGIN_MODE,                                // Margin calculation mode
   ACCOUNT_PROP_CURRENCY_DIGITS,                            // Number of digits for an account currency necessary for accurate display of trading results
   ACCOUNT_PROP_SERVER_TYPE,                                // Trade server type (MetaTrader5, MetaTrader4)
   ACCOUNT_PROP_FIFO_CLOSE                                  // Flag of a position closure by FIFO rule only
  };
#define ACCOUNT_PROP_INTEGER_TOTAL    (11)                  // Total number of integer properties
#define ACCOUNT_PROP_INTEGER_SKIP     (0)                   // Number of integer account properties not used in sorting
//+------------------------------------------------------------------+
```

**The new sorting criterion by integer properties has been**
**added to the block of possible account sorting criteria:**

```
//+------------------------------------------------------------------+
//| Possible account sorting criteria                                |
//+------------------------------------------------------------------+
#define FIRST_ACC_DBL_PROP            (ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_INTEGER_SKIP)
#define FIRST_ACC_STR_PROP            (ACCOUNT_PROP_INTEGER_TOTAL-ACCOUNT_PROP_INTEGER_SKIP+ACCOUNT_PROP_DOUBLE_TOTAL-ACCOUNT_PROP_DOUBLE_SKIP)
enum ENUM_SORT_ACCOUNT_MODE
  {
//--- Sort by integer properties
   SORT_BY_ACCOUNT_LOGIN = 0,                               // Sort by account number
   SORT_BY_ACCOUNT_TRADE_MODE,                              // Sort by trading account type
   SORT_BY_ACCOUNT_LEVERAGE,                                // Sort by leverage
   SORT_BY_ACCOUNT_LIMIT_ORDERS,                            // Sort by maximum acceptable number of existing pending orders
   SORT_BY_ACCOUNT_MARGIN_SO_MODE,                          // Sort by mode for setting the minimum acceptable margin level
   SORT_BY_ACCOUNT_TRADE_ALLOWED,                           // Sort by permission to trade for the current account
   SORT_BY_ACCOUNT_TRADE_EXPERT,                            // Sort by permission to trade for an EA
   SORT_BY_ACCOUNT_MARGIN_MODE,                             // Sort by margin calculation mode
   SORT_BY_ACCOUNT_CURRENCY_DIGITS,                         // Sort by number of digits for an account currency
   SORT_BY_ACCOUNT_SERVER_TYPE,                             // Sort by trade server type (MetaTrader5, MetaTrader4)
   SORT_BY_ACCOUNT_FIFO_CLOSE,                              // Sort by the flag of a position closure by FIFO rule only
//--- Sort by real properties
```

The symbol object properties are added in a similar way.

**The block of string properties features two new properties,**
**while the number of string properties**

**has been increased to 13:**

```
//+------------------------------------------------------------------+
//| Symbol string properties                                         |
//+------------------------------------------------------------------+
enum ENUM_SYMBOL_PROP_STRING
  {
   SYMBOL_PROP_NAME = (SYMBOL_PROP_INTEGER_TOTAL+SYMBOL_PROP_DOUBLE_TOTAL),   // Symbol name
   SYMBOL_PROP_BASIS,                                       // Name of the underlaying asset for a derivative symbol
   SYMBOL_PROP_CURRENCY_BASE,                               // Instrument base currency
   SYMBOL_PROP_CURRENCY_PROFIT,                             // Profit currency
   SYMBOL_PROP_CURRENCY_MARGIN,                             // Margin currency
   SYMBOL_PROP_BANK,                                        // Source of the current quote
   SYMBOL_PROP_DESCRIPTION,                                 // String description of a symbol
   SYMBOL_PROP_FORMULA,                                     // The formula used for custom symbol pricing
   SYMBOL_PROP_ISIN,                                        // The name of a trading symbol in the international system of securities identification numbers (ISIN)
   SYMBOL_PROP_PAGE,                                        // The web page containing symbol information
   SYMBOL_PROP_PATH,                                        // Location in the symbol tree
   SYMBOL_PROP_CATEGORY,                                    // Symbol category
   SYMBOL_PROP_EXCHANGE                                     // Name of an exchange a symbol is traded on
  };
#define SYMBOL_PROP_STRING_TOTAL     (13)                   // Total number of string properties
//+------------------------------------------------------------------+
```

**The two new criteria for sorting by symbol string properties has been added to the list of possible symbol sorting criteria:**

```
//--- Sort by string properties
   SORT_BY_SYMBOL_NAME = FIRST_SYM_STR_PROP,                // Sort by a symbol name
   SORT_BY_SYMBOL_BASIS,                                    // Sort by an underlying asset of a derivative
   SORT_BY_SYMBOL_CURRENCY_BASE,                            // Sort by a base currency of a symbol
   SORT_BY_SYMBOL_CURRENCY_PROFIT,                          // Sort by a profit currency
   SORT_BY_SYMBOL_CURRENCY_MARGIN,                          // Sort by a margin currency
   SORT_BY_SYMBOL_BANK,                                     // Sort by a feeder of the current quote
   SORT_BY_SYMBOL_DESCRIPTION,                              // Sort by a symbol string description
   SORT_BY_SYMBOL_FORMULA,                                  // Sort by the formula used for custom symbol pricing
   SORT_BY_SYMBOL_ISIN,                                     // Sort by the name of a symbol in the ISIN system
   SORT_BY_SYMBOL_PAGE,                                     // Sort by an address of the web page containing symbol information
   SORT_BY_SYMBOL_PATH,                                     // Sort by a path in the symbol tree
   SORT_BY_SYMBOL_CATEGORY,                                 // Sort by symbol category
   SORT_BY_SYMBOL_EXCHANGE                                  // Sort by a name of an exchange a symbol is traded on
  };
//+------------------------------------------------------------------+
```

Also, some new constants of the library text message indices have been added to work with new properties of objects, classes and methods.

**Let's**
**consider**

**the new constants in the Datas.mqh file:**

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
   MSG_LIB_PROP_NOT_SUPPORTED_MT5_LESS_2155,          // Property not supported in MetaTrader 5 versions lower than 2155
   MSG_LIB_PROP_NOT_SUPPORTED_POSITION,               // Property not supported for position
   MSG_LIB_PROP_NOT_SUPPORTED_PENDING,                // Property not supported for pending order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET,                 // Property not supported for market order
   MSG_LIB_PROP_NOT_SUPPORTED_MARKET_HIST,            // Property not supported for historical market order
   MSG_LIB_PROP_NOT_SET,                              // Value not set
   MSG_LIB_PROP_EMPTY,                                // Not set

   MSG_LIB_SYS_ERROR,                                 // Error
   MSG_LIB_SYS_NOT_SYMBOL_ON_SERVER,                  // Error. No such symbol on server
   MSG_LIB_SYS_NOT_SYMBOL_ON_LIST,                    // Error. No such symbol in the list of used symbols:
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
   MSG_LIB_SYS_FILE_RES_ALREADY_IN_LIST,              // This file already created and added to list:
   MSG_LIB_SYS_FAILED_CREATE_RES_LINK,                // Error. Failed to create object pointing to resource file
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
   MSG_LIB_SYS_FAILED_CREATE_TIMER,                   // Failed to create timer. Error:

   MSG_LIB_SYS_NO_TICKS_YET,                          // No ticks yet
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT,              // Could not create object structure
   MSG_LIB_SYS_FAILED_WRITE_UARRAY_TO_FILE,           // Could not write uchar array to file
   MSG_LIB_SYS_FAILED_LOAD_UARRAY_FROM_FILE,          // Could not load uchar array from file
   MSG_LIB_SYS_FAILED_CREATE_OBJ_STRUCT_FROM_UARRAY,  // Could not create object structure from uchar array
   MSG_LIB_SYS_FAILED_SAVE_OBJ_STRUCT_TO_UARRAY,      // Failed to save object structure to uchar array, error
   MSG_LIB_SYS_ERROR_INDEX,                           // Error. "index" value should be within 0 - 3
   MSG_LIB_SYS_ERROR_FAILED_CONV_TO_LOWERCASE,        // Failed to convert string to lowercase, error

//--- COrder
   MSG_ORD_BUY,                                       // Buy
   MSG_ORD_SELL,                                      // Sell
   MSG_ORD_TO_BUY,                                    // Buy order
   MSG_ORD_TO_SELL,                                   // Sell order
   MSG_DEAL_TO_BUY,                                   // Buy deal
   MSG_DEAL_TO_SELL,                                  // Sell deal
   MSG_ORD_MARKET,                                    // Market order
   MSG_ORD_HISTORY,                                   // Historical order
   MSG_ORD_DEAL,                                      // Deal
   MSG_ORD_POSITION,                                  // Position
   MSG_ORD_PENDING_ACTIVE,                            // Active pending order
   MSG_ORD_PENDING,                                   // Pending order
   MSG_ORD_UNKNOWN_TYPE,                              // Unknown order type
   MSG_POS_UNKNOWN_TYPE,                              // Unknown position type
   MSG_POS_UNKNOWN_DEAL,                              // Unknown deal type
   //---

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
   MSG_SYM_PROP_CAYEGORY,                             // Symbol category
   MSG_SYM_PROP_EXCHANGE,                             // Name of an exchange a symbol is traded on
   //---

   //---
   MSG_SYM_TRADE_MODE_DISABLED,                       // Trade disabled for symbol
   MSG_SYM_TRADE_MODE_LONGONLY,                       // Only long positions allowed
   MSG_SYM_TRADE_MODE_SHORTONLY,                      // Only short positions allowed
   MSG_SYM_TRADE_MODE_CLOSEONLY,                      // Enable close only
   MSG_SYM_TRADE_MODE_FULL,                           // No trading limitations

   MSG_SYM_MARKET_ORDER_DISABLED,                     // Market orders disabled
   MSG_SYM_LIMIT_ORDER_DISABLED,                      // Limit orders disabled
   MSG_SYM_STOP_ORDER_DISABLED,                       // Stop orders disabled
   MSG_SYM_STOP_LIMIT_ORDER_DISABLED,                 // StopLimit orders disabled
   MSG_SYM_SL_ORDER_DISABLED,                         // StopLoss orders disabled
   MSG_SYM_TP_ORDER_DISABLED,                         // TakeProfit orders disabled
   MSG_SYM_CLOSE_BY_ORDER_DISABLED,                   // CloseBy orders disabled
   //---

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
   MSG_ACC_PROP_FIFO_CLOSE,                           // Flag of a position closure by FIFO rule only
   //---

//--- CTrading
   MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED,           // Trade operations are not allowed in the terminal (the AutoTrading button is disabled)
   MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED,                 // EA is not allowed to trade (F7 --> Common --> Allow Automated Trading)
   MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED,            // Trading is disabled for the current account
   MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED,         // Trading on the trading server side is disabled for EAs on the current account
   MSG_LIB_TEXT_TERMINAL_NOT_CONNECTED,               // No connection to the trade server
   MSG_LIB_TEXT_REQUEST_REJECTED_DUE,                 // Request was rejected before sending to the server due to:
   MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR,                 // Insufficient funds for opening a position:
   MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED,            // Exceeded maximum allowed aggregate volume of orders and positions in one direction
   MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME,              // Request volume is less than the minimum acceptable one
   MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME,              // Request volume exceeds the maximum acceptable one
   MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED,             // Close by is disabled
   MSG_LIB_TEXT_INVALID_VOLUME_STEP,                  // Request volume is not a multiple of the minimum lot change step gradation
   MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL,             // Symbols of opposite positions are not equal

  };
```

**Also, let's add the texts corresponding to the declared**
**constants to the text message arrays:**

```
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
   {"Начало списка параметров","Beginning of event parameter list"},
   {"Конец списка параметров","End of parameter list"},
   {"Свойство не поддерживается","Property not supported"},
   {"Свойство не поддерживается в MQL4","Property not supported in MQL4"},
   {"Свойство не поддерживается в MetaTrader5 версии ниже 2155","Property not supported in MetaTrader 5, build lower than 2155"},
   {"Свойство не поддерживается у позиции","Property not supported for position"},
   {"Свойство не поддерживается у отложенного ордера","Property not supported for pending order"},
   {"Свойство не поддерживается у маркет-ордера","Property not supported for market order"},
   {"Свойство не поддерживается у исторического маркет-ордера","Property not supported for historical market order"},
   {"Значение не задано","Value not set"},
   {"Отсутствует","Not set"},

   {"Ошибка ","Error "},
   {"Ошибка. Такого символа нет на сервере","Error. No such symbol on server"},
   {"Ошибка. Такого символа нет в списке используемых символов: ","Error. This symbol is not in the list of symbols used: "},
   {"Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in market watch. Error: "},
   {"Не удалось получить текущие цены. Ошибка: ","Could not get current prices. Error: "},
   {"Не удалось получить коэффициенты взимания маржи. Ошибка: ","Failed to get margin rates. Error: "},
   {"Не удалось получить данные ","Failed to get data of "},

   {"Не удалось создать папку хранения файлов. Ошибка: ","Could not create file storage folder. Error: "},
   {"Ошибка. Не удалось добавить текущий объект-аккаунт в список-коллекцию","Error. Failed to add current account object to collection list"},
   {"Ошибка. Не удалось создать объект-аккаунт с данными текущего счёта","Error. Failed to create account object with current account data"},
   {"Не удалось открыть для записи файл ","Could not open file for writing: "},
   {"Ошибка входных данных: нет символа ","Input error: no "},
   {"Не удалось создать объект-символ ","Failed to create symbol object "},
   {"Не удалось добавить символ ","Failed to add "},

   {"Не удалось получить текущие цены по символу события ","Failed to get current prices by event symbol "},
   {"Такое событие уже есть в списке","This event already in the list"},
   {"Такой файл уже создан и добавлен в список: ","This file has already been created and added to list: "},
   {"Ошибка. Не удалось создать объект-указатель на файл ресурса","Error. Failed to create resource file link object"},

   {"Ошибка. Уже создан счётчик с идентификатором ","Error. Already created counter with id "},
   {"Не удалось создать счётчик таймера ","Failed to create timer counter "},

   {"Ошибка создания временного списка","Error creating temporary list"},
   {"Ошибка. Список не является списком рыночной коллекции","Error. The list is not a list of market collection"},
   {"Ошибка. Список не является списком исторической коллекции","Error. The list is not a list of history collection"},
   {"Не удалось добавить ордер в список","Could not add order to list"},
   {"Не удалось добавить сделку в список","Could not add deal to list"},
   {"Не удалось добавить контрольный ордер ","Failed to add control order "},
   {"Не удалось добавить контрольую позицию ","Failed to add control position "},
   {"Не удалось добавить модифицированный ордер в список изменённых ордеров","Could not add modified order to list of modified orders"},
   {"Не удалось создать таймер. Ошибка: ","Could not create timer. Error: "},


//--- COrder
   {"Buy","Buy"},
   {"Sell","Sell"},
   {"Ордер на покупку","Buy order"},
   {"Ордер на продажу","Sell order"},

   {"Сделка на покупку","Buy deal"},
   {"Сделка на продажу","Sell deal"},

   {"Маркет-ордер","Market order"},
   {"Исторический ордер","History order"},
   {"Сделка","Deal"},
   {"Позиция","Active position"},
   {"Установленный отложенный ордер","Active pending order"},
   {"Отложенный ордер","Pending order"},
   {"Неизвестный тип ордера","Unknown order type"},
   {"Неизвестный тип позиции","Unknown position type"},
   {"Неизвестный тип сделки","Unknown deal type"},
   //---

   {"Путь в дереве символов","Path in symbol tree"},
   {"Название категории или сектора, к которой принадлежит торговый символ","Name of sector or category trading symbol belongs to"},
   {"Название биржи или площадки, на которой торгуется символ","Name of exchange financial symbol traded in"},
   //---
   {"Форекс символ","Forex symbol"},

   {"Разрешены только операции закрытия позиций","Close only"},
   {"Нет ограничений на торговые операции","No trade restrictions"},
   {"Торговля рыночными ордерами запрещена","Trading through market orders prohibited"},
   {"Установка Limit-ордеров запрещена","Limit orders prohibited"},
   {"Установка Stop-ордеров запрещена","Stop orders prohibited"},
   {"Установка StopLimit-ордеров запрещена","StopLimit orders prohibited"},
   {"Установка StopLoss-ордеров запрещена","StopLoss orders prohibited"},
   {"Установка TakeProfit-ордеров запрещена","TakeProfit orders prohibited"},
   {"Установка CloseBy-ордеров запрещена","CloseBy orders prohibited"},
   //---
   {"Торговля по запросу","Execution by request"},
   {"Торговля по потоковым ценам","Instant execution"},
   {"Исполнение ордеров по рынку","Market execution"},
   {"Биржевое исполнение","Exchange execution"},
   //---

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
   {"Признак закрытия позиций только по правилу FIFO","Sign of closing positions only according to FIFO rule"},
   //---
   {"Баланс счета","Account balance"},

//--- CEngine
   {"С момента последнего запуска ЕА торговых событий не было","No trade events since the last launch of EA"},
   {"Не удалось получить описание последнего торгового события","Failed to get description of the last trading event"},
   {"Не удалось получить список открытых позиций","Failed to get open positions list"},
   {"Не удалось получить список установленных ордеров","Failed to get pending orders list"},
   {"Нет открытых позиций","No open positions"},
   {"Нет установленных ордеров","No placed orders"},
   {"В терминале нет разрешения на проведение торговых операций (отключена кнопка \"Авто-торговля\")","No permission to conduct trading operations in terminal (\"AutoTrading\" button disabled)"},
   {"Для советника нет разрешения на проведение торговых операций (F7 --> Общие --> \"Разрешить автоматическую торговлю\")","EA does not have permission to conduct trading operations (F7 --> Common --> \"Allow Automatic Trading\")"},
   {"Для текущего счёта запрещена торговля","Trading prohibited for the current account"},
   {"Для советников на текущем счёте запрещена торговля на стороне торгового сервера","From the side of trade server, trading for EA on the current account prohibited"},
   {"Нет связи с торговым сервером","No connection to trade server"},
   {"Запрос отклонён до отправки на сервер по причине:","Request rejected before being sent to server due to:"},
   {"Не хватает средств на открытие позиции: ","Not enough money to open position: "},
   {"Превышен максимальный совокупный объём ордеров и позиций в одном направлении","Exceeded maximum total volume of orders and positions in one direction"},
   {"Объём в запросе меньше минимально-допустимого","Volume in request less than minimum allowable"},
   {"Объём в запросе больше максимально-допустимого","Volume in request greater than maximum allowable"},
   {"Закрытие встречным запрещено","CloseBy orders prohibited"},
   {"Объём в запросе не кратен минимальной градации шага изменения лота","Volume in request not a multiple of minimum gradation of step for changing lot"},
   {"Символы встречных позиций не равны","Symbols of two opposite positions not equal"},

  };

//+---------------------------------------------------------------------+
//| Array of messages for trade server return codes (10004 - 10045)     |
//| (1) in user's country language                                      |
//| (2) in the international language                                   |
//+---------------------------------------------------------------------+
string messages_ts_ret_code[][TOTAL_LANG]=
  {
   {"Реквота","Requote"},                                                                                                                          // 10004
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},                                                   // 10005
   {"Запрос отклонен","Request rejected"},                                                                                                         // 10006
   {"Запрос отменен трейдером","Request canceled by trader"},                                                                                      // 10007
   {"Ордер размещен","Order placed"},                                                                                                              // 10008
   {"Заявка выполнена","Request completed"},                                                                                                       // 10009
   {"Заявка выполнена частично","Only part of request completed"},                                                                         // 10010
   {"Ошибка обработки запроса","Request processing error"},                                                                                        // 10011
   {"Запрос отменен по истечению времени","Request canceled by timeout"},                                                                          // 10012
   {"Неправильный запрос","Invalid request"},                                                                                                      // 10013
   {"Неправильный объем в запросе","Invalid volume in request"},                                                                                   // 10014
   {"Неправильная цена в запросе","Invalid price in request"},                                                                                     // 10015
   {"Неправильные стопы в запросе","Invalid stops in request"},                                                                                    // 10016
   {"Торговля запрещена","Trading disabled"},                                                                                                      // 10017
   {"Рынок закрыт","Market closed"},                                                                                                               // 10018
   {"Нет достаточных денежных средств для выполнения запроса","Not enough money to complete request"},                                     // 10019
   {"Цены изменились","Prices changed"},                                                                                                           // 10020
   {"Отсутствуют котировки для обработки запроса","No quotes to process request"},                                                         // 10021
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
   {"Достигнут лимит на количество отложенных ордеров","Number of pending orders reached limit"},                                           // 10033
   {"Достигнут лимит на объем ордеров и позиций для данного символа","Volume of orders and positions for symbol reached limit"},            // 10034
   {"Неверный или запрещённый тип ордера","Incorrect or prohibited order type"},                                                                   // 10035
   {"Позиция с указанным идентификатором уже закрыта","Position with specified identifier already closed"},                                 // 10036
   {"Неизвестный код возврата торгового сервера","Unknown trading server return code"},                                                     // 10037
   {"Закрываемый объем превышает текущий объем позиции","Close volume exceeds the current position volume"},                                       // 10038
   {"Для указанной позиции уже есть ордер на закрытие","Close order already exists for specified position"},                                       // 10039
   {"Достигнут лимит на количество открытых позиций","Number of positions reached limit"},                                                  // 10040
   {
    "Запрос на активацию отложенного ордера отклонен, а сам ордер отменен",                                                                        // 10041
    "Pending order activation request rejected, order canceled"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешены только длинные позиции\"",                                                // 10042
    "Request rejected, because \"Only long positions are allowed\" rule set for symbol"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешены только короткие позиции\"",                                               // 10043
    "Request rejected, because \"Only short positions are allowed\" rule set for symbol"
   },
   {
    "Запрос отклонен, так как на символе установлено правило \"Разрешено только закрывать существующие позиции\"",                                 // 10044
    "Request rejected because \"Only position closing is allowed\" rule set for symbol"
   },
   {
    "Запрос отклонен, так как для торгового счета установлено правило \"Разрешено закрывать существующие позиции только по правилу FIFO\"",        // 10045
    "Request rejected, because \"Position closing is allowed only by FIFO rule\" flag set for trading account"
   },
  };
//+------------------------------------------------------------------+
```

As we already know, all performed actions are aimed at adding new messages to the library and accessing a message by its index written in the
message constant.

Now let's improve the message class located in \\MQL5\\Include\\DoEasy\\Services\ **Message.mqh**.

**In the public class section, declare the method to stop playing any audio**
**file:**

```
public:
//--- (1) Display a text message in the journal, send a push notification and e-mail,
//--- (2) Display a message by ID, send a push notification and e-mail,
//--- (3) play an audio file
//--- (4) Stop playing any sound
   static bool       Out(const string text,const bool push=false,const bool mail=false,const string subject=NULL);
   static bool       OutByID(const int msg_id,const bool code=true);
   static bool       PlaySound(const string file_name);
   static bool       StopPlaySound(void);
   //--- Return (1) a message, (2) a code in the "(code)" format
```

Write its implementation outside the class body:

```
//+------------------------------------------------------------------+
//| Stop playing any sound                                           |
//+------------------------------------------------------------------+
bool CMessage::StopPlaySound(void)
  {
   bool res=::PlaySound(NULL);
   CMessage::m_global_error=(res ? ERR_SUCCESS : ::GetLastError());
   return res;
  }
//+------------------------------------------------------------------+
```

Here all is simple: when passing the NULL
constant as a file name, the [PlaySound()](https://www.mql5.com/en/docs/common/playsound) standard function
stops playing the currently active audio file.

To be able to play standard audio files included in the terminal delivery, we need to either specify a file name (to play sounds from the files
standard location:

**terminal\_folder\\Sounds**), or a file name with a prefix containing a double slash and a subfolder storing non-standard audio files.


Therefore,

**we should improve the PlaySound() method:**

```
//+------------------------------------------------------------------+
//| Play an audio file                                               |
//+------------------------------------------------------------------+
bool CMessage::PlaySound(const string file_name)
  {
   if(file_name==NULL)
      return true;
   string pref=(file_name==SND_ALERT       || file_name==SND_ALERT2   || file_name==SND_CONNECT  ||
                file_name==SND_DISCONNECT  || file_name==SND_EMAIL    || file_name==SND_EXPERT   ||
                file_name==SND_NEWS        || file_name==SND_OK       || file_name==SND_REQUEST  ||
                file_name==SND_STOPS       || file_name==SND_TICK     || file_name==SND_TIMEOUT  ||
                file_name==SND_WAIT        ?  "" : "\\Files\\");
   bool res=::PlaySound(pref+file_name);
   CMessage::m_global_error=(res ? ERR_SUCCESS : ::GetLastError());
   return res;
  }
//+------------------------------------------------------------------+
```

What do we have here?

If the file name
contains the name of a standard audio file, write an empty string to
the file name prefix.

Otherwise, add
the double slash and name of a subfolder containing library sounds to the prefix.

Then, the file name is composed of the prefix
and a name passed to the method.

Now when
passing the name to the PlaySound() function, the location of a sound file is selected automatically (either the standard sound
folder or the library audio files folder) and the appropriate file is played.

Now we have the new trade server return code. In the method for receiving a message from the text array by ID void CMessage::GetTextByID(), **changethe limits of trade server return codesby**
**adding 1 (10045 instead of 10044):**

```
      //--- Runtime errors (Economic calendar 5400 - 5402)
      msg_id>5399 && msg_id<5403    ?  messages_runtime_calendar[msg_id-5400][m_lang_num]         :
      //--- Trade server return codes (10004 - 10045)
      msg_id>10003 && msg_id<10046  ?  messages_ts_ret_code[msg_id-10004][m_lang_num]             :
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
```

**This concludes the changes in the library message class.**

Sometimes, we need to use an order type to find out the direction of a position when the order is activated. Suppose that we set a pending order of
ORDER\_TYPE\_BUY\_LIMIT type. As soon as it is activated, the ORDER\_TYPE\_BUY market order is placed leading to the DEAL\_ENTRY\_IN deal of
DEAL\_TYPE\_BUY type. The deal, in turn, leads to a position of POSITION\_TYPE\_BUY type.

In order to find out the position direction by a pending order type, we need to receive a market order type (in the current example, it is
ORDER\_TYPE\_BUY).

Previously, we defined it using the ternary operator, while comparing with a pending order type. However, it can be done easier: we can
simply take the excess from dividing the type of a checked order by 2. Type 0 (ORDER\_TYPE\_BUY) is always returned for even orders, while type 1
(ORDER\_TYPE\_SELL) is returned for odd ones. We have implemented that in some functions and methods.

**In the** (\\MQL5\\Include\\DoEasy\\Objects\\Orders\ **Order.mqh**) abstract order class:

```
//+------------------------------------------------------------------+
//| Return the type by direction                                     |
//+------------------------------------------------------------------+
long COrder::OrderTypeByDirection(void) const
  {
   ENUM_ORDER_STATUS status=(ENUM_ORDER_STATUS)this.GetProperty(ORDER_PROP_STATUS);
   if(status==ORDER_STATUS_MARKET_POSITION)
     {
      return (ENUM_ORDER_TYPE)this.OrderType();
     }
   else if(status==ORDER_STATUS_MARKET_PENDING || status==ORDER_STATUS_HISTORY_PENDING)
     {
      return ENUM_ORDER_TYPE(this.OrderType()%2);
     }
   else if(status==ORDER_STATUS_MARKET_ORDER || status==ORDER_STATUS_HISTORY_ORDER)
     {
      return this.OrderType();
     }
   else if(status==ORDER_STATUS_DEAL)
     {
      return
        (
         (ENUM_DEAL_TYPE)this.TypeOrder()==DEAL_TYPE_BUY ? ORDER_TYPE_BUY   :
         (ENUM_DEAL_TYPE)this.TypeOrder()==DEAL_TYPE_SELL ? ORDER_TYPE_SELL : WRONG_VALUE
        );
     }
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

and **in the library of service functions** (\\MQL5\\Include\\DoEasy\\Services\ **DELib.mqh**):

```
//+------------------------------------------------------------------+
//| Return position type by order type                               |
//+------------------------------------------------------------------+
ENUM_POSITION_TYPE PositionTypeByOrderType(ENUM_ORDER_TYPE type_order)
  {
   if(type_order==ORDER_TYPE_CLOSE_BY)
      return WRONG_VALUE;
   return ENUM_POSITION_TYPE(type_order%2);
  }
//+------------------------------------------------------------------+
```

The function returning an order name has also been changed in the same file of service functions.

The
flag indicating the need to display the order description has been added. Depending on the flag, the
returned market order type either receives the description
that this is a market order, or it receives none.

Order type descriptions have been added to the returned types of
pending orders:

```
//+------------------------------------------------------------------+
//| Return the order name                                            |
//+------------------------------------------------------------------+
string OrderTypeDescription(const ENUM_ORDER_TYPE type,bool as_order=true,bool need_prefix=true)
  {
   string pref=
     (
      !need_prefix ? "" :
      #ifdef __MQL5__ CMessage::Text(MSG_ORD_MARKET)
      #else/*__MQL4__*/(as_order ? CMessage::Text(MSG_ORD_MARKET) : CMessage::Text(MSG_ORD_POSITION)) #endif
     );
   return
     (
      type==ORDER_TYPE_BUY_LIMIT       ?  CMessage::Text(MSG_ORD_PENDING)+" Buy Limit"       :
      type==ORDER_TYPE_BUY_STOP        ?  CMessage::Text(MSG_ORD_PENDING)+" Buy Stop"        :
      type==ORDER_TYPE_SELL_LIMIT      ?  CMessage::Text(MSG_ORD_PENDING)+" Sell Limit"      :
      type==ORDER_TYPE_SELL_STOP       ?  CMessage::Text(MSG_ORD_PENDING)+" Sell Stop"       :
   #ifdef __MQL5__
      type==ORDER_TYPE_BUY_STOP_LIMIT  ?  CMessage::Text(MSG_ORD_PENDING)+" Buy Stop Limit"  :
      type==ORDER_TYPE_SELL_STOP_LIMIT ?  CMessage::Text(MSG_ORD_PENDING)+" Sell Stop Limit" :
      type==ORDER_TYPE_CLOSE_BY        ?  CMessage::Text(MSG_ORD_CLOSE_BY)                   :
   #else
      type==ORDER_TYPE_BALANCE         ?  CMessage::Text(MSG_LIB_PROP_BALANCE)               :
      type==ORDER_TYPE_CREDIT          ?  CMessage::Text(MSG_LIB_PROP_CREDIT)                :
   #endif
      type==ORDER_TYPE_BUY             ?  pref+" Buy"                                        :
      type==ORDER_TYPE_SELL            ?  pref+" Sell"                                       :
      CMessage::Text(MSG_ORD_UNKNOWN_TYPE)
     );
  }
//+------------------------------------------------------------------+
```

Due to the introduction of the new symbol and account object properties, we need to add these properties to the objects.

Open \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh** and make the necessary changes.

**In the protected class**
**section, add the definition of methods for receiving the two new properties:**

```
protected:
//--- Protected parametric constructor
                     CSymbol(ENUM_SYMBOL_STATUS symbol_status,const string name,const int index);

//--- Get and return integer properties of a selected symbol from its parameters
   bool              SymbolExists(const string name)     const;
   long              SymbolExists(void)                  const;
   long              SymbolCustom(void)                  const;
   long              SymbolChartMode(void)               const;
   long              SymbolMarginHedgedUseLEG(void)      const;
   long              SymbolOrderFillingMode(void)        const;
   long              SymbolOrderMode(void)               const;
   long              SymbolExpirationMode(void)          const;
   long              SymbolOrderGTCMode(void)            const;
   long              SymbolOptionMode(void)              const;
   long              SymbolOptionRight(void)             const;
   long              SymbolBackgroundColor(void)         const;
   long              SymbolCalcMode(void)                const;
   long              SymbolSwapMode(void)                const;
   long              SymbolDigitsLot(void);
   int               SymbolDigitsBySwap(void);
//--- Get and return real properties of a selected symbol from its parameters
   double            SymbolBidHigh(void)                 const;
   double            SymbolBidLow(void)                  const;
   double            SymbolVolumeReal(void)              const;
   double            SymbolVolumeHighReal(void)          const;
   double            SymbolVolumeLowReal(void)           const;
   double            SymbolOptionStrike(void)            const;
   double            SymbolTradeAccruedInterest(void)    const;
   double            SymbolTradeFaceValue(void)          const;
   double            SymbolTradeLiquidityRate(void)      const;
   double            SymbolMarginHedged(void)            const;
   bool              SymbolMarginLong(void);
   bool              SymbolMarginShort(void);
   bool              SymbolMarginBuyStop(void);
   bool              SymbolMarginBuyLimit(void);
   bool              SymbolMarginBuyStopLimit(void);
   bool              SymbolMarginSellStop(void);
   bool              SymbolMarginSellLimit(void);
   bool              SymbolMarginSellStopLimit(void);
//--- Get and return string properties of a selected symbol from its parameters
   string            SymbolBasis(void)                   const;
   string            SymbolBank(void)                    const;
   string            SymbolISIN(void)                    const;
   string            SymbolFormula(void)                 const;
   string            SymbolPage(void)                    const;
   string            SymbolCategory(void)                const;
   string            SymbolExchange(void)                const;
//--- Search for a symbol and return the flag indicating its presence on the server
   bool              Exist(void)                         const;

public:
```

**In the block for a simplified receiving of object properties of the public class section, add the definition of the two new methods:**

```
public:
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
//+------------------------------------------------------------------+
//--- Integer properties
   long              Status(void)                                 const { return this.GetProperty(SYMBOL_PROP_STATUS);                                      }
   int               IndexInMarketWatch(void)                     const { return (int)this.GetProperty(SYMBOL_PROP_INDEX_MW);                               }
   bool              IsCustom(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_CUSTOM);                                }
   color             ColorBackground(void)                        const { return (color)this.GetProperty(SYMBOL_PROP_BACKGROUND_COLOR);                     }
   ENUM_SYMBOL_CHART_MODE ChartMode(void)                         const { return (ENUM_SYMBOL_CHART_MODE)this.GetProperty(SYMBOL_PROP_CHART_MODE);          }
   bool              IsExist(void)                                const { return (bool)this.GetProperty(SYMBOL_PROP_EXIST);                                 }
   bool              IsExist(const string name)                   const { return this.SymbolExists(name);                                                   }
   bool              IsSelect(void)                               const { return (bool)this.GetProperty(SYMBOL_PROP_SELECT);                                }
   bool              IsVisible(void)                              const { return (bool)this.GetProperty(SYMBOL_PROP_VISIBLE);                               }
   long              SessionDeals(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_DEALS);                               }
   long              SessionBuyOrders(void)                       const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS);                          }
   long              SessionSellOrders(void)                      const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS);                         }
   long              Volume(void)                                 const { return this.GetProperty(SYMBOL_PROP_VOLUME);                                      }
   long              VolumeHigh(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH);                                  }
   long              VolumeLow(void)                              const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW);                                   }
   datetime          Time(void)                                   const { return (datetime)this.GetProperty(SYMBOL_PROP_TIME);                              }
   int               Digits(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS);                                 }
   int               DigitsLot(void)                              const { return (int)this.GetProperty(SYMBOL_PROP_DIGITS_LOTS);                            }
   int               Spread(void)                                 const { return (int)this.GetProperty(SYMBOL_PROP_SPREAD);                                 }
   bool              IsSpreadFloat(void)                          const { return (bool)this.GetProperty(SYMBOL_PROP_SPREAD_FLOAT);                          }
   int               TicksBookdepth(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TICKS_BOOKDEPTH);                        }
   ENUM_SYMBOL_CALC_MODE TradeCalcMode(void)                      const { return (ENUM_SYMBOL_CALC_MODE)this.GetProperty(SYMBOL_PROP_TRADE_CALC_MODE);      }
   ENUM_SYMBOL_TRADE_MODE TradeMode(void)                         const { return (ENUM_SYMBOL_TRADE_MODE)this.GetProperty(SYMBOL_PROP_TRADE_MODE);          }
   datetime          StartTime(void)                              const { return (datetime)this.GetProperty(SYMBOL_PROP_START_TIME);                        }
   datetime          ExpirationTime(void)                         const { return (datetime)this.GetProperty(SYMBOL_PROP_EXPIRATION_TIME);                   }
   int               TradeStopLevel(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_STOPS_LEVEL);                      }
   int               TradeFreezeLevel(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_TRADE_FREEZE_LEVEL);                     }
   ENUM_SYMBOL_TRADE_EXECUTION TradeExecutionMode(void)           const { return (ENUM_SYMBOL_TRADE_EXECUTION)this.GetProperty(SYMBOL_PROP_TRADE_EXEMODE);  }
   ENUM_SYMBOL_SWAP_MODE SwapMode(void)                           const { return (ENUM_SYMBOL_SWAP_MODE)this.GetProperty(SYMBOL_PROP_SWAP_MODE);            }
   ENUM_DAY_OF_WEEK  SwapRollover3Days(void)                      const { return (ENUM_DAY_OF_WEEK)this.GetProperty(SYMBOL_PROP_SWAP_ROLLOVER3DAYS);        }
   bool              IsMarginHedgedUseLeg(void)                   const { return (bool)this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED_USE_LEG);                 }
   int               ExpirationModeFlags(void)                    const { return (int)this.GetProperty(SYMBOL_PROP_EXPIRATION_MODE);                        }
   int               FillingModeFlags(void)                       const { return (int)this.GetProperty(SYMBOL_PROP_FILLING_MODE);                           }
   int               OrderModeFlags(void)                         const { return (int)this.GetProperty(SYMBOL_PROP_ORDER_MODE);                             }
   ENUM_SYMBOL_ORDER_GTC_MODE OrderModeGTC(void)                  const { return (ENUM_SYMBOL_ORDER_GTC_MODE)this.GetProperty(SYMBOL_PROP_ORDER_GTC_MODE);  }
   ENUM_SYMBOL_OPTION_MODE OptionMode(void)                       const { return (ENUM_SYMBOL_OPTION_MODE)this.GetProperty(SYMBOL_PROP_OPTION_MODE);        }
   ENUM_SYMBOL_OPTION_RIGHT OptionRight(void)                     const { return (ENUM_SYMBOL_OPTION_RIGHT)this.GetProperty(SYMBOL_PROP_OPTION_RIGHT);      }
//--- Real properties
   double            Bid(void)                                    const { return this.GetProperty(SYMBOL_PROP_BID);                                         }
   double            BidHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_BIDHIGH);                                     }
   double            BidLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_BIDLOW);                                      }
   double            Ask(void)                                    const { return this.GetProperty(SYMBOL_PROP_ASK);                                         }
   double            AskHigh(void)                                const { return this.GetProperty(SYMBOL_PROP_ASKHIGH);                                     }
   double            AskLow(void)                                 const { return this.GetProperty(SYMBOL_PROP_ASKLOW);                                      }
   double            Last(void)                                   const { return this.GetProperty(SYMBOL_PROP_LAST);                                        }
   double            LastHigh(void)                               const { return this.GetProperty(SYMBOL_PROP_LASTHIGH);                                    }
   double            LastLow(void)                                const { return this.GetProperty(SYMBOL_PROP_LASTLOW);                                     }
   double            VolumeReal(void)                             const { return this.GetProperty(SYMBOL_PROP_VOLUME_REAL);                                 }
   double            VolumeHighReal(void)                         const { return this.GetProperty(SYMBOL_PROP_VOLUMEHIGH_REAL);                             }
   double            VolumeLowReal(void)                          const { return this.GetProperty(SYMBOL_PROP_VOLUMELOW_REAL);                              }
   double            OptionStrike(void)                           const { return this.GetProperty(SYMBOL_PROP_OPTION_STRIKE);                               }
   double            Point(void)                                  const { return this.GetProperty(SYMBOL_PROP_POINT);                                       }
   double            TradeTickValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE);                            }
   double            TradeTickValueProfit(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_PROFIT);                     }
   double            TradeTickValueLoss(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_VALUE_LOSS);                       }
   double            TradeTickSize(void)                          const { return this.GetProperty(SYMBOL_PROP_TRADE_TICK_SIZE);                             }
   double            TradeContractSize(void)                      const { return this.GetProperty(SYMBOL_PROP_TRADE_CONTRACT_SIZE);                         }
   double            TradeAccuredInterest(void)                   const { return this.GetProperty(SYMBOL_PROP_TRADE_ACCRUED_INTEREST);                      }
   double            TradeFaceValue(void)                         const { return this.GetProperty(SYMBOL_PROP_TRADE_FACE_VALUE);                            }
   double            TradeLiquidityRate(void)                     const { return this.GetProperty(SYMBOL_PROP_TRADE_LIQUIDITY_RATE);                        }
   double            LotsMin(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MIN);                                  }
   double            LotsMax(void)                                const { return this.GetProperty(SYMBOL_PROP_VOLUME_MAX);                                  }
   double            LotsStep(void)                               const { return this.GetProperty(SYMBOL_PROP_VOLUME_STEP);                                 }
   double            VolumeLimit(void)                            const { return this.GetProperty(SYMBOL_PROP_VOLUME_LIMIT);                                }
   double            SwapLong(void)                               const { return this.GetProperty(SYMBOL_PROP_SWAP_LONG);                                   }
   double            SwapShort(void)                              const { return this.GetProperty(SYMBOL_PROP_SWAP_SHORT);                                  }
   double            MarginInitial(void)                          const { return this.GetProperty(SYMBOL_PROP_MARGIN_INITIAL);                              }
   double            MarginMaintenance(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_MAINTENANCE);                          }
   double            MarginLongInitial(void)                      const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_INITIAL);                         }
   double            MarginBuyStopInitial(void)                   const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_INITIAL);                     }
   double            MarginBuyLimitInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_INITIAL);                    }
   double            MarginBuyStopLimitInitial(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_INITIAL);                }
   double            MarginLongMaintenance(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_LONG_MAINTENANCE);                     }
   double            MarginBuyStopMaintenance(void)               const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOP_MAINTENANCE);                 }
   double            MarginBuyLimitMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_LIMIT_MAINTENANCE);                }
   double            MarginBuyStopLimitMaintenance(void)          const { return this.GetProperty(SYMBOL_PROP_MARGIN_BUY_STOPLIMIT_MAINTENANCE);            }
   double            MarginShortInitial(void)                     const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_INITIAL);                        }
   double            MarginSellStopInitial(void)                  const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_INITIAL);                    }
   double            MarginSellLimitInitial(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_INITIAL);                   }
   double            MarginSellStopLimitInitial(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_INITIAL);               }
   double            MarginShortMaintenance(void)                 const { return this.GetProperty(SYMBOL_PROP_MARGIN_SHORT_MAINTENANCE);                    }
   double            MarginSellStopMaintenance(void)              const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOP_MAINTENANCE);                }
   double            MarginSellLimitMaintenance(void)             const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_LIMIT_MAINTENANCE);               }
   double            MarginSellStopLimitMaintenance(void)         const { return this.GetProperty(SYMBOL_PROP_MARGIN_SELL_STOPLIMIT_MAINTENANCE);           }
   double            SessionVolume(void)                          const { return this.GetProperty(SYMBOL_PROP_SESSION_VOLUME);                              }
   double            SessionTurnover(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_TURNOVER);                            }
   double            SessionInterest(void)                        const { return this.GetProperty(SYMBOL_PROP_SESSION_INTEREST);                            }
   double            SessionBuyOrdersVolume(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_BUY_ORDERS_VOLUME);                   }
   double            SessionSellOrdersVolume(void)                const { return this.GetProperty(SYMBOL_PROP_SESSION_SELL_ORDERS_VOLUME);                  }
   double            SessionOpen(void)                            const { return this.GetProperty(SYMBOL_PROP_SESSION_OPEN);                                }
   double            SessionClose(void)                           const { return this.GetProperty(SYMBOL_PROP_SESSION_CLOSE);                               }
   double            SessionAW(void)                              const { return this.GetProperty(SYMBOL_PROP_SESSION_AW);                                  }
   double            SessionPriceSettlement(void)                 const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_SETTLEMENT);                    }
   double            SessionPriceLimitMin(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MIN);                     }
   double            SessionPriceLimitMax(void)                   const { return this.GetProperty(SYMBOL_PROP_SESSION_PRICE_LIMIT_MAX);                     }
   double            MarginHedged(void)                           const { return this.GetProperty(SYMBOL_PROP_MARGIN_HEDGED);                               }
   double            NormalizedPrice(const double price)          const;
   double            NormalizedLot(const double volume)           const;
   double            BidLast(void)                                const;
   double            BidLastHigh(void)                            const;
   double            BidLastLow(void)                             const;
//--- String properties
   string            Name(void)                                   const { return this.GetProperty(SYMBOL_PROP_NAME);                                        }
   string            Basis(void)                                  const { return this.GetProperty(SYMBOL_PROP_BASIS);                                       }
   string            CurrencyBase(void)                           const { return this.GetProperty(SYMBOL_PROP_CURRENCY_BASE);                               }
   string            CurrencyProfit(void)                         const { return this.GetProperty(SYMBOL_PROP_CURRENCY_PROFIT);                             }
   string            CurrencyMargin(void)                         const { return this.GetProperty(SYMBOL_PROP_CURRENCY_MARGIN);                             }
   string            Bank(void)                                   const { return this.GetProperty(SYMBOL_PROP_BANK);                                        }
   string            Description(void)                            const { return this.GetProperty(SYMBOL_PROP_DESCRIPTION);                                 }
   string            Formula(void)                                const { return this.GetProperty(SYMBOL_PROP_FORMULA);                                     }
   string            ISIN(void)                                   const { return this.GetProperty(SYMBOL_PROP_ISIN);                                        }
   string            Page(void)                                   const { return this.GetProperty(SYMBOL_PROP_PAGE);                                        }
   string            Path(void)                                   const { return this.GetProperty(SYMBOL_PROP_PATH);                                        }
   string            Category(void)                               const { return this.GetProperty(SYMBOL_PROP_CATEGORY);                                    }
   string            Exchange(void)                               const { return this.GetProperty(SYMBOL_PROP_EXCHANGE);                                    }

//+------------------------------------------------------------------+
```

**Add receiving these new properties in the closed parametric class constructor:**

```
//--- Save string properties
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_NAME)]                             = this.m_name;
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_BASE)]                    = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_BASE);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_PROFIT)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_PROFIT);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CURRENCY_MARGIN)]                  = ::SymbolInfoString(this.m_name,SYMBOL_CURRENCY_MARGIN);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_DESCRIPTION)]                      = ::SymbolInfoString(this.m_name,SYMBOL_DESCRIPTION);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PATH)]                             = ::SymbolInfoString(this.m_name,SYMBOL_PATH);
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BASIS)]                            = this.SymbolBasis();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_BANK)]                             = this.SymbolBank();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_ISIN)]                             = this.SymbolISIN();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_FORMULA)]                          = this.SymbolFormula();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_PAGE)]                             = this.SymbolPage();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_CATEGORY)]                         = this.SymbolCategory();
   this.m_string_prop[this.IndexProp(SYMBOL_PROP_EXCHANGE)]                         = this.SymbolExchange();
//--- Save additional integer properties
```

**Implement the methods for receiving the two new properties outside the class body:**

```
//+------------------------------------------------------------------+
//| Return a symbol category                                         |
//+------------------------------------------------------------------+
string CSymbol::SymbolCategory(void) const
  {
   return
     (
      #ifdef __MQL5__
         (
          ::TerminalInfoInteger(TERMINAL_BUILD)<2155 ? ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MT5_LESS_2155) :
          ::SymbolInfoString(this.m_name,SYMBOL_CATEGORY)
         )
      #else
         ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4)
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Return an exchange name                                          |
//| a symbol is traded on                                            |
//+------------------------------------------------------------------+
string CSymbol::SymbolExchange(void) const
  {
   return
     (
      #ifdef __MQL5__
         (
          ::TerminalInfoInteger(TERMINAL_BUILD)<2155 ? ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MT5_LESS_2155) :
          ::SymbolInfoString(this.m_name,SYMBOL_EXCHANGE)
         )
      #else
         ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED_MQL4)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

Here, check the terminal build for MQL5. If it is lower than 2155,

inform the current property is not supported in the version below 2155,
otherwise

return the new [SYMBOL\_EXCHANGE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_string)
property.

In case of MQL4, inform the property is not supported in MQL4.

**Add the return of descriptions of the two new properties**
**to the implementation of the method returning the description of a symbol string property:**

```
//+------------------------------------------------------------------+
//| Return the description of a symbol string property               |
//+------------------------------------------------------------------+
string CSymbol::GetPropertyDescription(ENUM_SYMBOL_PROP_STRING property)
  {
   return
     (
      property==SYMBOL_PROP_NAME             ?  CMessage::Text(MSG_SYM_PROP_NAME)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
          ": "+this.GetProperty(property)
         )  :
      property==SYMBOL_PROP_BASIS            ?  CMessage::Text(MSG_SYM_PROP_BASIS)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_BASE    ?  CMessage::Text(MSG_SYM_PROP_CURRENCY_BASE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_PROFIT  ?  CMessage::Text(MSG_SYM_PROP_CURRENCY_PROFIT)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CURRENCY_MARGIN  ?  CMessage::Text(MSG_SYM_PROP_CURRENCY_MARGIN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_BANK             ?  CMessage::Text(MSG_SYM_PROP_BANK)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_DESCRIPTION      ?  CMessage::Text(MSG_SYM_PROP_DESCRIPTION)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_FORMULA          ?  CMessage::Text(MSG_SYM_PROP_FORMULA)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_ISIN             ?  CMessage::Text(MSG_SYM_PROP_ISIN)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_PAGE             ?  CMessage::Text(MSG_SYM_PROP_PAGE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_PATH             ?  CMessage::Text(MSG_SYM_PROP_PATH)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_CATEGORY         ?  CMessage::Text(MSG_SYM_PROP_CAYEGORY)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      property==SYMBOL_PROP_EXCHANGE         ?  CMessage::Text(MSG_SYM_PROP_EXCHANGE)+
         (!this.SupportProperty(property) ?  ": "+CMessage::Text(MSG_LIB_PROP_NOT_SUPPORTED) :
         (this.GetProperty(property)=="" || this.GetProperty(property)==NULL  ?  ": ("+CMessage::Text(MSG_LIB_PROP_EMPTY)+")" : ": \""+this.GetProperty(property)+"\"")
         )  :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Now let's add the new property to the account object in \\MQL5\\Include\\DoEasy\\Objects\\Accounts\ **Account.mqh**.

**Write the new field in the account property structure:**

```
//+------------------------------------------------------------------+
//| Account class                                                    |
//+------------------------------------------------------------------+
class CAccount : public CBaseObj
  {
private:
   struct SData
     {
      //--- Account integer properties
      long           login;                        // ACCOUNT_LOGIN (Account number)
      int            trade_mode;                   // ACCOUNT_TRADE_MODE (Trading account type)
      long           leverage;                     // ACCOUNT_LEVERAGE (Leverage)
      int            limit_orders;                 // ACCOUNT_LIMIT_ORDERS (Maximum allowed number of active pending orders)
      int            margin_so_mode;               // ACCOUNT_MARGIN_SO_MODE (Mode of setting the minimum available margin level)
      bool           trade_allowed;                // ACCOUNT_TRADE_ALLOWED (Permission to trade for the current account from the server side)
      bool           trade_expert;                 // ACCOUNT_TRADE_EXPERT (Permission to trade for an EA from the server side)
      int            margin_mode;                  // ACCOUNT_MARGIN_MODE (Margin calculation mode)
      int            currency_digits;              // ACCOUNT_CURRENCY_DIGITS (Number of decimal places for the account currency)
      int            server_type;                  // Trade server type (MetaTrader 5, MetaTrader 4)
      bool           fifo_close;                   // The flag indicating that positions can be closed only by the FIFO rule
      //--- Account real properties
      double         balance;                      // ACCOUNT_BALANCE (Account balance in a deposit currency)
      double         credit;                       // ACCOUNT_CREDIT (Credit in a deposit currency)
      double         profit;                       // ACCOUNT_PROFIT (Current profit on an account in the account currency)
      double         equity;                       // ACCOUNT_EQUITY (Equity on an account in the deposit currency)
      double         margin;                       // ACCOUNT_MARGIN (Reserved margin on an account in a deposit currency)
      double         margin_free;                  // ACCOUNT_MARGIN_FREE (Free funds available for opening a position in a deposit currency)
      double         margin_level;                 // ACCOUNT_MARGIN_LEVEL (Margin level on an account in %)
      double         margin_so_call;               // ACCOUNT_MARGIN_SO_CALL (MarginCall)
      double         margin_so_so;                 // ACCOUNT_MARGIN_SO_SO (StopOut)
      double         margin_initial;               // ACCOUNT_MARGIN_INITIAL (Funds reserved on an account to ensure a guarantee amount for all pending orders)
      double         margin_maintenance;           // ACCOUNT_MARGIN_MAINTENANCE (Funds reserved on an account to ensure a minimum amount for all open positions)
      double         assets;                       // ACCOUNT_ASSETS (Current assets on an account)
      double         liabilities;                  // ACCOUNT_LIABILITIES (Current liabilities on an account)
      double         comission_blocked;            // ACCOUNT_COMMISSION_BLOCKED (Current sum of blocked commissions on an account)
      //--- Account string properties
      uchar          name[128];                    // ACCOUNT_NAME (Client name)
      uchar          server[64];                   // ACCOUNT_SERVER (Trade server name)
      uchar          currency[32];                 // ACCOUNT_CURRENCY (Deposit currency)
      uchar          company[128];                 // ACCOUNT_COMPANY (Name of a company serving an account)
     };
   SData             m_struct_obj;                                      // Account object structure
```

**In the block of methods for a simplified access to object properties of the public class section, add the**
**method returning the new property:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the account object properties  |
//+------------------------------------------------------------------+
//--- Return the account's integer properties
   ENUM_ACCOUNT_TRADE_MODE    TradeMode(void)                        const { return (ENUM_ACCOUNT_TRADE_MODE)this.GetProperty(ACCOUNT_PROP_TRADE_MODE);           }
   ENUM_ACCOUNT_STOPOUT_MODE  MarginSOMode(void)                     const { return (ENUM_ACCOUNT_STOPOUT_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_SO_MODE);     }
   ENUM_ACCOUNT_MARGIN_MODE   MarginMode(void)                       const { return (ENUM_ACCOUNT_MARGIN_MODE)this.GetProperty(ACCOUNT_PROP_MARGIN_MODE);         }
   long              Login(void)                                     const { return this.GetProperty(ACCOUNT_PROP_LOGIN);                                         }
   long              Leverage(void)                                  const { return this.GetProperty(ACCOUNT_PROP_LEVERAGE);                                      }
   long              LimitOrders(void)                               const { return this.GetProperty(ACCOUNT_PROP_LIMIT_ORDERS);                                  }
   long              TradeAllowed(void)                              const { return this.GetProperty(ACCOUNT_PROP_TRADE_ALLOWED);                                 }
   long              TradeExpert(void)                               const { return this.GetProperty(ACCOUNT_PROP_TRADE_EXPERT);                                  }
   long              CurrencyDigits(void)                            const { return this.GetProperty(ACCOUNT_PROP_CURRENCY_DIGITS);                               }
   long              ServerType(void)                                const { return this.GetProperty(ACCOUNT_PROP_SERVER_TYPE);                                   }
   long              FIFOClose(void)                                 const { return this.GetProperty(ACCOUNT_PROP_FIFO_CLOSE);                                    }
//--- Return the account's real properties
```

**Add saving the new account object property in the class constructor:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAccount::CAccount(void)
  {
//--- Initialize control data
   this.SetControlDataArraySizeLong(ACCOUNT_PROP_INTEGER_TOTAL);
   this.SetControlDataArraySizeDouble(ACCOUNT_PROP_DOUBLE_TOTAL);
   this.ResetChangesParams();
   this.ResetControlsParams();

//--- Save integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                              = ::AccountInfoInteger(ACCOUNT_LOGIN);
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                         = ::AccountInfoInteger(ACCOUNT_TRADE_MODE);
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                           = ::AccountInfoInteger(ACCOUNT_LEVERAGE);
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                       = ::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                     = ::AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                      = ::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                       = ::AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif ;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);
   this.m_long_prop[ACCOUNT_PROP_FIFO_CLOSE]                         = (#ifdef __MQL5__::TerminalInfoInteger(TERMINAL_BUILD)<2155 ? false : ::AccountInfoInteger(ACCOUNT_FIFO_CLOSE) #else false #endif );

//--- Save real properties
```

If the terminal build is below 2155, enter false. Otherwise, write the [ACCOUNT\_FIFO\_CLOSE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_integer)
account property value.

**Also, add saving the property to the method of updating the**
**Refresh() account properties:**

```
//+------------------------------------------------------------------+
//| Update all account data                                          |
//+------------------------------------------------------------------+
void CAccount::Refresh(void)
  {
//--- Initialize event data
   this.m_is_event=false;
   this.m_hash_sum=0;
//--- Update integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                                 = ::AccountInfoInteger(ACCOUNT_LOGIN);
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                            = ::AccountInfoInteger(ACCOUNT_TRADE_MODE);
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                              = ::AccountInfoInteger(ACCOUNT_LEVERAGE);
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                          = ::AccountInfoInteger(ACCOUNT_LIMIT_ORDERS);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                        = ::AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE);
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                         = ::AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                          = ::AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                           = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_MARGIN_MODE) #else ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif ;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                       = #ifdef __MQL5__::AccountInfoInteger(ACCOUNT_CURRENCY_DIGITS) #else 2 #endif ;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                           = (::TerminalInfoString(TERMINAL_NAME)=="MetaTrader 5" ? 5 : 4);
   this.m_long_prop[ACCOUNT_PROP_FIFO_CLOSE]                            = (#ifdef __MQL5__::TerminalInfoInteger(TERMINAL_BUILD)<2155 ? false : ::AccountInfoInteger(ACCOUNT_FIFO_CLOSE) #else false #endif );

//--- Update real properties
```

**Add filling the new property in the method of creating the**
**account object from the structure:**

```
//+------------------------------------------------------------------+
//| Create the account object from the structure                     |
//+------------------------------------------------------------------+
void CAccount::StructToObject(void)
  {
//--- Save integer properties
   this.m_long_prop[ACCOUNT_PROP_LOGIN]                              = this.m_struct_obj.login;
   this.m_long_prop[ACCOUNT_PROP_TRADE_MODE]                         = this.m_struct_obj.trade_mode;
   this.m_long_prop[ACCOUNT_PROP_LEVERAGE]                           = this.m_struct_obj.leverage;
   this.m_long_prop[ACCOUNT_PROP_LIMIT_ORDERS]                       = this.m_struct_obj.limit_orders;
   this.m_long_prop[ACCOUNT_PROP_MARGIN_SO_MODE]                     = this.m_struct_obj.margin_so_mode;
   this.m_long_prop[ACCOUNT_PROP_TRADE_ALLOWED]                      = this.m_struct_obj.trade_allowed;
   this.m_long_prop[ACCOUNT_PROP_TRADE_EXPERT]                       = this.m_struct_obj.trade_expert;
   this.m_long_prop[ACCOUNT_PROP_MARGIN_MODE]                        = this.m_struct_obj.margin_mode;
   this.m_long_prop[ACCOUNT_PROP_CURRENCY_DIGITS]                    = this.m_struct_obj.currency_digits;
   this.m_long_prop[ACCOUNT_PROP_SERVER_TYPE]                        = this.m_struct_obj.server_type;
   this.m_long_prop[ACCOUNT_PROP_FIFO_CLOSE]                         = this.m_struct_obj.fifo_close;
//--- Save real properties
```

**Add displaying the new property description in the method**
**returning the description of the account object integer properties:**

```
//+------------------------------------------------------------------+
//| Return the description of the account integer property           |
//+------------------------------------------------------------------+
string CAccount::GetPropertyDescription(ENUM_ACCOUNT_PROP_INTEGER property)
  {
   return
     (
      property==ACCOUNT_PROP_LOGIN           ?  CMessage::Text(MSG_ACC_PROP_LOGIN)+": "+(string)this.GetProperty(property)                            :
      property==ACCOUNT_PROP_TRADE_MODE      ?  CMessage::Text(MSG_ACC_PROP_TRADE_MODE)+": "+this.TradeModeDescription()                              :
      property==ACCOUNT_PROP_LEVERAGE        ?  CMessage::Text(MSG_ACC_PROP_LEVERAGE)+": "+(string)this.GetProperty(property)                         :
      property==ACCOUNT_PROP_LIMIT_ORDERS    ?  CMessage::Text(MSG_ACC_PROP_LIMIT_ORDERS)+": "+(string)this.GetProperty(property)                     :
      property==ACCOUNT_PROP_MARGIN_SO_MODE  ?  CMessage::Text(MSG_ACC_PROP_MARGIN_SO_MODE)+": "+this.MarginSOModeDescription()                       :
      property==ACCOUNT_PROP_TRADE_ALLOWED   ?  CMessage::Text(MSG_ACC_PROP_TRADE_ALLOWED)+": "+
                                                   (this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))  :
      property==ACCOUNT_PROP_TRADE_EXPERT    ?  CMessage::Text(MSG_ACC_PROP_TRADE_EXPERT)+": "+
                                                   (this.GetProperty(property) ? CMessage::Text(MSG_LIB_TEXT_YES) : CMessage::Text(MSG_LIB_TEXT_NO))  :
      property==ACCOUNT_PROP_MARGIN_MODE     ?  CMessage::Text(MSG_ACC_PROP_MARGIN_MODE)+": "+this.MarginModeDescription()                            :
      property==ACCOUNT_PROP_CURRENCY_DIGITS ?  CMessage::Text(MSG_ACC_PROP_CURRENCY_DIGITS)+": "+(string)this.GetProperty(property)                  :
      property==ACCOUNT_PROP_SERVER_TYPE     ?  CMessage::Text(MSG_ACC_PROP_SERVER_TYPE)+": "+(string)this.GetProperty(property)                      :
      property==ACCOUNT_PROP_FIFO_CLOSE      ?  CMessage::Text(MSG_ACC_PROP_FIFO_CLOSE)+": "+(string)this.GetProperty(property)                       :
      ""
     );
  }
//+------------------------------------------------------------------+
```

Keep in mind that adding a new property to the account object makes account objects previously saved in the common folder of all terminals
invalid. To avoid reading errors, make sure to remove all previously created account files in the common folder of all terminals in the
\\DoEasy\\Accounts\ subfolder. Subsequently, when replacing the account with a new one, the library saves the account objects in the
new format on its own.



To open the common folder of all terminals, use the editor command File --> Open Common Data Folder. In the explorer window, go to
the Files folder and then enter \\DoEasy\\Accounts\

To work with the trading class we are currently working on, we need to obtain the margin necessary for opening a position or placing an order.

**In the public section of the class, add the definition of the method**
**returning the margin required for opening a specified position or placing a pending order:**

```
public:
//--- Constructor
                     CAccount(void);
//--- Set account's (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_ACCOUNT_PROP_INTEGER property,long value)        { this.m_long_prop[property]=value;                                  }
   void              SetProperty(ENUM_ACCOUNT_PROP_DOUBLE property,double value)       { this.m_double_prop[this.IndexProp(property)]=value;                }
   void              SetProperty(ENUM_ACCOUNT_PROP_STRING property,string value)       { this.m_string_prop[this.IndexProp(property)]=value;                }
//--- Return (1) integer, (2) real and (3) string order properties from the account string property
   long              GetProperty(ENUM_ACCOUNT_PROP_INTEGER property)             const { return this.m_long_prop[property];                                 }
   double            GetProperty(ENUM_ACCOUNT_PROP_DOUBLE property)              const { return this.m_double_prop[this.IndexProp(property)];               }
   string            GetProperty(ENUM_ACCOUNT_PROP_STRING property)              const { return this.m_string_prop[this.IndexProp(property)];               }
//--- Return the flag of calculating MarginCall and StopOut levels in %
   bool              IsPercentsForSOLevels(void)                                 const { return this.MarginSOMode()==ACCOUNT_STOPOUT_MODE_PERCENT;          }
//--- Return the flag of supporting the property by the account object
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_INTEGER property)               { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_DOUBLE property)                { return true; }
   virtual bool      SupportProperty(ENUM_ACCOUNT_PROP_STRING property)                { return true; }

//--- Compare CAccount objects by all possible properties (for sorting the lists by a specified account object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CAccount objects by account properties (to search for equal account objects)
   bool              IsEqual(CAccount* compared_account) const;
//--- Update all account data
   virtual void      Refresh(void);
//--- (1) Save the account object to the file, (2), download the account object from the file
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);

//--- Return the margin required for opening a position or placing a pending order
   double            MarginForAction(const ENUM_ORDER_TYPE action,const string symbol,const double volume,const double price) const;

//+------------------------------------------------------------------+
```

**Write its implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Return the margin required for opening a position                |
//| or placing a pending order                                       |
//+------------------------------------------------------------------+
double CAccount::MarginForAction(const ENUM_ORDER_TYPE action,const string symbol,const double volume,const double price) const
  {
   double margin=EMPTY_VALUE;
   #ifdef __MQL5__
      return(!::OrderCalcMargin(action,symbol,volume,price,margin) ? EMPTY_VALUE : margin);
   #else
      return this.MarginFree()-::AccountFreeMarginCheck(symbol,action,volume);
   #endif
  }
//+------------------------------------------------------------------+
```

In MQL5, the margin required for opening a new position or placing a pending order is received using the [OrderCalcMargin()](https://www.mql5.com/en/docs/trading/ordercalcmargin)
function. If for some reason, the function returns an error, return the maximum [DBL\_MAX](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants)
value (this is the value of the EMPTY\_VALUE constant). Otherwise, return the
calculated required margin.

In MQL4, return the existing free margin minus the funds remaining after opening a specified position or placing an order calculated
using the

[AccountFreeMarginCheck()](https://docs.mql4.com/en/account/accountfreemargincheck "https://docs.mql4.com/en/account/accountfreemargincheck")
function.

**This concludes improving the account object class.**

Now let's improve the trading object base class. We need to add the ability to set any sounds for voicing any trading events.

**Create**
**the structure storing the flags of using sounds and audio file names for various events** in the private section of
the class to store data:

```
//+------------------------------------------------------------------+
//| Trading object class                                             |
//+------------------------------------------------------------------+
class CTradeObj
  {
private:
   struct SActionsFlags
     {
   private:
      bool                    m_use_sound_open;          // The flag of using the position opening/order placing sound
      bool                    m_use_sound_close;         // The flag of using the position closing/order removal sound
      bool                    m_use_sound_modify_sl;     // The flag of using the StopLoss position/order modification sound
      bool                    m_use_sound_modify_tp;     // The flag of using the TakeProfit position/order modification sound
      bool                    m_use_sound_modify_price;  // The flag of using the order placement price modification sound
      //---
      string                  m_sound_open;              // Position opening/order placing sound
      string                  m_sound_close;             // Position closing/order removal sound
      string                  m_sound_modify_sl;         // StopLoss position/order modification sound
      string                  m_sound_modify_tp;         // TakeProfit position/order modification sound
      string                  m_sound_modify_price;      // Order placement price modification sound
      //---
      string                  m_sound_open_err;          // Position opening/order placing error sound
      string                  m_sound_close_err;         // Position closing/order removal error sound
      string                  m_sound_modify_sl_err;     // StopLoss position/order modification error sound
      string                  m_sound_modify_tp_err;     // TakeProfit position/order modification error sound
      string                  m_sound_modify_price_err;  // Order placement price modification error sound
   public:
      //--- Method of placing/accessing flags and file names
      void                    UseSoundOpen(const bool flag)                      { this.m_use_sound_open=flag;             }
      void                    UseSoundClose(const bool flag)                     { this.m_use_sound_close=flag;            }
      void                    UseSoundModifySL(const bool flag)                  { this.m_use_sound_modify_sl=flag;        }
      void                    UseSoundModifyTP(const bool flag)                  { this.m_use_sound_modify_tp=flag;        }
      void                    UseSoundModifyPrice(const bool flag)               { this.m_use_sound_modify_price=flag;     }
      bool                    UseSoundOpen(void)                           const { return this.m_use_sound_open;           }
      bool                    UseSoundClose(void)                          const { return this.m_use_sound_close;          }
      bool                    UseSoundModifySL(void)                       const { return this.m_use_sound_modify_sl;      }
      bool                    UseSoundModifyTP(void)                       const { return this.m_use_sound_modify_tp;      }
      bool                    UseSoundModifyPrice(void)                    const { return this.m_use_sound_modify_price;   }
      //---
      void                    SoundOpen(const string sound)                      { this.m_sound_open=sound;                }
      void                    SoundClose(const string sound)                     { this.m_sound_close=sound;               }
      void                    SoundModifySL(const string sound)                  { this.m_sound_modify_sl=sound;           }
      void                    SoundModifyTP(const string sound)                  { this.m_sound_modify_tp=sound;           }
      void                    SoundModifyPrice(const string sound)               { this.m_sound_modify_price=sound;        }
      string                  SoundOpen(void)                              const { return this.m_sound_open;               }
      string                  SoundClose(void)                             const { return this.m_sound_close;              }
      string                  SoundModifySL(void)                          const { return this.m_sound_modify_sl;          }
      string                  SoundModifyTP(void)                          const { return this.m_sound_modify_tp;          }
      string                  SoundModifyPrice(void)                       const { return this.m_sound_modify_price;       }
      //---
      void                    SoundErrorOpen(const string sound)                 { this.m_sound_open_err=sound;            }
      void                    SoundErrorClose(const string sound)                { this.m_sound_close_err=sound;           }
      void                    SoundErrorModifySL(const string sound)             { this.m_sound_modify_sl_err=sound;       }
      void                    SoundErrorModifyTP(const string sound)             { this.m_sound_modify_tp_err=sound;       }
      void                    SoundErrorModifyPrice(const string sound)          { this.m_sound_modify_price_err=sound;    }
      string                  SoundErrorOpen(void)                         const { return this.m_sound_open_err;           }
      string                  SoundErrorClose(void)                        const { return this.m_sound_close_err;          }
      string                  SoundErrorModifySL(void)                     const { return this.m_sound_modify_sl_err;      }
      string                  SoundErrorModifyTP(void)                     const { return this.m_sound_modify_tp_err;      }
      string                  SoundErrorModifyPrice(void)                  const { return this.m_sound_modify_price_err;   }
     };
   struct SActions
     {
      SActionsFlags           Buy;
      SActionsFlags           BuyStop;
      SActionsFlags           BuyLimit;
      SActionsFlags           BuyStopLimit;
      SActionsFlags           Sell;
      SActionsFlags           SellStop;
      SActionsFlags           SellLimit;
      SActionsFlags           SellStopLimit;
     };
   SActions                   m_datas;
```

The structure consists of two nested structures: SActionsFlags
containing data on flags and flag names and SActions **m\_data**
featuring data on the SActionsFlags structure type for each position and order types. This enables us to access the structure field with flags and flag
names of any order or position providing us with greater flexibility in setting and using sounds for any events.

Also, **in the private section of the class, declare the class member variable storing the flag of a trading object using sounds for**
**trading events:**

```
   bool                       m_use_sound;                   // The flag of using sounds of the object trading events
```

This flag is to be the common on/off switch for using sounds for all trading events of a symbol trading object. The flag allows to simultaneously
enable/disable sounds for trading events of each symbol.

**In the public section of the class, declare all the necessary methods for**
**setting and receiving the flags of using sounds and audio file names for all trading events of a symbol base trading object:**

```
public:
//--- Constructor
                              CTradeObj();

//--- Set default values
   void                       Init(const string symbol,
                                   const ulong magic,
                                   const double volume,
                                   const ulong deviation,
                                   const int stoplimit,
                                   const datetime expiration,
                                   const bool async_mode,
                                   const ENUM_ORDER_TYPE_FILLING type_filling,
                                   const ENUM_ORDER_TYPE_TIME type_expiration,
                                   ENUM_LOG_LEVEL log_level);

//--- Set default sounds and flags of using sounds,
   void                       InitSounds(const bool use_sound=false,
                                         const string sound_open=NULL,
                                         const string sound_close=NULL,
                                         const string sound_sl=NULL,
                                         const string sound_tp=NULL,
                                         const string sound_price=NULL,
                                         const string sound_error=NULL);
//--- Allow working with sounds and set standard sounds
   void                       SetSoundsStandart(void);
//--- Set the flag of using the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       UseSoundOpen(const ENUM_ORDER_TYPE action,const bool flag);
   void                       UseSoundClose(const ENUM_ORDER_TYPE action,const bool flag);
   void                       UseSoundModifySL(const ENUM_ORDER_TYPE action,const bool flag);
   void                       UseSoundModifyTP(const ENUM_ORDER_TYPE action,const bool flag);
   void                       UseSoundModifyPrice(const ENUM_ORDER_TYPE action,const bool flag);
//--- Return the flag of using the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   bool                       UseSoundOpen(const ENUM_ORDER_TYPE action)            const;
   bool                       UseSoundClose(const ENUM_ORDER_TYPE action)           const;
   bool                       UseSoundModifySL(const ENUM_ORDER_TYPE action)        const;
   bool                       UseSoundModifyTP(const ENUM_ORDER_TYPE action)        const;
   bool                       UseSoundModifyPrice(const ENUM_ORDER_TYPE action)     const;

//--- Set the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       SetSoundOpen(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundClose(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundModifySL(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundModifyTP(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundModifyPrice(const ENUM_ORDER_TYPE action,const string sound);
//--- Set the error sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       SetSoundErrorOpen(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundErrorClose(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundErrorModifySL(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundErrorModifyTP(const ENUM_ORDER_TYPE action,const string sound);
   void                       SetSoundErrorModifyPrice(const ENUM_ORDER_TYPE action,const string sound);

//--- Return the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   string                     GetSoundOpen(const ENUM_ORDER_TYPE action)               const;
   string                     GetSoundClose(const ENUM_ORDER_TYPE action)              const;
   string                     GetSoundModifySL(const ENUM_ORDER_TYPE action)           const;
   string                     GetSoundModifyTP(const ENUM_ORDER_TYPE action)           const;
   string                     GetSoundModifyPrice(const ENUM_ORDER_TYPE action)        const;
//--- Return the error sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   string                     GetSoundErrorOpen(const ENUM_ORDER_TYPE action)          const;
   string                     GetSoundErrorClose(const ENUM_ORDER_TYPE action)         const;
   string                     GetSoundErrorModifySL(const ENUM_ORDER_TYPE action)      const;
   string                     GetSoundErrorModifyTP(const ENUM_ORDER_TYPE action)      const;
   string                     GetSoundErrorModifyPrice(const ENUM_ORDER_TYPE action)   const;

//--- Play the sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       PlaySoundOpen(const ENUM_ORDER_TYPE action);
   void                       PlaySoundClose(const ENUM_ORDER_TYPE action);
   void                       PlaySoundModifySL(const ENUM_ORDER_TYPE action);
   void                       PlaySoundModifyTP(const ENUM_ORDER_TYPE action);
   void                       PlaySoundModifyPrice(const ENUM_ORDER_TYPE action);
//--- Play the error sound of (1) opening/placing a specified position/order type,
//--- (2) closing/removal of a specified position/order type, (3) StopLoss modification for a specified position/order type,
//--- (4) TakeProfit modification for a specified position/order type, (5) placement price modification for a specified order type
   void                       PlaySoundErrorOpen(const ENUM_ORDER_TYPE action);
   void                       PlaySoundErrorClose(const ENUM_ORDER_TYPE action);
   void                       PlaySoundErrorModifySL(const ENUM_ORDER_TYPE action);
   void                       PlaySoundErrorModifyTP(const ENUM_ORDER_TYPE action);
   void                       PlaySoundErrorModifyPrice(const ENUM_ORDER_TYPE action);

//--- Set/return the flag of using sounds
   void                       SetUseSound(const bool flag)                             { this.m_use_sound=flag;               }
   bool                       IsUseSound(void)                                   const { return this.m_use_sound;             }

//--- (1) Return the margin calculation mode, (2) hedge account flag
```

Reset the common flag of using sounds in the class constructor
and

initialize the flags of using sounds and audio files into default values as
disabling sounds and empty file names:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTradeObj::CTradeObj(void) : m_magic(0),
                             m_deviation(5),
                             m_stop_limit(0),
                             m_expiration(0),
                             m_async_mode(false),
                             m_type_filling(ORDER_FILLING_FOK),
                             m_type_expiration(ORDER_TIME_GTC),
                             m_comment(::MQLInfoString(MQL_PROGRAM_NAME)+" by DoEasy"),
                             m_log_level(LOG_LEVEL_ERROR_MSG)
  {
   //--- Margin calculation mode
   this.m_margin_mode=
     (
      #ifdef __MQL5__ (ENUM_ACCOUNT_MARGIN_MODE)::AccountInfoInteger(ACCOUNT_MARGIN_MODE)
      #else /* MQL4 */ ACCOUNT_MARGIN_MODE_RETAIL_HEDGING #endif
     );
   //--- Set default sounds and flags of using sounds
   this.m_use_sound=false;
   this.InitSounds();
  }
//+------------------------------------------------------------------+
```

**Implement the method for initializing flags of using sounds and setting audio file names beyond the class body:**

```
//+------------------------------------------------------------------+
//| Set default sounds and flags of using sounds                     |
//+------------------------------------------------------------------+
void CTradeObj::InitSounds(const bool use_sound=false,
                           const string sound_open=NULL,
                           const string sound_close=NULL,
                           const string sound_sl=NULL,
                           const string sound_tp=NULL,
                           const string sound_price=NULL,
                           const string sound_error=NULL)
  {
   this.m_datas.Buy.UseSoundOpen(use_sound);
   this.m_datas.Buy.UseSoundClose(use_sound);
   this.m_datas.Buy.UseSoundModifySL(use_sound);
   this.m_datas.Buy.UseSoundModifyTP(use_sound);
   this.m_datas.Buy.UseSoundModifyPrice(use_sound);
   this.m_datas.Buy.SoundOpen(sound_open);
   this.m_datas.Buy.SoundClose(sound_close);
   this.m_datas.Buy.SoundModifySL(sound_sl);
   this.m_datas.Buy.SoundModifyTP(sound_tp);
   this.m_datas.Buy.SoundModifyPrice(sound_price);
   this.m_datas.Buy.SoundErrorClose(sound_error);
   this.m_datas.Buy.SoundErrorOpen(sound_error);
   this.m_datas.Buy.SoundErrorModifySL(sound_error);
   this.m_datas.Buy.SoundErrorModifyTP(sound_error);
   this.m_datas.Buy.SoundErrorModifyPrice(sound_error);

   this.m_datas.BuyStop.UseSoundOpen(use_sound);
   this.m_datas.BuyStop.UseSoundClose(use_sound);
   this.m_datas.BuyStop.UseSoundModifySL(use_sound);
   this.m_datas.BuyStop.UseSoundModifyTP(use_sound);
   this.m_datas.BuyStop.UseSoundModifyPrice(use_sound);
   this.m_datas.BuyStop.SoundOpen(sound_open);
   this.m_datas.BuyStop.SoundClose(sound_close);
   this.m_datas.BuyStop.SoundModifySL(sound_sl);
   this.m_datas.BuyStop.SoundModifyTP(sound_tp);
   this.m_datas.BuyStop.SoundModifyPrice(sound_price);
   this.m_datas.BuyStop.SoundErrorClose(sound_error);
   this.m_datas.BuyStop.SoundErrorOpen(sound_error);
   this.m_datas.BuyStop.SoundErrorModifySL(sound_error);
   this.m_datas.BuyStop.SoundErrorModifyTP(sound_error);
   this.m_datas.BuyStop.SoundErrorModifyPrice(sound_error);

   this.m_datas.BuyLimit.UseSoundOpen(use_sound);
   this.m_datas.BuyLimit.UseSoundClose(use_sound);
   this.m_datas.BuyLimit.UseSoundModifySL(use_sound);
   this.m_datas.BuyLimit.UseSoundModifyTP(use_sound);
   this.m_datas.BuyLimit.UseSoundModifyPrice(use_sound);
   this.m_datas.BuyLimit.SoundOpen(sound_open);
   this.m_datas.BuyLimit.SoundClose(sound_close);
   this.m_datas.BuyLimit.SoundModifySL(sound_sl);
   this.m_datas.BuyLimit.SoundModifyTP(sound_tp);
   this.m_datas.BuyLimit.SoundModifyPrice(sound_price);
   this.m_datas.BuyLimit.SoundErrorClose(sound_error);
   this.m_datas.BuyLimit.SoundErrorOpen(sound_error);
   this.m_datas.BuyLimit.SoundErrorModifySL(sound_error);
   this.m_datas.BuyLimit.SoundErrorModifyTP(sound_error);
   this.m_datas.BuyLimit.SoundErrorModifyPrice(sound_error);

   this.m_datas.BuyStopLimit.UseSoundOpen(use_sound);
   this.m_datas.BuyStopLimit.UseSoundClose(use_sound);
   this.m_datas.BuyStopLimit.UseSoundModifySL(use_sound);
   this.m_datas.BuyStopLimit.UseSoundModifyTP(use_sound);
   this.m_datas.BuyStopLimit.UseSoundModifyPrice(use_sound);
   this.m_datas.BuyStopLimit.SoundOpen(sound_open);
   this.m_datas.BuyStopLimit.SoundClose(sound_close);
   this.m_datas.BuyStopLimit.SoundModifySL(sound_sl);
   this.m_datas.BuyStopLimit.SoundModifyTP(sound_tp);
   this.m_datas.BuyStopLimit.SoundModifyPrice(sound_price);
   this.m_datas.BuyStopLimit.SoundErrorClose(sound_error);
   this.m_datas.BuyStopLimit.SoundErrorOpen(sound_error);
   this.m_datas.BuyStopLimit.SoundErrorModifySL(sound_error);
   this.m_datas.BuyStopLimit.SoundErrorModifyTP(sound_error);
   this.m_datas.BuyStopLimit.SoundErrorModifyPrice(sound_error);

   this.m_datas.Sell.UseSoundOpen(use_sound);
   this.m_datas.Sell.UseSoundClose(use_sound);
   this.m_datas.Sell.UseSoundModifySL(use_sound);
   this.m_datas.Sell.UseSoundModifyTP(use_sound);
   this.m_datas.Sell.UseSoundModifyPrice(use_sound);
   this.m_datas.Sell.SoundOpen(sound_open);
   this.m_datas.Sell.SoundClose(sound_close);
   this.m_datas.Sell.SoundModifySL(sound_sl);
   this.m_datas.Sell.SoundModifyTP(sound_tp);
   this.m_datas.Sell.SoundModifyPrice(sound_price);
   this.m_datas.Sell.SoundErrorClose(sound_error);
   this.m_datas.Sell.SoundErrorOpen(sound_error);
   this.m_datas.Sell.SoundErrorModifySL(sound_error);
   this.m_datas.Sell.SoundErrorModifyTP(sound_error);
   this.m_datas.Sell.SoundErrorModifyPrice(sound_error);

   this.m_datas.SellStop.UseSoundOpen(use_sound);
   this.m_datas.SellStop.UseSoundClose(use_sound);
   this.m_datas.SellStop.UseSoundModifySL(use_sound);
   this.m_datas.SellStop.UseSoundModifyTP(use_sound);
   this.m_datas.SellStop.UseSoundModifyPrice(use_sound);
   this.m_datas.SellStop.SoundOpen(sound_open);
   this.m_datas.SellStop.SoundClose(sound_close);
   this.m_datas.SellStop.SoundModifySL(sound_sl);
   this.m_datas.SellStop.SoundModifyTP(sound_tp);
   this.m_datas.SellStop.SoundModifyPrice(sound_price);
   this.m_datas.SellStop.SoundErrorClose(sound_error);
   this.m_datas.SellStop.SoundErrorOpen(sound_error);
   this.m_datas.SellStop.SoundErrorModifySL(sound_error);
   this.m_datas.SellStop.SoundErrorModifyTP(sound_error);
   this.m_datas.SellStop.SoundErrorModifyPrice(sound_error);

   this.m_datas.SellLimit.UseSoundOpen(use_sound);
   this.m_datas.SellLimit.UseSoundClose(use_sound);
   this.m_datas.SellLimit.UseSoundModifySL(use_sound);
   this.m_datas.SellLimit.UseSoundModifyTP(use_sound);
   this.m_datas.SellLimit.UseSoundModifyPrice(use_sound);
   this.m_datas.SellLimit.SoundOpen(sound_open);
   this.m_datas.SellLimit.SoundClose(sound_close);
   this.m_datas.SellLimit.SoundModifySL(sound_sl);
   this.m_datas.SellLimit.SoundModifyTP(sound_tp);
   this.m_datas.SellLimit.SoundModifyPrice(sound_price);
   this.m_datas.SellLimit.SoundErrorClose(sound_error);
   this.m_datas.SellLimit.SoundErrorOpen(sound_error);
   this.m_datas.SellLimit.SoundErrorModifySL(sound_error);
   this.m_datas.SellLimit.SoundErrorModifyTP(sound_error);
   this.m_datas.SellLimit.SoundErrorModifyPrice(sound_error);

   this.m_datas.SellStopLimit.UseSoundOpen(use_sound);
   this.m_datas.SellStopLimit.UseSoundClose(use_sound);
   this.m_datas.SellStopLimit.UseSoundModifySL(use_sound);
   this.m_datas.SellStopLimit.UseSoundModifyTP(use_sound);
   this.m_datas.SellStopLimit.UseSoundModifyPrice(use_sound);
   this.m_datas.SellStopLimit.SoundOpen(sound_open);
   this.m_datas.SellStopLimit.SoundClose(sound_close);
   this.m_datas.SellStopLimit.SoundModifySL(sound_sl);
   this.m_datas.SellStopLimit.SoundModifyTP(sound_tp);
   this.m_datas.SellStopLimit.SoundModifyPrice(sound_price);
   this.m_datas.SellStopLimit.SoundErrorClose(sound_error);
   this.m_datas.SellStopLimit.SoundErrorOpen(sound_error);
   this.m_datas.SellStopLimit.SoundErrorModifySL(sound_error);
   this.m_datas.SellStopLimit.SoundErrorModifyTP(sound_error);
   this.m_datas.SellStopLimit.SoundErrorModifyPrice(sound_error);
  }
//+------------------------------------------------------------------+
```

The method receives the flag of using trading object event sounds,
file names of

opening, closing,

StopLoss/TakeProfit
modification, order placement price and error
sounds. Next, the structure fields for each trading event are filled in for each order type sent to the server by passed values.

The flag of using sounds and error sound file name are set to be one for all enumerated trading events, while in case of trading events for each
one, sounds are set individually. However, if you wish, you can also set a custom sound for the error sounds of specific events using the
methods announced above, as well as set the flag for using a sound for each trading event of an object individually. The sounds and empty file
names are disabled by default.

Besides, the class features **the method setting standard sounds for all object trading events and allowing their usage for each**
**trading event:**

```
//+------------------------------------------------------------------+
//| Allow working with sounds and set standard sounds                |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundsStandart(void)
  {
   this.m_datas.Buy.UseSoundClose(true);
   this.m_datas.Buy.UseSoundOpen(true);
   this.m_datas.Buy.UseSoundModifySL(true);
   this.m_datas.Buy.UseSoundModifyTP(true);
   this.m_datas.Buy.UseSoundModifyPrice(true);
   this.m_datas.Buy.SoundOpen(SND_OK);
   this.m_datas.Buy.SoundClose(SND_OK);
   this.m_datas.Buy.SoundModifySL(SND_OK);
   this.m_datas.Buy.SoundModifyTP(SND_OK);
   this.m_datas.Buy.SoundModifyPrice(SND_OK);
   this.m_datas.Buy.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.Buy.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.Buy.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.Buy.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.Buy.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.BuyStop.UseSoundClose(true);
   this.m_datas.BuyStop.UseSoundOpen(true);
   this.m_datas.BuyStop.UseSoundModifySL(true);
   this.m_datas.BuyStop.UseSoundModifyTP(true);
   this.m_datas.BuyStop.UseSoundModifyPrice(true);
   this.m_datas.BuyStop.SoundOpen(SND_OK);
   this.m_datas.BuyStop.SoundClose(SND_OK);
   this.m_datas.BuyStop.SoundModifySL(SND_OK);
   this.m_datas.BuyStop.SoundModifyTP(SND_OK);
   this.m_datas.BuyStop.SoundModifyPrice(SND_OK);
   this.m_datas.BuyStop.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.BuyStop.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.BuyStop.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.BuyStop.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.BuyStop.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.BuyLimit.UseSoundClose(true);
   this.m_datas.BuyLimit.UseSoundOpen(true);
   this.m_datas.BuyLimit.UseSoundModifySL(true);
   this.m_datas.BuyLimit.UseSoundModifyTP(true);
   this.m_datas.BuyLimit.UseSoundModifyPrice(true);
   this.m_datas.BuyLimit.SoundOpen(SND_OK);
   this.m_datas.BuyLimit.SoundClose(SND_OK);
   this.m_datas.BuyLimit.SoundModifySL(SND_OK);
   this.m_datas.BuyLimit.SoundModifyTP(SND_OK);
   this.m_datas.BuyLimit.SoundModifyPrice(SND_OK);
   this.m_datas.BuyLimit.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.BuyLimit.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.BuyLimit.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.BuyLimit.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.BuyLimit.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.BuyStopLimit.UseSoundClose(true);
   this.m_datas.BuyStopLimit.UseSoundOpen(true);
   this.m_datas.BuyStopLimit.UseSoundModifySL(true);
   this.m_datas.BuyStopLimit.UseSoundModifyTP(true);
   this.m_datas.BuyStopLimit.UseSoundModifyPrice(true);
   this.m_datas.BuyStopLimit.SoundOpen(SND_OK);
   this.m_datas.BuyStopLimit.SoundClose(SND_OK);
   this.m_datas.BuyStopLimit.SoundModifySL(SND_OK);
   this.m_datas.BuyStopLimit.SoundModifyTP(SND_OK);
   this.m_datas.BuyStopLimit.SoundModifyPrice(SND_OK);
   this.m_datas.BuyStopLimit.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.BuyStopLimit.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.BuyStopLimit.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.BuyStopLimit.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.BuyStopLimit.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.Sell.UseSoundClose(true);
   this.m_datas.Sell.UseSoundOpen(true);
   this.m_datas.Sell.UseSoundModifySL(true);
   this.m_datas.Sell.UseSoundModifyTP(true);
   this.m_datas.Sell.UseSoundModifyPrice(true);
   this.m_datas.Sell.SoundOpen(SND_OK);
   this.m_datas.Sell.SoundClose(SND_OK);
   this.m_datas.Sell.SoundModifySL(SND_OK);
   this.m_datas.Sell.SoundModifyTP(SND_OK);
   this.m_datas.Sell.SoundModifyPrice(SND_OK);
   this.m_datas.Sell.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.Sell.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.Sell.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.Sell.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.Sell.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.SellStop.UseSoundClose(true);
   this.m_datas.SellStop.UseSoundOpen(true);
   this.m_datas.SellStop.UseSoundModifySL(true);
   this.m_datas.SellStop.UseSoundModifyTP(true);
   this.m_datas.SellStop.UseSoundModifyPrice(true);
   this.m_datas.SellStop.SoundOpen(SND_OK);
   this.m_datas.SellStop.SoundClose(SND_OK);
   this.m_datas.SellStop.SoundModifySL(SND_OK);
   this.m_datas.SellStop.SoundModifyTP(SND_OK);
   this.m_datas.SellStop.SoundModifyPrice(SND_OK);
   this.m_datas.SellStop.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.SellStop.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.SellStop.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.SellStop.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.SellStop.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.SellLimit.UseSoundClose(true);
   this.m_datas.SellLimit.UseSoundOpen(true);
   this.m_datas.SellLimit.UseSoundModifySL(true);
   this.m_datas.SellLimit.UseSoundModifyTP(true);
   this.m_datas.SellLimit.UseSoundModifyPrice(true);
   this.m_datas.SellLimit.SoundOpen(SND_OK);
   this.m_datas.SellLimit.SoundClose(SND_OK);
   this.m_datas.SellLimit.SoundModifySL(SND_OK);
   this.m_datas.SellLimit.SoundModifyTP(SND_OK);
   this.m_datas.SellLimit.SoundModifyPrice(SND_OK);
   this.m_datas.SellLimit.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.SellLimit.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.SellLimit.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.SellLimit.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.SellLimit.SoundErrorModifyPrice(SND_TIMEOUT);

   this.m_datas.SellStopLimit.UseSoundClose(true);
   this.m_datas.SellStopLimit.UseSoundOpen(true);
   this.m_datas.SellStopLimit.UseSoundModifySL(true);
   this.m_datas.SellStopLimit.UseSoundModifyTP(true);
   this.m_datas.SellStopLimit.UseSoundModifyPrice(true);
   this.m_datas.SellStopLimit.SoundOpen(SND_OK);
   this.m_datas.SellStopLimit.SoundClose(SND_OK);
   this.m_datas.SellStopLimit.SoundModifySL(SND_OK);
   this.m_datas.SellStopLimit.SoundModifyTP(SND_OK);
   this.m_datas.SellStopLimit.SoundModifyPrice(SND_OK);
   this.m_datas.SellStopLimit.SoundErrorClose(SND_TIMEOUT);
   this.m_datas.SellStopLimit.SoundErrorOpen(SND_TIMEOUT);
   this.m_datas.SellStopLimit.SoundErrorModifySL(SND_TIMEOUT);
   this.m_datas.SellStopLimit.SoundErrorModifyTP(SND_TIMEOUT);
   this.m_datas.SellStopLimit.SoundErrorModifyPrice(SND_TIMEOUT);
  }
//+------------------------------------------------------------------+
```

**Beyond the class body, implement the methods of playing sounds and setting flags and sounds for a certain trading event type:**

```
//+------------------------------------------------------------------+
//| Set the flag of using sounds                                     |
//| of opening/placing a specified position/order type               |
//+------------------------------------------------------------------+
void CTradeObj::UseSoundOpen(const ENUM_ORDER_TYPE action,const bool flag)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.UseSoundOpen(flag);           break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.UseSoundOpen(flag);       break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.UseSoundOpen(flag);      break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.UseSoundOpen(flag);          break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.UseSoundOpen(flag);      break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.UseSoundOpen(flag);     break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.UseSoundOpen(flag);  break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.UseSoundOpen(flag); break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the flag of using a sound                                    |
//| of closing/removal of a specified position/order type            |
//+------------------------------------------------------------------+
void CTradeObj::UseSoundClose(const ENUM_ORDER_TYPE action,const bool flag)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.UseSoundClose(flag);             break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.UseSoundClose(flag);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.UseSoundClose(flag);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.UseSoundClose(flag);    break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.UseSoundClose(flag);            break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.UseSoundClose(flag);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.UseSoundClose(flag);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.UseSoundClose(flag);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the flag of using a sound                                    |
//| of StopLoss modification for a specified position/order type     |
//+------------------------------------------------------------------+
void CTradeObj::UseSoundModifySL(const ENUM_ORDER_TYPE action,const bool flag)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.UseSoundModifySL(flag);             break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.UseSoundModifySL(flag);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.UseSoundModifySL(flag);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.UseSoundModifySL(flag);    break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.UseSoundModifySL(flag);            break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.UseSoundModifySL(flag);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.UseSoundModifySL(flag);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.UseSoundModifySL(flag);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the flag of using a sound                                    |
//| of TakeProfit modification for a specified position/order type   |
//+------------------------------------------------------------------+
void CTradeObj::UseSoundModifyTP(const ENUM_ORDER_TYPE action,const bool flag)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.UseSoundModifyTP(flag);             break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.UseSoundModifyTP(flag);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.UseSoundModifyTP(flag);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.UseSoundModifyTP(flag);    break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.UseSoundModifyTP(flag);            break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.UseSoundModifyTP(flag);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.UseSoundModifyTP(flag);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.UseSoundModifyTP(flag);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the flag of using a modification sound                       |
//| of the placement price for a specified order type                |
//+------------------------------------------------------------------+
void CTradeObj::UseSoundModifyPrice(const ENUM_ORDER_TYPE action,const bool flag)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.UseSoundModifyPrice(flag);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.UseSoundModifyPrice(flag);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.UseSoundModifyPrice(flag);    break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.UseSoundModifyPrice(flag);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.UseSoundModifyPrice(flag);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.UseSoundModifyPrice(flag);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of using the sound of opening/placing            |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
bool CTradeObj::UseSoundOpen(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.UseSoundOpen();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.UseSoundOpen();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.UseSoundOpen();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.UseSoundOpen();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.UseSoundOpen();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.UseSoundOpen();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.UseSoundOpen();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.UseSoundOpen();
      default: return false;
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of using the sound of closing/removal of         |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
bool CTradeObj::UseSoundClose(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.UseSoundClose();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.UseSoundClose();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.UseSoundClose();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.UseSoundClose();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.UseSoundClose();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.UseSoundClose();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.UseSoundClose();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.UseSoundClose();
      default: return false;
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of using a sound                                 |
//| of StopLoss modification for a specified position/order type     |
//+------------------------------------------------------------------+
bool CTradeObj::UseSoundModifySL(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.UseSoundModifySL();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.UseSoundModifySL();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.UseSoundModifySL();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.UseSoundModifySL();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.UseSoundModifySL();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.UseSoundModifySL();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.UseSoundModifySL();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.UseSoundModifySL();
      default: return false;
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of using a sound                                 |
//| of TakeProfit modification for a specified position/order type   |
//+------------------------------------------------------------------+
bool CTradeObj::UseSoundModifyTP(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.UseSoundModifyTP();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.UseSoundModifyTP();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.UseSoundModifyTP();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.UseSoundModifyTP();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.UseSoundModifyTP();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.UseSoundModifyTP();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.UseSoundModifyTP();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.UseSoundModifyTP();
      default: return false;
     }
  }
//+------------------------------------------------------------------+
//| Return the flag of using a modification sound                    |
//| of the placement price for a specified order type                |
//+------------------------------------------------------------------+
bool CTradeObj::UseSoundModifyPrice(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.UseSoundModifyPrice();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.UseSoundModifyPrice();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.UseSoundModifyPrice();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.UseSoundModifyPrice();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.UseSoundModifyPrice();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.UseSoundModifyPrice();
      default: return false;
     }
  }
//+------------------------------------------------------------------+
//| Set the sound of opening/placing                                 |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundOpen(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundOpen(sound);             break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundOpen(sound);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundOpen(sound);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundOpen(sound);    break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundOpen(sound);            break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundOpen(sound);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundOpen(sound);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundOpen(sound);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the sound of closing/removal of                              |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundClose(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundClose(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundClose(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundClose(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundClose(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundClose(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundClose(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundClose(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundClose(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set StopLoss modification sound of                               |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundModifySL(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundModifySL(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundModifySL(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundModifySL(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundModifySL(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundModifySL(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundModifySL(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundModifySL(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundModifySL(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set TakeProfit modification sound of                             |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundModifyTP(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundModifyTP(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundModifyTP(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundModifyTP(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundModifyTP(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundModifyTP(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundModifyTP(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundModifyTP(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundModifyTP(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set price modification sound                                     |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundModifyPrice(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundModifyPrice(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundModifyPrice(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundModifyPrice(sound);   break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundModifyPrice(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundModifyPrice(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundModifyPrice(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the error sound of opening/placing                           |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundErrorOpen(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundErrorOpen(sound);             break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundErrorOpen(sound);         break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundErrorOpen(sound);        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundErrorOpen(sound);    break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundErrorOpen(sound);            break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundErrorOpen(sound);        break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundErrorOpen(sound);       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundErrorOpen(sound);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set the error sound of closing/removal of                        |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundErrorClose(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundErrorClose(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundErrorClose(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundErrorClose(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundErrorClose(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundErrorClose(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundErrorClose(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundErrorClose(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundErrorClose(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set StopLoss modification error sound of                         |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundErrorModifySL(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundErrorModifySL(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundErrorModifySL(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundErrorModifySL(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundErrorModifySL(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundErrorModifySL(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundErrorModifySL(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundErrorModifySL(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundErrorModifySL(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set TakeProfit modification error sound of                       |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundErrorModifyTP(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   this.m_datas.Buy.SoundErrorModifyTP(sound);            break;
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundErrorModifyTP(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundErrorModifyTP(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundErrorModifyTP(sound);   break;
      case ORDER_TYPE_SELL             :   this.m_datas.Sell.SoundErrorModifyTP(sound);           break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundErrorModifyTP(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundErrorModifyTP(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundErrorModifyTP(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Set price modification error sound                               |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::SetSoundErrorModifyPrice(const ENUM_ORDER_TYPE action,const string sound)
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   this.m_datas.BuyStop.SoundErrorModifyPrice(sound);        break;
      case ORDER_TYPE_BUY_LIMIT        :   this.m_datas.BuyLimit.SoundErrorModifyPrice(sound);       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   this.m_datas.BuyStopLimit.SoundErrorModifyPrice(sound);   break;
      case ORDER_TYPE_SELL_STOP        :   this.m_datas.SellStop.SoundErrorModifyPrice(sound);       break;
      case ORDER_TYPE_SELL_LIMIT       :   this.m_datas.SellLimit.SoundErrorModifyPrice(sound);      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   this.m_datas.SellStopLimit.SoundErrorModifyPrice(sound);  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Return the sound of opening/placing                              |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundOpen(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundOpen();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundOpen();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundOpen();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundOpen();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundOpen();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundOpen();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundOpen();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundOpen();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return the sound of closing/removal of                           |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundClose(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundClose();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundClose();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundClose();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundClose();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundClose();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundClose();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundClose();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundClose();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return StopLoss modification sound of                            |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundModifySL(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundModifySL();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundModifySL();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundModifySL();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundModifySL();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundModifySL();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundModifySL();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundModifySL();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundModifySL();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return TakeProfit modification sound of                          |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundModifyTP(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundModifyTP();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundModifyTP();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundModifyTP();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundModifyTP();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundModifyTP();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundModifyTP();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundModifyTP();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundModifyTP();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return price modification sound                                  |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundModifyPrice(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundModifyPrice();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundModifyPrice();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundModifyPrice();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundModifyPrice();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundModifyPrice();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundModifyPrice();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return the error sound of opening/placing                        |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundErrorOpen(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundErrorOpen();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundErrorOpen();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundErrorOpen();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundErrorOpen();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundErrorOpen();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundErrorOpen();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundErrorOpen();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundErrorOpen();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return the error sound of closing/removal of                     |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundErrorClose(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundErrorClose();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundErrorClose();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundErrorClose();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundErrorClose();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundErrorClose();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundErrorClose();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundErrorClose();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundErrorClose();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return StopLoss modification error sound of                      |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundErrorModifySL(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundErrorModifySL();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundErrorModifySL();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundErrorModifySL();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundErrorModifySL();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundErrorModifySL();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundErrorModifySL();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundErrorModifySL();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundErrorModifySL();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return TakeProfit modification error sound of                    |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundErrorModifyTP(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   return this.m_datas.Buy.SoundErrorModifyTP();
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundErrorModifyTP();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundErrorModifyTP();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundErrorModifyTP();
      case ORDER_TYPE_SELL             :   return this.m_datas.Sell.SoundErrorModifyTP();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundErrorModifyTP();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundErrorModifyTP();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundErrorModifyTP();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Return price modification error sound                            |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
string CTradeObj::GetSoundErrorModifyPrice(const ENUM_ORDER_TYPE action) const
  {
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   return this.m_datas.BuyStop.SoundErrorModifyPrice();
      case ORDER_TYPE_BUY_LIMIT        :   return this.m_datas.BuyLimit.SoundErrorModifyPrice();
      case ORDER_TYPE_BUY_STOP_LIMIT   :   return this.m_datas.BuyStopLimit.SoundErrorModifyPrice();
      case ORDER_TYPE_SELL_STOP        :   return this.m_datas.SellStop.SoundErrorModifyPrice();
      case ORDER_TYPE_SELL_LIMIT       :   return this.m_datas.SellLimit.SoundErrorModifyPrice();
      case ORDER_TYPE_SELL_STOP_LIMIT  :   return this.m_datas.SellStopLimit.SoundErrorModifyPrice();
      default: return NULL;
     }
  }
//+------------------------------------------------------------------+
//| Play the sound of opening/placing                                |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundOpen(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundOpen());            break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundOpen());        break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundOpen());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundOpen());   break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundOpen());           break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundOpen());       break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundOpen());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundOpen());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the sound of closing/removal of                             |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundClose(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundClose());           break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundClose());       break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundClose());      break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundClose());  break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundClose());          break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundClose());      break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundClose());     break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundClose()); break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play StopLoss modification sound of                              |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifySL(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundModifySL());           break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundModifySL());       break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifySL());      break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifySL());  break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundModifySL());          break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundModifySL());      break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundModifySL());     break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifySL()); break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play TakeProfit modification sound of                            |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifyTP(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundModifyTP());           break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundModifyTP());       break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifyTP());      break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifyTP());  break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundModifyTP());          break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundModifyTP());      break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundModifyTP());     break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifyTP()); break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play price modification sound                                    |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundModifyPrice(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundModifyPrice());       break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundModifyPrice());      break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundModifyPrice());  break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundModifyPrice());      break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundModifyPrice());     break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundModifyPrice()); break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the error sound of opening/placing                          |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorOpen(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundErrorOpen());             break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorOpen());         break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorOpen());        break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorOpen());    break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundErrorOpen());            break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorOpen());        break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorOpen());       break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorOpen());   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play the error sound of closing/removal of                       |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorClose(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundErrorClose());            break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorClose());        break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorClose());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorClose());   break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundErrorClose());           break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorClose());       break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorClose());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorClose());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play StopLoss modification error sound of                        |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifySL(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundErrorModifySL());            break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifySL());        break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifySL());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifySL());   break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundErrorModifySL());           break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifySL());       break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifySL());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifySL());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play TakeProfit modification error sound of                      |
//| a specified position/order type                                  |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifyTP(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY              :   CMessage::PlaySound(this.m_datas.Buy.SoundErrorModifyTP());            break;
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifyTP());        break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifyTP());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifyTP());   break;
      case ORDER_TYPE_SELL             :   CMessage::PlaySound(this.m_datas.Sell.SoundErrorModifyTP());           break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifyTP());       break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifyTP());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifyTP());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
//| Play price modification error sound                              |
//| for a specified order type                                       |
//+------------------------------------------------------------------+
void CTradeObj::PlaySoundErrorModifyPrice(const ENUM_ORDER_TYPE action)
  {
   if(!this.m_use_sound)
      return;
   int index=action;
   switch(index)
     {
      case ORDER_TYPE_BUY_STOP         :   CMessage::PlaySound(this.m_datas.BuyStop.SoundErrorModifyPrice());        break;
      case ORDER_TYPE_BUY_LIMIT        :   CMessage::PlaySound(this.m_datas.BuyLimit.SoundErrorModifyPrice());       break;
      case ORDER_TYPE_BUY_STOP_LIMIT   :   CMessage::PlaySound(this.m_datas.BuyStopLimit.SoundErrorModifyPrice());   break;
      case ORDER_TYPE_SELL_STOP        :   CMessage::PlaySound(this.m_datas.SellStop.SoundErrorModifyPrice());       break;
      case ORDER_TYPE_SELL_LIMIT       :   CMessage::PlaySound(this.m_datas.SellLimit.SoundErrorModifyPrice());      break;
      case ORDER_TYPE_SELL_STOP_LIMIT  :   CMessage::PlaySound(this.m_datas.SellStopLimit.SoundErrorModifyPrice());  break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
```

Here all is similar to all methods, and everything is clear at first glance: the method whose name contains the type of a trading event receives a
type of an order (market order type for a position) whose event should be set, returned or played. Next, a sound of a specified trading event is
set/returned/played depending on the position/order type.

**This concludes the improvement of a symbol base trading object. It is time to develop the base trading class.**

For its correct operation, the trading class is to receive the pointers to the previously created collections. To do this, we will add the
method returning the pointer to the collection object to each collection class:

For the historical collection object:

```
public:
//--- Return itself
   CHistoryCollection *GetObject(void)                                                                   { return &this;   }
```

For the market collection object:

```
public:
//--- Return itself
   CMarketCollection *GetObject(void)                                                                    { return &this;   }
```

For the symbol collection:

```
public:
//--- Return itself
   CSymbolsCollection *GetObject(void)                                                                   { return &this;   }
```

The currently developed trading class is to provide centralized access to trading operations. It is to accumulate all verifications before
sending a trading order, handling trade server responses and making decisions on further actions depending on the trade server response
and the settings of the class trading method behavior.

In this article, we will create the basis of the class to be developed further. But first, let's check the trading limitations from the
program, terminal, account and symbol sides.

### Trading class

In \\MQL5\\Include\ **DoEasy\**, create the new class **CTrading** in the **Trading.mqh** file.

To make the
class work, we will need the lists of

some preliminarily created collections, while to store
errors when checking permissions, we will need [the \\
object](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint)[of \\
the dynamic array containing int or uint variables](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint) of the standard library.

**So let's include them to the**
**class file right away:**

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
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
```

**Let's have a look at the full listing of the class body and analyze its composition:**

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
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading
  {
private:
   CAccount            *m_account;           // Pointer to the current account object
   CSymbolsCollection  *m_symbols;           // Pointer to the symbol collection list
   CMarketCollection   *m_market;            // Pointer to the list of the collection of market orders and positions
   CHistoryCollection  *m_history;           // Pointer to the list of the collection of historical orders and deals
   CArrayInt            m_list_errors;       // Error list
   bool                 m_is_trade_enable;   // Flag enabling trading
   ENUM_LOG_LEVEL       m_log_level;         // Logging level

//--- Add the error code to the list
   bool                 AddErrorCodeToList(const int error_code);
//--- Return the symbol object by (1) position, (2) order ticket
   CSymbol             *GetSymbolObjByPosition(const ulong ticket,const string source_method);
   CSymbol             *GetSymbolObjByOrder(const ulong ticket,const string source_method);
//--- Return a symbol trading object by (1) position, (2) order ticket, (3) symbol name
   CTradeObj           *GetTradeObjByPosition(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjByOrder(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjBySymbol(const string symbol,const string source_method);
//--- Return the number of (1) all positions, (2) buy, (3) sell positions
   int                  PositionsTotalAll(void)          const;
   int                  PositionsTotalLong(void)         const;
   int                  PositionsTotalShort(void)        const;
//--- Return the number of (1) all pending orders, (2) buy, (3) sell pending orders
   int                  OrdersTotalAll(void)             const;
   int                  OrdersTotalLong(void)            const;
   int                  OrdersTotalShort(void)           const;
//--- Return the total volume of (1) buy, (2) sell positions
   double               PositionsTotalVolumeLong(void)   const;
   double               PositionsTotalVolumeShort(void)  const;
//--- Return the total volume of (1) buy, (2) sell orders
   double               OrdersTotalVolumeLong(void)      const;
   double               OrdersTotalVolumeShort(void)     const;
//--- Return the order direction by an operation type
   ENUM_ORDER_TYPE      DirectionByActionType(ENUM_ACTION_TYPE action) const;
//--- Check the presence of a (1) position, (2) order by ticket
   bool                 CheckPositionAvailablity(const ulong ticket,const string source_method);
   bool                 CheckOrderAvailablity(const ulong ticket,const string source_method);
//--- Set the desired sound for a trading object
   void                 SetSoundByMode(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,CTradeObj *trade_obj);
public:
//--- Constructor
                        CTrading();
//--- Get the pointers to the lists (make sure to call the method in program's OnInit() since the symbol collection list is created there)
   void                 OnInit(CAccount *account,CSymbolsCollection *symbols,CMarketCollection *market,CHistoryCollection *history)
                          {
                           this.m_account=account;
                           this.m_symbols=symbols;
                           this.m_market=market;
                           this.m_history=history;
                          }
//--- Return the error list
   CArrayInt           *GetListErrors(void)                                { return &this.m_list_errors; }
//--- Check trading limitations
   bool                 CheckTradeConstraints(const double volume,
                                              const ENUM_ACTION_TYPE action_type,
                                              const CSymbol *symbol_obj,
                                              const string source_method,
                                              double sl=0,
                                              double tp=0);
//--- Check if the funds are sufficient
   bool                 CheckMoneyFree(const ENUM_ORDER_TYPE order_type,const double volume,const double price,const CSymbol *symbol_obj,const string source_method) const;

//--- Set the following for symbol trading objects:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level
   void                 SetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol=NULL);
   void                 SetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol=NULL);
   void                 SetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol=NULL);
   void                 SetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol=NULL);
   void                 SetMagic(const ulong magic,const string symbol=NULL);
   void                 SetComment(const string comment,const string symbol=NULL);
   void                 SetDeviation(const ulong deviation,const string symbol=NULL);
   void                 SetVolume(const double volume=0,const string symbol=NULL);
   void                 SetExpiration(const datetime expiration=0,const string symbol=NULL);
   void                 SetAsyncMode(const bool mode=false,const string symbol=NULL);
   void                 SetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol=NULL);

//--- Set standard sounds (1 symbol=NULL) for trading objects of all symbols, (2 symbol!=NULL) for a symbol trading object
   void                 SetSoundsStandart(const string symbol=NULL);
//--- Set a sound for a specified order/position type and symbol
//--- 'mode' specifies an event a sound is set for
//--- (symbol=NULL) for trading objects of all symbols,
//--- (symbol!=NULL) for a trading object of a specified symbol
   void                 SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL);

//--- Open (1) Buy, (2) Sell position
   bool                 OpenBuy(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL);
   bool                 OpenSell(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL);
//--- Modify a position
   bool                 ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE);
//--- Close a position (1) fully, (2) partially, (3) by an opposite one
   bool                 ClosePosition(const ulong ticket);
   bool                 ClosePositionPartially(const ulong ticket,const double volume);
   bool                 ClosePositionBy(const ulong ticket,const ulong ticket_by);
//--- Set (1) BuyStop, (2) BuyLimit, (3) BuyStopLimit pending order
   bool                 PlaceBuyStop(const double volume,
                                     const string symbol,
                                     const double price,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   bool                 PlaceBuyLimit(const double volume,
                                     const string symbol,
                                     const double price,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   bool                 PlaceBuyStopLimit(const double volume,
                                     const string symbol,
                                     const double price_stop,
                                     const double price_limit,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
//--- Set (1) SellStop, (2) SellLimit, (3) SellStopLimit pending order
   bool                 PlaceSellStop(const double volume,
                                     const string symbol,
                                     const double price,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   bool                 PlaceSellLimit(const double volume,
                                     const string symbol,
                                     const double price,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
   bool                 PlaceSellStopLimit(const double volume,
                                     const string symbol,
                                     const double price_stop,
                                     const double price_limit,
                                     const double sl=0,
                                     const double tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC);
//--- Modify a pending order
   bool                 ModifyOrder(const ulong ticket,
                                    const double price=WRONG_VALUE,
                                    const double sl=WRONG_VALUE,
                                    const double tp=WRONG_VALUE,
                                    const double stoplimit=WRONG_VALUE,
                                    datetime expiration=WRONG_VALUE,
                                    ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE);
//--- Remove a pending order
   bool                 DeleteOrder(const ulong ticket);
  };
//+------------------------------------------------------------------+
```

Each of the class methods features an explanatory comment, so I believe, the structure of the class methods requires no additional
clarifications.

The class trading methods have almost the same signature as the methods of the symbol base trading object. The only differences are the
presence of a name of a symbol, on which a trading operation should be performed, in the position opening methods and in placing a pending
order.

**Let's consider the implementation of the class methods.**

Since there may be several limitations at once, the code of each error should be placed to the list and the list of detected limitations should be
displayed after checking all the necessary conditions. In order not to place the limitation value (detected in the various fragments of the
verification code) to the list twice, check the presence of the appropriate limitation in the list and place it only if there is none. In this
case, each limitation type is placed to the list once.

**Use the following method to check the presence of the limitation type in the list and add it to the list:**

```
//+------------------------------------------------------------------+
//| Add the error code to the list                                   |
//+------------------------------------------------------------------+
bool CTrading::AddErrorCodeToList(const int error_code)
  {
   this.m_list_errors.Sort();
   if(this.m_list_errors.Search(error_code)==WRONG_VALUE)
      return this.m_list_errors.Add(error_code);
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the error code, identifying the trading
limitation reason.

The sorted list flag is set for the flag, the
absence of such a code in the list is checked. If the check is successful,
the result of adding the code to the list using the

[Add() method](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint/carrayintadd) is
returned.

If the code is already present in the list, return false.

Since trading objects belong to a symbol (each symbol has its trading object), while many trading methods work with an order/position ticket,
we need to define a symbol, where such a position or an order is present, by a ticket.

**The methods returning a symbol object by a position ticket solve this task:**

```
//+------------------------------------------------------------------+
//| Return a symbol object by a position ticket                      |
//+------------------------------------------------------------------+
CSymbol *CTrading::GetSymbolObjByPosition(const ulong ticket,const string source_method)
  {
   //--- Get the list of open positions
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   //--- If failed to get the list of open positions, display the message and return NULL
   if(list==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_MARKET_POS_LIST));
      return NULL;
     }
   //--- If the list is empty (no open positions), display the message and return NULL
   if(list.Total()==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(MSG_ENG_NO_OPEN_POSITIONS));
      return NULL;
     }
   //--- Sort the list by a ticket
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   //--- If failed to get the list of open positions, display the message and return NULL
   if(list==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_MARKET_POS_LIST));
      return NULL;
     }
   //--- If the list is empty (no required ticket), display the message and return NULL
   if(list.Total()==0)
     {
      //--- Error. No open position with #ticket
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(MSG_LIB_SYS_ERROR_NO_OPEN_POSITION_WITH_TICKET),(string)ticket);
      return NULL;
     }
   //--- Get a position with #ticket from the obtained list
   COrder *pos=list.At(0);
   //--- If failed to get the position object, display the message and return NULL
   if(pos==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_POS_OBJ));
      return NULL;
     }
   //--- Get a symbol object by name
   CSymbol * symbol_obj=this.m_symbols.GetSymbolObjByName(pos.Symbol());
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Return a symbol object
   return symbol_obj;
  }
//+------------------------------------------------------------------+
```

**by an order ticket:**

```
//+------------------------------------------------------------------+
//| Return a symbol object by an order ticket                        |
//+------------------------------------------------------------------+
CSymbol *CTrading::GetSymbolObjByOrder(const ulong ticket,const string source_method)
  {
   //--- Get the list of placed orders
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   //--- If failed to get the list of placed orders, display the message and return NULL
   if(list==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_PENDING_ORD_LIST));
      return NULL;
     }
   //--- If the list is empty (no placed orders), display the message and return NULL
   if(list.Total()==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(MSG_ENG_NO_PLACED_ORDERS));
      return NULL;
     }
   //--- Sort the list by a ticket
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   //--- If failed to get the list of placed orders, display the message and return NULL
   if(list==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_PENDING_ORD_LIST));
      return NULL;
     }
   //--- If the list is empty (no required ticket), display the message and return NULL
   if(list.Total()==0)
     {
      //--- Error. No placed order with #ticket
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(source_method,CMessage::Text(MSG_LIB_SYS_ERROR_NO_PLACED_ORDER_WITH_TICKET),(string)ticket);
      return NULL;
     }
   //--- Get an order with #ticket from the obtained list
   COrder *ord=list.At(0);
   //--- If failed to get an object order, display the message and return NULL
   if(ord==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return NULL;
     }
   //--- Get a symbol object by name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(ord.Symbol());
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Return a symbol object
   return symbol_obj;
  }
//+------------------------------------------------------------------+
```

These two methods are almost identical. The only difference is that we search for a position by a ticket in the first one and we look for an order by a
ticket in the second one. The method receives a ticket of a desired position or order and the name of a method that method was called from. This
will allow us to see the method name in the journal in case of any error. The name of the auxiliary method provides no data for finding where the
error occurred (we need to specify the method the search for a symbol object was called from).

**For the class methods operation, we sometimes need to receive a symbol**
**trading object by a position ticket, an order ticket or a**
**symbol name:**

```
//+------------------------------------------------------------------+
//| Return a symbol trading object by a position ticket              |
//+------------------------------------------------------------------+
CTradeObj *CTrading::GetTradeObjByPosition(const ulong ticket,const string source_method)
  {
   //--- Get a symbol object by a position ticket
   CSymbol * symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Get and return the trading object from the symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   return trade_obj;
  }
//+------------------------------------------------------------------+
//| Return a symbol trading object by an order ticket                |
//+------------------------------------------------------------------+
CTradeObj *CTrading::GetTradeObjByOrder(const ulong ticket,const string source_method)
  {
   //--- Get a symbol object by an order ticket
   CSymbol * symbol_obj=this.GetSymbolObjByOrder(ticket,source_method);
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Get and return the trading object from the symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   return trade_obj;
  }
//+------------------------------------------------------------------+
//| Return a symbol trading object by a symbol name                  |
//+------------------------------------------------------------------+
CTradeObj *CTrading::GetTradeObjBySymbol(const string symbol,const string source_method)
  {
   //--- Get a symbol object by its name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return NULL;
     }
   //--- Get and return the trading object from the symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   return trade_obj;
  }
//+------------------------------------------------------------------+
```

All three auxiliary methods return a trading object for further work. All actions are commented in the method listings.

We need to know the number of open positions and orders placed in both directions to check some limitations and make trading decisions.

**The**
**method returning the number of open positions on the account:**

```
//+------------------------------------------------------------------+
//| Return the number of positions                                   |
//+------------------------------------------------------------------+
int CTrading::PositionsTotalAll(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
```

Here we receive the list of all open positions
and placed orders and sort the list by
"market position" order status.

Return the number of
objects in the obtained list. If failed to obtain the list, return zero.

**The method returning the number of open buy positions on the account:**

```
//+------------------------------------------------------------------+
//| Return the number of buy positions                               |
//+------------------------------------------------------------------+
int CTrading::PositionsTotalLong(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
```

Here we receive the list of all open positions
and placed orders and sort
the list by "market position" order status.

The obtained
list is sorted by buy position type.

Return the number of objects in the
obtained list. If failed to obtain the list, return zero.

**The method returning the number of open sell positions on the account:**

```
//+------------------------------------------------------------------+
//| Return the number of sell positions                              |
//+------------------------------------------------------------------+
int CTrading::PositionsTotalShort(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
```

Here we receive the list of all open positions
and placed orders and sort
the list by "market position" order status.

The obtained
list is sorted by sell position type.

Return the number of objects in the
obtained list. If failed to obtain the list, return zero.

**The method of receiving the number of all pending orders on an account, buy and sell orders:**

```
//+------------------------------------------------------------------+
//| Returns the number of pending orders                             |
//+------------------------------------------------------------------+
int CTrading::OrdersTotalAll(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
//| Return the number of buy pending orders                          |
//+------------------------------------------------------------------+
int CTrading::OrdersTotalLong(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_BUY,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
//| Return the number of sell pending orders                         |
//+------------------------------------------------------------------+
int CTrading::OrdersTotalShort(void) const
  {
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_SELL,EQUAL);
   return(list==NULL ? 0 : list.Total());
  }
//+------------------------------------------------------------------+
```

These three auxiliary methods are identical to the three methods discussed above. The only difference is that here we
sort the list by "active pending order" status and by buy and sell order
type direction.

Also, in order to define some limitations, we need to know the total volume of open positions and placed pending orders.

**The method returning the total volume of buy positions:**

```
//+------------------------------------------------------------------+
//| Return the total volume of buy positions                         |
//+------------------------------------------------------------------+
double CTrading::PositionsTotalVolumeLong(void) const
  {
   double vol=0;
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
   if(list==NULL || list.Total()==0)
      return 0;
   for(int i=0;i<list.Total();i++)
     {
      COrder *obj=list.At(i);
      if(obj==NULL)
         continue;
      vol+=obj.Volume();
     }
   return vol;
  }
//+------------------------------------------------------------------+
```

Here we receive the list of all open positions and placed orders,

as well as sort the list
by "market position" order status.

The
obtained list is sorted by buy position type.

If the list is
empty, return zero.

In
a loop by the obtained list, get the next order object and add
its volume to the

**vol** variable value.

Upon the loop completion, return the
obtained value in the

**vol** variable.

**The method returning the total volume of sell positions:**

```
//+------------------------------------------------------------------+
//| Return the total volume of sell positions                        |
//+------------------------------------------------------------------+
double CTrading::PositionsTotalVolumeShort(void) const
  {
   double vol=0;
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
   if(list==NULL || list.Total()==0)
      return 0;
   for(int i=0;i<list.Total();i++)
     {
      COrder *obj=list.At(i);
      if(obj==NULL)
         continue;
      vol+=obj.Volume();
     }
   return vol;
  }
//+------------------------------------------------------------------+
```

The method is similar to the previous one except for sorting by sell positions.

**The methods returning the total volume of placed pending buy**
**and sell orders:**

```
//+------------------------------------------------------------------+
//| Return the total volume of buy orders                            |
//+------------------------------------------------------------------+
double CTrading::OrdersTotalVolumeLong(void) const
  {
   double vol=0;
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_BUY,EQUAL);
   if(list==NULL || list.Total()==0)
      return 0;
   for(int i=0;i<list.Total();i++)
     {
      COrder *obj=list.At(i);
      if(obj==NULL)
         continue;
      vol+=obj.Volume();
     }
   return vol;
  }
//+------------------------------------------------------------------+
//| Return the total volume of sell orders                           |
//+------------------------------------------------------------------+
double CTrading::OrdersTotalVolumeShort(void) const
  {
   double vol=0;
   CArrayObj *list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_DIRECTION,ORDER_TYPE_SELL,EQUAL);
   if(list==NULL || list.Total()==0)
      return 0;
   for(int i=0;i<list.Total();i++)
     {
      COrder *obj=list.At(i);
      if(obj==NULL)
         continue;
      vol+=obj.Volume();
     }
   return vol;
  }
//+------------------------------------------------------------------+
```

The methods are identical to the total position volume methods. Although, the
lists are sorted by "market pending order" type and by an order
direction (buy or sell).

**The method returning the order direction by a** **trading operation type:**

```
//+------------------------------------------------------------------+
//| Return the order direction by an operation type                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE CTrading::DirectionByActionType(ENUM_ACTION_TYPE action) const
  {
   if(action>ACTION_TYPE_SELL_STOP_LIMIT)
      return WRONG_VALUE;
   return ENUM_ORDER_TYPE(action%2);
  }
//+------------------------------------------------------------------+
```

If the method receives the "close by" or "modification" type, return -1. Otherwise, return the excess of dividing a trade operation type by
2, which eventually results in 0 (ORDER\_TYPE\_BUY) or 1 (ORDER\_TYPE\_SELL).

**The method of checking a position existence by ticket:**

```
//+------------------------------------------------------------------+
//| Check if a position is present by ticket                         |
//+------------------------------------------------------------------+
bool CTrading::CheckPositionAvailablity(const ulong ticket,const string source_method)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_POSITION,EQUAL);
   list.Sort(SORT_BY_ORDER_TICKET);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   if(list.Total()==1)
      return true;
   if(this.m_log_level>LOG_LEVEL_NO_MSG)
      ::Print(source_method,CMessage::Text(MSG_LIB_SYS_ERROR_NO_OPEN_POSITION_WITH_TICKET),(string)ticket);
   return false;
  }
//+------------------------------------------------------------------+
```

Here, the method receives a ticked of a checked position and a source name.

Get
the list of all open positions and placed orders, as
well as sort the list by "market position" order status.


Set the flag of sorting by ticket for the list.

The obtained list is sorted by a ticket passed by the input.

If the list contains a single object (a position with a desired ticket), return

true.

Otherwise,
inform of the position absence and return

false.

**The method of checking a pending order existence by ticket:**

```
//+------------------------------------------------------------------+
//| Check the presence of an order by ticket                         |
//+------------------------------------------------------------------+
bool CTrading::CheckOrderAvailablity(const ulong ticket,const string source_method)
  {
   CArrayObj* list=this.m_market.GetList();
   list=CSelect::ByOrderProperty(list,ORDER_PROP_STATUS,ORDER_STATUS_MARKET_PENDING,EQUAL);
   list.Sort(SORT_BY_ORDER_TICKET);
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   if(list.Total()==1)
      return true;
   if(this.m_log_level>LOG_LEVEL_NO_MSG)
      ::Print(source_method,CMessage::Text(MSG_LIB_SYS_ERROR_NO_PLACED_ORDER_WITH_TICKET),(string)ticket);
   return false;
  }
//+------------------------------------------------------------------+
```

The method is identical to the previous one with the exception that the
list is sorted by "market pending order" status.

**The method assigning a specified sound to a trading object:**

```
//+------------------------------------------------------------------+
//| Set a necessary sound to a trading object                        |
//+------------------------------------------------------------------+
void CTrading::SetSoundByMode(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,CTradeObj *trade_obj)
  {
   switch(mode)
     {
      case MODE_SET_SOUND_OPEN               : trade_obj.SetSoundOpen(action,sound);               break;
      case MODE_SET_SOUND_CLOSE              : trade_obj.SetSoundClose(action,sound);              break;
      case MODE_SET_SOUND_MODIFY_SL          : trade_obj.SetSoundModifySL(action,sound);           break;
      case MODE_SET_SOUND_MODIFY_TP          : trade_obj.SetSoundModifyTP(action,sound);           break;
      case MODE_SET_SOUND_MODIFY_PRICE       : trade_obj.SetSoundModifyPrice(action,sound);        break;
      case MODE_SET_SOUND_ERROR_OPEN         : trade_obj.SetSoundErrorOpen(action,sound);          break;
      case MODE_SET_SOUND_ERROR_CLOSE        : trade_obj.SetSoundErrorClose(action,sound);         break;
      case MODE_SET_SOUND_ERROR_MODIFY_SL    : trade_obj.SetSoundErrorModifySL(action,sound);      break;
      case MODE_SET_SOUND_ERROR_MODIFY_TP    : trade_obj.SetSoundErrorModifyTP(action,sound);      break;
      case MODE_SET_SOUND_ERROR_MODIFY_PRICE : trade_obj.SetSoundErrorModifyPrice(action,sound);   break;
      default: break;
     }
  }
//+------------------------------------------------------------------+
```

The method receives the sound setting mode (each trading
event is to have its sound),

order type of a trading event a sound is set for, audio
file name (sound to be assigned to an order or position trading event) and the
pointer to a trading object the sound should be placed to (a target symbol custom trading event sounds are set for).

Depending
on the mode passed to the method, the appropriate methods of placing a trading object sound are set.

**The methods of placing symbol trading object parameters** we considered [in \\
the previous article](https://www.mql5.com/en/articles/7229) and the ones temporarily implemented in the library CEngine base object class:

```
   void                 TradingSetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetMagic(const ulong magic,const string symbol_name=NULL);
   void                 TradingSetComment(const string comment,const string symbol_name=NULL);
   void                 TradingSetDeviation(const ulong deviation,const string symbol_name=NULL);
   void                 TradingSetVolume(const double volume=0,const string symbol_name=NULL);
   void                 TradingSetExpiration(const datetime expiration=0,const string symbol_name=NULL);
   void                 TradingSetAsyncMode(const bool mode=false,const string symbol_name=NULL);
   void                 TradingSetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL);
```

have been moved to the trading class and revised in the CEngine class (they simply call the methods located here). We have already considered
these methods before, and there is no point in dwelling on them here. Find their implementation in the files attached below.

**The method setting standard trading object sounds:**

```
//+------------------------------------------------------------------+
//| Set standard sounds for a symbol trading object                  |
//+------------------------------------------------------------------+
void CTrading::SetSoundsStandart(const string symbol=NULL)
  {
   //--- Declare an empty symbol object
   CSymbol *symbol_obj=NULL;
   //--- If NULL is passed as a symbol name, set sounds for trading objects of all symbols
   if(symbol==NULL)
     {
      //--- Get the symbol list
      CArrayObj *list=this.m_symbols.GetList();
      if(list==NULL || list.Total()==0)
         return;
      //--- In a loop by the list of symbols
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         //--- get the next symbol object
         symbol_obj=list.At(i);
         if(symbol_obj==NULL)
            continue;
         //--- get a symbol trading object
         CTradeObj *trade_obj=symbol_obj.GetTradeObj();
         if(trade_obj==NULL)
            continue;
         //--- set standard sounds for a trading object
         trade_obj.SetSoundsStandart();
        }
     }
   //--- If a symbol name is passed
   else
     {
      //--- get a symbol trading object
      CTradeObj *trade_obj=this.GetTradeObjBySymbol(symbol,DFUN);
      if(trade_obj==NULL)
         return;
      //--- set standard sounds for a trading object
      trade_obj.SetSoundsStandart();
     }
  }
//+------------------------------------------------------------------+
```

**The method setting the sound of a specified trading event for a specified order:**

```
//+------------------------------------------------------------------+
//| Set a sound for a specified order/position type and symbol       |
//| 'mode' specifies an event a sound is set for                     |
//| (symbol=NULL) for trading objects of all symbols,                |
//| (symbol!=NULL) for a trading object of a specified symbol        |
//+------------------------------------------------------------------+
void CTrading::SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL)
  {
   //--- Declare an empty symbol object
   CSymbol *symbol_obj=NULL;
   //--- If NULL is passed as a symbol name, set sounds for trading objects of all symbols
   if(symbol==NULL)
     {
      //--- Get the symbol list
      CArrayObj *list=this.m_symbols.GetList();
      if(list==NULL || list.Total()==0)
         return;
      //--- In a loop by the list of symbols
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         //--- get the next symbol object
         symbol_obj=list.At(i);
         if(symbol_obj==NULL)
            continue;
         //--- get a symbol trading object
         CTradeObj *trade_obj=symbol_obj.GetTradeObj();
         if(trade_obj==NULL)
            continue;
         //--- set a sound of a necessary event for a trading object
         this.SetSoundByMode(mode,action,sound,trade_obj);
        }
     }
   //--- If a symbol name is passed
   else
     {
      //--- get a symbol trading object
      CTradeObj *trade_obj=this.GetTradeObjBySymbol(symbol,DFUN);
      if(trade_obj==NULL)
         return;
      //--- set a sound of a necessary event for a trading object
      this.SetSoundByMode(mode,action,sound,trade_obj);
     }
  }
//+------------------------------------------------------------------+
```

All actions in both methods are described in the code comments.

In this article, we are going to create the methods of initial verification of trading limitations. What can we assign to this category?
This is a check of the funds sufficiency for conducting a trading operation. If the funds are insufficient, there is no point in sending a
request. This type of limitation requires time for elimination. We need to wait for the increase of free funds in a natural way — a planned
position closure according to the trading strategy and getting a profit (loss) with the release of the margin required to support an
open position or forcibly close a loss-making position to return the margin only, or depositing to an account. For now, it is much easier
to return the flag of of the impossibility of fulfilling a trading request here.



Besides, there are some other trading limitations — disabling auto trading in the terminal, disabling auto trading for an EA in its
settings, disabling auto trading from the server side or for a specific symbol, as well as some other additional factors.



This is all we are going to implement for the trading class in this article.

**The method of checking the sufficiency of funds to perform a trading operation:**

```
//+------------------------------------------------------------------+
//| Check if the funds are sufficient                                |
//+------------------------------------------------------------------+
bool CTrading::CheckMoneyFree(const ENUM_ORDER_TYPE order_type,const double volume,const double price,const CSymbol *symbol_obj,const string source_method) const
  {
   ::ResetLastError();
   //--- Get the type of a market order by a trading operation type
   ENUM_ORDER_TYPE action=this.DirectionByActionType((ENUM_ACTION_TYPE)order_type);
   //--- Get the value of free funds to be left after conducting a trading operation
   double money_free=
     (
      //--- For MQL5, calculate the difference between free funds and the funds required to conduct a trading operation
      #ifdef __MQL5__  this.m_account.MarginFree()-this.m_account.MarginForAction(action,symbol_obj.Name(),volume,price)
      //--- For MQL4, use the operation result of the standard function returning the amount of funds left
      #else/*__MQL4__*/::AccountFreeMarginCheck(symbol_obj.Name(),order_type,volume) #endif
     );
   //--- If no free funds are left, inform of that and return 'false'
   if(money_free<=0 #ifdef __MQL4__ || ::GetLastError()==134 #endif )
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         //--- create a message text
         string message=
           (
            symbol_obj.Name()+" "+::DoubleToString(volume,symbol_obj.DigitsLot())+" "+
            (
             order_type>ORDER_TYPE_SELL ? OrderTypeDescription(order_type,false,false) :
             PositionTypeDescription(PositionTypeByOrderType(order_type))
            )+" ("+::DoubleToString(money_free,(int)this.m_account.CurrencyDigits())+")"
           );
         //--- display a journal message
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR),message);
        }
      return false;
     }
   //--- Verification successful
   return true;
  }
//+------------------------------------------------------------------+
```

**The method of checking limitations for conducting trading operations:**

```
//+------------------------------------------------------------------+
//| Check trading limitations                                        |
//+------------------------------------------------------------------+
bool CTrading::CheckTradeConstraints(const double volume,
                                     const ENUM_ACTION_TYPE action_type,
                                     const CSymbol *symbol_obj,
                                     const string source_method,
                                     double sl=0,
                                     double tp=0)
  {
//--- the result of conducting all checks
   bool res=true;
//--- Clear the error list (codes of found limitations)
   this.m_list_errors.Clear();
   this.m_list_errors.Sort();

//--- Check connection with the trade server (not in the test mode)
   if(!::TerminalInfoInteger(TERMINAL_CONNECTED))
     {
      if(!::MQLInfoInteger(MQL_TESTER))
        {
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_LIB_TEXT_TERMINAL_NOT_CONNECTED);
         res &=false;
        }
     }
//--- Check if trading is enabled for an account (if there is a connection with the trade server)
   else if(!this.m_account.TradeAllowed())
     {
      //--- add the error code to the list and write 'false' to the result
      this.AddErrorCodeToList(MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED);
      res &=false;
     }
//--- Check if trading is allowed for any EAs/scripts for the current account
   if(!this.m_account.TradeExpert())
     {
      //--- add the error code to the list and write 'false' to the result
      this.AddErrorCodeToList(MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED);
      res &=false;
     }
//--- Check if auto trading is allowed in the terminal.
//--- AutoTrading button (Options --> Expert Advisors --> "Allowed automated trading")
   if(!::TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
     {
      //--- add the error code to the list and write 'false' to the result
      this.AddErrorCodeToList(MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED);
      res &=false;
     }
//--- Check if auto trading is allowed for the current EA.
//--- (F7 --> Common --> Allow Automated Trading)
   if(!::MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      //--- add the error code to the list and write 'false' to the result
      this.AddErrorCodeToList(MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED);
      res &=false;
     }
//--- Check if trading is enabled on a symbol.
//--- If trading is disabled, add the error code to the list and write 'false' to the result
   if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)
     {
      this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_DISABLED);
      res &=false;
     }

//--- If not closing/removal/modification
   if(action_type>WRONG_VALUE && action_type<ACTION_TYPE_CLOSE_BY)
     {
      //--- In case of close-only, add the error code to the list and write 'false' to the result
      if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)
        {
         this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_CLOSEONLY);
         res &=false;
        }
      //--- Check the minimum volume
      if(volume<symbol_obj.LotsMin())
        {
         //--- The volume in a request is less than the minimum allowed one
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME);
         res &=false;
        }
      //--- Check the maximum volume
      else if(volume>symbol_obj.LotsMax())
        {
         //--- The volume in the request exceeds the maximum acceptable one
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME);
         res &=false;
        }
      //--- Check the minimum volume gradation
      double step=symbol_obj.LotsStep();
      if(fabs((int)round(volume/step)*step-volume)>0.0000001)
        {
         //--- The volume in the request is not a multiple of the minimum gradation of the lot change step
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_LIB_TEXT_INVALID_VOLUME_STEP);
         res &=false;
        }
     }

//--- When opening a position
   if(action_type>WRONG_VALUE && action_type<ACTION_TYPE_BUY_LIMIT)
     {
      //--- Check if sending market orders is allowed on a symbol.
      //--- If trading market orders is disabled, add the error code to the list and write 'false' to the result
      if(!symbol_obj.IsMarketOrdersAllowed())
        {
         this.AddErrorCodeToList(MSG_SYM_MARKET_ORDER_DISABLED);
         res &=false;
        }
     }
//--- When placing a pending order
   else if(action_type>ACTION_TYPE_SELL && action_type<ACTION_TYPE_CLOSE_BY)
     {
      //--- If there is a limitation on the number of pending orders on the account
      //--- and placing a new order exceeds the acceptable number
      if(this.m_account.LimitOrders()>0 && this.OrdersTotalAll()+1 > this.m_account.LimitOrders())
        {
         //--- The maximum number of pending orders is reached
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(10033);
         res &=false;
        }
      //--- Check if placing limit orders is allowed on a symbol.
      if(action_type==ACTION_TYPE_BUY_LIMIT || action_type==ACTION_TYPE_SELL_LIMIT)
        {
         //--- If it is not, add the error code to the list and write 'false' to the result
         if(!symbol_obj.IsLimitOrdersAllowed())
           {
            this.AddErrorCodeToList(MSG_SYM_LIMIT_ORDER_DISABLED);
            res &=false;
           }
        }
      //--- Check if placing stop orders is allowed on a symbol.
      else if(action_type==ACTION_TYPE_BUY_STOP || action_type==ACTION_TYPE_SELL_STOP)
        {
         //--- If placing stop orders is disabled, add the error code to the list and write 'false' to the result
         if(!symbol_obj.IsStopOrdersAllowed())
           {
            this.AddErrorCodeToList(MSG_SYM_STOP_ORDER_DISABLED);
            res &=false;
           }
        }
      //--- For MQL5, check if placing stop limit orders is allowed on a symbol.
      #ifdef __MQL5__
      else if(action_type==ACTION_TYPE_BUY_STOP_LIMIT || action_type==ACTION_TYPE_SELL_STOP_LIMIT)
        {
         //--- If it is not, add the error code to the list and write 'false' to the result
         if(!symbol_obj.IsStopLimitOrdersAllowed())
           {
            this.AddErrorCodeToList(MSG_SYM_STOP_LIMIT_ORDER_DISABLED);
            res &=false;
           }
        }
      #endif
     }

//--- In case of opening/placing/modification
   if(action_type>WRONG_VALUE && action_type!=ACTION_TYPE_CLOSE_BY)
     {
      //--- If not modification
      if(action_type!=ACTION_TYPE_MODIFY)
        {
         //--- When buying, check if long trading is enabled on a symbol
         if(this.DirectionByActionType(action_type)==ORDER_TYPE_BUY)
           {
            //--- If only selling is allowed, add the error code to the list and write 'false' to the result
            if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY)
              {
               this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_SHORTONLY);
               res &=false;
              }
            //--- If a symbol has the limitation on the total volume of an open position and pending orders in the same direction
            if(symbol_obj.VolumeLimit()>0)
              {
               //--- (If the total volume of placed long orders and open long positions)+open volume exceed the maximum one
               if(this.OrdersTotalVolumeLong()+this.PositionsTotalVolumeLong()+volume > symbol_obj.VolumeLimit())
                 {
                  //--- Exceeded maximum allowed aggregate volume of orders and positions in one direction
                  //--- add the error code to the list and write 'false' to the result
                  this.AddErrorCodeToList(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED);
                  res &=false;
                 }
              }
           }
         //--- When selling, check if short trading is enabled on a symbol
         else if(this.DirectionByActionType(action_type)==ORDER_TYPE_SELL)
           {
            //--- If only buying is allowed, add the error code to the list and write 'false' to the result
            if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY)
              {
               this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_LONGONLY);
               res &=false;
              }
            //--- If a symbol has the limitation on the total volume of an open position and pending orders in the same direction
            if(symbol_obj.VolumeLimit()>0)
              {
               //--- (If the total volume of placed short orders and open short positions)+open volume exceed the maximum one
               if(this.OrdersTotalVolumeShort()+this.PositionsTotalVolumeShort()+volume > symbol_obj.VolumeLimit())
                 {
                  //--- Exceeded maximum allowed aggregate volume of orders and positions in one direction
                  //--- add the error code to the list and write 'false' to the result
                  this.AddErrorCodeToList(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED);
                  res &=false;
                 }
              }
           }
        }
      //--- If the request features StopLoss and its placing is not allowed
      if(sl>0 && !symbol_obj.IsStopLossOrdersAllowed())
        {
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_SYM_SL_ORDER_DISABLED);
         res &=false;
        }
      //--- If the request features TakeProfit and its placing is not allowed
      if(tp>0 && !symbol_obj.IsTakeProfitOrdersAllowed())
        {
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_SYM_TP_ORDER_DISABLED);
         res &=false;
        }
     }

//--- When closing by an opposite position
   else if(action_type==ACTION_TYPE_CLOSE_BY)
     {
      //--- When closing by an opposite position is disabled
      if(!symbol_obj.IsCloseByOrdersAllowed())
        {
         //--- add the error code to the list and write 'false' to the result
         this.AddErrorCodeToList(MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED);
         res &=false;
        }
     }

//--- If there are limitations, display the header and the error list
   if(!res)
     {
      //--- Request was rejected before sending to the server due to:
      int total=this.m_list_errors.Total();
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         //--- For MQL5, first display the list header followed by the error list
         #ifdef __MQL5__
         ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_REQUEST_REJECTED_DUE));
         for(int i=0;i<total;i++)
            ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
         //--- For MQL4, the journal messages are displayed in the reverse order: the error list is followed by the list header
         #else
         for(int i=total-1;i>WRONG_VALUE;i--)
            ::Print((total>1 ? string(i+1)+". " : ""),CMessage::Text(m_list_errors.At(i)));
         ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_REQUEST_REJECTED_DUE));
         #endif
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

The method is quite spacious but all is already familiar here — checking the properties of the program, terminal, account or symbol, for which
the limitations are possible. If a limitation exists, add the code of the detected limitation to the list and add

false to the result flag. If, after all the checks, the variable has false,
display the header of the list of detected limitations and show all detected trading limitations under the header in a loop by the list of
limitation codes. If the variable storing the check results has

true (by which this variable was originally initialized), then all checks have been
passed and

true is returned.

In the current version of the trading class, we will stop at this point. Further on, when creating the error handling methods, all actions in
response to limitations and error are to be handled in this class.

**Class trading methods:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
bool CTrading::OpenBuy(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_BUY,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
//--- If the funds are insufficient, inform of that and exit
   if(!this.CheckMoneyFree(ORDER_TYPE_BUY,volume,symbol_obj.Ask(),symbol_obj,DFUN))
     {
      return false;
     }

//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.OpenPosition(POSITION_TYPE_BUY,volume,sl,tp,magic,trade_obj.GetDeviation(),comment);
  }
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
bool CTrading::OpenSell(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_SELL,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- If the funds are insufficient, inform of that and exit
   if(!this.CheckMoneyFree(ORDER_TYPE_SELL,volume,symbol_obj.Bid(),symbol_obj,DFUN))
     {
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.OpenPosition(POSITION_TYPE_SELL,volume,sl,tp,magic,trade_obj.GetDeviation(),comment);
  }
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
bool CTrading::ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE)
  {
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,ACTION_TYPE_MODIFY,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.ModifyPosition(ticket,sl,tp);
  }
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CTrading::ClosePosition(const ulong ticket)
  {
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,WRONG_VALUE,symbol_obj,DFUN))
     {
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.ClosePosition(ticket);
  }
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CTrading::ClosePositionPartially(const ulong ticket,const double volume)
  {
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,WRONG_VALUE,symbol_obj,DFUN))
     {
      return false;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.ClosePositionPartially(ticket,symbol_obj.NormalizedLot(volume));
  }
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CTrading::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,ACTION_TYPE_CLOSE_BY,symbol_obj,DFUN))
     {
      return false;
     }
   //--- trading object of a closed position
   CTradeObj *trade_obj_pos=this.GetTradeObjByPosition(ticket,DFUN);
   if(trade_obj_pos==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- check the presence of an opposite position
   if(!this.CheckPositionAvailablity(ticket_by,DFUN))
      return false;
   //--- trading object of an opposite position
   CTradeObj *trade_obj_pos_by=this.GetTradeObjByPosition(ticket_by,DFUN);
   if(trade_obj_pos_by==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- If a symbol of a closed position is not equal to an opposite position's one, inform of that and exit
   if(symbol_obj.Name()!=trade_obj_pos_by.GetSymbol())
     {
      //--- Symbols of opposite positions are not equal
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj_pos.ClosePositionBy(ticket,ticket_by);
  }
//+------------------------------------------------------------------+
//| Place BuyStop pending order                                      |
//+------------------------------------------------------------------+
bool CTrading::PlaceBuyStop(const double volume,
                           const string symbol,
                           const double price,
                           const double sl=0,
                           const double tp=0,
                           const ulong magic=WRONG_VALUE,
                           const string comment=NULL,
                           const datetime expiration=0,
                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_BUY_STOP,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.SetOrder(ORDER_TYPE_BUY_STOP,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
//| Place BuyLimit pending order                                     |
//+------------------------------------------------------------------+
bool CTrading::PlaceBuyLimit(const double volume,
                            const string symbol,
                            const double price,
                            const double sl=0,
                            const double tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_BUY_LIMIT,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.SetOrder(ORDER_TYPE_BUY_LIMIT,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
//| Place BuyStopLimit pending order                                 |
//+------------------------------------------------------------------+
bool CTrading::PlaceBuyStopLimit(const double volume,
                                const string symbol,
                                const double price_stop,
                                const double price_limit,
                                const double sl=0,
                                const double tp=0,
                                const ulong magic=WRONG_VALUE,
                                const string comment=NULL,
                                const datetime expiration=0,
                                const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_BUY_STOP_LIMIT,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   #ifdef __MQL5__
      //--- get a trading object from a symbol object
      CTradeObj *trade_obj=symbol_obj.GetTradeObj();
      if(trade_obj==NULL)
        {
         //--- Error. Failed to get trading object
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
         return false;
        }
      //--- Return the result of sending a trading request in a symbol trading object
      return trade_obj.SetOrder(ORDER_TYPE_BUY_STOP_LIMIT,volume,price_stop,sl,tp,price_limit,magic,expiration,type_time,comment);
   //--- MQL4
   #else
      return true;
   #endif
  }
//+------------------------------------------------------------------+
//| Place SellStop pending order                                     |
//+------------------------------------------------------------------+
bool CTrading::PlaceSellStop(const double volume,
                            const string symbol,
                            const double price,
                            const double sl=0,
                            const double tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_SELL_STOP,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.SetOrder(ORDER_TYPE_SELL_STOP,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
//| Place SellLimit pending order                                    |
//+------------------------------------------------------------------+
bool CTrading::PlaceSellLimit(const double volume,
                             const string symbol,
                             const double price,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=WRONG_VALUE,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_SELL_LIMIT,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.SetOrder(ORDER_TYPE_SELL_LIMIT,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
//| Place SellStopLimit pending order                                |
//+------------------------------------------------------------------+
bool CTrading::PlaceSellStopLimit(const double volume,
                                 const string symbol,
                                 const double price_stop,
                                 const double price_limit,
                                 const double sl=0,
                                 const double tp=0,
                                 const ulong magic=WRONG_VALUE,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(volume,ACTION_TYPE_SELL_STOP_LIMIT,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   #ifdef __MQL5__
      //--- get a trading object from a symbol object
      CTradeObj *trade_obj=symbol_obj.GetTradeObj();
      if(trade_obj==NULL)
        {
         //--- Error. Failed to get trading object
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
         return false;
        }
      //--- Return the result of sending a trading request in a symbol trading object
      return trade_obj.SetOrder(ORDER_TYPE_SELL_STOP_LIMIT,volume,price_stop,sl,tp,price_limit,magic,expiration,type_time,comment);
   //--- MQL4
   #else
      return true;
   #endif
  }
//+------------------------------------------------------------------+
//| Modify a pending order                                           |
//+------------------------------------------------------------------+
bool CTrading::ModifyOrder(const ulong ticket,
                          const double price=WRONG_VALUE,
                          const double sl=WRONG_VALUE,
                          const double tp=WRONG_VALUE,
                          const double stoplimit=WRONG_VALUE,
                          datetime expiration=WRONG_VALUE,
                          ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE)
  {
//--- Get a symbol object by an order ticket
   CSymbol *symbol_obj=this.GetSymbolObjByOrder(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,ACTION_TYPE_MODIFY,symbol_obj,DFUN,sl,tp))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.ModifyOrder(ticket,price,sl,tp,stoplimit,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Remove a pending order                                           |
//+------------------------------------------------------------------+
bool CTrading::DeleteOrder(const ulong ticket)
  {
//--- Get a symbol object by an order ticket
   CSymbol *symbol_obj=this.GetSymbolObjByOrder(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
   if(symbol_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- Update symbol quotes
   symbol_obj.RefreshRates();
//--- If there are trading limitations, inform of that and exit
   if(!this.CheckTradeConstraints(0,WRONG_VALUE,symbol_obj,DFUN))
     {
      return false;
     }
   //--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   //--- Return the result of sending a trading request in a symbol trading object
   return trade_obj.DeleteOrder(ticket);
  }
//+------------------------------------------------------------------+
```

We have considered the trading methods (sending trading requests to the server) [in \\
the previous article when creating the base trading object](https://www.mql5.com/en/articles/7229#node02). Here, the methods having a similar signature receive the necessary
parameters of a trading request, trading limitations examined above are checked and the appropriate method of a symbol base trading object
is called. The result of the working method is returned to the program.

At this stage, these are all the checks conducted before sending a trading request. We will expand them further on since we need to check
and handle the validity of the trading request parameters. Now it is assumed that all parameters are already valid.



**The trading class in its current implementation with the planned functionality is ready.**

Now we need to
access it from the library

**CEngine** base object and work with the class trading methods through it.

Open \\MQL5\\Include\\DoEasy\ **Engine.mqh** and make all the necessary changes in it.

First, **include the trading class file in it:**

```
//+------------------------------------------------------------------+
//|                                                       Engine.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "Services\TimerCounter.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\EventsCollection.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\ResourceCollection.mqh"
#include "Trading.mqh"
//+------------------------------------------------------------------+
```

**In the private section of the class, declare the trading class object:**

```
//+------------------------------------------------------------------+
//| Library basis class                                              |
//+------------------------------------------------------------------+
class CEngine
  {
private:
   CHistoryCollection   m_history;                       // Collection of historical orders and deals
   CMarketCollection    m_market;                        // Collection of market orders and deals
   CEventsCollection    m_events;                        // Event collection
   CAccountsCollection  m_accounts;                      // Account collection
   CSymbolsCollection   m_symbols;                       // Symbol collection
   CResourceCollection  m_resource;                      // Resource list
   CTrading             m_trading;                       // Trading class object
   CArrayObj            m_list_counters;                 // List of timer counters
   int                  m_global_error;                  // Global error code
   bool                 m_first_start;                   // First launch flag
   bool                 m_is_hedge;                      // Hedge account flag
   bool                 m_is_tester;                     // Flag of working in the tester
   bool                 m_is_market_trade_event;         // Account trading event flag
   bool                 m_is_history_trade_event;        // Account history trading event flag
   bool                 m_is_account_event;              // Account change event flag
   bool                 m_is_symbol_event;               // Symbol change event flag
   ENUM_TRADE_EVENT     m_last_trade_event;              // Last account trading event
   int                  m_last_account_event;            // Last event in the account properties
   int                  m_last_symbol_event;             // Last event in the symbol properties
//--- Return the counter index by id
```

The methods

```
//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level
   void                 SetTradeCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 SetTradeTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 SetTradeCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 SetTradeTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 SetTradeMagic(const ulong magic,const string symbol_name=NULL);
   void                 SetTradeComment(const string comment,const string symbol_name=NULL);
   void                 SetTradeDeviation(const ulong deviation,const string symbol_name=NULL);
   void                 SetTradeVolume(const double volume=0,const string symbol_name=NULL);
   void                 SetTradeExpiration(const datetime expiration=0,const string symbol_name=NULL);
   void                 SetTradeAsyncMode(const bool mode=false,const string symbol_name=NULL);
   void                 SetTradeLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL);
```

are renamed to

```
//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level
   void                 TradingSetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL);
   void                 TradingSetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL);
   void                 TradingSetMagic(const ulong magic,const string symbol_name=NULL);
   void                 TradingSetComment(const string comment,const string symbol_name=NULL);
   void                 TradingSetDeviation(const ulong deviation,const string symbol_name=NULL);
   void                 TradingSetVolume(const double volume=0,const string symbol_name=NULL);
   void                 TradingSetExpiration(const datetime expiration=0,const string symbol_name=NULL);
   void                 TradingSetAsyncMode(const bool mode=false,const string symbol_name=NULL);
   void                 TradingSetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL);
```

so that their name indicates that they belong to the trading class.

In the public section of the class, **add the methods for setting**
**standard sounds, sounds for a specified position/order type and a trading**
**event and** ****the method sending pointers to all the necessary****
****collection lists to the trading class**. Besides, declare** ****the****
****method for playing a sound by its description**:**

```
//--- Set standard sounds (symbol==NULL) for a symbol trading object, (symbol!=NULL) for trading objects of all symbols
   void                 SetSoundsStandart(const string symbol=NULL)
                          {
                           this.m_trading.SetSoundsStandart(symbol);
                          }

//--- Set a sound for a specified order/position type and symbol. 'mode' specifies an event a sound is set for
//--- (symbol=NULL) for trading objects of all symbols, (symbol!=NULL) for a trading object of a specified symbol
   void                 SetSound(const ENUM_MODE_SET_SOUND mode,const ENUM_ORDER_TYPE action,const string sound,const string symbol=NULL)
                          {
                           this.m_trading.SetSound(mode,action,sound,symbol);
                          }
//--- Play a sound by its description
   bool                 PlaySoundByDescription(const string sound_description);

//--- Pass the pointers to all the necessary collections to the trading class
   void                 TradingOnInit(void)
                          {
                           this.m_trading.OnInit(this.GetAccountCurrent(),m_symbols.GetObject(),m_market.GetObject(),m_history.GetObject());
                          }
```

Three methods simply call the trading class methods of the same name. **Let's consider the method of playing a sound by its**
**description below:**

```
//+------------------------------------------------------------------+
//| Play a sound by its description                                  |
//+------------------------------------------------------------------+
bool CEngine::PlaySoundByDescription(const string sound_description)
  {
   string file_name=NULL;
   //--- Get the list of resources
   CArrayObj* list=this.GetListResource();
   if(list==NULL)
      return false;
   //--- Get an index of a resource object by its description
   int index=this.m_resource.GetIndexResObjByDescription(sound_description);
   //--- If a resource object with a specified description is found in the list
   if(index>WRONG_VALUE)
     {
      //--- Get a resource object by its index in the list
      CResObj *res_obj=list.At(index);
      if(res_obj==NULL)
         return false;
      //--- Get a resource object file name
      file_name=res_obj.FileName();
     }
   //--- If there is no resource object with a specified description in the list, attempt to play the file by the name written in its description
   //--- To do this, make sure that either a standard audio file (macro substitution of its name),
   //--- or a name of a new *.wav audio file is passed as a description
   else if(::StringFind(sound_description,"SND_")==0 || StringSubstr(sound_description,StringLen(sound_description)-4)==".wav")
      file_name=sound_description;
   //--- Return the file playing result
   return(file_name!=NULL ? CMessage::PlaySound(file_name) : false);
  }
//+------------------------------------------------------------------+
```

Previously, we created [resource objects](https://www.mql5.com/en/articles/7195) where we
could store audio and image files. We have added the ability to add a description to each resource of that kind. Thus, it has become easy for us
to specify what audio file we need to play. For example, it is much easier to remember a custom file description than its name. If the file name
is "featureless" (like sound\_01.wav), we can set any description for it, for example, "sound of opening Buy on EURUSD".

The method allows specifying the necessary file description in its parameters. It detects a resource object with the matching
description and plays the sound. Optionally, the method is able to play standard sounds by the macro substitutions of their names we created
in Defines.mqh, as well as any unknown sound file. We simply need to pass its name as its description. Such a file should be located in the
terminal sandbox, have wav extension and have the correct path specified in its name.

The trading methods we temporarily placed in the CEngine class, as well as the methods setting trading object parameters and the ones
temporarily located in CEngine have been changed — all these methods, as well as their implementation, have been moved to a trading class,

**while the appropriate trading class methods are simply called from the CEngine class methods:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
bool CEngine::OpenBuy(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
   return this.m_trading.OpenBuy(volume,symbol,magic,sl,tp,comment);
  }
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
bool CEngine::OpenSell(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
   return this.m_trading.OpenSell(volume,symbol,magic,sl,tp,comment);
  }
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
bool CEngine::ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE)
  {
   return this.m_trading.ModifyPosition(ticket,sl,tp);
  }
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CEngine::ClosePosition(const ulong ticket)
  {
   return this.m_trading.ClosePosition(ticket);
  }
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionPartially(const ulong ticket,const double volume)
  {
   return this.m_trading.ClosePositionPartially(ticket,volume);
  }

//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   return this.m_trading.ClosePositionBy(ticket,ticket_by);
  }
//+------------------------------------------------------------------+
//| Place BuyStop pending order                                      |
//+------------------------------------------------------------------+
bool CEngine::PlaceBuyStop(const double volume,
                           const string symbol,
                           const double price,
                           const double sl=0,
                           const double tp=0,
                           const ulong magic=WRONG_VALUE,
                           const string comment=NULL,
                           const datetime expiration=0,
                           const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place BuyLimit pending order                                     |
//+------------------------------------------------------------------+
bool CEngine::PlaceBuyLimit(const double volume,
                            const string symbol,
                            const double price,
                            const double sl=0,
                            const double tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place BuyStopLimit pending order                                 |
//+------------------------------------------------------------------+
bool CEngine::PlaceBuyStopLimit(const double volume,
                                const string symbol,
                                const double price_stop,
                                const double price_limit,
                                const double sl=0,
                                const double tp=0,
                                const ulong magic=WRONG_VALUE,
                                const string comment=NULL,
                                const datetime expiration=0,
                                const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceBuyStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellStop pending order                                     |
//+------------------------------------------------------------------+
bool CEngine::PlaceSellStop(const double volume,
                            const string symbol,
                            const double price,
                            const double sl=0,
                            const double tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellLimit pending order                                    |
//+------------------------------------------------------------------+
bool CEngine::PlaceSellLimit(const double volume,
                             const string symbol,
                             const double price,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=WRONG_VALUE,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Place SellStopLimit pending order                                |
//+------------------------------------------------------------------+
bool CEngine::PlaceSellStopLimit(const double volume,
                                 const string symbol,
                                 const double price_stop,
                                 const double price_limit,
                                 const double sl=0,
                                 const double tp=0,
                                 const ulong magic=WRONG_VALUE,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC)
  {
   return this.m_trading.PlaceSellStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Modify a pending order                                           |
//+------------------------------------------------------------------+
bool CEngine::ModifyOrder(const ulong ticket,
                          const double price=WRONG_VALUE,
                          const double sl=WRONG_VALUE,
                          const double tp=WRONG_VALUE,
                          const double stoplimit=WRONG_VALUE,
                          datetime expiration=WRONG_VALUE,
                          ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE)
  {
   return this.m_trading.ModifyOrder(ticket,price,sl,tp,stoplimit,expiration,type_time);
  }
//+------------------------------------------------------------------+
//| Remove a pending order                                           |
//+------------------------------------------------------------------+
bool CEngine::DeleteOrder(const ulong ticket)
  {
   return this.m_trading.DeleteOrder(ticket);
  }
//+------------------------------------------------------------------+
//| Set the valid filling policy                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL)
  {
   this.m_trading.SetCorrectTypeFilling(type,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set the filling policy                                           |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL)
  {
   this.m_trading.SetTypeFilling(type,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set a correct order expiration type                              |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL)
  {
   this.m_trading.SetCorrectTypeExpiration(type,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set an order expiration type                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL)
  {
   this.m_trading.SetTypeExpiration(type,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set a magic number for trading objects of all symbols            |
//+------------------------------------------------------------------+
void CEngine::TradingSetMagic(const ulong magic,const string symbol_name=NULL)
  {
   this.m_trading.SetMagic(magic,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set a comment for trading objects of all symbols                 |
//+------------------------------------------------------------------+
void CEngine::TradingSetComment(const string comment,const string symbol_name=NULL)
  {
   this.m_trading.SetComment(comment,symbol_name);
  }

//+------------------------------------------------------------------+
//| Set a slippage                                                   |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetDeviation(const ulong deviation,const string symbol_name=NULL)
  {
   this.m_trading.SetDeviation(deviation,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set a volume for trading objects of all symbols                  |
//+------------------------------------------------------------------+
void CEngine::TradingSetVolume(const double volume=0,const string symbol_name=NULL)
  {
   this.m_trading.SetVolume(volume,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set an order expiration date                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetExpiration(const datetime expiration=0,const string symbol_name=NULL)
  {
   this.m_trading.SetExpiration(expiration,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set the flag of asynchronous sending of trading requests         |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetAsyncMode(const bool mode=false,const string symbol_name=NULL)
  {
   this.m_trading.SetAsyncMode(mode,symbol_name);
  }
//+------------------------------------------------------------------+
//| Set a logging level of trading requests                          |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::TradingSetLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL)
  {
   this.m_trading.SetLogLevel(log_level,symbol_name);
  }
//+------------------------------------------------------------------+
```

This concludes the improvement of the CEngine class. Now we need to check how all this works.

### Testing

To perform the test, let's use the EA from the previous article and save it to \\MQL5\\Experts\\TestDoEasy\ **Part22\**
under the name **TestDoEasyPart22.mq5**.

In the previous EA, the journal displayed a lot of textual verification data in the OnInit() handler. Currently, we do not need it, so let's
remove all redundant Print() and

add the trading class initialization and checking
a standard sound playback by macro substitution and a custom sound by
its description:

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
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_01",TextByLanguage("Звук упавшей монетки 1","Sound of falling coin 1"),sound_array_coin_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_02",TextByLanguage("Звук упавших монеток","Falling coins"),sound_array_coin_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_03",TextByLanguage("Звук монеток","Coins"),sound_array_coin_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_coin_04",TextByLanguage("Звук упавшей монетки 2","Sound of falling coin 2"),sound_array_coin_04);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_01",TextByLanguage("Звук щелчка по кнопке 1","Click on button sound 1"),sound_array_click_01);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_02",TextByLanguage("Звук щелчка по кнопке 2","Click on button sound 1"),sound_array_click_02);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_click_03",TextByLanguage("Звук щелчка по кнопке 3","Click on button sound 1"),sound_array_click_03);
   engine.CreateFile(FILE_TYPE_WAV,"sound_array_cash_machine_01",TextByLanguage("Звук кассового аппарата","Sound of cash machine"),sound_array_cash_machine_01);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_green",TextByLanguage("Изображение \"Зелёный светодиод\"","Image \"Green Spot lamp\""),img_array_spot_green);
   engine.CreateFile(FILE_TYPE_BMP,"img_array_spot_red",TextByLanguage("Изображение \"Красный светодиод\"","Image \"Red Spot lamp\""),img_array_spot_red);

//--- Pass all existing collections to the trading class
   engine.TradingOnInit();
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode();
//--- Set standard sounds for trading objects of all used symbols
   engine.SetSoundsStandart();
//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//--- Set controlled values for symbols
   //--- Get the list of all collection symbols
   CArrayObj *list=engine.GetListAllUsedSymbols();
   if(list!=NULL && list.Total()!=0)
     {
      //--- In a loop by the list, set the necessary values for tracked symbol properties
      //--- By default, the LONG_MAX value is set to all properties, which means "Do not track this property"
      //--- It can be enabled or disabled (by setting the value less than LONG_MAX or vice versa - set the LONG_MAX value) at any time and anywhere in the program
      for(int i=0;i<list.Total();i++)
        {
         CSymbol* symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- Set control of the symbol price increase by 100 points
         symbol.SetControlBidInc(100*symbol.Point());
         //--- Set control of the symbol price decrease by 100 points
         symbol.SetControlBidDec(100*symbol.Point());
         //--- Set control of the symbol spread increase by 40 points
         symbol.SetControlSpreadInc(40);
         //--- Set control of the symbol spread decrease by 40 points
         symbol.SetControlSpreadDec(40);
         //--- Set control of the current spread by the value of 40 points
         symbol.SetControlSpreadLevel(40);
        }
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

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

After compiling and launching the EA, the standard "ok.wav" sound is played. In 1/6 second, it is followed by the custom sound with the "Falling
coin 2" description.

To check the work of the methods checking trading limitations, we need to artificially create them.

For example:

1. disable Internet access (to simulate the loss of connection to the trade server),

2. disable trading in the EA settings (press F7 and uncheck "Allow Automated Trading" in the Common tab of the EA settings window),

3. disable auto trading in the terminal (the AutoTrading button).


Click the position opening button in the EA trading panel. The following entry appears in the journal:

```
2019.09.26 15:07:55.582 CTrading::OpenBuy: Request rejected before being sent to server due to:
2019.09.26 15:07:55.582 1. No permission to conduct trading operations in terminal ("AutoTrading" button disabled)
2019.09.26 15:07:55.582 2. No connection to trade server
2019.09.26 15:07:55.582 3. EA does not have permission to conduct trading operations (F7 --> Common --> "Allow Automatic Trading")
```

Let's eliminate the restrictions one by one.

After enabling the Internet connection, we obtain the following message when attempting to open a position:

```
2019.09.26 15:10:36.766 CTrading::OpenBuy: Request rejected before being sent to server due to:
2019.09.26 15:10:36.766 1. No permission to conduct trading operations in terminal ("AutoTrading" button disabled)
2019.09.26 15:10:36.766 2. EA does not have permission to conduct trading operations (F7 --> Common --> "Allow Automatic Trading")
```

Enable auto trading in the terminal clicking the AutoTrading button. When attempting to open a position, we obtain the following:

```
2019.09.26 15:13:03.424 CTrading::OpenBuy: Request rejected before being sent to server due to:
2019.09.26 15:13:03.424 EA does not have permission to conduct trading operations (F7 --> Common --> "Allow Automatic Trading")
```

Press F7 and allow the EA to trade in its settings. When attempting to open a position, we are finally successful:

```
2019.09.26 15:14:32.619 - Position is open: 2019.09.26 11:14:32.711 -
2019.09.26 15:14:32.619 EURUSD Opened 0.10 Buy #455179802 [0.10 Market-order Buy #455179802] at price 1.09441, Magic number 123
```

Other limitations can be checked in the tester or a demo account by creating a situation when one of the limitations is activated, for example, a
limitation by the maximum number of pending orders on the account.

### What's next?

In the next article, we will implement the verification of the trading order parameters validity.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7258#node00)

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

[Part 19. Class of \\
library messages](https://www.mql5.com/en/articles/7176)

[Part 20. Creating and storing program resources](https://www.mql5.com/en/articles/7195)

[Part 21. Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7258](https://www.mql5.com/ru/articles/7258)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7258.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7258/mql5.zip "Download MQL5.zip")(3585.29 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7258/mql4.zip "Download MQL4.zip")(3585.26 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/329337)**
(21)


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
6 Oct 2019 at 15:26

**Artyom Trishkin:**

I need to see the whole code. Throw it in a private message, or here if it's not a secret behind seven seals.

There's nothing there yet.

```
/********************************************************************\
|Only_BUY_or_only_SELL.mq4 |
|© 2019, Alexey Viktorov |
| https://www.mql5.com/en/users/alexeyvik/news |
\********************************************************************/
#property copyright "© 2019, Alexey Viktorov"
#property link      "https://www.mql5.com/en/users/alexeyvik/news"
#property version   "1.00"
#property strict
#define  ndd(A) NormalizeDouble(A, _Digits)
//---
#include <DoEasy\Engine.mqh>
CEngine        engine;
//---
enum MY_ORDER_TYPE
{
 BUY   = OP_BUY,
 SELL  = OP_SELL,
};

struct POZ
{
 double profit;
 ulong  ticket;
 double priceGrid;
 double price;
}poz[];

sinput  MY_ORDER_TYPE   ordType         = SELL;   // Direction of trade
sinput  double          lot             = 0.01;   // Lot size
 input  int             StepGrid        = 50;     // Grid spacing
 input  double          desiredProfit   = 10;     // Desired profit
sinput  bool            Should_I_open   = false;  // Should I open a new series of positions?
sinput  int             magick          = 1;      // Magick number
//---
int slip = 100;
double contract, stepGrid, priceGrid;
int totalPositions, levelNumb;//, levelBUY, levelSELL;

/*******************Expert initialization function*******************/
int OnInit()
{
 stepGrid = StepGrid*_Point;
 return(INIT_SUCCEEDED);
}/*******************************************************************/

/************************Expert tick function************************/
void OnTick()
{
//--- Initialisation of the last trade event
 static ENUM_TRADE_EVENT last_event = WRONG_VALUE;
//--- If working in the tester
 if(MQLInfoInteger(MQL_TESTER))
  {
   engine.OnTimer();
  }
//--- If the last trade event has changed
 if(engine.LastTradeEvent() != last_event)
  {
 //Print(__FUNCTION__, "***", last_event);
   last_event=engine.LastTradeEvent();
   switch(last_event)
    {
     case TRADE_EVENT_PENDING_ORDER_PLASED :
     Print(__FUNCTION__, "***", last_event);
     break;
     case TRADE_EVENT_PENDING_ORDER_REMOVED :

     break;
     case TRADE_EVENT_PENDING_ORDER_ACTIVATED :
      {
       Print(__FUNCTION__, "***", last_event);
       //OrderActivated();
     break;
      }
     case TRADE_EVENT_POSITION_OPENED :

     break;
     case TRADE_EVENT_POSITION_CLOSED :

     break;

     default :
      break;
    }
   last_event = TRADE_EVENT_NO_EVENT;
  }

 if(Should_I_open && OrdersTotal() == 0)
  {
   priceGrid = ndd(ordType == SELL ? Bid+stepGrid/2 : Bid-stepGrid/2);
   ENUM_ORDER_TYPE type = ENUM_ORDER_TYPE(ordType == OP_SELL ? OP_SELLLIMIT : OP_BUYSTOP);
   if(OrderSend(_Symbol, type, lot, priceGrid, slip, 0.0, 0.0, (string)priceGrid, magick) <= 0)
    Print(__FUNCTION__, GetLastError());
   priceGrid = ndd(ordType == SELL ? Bid-stepGrid/2 : Bid+stepGrid/2);
   type = ENUM_ORDER_TYPE(ordType == OP_SELL ? OP_SELLSTOP : OP_BUYLIMIT);
   if(OrderSend(_Symbol, type, lot, priceGrid, slip, 0.0, 0.0, (string)priceGrid, magick) <= 0)
    Print(__FUNCTION__, GetLastError());
  }
  return;
}/*******************************************************************/

void OrderActivated()
{
 Print(__FUNCTION__, "***", ENUM_TRADE_EVENT(engine.LastTradeEvent()));

}/*******************************************************************/

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
//--- Initialisation of the last trade event
 static ENUM_TRADE_EVENT last_event=WRONG_VALUE;
//--- If working in the tester
 if(MQLInfoInteger(MQL_TESTER))
  {
   engine.OnTimer();
  }
//--- If the last trade event has changed
 if(engine.LastTradeEvent()!=last_event)
  {
   last_event=engine.LastTradeEvent();
  }
}/*******************************************************************/

/***************************Timer function***************************/
void OnTimer()
{
 if(!MQLInfoInteger(MQL_TESTER))
  engine.OnTimer();
}/*******************************************************************/

void OnDeinit(const int reason)
{
 Comment("");
}/*******************************************************************/
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
6 Oct 2019 at 16:09

**Alexey Viktorov:**

Well, there's nothing there yet

I have to run your code to get an answer. I'll get back to you when I'm there.


![Alexey Viktorov](https://c.mql5.com/avatar/2017/4/58E3DFDD-D3B2.jpg)

**[Alexey Viktorov](https://www.mql5.com/en/users/alexeyvik)**
\|
6 Oct 2019 at 16:24

**Artyom Trishkin:**

I have to run your code to get an answer. I'll be there, I'll get back to you.

I know you're not home and I'm not rushing you.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
6 Oct 2019 at 16:53

**Alexey Viktorov:**

I know you're not home and I'm not rushing you.

No, on the contrary - I'm not at work, but at home ;)


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
6 Oct 2019 at 23:13

**Alexey Viktorov:**

I know you're not home and I'm not rushing you.

Yes, there was a bug in the [trade events](https://www.mql5.com/en/articles/232 "Article: Trading Events in MetaTrader 5") \- there was no flag returning the fact that the event happened.

Added it. An update will be in the next article. Thanks for the message.

![Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__5.png)[Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://www.mql5.com/en/articles/7286)

In the article, we continue the development of the trading class by implementing the control over incorrect trading order parameter values and voicing trading events.

![Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://c.mql5.com/2/37/MQL5-avatar-doeasy__3.png)[Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object](https://www.mql5.com/en/articles/7229)

In this article, we will start the development of the new library section - trading classes. Besides, we will consider the development of a unified base trading object for MetaTrader 5 and MetaTrader 4 platforms. When sending a request to the server, such a trading object implies that verified and correct trading request parameters are passed to it.

![Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization.png)[Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://www.mql5.com/en/articles/7290)

The first article is devoted to the creation of a toolkit for working with optimization reports, for importing them from the terminal, as well as for filtering and sorting the obtained data. MetaTrader 5 allows downloading optimization results, however our purpose is to add our own data to the optimization report.

![Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://c.mql5.com/2/37/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://www.mql5.com/en/articles/7195)

The article deals with storing data in the program's source code and creating audio and graphical files out of them. When developing an application, we often need audio and images. The MQL language features several methods of using such data.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/7258&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070450730000651791)

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