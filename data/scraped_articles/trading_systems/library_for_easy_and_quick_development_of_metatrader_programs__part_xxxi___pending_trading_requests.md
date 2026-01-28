---
title: Library for easy and quick development of MetaTrader programs (part XXXI): Pending trading requests - opening positions under certain conditions
url: https://www.mql5.com/en/articles/7521
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:12.137471
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/7521&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070430921611482548)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7521#node01)
- [Preparing data](https://www.mql5.com/en/articles/7521#node02)
- [Pending request object created on demand](https://www.mql5.com/en/articles/7521#node03)
- [Testing](https://www.mql5.com/en/articles/7521#node04)
- [What's next?](https://www.mql5.com/en/articles/7521#node05)


### Concept

While developing the library functionality, I have introduced the concept of trading using pending requests. The concept features two
operation options — handling trade server errors and usual sending of trading orders under programmatically set conditions. [Starting \\
with the article 26](https://www.mql5.com/en/articles/7394), I have been gradually implementing handling trade server errors using pending requests allowing to process
resending trading orders to the server in case fixing of an error requires resending the order to the server after correcting error
parameters and waiting.

Starting with this article, we are going to develop a functionality allowing to trade using pending requests under certain conditions.

This
library functionality allows users to programmatically create conditions, under which a trading order is sent to the server.

For
example:

1. Open Buy upon the occurrence or exceeding a certain time provided that the price has dropped below a specified value (two conditions
    related to symbol property values).

2. Close a position partially if a specified profit is exceeded (one condition related to an account property value).
3. If a position is closed by stop loss, open an opposite position (one condition related to an account event property).


The examples are simple but there may be plenty of conditions and their combinations. At this stage, we are going to develop control over
changes in properties of accounts, symbols and events occurring on the current account. The conditions from these three lists can be set in
any combination.

We will start from a simple thing — controlling changes of symbol and account property values. This will be followed by
controlling account events and reacting to them.

To let a pending request object work as part of a trading logic (sending trading orders under conditions), we need to implement additional
data into this object to store pending request activation conditions and methods of their control and handling. The data storage is to be
made in the form of a two-dimensional array. The first dimension is to store a condition number (there may be as many conditions as
necessary), while the second one is to contain all data of the condition whose number is specified in the first dimension — condition source
type (symbol, account or event), condition itself (create enumerations for each of the sources), comparison method
(>,<,==,!=,>=,<=), reference value of a tracked property and its current value.

Conditions set in pending request objects are to be controlled in the timer [of \\
the class managing pending requests](https://www.mql5.com/en/articles/7481#node04). Activated pending requests are sent to the server from the same class immediately after meeting
all the conditions set in a pending request object.

In the current article, we are going to create and check trading using pending requests — opening positions under certain conditions. We are
going to track only two conditions in the test EA — price and time. Conditions can be set either separately (by price or by time), or jointly (by
price and time).

### Preparing data

As usual, we start by adding indices of new library messages and appropriate texts.

**Write all the necessary message indices in the Datas.mqh**
**file:**

```
//--- CEvent
   MSG_EVN_EVENT,                                     // Event
   MSG_EVN_TYPE,                                      // Event type
```

...

```
//--- CAccount
   MSG_ACC_ACCOUNT,                                   // Account
   MSG_ACC_PROP_LOGIN,                                // Account number
```

...

```
   MSG_LIB_TEXT_REQUEST,                              // Pending request #
   MSG_LIB_TEXT_REQUEST_ACTIVATED,                    // Pending request activated: #
   MSG_LIB_TEXT_REQUEST_DATAS,                        // Trading request parameters
   MSG_LIB_TEXT_PEND_REQUEST_DATAS,                   // Pending trading request parameters
   MSG_LIB_TEXT_PEND_REQUEST_CREATED,                 // Pending request created
   MSG_LIB_TEXT_PEND_REQUEST_DELETED,                 // Removed due to expiration
   MSG_LIB_TEXT_PEND_REQUEST_EXECUTED,                // Removed due to execution
   MSG_LIB_TEXT_PEND_REQUEST_GETTING_FAILED,          // Failed to obtain a pending request object from the list
   MSG_LIB_TEXT_PEND_REQUEST_FAILED_ADD_PARAMS,       // Failed to add request activation parameters. Error:

   MSG_LIB_TEXT_PEND_REQUEST_PRICE_CREATE,            // Price at the moment of request generation
```

...

```
   MSG_LIB_TEXT_PEND_REQUEST_ACTUAL_EXPIRATION,       // Actual order lifetime

   MSG_LIB_TEXT_PEND_REQUEST_NO_FREE_IDS,             // No free IDs to create a pending request
   MSG_LIB_TEXT_PEND_REQUEST_ACTIVATION_TERMS,        // Activation conditions
   MSG_LIB_TEXT_PEND_REQUEST_CRITERION,               // Criterion
   MSG_LIB_TEXT_PEND_REQUEST_ADD_CRITERIONS,          // Added pending request activation conditions

  };
//+------------------------------------------------------------------+
```

**and add message texts corresponding to newly added indices:**

```
//--- CEvent
   {"Событие","Event"},
   {"Тип события","Event's type"},
```

...

```
//--- CAccount
   {"Аккаунт","Account"},
   {"Номер счёта","Account number"},
```

...

```
   {"Отложенный запрос #","Pending request #"},
   {"Активирован отложенный запрос: #","Pending request activated: #"},
   {"Параметры торгового запроса","Trade request parameters"},
   {"Параметры отложенного торгового запроса","Pending trade request parameters"},
   {"Создан отложенный запрос","Pending request created"},
   {"Удалён в связи с окончанием времени его действия","Deleted due to expiration"},
   {"Удалён в связи с его исполнением","Deleted due completed"},
   {"Не удалось получить объект-отложенный запрос из списка","Failed to get pending request object from list"},
   {"Не удалось добавить параметры активации запроса. Ошибка: ","Failed to add request activation parameters. Error: "},

   {"Цена в момент создания запроса","Price at time of request create"},
```

...

```
   {"Фактическое время жизни ордера","Actual of order lifetime"},

   {"Нет свободных идентификаторов для создания отложенного запроса","No free IDs to create a pending request"},
   {"Условия активации","Activation terms"},
   {"Критерий","Criterion"},
   {"Добавлены условия активации отложенного запроса","Pending request activation conditions added"},

  };
//+---------------------------------------------------------------------+
```

Since a single pending request object handles controlled conditions from completely different sources (in this case, these are account,
symbol and account events, then we can add something else), we need a data source whose parameters we track in order to track the activation of
a specified pending request activation condition. When tracking account and symbol parameters, they may have matching property indices
while the properties themselves are completely different. To avoid confusion, we will specify the data source, in which its property
values are tracked.

**In the Defines.mqh file, write the enumeration of pending request activation**
**sources:**

```
//+------------------------------------------------------------------+
//| Pending request type                                             |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_TYPE
  {
   PEND_REQ_TYPE_ERROR=PENDING_REQUEST_ID_TYPE_ERR,         // Pending request created based on the return code or error
   PEND_REQ_TYPE_REQUEST=PENDING_REQUEST_ID_TYPE_REQ,       // Pending request created by request
  };
//+------------------------------------------------------------------+
//| Pending request activation source                                |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_ACTIVATION_SOURCE
  {
   PEND_REQ_ACTIVATION_SOURCE_ACCOUNT,                      // Pending request activated by account data
   PEND_REQ_ACTIVATION_SOURCE_SYMBOL,                       // Pending request activated by symbol data
   PEND_REQ_ACTIVATION_SOURCE_EVENT,                        // Pending request activated by trading event data
  };
//+------------------------------------------------------------------+
//| Integer properties of a pending trading request                  |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_INTEGER
  {
```

**Also, add enumerations of possible criteria used to activate pending requests.**

For the criteria of activating
by account, symbol
and event properties, use separate enumerations:

```
//+------------------------------------------------------------------+
//| Possible criteria for activating requests by account properties  |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_ACTIVATE_BY_ACCOUNT_PROP
  {
   //--- long
   PEND_REQ_ACTIVATE_BY_ACCOUNT_EMPTY                    =  MSG_LIB_PROP_NOT_SET,                     // Value not set
   PEND_REQ_ACTIVATE_BY_ACCOUNT_LEVERAGE                 =  MSG_ACC_PROP_LEVERAGE,                    // Activate by a provided leverage
   PEND_REQ_ACTIVATE_BY_ACCOUNT_LIMIT_ORDERS             =  MSG_ACC_PROP_LIMIT_ORDERS,                // Activate by a maximum allowed number of active pending orders
   PEND_REQ_ACTIVATE_BY_ACCOUNT_TRADE_ALLOWED            =  MSG_ACC_PROP_TRADE_ALLOWED,               // Activate by the permission to trade for the current account from the server side
   PEND_REQ_ACTIVATE_BY_ACCOUNT_TRADE_EXPERT             =  MSG_ACC_PROP_TRADE_EXPERT,                // Activate by the permission to trade for an EA from the server side
   //--- double
   PEND_REQ_ACTIVATE_BY_ACCOUNT_BALANCE                  =  MSG_ACC_PROP_BALANCE,                     // Activate by an account balance in the deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_CREDIT                   =  MSG_ACC_PROP_CREDIT,                      // Activate by credit in a deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_PROFIT                   =  MSG_ACC_PROP_PROFIT,                      // Activate by the current profit on the account in the deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_EQUITY                   =  MSG_ACC_PROP_EQUITY,                      // Sort by an account equity in the deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN                   =  MSG_ACC_PROP_MARGIN,                      // Activate by an account reserved margin in the deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_FREE              =  MSG_ACC_PROP_MARGIN_FREE,                 // Activate by account free funds available for opening a position in the deposit currency
   PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_LEVEL             =  MSG_ACC_PROP_MARGIN_LEVEL,                // Activate by account margin level in %
   PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_INITIAL           =  MSG_ACC_PROP_MARGIN_INITIAL,              // Activate by funds reserved on an account to ensure a guarantee amount for all pending orders
   PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_MAINTENANCE       =  MSG_ACC_PROP_MARGIN_MAINTENANCE,          // Activate by funds reserved on an account to ensure a minimum amount for all open positions
   PEND_REQ_ACTIVATE_BY_ACCOUNT_ASSETS                   =  MSG_ACC_PROP_ASSETS,                      // Activate by the current assets on the account
   PEND_REQ_ACTIVATE_BY_ACCOUNT_LIABILITIES              =  MSG_ACC_PROP_LIABILITIES,                 // Activate by the current liabilities on the account
   PEND_REQ_ACTIVATE_BY_ACCOUNT_COMMISSION_BLOCKED       =  MSG_ACC_PROP_COMMISSION_BLOCKED           // Activate by the current amount of blocked commissions on the account
  };
//+------------------------------------------------------------------+
//| Possible criteria for activating requests by symbol properties   |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_ACTIVATE_BY_SYMBOL_PROP
  {
   PEND_REQ_ACTIVATE_BY_SYMBOL_EMPTY                     =  MSG_LIB_PROP_NOT_SET,                     // Value not set
   //--- double
   PEND_REQ_ACTIVATE_BY_SYMBOL_BID                       =  MSG_LIB_PROP_BID,                         // Activate by Bid - the best price at which a symbol can be sold
   PEND_REQ_ACTIVATE_BY_SYMBOL_ASK                       =  MSG_LIB_PROP_ASK,                         // Activate by Ask - best price, at which an instrument can be bought
   PEND_REQ_ACTIVATE_BY_SYMBOL_LAST                      =  MSG_LIB_PROP_LAST,                        // Activate by the last deal price
   //--- long
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_DEALS             =  MSG_SYM_PROP_SESSION_DEALS,               // Activate by number of deals in the current session
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_BUY_ORDERS        =  MSG_SYM_PROP_SESSION_BUY_ORDERS,          // Activate by number of Buy orders at the moment
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_SELL_ORDERS       =  MSG_SYM_PROP_SESSION_SELL_ORDERS,         // Activate by number of Sell orders at the moment
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUME                    =  MSG_SYM_PROP_VOLUME,                      // Activate by the last deal volume
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMEHIGH                =  MSG_SYM_PROP_VOLUMEHIGH,                  // Activate by maximum Volume per day
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMELOW                 =  MSG_SYM_PROP_VOLUMELOW,                   // Activate by minimum Volume per day
   PEND_REQ_ACTIVATE_BY_SYMBOL_TIME                      =  MSG_SYM_PROP_TIME,                        // Activate by the last quote time
   PEND_REQ_ACTIVATE_BY_SYMBOL_SPREAD                    =  MSG_SYM_PROP_SPREAD,                      // Activate by spread in points
   PEND_REQ_ACTIVATE_BY_SYMBOL_START_TIME                =  MSG_SYM_PROP_START_TIME,                  // Activate by an instrument trading start date (usually used for futures)
   PEND_REQ_ACTIVATE_BY_SYMBOL_EXPIRATION_TIME           =  MSG_SYM_PROP_EXPIRATION_TIME,             // Activate by an instrument trading completion date (usually used for futures)
   PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_STOPS_LEVEL         =  MSG_SYM_PROP_TRADE_STOPS_LEVEL,           // Activate by the minimum indent from the current close price (in points) for setting Stop orders
   PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_FREEZE_LEVEL        =  MSG_SYM_PROP_TRADE_FREEZE_LEVEL,          // Activate by trade operation freeze distance (in points)
   //--- double
   PEND_REQ_ACTIVATE_BY_SYMBOL_BIDHIGH                   =  MSG_SYM_PROP_BIDHIGH,                     // Activate by a maximum Bid of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_BIDLOW                    =  MSG_SYM_PROP_BIDLOW,                      // Activate by a minimum Bid of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_ASKHIGH                   =  MSG_SYM_PROP_ASKHIGH,                     // Activate by a maximum Ask of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_ASKLOW                    =  MSG_SYM_PROP_ASKLOW,                      // Activate by a minimum Ask of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_LASTHIGH                  =  MSG_SYM_PROP_LASTHIGH,                    // Activate by the maximum Last of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_LASTLOW                   =  MSG_SYM_PROP_LASTLOW,                     // Activate by the minimum Last of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUME_REAL               =  MSG_SYM_PROP_VOLUME_REAL,                 // Activate by Volume of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMEHIGH_REAL           =  MSG_SYM_PROP_VOLUMEHIGH_REAL,             // Activate by a maximum Volume of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMELOW_REAL            =  MSG_SYM_PROP_VOLUMELOW_REAL,              // Activate by a minimum Volume of the day
   PEND_REQ_ACTIVATE_BY_SYMBOL_OPTION_STRIKE             =  MSG_SYM_PROP_OPTION_STRIKE,               // Activate by an option execution price
   PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_ACCRUED_INTEREST    =  MSG_SYM_PROP_TRADE_ACCRUED_INTEREST,      // Activate by an accrued interest
   PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_FACE_VALUE          =  MSG_SYM_PROP_TRADE_FACE_VALUE,            // Activate by a face value – initial bond value set by an issuer
   PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_LIQUIDITY_RATE      =  MSG_SYM_PROP_TRADE_LIQUIDITY_RATE,        // Activate by a liquidity rate – the share of an asset that can be used for a margin
   PEND_REQ_ACTIVATE_BY_SYMBOL_SWAP_LONG                 =  MSG_SYM_PROP_SWAP_LONG,                   // Activate by a long swap value
   PEND_REQ_ACTIVATE_BY_SYMBOL_SWAP_SHORT                =  MSG_SYM_PROP_SWAP_SHORT,                  // Activate by a short swap value
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_VOLUME            =  MSG_SYM_PROP_SESSION_VOLUME,              // Activate by a summary volume of the current session deals
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_TURNOVER          =  MSG_SYM_PROP_SESSION_TURNOVER,            // Activate by a summary turnover of the current session
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_INTEREST          =  MSG_SYM_PROP_SESSION_INTEREST,            // Activate by a summary open interest
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_BUY_ORDERS_VOLUME =  MSG_SYM_PROP_SESSION_BUY_ORDERS_VOLUME,   // Activate by the current volume of Buy orders
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME=  MSG_SYM_PROP_SESSION_SELL_ORDERS_VOLUME,  // Activate by the current volume of Sell orders
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_OPEN              =  MSG_SYM_PROP_SESSION_OPEN,                // Activate by an open price of the current session
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_CLOSE             =  MSG_SYM_PROP_SESSION_CLOSE,               // Activate by a close price of the current session
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_AW                =  MSG_SYM_PROP_SESSION_AW,                  // Activate by an average weighted session price
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_SETTLEMENT  =  MSG_SYM_PROP_SESSION_PRICE_SETTLEMENT,    // Activate by a settlement price of the current session
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_LIMIT_MIN   =  MSG_SYM_PROP_SESSION_PRICE_LIMIT_MIN,     // Activate by a minimum session price
   PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_LIMIT_MAX   =  MSG_SYM_PROP_SESSION_PRICE_LIMIT_MAX,     // Activate by a maximum session price
  };
//+------------------------------------------------------------------+
//| Possible criteria for activating requests by events              |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_ACTIVATE_BY_EVENT
  {
   PEND_REQ_ACTIVATE_BY_EVENT_EMPTY                               =  MSG_LIB_PROP_NOT_SET,                        // Value not set
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_OPENED                     =  MSG_EVN_STATUS_MARKET_POSITION,              // Position opened
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED                     =  MSG_EVN_STATUS_HISTORY_POSITION,             // Position closed
   PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_PLASED                =  MSG_EVN_PENDING_ORDER_PLASED,                // Pending order placed
   PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_REMOVED               =  MSG_EVN_PENDING_ORDER_REMOVED,               // Pending order removed
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CREDIT                      =  MSG_EVN_ACCOUNT_CREDIT,                      // Accruing credit (3)
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CHARGE                      =  MSG_EVN_ACCOUNT_CHARGE,                      // Additional charges
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CORRECTION                  =  MSG_EVN_ACCOUNT_CORRECTION,                  // Correcting entry
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BONUS                       =  MSG_EVN_ACCOUNT_BONUS,                       // Charging bonuses
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION                   =  MSG_EVN_ACCOUNT_COMISSION,                   // Additional commissions
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_DAILY             =  MSG_EVN_ACCOUNT_COMISSION_DAILY,             // Commission charged at the end of a day
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_MONTHLY           =  MSG_EVN_ACCOUNT_COMISSION_MONTHLY,           // Commission charged at the end of a trading month
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_AGENT_DAILY       =  MSG_EVN_ACCOUNT_COMISSION_AGENT_DAILY,       // Agent commission charged at the end of a trading day
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY     =  MSG_EVN_ACCOUNT_COMISSION_AGENT_MONTHLY,     // Agent commission charged at the end of a month
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_INTEREST                    =  MSG_EVN_ACCOUNT_INTEREST,                    // Accruing interest on free funds
   PEND_REQ_ACTIVATE_BY_EVENT_BUY_CANCELLED                       =  MSG_EVN_BUY_CANCELLED,                       // Canceled buy deal
   PEND_REQ_ACTIVATE_BY_EVENT_SELL_CANCELLED                      =  MSG_EVN_SELL_CANCELLED,                      // Canceled sell deal
   PEND_REQ_ACTIVATE_BY_EVENT_DIVIDENT                            =  MSG_EVN_DIVIDENT,                            // Accruing dividends
   PEND_REQ_ACTIVATE_BY_EVENT_DIVIDENT_FRANKED                    =  MSG_EVN_DIVIDENT_FRANKED,                    // Accrual of franked dividend
   PEND_REQ_ACTIVATE_BY_EVENT_TAX                                 =  MSG_EVN_TAX,                                 // Tax accrual
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BALANCE_REFILL              =  MSG_EVN_BALANCE_REFILL,                      // Replenishing account balance
   PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BALANCE_WITHDRAWAL          =  MSG_EVN_BALANCE_WITHDRAWAL,                  // Withdrawing funds from an account
   PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_ACTIVATED             =  MSG_EVN_ACTIVATED_PENDING,                   // Pending order activated by price
   PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL     =  MSG_EVN_ACTIVATED_PENDING_PARTIALLY,         // Pending order partially activated by price
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_OPENED_PARTIAL             =  MSG_EVN_POSITION_OPENED_PARTIALLY,           // Position opened partially
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL             =  MSG_EVN_POSITION_CLOSED_PARTIALLY,           // Position closed partially
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_POS              =  MSG_EVN_POSITION_CLOSED_BY_POS,              // Position closed by an opposite one
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_POS      =  MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_POS,    // Position partially closed by an opposite one
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_SL               =  MSG_EVN_POSITION_CLOSED_BY_SL,               // Position closed by StopLoss
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_TP               =  MSG_EVN_POSITION_CLOSED_BY_TP,               // Position closed by TakeProfit
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_SL       =  MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_SL,     // Position closed partially by StopLoss
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_TP       =  MSG_EVN_POSITION_CLOSED_PARTIALLY_BY_TP,     // Position closed partially by TakeProfit
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_MARKET         =  MSG_EVN_POSITION_REVERSED_BY_MARKET,         // Position reversal by a new deal (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_PENDING        =  MSG_EVN_POSITION_REVERSED_BY_PENDING,        // Position reversal by activating a pending order (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL =  MSG_EVN_POSITION_REVERSE_PARTIALLY,          // Position reversal by partial market order execution (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOLUME_ADD_BY_MARKET       =  MSG_EVN_POSITION_VOLUME_ADD_BY_MARKET,       // Added volume to a position by a new deal (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOLUME_ADD_BY_PENDING      =  MSG_EVN_POSITION_VOLUME_ADD_BY_PENDING,      // Added volume to a position by activating a pending order (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE                  =  MSG_EVN_MODIFY_ORDER_PRICE,                  // Order price change
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_SL               =  MSG_EVN_MODIFY_ORDER_PRICE_SL,               // Changing order and StopLoss price
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_TP               =  MSG_EVN_MODIFY_ORDER_PRICE_TP,               // Order and TakeProfit price change
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_SL_TP            =  MSG_EVN_MODIFY_ORDER_PRICE_SL_TP,            // Changing order, StopLoss and TakeProfit price
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_SL_TP                  =  MSG_EVN_MODIFY_ORDER_SL_TP,                  // Changing order's StopLoss and TakeProfit price
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_SL                     =  MSG_EVN_MODIFY_ORDER_SL,                     // Modify StopLoss order
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_TP                     =  MSG_EVN_MODIFY_ORDER_TP,                     // Modify TakeProfit order
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_SL_TP               =  MSG_EVN_MODIFY_POSITION_SL_TP,               // Change position's StopLoss and TakeProfit
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_SL                  =  MSG_EVN_MODIFY_POSITION_SL,                  // Modify position's StopLoss
   PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_TP                  =  MSG_EVN_MODIFY_POSITION_TP,                  // Modify position's TakeProfit
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOL_ADD_BY_MARKET_PARTIAL  =  MSG_EVN_REASON_ADD_PARTIALLY,                // Added volume to a position by partial execution of a market order (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOL_ADD_BY_PENDING_PARTIAL =  MSG_EVN_REASON_ADD_BY_PENDING_PARTIALLY,     // Added volume to a position by partial activation of a pending order (netting)
   PEND_REQ_ACTIVATE_BY_EVENT_TRIGGERED_STOP_LIMIT_ORDER          =  MSG_EVN_REASON_STOPLIMIT_TRIGGERED,          // StopLimit order activation
   PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL=  MSG_EVN_REASON_REVERSE_BY_PENDING_PARTIALLY, // Position reversal by activating a pending order (netting)
  };
//+------------------------------------------------------------------+
```

The values of enumeration constants are equal to the constant values of text messages of the appropriate symbol, account and event
properties. This relieves us from the necessity to additionally identify a described constant as belonging to a symbol, account or event
when displaying messages in the journal. Instead, we are going to simply use the index of a constant itself to display a message.

Using three different enumerations of activation conditions, we can finally set any combination of constants out of three enumerations for
compiling a required pending request activation criteria.

**Add the function returning the description of a comparison type to the DELib.mqh service function file:**

```
//+------------------------------------------------------------------+
//| Return the comparison type description                           |
//+------------------------------------------------------------------+
string ComparisonTypeDescription(const ENUM_COMPARER_TYPE type)
  {
   switch((int)type)
     {
      case EQUAL           :  return " == ";
      case MORE            :  return " > ";
      case LESS            :  return " < ";
      case EQUAL_OR_MORE   :  return " >= ";
      case EQUAL_OR_LESS   :  return " <= ";
      default              :  return " != ";
     }
  }
//+------------------------------------------------------------------+
```

Names of enumeration constants featuring "STOP\_LOSS" and "TAKE\_PROFIT" strings have been changed in many library files. The occurrences of
these strings have been replaced with "SL" and "TP", respectively.

### Pending request object created on demand

[The basic object of the abstract pending request](https://www.mql5.com/en/articles/7454#node02) is now
inherited [from the base object of all library objects](https://www.mql5.com/en/articles/7071).

Include
the base object file of all library objects to the CPendRequest class file and make
the class inherit the base object:

```
//+------------------------------------------------------------------+
//|                                                  PendRequest.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
#property strict    // Necessary for mql4
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Object.mqh>
#include "..\..\Services\DELib.mqh"
#include "..\..\Objects\BaseObj.mqh"
//+------------------------------------------------------------------+
//| Abstract pending trading request class                           |
//+------------------------------------------------------------------+
class CPendRequest : public CBaseObj
  {
```

**In the private section of the class, declare the array for storing data**
**using tracked pending request activation criterion:**

```
//+------------------------------------------------------------------+
//| Abstract pending trading request class                           |
//+------------------------------------------------------------------+
class CPendRequest : public CBaseObj
  {
private:
   MqlTradeRequest   m_request;                                            // Trade request structure
   CPause            m_pause;                                              // Pause class object

/* Data on a pending request activation in the array:
   The first dimension contains the activation criteria number
   The second one features:

   m_activated_control[criterion number][0] - controlled property source
   m_activated_control[criterion number][1] - controlled property
   m_activated_control[criterion number][2] - type of comparing a controlled property with an actual value (=,>,<,!=,>=,<=)
   m_activated_control[criterion number][3] - property reference value for activation
   m_activated_control[criterion number][4] - actual property value
*/
   double            m_activated_control[][5];                             // Array of reference values of the pending request activation criterion

//--- Copy trading request data
   void              CopyRequest(const MqlTradeRequest &request);
```

In the same private section, add the methods returning a magic number
and ID set in the EA settings, as well as IDs
of the first and second groups. Also, declare the method for
returning the flag of the successful check of a controlled property with its actual value, the
method of comparing two controlled property values and the method
returning the number of decimal places for the tracked property for correct display of values in the journal:

```
//--- Return (1) the magic number, ID of the (2) magic number, (3) the first group, (4) the second group,
//--- (5) hedging account flag, (6) flag indicating the real property is equal to the value
   ulong             GetMagic(void)                                        const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC);                             }
   ushort            GetMagicID(void)                                      const { return CBaseObj::GetMagicID((uint)this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC)); }
   uchar             GetGroupID1(void)                                     const { return CBaseObj::GetGroupID1((uint)this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC));}
   uchar             GetGroupID2(void)                                     const { return CBaseObj::GetGroupID2((uint)this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC));}
   bool              IsHedge(void)                                         const { return this.m_is_hedge;                                                           }
   bool              IsEqualByMode(const int mode,const double value)      const;
   bool              IsEqualByMode(const int mode,const long value)        const;
//--- Return the flags indicating the pending request has completed changing each of the order/position parameters
   bool              IsCompletedVolume(void)                               const;
   bool              IsCompletedPrice(void)                                const;
   bool              IsCompletedStopLimit(void)                            const;
   bool              IsCompletedStopLoss(void)                             const;
   bool              IsCompletedTakeProfit(void)                           const;
   bool              IsCompletedTypeFilling(void)                          const;
   bool              IsCompletedTypeTime(void)                             const;
   bool              IsCompletedExpiration(void)                           const;

//--- Return the flag of a successful check of a controlled object property and the appropriate actual property
   bool              IsComparisonCompleted(const uint index)               const;
//--- Compare two data source values by a comparison type
   bool              IsCompared(const double actual_value,const double control_value,const ENUM_COMPARER_TYPE compare) const;

//--- Return the number of decimal places of a controlled property
   int               DigitsControlledValue(const uint index)               const;

public:
```

The methods of returning magic numberand
group IDs use same-name methods of the CBaseObj parent object we have inherited the base abstract pending request object from.

In the block of methods for a simplified access to request object properties of the public class section, add
declarations of all the necessary public methods we are going to consider further:

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the request object properties  |
//+------------------------------------------------------------------+
//--- Return (1) request structure, (2) status, (3) type, (4) price at the moment of the request generation,
//--- (5) request generation time, (6) next attempt activation time,
//--- (7) waiting time between requests, (8) current attempt index,
//--- (9) number of attempts, (10) request ID
//--- (11) result a request is based on,
//--- (12) order ticket, (13) position ticket, (14) trading operation type
   MqlTradeRequest      MqlRequest(void)                                   const { return this.m_request;                                                }
   ENUM_PEND_REQ_STATUS Status(void)                                       const { return (ENUM_PEND_REQ_STATUS)this.GetProperty(PEND_REQ_PROP_STATUS);  }
   ENUM_PEND_REQ_TYPE   TypeRequest(void)                                  const { return (ENUM_PEND_REQ_TYPE)this.GetProperty(PEND_REQ_PROP_TYPE);      }
   double               PriceCreate(void)                                  const { return this.GetProperty(PEND_REQ_PROP_PRICE_CREATE);                  }
   ulong                TimeCreate(void)                                   const { return this.GetProperty(PEND_REQ_PROP_TIME_CREATE);                   }
   ulong                TimeActivate(void)                                 const { return this.GetProperty(PEND_REQ_PROP_TIME_ACTIVATE);                 }
   ulong                WaitingMSC(void)                                   const { return this.GetProperty(PEND_REQ_PROP_WAITING);                       }
   uchar                CurrentAttempt(void)                               const { return (uchar)this.GetProperty(PEND_REQ_PROP_CURRENT_ATTEMPT);        }
   uchar                TotalAttempts(void)                                const { return (uchar)this.GetProperty(PEND_REQ_PROP_TOTAL);                  }
   uchar                ID(void)                                           const { return (uchar)this.GetProperty(PEND_REQ_PROP_ID);                     }
   int                  Retcode(void)                                      const { return (int)this.GetProperty(PEND_REQ_PROP_RETCODE);                  }
   ulong                Order(void)                                        const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER);                 }
   ulong                Position(void)                                     const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION);              }
   ENUM_TRADE_REQUEST_ACTIONS Action(void)                                 const { return (ENUM_TRADE_REQUEST_ACTIONS)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION);   }

//--- Return the actual (1) volume, (2) order, (3) limit order,
//--- (4) stoploss order and (5) takeprofit order prices, (6) order filling type,
//--- (7) order expiration type and (8) order lifetime
   double               ActualVolume(void)                                 const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_VOLUME);                 }
   double               ActualPrice(void)                                  const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_PRICE);                  }
   double               ActualStopLimit(void)                              const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_STOPLIMIT);              }
   double               ActualSL(void)                                     const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_SL);                     }
   double               ActualTP(void)                                     const { return this.GetProperty(PEND_REQ_PROP_ACTUAL_TP);                     }
   ENUM_ORDER_TYPE_FILLING ActualTypeFilling(void)                         const { return (ENUM_ORDER_TYPE_FILLING)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_FILLING); }
   ENUM_ORDER_TYPE_TIME ActualTypeTime(void)                               const { return (ENUM_ORDER_TYPE_TIME)this.GetProperty(PEND_REQ_PROP_ACTUAL_TYPE_TIME);       }
   datetime             ActualExpiration(void)                             const { return (datetime)this.GetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION);   }

//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) ID,
//--- (8) order ticket, (9) position ticket, (10) pending request type
   void                 SetPriceCreate(const double price)                       { this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);                   }
   void                 SetTimeCreate(const ulong time)
                          {
                           this.SetProperty(PEND_REQ_PROP_TIME_CREATE,time);
                           this.m_pause.SetTimeBegin(time);
                          }
   void                 SetTimeActivate(const ulong time)                        { this.SetProperty(PEND_REQ_PROP_TIME_ACTIVATE,time);                   }
   void                 SetWaitingMSC(const ulong miliseconds)
                          {
                           this.SetProperty(PEND_REQ_PROP_WAITING,miliseconds);
                           this.m_pause.SetWaitingMSC(miliseconds);
                          }
   void                 SetCurrentAttempt(const uchar number)                    { this.SetProperty(PEND_REQ_PROP_CURRENT_ATTEMPT,number);               }
   void                 SetTotalAttempts(const uchar number)                     { this.SetProperty(PEND_REQ_PROP_TOTAL,number);                         }
   void                 SetID(const uchar id)                                    { this.SetProperty(PEND_REQ_PROP_ID,id);                                }
   void                 SetOrder(const ulong ticket)                             { this.SetProperty(PEND_REQ_PROP_MQL_REQ_ORDER,ticket);                 }
   void                 SetPosition(const ulong ticket)                          { this.SetProperty(PEND_REQ_PROP_MQL_REQ_POSITION,ticket);              }
   void                 SetTypeRequest(const ENUM_PEND_REQ_TYPE type)            { this.SetProperty(PEND_REQ_PROP_TYPE,type);                            }

//--- Set the actual (1) volume, (2) order, (3) limit order,
//--- (4) stoploss order and (5) takeprofit order prices, (6) order filling type,
//--- (7) order expiration type and (8) order lifetime
   void                 SetActualVolume(const double volume)                     { this.SetProperty(PEND_REQ_PROP_ACTUAL_VOLUME,volume);                 }
   void                 SetActualPrice(const double price)                       { this.SetProperty(PEND_REQ_PROP_ACTUAL_PRICE,price);                   }
   void                 SetActualStopLimit(const double price)                   { this.SetProperty(PEND_REQ_PROP_ACTUAL_STOPLIMIT,price);               }
   void                 SetActualSL(const double price)                          { this.SetProperty(PEND_REQ_PROP_ACTUAL_SL,price);                      }
   void                 SetActualTP(const double price)                          { this.SetProperty(PEND_REQ_PROP_ACTUAL_TP,price);                      }
   void                 SetActualTypeFilling(const ENUM_ORDER_TYPE_FILLING type) { this.SetProperty(PEND_REQ_PROP_ACTUAL_TYPE_FILLING,type);             }
   void                 SetActualTypeTime(const ENUM_ORDER_TYPE_TIME type)       { this.SetProperty(PEND_REQ_PROP_ACTUAL_TYPE_TIME,type);                }
   void                 SetActualExpiration(const datetime expiration)           { this.SetProperty(PEND_REQ_PROP_ACTUAL_EXPIRATION,expiration);         }

//--- Set a controlled property and a comparison method for a request activation criteria data by its index - both the actual one and the one in the object of
//--- account, symbol or trading event property value (depends on 'source' value) for activating a pending request
   void                 SetNewActivationProperties(const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                                   const int property,
                                                   const double control_value,
                                                   const ENUM_COMPARER_TYPE comparer_type,
                                                   const double actual_value);

//--- Set a (1) controlled property, (2) comparison type, (3) object value and
//--- (4) actual controlled property value for activating a pending request
   bool                 SetActivationProperty(const uint index,const ENUM_PEND_REQ_ACTIVATION_SOURCE source,const int property);
   bool                 SetActivationComparerType(const uint index,const ENUM_COMPARER_TYPE comparer_type);
   bool                 SetActivationControlValue(const uint index,const double value);
   bool                 SetActivationActualValue(const uint index,const double value);

//--- Return (1) a pending request activation source, (2) controlled property, (3) comparison type,
//---  (4) object value,(5) actual controlled property value for activating a pending request
   ENUM_PEND_REQ_ACTIVATION_SOURCE GetActivationSource(const uint index)         const;
   int                  GetActivationProperty(const uint index)                  const;
   ENUM_COMPARER_TYPE   GetActivationComparerType(const uint index)              const;
   double               GetActivationControlValue(const uint index)              const;
   double               GetActivationActualValue(const uint index)               const;

//--- Return the flag of a successful check of all controlled object properties and the appropriate actual properties
   bool                 IsAllComparisonCompleted(void)  const;
```

The **SetTypeRequest()** method sets the "pending request type" property passed to the type method. "A pending request based on an
error code" or "a pending request created by request" can be used as a type. A pending request type is set in an object automatically within the
class constructor depending on the "error code" parameter. If the code is equal to zero, this is a pending request created by a program
request. Thus, the method is not used anywhere now. It is created in case you suddenly need to quickly change a pending request type from the
outside (personally, I have not been able to come up with a need for this yet).

Add declarations of the methods returning descriptions of controlled
properties to the appropriate block of methods:

```
//+------------------------------------------------------------------+
//| Descriptions of request object properties                        |
//+------------------------------------------------------------------+
//--- Get description of a request (1) integer, (2) real and (3) string property
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_INTEGER property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_DOUBLE property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_STRING property);

//--- Return the description of a (1) controlled property, (2) comparison type, (3) controlled property value in the object,
//--- (4) actual controlled property value for activating a pending request, (5) total number of activation conditions
   string               GetActivationPropertyDescription(const uint index)       const;
   string               GetActivationComparerTypeDescription(const uint index)   const;
   string               GetActivationControlValueDescription(const uint index)   const;
   string               GetActivationActualValueDescription(const uint index)    const;
   uint                 GetActivationCriterionTotal(void)                        const { return ::ArrayRange(this.m_activated_control,0); }

//--- Return the names of pending request object parameters
   string               StatusDescription(void)                const;
   string               TypeRequestDescription(void)           const;
   string               IDDescription(void)                    const;
   string               RetcodeDescription(void)               const;
   string               TimeCreateDescription(void)            const;
   string               TimeActivateDescription(void)          const;
   string               TimeWaitingDescription(void)           const;
   string               CurrentAttemptDescription(void)        const;
   string               TotalAttemptsDescription(void)         const;
   string               PriceCreateDescription(void)           const;

   string               TypeFillingActualDescription(void)     const;
   string               TypeTimeActualDescription(void)        const;
   string               ExpirationActualDescription(void)      const;
   string               VolumeActualDescription(void)          const;
   string               PriceActualDescription(void)           const;
   string               StopLimitActualDescription(void)       const;
   string               StopLossActualDescription(void)        const;
   string               TakeProfitActualDescription(void)      const;

//--- Return the names of trading request structures parameters in the request object
   string               MqlReqActionDescription(void)          const;
   string               MqlReqMagicDescription(void)           const;
   string               MqlReqOrderDescription(void)           const;
   string               MqlReqSymbolDescription(void)          const;
   string               MqlReqVolumeDescription(void)          const;
   string               MqlReqPriceDescription(void)           const;
   string               MqlReqStopLimitDescription(void)       const;
   string               MqlReqStopLossDescription(void)        const;
   string               MqlReqTakeProfitDescription(void)      const;
   string               MqlReqDeviationDescription(void)       const;
   string               MqlReqTypeOrderDescription(void)       const;
   string               MqlReqTypeFillingDescription(void)     const;
   string               MqlReqTypeTimeDescription(void)        const;
   string               MqlReqExpirationDescription(void)      const;
   string               MqlReqCommentDescription(void)         const;
   string               MqlReqPositionDescription(void)        const;
   string               MqlReqPositionByDescription(void)      const;

//--- Display (1) description of request properties (full_prop=true - all properties, false - only supported ones),
//--- (2) request activation parameters, (3) short message about the request, (4) short request name (3 and 4 - implementation in the class descendants)
   void                 Print(const bool full_prop=false);
   void                 PrintActivations(void);
   virtual void         PrintShort(void){;}
   virtual string       Header(void){return NULL;}
  };
//+------------------------------------------------------------------+
```

The **GetActivationCriterionTotal()** method returns the size of the first dimension of the activation conditions data array, in
other words, the number of activation conditions in the pending request object.

In the class constructor, set the zero size for the activation conditions
data array in the first dimension:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPendRequest::CPendRequest(const ENUM_PEND_REQ_STATUS status,
                           const uchar id,
                           const double price,
                           const ulong time,
                           const MqlTradeRequest &request,
                           const int retcode)
  {
   this.CopyRequest(request);
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_digits=(int)::SymbolInfoInteger(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL),SYMBOL_DIGITS);
   int dg=(int)DigitsLots(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL));
   this.m_digits_lot=(dg==0 ? 1 : dg);
   this.SetProperty(PEND_REQ_PROP_STATUS,status);
   this.SetProperty(PEND_REQ_PROP_ID,id);
   this.SetProperty(PEND_REQ_PROP_RETCODE,retcode);
   this.SetProperty(PEND_REQ_PROP_TYPE,this.GetProperty(PEND_REQ_PROP_RETCODE)>0 ? PEND_REQ_TYPE_ERROR : PEND_REQ_TYPE_REQUEST);
   this.SetProperty(PEND_REQ_PROP_TIME_CREATE,time);
   this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);
   this.m_pause.SetTimeBegin(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   this.m_pause.SetWaitingMSC(this.GetProperty(PEND_REQ_PROP_WAITING));
   ::ArrayResize(this.m_activated_control,0,10);
  }
//+------------------------------------------------------------------+
```

The size of the activation conditions data array changes automatically when adding each successive activation condition.

In the method displaying the full list of pending request object data, add
display of the activation condition list after displaying all its properties:

```
//+------------------------------------------------------------------+
//| Display the pending request properties in the journal            |
//+------------------------------------------------------------------+
void CPendRequest::Print(const bool full_prop=false)
  {
   int header_code=
     (
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_OPEN   ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN   :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_CLOSE  ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE  :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_SLTP   ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP   :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_PLACE  ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE  :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_REMOVE ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_MODIFY ? MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY :
      WRONG_VALUE
     );
   ::Print("============= \"",CMessage::Text(header_code),"\" =============");
   int beg=0, end=PEND_REQ_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_INTEGER prop=(ENUM_PEND_REQ_PROP_INTEGER)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=PEND_REQ_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_DOUBLE prop=(ENUM_PEND_REQ_PROP_DOUBLE)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   ::Print("------");
   beg=end; end+=PEND_REQ_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_STRING prop=(ENUM_PEND_REQ_PROP_STRING)i;
      if(!full_prop && !this.SupportProperty(prop)) continue;
      ::Print(this.GetPropertyDescription(prop));
     }
   this.PrintActivations();
   ::Print("================== ",CMessage::Text(MSG_LIB_PARAMS_LIST_END),": \"",CMessage::Text(header_code),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

Implementing the method displaying pending request activation conditions to the journal:

```
//+------------------------------------------------------------------+
//| Display request activation parameters in the journal             |
//+------------------------------------------------------------------+
void CPendRequest::PrintActivations(void)
  {
   //--- Get the size of the activation conditions data array's first dimension,
   //--- if it exceeds zero, send all data written in the data array to the journal
   int range=::ArrayRange(this.m_activated_control,0);
   if(range>0)
     {
      ::Print("--- ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ACTIVATION_TERMS)," ---");
      for(int i=0;i<range;i++)
        {
         ENUM_PEND_REQ_ACTIVATION_SOURCE source=(ENUM_PEND_REQ_ACTIVATION_SOURCE)this.m_activated_control[i][0];
         string type=
           (
            source==PEND_REQ_ACTIVATION_SOURCE_ACCOUNT   ?  CMessage::Text(MSG_ACC_ACCOUNT)     :
            source==PEND_REQ_ACTIVATION_SOURCE_SYMBOL    ?  CMessage::Text(MSG_LIB_PROP_SYMBOL) :
            source==PEND_REQ_ACTIVATION_SOURCE_EVENT     ?  CMessage::Text(MSG_EVN_EVENT)       :
            ""
           );
         ::Print(" - ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CRITERION)," #",string(i+1),". ",type,": ",this.GetActivationPropertyDescription(i));
        }
     }
   ::Print("");
  }
//+------------------------------------------------------------------+
```

The method for creating a new condition for activating a pending request in the activation conditions data array:

```
//+------------------------------------------------------------------+
//| Set a controlled property, values                                |
//| and comparison method for activating a pending request           |
//+------------------------------------------------------------------+
void CPendRequest::SetNewActivationProperties(const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                              const int property,
                                              const double control_value,
                                              const ENUM_COMPARER_TYPE comparer_type,
                                              const double actual_value)
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if(::ArrayResize(this.m_activated_control,range+1,10)==WRONG_VALUE)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_FAILED_ADD_PARAMS));
      return;
     }
   this.m_activated_control[range][0]=source;
   this.m_activated_control[range][1]=property;
   this.m_activated_control[range][2]=comparer_type;
   this.m_activated_control[range][3]=control_value;
   this.m_activated_control[range][4]=actual_value;
  }
//+---------------------------------------------------------------------+
```

Activation data source, activation
condition, controlled and actual
activation condition values, as well as comparison method are
passed to the method.

The
size of the activation conditions data array is increased by 1 and all the
necessary data in the array are filled with values passed in the method inputs. The method should be used only to add a new activation
condition.

The following methods are used to correct activation conditions already existing in the request object:

```
//+---------------------------------------------------------------------+
//| Set a controlled property to activate a request                     |
//+---------------------------------------------------------------------+
bool CPendRequest::SetActivationProperty(const uint index,const ENUM_PEND_REQ_ACTIVATION_SOURCE source,const int property)
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return false;
     }
   this.m_activated_control[index][0]=source;
   this.m_activated_control[index][1]=property;
   return true;
  }
//+------------------------------------------------------------------+
//| Set the object property comparison type                          |
//| with the actual one for a pending request activation             |
//+------------------------------------------------------------------+
bool CPendRequest::SetActivationComparerType(const uint index,const ENUM_COMPARER_TYPE comparer_type)
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return false;
     }
   this.m_activated_control[index][2]=comparer_type;
   return true;
  }
//+------------------------------------------------------------------+
//| Set the controlled property                                      |
//| value for activating a pending request                           |
//+------------------------------------------------------------------+
bool CPendRequest::SetActivationControlValue(const uint index,const double value)
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return false;
     }
   this.m_activated_control[index][3]=value;
   return true;
  }
//+------------------------------------------------------------------+
//| Set the actual value                                             |
//| of a controlled property in the request object                   |
//+------------------------------------------------------------------+
bool CPendRequest::SetActivationActualValue(const uint index,const double value)
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return false;
     }
   this.m_activated_control[index][4]=value;
   return true;
  }
//+------------------------------------------------------------------+
```

The method of setting the **SetActivationProperty()** controlled property receives the index and two condition parameters —
condition source (symbol, account or event) and the activation condition itself (from the appropriate enumerations considered above)
since the activation condition consists of the two parameters — property change source and type. Other methods of setting activation
values receive only index and value.

Activation condition number is used as an index. If there is only one condition, the index
should be equal to zero. In case of two conditions, the index should be equal to 0 or 1 depending on what condition we want to change, etc. When
passing the index going beyond the array first dimension, the invalid index entry appears in the journal and false
is returned.

The methods returning activation conditions parameters:

```
//+------------------------------------------------------------------+
//| Return a pending request activation source                       |
//+------------------------------------------------------------------+
ENUM_PEND_REQ_ACTIVATION_SOURCE CPendRequest::GetActivationSource(const uint index) const
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return WRONG_VALUE;
     }
   return (ENUM_PEND_REQ_ACTIVATION_SOURCE)this.m_activated_control[index][0];
  }
//+------------------------------------------------------------------+
//| Return a controlled property to activate a request               |
//+------------------------------------------------------------------+
int CPendRequest::GetActivationProperty(const uint index) const
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return WRONG_VALUE;
     }
   return (int)this.m_activated_control[index][1];
  }
//+------------------------------------------------------------------+
//| Return the object property comparison type                       |
//| with the actual one for a pending request activation             |
//+------------------------------------------------------------------+
ENUM_COMPARER_TYPE CPendRequest::GetActivationComparerType(const uint index) const
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return WRONG_VALUE;
     }
   return (ENUM_COMPARER_TYPE)this.m_activated_control[index][2];
  }
//+------------------------------------------------------------------+
//| Return the controlled property                                   |
//| value for activating a pending request                           |
//+------------------------------------------------------------------+
double CPendRequest::GetActivationControlValue(const uint index) const
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return EMPTY_VALUE;
     }
   return this.m_activated_control[index][3];
  }
//+------------------------------------------------------------------+
//| Return the actual value                                          |
//| of a controlled property in the request object                   |
//+------------------------------------------------------------------+
double CPendRequest::GetActivationActualValue(const uint index) const
  {
   int range=::ArrayRange(this.m_activated_control,0);
   if((int)index>range-1 || range==0)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(4002));
      return EMPTY_VALUE;
     }
   return this.m_activated_control[index][4];
  }
//+------------------------------------------------------------------+
```

Here all is similar to setting the activation properties except that all condition properties are returned one at a time therefore passing a
requested activation condition index in each method is sufficient. In case an invalid index is passed, the invalid index entry appears in
the journal. The value of -1 is returned for the methods returning integer values, while EMPTY\_VALUE
is returned for methods returning real values.

The method of comparing two values by a specified comparison type:

```
//+------------------------------------------------------------------+
//| Compare two data source values by a comparison type              |
//+------------------------------------------------------------------+
bool CPendRequest::IsCompared(const double actual_value,const double control_value,const ENUM_COMPARER_TYPE compare) const
  {
   switch((int)compare)
     {
      case EQUAL           :  return(actual_value<control_value || actual_value>control_value ? false : true);
      case NO_EQUAL        :  return(actual_value<control_value || actual_value>control_value ? true : false);
      case MORE            :  return(actual_value>control_value ? true  : false);
      case LESS            :  return(actual_value<control_value ? true  : false);
      case EQUAL_OR_MORE   :  return(actual_value<control_value ? false : true);
      case EQUAL_OR_LESS   :  return(actual_value>control_value ? false : true);
      default              :  break;
     }
   return false;
  }
//+------------------------------------------------------------------+
```

The method receives the current value of a compared property, the
reference value the current value is compared with and the
comparison type.

Depending on the comparison type,
the current property values are compared with its reference value returning the comparison result.

The method returning the flag of successful comparison of an activation condition by its index in the activation conditions data array:

```
//+----------------------------------------------------------------------+
//| Return the flag of a successful check of a controlled object property|
//| and the appropriate real property                                    |
//+----------------------------------------------------------------------+
bool CPendRequest::IsComparisonCompleted(const uint index) const
  {
//--- If the controlled property is not set, return 'false'
   if(this.m_activated_control[index][1]==MSG_LIB_PROP_NOT_SET)
      return false;
//--- Return the result of the specified comparison of a controlled property value with a real one
   ENUM_COMPARER_TYPE comparer=(ENUM_COMPARER_TYPE)this.m_activated_control[index][2];
   return this.IsCompared(this.m_activated_control[index][4],this.m_activated_control[index][3],comparer);
  }
//+------------------------------------------------------------------+
```

The method returns the activation flag of one of the pending request activation conditions. The method input passes the index of a checked
condition in the activation conditions data array. The comparison is performed using the **IsCompared()** method considered above,
and the comparison result is returned.

The method returning the flag of successful check of all activation conditions created for a request object:

```
//+------------------------------------------------------------------+
//| Return the flag of successful check of all controlled properties |
//| of the object and the appropriate actual properties              |
//+------------------------------------------------------------------+
bool CPendRequest::IsAllComparisonCompleted(void) const
  {
   bool res=true;
   int range=::ArrayRange(this.m_activated_control,0);
   if(range==0)
      return false;
   for(int i=0;i<range;i++)
      res &=this.IsComparisonCompleted(i);
   return res;
  }
//+-------------------------------------------------------------------+
```

This is a universal method allowing to check any pending request object for its activation time.

Here, in
the array's first dimension loop of the activation conditions data, the **IsComparisonCompleted()** method is used
to add the successful check flag to the check result ( **res** variable).
The check defines whether the controlled property loop matches the index. The result of checking all conditions is returned upon the loop
completion. If at least one of the conditions is not met or the data array is of zero size in the first dimension, the result is false.

The method returning the number of decimal places for the correct display of an activation condition description in the journal:

```
//+-------------------------------------------------------------------+
//|Return the number of decimal places of a controlled property       |
//+-------------------------------------------------------------------+
int CPendRequest::DigitsControlledValue(const uint index) const
  {
   int dg=0;
   //--- Depending on the activation condition source, check the activation conditions
   //--- and write the required number of decimal places to the result
   switch((int)this.m_activated_control[index][0])
     {
      //--- Account. If an activation condition is an integer value, then 0,
      //--- if it is a real value, then the number of decimal places in the current currency
      case PEND_REQ_ACTIVATION_SOURCE_ACCOUNT      :
         dg=(this.m_activated_control[index][1]<PEND_REQ_ACTIVATE_BY_ACCOUNT_BALANCE  ?  0 : this.m_digits_currency);
        break;
      //--- Symbol. Depending on a condition, write either a number of decimal places in a symbol quote,
      //--- or a number of decimal places in the current currency, or a number of decimal places in the lot value, or zero
      case PEND_REQ_ACTIVATION_SOURCE_SYMBOL       :
         //--- digits
         if(
           (this.m_activated_control[index][1]<PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_DEALS &&
            this.m_activated_control[index][1]>PEND_REQ_ACTIVATE_BY_SYMBOL_EMPTY)                     ||
            this.m_activated_control[index][1]==PEND_REQ_ACTIVATE_BY_SYMBOL_OPTION_STRIKE             ||
            this.m_activated_control[index][1]>PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME ||
           (this.m_activated_control[index][1]>PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_FREEZE_LEVEL &&
            this.m_activated_control[index][1]<PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUME_REAL)
           ) dg=this.m_digits;
         //--- не digits
         else if(
            this.m_activated_control[index][1]>PEND_REQ_ACTIVATE_BY_SYMBOL_LASTLOW)
           {
            //--- digits currency
            if(
               (this.m_activated_control[index][1]>PEND_REQ_ACTIVATE_BY_SYMBOL_OPTION_STRIKE    &&
                this.m_activated_control[index][1]<PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_VOLUME)  ||
               this.m_activated_control[index][1]==PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_TURNOVER
              ) dg=this.m_digits_currency;
            //--- digits lots
            else
               dg=(this.m_digits_lot==0 ? 1 : this.m_digits_lot);
           }
         //--- 0
         else
            dg=0;
        break;
      //--- Default is zero
      default:
        break;
     }
   return dg;
  }
//+------------------------------------------------------------------+
```

The method checks the activation source and the activation condition depending on the source. Depending on the activation condition, the
system returns either a number of decimal places in a symbol quote value, or a number of decimal places in the current account currency, or a
number of decimal places in the symbol lot value, or zero.

The method returning a text description of a controlled property:

```
//+------------------------------------------------------------------+
//| Return the controlled property description by index              |
//+------------------------------------------------------------------+
string CPendRequest::GetActivationPropertyDescription(const uint index) const
  {
   //--- Get the activation source and, depending on that source, create a description text for a type of comparison with the reference value
   ENUM_PEND_REQ_ACTIVATION_SOURCE source=(ENUM_PEND_REQ_ACTIVATION_SOURCE)this.m_activated_control[index][0];
   string value=
     (
      source==PEND_REQ_ACTIVATION_SOURCE_EVENT     ?  "" :
      (
       this.m_activated_control[index][1]==MSG_LIB_PROP_NOT_SET ? ""  :
       this.GetActivationComparerTypeDescription(index)+this.GetActivationControlValueDescription(index)
      )
     );
   //--- Return the activation conditions description + comparison type + controlled value
   return
     (
      source==PEND_REQ_ACTIVATION_SOURCE_ACCOUNT   ?  CMessage::Text((ENUM_PEND_REQ_ACTIVATE_BY_ACCOUNT_PROP)this.m_activated_control[index][1])+value   :
      source==PEND_REQ_ACTIVATION_SOURCE_SYMBOL    ?  CMessage::Text((ENUM_PEND_REQ_ACTIVATE_BY_SYMBOL_PROP)this.m_activated_control[index][1])+value    :
      source==PEND_REQ_ACTIVATION_SOURCE_EVENT     ?  CMessage::Text((ENUM_PEND_REQ_ACTIVATE_BY_EVENT)this.m_activated_control[index][1])+value          :
      ""
     );
  }
//+------------------------------------------------------------------+
```

The method receives the condition index in the activation conditions data array. Get the activation source by the index and obtain the
remaining text messages depending on it. These messages are to be used to arrange and return the final text.

The method returning the comparison type description:

```
//+------------------------------------------------------------------+
//| Return the comparison type description                           |
//+------------------------------------------------------------------+
string CPendRequest::GetActivationComparerTypeDescription(const uint index) const
  {
   return ComparisonTypeDescription((ENUM_COMPARER_TYPE)this.m_activated_control[index][2]);
  }
//+------------------------------------------------------------------+
```

The method simply returns the text description of a comparison type set in the data array by the activation condition index passed by the
parameter to the method.

The method returning a controlled property value description in a request object:

```
//+------------------------------------------------------------------+
//| Return a controlled property value description in an object      |
//+------------------------------------------------------------------+
string CPendRequest::GetActivationControlValueDescription(const uint index) const
  {
   return
     (
      this.m_activated_control[index][3]!=EMPTY_VALUE ?
      (
       this.m_activated_control[index][0]==PEND_REQ_ACTIVATION_SOURCE_SYMBOL &&
       this.m_activated_control[index][1]==PEND_REQ_ACTIVATE_BY_SYMBOL_TIME   ?
       ::TimeToString((ulong)this.m_activated_control[index][3])              :
       ::DoubleToString(this.m_activated_control[index][3],this.DigitsControlledValue(index))
      )  :  ""
     );
  }
//+------------------------------------------------------------------+
```

The method receives the condition index.

A controlled property
value set to the array by a specified index is checked. If it is equal to a specified index and not equal to an "empty value" (EMPTY\_VALUE),
conditionand
its type are checked. If a symbol time is checked as a result,
the time text description is returned, otherwise,
the text description of an integer or real value is
returned with the correct number of decimal places.

The method returning an actual controlled property value description in a request object:

```
//+------------------------------------------------------------------+
//|Return an actual controlled property value description            |
//+------------------------------------------------------------------+
string CPendRequest::GetActivationActualValueDescription(const uint index) const
  {
   return
     (
      this.m_activated_control[index][4]!=EMPTY_VALUE ?
      (
       this.m_activated_control[index][0]==PEND_REQ_ACTIVATION_SOURCE_SYMBOL &&
       this.m_activated_control[index][1]==PEND_REQ_ACTIVATE_BY_SYMBOL_TIME   ?
       ::TimeToString((ulong)this.m_activated_control[index][4])              :
       ::DoubleToString(this.m_activated_control[index][4],this.DigitsControlledValue(index))
      )  :  ""
     );
  }
//+------------------------------------------------------------------+
```

The method is identical to the previous one, except that the data is obtained by index 4 in the second dimension of the activation conditions
data array. These are all the changes of the abstract pending request's base object.

Now let's make some improvements in the classes of the descendant objects of the abstract request base object.

Since we have implemented two types of pending requests — by error code and by request, the second type of objects does not imply the presence of
some properties — such as the server return code (it is always equal to zero here), request activation time (the time in requests of the second
type can be specified as one of the request activation conditions and is located in the activation conditions data array of a pending trading
request), waiting time (not used here at all) and the current attempt index (one attempt is given here, a standard trading order is sent
afterwards and handled by the trade server return code).

In this regard, let's supplement all descendant objects of the pending request base object, namely their methods returning support for
integer properties by the object and add calling the method displaying the list of pending request activation conditions in the journal to
the **PrintShort()** method of each of the descendant objects.

Add
the following changes to the PendReqOpen.mqh, PendReqClose.mqh, PendReqSLTP.mqh, PendReqPlace.mqh, PendReqRemove.mqh and
PendReqModify.mqh files of the abstract pending request base object (the CPendReqOpen class is used as an example):

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(
      (this.GetProperty(PEND_REQ_PROP_TYPE)==PEND_REQ_TYPE_REQUEST   &&
       (property==PEND_REQ_PROP_RETCODE                              ||
        property==PEND_REQ_PROP_TIME_ACTIVATE                        ||
        property==PEND_REQ_PROP_WAITING                              ||
        property==PEND_REQ_PROP_CURRENT_ATTEMPT
       )
      )                                                              ||
      property==PEND_REQ_PROP_MQL_REQ_ORDER                          ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION                       ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY                    ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION                     ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
```

The system makes sure this is the object created by request.
If yes, the properties mentioned above are excluded.

```
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqOpen::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
   string price=CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE),this.m_digits);
   string sl=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL),this.m_digits) : "";
   string tp=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP),this.m_digits) : "";
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN)+": "+
   "\n- "+params+", "+price+sl+tp+
   "\n- "+time+", "+attempts+", "+wait+", "+end;
   ::Print(message);
   this.PrintActivations();
  }
//+------------------------------------------------------------------+
```

After the "+end" string, we removed adding the text wrapping code (+"\\n"), as well as added calling the method displaying the list of activation
conditions after the ::Print(message); string. If the condition array is of zero size (in the objects created by the error code), the
PrintActivations() prints nothing except for the text wrapping code ("\\n"). Otherwise, the method displays the full list of all
conditions written in the data array.

Some classes have undergone minor changes related to the journal display. There is no point in
dwelling on them here. You can find them in the attachments.

Now let's have a look at the trading classes.

In the CTrading base trading class, move
the three class member variables and the GetFreeID() method from the private section to the protected one:

```
private:
   CArrayInt            m_list_errors;                   // Error list
   bool                 m_is_trade_disable;              // Flag disabling trading
   bool                 m_use_sound;                     // The flag of using sounds of the object trading events
   uchar                m_total_try;                     // Number of trading attempts
   MqlTradeRequest      m_request;                       // Trading request prices
   ENUM_TRADE_REQUEST_ERR_FLAGS m_error_reason_flags;    // Flags of error source in a trading method
   ENUM_ERROR_HANDLING_BEHAVIOR m_err_handling_behavior; // Behavior when handling error

//--- Add the error code to the list
```

These variables and methods are necessary in the descendant class. Therefore, they
should be located in the protected section so that the descendant class is able to access them (they are not needed in the public
section — the outside access to them is disabled). In the protected section of the
class, add the method returning the flag of a market order/position with a pending request ID.

As a result, the protected class
section looks as follows:

```
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading : public CBaseObj
  {
protected:
   CAccount            *m_account;                       // Pointer to the current account object
   CSymbolsCollection  *m_symbols;                       // Pointer to the symbol collection list
   CMarketCollection   *m_market;                        // Pointer to the list of the collection of market orders and positions
   CHistoryCollection  *m_history;                       // Pointer to the list of the collection of historical orders and deals
   CEventsCollection   *m_events;                        // Pointer to the event collection list
   CArrayObj            m_list_request;                  // List of pending requests
   uchar                m_total_try;                     // Number of trading attempts
   MqlTradeRequest      m_request;                       // Trade request structure
   ENUM_TRADE_REQUEST_ERR_FLAGS m_error_reason_flags;    // Flags of error source in a trading method
//--- Look for the first free pending request ID
   int                  GetFreeID(void);
//--- Return the flag of a market order/position with a pending request ID
   bool                 IsPresentOrderByID(const uchar id);
private:
```

The variables and methods moved from the private section
and the new method definition are highlighted in color here.

In the public section of the class, add the declaration of the method
returning the pointer to the request object by its ID in the list and the
declaration of the method returning the level of logging a symbol trading object:

```
//--- Create a pending request
   bool                 CreatePendingRequest(const ENUM_PEND_REQ_STATUS status,
                                             const uchar id,
                                             const uchar attempts,
                                             const ulong wait,
                                             const MqlTradeRequest &request,
                                             const int retcode,
                                             CSymbol *symbol_obj,
                                             COrder *order);

//--- Return (1) the pointer to the request object by its ID in the list,
//--- (2) the logging level of a symbol trading object
   CPendRequest        *GetPendRequestByID(const uchar id);
   ENUM_LOG_LEVEL       GetTradeObjLogLevel(const string symbol_name);

  };
//+------------------------------------------------------------------+
```

Let's implement these methods outside the class body.

Implementing the method returning the logging level of a symbol trading object:

```
//+------------------------------------------------------------------+
//| Return the logging level of a symbol trading object              |
//+------------------------------------------------------------------+
ENUM_LOG_LEVEL CTrading::GetTradeObjLogLevel(const string symbol_name)
  {
   CTradeObj *trade_obj=this.GetTradeObjBySymbol(symbol_name,DFUN);
   return(trade_obj!=NULL ? trade_obj.GetLogLevel() : LOG_LEVEL_NO_MSG);
  }
//+------------------------------------------------------------------+
```

The method receives a name of a symbol whose trading object logging level should be received. Get
a trading object from a symbol object. If the object has been received,
return the object's logging level, otherwise return
the logging disabled status.

Implementing the method returning the pointer to the request object by its ID in the list:

```
//+------------------------------------------------------------------+
//| Return the pointer to the request object by its ID in the list   |
//+------------------------------------------------------------------+
CPendRequest* CTrading::GetPendRequestByID(const uchar id)
  {
   int index=this.GetIndexPendingRequestByID(id);
   if(index==WRONG_VALUE)
      return NULL;
   return this.m_list_request.At(index);
  }
//+------------------------------------------------------------------+
```

The method receives the request ID, then obtains the pending request object
index in the list by its ID. If there is no object in the list, return NULL.
Otherwise, return
the object from the list by its obtained index in the list.

Implementing the method returning the flag of a market order/position with a pending request ID:

```
//+------------------------------------------------------------------+
//| Return the flag of a market order/position                       |
//| with a pending request ID                                        |
//+------------------------------------------------------------------+
bool CTrading::IsPresentOrderByID(const uchar id)
  {
   CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
   return(list==NULL ? false : list.Total()!=0);
  }
//+------------------------------------------------------------------+
```

The method receives a pending request ID. Next, receive the list of market
orders/positions sorted by a pending request ID and its value. If the list is not received or is empty (no orders/positions with the
desired ID), return false, otherwise return true.

Let's add yet another check to the method returning the
unoccupied ID:

```
//+------------------------------------------------------------------+
//| Look for the first free pending request ID                       |
//+------------------------------------------------------------------+
int CTrading::GetFreeID(void)
  {
   int id=WRONG_VALUE;
   CPendRequest *element=new CPendRequest();
   if(element==NULL)
      return 0;
   for(int i=1;i<256;i++)
     {
      element.SetID((uchar)i);
      this.m_list_request.Sort(SORT_BY_PEND_REQ_ID);
      if(this.m_list_request.Search(element)==WRONG_VALUE)
        {
         if(this.IsPresentOrderByID((uchar)i))
            continue;
         id=i;
         break;
        }
     }
   delete element;
   return id;
  }
//+------------------------------------------------------------------+
```

Why do we need yet another check for an order/position having the
appropriate ID? If we have an activated pending request and a position is opened based on it, the request is removed from the lists of
pending requests and its ID becomes available for use when creating new pending requests.

When creating a new pending request, its ID
will be equal to the one used by a currently opened position. When the new pending request activation conditions are met, the presence of a
position with the same ID is checked (it should be present as it has been opened using the previous ID) and the new pending request is simply
removed. Since the position with the same ID exists, the request is considered executed. In other words, the request is removed instead of
sending a trading order to the server.

There are a couple of solutions for avoiding such situations: introduce an additional identification defining whether this is another
request with the same ID the open position has or simply check the presence of a position with the same ID if there is no pending request with the
same ID in the list.

The second option seems less resource-intensive to me, although it has a limitation since it is impossible to use an
unoccupied ID till a position with the same ID is closed. In other words, we have a strict limitation of 255 positions with different pending
request IDs.

**This concludes the improvements of the main trading class.**

Now let's finalize the **CTradingControl** trading management class which is a descendant of the **CTrading**
main trading class.

While developing the pending requests management class [in the \\
previous article](https://www.mql5.com/en/articles/7481#node04), we introduced handling pending request objects in the class timer.

Since we handled a single type
of pending requests created using the server return code, it is sufficient to place the entire handling code in the class timer.

Today we will add handling the second type of pending requests created by a program request.

This means we need to
make two handlers — the first one is for requests created by an error code, while the second one is for the ones created by request.


Therefore, we will introduce two pending requests object handlers divided by the type of handled requests, while the identical code for
both handlers is made in a separate method. In this case, we only need to check a request type in the timer and call the appropriate handler to
handle both types of pending requests.

Let's make all the necessary additions in the class body and analyze them:

```
//+------------------------------------------------------------------+
//| Class for managing pending trading requests                      |
//+------------------------------------------------------------------+
class CTradingControl : public CTrading
  {
private:
//--- Set actual order/position data to a pending request object
   void                 SetOrderActualProperties(CPendRequest *req_obj,const COrder *order);
//--- Handler of pending requests created (1) by error code, (2) by request
   void                 OnPReqByErrCodeHandler(CPendRequest *req_obj,const int index);
   void                 OnPReqByRequestHandler(CPendRequest *req_obj,const int index);
//--- Check a pending request relevance (activated or not)
   bool                 CheckPReqRelevance(CPendRequest *req_obj,const MqlTradeRequest &request,const int index);
//--- Update relevant values of controlled properties in pending request objects,
   void                 RefreshControlActualDatas(CPendRequest *req_obj,const CSymbol *symbol);
//--- Return the relevant (1) account, (2) symbol, (3) event data to control activation
   double               GetActualDataAccount(const int property);
   double               GetActualDataSymbol(const int property,const CSymbol *symbol);
   double               GetActualDataEvent(const int property);

public:
//--- Return itself
   CTradingControl     *GetObject(void)            { return &this;   }
//--- Timer
   virtual void         OnTimer(void);
//--- Constructor
                        CTradingControl();
//--- (1) Create a pending request (1) to open a position, (2) to place a pending order
   template<typename SL,typename TP>
   int                  OpenPositionPending(const ENUM_POSITION_TYPE type,
                                           const double volume,
                                           const string symbol,
                                           const ulong magic=ULONG_MAX,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const uchar group_id1=0,
                                           const uchar group_id2=0,
                                           const string comment=NULL,
                                           const ulong deviation=ULONG_MAX,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   int                  PlaceOrderPending( const ENUM_ORDER_TYPE order_type,
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
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
//--- Set pending request activation criteria
   bool                 SetNewActivationProperties(const uchar id,
                                                   const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                                   const int property,
                                                   const double control_value,
                                                   const ENUM_COMPARER_TYPE comparer_type,
                                                   const double actual_value);
  };
//+------------------------------------------------------------------+
```

Since now we need to set data on controlled parameters to a pending request object (while we have previously also added relevant data on the order
the request is based on), rename the SetActualProperties()
method for setting relevant order data to a request object to **SetOrderActualProperties()** to avoid confusion.

In this article, we are dealing only with opening positions using pending requests, therefore the method of creating a pending request
remains out of the scope of the current article.

Let's consider the method of creating a pending request for opening a position:

```
//+------------------------------------------------------------------+
//| Create a pending request for opening a position                  |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CTradingControl::OpenPositionPending(const ENUM_POSITION_TYPE type,
                                         const double volume,
                                         const string symbol,
                                         const ulong magic=ULONG_MAX,
                                         const SL sl=0,
                                         const TP tp=0,
                                         const uchar group_id1=0,
                                         const uchar group_id2=0,
                                         const string comment=NULL,
                                         const ulong deviation=ULONG_MAX,
                                         const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Exit if the global trading ban flag is set
   if(this.IsTradingDisable())
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return false;
     }
//--- Set the trading request result as 'true' and the error flag as "no errors"
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)type;
   ENUM_ACTION_TYPE action=(ENUM_ACTION_TYPE)order_type;
//--- Get a symbol object by a symbol name.
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
   //--- Look for the least of the possible IDs. If failed to find, return 'false'
   int id=this.GetFreeID();
   if(id<1)
     {
      //--- No free IDs to create a pending request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_NO_FREE_IDS));
      return false;
     }

//--- Write the volume, deviation, comment and filling type to the request structure
   this.m_request.volume=volume;
   this.m_request.deviation=(deviation==ULONG_MAX ? trade_obj.GetDeviation() : deviation);
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : trade_obj.GetTypeFilling());
//--- Write pending request object ID to the magic number, add group IDs to the magic number value
//--- and fill in the remaining unfilled trading request structure fields
   uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
   this.SetPendReqID((uchar)id,mn);
   if(group_id1>0)
      this.SetGroupID1(group_id1,mn);
   if(group_id2>0)
      this.SetGroupID2(group_id2,mn);
   this.m_request.magic=mn;
   this.m_request.action=TRADE_ACTION_DEAL;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.type=order_type;
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_OPEN,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,NULL))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The method is a truncated version of the method for opening a position [from \\
the article 26](https://www.mql5.com/en/articles/7394#node03) (and subsequent ones) with the creation of a pending request in case of a trade server error. It is commented in detail so
there is no point in dwelling on it here.

The method receives all the necessary data for opening a position. The trading request
structure fields are filled in and sent to the pending request creation method.

If a pending request is created successfully, the ID of a
newly created pending request is returned, otherwise -1 is returned.

The calculated maximum possible waiting time is used here as
a difference between the maximum possible time in the terminal and the current time as a delay between repeated attempts. Thus, the
maximum possible lifetime is used for a pending request (up to 31.12.3000).

The method setting pending request activation criteria:

```
//+------------------------------------------------------------------+
//| Set pending request activation criteria                          |
//+------------------------------------------------------------------+
bool CTradingControl::SetNewActivationProperties(const uchar id,
                                                const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                                const int property,
                                                const double control_value,
                                                const ENUM_COMPARER_TYPE comparer_type,
                                                const double actual_value)
  {
   CPendRequest *req_obj=this.GetPendRequestByID(id);
   if(req_obj==NULL)
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_GETTING_FAILED));
      return false;
     }
   req_obj.SetNewActivationProperties(source,property,control_value,comparer_type,actual_value);
   return true;
  }
//+------------------------------------------------------------------+
```

The method receives an ID of a pending request a new activation condition should be added to, a request activation source (symbol, account or
event), activation condition, reference value, comparison type and actual value of a property controlled for activating the request.


Next, receive a pending request object by the ID passed to the method
and create a new activation condition for it with the parameters
passed to the method.

The method of checking the pending request relevance:

```
//+------------------------------------------------------------------+
//| Checking the pending request relevance                           |
//+------------------------------------------------------------------+
bool CTradingControl::CheckPReqRelevance(CPendRequest *req_obj,const MqlTradeRequest &request,const int index)
  {
   //--- If this is a position opening or placing a pending order
   if((req_obj.Action()==TRADE_ACTION_DEAL && req_obj.Position()==0) || req_obj.Action()==TRADE_ACTION_PENDING)
     {
      //--- Get the pending request ID
      uchar id=this.GetPendReqID((uint)request.magic);
      //--- Get the list of orders/positions containing the order/position with the pending request ID
      CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
      if(::CheckPointer(list)==POINTER_INVALID)
         return false;
      //--- If the order/position is present, the request is handled: remove it and proceed to the next (leave the method for the external loop)
      if(list.Total()>0)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
         this.m_list_request.Delete(index);
         return false;
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
            return false;
         //--- If the market has no such position, the request is handled: remove it and proceed to the next (leave the method for the external loop)
         if(list.Total()==0)
           {
            if(this.m_log_level>LOG_LEVEL_NO_MSG)
               ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
            this.m_list_request.Delete(index);
            return false;
           }
         //--- Otherwise, if the position still exists, this is a partial closure
         else
           {
            //--- Get the list of all account trading events
            list=this.m_events.GetList();
            if(list==NULL)

               return false;
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
                  //--- If a position ticket in a trading event coincides with the ticket in a pending trading request
                  if(event.PositionID()==req_obj.Position())
                    {
                     //--- Get a position object from the list of market positions
                     CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_POSITION_ID,req_obj.Position(),EQUAL);
                     if(list_orders==NULL || list_orders.Total()==0)
                        break;
                     COrder *order=list_orders.At(list_orders.Total()-1);
                     if(order==NULL)
                        break;
                     //--- Set actual position data to the pending request object
                     this.SetOrderActualProperties(req_obj,order);
                     //--- If (executed request volume + unexecuted request volume) is equal to the requested volume in a pending request -
                     //--- the request is handled: remove it and break the loop by the list of account trading events
                     if(req_obj.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME)==event.VolumeOrderExecuted()+event.VolumeOrderCurrent())
                       {
                        if(this.m_log_level>LOG_LEVEL_NO_MSG)
                           ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                        this.m_list_request.Delete(index);
                        break;
                       }
                    }
                 }
              }
            //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one (leave the method for the external loop)
            if(::CheckPointer(req_obj)==POINTER_INVALID)
               return false;
           }
        }
      //--- If this is a modification of position stop orders
      if(req_obj.Action()==TRADE_ACTION_SLTP)
        {
         //--- Get the list of all account trading events
         list=this.m_events.GetList();
         if(list==NULL)
            return false;
         //--- In the loop from the end of the account trading event list
         int events_total=list.Total();
         for(int j=events_total-1; j>WRONG_VALUE; j--)
           {
            //--- get the next trading event
            CEvent *event=list.At(j);
            if(event==NULL)
               continue;
            //--- If this is a change of the position's stop orders
            if(event.TypeEvent()>TRADE_EVENT_MODIFY_ORDER_TP)
              {
               //--- If a position ticket in a trading event coincides with the ticket in a pending trading request
               if(event.PositionID()==req_obj.Position())
                 {
                  //--- Get a position object from the list of market positions
                  CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_POSITION_ID,req_obj.Position(),EQUAL);
                  if(list_orders==NULL || list_orders.Total()==0)
                     break;
                  COrder *order=list_orders.At(list_orders.Total()-1);
                  if(order==NULL)
                     break;
                  //--- Set actual position data to the pending request object
                  this.SetOrderActualProperties(req_obj,order);
                  //--- If all modifications have worked out -
                  //--- the request is handled: remove it and break the loop by the list of account trading events
                  if(req_obj.IsCompleted())
                    {
                     if(this.m_log_level>LOG_LEVEL_NO_MSG)
                        ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                     this.m_list_request.Delete(index);
                     break;
                    }
                 }
              }
           }
         //--- If a handled pending request object was removed by the trading event list in the loop, move on to the next one (leave the method for the external loop)
         if(::CheckPointer(req_obj)==POINTER_INVALID)
            return false;
        }
      //--- If this is a pending order removal
      if(req_obj.Action()==TRADE_ACTION_REMOVE)
        {
         //--- Get the list of removed pending orders from the historical list
         list=this.m_history.GetList(ORDER_PROP_STATUS,ORDER_STATUS_HISTORY_PENDING,EQUAL);
         if(::CheckPointer(list)==POINTER_INVALID)
            return false;
         //--- Leave a single order with the necessary ticket in the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,req_obj.Order(),EQUAL);
         //--- If the order is present, the request is handled: remove it and proceed to the next (leave the method for the external loop)
         if(list.Total()>0)
           {
            if(this.m_log_level>LOG_LEVEL_NO_MSG)
               ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
            this.m_list_request.Delete(index);
            return false;
           }
        }
      //--- If this is a pending order modification
      if(req_obj.Action()==TRADE_ACTION_MODIFY)
        {
         //--- Get the list of all account trading events
         list=this.m_events.GetList();
         if(list==NULL)
            return false;
         //--- In the loop from the end of the account trading event list
         int events_total=list.Total();
         for(int j=events_total-1; j>WRONG_VALUE; j--)
           {
            //--- get the next trading event
            CEvent *event=list.At(j);
            if(event==NULL)
               continue;
            //--- If this event involves any change of modified pending order parameters
            if(event.TypeEvent()>TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER && event.TypeEvent()<TRADE_EVENT_MODIFY_POSITION_SL_TP)
              {
               //--- If an order ticket in a trading event coincides with the ticket in a pending trading request
               if(event.TicketOrderEvent()==req_obj.Order())
                 {
                  //--- Get an order object from the list
                  CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Order(),EQUAL);
                  if(list_orders==NULL || list_orders.Total()==0)
                     break;
                  COrder *order=list_orders.At(0);
                  if(order==NULL)
                     break;
                  //--- Set actual order data to the pending request object
                  this.SetOrderActualProperties(req_obj,order);
                  //--- If all modifications have worked out -
                  //--- the request is handled: remove it and break the loop by the list of account trading events
                  if(req_obj.IsCompleted())
                    {
                     if(this.m_log_level>LOG_LEVEL_NO_MSG)
                        ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_EXECUTED));
                     this.m_list_request.Delete(index);
                     break;
                    }
                 }
              }
           }
        }
     }
   //--- Exit if the pending request object has been removed after checking its operation (leave the method for the external loop)
   return(::CheckPointer(req_obj)==POINTER_INVALID ? false : true);
  }
//+------------------------------------------------------------------+
```

The method checks the execution of a pending request and removes it after the execution is confirmed. We have already considered this code [within \\
the code of the trading management class timer](https://www.mql5.com/en/articles/7481#node04). Since we have divided handling pending request objects into two handlers by pending
request types and that code is similar for both handlers, we have put it into a separate method. It is to be called in each handler.

The handler of pending requests created by error code:

```
//+------------------------------------------------------------------+
//| Handler of pending requests created by error code                |
//+------------------------------------------------------------------+
void CTradingControl::OnPReqByErrCodeHandler(CPendRequest *req_obj,const int index)
  {
   //--- get the request structure and the symbol object a trading operation should be performed for
   MqlTradeRequest request=req_obj.MqlRequest();
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(request.symbol);
   if(symbol_obj==NULL || !symbol_obj.RefreshRates())
      return;

   //--- Set the flag disabling trading in the terminal by two properties simultaneously
   //--- (the AutoTrading button in the terminal and the Allow Automated Trading option in the EA settings)
   //--- If any of the two properties is 'false', the flag is 'false' as well
   bool terminal_trade_allowed=::TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   terminal_trade_allowed &=::MQLInfoInteger(MQL_TRADE_ALLOWED);
   //--- if the error has been caused by trading disabled on the terminal side and has been eliminated
   if(req_obj.Retcode()==10027 && terminal_trade_allowed)
     {
      //--- if the current attempt has not exceeded the defined number of trading attempts yet
      if(req_obj.CurrentAttempt()<req_obj.TotalAttempts()+1)
        {
         //--- Set the request creation time equal to its creation time minus waiting time, i.e. send the request immediately
         //--- Also, decrease the number of a successful attempt since during the next attempt, its number is increased, and if this is the last attempt,
         //--- it is not executed. However, this is related to fixing the error cause by a user, which means we need to give more time for the last attempt
         req_obj.SetTimeCreate(req_obj.TimeCreate()-req_obj.WaitingMSC());
         req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()>0 ? req_obj.CurrentAttempt()-1 : 0));
        }
     }

   //--- if the current attempt exceeds the defined number of trading attempts,
   //--- or the current time exceeds the waiting time of all attempts
   //--- remove the current request object and proceed to the next (leave the method for the external loop)
   if(req_obj.CurrentAttempt()>req_obj.TotalAttempts() || req_obj.CurrentAttempt()>=UCHAR_MAX ||
      (long)symbol_obj.Time()>long(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.TotalAttempts()+1)))
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(req_obj.Header(),": ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_DELETED));
      this.m_list_request.Delete(index);
      return;
     }
   //--- Check the relevance of a pending request and exit to the external loop if the request is handled or an error occurs
   if(!this.CheckPReqRelevance(req_obj,request,index))
      return;

   //--- Set the request activation time in the request object
   req_obj.SetTimeActivate(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.CurrentAttempt()+1));

   //--- If the current time is less than the request activation time,
   //--- this is not the request time - move on to the next request in the list (leave the method for the external loop)
   if((long)symbol_obj.Time()<(long)req_obj.TimeActivate())
      return;

   //--- Set the attempt number in the request object
   req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

   //--- Display the number of a trading attempt in the journal

   if(this.m_log_level>LOG_LEVEL_NO_MSG)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_RE_TRY_N)+(string)req_obj.CurrentAttempt()+":");
      req_obj.PrintShort();
     }

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
         this.DeleteOrder(request.order);
         break;
      //---
      default:
         break;
     }
  }
//+------------------------------------------------------------------+
```

We have also considered this code [within the trading management class timer](https://www.mql5.com/en/articles/7481#node04).
Its only difference is that handling a request activation check is moved to calling an
appropriate method.

The handler of pending requests created by request:

```
//+------------------------------------------------------------------+
//| The handler of pending requests created by request               |
//+------------------------------------------------------------------+
void CTradingControl::OnPReqByRequestHandler(CPendRequest *req_obj,const int index)
  {
   //--- get the request structure and the symbol object a trading operation should be performed for
   MqlTradeRequest request=req_obj.MqlRequest();
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(request.symbol);
   if(symbol_obj==NULL || !symbol_obj.RefreshRates())
      return;
   //--- Check the relevance of a pending request and exit to the external loop if the request is handled or an error occurs
   if(!this.CheckPReqRelevance(req_obj,request,index))
      return;

   //--- Update relevant data on request activation conditions
   this.RefreshControlActualDatas(req_obj,symbol_obj);

   //--- If all pending request activation conditions are met
   if(req_obj.IsAllComparisonCompleted())
     {
      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));
      //--- Display the request activation message in the journal
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
        {
         ::Print(CMessage::Text(MSG_LIB_TEXT_REQUEST_ACTIVATED)+(string)req_obj.ID()+":");
         req_obj.PrintShort();
        }
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

This method is slightly simpler than the previous one, since it only checks the activation and the moment of sending a trading order (meeting
pending request activation conditions).

The method of checking the request
activation is also called in it. Also, the system tracks whether pending request activation conditions set in its parameters are
triggered. If all conditions are triggered, a trading order is sent
to the server.

The method updating relevant values of controlled properties in pending request objects:

```
//+------------------------------------------------------------------+
//| Update relevant values of controlled properties                  |
//| in pending request objects                                       |
//+------------------------------------------------------------------+
void CTradingControl::RefreshControlActualDatas(CPendRequest *req_obj,const CSymbol *symbol)
  {
//--- Exit if a request object has a request type based on an error code
   if(req_obj.GetProperty(PEND_REQ_PROP_TYPE)==PEND_REQ_TYPE_ERROR)
      return;
   double res=EMPTY_VALUE;
//--- In the loop by all request object activation conditions,
   uint total=req_obj.GetActivationCriterionTotal();
   for(uint i=0;i<total;i++)
     {
      //--- get the activation source
      ENUM_PEND_REQ_ACTIVATION_SOURCE source=req_obj.GetActivationSource(i);
      //--- receive the current value of a controlled property
      double value=req_obj.GetActivationActualValue(i),actual=EMPTY_VALUE;
      //--- Depending on the activation source,
      //--- write the current value of a controlled property to the activation conditions data array
      switch((int)source)
        {
         //--- Account
         case PEND_REQ_ACTIVATION_SOURCE_ACCOUNT   :
            actual=this.GetActualDataAccount(req_obj.GetActivationProperty(i));
            req_obj.SetActivationActualValue(i,(actual!=EMPTY_VALUE ? actual : value));
           break;
         //--- Symbol
         case PEND_REQ_ACTIVATION_SOURCE_SYMBOL    :
            actual=this.GetActualDataSymbol(req_obj.GetActivationProperty(i),symbol);
            req_obj.SetActivationActualValue(i,(actual!=EMPTY_VALUE ? actual : value));
           break;
         //--- Event
         case PEND_REQ_ACTIVATION_SOURCE_EVENT     :
            actual=this.GetActualDataEvent(req_obj.GetActivationProperty(i));
            req_obj.SetActivationActualValue(i,(actual!=EMPTY_VALUE ? actual : value));
           break;
         //--- Default is EMPTY_VALUE
         default:
           break;
        }
     }
  }
//+------------------------------------------------------------------+
```

The method receives the array size of the activation conditions data. Besides, we move along all the conditions in the loop. Depending
on the activation conditions source, we obtain the actual (current) data
from the appropriate collections and write them back to the activation
conditions data array of the pending request object.

The method returning the relevant account data:

```
//+------------------------------------------------------------------+
//| Return the relevant account data to control activation           |
//+------------------------------------------------------------------+
double CTradingControl::GetActualDataAccount(const int property)
  {
   switch(property)
    {
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_LEVERAGE             : return (double)this.m_account.Leverage();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_LIMIT_ORDERS         : return (double)this.m_account.LimitOrders();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_TRADE_ALLOWED        : return (double)this.m_account.TradeAllowed();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_TRADE_EXPERT         : return (double)this.m_account.TradeExpert();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_BALANCE              : return this.m_account.Balance();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_CREDIT               : return this.m_account.Credit();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_PROFIT               : return this.m_account.Profit();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_EQUITY               : return this.m_account.Equity();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN               : return this.m_account.Margin();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_FREE          : return this.m_account.MarginFree();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_LEVEL         : return this.m_account.MarginLevel();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_INITIAL       : return this.m_account.MarginInitial();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_MARGIN_MAINTENANCE   : return this.m_account.MarginMaintenance();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_ASSETS               : return this.m_account.Assets();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_LIABILITIES          : return this.m_account.Liabilities();
     case PEND_REQ_ACTIVATE_BY_ACCOUNT_COMMISSION_BLOCKED   : return this.m_account.ComissionBlocked();
     default: return EMPTY_VALUE;
    }
  }
//+------------------------------------------------------------------+
```

Depending on an account type, return the value of the appropriate
account object property according to the account condition types enumeration.

The method returning the relevant symbol data:

```
//+------------------------------------------------------------------+
//| Return the relevant symbol data to control activation            |
//+------------------------------------------------------------------+
double CTradingControl::GetActualDataSymbol(const int property,const CSymbol *symbol)
  {
   switch(property)
    {
     case PEND_REQ_ACTIVATE_BY_SYMBOL_BID                         : return symbol.Bid();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_ASK                         : return symbol.Ask();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_LAST                        : return symbol.Last();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_DEALS               : return (double)symbol.SessionDeals();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_BUY_ORDERS          : return (double)symbol.SessionBuyOrders();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_SELL_ORDERS         : return (double)symbol.SessionSellOrders();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUME                      : return (double)symbol.Volume();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMEHIGH                  : return (double)symbol.VolumeHigh();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMELOW                   : return (double)symbol.VolumeLow();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TIME                        : return (double)symbol.Time()/1000;
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SPREAD                      : return symbol.Spread();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_START_TIME                  : return (double)symbol.StartTime();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_EXPIRATION_TIME             : return (double)symbol.ExpirationTime();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_STOPS_LEVEL           : return symbol.TradeStopLevel();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_FREEZE_LEVEL          : return symbol.TradeFreezeLevel();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_BIDHIGH                     : return symbol.BidHigh();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_BIDLOW                      : return symbol.BidLow();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_ASKHIGH                     : return symbol.AskHigh();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_ASKLOW                      : return symbol.AskLow();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_LASTHIGH                    : return symbol.LastHigh();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_LASTLOW                     : return symbol.LastLow();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUME_REAL                 : return symbol.VolumeReal();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMEHIGH_REAL             : return symbol.VolumeHighReal();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_VOLUMELOW_REAL              : return symbol.VolumeLowReal();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_OPTION_STRIKE               : return symbol.OptionStrike();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_ACCRUED_INTEREST      : return symbol.TradeAccuredInterest();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_FACE_VALUE            : return symbol.TradeFaceValue();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_TRADE_LIQUIDITY_RATE        : return symbol.TradeLiquidityRate();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SWAP_LONG                   : return symbol.SwapLong();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SWAP_SHORT                  : return symbol.SwapShort();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_VOLUME              : return symbol.SessionVolume();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_TURNOVER            : return symbol.SessionTurnover();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_INTEREST            : return symbol.SessionInterest();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_BUY_ORDERS_VOLUME   : return symbol.SessionBuyOrdersVolume();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_SELL_ORDERS_VOLUME  : return symbol.SessionSellOrdersVolume();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_OPEN                : return symbol.SessionOpen();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_CLOSE               : return symbol.SessionClose();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_AW                  : return symbol.SessionAW();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_SETTLEMENT    : return symbol.SessionPriceSettlement();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_LIMIT_MIN     : return symbol.SessionPriceLimitMin();
     case PEND_REQ_ACTIVATE_BY_SYMBOL_SESSION_PRICE_LIMIT_MAX     : return symbol.SessionPriceLimitMax();
     default: return EMPTY_VALUE;
    }
  }
//+------------------------------------------------------------------+
```

Depending on an account type and according to the symbol
condition types enumeration, return the value of the appropriate symbol object
property, the pointer to which is passed to the method.

The method returning the relevant event data:

```
//+------------------------------------------------------------------+
//| Return the relevant event data to control activation             |
//+------------------------------------------------------------------+
double CTradingControl::GetActualDataEvent(const int property)
  {
   if(this.m_events.IsEvent())
     {
      ENUM_TRADE_EVENT event=this.m_events.GetLastTradeEvent();
      switch(property)
       {
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_OPENED                       : return event==TRADE_EVENT_POSITION_OPENED;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED                       : return event==TRADE_EVENT_POSITION_CLOSED;
        case PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_PLASED                  : return event==TRADE_EVENT_PENDING_ORDER_PLASED;
        case PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_REMOVED                 : return event==TRADE_EVENT_PENDING_ORDER_REMOVED;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CREDIT                        : return event==TRADE_EVENT_ACCOUNT_CREDIT;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CHARGE                        : return event==TRADE_EVENT_ACCOUNT_CHARGE;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_CORRECTION                    : return event==TRADE_EVENT_ACCOUNT_CORRECTION;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BONUS                         : return event==TRADE_EVENT_ACCOUNT_BONUS;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION                     : return event==TRADE_EVENT_ACCOUNT_COMISSION;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_DAILY               : return event==TRADE_EVENT_ACCOUNT_COMISSION_DAILY;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_MONTHLY             : return event==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_AGENT_DAILY         : return event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY       : return event==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_INTEREST                      : return event==TRADE_EVENT_ACCOUNT_INTEREST;
        case PEND_REQ_ACTIVATE_BY_EVENT_BUY_CANCELLED                         : return event==TRADE_EVENT_BUY_CANCELLED;
        case PEND_REQ_ACTIVATE_BY_EVENT_SELL_CANCELLED                        : return event==TRADE_EVENT_SELL_CANCELLED;
        case PEND_REQ_ACTIVATE_BY_EVENT_DIVIDENT                              : return event==TRADE_EVENT_DIVIDENT;
        case PEND_REQ_ACTIVATE_BY_EVENT_DIVIDENT_FRANKED                      : return event==TRADE_EVENT_DIVIDENT_FRANKED;
        case PEND_REQ_ACTIVATE_BY_EVENT_TAX                                   : return event==TRADE_EVENT_TAX;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BALANCE_REFILL                : return event==TRADE_EVENT_ACCOUNT_BALANCE_REFILL;
        case PEND_REQ_ACTIVATE_BY_EVENT_ACCOUNT_BALANCE_WITHDRAWAL            : return event==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_ACTIVATED               : return event==TRADE_EVENT_PENDING_ORDER_ACTIVATED;
        case PEND_REQ_ACTIVATE_BY_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL       : return event==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_OPENED_PARTIAL               : return event==TRADE_EVENT_POSITION_OPENED_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL               : return event==TRADE_EVENT_POSITION_CLOSED_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_POS                : return event==TRADE_EVENT_POSITION_CLOSED_BY_POS;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_POS        : return event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_SL                 : return event==TRADE_EVENT_POSITION_CLOSED_BY_SL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_BY_TP                 : return event==TRADE_EVENT_POSITION_CLOSED_BY_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_SL         : return event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_CLOSED_PARTIAL_BY_TP         : return event==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_MARKET           : return event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_PENDING          : return event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL   : return event==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOLUME_ADD_BY_MARKET         : return event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOLUME_ADD_BY_PENDING        : return event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE                    : return event==TRADE_EVENT_MODIFY_ORDER_PRICE;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_SL                 : return event==TRADE_EVENT_MODIFY_ORDER_PRICE_SL;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_TP                 : return event==TRADE_EVENT_MODIFY_ORDER_PRICE_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_PRICE_SL_TP              : return event==TRADE_EVENT_MODIFY_ORDER_PRICE_SL_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_SL_TP                    : return event==TRADE_EVENT_MODIFY_ORDER_SL_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_SL                       : return event==TRADE_EVENT_MODIFY_ORDER_SL;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_ORDER_TP                       : return event==TRADE_EVENT_MODIFY_ORDER_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_SL_TP                 : return event==TRADE_EVENT_MODIFY_POSITION_SL_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_SL                    : return event==TRADE_EVENT_MODIFY_POSITION_SL;
        case PEND_REQ_ACTIVATE_BY_EVENT_MODIFY_POSITION_TP                    : return event==TRADE_EVENT_MODIFY_POSITION_TP;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOL_ADD_BY_MARKET_PARTIAL    : return event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_VOL_ADD_BY_PENDING_PARTIAL   : return event==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL;
        case PEND_REQ_ACTIVATE_BY_EVENT_TRIGGERED_STOP_LIMIT_ORDER            : return event==TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER;
        case PEND_REQ_ACTIVATE_BY_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL  : return event==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL;
        default: return EMPTY_VALUE;
       }
     }
   return EMPTY_VALUE;
  }
//+------------------------------------------------------------------+
```

Depending on a condition type and the
current presence of a new event on an account, obtain the last event on the
account. According to the enumeration of event condition types, return the flag of equality of the last event to the value controlled
in the pending request object (return the flag of an occurred controlled event).

The class timer has now become much more compact:

```
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CTradingControl::OnTimer(void)
  {
   //--- In a loop by the list of pending requests
   int total=this.m_list_request.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {
      //--- receive the next request object
      CPendRequest *req_obj=this.m_list_request.At(i);
      if(req_obj==NULL)
         continue;
      //--- If a request object is created by an error code, use the handler of objects created by the error code
      if(req_obj.TypeRequest()==PEND_REQ_TYPE_ERROR)
         this.OnPReqByErrCodeHandler(req_obj,i);
      //--- Otherwise, this is an object created by request - use the handler of objects created by request
      else
         this.OnPReqByRequestHandler(req_obj,i);
     }
  }
//+------------------------------------------------------------------+
```

Now, in the class timer, check a type of a request object obtained from the list of pending request objects. Depending on its type, call the
corresponding handler of pending requests we examined above.

**These are all the improvements of the trading management class at the moment.**

You can find the full code in the
attachments.

Let's make additions to the public section of the **CEngine** library base object class.

To be able to receive a logging level of trading objects to retrieve library-based program messages from them, **add**
**the method of receiving a trading object logging level by symbol:**

```
   void                 TradingSetTotalTry(const uchar attempts)                       { this.m_trading.SetTotalTry(attempts);                     }

//--- Return the logging level of a trading class symbol trading object
   ENUM_LOG_LEVEL       TradingGetLogLevel(const string symbol_name)                   { return this.m_trading.GetTradeObjLogLevel(symbol_name);   }

//--- Set standard sounds (symbol==NULL) for a symbol trading object, (symbol!=NULL) for trading objects of all symbols
```

The method returns the operation result of the **GetTradeObjLogLevel()** trading management class method.

Declare the methods of creating a pending request for opening Buy
and Sell positions, as well as the method of setting a new pending
request activation condition and write the method returning the pointer
to a pending request object by its ID:

```
//--- Remove a pending order
   bool                 DeleteOrder(const ulong ticket);

//--- Create a pending request (1) to open Buy and (2) Sell positions
   template<typename SL,typename TP>
   int                  OpenBuyPending(const double volume,
                                       const string symbol,
                                       const ulong magic=ULONG_MAX,
                                       const SL sl=0,
                                       const TP tp=0,
                                       const uchar group_id1=0,
                                       const uchar group_id2=0,
                                       const string comment=NULL,
                                       const ulong deviation=ULONG_MAX,
                                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename SL,typename TP>
   int                  OpenSellPending(const double volume,
                                       const string symbol,
                                       const ulong magic=ULONG_MAX,
                                       const SL sl=0,
                                       const TP tp=0,
                                       const uchar group_id1=0,
                                       const uchar group_id2=0,
                                       const string comment=NULL,
                                       const ulong deviation=ULONG_MAX,
                                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

//--- Set pending request activation criteria
   bool                 SetNewActivationProperties(const uchar id,
                                                   const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                                   const int property,
                                                   const double control_value,
                                                   const ENUM_COMPARER_TYPE comparer_type,
                                                   const double actual_value);

//--- Return the pointer to the request object by its ID in the list
   CPendRequest        *GetPendRequestByID(const uchar id)              { return this.m_trading.GetPendRequestByID(id);       }

//--- Return event (1) milliseconds, (2) reason and (3) source from its 'long' value
```

The **GetPendRequestByID()** method returning the pointer to a pending request object returns the result of the same-name trading
management class method operation.

Implementing the method of creating a pending request for opening a Buy position:

```
//+------------------------------------------------------------------+
//| Create a pending request for opening a Buy position              |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CEngine::OpenBuyPending(const double volume,
                            const string symbol,
                            const ulong magic=ULONG_MAX,
                            const SL sl=0,
                            const TP tp=0,
                            const uchar group_id1=0,
                            const uchar group_id2=0,
                            const string comment=NULL,
                            const ulong deviation=ULONG_MAX,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.OpenPositionPending(POSITION_TYPE_BUY,volume,symbol,magic,sl,tp,group_id1,group_id2,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+
```

The method calls the method of creating a pending request for opening a trading management class position. Pass the POSITION\_TYPE\_BUY
constant as an opened position type (together with other parameters of a future position passed to the method).

Implementing the method of creating a pending request for opening a Sell position:

```
//+------------------------------------------------------------------+
//| Create a pending request for opening a Sell position             |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CEngine::OpenSellPending(const double volume,
                            const string symbol,
                            const ulong magic=ULONG_MAX,
                            const SL sl=0,
                            const TP tp=0,
                            const uchar group_id1=0,
                            const uchar group_id2=0,
                            const string comment=NULL,
                            const ulong deviation=ULONG_MAX,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.OpenPositionPending(POSITION_TYPE_SELL,volume,symbol,magic,sl,tp,group_id1,group_id2,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+
```

The method calls the method of creating a pending request for opening a trading management class position. Pass the POSITION\_TYPE\_SELL
constant as an opened position type (together with other parameters of a future position passed to the method).

Implementing the method of setting a new activation condition in a pending request object:

```
//+------------------------------------------------------------------+
//| Set pending request activation criteria                          |
//+------------------------------------------------------------------+
bool CEngine::SetNewActivationProperties(const uchar id,
                                         const ENUM_PEND_REQ_ACTIVATION_SOURCE source,
                                         const int property,
                                         const double control_value,
                                         const ENUM_COMPARER_TYPE comparer_type,
                                         const double actual_value)
  {
   return this.m_trading.SetNewActivationProperties(id,source,property,control_value,comparer_type,actual_value);
  }
//+------------------------------------------------------------------+
```

The method calls the method of adding a new activation condition to the pending request object of the trading management class. These are all
the library improvements for now.

### Testing

To test applying pending requests for opening positions, use the EA from the previous article and save it in the new folder
\\MQL5\\Experts\\TestDoEasy\ **Part31\** under the name **TestDoEasyPart31.mq5**.

To check operation of pending requests by conditions, we will introduce additional buttons in the trading panel of the test EA. The buttons
are marked as **P** — price condition and **T** — time condition. Pending requests are created when clicking **Buy**
or **Sell** provided that **P** or **T** (or both) are pressed. If both are pressed, the pending request has two activation
conditions — by price and time.

Also, let's add two inputs to set the distance from the current price for specifying a controlled price and a number of bars of the current
timeframe for setting the request activation time.

Thus, if **Buy** and **P** buttons are pressed, the distance below the current price is set from it for the number of points
specified in the settings. This value is set as a reference one for triggering a pending request — when the price is equal or below the
calculated one, the pending request is activated.

If the **T** button is pressed, the time calculated as the current time + the
time of a specified number of bars of the current timeframe is added to the current time. This time is set as a reference one for triggering the
pending request — when the current time becomes equal to or exceeds the calculated one, the pending request is activated.

If both **P** and **T** buttons are triggered, both conditions should be met at once for the pending request activation.

To open a **Sell** position, the controlled price is calculated as the current price + the specified number of points in the
settings. For activating the pending request, the current price should exceed the one present at the moment of creating the pending request
(pressing the **Sell** button).

**Add the indent distance of the reference price of the request activation**
**from the current price at the moment of creating the request and the**
**number of delay bars for setting the pending request activation time:**

```
//--- input variables
input    ushort            InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  150;  // StopLoss in points
input    uint              InpTakeProfit        =  150;  // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpDistancePReq      =  50;   // Distance for Pending Request's activate (points)
input    uint              InpBarsDelayPReq     =  5;    // Bars delay for Pending Request's activate (current timeframe)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
input    uchar             InpTotalAttempts     =  5;    // Number of trading attempts
sinput   double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)
sinput   uint              InpButtShiftX        =  0;    // Buttons X shift
sinput   uint              InpButtShiftY        =  10;   // Buttons Y shift
input    uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input    uint              InpTrailingStep      =  20;   // Trailing Step (points)
input    uint              InpTrailingStart     =  0;    // Trailing Start (points)
input    uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input    uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
sinput   ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;   // Mode of used symbols list
sinput   string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)
sinput   bool              InpUseSounds         =  true; // Use sounds
```

**Add the appropriate variables for storing the activation price indent**
**and delays in bars to the block of EA global variables** to
set the pending request activation time, as well as the flags of pending request
button states:

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
uint           distance_pending_request;
uint           bars_delay_pending_request;
uint           slippage;
bool           trailing_on;
bool           pending_buy;
bool           pending_buy_limit;
bool           pending_buy_stop;
bool           pending_buy_stoplimit;
bool           pending_close_buy;
bool           pending_close_buy2;
bool           pending_close_buy_by_sell;
bool           pending_sell;
bool           pending_sell_limit;
bool           pending_sell_stop;
bool           pending_sell_stoplimit;
bool           pending_close_sell;
bool           pending_close_sell2;
bool           pending_close_sell_by_buy;
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

In the EA's OnInit() handler, assign correct input values to variables
and reset states of pending request buttons:

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
   distance_pending_request=(InpDistancePReq<5 ? 5 : InpDistancePReq);
   bars_delay_pending_request=(InpBarsDelayPReq<1 ? 1 : InpBarsDelayPReq);
//--- Initialize random group numbers
   group1=0;
   group2=0;
   srand(GetTickCount());

//--- Initialize DoEasy library
   OnInitDoEasy();

//--- Check and remove remaining EA graphical objects
   if(IsPresentObects(prefix))
      ObjectsDeleteAll(0,prefix);

//--- Create the button panel
   if(!CreateButtons(InpButtShiftX,InpButtShiftY))
      return INIT_FAILED;
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);
//--- Reset states of the buttons for working using pending requests
   for(int i=0;i<14;i++)
     {
      ButtonState(butt_data[i].name+"_PRICE",false);
      ButtonState(butt_data[i].name+"_TIME",false);
     }

//--- Check playing a standard sound by macro substitution and a custom sound by description
   engine.PlaySoundByDescription(SND_OK);
   Sleep(600);
   engine.PlaySoundByDescription(TextByLanguage("Звук упавшей монетки 2","Falling coin 2"));

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
```

For the distance of the pending request activation price, the distance of less than five points is equal to five points, while the minimum
delay of at least one bar of the current timeframe is set for the delay in bars.

The buttons for enabling pending requests are simply made
inactive. Since this is a test EA, these buttons are needed for test reasons only. We do not need to track their status.

In the function of creating the panel buttons, add the variable storing
the width of new buttons. Also, create the new buttons enabling pending
requests in yet another loop:

```
//+------------------------------------------------------------------+
//| Create the buttons panel                                         |
//+------------------------------------------------------------------+
bool CreateButtons(const int shift_x=20,const int shift_y=0)
  {
   int h=18,w=82,offset=2,wpt=14;
   int cx=offset+shift_x+wpt*2+2,cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+3*h+1;
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

   h=18; offset=2;
   cx=offset+shift_x; cy=offset+shift_y+(h+1)*(TOTAL_BUTT/2)+3*h+1;
   x=cx; y=cy;
   shift=0;
   for(int i=0;i<14;i++)
     {
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name+"_PRICE",((i>6 && i<11) || i>10 ? x+wpt*2+w*2+5 : x),y,wpt,h,"P",(i<4 ? clrGreen : i>6 && i<11 ? clrChocolate : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text+" \"P\"");
         return false;
        }
      if(!ButtonCreate(butt_data[i].name+"_TIME",((i>6 && i<11) || i>10 ? x+wpt*2+w*2+5+wpt+1 : x+wpt+1),y,wpt,h,"T",(i<4 ? clrGreen : i>6 && i<11 ? clrChocolate : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text+" \"T\"");
         return false;
        }
     }
   ChartRedraw(0);
   return true;
  }
//+------------------------------------------------------------------+
```

In the function setting the states of buttons (active button color), add
setting the color of active buttons for trading using pending requests:

```
//+------------------------------------------------------------------+
//| Set the button status                                            |
//+------------------------------------------------------------------+
void ButtonState(const string name,const bool state)
  {
   ObjectSetInteger(0,name,OBJPROP_STATE,state);
//--- Trailing activation button
   if(name==butt_data[TOTAL_BUTT-1].name)
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'220,255,240');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
//--- Buttons enabling pending requests
   if(StringFind(name,"_PRICE")>0 || StringFind(name,"_TIME")>0)
     {
      if(state)
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'255,220,90');
      else
         ObjectSetInteger(0,name,OBJPROP_BGCOLOR,C'240,240,240');
     }
  }
//+------------------------------------------------------------------+
```

Add the codes handling pressing the buttons for working with pending requests
to the function handling button pressing:

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
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
         //--- If the pending request creation buttons are not pressed, open Buy
         if(!pending_buy)
            engine.OpenBuy(lot,Symbol(),magic,stoploss,takeprofit);   // No comment - the default comment is to be set
         //--- Otherwise, create a pending request for opening a Buy position
         else
           {
            int id=engine.OpenBuyPending(lot,Symbol(),magic,stoploss,takeprofit);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY)+"_PRICE"))
                 {
                  double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                  double control_value=NormalizeDouble(ask-distance_pending_request*SymbolInfoDouble(NULL,SYMBOL_POINT),(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS));
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_ASK,control_value,EQUAL_OR_LESS,ask);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY)+"_TIME"))
                 {
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
              }
            CPendRequest *req_obj=engine.GetPendRequestByID((uchar)id);
            if(req_obj==NULL)
               return;
            if(engine.TradingGetLogLevel(Symbol())>LOG_LEVEL_NO_MSG)
              {
               ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ADD_CRITERIONS)," #",req_obj.ID(),":");
               req_obj.PrintActivations();
              }
           }
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set BuyLimit
         if(!pending_buy_limit)
            engine.PlaceBuyLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStop
         if(!pending_buy_stop)
            engine.PlaceBuyStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStopLimit
         if(!pending_buy_stoplimit)
            engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStopLimit","Pending order BuyStopLimit"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- If the pending request creation buttons are not pressed, open Sell
         if(!pending_sell)
            engine.OpenSell(lot,Symbol(),magic,stoploss,takeprofit);  // No comment - the default comment is to be set
         //--- Otherwise, create a pending request for opening a Sell position
         else
           {
            int id=engine.OpenSellPending(lot,Symbol(),magic,stoploss,takeprofit);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL)+"_PRICE"))
                 {
                  double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                  double control_value=NormalizeDouble(bid+distance_pending_request*SymbolInfoDouble(NULL,SYMBOL_POINT),(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS));
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_BID,control_value,EQUAL_OR_MORE,bid);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL)+"_TIME"))
                 {
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
              }
            CPendRequest *req_obj=engine.GetPendRequestByID((uchar)id);
            if(req_obj==NULL)
               return;
            if(engine.TradingGetLogLevel(Symbol())>LOG_LEVEL_NO_MSG)
              {
               ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ADD_CRITERIONS)," #",req_obj.ID(),":");
               req_obj.PrintActivations();
              }
           }
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set SellLimit
         if(!pending_sell_limit)
            engine.PlaceSellLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellLimit","Pending SellLimit order"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set SellStop
         if(!pending_sell_stop)
            engine.PlaceSellStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set SellStopLimit
         if(!pending_sell_stoplimit)
            engine.PlaceSellStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
         //--- Otherwise, do nothing in this version
         else
           {

           }
        }
      //--- If the BUTT_CLOSE_BUY button is pressed: Close Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list and for the current symbol only
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            //--- Get the Buy position object and close a position by ticket
            COrder* position=list.At(index);
            if(position!=NULL)
               engine.ClosePosition((ulong)position.Ticket());
           }
        }
      //--- If the BUTT_CLOSE_BUY2 button is pressed: Close the half of the Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Buy positions from the list and for the current symbol only
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Buy position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            //--- Close the Buy position partially
            if(position!=NULL)
               engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
           }
        }
      //--- If the BUTT_CLOSE_BUY_BY_SELL button is pressed: Close Buy with the maximum profit by the opposite Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL))
        {
         //--- In case of a hedging account
         if(engine.IsHedge())
           {
            CArrayObj *list_buy=NULL, *list_sell=NULL;
            //--- Get the list of all open positions
            CArrayObj* list=engine.GetListMarketPosition();
            if(list==NULL)
               return;
            //--- Select only current symbol positions from the list
            list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);

            //--- Select only Buy positions from the list
            list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
            if(list_buy==NULL)
               return;
            //--- Sort the list by profit considering commission and swap
            list_buy.Sort(SORT_BY_ORDER_PROFIT_FULL);
            //--- Get the index of the Buy position with the maximum profit
            int index_buy=CSelect::FindOrderMax(list_buy,ORDER_PROP_PROFIT_FULL);

            //--- Select only Sell positions from the list
            list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
            if(list_sell==NULL)
               return;
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
               //--- Close the Buy position by the opposite Sell one
               if(position_buy!=NULL && position_sell!=NULL)
                  engine.ClosePositionBy((ulong)position_buy.Ticket(),(ulong)position_sell.Ticket());
              }
           }
        }

      //--- If the BUTT_CLOSE_SELL button is pressed: Close Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list and for the current symbol only
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            //--- Get the Sell position object and close a position by ticket
            COrder* position=list.At(index);
            if(position!=NULL)
               engine.ClosePosition((ulong)position.Ticket());
           }
        }
      //--- If the BUTT_CLOSE_SELL2 button is pressed: Close the half of the Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL2))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only Sell positions from the list and for the current symbol only
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
         list=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
         //--- Sort the list by profit considering commission and swap
         list.Sort(SORT_BY_ORDER_PROFIT_FULL);
         //--- Get the index of the Sell position with the maximum profit
         int index=CSelect::FindOrderMax(list,ORDER_PROP_PROFIT_FULL);
         if(index>WRONG_VALUE)
           {
            COrder* position=list.At(index);
            //--- Close the Sell position partially
            if(position!=NULL)
               engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
           }
        }
      //--- If the BUTT_CLOSE_SELL_BY_BUY button is pressed: Close Sell with the maximum profit by the opposite Buy with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY))
        {
         //--- In case of a hedging account
         if(engine.IsHedge())
           {
            CArrayObj *list_buy=NULL, *list_sell=NULL;
            //--- Get the list of all open positions
            CArrayObj* list=engine.GetListMarketPosition();
            if(list==NULL)
               return;
            //--- Select only current symbol positions from the list
            list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);

            //--- Select only Sell positions from the list
            list_sell=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_SELL,EQUAL);
            if(list_sell==NULL)
               return;
            //--- Sort the list by profit considering commission and swap
            list_sell.Sort(SORT_BY_ORDER_PROFIT_FULL);
            //--- Get the index of the Sell position with the maximum profit
            int index_sell=CSelect::FindOrderMax(list_sell,ORDER_PROP_PROFIT_FULL);

            //--- Select only Buy positions from the list
            list_buy=CSelect::ByOrderProperty(list,ORDER_PROP_TYPE,POSITION_TYPE_BUY,EQUAL);
            if(list_buy==NULL)
               return;
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
               //--- Close the Sell position by the opposite Buy one
               if(position_sell!=NULL && position_buy!=NULL)
                  engine.ClosePositionBy((ulong)position_sell.Ticket(),(ulong)position_buy.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_ALL is pressed: Close all positions starting with the one with the least profit
      else if(button==EnumToString(BUTT_CLOSE_ALL))
        {
         //--- Get the list of all open positions
         CArrayObj* list=engine.GetListMarketPosition();
         //--- Select only current symbol positions from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
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
               engine.ClosePosition((ulong)position.Ticket());
              }
           }
        }
      //--- If the BUTT_DELETE_PENDING button is pressed: Remove pending orders starting from the oldest one
      else if(button==EnumToString(BUTT_DELETE_PENDING))
        {
         //--- Get the list of all orders
         CArrayObj* list=engine.GetListMarketPendings();
         //--- Select only current symbol orders from the list
         list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
         if(list!=NULL)
           {
            //--- Sort the list by placement time
            list.Sort(SORT_BY_ORDER_TIME_OPEN);
            int total=list.Total();
            //--- In a loop from an order with the longest time
            for(int i=total-1;i>=0;i--)
              {
               COrder* order=list.At(i);
               if(order==NULL)
                  continue;
               //--- delete the order by its ticket
               engine.DeleteOrder((ulong)order.Ticket());
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
      //--- "Unpress" the button (if this is neither a trailing button, nor the buttons enabling pending requests)
      if(button!=EnumToString(BUTT_TRAILING_ALL) && StringFind(button,"_PRICE")<0 && StringFind(button,"_TIME")<0)
         ButtonState(button_name,false);
      //--- If the BUTT_TRAILING_ALL button or the buttons enabling pending requests are pressed
      else
        {
         //--- Set the active button color for the button enabling trailing
         if(button==EnumToString(BUTT_TRAILING_ALL))
           {
            ButtonState(button_name,true);
            trailing_on=true;
           }

         //--- Buying
         //--- Set the active button color for the button enabling pending requests for opening Buy by price or time
         if(button==EnumToString(BUTT_BUY)+"_PRICE" || button==EnumToString(BUTT_BUY)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_buy=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyLimit by price or time
         if(button==EnumToString(BUTT_BUY_LIMIT)+"_PRICE" || button==EnumToString(BUTT_BUY_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_buy_limit=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyStop by price or time
         if(button==EnumToString(BUTT_BUY_STOP)+"_PRICE" || button==EnumToString(BUTT_BUY_STOP)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_buy_stop=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyStopLimit by price or time
         if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE" || button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_buy_stoplimit=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_buy=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing 1/2 Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY2)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY2)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_buy2=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Buy by an opposite Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_buy_by_sell=true;
           }

         //--- Selling
         //--- Set the active button color for the button enabling pending requests for opening Sell by price or time
         if(button==EnumToString(BUTT_SELL)+"_PRICE" || button==EnumToString(BUTT_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_sell=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing SellLimit by price or time
         if(button==EnumToString(BUTT_SELL_LIMIT)+"_PRICE" || button==EnumToString(BUTT_SELL_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_sell_limit=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing SellStop by price or time
         if(button==EnumToString(BUTT_SELL_STOP)+"_PRICE" || button==EnumToString(BUTT_SELL_STOP)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_sell_stop=true;

           }
         //--- Set the active button color for the button enabling pending requests for placing SellStopLimit by price or time
         if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE" || button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_sell_stoplimit=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_sell=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing 1/2 Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL2)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL2)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_sell2=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Sell by an opposite Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME")
           {
            ButtonState(button_name,true);
            pending_close_sell_by_buy=true;
           }
        }
      //--- re-draw the chart
      ChartRedraw();
     }
   //--- Return a color for the inactive buttons
   else
     {
      //--- trailing button
      if(button==EnumToString(BUTT_TRAILING_ALL))
        {
         ButtonState(button_name,false);
         trailing_on=false;
        }

      //--- Buying
      //--- the button enabling pending requests for opening Buy by price
      if(button==EnumToString(BUTT_BUY)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for opening Buy by time
      if(button==EnumToString(BUTT_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyLimit by price
      if(button==EnumToString(BUTT_BUY_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_buy_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyLimit by time
      if(button==EnumToString(BUTT_BUY_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_buy_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyStop by price
      if(button==EnumToString(BUTT_BUY_STOP)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_buy_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyStop by time
      if(button==EnumToString(BUTT_BUY_STOP)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_buy_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyStopLimit by price
      if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_buy_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyStopLimit by time
      if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_buy_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Buy by price
      if(button==EnumToString(BUTT_CLOSE_BUY)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Buy by time
      if(button==EnumToString(BUTT_CLOSE_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing 1/2 Buy by price
      if(button==EnumToString(BUTT_CLOSE_BUY2)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_buy2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY2)+"_TIME"));
        }
      //--- the button enabling pending requests for closing 1/2 Buy by time
      if(button==EnumToString(BUTT_CLOSE_BUY2)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_buy2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY2)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Buy by an opposite Sell by price
      if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_buy_by_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Buy by an opposite Sell by time
      if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_buy_by_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE"));
        }

      //--- Selling
      //--- the button enabling pending requests for opening Sell by price
      if(button==EnumToString(BUTT_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for opening Sell by time
      if(button==EnumToString(BUTT_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellLimit by price
      if(button==EnumToString(BUTT_SELL_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_sell_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellLimit by time
      if(button==EnumToString(BUTT_SELL_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_sell_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellStop by price
      if(button==EnumToString(BUTT_SELL_STOP)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_sell_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellStop by time
      if(button==EnumToString(BUTT_SELL_STOP)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_sell_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellStopLimit by price
      if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_sell_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellStopLimit by time
      if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_sell_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Sell by price
      if(button==EnumToString(BUTT_CLOSE_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Sell by time
      if(button==EnumToString(BUTT_CLOSE_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing 1/2 Sell by price
      if(button==EnumToString(BUTT_CLOSE_SELL2)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_sell2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL2)+"_TIME"));
        }
      //--- the button enabling pending requests for closing 1/2 Sell by time
      if(button==EnumToString(BUTT_CLOSE_SELL2)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_sell2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL2)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Sell by an opposite Buy by price
      if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE")
        {
         ButtonState(button_name,false);
         pending_close_sell_by_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Sell by an opposite Buy by time
      if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pending_close_sell_by_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE"));
        }
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

The function is quite large but the code is commented in detail and needs no explanation. If you have any questions, feel free to ask them in the
comments below.

Let's compile the EA. By default, the shift of the pending request price is equal to 50 points while delay in bars is equal to five bars. Leave these
settings unchanged and launch the EA in the strategy tester.

Enable the pending request activation buttons for opening a Buy position
by price and time. Then wait for activation of pending requests.

After that, enable the pending request activation button for opening a
Sell position by time only and wait for the pending request activation:

![](https://c.mql5.com/2/37/pUuYcSiC6R.gif)

As we can see from the journal entries, buy pending requests are generated and activation conditions are set for them. When the price and time
reach the specified conditions, both pending requests are activated and pending request objects are removed due to their activation.


Then we create a sell pending request which is activated after five bars. The request is removed as an executed one after a position is opened.

### What's next?

In the next article, we will continue the development of the pending trading request concept and implement placing pending orders by
condition.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7521#node00)

**Previous articles within the series:**

[Part 1. Concept, data management](https://www.mql5.com/en/articles/5654)

[Part 2. Collection of historical orders and deals](https://www.mql5.com/en/articles/5669)

[Part \\
3\. Collection of market orders and positions, arranging the search](https://www.mql5.com/en/articles/5687)

[Part 4. \\
Trading events. Concept](https://www.mql5.com/en/articles/5724)

[Part 5. Classes and collection of trading events. \\
Sending events to the program](https://www.mql5.com/en/articles/6211)

[Part 6. Netting account events](https://www.mql5.com/en/articles/6383)

[Part \\
7\. StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

[Part \\
8\. Order and position modification events](https://www.mql5.com/en/articles/6595)

[Part 9. Compatibility with MQL4 — \\
Preparing data](https://www.mql5.com/en/articles/6651)

[Part 10. Compatibility with MQL4 - Events of opening a position and \\
activating pending orders](https://www.mql5.com/en/articles/6767)

[Part 11. Compatibility with MQL4 - Position closure \\
events](https://www.mql5.com/en/articles/6921)

[Part 12. Account object class and account object collection](https://www.mql5.com/en/articles/6952)

[Part 13. Account object events](https://www.mql5.com/en/articles/6995)

[Part \\
14\. Symbol object](https://www.mql5.com/en/articles/7014)

[Part 15. Symbol object collection](https://www.mql5.com/en/articles/7041)

[Part \\
16\. Symbol collection events](https://www.mql5.com/en/articles/7071)

[Part 17. Interactivity of library objects](https://www.mql5.com/en/articles/7124)

[Part 18. Interactivity of account and any other library objects](https://www.mql5.com/en/articles/7149)

[Part \\
19\. Class of library messages](https://www.mql5.com/en/articles/7176)

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
26\. Working with pending trading requests - First implementation (opening positions)](https://www.mql5.com/en/articles/7394)

[Part \\
27\. Working with pending trading requests - Placing pending orders](https://www.mql5.com/en/articles/7418)

[Part 28. \\
Working with pending trading requests - Closure, removal and modification](https://www.mql5.com/en/articles/7438)

[Part \\
29\. Working with pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

[Part 30. \\
Pending trading requests - managing request objects](https://www.mql5.com/en/articles/7481)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7521](https://www.mql5.com/ru/articles/7521)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7521.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7521/mql5.zip "Download MQL5.zip")(3659.61 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7521/mql4.zip "Download MQL4.zip")(3659.6 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/336949)**

![Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://www.mql5.com/en/articles/7536)

We continue the development of the functionality allowing users to trade using pending requests. In this article, we are going to implement the ability to place pending orders under certain conditions.

![Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://c.mql5.com/2/37/Article_Logo__2.png)[Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://www.mql5.com/en/articles/7417)

In this article, we will discuss the idea of creating a multicurrency monitor of trading signals and will develop a future application structure along with its prototype, as well as create its framework for further operation. The article presents a step-by-step creation of a flexible multicurrency application which will enable the generation of trading signals and which will assist traders in finding the desired signals.

![Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://c.mql5.com/2/38/OLAP_02.png)[Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://www.mql5.com/en/articles/7535)

In this article we will continue dealing with the OLAP technology applied to trading. We will expand the functionality presented in the first two articles. This time we will consider the operational analysis of quotes. We will put forward and test the hypotheses on trading strategies based on aggregated historical data. The article presents Expert Advisors for studying bar patterns and adaptive trading.

![Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://c.mql5.com/2/37/jlp_0d3zw11j.png)[Econometric approach to finding market patterns: Autocorrelation, Heat Maps and Scatter Plots](https://www.mql5.com/en/articles/5451)

The article presents an extended study of seasonal characteristics: autocorrelation heat maps and scatter plots. The purpose of the article is to show that "market memory" is of seasonal nature, which is expressed through maximized correlation of increments of arbitrary order.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/7521&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070430921611482548)

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