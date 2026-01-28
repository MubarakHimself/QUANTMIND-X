---
title: Library for easy and quick development of MetaTrader programs (part XXV): Handling errors returned by the trade server
url: https://www.mql5.com/en/articles/7365
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:38:08.774246
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7365&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070443797923436013)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7365#node01)
- [Implementation](https://www.mql5.com/en/articles/7365#node02)
- [Testing](https://www.mql5.com/en/articles/7365#node03)
- [What's next?](https://www.mql5.com/en/articles/7365#node04)


### Concept

We have already implemented the [verification of the valid parameters](https://www.mql5.com/en/articles/7286)
of the terminal, account and trading symbol, as well as the [auto correction of invalid \\
trading order parameters](https://www.mql5.com/en/articles/7326). Now it only remains to implement the handling of server responses to sent trading orders.


After we send a trading order to the server, we need to check the response. The server returns the absence of errors or the error codes we need to
handle.

We are going to perform the handling in exactly the same way as we process invalid trading order parameters:

- **No error** — an order has been successfully queued for execution,

- **Disable trading for the EA** — for example, a complete ban on trading operations from the server side,

- **Exit the trading method** — for example, it is impossible to successfully send an order to the server, a position is already
closed or a pending order is removed,

- **Correct trading request parameters and repeat** — there are some invalid values in trading order parameters. Most
likely, the data has been changed when preparing a server request, and now the appropriate adjustment is required,

- **Update data and repeat** — server data has changed, however there is no need to adjust trading request values,

- **Wait and repeat** — waiting is required, for example, if the price is close to one of position stop levels, the FreezeLevel
parameter disables modification since the stop order may have already been activated. Waiting allows you to wait either for a stop
order activation and canceling a trading request, or the price leaving the freeze area, so that the order is successfully sent to the
server,

- **Create a pending request** — to be discussed in the next article.


The return codes are more numerous compared to the ones implemented for fixing possible errors in a trading order. Besides, not every code can
be corrected to repeat a request. To exclude errors that can be fixed, we will try to handle and send them back to a trading order.

In the methods of sending trading requests, arrange the loop for resending a trading order to the server in the methods of sending trading
requests. In other words, if we receive an error after the first request to the server, we will send the trading order as many times as there are
defined trading attempts for the trading class — either till the order is successfully sent to the server or before all attempts are made.


If all attempts of sending the order to the server are unsuccessful, return false from the
trading method. In this case, we are able to see the last error code in the calling program. The code is returned by the trade server, so that you
are able to decide on handling the error.

Now it is time for implementation.

### Implementation

In the section of the simplified access to the account object properties of the **CAccount** account class in the **Account.mqh**
file, **add the method returning the flag of working on the hedge type account:**

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the account object properties  |
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
   bool              IsHedge(void)                                   const { return this.MarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;                        }
//--- Return the account's real properties
```

In the **Defines.mqh** file, add the macro substitution to specify the
default number of trading attempts for a trading class.

In this article, we are going to prepare for creation of pending
requests, so we need the timer for the trading class.

Therefore, let's write
the trading class timer parameters right away:

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
#define TOTAL_TRY                      (5)                        // Default number of trading attempts
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
#define COLLECTION_ORD_PAUSE           (250)                      // Orders and deals collection timer pause in milliseconds
#define COLLECTION_ORD_COUNTER_STEP    (16)                       // Increment of the orders and deals collection timer counter
#define COLLECTION_ORD_COUNTER_ID      (1)                        // Orders and deals collection timer counter ID
//--- Parameters of the account collection timer
#define COLLECTION_ACC_PAUSE           (1000)                     // Account collection timer pause in milliseconds
#define COLLECTION_ACC_COUNTER_STEP    (16)                       // Account timer counter increment
#define COLLECTION_ACC_COUNTER_ID      (2)                        // Account timer counter ID
//--- Symbol collection timer 1 parameters
#define COLLECTION_SYM_PAUSE1          (100)                      // Pause of the symbol collection timer 1 in milliseconds (for scanning market watch symbols)
#define COLLECTION_SYM_COUNTER_STEP1   (16)                       // Increment of the symbol timer 1 counter
#define COLLECTION_SYM_COUNTER_ID1     (3)                        // Symbol timer 1 counter ID
//--- Symbol collection timer 2 parameters
#define COLLECTION_SYM_PAUSE2          (300)                      // Pause of the symbol collection timer 2 in milliseconds (for events of the market watch symbol list)
#define COLLECTION_SYM_COUNTER_STEP2   (16)                       // Increment of the symbol timer 2 counter
#define COLLECTION_SYM_COUNTER_ID2     (4)                        // Symbol timer 2 counter ID
//--- Trading class timer parameters
#define COLLECTION_REQ_PAUSE           (300)                      // Trading class timer pause in milliseconds
#define COLLECTION_REQ_COUNTER_STEP    (16)                       // Trading class timer counter increment
#define COLLECTION_REQ_COUNTER_ID      (5)                        // Trading class timer counter ID
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x7779)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777A)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777B)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777C)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777D)                   // Symbol collection list ID
//--- Data parameters for file operations
#define DIRECTORY                      ("DoEasy\\")               // Library directory for storing object folders
#define RESOURCE_DIR                   ("DoEasy\\Resource\\")     // Library directory for storing resource folders
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
#define SYMBOLS_COMMON_TOTAL           (1000)                     // Total number of working symbols
//+------------------------------------------------------------------+
```

Add two flags to the list of flags of the trade server error handling methods — the
flag of the pending order price error and the flag of the stop limit order
price error. Also, add the method of correcting trading order
parameters to the methods for handling errors and trade server return codes:

```
//+------------------------------------------------------------------+
//| Flags indicating the trading request error handling methods      |
//+------------------------------------------------------------------+
enum ENUM_TRADE_REQUEST_ERR_FLAGS
  {
   TRADE_REQUEST_ERR_FLAG_NO_ERROR                 =  0,    // No error
   TRADE_REQUEST_ERR_FLAG_FATAL_ERROR              =  1,    // Disable trading for an EA (critical error) - exit
   TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR             =  2,    // Library internal error - exit
   TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST            =  4,    // Error in the list - handle (ENUM_ERROR_CODE_PROCESSING_METHOD)
   TRADE_REQUEST_ERR_FLAG_PRICE_ERROR              =  8,    // Placement price error
   TRADE_REQUEST_ERR_FLAG_LIMIT_ERROR              =  16,   // Limit order price error
  };
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

**Write the new message indices in the Datas.mqh file:**

```
//--- CTrading
   MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED,           // Trade operations are not allowed in the terminal (the AutoTrading button is disabled)
   MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED,                 // EA is not allowed to trade (F7 --> Common --> Allow Automated Trading)
   MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED,            // Trading is disabled for the current account
   MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED,         // Trading on the trading server side is disabled for EAs on the current account
   MSG_LIB_TEXT_REQUEST_REJECTED_DUE,                 // Request was rejected before sending to the server due to:
   MSG_LIB_TEXT_INVALID_REQUEST,                      // Invalid request:
   MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR,                 // Insufficient funds for performing a trade
   MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED,            // Exceeded maximum allowed aggregate volume of orders and positions in one direction
   MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME,              // Request volume is less than the minimum acceptable one
   MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME,              // Request volume exceeds the maximum acceptable one
   MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED,             // Close by is disabled
   MSG_LIB_TEXT_INVALID_VOLUME_STEP,                  // Request volume is not a multiple of the minimum lot change step gradation
   MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL,             // Symbols of opposite positions are not equal
   MSG_LIB_TEXT_SL_LESS_STOP_LEVEL,                   // StopLoss violates requirements for symbol's StopLevel
   MSG_LIB_TEXT_TP_LESS_STOP_LEVEL,                   // TakeProfit violates requirements for symbol's StopLevel
   MSG_LIB_TEXT_PRICE_LESS_STOP_LEVEL,                // Order distance in points is less than a value allowed by symbol's StopLevel parameter
   MSG_LIB_TEXT_LIMIT_LESS_STOP_LEVEL,                // Limit order distance in points relative to a stop order is less than a value allowed by symbol's StopLevel parameter
   MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL,                 // The distance from the price to StopLoss is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL,                 // The distance from the price to TakeProfit is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL,                 // The distance from the price to an order activation level is less than a value allowed by symbol's FreezeLevel parameter
   MSG_LIB_TEXT_UNSUPPORTED_SL_TYPE,                  // Unsupported StopLoss parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_TP_TYPE,                  // Unsupported TakeProfit parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PR_TYPE,                  // Unsupported price parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PL_TYPE,                  // Unsupported limit order price parameter type (should be 'int' or 'double')
   MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ,        // Unsupported price parameter type in a request
   MSG_LIB_TEXT_TRADING_DISABLE,                      // Trading disabled for the EA until the reason is eliminated
   MSG_LIB_TEXT_TRADING_OPERATION_ABORTED,            // Trading operation is interrupted
   MSG_LIB_TEXT_CORRECTED_TRADE_REQUEST,              // Correcting trading request parameters
   MSG_LIB_TEXT_CREATE_PENDING_REQUEST,               // Creating a pending request
   MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT,          // Unable to correct a lot
   MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ,           // Failed to create a pending request
   MSG_LIB_TEXT_TRY_N,                                // Trading attempt #

  };
```

**and message texts:**

```
   {"Дистанция установки ордера в пунктах меньше разрешённой параметром StopLevel символа","Distance to place order in points less than allowed by symbol's StopLevel"},
   {"Дистанция установки лимит-ордера относительно стоп-ордера меньше разрешённой параметром StopLevel символа","Distance to place limit order relative to stop order less than allowed by symbol's StopLevel"},
   {"Дистанция от цены до StopLoss меньше разрешённой параметром FreezeLevel символа","Distance from price to StopLoss less than allowed by symbol's FreezeLevel"},
   {"Дистанция от цены до TakeProfit меньше разрешённой параметром FreezeLevel символа","Distance from price to TakeProfit less than allowed by symbol's FreezeLevel"},
   {"Дистанция от цены до цены срабатывания ордера меньше разрешённой параметром FreezeLevel символа","Distance from price to order triggering price less than allowed by symbol's FreezeLevel"},
   {"Неподдерживаемый тип параметра StopLoss (необходимо int или double)","Unsupported StopLoss parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра TakeProfit (необходимо int или double)","Unsupported TakeProfit parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра цены (необходимо int или double)","Unsupported price parameter type (int or double required)"},
   {"Неподдерживаемый тип параметра цены limit-ордера (необходимо int или double)","Unsupported type of price parameter for limit order (int or double required)"},
   {"Неподдерживаемый тип параметра цены в запросе","Unsupported price parameter type in request"},
   {"Торговля отключена для эксперта до устранения причины запрета","Trading for expert disabled till this ban eliminated"},
   {"Торговая операция прервана","Trading operation aborted"},
   {"Корректировка параметров торгового запроса ...","Correction of trade request parameters ..."},
   {"Создание отложенного запроса","Create pending request"},
   {"Нет возможности скорректировать лот","Unable to correct lot"},
   {"Не удалось создать отложенный запрос","Failed to create pending request"},
   {"Торговая попытка #","Trading attempt #"},

  };
```

The **TradeObj.mqh** base trading object file has undergone minor changes.

**The method for placing a pending order**
**features the order type parameter defining the execution**(the
default one was used before that):

```
//--- Place an order
   bool                       SetOrder(const ENUM_ORDER_TYPE type,
                                       const double volume,
                                       const double price,
                                       const double sl=0,
                                       const double tp=0,
                                       const double price_stoplimit=0,
                                       const ulong magic=ULONG_MAX,
                                       const string comment=NULL,
                                       const datetime expiration=0,
                                       const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

Now if the passed value exceeds -1, the
value passed to the method is used. Otherwise, the
default parameter value is applied:

```
//+------------------------------------------------------------------+
//| Set an order                                                     |
//+------------------------------------------------------------------+
bool CTradeObj::SetOrder(const ENUM_ORDER_TYPE type,
                         const double volume,
                         const double price,
                         const double sl=0,
                         const double tp=0,
                         const double price_stoplimit=0,
                         const ulong magic=ULONG_MAX,
                         const string comment=NULL,
                         const datetime expiration=0,
                         const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                         const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   ::ResetLastError();
   //--- If an invalid order type has been passed, write the error code and description, send the message to the journal and return 'false'
   if(type==ORDER_TYPE_BUY || type==ORDER_TYPE_SELL || type==ORDER_TYPE_CLOSE_BY
      #ifdef __MQL4__ || type==ORDER_TYPE_BUY_STOP_LIMIT || type==ORDER_TYPE_SELL_STOP_LIMIT #endif )
     {
      this.m_result.retcode=MSG_LIB_SYS_INVALID_ORDER_TYPE;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_INVALID_ORDER_TYPE),OrderTypeDescription(type));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action      =  TRADE_ACTION_PENDING;
   this.m_request.symbol      =  this.m_symbol;
   this.m_request.magic       =  (magic==ULONG_MAX ? this.m_magic : magic);
   this.m_request.volume      =  volume;
   this.m_request.type        =  type;
   this.m_request.stoplimit   =  price_stoplimit;
   this.m_request.price       =  price;
   this.m_request.sl          =  sl;
   this.m_request.tp          =  tp;
   this.m_request.expiration  =  expiration;
   this.m_request.type_time   =  (type_time>WRONG_VALUE ? type_time : this.m_type_time);
   this.m_request.type_filling=  (type_filling>WRONG_VALUE ? type_filling : this.m_type_filling);
   this.m_request.comment     =  (comment==NULL ? this.m_comment : comment);
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
      this.m_result.order=ticket;
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

The prices in trading orders have also been corrected. Previously, if the chart was based on Last prices, the price in the trading order was set
as Ask and Last. Now it is always Ask and Bid, regardless of prices used to construct a chart.

You can find other minor changes in the files attached below. There is no point in dwelling on them here.

In the **Trading.mqh** file, in the private section of the **CTrading** trading class, add the
list of pending requests and the variable for storing the number of
trading attempts:

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

In the future, we will use the list of trading requests to store the objects of the pending request class, while the
**m\_total\_try** variable is to contain the number of trading attemptsset
by default for a trading class in its constructor:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTrading::CTrading()
  {
   this.m_list_errors.Clear();
   this.m_list_errors.Sort();
   this.m_list_request.Clear();
   this.m_list_request.Sort();
   this.m_total_try=TOTAL_TRY;
   this.m_log_level=LOG_LEVEL_ALL_MSG;
   this.m_is_trade_disable=false;
   this.m_err_handling_behavior=ERROR_HANDLING_BEHAVIOR_CORRECT;
   ::ZeroMemory(this.m_request);
  }
//+------------------------------------------------------------------+
```

Here, clear the list of pending requests and set the sorted list flag for it.

**Add the price of placing a limit order for a StopLimit type order to the**
**parameters of the method for checking the price relative to StopLevel:**

```
bool CheckPriceByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj,const double limit=0);
```

**Add the check to the method itself:**

```
//+------------------------------------------------------------------+
//| Return the flag checking the validity of the distance            |
//| from the price to the placement level by StopLevel               |
//+------------------------------------------------------------------+
bool CTrading::CheckPriceByStopLevel(const ENUM_ORDER_TYPE order_type,const double price,const CSymbol *symbol_obj,const double limit=0)
  {
   double lv=symbol_obj.TradeStopLevel()*symbol_obj.Point();
   double pr=(this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY ? symbol_obj.Ask() : symbol_obj.Bid());
   return
     (limit==0 ?
      //--- Order placement prices relative to the price
      (
       order_type==ORDER_TYPE_SELL_STOP         ||
       order_type==ORDER_TYPE_SELL_STOP_LIMIT   ||
       order_type==ORDER_TYPE_BUY_LIMIT         ?  price<(pr-lv)     :
       order_type==ORDER_TYPE_BUY_STOP          ||
       order_type==ORDER_TYPE_BUY_STOP_LIMIT    ||
       order_type==ORDER_TYPE_SELL_LIMIT        ?  price>(pr+lv)     :
       true
      ) :
      //--- Limit order placement prices relative to the stop order price
      (
       order_type==ORDER_TYPE_BUY_STOP_LIMIT    ?  limit<(price-lv)  :
       order_type==ORDER_TYPE_SELL_STOP_LIMIT   ?  limit>(price+lv)  :
      true
      )
     );
  }
//+------------------------------------------------------------------+
```

Here, if the limit order price is equal to zero, check the
prices of stop and limit orders, otherwise check stop limit order prices
(the limit order price relative to the stop order price a stop limit order is activated at).

**Pass the error code to the method returning the way of handling errors**
**and add the pointer to the trading object in the error correction**
**method:**

```
//--- Return the error handling method
   ENUM_ERROR_CODE_PROCESSING_METHOD   ResultProccessingMethod(const uint result_code);
//--- Correct errors
   ENUM_ERROR_CODE_PROCESSING_METHOD   RequestErrorsCorrecting(MqlTradeRequest &request,const ENUM_ORDER_TYPE order_type,const uint spread_multiplier,CSymbol *symbol_obj,CTradeObj *trade_obj);
```

Since we have multiple methods of opening positions and placing orders, all of them turned out to be almost the same. The difference is only in the
types of opened positions and placed orders.

To avoid writing the same code for each method, **declare and implement two private**
**methods — for opening a position and placing**
**pending orders:**

```
//--- (1) Open a position, (2) place a pending order
   template<typename SL,typename TP>
   bool                 OpenPosition(const ENUM_POSITION_TYPE type,
                                    const double volume,
                                    const string symbol,
                                    const ulong magic=ULONG_MAX,
                                    const SL sl=0,
                                    const TP tp=0,
                                    const string comment=NULL,
                                    const ulong deviation=ULONG_MAX);
   template<typename PS,typename PL,typename SL,typename TP>
   bool                 PlaceOrder( const ENUM_ORDER_TYPE order_type,
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
public:
//--- Constructor
```

**In the public section of the class, declare the timer we**
**will need to work with the class of pending requests, the method returning the list**
**of pending requests and the method setting the number of trading attempts:**

```
public:
//--- Constructor
                        CTrading();
//--- Timer
   void                 OnTimer(void);
//--- Get the pointers to the lists (make sure to call the method in program's OnInit() since the symbol collection list is created there)
   void                 OnInit(CAccount *account,CSymbolsCollection *symbols,CMarketCollection *market,CHistoryCollection *history)
                          {
                           this.m_account=account;
                           this.m_symbols=symbols;
                           this.m_market=market;
                           this.m_history=history;
                          }
//--- Return the list of (1) errors and (2) pending requests
   CArrayInt           *GetListErrors(void)                             { return &this.m_list_errors; }
   CArrayObj           *GetListRequests(void)                           { return &this.m_list_request;}
//--- Set the number of trading attempts
   void                 SetTotalTry(const uchar number)                 { this.m_total_try=number;    }
//--- Check limitations and errors
```

**Let's improve the specification of the method for closing positions by a closed**
**volume. The default is WRONG\_VALUE** **—**
**full position closure, otherwise** **— partial closure by a specified volume:**

```
bool ClosePosition(const ulong ticket,const double volume=WRONG_VALUE,const string comment=NULL,const ulong deviation=ULONG_MAX);
```

**In the specifications of pending order placement methods, add [the \**\
**types of executing orders by excess](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling)**. Previously, the default value set for the class was used. Now, the order execution type
value is selected based on the value passed to the method. In case of WRONG\_VALUE, the
specified value is set by default, otherwise the one passed to the method is applied:

```
//--- Set (1) BuyStop, (2) BuyLimit, (3) BuyStopLimit pending order
   template<typename PS,typename SL,typename TP>
   bool                 PlaceBuyStop(const double volume,
                                           const string symbol,
                                           const PS price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename SL,typename TP>
   bool                 PlaceBuyLimit(const double volume,
                                           const string symbol,
                                           const PS price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   bool                 PlaceBuyStopLimit(const double volume,
                                           const string symbol,
                                           const PS price_stop,
                                           const PL price_limit,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

//--- Set (1) SellStop, (2) SellLimit, (3) SellStopLimit pending order
   template<typename PS,typename SL,typename TP>
   bool                 PlaceSellStop(const double volume,
                                           const string symbol,
                                           const PS price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename SL,typename TP>
   bool                 PlaceSellLimit(const double volume,
                                           const string symbol,
                                           const PS price,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   bool                 PlaceSellStopLimit(const double volume,
                                           const string symbol,
                                           const PS price_stop,
                                           const PL price_limit,
                                           const SL sl=0,
                                           const TP tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const datetime expiration=0,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
//--- Modify a pending order
   template<typename PS,typename PL,typename SL,typename TP>
   bool                 ModifyOrder(const ulong ticket,
                                          const PS price=WRONG_VALUE,
                                          const SL sl=WRONG_VALUE,
                                          const TP tp=WRONG_VALUE,
                                          const PL limit=WRONG_VALUE,
                                          datetime expiration=WRONG_VALUE,
                                          const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                          const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

**Let's implement the timer** **. So far, we will prepare a workpiece of handling the list of pending requests:**

```
//+------------------------------------------------------------------+
//| Timer                                                            |
//+------------------------------------------------------------------+
void CTrading::OnTimer(void)
  {
   int total=this.m_list_request.Total();
   for(int i=total-1;i>WRONG_VALUE;i--)
     {

     }
  }
//+------------------------------------------------------------------+
```

**Implementing the method returning the ways to handle trade server return codes:**

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

      //--- No quotes to process the request
      case 10021  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000;    // ERROR_CODE_PROCESSING_METHOD_WAIT;
      //--- Too frequent requests
      case 10024  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT
      //--- An order or a position is frozen
      case 10029  :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)10000;   // ERROR_CODE_PROCESSING_METHOD_WAIT;

      //--- Request handling error
      case 10011  :  return ERROR_CODE_PROCESSING_METHOD_PENDING;
      //--- Auto trading disabled by the client terminal
      case 10027  :  return ERROR_CODE_PROCESSING_METHOD_PENDING;
      //--- No connection to the trade server
      case 10031  :  return ERROR_CODE_PROCESSING_METHOD_PENDING;

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

Here all is simple: the method receives the code obtained from the server after
sending a trading request to it. Then the codes indicating an error can be fixed are handled by the error fixing method, the codes
requiring data update and re-sending a request are handled appropriately, etc.

Since MQL5 and MQL4 servers return different error
codes, the method features the conditional compilation for MQL4 and MQL5.


All codes requiring the same type of handling are grouped in a single case of the switch
operator and return the unified method of handling the trade server return code.

**Implementing the method of handling trade server errors:**

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
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_TRADE_MODE_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_TRADE_MODE_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Close only
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_TRADE_MODE_CLOSEONLY))
     {
      trade_obj.SetResultRetcode(MSG_SYM_TRADE_MODE_CLOSEONLY);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Market orders are disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_MARKET_ORDER_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_MARKET_ORDER_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Limit orders are disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_LIMIT_ORDER_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_LIMIT_ORDER_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Stop orders are disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_STOP_ORDER_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_STOP_ORDER_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- StopLimit orders are disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_STOP_LIMIT_ORDER_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_STOP_LIMIT_ORDER_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Sell only
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_TRADE_MODE_SHORTONLY))
     {
      trade_obj.SetResultRetcode(MSG_SYM_TRADE_MODE_SHORTONLY);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Buy only
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_TRADE_MODE_LONGONLY))
     {
      trade_obj.SetResultRetcode(MSG_SYM_TRADE_MODE_LONGONLY);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- CloseBy orders are disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_SYM_CLOSE_BY_ORDER_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_SYM_CLOSE_BY_ORDER_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Exceeded maximum allowed aggregate volume of orders and positions in one direction
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Close by is disabled
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Symbols of opposite positions are not equal
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Unsupported price parameter type in a request
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Trading disabled for the EA until the reason is eliminated
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(MSG_LIB_TEXT_TRADING_DISABLE))
     {
      trade_obj.SetResultRetcode(MSG_LIB_TEXT_TRADING_DISABLE);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- The maximum number of pending orders is reached
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(10033))
     {
      trade_obj.SetResultRetcode(10033);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }
//--- Reached the maximum order and position volume for this symbol
//--- write the error code to the base trading class object and return "exit from the trading method"
   if(this.IsPresentErorCode(10034))
     {
      trade_obj.SetResultRetcode(10034);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      return ERROR_CODE_PROCESSING_METHOD_EXIT;
     }

//--- Correcting trading request parameters
//--- Price, according to which stop orders are placed
   double price_set=(this.IsPresentErrorFlag(TRADE_REQUEST_ERR_FLAG_PRICE_ERROR) ? request.price : request.stoplimit);
//--- First, adjust stop orders relative to the order/position level
   if(this.IsPresentErorCode(MSG_LIB_TEXT_SL_LESS_STOP_LEVEL))
      request.sl=this.CorrectStopLoss(order_type,price_set,request.sl,symbol_obj,spread_multiplier);
   if(this.IsPresentErorCode(MSG_LIB_TEXT_TP_LESS_STOP_LEVEL))
      request.tp=this.CorrectTakeProfit(order_type,price_set,request.tp,symbol_obj,spread_multiplier);
//--- Pending orders price
   double shift=0;
   if(this.IsPresentErrorFlag(TRADE_REQUEST_ERR_FLAG_PRICE_ERROR))
     {
      price_set=request.price;
      request.price=this.CorrectPricePending(order_type,price_set,0,symbol_obj,spread_multiplier);
      shift=request.price-price_set;
      //--- If this is not a stop limit order, move stop orders by the calculated correcting order level shift
      if(request.stoplimit==0)
        {
         if(request.sl>0)
            request.sl=this.CorrectStopLoss(order_type,request.price,request.sl+shift,symbol_obj,spread_multiplier);
         if(request.tp>0)
            request.tp=this.CorrectTakeProfit(order_type,request.price,request.tp+shift,symbol_obj,spread_multiplier);
        }
     }
//--- The specified type of order execution by balance is not supported
   if(this.IsPresentErorCode(10030))
      request.type_filling=symbol_obj.GetCorrectTypeFilling();
//--- Invalid order expiration in a request -
   if(this.IsPresentErorCode(10022))
     {
      //--- if the expiration type is not supported as set by the expiration date and the expiration data is defined, reset the expiration date
      if(!symbol_obj.IsExpirationModeSpecified() && request.expiration>0)
         request.expiration=0;
     }
//--- View the list of remaining errors and correct trading request parameters
   for(int i=0;i<total;i++)
     {
      int err=this.m_list_errors.At(i);
      if(err==NULL)
         continue;
      switch(err)
        {
         //--- Correct an invalid volume and disabling stop levels in a trading request
         case MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME :
         case MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME :
         case MSG_LIB_TEXT_INVALID_VOLUME_STEP     :  request.volume=symbol_obj.NormalizedLot(request.volume);                                              break;
         case MSG_SYM_SL_ORDER_DISABLED            :  request.sl=0;                                                                                         break;
         case MSG_SYM_TP_ORDER_DISABLED            :  request.tp=0;                                                                                         break;

         //--- If unable to select the position lot, return "abort trading attempt" since the funds are insufficient even for the minimum lot
         case MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR    :  request.volume=this.CorrectVolume(request.price,order_type,symbol_obj,DFUN);
                                                      if(request.volume==0)
                                                        {
                                                         trade_obj.SetResultRetcode(MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT);
                                                         trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                         return ERROR_CODE_PROCESSING_METHOD_EXIT;                                                                                      break;
                                                        }
         //--- No quotes to process the request
         case 10021                                :  trade_obj.SetResultRetcode(10021);
                                                      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                      return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000; // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds
         //--- No connection to the trade server
         case 10031                                :  trade_obj.SetResultRetcode(10031);
                                                      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
                                                      return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000; // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds

         //--- Proximity to the order activation level is handled by five-second waiting - during this time, the price may go beyond the freeze level
         case MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL    :
         case MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL    :
         case MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL    :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000; // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds
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

The code comments of the method listing feature all actions directed at handling of errors returned by the trade server.

**Implement the private method for opening a position:**

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
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(this.m_request.volume,symbol_obj.Ask(),action,order_type,symbol_obj,trade_obj,DFUN,0,this.m_request.sl,this.m_request.tp);
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
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
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
         //--- If "Wait and repeat" is received as a result of sending a request -
         //--- in this implementation, we wait the number of milliseconds equal to the 'method' value and move on to the next iteration
         if(method==ERROR_CODE_PROCESSING_METHOD_WAIT)
           {
            ::Sleep(method);
            continue;
           }
         //--- If "Create a pending request" is received as a result of sending a request -
         //--- create a pending request with the trading request parameters and end the attempt loop
         if(method==ERROR_CODE_PROCESSING_METHOD_PENDING)
           {
            break;
           }
        }
     }
//--- Return the result of sending a trading request in a symbol trading object
   return res;
  }
//+------------------------------------------------------------------+
```

This method is commented in details directly in the listing and is to be used to open Buy
and Sell positions:

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CTrading::OpenBuy(const double volume,
                       const string symbol,
                       const ulong magic=ULONG_MAX,
                       const SL sl=0,
                       const TP tp=0,
                       const string comment=NULL,
                       const ulong deviation=ULONG_MAX)
  {
//--- Return the result of sending a trading request from the OpenPosition() method
   return this.OpenPosition(POSITION_TYPE_BUY,volume,symbol,magic,sl,tp,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CTrading::OpenSell(const double volume,
                        const string symbol,
                        const ulong magic=ULONG_MAX,
                        const SL sl=0,
                        const TP tp=0,
                        const string comment=NULL,
                        const ulong deviation=ULONG_MAX)
  {
//--- Return the result of sending a trading request from the OpenPosition() method
   return this.OpenPosition(POSITION_TYPE_SELL,volume,symbol,magic,sl,tp,comment,deviation);
  }
//+------------------------------------------------------------------+
```

The common private method for opening a position indicating the opened
position type is simply called in these methods.

**Implementing the private method for placing pending orders:**

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

//--- In case of trading limitations, funds insufficiency,
//--- there are limitations on StopLevel - play the error sound and exit
   this.m_request.volume=volume;
   this.m_request.type_filling=type_filling;
   this.m_request.type_time=type_time;
   this.m_request.expiration=expiration;
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
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
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
         //--- If "Wait and repeat" is received as a result of sending a request -
         //--- in this implementation, we wait the number of milliseconds equal to the 'method' value and move on to the next iteration
         if(method==ERROR_CODE_PROCESSING_METHOD_WAIT)
           {
            Sleep(method);
            continue;
           }
         //--- If "Create a pending request" is received as a result of sending a request -
         //--- create a pending request with the trading request parameters and end the attempt loop
         if(method==ERROR_CODE_PROCESSING_METHOD_PENDING)
           {
            break;
           }
        }
     }
//--- Return the result of sending a trading request in a symbol trading object
   return res;
  }
//+------------------------------------------------------------------+
```

This method is commented in details directly in the listing and is to be used to set various
types of pending orders:

```
//+------------------------------------------------------------------+
//| Place BuyStop pending order                                      |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
bool CTrading::PlaceBuyStop(const double volume,
                            const string symbol,
                            const PS price,
                            const SL sl=0,
                            const TP tp=0,
                            const ulong magic=ULONG_MAX,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_BUY_STOP,volume,symbol,price,0,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place BuyLimit pending order                                     |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
bool CTrading::PlaceBuyLimit(const double volume,
                             const string symbol,
                             const PS price,
                             const SL sl=0,
                             const TP tp=0,
                             const ulong magic=ULONG_MAX,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_BUY_LIMIT,volume,symbol,price,0,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place BuyStopLimit pending order                                 |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
bool CTrading::PlaceBuyStopLimit(const double volume,
                                 const string symbol,
                                 const PS price_stop,
                                 const PL price_limit,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const ulong magic=ULONG_MAX,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
#ifdef __MQL5__
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_BUY_STOP_LIMIT,volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time,type_filling);
//--- MQL4
#else
   return true;
#endif
  }
//+------------------------------------------------------------------+
//| Place SellStop pending order                                     |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
bool CTrading::PlaceSellStop(const double volume,
                             const string symbol,
                             const PS price,
                             const SL sl=0,
                             const TP tp=0,
                             const ulong magic=ULONG_MAX,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_SELL_STOP,volume,symbol,price,0,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place SellLimit pending order                                    |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
bool CTrading::PlaceSellLimit(const double volume,
                              const string symbol,
                              const PS price,
                              const SL sl=0,
                              const TP tp=0,
                              const ulong magic=ULONG_MAX,
                              const string comment=NULL,
                              const datetime expiration=0,
                              const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                              const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_SELL_LIMIT,volume,symbol,price,0,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place SellStopLimit pending order                                |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
bool CTrading::PlaceSellStopLimit(const double volume,
                                  const string symbol,
                                  const PS price_stop,
                                  const PL price_limit,
                                  const SL sl=0,
                                  const TP tp=0,
                                  const ulong magic=ULONG_MAX,
                                  const string comment=NULL,
                                  const datetime expiration=0,
                                  const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                  const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
#ifdef __MQL5__
//--- Return the result of sending a trading request using the PlaceOrder() method
   return this.PlaceOrder(ORDER_TYPE_SELL_STOP_LIMIT,volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time,type_filling);
   //--- MQL4
#else
   return true;
#endif
  }
//+------------------------------------------------------------------+
```

The remaining methods are necessary for closing positions and removing pending orders. The methods modifying positions and orders are
similar to private methods for opening positions/placing pending orders. All method codes have detailed comments. All files are attached
below.

**This concludes all the work with the trading class at this stage.**

Now it only remains to make some changes in the library's **CEngine** base object class.

Considering the floating nature of the minimum stop and pending order levels (StopLevel), we need to set the spread multiplier since a spread
multiplied by a certain value is often used in such cases to specify acceptable stop order distance. This means we need the method allowing to
set the spread multiplier for the trading class.

**Declare the following method**
**in the public section of the class:**

```
//--- Set the spread multiplier for symbol trading objects in the symbol collection
   void                 SetSpreadMultiplier(const uint value=1,const string symbol=NULL)  { this.m_trading.SetSpreadMultiplier(value,symbol);   }

//--- Open (1) Buy, (2) Sell position
```

The method simply calls the trading class method of the same name we already examined [in \\
the previous article](https://www.mql5.com/en/articles/7326) allowing to set both a single common multiplier for all used symbols and individual multipliers for specified
symbols.

Since the trading class will soon use the timer to work with pending requests, **create**
**the new timer counter for the trading class in the CEngine class constructor:**

```
//+------------------------------------------------------------------+
//| CEngine constructor                                              |
//+------------------------------------------------------------------+
CEngine::CEngine() : m_first_start(true),
                     m_last_trade_event(TRADE_EVENT_NO_EVENT),
                     m_last_account_event(WRONG_VALUE),
                     m_last_symbol_event(WRONG_VALUE),
                     m_global_error(ERR_SUCCESS)
  {
   this.m_is_hedge=#ifdef __MQL4__ true #else bool(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) #endif;
   this.m_is_tester=::MQLInfoInteger(MQL_TESTER);

   this.m_list_counters.Sort();
   this.m_list_counters.Clear();
   this.CreateCounter(COLLECTION_ORD_COUNTER_ID,COLLECTION_ORD_COUNTER_STEP,COLLECTION_ORD_PAUSE);
   this.CreateCounter(COLLECTION_ACC_COUNTER_ID,COLLECTION_ACC_COUNTER_STEP,COLLECTION_ACC_PAUSE);

   this.CreateCounter(COLLECTION_SYM_COUNTER_ID1,COLLECTION_SYM_COUNTER_STEP1,COLLECTION_SYM_PAUSE1);
   this.CreateCounter(COLLECTION_SYM_COUNTER_ID2,COLLECTION_SYM_COUNTER_STEP2,COLLECTION_SYM_PAUSE2);
   this.CreateCounter(COLLECTION_REQ_COUNTER_ID,COLLECTION_REQ_COUNTER_STEP,COLLECTION_REQ_PAUSE);

   ::ResetLastError();
   #ifdef __MQL5__
      if(!::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   //---__MQL4__
   #else
      if(!this.IsTester() && !::EventSetMillisecondTimer(TIMER_FREQUENCY))
        {
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_FAILED_CREATE_TIMER),(string)::GetLastError());
         this.m_global_error=::GetLastError();
        }
   #endif
   //---
  }
//+------------------------------------------------------------------+
```

**In the CEngine class timer, add the block of working with the trading**
**class timer:**

```
//+------------------------------------------------------------------+
//| CEngine timer                                                    |
//+------------------------------------------------------------------+
void CEngine::OnTimer(void)
  {
//--- Timer of the collections of historical orders and deals, as well as of market orders and positions
   int index=this.CounterIndex(COLLECTION_ORD_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the order, deal and position collections events
            if(counter.IsTimeDone())
               this.TradeEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.TradeEventsControl();
        }
     }
//--- Account collection timer
   index=this.CounterIndex(COLLECTION_ACC_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the account collection events
            if(counter.IsTimeDone())
               this.AccountEventsControl();
           }
         //--- If this is a tester, work with collection events by tick
         else
            this.AccountEventsControl();
        }
     }

//--- Timer 1 of the symbol collection (updating symbol quote data in the collection)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID1);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over, update quote data of all symbols in the collection
            if(counter.IsTimeDone())
               this.m_symbols.RefreshRates();
           }
         //--- In case of a tester, update quote data of all collection symbols by tick
         else
            this.m_symbols.RefreshRates();
        }
     }
//--- Timer 2 of the symbol collection (updating all data of all symbols in the collection and tracking symbl and symbol search events in the market watch window)
   index=this.CounterIndex(COLLECTION_SYM_COUNTER_ID2);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If the pause is over
            if(counter.IsTimeDone())
              {
               //--- update data and work with events of all symbols in the collection
               this.SymbolEventsControl();
               //--- When working with the market watch list, check the market watch window events
               if(this.m_symbols.ModeSymbolsList()==SYMBOLS_MODE_MARKET_WATCH)
                  this.MarketWatchEventsControl();
              }
           }
         //--- If this is a tester, work with events of all symbols in the collection by tick
         else
            this.SymbolEventsControl();
        }
     }
//--- Trading class timer
   index=this.CounterIndex(COLLECTION_REQ_COUNTER_ID);
   if(index>WRONG_VALUE)
     {
      CTimerCounter* counter=this.m_list_counters.At(index);
      if(counter!=NULL)
        {
         //--- If this is not a tester
         if(!this.IsTester())
           {
            //--- If unpaused, work with the list of pending requests
            if(counter.IsTimeDone())
               this.m_trading.OnTimer();
           }
         //--- In case of the tester, work with the list of pending orders by tick
         else
            this.m_trading.OnTimer();
        }
     }
  }
//+------------------------------------------------------------------+
```

**Slightly change the method for the complete position closure:**

```
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CEngine::ClosePosition(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX)
  {
   return this.m_trading.ClosePosition(ticket,WRONG_VALUE,comment,deviation);
  }
//+------------------------------------------------------------------+
```

Since we now have the common position closure method both for the complete and partial closure, we need to pass -1 as the closed position volume
for the complete position closure.

**This concludes the necessary changes and improvements for handling the trade server return codes.**

### Testing

To check handling of errors returned by the trade server, it would be reasonable to set the trading conditions causing errors, for example
execution delays. During the delay, the price changes causing the appropriate error.

To perform the test, let's use the EA from the [previous article](https://www.mql5.com/en/articles/7326#node04)
and save it to \\MQL5\\Experts\\TestDoEasy\ **Part25\** under the name **TestDoEasyPart25.mq5**.

We can launch the EA with no changes, but let's make some improvements first.

**In the block of the EA inputs, change**
**the default slippage from zero to five points and add the spread**
**multiplier:**

```
//--- input variables
input    ulong             InpMagic             =  123;  // Magic number
input    double            InpLots              =  0.1;  // Lots
input    uint              InpStopLoss          =  50;   // StopLoss in points
input    uint              InpTakeProfit        =  50;   // TakeProfit in points
input    uint              InpDistance          =  50;   // Pending orders distance (points)
input    uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input    uint              InpSlippage          =  5;    // Slippage in points
input    uint              InpSpreadMultiplier  =  1;    // Spread multiplier for adjusting stop-orders by StopLevel
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
```

In the library initialization function, set the spread multiplier for all
trading objects of all used symbols and comment out the block setting control
over the increase in symbol parameter values to avoid tracking and sending redundant entries to the tester journal:

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
//--- Set synchronous passing of orders for all used symbols
   engine.TradingSetAsyncMode(false);
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

Set the execution delay of 4 seconds in the strategy tester.

To do this, select "Custom delay ..." in the drop-down menu

![](https://c.mql5.com/2/37/NC1xkEXoaO.png)

... and enter 4000 milliseconds in the new input field:

![](https://c.mql5.com/2/37/terminal64_YityFkgFof.png)

Now, all trading orders sent to the server are delayed for four seconds in the tester.

Launch the EA in visual mode and attempt to open several positions and then close them in the fast market:

![](https://c.mql5.com/2/37/qNHyfomp3v.gif)

As we can see, it is not always possible to open a position as we get a requote. The EA makes the necessary number of trading attempts (the
default is not more than five). This is confirmed by the "Trading attempt" entries specifying the number of an attempt and featuring the
"Requote" clarification. When closing positions simultaneously, we obtain requotes again. The last position after five attempts was
never closed. I managed to close it manually after a few unsuccessful attempts. Anyway, the EA ran the algorithm built into the library with
the specified number of repeated trading attempts.

In the last MetaTrader 5 versions (starting from the build 2201), the tester features the ability to set parameters of a symbol the test
is performed on. Thus, it is possible to set trading limitations on a symbol and test the library behavior when the symbol limitations
are detected.

To call the symbol settings window, click the button to the right of the tested timeframe selection:

![](https://c.mql5.com/2/37/terminal64_IOYLnG2vXD.png)

Allow opening long positions only for a symbol and set the volume limitation of simultaneously opened positions and placed pending orders in
one direction to 0.5.

![](https://c.mql5.com/2/37/terminal64_FEADGVnfey.png)

Thus, we will be able to use long positions only and have the maximum total volume of buy positions and orders of no more than 0.5 lot in the market.
In other words, when opening a position with the lot of 0.1, we are able to open five positions only or place a single pending buy order and open
four positions:

![](https://c.mql5.com/2/37/Tvo9Cpy7N5.gif)

For more authenticity, we could disable auto closure of positions when a specified profit is exceeded. However, we can see that we were unable
to open a short position and received the warning that only buy positions are allowed on a symbol. Further on, when trying to open a number of
positions with their total volume exceeding 0.5 lots, we receive the message of the inability to open a position due to exceeding the maximum
total volume of positions and orders in one direction.

You can test this and many other features related to symbol parameters in the terminal beta version tester starting with the build 2201.


To get the latest beta version of the terminal, simply connect to MetaQuotes-Demo and select Latest Beta Version in the Help menu:

![](https://c.mql5.com/2/37/terminal64_SQnGhXeRWf.png)

### What's next?

In the next article, we will implement pending trading requests.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7365#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7365](https://www.mql5.com/ru/articles/7365)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7365.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7365/mql5.zip "Download MQL5.zip")(3603.44 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7365/mql4.zip "Download MQL4.zip")(3603.45 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/331543)**
(15)


![Igor Ryabchikov](https://c.mql5.com/avatar/avatar_na2.png)

**[Igor Ryabchikov](https://www.mql5.com/en/users/rigal)**
\|
22 Dec 2019 at 16:00

**Artyom Trishkin:**

**About the information recorded in the Magic Number value:**

you can use a different magik for each group to create different groups. For example, if the advisor's magic number is 123, then the magic number of the first group will be 124, the magic number of the second group will be 125, the magic number of the third group will be 126, and so on.

The library offers a different way of creating different groups - the number of each subgroup is stored directly in the Magic Number value. Then the EA's magic number is also a group identifier, but it is placed in an independent group called MagicID - the EA's magic number identifier. And there are two more groups. Each of them has 15 subgroups. And each of the subgroups can have its own identifier.

This will give **more** flexibility when working with groups.

For example: We want to move a grid of pending orders behind the price - add them to group 1 in subgroup 1. Group 1 moves behind the price. Subgroup 1 moves along the MA. Now we want to move some of those orders that move behind the price (group 1) by Parabolic SAR. We give them subgroup 2. Then group 1 moves after the price, but subgroup 1 moves by MA, and subgroup 2 moves by Parabolic SAR.

Orders are triggered, turning into positions - you can set your own groups for modification of Stop Loss, and your own subgroups in this group for modification by different values. Modification algorithms are written in subgroups.

In general - a flight of fancy. You can also use a simple magik, but you will have to invent the logic of tracking different groups yourself.

**On the second question:**

There is a CSelect class. It is available from the programme, and provides methods of selection and search you are writing about from all existing collections: Account, Event, Order, Symbol.

You can select the objects of each collection into a list based on all criteria. In the created list, you can reselect by refining criteria, you can find the maximum and minimum values for the selection criterion.

However, further on there will be custom functions (much later) for quick and convenient access to all properties of all collections and search in them.

But for now - only through CSelect, and when you need it. The class is static, so access to its methods via ":::" For example, CSelect::ByOrderProperty().

Yes, by the way, there is an example of its use in the programme right in the test EA - for example, in its trailing functions:

I figured out CSelect, but it turns out that I have to make a choice in each place where we need aggregate values: say, I need to calculate the total profit of a grid of positions in order to trailing it, this is one function. And in another I'm, say, adjusting the total take of a grid after opening, or closing one of the orders - and for that I need the total profit and aggregate volume.

And somewhere else I need to make a decision whether to open a new grid or continue the current one - I need to know how many orders I have already opened in this group.

All these values are calculated by one run through the list, which can be selected by a series of CSelects. But since I need them in different places, I would have to start an aggregate construction each time, recalculate at the beginning of a tick and then use them everywhere.

I thought it would be good to add such a construct to the library, with the ability to allocate it by a set of selection criteria, such as symbol, magick (with groups), order type.

And let it collect simple statistics: [number of orders](https://www.mql5.com/en/docs/trading/orderstotal "MQL5 documentation: OrdersTotal function"), total volume, total profit... a collection of orders, by the way, since we have already selected it.

Otherwise, either to create an auxiliary structure each time, or CSelects and iteration in each place.

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
22 Dec 2019 at 17:49

**rigal:**

I figured out CSelect, but it turns out that I have to make a choice in each place where we need aggregate values: say, I need to calculate the total profit of a grid of positions in order to trawl it, that's one function. And in another, say, I'm adjusting the total take of a grid after opening or closing one of the orders - and for that I need the total profit and aggregate volume.

And somewhere else I need to make a decision whether to open a new grid or continue the current one - I need to know how many orders I already have open in this group.

All these values are calculated by one run through the list, which can be selected by a series of CSelects. But since they are needed in different places, I have to start an aggregating construction each time, recalculate it at the beginning of a tick and then use it everywhere.

I thought it would be good to add such a construction to the library, with the possibility to allocate it by a set of selection criteria, e.g. symbol, magick (with groups), order type.

And let it collect simple statistics: [number of orders](https://www.mql5.com/en/docs/trading/orderstotal "MQL5 documentation: OrdersTotal function"), total volume, total profit... a collection of orders, by the way, since we've sampled it already.

Otherwise, either to create an auxiliary structure each time, or CSelects and iteration in each place.

And you propose to make the calculations in the library itself heavier? There's already a lot of things to be calculated there.

Why not do what you suggest directly in the EA instead of the library? There's no difference where to calculate it all. But someone needs it (you) and someone doesn't. And why does he need unnecessary calculations?

In order not to count the same thing in different functions every time, you should create global lists in the Expert Advisor, where you can put the information you need everywhere. And in separate functions, and only as needed, take these public lists, and from them get the data necessary only inside the function, and return them.

The library provides a set of data, gives the possibility of selecting this data in any combination, and gives tools for solving tasks that are not quick enough to do from scratch.

![Igor Ryabchikov](https://c.mql5.com/avatar/avatar_na2.png)

**[Igor Ryabchikov](https://www.mql5.com/en/users/rigal)**
\|
23 Dec 2019 at 04:43

**Artyom Trishkin:**

And you're suggesting that the calculations in the library itself be weighted down? There's already a lot of counting going on there.

Why not do what you suggest directly in the EA instead of the library? There is no difference where to calculate it all. But someone needs it (you) and someone doesn't. And why does he need extra calculations?

In order not to count the same thing in different functions every time, you should create global lists in the Expert Advisor, where you can put the information you need everywhere. And in separate functions, and only as needed, take these public lists, and from them get the data necessary only inside the function, and return them.

The library provides a set of data, provides the possibility of selecting this data in any combination, and gives tools for solving tasks that are not quick enough to do from scratch.

Well, I propose to make it optional.

To start a "subscription".

But if this idea doesn't seem worthy of consideration to you, I will certainly write the wrapper myself.

Thanks again for what has already been done - and ahead for all that is still in the plans (I here and there meet reservations about plans to continue the development of the library).

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
23 Dec 2019 at 05:16

**rigal:**

Well I suggest counting optionally.

Make a "subscription."

But if you do not find this idea worthy of consideration, I will certainly write the wrapper myself.

Thanks again for what's already done - and ahead for whatever else is in the pipeline (I've seen reservations here and there about plans to continue developing the library)

There will be similar functionality in the plans. And, yes, there's still a lot of planning - less than a third of it is done.


![Igor Ryabchikov](https://c.mql5.com/avatar/avatar_na2.png)

**[Igor Ryabchikov](https://www.mql5.com/en/users/rigal)**
\|
24 Dec 2019 at 07:19

I'm watching closely :)


![Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://c.mql5.com/2/37/MQL5-avatar-season_research.png)[Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://www.mql5.com/en/articles/7038)

In this article we will view seasonal characteristics of financial time series using Boxplot diagrams. Each separate boxplot (or box-and-whiskey diagram) provides a good visualization of how values are distributed along the dataset. Boxplots should not be confused with the candlestick charts, although they can be visually similar.

![Extending Strategy Builder Functionality](https://c.mql5.com/2/37/Article_Logo__1.png)[Extending Strategy Builder Functionality](https://www.mql5.com/en/articles/7361)

In the previous two articles, we discussed the application of Merrill patterns to various data types. An application was developed to test the presented ideas. In this article, we will continue working with the Strategy Builder, to improve its efficiency and to implement new features and capabilities.

![Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://c.mql5.com/2/37/MQL5-avatar-doeasy__14.png)[Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://www.mql5.com/en/articles/7394)

In this article, we are going to store some data in the value of the orders and positions magic number and start the implementation of pending requests. To check the concept, let's create the first test pending request for opening market positions when receiving a server error requiring waiting and sending a repeated request.

![Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__6.png)[Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters](https://www.mql5.com/en/articles/7326)

In this article, we will have a look at the handler of invalid trading order parameters and improve the trading event class. Now all trading events (both single ones and the ones occurred simultaneously within one tick) will be defined in programs correctly.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/7365&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070443797923436013)

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