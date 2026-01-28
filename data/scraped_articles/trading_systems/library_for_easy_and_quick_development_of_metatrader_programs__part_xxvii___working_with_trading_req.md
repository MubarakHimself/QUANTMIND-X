---
title: Library for easy and quick development of MetaTrader programs (part XXVII): Working with trading requests - placing pending orders
url: https://www.mql5.com/en/articles/7418
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:57.481320
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/7418&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070441199468221922)

MetaTrader 5 / Examples


### Contents

- [Preparing data](https://www.mql5.com/en/articles/7418#node01)
- [Eliminating trading class shortcomings and creating a pending request for placing orders](https://www.mql5.com/en/articles/7418#node02)
- [Testing](https://www.mql5.com/en/articles/7418#node03)
- [What's next?](https://www.mql5.com/en/articles/7418#node04)


In the [previous article](https://www.mql5.com/en/articles/7394), we started the implementation of pending trading
requests and created the first pending request for opening a position in case an error is received in the trading class after sending a
request to the server. In this article, we will resume the development of pending requests and implement creation of a pending request in
case of errors occurred when setting pending orders.

While testing the trading class, I have detected some shortcomings. In particular, when initializing symbol's trading objects in the class
constructor, hard-set default values were set for them. Not all of those values may have been supported in the symbol specification. This
caused server errors when attempting to place pending orders — the server activated the unsupported order expiration type error. This
error was not corrected anywhere and eventually led to the inability to place a pending order. When sending a trading request with default
values, some unsupported data was also sent to the trading request. To solve this issue, I had to specify correct data corresponding to the
appropriate symbol specification directly in the trading request.

This required the knowledge of a symbol specification and the manual
input of accurate values directly in the program code instead of auto correction of values by the library itself. To simplify things, we are
going to make minor changes in the trading class handling logic. All symbol trading objects are to be initialized by auto selection of
correct values in the EA's OnInit() handler. The -1 values are passed to the trading methods of the trading class by default for order filling
and expiration types indicating it is time to use preset correct default values. If another value is passed from the program, it is applied
instead. If the value turns out to be invalid, it is corrected when handling errors in the trading class.

### Preparing data

Apart from fixing the trading class, we are also going to add the request description (displaying all its parameters in the journal) to the
pending request object class. This makes it easier to test working with pending request objects.

First, add all the necessary messages
to the library message array.

**Open the Datas.mqh file and add indices of new messages to it:**

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
   MSG_LIB_PROP_AS_IN_ORDER,                          // According to the order expiration mode

   MSG_LIB_SYS_ERROR,                                 // Error
```

...

```
   MSG_LIB_TEXT_FAILING_CREATE_PENDING_REQ,           // Failed to create a pending request
   MSG_LIB_TEXT_TRY_N,                                // Trading attempt #
   MSG_LIB_TEXT_RE_TRY_N,                             // Repeated trading attempt #

   MSG_LIB_TEXT_REQUEST_ACTION,                       // Type of a performed action
   MSG_LIB_TEXT_REQUEST_MAGIC,                        // EA stamp (magic number)
   MSG_LIB_TEXT_REQUEST_ORDER,                        // Order ticket
   MSG_LIB_TEXT_REQUEST_SYMBOL,                       // Name of a trading instrument
   MSG_LIB_TEXT_REQUEST_VOLUME,                       // Requested volume of a deal in lots
   MSG_LIB_TEXT_REQUEST_PRICE,                        // Price
   MSG_LIB_TEXT_REQUEST_STOPLIMIT,                    // StopLimit
   MSG_LIB_TEXT_REQUEST_SL,                           // Stop Loss
   MSG_LIB_TEXT_REQUEST_TP,                           // Take Profit
   MSG_LIB_TEXT_REQUEST_DEVIATION,                    // Maximum price deviation
   MSG_LIB_TEXT_REQUEST_TYPE,                         // Order type
   MSG_LIB_TEXT_REQUEST_TYPE_FILLING,                 // Order filling type
   MSG_LIB_TEXT_REQUEST_TYPE_TIME,                    // Order lifetime type
   MSG_LIB_TEXT_REQUEST_EXPIRATION,                   // Order expiration date
   MSG_LIB_TEXT_REQUEST_COMMENT,                      // Order comment
   MSG_LIB_TEXT_REQUEST_POSITION,                     // Position ticket
   MSG_LIB_TEXT_REQUEST_POSITION_BY,                  // Opposite position ticket

   MSG_LIB_TEXT_REQUEST_ACTION_DEAL,                  // Place a market order
   MSG_LIB_TEXT_REQUEST_ACTION_PENDING,               // Place a pending order
   MSG_LIB_TEXT_REQUEST_ACTION_SLTP,                  // Change open position Stop Loss and Take Profit
   MSG_LIB_TEXT_REQUEST_ACTION_MODIFY,                // Change parameters of the previously placed trading order
   MSG_LIB_TEXT_REQUEST_ACTION_REMOVE,                // Remove previously placed pending order
   MSG_LIB_TEXT_REQUEST_ACTION_CLOSE_BY,              // Close a position by an opposite one
   MSG_LIB_TEXT_REQUEST_ACTION_UNCNOWN,               // Unknown trading operation type

   MSG_LIB_TEXT_REQUEST_ORDER_FILLING_FOK,            // Order is executed in the specified volume only, otherwise it is canceled
   MSG_LIB_TEXT_REQUEST_ORDER_FILLING_IOK,            // Order is filled within an available volume, while the unfilled one is canceled
   MSG_LIB_TEXT_REQUEST_ORDER_FILLING_RETURN,         // Order is filled within an available volume, while the unfilled one remains

   MSG_LIB_TEXT_REQUEST_ORDER_TIME_GTC,               // Order is valid till explicitly canceled
   MSG_LIB_TEXT_REQUEST_ORDER_TIME_DAY,               // Order is valid only during the current trading day
   MSG_LIB_TEXT_REQUEST_ORDER_TIME_SPECIFIED,         // Order is valid till the expiration date
   MSG_LIB_TEXT_REQUEST_ORDER_TIME_SPECIFIED_DAY,     // Order is valid till 23:59:59 of a specified day

   MSG_LIB_TEXT_REQUEST_DATAS,                        // Trading request parameters
   MSG_LIB_TEXT_PEND_REQUEST_DATAS,                   // Pending trading request parameters
   MSG_LIB_TEXT_PEND_REQUEST_CREATED,                 // Pending request created
   MSG_LIB_TEXT_PEND_REQUEST_DELETED,                 // Pending request is removed due to its expiration

   MSG_LIB_TEXT_PEND_REQUEST_PRICE_CREATE,            // Price at the moment of request generation
   MSG_LIB_TEXT_PEND_REQUEST_TIME_CREATE,             // Request creation time
   MSG_LIB_TEXT_PEND_REQUEST_TIME_ACTIVATE,           // Request activation time
   MSG_LIB_TEXT_PEND_REQUEST_WAITING,                 // Waiting time between trading attempts
   MSG_LIB_TEXT_PEND_REQUEST_CURRENT_ATTEMPT,         // Current trading attempt
   MSG_LIB_TEXT_PEND_REQUEST_TOTAL_ATTEMPTS,          // Total number of trading attempts
   MSG_LIB_TEXT_PEND_REQUEST_ID,                      // Trading request ID
   MSG_LIB_TEXT_PEND_REQUEST_RETCODE,                 // Return code a request is based on
   MSG_LIB_TEXT_PEND_REQUEST_TYPE,                    // Pending request type

   MSG_LIB_TEXT_PEND_REQUEST_BY_ERROR,                // Pending request generated based on the server return code
   MSG_LIB_TEXT_PEND_REQUEST_BY_REQUEST,              // Pending request created by request
   MSG_LIB_TEXT_PEND_REQUEST_WAITING_ONSET,           // Wait for the first trading attempt

  };
```

**and message texts corresponding to the indices:**

```
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
   {"В соответствии с режимом истечения ордера","In accordance with the order expiration mode"},

   {"Ошибка ","Error"},
```

...

```
   {"Не удалось создать отложенный запрос","Failed to create pending request"},
   {"Торговая попытка #","Trading attempt #"},
   {"Повторная торговая попытка #","Retry trading attempt #"},

   {"Тип выполняемого действия","Trade operation type"},
   {"Штамп эксперта (magic number)","Expert Advisor ID (magic number)"},
   {"Тикет ордера","Order ticket"},
   {"Имя торгового инструмента","Trade symbol"},
   {"Запрашиваемый объем сделки в лотах","Requested volume for a deal in lots"},
   {"Цена","Price"},
   {"Уровень StopLimit ордера","StopLimit level of the order"},
   {"Уровень Stop Loss ордера","Stop Loss level of the order"},
   {"Уровень Take Profit ордера","Take Profit level of the order"},
   {"Максимальное отклонение от цены","Maximal deviation from the price"},
   {"Тип ордера","Order type"},
   {"Тип ордера по исполнению","Order execution type"},
   {"Тип ордера по времени действия","Order expiration type"},
   {"Срок истечения ордера","Order expiration time"},
   {"Комментарий к ордеру","Order comment"},
   {"Тикет позиции","Position ticket"},
   {"Тикет встречной позиции","Opposite position ticket"},

   {"Поставить рыночный ордер","Place market order"},
   {"Установить отложенный ордер","Place pending order"},
   {"Изменить значения Stop Loss и Take Profit у открытой позиции","Modify Stop Loss and Take Profit values of an opened position"},
   {"Изменить параметры ранее установленного торгового ордера","Modify the parameters of the order placed previously"},
   {"Удалить ранее выставленный отложенный ордер","Delete the pending order placed previously"},
   {"Закрыть позицию встречной","Close a position by an opposite one"},
   {"Неизвестный тип торговой операции","Unknown trade action type"},

   {"Ордер исполняется исключительно в указанном объеме, иначе отменяется (FOK)","The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)"},
   {"Ордер исполняется на доступный объем, неисполненный отменяется (IOK)","The order is executed on the available volume, the unfulfilled is canceled (IOK)"},
   {"Ордер исполняется на доступный объем, неисполненный остаётся (Return)","The order is executed at an available volume, unfulfilled remains in the market (Return)"},

   {"Ордер действителен до явной отмены","Good till cancel order"},
   {"Ордер действителен только в течение текущего торгового дня","Good till current trade day order"},
   {"Ордер действителен до даты истечения","Good till expired order"},
   {"Ордер действителен до 23:59:59 указанного дня","The order will be effective till 23:59:59 of the specified day"},
   {"Параметры торгового запроса","Trade request's parameters"},
   {"Параметры отложенного торгового запроса","Pending trade request's parameters"},
   {"Создан отложенный запрос","Pending request created"},
   {"Отложенный запрос удалён в связи с окончанием времени его действия","Pending request deleted due to expiration"},

   {"Цена в момент создания запроса: ","Price at time of request create: "},
   {"Время создания запроса: ","Request creation time: "},
   {"Время активации запроса: ","Request activation time: "},
   {"Время ожидания между торговыми попытками: ","Waiting time between trading attempts: "},
   {"Текущая торговая попытка: ","Current trading attempt: "},
   {"Общее количество торговых попыток: ","Total trade attempts: "},
   {"Идентификатор торгового запроса: ","Trade request ID: "},
   {"Код возврата, на основании которого создан запрос: ","Return code based on which the request was created: "},
   {"Тип отложенного запроса: ","Pending request type: "},
   {"Отложенный запрос, созданный по коду возврата сервера","Pending request that was created as a result of the server code"},
   {"Отложенный запрос, созданный по запросу","Pending request created by request"},
   {"Ожидание наступления времени первой торговой попытки","Waiting for the onset time of the first trading attempt"},

  };
//+---------------------------------------------------------------------+
```

Apart from the above shortcomings, I have noticed that collection IDs overlap with object type IDs in the standard library. In particular, the
ID of the COLLECTION\_HISTORY\_ID (historical orders and deals collection) is **0x7779** [corresponding](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clisttype) to the standard library's type of the CList [dynamic list of CObject class instances and its descendants](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist). It is unreasonable for object IDs to have the same value.

Below is a list of standard library object IDs and the corresponding hexadecimal values:

| Base class           **CObject** | [Type \<br> = 0](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjecttype) |
| Data collections    **CArrayChar** | [Type \<br> = 0x77](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraychar/carraychartype) |
| Data collections    **CArrayShort** | [Type \<br> = 0x82](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayshort/carrayshorttype) |
| Data collections    **CArrayInt** | [Type = \<br> 0x82](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayint/carrayinttype) |
| Data collections    **CArrayLong** | [Type = 0x84](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraylong/carraylongtype) |
| Data collections    **CArrayFloat** | [Type \<br> = 0x87](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayfloat/carrayfloattype) |
| Data collections    **CArrayDouble** | [Type \<br> = 0x87](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraydouble/carraydoubletype) |
| Data collections    **CArrayString** | [Type \<br> = 0x89](https://www.mql5.com/en/docs/standardlibrary/datastructures/carraystring/carraystringtype) |
| Data collections    **CArrayObj** | [Type \<br> = 0x7778](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjtype) |
| Data collections    **CList** | [Type \<br> = 0x7779](https://www.mql5.com/en/docs/standardlibrary/datastructures/clist/clisttype) |
| Graphical objects Base class **CChartObject** | [Type \<br> = 0x8888](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes/cchartobject/cchartobjecttype) |
| Price charts     **CChart** | [Type \<br> = 0x1111](https://www.mql5.com/en/docs/standardlibrary/cchart/cchartype) |

The list is most probably incomplete. As we can see, the list object
type matches the ID of the historical orders and deals collection set for the library.

**Let's fix this in the Defines.mqh file by**
**increasing the values of all collection IDs by 1:**

```
//--- Collection list IDs
#define COLLECTION_HISTORY_ID          (0x777A)                   // Historical collection list ID
#define COLLECTION_MARKET_ID           (0x777B)                   // Market collection list ID
#define COLLECTION_EVENTS_ID           (0x777C)                   // Event collection list ID
#define COLLECTION_ACCOUNT_ID          (0x777D)                   // Account collection list ID
#define COLLECTION_SYMBOLS_ID          (0x777E)                   // Symbol collection list ID
//--- Data parameters for file operations
```

Since I want to implement the ability to trade using pending requests, the two types of pending requests are to be implemented:

- a request generated based on the trade server error code (these are the requests we are currently implementing);

- a pending request created by request from a program (trading by pending requests, which is to be implemented later).


Therefore, **I am going to introduce the concept of "request type" and the IDs**
**matching it to separate request types:**

```
//--- Symbol parameters
#define CLR_DEFAULT                    (0xFF000000)               // Default color
#define SYMBOLS_COMMON_TOTAL           (1000)                     // Total number of working symbols
//--- Pending request type IDs
#define PENDING_REQUEST_ID_TYPE_ERR    (1)                        // Type of a pending request created based on the server return code
#define PENDING_REQUEST_ID_TYPE_REQ    (2)                        // Type of a pending request created by request
//+------------------------------------------------------------------+
```

**At the very end of the Defines.mqh file, add the enumeration of pending request types:**

```
//+------------------------------------------------------------------+
//| Pending request type                                             |
//+------------------------------------------------------------------+
enum ENUM_PENDING_REQUEST_TYPE
  {
   PENDING_REQUEST_TYPE_ERROR=PENDING_REQUEST_ID_TYPE_ERR,  // Pending request created based on the return code or error
   PENDING_REQUEST_TYPE_REQUEST=PENDING_REQUEST_ID_TYPE_REQ,// Pending request created by request
  };
//+------------------------------------------------------------------+
```

To display trading request descriptions in the journal, we need to prepare the appropriate functions.

In the **DELib.mqh**
file of service functions, add all the necessary functions for generating a message from the pre-defined texts set in the Datas.mqh file and
the values of the properties displayed by the functions.

**The functions for displaying the order filling mode**
**and expiration type:**

```
//+------------------------------------------------------------------+
//| Return the order filling mode description                        |
//+------------------------------------------------------------------+
string OrderTypeFillingDescription(const ENUM_ORDER_TYPE_FILLING type)
  {
   return
     (
      type==ORDER_FILLING_FOK    ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_FILLING_FOK)   :
      type==ORDER_FILLING_IOC    ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_FILLING_IOK)   :
      type==ORDER_FILLING_RETURN ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_FILLING_RETURN):
      type==WRONG_VALUE          ? "WRONG_VALUE"   :  EnumToString(type)
     );
  }
//+------------------------------------------------------------------+
//| Return the order expiration type description                     |
//+------------------------------------------------------------------+
string OrderTypeTimeDescription(const ENUM_ORDER_TYPE_TIME type)
  {
   return
     (
      type==ORDER_TIME_GTC             ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_TIME_GTC)            :
      type==ORDER_TIME_DAY             ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_TIME_DAY)            :
      type==ORDER_TIME_SPECIFIED       ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_TIME_SPECIFIED)      :
      type==ORDER_TIME_SPECIFIED_DAY   ?  CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER_TIME_SPECIFIED_DAY)  :
      type==WRONG_VALUE                ? "WRONG_VALUE"   :  EnumToString(type)
     );
  }
//+------------------------------------------------------------------+
```

**The functions for displaying the description of the [MqlTradeRequest \**\
**trading request structure](https://www.mql5.com/en/docs/constants/structures/mqltraderequest):**

```
//+------------------------------------------------------------------+
//| Display the trading request description in the journal           |
//+------------------------------------------------------------------+
void PrintRequestDescription(const MqlTradeRequest &request)
  {
   string datas=
     (
      " - "+RequestActionDescription(request)+"\n"+
      " - "+RequestMagicDescription(request)+"\n"+
      " - "+RequestOrderDescription(request)+"\n"+
      " - "+RequestSymbolDescription(request)+"\n"+
      " - "+RequestVolumeDescription(request)+"\n"+
      " - "+RequestPriceDescription(request)+"\n"+
      " - "+RequestStopLimitDescription(request)+"\n"+
      " - "+RequestStopLossDescription(request)+"\n"+
      " - "+RequestTakeProfitDescription(request)+"\n"+
      " - "+RequestDeviationDescription(request)+"\n"+
      " - "+RequestTypeDescription(request)+"\n"+
      " - "+RequestTypeFillingDescription(request)+"\n"+
      " - "+RequestTypeTimeDescription(request)+"\n"+
      " - "+RequestExpirationDescription(request)+"\n"+
      " - "+RequestCommentDescription(request)+"\n"+
      " - "+RequestPositionDescription(request)+"\n"+
      " - "+RequestPositionByDescription(request)
     );
   Print("================== ",CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)," ==================\n",datas,"\n");
  }
//+------------------------------------------------------------------+
//| Return the executed action type description                      |
//+------------------------------------------------------------------+
string RequestActionDescription(const MqlTradeRequest &request)
  {
   int code_descr=
     (
      request.action==TRADE_ACTION_DEAL      ?  MSG_LIB_TEXT_REQUEST_ACTION_DEAL       :
      request.action==TRADE_ACTION_PENDING   ?  MSG_LIB_TEXT_REQUEST_ACTION_PENDING    :
      request.action==TRADE_ACTION_SLTP      ?  MSG_LIB_TEXT_REQUEST_ACTION_SLTP       :
      request.action==TRADE_ACTION_MODIFY    ?  MSG_LIB_TEXT_REQUEST_ACTION_MODIFY     :
      request.action==TRADE_ACTION_REMOVE    ?  MSG_LIB_TEXT_REQUEST_ACTION_REMOVE     :
      request.action==TRADE_ACTION_CLOSE_BY  ?  MSG_LIB_TEXT_REQUEST_ACTION_CLOSE_BY   :
      MSG_LIB_TEXT_REQUEST_ACTION_UNCNOWN
     );
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_ACTION)+": "+CMessage::Text(code_descr);
  }
//+------------------------------------------------------------------+
//| Return the magic number value description                        |
//+------------------------------------------------------------------+
string RequestMagicDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_ORD_MAGIC)+": "+(string)request.magic;
  }
//+------------------------------------------------------------------+
//| Return the order ticket value description                        |
//+------------------------------------------------------------------+
string RequestOrderDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER)+": "+(request.order>0 ? (string)request.order : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the trading instrument name description                   |
//+------------------------------------------------------------------+
string RequestSymbolDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_SYMBOL)+": "+request.symbol;
  }
//+------------------------------------------------------------------+
//| Return the request volume description                            |
//+------------------------------------------------------------------+
string RequestVolumeDescription(const MqlTradeRequest &request)
  {
   int dg=(int)DigitsLots(request.symbol);
   int dgl=(dg==0 ? 1 : dg);
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_VOLUME)+": "+(request.volume>0 ? DoubleToString(request.volume,dgl) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request price value description                       |
//+------------------------------------------------------------------+
string RequestPriceDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+": "+(request.price>0 ? DoubleToString(request.price,(int)SymbolInfoInteger(request.symbol,SYMBOL_DIGITS)) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request StopLimit order price description             |
//+------------------------------------------------------------------+
string RequestStopLimitDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_STOPLIMIT)+": "+(request.stoplimit>0 ? DoubleToString(request.stoplimit,(int)SymbolInfoInteger(request.symbol,SYMBOL_DIGITS)) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request StopLoss order price description              |
//+------------------------------------------------------------------+
string RequestStopLossDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+": "+(request.sl>0 ? DoubleToString(request.sl,(int)SymbolInfoInteger(request.symbol,SYMBOL_DIGITS)) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request TakeProfit order price description            |
//+------------------------------------------------------------------+
string RequestTakeProfitDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+": "+(request.tp>0 ? DoubleToString(request.tp,(int)SymbolInfoInteger(request.symbol,SYMBOL_DIGITS)) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request deviation size description                    |
//+------------------------------------------------------------------+
string RequestDeviationDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_DEVIATION)+": "+(string)request.deviation;
  }
//+------------------------------------------------------------------+
//| Return the request order type description                        |
//+------------------------------------------------------------------+
string RequestTypeDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE)+": "+OrderTypeDescription(request.type);
  }
//+------------------------------------------------------------------+
//| Return the request order filling mode description                |
//+------------------------------------------------------------------+
string RequestTypeFillingDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE_FILLING)+": "+OrderTypeFillingDescription(request.type_filling);
  }
//+------------------------------------------------------------------+
//| Return the request order lifetime type description               |
//+------------------------------------------------------------------+
string RequestTypeTimeDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE_TIME)+": "+OrderTypeTimeDescription(request.type_time);
  }
//+------------------------------------------------------------------+
//| Return the request order expiration time description             |
//+------------------------------------------------------------------+
string RequestExpirationDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_EXPIRATION)+": "+(request.expiration>0 ? TimeToString(request.expiration) : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request order comment description                     |
//+------------------------------------------------------------------+
string RequestCommentDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_COMMENT)+": "+(request.comment!="" && request.comment!=NULL ? "\""+request.comment+"\"" : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request position ticket description                   |
//+------------------------------------------------------------------+
string RequestPositionDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_POSITION)+": "+(request.position>0 ? (string)request.position : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request opposite position ticket description          |
//+------------------------------------------------------------------+
string RequestPositionByDescription(const MqlTradeRequest &request)
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_POSITION_BY)+": "+(request.position_by>0 ? (string)request.position_by : CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
```

Now let's fix the detected trading class drawbacks and prepare for the creation of a pending request object for setting pending orders.

### Eliminating trading class shortcomings and creating a pending request for placing orders

I have noticed an interesting feature of some symbols whose charts are based on Last prices. Sometimes, they do not have Ask and Bid prices,
or one of them. To obtain the price anyway, I had to add additional methods (and change the existing ones) in the **CSymbol** abstract
symbol object class to check the chart construction type. If the chart is based on Last prices, the presence of Ask and Bid prices is checked.
If present, these prices are used, otherwise the Last price is applied.

In the block of methods for a simplified access to the symbol object properties in the **Symbol.mqh** file, change
the type of the method returning time. Since the time is returned in milliseconds, the type should be 'ulong' instead of 'datetime'.


Also, declare the three additional methods whose purpose has been
described above:

```
//+------------------------------------------------------------------+
//| Methods of a simplified access to the order object properties    |
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
   long              Time(void)                                   const { return (datetime)this.GetProperty(SYMBOL_PROP_TIME);                              }
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
   double            AskLast(void)                                const;
   double            AskLastHigh(void)                            const;
   double            AskLastLow(void)                             const;
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

We already have the similar three methods for the Bid price, although without the price presence check. Let's make changes to them:

```
//+------------------------------------------------------------------+
//| Return Bid or Last price                                         |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::BidLast(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetProperty(SYMBOL_PROP_BID)  :
     (this.GetProperty(SYMBOL_PROP_BID)==0    ? this.GetProperty(SYMBOL_PROP_LAST) : this.GetProperty(SYMBOL_PROP_BID))
     );
  }
//+------------------------------------------------------------------+
//| Return maximum Bid or Last price for a day                       |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::BidLastHigh(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID   ? this.GetProperty(SYMBOL_PROP_BIDHIGH)   :
     (this.GetProperty(SYMBOL_PROP_BIDHIGH)==0  ? this.GetProperty(SYMBOL_PROP_LASTHIGH)  : this.GetProperty(SYMBOL_PROP_BIDHIGH))
     );
  }
//+------------------------------------------------------------------+
//| Return minimum Bid or Last price for a day                       |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::BidLastLow(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID   ? this.GetProperty(SYMBOL_PROP_BIDLOW)  :
     (this.GetProperty(SYMBOL_PROP_BIDLOW)==0   ? this.GetProperty(SYMBOL_PROP_LASTLOW) : this.GetProperty(SYMBOL_PROP_BIDLOW))
     );
  }
//+------------------------------------------------------------------+
```

As mentioned above, here we check the chart construction type. If the chart is based on Bid prices, return the appropriate Bid prices. If the
chart is based on Last prices, additionally check for the zero Bid price of a returned symbol property. If it is equal to zero, use the
appropriate Last price, otherwise use the appropriate Bid price.

**Let's write the same implementation for the methods returning Ask prices:**

```
//+------------------------------------------------------------------+
//| Return Ask or Last price                                         |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::AskLast(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID ? this.GetProperty(SYMBOL_PROP_ASK)  :
     (this.GetProperty(SYMBOL_PROP_ASK)==0    ? this.GetProperty(SYMBOL_PROP_LAST) : this.GetProperty(SYMBOL_PROP_ASK))
     );
  }
//+------------------------------------------------------------------+
//| Return maximum Ask or Last price for a day                       |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::AskLastHigh(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID   ? this.GetProperty(SYMBOL_PROP_ASKHIGH)   :
     (this.GetProperty(SYMBOL_PROP_ASKHIGH)==0  ? this.GetProperty(SYMBOL_PROP_LASTHIGH)  : this.GetProperty(SYMBOL_PROP_ASKHIGH))
     );
  }
//+------------------------------------------------------------------+
//| Return minimum Ask or Last price for a day                       |
//| depending on the chart construction method and price availability|
//+------------------------------------------------------------------+
double CSymbol::AskLastLow(void) const
  {
   return
     (
      this.ChartMode()==SYMBOL_CHART_MODE_BID   ? this.GetProperty(SYMBOL_PROP_ASKLOW)  :
     (this.GetProperty(SYMBOL_PROP_ASKLOW)==0   ? this.GetProperty(SYMBOL_PROP_LASTLOW) : this.GetProperty(SYMBOL_PROP_ASKLOW))
     );
  }
//+------------------------------------------------------------------+
```

These are the methods to be used in the library when calculating price levels for obtaining Ask or Last, as well as Bid or Last prices.

When displaying messages about trading events in the journal, we now have additional notations for magic number IDs and two groups provided
that additional data is stored in the magic number value ( [implemented in the previous \\
article](https://www.mql5.com/en/articles/7394#node02)). But if the magic number also contains the pending request ID, it has not been displayed in the journal. Let's fix this. Add a
couple of strings to the appropriate brief event description method of each of the event classes in the **EventModify.mqh**, **EventOrderPlaced.mqh**,
**EventOrderRemoved.mqh**, **EventPositionClose.mqh** and **EventPositionOpen.mqh** files.

**Replace the**
**following string in each of the files:**

```
string magic=(this.Magic()!=0 ? ", "+CMessage::Text(MSG_ORD_MAGIC)+" "+(string)this.Magic()+magic_id+group_id1+group_id2 : "");
```

**with the following two strings:**

```
string pend_req_id=(this.GetPendReqID()>0 ? ", ID: "+(string)this.GetPendReqID() : "");
string magic=(this.Magic()!=0 ? ", "+CMessage::Text(MSG_ORD_MAGIC)+" "+(string)this.Magic()+magic_id+group_id1+group_id2+pend_req_id : "");
```

Thus, we have added the description of the pending event ID (if any) to the magic number description.

**Add the methods returning the descriptions of the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)**
**trading request structure fields to the public section of the symbol's CTradeObj trading object class in the TradeObj.mqh file:**

```
//--- Return the description of the (1) executed action type, (2) magic number, (3) order ticket, (4) volume,
//--- (5) open, (6) StopLimit order, (7) StopLoss, (8) TakeProfit price, (9) deviation,
//--- type of (10) order, (11) execution, (12) lifetime, (13) order expiration date,
//--- (14) comment, (15) position ticket, (16) opposite position ticket
   string                     GetRequestActionDescription(void)                  const { return RequestActionDescription(this.m_request);       }
   string                     GetRequestMagicDescription(void)                   const { return RequestMagicDescription(this.m_request);        }
   string                     GetRequestOrderDescription(void)                   const { return RequestOrderDescription(this.m_request);        }
   string                     GetRequestSymbolDescription(void)                  const { return RequestSymbolDescription(this.m_request);       }
   string                     GetRequestVolumeDescription(void)                  const { return RequestVolumeDescription(this.m_request);       }
   string                     GetRequestPriceDescription(void)                   const { return RequestPriceDescription(this.m_request);        }
   string                     GetRequestStopLimitDescription(void)               const { return RequestStopLimitDescription(this.m_request);    }
   string                     GetRequestStopLossDescription(void)                const { return RequestStopLossDescription(this.m_request);     }
   string                     GetRequestTakeProfitDescription(void)              const { return RequestTakeProfitDescription(this.m_request);   }
   string                     GetRequestDeviationDescription(void)               const { return RequestDeviationDescription(this.m_request);    }
   string                     GetRequestTypeDescription(void)                    const { return RequestTypeDescription(this.m_request);         }
   string                     GetRequestTypeFillingDescription(void)             const { return RequestTypeFillingDescription(this.m_request);  }
   string                     GetRequestTypeTimeDescription(void)                const { return RequestTypeTimeDescription(this.m_request);     }
   string                     GetRequestExpirationDescription(void)              const { return RequestExpirationDescription(this.m_request);   }
   string                     GetRequestCommentDescription(void)                 const { return RequestCommentDescription(this.m_request);      }
   string                     GetRequestPositionDescription(void)                const { return RequestPositionDescription(this.m_request);     }
   string                     GetRequestPositionByDescription(void)              const { return RequestPositionByDescription(this.m_request);   }

//--- Open a position
```

The methods simply call the appropriate functions we have previously created in the file of the library service functions.

Previously, I have overlooked passing the order filling type to the position opening methods.

Let's **add this**
**parameter to the declaration of the method of opening class positions:**

```
//--- Open a position
   bool                       OpenPosition(const ENUM_POSITION_TYPE type,
                                           const double volume,
                                           const double sl=0,
                                           const double tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const string comment=NULL,
                                           const ulong deviation=ULONG_MAX,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

**and to the method implementation:**

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
```

Since I am going to implement the ability to trade using pending requests in the future, I have introduced the concept of a pending request type.

In the **Trading.mqh** trading class file (namely, in the **CPendingReq** pending request class), **add**
**the class member variable to the private section to store the pending request type:**

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
```

In the public class section, **add the method returning the server return**
**code the pending request is based on, the methods returning the**
**descriptions of the pending request properties and the request type,**
**as well as the method returning all trading request data to the journal:**

```
public:
//--- Return (1) the request structure, (2) the price at the moment of the request generation,
//--- (3) request generation time, (4) current attempt time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
   MqlTradeRequest      MqlRequest(void)                       const { return this.m_request;         }
   double               PriceCreate(void)                      const { return this.m_price_create;    }
   ulong                TimeCreate(void)                       const { return this.m_time_create;     }
   ulong                TimeActivate(void)                     const { return this.m_time_activate;   }
   ulong                WaitingMSC(void)                       const { return this.m_waiting_msc;     }
   uchar                CurrentAttempt(void)                   const { return this.m_current_attempt; }
   uchar                TotalAttempts(void)                    const { return this.m_total_attempts;  }
   uchar                ID(void)                               const { return this.m_id;              }
   int                  Retcode(void)                          const { return this.m_retcode;         }
//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) request ID
   void                 SetPriceCreate(const double price)           { this.m_price_create=price;     }
   void                 SetTimeCreate(const ulong time)              { this.m_time_create=time;       }
   void                 SetTimeActivate(const ulong time)            { this.m_time_activate=time;     }
   void                 SetWaitingMSC(const ulong miliseconds)       { this.m_waiting_msc=miliseconds;}
   void                 SetCurrentAttempt(const uchar number)        { this.m_current_attempt=number; }
   void                 SetTotalAttempts(const uchar number)         { this.m_total_attempts=number;  }
   void                 SetID(const uchar id)                        { this.m_id=id;                  }

//--- Return the description of the (1) request structure, (2) the price at the moment of the request generation,
//--- (3) request generation time, (4) current attempt time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
   string               MqlRequestDescription(void)            const { return RequestActionDescription(this.m_request); }
   string               TypeDescription(void)                  const
                          {
                           return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TYPE) +
                             (this.Type()==PENDING_REQUEST_ID_TYPE_ERR           ?
                              CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_BY_ERROR) :
                              CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_BY_REQUEST)
                             );
                          }
   string               PriceCreateDescription(void)           const
                          {
                           return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_PRICE_CREATE)+": "+
                                  ::DoubleToString(this.PriceCreate(),(int)::SymbolInfoInteger(this.m_request.symbol,SYMBOL_DIGITS));
                          }
   string               TimeCreateDescription(void)            const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TIME_CREATE)+TimeMSCtoString(this.TimeCreate());    }
   string               TimeActivateDescription(void)          const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TIME_ACTIVATE)+TimeMSCtoString(this.TimeActivate());}
   string               WaitingMSCDescription(void)            const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_WAITING)+(string)this.WaitingMSC();                 }
   string               CurrentAttemptDescription(void)        const
                          {
                           return
                             (CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CURRENT_ATTEMPT)+
                              (this.CurrentAttempt()==0 ? CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_WAITING_ONSET) : (string)this.CurrentAttempt())
                             );
                          }
   string               TotalAttemptsDescription(void)         const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TOTAL_ATTEMPTS)+(string)this.TotalAttempts();       }
   string               IdDescription(void)                    const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ID)+(string)this.ID();                              }
   string               RetcodeDescription(void)               const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_RETCODE)+(string)this.Retcode();                    }
   string               ReasonDescription(void)                const { return CMessage::Text(this.m_retcode);  }

//--- Return a request type
   virtual int          Type(void)                             const { return this.m_type;   }
//--- Display request data in the journal
   void                 Print(void);
//--- Constructors
                        CPendingReq(void){;}
                        CPendingReq(const uchar id,const double price,const ulong time,const MqlTradeRequest &request,const int retcode);
  };
//+------------------------------------------------------------------+
```

The methods returning descriptions of the pending request object properties simply generate a composite description out of the header
describing the property and its value. The Type() method returning the pending request type is made virtual since [the \\
same virtual method has already been defined](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjecttype) for the CObject base object the pending request object is derived from. To implement the
return of the derived object type, the method should be redefined in the descendant. I have done exactly that, and now this method returns the **m\_type**
variable value defined in the pending request class.

**In the class constructor, set the value for a pending request type:**

```
//+------------------------------------------------------------------+
//| Parametric constructor                                           |
//+------------------------------------------------------------------+
CPendingReq::CPendingReq(const uchar id,const double price,const ulong time,const MqlTradeRequest &request,const int retcode) : m_price_create(price),
                                                                                                                                m_time_create(time),
                                                                                                                                m_id(id),
                                                                                                                                m_retcode(retcode)
  {
   this.CopyRequest(request);
   this.m_type=(retcode>0 ? PENDING_REQUEST_ID_TYPE_ERR : PENDING_REQUEST_ID_TYPE_REQ);
  }
//+------------------------------------------------------------------+
```

Since pending requests are created by a server return code and a program request, it is sufficient to know the server response code to define a
pending request type. This is exactly what we do here: if the return code exceeds
zero (the server has returned an error), this request has been
generated by the server return code. If the code is zero, a pending
request object has been created by a program request.

Improve the Compare() virtual method of the pending request object.

Previously, it always compared objects according
to a single property — request ID:

```
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

After introducing pending request types, we need to compare objects by two properties — ID and type.

To implement
this, change the method of comparing two objects:

```
//+------------------------------------------------------------------+
//| Compare CPendingReq objects by properties                        |
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

Compared properties from the **mode** variable are
selected here. If 0, the objects are compared by
IDs. If not 0, they are compared by object types.

Also beyond the class body, **write the method displaying the full description of all pending request object properties to the journal:**

```
//+------------------------------------------------------------------+
//| Display request data in the journal                              |
//+------------------------------------------------------------------+
void CPendingReq::Print(void)
  {
   string action=" - "+RequestActionDescription(this.m_request)+"\n";
   string symbol="",order="",volume="",price="",stoplimit="",sl="",tp="",deviation="",type="",type_filling="";
   string type_time="",expiration="",position="",position_by="",magic="",comment="",request_data="";
   string type_req=" - "+this.TypeDescription()+"\n";

   if(this.m_request.action==TRADE_ACTION_DEAL)
     {
      symbol=" - "+RequestSymbolDescription(this.m_request)+"\n";
      volume=" - "+RequestVolumeDescription(this.m_request)+"\n";
      price=" - "+RequestPriceDescription(this.m_request)+"\n";
      sl=" - "+RequestStopLossDescription(this.m_request)+"\n";
      tp=" - "+RequestTakeProfitDescription(this.m_request)+"\n";
      deviation=" - "+RequestDeviationDescription(this.m_request)+"\n";
      type=" - "+RequestTypeDescription(this.m_request)+"\n";
      type_filling=" - "+RequestTypeFillingDescription(this.m_request)+"\n";
      magic=" - "+RequestMagicDescription(this.m_request)+"\n";
      comment=" - "+RequestCommentDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+symbol+volume+price+sl+tp+deviation+type+type_filling+magic+comment+
         " ==================\n"
        );
     }
   else if(this.m_request.action==TRADE_ACTION_SLTP)
     {
      symbol=" - "+RequestSymbolDescription(this.m_request)+"\n";
      sl=" - "+RequestStopLossDescription(this.m_request)+"\n";
      tp=" - "+RequestTakeProfitDescription(this.m_request)+"\n";
      position=" - "+RequestPositionDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+symbol+sl+tp+position+
         " ==================\n"
        );
     }
   else if(this.m_request.action==TRADE_ACTION_PENDING)
     {
      symbol=" - "+RequestSymbolDescription(this.m_request)+"\n";
      volume=" - "+RequestVolumeDescription(this.m_request)+"\n";
      price=" - "+RequestPriceDescription(this.m_request)+"\n";
      stoplimit=" - "+RequestStopLimitDescription(this.m_request)+"\n";
      sl=" - "+RequestStopLossDescription(this.m_request)+"\n";
      tp=" - "+RequestTakeProfitDescription(this.m_request)+"\n";
      type=" - "+RequestTypeDescription(this.m_request)+"\n";
      type_filling=" - "+RequestTypeFillingDescription(this.m_request)+"\n";
      type_time=" - "+RequestTypeTimeDescription(this.m_request)+"\n";
      expiration=" - "+RequestExpirationDescription(this.m_request)+"\n";
      magic=" - "+RequestMagicDescription(this.m_request)+"\n";
      comment=" - "+RequestCommentDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+symbol+volume+price+stoplimit+sl+tp+type+type_filling+type_time+expiration+magic+comment+
         " ==================\n"
        );
     }
   else if(this.m_request.action==TRADE_ACTION_MODIFY)
     {
      order=" - "+RequestOrderDescription(this.m_request)+"\n";
      price=" - "+RequestPriceDescription(this.m_request)+"\n";
      sl=" - "+RequestStopLossDescription(this.m_request)+"\n";
      tp=" - "+RequestTakeProfitDescription(this.m_request)+"\n";
      type_time=" - "+RequestTypeTimeDescription(this.m_request)+"\n";
      expiration=" - "+RequestExpirationDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+order+price+sl+tp+type_time+expiration+
         " ==================\n"
        );
     }
   else if(this.m_request.action==TRADE_ACTION_REMOVE)
     {
      order=" - "+RequestOrderDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+order+
         " ==================\n"
        );
     }
   else if(this.m_request.action==TRADE_ACTION_CLOSE_BY)
     {
      position=" - "+RequestPositionDescription(this.m_request)+"\n";
      position_by=" - "+RequestPositionByDescription(this.m_request)+"\n";
      magic=" - "+RequestMagicDescription(this.m_request)+"\n";
      request_data=
        ("================== "+
         CMessage::Text(MSG_LIB_TEXT_REQUEST_DATAS)+" ==================\n"+
         action+position+position_by+magic+
         " ==================\n"
        );
     }
   string datas=
     (
      " - "+this.TypeDescription()+"\n"+
      " - "+this.IdDescription()+"\n"+
      " - "+this.RetcodeDescription()+" \""+this.ReasonDescription()+"\"\n"+
      " - "+this.TimeCreateDescription()+"\n"+
      " - "+this.PriceCreateDescription()+"\n"+
      " - "+this.TimeActivateDescription()+"\n"+
      " - "+this.WaitingMSCDescription()+" ("+TimeToString(this.WaitingMSC()/1000,TIME_MINUTES|TIME_SECONDS)+")"+"\n"+
      " - "+this.CurrentAttemptDescription()+"\n"+
      " - "+this.TotalAttemptsDescription()+"\n"
     );
   ::Print("================== ",CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_DATAS)," ==================\n",datas,request_data);
  }
//+------------------------------------------------------------------+
```

In the method, descriptions of all object properties are gathered in string variables including field descriptions for the object's
trading request structure. The number of displayed data depends on the type of the action in the trading request since different actions
require different number of trading request structure fields. Therefore, the 'action' field value is checked and only the appropriate
fields are displayed. The description of the object variables is displayed first followed by the description of the request structure
fields. Thus, all pending request object properties are displayed in the journal according to the request trading action type (action).

Previously, we added the additional property — order filling type — to the position opening method in the **CTradeObj**
class.

Now, **let's add the same property to the private position opening method definition in the CTrading class:**

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
                                    const ulong deviation=ULONG_MAX,
                                    const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

**Add the same properties to the definition of public Buy and Sell positions opening methods:**

```
//--- Open (1) Buy, (2) Sell position
   template<typename SL,typename TP>
   bool                 OpenBuy(const double volume,
                                const string symbol,
                                const ulong magic=ULONG_MAX,
                                const SL sl=0,
                                const TP tp=0,
                                const string comment=NULL,
                                const ulong deviation=ULONG_MAX,
                                const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

   template<typename SL,typename TP>
   bool                 OpenSell(const double volume,
                                 const string symbol,
                                 const ulong magic=ULONG_MAX,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const string comment=NULL,
                                 const ulong deviation=ULONG_MAX,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

Also, let's add the same parameters when implementing these methods outside the class body:

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
                       const ulong deviation=ULONG_MAX,
                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request from the OpenPosition() method
   return this.OpenPosition(POSITION_TYPE_BUY,volume,symbol,magic,sl,tp,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+

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
                        const ulong deviation=ULONG_MAX,
                        const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- Return the result of sending a trading request from the OpenPosition() method
   return this.OpenPosition(POSITION_TYPE_SELL,volume,symbol,magic,sl,tp,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+
```

**In the private section of the class, declare the method returning the**
**request object index in the list by ID:**

```
//--- Look for the first free pending request ID
   int                  GetFreeID(void);
//--- Return the request object index in the list by ID
   int                  GetIndexPendingRequestByID(const uchar id);

public:
```

I have slightly changed the implementation of the class timer.

**Below is the full timer implementation code with its logic**
**described in the comments:**

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
           {
            ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_DELETED));
            req_obj.Print();
           }
         this.m_list_request.Delete(i);
         continue;
        }

      //--- Get the pending request ID
      uchar id=this.GetPendReqID((uint)request.magic);
      //--- Get the list of orders/positions containing the order/position with the pending request ID
      CArrayObj *list=this.m_market.GetList(ORDER_PROP_PEND_REQ_ID,id,EQUAL);
      if(::CheckPointer(list)==POINTER_INVALID)
         continue;
      //--- If the order/position is present, the request is handled: remove it and proceed to the next
      if(list.Total()>0)
        {
         this.m_list_request.Delete(i);
         continue;
        }

      //--- Set the request activation time in the request object
      req_obj.SetTimeActivate(req_obj.TimeCreate()+req_obj.WaitingMSC()*(req_obj.CurrentAttempt()+1));

      //--- If the current time is less than the request activation time,
      //--- this is not the request time - move on to the next request in the list
      if((long)symbol_obj.Time()<(long)req_obj.TimeActivate())
         continue;

      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

      //--- Display the number of the trading attempt in the journal
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_TEXT_RE_TRY_N)+(string)req_obj.CurrentAttempt());

      //--- Depending on the type of action performed in the trading request
      switch(request.action)
        {
         //--- Open a position
         case TRADE_ACTION_DEAL :
            this.OpenPosition((ENUM_POSITION_TYPE)request.type,request.volume,request.symbol,request.magic,request.sl,request.tp,request.comment,request.deviation,request.type_filling);
            break;
         //--- Place a pending order
         case TRADE_ACTION_PENDING :
            this.PlaceOrder(request.type,request.volume,request.symbol,request.price,request.stoplimit,request.sl,request.tp,request.magic,request.comment,request.expiration,request.type_time,request.type_filling);
            break;
         //---
         default:
            break;
        }
     }
  }
//+------------------------------------------------------------------+
```

I hope, everything is clear here.

**In the RequestErrorsCorrecting() error correction method, add**
**expiration type correctionwhen receiving the "invalid order expiration**
**date" error (only a part of the code with corrections):**

```
//--- The specified type of order execution by residue is not supported
   if(this.IsPresentErorCode(10030))
      request.type_filling=symbol_obj.GetCorrectTypeFilling();
//--- Invalid order expiration in a request -
   if(this.IsPresentErorCode(10022))
     {
      //--- Set a correct order expiration type
      request.type_time=symbol_obj.GetCorrectTypeExpiration();
      //--- if the expiration type is not supported as set by the expiration date and the expiration data is defined, reset the expiration date
      if(!symbol_obj.IsExpirationModeSpecified() && request.expiration>0)
         request.expiration=0;
     }
//--- View the list of remaining errors and correct trading request parameters
```

Previously, we added the new methods for obtaining Ask prices to the symbol object and adjusted the methods for obtaining Bid prices. Now we need to
replace all occurrences of the "Ask()" and "Bid()" strings in the entire listing with "AskLast()" and "BidLast()", respectively. The most
convenient way to do this is applying the search and replace function (Ctrl+H) to the entire code. Thus, we are going to use the auto selection
of suitable prices wherever we need symbol object's Ask and Bid prices.

For example, **the method of setting the trading request price now looks as follows with**
**the replaced prices:**

```
//+------------------------------------------------------------------+
//| Set trading request prices                                       |
//+------------------------------------------------------------------+
template <typename PS,typename SL,typename TP,typename PL>
bool CTrading::SetPrices(const ENUM_ORDER_TYPE action,const PS price,const SL sl,const TP tp,const PL limit,const string source_method,CSymbol *symbol_obj)
  {
//--- Reset prices
   ::ZeroMemory(this.m_request);
//--- Update all data by symbol
   if(!symbol_obj.RefreshRates())
     {
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      return false;
     }
//--- Open/close price
   if(price>0)
     {
      //--- price parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(price)=="double")
         this.m_request.price=::NormalizeDouble(price,symbol_obj.Digits());
      //--- price parameter type (int) - the distance has been passed
      else if(typename(price)=="int" || typename(price)=="uint" || typename(price)=="long" || typename(price)=="ulong")
        {
         //--- Calculate the order price
         switch((int)action)
           {
            //--- Pending order
            case ORDER_TYPE_BUY_LIMIT       :   this.m_request.price=::NormalizeDouble(symbol_obj.AskLast()-price*symbol_obj.Point(),symbol_obj.Digits()); break;
            case ORDER_TYPE_BUY_STOP        :
            case ORDER_TYPE_BUY_STOP_LIMIT  :   this.m_request.price=::NormalizeDouble(symbol_obj.AskLast()+price*symbol_obj.Point(),symbol_obj.Digits()); break;

            case ORDER_TYPE_SELL_LIMIT      :   this.m_request.price=::NormalizeDouble(symbol_obj.BidLast()+price*symbol_obj.Point(),symbol_obj.Digits()); break;
            case ORDER_TYPE_SELL_STOP       :
            case ORDER_TYPE_SELL_STOP_LIMIT :   this.m_request.price=::NormalizeDouble(symbol_obj.BidLast()-price*symbol_obj.Point(),symbol_obj.Digits()); break;
            //--- Default - current position open prices
            default  :  this.m_request.price=
              (
               this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY ? ::NormalizeDouble(symbol_obj.AskLast(),symbol_obj.Digits()) :
               ::NormalizeDouble(symbol_obj.BidLast(),symbol_obj.Digits())
              ); break;
           }
        }
      //--- unsupported price types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_PR_TYPE));
         return false;
        }
     }
   //--- If no price is specified, use the current prices
   else
     {
      this.m_request.price=
        (
         this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY ?
         ::NormalizeDouble(symbol_obj.AskLast(),symbol_obj.Digits())          :
         ::NormalizeDouble(symbol_obj.BidLast(),symbol_obj.Digits())
        );
     }

//--- StopLimit order price or distance
   if(limit>0)
     {
      //--- limit order price parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(limit)=="double")
         this.m_request.stoplimit=::NormalizeDouble(limit,symbol_obj.Digits());
      //--- limit order price parameter type (int) - the distance has been passed
      else if(typename(limit)=="int" || typename(limit)=="uint" || typename(limit)=="long" || typename(limit)=="ulong")
        {
         //--- Calculate a limit order price
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_request.stoplimit=::NormalizeDouble(this.m_request.price-limit*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_request.stoplimit=::NormalizeDouble(this.m_request.price+limit*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported limit order price types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_PL_TYPE));
         return false;
        }
     }

//--- Order price stop order prices are calculated from
   double price_open=
     (
      (action==ORDER_TYPE_BUY_STOP_LIMIT || action==ORDER_TYPE_SELL_STOP_LIMIT) && limit>0 ? this.m_request.stoplimit : this.m_request.price
     );

//--- StopLoss
   if(sl>0)
     {
      //--- StopLoss parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(sl)=="double")
         this.m_request.sl=::NormalizeDouble(sl,symbol_obj.Digits());
      //--- StopLoss parameter type (int) - calculate the placement distance
      else if(typename(sl)=="int" || typename(sl)=="uint" || typename(sl)=="long" || typename(sl)=="ulong")
        {
         //--- Calculate the StopLoss price
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_request.sl=::NormalizeDouble(price_open-sl*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_request.sl=::NormalizeDouble(price_open+sl*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported StopLoss types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_SL_TYPE));
         return false;
        }
     }

//--- TakeProfit
   if(tp>0)
     {
      //--- TakeProfit parameter type (double) - normalize the price up to Digits(), since the price has been passed
      if(typename(tp)=="double")
         this.m_request.tp=::NormalizeDouble(tp,symbol_obj.Digits());
      //--- TakeProfit parameter type (int) - calculate the placement distance
      else if(typename(tp)=="int" || typename(tp)=="uint" || typename(tp)=="long" || typename(tp)=="ulong")
        {
         if(this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY)
            this.m_request.tp=::NormalizeDouble(price_open+tp*symbol_obj.Point(),symbol_obj.Digits());
         else
            this.m_request.tp=::NormalizeDouble(price_open-tp*symbol_obj.Point(),symbol_obj.Digits());
        }
      //--- unsupported TakeProfit types - display the message and return 'false'
      else
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_UNSUPPORTED_TP_TYPE));
         return false;
        }
     }

//--- All prices are recorded
   return true;
  }
//+------------------------------------------------------------------+
```

There is no point in displaying all codes with the performed replacements here. They are all fixed already and attached below.

**In the implementation of the private method for placing a pending order, add the**
**block for creating a pending trading request in case of a server error:**

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
         //--- If "Wait and repeat" is received as a result of sending a request
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
               //--- Write the request ID to the magic number, while a symbol name is set in the request structure,
               //--- set the trading operation and order types, comment and position type, filling and expiration type
               uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
               this.SetPendReqID((uchar)id,mn);
               this.m_request.magic=mn;
               this.m_request.symbol=symbol_obj.Name();
               this.m_request.action=TRADE_ACTION_PENDING;
               this.m_request.type=order_type;
               this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
               this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : trade_obj.GetTypeExpiration());
               this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : trade_obj.GetTypeFilling());
               //--- Pass the number of trading attempts to the pending request
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

I believe, all actions related to the development of the pending request are described in the code comments and are easy to understand. In
any case, you are welcome to use the comments section.

To simplify debugging (namely to be able to see the pending request generation result), **in the method of creating the**
**pending request object, add displaying**
**the properties of a newly created request in the journal:**

```
//+------------------------------------------------------------------+
//| Create a pending request                                         |
//+------------------------------------------------------------------+
bool CTrading::CreatePendingRequest(const uchar id,const uchar attempts,const ulong wait,const MqlTradeRequest &request,const int retcode,CSymbol *symbol_obj)
  {
   //--- Create a new pending request object
   CPendingReq *req_obj=new CPendingReq(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);
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

   if(this.m_log_level>LOG_LEVEL_NO_MSG)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CREATED)," #",req_obj.ID(),":");
      req_obj.Print();
     }

   return true;
  }
//+------------------------------------------------------------------+
```

At the very end of the trading class listing, **implement the method returning the request object index in the list by ID:**

```
//+------------------------------------------------------------------+
//| Return the request object index in the list by ID                |
//+------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByID(const uchar id)
  {
   CPendingReq *req=new CPendingReq();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetID(id);
   this.m_list_request.Sort();
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+------------------------------------------------------------------+
```

The method receives the necessary ID, a
temporary request object is created and the ID passed to the method is set
for it.

Next, the sorted list flag is set for the list containing request objects.
By default, the sorting mode is equal to zero. This is the mode used to arrange the objects comparison by ID in the Compare() virtual method of
the **CPendingReq** class. Therefore, it is now possible to use the [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch)
object searching method in the [dynamic array \\
of pointers to objects](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj). The method returns the obtained object index in the list or -1 if the object is not found. Before exiting the
method, remove the temporary request object and
return the obtained index of the detected object or -1.

Now all we have to do is **supplement the library's CEngine base object class with an additional parameter**
**specifying the order filling type. The parameter is added to the definitions of the class methods for sending trading requests.**

```
//--- Open (1) Buy, (2) Sell position
   template<typename SL,typename TP>
   bool                 OpenBuy(const double volume,
                                 const string symbol,
                                 const ulong magic=ULONG_MAX,
                                 SL sl=0,
                                 TP tp=0,
                                 const string comment=NULL,
                                 const ulong deviation=ULONG_MAX,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename SL,typename TP>
   bool                 OpenSell(const double volume,
                                 const string symbol,
                                 const ulong magic=ULONG_MAX,
                                 SL sl=0,
                                 TP tp=0,
                                 const string comment=NULL,
                                 const ulong deviation=ULONG_MAX,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

//--- Set (1) BuyStop, (2) BuyLimit, (3) BuyStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyStop(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceBuyLimit(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceBuyStopLimit(const double volume,
                                     const string symbol,
                                     const PR price_stop,
                                     const PL price_limit,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
//--- Set (1) SellStop, (2) SellLimit, (3) SellStopLimit pending order
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellStop(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PR,typename SL,typename TP>
   bool                 PlaceSellLimit(const double volume,
                                     const string symbol,
                                     const PR price,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PR,typename PL,typename SL,typename TP>
   bool                 PlaceSellStopLimit(const double volume,
                                     const string symbol,
                                     const PR price_stop,
                                     const PL price_limit,
                                     const SL sl=0,
                                     const TP tp=0,
                                     const ulong magic=ULONG_MAX,
                                     const string comment=NULL,
                                     const datetime expiration=0,
                                     const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                     const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
```

If the default value is -1, correct values of order filling types are taken from symbol trading objects a trading operation is to be performed
on.

**Add the same parameters to the implementation codes of these trading methods:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CEngine::OpenBuy(const double volume,
                      const string symbol,
                      const ulong magic=ULONG_MAX,
                      SL sl=0,TP tp=0,
                      const string comment=NULL,
                      const ulong deviation=ULONG_MAX,
                      const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.OpenBuy(volume,symbol,magic,sl,tp,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
bool CEngine::OpenSell(const double volume,
                       const string symbol,
                       const ulong magic=ULONG_MAX,
                       SL sl=0,TP tp=0,
                       const string comment=NULL,
                       const ulong deviation=ULONG_MAX,
                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.OpenSell(volume,symbol,magic,sl,tp,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Place BuyStop pending order                                      |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceBuyStop(const double volume,
                           const string symbol,
                           const PR price,
                           const SL sl=0,
                           const TP tp=0,
                           const ulong magic=WRONG_VALUE,
                           const string comment=NULL,
                           const datetime expiration=0,
                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceBuyStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place BuyLimit pending order                                     |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceBuyLimit(const double volume,
                            const string symbol,
                            const PR price,
                            const SL sl=0,
                            const TP tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceBuyLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place BuyStopLimit pending order                                 |
//+------------------------------------------------------------------+
template<typename PR,typename PL,typename SL,typename TP>
bool CEngine::PlaceBuyStopLimit(const double volume,
                                const string symbol,
                                const PR price_stop,
                                const PL price_limit,
                                const SL sl=0,
                                const TP tp=0,
                                const ulong magic=WRONG_VALUE,
                                const string comment=NULL,
                                const datetime expiration=0,
                                const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceBuyStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place SellStop pending order                                     |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceSellStop(const double volume,
                            const string symbol,
                            const PR price,
                            const SL sl=0,
                            const TP tp=0,
                            const ulong magic=WRONG_VALUE,
                            const string comment=NULL,
                            const datetime expiration=0,
                            const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceSellStop(volume,symbol,price,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place SellLimit pending order                                    |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP>
bool CEngine::PlaceSellLimit(const double volume,
                             const string symbol,
                             const PR price,
                             const SL sl=0,
                             const TP tp=0,
                             const ulong magic=WRONG_VALUE,
                             const string comment=NULL,
                             const datetime expiration=0,
                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceSellLimit(volume,symbol,price,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Place SellStopLimit pending order                                |
//+------------------------------------------------------------------+
template<typename PR,typename PL,typename SL,typename TP>
bool CEngine::PlaceSellStopLimit(const double volume,
                                 const string symbol,
                                 const PR price_stop,
                                 const PL price_limit,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const ulong magic=WRONG_VALUE,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.PlaceSellStopLimit(volume,symbol,price_stop,price_limit,sl,tp,magic,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
```

**These are all the necessary adjustments and changes for now.**

### Testing

To test pending requests for placing pending orders, let's take the [EA \\
from the previous article](https://www.mql5.com/en/articles/7394#node04) and save it to \\MQL5\\Experts\\TestDoEasy\ **Part27\** under the name **TestDoEasyPart27.mq5**.

**In the EA's library initialization function, add setting correct**
**values of order filling and expiration types for all trading objects of all symbols used in the EA:**

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
//--- Set correct order expiration and filling types to all trading objects
   engine.TradingSetCorrectTypeExpiration();
   engine.TradingSetCorrectTypeFilling();

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

**Oddly enough, these are all the changes in the EA. All other changes have been implemented to the library codes.**

To test pending requests for placing pending orders, we will do exactly the same thing as the last time — disable the Internet, try to place a
pending order, get the trade server connection error and get the journal message informing of a pending request generation and its
parameters. Then enable the Internet back and get activation of a pending request, as well as placing the requested pending order.

Let's check.

Compile and launch the EA. Turn off the Internet and wait till the following image appears in the lower right corner of the terminal:

![](https://c.mql5.com/2/37/WkjYQlsw9Y__1.png)

![](https://c.mql5.com/2/37/wTmk55f43l.gif)

After disabling the Internet and clicking Sell, the trade server returns the error and the following error and pending request entries are
displayed in the journal.

Then enable the Internet to restore the connection to the trade server:

![](https://c.mql5.com/2/37/terminal64_SwjtIhMXo0__1.png)

As soon as the connection is restored, the library starts handling a pending request sending it to the server.


As a result, the pending order is placed and the following journal entries appear:

```
2019.12.05 16:38:32.591 CTrading::PlaceOrder<uint,int,uint,uint>: Invalid request:
2019.12.05 16:38:32.591 No connection with the trade server
2019.12.05 16:38:32.591 Correction of trade request parameters ...
2019.12.05 16:38:32.610 Trading attempt #1. Error: No connection with the trade server
2019.12.05 16:38:32.610 Pending request created #1:
2019.12.05 16:38:32.610 ================== Pending trade request's parameters ==================
2019.12.05 16:38:32.610  - Pending request type: Pending request that was created as a result of the server code
2019.12.05 16:38:32.610  - Trade request ID: 1
2019.12.05 16:38:32.610  - Return code based on which the request was created: 10031 "No connection with the trade server"
2019.12.05 16:38:32.610  - Request creation time: 2019.12.05 11:37:39.054
2019.12.05 16:38:32.610  - Price at time of request create: : 1.10913
2019.12.05 16:38:32.610  - Request activation time: 2019.12.05 11:37:59.054
2019.12.05 16:38:32.610  - Waiting time between trading attempts: 20000 (00:00:20)
2019.12.05 16:38:32.610  - Current trading attempt: Waiting for the onset time of the first trading attempt
2019.12.05 16:38:32.610  - Total trade attempts: 5
2019.12.05 16:38:32.610 ================== Trade request's parameters ==================
2019.12.05 16:38:32.610  - Trade operation type: Place pending order
2019.12.05 16:38:32.610  - Trade symbol: EURUSD
2019.12.05 16:38:32.610  - Requested volume for a deal in lots: 0.10
2019.12.05 16:38:32.610  - Price: 1.10963
2019.12.05 16:38:32.610  - StopLimit level of the order: Value not set
2019.12.05 16:38:32.610  - Stop Loss level of the order: 1.11113
2019.12.05 16:38:32.610  - Take Profit level of the order: 1.10813
2019.12.05 16:38:32.610  - Order type: Pending order Sell Limit
2019.12.05 16:38:32.610  - Order execution type: The order is executed exclusively in the specified volume, otherwise it is canceled (FOK)
2019.12.05 16:38:32.610  - Order expiration type: Good till cancel order
2019.12.05 16:38:32.610  - Order expiration time: Value not set
2019.12.05 16:38:32.610  - Magic number: 24379515
2019.12.05 16:38:32.610  - Order comment: "Pending order SellLimit"
2019.12.05 16:38:32.610  ==================
2019.12.05 16:38:32.610
2019.12.05 16:38:45.185 Retry trading attempt #1
2019.12.05 16:38:45.185 CTrading::PlaceOrder<double,double,double,double>: Invalid request:
2019.12.05 16:38:45.185 Trading is prohibited for the current account
2019.12.05 16:38:45.185 Correction of trade request parameters ...
2019.12.05 16:38:45.185 Trading operation aborted
2019.12.05 16:38:45.512 Retry trading attempt #2
2019.12.05 16:38:45.512 CTrading::PlaceOrder<double,double,double,double>: Invalid request:
2019.12.05 16:38:45.512 Trading is prohibited for the current account
2019.12.05 16:38:45.512 Correction of trade request parameters ...
2019.12.05 16:38:45.512 Trading operation aborted
2019.12.05 16:38:45.852 Retry trading attempt #3
2019.12.05 16:38:46.405 - Pending order placed: 2019.12.05 11:38:45.933 -
2019.12.05 16:38:46.405 EURUSD Placed 0.10 Pending order Sell Limit #491179168 at price 1.10963, sl 1.11113, tp 1.10813, Magic number 24379515 (123), G1: 4, G2: 7, ID: 1
2019.12.05 16:38:46.472 OnDoEasyEvent: Pending order placed
```

First, we obtain the "No connection with trade server" error.

Next, obtain the message about creation of a pending trading request with ID #1
containing all the object parameters and request
structure parameters within that object.

After that, two
repeated trading attempts #1 and #2 are made. The attempts are sent from a pending trading request and are followed by the error of disabled
trading on the account (trading on the account is not enabled yet after connection is restored).

The
third attempt sent from the pending trading request object has turned out to be successful and the pending order has been placed.

In the description of the pending order's magic number, we have the magic number 24379515 followed by the ID of the magic number set in the EA
parameters (123), the first group ID "G1: 4", the second group ID "G2: 7" and the pending request ID "ID: 1"

**Please note:**

Do not use
the results of the trading class with the pending requests, described in the article, and the attached test EA in real trading!

The
article, its accompanying materials and results are meant only as a test of the pending requests concept. In its current state, it is not
a finished product and is in no way intended for real trading. Instead, it is only meant for
the demo mode or the tester.

### What's next?

In the next article, we will continue working on basic pending request features — modifying, deleting and closing orders/positions.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7418#node00)

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
26\. Working with pending trading requests - first implementation](https://www.mql5.com/en/articles/7394)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7418](https://www.mql5.com/ru/articles/7418)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7418.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7418/mql5.zip "Download MQL5.zip")(3617 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7418/mql4.zip "Download MQL4.zip")(3617.01 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/332917)**
(47)


![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
9 Jan 2021 at 00:39

```
//+------------------------------------------------------------------+
//| Compare CPendingReq objects by properties |
//+------------------------------------------------------------------+
int CPendingReq::Compare(const CObject *node,const int mode=0) const
  {
   const CPendingReq *compared_req=node;
   return
     (
      //--- Compare by ID
      mode==0  ?
      (this.ID()> compared_req.ID() ? 1 : this.ID()< compared_req.ID() ? -1 : 0)   :
      //--- Compare by type
      (this.Type()> compared_req.Type() ? 1 : this.Type()< compared_req.Type() ? -1 : 0)
     );
  }
//+------------------------------------------------------------------+
```

Type or Id cannot be equal?

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
9 Jan 2021 at 06:31

**BillionerClub:**

Type or Id cannot be equal?

I don't understand the meaning of the question. Please clarify it.

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
9 Jan 2021 at 10:27

**Artyom Trishkin:**

I don't understand the meaning of the question. Please clarify.

When comparing there is always a situation when equality is possible

```
this.Type()>=compared_req.Type()
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
9 Jan 2021 at 12:17

**BillionerClub:**

When you compare, there's always a situation where equality is possible.

Here three conditions are checked:

```
this.Type()>compared_req.Type() ? 1 : this.Type()<compared_req.Type() ? -1 : 0
```

Greater than (1), less than (-1), rest (0)

![BillionerClub](https://c.mql5.com/avatar/avatar_na2.png)

**[BillionerClub](https://www.mql5.com/en/users/billionerclub)**
\|
9 Jan 2021 at 13:41

**Artyom Trishkin:**

Three conditions are tested here:

Greater than (1), less than (-1), rest (0)

man, I was a fool and didn't see the meaning of the code)))

![Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__1.png)[Continuous Walk-Forward Optimization (Part 2): Mechanism for creating an optimization report for any robot](https://www.mql5.com/en/articles/7452)

The first article within the Walk-Through Optimization series described the creation of a DLL to be used in our auto optimizer. This continuation is entirely devoted to the MQL5 language.

![Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://c.mql5.com/2/37/MQL5-avatar-doeasy__14.png)[Library for easy and quick development of MetaTrader programs (part XXVI): Working with pending trading requests - first implementation (opening positions)](https://www.mql5.com/en/articles/7394)

In this article, we are going to store some data in the value of the orders and positions magic number and start the implementation of pending requests. To check the concept, let's create the first test pending request for opening market positions when receiving a server error requiring waiting and sending a repeated request.

![SQLite: Native handling of SQL databases in MQL5](https://c.mql5.com/2/37/database-mql5.png)[SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463)

The development of trading strategies is associated with handling large amounts of data. Now, you are able to work with databases using SQL queries based on SQLite directly in MQL5. An important feature of this engine is that the entire database is placed in a single file located on a user's PC.

![Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://c.mql5.com/2/37/MQL5-avatar-season_research.png)[Exploring Seasonal Patterns of Financial Time Series with Boxplot](https://www.mql5.com/en/articles/7038)

In this article we will view seasonal characteristics of financial time series using Boxplot diagrams. Each separate boxplot (or box-and-whiskey diagram) provides a good visualization of how values are distributed along the dataset. Boxplots should not be confused with the candlestick charts, although they can be visually similar.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/7418&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070441199468221922)

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