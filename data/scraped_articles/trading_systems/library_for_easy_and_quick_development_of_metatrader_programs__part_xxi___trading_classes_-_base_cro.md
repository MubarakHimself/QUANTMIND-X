---
title: Library for easy and quick development of MetaTrader programs (part XXI): Trading classes - Base cross-platform trading object
url: https://www.mql5.com/en/articles/7229
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:38:54.840292
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kprvgnlwhlhenczklrxxefintwxrsemq&ssn=1769186332690644718&ssn_dr=0&ssn_sr=0&fv_date=1769186332&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7229&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXI)%3A%20Trading%20classes%20-%20Base%20cross-platform%20trading%20object%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918633229449039&fz_uniq=5070453268326323739&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7229#node01)
- [Creating the base trading object](https://www.mql5.com/en/articles/7229#node02)
- [Testing the base trading object](https://www.mql5.com/en/articles/7229#node03)
- [What's next?](https://www.mql5.com/en/articles/7229#node04)


This article starts a new broad topic — trading classes.

### Concept

It is nice to have an easy access to various data at any time. However, that data is useless if we cannot apply it in trading. This means we need
trading functionality along with the already existing one.

This section will be relatively large, and we will do everything step by step.

- We should be able to send any trading requests from any platform, be it MetaTrader 5 or MetaTrader 4, without even thinking about
differences between them. Everything should be unified.
- First, we need to verify trading requests in order not to load the server with deliberately erroneous requests.

- We need to consider and correctly handle the return codes of the trade server. What does an EA do while sending a request to the server? It
maintains the 'request-response' dialog with the server. Our task is to correctly arrange such a "communication channel", i.e.
create the methods of handling trade server responses.
- We need to create several options of handling server responses since sometimes we need to open a position "preferably at any cost". To
do this, we need to arrange a repeated sending of a request to the server in case of a refusal to place an order — we can either adjust the
trading request parameters or re-send it, or leave all the parameters intact but wait for the right moment when the request with these
parameters is passed to send it immediately. Besides, we need to consider the price level in order not to re-send an order at a knowingly
worse price.



Sometimes, we need to send a trading request and continue work regardless of a request result.

- Besides, we need to arrange the work with trading classes so that to avoid issues when placing a library-based program to [MQL5 \\
Market](https://www.mql5.com/en/market). The program should pass [all the checks](https://www.mql5.com/en/articles/2555) smoothly.


This is my current plan regarding trading classes.

In this article, we will consider the development of the base trading
object. This is a class sending a trading request to the server in the same manner regardless of the platform. When sending a request to the
server, such a trading object implies that verified and correct trading request parameters are passed to it. The object features no
verification of parameters. Instead, they will be verified in the base trading class to be developed later.

It should be noted that selecting an order or a position by ticket is to be implemented in the current trading object for now. After
creating the base trading class, the feature will be re-located to it.

Since the entire trading is directly tied to a symbol, the base trading object will be a part of a symbol object we considered [in \\
the article 14](https://www.mql5.com/en/articles/7014). The access to symbol trading objects will be arranged later — in the base trading class. In this article, we will arrange a
temporary access to symbol trading objects from the

**CEngine** library base class we considered [in the article 3](https://www.mql5.com/en/articles/5687#node02).
This is the class where all the environment data is accumulated. It features all account and symbol properties necessary to work with
trading classes.

### Creating a base trading object

To log the work of trading classes, we need to create an enumeration of logging levels in the **Defines.mqh**
library file.

**Add the necessary enumeration at the very end of the**
**listing:**

```
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
```

To display messages in the journal, we need message texts and their indices in the list of library messages.

**Add**
**the necessary indices to the Datas.mqh file:**

```
   MSG_LIB_SYS_NOT_SYMBOL_ON_SERVER,                  // Error. No such symbol on server
   MSG_LIB_SYS_NOT_SYMBOL_ON_LIST,                    // Error. No such symbol in the list of used symbols:
   MSG_LIB_SYS_FAILED_PUT_SYMBOL,                     // Failed to place to market watch. Error:
   MSG_LIB_SYS_ERROR_NOT_POSITION,                    // Error. Not a position:
   MSG_LIB_SYS_ERROR_NO_OPEN_POSITION_WITH_TICKET,    // Error. No open position with ticket #
   MSG_LIB_SYS_ERROR_NO_PLACED_ORDER_WITH_TICKET,     // Error. No placed order with ticket #
   MSG_LIB_SYS_ERROR_FAILED_CLOSE_POS,                // Failed to closed position. Error
   MSG_LIB_SYS_ERROR_FAILED_MODIFY_ORD,               // Failed to modify order. Error
   MSG_LIB_SYS_ERROR_UNABLE_PLACE_WITHOUT_TIME_SPEC,  // Error: Cannot place order without explicitly specified expiration time
   MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ,            // Error. Failed to get trading object
   MSG_LIB_SYS_ERROR_FAILED_GET_POS_OBJ,              // Error. Failed to get position object
   MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ,              // Error. Failed to get order object
   MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ,              // Error. Failed to get symbol object
   MSG_LIB_SYS_ERROR_CODE_OUT_OF_RANGE,               // Return code out of range of error codes
   MSG_LIB_TEXT_FAILED_ADD_TO_LIST,                   // failed to add to list
   MSG_LIB_TEXT_TIME_UNTIL_THE_END_DAY,               // Order lifetime till the end of the current day to be used
   MSG_LIB_TEXT_SUNDAY,                               // Sunday
   MSG_ACC_MARGIN_MODE_RETAIL_EXCHANGE,               // Exchange markets mode
   MSG_ACC_UNABLE_CLOSE_BY,                           // Close by is available only on hedging accounts
   MSG_ACC_SAME_TYPE_CLOSE_BY,                        // Error. Positions for close by are of the same type

//--- CEngine
   MSG_ENG_NO_TRADE_EVENTS,                           // There have been no trade events since the last launch of EA
   MSG_ENG_FAILED_GET_LAST_TRADE_EVENT_DESCR,         // Failed to get description of the last trading event
   MSG_ENG_FAILED_GET_MARKET_POS_LIST,                // Failed to get the list of open positions
   MSG_ENG_FAILED_GET_PENDING_ORD_LIST,               // Failed to get the list of placed orders
   MSG_ENG_NO_OPEN_POSITIONS,                         // No open positions
   MSG_ENG_NO_PLACED_ORDERS,                          // No placed orders

  };
```

Only the parts of the file with "geolocation" (where the indices enumeration constants should be added) are displayed here.

Now **add the required messages, whose indices we have just defined, to the array of text messages**:

```
   {"Ошибка. Такого символа нет на сервере","Error. No such symbol on server"},
   {"Ошибка. Такого символа нет в списке используемых символов: ","Error. This symbol is not in the list of symbols used: "},
   {"Не удалось поместить в обзор рынка. Ошибка: ","Failed to put in market watch. Error: "},
   {"Ошибка. Не позиция: ","Error. Not position: "},
   {"Ошибка. Нет открытой позиции с тикетом #","Error. No open position with ticket #"},
   {"Ошибка. Нет установленного ордера с тикетом #","Error. No placed order with ticket #"},
   {"Не удалось закрыть позицию. Ошибка ","Could not close position. Error "},
   {"Не удалось модифицировать ордер. Ошибка ","Failed to modify order. Error "},
   {"Ошибка: невозможно разместить ордер без явно заданного его времени истечения","Error: Unable to place order without explicitly specified expiration time"},
   {"Ошибка. Не удалось получить торговый объект","Error. Failed to get trade object"},
   {"Ошибка. Не удалось получить объект-позицию","Error. Failed to get position object"},
   {"Ошибка. Не удалось получить объект-ордер","Error. Failed to get order object"},
   {"Ошибка. Не удалось получить объект-символ","Error. Failed to get symbol object"},
   {"Код возврата вне заданного диапазона кодов ошибок","Return code out of range of error codes"},
   {"не удалось добавить в список","failed to add to list"},
   {"Будет использоваться время действия ордера до конца текущего дня","Order validity time until the end of the current day will be used"},

   {"Воскресение","Sunday"},
   {"Биржевой рынок","Exchange market mode"},
   {"Закрытие встречным доступно только на счетах с типом \"Хеджинг\"","Close by opposite position iavailable only on \"Hedging\" accounts"},
   {"Ошибка. Позиции для встречного закрытия имеют один и тот же тип","Error. Positions of the same type in counterclosure request"},

//--- CEngine
   {"С момента последнего запуска ЕА торговых событий не было","No trade events since the last launch of EA"},
   {"Не удалось получить описание последнего торгового события","Failed to get description of the last trading event"},
   {"Не удалось получить список открытых позиций","Failed to get open positions list"},
   {"Не удалось получить список установленных ордеров","Failed to get pending orders list"},
   {"Нет открытых позиций","No open positions"},
   {"Нет установленных ордеров","No placed orders"},

  };
```

Just like with defining indices constants, only certain areas for adding the necessary message texts are displayed here. In the attachments,
you can find and analyze the full version of the improved Datas.mqh.

When sending position closure requests, we need to know a type of the order opposite to the direction of the closed position (in MQL5, closing is
performed by opening an opposite position, while an order (not position) type is sent to a trading request).

In the service functions file of the **DELib.mqh** library, **write two functions for**
**receiving an order type by a position direction and  a type of an order**
**opposite to the position direction:**

```
//+------------------------------------------------------------------+
//| Return an order type by a position type                          |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE OrderTypeByPositionType(ENUM_POSITION_TYPE type_position)
  {
   return(type_position==POSITION_TYPE_BUY ? ORDER_TYPE_BUY :  ORDER_TYPE_SELL);
  }
//+------------------------------------------------------------------+
//| Return a reverse order type by a position type                   |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE OrderTypeOppositeByPositionType(ENUM_POSITION_TYPE type_position)
  {
   return(type_position==POSITION_TYPE_BUY ? ORDER_TYPE_SELL :  ORDER_TYPE_BUY);
  }
//+------------------------------------------------------------------+
```

**Now that we have prepared all the data, let's tackle the trading object class itself.**

In \\MQL5\\Include\\DoEasy\ **Objects\**, create **Trade\** subfolder and the new class **CTradeObj** in the **TradeObj.mqh**
file in it.

**Include the file of service functions to the newly created file:**

```
//+------------------------------------------------------------------+
//|                                                     TradeObj.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
```

**Add all the necessary class member variables and methods to the class file:**

```
//+------------------------------------------------------------------+
//|                                                     TradeObj.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "..\..\Services\DELib.mqh"
//+------------------------------------------------------------------+
//| Trading object class                                             |
//+------------------------------------------------------------------+
class CTradeObj
  {
private:
   MqlTick                    m_tick;                                            // Tick structure for receiving prices
   MqlTradeRequest            m_request;                                         // Trade request structure
   MqlTradeResult             m_result;                                          // trade request execution result
   ENUM_ACCOUNT_MARGIN_MODE   m_margin_mode;                                     // Margin calculation mode
   ENUM_ORDER_TYPE_FILLING    m_type_filling;                                    // Filling policy
   ENUM_ORDER_TYPE_TIME       m_type_expiration;                                 // Order expiration type
   int                        m_symbol_expiration_flags;                         // Flags of order expiration modes for a trading object symbol
   ulong                      m_magic;                                           // Magic number
   string                     m_symbol;                                          // Symbol
   string                     m_comment;                                         // Comment
   ulong                      m_deviation;                                       // Slippage in points
   double                     m_volume;                                          // Volume
   datetime                   m_expiration;                                      // Order expiration time (for ORDER_TIME_SPECIFIED type order)
   bool                       m_async_mode;                                      // Flag of asynchronous sending of a trade request
   ENUM_LOG_LEVEL             m_log_level;                                       // Logging level
   int                        m_stop_limit;                                      // Distance of placing a StopLimit order in points
public:
//--- Constructor
                              CTradeObj();;

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

//--- (1) Return the margin calculation mode, (2) hedge account flag
   ENUM_ACCOUNT_MARGIN_MODE   GetMarginMode(void)                                const { return this.m_margin_mode;           }
   bool                       IsHedge(void) const { return this.GetMarginMode()==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;          }
//--- (1) Set, (2) return the error logging level
   void                       SetLogLevel(const ENUM_LOG_LEVEL level)                  { this.m_log_level=level;              }
   ENUM_LOG_LEVEL             GetLogLevel(void)                                  const { return this.m_log_level;             }
//--- (1) Set, (2) return the filling policy
   void                       SetTypeFilling(const ENUM_ORDER_TYPE_FILLING type)       { this.m_type_filling=type;            }
   ENUM_ORDER_TYPE_FILLING    GetTypeFilling(void)                               const { return this.m_type_filling;          }
//--- (1) Set, (2) return order expiration type
   void                       SetTypeExpiration(const ENUM_ORDER_TYPE_TIME type)       { this.m_type_expiration=type;         }
   ENUM_ORDER_TYPE_TIME       GetTypeExpiration(void)                            const { return this.m_type_expiration;       }
//--- (1) Set, (2) return the magic number
   void                       SetMagic(const ulong magic)                              { this.m_magic=magic;                  }
   ulong                      GetMagic(void)                                     const { return this.m_magic;                 }
//--- (1) Set, (2) return a symbol
   void                       SetSymbol(const string symbol)                           { this.m_symbol=symbol;                }
   string                     GetSymbol(void)                                    const { return this.m_symbol;                }
//--- (1) Set, (2) return a comment
   void                       SetComment(const string comment)                         { this.m_comment=comment;              }
   string                     GetComment(void)                                   const { return this.m_comment;               }
//--- (1) Set, (2) return slippage
   void                       SetDeviation(const ulong deviation)                      { this.m_deviation=deviation;          }
   ulong                      GetDeviation(void)                                 const { return this.m_deviation;             }
//--- (1) Set, (2) return volume
   void                       SetVolume(const double volume)                           { this.m_volume=volume;                }
   double                     GetVolume(void)                                    const { return this.m_volume;                }
//--- (1) Set, (2) return order expiration date
   void                       SetExpiration(const datetime time)                       { this.m_expiration=time;              }
   datetime                   GetExpiration(void)                                const { return this.m_expiration;            }
//--- (1) Set, (2) return the flag of the asynchronous sending of a trading request
   void                       SetAsyncMode(const bool async)                           { this.m_async_mode=async;             }
   bool                       GetAsyncMode(void)                                 const { return this.m_async_mode;            }

//--- Last request data:
//--- Return (1) executed action type, (2) magic number, (3) order ticket, (4) volume,
//--- (5) open, (6) StopLimit order, (7) StopLoss, (8) TakeProfit price, (9) deviation,
//--- type of (10) order, (11) execution, (12) lifetime, (13) order expiration date,
//--- (14) comment, (15) position ticket, (16) opposite position ticket
   ENUM_TRADE_REQUEST_ACTIONS GetLastRequestAction(void)                         const { return this.m_request.action;        }
   ulong                      GetLastRequestMagic(void)                          const { return this.m_request.magic;         }
   ulong                      GetLastRequestOrder(void)                          const { return this.m_request.order;         }
   double                     GetLastRequestVolume(void)                         const { return this.m_request.volume;        }
   double                     GetLastRequestPrice(void)                          const { return this.m_request.price;         }
   double                     GetLastRequestStopLimit(void)                      const { return this.m_request.stoplimit;     }
   double                     GetLastRequestStopLoss(void)                       const { return this.m_request.sl;            }
   double                     GetLastRequestTakeProfit(void)                     const { return this.m_request.tp;            }
   ulong                      GetLastRequestDeviation(void)                      const { return this.m_request.deviation;     }
   ENUM_ORDER_TYPE            GetLastRequestType(void)                           const { return this.m_request.type;          }
   ENUM_ORDER_TYPE_FILLING    GetLastRequestTypeFilling(void)                    const { return this.m_request.type_filling;  }
   ENUM_ORDER_TYPE_TIME       GetLastRequestTypeTime(void)                       const { return this.m_request.type_time;     }
   datetime                   GetLastRequestExpiration(void)                     const { return this.m_request.expiration;    }
   string                     GetLastRequestComment(void)                        const { return this.m_request.comment;       }
   ulong                      GetLastRequestPosition(void)                       const { return this.m_request.position;      }
   ulong                      GetLastRequestPositionBy(void)                     const { return this.m_request.position_by;   }

//--- Data on the last request result:
//--- Return (1) operation result code, (2) performed deal ticket, (3) placed order ticket,
//--- (4) deal volume confirmed by a broker, (5) deal price confirmed by a broker,
//--- (6) current market Bid (requote) price, (7) current market Ask (requote) price
//--- (8) broker comment to operation (by default, it is filled by the trade server return code description),
//--- (9) request ID set by the terminal when sending, (10) external trading system return code
   uint                       GetResultRetcode(void)                             const { return this.m_result.retcode;        }
   ulong                      GetResultDeal(void)                                const { return this.m_result.deal;           }
   ulong                      GetResultOrder(void)                               const { return this.m_result.order;          }
   double                     GetResultVolume(void)                              const { return this.m_result.volume;         }
   double                     GetResultPrice(void)                               const { return this.m_result.price;          }
   double                     GetResultBid(void)                                 const { return this.m_result.bid;            }
   double                     GetResultAsk(void)                                 const { return this.m_result.ask;            }
   string                     GetResultComment(void)                             const { return this.m_result.comment;        }
   uint                       GetResultRequestID(void)                           const { return this.m_result.request_id;     }
   uint                       GetResultRetcodeEXT(void)                          const { return this.m_result.retcode_external;}

//--- Open a position
   bool                       OpenPosition(const ENUM_POSITION_TYPE type,
                                           const double volume,
                                           const double sl=0,
                                           const double tp=0,
                                           const ulong magic=ULONG_MAX,
                                           const ulong deviation=ULONG_MAX,
                                           const string comment=NULL);
//--- Close a position
   bool                       ClosePosition(const ulong ticket,
                                            const ulong deviation=ULONG_MAX,
                                            const string comment=NULL);
//--- Close a position partially
   bool                       ClosePositionPartially(const ulong ticket,
                                                     const double volume,
                                                     const ulong deviation=ULONG_MAX,
                                                     const string comment=NULL);
//--- Close a position by an opposite one
   bool                       ClosePositionBy(const ulong ticket,const ulong ticket_by);
//--- Modify a position
   bool                       ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE);
//--- Place an order
   bool                       SetOrder(const ENUM_ORDER_TYPE type,
                                       const double volume,
                                       const double price,
                                       const double sl=0,
                                       const double tp=0,
                                       const double price_stoplimit=0,
                                       const ulong magic=ULONG_MAX,
                                       const datetime expiration=0,
                                       const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC,
                                       const string comment=NULL);
//--- Remove an order
   bool                       DeleteOrder(const ulong ticket);
//--- Modify an order
   bool                       ModifyOrder(const ulong ticket,
                                          const double price=WRONG_VALUE,
                                          const double sl=WRONG_VALUE,
                                          const double tp=WRONG_VALUE,
                                          const double price_stoplimit=WRONG_VALUE,
                                          const datetime expiration=WRONG_VALUE,
                                          const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE);

  };
//+------------------------------------------------------------------+
```

Let's have a closer look at what we wrote here.

To get the current prices, we need to access the properties of a symbol, on which a trading request is to be sent. Since we need relevant prices,
we should get them immediately before sending a trading request. This is why we place the

**m\_tick** variable with the **MqlTick** structure type directly in the base trading object. We can pass it from a symbol object but it
would be better to do without redundant (albeit minimum) costs of passing a property to a trading object.

The **m\_request** variable with the [trading \\
request structure](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) type of **MqlTradeRequest** is used to fill in all trading request properties and send them to the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend)
function. The **m\_result** variable with the [trading \\
request result structure](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) of **MqlTradeResult** type is passed to the function as well. It is filled by the server when receiving an
answer from the trade server. If the result of sending an order to the server is incorrect, we can always read the fields of the trade request
result structure to understand what happened.

I believe, other class member variables require no explanations.

**Let's have a look at the implementation of the class methods.**

The methods of setting and getting the trading request properties (Set and Get
methods) are written in the class body. All they do is write the value passed to the method to the appropriate variable or return the
value of the appropriate variable. These methods work only with the variables storing the default values. In other words, these methods
allow you to set the required property of a trading request. Further on, it will have the defined value as a default one. If you need to use
another value for a trading request once, the methods of sending trading requests feature passing the required values for a one-time use of
the value passed to the method.

The methods returning the parameters of the last trading request,
allow defining what value has been passed to a property of the last trading request and take steps to eliminate errors or use these values for
the next server request.

The methods simply return the contents of the trading request structure fields corresponding to the method. Before sending a request,
this structure (some of its fields corresponding to a trading request) are filled in and passed to the function of sending a request to the
server. This is the structure, from which we get the values that have been last filled.

The methods returning the trading request result allow
getting data on the trading request handling result. If the request turns out to be erroneous, we can see clarification on the error code in

**retcode**. Alternatively, the structure is filled with data on an open position or placed pending order, while **request\_id** features
the code of a request, which can be then analyzed in the

[OnTradeTransaction()](https://www.mql5.com/en/docs/event_handlers/ontradetransaction) handler to link the trading
request sent to the server via

[OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) with the request result.

In this
library, we do not use OnTradeTransaction() due to its absence in MQL4. We will analyze the asynchronous sending of requests and their
results on our own.

**Class constructor:**

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
  }
//+------------------------------------------------------------------+
```

Set initializing values in its initialization list:

- magic number is zero,

- slippage is five points,

- StopLimit order price is zero (no price),

- order expiration time is zero (unlimited),
- [asynchronous sending of trading requests](https://www.mql5.com/en/docs/trading/ordersendasync) is
disabled,



- order filling policy "Fill or Kill",

- order expiration time — unlimited
- in the order comment, set the program name + " by DoEasy",
- mode of logging trading class operation — errors only.

In the class body, add the margin calculation method set for the account to the **m\_margin\_mode** variable.


For MQL5, get the required value using the

[AccountInfoInteger()](https://www.mql5.com/en/docs/account/accountinfointeger) function with the [ACCOUNT\_MARGIN\_MODE](https://www.mql5.com/en/docs/constants/environment_state/accountinformation)
property ID.

For MQL4, set the hedging mode of margin calculation right away ( [ACCOUNT\_MARGIN\_MODE\_RETAIL\_HEDGING](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_margin_mode)).

We will be able to pass the required values of the necessary properties to trading methods for filling in a trading request. But often we
do not need to fill out all the necessary properties — they usually should be unchanged for each trading request. Therefore, we should be
able to initialize the default variables and select the values to be used for a trading request in trading methods — either the one passed
to the method of sending a request to the server or the one set by default.

**Write the method of initializing trading request default parameters:**

```
//+------------------------------------------------------------------+
//| Set default values                                               |
//+------------------------------------------------------------------+
void CTradeObj::Init(const string symbol,
                     const ulong magic,
                     const double volume,
                     const ulong deviation,
                     const int stoplimit,
                     const datetime expiration,
                     const bool async_mode,
                     const ENUM_ORDER_TYPE_FILLING type_filling,
                     const ENUM_ORDER_TYPE_TIME type_expiration,
                     ENUM_LOG_LEVEL log_level)
  {
   this.SetSymbol(symbol);
   this.SetMagic(magic);
   this.SetDeviation(deviation);
   this.SetVolume(volume);
   this.SetExpiration(expiration);
   this.SetTypeFilling(type_filling);
   this.SetTypeExpiration(type_expiration);
   this.SetAsyncMode(async_mode);
   this.SetLogLevel(log_level);
   this.m_symbol_expiration_flags=(int)::SymbolInfoInteger(this.m_symbol,SYMBOL_EXPIRATION_MODE);
   this.m_volume=::SymbolInfoDouble(this.m_symbol,SYMBOL_VOLUME_MIN);
  }
//+------------------------------------------------------------------+
```

The method receives the required values of trading request parameters. In the method body, the passed values are set for the appropriate
variables using their setting methods considered above.

[The flags of \\
allowed order expiration modes](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#symbol_expiration_mode) are set using the [SymbolInfoInteger()](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger)
function with the [SYMBOL\_EXPIRATION\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) property
ID. The applied volume is the one minimally accepted for a symbol using the

[SymbolInfoDouble()](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function with the [SYMBOL\_VOLUME\_MIN](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)
property ID.

**Position opening method:**

```
//+------------------------------------------------------------------+
//| Open a position                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::OpenPosition(const ENUM_POSITION_TYPE type,
                             const double volume,
                             const double sl=0,
                             const double tp=0,
                             const ulong magic=ULONG_MAX,
                             const ulong deviation=ULONG_MAX,
                             const string comment=NULL)
  {
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
   this.m_request.type     =  OrderTypeByPositionType(type);
   this.m_request.price    =  (type==POSITION_TYPE_BUY ? this.m_tick.ask : this.m_tick.bid);
   this.m_request.volume   =  volume;
   this.m_request.sl       =  sl;
   this.m_request.tp       =  tp;
   this.m_request.deviation=  (deviation==ULONG_MAX ? this.m_deviation : deviation);
   this.m_request.comment  =  (comment==NULL ? this.m_comment : comment);
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         (::OrderSend(m_request.symbol,m_request.type,m_request.volume,m_request.price,(int)m_request.deviation,m_request.sl,m_request.tp,m_request.comment,(int)m_request.magic,m_request.expiration,clrNONE)!=WRONG_VALUE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the position type, its volume, StopLoss and TakeProfit prices, position magic number, slippage value and comment.


The default values are set for StopLoss, TakeProfit, magic number, slippage and comment. If these values are left unchanged when the method
is called, then the values set by default in the Init() method or set directly from the program by the methods of setting default values, which
we examined above, are used for these values. The entire method logic is written in the code comments.

The result sent by the OrderTypeByPositionType() function
we wrote in DELib.mqh for receiving an order type by a position type is sent to the field of the structure of a trading request storing the order
type. The method does not verify the parameters passed to it. It is assumed they have already been verified.

For MQL4, we also do not change anything yet when returning the
result of sending a request to the server and do not fill in the structure of the trading request result. Currently, we need to quickly gather
trading methods for testing. We will put everything in order in the coming articles.

**Position closing method:**

```
//+------------------------------------------------------------------+
//| Close a position                                                 |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePosition(const ulong ticket,
                              const ulong deviation=ULONG_MAX,
                              const string comment=NULL)
  {
   //--- If failed to select a position. Write the error code and description, send the message to the journal and return 'false'
   if(!::PositionSelectByTicket(ticket))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_POS),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get a position symbol
   string symbol=::PositionGetString(POSITION_SYMBOL);
   //--- If failed to get the current prices, write the error code and description, send the message to the journal and return 'false'
   if(!::SymbolInfoTick(symbol,this.m_tick))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_PRICE),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get a position type and an order type inverse of the position type
   ENUM_POSITION_TYPE position_type=(ENUM_POSITION_TYPE)::PositionGetInteger(POSITION_TYPE);
   ENUM_ORDER_TYPE type=OrderTypeOppositeByPositionType(position_type);
   //--- Get a position volume and magic number
   double position_volume=::PositionGetDouble(POSITION_VOLUME);
   ulong magic=::PositionGetInteger(POSITION_MAGIC);
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action   =  TRADE_ACTION_DEAL;
   this.m_request.symbol   =  symbol;
   this.m_request.magic    =  magic;
   this.m_request.type     =  type;
   this.m_request.price    =  (position_type==POSITION_TYPE_SELL ? this.m_tick.ask : this.m_tick.bid);
   this.m_request.volume   =  position_volume;
   this.m_request.deviation=  (deviation==ULONG_MAX ? this.m_deviation : deviation);
   this.m_request.comment  =  (comment==NULL ? this.m_comment : comment);
   //--- In case of a hedging account, write the ticket of a closed position to the structure
   if(this.IsHedge())
      this.m_request.position=::PositionGetInteger(POSITION_TICKET);
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderClose((int)m_request.position,m_request.volume,m_request.price,(int)m_request.deviation,clrNONE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a closed position, slippage and comment.

Here (as well as in other trading methods)
everything is similar to the position opening method discussed above.

**The method for position partial closing:**

```
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePositionPartially(const ulong ticket,
                                       const double volume,
                                       const ulong deviation=ULONG_MAX,
                                       const string comment=NULL)
  {
   //--- If failed to select a position. Write the error code and description, send the message to the journal and return 'false'
   if(!::PositionSelectByTicket(ticket))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_POS),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get a position symbol
   string symbol=::PositionGetString(POSITION_SYMBOL);
   //--- If failed to get the current prices, write the error code and description, send the message to the journal and return 'false'
   if(!::SymbolInfoTick(symbol,this.m_tick))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_GET_PRICE),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get a position type and an order type inverse of the position type
   ENUM_POSITION_TYPE position_type=(ENUM_POSITION_TYPE)::PositionGetInteger(POSITION_TYPE);
   ENUM_ORDER_TYPE type=OrderTypeOppositeByPositionType(position_type);
   //--- Get a position volume and magic number
   double position_volume=::PositionGetDouble(POSITION_VOLUME);
   ulong magic=::PositionGetInteger(POSITION_MAGIC);
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action   =  TRADE_ACTION_DEAL;
   this.m_request.position =  ticket;
   this.m_request.symbol   =  symbol;
   this.m_request.magic    =  magic;
   this.m_request.type     =  type;
   this.m_request.price    =  (position_type==POSITION_TYPE_SELL ? this.m_tick.ask : this.m_tick.bid);
   this.m_request.volume   =  (volume<position_volume ? volume : position_volume);
   this.m_request.deviation=  (deviation==ULONG_MAX ? this.m_deviation : deviation);
   this.m_request.comment  =  (comment==NULL ? this.m_comment : comment);
   //--- In case of a hedging account, write the ticket of a closed position to the structure
   if(this.IsHedge())
      this.m_request.position=::PositionGetInteger(POSITION_TICKET);
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderClose((int)m_request.position,m_request.volume,m_request.price,(int)m_request.deviation,clrNONE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method passes the ticket of a closed position, a volume to be closed, a slippage and a comment.

If
a closing volume passed to the method exceeds the existing position volume, the position is closed in full.

**The method of closing a position by an opposite one:**

```
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CTradeObj::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   #ifdef __MQL5__
   //--- If this is not a hedging account.
   if(::AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
     {
      //--- Close by is available only on hedging accounts.
      //---Write the error code and description, send the message to the journal and return 'false'
      this.m_result.retcode=MSG_ACC_UNABLE_CLOSE_BY;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_ACC_UNABLE_CLOSE_BY));
      return false;
     }
   #endif
//--- Closed position
   //--- If failed to select a position, write the error code and description, send the message to the journal and return 'false'
   if(!::PositionSelectByTicket(ticket))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_POS),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get a type and magic of a closed position
   ENUM_POSITION_TYPE position_type=(ENUM_POSITION_TYPE)::PositionGetInteger(POSITION_TYPE);
   ulong magic=::PositionGetInteger(POSITION_MAGIC);

//--- Opposite position
   //--- If failed to select a position, write the error code and description, send the message to the journal and return 'false'
   if(!::PositionSelectByTicket(ticket_by))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket_by,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_POS),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Get an opposite position type
   ENUM_POSITION_TYPE position_type_by=(ENUM_POSITION_TYPE)::PositionGetInteger(POSITION_TYPE);
   //--- If types of a closed and an opposite position match, write the error code and description, send the message to the journal and return 'false'
   if(position_type==position_type_by)
     {
      this.m_result.retcode=MSG_ACC_SAME_TYPE_CLOSE_BY;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,CMessage::Text(MSG_ACC_SAME_TYPE_CLOSE_BY));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   this.m_request.action      =  TRADE_ACTION_CLOSE_BY;
   this.m_request.position    =  ticket;
   this.m_request.position_by =  ticket_by;
   this.m_request.magic       =  magic;
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderCloseBy((int)m_request.position,(int)m_request.position_by,clrNONE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a closed position and a ticket of an opposite one.

**The method of modifying position's stop levels:**

```
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
bool CTradeObj::ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE)
  {
   //--- If all default values are passed, there is nothing to be modified
   if(sl==WRONG_VALUE && tp==WRONG_VALUE)
     {
      //--- There are no changes in the request - write the error code and description, send the message to the journal and return 'false'
      this.m_result.retcode= #ifdef __MQL5__ TRADE_RETCODE_NO_CHANGES #else 10025 #endif ;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,CMessage::Text(this.m_result.retcode),CMessage::Retcode(this.m_result.retcode));
      return false;
     }
   //--- If failed to select a position, write the error code and description, send the message to the journal and return 'false'
   if(!::PositionSelectByTicket(ticket))
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_POS),CMessage::Text(this.m_result.retcode));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   m_request.action  =  TRADE_ACTION_SLTP;
   m_request.position=  ticket;
   m_request.symbol  =  ::PositionGetString(POSITION_SYMBOL);
   m_request.magic   =  ::PositionGetInteger(POSITION_MAGIC);
   m_request.sl      =  (sl==WRONG_VALUE ? ::PositionGetDouble(POSITION_SL) : sl);
   m_request.tp      =  (tp==WRONG_VALUE ? ::PositionGetDouble(POSITION_TP) : tp);
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderModify((int)m_request.position,::OrderOpenPrice(),m_request.sl,m_request.tp,::OrderExpiration(),clrNONE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a modified position and new StopLoss and TakeProfit levels.

**The method for placing a pending order:**

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
                         const datetime expiration=0,
                         const ENUM_ORDER_TYPE_TIME type_time=ORDER_TIME_GTC,
                         const string comment=NULL)
  {
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
   m_request.action     =  TRADE_ACTION_PENDING;
   m_request.symbol     =  this.m_symbol;
   m_request.magic      =  (magic==ULONG_MAX ? this.m_magic : magic);
   m_request.volume     =  volume;
   m_request.type       =  type;
   m_request.stoplimit  =  price_stoplimit;
   m_request.price      =  price;
   m_request.sl         =  sl;
   m_request.tp         =  tp;
   m_request.type_time  =  type_time;
   m_request.expiration =  expiration;
   m_request.comment    =  (comment==NULL ? this.m_comment : comment);
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         (::OrderSend(m_request.symbol,m_request.type,m_request.volume,m_request.price,(int)m_request.deviation,m_request.sl,m_request.tp,m_request.comment,(int)m_request.magic,m_request.expiration,clrNONE)!=WRONG_VALUE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the type of a pending order, its volume, open price, StopLoss, TakeProfit and StopLimit order prices, magic number,
order lifetime, order expiration type and comment.

**The method for removing a pending order:**

```
//+------------------------------------------------------------------+
//| Remove an order                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::DeleteOrder(const ulong ticket)
  {
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   m_request.action  =  TRADE_ACTION_REMOVE;
   m_request.order   =  ticket;
   //--- Return the result of sending a request to the server
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderDelete((int)m_request.order,clrNONE)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a removed order.

**The method for modifying a pending order:**

```
//+------------------------------------------------------------------+
//| Modify an order                                                  |
//+------------------------------------------------------------------+
bool CTradeObj::ModifyOrder(const ulong ticket,
                            const double price=WRONG_VALUE,
                            const double sl=WRONG_VALUE,
                            const double tp=WRONG_VALUE,
                            const double price_stoplimit=WRONG_VALUE,
                            const datetime expiration=WRONG_VALUE,
                            const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE)
  {
   //--- If failed to select an order, write the error code and description, send the message to the journal and return 'false'
   #ifdef __MQL5__
   if(!::OrderSelect(ticket))
   #else
   if(!::OrderSelect((int)ticket,SELECT_BY_TICKET))
   #endif
     {
      this.m_result.retcode=::GetLastError();
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,"#",(string)ticket,": ",CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_SELECT_ORD),CMessage::Text(this.m_result.retcode));
      return false;
     }
   double order_price=::OrderGetDouble(ORDER_PRICE_OPEN);
   double order_sl=::OrderGetDouble(ORDER_SL);
   double order_tp=::OrderGetDouble(ORDER_TP);
   double order_stoplimit=::OrderGetDouble(ORDER_PRICE_STOPLIMIT);
   ENUM_ORDER_TYPE_TIME order_type_time=(ENUM_ORDER_TYPE_TIME)::OrderGetInteger(ORDER_TYPE_TIME);
   datetime order_expiration=(datetime)::OrderGetInteger(ORDER_TIME_EXPIRATION);
   //--- If the default values are passed and the price is equal to the price set in the order, the request is unchanged
   //---Write the error code and description, send the message to the journal and return 'false'
   if(price==order_price && sl==WRONG_VALUE && tp==WRONG_VALUE && price_stoplimit==WRONG_VALUE && type_time==WRONG_VALUE && expiration==WRONG_VALUE)
     {
      this.m_result.retcode = #ifdef __MQL5__ TRADE_RETCODE_NO_CHANGES #else 10025 #endif ;
      this.m_result.comment=CMessage::Text(this.m_result.retcode);
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         Print(DFUN,CMessage::Text(this.m_result.retcode),CMessage::Retcode(this.m_result.retcode));
      return false;
     }
   //--- Clear the structures
   ::ZeroMemory(this.m_request);
   ::ZeroMemory(this.m_result);
   //--- Fill in the request structure
   m_request.action     =  TRADE_ACTION_MODIFY;
   m_request.order      =  ticket;
   m_request.price      =  (price==WRONG_VALUE ? order_price : price);
   m_request.sl         =  (sl==WRONG_VALUE ? order_sl : sl);
   m_request.tp         =  (tp==WRONG_VALUE ? order_tp : tp);
   m_request.stoplimit  =  (price_stoplimit==WRONG_VALUE ? order_stoplimit : price_stoplimit);
   m_request.type_time  =  (type_time==WRONG_VALUE ? order_type_time : type_time);
   m_request.expiration =  (expiration==WRONG_VALUE ? order_expiration : expiration);
   //--- Return an order modification result
   return
     (
      #ifdef __MQL5__
         !this.m_async_mode ? ::OrderSend(this.m_request,this.m_result) : ::OrderSendAsync(this.m_request,this.m_result)
      #else
         ::OrderModify((int)m_request.order,m_request.price,m_request.sl,m_request.tp,m_request.expiration,clrNONE)
      #endif
     );
   Print(DFUN);
  }
//+------------------------------------------------------------------+
```

The method receives a modified order ticket, new price values and StopLoss, TakeProfit, StopLimit order levels, as well as order lifetime
and expiration type.

All the methods feature identical checks of default values passed to the method. All actions are commented in the code. The comments are of the
same type, so there is no point in dwelling on them.

**This concludes the creation of the minimum functionality of the base trading class.**

Since we send any trading requests in relation to a symbol, let's place the base trading object in the symbol object and make it accessible from
the outside.

Open the symbol object file \\MQL5\\Include\\DoEasy\\Objects\\Symbols\ **Symbol.mqh** and **include**
**the TradeObj.mqh trading object file to it:**

```
//+------------------------------------------------------------------+
//|                                                       Symbol.mqh |
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
#include "..\BaseObj.mqh"
#include "..\Trade\TradeObj.mqh"
//+------------------------------------------------------------------+
```

**In the private section, declare the variable object of the trading class:**

```
//+------------------------------------------------------------------+
//| Abstract symbol class                                            |
//+------------------------------------------------------------------+
class CSymbol : public CBaseObj
  {
private:
   struct MqlMarginRate
     {
      double         Initial;                                  // initial margin rate
      double         Maintenance;                              // maintenance margin rate
     };
   struct MqlMarginRateMode
     {
      MqlMarginRate  Long;                                     // MarginRate of long positions
      MqlMarginRate  Short;                                    // MarginRate of short positions
      MqlMarginRate  BuyStop;                                  // MarginRate of BuyStop orders
      MqlMarginRate  BuyLimit;                                 // MarginRate of BuyLimit orders
      MqlMarginRate  BuyStopLimit;                             // MarginRate of BuyStopLimit orders
      MqlMarginRate  SellStop;                                 // MarginRate of SellStop orders
      MqlMarginRate  SellLimit;                                // MarginRate of SellLimit orders
      MqlMarginRate  SellStopLimit;                            // MarginRate of SellStopLimit orders
     };
   MqlMarginRateMode m_margin_rate;                            // Margin ratio structure
   MqlBookInfo       m_book_info_array[];                      // Array of the market depth data structures

   long              m_long_prop[SYMBOL_PROP_INTEGER_TOTAL];   // Integer properties
   double            m_double_prop[SYMBOL_PROP_DOUBLE_TOTAL];  // Real properties
   string            m_string_prop[SYMBOL_PROP_STRING_TOTAL];  // String properties
   bool              m_is_change_trade_mode;                   // Flag of changing trading mode for a symbol
   CTradeObj         m_trade;                                  // Trading class object
//--- Return the index of the array the symbol's (1) double and (2) string properties are located at
   int               IndexProp(ENUM_SYMBOL_PROP_DOUBLE property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL;                                    }
   int               IndexProp(ENUM_SYMBOL_PROP_STRING property)  const { return(int)property-SYMBOL_PROP_INTEGER_TOTAL-SYMBOL_PROP_DOUBLE_TOTAL;           }
//--- (1) Fill in all the "margin ratio" symbol properties, (2) initialize the ratios
   bool              MarginRates(void);
   void              InitMarginRates(void);
//--- Reset all symbol object data
   void              Reset(void);
//--- Return the current day of the week
   ENUM_DAY_OF_WEEK  CurrentDayOfWeek(void)              const;

public:
//--- Default constructor
```

Declare the two methods in the public section of the class:

**the method returning the correct filling policy and the method**
**returning the correct order expiration type:**

```
public:
//--- Set (1) integer, (2) real and (3) string symbol properties
   void              SetProperty(ENUM_SYMBOL_PROP_INTEGER property,long value)   { this.m_long_prop[property]=value;                                        }
   void              SetProperty(ENUM_SYMBOL_PROP_DOUBLE property,double value)  { this.m_double_prop[this.IndexProp(property)]=value;                      }
   void              SetProperty(ENUM_SYMBOL_PROP_STRING property,string value)  { this.m_string_prop[this.IndexProp(property)]=value;                      }
//--- Return (1) integer, (2) real and (3) string symbol properties from the properties array
   long              GetProperty(ENUM_SYMBOL_PROP_INTEGER property)        const { return this.m_long_prop[property];                                       }
   double            GetProperty(ENUM_SYMBOL_PROP_DOUBLE property)         const { return this.m_double_prop[this.IndexProp(property)];                     }
   string            GetProperty(ENUM_SYMBOL_PROP_STRING property)         const { return this.m_string_prop[this.IndexProp(property)];                     }

//--- Return the flag of a symbol supporting the property
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_INTEGER property)          { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_DOUBLE property)           { return true; }
   virtual bool      SupportProperty(ENUM_SYMBOL_PROP_STRING property)           { return true; }

//--- Return the flag of allowing (1) market, (2) limit, (3) stop (4) and stop limit orders,
//--- the flag of allowing setting (5) StopLoss and (6) TakeProfit orders, (7) as well as closing by an opposite order
   bool              IsMarketOrdersAllowed(void)            const { return((this.OrderModeFlags() & SYMBOL_ORDER_MARKET)==SYMBOL_ORDER_MARKET);             }
   bool              IsLimitOrdersAllowed(void)             const { return((this.OrderModeFlags() & SYMBOL_ORDER_LIMIT)==SYMBOL_ORDER_LIMIT);               }
   bool              IsStopOrdersAllowed(void)              const { return((this.OrderModeFlags() & SYMBOL_ORDER_STOP)==SYMBOL_ORDER_STOP);                 }
   bool              IsStopLimitOrdersAllowed(void)         const { return((this.OrderModeFlags() & SYMBOL_ORDER_STOP_LIMIT)==SYMBOL_ORDER_STOP_LIMIT);     }
   bool              IsStopLossOrdersAllowed(void)          const { return((this.OrderModeFlags() & SYMBOL_ORDER_SL)==SYMBOL_ORDER_SL);                     }
   bool              IsTakeProfitOrdersAllowed(void)        const { return((this.OrderModeFlags() & SYMBOL_ORDER_TP)==SYMBOL_ORDER_TP);                     }
   bool              IsCloseByOrdersAllowed(void)           const;

//--- Return the (1) FOK and (2) IOC filling flag
   bool              IsFillingModeFOK(void)                 const { return((this.FillingModeFlags() & SYMBOL_FILLING_FOK)==SYMBOL_FILLING_FOK);             }
   bool              IsFillingModeIOC(void)                 const { return((this.FillingModeFlags() & SYMBOL_FILLING_IOC)==SYMBOL_FILLING_IOC);             }

//--- Return the flag of order expiration: (1) GTC, (2) DAY, (3) Specified and (4) Specified Day
   bool              IsExpirationModeGTC(void)              const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_GTC)==SYMBOL_EXPIRATION_GTC);    }
   bool              IsExpirationModeDAY(void)              const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_DAY)==SYMBOL_EXPIRATION_DAY);    }
   bool              IsExpirationModeSpecified(void)        const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_SPECIFIED)==SYMBOL_EXPIRATION_SPECIFIED);          }
   bool              IsExpirationModeSpecifiedDay(void)     const { return((this.ExpirationModeFlags() & SYMBOL_EXPIRATION_SPECIFIED_DAY)==SYMBOL_EXPIRATION_SPECIFIED_DAY);  }

//--- Return the description of allowing (1) market, (2) limit, (3) stop and (4) stop limit orders,
//--- the description of allowing (5) StopLoss and (6) TakeProfit orders, (7) as well as closing by an opposite order
   string            GetMarketOrdersAllowedDescription(void)      const;
   string            GetLimitOrdersAllowedDescription(void)       const;
   string            GetStopOrdersAllowedDescription(void)        const;
   string            GetStopLimitOrdersAllowedDescription(void)   const;
   string            GetStopLossOrdersAllowedDescription(void)    const;
   string            GetTakeProfitOrdersAllowedDescription(void)  const;
   string            GetCloseByOrdersAllowedDescription(void)     const;

//--- Return the description of allowing the filling type (1) FOK and (2) IOC, (3) as well as allowed order expiration modes
   string            GetFillingModeFOKAllowedDescrioption(void)   const;
   string            GetFillingModeIOCAllowedDescrioption(void)   const;

//--- Return the description of order expiration: (1) GTC, (2) DAY, (3) Specified and (4) Specified Day
   string            GetExpirationModeGTCDescription(void)        const;
   string            GetExpirationModeDAYDescription(void)        const;
   string            GetExpirationModeSpecifiedDescription(void)  const;
   string            GetExpirationModeSpecDayDescription(void)    const;

//--- Return the description of the (1) status, (2) price type for constructing bars,
//--- (3) method of calculating margin, (4) instrument trading mode,
//--- (5) deal execution mode for a symbol, (6) swap calculation mode,
//--- (7) StopLoss and TakeProfit lifetime, (8) option type, (9) option rights
//--- flags of (10) allowed order types, (11) allowed filling types,
//--- (12) allowed order expiration modes
   string            GetStatusDescription(void)                   const;
   string            GetChartModeDescription(void)                const;
   string            GetCalcModeDescription(void)                 const;
   string            GetTradeModeDescription(void)                const;
   string            GetTradeExecDescription(void)                const;
   string            GetSwapModeDescription(void)                 const;
   string            GetOrderGTCModeDescription(void)             const;
   string            GetOptionTypeDescription(void)               const;
   string            GetOptionRightDescription(void)              const;
   string            GetOrderModeFlagsDescription(void)           const;
   string            GetFillingModeFlagsDescription(void)         const;
   string            GetExpirationModeFlagsDescription(void)      const;

//--- Return (1) execution type, (2) order expiration type equal to 'type' if it is available on a symbol, otherwise - the correct option
   ENUM_ORDER_TYPE_FILLING GetCorrectTypeFilling(const uint type=ORDER_FILLING_RETURN);
   ENUM_ORDER_TYPE_TIME    GetCorrectTypeExpiration(uint expiration=ORDER_TIME_GTC);
//+------------------------------------------------------------------+
```

**Let's write their implementation outside the class body:**

```
//+------------------------------------------------------------------+
//| Return an order expiration type equal to 'type',                 |
//| if it is available on a symbol, otherwise, the correct option    |
//| https://www.mql5.com/ru/forum/170952/page4#comment_4128864       |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING CSymbol::GetCorrectTypeFilling(const uint type=ORDER_FILLING_RETURN)
  {
  const ENUM_SYMBOL_TRADE_EXECUTION exe_mode=this.TradeExecutionMode();
  const int filling_mode=this.FillingModeFlags();

  return(
         (filling_mode == 0 || (type >= ORDER_FILLING_RETURN) || ((filling_mode & (type + 1)) != type + 1)) ?
         (((exe_mode == SYMBOL_TRADE_EXECUTION_EXCHANGE) || (exe_mode == SYMBOL_TRADE_EXECUTION_INSTANT)) ?
           ORDER_FILLING_RETURN : ((filling_mode == SYMBOL_FILLING_IOC) ? ORDER_FILLING_IOC : ORDER_FILLING_FOK)) :
         (ENUM_ORDER_TYPE_FILLING)type
        );
  }
//+------------------------------------------------------------------+
//| Return order expiration type equal to 'expiration'               |
//| if it is available on Symb symbol, otherwise - the correct option|
//| https://www.mql5.com/en/forum/170952/page4#comment_4128871       |
//| Application:                                                     |
//| Request.type_time = GetExpirationType((uint)Expiration);         |
//| 'Expiration' can be datetime                                     |
//| if(Expiration > ORDER_TIME_DAY) Request.expiration = Expiration; |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_TIME CSymbol::GetCorrectTypeExpiration(uint expiration=ORDER_TIME_GTC)
  {
#ifdef __MQL5__
  const int expiration_mode=this.ExpirationModeFlags();
  if((expiration > ORDER_TIME_SPECIFIED_DAY) || (((expiration_mode >> expiration) & 1) == 0))
    {
      if((expiration < ORDER_TIME_SPECIFIED) || (expiration_mode < SYMBOL_EXPIRATION_SPECIFIED))
         expiration=ORDER_TIME_GTC;
      else if(expiration > ORDER_TIME_DAY)
         expiration=ORDER_TIME_SPECIFIED;

      uint i=1 << expiration;
      while((expiration <= ORDER_TIME_SPECIFIED_DAY) && ((expiration_mode & i) != i))
        {
         i <<= 1;
         expiration++;
        }
     }
#endif
  return (ENUM_ORDER_TYPE_TIME)expiration;
  }
//+------------------------------------------------------------------+
```

To avoid re-inventing the wheel, I used the method logic described by **fxsaber** Forum member. The code headers
feature links to the posts containing the codes.

Understanding all the intricacies of this logic is not the easiest experience, but knowing the person who published the functions as a trusted
developer, I decided that I could rely on him. Of course, it is possible to divide the entire logic of the methods into separate elements,
obtain extensive methods and describe their entire logic. But it is easier to implement descriptions of methods:

The methods receive the necessary filling policy and order expiration type. If a symbol supports this policy or type, it is returned. If
the necessary modes are not supported on a symbol, the allowed modes are returned. Thus, the methods always return supported modes — correct
filling policy or order expiration modes.

In the block with the methods of a simplified access to symbol object integer properties of the public section, **add declaration**
**of the method returning a normalized lot:**

```
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
```

At the very end of the class body, **add the method returning a trading**
**object belonging to a symbol object:**

```
   //--- The average weighted session price
   //--- setting the controlled session average weighted price (1) increase, (2) decrease and (3) control value
   //--- getting (4) the change value of the average weighted session price,
   //--- getting the flag of the average weighted session price change exceeding the (5) increase, (6) decrease value
   void              SetControlSessionPriceAWInc(const double value)             { this.SetControlledValueINC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWDec(const double value)             { this.SetControlledValueDEC(SYMBOL_PROP_SESSION_AW,::fabs(value));                 }
   void              SetControlSessionPriceAWLevel(const double value)           { this.SetControlledValueLEVEL(SYMBOL_PROP_SESSION_AW,::fabs(value));               }
   double            GetValueChangedSessionPriceAW(void)                   const { return this.GetPropDoubleChangedValue(SYMBOL_PROP_SESSION_AW);                    }
   bool              IsIncreasedSessionPriceAW(void)                       const { return (bool)this.GetPropDoubleFlagINC(SYMBOL_PROP_SESSION_AW);                   }
   bool              IsDecreasedSessionPriceAW(void)                       const { return (bool)this.GetPropDoubleFlagDEC(SYMBOL_PROP_SESSION_AW);                   }

//--- Return a trading object
   CTradeObj        *GetTradeObj(void)                                           { return &this.m_trade; }

  };
//+------------------------------------------------------------------+
```

Since a trading object is generated immediately when creating a symbol object and the trading object has starting initializing values of all of
its fields, it should be initialized using the necessary default values. To achieve this, at the very end of the CSymbol class constructor,

**call the Init() method of the trading object with the required default values:**

```

//--- Fill in the symbol current data
   for(int i=0;i<SYMBOL_PROP_INTEGER_TOTAL;i++)
      this.m_long_prop_event[i][3]=this.m_long_prop[i];
   for(int i=0;i<SYMBOL_PROP_DOUBLE_TOTAL;i++)
      this.m_double_prop_event[i][3]=this.m_double_prop[i];

//--- Update the base object data and search for changes
   CBaseObj::Refresh();
//---
   if(!select)
      this.RemoveFromMarketWatch();

//--- Initializing default values of a trading object
   this.m_trade.Init(this.Name(),0,this.LotsMin(),5,0,0,false,this.GetCorrectTypeFilling(),this.GetCorrectTypeExpiration(),LOG_LEVEL_ERROR_MSG);
  }
//+------------------------------------------------------------------+
```

When calling the method to the trading object, pass:

- a symbol name,
- minimum allowable lot for the symbol,
- the slippage of five points,
- the StopLoss price equal to zero — no StopLoss,
- TakeProfit price equal to zero — no TakeProfit,
- the flag of asynchronous sending of trading requests equal to false —
synchronous sending,
- receive the correct order filling policy right away and set it for the trading object,
- get the correct mode of order lifetime and set it for a trading object,
- set the logging level of trading methods to "errors only"

These values are set for the trading object by default but they can always be changed using Set methods described above for each of the
properties separately. Alternatively, you can leave the default values but pass another parameter to it when calling a trading method
so that it is used once when sending the request to the server.

**Implement the lot normalization method outside the class body:**

```
//+------------------------------------------------------------------+
//| Return a normalized lot considering symbol properties            |
//+------------------------------------------------------------------+
double CSymbol::NormalizedLot(const double volume) const
  {
   double ml=this.LotsMin();
   double mx=this.LotsMax();
   double ln=::NormalizeDouble(volume,this.DigitsLot());
   return(ln<ml ? ml : ln>mx ? mx : ln);
  }
//+------------------------------------------------------------------+
```

The method receives the lot value required for normalization. Next, get minimum and maximum lots allowed for a symbol, normalize the lot
value passed to the method and define the value to be returned by simply comparing the normalized value with the minimum and maximum lot. If
the lot passed to the method is less or more than the min/max symbol lot, return the min/max lot accordingly. Otherwise, return the
normalized lot considering the number of decimal places in the lot value (DigitsLot() method).

**We have improved the CSymbol class.**

Now we need to test the trading methods. Since we have no base trading class yet, we will temporarily add methods to the **CEngine** library
base object class to access a trading object of a necessary symbol. Since this is the object where we have full access to all important library
collections, this is where we are to place the methods for testing the trading object.

Note that the methods in the class are temporary. Later on, we will implement a full-fledged trading class where all the methods required
for checking values and trading are to be located.

**All methods that are currently necessary for testing the trading object**
**are added to the public section of the CEngine class:**

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

//--- Return a symbol trading object by (1) position, (2) order ticket
   CTradeObj           *GetTradeObjByPosition(const ulong ticket);
   CTradeObj           *GetTradeObjByOrder(const ulong ticket);

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

//--- Return event (1) milliseconds, (2) reason and (3) source from its 'long' value
   ushort               EventMSC(const long lparam)               const { return this.LongToUshortFromByte(lparam,0);         }
   ushort               EventReason(const long lparam)            const { return this.LongToUshortFromByte(lparam,1);         }
   ushort               EventSource(const long lparam)            const { return this.LongToUshortFromByte(lparam,2);         }

//--- Constructor/destructor
                        CEngine();
                       ~CEngine();
  };
//+------------------------------------------------------------------+
```

Implement declared methods outside the class body.

**The method for opening Buy position:**

```
//+------------------------------------------------------------------+
//| Open Buy position                                                |
//+------------------------------------------------------------------+
bool CEngine::OpenBuy(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.OpenPosition(POSITION_TYPE_BUY,volume,sl,tp,magic,trade_obj.GetDeviation(),comment);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a volume of an opened position (required),
- a symbol, on which a position should be opened (required),
- a magic number to be assigned to an open position (the default is 0),
- StopLoss (not set by default),
- TakeProfit (not set by default),
- a position comment (the default is program name+" by DoEasy")

Get a symbol object by a symbol name. If
failed to get the object, display the appropriate message and return false.

Get the trading object from a symbol object. If
failed to get the object, display the appropriate message and return false.

Return the operation result of the method for opening a trading object position
considered above.

**The method for opening a Sell position:**

```
//+------------------------------------------------------------------+
//| Open a Sell position                                             |
//+------------------------------------------------------------------+
bool CEngine::OpenSell(const double volume,const string symbol,const ulong magic=ULONG_MAX,double sl=0,double tp=0,const string comment=NULL)
  {
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.OpenPosition(POSITION_TYPE_SELL,volume,sl,tp,magic,trade_obj.GetDeviation(),comment);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a volume of an opened position (required),
- a symbol, on which a position should be opened (required),
- a magic number to be assigned to an open position (the default is 0),
- StopLoss (not set by default),
- TakeProfit (not set by default),
- a position comment (the default is program name+" by DoEasy")

Get a symbol object by a symbol name. If
failed to get the object, display the appropriate message and return false.

Get the trading object from a symbol object. If
failed to get the object, display the appropriate message and return false.

Return the operation result of the method for opening a trading object position
considered above.

**The method for modifying position's StopLoss and TakeProfit:**

```
//+------------------------------------------------------------------+
//| Modify a position                                                |
//+------------------------------------------------------------------+
bool CEngine::ModifyPosition(const ulong ticket,const double sl=WRONG_VALUE,const double tp=WRONG_VALUE)
  {
   CTradeObj *trade_obj=this.GetTradeObjByPosition(ticket);
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.ModifyPosition(ticket,sl,tp);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a modified position ticket (required),
- new StopLoss (the default is no changes),
- new TakeProfit (the default is no changes)

Get the trading object by a position ticket using the GetTradeObjByPosition()
method considered below.

If failed to get the object,
display the appropriate message and return

false.

Return the operation
result of the method for modifying a trading object position considered above.

**The method for closing a position in full:**

```
//+------------------------------------------------------------------+
//| Close a position in full                                         |
//+------------------------------------------------------------------+
bool CEngine::ClosePosition(const ulong ticket)
  {
   CTradeObj *trade_obj=this.GetTradeObjByPosition(ticket);
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.ClosePosition(ticket);
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a closed position.

Get the trading object by a position ticket using the GetTradeObjByPosition()
method considered below.

If failed to get the object,
display the appropriate message and return

false.

Return the operation
result of the method for closing a trading object position considered above.

**The method for position partial closing:**

```
//+------------------------------------------------------------------+
//| Close a position partially                                       |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionPartially(const ulong ticket,const double volume)
  {
   CTradeObj *trade_obj=this.GetTradeObjByPosition(ticket);
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   CSymbol *symbol=this.GetSymbolObjByName(trade_obj.GetSymbol());
   return trade_obj.ClosePositionPartially(ticket,symbol.NormalizedLot(volume));
  }
//+------------------------------------------------------------------+
```

The method receives a position ticket and a closed volume.

Get the trading object by a position ticket using the GetTradeObjByPosition()
method considered below.

If failed to get the object,
display the appropriate message and return

false.

Get a symbol object by
a trading object symbol name.

Return the operation result of the
method for partial closing of a trading object position considered above. The
method receives a normalized closed volume.

**The method of closing a position by an opposite one:**

```
//+------------------------------------------------------------------+
//| Close a position by an opposite one                              |
//+------------------------------------------------------------------+
bool CEngine::ClosePositionBy(const ulong ticket,const ulong ticket_by)
  {
   CTradeObj *trade_obj_pos=this.GetTradeObjByPosition(ticket);
   if(trade_obj_pos==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   CTradeObj *trade_obj_by=this.GetTradeObjByPosition(ticket_by);
   if(trade_obj_by==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj_pos.ClosePositionBy(ticket,ticket_by);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a closed position ticket,
- an opposite position ticket

Get the trading object by a position ticket using the
GetTradeObjByPosition() considered below.

If failed to
get the object, display the appropriate message and return false.

Get the trading object by an opposite position ticket using the
GetTradeObjByPosition() considered below.

If failed to
get the object, display the appropriate message and return false.

Return the operation result of the method for closing a trading object position by
an opposite one.

**The method for placing BuyStop pending order:**

```
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
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.SetOrder(ORDER_TYPE_BUY_STOP,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a volume of a placed order (required),
- order symbol (required),
- order price (required),
- StopLoss (not set by default),
- TakeProfit (not set by default),
- a magic number of a placed order (the default is 0),
- a placed order comment (the default is program name+" by DoEasy"),
- a placed order lifetime (the default is unlimited),
- a placed order lifetime type (the default is until explicit cancelation)

Get a symbol object by a symbol name. If
failed to get the object, display the appropriate message and return false.

Get the trading object from a symbol object. If
failed to get the object, display the appropriate message and return false.

Return the operation result of the pending order placing method of the trading
object considered above.

**The method for placing BuyLimit pending order:**

```
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
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.SetOrder(ORDER_TYPE_BUY_LIMIT,volume,price,sl,tp,0,magic,expiration,type_time,comment);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a volume of a placed order (required),
- order symbol (required),
- order price (required),
- StopLoss (not set by default),
- TakeProfit (not set by default),
- a magic number of a placed order (the default is 0),
- a placed order comment (the default is program name+" by DoEasy"),
- a placed order lifetime (the default is unlimited),
- a placed order lifetime type (the default is until explicit cancelation)

Get a symbol object by a symbol name. If
failed to get the object, display the appropriate message and return false.

Get the trading object from a symbol object. If
failed to get the object, display the appropriate message and return false.

Return the operation result of the pending order placing method of the trading
object considered above.

**The method for placing BuyStopLimit pending order:**

```
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
   #ifdef __MQL5__
      CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
      if(symbol_obj==NULL)
        {
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
         return false;
        }
      CTradeObj *trade_obj=symbol_obj.GetTradeObj();
      if(trade_obj==NULL)
        {
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
         return false;
        }
      return trade_obj.SetOrder(ORDER_TYPE_BUY_STOP_LIMIT,volume,price_stop,sl,tp,price_limit,magic,expiration,type_time,comment);
   //--- MQL4
   #else
      return true;
   #endif
  }
//+------------------------------------------------------------------+
```

The method receives:

- a volume of a placed order (required),
- order symbol (required),
- BuyStop order price (required),
- BuyLimit order price after BuyStop order activation (required),
- StopLoss (not set by default),
- TakeProfit (not set by default),
- a magic number of a placed order (the default is 0),
- a placed order comment (the default is program name+" by DoEasy"),
- a placed order lifetime (the default is unlimited),
- a placed order lifetime type (the default is until explicit cancelation)

For MQL5:

Get a symbol object by a symbol name. If
failed to get the object, display the appropriate message and return false.

Get the trading object from a symbol object. If
failed to get the object, display the appropriate message and return false.

Return the operation result of the pending order placing method of the trading
object considered above.

For MQL4:

Do nothing — return true.

**The methods for placing** **SellStop,** **SellLimit and SellStopLimit pending orders:**

```
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
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.SetOrder(ORDER_TYPE_SELL_STOP,volume,price,sl,tp,0,magic,expiration,type_time,comment);
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
   CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
   if(symbol_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
      return false;
     }
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.SetOrder(ORDER_TYPE_SELL_LIMIT,volume,price,sl,tp,0,magic,expiration,type_time,comment);
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
   #ifdef __MQL5__
      CSymbol *symbol_obj=this.GetSymbolObjByName(symbol);
      if(symbol_obj==NULL)
        {
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_NOT_SYMBOL_ON_LIST));
         return false;
        }
      CTradeObj *trade_obj=symbol_obj.GetTradeObj();
      if(trade_obj==NULL)
        {
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
         return false;
        }
      return trade_obj.SetOrder(ORDER_TYPE_SELL_STOP_LIMIT,volume,price_stop,sl,tp,price_limit,magic,expiration,type_time,comment);
   //--- MQL4
   #else
      return true;
   #endif
  }
//+------------------------------------------------------------------+
```

These methods are similar to the ones placing pending Buy orders.

**The method for modifying a pending order:**

```
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
   CTradeObj *trade_obj=this.GetTradeObjByOrder(ticket);
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.ModifyOrder(ticket,price,sl,tp,stoplimit,expiration,type_time);
  }
//+------------------------------------------------------------------+
```

The method receives:

- a modified order ticket (required),
- new pending order price (the default is no changes),
- new pending order StopLoss price (the default is no changes),
- new pending order TakeProfit price (the default is no changes),
- new pending order StopLimit price (the default is no changes),
- new time of a pending order expiration (the default is no changes),
- new mode of a pending order lifetime (the default is no changes)

Get a trading object by a modified order ticket using the
GetTradeObjByOrder() method considered below.

If failed
to get the object, display the appropriate message and return false.

Return the operation result of the pending order modification method of the
trading object considered above.

**The method for removing a pending order:**

```
//+------------------------------------------------------------------+
//| Remove a pending order                                           |
//+------------------------------------------------------------------+
bool CEngine::DeleteOrder(const ulong ticket)
  {
   CTradeObj *trade_obj=this.GetTradeObjByOrder(ticket);
   if(trade_obj==NULL)
     {
      //--- Error. Failed to get trading object
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   return trade_obj.DeleteOrder(ticket);
  }
//+------------------------------------------------------------------+
```

The method receives the ticket of a removed order.

Get a trading object by an order ticket using the
GetTradeObjByOrder() method considered below.

If
failed to get the object, display the appropriate message and return false.

Return the operation result of the pending order removal method of the trading
object considered above.

**The methods returning a symbol trading object by a position and**
**order ticket:**

```
//+------------------------------------------------------------------+
//| Return a symbol trading object by a position ticket              |
//+------------------------------------------------------------------+
CTradeObj *CEngine::GetTradeObjByPosition(const ulong ticket)
  {
   //--- Get the list of open positions
   CArrayObj *list=this.GetListMarketPosition();
   //--- If failed to get the list of open positions, display the message and return NULL
   if(list==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_MARKET_POS_LIST));
      return NULL;
     }
   //--- If the list is empty (no open positions), display the message and return NULL
   if(list.Total()==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_ENG_NO_OPEN_POSITIONS));
      return NULL;
     }
   //--- Sort the list by a ticket
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   //--- If failed to get the list of open positions, display the message and return NULL
   if(list==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_MARKET_POS_LIST));
      return NULL;
     }
   //--- If the list is empty (no required ticket), display the message and return NULL
   if(list.Total()==0)
     {
      //--- Error. No open position with #ticket
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_NO_OPEN_POSITION_WITH_TICKET),(string)ticket);
      return NULL;
     }
   //--- Get a position with #ticket from the obtained list
   COrder *pos=list.At(0);
   //--- If failed to get the position object, display the message and return NULL
   if(pos==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_POS_OBJ));
      return NULL;
     }
   //--- Get a symbol object by name
   CSymbol * symbol_obj=this.GetSymbolObjByName(pos.Symbol());
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Get and return the trading object from the symbol object
   CTradeObj *obj=symbol_obj.GetTradeObj();
   return obj;
  }
//+------------------------------------------------------------------+
//| Return a symbol trading object by an order ticket                |
//+------------------------------------------------------------------+
CTradeObj *CEngine::GetTradeObjByOrder(const ulong ticket)
  {
   //--- Get the list of placed orders
   CArrayObj *list=this.GetListMarketPendings();
   //--- If failed to get the list of placed orders, display the message and return NULL
   if(list==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_PENDING_ORD_LIST));
      return NULL;
     }
   //--- If the list is empty (no placed orders), display the message and return NULL
   if(list.Total()==0)
     {
      ::Print(DFUN,CMessage::Text(MSG_ENG_NO_PLACED_ORDERS));
      return NULL;
     }
   //--- Sort the list by a ticket
   list=CSelect::ByOrderProperty(list,ORDER_PROP_TICKET,ticket,EQUAL);
   //--- If failed to get the list of placed orders, display the message and return NULL
   if(list==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_ENG_FAILED_GET_PENDING_ORD_LIST));
      return NULL;
     }
   //--- If the list is empty (no required ticket), display the message and return NULL
   if(list.Total()==0)
     {
      //--- Error. No placed order with #ticket
      ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_NO_PLACED_ORDER_WITH_TICKET),(string)ticket);
      return NULL;
     }
   //--- Get an order with #ticket from the obtained list
   COrder *ord=list.At(0);
   //--- If failed to get an object order, display the message and return NULL
   if(ord==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return NULL;
     }
   //--- Get a symbol object by name
   CSymbol *symbol_obj=this.GetSymbolObjByName(ord.Symbol());
   //--- If failed to get the symbol object, display the message and return NULL
   if(symbol_obj==NULL)
     {
      ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return NULL;
     }
   //--- Get and return the trading object from the symbol object
   CTradeObj *obj=symbol_obj.GetTradeObj();
   return obj;
  }
//+--------------------------------------------------------------------+
```

Both methods are almost identical, except that in the first one,
we get the list of all open positions, while

in the second one, we get the list of all placed pending orders. The
remaining logic is completely identical for both methods and commented in the code comments.



**The methods for setting** **the**
**filling policy and valid filling policy in**
**trading objects of all symbols** **located in the symbol collection list or for a single specified symbol:**

```
//+------------------------------------------------------------------+
//| Set the valid filling policy                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeCorrectTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL)
  {
   //--- Declare the empty pointer to a symbol object
   CSymbol *symbol=NULL;
   //--- If a symbol name passed in the method inputs is not set, specify a filling policy for all symbols
   if(symbol_name==NULL)
     {
      //--- get the list of all used symbols
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      //--- In a loop by the list of symbol objects
      for(int i=0;i<total;i++)
        {
         //--- get the next symbol object
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- get a trading object from a symbol object
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         //--- set correct filling policy to the trading object (the default is "fill or kill")
         obj.SetTypeFilling(symbol.GetCorrectTypeFilling(type));
        }
     }
   //--- If a symbol name is specified in the method inputs, set the filling policy only for the specified symbol
   else
     {
      //--- Get a symbol object by a symbol name
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      //--- get a trading object from a symbol object
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      //--- set correct filling policy to the trading object (the default is "fill or kill")
      obj.SetTypeFilling(symbol.GetCorrectTypeFilling(type));
     }
  }
//+------------------------------------------------------------------+
//| Set the filling policy                                           |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeTypeFilling(const ENUM_ORDER_TYPE_FILLING type=ORDER_FILLING_FOK,const string symbol_name=NULL)
  {
   //--- Declare the empty pointer to a symbol object
   CSymbol *symbol=NULL;
   //--- If a symbol name passed in the method inputs is not set, specify a filling policy for all symbols
   if(symbol_name==NULL)
     {
      //--- get the list of all used symbols
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      //--- In a loop by the list of symbol objects
      for(int i=0;i<total;i++)
        {
         //--- get the next symbol object
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         //--- get a trading object from a symbol object
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         //--- for the trading object, set a filling policy passed to the method in the inputs (the default is "fill or kill")
         obj.SetTypeFilling(type);
        }
     }
   //--- If a symbol name is specified in the method inputs, set the filling policy only for the specified symbol
   else
     {
      //--- Get a symbol object by a symbol name
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      //--- get a trading object from a symbol object
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      //--- for the trading object, set a filling policy passed to the method in the inputs (the default is "fill or kill")
      obj.SetTypeFilling(type);
     }
  }
//+------------------------------------------------------------------+
```

The methods receive the filling policy (the default is "fill or kill") and a symbol (all symbols from the symbol collection are used by
default).

The methods logic is displayed in the code comments and is quite comprehensible. If you have any questions, feel free to ask them in the
comments below.

Other methods for setting default values for symbol trading objects have the same logic and feature no comments. In any case, you can
study the logic using these two methods.

**All the remaining methods for setting default values to symbol trading objects:**

```
//+------------------------------------------------------------------+
//| Set a correct order expiration type                              |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeCorrectTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetTypeExpiration(symbol.GetCorrectTypeExpiration(type));
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetTypeExpiration(symbol.GetCorrectTypeExpiration(type));
     }
  }
//+------------------------------------------------------------------+
//| Set an order expiration type                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeTypeExpiration(const ENUM_ORDER_TYPE_TIME type=ORDER_TIME_GTC,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetTypeExpiration(type);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetTypeExpiration(type);
     }
  }
//+------------------------------------------------------------------+
//| Set a magic number for trading objects of all symbols            |
//+------------------------------------------------------------------+
void CEngine::SetTradeMagic(const ulong magic,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetMagic(magic);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetMagic(magic);
     }
  }
//+------------------------------------------------------------------+
//| Set a comment for trading objects of all symbols                 |
//+------------------------------------------------------------------+
void CEngine::SetTradeComment(const string comment,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetComment(comment);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetComment(comment);
     }
  }
//+------------------------------------------------------------------+
//| Set a slippage                                                   |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeDeviation(const ulong deviation,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetDeviation(deviation);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetDeviation(deviation);
     }
  }
//+------------------------------------------------------------------+
//| Set a volume for trading objects of all symbols                  |
//+------------------------------------------------------------------+
void CEngine::SetTradeVolume(const double volume=0,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetVolume(volume!=0 ? symbol.NormalizedLot(volume) : symbol.LotsMin());
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetVolume(volume!=0 ? symbol.NormalizedLot(volume) : symbol.LotsMin());
     }
  }
//+------------------------------------------------------------------+
//| Set an order expiration date                                     |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeExpiration(const datetime expiration=0,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetExpiration(expiration);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetExpiration(expiration);
     }
  }
//+------------------------------------------------------------------+
//| Set the flag of asynchronous sending of trading requests         |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeAsyncMode(const bool mode=false,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetAsyncMode(mode);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetAsyncMode(mode);
     }
  }
//+------------------------------------------------------------------+
//| Set a logging level of trading requests                          |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CEngine::SetTradeLogLevel(const ENUM_LOG_LEVEL log_level=LOG_LEVEL_ERROR_MSG,const string symbol_name=NULL)
  {
   CSymbol *symbol=NULL;
   if(symbol_name==NULL)
     {
      CArrayObj *list=this.GetListAllUsedSymbols();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol=list.At(i);
         if(symbol==NULL)
            continue;
         CTradeObj *obj=symbol.GetTradeObj();
         if(obj==NULL)
            continue;
         obj.SetLogLevel(log_level);
        }
     }
   else
     {
      symbol=this.GetSymbolObjByName(symbol_name);
      if(symbol==NULL)
         return;
      CTradeObj *obj=symbol.GetTradeObj();
      if(obj==NULL)
         return;
      obj.SetLogLevel(log_level);
     }
  }
//+------------------------------------------------------------------+
```

**We have prepared all auxiliary temporary methods in the CEngine class for testing symbol trading objects.**

The existing cross-platform trading methods (albeit still in their infancy) allow us to avoid conditional compilation for MQL5 or MQL4
in the test EA. Now all trading functions of the test EA remain the same for any platform. Further on, we will improve working with the
library trading classes to obtain the entire functionality for the efficient work with our programs.



### Testing the base trading object

To test symbol trading objects, [we are going to use the test EA from the \\
previous article](https://www.mql5.com/en/articles/7195#node04) and adjust its trading functions for working with symbol trading objects. Keep in mind that we do not have any checks of
trading request values yet, but this allows us to test a response to invalid parameters. Such a response will be implemented later.

Save the EA in \\MQL5\\Experts\ **TestDoEasy\\Part21\** under the name **TestDoEasyPart21.mq5**.

**First, remove the inclusion of the standard library CTrade trading class**
**and declaration of a trading object of CTrade class type:**

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart20.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
#ifdef __MQL5__
#include <Trade\Trade.mqh>
#endif
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
input ulong             InpMagic             =  123;  // Magic number
input double            InpLots              =  0.1;  // Lots
input uint              InpStopLoss          =  50;   // StopLoss in points
input uint              InpTakeProfit        =  50;   // TakeProfit in points
input uint              InpDistance          =  50;   // Pending orders distance (points)
input uint              InpDistanceSL        =  50;   // StopLimit orders distance (points)
input uint              InpSlippage          =  0;    // Slippage in points
input double            InpWithdrawal        =  10;   // Withdrawal funds (in tester)
input uint              InpButtShiftX        =  40;   // Buttons X shift
input uint              InpButtShiftY        =  10;   // Buttons Y shift
input uint              InpTrailingStop      =  50;   // Trailing Stop (points)
input uint              InpTrailingStep      =  20;   // Trailing Step (points)
input uint              InpTrailingStart     =  0;    // Trailing Start (points)
input uint              InpStopLossModify    =  20;   // StopLoss for modification (points)
input uint              InpTakeProfitModify  =  60;   // TakeProfit for modification (points)
input ENUM_SYMBOLS_MODE InpModeUsedSymbols   =  SYMBOLS_MODE_CURRENT;   // Mode of used symbols list
input string            InpUsedSymbols       =  "EURUSD,AUDUSD,EURAUD,EURCAD,EURGBP,EURJPY,EURUSD,GBPUSD,NZDUSD,USDCAD,USDJPY";  // List of used symbols (comma - separator)

//--- global variables
CEngine        engine;
#ifdef __MQL5__
CTrade         trade;
#endif
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
int            used_symbols_mode;
string         used_symbols;
string         array_used_symbols[];
//+------------------------------------------------------------------+
```

**In the OnInit() handler, remove setting the parameters to the 'trade'**
**objects of the CTrade trading class:**

```
//--- Set trailing activation button status
   ButtonState(butt_data[TOTAL_BUTT-1].name,trailing_on);

//--- Set CTrade trading class parameters
#ifdef __MQL5__
   trade.SetDeviationInPoints(slippage);
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetMarginMode();
   trade.LogLevel(LOG_LEVEL_NO);
#endif
//--- Create and check the resource files
```

Then use the search ( **Ctrl+F**) to look for the " **trade**" string and replace calling the trading methods of the
standard library with the ones you need.

For example, replace this one:

```
                  COrder* position=list_positions.At(index);
                  if(position!=NULL)
                    {
                     //--- Get a ticket of a position with the highest profit and close the position by a ticket
                     #ifdef __MQL5__
                        trade.PositionClose(position.Ticket());
                     #else
                        PositionClose(position.Ticket(),position.Volume());
                     #endif
                    }
```

with this:

```
                  COrder* position=list_positions.At(index);
                  if(position!=NULL)
                    {
                     //--- Get a ticket of a position with the highest profit and close the position by a ticket
                     engine.ClosePosition(position.Ticket());
                    }
```

While finding the calls of the standard library trading methods, simply replace them with the calls of your methods.

**Let's consider the resulting panel buttons pressing handler. All calls of the new trading methods are highlighted**
**in color:**

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   string comment="";
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
         engine.OpenBuy(lot,Symbol(),magic_number,sl,tp);   // No comment - the default comment is to be set
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
         engine.PlaceBuyLimit(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
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
         engine.PlaceBuyStop(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
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
         engine.PlaceBuyStopLimit(lot,Symbol(),price_set_stop,price_set_limit,sl,tp,magic_number,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- Get the correct StopLoss and TakeProfit prices relative to StopLevel
         double sl=CorrectStopLoss(Symbol(),ORDER_TYPE_SELL,0,stoploss);
         double tp=CorrectTakeProfit(Symbol(),ORDER_TYPE_SELL,0,takeprofit);
         //--- Open Sell position
         engine.OpenSell(lot,Symbol(),magic_number,sl,tp);  // No comment - the default comment is to be set
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
         engine.PlaceSellLimit(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный SellLimit","Pending SellLimit order"));
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
         engine.PlaceSellStop(lot,Symbol(),price_set,sl,tp,magic_number,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
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
         engine.PlaceSellStopLimit(lot,Symbol(),price_set_stop,price_set_limit,sl,tp,magic_number,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
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
               //--- If this is a hedge account, close the half of the Buy position by the ticket
               if(engine.IsHedge())
                  engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
               //--- If this is a netting account, open a Sell position with the half of the Buy position volume
               else
                  engine.OpenSell(NormalizeLot(position.Symbol(),position.Volume()/2.0),Symbol(),magic_number,position.StopLoss(),position.TakeProfit(),"Частичное закрытие Buy #"+(string)position.Ticket());
              }
           }
        }
      //--- If the BUTT_CLOSE_BUY_BY_SELL button is pressed: Close Buy with the maximum profit by the opposite Sell with the maximum profit
      else if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL))
        {
         //--- In case of a hedging account
         if(engine.IsHedge())
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
         //--- Select only Sell positions from the list
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
               //--- If this is a hedge account, close the half of the Sell position by the ticket
               if(engine.IsHedge())
                  engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
               //--- If this is a netting account, open a Buy position with the half of the Sell position volume
               else
                  engine.OpenBuy(NormalizeLot(position.Symbol(),position.Volume()/2.0),Symbol(),position.Magic(),position.StopLoss(),position.TakeProfit(),"Partial closure Buy #"+(string)position.Ticket());
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
               engine.ClosePositionBy((ulong)position_sell.Ticket(),(ulong)position_buy.Ticket());
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
               engine.ClosePosition((ulong)position.Ticket());
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
            list.Sort(SORT_BY_ORDER_TIME_OPEN);
            int total=list.Total();
            //--- In the loop from the position with the most amount of time
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

We are not going to consider other improved EA functions that call CTrade trading class methods here. You can find all the data in the files
attached below.

Now we will simply compile the EA and launch it in the tester.

**Click various panel buttons and make sure that the trading**
**objects are operable:**

![](https://c.mql5.com/2/37/ZQbQi1Hh68.gif)

Our first symbol trading objects are working as intended.

Multiple improvements are yet to be implemented to make working
with them efficient and convenient.

### What's next?

Our next objective is the development of a full-fledged class to be used when accessing symbol trading objects.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7229#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7229](https://www.mql5.com/ru/articles/7229)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7229.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7229/mql5.zip "Download MQL5.zip")(3567.28 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7229/mql4.zip "Download MQL4.zip")(3567.28 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/328634)**
(26)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Oct 2020 at 01:35

Problem: the **CEvent::GetPendReqID()** method I need above is **protected**!! Any better ideas without me having to changw the DoEasy [source code](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development")? In my humble opinion, these methods should be **public** ;-)

```
class CEvent : public CObject
  {
//...
protected:
//--- Return (1) the specified magic number, the ID of (2) the first group, (3) second group, (4) pending request from the magic number value
   ushort            GetMagicID(void)                          const { return ushort(this.Magic() & 0xFFFF);                                 }
   uchar             GetGroupID1(void)                         const { return uchar(this.Magic()>>16) & 0x0F;                                }
   uchar             GetGroupID2(void)                         const { return uchar((this.Magic()>>16) & 0xF0)>>4;                           }
   uchar             GetPendReqID(void)                        const { return uchar(this.Magic()>>24) & 0xFF;                                }
//...
};
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
8 Oct 2020 at 07:31

**Dima Diall :**

Problem: the **CEvent::GetPendReqID()** method I need above is **protected** !! Any better ideas without me having to changw the DoEasy source code? In my humble opinion, these methods should be **public** ;-)

You need to watch the CEngine class - only it gives access to the library to user programs.

All other classes are for the needs of the library, and are not intended for users, with the exception of the library service [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") that are available in the program after connecting the library to it.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Oct 2020 at 16:22

**Artyom Trishkin:**

You need to watch the CEngine class - only it gives access to the library to user programs.

All other classes are for the needs of the library, and are not intended for users, with the exception of the library service [functions](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined Macrosubstitutions") that are available in the program after connecting the library to it.

Can you please give me an example? I am looking at **CEngine** class and see that it is possible to extract a list of events, check their type etc... so I can access each event but don't find any obvious way to read specific event details packed in the magic number (groups & request ID) from the CEngine object \-\- as I see it, I still need to read this information directly from the **CEvent** object instances as in my event handler example above in my robot's even-handler method, i.e. CEvent:GetPendReq()

```
class CEngine
  {
//...
//...
//--- Return (1) the list of order, deal and position events, (2) base trading event object by index and the (3) number of new trading events
   CArrayObj           *GetListAllOrdersEvents(void)                    { return this.m_events.GetList();                     }
   CEventBaseObj       *GetTradeEventByIndex(const int index)           { return this.m_events.GetTradeEventByIndex(index);   }
   int                  GetTradeEventsTotal(void)                 const { return this.m_events.GetTradeEventsTotal();         }
//--- Reset the last trading event
   void                 ResetLastTradeEvent(void)                       { this.m_events.ResetLastTradeEvent();                }
//--- Return the (1) last trading event, (2) the last event in the account properties and (3) the last event in symbol properties
   ENUM_TRADE_EVENT     LastTradeEvent(void)                      const { return this.m_last_trade_event;                     }
   int                  LastAccountEvent(void)                    const { return this.m_last_account_event;                   }
   int                  LastSymbolsEvent(void)                    const { return this.m_last_symbol_event;                    }
//--- Return the (1) hedge account, (2) working in the tester, (3) account event, (4) symbol event and (5) trading event flag
   bool                 IsHedge(void)                             const { return this.m_is_hedge;                             }
   bool                 IsTester(void)                            const { return this.m_is_tester;                            }
   bool                 IsAccountsEvent(void)                     const { return this.m_accounts.IsEvent();                   }
   bool                 IsSymbolsEvent(void)                      const { return this.m_symbols.IsEvent();                    }
   bool                 IsTradeEvent(void)                        const { return this.m_events.IsEvent();                     }
//...
//...
  };
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
9 Oct 2020 at 12:13

**Dima Diall :**

Не могли бы вы привести мне пример? Я смотрю на класс **CEngine** и вижу, что можно извлечь список событий, проверить их тип и т. Д., Поэтому я могу получить доступ к каждому событию, но не нахожу очевидного способа прочитать конкретные детали события, упакованные в магическое число (группы и идентификатор запроса) из объекта CEngine  \- как я вижу, мне все еще нужно читать эту информацию непосредственно из **экземпляров** объекта **CEvent,** как в моем примере обработчика событий выше в методе обработчика четных событий моего робота, то есть CEvent ::GetPendReqID ()

Wait a little, please. The next article in the ru-segment will be about advisors, and there I will try to explain.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
10 Oct 2020 at 01:06

**Artyom Trishkin:**

Wait a little, please. The next article in the ru-segment will be about advisors, and there I will try to explain.

OK, cool - thank you...

![Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://c.mql5.com/2/37/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part XXII): Trading classes - Base trading class, verification of limitations](https://www.mql5.com/en/articles/7258)

In this article, we will start the development of the library base trading class and add the initial verification of permissions to conduct trading operations to its first version. Besides, we will slightly expand the features and content of the base trading class.

![Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://c.mql5.com/2/37/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part XX): Creating and storing program resources](https://www.mql5.com/en/articles/7195)

The article deals with storing data in the program's source code and creating audio and graphical files out of them. When developing an application, we often need audio and images. The MQL language features several methods of using such data.

![Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__5.png)[Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://www.mql5.com/en/articles/7286)

In the article, we continue the development of the trading class by implementing the control over incorrect trading order parameter values and voicing trading events.

![Building an Expert Advisor using separate modules](https://c.mql5.com/2/37/mql5_avatar_adviser_modules.png)[Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)

When developing indicators, Expert Advisors and scripts, developers often need to create various pieces of code, which are not directly related to the trading strategy. In this article, we consider a way to create Expert Advisors using earlier created blocks, such as trailing, filtering and scheduling code, among others. We will see the benefits of this programming approach.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/7229&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070453268326323739)

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