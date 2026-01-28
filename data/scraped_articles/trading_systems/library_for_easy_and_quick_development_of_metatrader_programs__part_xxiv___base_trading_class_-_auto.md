---
title: Library for easy and quick development of MetaTrader programs (part XXIV): Base trading class - auto correction of invalid parameters
url: https://www.mql5.com/en/articles/7326
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:38:19.323319
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hvltcxhpwaxjgffelalolxgnpucuizha&ssn=1769186297385414315&ssn_dr=0&ssn_sr=0&fv_date=1769186297&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7326&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXIV)%3A%20Base%20trading%20class%20-%20auto%20correction%20of%20invalid%20parameters%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691862972402683&fz_uniq=5070446173040350712&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7326#node01)
- [Correcting trading event definitions](https://www.mql5.com/en/articles/7326#node02)
- [Handler of trade request parameter errors](https://www.mql5.com/en/articles/7326#node03)
- [Testing](https://www.mql5.com/en/articles/7326#node04)
- [What's next?](https://www.mql5.com/en/articles/7326#node05)

[In the previous article](https://www.mql5.com/en/articles/7286#node02), we added control over invalid parameters to
the trading class. The validity of values passed to the trading methods is checked. If any of the parameters turns out to be invalid, the
return from the trading method is performed accompanied by the error message. This behavior does not allow the EA to overload a trade server
with deliberately invalid orders. On the other hand, it does not give full control over the EA behavior. Instead, we may check if invalid
values can be corrected. If yes, then it would be reasonable to correct them and send the adjusted trading order to the server.

Generally, the EA should be able to act according to the circumstances while following the user-defined logic of handling errors in trading orders.
Thus, we may give the following instructions to the EA when a trading order error is detected:

1. Simply exit the trading method letting the user create the handler of invalid parameters of an erroneous order on their own.
2. If an invalid value of a trading order can be fixed, do that right away and send it to the server,

3. If a situation and an error are appropriate, repeat a trading request after a pause or simply repeat the request with the same
    parameters.

Handling errors in trading order parameters may lead to one of several outcomes:

- Inability to continue trading using the EA before an error source is eliminated by the user.

- Inability to send a trading order — exit from the trading method.
- Correcting invalid values and sending a fixed trading order.
- Immediate sending of a trading order with the initial parameters (here the assumption is made that trading conditions have improved).
- Waiting, updating quote data and sending a trading order with the initial parameters.

In this article, we are going to develop the trading order error handler that will check errors and their sources, as well as return the error
handling method:

- disabling trading operations,

- interrupting a trading operation,
- correcting invalid parameters,
- trading request with initial parameters,
- trading request after waiting (temporary solution),
- creating a pending trading request (in subsequent articles)


### Concept

Disabling trading operations is necessary when trading is disabled on the server either completely, or for EAs only making trading requests
useless. In this case, the EA can only be used as an analytical assistant. To achieve this, we need to have the global flag that is to set during
the first trading attempt and determining the impossibility of trading.

Interrupting a trading operation: In case of any error, exit the trading method providing the user the ability to continue trading attempts with the
same parameters.

Correcting invalid parameters works as follows: when checking the trading request validity, we compile a list of all detected errors. The method of
checking parameters looks through all errors from the list and returns the trading method behavior code. If an error makes further trading
impossible, the method returns the code of exiting the trading method since sending a trading order will still yield no positive result. If
the error can be fixed, the methods of correcting the appropriate trading order values are called and the result of the successful
verification is returned. Besides, the method returns the "Wait and repeat", "Update data and repeat" and "Create a pending request"
trading method behavior codes.

What does this mean?

The "Wait and repeat" behavior may be necessary when the market is close to one of the order stop levels or its activation price, while we try to
change stop levels or remove an order/close a position. If the stop level activation price is within the trading operations freeze area, the
server returns the ban on changing the order values. In this case, there is only one solution — simply wait a while hoping that the price leaves
the area. After that, send a trading request to change order/position parameters or remove an order after waiting.

The "Update data and
repeat" behavior may be necessary if the prices have become outdated and we have received a requote during the processing of a trading order.

"Create a pending request" behavior. What does this mean?

If we have a close a look at the previous two handling methods, it
becomes clear that, while waiting, we simply wait in the trading method till the waiting time is over. Such a behavior is justified if we do not
have to analyze the trading environment while waiting. To free the program from the necessity to "stand" inside the trading method, we
simply create a pending trading request containing the necessary parameters and waiting time with the number of repeats.

Thus, creating a
pending request fully eliminates the need in the "Update data and repeat" and "Wait and repeat" behaviors. These two behaviors are
essentially pending trading requests (with the minimum and specified waiting times). Besides, the ability to make pending requests in the
program, provides users with yet another method of conducting trading operations. We will leave implementation of pending requests for
subsequent articles.

Before we start, I want to remind you that we started making changes in the trading event definition [in \\
the previous article](https://www.mql5.com/en/articles/7286):

I have received several user reports on the error detected when receiving the last [trading \\
event](https://www.mql5.com/en/articles/5724). The test EA, that relates to the articles describing how to receive trading events, obtains data on the occurred trading
event by comparing the previous event value with the current one. This would be sufficient for the purpose of testing the tracking of
trading events by the library since I did not intend to use the unfinished library version in custom applications when writing articles
on trading events. But it turned out that obtaining info on trading events is in high demand and it is important to know the exact last
event that occurred.

The implemented method of obtaining a trading event may skip some events. For example, if you set a pending order two times in a row, the
second one is not tracked in the program (the library tracks all events) since it matches the penultimate one ("placing a pending
order") though the orders themselves may actually differ.

Therefore, we will fix this behavior. Today we will implement a simple flag informing the program of an event, so that we are able to view data on the
event in the program. In the next article, we will complete obtaining trading events in the program by creating a full list of all events
occurred simultaneously and sending it to the program. Thus, we will be able not only to find out about an occurred trading event but also
view all events occurred simultaneously as it is done for [account](https://www.mql5.com/en/articles/6995) and [symbol \\
collection events](https://www.mql5.com/en/articles/7071).

So, let's complete the work on changing this functionality before resuming our work on the trading class.

### Correcting trading event definitions

Since all our objects are actually based on the [base object of all \\
library objects](https://www.mql5.com/en/articles/7071#node01) featuring the list of events and the method returning the object's occurred event flag, we add all trading events to the
base object event list, while the event flag can be obtained from the class using the **IsEvent()** method. Event flags are
automatically set by the class. However, we should be able to set the flag of an occurred trading event from other classes and their event
handlers.

To do this, **add**
**the method of setting the base object event flag to the CEventBaseObj class in the BaseObj.mqh file:**

```
//--- Set/return the occurred event flag to the object data
   void              SetEvent(const bool flag)                       { this.m_is_event=flag;                   }
   bool              IsEvent(void)                             const { return this.m_is_event;                 }
```

Now when a new event appears in the **CEventsCollection** trading event collection class, we need to create the event
description, place it to the list of new events of the base class of all objects and set the new event flag.

Thus, descriptions of all newly occurred events are placed to the list of symbol collection's base class trading events. From that list, we can
easily read the list in the program and handle each event in it.

Let's make all the necessary improvements to the **EventsCollection.mqh** trading events class file.

**Add the definition of the two new methods to the public section of the class**— **the method of receiving the base event object by its index in the list and**

**the**
**method returning the number of new events:**

```
//--- Return (1) the last trading event on an account, (2) base event object by index and (3) number of new events
   ENUM_TRADE_EVENT  GetLastTradeEvent(void)                const { return this.m_trade_event;                 }
   CEventBaseObj    *GetTradeEventByIndex(const int index)        { return this.GetEvent(index,false);         }
   int               GetTradeEventsTotal(void)              const { return this.m_list_events.Total();         }
//--- Reset the last trading event
```

The method, returning the base event object by index, calls the **GetEvent()** base object method featuring the
required event index and the reset flag (false)
checking the index going beyond the event list in order not to correct the returned event if the index goes beyond the event list. In other words, if we
pass a non-existent index, the method returns NULL. If we passed true
to the flag value, the method would return the last event we do not need here.

The method returning the number of new events simply returns
the size of the base object list.

Since the lists of historical and market orders and positions are constantly viewed in the timer's trading event collection class, **we need to clear**
**the list of base trading events and set the sorted list flag**
**in the Refresh() method:**

```
//+------------------------------------------------------------------+
//| Update the event list                                            |
//+------------------------------------------------------------------+
void CEventsCollection::Refresh(CArrayObj* list_history,
                                CArrayObj* list_market,
                                CArrayObj* list_changes,
                                CArrayObj* list_control,
                                const bool is_history_event,
                                const bool is_market_event,
                                const int  new_history_orders,
                                const int  new_market_pendings,
                                const int  new_market_positions,
                                const int  new_deals,
                                const double changed_volume)
  {
//--- Exit if the lists are empty
   if(list_history==NULL || list_market==NULL)
      return;
//---
   this.m_is_event=false;
   this.m_list_events.Clear();
   this.m_list_events.Sort();
//--- If the event is in the market environment
   if(is_market_event)
     {
```

**After the event sending string in all methods of creating the**
**new CreateNewEvent() event, we need to add the event to the list of base events:**

```
         event.SendEvent();
         CBaseObj::EventAdd(this.m_trade_event,order.Ticket(),order.Price(),order.Symbol());
```

This has already been set in the method listings, so there is no point in dwelling on it here to save the article space. All can be found in the
attached files.

Now, in the public section of the class of the **CEngine** library base object, add the method
returning the base event object by its index in the list and the method
returning the number of new events:

```
//--- Return (1) the list of order, deal and position events, (2) base trading event object by index and the (3) number of new trading events
   CArrayObj           *GetListAllOrdersEvents(void)                    { return this.m_events.GetList();                     }
   CEventBaseObj       *GetTradeEventByIndex(const int index)           { return this.m_events.GetTradeEventByIndex(index);   }
   int                  GetTradeEventsTotal(void)                 const { return this.m_events.GetTradeEventsTotal();         }
//--- Reset the last trading event
```

These methods simply call same-name trading events collection class methods described above.

These are all the necessary changes allowing you to track any events occurred simultaneously and sent to the program in one bundle. This will be
seen later — when testing the functionality described in the article.

**Now we can start further refinement of the trading class.**

### Handler of trade request parameter errors

First, **add the indices of necessary messages to the Datas.mqh file:**

```
   MSG_LIB_TEXT_REQUEST_REJECTED_DUE,                 // Request was rejected before sending to the server due to:
   MSG_LIB_TEXT_INVALID_REQUEST,                      // Invalid request:
   MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR,                 // Insufficient funds for performing a trade

   MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ,        // Unsupported price parameter type in a request
   MSG_LIB_TEXT_TRADING_DISABLE,                      // Trading disabled for the EA until the reason is eliminated
   MSG_LIB_TEXT_TRADING_OPERATION_ABORTED,            // Trading operation is interrupted
   MSG_LIB_TEXT_CORRECTED_TRADE_REQUEST,              // Correcting trading request parameters
   MSG_LIB_TEXT_CREATE_PENDING_REQUEST,               // Creating a pending request
   MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT,          // Unable to correct a lot

  };
```

**and the texts corresponding to the indices:**

```
   {"Запрос отклонён до отправки на сервер по причине:","Request rejected before being sent to server due to:"},
   {"Ошибочный запрос:","Invalid request:"},
   {"Недостаточно средств для совершения торговой операции","Not enough money to perform trading operation"},

   {"Неподдерживаемый тип параметра цены в запросе","Unsupported price parameter type in request"},
   {"Торговля отключена для эксперта до устранения причины запрета","Trading for the expert is disabled until this ban is eliminated"},
   {"Торговая операция прервана","Trading operation aborted"},
   {"Корректировка параметров торгового запроса ...","Correction of trade request parameters ..."},
   {"Создание отложенного запроса","Create pending request"},
   {"Нет возможности скорректировать лот","Unable to correct the lot"},

  };
```

In the **Defines.mqh** file, add enumerations we need to define and return the ways of handling errors in trading requests and
errors returned by the trade server.

To set a behavior activated when receiving an error directly for the EA, **add the enumeration describing the possible EA**
**behavior when detecting an error in the trading request or when an error is returned by the trade server:**

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

It will be possible to set the EA preferable behavior when handling errors in the EA settings by specifying one of the enumeration
parameters.

Various error handling methods are possible when checking the trading request parameter values. To find out what errors are detected when
checking trading order parameters and what trading conditions affect these errors, **add the enumeration with flags of possible error**
**handling methods:**

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
  };
//+------------------------------------------------------------------+
```

If we check trading order parameters and the ability to execute it, we add the error handling behavior flags:

- 0 — no error, the trading order can be sent,
- 1 — critical error — there is no point in trading attempts, the EA should be switched to the non-trading analytical assistant mode,
- 2 — something went wrong, and there was a failure in the library - just interrupt further execution of the trading method to avoid the
malfunctioning of the trading class,
- 4 — the error can be fixed, and it is written to the error list to call the method for fixing them.

The error-checking method will return the ways to correctly handle detected errors.

To do this, **add the**
**enumeration of possible methods of handling trading order errors, as well as the ones returned by the trade server:**

```
//+------------------------------------------------------------------+
//| The methods of handling errors and server return codes           |
//+------------------------------------------------------------------+
enum ENUM_ERROR_CODE_PROCESSING_METHOD
  {
   ERROR_CODE_PROCESSING_METHOD_OK,                         // No errors
   ERROR_CODE_PROCESSING_METHOD_DISABLE,                    // Disable trading for the EA
   ERROR_CODE_PROCESSING_METHOD_EXIT,                       // Exit the trading method
   ERROR_CODE_PROCESSING_METHOD_REFRESH,                    // Update data and repeat
   ERROR_CODE_PROCESSING_METHOD_WAIT,                       // Wait and repeat
   ERROR_CODE_PROCESSING_METHOD_PENDING,                    // Create a pending request
  };
//+------------------------------------------------------------------+
```

The methods of handling errors can be comprehended from the descriptions of the enumeration constants.

Also, improve the base [trading object](https://www.mql5.com/en/articles/7229).

Depending on the method of constructing bars on the chart, trading is performed either by Ask and Bid prices or by Ask and Last ones. Currently,
only trading by Ask and Bid prices is arranged in the base trading class. Let's add the ability to check the prices used to construct the
chart. Also, adjust the prices we are going to use to trade. Besides, MQL5 features the structure of the [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult)
trading request result, as well as 'retcode' and 'comment' fields containing the error code and the error code description, respectively. This
allows checking the codes returned by the trade server after sending a trading order to the server. MQL4 has no such feature, so the error
code should be read by the GetLastError() function returning the last error code. Since our library is a multi-platform one, in case of
MQL4 we need to fill in the fields of the trade request structure after sending it to the server.

When checking the stop order distance
relative to the price, we also consider the distances of minimum acceptable stop levels (StopLevel) set for a symbol. If StopLevel
value returned by the [SymbolInfoInteger()](https://www.mql5.com/en/docs/marketinformation/symbolinfointeger) function
with the [SYMBOL\_TRADE\_STOPS\_LEVEL](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer)
property ID is equal to zero, this does not mean the absence of the minimum shift in points of the price for stop orders. It only means the level is
floating. Thus, in order to correct the value of the stop level shift from the price, we need to select the level "in situ" or use the
current spread multiplied by a certain value as the shift value. The double spread is usually used for a smooth stop levels adjustment.
Add the multiplier, as well as its return and setting methods, to the trading object to be able to set the multiplier to each symbol's
trading object.

Add the necessary changes to the **CTradeObj** trading object base class in the **TradeObj.mqh** file.

**In the private section of the class, declare two class member variables to**
**store the price type for constructing bars and the spread**
**multiplier to adjust stop order levels:**

```
   SActions                   m_datas;
   MqlTick                    m_tick;                                            // Tick structure for receiving prices
   MqlTradeRequest            m_request;                                         // Trade request structure
   MqlTradeResult             m_result;                                          // trade request execution result
   ENUM_SYMBOL_CHART_MODE     m_chart_mode;                                      // Price type for constructing bars
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
   bool                       m_use_sound;                                       // The flag of using sounds of the object trading events
   uint                       m_multiplier;                                      // The spread multiplier to adjust levels relative to StopLevel

```

**In the public section of the class, add the method setting the spread**
**multiplier and the method returning the multiplier value:**

```
public:
//--- Constructor
                              CTradeObj();

//--- Set/return the spread multiplier
   void                       SetSpreadMultiplier(const uint value)        { this.m_multiplier=(value==0 ? 1 : value);  }
   uint                       SpreadMultiplier(void)                 const { return this.m_multiplier;                  }
//--- Set default values
```

When setting the spread multiplier, check if the value passed to the method is equal to zero. If yes, assign the value of 1.

Also, **in the public section of the class, add two methods — the**
**one setting the trading request error code and the one setting the**
**trading request error code description:**

```
//--- Set the error code in the last request result
   void                       SetResultRetcode(const uint retcode)                     { this.m_result.retcode=retcode;       }
   void                       SetResultComment(const string comment)                   { this.m_result.comment=comment;       }
//--- Data on the last request result:
```

**In the class constructor, assign the default value of 1 to the spread**
**multiplier:**

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
   //--- Spread multiplier
   this.m_multiplier=1;
   //--- Set default sounds and flags of using sounds
   this.m_use_sound=false;
   this.InitSounds();
  }
//+------------------------------------------------------------------+
```

In the **Init()** method defining the default values of the trading object parameters, **set the m\_chart\_mode variable**
**value storing the bar construction prices:**

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
   this.m_chart_mode=#ifdef __MQL5__ (ENUM_SYMBOL_CHART_MODE)::SymbolInfoInteger(this.m_symbol,SYMBOL_CHART_MODE) #else SYMBOL_CHART_MODE_BID #endif ;
  }
//+------------------------------------------------------------------+
```

Here for MQL5, we
obtain data using the SymbolInfoInteger() function with the [SYMBOL\_CHART\_MODE](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_chart_mode)
ID, while in case of MQL4, write
that the bars are constructed by Bid price.

Now we should add filling the trade server return structure to each trading method.

**Let's use the position opening**
**method as an example:**

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
                             const ulong deviation=ULONG_MAX)
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
   this.m_request.type     =  OrderTypeByPositionType(type);
   this.m_request.price    =  (type==POSITION_TYPE_BUY ? this.m_tick.ask : (this.m_chart_mode==SYMBOL_CHART_MODE_BID ? this.m_tick.bid : this.m_tick.last));
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
   if(ticket!=WRONG_VALUE)
     {
      ::SymbolInfoTick(this.m_symbol,this.m_tick);
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
      ::SymbolInfoTick(this.m_symbol,this.m_tick);
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

Here for MQL5, we return
the result of the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function operation as before,
while in case of MQL4, we check the ticket
number returned by the [MQL4 \\
order sending function](https://docs.mql4.com/en/trading/ordersend "https://docs.mql4.com/en/trading/ordersend"). If a trading request is executed successfully, the function returns the open order ticket. The error
brings WRONG\_VALUE. Therefore, make sure the function returns the value other than
-1. If yes, update symbol quotes, fill in the trading request result structure using the appropriate data and return true
— the function is executed successfully.

If the order
sending function returns -1, write the last error code, the
current prices and the last error code definition to the
trading request result structure. The remaining structure fields are left equal to zero.
As a result, return false —
trading order sending error.

Thanks to this refinement, we can see the request result using the class methods regardless of the outcome of sending the trading order:

```
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
```

The rest of the trading methods are finalized in a similar way, and there is no point in considering them here. You can find all you need in the
files attached below.

In the **CAccount** account object class of the **Account.mqh** file, **improve**
**the method returning the margin required for opening a position or setting a pending order:**

```
//+------------------------------------------------------------------+
//| Return the margin required for opening a position                |
//| or placing a pending order                                       |
//+------------------------------------------------------------------+
double CAccount::MarginForAction(const ENUM_ORDER_TYPE action,const string symbol,const double volume,const double price) const
  {
   double margin=EMPTY_VALUE;
   #ifdef __MQL5__
      return(!::OrderCalcMargin(ENUM_ORDER_TYPE(action%2),symbol,volume,price,margin) ? EMPTY_VALUE : margin);
   #else
      return this.MarginFree()-::AccountFreeMarginCheck(symbol,ENUM_ORDER_TYPE(action%2),volume);
   #endif
  }
//+------------------------------------------------------------------+
```

All we need to add here is a conversion of the order type passed to the method into two possible values — ORDER\_TYPE\_BUY or ORDER\_TYPE\_SELL,
since MQL5 and MQL4 functions the method works with require only this type of order.

As you may remember, the
remainder of dividing the order type constant by 2 always returns one of the two values:

- either 0 (ORDER\_TYPE\_BUY),
- or 1 (ORDER\_TYPE\_SELL).


This is exactly what we need to make a conversion into the correct order type.

We have already created the custom structure for filling in the trading order price parameters in the **CTrading** class
of the **Trading.mqh** file:

```
   struct SDataPrices
     {
      double            open;                // Open price
      double            limit;               // Limit order price
      double            sl;                  // StopLoss price
      double            tp;                  // TakeProfit price
     };
   SDataPrices          m_req_price;         // Trade request prices
```

However, MQL features the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)
structure. So, to avoid the redundant structure,

**replace the custom**
**structure with the standard one in the private section of the class** **, as well as**

**declare**
**the class member variable for storing error source flags in the trading request and**


**the variable for storing the EA behavior in case of errors occurring when sending trading orders:**

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
   CArrayInt            m_list_errors;                   // Error list
   bool                 m_is_trade_disable;              // Flag disabling trading
   bool                 m_use_sound;                     // The flag of using sounds of the object trading events
   ENUM_LOG_LEVEL       m_log_level;                     // Logging level
   MqlTradeRequest      m_request;                       // Trade request prices
   ENUM_TRADE_REQUEST_ERR_FLAGS m_error_reason_flags;    // Flags of error source in a trading method
   ENUM_ERROR_HANDLING_BEHAVIOR m_err_handling_behavior; // Behavior when handling error
//--- Add the error code to the list
   bool                 AddErrorCodeToList(const int error_code);
```

Also, in the private class section, write the method returning the flag
within the variable storing error sources flags,

the method
returning the error code presence in the error list, as well as

the
methods of placing and returning actions when handling errors:

```
//--- Return the flag presence in the trading event error reason
   bool                 IsPresentErrorFlag(const int code)     const { return (this.m_error_reason_flags & code)==code;                               }
//--- Return the error code in the list
   bool                 IsPresentErorCode(const int code)            { this.m_list_errors.Sort(); return this.m_list_errors.Search(code)>WRONG_VALUE; }
//--- Set/return the error handling action
   void                 SetErrorHandlingBehavior(const ENUM_ERROR_HANDLING_BEHAVIOR behavior) { this.m_err_handling_behavior=behavior;                }
   ENUM_ERROR_HANDLING_BEHAVIOR  ErrorHandlingBehavior(void)   const { return this.m_err_handling_behavior;                                           }

//--- Check trading limitations
```

**Remove the code for displaying the funds insufficiency message**
**in the journal from the method checking the funds sufficiency:**

```
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
            ::Print(source_method,CMessage::Text(MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR),": ",message);
         this.AddErrorCodeToList(MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR);
        }
      return false;
     }
```

Now the funds insufficiency message will be displayed from another method.

In the current method, simply **add**
**the flag instructing to search for the error in the error list and add the**
**error code to the error list:**

```
//+------------------------------------------------------------------+
//| Check if the funds are sufficient                                |
//+------------------------------------------------------------------+
bool CTrading::CheckMoneyFree(const double volume,
                              const double price,
                              const ENUM_ORDER_TYPE order_type,
                              const CSymbol *symbol_obj,
                              const string source_method,
                              const bool mess=true)
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
      #else/*__MQL4__*/::AccountFreeMarginCheck(symbol_obj.Name(),action,volume) #endif
     );
   //--- If no free funds are left, inform of that and return 'false'
   if(money_free<=0 #ifdef __MQL4__ || ::GetLastError()==134 #endif )
     {
      this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR);
      return false;
     }
   //--- Verification successful
   return true;
  }
//+------------------------------------------------------------------+
```

Declare the methods correcting the stop and pending order prices, the method
correcting the volume in the trading order, the method specifying how to
handle the error and the method correcting trading order errors:

```
//--- Return the correct (1) StopLoss, (2) TakeProfit and (3) order placement price
   double               CorrectStopLoss(const ENUM_ORDER_TYPE order_type,
                                       const double price_set,
                                       const double stop_loss,
                                       const CSymbol *symbol_obj,
                                       const uint spread_multiplier=1);
   double               CorrectTakeProfit(const ENUM_ORDER_TYPE order_type,
                                       const double price_set,
                                       const double take_profit,
                                       const CSymbol *symbol_obj,
                                       const uint spread_multiplier=1);
   double               CorrectPricePending(const ENUM_ORDER_TYPE order_type,
                                       const double price_set,
                                       const double price,
                                       const CSymbol *symbol_obj,
                                       const uint spread_multiplier=1);
//--- Return the volume, at which it is possible to open a position
   double               CorrectVolume(const double price,
                                       const ENUM_ORDER_TYPE order_type,
                                       const CSymbol *symbol_obj,
                                       const string source_method);

//--- Return the error handling method
   ENUM_ERROR_CODE_PROCESSING_METHOD   ResultProccessingMethod(void);
//--- Correct errors
   ENUM_ERROR_CODE_PROCESSING_METHOD   RequestErrorsCorrecting(MqlTradeRequest &request,const ENUM_ORDER_TYPE order_type,const uint spread_multiplier,CSymbol *symbol_obj);

public:
```

**Complement the specification of the method for checking**
**limitations and errors, and replace the returned type from 'bool' to ENUM\_ERROR\_CODE\_PROCESSING\_METHOD:**

```
//--- Check limitations and errors
   ENUM_ERROR_CODE_PROCESSING_METHOD CheckErrors(const double volume,
                                                 const double price,
                                                 const ENUM_ACTION_TYPE action,
                                                 const ENUM_ORDER_TYPE order_type,
                                                 CSymbol *symbol_obj,
                                                 const CTradeObj *trade_obj,
                                                 const string source_method,
                                                 const double limit=0,
                                                 double sl=0,
                                                 double tp=0);
```

The method has now become more complete — it immediately checks for possible methods for correcting errors in a trading order, and now the
method returns the way to handle detected errors. Previously, it simply returned the check success flag.

**Declare the method of setting the spread multiplier:**

```
//--- Set the following for symbol trading objects:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level, (12) spread multiplier
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
   void                 SetSpreadMultiplier(const uint value=1,const string symbol=NULL);
```

**Add the methods of setting and returning**
**the flag enabling trading for an EA:**

```
//--- Set/return the flag enabling sounds
   void                 SetUseSounds(const bool flag);
   bool                 IsUseSounds(void)                            const { return this.m_use_sound;       }

//--- Set/return the flag enabling trading
   void                 SetTradingDisableFlag(const bool flag)             { this.m_is_trade_disable=flag;  }
   bool                 IsTradingDisable(void)                       const { return this.m_is_trade_disable;}

//--- Open (1) Buy, (2) Sell position
```

There may be errors preventing further trading upon their detection, for example, complete ban on trading for an account. This flag is set when
such errors are detected preventing from sending any further useless trading orders.

**In the class constructor, reset the flag disabling trading**
**and set the default EA behavior in case of trading requests as**
**"correct parameters":**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CTrading::CTrading()
  {
   this.m_list_errors.Clear();
   this.m_list_errors.Sort();
   this.m_log_level=LOG_LEVEL_ALL_MSG;
   this.m_is_trade_disable=false;
   this.m_err_handling_behavior=ERROR_HANDLING_BEHAVIOR_CORRECT;
   ::ZeroMemory(this.m_request);
  }
//+------------------------------------------------------------------+
```

The EA behavior during errors in trading methods can then be set from the EA settings. However, we will use the method of auto correction till
all handlers are ready.

**The current implementation of the methods handling trade server return codes returns**
**only the success flag:**

```
//+------------------------------------------------------------------+
//| Return the error handling method                                 |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::ResultProccessingMethod(void)
  {
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

Why? We do not consider this method in the current article as we will implement handling trade server return codes in the next article. However,
the method has already been described and implemented, albeit in its lite form.

**Implementing the method of correcting errors in a trading order:**

```
//+------------------------------------------------------------------+
//| Correct errors                                                   |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::RequestErrorsCorrecting(MqlTradeRequest &request,
                                                                    const ENUM_ORDER_TYPE order_type,
                                                                    const uint spread_multiplier,
                                                                    CSymbol *symbol_obj)
  {
//--- The empty error list means no errors are detected, return success
   int total=this.m_list_errors.Total();
   if(total==0)
      return ERROR_CODE_PROCESSING_METHOD_OK;
//--- In the current implementation, all these codes are temporarily handled by interrupting a trading request
   if(
      this.IsPresentErorCode(MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED)       || // Trading is disabled for the current account
      this.IsPresentErorCode(MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED)    || // Trading on the trading server side is disabled for EAs on the current account
      this.IsPresentErorCode(MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED)      || // Trading operations are disabled in the terminal
      this.IsPresentErorCode(MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)            || // Trading operations are disabled for the EA
      this.IsPresentErorCode(MSG_SYM_TRADE_MODE_DISABLED)                  || // Trading on a symbol is disabled
      this.IsPresentErorCode(MSG_SYM_TRADE_MODE_CLOSEONLY)                 || // Close only
      this.IsPresentErorCode(MSG_SYM_MARKET_ORDER_DISABLED)                || // Market orders disabled
      this.IsPresentErorCode(MSG_SYM_LIMIT_ORDER_DISABLED)                 || // Limit orders are disabled
      this.IsPresentErorCode(MSG_SYM_STOP_ORDER_DISABLED)                  || // Stop orders are disabled
      this.IsPresentErorCode(MSG_SYM_STOP_LIMIT_ORDER_DISABLED)            || // StopLimit orders are disabled
      this.IsPresentErorCode(MSG_SYM_TRADE_MODE_SHORTONLY)                 || // Only short positions are allowed
      this.IsPresentErorCode(MSG_SYM_TRADE_MODE_LONGONLY)                  || // Only long positions are allowed
      this.IsPresentErorCode(MSG_SYM_CLOSE_BY_ORDER_DISABLED)              || // CloseBy orders are disabled
      this.IsPresentErorCode(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED)       || // Exceeded maximum allowed aggregate volume of orders and positions in one direction
      this.IsPresentErorCode(MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED)        || // Close by is disabled
      this.IsPresentErorCode(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL)        || // Symbols of opposite positions are not equal
      this.IsPresentErorCode(MSG_LIB_TEXT_UNSUPPORTED_PRICE_TYPE_IN_REQ)   || // Unsupported price parameter type in a request
      this.IsPresentErorCode(MSG_LIB_TEXT_TRADING_DISABLE)                 || // Trading disabled for the EA until the reason is eliminated
      this.IsPresentErorCode(10006)                                        || // Request rejected
      this.IsPresentErorCode(10011)                                        || // Request handling error
      this.IsPresentErorCode(10012)                                        || // Request rejected due to expiration
      this.IsPresentErorCode(10013)                                        || // Invalid request
      this.IsPresentErorCode(10017)                                        || // Trading disabled
      this.IsPresentErorCode(10018)                                        || // Market closed
      this.IsPresentErorCode(10023)                                        || // Order status changed
      this.IsPresentErorCode(10025)                                        || // No changes in the request
      this.IsPresentErorCode(10026)                                        || // Auto trading disabled by server
      this.IsPresentErorCode(10027)                                        || // Auto trading disabled by client terminal
      this.IsPresentErorCode(10032)                                        || // Transaction is allowed for live accounts only
      this.IsPresentErorCode(10033)                                        || // The maximum number of pending orders is reached
      this.IsPresentErorCode(10034)                                           // You have reached the maximum order and position volume for this symbol
     ) return ERROR_CODE_PROCESSING_METHOD_EXIT;
//--- View the full list of errors and correct trading request parameters
   for(int i=0;i<total;i++)
     {
      int err=this.m_list_errors.At(i);
      if(err==NULL)
         continue;
      switch(err)
        {
         //--- Correct an invalid volume and stop levels in a trading request
         case MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME :
         case MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME :
         case MSG_LIB_TEXT_INVALID_VOLUME_STEP     :  request.volume=symbol_obj.NormalizedLot(request.volume);                                              break;
         case MSG_SYM_SL_ORDER_DISABLED            :  request.sl=0;                                                                                         break;
         case MSG_SYM_TP_ORDER_DISABLED            :  request.tp=0;                                                                                         break;
         case MSG_LIB_TEXT_PR_LESS_STOP_LEVEL      :  request.price=this.CorrectPricePending(order_type,request.price,0,symbol_obj,spread_multiplier);      break;
         case MSG_LIB_TEXT_SL_LESS_STOP_LEVEL      :  request.sl=this.CorrectStopLoss(order_type,request.price,request.sl,symbol_obj,spread_multiplier);    break;
         case MSG_LIB_TEXT_TP_LESS_STOP_LEVEL      :  request.tp=this.CorrectTakeProfit(order_type,request.price,request.tp,symbol_obj,spread_multiplier);  break;
         //--- If unable to select the position lot, return "abort trading attempt" since the funds are insufficient even for the minimum lot
         case MSG_LIB_TEXT_NOT_ENOUTH_MONEY_FOR    :  request.volume=this.CorrectVolume(request.volume,request.price,order_type,symbol_obj,DFUN);
                                                      if(request.volume==0)
                                                         return ERROR_CODE_PROCESSING_METHOD_EXIT;                                                                                      break;
         //--- Proximity to the order activation level is handled by five-second waiting - during this time, the price may go beyond the freeze level
         case MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL    :
         case MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL    :
         case MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL    :  return (ENUM_ERROR_CODE_PROCESSING_METHOD)5000; // ERROR_CODE_PROCESSING_METHOD_WAIT - wait 5 seconds
         default:
           break;
        }
     }
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

The method logic is described in the code comments. In short: when detecting error codes that cannot be handled yet, we return the "abort
trading attempt" handling method. In case of errors that can be corrected, correct the parameter values and return ОК.

**Improving the method checking trading limitations and trading request errors:**

```
//+------------------------------------------------------------------+
//| Check limitations and errors                                     |
//+------------------------------------------------------------------+
ENUM_ERROR_CODE_PROCESSING_METHOD CTrading::CheckErrors(const double volume,
                                                        const double price,
                                                        const ENUM_ACTION_TYPE action,
                                                        const ENUM_ORDER_TYPE order_type,
                                                        CSymbol *symbol_obj,
                                                        const CTradeObj *trade_obj,
                                                        const string source_method,
                                                        const double limit=0,
                                                        double sl=0,
                                                        double tp=0)
  {
//--- Check the previously set flag disabling trading for an EA
   if(this.IsTradingDisable())
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_FATAL_ERROR;
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

//--- If there are limitations, display the header and the error list
   if(!res)
     {
      //--- Request was rejected before sending to the server due to:
      int total=this.m_list_errors.Total();
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
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
      //--- If the action is performed at the "abort trading operation" error
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
         return ERROR_CODE_PROCESSING_METHOD_EXIT;
      //--- If the action is performed at the "create a pending request" error
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_PENDING_REQUEST)
         return ERROR_CODE_PROCESSING_METHOD_PENDING;
      //--- If the action is performed at the "correct parameters" error
      if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_CORRECT)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_CORRECTED_TRADE_REQUEST));
         //--- Return the result of an attempt to correct the request parameters
         return this.RequestErrorsCorrecting(this.m_request,order_type,trade_obj.SpreadMultiplier(),symbol_obj);
        }
     }
//--- No limitations
   return ERROR_CODE_PROCESSING_METHOD_OK;
  }
//+------------------------------------------------------------------+
```

The complemented code is marked in yellow. Now, the flag disabling trading is first checked in the method. If it is set, the "disable trading
for EA" error handling type is returned. Next, depending on the specified EA behavior during errors and according to the error code, the
required error handling method is returned. If there are no errors, the code that
requires no error handling is returned.

The method of checking trading limitations has undergone multiple similar changes related to adding the necessary flags indicating the
presence of various error types and ways to handle them.

All actions performed in the method, as well as their logic, are described in the
code comments in great details. Therefore, let's have a look at the finalized method:

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
//--- Check connection with the trade server (not in the test mode)
   if(!::TerminalInfoInteger(TERMINAL_CONNECTED))
     {
      if(!::MQLInfoInteger(MQL_TESTER))
        {
         //--- Write the error code to the list and return 'false' - there is no point in further checks
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(10031);
         return false;
        }
     }
//--- Check if trading is enabled for an account (if there is a connection with the trade server)
   else if(!this.m_account.TradeAllowed())
     {
      //--- Write the error code to the list and return 'false' - there is no point in further checks
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_LIB_TEXT_ACCOUNT_NOT_TRADE_ENABLED);
      return false;
     }
//--- Check if trading is allowed for any EAs/scripts for the current account
   if(!this.m_account.TradeExpert())
     {
      //--- Write the error code to the list and return 'false' - there is no point in further checks
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_LIB_TEXT_ACCOUNT_EA_NOT_TRADE_ENABLED);
      return false;
     }
//--- Check if auto trading is allowed in the terminal.
//--- AutoTrading button (Options --> Expert Advisors --> "Allowed automated trading")
   if(!::TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
     {
      //--- Write the error code to the list and return 'false' - there is no point in further checks
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED);
      return false;
     }
//--- Check if auto trading is allowed for the current EA.
//--- (F7 --> Common --> Allow Automated Trading)
   if(!::MQLInfoInteger(MQL_TRADE_ALLOWED))
     {
      //--- Write the error code to the list and return 'false' - there is no point in further checks
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED);
      return false;
     }
//--- Check if trading is enabled on a symbol.
//--- If trading is disabled, write the error code to the list and return 'false' - there is no point in further checks
   if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_DISABLED)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
      this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_DISABLED);
      return false;
     }

//--- If not closing/removal/modification
   if(action_type<ACTION_TYPE_CLOSE_BY)
     {
      //--- In case of close-only, write the error code to the list and return 'false' - there is no point in further checks
      if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_CLOSEONLY)
        {
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_CLOSEONLY);
         return false;
        }
      //--- Check the minimum volume
      if(volume<symbol_obj.LotsMin())
        {
         //--- The volume in a request is less than the minimum allowed one.
         //--- add the error code to the list
         this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_LESS_MIN_VOLUME);
         //--- If the EA behavior during the trading error is set to "abort trading operation",
         //--- return 'false' - there is no point in further checks
         if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
            return false;
         //--- If the EA behavior during a trading error is set to
         //--- "correct parameters" or "create a pending request",
         //--- write 'false' to the result
         else res &=false;
        }
      //--- Check the maximum volume
      else if(volume>symbol_obj.LotsMax())
        {
         //--- The volume in the request exceeds the maximum acceptable one.
         //--- add the error code to the list
         this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_LIB_TEXT_REQ_VOL_MORE_MAX_VOLUME);
         //--- If the EA behavior during the trading error is set to "abort trading operation",
         //--- return 'false' - there is no point in further checks
         if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
            return false;
         //--- If the EA behavior during a trading error is set to
         //--- "correct parameters" or "create a pending request",
         //--- write 'false' to the result
         else res &=false;
        }
      //--- Check the minimum volume gradation
      double step=symbol_obj.LotsStep();
      if(fabs((int)round(volume/step)*step-volume)>0.0000001)
        {
         //--- The volume in the request is not a multiple of the minimum gradation of the lot change step
         //--- add the error code to the list
         this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_LIB_TEXT_INVALID_VOLUME_STEP);
         //--- If the EA behavior during the trading error is set to "abort trading operation",
         //--- return 'false' - there is no point in further checks
         if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
            return false;
         //--- If the EA behavior during a trading error is set to
         //--- "correct parameters" or "create a pending request",
         //--- write 'false' to the result
         else res &=false;
        }
     }

//--- When opening a position
   if(action_type<ACTION_TYPE_BUY_LIMIT)
     {
      //--- Check if sending market orders is allowed on a symbol.
      //--- If using market orders is disabled, write the error code to the list and return 'false' - there is no point in further checks
      if(!symbol_obj.IsMarketOrdersAllowed())
        {
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_SYM_MARKET_ORDER_DISABLED);
         return false;
        }
     }
//--- When placing a pending order
   else if(action_type>ACTION_TYPE_SELL && action_type<ACTION_TYPE_CLOSE_BY)
     {
      //--- If there is a limitation on the number of pending orders on an account and placing a new order exceeds it
      if(this.m_account.LimitOrders()>0 && this.OrdersTotalAll()+1 > this.m_account.LimitOrders())
        {
         //--- The limit on the number of pending orders is reached - write the error code to the list and return 'false' - there is no point in further checks
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(10033);
         return false;
        }
      //--- Check if placing limit orders is allowed on a symbol.
      if(action_type==ACTION_TYPE_BUY_LIMIT || action_type==ACTION_TYPE_SELL_LIMIT)
        {
         //--- If setting limit orders is disabled, write the error code to the list and return 'false' - there is no point in further checks
         if(!symbol_obj.IsLimitOrdersAllowed())
           {
            this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_SYM_LIMIT_ORDER_DISABLED);
            return false;
           }
        }
      //--- Check if placing stop orders is allowed on a symbol.
      else if(action_type==ACTION_TYPE_BUY_STOP || action_type==ACTION_TYPE_SELL_STOP)
        {
         //--- If setting stop orders is disabled, write the error code to the list and return 'false' - there is no point in further checks
         if(!symbol_obj.IsStopOrdersAllowed())
           {
            this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_SYM_STOP_ORDER_DISABLED);
            return false;
           }
        }
      //--- For MQL5, check if placing stop limit orders is allowed on a symbol.
      #ifdef __MQL5__
      else if(action_type==ACTION_TYPE_BUY_STOP_LIMIT || action_type==ACTION_TYPE_SELL_STOP_LIMIT)
        {
         //--- If setting stop limit orders is disabled, write the error code to the list and return 'false' - there is no point in further checks
         if(!symbol_obj.IsStopLimitOrdersAllowed())
           {
            this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_SYM_STOP_LIMIT_ORDER_DISABLED);
            return false;
           }
        }
      #endif
     }

//--- In case of opening/placing/modification
   if(action_type!=ACTION_TYPE_CLOSE_BY)
     {
      //--- If not modification
      if(action_type!=ACTION_TYPE_MODIFY)
        {
         //--- When buying, check if long trading is enabled on a symbol
         if(this.DirectionByActionType(action_type)==ORDER_TYPE_BUY)
           {
            //--- If only short positions are enabled, write the error code to the list and return 'false' - there is no point in further checks
            if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_SHORTONLY)
              {
               this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
               this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_SHORTONLY);
               return false;
              }
            //--- If a symbol has the limitation on the total volume of an open position and pending orders in the same direction
            if(symbol_obj.VolumeLimit()>0)
              {
               //--- (If the total volume of placed long orders and open long positions)+open volume exceed the maximum one
               if(this.OrdersTotalVolumeLong()+this.PositionsTotalVolumeLong()+volume > symbol_obj.VolumeLimit())
                 {
                  //--- Exceeded maximum allowed aggregate volume of orders and positions in one direction
                  //--- write the error code to the list and return 'false' - there is no point in further checks
                  this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
                  this.AddErrorCodeToList(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED);
                  return false;
                 }
              }
           }
         //--- When selling, check if short trading is enabled on a symbol
         else if(this.DirectionByActionType(action_type)==ORDER_TYPE_SELL)
           {
            //--- If only long positions are enabled, write the error code to the list and return 'false' - there is no point in further checks
            if(symbol_obj.TradeMode()==SYMBOL_TRADE_MODE_LONGONLY)
              {
               this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
               this.AddErrorCodeToList(MSG_SYM_TRADE_MODE_LONGONLY);
               return false;
              }
            //--- If a symbol has the limitation on the total volume of an open position and pending orders in the same direction
            if(symbol_obj.VolumeLimit()>0)
              {
               //--- (If the total volume of placed short orders and open short positions)+open volume exceed the maximum one
               if(this.OrdersTotalVolumeShort()+this.PositionsTotalVolumeShort()+volume > symbol_obj.VolumeLimit())
                 {
                  //--- Exceeded maximum allowed aggregate volume of orders and positions in one direction
                  //--- write the error code to the list and return 'false' - there is no point in further checks
                  this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
                  this.AddErrorCodeToList(MSG_LIB_TEXT_MAX_VOLUME_LIMIT_EXCEEDED);
                  return false;
                 }
              }
           }
        }
      //--- If the request features StopLoss and its placing is not allowed
      if(sl>0 && !symbol_obj.IsStopLossOrdersAllowed())
        {
         //--- add the error code to the list
         this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_SYM_SL_ORDER_DISABLED);
         //--- If the EA behavior during the trading error is set to "abort trading operation",
         //--- return 'false' - there is no point in further checks
         if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
            return false;
         //--- If the EA behavior during a trading error is set to
         //--- "correct parameters" or "create a pending request",
         //--- write 'false' to the result
         else res &=false;
        }
      //--- If the request features TakeProfit and its placing is not allowed
      if(tp>0 && !symbol_obj.IsTakeProfitOrdersAllowed())
        {
         //--- add the error code to the list
         this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_SYM_TP_ORDER_DISABLED);
         //--- If the EA behavior during the trading error is set to "abort trading operation",
         //--- return 'false' - there is no point in further checks
         if(this.m_err_handling_behavior==ERROR_HANDLING_BEHAVIOR_BREAK)
            return false;
         //--- If the EA behavior during a trading error is set to
         //--- "correct parameters" or "create a pending request",
         //--- write 'false' to the result
         else res &=false;
        }
     }

//--- When closing by an opposite position
   else if(action_type==ACTION_TYPE_CLOSE_BY)
     {
      //--- When closing by an opposite position is disabled
      if(!symbol_obj.IsCloseByOrdersAllowed())
        {
         //--- write the error code to the list and return 'false'
         this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
         this.AddErrorCodeToList(MSG_LIB_TEXT_CLOSE_BY_ORDERS_DISABLED);
         return false;
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

**In the method returning the parameter values by StopLevel and FreezeLevel, add**
**the flag indicating that an error should be viewed in the error list to each detected error:**

```
//+------------------------------------------------------------------+
//| Check parameter values by StopLevel and FreezeLevel              |
//+------------------------------------------------------------------+
bool CTrading::CheckLevels(const ENUM_ACTION_TYPE action,
                           const ENUM_ORDER_TYPE order_type,
                           double price,
                           double limit,
                           double sl,
                           double tp,
                           const CSymbol *symbol_obj,
                           const string source_method)
  {
//--- the result of conducting all checks
   bool res=true;
//--- StopLevel
//--- If this is not a position closure/order removal
   if(action!=ACTION_TYPE_CLOSE && action!=ACTION_TYPE_CLOSE_BY)
     {
      //--- When placing a pending order
      if(action>ACTION_TYPE_SELL)
        {
         //--- If the placement distance in points is less than StopLevel
         if(!this.CheckPriceByStopLevel(order_type,price,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_PR_LESS_STOP_LEVEL);
            res &=false;
           }
        }
      //--- If StopLoss is present
      if(sl>0)
        {
         //--- If StopLoss distance in points from the open price is less than StopLevel
         double price_open=(action==ACTION_TYPE_BUY_STOP_LIMIT || action==ACTION_TYPE_SELL_STOP_LIMIT ? limit : price);
         if(!this.CheckStopLossByStopLevel(order_type,price_open,sl,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_SL_LESS_STOP_LEVEL);
            res &=false;
           }
        }
      //--- If TakeProfit is present
      if(tp>0)
        {
         double price_open=(action==ACTION_TYPE_BUY_STOP_LIMIT || action==ACTION_TYPE_SELL_STOP_LIMIT ? limit : price);
         //--- If TakeProfit distance in points from the open price is less than StopLevel
         if(!this.CheckTakeProfitByStopLevel(order_type,price_open,tp,symbol_obj))
           {
            //--- add the error code to the list and write 'false' to the result
            this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
            this.AddErrorCodeToList(MSG_LIB_TEXT_TP_LESS_STOP_LEVEL);
            res &=false;
           }
        }
     }
//--- FreezeLevel
//--- If this is a position closure/order removal/modification
   if(action>ACTION_TYPE_SELL_STOP_LIMIT)
     {
      //--- If this is a position
      if(order_type<ORDER_TYPE_BUY_LIMIT)
        {
         //--- StopLoss modification
         if(sl>0)
           {
            //--- If the distance from the price to StopLoss is less than FreezeLevel
            if(!this.CheckStopLossByFreezeLevel(order_type,sl,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
               this.AddErrorCodeToList(MSG_LIB_TEXT_SL_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
         //--- TakeProfit modification
         if(tp>0)
           {
            //--- If the distance from the price to StopLoss is less than FreezeLevel
            if(!this.CheckTakeProfitByFreezeLevel(order_type,tp,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
               this.AddErrorCodeToList(MSG_LIB_TEXT_TP_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
        }
      //--- If this is a pending order
      else
        {
         //--- Placement price modification
         if(price>0)
           {
            //--- If the distance from the price to the order activation price is less than FreezeLevel
            if(!this.CheckPriceByFreezeLevel(order_type,price,symbol_obj))
              {
               //--- add the error code to the list and write 'false' to the result
               this.m_error_reason_flags &=TRADE_REQUEST_ERR_FLAG_ERROR_IN_LIST;
               this.AddErrorCodeToList(MSG_LIB_TEXT_PR_LESS_FREEZE_LEVEL);
               res &=false;
              }
           }
        }
     }
   return res;
  }
//+------------------------------------------------------------------+
```

**In the method of setting trading request prices, add updating the**
**prices and exiting in case of an update error with the appropriate error code:**

```
//+------------------------------------------------------------------+
//| Set trading request prices                                       |
//+------------------------------------------------------------------+
template <typename PR,typename SL,typename TP,typename PL>
bool CTrading::SetPrices(const ENUM_ORDER_TYPE action,const PR price,const SL sl,const TP tp,const PL limit,const string source_method,CSymbol *symbol_obj)
  {
//--- Reset prices
   ::ZeroMemory(this.m_request);
//--- Update all data by symbol
   if(!symbol_obj.RefreshRates())
     {
      this.AddErrorCodeToList(10021);
      return false;
     }

//--- Open/close price
```

**Also, price calculation in the method of setting trading request prices has also been changed:**

```
         //--- Calculate the order price
         switch((int)action)
           {
            //--- Pending order
            case ORDER_TYPE_BUY_LIMIT       :  this.m_request.price=::NormalizeDouble(symbol_obj.Ask()-price*symbol_obj.Point(),symbol_obj.Digits());      break;
            case ORDER_TYPE_BUY_STOP        :
            case ORDER_TYPE_BUY_STOP_LIMIT  :  this.m_request.price=::NormalizeDouble(symbol_obj.Ask()+price*symbol_obj.Point(),symbol_obj.Digits());      break;

            case ORDER_TYPE_SELL_LIMIT      :  this.m_request.price=::NormalizeDouble(symbol_obj.BidLast()+price*symbol_obj.Point(),symbol_obj.Digits());  break;
            case ORDER_TYPE_SELL_STOP       :
            case ORDER_TYPE_SELL_STOP_LIMIT :  this.m_request.price=::NormalizeDouble(symbol_obj.BidLast()-price*symbol_obj.Point(),symbol_obj.Digits());  break;
            //--- Default - current position open prices
            default  :  this.m_request.price=
              (
               this.DirectionByActionType((ENUM_ACTION_TYPE)action)==ORDER_TYPE_BUY ? ::NormalizeDouble(symbol_obj.Ask(),symbol_obj.Digits()) :
               ::NormalizeDouble(symbol_obj.BidLast(),symbol_obj.Digits())
              ); break;
           }
```

The method of the Bid() symbol object class has been replaced with the BidLast()
method returning either Bid or Last price depending on the chart construction mode.

**The method setting the spread multiplier for trading objects of all symbols:**

```
//+------------------------------------------------------------------+
//| Set the spread multiplier                                        |
//| for trading objects of all symbols                               |
//+------------------------------------------------------------------+
void CTrading::SetSpreadMultiplier(const uint value=1,const string symbol=NULL)
  {
   CSymbol *symbol_obj=NULL;
   if(symbol==NULL)
     {
      CArrayObj *list=this.m_symbols.GetList();
      if(list==NULL || list.Total()==0)
         return;
      int total=list.Total();
      for(int i=0;i<total;i++)
        {
         symbol_obj=list.At(i);
         if(symbol_obj==NULL)
            continue;
         CTradeObj *trade_obj=symbol_obj.GetTradeObj();
         if(trade_obj==NULL)
            continue;
         trade_obj.SetSpreadMultiplier(value);
        }
     }
   else
     {
      CTradeObj *trade_obj=this.GetTradeObjBySymbol(symbol,DFUN);
      if(trade_obj==NULL)
         return;
      trade_obj.SetSpreadMultiplier(value);
     }
  }
//+------------------------------------------------------------------+
```

The method receives the multiplier value (default is 1) and a symbol name (default is NULL).

If NULL is passed as a symbol, the multiplier is set for trading objects of all symbols
of the existing symbol collection.

Otherwise, the value is assigned to a trading object of a symbol whose name has been passed to the
method.

Due to the new error handling, all trading methods have been refined.

**Let's consider the code of the Buy position opening**
**method:**

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
//--- Set the trading request result as 'true' and the error flag as "no errors"
   bool res=true;
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_BUY;
   ENUM_ORDER_TYPE order_type=ORDER_TYPE_BUY;
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
//--- If failed to set - write the "internal error" flag, display the message in the journal and return 'false'
   if(!this.SetPrices(order_type,0,sl,tp,0,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
     }

//--- Write the volume to the request structure
   this.m_request.volume=volume;
//--- Get the method of handling errors from the CheckErrors() method while checking for errors
   ENUM_ERROR_CODE_PROCESSING_METHOD method=this.CheckErrors(this.m_request.volume,symbol_obj.Ask(),action,order_type,symbol_obj,trade_obj,DFUN,0,this.m_request.sl,this.m_request.tp);
//--- In case of trading limitations, funds insufficiency,
//--- if there are limitations by StopLevel or FreezeLevel ...
   if(method!=ERROR_CODE_PROCESSING_METHOD_OK)
     {
      //--- If trading is disabled completely, display a journal message, play the error sound and exit
      if(method==ERROR_CODE_PROCESSING_METHOD_DISABLE)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "abort trading operation" - display a journal message, play the error sound and exit
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
        {
         if(this.m_log_level>LOG_LEVEL_NO_MSG)
            ::Print(CMessage::Text(MSG_LIB_TEXT_TRADING_OPERATION_ABORTED));
         if(this.IsUseSounds())
            trade_obj.PlaySoundError(action,order_type);
         return false;
        }
      //--- If the check result is "waiting", display the message in the journal
      if(method==ERROR_CODE_PROCESSING_METHOD_EXIT)
        {
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

//--- Send the request
   res=trade_obj.OpenPosition(POSITION_TYPE_BUY,this.m_request.volume,this.m_request.sl,this.m_request.tp,magic,comment,deviation);
//--- If the request is successful, play the success sound set for a symbol trading object for this type of trading operation
   if(res)
     {
      if(this.IsUseSounds())
         trade_obj.PlaySoundSuccess(action,order_type);
     }
//--- If the request is not successful, play the error sound set for a symbol trading object for this type of trading operation
   else
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_SYS_ERROR),": ",CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.IsUseSounds())
         trade_obj.PlaySoundError(action,order_type);
     }
//--- Return the result of sending a trading request in a symbol trading object
   return res;
  }
//+------------------------------------------------------------------+
```

All clarifications are made in details in the code comments. Other trading methods have been improved in a similar way. I hope, everything is
clear here. In any case, you are welcome to use the comments section.

**The methods returning calculated valid stop and pending order prices:**

```
//+------------------------------------------------------------------+
//| Return correct StopLoss relative to StopLevel                    |
//+------------------------------------------------------------------+
double CTrading::CorrectStopLoss(const ENUM_ORDER_TYPE order_type,const double price_set,const double stop_loss,const CSymbol *symbol_obj,const uint spread_multiplier=1)
  {
   if(stop_loss==0) return 0;
   uint lv=(symbol_obj.TradeStopLevel()==0 ? symbol_obj.Spread()*spread_multiplier : symbol_obj.TradeStopLevel());
   double price=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : order_type==ORDER_TYPE_SELL ? symbol_obj.Ask() : price_set);
   return
     (this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY            ?
      ::NormalizeDouble(fmin(price-lv*symbol_obj.Point(),stop_loss),symbol_obj.Digits())  :
      ::NormalizeDouble(fmax(price+lv*symbol_obj.Point(),stop_loss),symbol_obj.Digits())
     );
  }
//+------------------------------------------------------------------+
//| Return correct TakeProfit relative to StopLevel                  |
//+------------------------------------------------------------------+
double CTrading::CorrectTakeProfit(const ENUM_ORDER_TYPE order_type,const double price_set,const double take_profit,const CSymbol *symbol_obj,const uint spread_multiplier=1)
  {
   if(take_profit==0) return 0;
   uint lv=(symbol_obj.TradeStopLevel()==0 ? symbol_obj.Spread()*spread_multiplier : symbol_obj.TradeStopLevel());
   double price=(order_type==ORDER_TYPE_BUY ? symbol_obj.BidLast() : order_type==ORDER_TYPE_SELL ? symbol_obj.Ask() : price_set);
   return
     (this.DirectionByActionType((ENUM_ACTION_TYPE)order_type)==ORDER_TYPE_BUY             ?
      ::NormalizeDouble(fmax(price+lv*symbol_obj.Point(),take_profit),symbol_obj.Digits()) :
      ::NormalizeDouble(fmin(price-lv*symbol_obj.Point(),take_profit),symbol_obj.Digits())
     );
  }
//+------------------------------------------------------------------+
//| Return the correct order placement price                         |
//| relative to StopLevel                                            |
//+------------------------------------------------------------------+
double CTrading::CorrectPricePending(const ENUM_ORDER_TYPE order_type,const double price_set,const double price,const CSymbol *symbol_obj,const uint spread_multiplier=1)
  {
   uint lv=(symbol_obj.TradeStopLevel()==0 ? symbol_obj.Spread()*spread_multiplier : symbol_obj.TradeStopLevel());
   double pp=0;
   switch((int)order_type)
     {
      case ORDER_TYPE_BUY_LIMIT        :  pp=(price==0 ? symbol_obj.Ask()     : price); return ::NormalizeDouble(fmin(pp-lv*symbol_obj.Point(),price_set),symbol_obj.Digits());
      case ORDER_TYPE_BUY_STOP         :
      case ORDER_TYPE_BUY_STOP_LIMIT   :  pp=(price==0 ? symbol_obj.Ask()     : price); return ::NormalizeDouble(fmax(pp+lv*symbol_obj.Point(),price_set),symbol_obj.Digits());
      case ORDER_TYPE_SELL_LIMIT       :  pp=(price==0 ? symbol_obj.BidLast() : price); return ::NormalizeDouble(fmax(pp+lv*symbol_obj.Point(),price_set),symbol_obj.Digits());
      case ORDER_TYPE_SELL_STOP        :
      case ORDER_TYPE_SELL_STOP_LIMIT  :  pp=(price==0 ? symbol_obj.BidLast() : price); return ::NormalizeDouble(fmin(pp-lv*symbol_obj.Point(),price_set),symbol_obj.Digits());
      default                          :  if(this.m_log_level>LOG_LEVEL_NO_MSG) ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_INVALID_ORDER_TYPE),::EnumToString(order_type)); return 0;
     }
  }
//+------------------------------------------------------------------+
```

Here all should be clear even without code comments — the prices passed to the methods are compared to the price obtained as a StopLevel shift
from the open price. The valid (higher/lower depending on an order type) price is returned to the calling program.

**The method returning the volume, at which it is possible to open a position:**

```
//+------------------------------------------------------------------+
//| Return the volume, at which it is possible to open a position    |
//+------------------------------------------------------------------+
double CTrading::CorrectVolume(const double price,const ENUM_ORDER_TYPE order_type,CSymbol *symbol_obj,const string source_method)
  {
//--- If funds are insufficient for the minimum lot, inform of that and return zero
   if(!this.CheckMoneyFree(symbol_obj.LotsMin(),price,order_type,symbol_obj,source_method))
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT));
      return 0;
     }

//--- Update account and symbol data
   this.m_account.Refresh();
   symbol_obj.RefreshRates();
//--- Calculate the lot, which is closest to the acceptable one
   double vol=symbol_obj.NormalizedLot(this.m_account.Equity()*this.m_account.Leverage()/symbol_obj.TradeContractSize()/(symbol_obj.CurrencyBase()=="USD" ? 1.0 : symbol_obj.BidLast()));
//--- Calculate a sufficient lot
   double margin=this.m_account.MarginForAction(order_type,symbol_obj.Name(),1.0,price);
   if(margin!=EMPTY_VALUE)
      vol=symbol_obj.NormalizedLot(this.m_account.MarginFree()/margin);

//--- If the calculated lot is invalid or the margin calculation returns an error
   if(!this.CheckMoneyFree(vol,price,order_type,symbol_obj,source_method))
     {
      //--- In the do-while loop, while the calculated valid volume exceeds the minimum lot
      do
        {
         //--- Subtract the minimum lot from the valid lot value
         vol-=symbol_obj.LotsStep();
         //--- If the calculated lot allows opening a position/setting an order, return the lot value
         if(this.CheckMoneyFree(symbol_obj.NormalizedLot(vol),price,order_type,symbol_obj,source_method))
            return vol;
        }
      while(vol>symbol_obj.LotsMin() && !::IsStopped());
     }
//--- If the lot is calculated correctly, return the calculated lot
   else
      return vol;
//--- If the current stage is reached, the funds are insufficient. Inform of that and return zero
   if(this.m_log_level>LOG_LEVEL_NO_MSG)
      ::Print(CMessage::Text(MSG_LIB_TEXT_NOT_POSSIBILITY_CORRECT_LOT));
   return 0;
  }
//+------------------------------------------------------------------+
```

The code is also commented here.

First, we check the ability to open by a
minimum lot. If this is impossible, further calculations are meaningless — return zero.

Next, we calculate
approximate allowable lot (in order not to "select" the required lot from its maximum value in case of its possible adjustment).


Next, calculate the maximum lot allowing to open a position using all available functions.
Why so? If the funds are insufficient for opening a position, this implies that the required volume was large, which means we need to
calculate the maximum possible volume.

In this calculation, we use the [OrderCalcMargin()](https://www.mql5.com/en/docs/trading/ordercalcmargin)
function that may return false in case of an error, while the MarginForAction() method of
the **CAccount** class using this function returns EMPTY\_VALUE
corresponding to the [DBL\_MAX](https://www.mql5.com/en/docs/constant_indices) constant value (the maximum value
that can be represented by the double type). If we receive this value, there has
been an error and the lot has not been calculated.

In this case (not only in case of an error but also when checking the calculation
validity), we will use the "selection" of a required maximum lot by simply subtracting the lot step from the calculated maximum possible
volume in the trading order. This is where we need the previously calculated approximate
available volume. If we have been unable to calculate the exact volume, the lot decrease loop starts from its nearest lot (rather than
its maximum lot set for the symbol) greatly reducing the number of the loop iterations.

By the way, during the check, I have not received the [OrderCalcMargin()](https://www.mql5.com/en/docs/trading/ordercalcmargin)
function errors when calculating the lot. However, the invalid calculations have still occurred (approximately, by one lot change step).

**This concludes the trading class changes and improvements.**

### Testing

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/7286#node03)
and save it to \\MQL5\\Experts\\TestDoEasy\ **Part24\** under the name **TestDoEasyPart24.mq5**.

**Add the flag of working in the strategy tester to the list**
**of global variables:**

```
//--- global variables
CEngine        engine;
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
bool           testing;
//+------------------------------------------------------------------+
```

**In the OnInit() handler, set the value of the flag of working in the**
**strategy tester:**

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
```

In order to send events to the **OnDoEasyEvent()** library events handler while working in the tester, we need the special
function **EventsHandling()** .

**It has undergone a minor**
**improvement:**

```
//+------------------------------------------------------------------+
//| Working with events in the tester                                |
//+------------------------------------------------------------------+
void EventsHandling(void)
  {
//--- If a trading event is present
   if(engine.IsTradeEvent())
     {
      //--- Number of trading events occurred simultaneously
      int total=engine.GetTradeEventsTotal();
      for(int i=0;i<total;i++)
        {
         //--- Get the next event from the list of simultaneously occurred events by index
         CEventBaseObj *event=engine.GetTradeEventByIndex(i);
         if(event==NULL)
            continue;
         long   lparam=i;
         double dparam=event.DParam();
         string sparam=event.SParam();
         OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
        }
     }
//--- If there is an account event
   if(engine.IsAccountsEvent())
     {
      //--- Get the list of all account events occurred simultaneously
      CArrayObj* list=engine.GetListAccountEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
//--- If there is a symbol collection event
   if(engine.IsSymbolsEvent())
     {
      //--- Get the list of all symbol events occurred simultaneously
      CArrayObj* list=engine.GetListSymbolsEvents();
      if(list!=NULL)
        {
         //--- Get the next event in a loop
         int total=list.Total();
         for(int i=0;i<total;i++)
           {
            //--- take an event from the list
            CEventBaseObj *event=list.At(i);
            if(event==NULL)
               continue;
            //--- Send an event to the event handler
            long lparam=event.LParam();
            double dparam=event.DParam();
            string sparam=event.SParam();
            OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(),lparam,dparam,sparam);
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The code comments are quite clear here.

Since we have now created the list of new trading events, **we get**
**each event from the list of all new trading events in the OnDoEasyEvent() library events handler by an event index and simply display**
**descriptions of all events obtained from the list in the journal:**

```
//+------------------------------------------------------------------+
//| Handling DoEasy library events                                   |
//+------------------------------------------------------------------+
void OnDoEasyEvent(const int id,
                   const long &lparam,
                   const double &dparam,
                   const string &sparam)
  {
   int idx=id-CHARTEVENT_CUSTOM;
//--- Retrieve (1) event time milliseconds, (2) reason and (3) source from lparam, as well as (4) set the exact event time
   ushort msc=engine.EventMSC(lparam);
   ushort reason=engine.EventReason(lparam);
   ushort source=engine.EventSource(lparam);
   long time=TimeCurrent()*1000+msc;

//--- Handling symbol events
   if(source==COLLECTION_SYMBOLS_ID)
     {
      CSymbol *symbol=engine.GetSymbolObjByName(sparam);
      if(symbol==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=(idx<SYMBOL_PROP_INTEGER_TOTAL ? 0 : symbol.Digits());
      //--- Event text description
      string id_descr=(idx<SYMBOL_PROP_INTEGER_TOTAL ? symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_INTEGER)idx) : symbol.GetPropertyDescription((ENUM_SYMBOL_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Check event reasons and display its description in the journal
      if(reason==BASE_EVENT_REASON_INC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(symbol.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling account events
   else if(source==COLLECTION_ACCOUNT_ID)
     {
      CAccount *account=engine.GetAccountCurrent();
      if(account==NULL)
         return;
      //--- Number of decimal places in the event value - in case of a 'long' event, it is 0, otherwise - Digits() of a symbol
      int digits=int(idx<ACCOUNT_PROP_INTEGER_TOTAL ? 0 : account.CurrencyDigits());
      //--- Event text description
      string id_descr=(idx<ACCOUNT_PROP_INTEGER_TOTAL ? account.GetPropertyDescription((ENUM_ACCOUNT_PROP_INTEGER)idx) : account.GetPropertyDescription((ENUM_ACCOUNT_PROP_DOUBLE)idx));
      //--- Property change text value
      string value=DoubleToString(dparam,digits);

      //--- Checking event reasons and handling the increase of funds by a specified value,

      //--- In case of a property value increase
      if(reason==BASE_EVENT_REASON_INC)
        {
         //--- Display an event in the journal
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
         //--- if this is an equity increase
         if(idx==ACCOUNT_PROP_EQUITY)
           {
            //--- Get the list of all open positions
            CArrayObj* list_positions=engine.GetListMarketPosition();
            //--- Select positions with the profit exceeding zero
            list_positions=CSelect::ByOrderProperty(list_positions,ORDER_PROP_PROFIT_FULL,0,MORE);
            if(list_positions!=NULL)
              {
               //--- Sort the list by profit considering commission and swap
               list_positions.Sort(SORT_BY_ORDER_PROFIT_FULL);
               //--- Get the position index with the highest profit
               int index=CSelect::FindOrderMax(list_positions,ORDER_PROP_PROFIT_FULL);
               if(index>WRONG_VALUE)
                 {
                  COrder* position=list_positions.At(index);
                  if(position!=NULL)
                    {
                     //--- Get a ticket of a position with the highest profit and close the position by a ticket
                     engine.ClosePosition(position.Ticket());
                    }
                 }
              }
           }
        }
      //--- Other events are simply displayed in the journal
      if(reason==BASE_EVENT_REASON_DEC)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_MORE_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_LESS_THEN)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
      if(reason==BASE_EVENT_REASON_EQUALS)
        {
         Print(account.EventDescription(idx,(ENUM_BASE_EVENT_REASON)reason,source,value,id_descr,digits));
        }
     }

//--- Handling market watch window events
   else if(idx>MARKET_WATCH_EVENT_NO_EVENT && idx<SYMBOL_EVENTS_NEXT_CODE)
     {
      //--- Market Watch window event
      string descr=engine.GetMWEventDescription((ENUM_MW_EVENT)idx);
      string name=(idx==MARKET_WATCH_EVENT_SYMBOL_SORT ? "" : ": "+sparam);
      Print(TimeMSCtoString(lparam)," ",descr,name);
     }
//--- Handling trading events
   else if(idx>TRADE_EVENT_NO_EVENT && idx<TRADE_EVENTS_NEXT_CODE)
     {
      //--- Get the list of trading events
      CArrayObj *list=engine.GetListAllOrdersEvents();
      if(list==NULL)
         return;
      //--- get the event index shift relative to the end of the list
      //--- in the tester, the shift is passed by the lparam parameter to the event handler
      //--- outside the tester, events are sent one by one and handled in OnChartEvent()
      int shift=(testing ? (int)lparam : 0);
      CEvent *event=list.At(list.Total()-1-shift);
      if(event==NULL)
      return;
      //--- Accrue the credit
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CREDIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional charges
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CHARGE)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Correction
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_CORRECTION)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Enumerate bonuses
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BONUS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Additional commissions
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_DAILY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_MONTHLY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Daily agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_DAILY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Monthly agent commission
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_COMISSION_AGENT_MONTHLY)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Interest rate
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_INTEREST)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled buy deal
      if(event.TypeEvent()==TRADE_EVENT_BUY_CANCELLED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Canceled sell deal
      if(event.TypeEvent()==TRADE_EVENT_SELL_CANCELLED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Dividend operations
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Accrual of franked dividend
      if(event.TypeEvent()==TRADE_EVENT_DIVIDENT_FRANKED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Tax charges
      if(event.TypeEvent()==TRADE_EVENT_TAX)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Replenishing account balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_REFILL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Withdrawing funds from balance
      if(event.TypeEvent()==TRADE_EVENT_ACCOUNT_BALANCE_WITHDRAWAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }

      //--- Pending order placed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_PLASED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order removed
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_REMOVED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Pending order partially activated by price
      if(event.TypeEvent()==TRADE_EVENT_PENDING_ORDER_ACTIVATED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position opened partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_OPENED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_POS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_BY_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by partial market order execution (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_MARKET_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position reversal by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_REVERSED_BY_PENDING_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by a new deal (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial execution of a market order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_MARKET_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by activating a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Added volume to a position by partial activation of a pending order (netting)
      if(event.TypeEvent()==TRADE_EVENT_POSITION_VOLUME_ADD_BY_PENDING_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position partially closed by an opposite one
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_POS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by StopLoss
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_SL)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Position closed partially by TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_POSITION_CLOSED_PARTIAL_BY_TP)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- StopLimit order activation
      if(event.TypeEvent()==TRADE_EVENT_TRIGGERED_STOP_LIMIT_ORDER)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and StopLoss price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order, StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_PRICE_STOP_LOSS_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss and TakeProfit price
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_STOP_LOSS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing order's TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_ORDER_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position's StopLoss and TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position StopLoss
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_STOP_LOSS)
        {
         Print(DFUN,event.TypeEventDescription());
        }
      //--- Changing position TakeProfit
      if(event.TypeEvent()==TRADE_EVENT_MODIFY_POSITION_TAKE_PROFIT)
        {
         Print(DFUN,event.TypeEventDescription());
        }
     }
  }
//+------------------------------------------------------------------+
```

For more simplicity, we just receive an event from the list by its index (for the tester, the index is passed in the **lparam** parameter by the **EventsHandling()** function, while on a demo and real accounts, the index is always equal to zero since every
event is sent to OnChartEvent() as an independent event rather than from the list) and display the description of an obtained event in the
journal.

It is up to you to arrange their handling. You may implement the handling directly in the same code or declare the list of event flags setting
the flags of occurred events here, while the actual handling is performed in separate functions.

These are all the changes and improvements necessary to control all trading events occurred simultaneously. The library already features all
the necessary things for auto correction of trading request parameter errors. No changes are required in the EA (for now). Further on, after
creating all ways of handling errors, we are going to introduce an additional input indicating the EA behavior during errors.

Compile the EA and launch it in the tester. Place several pending orders and remove all of them in a single loop:

![](https://c.mql5.com/2/37/4m5ss6HmWh.gif)

The EA displays four events in the journal. These events occurred when removing four pending orders in a single loop after clicking "Delete
pending".

Now let's set a bigger lot in the EA settings in the strategy tester (for example, 100.0) and try to set a pending order or open a position:

![](https://c.mql5.com/2/37/eiho4xrdMY.gif)

After trying to set a pending order and open a position with the volumes of 100.0 lots, we have obtained journal messages informing of funds
insufficiency and volume adjustment. The order was set and the position was opened after that.

### What's next?

In the next article, we will implement handling errors returned by the trade server.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7326#node00)

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

[Part 9. Compatibility with MQL4 - \\
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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7326](https://www.mql5.com/ru/articles/7326)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7326.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7326/mql5.zip "Download MQL5.zip")(3598.41 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7326/mql4.zip "Download MQL4.zip")(3598.33 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/331116)**
(26)


![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Oct 2020 at 00:09

Hi Artyom -- in working closer with this code, I noticed this peculiarity of the 'shift' value implemented in EventsHandling() and OnDoEasyEvent() for correctly handling trade events when running in the tester... I understand, as you point out in the article, when running live **trade events** are delivered one by one from OnChartEvent() as they are triggered by the engine, whereas in [testing mode](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_mql5_info_integer "MQL5 documentation: Running MQL5 Program Properties") they are grouped together and delivered as a list...

My question is: wouldn't it be better to implement a dedicated function parameter in the event-handler rather than sacrificing 'lparam' which can contain useful information for the event-handler? I also think it makes the code simpler / more readable; do you agree?

PS: Anyway, I am finding this library to be really powerful but also complex and difficult to wrap one's head around, however once mastered it should enable developing all sorts of EA-powered strategies very quickly... Beside the huge learning curve, I am also noticing the back-testing performance to be rather slow, so I hope you can address this point once you complete the feature-set you envisioned for DoEasy.

```
void EventsHandling(void)
  {
//--- If a trading event is present
   if(engine.IsTradeEvent())
     {
      //--- Number of trading events occurred simultaneously
      int total = engine.GetTradeEventsTotal();
      for (int i = 0; i < total; i++)
        {
         //--- Get the next event from the list of simultaneously occurred events by index
         CEventBaseObj *event = engine.GetTradeEventByIndex(i);
         if(event == NULL)
            continue;
         int   shift  = i;
         long   lparam = event.LParam();
         double dparam = event.DParam();
         string sparam = event.SParam();
         OnDoEasyEvent(CHARTEVENT_CUSTOM+event.ID(), lparam, dparam, sparam, shift);
        }
     }
   //...
   //...
  }

void OnDoEasyEvent(const int id, const long &lparam, const double &dparam, const string &sparam, int shift=0)
  {
   //...
   //...
//--- Handling trading events
   if(idx > TRADE_EVENT_NO_EVENT && idx < TRADE_EVENTS_NEXT_CODE)
     {
      //--- Get the list of trading events
      CArrayObj *list = engine.GetListAllOrdersEvents();
      if(list == NULL) return;

      //--- get the event index shift relative to the end of the list
      //--- in the tester, the shift is passed by the lparam parameter to the event handler
      //--- outside the tester, events are sent one by one and handled in OnChartEvent()
      int shift=(testing ? (int)lparam : 0);

      CEvent *event=list.At(list.Total()-1-shift);
      if(event==NULL) return;
   //...
   //...
  }
```

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
8 Oct 2020 at 07:56

**Dima Diall :**

Hi Artyom -- in working closer with this code, I noticed this peculiarity of the ' shift ' value implemented in  EventsHandling()  and  OnDoEasyEvent()  for correctly handling trade events when running in the tester... I understand, as you point out in the article, when running live **trade events** are delivered one by one from  OnChartEvent()  as they are triggered by the engine, whereas in [testing mode](https://www.mql5.com/en/docs/constants/environment_state/mql5_programm_info#enum_mql5_info_integer "MQL5 documentation: Running MQL5 Program Properties") they are grouped together and delivered as a list...

My question is: wouldn't it be better to implement a dedicated function parameter in the event-handler rather than sacrificing ' lparam ' which can contain useful information for the event-handler? I also think it makes the code simpler / more readable; do you agree?

PS: Anyway, I am finding this library to be really powerful but also complex and difficult to wrap one's head around, however once mastered it should enable developing all sorts of EA-powered strategies very quickly... Beside the huge learning curve, I am also noticing the back-testing performance to be rather slow, so I hope you can address this point once you complete the feature-set you envisioned for DoEasy.

No. Here I did not plan to redo anything, and most likely I will not. All the necessary data is already delivered to event objects, and the rest of the data is already taken from those objects whose event was registered.

![Moon Domain - Unipessoal Lda](https://c.mql5.com/avatar/2020/8/5F384774-67B1.jpg)

**[Dmitri Diall](https://www.mql5.com/en/users/ddiall)**
\|
8 Oct 2020 at 14:51

**Artyom Trishkin:**

No. Here I did not plan to redo anything, and most likely I will not. All the necessary data is already delivered to event objects, and the rest of the data is already taken from those objects whose event was registered.

OK, fair enough... I agree that all necessary data is in the event objects.

![leonerd](https://c.mql5.com/avatar/2017/5/5919A02E-9AEB.jpg)

**[leonerd](https://www.mql5.com/en/users/leonerd)**
\|
20 Dec 2020 at 12:48

Could you please give me a code example to pull the [order/position ticket](https://www.mql5.com/en/docs/trading/ordergetticket "MQL5 documentation: OrderGetTicket function") and other properties after getting the last trade event?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
21 Dec 2020 at 12:58

**leonerd:**

Could you please provide some sample code so that when you get the last trade event, you can pull the [order/position ticket](https://www.mql5.com/en/docs/trading/ordergetticket "MQL5 documentation: OrderGetTicket function") and other properties?

Not until a week from now - not on site.


![Extending Strategy Builder Functionality](https://c.mql5.com/2/37/Article_Logo__1.png)[Extending Strategy Builder Functionality](https://www.mql5.com/en/articles/7361)

In the previous two articles, we discussed the application of Merrill patterns to various data types. An application was developed to test the presented ideas. In this article, we will continue working with the Strategy Builder, to improve its efficiency and to implement new features and capabilities.

![Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization.png)[Continuous Walk-Forward Optimization (Part 1): Working with Optimization Reports](https://www.mql5.com/en/articles/7290)

The first article is devoted to the creation of a toolkit for working with optimization reports, for importing them from the terminal, as well as for filtering and sorting the obtained data. MetaTrader 5 allows downloading optimization results, however our purpose is to add our own data to the optimization report.

![Library for easy and quick development of MetaTrader programs (part XXV): Handling errors returned by the trade server](https://c.mql5.com/2/37/MQL5-avatar-doeasy__12.png)[Library for easy and quick development of MetaTrader programs (part XXV): Handling errors returned by the trade server](https://www.mql5.com/en/articles/7365)

After we send a trading order to the server, we need to check the error codes or the absence of errors. In this article, we will consider handling errors returned by the trade server and prepare for creating pending trading requests.

![Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://c.mql5.com/2/37/MQL5-avatar-doeasy__5.png)[Library for easy and quick development of MetaTrader programs (part XXIII): Base trading class - verification of valid parameters](https://www.mql5.com/en/articles/7286)

In the article, we continue the development of the trading class by implementing the control over incorrect trading order parameter values and voicing trading events.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/7326&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070446173040350712)

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