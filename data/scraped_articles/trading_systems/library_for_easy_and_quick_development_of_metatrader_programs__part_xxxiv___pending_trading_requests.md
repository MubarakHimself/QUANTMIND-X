---
title: Library for easy and quick development of MetaTrader programs (part XXXIV): Pending trading requests - removing and modifying orders and positions under certain conditions
url: https://www.mql5.com/en/articles/7569
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:36:39.178573
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/7569&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070423882160084372)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7569#node01)
- [Implementation](https://www.mql5.com/en/articles/7569#node02)
- [Testing](https://www.mql5.com/en/articles/7569#node03)
- [What's next?](https://www.mql5.com/en/articles/7569#node04)


### Concept

This article completes the section dedicated to trading using pending requests. We are going to develop the functionality for removing
pending orders, as well as for modifying StopLoss and TakeProfit levels and pending order parameters.

Thus, we are going to have the entire functionality enabling us to develop simple custom strategies, or rather EA behavior logic activated upon
user-defined conditions. After the graphical shell is ready, this preparatory work will provide us with the appropriate tools, for
example, for creating a visual constructor of the EA behavior directly from the EA itself during its work (probably, I will make a simple
example after the graphical shell of the library is ready).

At the moment, we are able to create additional types of pending orders. For example, it is possible to create a StopLimit order for MQL4. I
will do that after I have enough library functionality.

As a result, it will be possible to create completely new types of pending
orders, such as BuyTime, SellTime, BuyTimeStop, SellTimeStop, etc.

Some graphical constructions are still missing preventing us
from creating full-fledged custom orders. We will get back to that task after we have the appropriate library functionality.

In the
meantime, I am going to fix some shortcomings in the descendant classes of the abstract pending request object and create the missing
trading functionality planned for now.

### Implementation

As it turns out, we do not have the function that simply displays a pending order name. But we have the OrderTypeDescription() function
displaying its description+name. This means we should simply remove the description text from the function return result leaving only the
order name.

Let's improve the function returning the order name in the \\MQL5\\Include\\DoEasy\ **Services\\DELib.mqh**
file of service functions:

```
//+------------------------------------------------------------------+
//| Return the order name                                            |
//+------------------------------------------------------------------+
string OrderTypeDescription(const ENUM_ORDER_TYPE type,bool as_order=true,bool prefix_for_market_order=true,bool descr=true)
  {
   string pref=
     (
      !prefix_for_market_order ? "" :
      #ifdef __MQL5__ CMessage::Text(MSG_ORD_MARKET)
      #else/*__MQL4__*/(as_order ? CMessage::Text(MSG_ORD_MARKET) : CMessage::Text(MSG_ORD_POSITION)) #endif
     );
   return
     (
      type==ORDER_TYPE_BUY_LIMIT       ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Buy Limit"       :
      type==ORDER_TYPE_BUY_STOP        ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Buy Stop"        :
      type==ORDER_TYPE_SELL_LIMIT      ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Sell Limit"      :
      type==ORDER_TYPE_SELL_STOP       ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Sell Stop"       :
   #ifdef __MQL5__
      type==ORDER_TYPE_BUY_STOP_LIMIT  ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Buy Stop Limit"  :
      type==ORDER_TYPE_SELL_STOP_LIMIT ?  (descr ? CMessage::Text(MSG_ORD_PENDING) : "")+" Sell Stop Limit" :
      type==ORDER_TYPE_CLOSE_BY        ?  CMessage::Text(MSG_ORD_CLOSE_BY)                                  :
   #else
      type==ORDER_TYPE_BALANCE         ?  CMessage::Text(MSG_LIB_PROP_BALANCE)                              :
      type==ORDER_TYPE_CREDIT          ?  CMessage::Text(MSG_LIB_PROP_CREDIT)                               :
   #endif
      type==ORDER_TYPE_BUY             ?  pref+" Buy"                                                       :
      type==ORDER_TYPE_SELL            ?  pref+" Sell"                                                      :
      CMessage::Text(MSG_ORD_UNKNOWN_TYPE)
     );
  }
//+------------------------------------------------------------------+
```

The function always returns the "pending order" text before
the pending order type.

The flag indicating the necessity
to display the "pending order" text has been implemented so that the function is able to display the order type description without
the preliminary text. Thus, if the flag is disabled (the value is false), the "pending
order" preliminary text is not displayed.

In all files of the base pending request descendant objects, namely in the methods of displaying a brief request name, fix the display
of the brief description of a pending request — add order/position type
to the request description text. After that, add the ticket (if
available in the class) followed by a pending request ID separated by
comma. The current structure is not perfect since a request description is displayed first followed by the ID and the ticket.

**The changes in the pending request object class for opening a position:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqOpen::Header(void)
  {
   string type=PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN)+" "+type+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The changes in the pending request object class for closing a position:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqClose::Header(void)
  {
   string type=PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE)+" "+type+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The changes in the pending request object class for modifying position's StopLoss and/or TakeProfit levels:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqSLTP::Header(void)
  {
   string type=PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP)+" "+type+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The changes in the pending request object class for placing a pending order:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqPlace::Header(void)
  {
   string type=OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE),true,false,false);
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE)+type+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The changes in the pending request object class for removing a pending order:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqRemove::Header(void)
  {
   string type=OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE),true,false,false);
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE)+type+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER)+", ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

**The changes in the pending request object class for modifying pending order properties:**

```
//+------------------------------------------------------------------+
//| Return the short request name                                    |
//+------------------------------------------------------------------+
string CPendReqModify::Header(void)
  {
   string type=OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE),true,false,false);
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY)+type+", #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER)+" ID #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
```

In the **TradingControl.mqh** file of the **CTradingControl** trading management class (namely in its public section),
add the declaration of the methods for creating a pending request to remove a
pending order, modify position's StopLoss/TakeProfit
and modify pending order parameters:

```
//--- Create a pending request (1) for full and partial position closure, (2) for closing a position by an opposite one, (3) for removing an order
   int                  CreatePReqClose(const ulong ticket,const double volume=WRONG_VALUE,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  CreatePReqCloseBy(const ulong ticket,const ulong ticket_by);
   int                  CreatePreqDelete(const ulong ticket);

//--- Create a pending request to modify (1) position's stop orders, (2) an order
   template<typename SL,typename TP>
   int                  CreatePReqModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   int                  CreatePReqModifyOrder(const ulong ticket,
                                              const PS price=WRONG_VALUE,
                                              const SL sl=WRONG_VALUE,
                                              const TP tp=WRONG_VALUE,
                                              const PL limit=WRONG_VALUE,
                                              datetime expiration=WRONG_VALUE,
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

Implement them beyond the class body:

```
//+------------------------------------------------------------------+
//| Create a pending request to remove a pending order               |
//+------------------------------------------------------------------+
int CTradingControl::CreatePreqDelete(const ulong ticket)
  {
//--- If the global trading ban flag is set, exit and return WRONG_VALUE
   if(this.IsTradingDisable())
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return WRONG_VALUE;
     }
//--- Set the error flag as "no errors"
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_CLOSE;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return WRONG_VALUE;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by an order ticket
   CSymbol *symbol_obj=this.GetSymbolObjByOrder(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return WRONG_VALUE;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return WRONG_VALUE;
     }
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return WRONG_VALUE;
     }

//--- Look for the least of the possible IDs. If failed to find, return WRONG_VALUE
   int id=this.GetFreeID();
   if(id<1)
     {
      //--- No free IDs to create a pending request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_NO_FREE_IDS));
      return WRONG_VALUE;
     }

   //--- Set the trading operation type, as well as deleted order's symbol and ticket in the request structure
   this.m_request.action=TRADE_ACTION_REMOVE;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.order=ticket;
   this.m_request.type=order_type;
   this.m_request.volume=order.Volume();
   this.m_request.price=order.PriceOpen();
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_REMOVE,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,order))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
//| Create a pending request to modify position's stop orders        |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CTradingControl::CreatePReqModifyPosition(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE)
  {
//--- If the global trading ban flag is set, exit and return WRONG_VALUE
   if(this.IsTradingDisable())
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return WRONG_VALUE;
     }
//--- Set the error flag as "no errors"
   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ACTION_TYPE action=ACTION_TYPE_MODIFY;
//--- Get an order object by ticket
   COrder *order=this.GetOrderObjByTicket(ticket);
   if(order==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_ORD_OBJ));
      return WRONG_VALUE;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return WRONG_VALUE;
     }
//--- Get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return WRONG_VALUE;
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
      return WRONG_VALUE;
     }
//--- Look for the least of the possible IDs. If failed to find, return 'false'
   int id=this.GetFreeID();
   if(id<1)
      return WRONG_VALUE;

//--- Write a type of a conducted operation, as well as a symbol and a ticket of a modified position to the request structure
   this.m_request.action=TRADE_ACTION_SLTP;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.position=ticket;
   this.m_request.type=order_type;
   this.m_request.volume=order.Volume();
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_SLTP,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,order))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
//| Create a pending request to modify a pending order               |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
int CTradingControl::CreatePReqModifyOrder(const ulong ticket,
                                           const PS price=WRONG_VALUE,
                                           const SL sl=WRONG_VALUE,
                                           const TP tp=WRONG_VALUE,
                                           const PL limit=WRONG_VALUE,
                                           datetime expiration=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
//--- If the global trading ban flag is set, exit and return WRONG_VALUE
   if(this.IsTradingDisable())
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return WRONG_VALUE;
     }
//--- Set the error flag as "no errors"
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
//--- Look for the least of the possible IDs. If failed to find, return 'false'
   int id=this.GetFreeID();
   if(id<1)
      return WRONG_VALUE;

//--- Write the magic number, volume, filling type, as well as expiration date and type to the request structure
   this.m_request.magic=order.GetMagicID((uint)order.Magic());
   this.m_request.volume=order.Volume();
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : order.TypeFilling());
   this.m_request.expiration=(expiration>WRONG_VALUE ? expiration : order.TimeExpiration());
   this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : order.TypeTime());

   //--- Set the trading operation type, as well as modified order's symbol and ticket in the request structure
   this.m_request.action=TRADE_ACTION_MODIFY;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.order=ticket;
   this.m_request.type=order_type;
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_MODIFY,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,order))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The logic of methods is absolutely identical to the previously created methods of generating pending requests for opening/closing
positions/placing pending orders under certain conditions. The code is commented in details, so there is no point in dwelling on these
methods again.

Now let's add access from the program to the created methods. To do this, write the call of these methods from the library base object class
methods.

In the **CEngine** class, add the declaration of methods for creating pending requests to
remove a pending order, modify position's StopLoss/TakeProfit
and modify the pending order parameters:

```
//--- Create a pending request for closing a position (1) fully, (2) partially, (3) by an opposite one, (4) for removing an order
   int                  ClosePositionPending(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  ClosePositionPartiallyPending(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  ClosePositionByPending(const ulong ticket,const ulong ticket_by);
   int                  DeleteOrderPending(const ulong ticket);

//--- Create a pending request to modify (1) a position, (2) an order
   template<typename SL,typename TP>
   int                  ModifyPositionPending(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE,const string comment=NULL);
   template<typename PR,typename SL,typename TP,typename PL>
   int                  ModifyOrderPending(const ulong ticket,
                                           const PR price=WRONG_VALUE,
                                           const SL sl=WRONG_VALUE,
                                           const TP tp=WRONG_VALUE,
                                           const PL stoplimit=WRONG_VALUE,
                                           datetime expiration=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                           const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

```

Implement declared methods outside the class body:

```
//+------------------------------------------------------------------+
//| Create a pending request to remove a pending order               |
//+------------------------------------------------------------------+
int CEngine::DeleteOrderPending(const ulong ticket)
  {
   return this.m_trading.CreatePreqDelete(ticket);
  }
//+------------------------------------------------------------------+
//| Create a pending request to modify a position                    |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CEngine::ModifyPositionPending(const ulong ticket,const SL sl=WRONG_VALUE,const TP tp=WRONG_VALUE,const string comment=NULL)
  {
   return this.m_trading.CreatePReqModifyPosition(ticket,sl,tp);
  }
//+------------------------------------------------------------------+
//| Create a pending request to modify an order                      |
//+------------------------------------------------------------------+
template<typename PR,typename SL,typename TP,typename PL>
int CEngine::ModifyOrderPending(const ulong ticket,
                                const PR price=WRONG_VALUE,
                                const SL sl=WRONG_VALUE,
                                const TP tp=WRONG_VALUE,
                                const PL stoplimit=WRONG_VALUE,
                                datetime expiration=WRONG_VALUE,
                                const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqModifyOrder(ticket,price,sl,tp,stoplimit,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
```

The methods simply return the result of calling the appropriate methods of creating pending requests of the CTradingControl class we wrote
before.

**These are all the improvements required for adding the functionality for working with pending requests to remove orders and modify orders**
**and positions.**

Also, some minor changes have been made in the codes of the improved classes (names of template parameters).
However, they are related only to the visual perception of the method code, so there is no point in considering them here.

### Testing

To perform the created functionality, use the [EA from the previous \\
article](https://www.mql5.com/en/articles/7554#node03) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part34\** under the name **TestDoEasyPart34.mq5**.

Like the previous EAs intended to test pending requests, we will create the buttons for activating pending request modes — for the buttons
removing all pending orders (Delete pending), closing all positions (Close all) and setting StopLoss and TakeProfit for orders and
positions having no stop levels (Set StopLoss and Set TakeProfit).

Since pressing these buttons leads to the batch handling of all the existing orders and positions, enabling the appropriate activation
buttons allows us to check pending requests for multiple orders and positions at once.

In the current version, the trading management buttons of the test EA's trading panel are not very convenient — the button of withdrawing
earned funds is located between the buttons of removing orders and closing positions and the buttons for placing stop orders. Let's
set this button lower — after the Set TakeProfit button.
To do this, simply change the location of constants in the enumeration of all buttons of the test EA's trading panel:

```
//+------------------------------------------------------------------+
//|                                             TestDoEasyPart34.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//--- includes
#include <DoEasy\Engine.mqh>
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
   BUTT_SET_STOP_LOSS,
   BUTT_SET_TAKE_PROFIT,
   BUTT_PROFIT_WITHDRAWAL,
   BUTT_TRAILING_ALL
  };
#define TOTAL_BUTT   (20)
```

The list of global variables receives the flags indicating states of the buttons
activating modes of working with pending requests to remove pending orders, close positions and modify stop levels of orders and positions,
as well as add two variables for storing Point and Digits values of the current
symbol:

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
bool           pressed_pending_buy;
bool           pressed_pending_buy_limit;
bool           pressed_pending_buy_stop;
bool           pressed_pending_buy_stoplimit;
bool           pressed_pending_close_buy;
bool           pressed_pending_close_buy2;
bool           pressed_pending_close_buy_by_sell;
bool           pressed_pending_sell;
bool           pressed_pending_sell_limit;
bool           pressed_pending_sell_stop;
bool           pressed_pending_sell_stoplimit;
bool           pressed_pending_close_sell;
bool           pressed_pending_close_sell2;
bool           pressed_pending_close_sell_by_buy;
bool           pressed_pending_delete_all;
bool           pressed_pending_close_all;
bool           pressed_pending_sl;
bool           pressed_pending_tp;
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
double         g_point;
int            g_digits;
//+------------------------------------------------------------------+
```

In the EA's **OnInit()** handler, assign Point and Digits values of the
current symbol to the corresponding variables:

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
   g_point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   g_digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
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

In the button generation function, improve the code to
create additional buttons for activating the mode of working with pending requests:

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
   for(int i=0;i<18;i++)
     {
      y=(cy-(i-(i>6 ? 7 : 0))*(h+1));
      if(!ButtonCreate(butt_data[i].name+"_PRICE",((i>6 && i<14) || i>17 ? x+wpt*2+w*2+5 : x),y,wpt,h,"P",(i<4 ? clrGreen : i>6 && i<11 ? clrChocolate : clrBlue)))
        {
         Alert(TextByLanguage("Не удалось создать кнопку \"","Could not create button \""),butt_data[i].text+" \"P\"");
         return false;
        }
      if(!ButtonCreate(butt_data[i].name+"_TIME",((i>6 && i<14) || i>17 ? x+wpt*2+w*2+5+wpt+1 : x+wpt+1),y,wpt,h,"T",(i<4 ? clrGreen : i>6 && i<11 ? clrChocolate : clrBlue)))
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

All the changes affect only the management of a serial number
of a created button to set the necessary coordinates.

The main changes in the EA are related to the function for handling pressing of the EA trading panel buttons.


Add the codes for handling pressed buttons of the trading panel:

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
         if(!pressed_pending_buy)
            engine.OpenBuy(lot,Symbol(),magic,stoploss,takeprofit);   // No comment - the default comment is to be set
         //--- Otherwise, create a pending request for opening a Buy position
         else
           {
            int id=engine.OpenBuyPending(lot,Symbol(),magic,stoploss,takeprofit);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_BUY,EQUAL_OR_LESS,ask,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set BuyLimit
         if(!pressed_pending_buy_limit)
            engine.PlaceBuyLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyLimit","Pending BuyLimit order"));
         //--- Otherwise, create a pending request to place a BuyLimit order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceBuyLimitPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_BUY_LIMIT,EQUAL_OR_LESS,ask,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStop
         if(!pressed_pending_buy_stop)
            engine.PlaceBuyStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
         //--- Otherwise, create a pending request to place a BuyStop order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceBuyStopPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_BUY_STOP,EQUAL_OR_LESS,ask,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStopLimit
         if(!pressed_pending_buy_stoplimit)
            engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
         //--- Otherwise, create a pending request to place a BuyStopLimit order with the placement distances
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceBuyStopLimitPending(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_BUY_STOP_LIMIT,EQUAL_OR_LESS,ask,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_SELL button is pressed: Open Sell position
      else if(button==EnumToString(BUTT_SELL))
        {
         //--- If the pending request creation buttons are not pressed, open Sell
         if(!pressed_pending_sell)
            engine.OpenSell(lot,Symbol(),magic,stoploss,takeprofit);  // No comment - the default comment is to be set
         //--- Otherwise, create a pending request for opening a Sell position
         else
           {
            int id=engine.OpenSellPending(lot,Symbol(),magic,stoploss,takeprofit);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SELL,EQUAL_OR_MORE,bid,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_SELL_LIMIT button is pressed: Set SellLimit
      else if(button==EnumToString(BUTT_SELL_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set SellLimit
         if(!pressed_pending_sell_limit)
            engine.PlaceSellLimit(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellLimit","Pending SellLimit order"));
         //--- Otherwise, create a pending request to place a SellLimit order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceSellLimitPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SELL_LIMIT,EQUAL_OR_MORE,bid,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set SellStop
         if(!pressed_pending_sell_stop)
            engine.PlaceSellStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
         //--- Otherwise, create a pending request to place a SellStop order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceSellStopPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SELL_STOP,EQUAL_OR_MORE,bid,TimeCurrent());
              }
           }
        }
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set SellStopLimit
         if(!pressed_pending_sell_stoplimit)
            engine.PlaceSellStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
         //--- Otherwise, create a pending request to place a SellStopLimit order with the placement distances
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceSellStopLimitPending(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
               ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
               SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SELL_STOP_LIMIT,EQUAL_OR_MORE,bid,TimeCurrent());
              }
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
              {
               //--- If the pending request creation buttons are not pressed, close a position
               if(!pressed_pending_close_buy)
                  engine.ClosePosition((ulong)position.Ticket());
               //--- Otherwise, create a pending request for closing a position by ticket
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.ClosePositionPending(position.Ticket());
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                     double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_BUY,EQUAL_OR_MORE,bid,TimeCurrent());
                    }
                 }
              }
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
            //--- Get the Buy position object and close a position by ticket
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- If the pending request creation buttons are not pressed, close a position by ticket
               if(!pressed_pending_close_buy2)
                  engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
               //--- Otherwise, create a pending request for closing a position partially by ticket
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.ClosePositionPartiallyPending(position.Ticket(),position.Volume()/2.0);
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                     double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_BUY2,EQUAL_OR_MORE,bid,TimeCurrent());
                    }
                 }
              }
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
               if(position_buy!=NULL && position_sell!=NULL)
                 {
                  //--- If the pending request creation buttons are not pressed, close positions by ticket
                  if(!pressed_pending_close_buy_by_sell)
                     engine.ClosePositionBy((ulong)position_buy.Ticket(),(ulong)position_sell.Ticket());
                  //--- Otherwise, create a pending request for closing a Buy position by an opposite Sell one
                  //--- and set the conditions depending on active buttons
                  else
                    {
                     int id=engine.ClosePositionByPending(position_buy.Ticket(),position_sell.Ticket());
                     if(id>0)
                       {
                        //--- set the pending request activation price and time, as well as activation parameters
                        double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                        double price_activation=NormalizeDouble(bid+distance_pending_request*g_point,g_digits);
                        ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                        SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_BUY_BY_SELL,EQUAL_OR_MORE,bid,TimeCurrent());
                       }
                    }
                 }
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
              {
               //--- If the pending request creation buttons are not pressed, close a position
               if(!pressed_pending_close_sell)
                  engine.ClosePosition((ulong)position.Ticket());
               //--- Otherwise, create a pending request for closing a position by ticket
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.ClosePositionPending(position.Ticket());
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                     double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_SELL,EQUAL_OR_LESS,ask,TimeCurrent());
                    }
                 }
              }
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
            //--- Get the Sell position object and close a position by ticket
            COrder* position=list.At(index);
            if(position!=NULL)
              {
               //--- If the pending request creation buttons are not pressed, close a position by ticket
               if(!pressed_pending_close_sell2)
                  engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
               //--- Otherwise, create a pending request for closing a position partially by ticket
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.ClosePositionPartiallyPending(position.Ticket(),position.Volume()/2.0);
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                     double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_SELL2,EQUAL_OR_LESS,ask,TimeCurrent());
                    }
                 }
              }
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
               if(position_sell!=NULL && position_buy!=NULL)
                 {
                  //--- If the pending request creation buttons are not pressed, close positions by ticket
                  if(!pressed_pending_close_sell_by_buy)
                     engine.ClosePositionBy((ulong)position_sell.Ticket(),(ulong)position_buy.Ticket());
                  //--- Otherwise, create a pending request for closing a Sell position by an opposite Buy one
                  //--- and set the conditions depending on active buttons
                  else
                    {
                     int id=engine.ClosePositionByPending(position_sell.Ticket(),position_buy.Ticket());
                     if(id>0)
                       {
                        //--- set the pending request activation price and time, as well as activation parameters
                        double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                        double price_activation=NormalizeDouble(ask-distance_pending_request*g_point,g_digits);
                        ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                        SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_SELL_BY_BUY,EQUAL_OR_LESS,ask,TimeCurrent());
                       }
                    }
                 }
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
               //--- If the pending request creation buttons are not pressed, close each position by its ticket
               if(!pressed_pending_close_all)
                  engine.ClosePosition((ulong)position.Ticket());
               //--- Otherwise, create a pending request for closing each position
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.ClosePositionPending(position.Ticket());
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double price=SymbolInfoDouble(NULL,SYMBOL_BID);
                     double price_activation=NormalizeDouble(position.PriceOpen()+distance_pending_request*g_point,g_digits);
                     ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
                     if(position.TypeOrder()==POSITION_TYPE_SELL)
                       {
                        price=SymbolInfoDouble(NULL,SYMBOL_ASK);
                        price_activation=NormalizeDouble(position.PriceOpen()-distance_pending_request*g_point,g_digits);
                        comparer=EQUAL_OR_LESS;
                       }
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_CLOSE_ALL,comparer,price,TimeCurrent());
                    }
                 }
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
               //--- If the pending request creation buttons are not pressed, remove each order by its ticket
               if(!pressed_pending_delete_all)
                  engine.DeleteOrder((ulong)order.Ticket());
               //--- Otherwise, create a pending request for removing each order
               //--- and set the conditions depending on active buttons
               else
                 {
                  int id=engine.DeleteOrderPending(order.Ticket());
                  if(id>0)
                    {
                     //--- set the pending request activation price and time, as well as activation parameters
                     double price=SymbolInfoDouble(NULL,SYMBOL_ASK);
                     double price_activation=NormalizeDouble(order.PriceOpen()+(distance_pending+distance_pending_request)*g_point,g_digits);
                     ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
                     if(order.TypeByDirection()==ORDER_TYPE_SELL)
                       {
                        price=SymbolInfoDouble(NULL,SYMBOL_BID);
                        price_activation=NormalizeDouble(order.PriceOpen()-(distance_pending+distance_pending_request)*g_point,g_digits);
                        comparer=EQUAL_OR_LESS;
                       }
                     ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                     SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_DELETE_PENDING,comparer,price,TimeCurrent());
                    }
                 }
              }
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
            pressed_pending_buy=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyLimit by price or time
         if(button==EnumToString(BUTT_BUY_LIMIT)+"_PRICE" || button==EnumToString(BUTT_BUY_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_buy_limit=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyStop by price or time
         if(button==EnumToString(BUTT_BUY_STOP)+"_PRICE" || button==EnumToString(BUTT_BUY_STOP)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_buy_stop=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing BuyStopLimit by price or time
         if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE" || button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_buy_stoplimit=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_buy=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing 1/2 Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY2)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY2)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_buy2=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Buy by an opposite Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE" || button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_buy_by_sell=true;
           }

         //--- Selling
         //--- Set the active button color for the button enabling pending requests for opening Sell by price or time
         if(button==EnumToString(BUTT_SELL)+"_PRICE" || button==EnumToString(BUTT_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_sell=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing SellLimit by price or time
         if(button==EnumToString(BUTT_SELL_LIMIT)+"_PRICE" || button==EnumToString(BUTT_SELL_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_sell_limit=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing SellStop by price or time
         if(button==EnumToString(BUTT_SELL_STOP)+"_PRICE" || button==EnumToString(BUTT_SELL_STOP)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_sell_stop=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing SellStopLimit by price or time
         if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE" || button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_sell_stoplimit=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_sell=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing 1/2 Sell by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL2)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL2)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_sell2=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing Sell by an opposite Buy by price or time
         if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE" || button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_sell_by_buy=true;
           }
         //--- Set the active button color for the button enabling pending requests for removing orders by price or time
         if(button==EnumToString(BUTT_DELETE_PENDING)+"_PRICE" || button==EnumToString(BUTT_DELETE_PENDING)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_delete_all=true;
           }
         //--- Set the active button color for the button enabling pending requests for closing positions by price or time
         if(button==EnumToString(BUTT_CLOSE_ALL)+"_PRICE" || button==EnumToString(BUTT_CLOSE_ALL)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_close_all=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing StopLoss by price or time
         if(button==EnumToString(BUTT_SET_STOP_LOSS)+"_PRICE" || button==EnumToString(BUTT_SET_STOP_LOSS)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_sl=true;
           }
         //--- Set the active button color for the button enabling pending requests for placing TakeProfit by price or time
         if(button==EnumToString(BUTT_SET_TAKE_PROFIT)+"_PRICE" || button==EnumToString(BUTT_SET_TAKE_PROFIT)+"_TIME")
           {
            ButtonState(button_name,true);
            pressed_pending_tp=true;
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
         pressed_pending_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for opening Buy by time
      if(button==EnumToString(BUTT_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyLimit by price
      if(button==EnumToString(BUTT_BUY_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyLimit by time
      if(button==EnumToString(BUTT_BUY_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyStop by price
      if(button==EnumToString(BUTT_BUY_STOP)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyStop by time
      if(button==EnumToString(BUTT_BUY_STOP)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing BuyStopLimit by price
      if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing BuyStopLimit by time
      if(button==EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_buy_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Buy by price
      if(button==EnumToString(BUTT_CLOSE_BUY)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Buy by time
      if(button==EnumToString(BUTT_CLOSE_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing 1/2 Buy by price
      if(button==EnumToString(BUTT_CLOSE_BUY2)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY2)+"_TIME"));
        }
      //--- the button enabling pending requests for closing 1/2 Buy by time
      if(button==EnumToString(BUTT_CLOSE_BUY2)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY2)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Buy by an opposite Sell by price
      if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy_by_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Buy by an opposite Sell by time
      if(button==EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_buy_by_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_BUY_BY_SELL)+"_PRICE"));
        }

      //--- Selling
      //--- the button enabling pending requests for opening Sell by price
      if(button==EnumToString(BUTT_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for opening Sell by time
      if(button==EnumToString(BUTT_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellLimit by price
      if(button==EnumToString(BUTT_SELL_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellLimit by time
      if(button==EnumToString(BUTT_SELL_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_limit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellStop by price
      if(button==EnumToString(BUTT_SELL_STOP)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellStop by time
      if(button==EnumToString(BUTT_SELL_STOP)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_stop=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing SellStopLimit by price
      if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing SellStopLimit by time
      if(button==EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_sell_stoplimit=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Sell by price
      if(button==EnumToString(BUTT_CLOSE_SELL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Sell by time
      if(button==EnumToString(BUTT_CLOSE_SELL)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing 1/2 Sell by price
      if(button==EnumToString(BUTT_CLOSE_SELL2)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL2)+"_TIME"));
        }
      //--- the button enabling pending requests for closing 1/2 Sell by time
      if(button==EnumToString(BUTT_CLOSE_SELL2)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell2=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL2)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing Sell by an opposite Buy by price
      if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell_by_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME"));
        }
      //--- the button enabling pending requests for closing Sell by an opposite Buy by time
      if(button==EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_sell_by_buy=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_SELL_BY_BUY)+"_PRICE"));
        }

      //--- the button enabling pending requests for removing orders by price
      if(button==EnumToString(BUTT_DELETE_PENDING)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_delete_all=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_DELETE_PENDING)+"_TIME"));
        }
      //--- the button enabling pending requests for removing orders by time
      if(button==EnumToString(BUTT_DELETE_PENDING)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_delete_all=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_DELETE_PENDING)+"_PRICE"));
        }

      //--- the button enabling pending requests for closing positions by price
      if(button==EnumToString(BUTT_CLOSE_ALL)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_close_all=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_ALL)+"_TIME"));
        }
      //--- the button enabling pending requests for closing positions by time
      if(button==EnumToString(BUTT_CLOSE_ALL)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_close_all=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_CLOSE_ALL)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing StopLoss by price
      if(button==EnumToString(BUTT_SET_STOP_LOSS)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_sl=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SET_STOP_LOSS)+"_TIME"));
        }
      //--- the button enabling pending requests for placing StopLoss by time
      if(button==EnumToString(BUTT_SET_STOP_LOSS)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_sl=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SET_STOP_LOSS)+"_PRICE"));
        }

      //--- the button enabling pending requests for placing TakeProfit by price
      if(button==EnumToString(BUTT_SET_TAKE_PROFIT)+"_PRICE")
        {
         ButtonState(button_name,false);
         pressed_pending_tp=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SET_TAKE_PROFIT)+"_TIME"));
        }
      //--- the button enabling pending requests for placing TakeProfit by time
      if(button==EnumToString(BUTT_SET_TAKE_PROFIT)+"_TIME")
        {
         ButtonState(button_name,false);
         pressed_pending_tp=(ButtonState(button_name) | ButtonState(prefix+EnumToString(BUTT_SET_TAKE_PROFIT)+"_PRICE"));
        }
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

In the functions for placing StopLoss and TakeProfit,
add the code blocks for creating pending requests to set StopLoss/TakeProfit
to all orders and positions:

```
//+------------------------------------------------------------------+
//| Set StopLoss to all orders and positions                         |
//+------------------------------------------------------------------+
void SetStopLoss(void)
  {
   if(stoploss_to_modify==0)
      return;
//--- Set StopLoss to all positions where it is absent
   //--- Get the list of all positions
   CArrayObj* list=engine.GetListMarketPosition();
   //--- Select only current symbol positions from the list
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
   //--- select positions with zero StopLoss from the list
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
      //--- If the pending request creation buttons are not pressed, set StopLoss for each position by its ticket
      if(!pressed_pending_sl)
         engine.ModifyPosition((ulong)position.Ticket(),sl,-1);
      //--- Otherwise, create a pending request for setting StopLoss for each position
      //--- and set the conditions depending on active buttons
      else
        {
         int id=engine.ModifyPositionPending(position.Ticket(),sl,-1);
         if(id>0)
           {
            //--- set the pending request activation price and time, as well as activation parameters
            double price=SymbolInfoDouble(NULL,SYMBOL_BID);
            double price_activation=NormalizeDouble(position.PriceOpen()+distance_pending_request*g_point,g_digits);
            ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
            if(position.TypeByDirection()==ORDER_TYPE_SELL)
              {
               price=SymbolInfoDouble(NULL,SYMBOL_ASK);
               price_activation=NormalizeDouble(position.PriceOpen()-distance_pending_request*g_point,g_digits);
               comparer=EQUAL_OR_LESS;
              }
            ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
            SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SET_STOP_LOSS,comparer,price,TimeCurrent());
           }
        }
     }
//--- Set StopLoss to all pending orders where it is absent
   //--- Get the list of all orders
   list=engine.GetListMarketPendings();
   //--- Select only current symbol positions from the list
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
   //--- select orders with zero StopLoss from the list
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
      //--- If the pending request creation buttons are not pressed, set StopLoss for each order by its ticket
      if(!pressed_pending_sl)
         engine.ModifyOrder((ulong)order.Ticket(),-1,sl,-1,-1);
      //--- Otherwise, create a pending request for setting StopLoss for each order
      //--- and set the conditions depending on active buttons
      else
        {
         int id=engine.ModifyOrderPending(order.Ticket(),-1,sl,-1,-1);
         if(id>0)
           {
            //--- set the pending request activation price and time, as well as activation parameters
            double price=SymbolInfoDouble(NULL,SYMBOL_ASK);
            double price_activation=NormalizeDouble(order.PriceOpen()+(distance_pending+distance_pending_request)*g_point,g_digits);
            ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
            if(order.TypeByDirection()==ORDER_TYPE_SELL)
              {
               price=SymbolInfoDouble(NULL,SYMBOL_BID);
               price_activation=NormalizeDouble(order.PriceOpen()-(distance_pending+distance_pending_request)*g_point,g_digits);
               comparer=EQUAL_OR_LESS;
              }
            ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
            SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SET_STOP_LOSS,comparer,price,TimeCurrent());
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Set TakeProfit to all orders and positions                       |
//+------------------------------------------------------------------+
void SetTakeProfit(void)
  {
   if(takeprofit_to_modify==0)
      return;
//--- Set TakeProfit to all positions where it is absent
   //--- Get the list of all positions
   CArrayObj* list=engine.GetListMarketPosition();
   //--- Select only current symbol positions from the list
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
   //--- select positions with zero TakeProfit from the list
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
      //--- If the pending request creation buttons are not pressed, set TakeProfit for each position by its ticket
      if(!pressed_pending_tp)
         engine.ModifyPosition((ulong)position.Ticket(),-1,tp);
      //--- Otherwise, create a pending request for setting TakeProfit for each position
      //--- and set the conditions depending on active buttons
      else
        {
         int id=engine.ModifyPositionPending(position.Ticket(),-1,tp);
         if(id>0)
           {
            //--- set the pending request activation price and time, as well as activation parameters
            double price=SymbolInfoDouble(NULL,SYMBOL_BID);
            double price_activation=NormalizeDouble(position.PriceOpen()+distance_pending_request*g_point,g_digits);
            ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
            if(position.TypeByDirection()==ORDER_TYPE_SELL)
              {
               price=SymbolInfoDouble(NULL,SYMBOL_ASK);
               price_activation=NormalizeDouble(position.PriceOpen()-distance_pending_request*g_point,g_digits);
               comparer=EQUAL_OR_LESS;
              }
            ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
            SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SET_TAKE_PROFIT,comparer,price,TimeCurrent());
           }
        }
     }
//--- Set TakeProfit to all pending orders where it is absent
   //--- Get the list of all orders
   list=engine.GetListMarketPendings();
   //--- Select only current symbol orders from the list
   list=CSelect::ByOrderProperty(list,ORDER_PROP_SYMBOL,Symbol(),EQUAL);
   //--- select orders with zero TakeProfit from the list
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
      //--- If the pending request creation buttons are not pressed, set TakeProfit for each order by its ticket
      if(!pressed_pending_sl)
         engine.ModifyOrder((ulong)order.Ticket(),-1,-1,tp,-1);
      //--- Otherwise, create a pending request for setting TakeProfit for each order
      //--- and set the conditions depending on active buttons
      else
        {
         int id=engine.ModifyOrderPending(order.Ticket(),-1,-1,tp,-1);
         if(id>0)
           {
            //--- set the pending request activation price and time, as well as activation parameters
            double price=SymbolInfoDouble(NULL,SYMBOL_ASK);
            double price_activation=NormalizeDouble(order.PriceOpen()+(distance_pending+distance_pending_request)*g_point,g_digits);
            ENUM_COMPARER_TYPE comparer=EQUAL_OR_MORE;
            if(order.TypeByDirection()==ORDER_TYPE_SELL)
              {
               price=SymbolInfoDouble(NULL,SYMBOL_BID);
               price_activation=NormalizeDouble(order.PriceOpen()-(distance_pending+distance_pending_request)*g_point,g_digits);
               comparer=EQUAL_OR_LESS;
              }
            ulong  time_activation=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
            SetPReqCriterion((uchar)id,price_activation,time_activation,BUTT_SET_TAKE_PROFIT,comparer,price,TimeCurrent());
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

The code blocks for creating pending requests are similar in all considered functions in terms of logic. They are commented in detail in the
code, so leave them for independent study. If you have any questions, feel free to ask them in the comments.

Compile the EA and launch it in the tester in the visualization mode. To check order removal as well as order and position modification, open two
sell positions and place a sell pending order without StopLoss and TakeProfit levels. Next, create pending requests to modify stop levels
of orders and positions by price value. Wait for activation of pending requests and placing specified stop levels, and remove orders and
positions.

Then open two buy positions and place a buy pending order. After that, create pending requests to remove orders and close positions by time.

![](https://c.mql5.com/2/38/AmDTF64738.gif)

As we can see, stop levels were set at the intersection of a given pending request activation price level. The positions were closed after a
specified time and the order was removed.

The work is not polished yet. There are issues with simultaneous creation of several pending requests for the same ticket as these
requests do not always work out correctly. Currently, the logic works correctly only if there is one pending request for each position
or order. After a pending request is activated, executed and removed, it is possible to create a new pending request for this position or
order (if they are still active).

I plan to fix this issue gradually along with the further development of the library functionality as soon as I already have some graphical
library objects.

### What's next?

In the next article, we will start the development of the library functionality for storing, handling and receiving price data.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7569#node00)

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
26\. Working with pending trading requests - First implementation (opening positions)](https://www.mql5.com/en/articles/7394)

[Part \\
27\. Working with pending trading requests - Placing pending orders](https://www.mql5.com/en/articles/7418)

[Part 28. \\
Working with pending trading requests - Closure, removal and modification](https://www.mql5.com/en/articles/7438)

[Part \\
29\. Working with pending trading requests - request object classes](https://www.mql5.com/en/articles/7454)

[Part 30. \\
Pending trading requests - managing request objects](https://www.mql5.com/en/articles/7481)

[Part 31. Pending trading \\
requests - opening positions under certain conditions](https://www.mql5.com/en/articles/7521)

[Part 32. Pending trading \\
requests - placing pending orders under certain conditions](https://www.mql5.com/en/articles/7536)

[Part 33. Pending \\
trading requests - closing positions (full, partial or by an opposite one) under certain conditions](https://www.mql5.com/en/articles/7554)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7569](https://www.mql5.com/ru/articles/7569)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7569.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7569/mql5.zip "Download MQL5.zip")(3666.8 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7569/mql4.zip "Download MQL4.zip")(3666.79 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/339754)**
(30)


![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
19 Aug 2022 at 13:56

**Artyom Trishkin [#](https://www.mql5.com/ru/forum/332197#comment_41488639):**

You can disable optimisation at compile time. It will be faster

Yes, it will indeed be faster. But only twice as fast.

On my laptop the compilation time was up to 60 seconds, now, after disabling optimisation, it comes down to 30 seconds. And if we take into account that this is only the 34th part (not the last one), it's still a lot. After all, the library will only grow further....

I thought that it could be easily made as a DLL (for example) and connected to the Expert Advisor. But I tried, and... it's not quite clear how to do it...

![Aliaksandr Hryshyn](https://c.mql5.com/avatar/2016/2/56CF9FD9-71DB.jpg)

**[Aliaksandr Hryshyn](https://www.mql5.com/en/users/greshnik1)**
\|
19 Aug 2022 at 14:04

**MQL\_User [#](https://www.mql5.com/ru/forum/332197#comment_41510034):**

I thought that it could be easily made as a DLL (for example) and connected to the Expert Advisor. But I tried, and... it's not quite clear how to do it...

There are dll analogues, see documentation

[https://www.mql5.com/en/docs/basis/function/export](https://www.mql5.com/en/docs/basis/function/export "https://www.mql5.com/en/docs/basis/function/export")

![MQL_User](https://c.mql5.com/avatar/avatar_na2.png)

**[MQL\_User](https://www.mql5.com/en/users/mql_user)**
\|
20 Aug 2022 at 14:29

**Aliaksandr Hryshyn [#](https://www.mql5.com/ru/forum/332197#comment_41510083):**

There are dll analogues, see documentation

h [ttps://](https://www.mql5.com/en/docs/basis/function/export "https://www.mql5.com/en/docs/basis/function/export") www.mql5.com/ru/docs/basis/function/export

Thanks.

That seems to be exactly what I need. At least theoretically. But in practice, when I try to compile the Engine file with #property library, I can' t export methods (which are essentially functions) - I get an error. It looks like we need to add more functions (exportable) that call these methods. Besides, there will be no tooltips in the programme that imports these functions.

In general, all this is not quite as I would like it to be....

![theonementor](https://c.mql5.com/avatar/2024/6/666AE7F9-6516.png)

**[theonementor](https://www.mql5.com/en/users/theonementor)**
\|
24 Jul 2024 at 20:35

```
The work is not polished yet. There are issues with simultaneous creation of several pending requests for the same ticket as these requests do not always work out correctly. Currently, the logic works correctly only if there is one pending request for each position or order. After a pending request is activated, executed and removed, it is possible to create a new pending request for this position or order (if they are still active).
I plan to fix this issue gradually along with the further development of the library functionality as soon as I already have some graphical library objects.
```

Did you ever polish and fix this issue in future chapters or did you forgot about this when you developed the graphical section? If you fixed and polished can you tell me in which chapter it is?

Thanks

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
25 Jul 2024 at 04:04

**theonementor [#](https://www.mql5.com/en/forum/339754/page2#comment_54087815):**

Did you ever polish and fix this issue in future chapters or did you forgot about this when you developed the graphical section? If you fixed and polished can you tell me in which chapter it is?

Thanks

No, unfortunately I haven't worked on this issue yet

![Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download__1.png)[Applying network functions, or MySQL without DLL: Part II - Program for monitoring changes in signal properties](https://www.mql5.com/en/articles/7495)

In the previous part, we considered the implementation of the MySQL connector. In this article, we will consider its application by implementing the service for collecting signal properties and the program for viewing their changes over time. The implemented example has practical sense if users need to observe changes in properties that are not displayed on the signal's web page.

![Applying network functions, or MySQL without DLL: Part I - Connector](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download.png)[Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)

MetaTrader 5 has received network functions recently. This opened up great opportunities for programmers developing products for the Market. Now they can implement things that required dynamic libraries before. In this article, we will consider them using the implementation of the MySQL as an example.

![Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://c.mql5.com/2/37/Article_Logo__3.png)[Multicurrency monitoring of trading signals (Part 2): Implementation of the visual part of the application](https://www.mql5.com/en/articles/7528)

In the previous article, we created the application framework, which we will use as the basis for all further work. In this part, we will proceed with the development: we will create the visual part of the application and will configure basic interaction of interface elements.

![How to create 3D graphics using DirectX in MetaTrader 5](https://c.mql5.com/2/39/MQL5-avatar-directx_yellow.png)[How to create 3D graphics using DirectX in MetaTrader 5](https://www.mql5.com/en/articles/7708)

3D graphics provide excellent means for analyzing huge amounts of data as they enable the visualization of hidden patterns. These tasks can be solved directly in MQL5, while DireсtX functions allow creating three-dimensional object. Thus, it is even possible to create programs of any complexity, even 3D games for MetaTrader 5. Start learning 3D graphics by drawing simple three-dimensional shapes.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qvkoqoqsweqbclaeftbuispfdzafsysl&ssn=1769186196788102243&ssn_dr=1&ssn_sr=0&fv_date=1769186196&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7569&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXXIV)%3A%20Pending%20trading%20requests%20-%20removing%20and%20modifying%20orders%20and%20positions%20under%20certain%20conditions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918619709032229&fz_uniq=5070423882160084372&sv=2552)

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