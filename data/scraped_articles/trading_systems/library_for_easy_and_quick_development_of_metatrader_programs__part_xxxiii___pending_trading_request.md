---
title: Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions
url: https://www.mql5.com/en/articles/7554
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:36:50.265294
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/7554&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070426218622293407)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7554#node01)
- [Implementation](https://www.mql5.com/en/articles/7554#node02)
- [Testing](https://www.mql5.com/en/articles/7554#node03)
- [What's next?](https://www.mql5.com/en/articles/7554#node04)


We continue the development of the library functionality featuring trading using pending requests. We have already implemented sending
conditional trading requests for [opening positions](https://www.mql5.com/en/articles/7521) and [placing \\
pending orders](https://www.mql5.com/en/articles/7536). Now let's supplement the functionality with the ability to close positions under specified conditions. We are going to
implement three types of closing positions: full closure, partial closure and closure by an opposite position.

### Concept

As we develop the library functionality for trading using pending requests, we gradually identify the bottlenecks of the already complete
functionality, as well as errors and other shortcomings, and fix erroneous methods or invalid logic.

For example, in order to make sure a pending request was already activated and should have been deleted, we checked the last trading event on
the account. If the data set in the pending request object matched the last event, the request was deemed to be complete and it was removed. It
turned out that such logic was not always correct. For example, when closing positions partially using pending requests, when it remains to
close the last part of the open position (the previous closure was on 0.01 lot, while the remaining part is also equal to 0.01 lot), the method
for checking the trading request relevance considered the request to be already activated — its data coincided with the previous closure.

Thinking about how to control this situation, I came to the conclusion that it is easier not to track the time of the event creation, the
corresponding trading request execution time and other parameters, but simply check the last trading event only when the occurred account
trading event is firmly established. Fortunately, we already implemented that long time ago, and we are able to use the method of the event
class returning the flag of a new event present on the account. In such case, we will not confuse the past event with the current one — the check
occurs only at the moment the occurrence of a new event is established (immediately after the occurrence).

Since pending trading
requests may be a part of the trading strategy functionality in the future, it is advisable to have access to all created pending request
objects waiting for their activation conditions. For more convenient selection and sorting of the required objects, we are going to add the
ability to search and sort by pending request object properties located in the list of requests. This makes it possible to select, display,
sort (including the use of GUI) and manage required objects in the program. In other words, you can change, delete and modify them.

### Implementation

In the **PendRequest.mqh** file of the abstract pending request class, namely in its constructor, add
initialization (setting all fields to zero) of the trading request structure:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CPendRequest::CPendRequest(const ENUM_PEND_REQ_STATUS status,
                           const uchar id,
                           const double price,
                           const ulong time,
                           const MqlTradeRequest &request,
                           const int retcode)
  {
   ::ZeroMemory(this.m_request);
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
   this.m_follow=true;
  }
//+------------------------------------------------------------------+
```

Without setting all structure fields to zero, an invalid pending request type was sometimes created, since when creating an object for closing a
position, a pending request object for closing a position by an opposite one is created if the position\_by field has a non-zero value in the
trading request structure. Without a preliminary reset of the fields, a request for closing a position by an opposite one was sometimes
created instead of a simple position closure. However, this is justified since we should never forget that a simple declaration of a
variable without its initialization may subsequently lead to unpredictable results. This fact was confirmed again when I forgot to
initialize the structure of the trading request in the class constructor.

In the **PendReqControl.mqh** file of the trading management class, namely in its public section, declare the two methods — the
method of creating a pending request for full and partial position closure and the
method for closing a position by an opposite one:

```
public:
//--- Return itself
   CTradingControl     *GetObject(void)            { return &this;   }
//--- Timer
   virtual void         OnTimer(void);
//--- Constructor
                        CTradingControl();
//--- (1) Create a pending request (1) to open a position, (2) to place a pending order
   template<typename SL,typename TP>
   int                  CreatePReqPosition(const ENUM_POSITION_TYPE type,
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
   int                  CreatePReqOrder(const ENUM_ORDER_TYPE order_type,
                                        const double volume,
                                        const string symbol,
                                        const PS price_set,
                                        const PL price_limit=0,
                                        const SL sl=0,
                                        const TP tp=0,
                                        const ulong magic=ULONG_MAX,
                                        const uchar group_id1=0,
                                        const uchar group_id2=0,
                                        const string comment=NULL,
                                        const datetime expiration=0,
                                        const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                        const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
//--- Create a pending request (1) for full and partial position closure, (2) for closing a position by an opposite one
   int                  CreatePReqClose(const ulong ticket,const double volume=WRONG_VALUE,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  CreatePReqCloseBy(const ulong ticket,const ulong ticket_by);

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

In the method for checking the relevance of a pending request, improve the
block of handling pending request objects when closing positions partially or by an opposite one:

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
            //--- If there is an event
            if(this.m_events.IsEvent())
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
                     if(event.TicketFirstOrderPosition()==req_obj.Position())
                       {
                        //--- Get a position object from the list of market positions
                        CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Position(),EQUAL);
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
               if(event.TicketFirstOrderPosition()==req_obj.Position())
                 {
                  //--- Get a position object from the list of market positions
                  CArrayObj *list_orders=this.m_market.GetList(ORDER_PROP_TICKET,req_obj.Position(),EQUAL);
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

Here we added the check whether the flag of an occurred account event is currently
set to always be able to handle the last trading event and not affect the previous one located in the list of account trading events. In
this case, a newly created pending request object is deemed activated and is immediately removed. This is the outcome we want to avoid.

Beyond the class body, write the implementation of methods creating
pending requests for full and partial position closure and closing a
position by an opposite one:

```
//+------------------------------------------------------------------+
//| Create a pending request for closing a position                  |
//+------------------------------------------------------------------+
int CTradingControl::CreatePReqClose(const ulong ticket,const double volume=WRONG_VALUE,const string comment=NULL,const ulong deviation=ULONG_MAX)
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
      return false;
     }
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)order.TypeOrder();
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   //--- If failed to get the symbol object, display the message and return 'false'
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
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
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

//--- Write a volume, deviation and a comment to the request structure
   this.m_request.deviation=(deviation==ULONG_MAX ? trade_obj.GetDeviation() : deviation);
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.volume=(volume==WRONG_VALUE || volume>order.Volume() ? order.Volume() : symbol_obj.NormalizedLot(volume));
//--- Write a magic number, a symbol name,
//--- a trading operation type, as well as order type and ticket to the request structure
   this.m_request.magic=order.Magic();
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.action=TRADE_ACTION_DEAL;
   this.m_request.type=order_type;
   this.m_request.position=ticket;
   this.m_request.position_by=0;
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_CLOSE,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,order))
      return id;
   return WRONG_VALUE;
  }
//+--------------------------------------------------------------------+
//| Create a pending request for closing a position by an opposite one |
//+--------------------------------------------------------------------+
int CTradingControl::CreatePReqCloseBy(const ulong ticket,const ulong ticket_by)
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
   ENUM_ACTION_TYPE action=ACTION_TYPE_CLOSE_BY;
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
//--- Get a symbol object by a position ticket
   CSymbol *symbol_obj=this.GetSymbolObjByPosition(ticket,DFUN);
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return false;
     }
//--- trading object of a closed position
   CTradeObj *trade_obj_pos=this.GetTradeObjByPosition(ticket,DFUN);
   if(trade_obj_pos==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
   if(!this.m_account.IsHedge())
     {
      trade_obj_pos.SetResultRetcode(MSG_ACC_UNABLE_CLOSE_BY);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      return false;
     }
//--- check the presence of an opposite position
   if(!this.CheckPositionAvailablity(ticket_by,DFUN))
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_SYS_ERROR_POSITION_BY_ALREADY_CLOSED);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      return false;
     }
//--- trading object of an opposite position
   CTradeObj *trade_obj_pos_by=this.GetTradeObjByPosition(ticket_by,DFUN);
   if(trade_obj_pos_by==NULL)
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN_ERR_LINE,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return false;
     }
//--- If a symbol of a closed position is not equal to an opposite position's one, inform of that and exit
   if(symbol_obj.Name()!=trade_obj_pos_by.GetSymbol())
     {
      trade_obj_pos.SetResultRetcode(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_CLOSE_BY_SYMBOLS_UNEQUAL));
      return false;
     }
//--- Update symbol quotes
   if(!symbol_obj.RefreshRates())
     {
      trade_obj_pos.SetResultRetcode(10021);
      trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
      this.AddErrorCodeToList(10021);  // No quotes to handle the request
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));
      return false;
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

//--- Write the trading operation type, symbol, tickets of two positions, type and volume of a closed position to the request structure
   this.m_request.action=TRADE_ACTION_CLOSE_BY;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.position=ticket;
   this.m_request.position_by=ticket_by;
   this.m_request.type=order_type;
   this.m_request.volume=order.Volume();
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_CLOSE,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,order))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The methods are identical to all previously considered methods of creating pending requests for opening positions and placing pending
orders. We have already considered them in previous articles. Besides, the code of the methods is commented in sufficient detail, so there
is no point in dwelling on it here.

In the **Trading.mqh** file of the library base trading object class, move
the methods from the private section of the class to the protected one:

```
private:
   CArrayInt            m_list_errors;                   // Error list
   bool                 m_is_trade_disable;              // Flag disabling trading
   bool                 m_use_sound;                     // The flag of using sounds of the object trading events
   ENUM_ERROR_HANDLING_BEHAVIOR m_err_handling_behavior; // Behavior when handling error

//--- Add the error code to the list
   bool                 AddErrorCodeToList(const int error_code);
//--- Return the symbol object by (1) position, (2) order ticket
   CSymbol             *GetSymbolObjByPosition(const ulong ticket,const string source_method);
   CSymbol             *GetSymbolObjByOrder(const ulong ticket,const string source_method);
//--- Return a symbol trading object by (1) position, (2) order ticket, (3) symbol name
   CTradeObj           *GetTradeObjByPosition(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjByOrder(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjBySymbol(const string symbol,const string source_method);
//--- Return an order object by ticket
   COrder              *GetOrderObjByTicket(const ulong ticket);

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
   ENUM_ORDER_TYPE      DirectionByActionType(const ENUM_ACTION_TYPE action)  const;
//--- Check the presence of a (1) position, (2) order by ticket
   bool                 CheckPositionAvailablity(const ulong ticket,const string source_method);
   bool                 CheckOrderAvailablity(const ulong ticket,const string source_method);
//--- Set the desired sound for a trading object
```

Now the relocated methods are in the protected section of the class:

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

//--- Add the error code to the list
   bool                 AddErrorCodeToList(const int error_code);
//--- Look for the first free pending request ID
   int                  GetFreeID(void);
//--- Return the flag of a market order/position with a pending request ID
   bool                 IsPresentOrderByID(const uchar id);
//--- Return an order object by ticket
   COrder              *GetOrderObjByTicket(const ulong ticket);
//--- Return the symbol object by (1) position, (2) order ticket
   CSymbol             *GetSymbolObjByPosition(const ulong ticket,const string source_method);
   CSymbol             *GetSymbolObjByOrder(const ulong ticket,const string source_method);
//--- Return a symbol trading object by (1) position, (2) order ticket, (3) symbol name
   CTradeObj           *GetTradeObjByPosition(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjByOrder(const ulong ticket,const string source_method);
   CTradeObj           *GetTradeObjBySymbol(const string symbol,const string source_method);
//--- Check the presence of a (1) position, (2) order by ticket
   bool                 CheckPositionAvailablity(const ulong ticket,const string source_method);
   bool                 CheckOrderAvailablity(const ulong ticket,const string source_method);

private:
```

These methods are used by the CTradingControl child class and should be located in the protected section.

In the **CEngine** library base object class, namely in its public section, add
the method returning the full list of all pending requests:

```
//--- Return (1) the list of references to resources, (2) resource object index by its description
   CArrayObj           *GetListResource(void)                                 { return this.m_resource.GetList();                               }
   int                  GetIndexResObjByDescription(const string file_name)   { return this.m_resource.GetIndexResObjByDescription(file_name);  }

//--- Return the list of pending requests
   CArrayObj           *GetListPendingRequests(void)                          { return this.m_trading.GetListRequests();                        }

//--- Set the following for the trading classes:
//--- (1) correct filling policy, (2) filling policy,
//--- (3) correct order expiration type, (4) order expiration type,
//--- (5) magic number, (6) comment, (7) slippage, (8) volume, (9) order expiration date,
//--- (10) the flag of asynchronous sending of a trading request, (11) logging level, (12) number of trading attempts
```

The method returns the list of pending requests by calling the method of the GetListRequests() trading class.

Now the method allows us to get the full list of existing pending requests that can be sorted and searched using the search and sorting methods
to be developed below.

In the public section of the class, declare three methods for creating pending requests:

for a full
position closure, for a partial position closure and for closing
by an opposite position:

```
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

//--- Create a pending request for closing a position (1) fully, (2) partially, (3) by an opposite one
   int                  ClosePositionPending(const ulong ticket,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  ClosePositionPartiallyPending(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=ULONG_MAX);
   int                  ClosePositionByPending(const ulong ticket,const ulong ticket_by);

//--- Create a pending request to place a (1) BuyLimit, (2) BuyStop and (3) BuyStopLimit order
```

Implement them beyond the class body:

```
//+------------------------------------------------------------------+
//| Create a pending request for closing a position in full          |
//+------------------------------------------------------------------+
int CEngine::ClosePositionPending(const ulong ticket,const string comment=NULL,const ulong deviation=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqClose(ticket,WRONG_VALUE,comment,deviation);
  }
//+------------------------------------------------------------------+
//| Create a pending request for closing a position partially        |
//+------------------------------------------------------------------+
int CEngine::ClosePositionPartiallyPending(const ulong ticket,const double volume,const string comment=NULL,const ulong deviation=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqClose(ticket,volume,comment,deviation);
  }
//+--------------------------------------------------------------------+
//| Create a pending request for closing a position by an opposite one |
//+--------------------------------------------------------------------+
int CEngine::ClosePositionByPending(const ulong ticket,const ulong ticket_by)
  {
   return this.m_trading.CreatePReqCloseBy(ticket,ticket_by);
  }
//+------------------------------------------------------------------+
```

The methods simply call the appropriate methods of creating pending requests of the CTradingControl class.

To
create a pending request for a full position closure, the CreatePReqClose() method of the trading management class receives
WRONG\_VALUE as a closed volume, while a
closed volume passed to the method as an input is used for a partial closure.

Now let's create the methods for searching and sorting pending request objects in the list of pending requests.


The \\MQL5\\Include\\DoEasy\ **Services\\Select.mqh** file receives the abstract
pending request object class. Declare the methods for working with pending
requests:

```
//+------------------------------------------------------------------+
//|                                                       Select.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/ru/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Arrays\ArrayObj.mqh>
#include "..\Objects\Orders\Order.mqh"
#include "..\Objects\Events\Event.mqh"
#include "..\Objects\Accounts\Account.mqh"
#include "..\Objects\Symbols\Symbol.mqh"
#include "..\Objects\PendRequest\PendRequest.mqh"
//+------------------------------------------------------------------+
//| Storage list                                                     |
//+------------------------------------------------------------------+
CArrayObj   ListStorage; // Storage object for storing sorted collection lists
//+------------------------------------------------------------------+
//| Class for sorting objects meeting the criterion                  |
//+------------------------------------------------------------------+
class CSelect
  {
private:
   //--- Method for comparing two values
   template<typename T>
   static bool       CompareValues(T value1,T value2,ENUM_COMPARER_TYPE mode);
public:
//+------------------------------------------------------------------+
//| Methods of working with orders                                   |
//+------------------------------------------------------------------+
   //--- Return the list of orders with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByOrderProperty(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the order index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMax(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
   //--- Return the order index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_INTEGER property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_DOUBLE property);
   static int        FindOrderMin(CArrayObj *list_source,ENUM_ORDER_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with events                                   |
//+------------------------------------------------------------------+
   //--- Return the list of events with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByEventProperty(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMax(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_INTEGER property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_DOUBLE property);
   static int        FindEventMin(CArrayObj *list_source,ENUM_EVENT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with accounts                                 |
//+------------------------------------------------------------------+
   //--- Return the list of accounts with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByAccountProperty(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the event index with the maximum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMax(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
   //--- Return the event index with the minimum value of the event's (1) integer, (2) real and (3) string properties
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_INTEGER property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_DOUBLE property);
   static int        FindAccountMin(CArrayObj *list_source,ENUM_ACCOUNT_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with symbols                                  |
//+------------------------------------------------------------------+
   //--- Return the list of symbols with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *BySymbolProperty(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the symbol index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMax(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
   //--- Return the symbol index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_INTEGER property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_DOUBLE property);
   static int        FindSymbolMin(CArrayObj *list_source,ENUM_SYMBOL_PROP_STRING property);
//+------------------------------------------------------------------+
//| Methods of working with pending requests                         |
//+------------------------------------------------------------------+
   //--- Return the list of pending requests with one out of (1) integer, (2) real and (3) string properties meeting a specified criterion
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode);
   static CArrayObj *ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode);
   //--- Return the pending request index with the maximum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property);
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property);
   static int        FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property);
   //--- Return the pending request index with the minimum value of the order's (1) integer, (2) real and (3) string properties
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property);
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property);
   static int        FindPendReqMin(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property);
//---
  };
//+------------------------------------------------------------------+
```

Implement the methods for sorting and searching in the list of pending requests beyond the class body:

```
//+------------------------------------------------------------------+
//| Methods of working with lists of pending trading requests        |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Return the list of requests with one integer                     |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property,long value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   int total=list_source.Total();
   for(int i=0; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      long obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of requests with one real                        |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property,double value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CPendRequest *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      double obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the list of requests with one string                      |
//| property meeting the specified criterion                         |
//+------------------------------------------------------------------+
CArrayObj *CSelect::ByPendReqProperty(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property,string value,ENUM_COMPARER_TYPE mode)
  {
   if(list_source==NULL) return NULL;
   CArrayObj *list=new CArrayObj();
   if(list==NULL) return NULL;
   list.FreeMode(false);
   ListStorage.Add(list);
   for(int i=0; i<list_source.Total(); i++)
     {
      CPendRequest *obj=list_source.At(i);
      if(!obj.SupportProperty(property)) continue;
      string obj_prop=obj.GetProperty(property);
      if(CompareValues(obj_prop,value,mode)) list.Add(obj);
     }
   return list;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the maximum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CPendRequest *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      long obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the maximum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CPendRequest *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      double obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the maximum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMax(CArrayObj *list_source,ENUM_PEND_REQ_PROP_STRING property)
  {
   if(list_source==NULL) return WRONG_VALUE;
   int index=0;
   CPendRequest *max_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      max_obj=list_source.At(index);
      string obj2_prop=max_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,MORE)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the minimum integer property value                          |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMin(CArrayObj* list_source,ENUM_PEND_REQ_PROP_INTEGER property)
  {
   int index=0;
   CPendRequest *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      long obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      long obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the minimum real property value                             |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMin(CArrayObj* list_source,ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   int index=0;
   CPendRequest *min_obj=NULL;
   int total=list_source.Total();
   if(total== 0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      double obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      double obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
//| Return the listed request index                                  |
//| with the minimum string property value                           |
//+------------------------------------------------------------------+
int CSelect::FindPendReqMin(CArrayObj* list_source,ENUM_PEND_REQ_PROP_STRING property)
  {
   int index=0;
   CPendRequest *min_obj=NULL;
   int total=list_source.Total();
   if(total==0) return WRONG_VALUE;
   for(int i=1; i<total; i++)
     {
      CPendRequest *obj=list_source.At(i);
      string obj1_prop=obj.GetProperty(property);
      min_obj=list_source.At(index);
      string obj2_prop=min_obj.GetProperty(property);
      if(CompareValues(obj1_prop,obj2_prop,LESS)) index=i;
     }
   return index;
  }
//+------------------------------------------------------------------+
```

The methods were described in detail in the [third article when \\
considering the implementation of the search by library collections](https://www.mql5.com/en/articles/5687#node01).

The only difference in the logic of the current methods is
that the search and sorting methods work with objects and pending request data of the CPendRequest class.

**These are all the changes of the library classes for arranging closing positions under certain conditions using pending trading requests.**

### Testing

To test closing positions under certain conditions, [use the EA from the \\
previous article](https://www.mql5.com/en/articles/7536#node03) and save it in \\MQL5\\Experts\\TestDoEasy\ **Part33\** under the name **TestDoEasyPart33.mq5**.

In the block of EA global variables, I have changed the names of
variables storing the flags of states of the buttons  activating trading modes using pending requests:

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

Now these variables have more readable names:

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

I used **Ctrl+H** to search for "pending\_" throughout the entire text and replace it with "pressed\_pending\_" in order to rename all
these variables within the entire EA code.

The **PressButtonEvents()** function handling EA button pressing features similar
code blocks for setting activation conditions for newly created objects of pending trading requests:

```
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
        }
      //--- If the BUTT_BUY_LIMIT button is pressed: Place BuyLimit
      else if(button==EnumToString(BUTT_BUY_LIMIT))
        {
```

To reduce the amount of code, it would be reasonable to put all repeating code blocks into a separate function which is to accept required
parameters for setting activation conditions to pending request objects.

Let's implement the following function:

```
//+------------------------------------------------------------------+
//| Set pending request activation conditions                        |
//+------------------------------------------------------------------+
void SetPReqCriterion(const uchar id,const double price_activation,const ulong time_activation,ENUM_BUTTONS button,ENUM_COMPARER_TYPE comp_type,const double price_curr,const datetime time_curr)
  {
   double point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   int    digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
//--- If the price criterion is selected
   if(ButtonState(prefix+EnumToString(button)+"_PRICE"))
     {
      //--- set the pending request activation price
      engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_BID,price_activation,comp_type,price_curr);
     }
//--- If the time criterion is selected
   if(ButtonState(prefix+EnumToString(button)+"_TIME"))
     {
      //--- set the pending request activation time
      engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,time_activation,EQUAL_OR_MORE,time_curr);
     }
//--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
   CPendRequest *req_obj=engine.GetPendRequestByID((uchar)id);
   if(req_obj==NULL)
      return;
   if(engine.TradingGetLogLevel(Symbol())>LOG_LEVEL_NO_MSG)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ADD_CRITERIONS),", ID #",req_obj.ID(),":");
      req_obj.PrintActivations();
     }
  }
//+------------------------------------------------------------------+
```

The function receives the ID of a new pending request object, request price and activation time, pressed button name constant, comparison
type and the current price and time.

Depending on the pressed button name, request object activation conditions are set in it and a
message is displayed in the journal informing of adding activation conditions for a pending request.

Now, in the **PressButtonEvents()** function, replace the code blocks of the same type described above with
calling a new function for setting pending request activation conditions, as well as improve
handling pressing position closure buttons:

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
//+------------------------------------------------------------------+
void PressButtonEvents(const string button_name)
  {
   bool comp_magic=true;   // Temporary variable selecting the composite magic number with random group IDs
   string comment="";
   double point=SymbolInfoDouble(NULL,SYMBOL_POINT);
   int    digits=(int)SymbolInfoInteger(NULL,SYMBOL_DIGITS);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
            engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStopLimit","Pending order BuyStopLimit"));
         //--- Otherwise, create a pending request to place a BuyStopLimit order with the placement distances
         //--- and set the conditions depending on active buttons
         else
           {
            int id=engine.PlaceBuyStopLimitPending(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- set the pending request activation price and time, as well as set activation parameters
               double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
               double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
               //--- set the pending request activation price and time, as well as set activation parameters
               double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
               double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
                     //--- set the pending request activation price and time, as well as set activation parameters
                     double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                     double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
                     //--- set the pending request activation price and time, as well as set activation parameters
                     double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                     double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
                        //--- set the pending request activation price and time, as well as set activation parameters
                        double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
                        double price_activation=NormalizeDouble(bid+distance_pending_request*point,digits);
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
                     //--- set the pending request activation price and time, as well as set activation parameters
                     double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                     double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
                     //--- set the pending request activation price and time, as well as set activation parameters
                     double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                     double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
                        //--- set the pending request activation price and time, as well as set activation parameters
                        double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
                        double price_activation=NormalizeDouble(ask-distance_pending_request*point,digits);
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
      //--- re-draw the chart
      ChartRedraw();
     }
  }
//+------------------------------------------------------------------+
```

All replaced code blocks, as well as newly
added ones, are commented in detail and require no further explanations.

If you have any questions, feel free to ask
them in the comments.

Let's compile the EA and test pending requests in relation to various types of closing positions (partial, full and by an opposite one). To do
this, launch the EA in the visual tester and do the following:

1. open a sell position and create a pending request for closing it partially by price;

2. after the partial closure, open a buy position and create a pending request for closing it by the opposite one (the half-closed short
    position) by price;

3. after the partial closure of the long position by the opposite sell one, create a new pending request for the full closure of the long
    position under the condition that the request is activated by time.

![](https://c.mql5.com/2/38/FoGlbrlwqr.gif)

As we can see, all requests are handled according to the given conditions and are removed after activation.

### What's next?

In the next article, we will continue the development of the pending trading request concept and implement removal of pending orders, as
well as modifying orders and positions under certain conditions.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7554#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7554](https://www.mql5.com/ru/articles/7554)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7554.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7554/mql5.zip "Download MQL5.zip")(3664.44 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7554/mql4.zip "Download MQL4.zip")(3664.45 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/338356)**

![How to create 3D graphics using DirectX in MetaTrader 5](https://c.mql5.com/2/39/MQL5-avatar-directx_yellow.png)[How to create 3D graphics using DirectX in MetaTrader 5](https://www.mql5.com/en/articles/7708)

3D graphics provide excellent means for analyzing huge amounts of data as they enable the visualization of hidden patterns. These tasks can be solved directly in MQL5, while DireсtX functions allow creating three-dimensional object. Thus, it is even possible to create programs of any complexity, even 3D games for MetaTrader 5. Start learning 3D graphics by drawing simple three-dimensional shapes.

![Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://c.mql5.com/2/38/OLAP_02.png)[Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://www.mql5.com/en/articles/7535)

In this article we will continue dealing with the OLAP technology applied to trading. We will expand the functionality presented in the first two articles. This time we will consider the operational analysis of quotes. We will put forward and test the hypotheses on trading strategies based on aggregated historical data. The article presents Expert Advisors for studying bar patterns and adaptive trading.

![Applying network functions, or MySQL without DLL: Part I - Connector](https://c.mql5.com/2/37/kisspng-computer-icons-application-programming-interface-c-database-administrator-icon-free-download.png)[Applying network functions, or MySQL without DLL: Part I - Connector](https://www.mql5.com/en/articles/7117)

MetaTrader 5 has received network functions recently. This opened up great opportunities for programmers developing products for the Market. Now they can implement things that required dynamic libraries before. In this article, we will consider them using the implementation of the MySQL as an example.

![Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy.png)[Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions](https://www.mql5.com/en/articles/7536)

We continue the development of the functionality allowing users to trade using pending requests. In this article, we are going to implement the ability to place pending orders under certain conditions.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/7554&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070426218622293407)

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