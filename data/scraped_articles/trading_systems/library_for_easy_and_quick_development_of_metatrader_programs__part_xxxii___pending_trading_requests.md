---
title: Library for easy and quick development of MetaTrader programs (part XXXII): Pending trading requests - placing orders under certain conditions
url: https://www.mql5.com/en/articles/7536
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:00.560822
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=epfzenkjuzweielqamnyodtpkzqiratk&ssn=1769186218158442233&ssn_dr=0&ssn_sr=0&fv_date=1769186218&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7536&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Library%20for%20easy%20and%20quick%20development%20of%20MetaTrader%20programs%20(part%20XXXII)%3A%20Pending%20trading%20requests%20-%20placing%20orders%20under%20certain%20conditions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918621835967645&fz_uniq=5070428653868750250&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7536#node01)
- [Implementation](https://www.mql5.com/en/articles/7536#node02)
- [Testing](https://www.mql5.com/en/articles/7536#node03)
- [What's next?](https://www.mql5.com/en/articles/7536#node04)


[In the previous article,](https://www.mql5.com/en/articles/7521) we added the ability to send trading requests by
predefined conditions. Upon the occurrence of a given condition (or a set of conditions), a trading order to open a position is sent. There
may be multiple conditions in different combinations from the lists of account, symbol and event status conditions.

Trading orders
are sent using [pending trading requests](https://www.mql5.com/en/articles/7394) upon meeting all the conditions set in
the pending request object.

In this article, we continue the development of the concept and create the functionality allowing to place pending orders using pending
trading requests featuring all conditions that are necessary for placing a pending order.

### Concept

The pending request object features the array storing all of its activation conditions. [The \\
trading management class](https://www.mql5.com/en/articles/7481#node04) (namely, its timer) allows constant viewing of the list of pending trading requests. When it is time to
activate a pending trading request (all predefined activation conditions are met), a trading order is sent to the server. Its parameters
are set in the triggered pending request.

To open positions, you just need to control the occurrence of specified conditions. As soon as they occur, the trading order to open a
position is sent to the server.

However, there is one complication regarding placing pending orders using pending request objects: an
order is placed at a distance from the price, while a position is set at the appropriate current price.

Therefore, in order to work with
placing pending orders under certain conditions, we also need to consider the distance of placing the pending order. This entails a
question: when creating a pending request, we specify the distance of a future pending order. BUT... from which price? From the price
present at the moment of a pending request? Or from the price present when fulfilling all conditions set in the request object for its
activation? After all, at the moment all conditions are met, the price may move far from the level the pending request was created at, while we
are able to know the future price only in one case — when the only activation condition of a pending request is a specified price value. In other
cases, the future price we need to set an order from is unknown.

Let's do it the following way: when creating a pending request, we will specify the pending order distance. We can always view the distance using
the difference **between the current price at the moment of creating a pending request** (the current Ask or Bid price is set
in the property depending on the future order direction) and **the price of placing a pending order** (also set in the pending
request object properties). In other words, we are able to calculate a new pending order price at any price value at the moment of the pending
request activation or leave the price specified when creating the pending request.

In the first case, the order price is recalculated relative to the current price at the moment of the pending request activation, while in the
second one, a trading order for placing a pending order relative to the price the pending request has been based on is sent to the server. In
this option, the price is adjusted if it becomes invalid while waiting for the pending request activation.

### Implementation

In the **PendRequest.mqh** file, namely in the private section of the **CPendRequest** abstract pending request
object class, add the class member variable for storing the flag of
shifting the pending order distance reference point following the price:

```
//+------------------------------------------------------------------+
//| Abstract pending trading request class                           |
//+------------------------------------------------------------------+
class CPendRequest : public CBaseObj
  {
private:
   MqlTradeRequest   m_request;                                            // Trade request structure
   CPause            m_pause;                                              // Pause class object
   bool              m_follow;                                             // The flag of the pending order distance reference point following the price
/* Data on a pending request activation in the array:
```

If the variable is true, the order price is recalculated relative to the current
price at the moment of the pending request activation. Otherwise, the pending order is set at the price set in the pending request object
properties and adjusted in case the order price becomes invalid due to a change of the current price relative to the pending request one.

In the protected section of the class, declare the method of placing
pending order prices according to the shift:

```
//--- Return the number of decimal places of a controlled property
   int               DigitsControlledValue(const uint index)               const;
//--- Set a new value changed by the shift (+/-) for all order prices
   void              SetAllMqlPrices(const double shift);

public:
```

Declare the method for adjusting pending order prices relative to the
current price in the block of methods for a simplified access to the request object properties in the public section of the class, as
well as write the methods of placing new order prices to the pending request object
properties and the methods of setting/receiving the flag
of the order price reference point following the price:

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

//--- Modify order prices by the current price
   void                 CorrectMqlPricesByCurrentPrice(const double price);

//--- Set (1) the price when creating a request, (2) setting, (3) StopLoss, (4) TakeProfit, (5) stoplimit,
//--- (6) request creation time, (7) current attempt time, (8) waiting time between requests, (9) current attempt index,
//---  (10) number of attempts,(11) id, (12) order ticket, (13) position ticket, (14) pending request type
   void                 SetPriceCreate(const double price)           { this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);                                              }
   void                 SetMqlPrice(const double price)              { this.SetProperty(PEND_REQ_PROP_MQL_REQ_PRICE,price); this.m_request.price=price;                 }
   void                 SetMqlSL(const double sl)                    { this.SetProperty(PEND_REQ_PROP_MQL_REQ_SL,sl); this.m_request.sl=sl;                             }
   void                 SetMqlTP(const double tp)                    { this.SetProperty(PEND_REQ_PROP_MQL_REQ_TP,tp); this.m_request.tp=tp;                             }
   void                 SetMqlStopLimit(const double stoplimit)      { this.SetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT,stoplimit); this.m_request.stoplimit=stoplimit; }
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
//--- Return/set the flag of the pending order distance reference point following the price
   bool                 IsFollowThePrice(void)                                   const { return this.m_follow; }
   void                 SetFollowThePrice(const bool flag)                             { this.m_follow=flag;   }

//+------------------------------------------------------------------+
//| Descriptions of request object properties                        |
//+------------------------------------------------------------------+
```

In the class constructor, set the flag of the order distance
reference point following the price:

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

Implement the method for setting new values to all order prices beyond the class body:

```
//+------------------------------------------------------------------+
//| Set a new value changed by the shift (+/-),                      |
//| for all order prices (+/-)                                       |
//+------------------------------------------------------------------+
void CPendRequest::SetAllMqlPrices(const double shift)
  {
   this.SetMqlPrice(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE)-shift);
   if(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)!=0)
      this.SetMqlSL(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)-shift);
   if(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)!=0)
      this.SetMqlTP(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)-shift);
   if(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)!=0)
      this.SetMqlStopLimit(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)-shift);
  }
//+------------------------------------------------------------------+
```

The method receives the price shift, and the methods described
above are used to set new prices in each pending request object property corresponding to pending order price types calculated as (the
current property value minus the shift).

For
StopLoss, TakeProfit and StopLimit order prices, the existence of the price is preliminarily checked, and the shift is set only if the price
set in the pending request object properties has a non-zero value.

Implementing the method for adjusting the prices of a placed pending order by the current price at the moment of the pending request activation:

```
//+------------------------------------------------------------------+
//| Adjust order prices by the current price                         |
//+------------------------------------------------------------------+
void CPendRequest::CorrectMqlPricesByCurrentPrice(const double price)
  {
   ENUM_ORDER_TYPE type=this.m_request.type;
   if(!this.m_follow || (type<ORDER_TYPE_BUY_LIMIT && type>ORDER_TYPE_SELL_STOP_LIMIT))
      return;
   this.SetAllMqlPrices(this.PriceCreate()-price);
  }
//+------------------------------------------------------------------+
```

The method receives the current price the pending order should
be placed from. If the
flag of following the price by the order distance reference point is not set
or the pending order is not added to the trading request structure of the pending request object,
leave the method.

Next, call the method described above for changing all pending order prices. It receives the shift
calculated as the price at the moment of creating the pending request object
minus the current price passed to the method.

Now let's make additions and improvements in the **PendReqControl.mqh** file of the **CPendReqControl** trading
management class.

Rename the OpenPositionPending() and PlaceOrderPending() public methods of creating pending requests to CreatePReqPosition()
and CreatePReqOrder() respectively. I believe, these method names
reflect the idea behind them (creating a pending request) more accurately.

In the **CreatePReqOrder()** method inputs, add passing group IDs:

```
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
//--- Set pending request activation criteria
```

Make an addition to the handler of pending requests created by
request:

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
      //--- Adjust prices for a pending order relative to the current price and get the request again
      if(request.action==TRADE_ACTION_PENDING)
        {
         req_obj.CorrectMqlPricesByCurrentPrice(PositionTypeByOrderType(request.type)==POSITION_TYPE_BUY ? symbol_obj.AskLast() : symbol_obj.BidLast());
         request=req_obj.MqlRequest();
        }
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

Here, if the trading operation type set in the trading request structure of the
pending request object is equal to "place a pending order", call the method
for adjusting pending order prices set in the pending request object properties. As a result, the pending order prices in the
request object are either adjusted relative to the current price or
not — this depends on the flag of the pending order distance reference point following the price in the pending request object. We have
discussed this behavior above.

Let's slightly improve the method of creating a pending request for opening a position. When developing it using the copy-paste method, I made a
blunder — the method should return the integer value of the pending request ID,
while it currently returns false in case of an error. Let's
change it to WRONG\_VALUE:

```
//+------------------------------------------------------------------+
//| Create a pending request for opening a position                  |
//+------------------------------------------------------------------+
template<typename SL,typename TP>
int CTradingControl::CreatePReqPosition(const ENUM_POSITION_TYPE type,
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
//--- If the global trading ban flag is set, exit and return WRONG_VALUE
   if(this.IsTradingDisable())
     {
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_TEXT_TRADING_DISABLE));
      return WRONG_VALUE;
     }
//--- Set the error flag as "no errors"

   this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_NO_ERROR;
   ENUM_ORDER_TYPE order_type=(ENUM_ORDER_TYPE)type;
   ENUM_ACTION_TYPE action=(ENUM_ACTION_TYPE)order_type;
//--- Get a symbol object by a symbol name.
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
//--- If failed to get - write the "internal error" flag, display the message in the journal and return WRONG_VALUE
   if(symbol_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_SYM_OBJ));
      return WRONG_VALUE;
     }
//--- get a trading object from a symbol object
   CTradeObj *trade_obj=symbol_obj.GetTradeObj();
//--- If failed to get - write the "internal error" flag, display the message in the journal and return WRONG_VALUE
   if(trade_obj==NULL)
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(MSG_LIB_SYS_ERROR_FAILED_GET_TRADE_OBJ));
      return WRONG_VALUE;
     }
//--- Set the prices
//--- If failed to set - write the "internal error" flag, set the error code in the return structure,
//--- display the message in the journal and return WRONG_VALUE
   if(!this.SetPrices(order_type,0,sl,tp,0,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
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

Implement the method of creating a pending request for placing a pending order:

```
//+------------------------------------------------------------------+
//| Create a pending request to place a pending order                |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
int CTradingControl::CreatePReqOrder(const ENUM_ORDER_TYPE order_type,
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
   ENUM_ACTION_TYPE action=(ENUM_ACTION_TYPE)order_type;
//--- Get a symbol object by a symbol name
   CSymbol *symbol_obj=this.m_symbols.GetSymbolObjByName(symbol);
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
//--- display the message in the journal and return WRONG_VALUE
   if(!this.SetPrices(order_type,price_set,sl,tp,price_limit,DFUN,symbol_obj))
     {
      this.m_error_reason_flags=TRADE_REQUEST_ERR_FLAG_INTERNAL_ERR;
      trade_obj.SetResultRetcode(10021);
      trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(DFUN,CMessage::Text(10021));   // No quotes to process the request
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

//--- Write the volume, comment, as well as expiration and filling types to the request structure
   this.m_request.volume=volume;
   this.m_request.comment=(comment==NULL ? trade_obj.GetComment() : comment);
   this.m_request.type_time=(type_time>WRONG_VALUE ? type_time : trade_obj.GetTypeExpiration());
   this.m_request.type_filling=(type_filling>WRONG_VALUE ? type_filling : trade_obj.GetTypeFilling());
//--- Write the request ID to the magic number, while a symbol name is set in the request structure,
//--- trading operation and order types
   uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
   this.SetPendReqID((uchar)id,mn);
   if(group_id1>0)
      this.SetGroupID1(group_id1,mn);
   if(group_id2>0)
      this.SetGroupID2(group_id2,mn);
   this.m_request.magic=mn;
   this.m_request.symbol=symbol_obj.Name();
   this.m_request.action=TRADE_ACTION_PENDING;
   this.m_request.type=order_type;
//--- As a result of creating a pending trading request, return either its ID or -1 if unsuccessful
   if(this.CreatePendingRequest(PEND_REQ_STATUS_PLACE,(uchar)id,1,ulong(END_TIME-(ulong)::TimeCurrent()),this.m_request,0,symbol_obj,NULL))
      return id;
   return WRONG_VALUE;
  }
//+------------------------------------------------------------------+
```

The method is thoroughly described in the code comments. We have already considered a similar method of creating a pending request for
opening a position, therefore there is no point in dwelling on it here. If you have any questions, feel free to ask them in the comments.

When creating a pending request, we need to set the price at the moment of its creation in the pending request object. We need to set different
prices for different types of order. In case of buy orders, it is the current Ask, while in case of sell ones, it is the current Bid.

To do
this, make changes to the **CreatePendingRequest()** method of creating a pending request in the **Trading.mqh** file
of the **CTrading** base trading object class:

```
//+------------------------------------------------------------------+
//| Create a pending request                                         |
//+------------------------------------------------------------------+
bool CTrading::CreatePendingRequest(const ENUM_PEND_REQ_STATUS status,
                                    const uchar id,
                                    const uchar attempts,
                                    const ulong wait,
                                    const MqlTradeRequest &request,
                                    const int retcode,
                                    CSymbol *symbol_obj,
                                    COrder *order)
  {
   //--- Create a new pending request object depending on a request status
   CPendRequest *req_obj=NULL;
   double price=(PositionTypeByOrderType(request.type)==POSITION_TYPE_BUY ? symbol_obj.AskLast() : symbol_obj.BidLast());
   switch(status)
     {
      case PEND_REQ_STATUS_OPEN     : req_obj=new CPendReqOpen(id,price,symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_CLOSE    : req_obj=new CPendReqClose(id,price,symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_SLTP     : req_obj=new CPendReqSLTP(id,price,symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_PLACE    : req_obj=new CPendReqPlace(id,price,symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_REMOVE   : req_obj=new CPendReqRemove(id,price,symbol_obj.Time(),request,retcode);  break;
      case PEND_REQ_STATUS_MODIFY   : req_obj=new CPendReqModify(id,price,symbol_obj.Time(),request,retcode);  break;
      default: req_obj=NULL;
        break;
     }
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
   //--- Fill in the properties of a successfully created object by the values passed to the method
   req_obj.SetTimeActivate(symbol_obj.Time()+wait);
   req_obj.SetWaitingMSC(wait);
   req_obj.SetCurrentAttempt(0);
   req_obj.SetTotalAttempts(attempts);
   if(order!=NULL)
     {
      req_obj.SetActualVolume(order.Volume());
      req_obj.SetActualPrice(order.PriceOpen());
      req_obj.SetActualStopLimit(order.PriceStopLimit());
      req_obj.SetActualSL(order.StopLoss());
      req_obj.SetActualTP(order.TakeProfit());
      req_obj.SetActualTypeFilling(order.TypeFilling());
      req_obj.SetActualTypeTime(order.TypeTime());
      req_obj.SetActualExpiration(order.TimeExpiration());
     }
   else
     {
      req_obj.SetActualVolume(request.volume);
      req_obj.SetActualPrice(request.price);
      req_obj.SetActualStopLimit(request.stoplimit);
      req_obj.SetActualSL(request.sl);
      req_obj.SetActualTP(request.tp);
      req_obj.SetActualTypeFilling(request.type_filling);
      req_obj.SetActualTypeTime(request.type_time);
      req_obj.SetActualExpiration(request.expiration);
     }
   //--- Display a brief description of a created pending request
   if(this.m_log_level>LOG_LEVEL_NO_MSG)
     {
      ::Print(CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CREATED)," #",req_obj.ID(),":");
      req_obj.PrintShort();
     }
   //--- successful
   return true;
  }
//+------------------------------------------------------------------+
```

Here we use the function of defining the position type by
PositionTypeByOrderType() order type to define the order direction. In case of a buy order, use the Ask price, in case of a sell order, use
the Bid price. When creating a pending request,
pass the obtained price to its creation method.

Now we only have to implement the access to the created functionality. In the public section of the **CEngine**
library main object, declare the methods of creating pending requests for placing all
order types:

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

//--- Create a pending request to place a (1) BuyLimit, (2) BuyStop and (3) BuyStopLimit order
   template<typename PS,typename SL,typename TP>
   int                  PlaceBuyLimitPending(const double volume,
                                             const string symbol,
                                             const PS price_set,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
                                             const string comment=NULL,
                                             const datetime expiration=0,
                                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename SL,typename TP>
   int                  PlaceBuyStopPending( const double volume,
                                             const string symbol,
                                             const PS price_set,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
                                             const string comment=NULL,
                                             const datetime expiration=0,
                                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   int                  PlaceBuyStopLimitPending(const double volume,
                                             const string symbol,
                                             const PS price_stop,
                                             const PL price_limit,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
                                             const string comment=NULL,
                                             const datetime expiration=0,
                                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);

//--- Create a pending request to place a (1) SellLimit, (2) SellStop, (3) SellStopLimit order
   template<typename PS,typename SL,typename TP>
   int                  PlaceSellLimitPending(const double volume,
                                             const string symbol,
                                             const PS price_set,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
                                             const string comment=NULL,
                                             const datetime expiration=0,
                                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename SL,typename TP>
   int                  PlaceSellStopPending(const double volume,
                                             const string symbol,
                                             const PS price_set,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
                                             const string comment=NULL,
                                             const datetime expiration=0,
                                             const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                             const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE);
   template<typename PS,typename PL,typename SL,typename TP>
   int                  PlaceSellStopLimitPending(const double volume,
                                             const string symbol,
                                             const PS price_stop,
                                             const PL price_limit,
                                             const SL sl=0,
                                             const TP tp=0,
                                             const ulong magic=ULONG_MAX,
                                             const uchar group_id1=0,
                                             const uchar group_id2=0,
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
```

Beyond the class body, implement all these methodswhile
renaming the methods of creating pending requests for opening positions along the way (we have done that previously):

```
//+------------------------------------------------------------------+
//| Create a pending request for opening a Buy position              |
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
   return this.m_trading.CreatePReqPosition(POSITION_TYPE_BUY,volume,symbol,magic,sl,tp,group_id1,group_id2,comment,deviation,type_filling);
  }
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
   return this.m_trading.CreatePReqPosition(POSITION_TYPE_SELL,volume,symbol,magic,sl,tp,group_id1,group_id2,comment,deviation,type_filling);
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a BuyLimit order               |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
int CEngine::PlaceBuyLimitPending(const double volume,
                                  const string symbol,
                                  const PS price_set,
                                  const SL sl=0,
                                  const TP tp=0,
                                  const ulong magic=ULONG_MAX,
                                  const uchar group_id1=0,
                                  const uchar group_id2=0,
                                  const string comment=NULL,
                                  const datetime expiration=0,
                                  const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                  const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqOrder(ORDER_TYPE_BUY_LIMIT,volume,symbol,price_set,0,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a BuyStop order                |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
int CEngine::PlaceBuyStopPending(const double volume,
                                 const string symbol,
                                 const PS price_set,
                                 const SL sl=0,
                                 const TP tp=0,
                                 const ulong magic=ULONG_MAX,
                                 const uchar group_id1=0,
                                 const uchar group_id2=0,
                                 const string comment=NULL,
                                 const datetime expiration=0,
                                 const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                 const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqOrder(ORDER_TYPE_BUY_STOP,volume,symbol,price_set,0,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a BuyStopLimit order           |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
int CEngine::PlaceBuyStopLimitPending(const double volume,
                                      const string symbol,
                                      const PS price_stop,
                                      const PL price_limit,
                                      const SL sl=0,
                                      const TP tp=0,
                                      const ulong magic=ULONG_MAX,
                                      const uchar group_id1=0,
                                      const uchar group_id2=0,
                                      const string comment=NULL,
                                      const datetime expiration=0,
                                      const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                      const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return
     (
      #ifdef __MQL4__ WRONG_VALUE #else
      this.m_trading.CreatePReqOrder(ORDER_TYPE_BUY_STOP_LIMIT,volume,symbol,price_stop,price_limit,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling);
      #endif
     );
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a SellLimit order              |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
int CEngine::PlaceSellLimitPending(const double volume,
                                   const string symbol,
                                   const PS price_set,
                                   const SL sl=0,
                                   const TP tp=0,
                                   const ulong magic=ULONG_MAX,
                                   const uchar group_id1=0,
                                   const uchar group_id2=0,
                                   const string comment=NULL,
                                   const datetime expiration=0,
                                   const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                   const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqOrder(ORDER_TYPE_SELL_LIMIT,volume,symbol,price_set,0,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a SellStop order               |
//+------------------------------------------------------------------+
template<typename PS,typename SL,typename TP>
int CEngine::PlaceSellStopPending(const double volume,
                                  const string symbol,
                                  const PS price_set,
                                  const SL sl=0,
                                  const TP tp=0,
                                  const ulong magic=ULONG_MAX,
                                  const uchar group_id1=0,
                                  const uchar group_id2=0,
                                  const string comment=NULL,
                                  const datetime expiration=0,
                                  const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                  const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return this.m_trading.CreatePReqOrder(ORDER_TYPE_SELL_STOP,volume,symbol,price_set,0,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling);
  }
//+------------------------------------------------------------------+
//| Create a pending request to place a SellStopLimit order          |
//+------------------------------------------------------------------+
template<typename PS,typename PL,typename SL,typename TP>
int CEngine::PlaceSellStopLimitPending(const double volume,
                                       const string symbol,
                                       const PS price_stop,
                                       const PL price_limit,
                                       const SL sl=0,
                                       const TP tp=0,
                                       const ulong magic=ULONG_MAX,
                                       const uchar group_id1=0,
                                       const uchar group_id2=0,
                                       const string comment=NULL,
                                       const datetime expiration=0,
                                       const ENUM_ORDER_TYPE_TIME type_time=WRONG_VALUE,
                                       const ENUM_ORDER_TYPE_FILLING type_filling=WRONG_VALUE)
  {
   return
     (
      #ifdef __MQL4__ WRONG_VALUE #else
      this.m_trading.CreatePReqOrder(ORDER_TYPE_SELL_STOP_LIMIT,volume,symbol,price_stop,price_limit,sl,tp,magic,group_id1,group_id2,comment,expiration,type_time,type_filling)
      #endif
     );
  }
//+------------------------------------------------------------------+
```

Here, the methods of creating pending requests for placing pending orders return the result of the method for creating a pending request of the **CTradingControl**
trading mamagement class receiving required pending order types corresponding to the method the pending request is created from. For MQL4,
return WRONG\_VALUE till we do not have the class of the pending StopLimit order object for
MQL4.

**These are all the changes required for placing pending orders under conditions using pending trading requests.**

### Testing

To perform the test, let's use [the EA from the previous article](https://www.mql5.com/en/articles/7521#node04)
and save it to \\MQL5\\Experts\\TestDoEasy\ **Part32\** under the name **TestDoEasyPart32.mq5**.

All we need to add to it is control over the states of the buttons managing pending requests activation for the appropriate pending order
placement buttons. If **P** or **T** (price and time condition) near the pending order placement button is pressed, such an order
is not placed immediately. Instead, a pending request is created. Its activation by a specified condition leads to placing the pending
order. The order is set relative to the price the pending request was activated at.

In the function that handles pressing the test EA's trading panel buttons, add two variables for storing the Point()
and Digits() values of the current symbol, as well as add
handling pressing the trading panel buttons for creating pending requests for placing all pending order types:

```
//+------------------------------------------------------------------+
//| Handle pressing the buttons                                      |
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
         //--- Otherwise, create a pending request to place a BuyLimit order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
            int id=engine.PlaceBuyLimitPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(ask-distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_ASK,price_act,EQUAL_OR_LESS,ask);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_LIMIT)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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
      //--- If the BUTT_BUY_STOP button is pressed: Set BuyStop
      else if(button==EnumToString(BUTT_BUY_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStop
         if(!pending_buy_stop)
            engine.PlaceBuyStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStop","Pending BuyStop order"));
         //--- Otherwise, create a pending request to place a BuyStop order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
            int id=engine.PlaceBuyStopPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(ask-distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_ASK,price_act,EQUAL_OR_LESS,ask);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_STOP)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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
      //--- If the BUTT_BUY_STOP_LIMIT button is pressed: Set BuyStopLimit
      else if(button==EnumToString(BUTT_BUY_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set BuyStopLimit
         if(!pending_buy_stoplimit)
            engine.PlaceBuyStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный BuyStopLimit","Pending BuyStopLimit order"));
         //--- Otherwise, create a pending request to place a BuyStopLimit order with the placement distances
         //--- and set the conditions depending on active buttons
         else
           {
            double ask=SymbolInfoDouble(NULL,SYMBOL_ASK);
            int id=engine.PlaceBuyStopLimitPending(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(ask-distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_ASK,price_act,EQUAL_OR_LESS,ask);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_BUY_STOP_LIMIT)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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
         //--- Otherwise, create a pending request to place a SellLimit order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
            int id=engine.PlaceSellLimitPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(bid+distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_BID,price_act,EQUAL_OR_MORE,bid);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_LIMIT)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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
      //--- If the BUTT_SELL_STOP button is pressed: Set SellStop
      else if(button==EnumToString(BUTT_SELL_STOP))
        {
         //--- If the pending request creation buttons are not pressed, set SellStop
         if(!pending_sell_stop)
            engine.PlaceSellStop(lot,Symbol(),distance_pending,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStop","Pending SellStop order"));
         //--- Otherwise, create a pending request to place a SellStop order with the placement distance
         //--- and set the conditions depending on active buttons
         else
           {
            double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
            int id=engine.PlaceSellStopPending(lot,Symbol(),distance_pending,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(bid+distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_BID,price_act,EQUAL_OR_MORE,bid);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_STOP)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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
      //--- If the BUTT_SELL_STOP_LIMIT button is pressed: Set SellStopLimit
      else if(button==EnumToString(BUTT_SELL_STOP_LIMIT))
        {
         //--- If the pending request creation buttons are not pressed, set SellStopLimit
         if(!pending_sell_stoplimit)
            engine.PlaceSellStopLimit(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic,TextByLanguage("Отложенный SellStopLimit","Pending SellStopLimit order"));
         //--- Otherwise, create a pending request to place a SellStopLimit order with the placement distances
         //--- and set the conditions depending on active buttons
         else
           {
            double bid=SymbolInfoDouble(NULL,SYMBOL_BID);
            int id=engine.PlaceSellStopLimitPending(lot,Symbol(),distance_pending,distance_stoplimit,stoploss,takeprofit,magic);
            if(id>0)
              {
               //--- If the price criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_PRICE"))
                 {
                  //--- set the pending request activation price
                  double price_act=NormalizeDouble(bid+distance_pending_request*point,digits);
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_BID,price_act,EQUAL_OR_MORE,bid);
                 }
               //--- If the time criterion is selected
               if(ButtonState(prefix+EnumToString(BUTT_SELL_STOP_LIMIT)+"_TIME"))
                 {
                  //--- set the pending request activation time
                  ulong control_time=TimeCurrent()+bars_delay_pending_request*PeriodSeconds();
                  engine.SetNewActivationProperties((uchar)id,PEND_REQ_ACTIVATION_SOURCE_SYMBOL,PEND_REQ_ACTIVATE_BY_SYMBOL_TIME,control_time,EQUAL_OR_MORE,TimeCurrent());
                 }
               //--- Get a newly created pending request by ID and display the message about adding the conditions to the journal
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

The codes of button pressing blocks feature detailed comments so there is no point in dwelling on them here. If you have any questions, feel
free to ask them in the comments.

**These are all the necessary changes to the test EA.**

Compile the EA and launch it in the tester in the visualization mode.

Simply enable the buttons of creating pending
requests for placing pending orders and see how pending requests are executed:

![](https://c.mql5.com/2/38/Uo7Lqyd2mH.gif)

A pending request for placing a pending order by price and time was created first, while the remaining pending requests were created by time
only. As we can see, all pending requests were activated upon the occurrence of their activation conditions: the first one — by price and
time, while the following ones — by their activation time. Thus, everything works as planned.

### What's next?

In the next article, we will continue the development of the pending trading request concept and implement closing positions (full,
partial and closing by an opposite one) by condition.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7536#node00)

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

[Part 31. Pending trading \\
requests - opening positions under certain conditions](https://www.mql5.com/en/articles/7521)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7536](https://www.mql5.com/ru/articles/7536)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7536.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7536/mql5.zip "Download MQL5.zip")(3662.36 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7536/mql4.zip "Download MQL4.zip")(3662.36 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/337462)**
(22)


![Marco Klaus Gerhard Niese](https://c.mql5.com/avatar/2021/3/606060F6-72B8.png)

**[Marco Klaus Gerhard Niese](https://www.mql5.com/en/users/mkgone)**
\|
21 May 2020 at 16:30

**Artyom Trishkin:**

OK. I'll check.

Thanks.

Any news?


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Sep 2020 at 08:47

**marco :**

Any news?

No, I haven't done it yet. All in turn, unfortunately.

![Anton Iaroshenko](https://c.mql5.com/avatar/2023/12/65894499-3534.jpg)

**[Anton Iaroshenko](https://www.mql5.com/en/users/topseller)**
\|
18 Apr 2023 at 12:43

Script fails to compile, finds 7 errors. Details in the image

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
18 Apr 2023 at 12:59

**TopSeller [#](https://www.mql5.com/ru/forum/331265#comment_46335142):**

Script fails to compile, finds 7 errors. Details in the image

Right now I don't have access to the computer where the files of past versions of the library are. Just go to the declaration of variables that are not allowed to be accessed because they are in the private section of the class, and specify aprotected section for them. Then return the private section:

protected:

variable

private:

This is my error, which occurred inadvertently, but it was missed by the compiler of the build of the terminal that I had at that time.

![Anton Iaroshenko](https://c.mql5.com/avatar/2023/12/65894499-3534.jpg)

**[Anton Iaroshenko](https://www.mql5.com/en/users/topseller)**
\|
18 Apr 2023 at 14:51

[@Artyom Trishkin](https://www.mql5.com/en/users/artmedia70), I'm sorry, I'm not very good at programming. I changed private to protected, but the errors remain.  Could you please post a working version of the script when you have time to do so?


![Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://c.mql5.com/2/38/OLAP_02.png)[Applying OLAP in trading (part 3): Analyzing quotes for the development of trading strategies](https://www.mql5.com/en/articles/7535)

In this article we will continue dealing with the OLAP technology applied to trading. We will expand the functionality presented in the first two articles. This time we will consider the operational analysis of quotes. We will put forward and test the hypotheses on trading strategies based on aggregated historical data. The article presents Expert Advisors for studying bar patterns and adaptive trading.

![Library for easy and quick development of MetaTrader programs (part XXXI): Pending trading requests - opening positions under certain conditions](https://c.mql5.com/2/37/MQL5-avatar-doeasy__19.png)[Library for easy and quick development of MetaTrader programs (part XXXI): Pending trading requests - opening positions under certain conditions](https://www.mql5.com/en/articles/7521)

Starting with this article, we are going to develop a functionality allowing users to trade using pending requests under certain conditions, for example, when reaching a certain time limit, exceeding a specified profit or closing a position by stop loss.

![Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://c.mql5.com/2/38/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part XXXIII): Pending trading requests - closing positions under certain conditions](https://www.mql5.com/en/articles/7554)

We continue the development of the library functionality featuring trading using pending requests. We have already implemented sending conditional trading requests for opening positions and placing pending orders. In the current article, we will implement conditional position closure – full, partial and closing by an opposite position.

![Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://c.mql5.com/2/37/Article_Logo__2.png)[Multicurrency monitoring of trading signals (Part 1): Developing the application structure](https://www.mql5.com/en/articles/7417)

In this article, we will discuss the idea of creating a multicurrency monitor of trading signals and will develop a future application structure along with its prototype, as well as create its framework for further operation. The article presents a step-by-step creation of a flexible multicurrency application which will enable the generation of trading signals and which will assist traders in finding the desired signals.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/7536&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070428653868750250)

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