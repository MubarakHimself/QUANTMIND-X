---
title: Library for easy and quick development of MetaTrader programs (part XXIX): Pending trading requests - request object classes
url: https://www.mql5.com/en/articles/7454
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T19:37:34.454693
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/7454&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070435684730213835)

MetaTrader 5 / Examples


### Contents

- [Concept](https://www.mql5.com/en/articles/7454#node01)
- [Base object of the abstract pending trading request](https://www.mql5.com/en/articles/7454#node02)
- [Descendant objects of the pending request base object](https://www.mql5.com/en/articles/7454#node03)
- [Testing](https://www.mql5.com/en/articles/7454#node04)
- [What's next?](https://www.mql5.com/en/articles/7454#node05)


### Concept

In the three previous articles, we checked the concept of managing trading methods of the trading class using pending requests.

[A \\
pending request](https://www.mql5.com/en/articles/7394) is, in fact, a common trading order executed by a certain condition. We checked the condition of delaying sending a
trading order in trading methods when receiving the server error whose handling requires some waiting before resending the request to the
server. Naturally, these are not all the conditions, under which pending requests can be used. Conditions may also include price levels,
upon reaching which a trade order is sent. It may also be a combination of conditions, for example, some threshold values of symbol
properties. Upon reaching these values, a trading order is sent to the server (a stop limit order is a good example of a pending trading
request for placing a limit order when the price reaches a stop order level).

In general, pending trading requests allow us to create some
kind of behavior logic for sending trading orders to the server.

However, in order to fit all this to the pending request object code, we need it to conform the [general \\
concept of library objects](https://www.mql5.com/en/articles/5654). This makes objects easily extensible in order to insert new properties into them. At the current stage, the
code for handling pending trading requests is located directly in the [trading class](https://www.mql5.com/en/articles/7258#node03)
listing. This is done purely to check the concept and is conceptually incorrect for further use. I have done this on purpose in order to
quickly check everything before wrapping the construction into the correct form.

In the current article, we are going to create the
base class of the abstract pending trading request object and the descendant object classes of the base request object. The base object is to
contain properties that are common for all property request objects, while the descendant objects contain individual properties native
to the status of each child object. This is done to all library objects and the current case is no exception.

But first, as usual, let's create the library messages necessary for handling the objects.

**Write the new library message indices in the Datas.mqh**
**file:**

...

```
   MSG_LIB_TEXT_AND_PAUSE,                            //  and pause
   MSG_LIB_TEXT_ALREADY_EXISTS,                       // already exists
   MSG_LIB_TEXT_CREATED,                              // Created
   MSG_LIB_TEXT_ATTEMPTS,                             // Attempts
   MSG_LIB_TEXT_WAIT,                                 // Wait
   MSG_LIB_TEXT_END,                                  // End
```

...

```
   MSG_LIB_TEXT_REQUEST,                              // Pending request #
   MSG_LIB_TEXT_REQUEST_DATAS,                        // Trading request parameters
   MSG_LIB_TEXT_PEND_REQUEST_DATAS,                   // Pending trading request parameters
```

...

```
   MSG_LIB_TEXT_PEND_REQUEST_BY_ERROR,                // Pending request generated based on the server return code
   MSG_LIB_TEXT_PEND_REQUEST_BY_REQUEST,              // Pending request created by request
   MSG_LIB_TEXT_PEND_REQUEST_WAITING_ONSET,           // Wait for the first trading attempt

   MSG_LIB_TEXT_PEND_REQUEST_STATUS,                  // Pending request status
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN,             // Pending request to open a position
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE,            // Pending request to close a position
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP,             // Pending request to modify position stop orders
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE,            // Pending request to place a pending order
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE,           // Pending request to delete a pending order
   MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY,           // Pending request to modify pending order parameters
  };
```

**and the messages corresponding to the new indices:**

```
   {" и паузой "," and pause "},
   {" уже существует"," already exists"},
   {"Создан","Created"},
   {"Попыток","Attempts"},
   {"Ожидание","Wait"},
   {"Окончание","End"},
```

...

```
   {"Отложенный запрос #","Pending request #"},
   {"Параметры торгового запроса","Trade request parameters"},
   {"Параметры отложенного торгового запроса","Pending trade request parameters"},
```

...

```
   {"Отложенный запрос, созданный по коду возврата сервера","Pending request created as a result of server code"},
   {"Отложенный запрос, созданный по запросу","Pending request created by request"},
   {"Ожидание наступления времени первой торговой попытки","Waiting for onset time of the first trading attempt"},

   {"Статус отложенного запроса","Pending request status"},
   {"Отложенный запрос на открытие позиции","Pending request to open position"},
   {"Отложенный запрос на закрытие позиции","Pending request to close position"},
   {"Отложенный запрос на модификацию стоп-приказов позиции","Pending request to modify position stop orders"},
   {"Отложенный запрос на установку отложенного ордера","Pending request to place pending order"},
   {"Отложенный запрос на удаление отложенного ордера","Pending request to remove pending order"},
   {"Отложенный запрос на модификацию параметров отложенного ордера","Pending request to modify pending order parameters"},
  };
```

As already mentioned above, the base object of a pending trading request is a general abstract request containing the properties present in
all trading requests, while the task of clarifying the properties is assigned to descendant objects of the base object.

Thus, we are
going to have a single base request object and six descendant objects clarifying the properties of a pending request by type of a performed
trading operation:

- opening a position (a new one on a hedge account, while adding volume and reversing on a netting account);

- modifying stop orders of an open position;
- closing a position — full and partial closure, as well as closing by an opposite order (on a hedge account);

- placing a pending order;
- modifying pending order properties — stop order prices, order placement price, StopLimit order price, order lifetime, filling type and
expiration type;

- removing a previously placed pending order.


### Base object of the abstract pending trading request

Let's create enumerations of the pending trading request object properties, like for all the previous library objects.

**In the Defines.mqh file, add enumerations of object statuses,**
**as well as of integer, real**
**and string properties:**

```
//+------------------------------------------------------------------+
//| Data for working with pending trading requests                   |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Pending request status                                           |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_STATUS
  {
   PEND_REQ_STATUS_OPEN,                                    // Pending request to open a position
   PEND_REQ_STATUS_CLOSE,                                   // Pending request to close a position
   PEND_REQ_STATUS_SLTP,                                    // Pending request to modify open position stop orders
   PEND_REQ_STATUS_PLACE,                                   // Pending request to place a pending order
   PEND_REQ_STATUS_REMOVE,                                  // Pending request to delete a pending order
   PEND_REQ_STATUS_MODIFY                                   // Pending request to modify a placed pending order
  };
//+------------------------------------------------------------------+
//| Pending request type                                             |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_TYPE
  {
   PEND_REQ_TYPE_ERROR=PENDING_REQUEST_ID_TYPE_ERR,         // Pending request created based on the return code or error
   PEND_REQ_TYPE_REQUEST=PENDING_REQUEST_ID_TYPE_REQ,       // Pending request created by request
  };
//+------------------------------------------------------------------+
//| Integer properties of a pending trading request                  |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_INTEGER
  {
   PEND_REQ_PROP_STATUS = 0,                                // Trading request status (from the ENUM_PEND_REQ_STATUS enumeration)
   PEND_REQ_PROP_TYPE,                                      // Trading request type (from the ENUM_PEND_REQ_TYPE enumeration)
   PEND_REQ_PROP_ID,                                        // Trading request ID
   PEND_REQ_PROP_RETCODE,                                   // Result a request is based on
   PEND_REQ_PROP_TIME_CREATE,                               // Request creation time
   PEND_REQ_PROP_TIME_ACTIVATE,                             // Next attempt activation time
   PEND_REQ_PROP_WAITING,                                   // Waiting time between requests
   PEND_REQ_PROP_CURENT,                                    // Current attempt index
   PEND_REQ_PROP_TOTAL,                                     // Number of attempts
   //--- MqlTradeRequest
   PEND_REQ_PROP_MQL_REQ_ACTION,                            // Type of a performed action in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE,                              // Order type in the request structure
   PEND_REQ_PROP_MQL_REQ_MAGIC,                             // EA stamp (magic number ID) in the request structure
   PEND_REQ_PROP_MQL_REQ_ORDER,                             // Order ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_POSITION,                          // Position ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_POSITION_BY,                       // Opposite position ticket in the request structure
   PEND_REQ_PROP_MQL_REQ_DEVIATION,                         // Maximum acceptable deviation from a requested price in the request structure
   PEND_REQ_PROP_MQL_REQ_EXPIRATION,                        // Order expiration time (for ORDER_TIME_SPECIFIED type orders) in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE_FILLING,                      // Order filling type in the request structure
   PEND_REQ_PROP_MQL_REQ_TYPE_TIME,                         // Order lifetime type in the request structure
  };
#define PEND_REQ_PROP_INTEGER_TOTAL (19)                    // Total number of integer event properties
#define PEND_REQ_PROP_INTEGER_SKIP  (0)                     // Number of request properties not used in sorting
//+------------------------------------------------------------------+
//| Real properties of a pending trading request                     |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_DOUBLE
  {
   PEND_REQ_PROP_PRICE_CREATE = PEND_REQ_PROP_INTEGER_TOTAL,// Price at the moment of a request generation
   //--- MqlTradeRequest
   PEND_REQ_PROP_MQL_REQ_VOLUME,                            // Requested volume of a deal in lots in the request structure
   PEND_REQ_PROP_MQL_REQ_PRICE,                             // Price in the request structure
   PEND_REQ_PROP_MQL_REQ_STOPLIMIT,                         // StopLimit level in the request structure
   PEND_REQ_PROP_MQL_REQ_SL,                                // Stop Loss level in the request structure
   PEND_REQ_PROP_MQL_REQ_TP,                                // Take Profit level in the request structure
  };
#define PEND_REQ_PROP_DOUBLE_TOTAL  (6)                     // Total number of event's real properties
#define PEND_REQ_PROP_DOUBLE_SKIP   (0)                     // Number of order properties not used in sorting
//+------------------------------------------------------------------+
//| String properties of a pending trading request                   |
//+------------------------------------------------------------------+
enum ENUM_PEND_REQ_PROP_STRING
  {
   //--- MqlTradeRequest
   PEND_REQ_PROP_MQL_REQ_SYMBOL = (PEND_REQ_PROP_INTEGER_TOTAL+PEND_REQ_PROP_DOUBLE_TOTAL),  // Trading instrument name in the request structure
   PEND_REQ_PROP_MQL_REQ_COMMENT                            // Order comment in the request structure
  };
#define PEND_REQ_PROP_STRING_TOTAL  (2)                     // Total number of event's string properties
//+------------------------------------------------------------------+
```

Macro substitutions specifying the number of each of property types are
used to calculate the property indexing in the object property arrays.

We have considered this [in \\
the first article](https://www.mql5.com/en/articles/5654#node04) when describing the library generation. There is no point in dwelling on that here.

As we can see in the listing, the object properties include both the properties belonging to a pending request creation parameters and the
properties specified in the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest)
trading request structure where all the parameters of a request sent to the server are set.

Since we want to sort and
look for objects in the list by any of its parameters (just like all the previous library objects), we copy the entire content of the trading
request structure to the appropriate object properties repeating the purpose of the structure fields. In this case, we can look for and sort
pending requests in the list by any property.

**In order to search and sort pending request objects, list all possible object sorting criteria:**

```
//+------------------------------------------------------------------+
//| Possible pending request sorting criteria                        |
//+------------------------------------------------------------------+
#define FIRST_PREQ_DBL_PROP         (PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_INTEGER_SKIP)
#define FIRST_PREQ_STR_PROP         (PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_INTEGER_SKIP+PEND_REQ_PROP_DOUBLE_TOTAL-PEND_REQ_PROP_DOUBLE_SKIP)
enum ENUM_SORT_PEND_REQ_MODE
  {
//--- Sort by integer properties
   SORT_BY_PEND_REQ_STATUS = 0,                             // Sort by a trading request status (from the ENUM_PEND_REQ_STATUS enumeration)
   SORT_BY_PEND_REQ_TYPE,                                   // Sort by a trading request type (from the ENUM_PEND_REQ_TYPE enumeration)
   SORT_BY_PEND_REQ_ID,                                     // Sort by a trading request ID
   SORT_BY_PEND_REQ_RETCODE,                                // Sort by a result a request is based on
   SORT_BY_PEND_REQ_TIME_CREATE,                            // Sort by a request generation time
   SORT_BY_PEND_REQ_TIME_ACTIVATE,                          // Sort by next attempt activation time
   SORT_BY_PEND_REQ_WAITING,                                // Sort by a waiting time between requests
   SORT_BY_PEND_REQ_CURENT,                                 // Sort by the current attempt index
   SORT_BY_PEND_REQ_TOTAL,                                  // Sort by a number of attempts
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_REQ_ACTION,                         // Sort by a type of a performed action in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE,                           // Sort by an order type in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_MAGIC,                          // Sort by an EA stamp (magic number ID) in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_ORDER,                          // Sort by an order ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_POSITION,                       // Sort by a position ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_POSITION_BY,                    // Sort by an opposite position ticket in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_DEVIATION,                      // Sort by a maximum acceptable deviation from a requested price in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_EXPIRATION,                     // Sort by an order expiration time (for ORDER_TIME_SPECIFIED type orders) in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE_FILLING,                   // Sort by an order filling type in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TYPE_TIME,                      // Sort by order lifetime type in the request structure
//--- Sort by real properties
   SORT_BY_PEND_REQ_PRICE_CREATE = FIRST_PREQ_DBL_PROP,     // Sort by a price at the moment of a request generation
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_REQ_VOLUME,                         // Sort by a requested volume of a deal in lots in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_PRICE,                          // Sort by a price in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_STOPLIMIT,                      // Sort by StopLimit order level in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_SL,                             // Sort by StopLoss order level in the request structure
   SORT_BY_PEND_REQ_MQL_REQ_TP,                             // Sort by TakeProfit order level in the request structure
//--- Sort by string properties
   //--- MqlTradeRequest
   SORT_BY_PEND_REQ_MQL_SYMBOL = FIRST_PREQ_STR_PROP,       // Sort by a trading instrument name in the request structure
   SORT_BY_PEND_REQ_MQL_COMMENT                             // Sort by an order comment in the request structure
  };
//+------------------------------------------------------------------+
```

In a separate file, create a new class of the abstract pending request base object.

In the \\MQL5\\Include\\DoEasy\ **Objects\** library directory, create a new **PendRequest\** subfolder to store files
of pending trading request classes.

**Create a new class of the abstract pending request base object in the PendRequest.mqh file of the PendRequest folder:**

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
//+------------------------------------------------------------------+
//| Abstract pending trading request class                           |
//+------------------------------------------------------------------+
class CPendRequest : public CObject
  {
private:
   MqlTradeRequest   m_request;                                         // Trade request structure
//--- Copy trading request data
   void              CopyRequest(const MqlTradeRequest &request);
//--- Return the index of the array the request (1) double and (2) string properties are actually located at
   int               IndexProp(ENUM_PEND_REQ_PROP_DOUBLE property)const { return(int)property-PEND_REQ_PROP_INTEGER_TOTAL;                               }
   int               IndexProp(ENUM_PEND_REQ_PROP_STRING property)const { return(int)property-PEND_REQ_PROP_INTEGER_TOTAL-PEND_REQ_PROP_DOUBLE_TOTAL;    }
protected:
   int               m_digits;                                          // Number of decimal places in a quote
   int               m_digits_lot;                                      // Number of decimal places in the symbol lot value
   bool              m_is_hedge;                                        // Hedging account flag
   long              m_long_prop[PEND_REQ_PROP_INTEGER_TOTAL];          // Request integer properties
   double            m_double_prop[PEND_REQ_PROP_DOUBLE_TOTAL];         // Request real properties
   string            m_string_prop[PEND_REQ_PROP_STRING_TOTAL];         // Request string properties
//--- Protected parametric constructor
                     CPendRequest(const ENUM_PEND_REQ_STATUS status,
                                  const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode);
//--- Return (1) the magic number specified in the settings, (2) hedging account flag
   ushort            GetMagicID(void)                                      const { return ushort(this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC) & 0xFFFF);}
   bool              IsHedge(void)                                         const { return this.m_is_hedge;                                               }
public:
//--- Default constructor
                     CPendRequest(){;}
//--- Set request (1) integer, (2) real and (3) string properties
   void              SetProperty(ENUM_PEND_REQ_PROP_INTEGER property,long value) { this.m_long_prop[property]=value;                                     }
   void              SetProperty(ENUM_PEND_REQ_PROP_DOUBLE property,double value){ this.m_double_prop[this.IndexProp(property)]=value;                   }
   void              SetProperty(ENUM_PEND_REQ_PROP_STRING property,string value){ this.m_string_prop[this.IndexProp(property)]=value;                   }
//--- Return (1) integer, (2) real and (3) string request properties from the properties array
   long              GetProperty(ENUM_PEND_REQ_PROP_INTEGER property)      const { return this.m_long_prop[property];                                    }
   double            GetProperty(ENUM_PEND_REQ_PROP_DOUBLE property)       const { return this.m_double_prop[this.IndexProp(property)];                  }
   string            GetProperty(ENUM_PEND_REQ_PROP_STRING property)       const { return this.m_string_prop[this.IndexProp(property)];                  }

//--- Return the flag of the request supporting the property
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)        { return true; }
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)         { return true; }
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property)         { return true; }

//--- Compare CPendRequest objects by a specified property (to sort the lists by a specified request object property)
   virtual int       Compare(const CObject *node,const int mode=0) const;
//--- Compare CPendRequest objects by all properties (to search for equal request objects)
   bool              IsEqual(CPendRequest* compared_obj);

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
   uchar                CurrentAttempt(void)                               const { return (uchar)this.GetProperty(PEND_REQ_PROP_CURENT);                 }
   uchar                TotalAttempts(void)                                const { return (uchar)this.GetProperty(PEND_REQ_PROP_TOTAL);                  }
   uchar                ID(void)                                           const { return (uchar)this.GetProperty(PEND_REQ_PROP_ID);                     }
   int                  Retcode(void)                                      const { return (int)this.GetProperty(PEND_REQ_PROP_RETCODE);                  }
   ulong                Order(void)                                        const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER);                 }
   ulong                Position(void)                                     const { return this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION);              }
   ENUM_TRADE_REQUEST_ACTIONS Action(void)                                 const { return (ENUM_TRADE_REQUEST_ACTIONS)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION);}

//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) ID,
//--- (8) order ticket, (9) position ticket
   void                 SetPriceCreate(const double price)                       { this.SetProperty(PEND_REQ_PROP_PRICE_CREATE,price);                   }
   void                 SetTimeCreate(const ulong time)                          { this.SetProperty(PEND_REQ_PROP_TIME_CREATE,time);                     }
   void                 SetTimeActivate(const ulong time)                        { this.SetProperty(PEND_REQ_PROP_TIME_ACTIVATE,time);                   }
   void                 SetWaitingMSC(const ulong miliseconds)                   { this.SetProperty(PEND_REQ_PROP_WAITING,miliseconds);                  }
   void                 SetCurrentAttempt(const uchar number)                    { this.SetProperty(PEND_REQ_PROP_CURENT,number);                        }
   void                 SetTotalAttempts(const uchar number)                     { this.SetProperty(PEND_REQ_PROP_TOTAL,number);                         }
   void                 SetID(const uchar id)                                    { this.SetProperty(PEND_REQ_PROP_ID,id);                                }
   void                 SetOrder(const ulong ticket)                             { this.SetProperty(PEND_REQ_PROP_MQL_REQ_ORDER,ticket);                 }
   void                 SetPosition(const ulong ticket)                          { this.SetProperty(PEND_REQ_PROP_MQL_REQ_POSITION,ticket);              }

//+------------------------------------------------------------------+
//| Descriptions of request object properties                        |
//+------------------------------------------------------------------+
//--- Get description of a request (1) integer, (2) real and (3) string property
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_INTEGER property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_DOUBLE property);
   string               GetPropertyDescription(ENUM_PEND_REQ_PROP_STRING property);

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
//--- (2) short event message (implementation in the class descendants) in the journal
   void                 Print(const bool full_prop=false);
   virtual void         PrintShort(void){;}
  };
//+------------------------------------------------------------------+
```

We have already created identical objects many times with a detailed description of their structure. I think, there is no point in dwelling
here on the object structure principles. Just like all other similar objects, it features three arrays for storing integer, real and string
properties. Besides, it features the methods of accessing these properties by specifying the property constant — GetProperty(), as well
as simplified ones with self-explanatory method names. There are also methods displaying descriptions of all object properties. In
general, everything is standard for library objects. You can explore the structure [of \\
the library objects in the very first article](https://www.mql5.com/en/articles/5654#node04).

**Write the closed class constructor outside the class body:**

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
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
  }
//+------------------------------------------------------------------+
```

First of all, copy trading request structure data to the arrays of
integer, real and string properties using the CopyRequest() method we are to consider below.

Next, set
the hedging account flag, set the number of decimal places in a symbol quote and in a symbol lot value (this is required for the correct
display of the info in the journal). If the number of decimal places in the lot value is zero, only one decimal place is displayed for more
convenience: "1.0" is shown instead of "1".

Next, simply set passed values of some properties to the class constructor when creating a
request object for the values of the appropriate object properties.

Thus, we fill all the properties of a pending request object
immediately upon its creation.

**The method of comparing two pending request objects by a specified property (mode):**

```
//+------------------------------------------------------------------+
//| Compare CPendRequest objects by a specified property             |
//+------------------------------------------------------------------+
int CPendRequest::Compare(const CObject *node,const int mode=0) const
  {
   const CPendRequest *compared_obj=node;
//--- compare integer properties of two events
   if(mode<PEND_REQ_PROP_INTEGER_TOTAL)
     {
      long value_compared=compared_obj.GetProperty((ENUM_PEND_REQ_PROP_INTEGER)mode);
      long value_current=this.GetProperty((ENUM_PEND_REQ_PROP_INTEGER)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare integer properties of two objects
   if(mode<PEND_REQ_PROP_DOUBLE_TOTAL+PEND_REQ_PROP_INTEGER_TOTAL)
     {
      double value_compared=compared_obj.GetProperty((ENUM_PEND_REQ_PROP_DOUBLE)mode);
      double value_current=this.GetProperty((ENUM_PEND_REQ_PROP_DOUBLE)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
//--- compare string properties of two objects
   else if(mode<PEND_REQ_PROP_DOUBLE_TOTAL+PEND_REQ_PROP_INTEGER_TOTAL+PEND_REQ_PROP_STRING_TOTAL)
     {
      string value_compared=compared_obj.GetProperty((ENUM_PEND_REQ_PROP_STRING)mode);
      string value_current=this.GetProperty((ENUM_PEND_REQ_PROP_STRING)mode);
      return(value_current>value_compared ? 1 : value_current<value_compared ? -1 : 0);
     }
   return 0;
  }
//+------------------------------------------------------------------+
```

**The method of a full comparison of two pending request objects by all its properties:**

```
//+------------------------------------------------------------------+
//| Compare CPendRequest objects by all properties                   |
//+------------------------------------------------------------------+
bool CPendRequest::IsEqual(CPendRequest *compared_obj)
  {
   int beg=0, end=PEND_REQ_PROP_INTEGER_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_INTEGER prop=(ENUM_PEND_REQ_PROP_INTEGER)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=PEND_REQ_PROP_DOUBLE_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_DOUBLE prop=(ENUM_PEND_REQ_PROP_DOUBLE)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
   beg=end; end+=PEND_REQ_PROP_STRING_TOTAL;
   for(int i=beg; i<end; i++)
     {
      ENUM_PEND_REQ_PROP_STRING prop=(ENUM_PEND_REQ_PROP_STRING)i;
      if(this.GetProperty(prop)!=compared_obj.GetProperty(prop)) return false;
     }
//--- All properties are equal
   return true;
  }
//+------------------------------------------------------------------+
```

We have also considered such methods when creating the previous library objects.

As you may remember, the **Compare()**
method of a request object used to compare by a specified property compares the specified properties of the current current CPendRequest class
object with the object of the same class passed to the method. If the current object value exceeds the value of a compared object, the method
returns 1, if its is less than the value of a compared object, the method returns -1, otherwise 0. This is necessary for working with lists of
pointers to objects — with its [Search()](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch)
method using [the Compare() virtual method of the \\
standard library base object](https://www.mql5.com/en/docs/standardlibrary/cobject/cobjectcompare) for a quick comparison. Since the method is virtual, it should be redefined in the CObject class
descendants. This is exactly what I have done here.

The **IsEqual()** method compares each property of two compared objects one
by one and returns false as soon as inequality is detected — the objects are not equal. Upon
completion of three loops by all properties of the two compared objects, true is returned
— all object properties are equal.

**The method for copying the trading request structure to the arrays of pending request object properties:**

```
//+------------------------------------------------------------------+
//| Copy trading request data                                        |
//+------------------------------------------------------------------+
void CPendRequest::CopyRequest(const MqlTradeRequest &request)
  {
//--- Copy a passed structure to an object structure
   this.m_request=request;

//--- Integer properties of a trading request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_ACTION,request.action);             // Type of a performed action in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_TYPE,request.type);                 // Order type in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC,request.magic);               // EA stamp (magic number ID) in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_ORDER,request.order);               // Order ticket in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_POSITION,request.position);         // Position ticket in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_POSITION_BY,request.position_by);   // Opposite position ticket in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_DEVIATION,request.deviation);       // Maximum acceptable deviation from a requested price in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_EXPIRATION,request.expiration);     // Order expiration time (for ORDER_TIME_SPECIFIED type orders) in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_FILLING,request.type_filling); // Order filling type in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_TIME,request.type_time);       // Order lifetime type in the request structure
//--- Real properties of a trading request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME,request.volume);             // Requested volume of a deal in lots in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_PRICE,request.price);               // Price in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT,request.stoplimit);       // StopLimit level in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_SL,request.sl);                     // Stop Loss level in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_TP,request.tp);                     // Take Profit level in the request structure
//--- String properties of a trading request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL,request.symbol);             // Trading instrument name in the request structure
   this.SetProperty(PEND_REQ_PROP_MQL_REQ_COMMENT,request.comment);           // Order comment in the request structure
  }
//+------------------------------------------------------------------+
```

Here we first copy the values of all the fields of the trading request structure
passed to the methodto the same request object structure.
Then the values of all structure fields are set for object properties using the SetProperty() methods for placing properties.

**The methods returning descriptions of integer, real**
**and string properties of a pending request object:**

```
//+------------------------------------------------------------------+
//| Return the description of the request integer property           |
//+------------------------------------------------------------------+
string CPendRequest::GetPropertyDescription(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   return
     (
      property==PEND_REQ_PROP_STATUS               ?  this.StatusDescription()            :
      property==PEND_REQ_PROP_TYPE                 ?  this.TypeRequestDescription()       :
      property==PEND_REQ_PROP_ID                   ?  this.IDDescription()                :
      property==PEND_REQ_PROP_RETCODE              ?  this.RetcodeDescription()           :
      property==PEND_REQ_PROP_TIME_CREATE          ?  this.TimeCreateDescription()        :
      property==PEND_REQ_PROP_TIME_ACTIVATE        ?  this.TimeActivateDescription()      :
      property==PEND_REQ_PROP_WAITING              ?  this.TimeWaitingDescription()       :
      property==PEND_REQ_PROP_CURENT               ?  this.CurrentAttemptDescription()    :

      property==PEND_REQ_PROP_TOTAL                ?  this.TotalAttemptsDescription()     :
      //--- MqlTradeRequest
      property==PEND_REQ_PROP_MQL_REQ_ACTION       ?  this.MqlReqActionDescription()      :
      property==PEND_REQ_PROP_MQL_REQ_TYPE         ?  this.MqlReqTypeOrderDescription()   :
      property==PEND_REQ_PROP_MQL_REQ_MAGIC        ?  this.MqlReqMagicDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_ORDER        ?  this.MqlReqOrderDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_POSITION     ?  this.MqlReqPositionDescription()    :
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ?  this.MqlReqPositionByDescription()  :
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION    ?  this.MqlReqDeviationDescription()   :
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ?  this.MqlReqExpirationDescription()  :
      property==PEND_REQ_PROP_MQL_REQ_TYPE_FILLING ?  this.MqlReqTypeFillingDescription() :
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME    ?  this.MqlReqTypeTimeDescription()    :
      ::EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the request real property              |
//+------------------------------------------------------------------+
string CPendRequest::GetPropertyDescription(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return
     (
      property==PEND_REQ_PROP_PRICE_CREATE      ?  this.PriceCreateDescription()          :
      //--- MqlTradeRequest
      property==PEND_REQ_PROP_MQL_REQ_VOLUME    ?  this.MqlReqVolumeDescription()         :
      property==PEND_REQ_PROP_MQL_REQ_PRICE     ?  this.MqlReqPriceDescription()          :
      property==PEND_REQ_PROP_MQL_REQ_STOPLIMIT ?  this.MqlReqStopLimitDescription()      :
      property==PEND_REQ_PROP_MQL_REQ_SL        ?  this.MqlReqStopLossDescription()       :
      property==PEND_REQ_PROP_MQL_REQ_TP        ?  this.MqlReqTakeProfitDescription()     :
      ::EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
//| Return the description of the request string property            |
//+------------------------------------------------------------------+
string CPendRequest::GetPropertyDescription(ENUM_PEND_REQ_PROP_STRING property)
  {
   return
     (
      property==PEND_REQ_PROP_MQL_REQ_SYMBOL    ?  this.MqlReqSymbolDescription()         :
      property==PEND_REQ_PROP_MQL_REQ_COMMENT   ?  this.MqlReqCommentDescription()        :
      ::EnumToString(property)
     );
  }
//+------------------------------------------------------------------+
```

The method receives the property and its string description is
returned depending on the property value **using the methods returning the property description:**

```
//+------------------------------------------------------------------+
//| Return the pending request status name                           |
//+------------------------------------------------------------------+
string CPendRequest::StatusDescription(void) const
  {
   int code_descr=
     (
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_OPEN   ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_OPEN     :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_CLOSE  ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE    :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_SLTP   ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP     :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_PLACE  ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE    :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_REMOVE ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE   :
      this.GetProperty(PEND_REQ_PROP_STATUS)==PEND_REQ_STATUS_MODIFY ?  MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY   :
      MSG_EVN_STATUS_UNKNOWN
     );
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS)+": "+CMessage::Text(code_descr);
  }
//+------------------------------------------------------------------+
//| Return the pending request type name                             |
//+------------------------------------------------------------------+
string CPendRequest::TypeRequestDescription(void) const
  {
   int code_descr=
     (
      this.GetProperty(PEND_REQ_PROP_TYPE)==PEND_REQ_TYPE_ERROR   ?  MSG_LIB_TEXT_PEND_REQUEST_BY_ERROR     :
      this.GetProperty(PEND_REQ_PROP_TYPE)==PEND_REQ_TYPE_REQUEST ?  MSG_LIB_TEXT_PEND_REQUEST_BY_REQUEST   :
      MSG_SYM_MODE_UNKNOWN
     );
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TYPE)+": "+CMessage::Text(code_descr);
  }
//+------------------------------------------------------------------+
//| Return the pending request ID description                        |
//+------------------------------------------------------------------+
string CPendRequest::IDDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ID)+": #"+(string)this.GetProperty(PEND_REQ_PROP_ID);
  }
//+------------------------------------------------------------------+
//| Return the description of an error code a request is based on    |
//+------------------------------------------------------------------+
string CPendRequest::RetcodeDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_RETCODE)+": "+
          CMessage::Text((int)this.GetProperty(PEND_REQ_PROP_RETCODE))+
          " ("+(string)this.GetProperty(PEND_REQ_PROP_RETCODE)+")";
  }
//+------------------------------------------------------------------+
//| Return the description of a pending request creation time        |
//+------------------------------------------------------------------+
string CPendRequest::TimeCreateDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TIME_CREATE)+": "+::TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
  }
//+------------------------------------------------------------------+
//| Return the description of a pending request activation time      |
//+------------------------------------------------------------------+
string CPendRequest::TimeActivateDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TIME_ACTIVATE)+": "+::TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_ACTIVATE));
  }
//+----------------------------------------------------------------------+
//| Return the description of a time spent waiting for a pending request |
//+----------------------------------------------------------------------+
string CPendRequest::TimeWaitingDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_WAITING)+": "+
          (string)this.GetProperty(PEND_REQ_PROP_WAITING)+
          " ("+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_MINUTES|TIME_SECONDS)+")";
  }
//+------------------------------------------------------------------+
//| Return the description of the current pending request attempt    |
//+------------------------------------------------------------------+
string CPendRequest::CurrentAttemptDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_CURRENT_ATTEMPT)+": "+(string)this.GetProperty(PEND_REQ_PROP_CURENT);
  }
//+------------------------------------------------------------------+
//| Return the description of a number of pending request attempts   |
//+------------------------------------------------------------------+
string CPendRequest::TotalAttemptsDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TOTAL_ATTEMPTS)+": "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
  }
//+------------------------------------------------------------------+
//| Return the description of a price when creating a request        |
//+------------------------------------------------------------------+
string CPendRequest::PriceCreateDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_PRICE_CREATE)+": "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_PRICE_CREATE),this.m_digits);
  }
//+------------------------------------------------------------------+
//| Return the executed action type description                      |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqActionDescription(void) const
  {
   int code_descr=
     (
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_DEAL       ?  MSG_LIB_TEXT_REQUEST_ACTION_DEAL       :
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_PENDING    ?  MSG_LIB_TEXT_REQUEST_ACTION_PENDING    :
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_SLTP       ?  MSG_LIB_TEXT_REQUEST_ACTION_SLTP       :
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_MODIFY     ?  MSG_LIB_TEXT_REQUEST_ACTION_MODIFY     :
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_REMOVE     ?  MSG_LIB_TEXT_REQUEST_ACTION_REMOVE     :
      this.GetProperty(PEND_REQ_PROP_MQL_REQ_ACTION)==TRADE_ACTION_CLOSE_BY   ?  MSG_LIB_TEXT_REQUEST_ACTION_CLOSE_BY   :
      MSG_LIB_TEXT_REQUEST_ACTION_UNCNOWN
     );
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_ACTION)+": "+CMessage::Text(code_descr);
  }
//+------------------------------------------------------------------+
//| Return the magic number value description                        |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqMagicDescription(void) const
  {
   return CMessage::Text(MSG_ORD_MAGIC)+": "+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC)+
          (this.GetMagicID()!=this.GetProperty(PEND_REQ_PROP_MQL_REQ_MAGIC) ? " ("+(string)this.GetMagicID()+")" : "");
  }
//+------------------------------------------------------------------+
//| Return the order ticket value description                        |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqOrderDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_ORDER)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER)>0 ?
           "#"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request position ticket description                   |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqPositionDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_POSITION)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION)>0 ?
           (string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request opposite position ticket description          |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqPositionByDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_POSITION_BY)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION_BY)>0 ?
           (string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION_BY) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request deviation size description                    |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqDeviationDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_DEVIATION)+": "+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_DEVIATION);
  }
//+------------------------------------------------------------------+
//| Return the request order type description                        |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqTypeOrderDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE)+": "+OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
  }
//+------------------------------------------------------------------+
//| Return the request order filling mode description                |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqTypeFillingDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE_FILLING)+": "+OrderTypeFillingDescription((ENUM_ORDER_TYPE_FILLING)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_FILLING));
  }
//+------------------------------------------------------------------+
//| Return the request order lifetime type description               |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqTypeTimeDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TYPE_TIME)+": "+OrderTypeTimeDescription((ENUM_ORDER_TYPE_TIME)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_TIME));
  }
//+------------------------------------------------------------------+
//| Return the request order expiration time description             |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqExpirationDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_EXPIRATION)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_EXPIRATION)>0 ?
           ::TimeToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_EXPIRATION)) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request volume description                            |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqVolumeDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_VOLUME)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME)>0 ?
           ::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request price value description                       |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqPriceDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE)>0 ?
           ::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE),this.m_digits) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request StopLimit order price description             |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqStopLimitDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_STOPLIMIT)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)>0 ?
           ::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT),this.m_digits) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request StopLoss order price description              |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqStopLossDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)>0 ?
           ::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL),this.m_digits) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the request TakeProfit order price description            |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqTakeProfitDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)>0 ?
           ::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP),this.m_digits) :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
//| Return the description of a trading instrument name in a request |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqSymbolDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_SYMBOL)+": "+this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL);
  }
//+------------------------------------------------------------------+
//| Return the request order comment description                     |
//+------------------------------------------------------------------+
string CPendRequest::MqlReqCommentDescription(void) const
  {
   return CMessage::Text(MSG_LIB_TEXT_REQUEST_COMMENT)+": "+
          (this.GetProperty(PEND_REQ_PROP_MQL_REQ_COMMENT)!="" && this.GetProperty(PEND_REQ_PROP_MQL_REQ_COMMENT)!=NULL ?
           "\""+this.GetProperty(PEND_REQ_PROP_MQL_REQ_COMMENT)+"\"" :
           CMessage::Text(MSG_LIB_PROP_NOT_SET));
  }
//+------------------------------------------------------------------+
```

The value of the passed property is checked in the methods, its text description is created and the generated text line is returned.

**The Print() method is used to display all the object properties in the journal:**

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
   ::Print("================== ",CMessage::Text(MSG_LIB_PARAMS_LIST_END),": \"",CMessage::Text(header_code),"\" ==================\n");
  }
//+------------------------------------------------------------------+
```

We have created such methods for all the previous library objects and analyzed their structure from the very beginning of the library
description, so I will provide only the following brief description here:

Move by three object property arrays in three loops, get each
successive object property and display its description in the journal depending on the flag of displaying only supported object
properties.

The **PrintShort()** method is meant for displaying a short request object description and is made virtual. It does nothing in the
base object and should be re-defined in descendant objects so that each descendant is able to display only its own properties in the journal.

We have created the base object of the abstract pending request.

In order to clarify what request should be generated, we
need to create six descendant objects based on the abstract request object. They will be created in the trading methods of the trading class
when handling errors that require waiting. Besides, they will be created as independent trading requests when creating a trading logic
using pending trading requests.

### Descendant objects of the pending request base object

Each trading method of the trading class features the blocks of creating pending requests. We are going to create the appropriate pending
requests based on the operation carried out by the trading method. But for now, we only have a base abstract pending request. In order to
generate various trading requests by the type of executed operation, we need to create six descendants where each descendant class sends
only its own trading operation to the server. I have already provided the list of such objects above:

- opening a position
- modifying stop orders of an open position
- closing a position
- placing a pending order
- modifying pending order properties
- removing a previously placed pending order

As you may remember, the protected constructor of the base abstract request features the parameter
indicating the generated request status:

```
//--- Protected parametric constructor
                     CPendRequest(const ENUM_PEND_REQ_STATUS status,
                                  const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode);
```

We will display this status (passing from the inherited object constructor to the base object one) when creating a new pending request. This
allows us to set what request type we are generating by an executed operation. We analyzed this behavior in the second article when [creating \\
objects of historical orders and deals](https://www.mql5.com/en/articles/5669#node01). Therefore, we will not stop at this point but instead consider all six pending
request objects by the type of executed trading operations.

All classes of the base trading request descendant objects are created in
the same folder where the base object is located:

\\MQL5\\Include\\DoEasy\\Objects\ **PendRequest\**

**The class of the pending request object for opening a position** **(PendReqOpen.mqh class file):**

```
//+------------------------------------------------------------------+
//|                                                  PendReqOpen.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request for opening a position                           |
//+------------------------------------------------------------------+
class CPendReqOpen : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqOpen(const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode) : CPendRequest(PEND_REQ_STATUS_OPEN,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION     ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return(property==PEND_REQ_PROP_MQL_REQ_STOPLIMIT ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
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
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

The method is quite primitive. Its only function is to pass the status of a created object (opening a position) and all inputs of its
constructor to
the base object constructor in the initialization list:

```
//--- Constructor
   CPendReqOpen(const uchar id,
                const double price,
                const ulong time,
                const MqlTradeRequest &request,
                const int retcode) : CPendRequest(PEND_REQ_STATUS_OPEN,id,price,time,request,retcode) {}
```

**The virtual methods returning the flag of supporting some integer,**
**real and string**
**properties by the object:**

```
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION     ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return(property==PEND_REQ_PROP_MQL_REQ_STOPLIMIT ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqOpen::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
```

If a descendant object of the base abstract pending request does not support any property, the method returns 'false', otherwise — 'true'.
We have already considered this [in the second article when considering \\
generating objects of historical orders and deals](https://www.mql5.com/en/articles/5669#node01).

**The virtual method of creating and displaying a brief request description:**

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
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

The lines describing some parameters used by the object are created in the method. The final message is compiled of them and displayed in the
journal. The method is virtual and implemented in its own way in each of the descendant objects of the abstract request base object.

**The class of the pending request object for modifying stop orders of an open position** **(PendReqSLTP.mqh class file):**

```
//+------------------------------------------------------------------+
//|                                                  PendReqSLTP.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request to modify position stop orders                   |
//+------------------------------------------------------------------+
class CPendReqSLTP : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqSLTP(const uchar id,
                                  const double price,
                                  const ulong time,
                                  const MqlTradeRequest &request,
                                  const int retcode) : CPendRequest(PEND_REQ_STATUS_SLTP,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqSLTP::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ||
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION    ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_FILLING ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqSLTP::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   if(property==PEND_REQ_PROP_PRICE_CREATE         ||
      property==PEND_REQ_PROP_MQL_REQ_SL           ||
      property==PEND_REQ_PROP_MQL_REQ_TP
     ) return true;
   return false;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqSLTP::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqSLTP::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE))+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION);
   string sl=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL),this.m_digits) : "";
   string tp=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP),this.m_digits) : "";
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_SLTP)+": "+
   "\n- "+params+sl+tp+
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request object for closing a position** **(PendReqClose.mqh class file):**

```
//+------------------------------------------------------------------+
//|                                                 PendReqClose.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request to close a position                              |
//+------------------------------------------------------------------+
class CPendReqClose : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqClose(const uchar id,
                                   const double price,
                                   const ulong time,
                                   const MqlTradeRequest &request,
                                   const int retcode) : CPendRequest(PEND_REQ_STATUS_CLOSE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqClose::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if((
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  &&
      this.GetProperty(property)==0)               ||
      property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_EXPIRATION   ||
      property==PEND_REQ_PROP_MQL_REQ_TYPE_TIME
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqClose::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_STOPLIMIT    ||
      property==PEND_REQ_PROP_MQL_REQ_SL           ||
      property==PEND_REQ_PROP_MQL_REQ_TP
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqClose::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqClose::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 PositionTypeDescription((ENUM_POSITION_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE))+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION);
   string pos_by=(this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION_BY)>0 ? ", "+CMessage::Text(MSG_ORD_DEAL_OUT_BY)+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_POSITION_BY) : "");
   string price=CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_PRICE_CREATE),this.m_digits);
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_CLOSE)+": "+
   "\n- "+params+", "+price+pos_by+
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request object for placing a pending order** **(PendReqPlace.mqh class file):**

```
//+------------------------------------------------------------------+
//|                                                 PendReqPlace.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request to place a pending order                         |
//+------------------------------------------------------------------+
class CPendReqPlace : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqPlace(const uchar id,
                                   const double price,
                                   const ulong time,
                                   const MqlTradeRequest &request,
                                   const int retcode) : CPendRequest(PEND_REQ_STATUS_PLACE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqPlace::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_ORDER        ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION     ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqPlace::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqPlace::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqPlace::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE));
   string price=CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE),this.m_digits);
   string stoplimit=this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_STOPLIMIT)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT),this.m_digits) : "";
   string sl=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL),this.m_digits) : "";
   string tp=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP),this.m_digits) : "";
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_PLACE)+": "+
   "\n- "+params+", "+price+stoplimit+sl+tp+
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request object for modifying parameters of a placed pending order** **(PendReqModify.mqh class**
**file):**

```
//+------------------------------------------------------------------+
//|                                                PendReqModify.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request to modify pending order parameters               |
//+------------------------------------------------------------------+
class CPendReqModify : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqModify(const uchar id,
                                    const double price,
                                    const ulong time,
                                    const MqlTradeRequest &request,
                                    const int retcode) : CPendRequest(PEND_REQ_STATUS_MODIFY,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqModify::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   if(property==PEND_REQ_PROP_MQL_REQ_POSITION     ||
      property==PEND_REQ_PROP_MQL_REQ_POSITION_BY  ||
      property==PEND_REQ_PROP_MQL_REQ_DEVIATION
     ) return false;
   return true;
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqModify::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return(property==PEND_REQ_PROP_MQL_REQ_VOLUME ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqModify::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqModify::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE))+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER);
   string price=CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE),this.m_digits);
   string stoplimit=this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_STOPLIMIT)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT),this.m_digits) : "";
   string sl=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_SL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_SL),this.m_digits) : "";
   string tp=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_TP)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_TP),this.m_digits) : "";
   string expiration=this.GetProperty(PEND_REQ_PROP_MQL_REQ_EXPIRATION)>0 ? this.MqlReqExpirationDescription() : "";
   string type_filling=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_FILLING)>WRONG_VALUE ? "\n- "+this.MqlReqTypeFillingDescription() : "";
   string type_time=this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE_TIME)>WRONG_VALUE ? "\n- "+this.MqlReqTypeTimeDescription() : "";
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_MODIFY)+": "+
   "\n- "+params+", "+price+stoplimit+sl+tp+expiration+type_filling+type_time+
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

**The class of the pending request object for removing a pending order** **(PendReqRemove.mqh class file):**

```
//+------------------------------------------------------------------+
//|                                                PendReqRemove.mqh |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                             https://mql5.com/en/users/artmedia70 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://mql5.com/en/users/artmedia70"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include "PendRequest.mqh"
//+------------------------------------------------------------------+
//| Pending request to remove a pending order                        |
//+------------------------------------------------------------------+
class CPendReqRemove : public CPendRequest
  {
public:
//--- Constructor
                     CPendReqRemove(const uchar id,
                                    const double price,
                                    const ulong time,
                                    const MqlTradeRequest &request,
                                    const int retcode) : CPendRequest(PEND_REQ_STATUS_REMOVE,id,price,time,request,retcode) {}

//--- Supported deal properties (1) real, (2) integer
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property);
   virtual bool      SupportProperty(ENUM_PEND_REQ_PROP_STRING property);
//--- Display a brief message with request data in the journal
   virtual void      PrintShort(void);
  };
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| integer property, otherwise return 'false'                       |
//+------------------------------------------------------------------+
bool CPendReqRemove::SupportProperty(ENUM_PEND_REQ_PROP_INTEGER property)
  {
   return(property>PEND_REQ_PROP_MQL_REQ_ORDER ? false : true);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| real property, otherwise return 'false'                          |
//+------------------------------------------------------------------+
bool CPendReqRemove::SupportProperty(ENUM_PEND_REQ_PROP_DOUBLE property)
  {
   return(property==PEND_REQ_PROP_PRICE_CREATE || property==PEND_REQ_PROP_MQL_REQ_PRICE ? true : false);
  }
//+------------------------------------------------------------------+
//| Return 'true' if an order supports a passed                      |
//| string property, otherwise return 'false'                        |
//+------------------------------------------------------------------+
bool CPendReqRemove::SupportProperty(ENUM_PEND_REQ_PROP_STRING property)
  {
   return true;
  }
//+------------------------------------------------------------------+
//| Display a brief message with request data in the journal         |
//+------------------------------------------------------------------+
void CPendReqRemove::PrintShort(void)
  {
   string params=this.GetProperty(PEND_REQ_PROP_MQL_REQ_SYMBOL)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_VOLUME),this.m_digits_lot)+" "+
                 OrderTypeDescription((ENUM_ORDER_TYPE)this.GetProperty(PEND_REQ_PROP_MQL_REQ_TYPE))+" #"+(string)this.GetProperty(PEND_REQ_PROP_MQL_REQ_ORDER);
   string price=CMessage::Text(MSG_LIB_TEXT_REQUEST_PRICE)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_PRICE),this.m_digits);
   string stoplimit=this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT)>0 ? ", "+CMessage::Text(MSG_LIB_TEXT_REQUEST_STOPLIMIT)+" "+::DoubleToString(this.GetProperty(PEND_REQ_PROP_MQL_REQ_STOPLIMIT),this.m_digits) : "";
   string time=this.IDDescription()+", "+CMessage::Text(MSG_LIB_TEXT_CREATED)+" "+TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE));
   string attempts=CMessage::Text(MSG_LIB_TEXT_ATTEMPTS)+" "+(string)this.GetProperty(PEND_REQ_PROP_TOTAL);
   string wait=CMessage::Text(MSG_LIB_TEXT_WAIT)+" "+::TimeToString(this.GetProperty(PEND_REQ_PROP_WAITING)/1000,TIME_SECONDS);
   string end=CMessage::Text(MSG_LIB_TEXT_END)+" "+
              TimeMSCtoString(this.GetProperty(PEND_REQ_PROP_TIME_CREATE)+this.GetProperty(PEND_REQ_PROP_WAITING)*this.GetProperty(PEND_REQ_PROP_TOTAL));
   //---
   string message=CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_STATUS_REMOVE)+": "+
   "\n- "+params+", "+price+stoplimit+
   "\n- "+time+", "+attempts+", "+wait+", "+end+"\n";
   ::Print(message);
  }
//+------------------------------------------------------------------+
```

The most important distinguishing feature of all these classes from each other is passing
a pending
request status to the parent object constructor \- it is the status that determines what type of performed trading
operation each of these objects is to conduct. The remaining class methods serve only auxiliary purposes and do not affect the trading
result.

This completes the creation of new classes of pending trading requests.

Now we need to remove the temporary pending
request class from the trading class file and make additional changes to work with new objects of pending trading requests.

**Open Trading.mqh and completely remove the previous pending request class from**
**its listing:**

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
#include "Collections\EventsCollection.mqh"
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
//--- Copy trading request data
   void                 CopyRequest(const MqlTradeRequest &request)  { this.m_request=request;        }
//--- Compare CPendingReq objects by IDs
   virtual int          Compare(const CObject *node,const int mode=0) const;

public:
//--- Return (1) the request structure, (2) the price at the moment of the request generation,
//--- (3) request generation time, (4) next attempt activation time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
//--- (9) result a request is based on,
//--- (10) order ticket, (11) position ticket, (12) trading operation type
   MqlTradeRequest      MqlRequest(void)                       const { return this.m_request;         }
   double               PriceCreate(void)                      const { return this.m_price_create;    }
   ulong                TimeCreate(void)                       const { return this.m_time_create;     }
   ulong                TimeActivate(void)                     const { return this.m_time_activate;   }
   ulong                WaitingMSC(void)                       const { return this.m_waiting_msc;     }
   uchar                CurrentAttempt(void)                   const { return this.m_current_attempt; }
   uchar                TotalAttempts(void)                    const { return this.m_total_attempts;  }
   uchar                ID(void)                               const { return this.m_id;              }
   int                  Retcode(void)                          const { return this.m_retcode;         }
   ulong                Order(void)                            const { return this.m_request.order;   }
   ulong                Position(void)                         const { return this.m_request.position;}
   ENUM_TRADE_REQUEST_ACTIONS Action(void)                     const { return this.m_request.action;  }

//--- Set (1) the price when creating a request, (2) request creation time,
//--- (3) current attempt time, (4) waiting time between requests,
//--- (5) current attempt index, (6) number of attempts, (7) ID,
//--- (8) order ticket, (9) position ticket
   void                 SetPriceCreate(const double price)           { this.m_price_create=price;     }
   void                 SetTimeCreate(const ulong time)              { this.m_time_create=time;       }
   void                 SetTimeActivate(const ulong time)            { this.m_time_activate=time;     }
   void                 SetWaitingMSC(const ulong miliseconds)       { this.m_waiting_msc=miliseconds;}
   void                 SetCurrentAttempt(const uchar number)        { this.m_current_attempt=number; }
   void                 SetTotalAttempts(const uchar number)         { this.m_total_attempts=number;  }
   void                 SetID(const uchar id)                        { this.m_id=id;                  }
   void                 SetOrder(const ulong ticket)                 { this.m_request.order=ticket;   }
   void                 SetPosition(const ulong ticket)              { this.m_request.position=ticket;}

//--- Return the description of the (1) request structure, (2) the price at the moment of the request generation,
//--- (3) request generation time, (4) current attempt time,
//--- (5) waiting time between requests, (6) current attempt index,
//--- (7) number of attempts, (8) request ID
   string               MqlRequestDescription(void)            const { return RequestActionDescription(this.m_request); }
   string               TypeDescription(void)                  const
                          {
                           return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_TYPE) +
                             (this.Type()==PENDING_REQUEST_ID_TYPE_ERR           ?
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
   string               IdDescription(void)                    const { return CMessage::Text(MSG_LIB_TEXT_PEND_REQUEST_ID)+"#"+(string)this.ID();                          }
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
//| Compare CPendingReq objects by properties                        |
//+------------------------------------------------------------------+
int CPendingReq::Compare(const CObject *node,const int mode=0) const
  {
   const CPendingReq *compared_req=node;
   return
     (
      //--- Compare by ID
      mode==SORT_BY_PEND_REQ_ID        ?
      (this.ID()>compared_req.ID()     ? 1 : this.ID()<compared_req.ID() ? -1 : 0)     :
      //--- Compare by type
      mode==SORT_BY_PEND_REQ_TYPE      ?
      (this.Type()>compared_req.Type() ? 1 : this.Type()<compared_req.Type() ? -1 : 0) :
      //--- Compare by ticket
      (
       //--- modifying position sl, tp, opening/closing a position or closing by an opposite one
       this.m_request.action==TRADE_ACTION_SLTP    || this.m_request.action==TRADE_ACTION_DEAL     || this.m_request.action==TRADE_ACTION_CLOSE_BY ?
       (this.m_request.position>compared_req.m_request.position ? 1 : this.m_request.position<compared_req.m_request.position ? -1 : 0)            :
       //--- modifying parameters, placing/removing a pending order
       this.m_request.action==TRADE_ACTION_MODIFY  || this.m_request.action==TRADE_ACTION_PENDING  || this.m_request.action==TRADE_ACTION_REMOVE   ?
       (this.m_request.order>compared_req.m_request.order ? 1 : this.m_request.order<compared_req.m_request.order ? -1 : 0)                        :
       //--- otherwise
       0
      )
     );
  }
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
//+------------------------------------------------------------------+
//| Trading class                                                    |
//+------------------------------------------------------------------+
class CTrading
  {
```

**To let the trading class see the objects of new classes of pending requests, connect**
**them to the trading class file listing:**

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
#include "Objects\PendRequest\PendReqOpen.mqh"
#include "Objects\PendRequest\PendReqSLTP.mqh"
#include "Objects\PendRequest\PendReqClose.mqh"
#include "Objects\PendRequest\PendReqPlace.mqh"
#include "Objects\PendRequest\PendReqModify.mqh"
#include "Objects\PendRequest\PendReqRemove.mqh"
#include "Collections\AccountsCollection.mqh"
#include "Collections\SymbolsCollection.mqh"
#include "Collections\MarketCollection.mqh"
#include "Collections\HistoryCollection.mqh"
#include "Collections\EventsCollection.mqh"
//+------------------------------------------------------------------+
```

The method of creating a new pending request now additionally passes the status of a created trading request.

**Add**
**the status to the method declaration:**

```
//--- Create a pending request
   bool                 CreatePendingRequest(const ENUM_PEND_REQ_STATUS status,
                                             const uchar id,
                                             const uchar attempts,
                                             const ulong wait,
                                             const MqlTradeRequest &request,
                                             const int retcode,
                                             CSymbol *symbol_obj);
```

**Also, fix the method for implementing a new pending request:**

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
                                    CSymbol *symbol_obj)
  {
   //--- Create a new pending request object depending on a request status
   CPendRequest *req_obj=NULL;
   switch(status)
     {
      case PEND_REQ_STATUS_OPEN     : req_obj=new CPendReqOpen(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_CLOSE    : req_obj=new CPendReqClose(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_SLTP     : req_obj=new CPendReqSLTP(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);    break;
      case PEND_REQ_STATUS_PLACE    : req_obj=new CPendReqPlace(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);   break;
      case PEND_REQ_STATUS_REMOVE   : req_obj=new CPendReqRemove(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);  break;
      case PEND_REQ_STATUS_MODIFY   : req_obj=new CPendReqModify(id,symbol_obj.BidLast(),symbol_obj.Time(),request,retcode);  break;
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
   //--- Filled in the fields of a successfully created object by the values passed to the method
   req_obj.SetTimeActivate(symbol_obj.Time()+wait);
   req_obj.SetWaitingMSC(wait);
   req_obj.SetCurrentAttempt(0);
   req_obj.SetTotalAttempts(attempts);

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

Now the method additionally receives the status of a created
pending request object.

Previously, the pending request object was of the CPendingReq class type. Now the class has been completely
removed, so currently we have a new class of the **CPendRequest** pending request object and its descendants.

Therefore,
when creating an empty object, we use the new object of the CPendRequest
base abstract request.

Next, depending on a status passed to the method, create the
appropriate new pending request object.

If the object is successfully created, the
journal displays a brief message featuring the parameters of a created pending trading request.

Since we now have a new object of a pending request class, replace all occurrences of the previous CPendingReq class with **CPendRequest**
in the trading class listing. This can be conveniently done using **Ctrl+H** (search and replace).

When creating new pending request objects, they are to obtain a bit more data on a conducted trading operation. Basically, this is necessary
for the display in the journal. The data is passed by writing additional data to the trading request structure when creating a new object of a
pending request.

**Specify the**
**status of the created request in the pending request creation block of the position opening method:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the trading request magic number, has no pending request ID
         if(this.GetPendReqID((uint)magic)==0)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the pending request object ID to the magic number and fill in the remaining unfilled fields of the trading request structure
            uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
            this.SetPendReqID((uchar)id,mn);
            this.m_request.magic=mn;
            this.m_request.action=TRADE_ACTION_DEAL;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.type=order_type;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_OPEN,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

**In the method modifying stop orders of an open position, namely in**
**the pending request creation block, add passing** ****an order type** and a position volume to the**
**request object via the structure, as well as set the created request status:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true));
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write a type of a conducted operation, as well as a symbol and a ticket of a modified position to the request structure
            this.m_request.action=TRADE_ACTION_SLTP;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.type=order_type;
            this.m_request.volume=order.Volume();
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_SLTP,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

**In the position closing method, namely in its block of creating a**
**pending request, add passing** **an order type and a position volume** **to the request**
**object via the structure, as well as set the created request status:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the trading operation type, symbol, ticket and volume to the request structure
            this.m_request.action=TRADE_ACTION_DEAL;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.volume=(volume==WRONG_VALUE || volume>order.Volume() ? order.Volume() : volume);
            this.m_request.type=order_type;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_CLOSE,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
```

A position volume may be negative in the method (-1)
indicating the position is closed in full. If the volume is specified as a certain value, the position is closed partially. All this is
handled in the base trading object of a symbol when sending a trading request to the server.

Here we immediately check the volume passed
to the method and set the required volume to the trading request structure.

**In the method of closing a position by an opposite one, in its**
**block of creating a pending request, add passing** **an order type and a position volume to the request object**
**via the structure, as well as set the created request status:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the position ticket is not present in the list
         if(this.GetIndexPendingRequestByPosition(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj_pos.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj_pos.SetResultRetcode(code);
               trade_obj_pos.SetResultComment(CMessage::Text(trade_obj_pos.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the trading operation type, symbol and tickets of two positions to the request structure
            this.m_request.action=TRADE_ACTION_CLOSE_BY;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.position=ticket;
            this.m_request.position_by=ticket_by;
            this.m_request.type=order_type;
            this.m_request.volume=order.Volume();
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_CLOSE,(uchar)id,attempts,wait,this.m_request,trade_obj_pos.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

**Specify the**
**status of the created request in the pending request creation block of the pending order placement method:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the trading request magic number, has no pending request ID
         if(this.GetPendReqID((uint)magic)==0)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Write the request ID to the magic number, while a symbol name is set in the request structure,
            //--- trading operation and order types
            uint mn=(magic==ULONG_MAX ? (uint)trade_obj.GetMagic() : (uint)magic);
            this.SetPendReqID((uchar)id,mn);
            this.m_request.magic=mn;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.action=TRADE_ACTION_PENDING;
            this.m_request.type=order_type;
               //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_PLACE,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

**In the method modifying pending order parameters, namely in its**
**pending request creation block,** **add passing** **an order type to the request object via the structure, as well as set**
**the created request status:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the order ticket is not present in the list
         if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type,(sl<0 ? false : true),(tp<0 ? false : true),(price>0 || limit>0 ? true : false));
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Set the trading operation type, as well as modified order's symbol and ticket in the request structure
            this.m_request.action=TRADE_ACTION_MODIFY;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.order=ticket;
            this.m_request.type=order_type;
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_MODIFY,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

**In the method removing a pending order, namely in its pending request**
**creation block, add passing** **an order type and volume and its price,**
**as well as set the created request status:**

```
      //--- If the check result is "waiting" - set the last error code to the return structure and display the message in the journal,
      //--- create a pending request and return 'false'
      if(method>ERROR_CODE_PROCESSING_METHOD_REFRESH)
        {
         //--- If the pending request object with the order ticket is not present in the list
         if(this.GetIndexPendingRequestByOrder(ticket)==WRONG_VALUE)
           {
            //--- Play the error sound
            if(this.IsUseSounds())
               trade_obj.PlaySoundError(action,order_type);
            //--- set the last error code to the return structure
            int code=this.m_list_errors.At(this.m_list_errors.Total()-1);
            if(code!=NULL)
              {
               if(code==MSG_LIB_TEXT_TERMINAL_NOT_TRADE_ENABLED || code==MSG_LIB_TEXT_EA_NOT_TRADE_ENABLED)
                  code=10027;
               trade_obj.SetResultRetcode(code);
               trade_obj.SetResultComment(CMessage::Text(trade_obj.GetResultRetcode()));
              }
            //--- Waiting time in milliseconds:
            //--- for the "Wait and repeat" handling method, the waiting value corresponds to the 'method' value,
            //--- for the "Create a pending request" handling method - till there is a zero waiting time
            ulong wait=method;
            //--- Look for the least of the possible IDs. If failed to find
            //--- or in case of an error while updating the current symbol data, return 'false'
            int id=this.GetFreeID();
            if(id<1 || !symbol_obj.RefreshRates())
               return false;
            //--- Set the trading operation type, as well as deleted order's symbol and ticket in the request structure
            this.m_request.action=TRADE_ACTION_REMOVE;
            this.m_request.symbol=symbol_obj.Name();
            this.m_request.order=ticket;
            this.m_request.type=order_type;
            this.m_request.volume=order.Volume();
            this.m_request.price=order.PriceOpen();
            //--- Set the number of trading attempts and create a pending request
            uchar attempts=(this.m_total_try < 1 ? 1 : this.m_total_try);
            this.CreatePendingRequest(PEND_REQ_STATUS_REMOVE,(uchar)id,attempts,wait,this.m_request,trade_obj.GetResultRetcode(),symbol_obj);
           }
         return false;
        }
```

We have two such blocks per each trading method. Therefore, we need to insert the changes to each of them.

**Set sorting types in the methods searching for a free ID and**
**returning request object indices:**

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
         id=i;
         break;
        }
     }
   delete element;
   return id;
  }
//+------------------------------------------------------------------+
//| Return the request object index in the list by ID                |
//+------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByID(const uchar id)
  {
   CPendRequest *req=new CPendRequest();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetID(id);
   this.m_list_request.Sort(SORT_BY_PEND_REQ_ID);
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+------------------------------------------------------------------+
//| Return the request object index in the list by the order ticket  |
//+------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByOrder(const ulong ticket)
  {
   CPendRequest *req=new CPendRequest();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetOrder(ticket);
   this.m_list_request.Sort(SORT_BY_PEND_REQ_MQL_REQ_ORDER);
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+-------------------------------------------------------------------+
//| Return the request object index in the list by the position ticket|
//+-------------------------------------------------------------------+
int CTrading::GetIndexPendingRequestByPosition(const ulong ticket)
  {
   CPendRequest *req=new CPendRequest();
   if(req==NULL)
      return WRONG_VALUE;
   req.SetPosition(ticket);
   this.m_list_request.Sort(SORT_BY_PEND_REQ_MQL_REQ_POSITION);
   int index=this.m_list_request.Search(req);
   delete req;
   return index;
  }
//+------------------------------------------------------------------+
```

To search for an object by a specified property in the list using the [Search() \\
method of the class of dynamic pointers to the CObject class instances](https://www.mql5.com/en/docs/standardlibrary/datastructures/carrayobj/carrayobjsearch), we need to apply the [Sort() \\
method of the base class of the CArray dynamic array of variables](https://www.mql5.com/en/docs/standardlibrary/datastructures/carray/carraysort) to specify the sorting type of the list where
the search is performed since the search is conducted only in sorted arrays. This is exactly what we do here.

Now improve the display of messages with a trading attempt number in the journal in the class timer.

**Replace the code**

```
      //--- Set the attempt number in the request object
      req_obj.SetCurrentAttempt(uchar(req_obj.CurrentAttempt()+1));

      //--- Display the number of a trading attempt in the journal

      if(this.m_log_level>LOG_LEVEL_NO_MSG)
         ::Print(CMessage::Text(MSG_LIB_TEXT_REQUEST)+(string)req_obj.GetProperty(PEND_REQ_PROP_ID)+" "+CMessage::Text(MSG_LIB_TEXT_RE_TRY_N)+(string)req_obj.CurrentAttempt());

      //--- Depending on the type of action performed in the trading request
      switch(request.action)
        {
```

**with the following one:**

```
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
```

This provides us with more informative journal message about a pending request object the next trading attempt is conducted for.

**This concludes all the necessary changes to let the library work with pending trading request objects.**

### Testing

To test new pending request objects, use the EA from the previous article and save it in \\MQL5\\Experts\\TestDoEasy\ **Part29\**
under the name **TestDoEasyPart29.mq5**.

All we need to do is remove a redundant check in the blocks of closing half of Buy and Sell positions since all is done in trading methods of the
tading class.

**In the block of partial closing of a Buy position, replace the code:**

```
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
                  engine.OpenSell(NormalizeLot(position.Symbol(),position.Volume()/2.0),position.Symbol(),position.Magic(),position.StopLoss(),position.TakeProfit(),"Частичное закрытие Buy #"+(string)position.Ticket());
              }
           }
        }
```

**with the following one:**

```
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
            //--- Close the Buy position partially
            if(position!=NULL)
               engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
           }
        }
```

**Also, re-write the partial Sell position closure the same way:**

```
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
            //--- Close the Sell position partially
            if(position!=NULL)
               engine.ClosePositionPartially((ulong)position.Ticket(),position.Volume()/2.0);
           }
        }
```

Compile and launch the EA.

Currently, I have four active pending orders. Let's disable the AutoTrading button and click the test EA's "Delete pending" button to remove all
pending orders. The following messages are displayed in the journal:

```
2019.12.19 04:25:04.385 CTrading::DeleteOrder: Invalid request:
2019.12.19 04:25:04.385 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.19 04:25:04.385 Correction of trade request parameters ...
2019.12.19 04:25:04.385 Pending request created #1:
2019.12.19 04:25:04.385 Pending request to remove a pending order:
2019.12.19 04:25:04.385 - EURUSD 0.10 Pending order Buy Limit #497568391, Price 1.11076
2019.12.19 04:25:04.385 - Pending request ID: #1, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:04.385
2019.12.19 04:25:04.386 CTrading::DeleteOrder: Invalid request:
2019.12.19 04:25:04.386 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.19 04:25:04.386 Correction of trade request parameters ...
2019.12.19 04:25:04.386 Pending request created #2:
2019.12.19 04:25:04.386 Pending request to remove a pending order:
2019.12.19 04:25:04.386 - EURUSD 0.10 Pending order Buy Limit #497563913, Price 1.11099
2019.12.19 04:25:04.386 - Pending request ID: #2, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:04.386
2019.12.19 04:25:04.386 CTrading::DeleteOrder: Invalid request:
2019.12.19 04:25:04.386 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.19 04:25:04.386 Correction of trade request parameters ...
2019.12.19 04:25:04.386 Pending request created #3:
2019.12.19 04:25:04.386 Pending request to remove a pending order:
2019.12.19 04:25:04.386 - EURUSD 0.10 Pending order Sell Limit #496816788, Price 1.11829
2019.12.19 04:25:04.386 - Pending request ID: #3, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:04.386
2019.12.19 04:25:04.386 CTrading::DeleteOrder: Invalid request:
2019.12.19 04:25:04.386 There is no permission to conduct trading operations in the terminal (the "AutoTrading" button is disabled)
2019.12.19 04:25:04.386 Correction of trade request parameters ...
2019.12.19 04:25:04.387 Pending request created #4:
2019.12.19 04:25:04.387 Pending request to remove a pending order:
2019.12.19 04:25:04.387 - EURUSD 0.10 Pending order Sell Limit #496816707, Price 1.11784
2019.12.19 04:25:04.387 - Pending request ID: #4, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:04.387
```

As we can see, the EA has made four attempts to remove four pending orders. After facing the error of disabled trading from the terminal side,
it created four pending requests accompanied by short messages with their parameters.

Then it is time to wait.

After some time (20 seconds, although that depends on the arrival of ticks), the messages from pending
request objects about repeated attempts to remove the four orders are displayed in the journal:

```
2019.12.19 04:25:38.780 Retry trading attempt #1:
2019.12.19 04:25:38.780 Pending request to remove a pending order:
2019.12.19 04:25:38.780 - EURUSD 0.10 Pending order Sell Limit #496816707, Price 1.11784
2019.12.19 04:25:38.780 - Pending request ID: #4, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:38.780
2019.12.19 04:25:38.780 Retry trading attempt #1:
2019.12.19 04:25:38.780 Pending request to remove a pending order:
2019.12.19 04:25:38.780 - EURUSD 0.10 Pending order Buy Limit #497563913, Price 1.11099
2019.12.19 04:25:38.780 - Pending request ID: #2, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:38.780
2019.12.19 04:25:38.781 Retry trading attempt #1:
2019.12.19 04:25:38.781 Pending request to remove a pending order:
2019.12.19 04:25:38.781 - EURUSD 0.10 Pending order Sell Limit #496816788, Price 1.11829
2019.12.19 04:25:38.781 - Pending request ID: #3, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:38.781
2019.12.19 04:25:39.103 Retry trading attempt #1:
2019.12.19 04:25:39.103 Pending request to remove a pending order:
2019.12.19 04:25:39.103 - EURUSD 0.10 Pending order Buy Limit #497568391, Price 1.11076
2019.12.19 04:25:39.103 - Pending request ID: #1, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:39.103
2019.12.19 04:25:42.006 Retry trading attempt #2:
2019.12.19 04:25:42.006 Pending request to remove a pending order:
2019.12.19 04:25:42.006 - EURUSD 0.10 Pending order Buy Limit #497568391, Price 1.11076
2019.12.19 04:25:42.006 - Pending request ID: #1, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:42.006
2019.12.19 04:25:42.007 Retry trading attempt #2:
2019.12.19 04:25:42.007 Pending request to remove a pending order:
2019.12.19 04:25:42.007 - EURUSD 0.10 Pending order Buy Limit #497563913, Price 1.11099
2019.12.19 04:25:42.007 - Pending request ID: #2, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:42.007
2019.12.19 04:25:42.007 Retry trading attempt #2:
2019.12.19 04:25:42.007 Pending request to remove a pending order:
2019.12.19 04:25:42.007 - EURUSD 0.10 Pending order Sell Limit #496816788, Price 1.11829
2019.12.19 04:25:42.007 - Pending request ID: #3, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:42.007
2019.12.19 04:25:42.008 Retry trading attempt #2:
2019.12.19 04:25:42.008 Pending request to remove a pending order:
2019.12.19 04:25:42.008 - EURUSD 0.10 Pending order Sell Limit #496816707, Price 1.11784
2019.12.19 04:25:42.008 - Pending request ID: #4, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:25:42.008
2019.12.19 04:26:03.026 Retry trading attempt #3:
2019.12.19 04:26:03.026 Pending request to remove a pending order:
2019.12.19 04:26:03.026 - EURUSD 0.10 Pending order Buy Limit #497568391, Price 1.11076
2019.12.19 04:26:03.026 - Pending request ID: #1, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:03.026
2019.12.19 04:26:03.027 Retry trading attempt #3:
2019.12.19 04:26:03.027 Pending request to remove a pending order:
2019.12.19 04:26:03.027 - EURUSD 0.10 Pending order Buy Limit #497563913, Price 1.11099
2019.12.19 04:26:03.027 - Pending request ID: #2, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:03.027
2019.12.19 04:26:03.027 Retry trading attempt #3:
2019.12.19 04:26:03.028 Pending request to remove a pending order:
2019.12.19 04:26:03.028 - EURUSD 0.10 Pending order Sell Limit #496816788, Price 1.11829
2019.12.19 04:26:03.028 - Pending request ID: #3, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:03.028
2019.12.19 04:26:03.028 Retry trading attempt #3:
2019.12.19 04:26:03.028 Pending request to remove a pending order:
2019.12.19 04:26:03.028 - EURUSD 0.10 Pending order Sell Limit #496816707, Price 1.11784
2019.12.19 04:26:03.028 - Pending request ID: #4, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:03.028
2019.12.19 04:26:22.357 Retry trading attempt #4:
2019.12.19 04:26:22.357 Pending request to remove a pending order:
2019.12.19 04:26:22.357 - EURUSD 0.10 Pending order Buy Limit #497568391, Price 1.11076
2019.12.19 04:26:22.357 - Pending request ID: #1, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:22.357
2019.12.19 04:26:22.358 Retry trading attempt #4:
2019.12.19 04:26:22.358 Pending request to remove a pending order:
2019.12.19 04:26:22.358 - EURUSD 0.10 Pending order Buy Limit #497563913, Price 1.11099
2019.12.19 04:26:22.358 - Pending request ID: #2, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:22.358
2019.12.19 04:26:22.358 Retry trading attempt #4:
2019.12.19 04:26:22.358 Pending request to remove a pending order:
2019.12.19 04:26:22.358 - EURUSD 0.10 Pending order Sell Limit #496816788, Price 1.11829
2019.12.19 04:26:22.358 - Pending request ID: #3, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
2019.12.19 04:26:22.358
2019.12.19 04:26:22.359 Retry trading attempt #4:
2019.12.19 04:26:22.359 Pending request to remove a pending order:
2019.12.19 04:26:22.359 - EURUSD 0.10 Pending order Sell Limit #496816707, Price 1.11784
2019.12.19 04:26:22.359 - Pending request ID: #4, Created 2019.12.18 23:25:01.920, Attempts 5, Wait 00:00:20, End 2019.12.18 23:26:41.920
```

Wait for the next trading attempt again. When the time comes, pending request objects are removed due to the expiration of the allotted time for
all trading attempts:

```
2019.12.19 04:26:44.004 Pending request ID: #1: Deleted due to expiration
2019.12.19 04:26:44.004 Pending request ID: #2: Deleted due to expiration
2019.12.19 04:26:44.004 Pending request ID: #3: Deleted due to expiration
2019.12.19 04:26:44.004 Pending request ID: #4: Deleted due to expiration
```

If the AutoTrading button was clicked within the time allotted for all trading attempts, the pending request objects would send a command to
the server to remove pending orders and they would be removed.

In exactly the same way, I checked pending trading request objects for closing/opening positions, placing orders, modifying position
stop orders and pending order parameters and closing a position by an opposite one. It turned out that pending trading requests work
correctly in most cases. However, if we first try to modify a pending order and then modify it again, such an attempt will fail. The same goes
for conducting two and more trading operations using the same ticket. Pending requests will simply be removed immediately after creation
with the "Executed" reason. In the previous articles, checks of pending request activations were made only by a ticket in the trading class
timer. The check method detected in the account history that an operation with a certain ticket had just been carried out. Accordingly, the
system assumed that everything worked, and it was time to remove this request object. We will fix that later.

**Please note:**

Developing the class
of pending request objects is currently a work in progress. Therefore, do not use the results described in the article and the attached
test EA in real trading!

The article and its accompanying materials are not a finished product and are in no way intended for real
trading. Instead, they are only meant for the demo
mode or the tester.

### What's next?

In the next article, we will create the class of managing pending trading requests and get rid of the inability to conduct the same trading
operation with the same order or position several times in a row.

All files of the current version of the library are attached below together with the test EA files for you to test and download.

Leave
your questions, comments and suggestions in the comments.

[Back to contents](https://www.mql5.com/en/articles/7454#node00)

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

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7454](https://www.mql5.com/ru/articles/7454)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7454.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/7454/mql5.zip "Download MQL5.zip")(3635.64 KB)

[MQL4.zip](https://www.mql5.com/en/articles/download/7454/mql4.zip "Download MQL4.zip")(3635.64 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/334758)**
(2)


![Vladimir Pastushak](https://c.mql5.com/avatar/2025/10/690362ed-8db6.png)

**[Vladimir Pastushak](https://www.mql5.com/en/users/voldemar)**
\|
24 Dec 2019 at 10:53

There you go Artem, the further into the forest the more code.....

Thanks!

![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
24 Dec 2019 at 11:08

**Vladimir Pastushak:**

You've got it all screwed up Artem, the further into the woods the more code....

Thanks!

:) It would be cool if the further into the forest the less..., but I can't do that yet ;)

![Library for easy and quick development of MetaTrader programs (part XXX): Pending trading requests - managing request objects](https://c.mql5.com/2/37/MQL5-avatar-doeasy__18.png)[Library for easy and quick development of MetaTrader programs (part XXX): Pending trading requests - managing request objects](https://www.mql5.com/en/articles/7481)

In the previous article, we have created the classes of pending request objects corresponding to the general concept of library objects. This time, we are going to deal with the class allowing the management of pending request objects.

![Neural Networks Made Easy](https://c.mql5.com/2/48/Neural_networks_made_easy_001.png)[Neural Networks Made Easy](https://www.mql5.com/en/articles/7447)

Artificial intelligence is often associated with something fantastically complex and incomprehensible. At the same time, artificial intelligence is increasingly mentioned in everyday life. News about achievements related to the use of neural networks often appear in different media. The purpose of this article is to show that anyone can easily create a neural network and use the AI achievements in trading.

![Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://c.mql5.com/2/37/MQL5-avatar-continuous_optimization__2.png)[Continuous Walk-Forward Optimization (Part 3): Adapting a Robot to Auto Optimizer](https://www.mql5.com/en/articles/7490)

The third part serves as a bridge between the previous two parts: it describes the mechanism of interaction with the DLL considered in the first article and the objects for report downloading, which were described in the second article. We will analyze the process of wrapper creation for a class which is imported from DLL and which forms an XML file with the trading history. We will also consider a method for interacting with this wrapper.

![Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://c.mql5.com/2/37/MQL5-avatar-doeasy__16.png)[Library for easy and quick development of MetaTrader programs (part XXVIII): Closure, removal and modification of pending trading requests](https://www.mql5.com/en/articles/7438)

This is the third article about the concept of pending requests. We are going to complete the tests of pending trading requests by creating the methods for closing positions, removing pending orders and modifying position and pending order parameters.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/7454&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070435684730213835)

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