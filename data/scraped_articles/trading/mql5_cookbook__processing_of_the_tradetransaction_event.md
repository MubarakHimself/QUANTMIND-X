---
title: MQL5 Cookbook: Processing of the TradeTransaction Event
url: https://www.mql5.com/en/articles/1111
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:19:47.099748
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/1111&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069347842528576401)

MetaTrader 5 / Examples


### Introduction

In this article I would like to introduce one of the ways to control trade events using the means of MQL5. I must mention that a few articles have already been dedicated to this topic. ["Processing of trade events in Expert Advisor using the OnTrade() function"](https://www.mql5.com/en/articles/40) is one of them. I am not going to repeat other authors and will use another handler - OnTradeTransaction().

I would like to draw the readers' attention to the following point. The current version of the MQL5 language formally counts 14 event handlers of the Client Terminal. In addition, a programmer has a possibility to create custom events with EventChartCustom() and process them with OnChartEvent(). However, the term 'Event-driven programming' (EDP) is not mentioned in Documentation at all. It is strange, given the fact, that any program in MQL5 is created based on the EDP principles. For instance, the user is offered to make a choice at the step 'Events handler for the Expert' in a template of any Expert Advisor.

It is obvious that the mechanism of the Event-driven Programming is used in MQL5 one way or another. The language may contain program blocks consisting of two parts: selecting and processing of an event. Moreover, if we are talking about events of the Client Terminal, a programmer has control only over the second part, i.e. the event handler. To be fair, there are exceptions for some events. Timer and custom event are among them. Control of these events is left entirely for the programmer.

### 1\. TradeTransaction Event

Before we go in depth into our topic, let us refer to the official information.

According to Documentation, the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event is a result of certain operations with a trading account. An operation itself consists of a number of stages determined by transactions. For instance, opening a position with a market order, one of the most popular operations with a trading account, is implemented in the following stages:

1. Make a trade request;
2. Verify the trade request;
3. Send the trade request to the server;
4. Receive a response about the trade order execution on the server.

Such sequence, though, only shows the logic of work of the terminal-server pair which is reflected in the strings of the EA code. From the point of view of the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) trade event, opening a position on the market happens the following way:

1. MQL5-program receives a notification from the server about the result of the completed request;

2. The request in the form of an order with a unique ticket gets included into the list of open orders;
3. The order gets deleted from the list of open orders after execution;

4. Then, the order goes to the account history;

5. The account history also contains data on the deal that the order execution results in.

So, opening a position requires five calls for the OnTradeTransaction() handler.

We shall discuss the program code in detail a little later and now we are going to have a close look at the header of the function. It has three input parameters.

```
void  OnTradeTransaction(
   const MqlTradeTransaction&    trans,        // structure of the trade transaction
   const MqlTradeRequest&        request,      // structure of the request
   const MqlTradeResult&         result        // structure of the response
   );
```

These parameters are described in detail in Documentation. I would like to note that a parameter of the trade transaction structure is a kind of cast of the information that the handler receives during the current call.

I must also say a few words about the type of the trade transaction as we are going to come across it a lot.

In MQL5, ENUM\_TRADE\_TRANSACTION\_TYPE is a special enumeration which is responsible for the the type of the trade transaction. To find out what type a trade transaction belongs to, we need to refer to the parameter-constant of the MqlTradeTransaction type.

```
struct MqlTradeTransaction
  {
   ulong                         deal;             // Ticket of the deal
   ulong                         order;            // Ticket of the order
   string                        symbol;           // Trade symbol
   ENUM_TRADE_TRANSACTION_TYPE   type;             // Type of the trade transaction
   ENUM_ORDER_TYPE               order_type;       // Type of the order
   ENUM_ORDER_STATE              order_state;      // Status of the order
   ENUM_DEAL_TYPE                deal_type;        // Type of the deal
   ENUM_ORDER_TYPE_TIME          time_type;        // Order expiration type
   datetime                      time_expiration;  // Order expiration time
   double                        price;            // Price
   double                        price_trigger;    // Price that triggers the Stop Limit order
   double                        price_sl;         // Level of Stop Loss
   double                        price_tp;         // Level of Take Profit
   double                        volume;           // Volume in lots
  };
```

The fourth field of the structure is the very enumeration we are looking for.

### 2\. Processing of Positions

Virtually all trade operations that concern processing of positions entail five calls of the OnTradeTransaction() handler. Among them are:

- opening of a position;

- position;

- position reversal;
- adding lots to the position;
- partial closing of a position.

Modifying a position is the only trade operation that calls the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event handler twice.

Since there is no information about what transaction types are responsible for certain trade operations, we are going to find it out through trial and error.

Before that, we will have to create a template of the Expert that will contain the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event handler. I named my version of the template TradeProcessor.mq5. I added a feature that enables displaying information on the values of the structure fields in the log. These values are the parameters of the event - handler. Analyzing those records will be time consuming, but in the end it will pay off by presenting the complete picture of events.

We need to launch the Expert in the debug mode on any of the charts in MetaTrader 5 terminal.

Open a position manually and take a look at the code. The first call of the handler will be like this (Fig. 1).

![Fig.1. The type field is equal to TRADE_TRANSACTION_REQUEST](https://c.mql5.com/2/11/1__29.png)

Fig.1. The type field is equal to TRADE\_TRANSACTION\_REQUEST

The following entries will appear in the log:

```
IO      0       17:37:53.233    TradeProcessor (EURUSD,H1)      ---===Transaction===---
NK      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
RR      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
DE      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ticket of the order: 0
JS      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
JN      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
FD      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Price: 0.0000
FN      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
HF      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
FQ      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
RR      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Trade symbol:
HD      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
GS      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
DN      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Type of the trade transaction TRADE_TRANSACTION_REQUEST
FK      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Volume in lots: 0.00
```

In this block only the record concerning the transaction type is of interest to us. As we can see, this type belongs to the (TRADE\_TRANSACTION\_REQUEST) request.

Information about the request details can be obtained in the block "Request".

```
QG      0       17:37:53.233    TradeProcessor (EURUSD,H1)      ---===Request===---
HL      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Type of the trade operation: TRADE_ACTION_DEAL
EE      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Comment to the order:
JP      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Deviation from the requested price: 0
GS      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Order expiration time: 1970.01.01 00:00
LF      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Magic number of the EA: 0
FM      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
EJ      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Price: 1.3137
QR      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Stop Loss level of the order: 0.0000
IJ      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Take Profit level of the order: 0.0000
KK      0       17:37:53.233    TradeProcessor (EURUSD,H1)      StopLimit level of the order: 0.0000
FS      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Trade symbol: EURUSD
RD      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
```

Data on the result of the request execution will get to the block "Response".

```
KG      0       17:37:53.233    TradeProcessor (EURUSD,H1)      ---===Response===---
JR      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Code of the operation result: 10009
GD      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ticket of the deal: 15258202
NR      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
EF      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Volume of the deal: 0.11
MN      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Price of the deal: 1.3137
HJ      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Bid: 1.3135
PM      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Ask: 1.3137
OG      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Comment to the operation:
RQ      0       17:37:53.233    TradeProcessor (EURUSD,H1)      Request ID: 1
```

Having analyzed other parameters of the handler such as structures of the request and response, you can get additional information about the request at the first call.

The second call concerns adding the order to the list of open orders (Fig. 2).

![Fig.2. The type field is equal to TRADE_TRANSACTION_ORDER_ADD](https://c.mql5.com/2/11/2__6.png)

Fig.2. The type field is equal to TRADE\_TRANSACTION\_ORDER\_ADD

The block "Transaction" is the only one we need in the log.

```
MJ      0       17:41:12.280    TradeProcessor (EURUSD,H1)      ---===Transaction===---
JN      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
FG      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
LM      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
LI      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
LP      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
QN      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Price: 1.3137
PD      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
NL      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
PG      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
DL      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Trade symbol: EURUSD
JK      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
QD      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
IQ      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_ORDER_ADD
PL      0       17:41:12.280    TradeProcessor (EURUSD,H1)      Volume in lots: 0.11
```

The order, as we can see, has already received its ticket and other parameters (symbol, price and volume) and is included in the list of open orders.

The third call of the event handler is connected with the deletion of the order from the list of the open ones (Fig. 3).

![Fig.3. The type field is equal to TRADE_TRANSACTION_ORDER_DELETE](https://c.mql5.com/2/11/3__4.png)

Fig.3. The type field is equal to TRADE\_TRANSACTION\_ORDER\_DELETE

The block "Transaction" is the only one we need in the log.

```
PF      0       17:52:36.722    TradeProcessor (EURUSD,H1)      ---===Transaction===---
OE      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
KL      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
EH      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
QM      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
QK      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
HS      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Price: 1.3137
MH      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
OP      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
EJ      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
IH      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Trade symbol: EURUSD
KP      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
LO      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
HG      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_ORDER_DELETE
CG      0       17:52:36.722    TradeProcessor (EURUSD,H1)      Volume in lots: 0.11
```

There is no new information in this block except the type of transaction.

The handler is called for the fourth time when a new historical order appears in the history (Fig. 4).

![Fig.4. The type field is equal to TRADE_TRANSACTION_HISTORY_ADD](https://c.mql5.com/2/11/4__5.png)

Fig.4. The type field is equal to TRADE\_TRANSACTION\_HISTORY\_ADD

We can get the relevant information from the block "Transaction".

```
QO      0       17:57:32.234    TradeProcessor (EURUSD,H1)      ---===Transaction==---
RJ      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
NS      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
DQ      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
EH      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_FILLED
RL      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
KJ      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Price: 1.3137
NO      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
PI      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
FS      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
JS      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Trade symbol: EURUSD
LG      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
KP      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
OL      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_HISTORY_ADD
JH      0       17:57:32.234    TradeProcessor (EURUSD,H1)      Volume in lots: 0.00
```

At this stage, we can see that the order has been executed.

Finally, the last (the fifth) call takes place when a deal is added to the history (Fig. 5).

![Fig.5. The type field is equal to TRADE_TRANSACTION_DEAL_ADD](https://c.mql5.com/2/11/5__4.png)

Fig.5. The type field is equal to TRADE\_TRANSACTION\_DEAL\_ADD

In the log, again, we are interested only in the block "Transaction".

```
OE      0       17:59:40.718    TradeProcessor (EURUSD,H1)      ---===Transaction===---
MS      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Ticket of the deal: 15258202
RJ      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
HN      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535869
LK      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
LE      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
MM      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Price: 1.3137
PF      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
NN      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
PI      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
DJ      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Trade symbol: EURUSD
JM      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
QI      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
CK      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_DEAL_ADD
RQ      0       17:59:40.718    TradeProcessor (EURUSD,H1)      Volume in lots: 0.11
```

The important string in this block is the ticket of the deal.

I am going to present a scheme of the transaction. For the positions there will be only two of them. The first one looks like the one on Fig. 6.

![Fig.6. The first scheme of the transaction process](https://c.mql5.com/2/11/11__1.png)

Fig.6. The first scheme of the transaction process

All trade operations connected with processing of positions take place in accordance with this scheme. The only exception here is the operation of modifying a position. The last operation includes processing of the following two transactions (Fig. 7).

![Fig.7. The second scheme of the transaction process](https://c.mql5.com/2/11/12__1.png)

Fig.7. The second scheme of the transaction process

So, modification of a position cannot be traced in the history of deals and orders.

That is pretty much all about positions.

### 3\. Processing of Pending Orders

With regard to pending orders, it should be noted that operations with them take fewer transactions. At the same time, there are more combinations of transaction types when working with orders.

To modify an order, the handler is called twice, similar to modifying a position. Placing and deleting an order takes three calls. The [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event occurs four times at deleting the order or its execution.

Now we are going to place a pending order. We need to launch the Expert in the debug mode on any of the charts in MetaTrader 5 terminal again.

The first call of the handler will be connected with the request (Fig. 8).

![Fig.8. The type field is equal to TRADE_TRANSACTION_REQUEST](https://c.mql5.com/2/11/1__30.png)

Fig.8. The type field is equal to TRADE\_TRANSACTION\_REQUEST

The log will contain the following entries:

```
IO      0       18:13:33.195    TradeProcessor (EURUSD,H1)      ---===Transaction===---
NK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
RR      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
DE      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ticket of the order: 0
JS      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
JN      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY
FD      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Price: 0.0000
FN      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
HF      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
FQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
RR      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Trade symbol:
HD      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
GS      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
DN      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_REQUEST
FK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Volume in lots: 0.00
NS      0       18:13:33.195    TradeProcessor (EURUSD,H1)

QG      0       18:13:33.195    TradeProcessor (EURUSD,H1)      ---===Request==---
IQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Type of the trade operation: TRADE_ACTION_PENDING
OE      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Order comment:
PQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Deviation from the requested price: 0
QS      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Order expiration time: 1970.01.01 00:00
FI      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Magic number of the EA: 0
CM      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535983
PK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Price: 1.6500
KR      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Stop Loss level of the order: 0.0000
OI      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Take Profit level of the order: 0.0000
QK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      StopLimit level of the order: 0.0000
QQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Trade symbol: GBPUSD
RD      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY_LIMIT
LS      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Order execution type: ORDER_FILLING_RETURN
MN      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
IK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Volume in lots: 0.14
NS      0       18:13:33.195    TradeProcessor (EURUSD,H1)
CD      0       18:13:33.195    TradeProcessor (EURUSD,H1)      ---===Response===---
RQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Code of the operation result: 10009
JI      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
GM      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535983
LF      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Volume of the deal: 0.14
JN      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Price of the deal: 0.0000
MK      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Bid: 0.0000
CM      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Ask: 0.0000
IG      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Comment to the operation:
DQ      0       18:13:33.195    TradeProcessor (EURUSD,H1)      Request ID: 1
```

The second call of the handler will add the order to the list of the open ones (Fig. 9).

![Fig.9. The type field is equal to TRADE_TRANSACTION_ORDER_ADDED](https://c.mql5.com/2/11/2__7.png)

Fig.9. The type field is equal to TRADE\_TRANSACTION\_ORDER\_ADDED

In the log we need to see only the "Transaction" block.

```
HJ      0       18:17:02.886    TradeProcessor (EURUSD,H1)      ---===Transaction===---
GQ      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
CH      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
RL      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535983
II      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_STARTED
OG      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY_LIMIT
GL      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Price: 1.6500
IE      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
CO      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
IF      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
PL      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Trade symbol: GBPUSD
OL      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
HJ      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
LF      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_ORDER_ADD
FR      0       18:17:02.886    TradeProcessor (EURUSD,H1)      Volume in lots: 0.14
```

The third call of the handler will renew the data according to the placed order (Fig. 10).

In particular, the order status will receive the value of ORDER\_STATE\_PLACED.

![Fig.10. The type field is equal to TRADE_TRANSACTION_ORDER_UPDATE](https://c.mql5.com/2/11/8__7.png)

Fig.10. The type field is equal to TRADE\_TRANSACTION\_ORDER\_UPDATE

In the log, we need to see the records for the "Transaction" block.

```
HS      0       18:21:27.004    TradeProcessor (EURUSD,H1)      ---===Transaction==---
GF      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Ticket of the deal: 0
CO      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Type of the deal: DEAL_TYPE_BUY
RE      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Ticket of the order: 22535983
KM      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Status of the order: ORDER_STATE_PLACED
QH      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Type of the order: ORDER_TYPE_BUY_LIMIT
EG      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Price: 1.6500
GL      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Level of Stop Loss: 0.0000
ED      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Level of Take Profit: 0.0000
GO      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Price that triggers the Stop Limit order: 0.0000
RE      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Trade symbol: GBPUSD
QS      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Pending order expiration time: 1970.01.01 00:00
JS      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Order expiration type: ORDER_TIME_GTC
RD      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Type of the trade transaction: TRADE_TRANSACTION_ORDER_UPDATE
JK      0       18:21:27.004    TradeProcessor (EURUSD,H1)      Volume in lots: 0.14
```

The most important string here is the status of the order.

Unlike processing of positions, processing of pending orders cannot be implemented by a scheme. Every trade operation connected with a pending order will be unique from the point of view of trading transactions types.

Placing a pending order will take three transactions (Fig. 11).

![Fig.11. Transactions, processing placement of a pending order](https://c.mql5.com/2/11/13__1.png)

Fig.11. Transactions, processing placement of a pending order

Modifying a pending order will generate two transactions (Fig. 12).

![Fig.12 Transactions, processing a modification of a pending order](https://c.mql5.com/2/11/14__1.png)

Fig.12. Transactions, processing modifying of a pending order

If the pending order is to be deleted, then the OnTradeTransaction() handler will be called four times (Fig. 13).

![Fig.13. Transactions, processing deletion of a pending order](https://c.mql5.com/2/11/15__3.png)

Fig.13. Transactions, processing deletion of a pending order

Deletion of a pending order is defined by the following scheme (Fig. 14).

![Fig.14. Transactions, processing a deletion of a pending order](https://c.mql5.com/2/11/16__1.png)

Fig.14. Transactions, processing a deletion of a pending order

Triggering of a pending order, the last trade operation, will induce four different transactions (Fig. 15).

![Fig.15. Transactions, processing activation of a pending order](https://c.mql5.com/2/11/17.png)

Fig.15. Transactions, processing activation of a pending order

I am not going to bring the log entries for every combination of transactions. If the reader feels so inclined, they can study them by executing the code.

### 4\. Universal Handler

Let us take a look at the program that can work with the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event through the eyes of the end user. The end user is very likely to need a program that can flawlessly work both with the orders and positions. A programmer must write the code for OnTradeTransaction() the way that will let it identify all transactions and their combinations regardless of what was processed - a position or an order. Ideally, the program should be able to indicate what operation was executed upon completion of processing the series of transactions.

In the example below, a sequential processing of transaction is used. The developer of MQL5 states the following though:

...A trade request manually sent from the terminal or via [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend)/ [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) functions can generate several consecutive transactions on the trade server. Priority of these transactions arrival at the terminal is not guaranteed, thus trading algorithm should not be based on the presumption that one group of transactions will arrive after another one. In addition to that, transactions can be lost at delivery from the server to the terminal...

So, if the requirement is to write a program working close to the ideal, you can improve the suggested example and make processing of transactions independent of the arrival order of transactions.

In general, positions and orders can have common types of transactions. There are 11 types of transactions. Only four out of that number have something to do with trading from the terminal:

- TRADE\_TRANSACTION\_DEAL\_UPDATE;
- TRADE\_TRANSACTION\_DEAL\_DELETE;
- TRADE\_TRANSACTION\_HISTORY\_UPDATE;
- TRADE\_TRANSACTION\_HISTORY\_DELETE.

We are not going to discuss them in this article. These types, according to the developer, were designed for extending functionality on the trading server side. I must admit that I have not previously dealt with such types before.

That leaves us with seven full-featured types that get processed in the OnTradeTransaction() most often.

In the body of the handler, the segment, defining the current transaction type will have an important role.

```
//--- ========== Types of transaction [START]
   switch(trans_type)
     {
      //--- 1) if it is a request
      case TRADE_TRANSACTION_REQUEST:
        {

         //---
         break;
        }
      //--- 2) if it is an addition of a new open order
      case TRADE_TRANSACTION_ORDER_ADD:
        {

         //---
         break;
        }
      //--- 3) if it is a deletion of an order from the list of open ones
      case TRADE_TRANSACTION_ORDER_DELETE:
        {

         //---
         break;
        }
      //--- 4) if it is an addition of a new order to the history
      case TRADE_TRANSACTION_HISTORY_ADD:
        {

         //---
         break;
        }
      //--- 5) if it is an addition of a deal to history
      case TRADE_TRANSACTION_DEAL_ADD:
        {

         //---
         break;
        }
      //--- 6) if it is a modification of a position
      case TRADE_TRANSACTION_POSITION:
        {

         //---
         break;
        }
      //--- 7) if it is a modification of an open order
      case TRADE_TRANSACTION_ORDER_UPDATE:
        {

         //---
         break;
        }
     }
//--- ========== Types of transactions [END]
```

We will try to define what trade operation we are processing by the current transaction type. To find out what we are working with - a position or an order, we shall delegate memorizing the type of the trade operation to the case-module of processing the request.

The module itself will look as follows:

```
//--- 1) if it is a request
      case TRADE_TRANSACTION_REQUEST:
        {
         //---
         last_action=request.action;
         string action_str;

         //--- what is the request for?
         switch(last_action)
           {
            //--- а) on market
            case TRADE_ACTION_DEAL:
              {
               action_str="place a market order";
               trade_obj=TRADE_OBJ_POSITION;
               break;
              }
            //--- б) place a pending order
            case TRADE_ACTION_PENDING:
              {
               action_str="place a pending order";
               trade_obj=TRADE_OBJ_ORDER;
               break;
              }
            //--- в) modify position
            case TRADE_ACTION_SLTP:
              {
               trade_obj=TRADE_OBJ_POSITION;
               //---
               StringConcatenate(action_str,request.symbol,": modify the levels of Stop Loss",
                                 " and Take Profit");

               //---
               break;
              }
            //--- г) modify order
            case TRADE_ACTION_MODIFY:
              {
               action_str="modify parameters of the pending order";
               trade_obj=TRADE_OBJ_ORDER;
               break;
              }
            //--- д) delete order
            case TRADE_ACTION_REMOVE:
              {
               action_str="delete pending order";
               trade_obj=TRADE_OBJ_ORDER;
               break;
              }
           }
         //---
         if(InpIsLogging)
            Print("Request received: "+action_str);

         //---
         break;
        }
```

Changing a few variables is not difficult in this case.

```
static ENUM_TRADE_REQUEST_ACTIONS last_action; // market operation at the first pass
```

The last\_action variable will memorize why the event handler was launched at all.

```
static ENUM_TRADE_OBJ trade_obj;               // specifies the trade object at the first pass
```

The variable trade\_obj will keep in memory what was processed - a position or an order. For that we shall create the ENUM\_TRADE\_OBJ enumeration.

After that, we are going to proceed to the module which will process transactions of the TRADE\_TRANSACTION\_ORDER\_ADD type:

```
//--- 2) if it is an addition of a new open order
      case TRADE_TRANSACTION_ORDER_ADD:
        {
         if(InpIsLogging)
           {
            if(trade_obj==TRADE_OBJ_POSITION)
               Print("Open a new market order: "+
                     EnumToString(trans.order_type));
            //---
            else if(trade_obj==TRADE_OBJ_ORDER)
               Print("Place a new pending order: "+
                     EnumToString(trans.order_type));
           }
         //---
         break;
        }
```

This module is rather simple. Since position was processed at the first step, a log entry "Open a new market order" will appear at the current one, otherwise "Place a new pending order". There are no more actions other than informative in this block.

Now it is the turn of the third module which processes the TRADE\_TRANSACTION\_ORDER\_DELETE type:

```
//--- 3) if it is a deletion of an order from the list of open ones
      case TRADE_TRANSACTION_ORDER_DELETE:
        {
         if(InpIsLogging)
            PrintFormat("Order deleted from the list of open ones: #%d, "+
                        EnumToString(trans.order_type),trans.order);
         //---
         break;
        }
```

This module also has only an informative role.

The fourth case-module processes the TRADE\_TRANSACTION\_HISTORY\_ADD type:

```
//--- 4) if it is an addition of a new order to the history
      case TRADE_TRANSACTION_HISTORY_ADD:
        {
         if(InpIsLogging)
            PrintFormat("Order added to the history: #%d, "+
                        EnumToString(trans.order_type),trans.order);

         //--- if a pending order is being processed
         if(trade_obj==TRADE_OBJ_ORDER)
           {
            //--- if it is the third pass
            if(gTransCnt==2)
              {
               //--- if the order was canceled, check the deals
               datetime now=TimeCurrent();

               //--- request the history of orders and deals
               HistorySelect(now-PeriodSeconds(PERIOD_H1),now);

               //--- attempt to find a deal for the order
               CDealInfo myDealInfo;
               int all_deals=HistoryDealsTotal();
               //---
               bool is_found=false;
               for(int deal_idx=all_deals;deal_idx>=0;deal_idx--)
                  if(myDealInfo.SelectByIndex(deal_idx))
                     if(myDealInfo.Order()==trans.order)
                        is_found=true;

               //--- if the deal was not found
               if(!is_found)
                 {
                  is_to_reset_cnt=true;
                  //---
                  PrintFormat("Order canceled: #%d",trans.order);
                 }
              }
            //--- if it is the fourth pass
            if(gTransCnt==3)
              {
               is_to_reset_cnt=true;
               PrintFormat("Order deleted: #%d",trans.order);
              }
           }
         //---
         break;
        }
```

In addition to the record that the order was added to the history, this module carries out a check if we have been working with a pending order initially. In case we have done, we need to find out what number the current pass of the handler has. The thing is that this type of transaction can appear at the third pass when working with the order, if a pending order was canceled. At the fourth pass this type can appear when a pending order was deleted.

At the strings of the module checking the third pass, we need to refer to the history of deals once again. If a deal for the current order is not found, then we will consider such an order canceled.

The fifth case-module processes the TRADE\_TRANSACTION\_DEAL\_ADD type. It is the largest block of the program by the size of the strings.

The deal is checked in this block. It is important to choose a deal by the ticket to get access to its properties. A deal type can provide information if position was open, closed etc. Information about triggering of a pending order can be retrieved there too. It is the only case when a pending order could generate a deal in the work context of the event handler [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction).

```
//--- 5) if it is an addition of a deal to history
      case TRADE_TRANSACTION_DEAL_ADD:
        {
         is_to_reset_cnt=true;
         //---
         ulong deal_ticket=trans.deal;
         ENUM_DEAL_TYPE deal_type=trans.deal_type;
         //---
         if(InpIsLogging)
            PrintFormat("Deal added to history: #%d, "+EnumToString(deal_type),deal_ticket);

         if(deal_ticket>0)
           {
            datetime now=TimeCurrent();

            //--- request the history of orders and deals
            HistorySelect(now-PeriodSeconds(PERIOD_H1),now);

            //--- select a deal by the ticket
            if(HistoryDealSelect(deal_ticket))
              {
               //--- check the deal
               CDealInfo myDealInfo;
               myDealInfo.Ticket(deal_ticket);
               long order=myDealInfo.Order();

               //--- parameters of the deal
               ENUM_DEAL_ENTRY  deal_entry=myDealInfo.Entry();
               double deal_vol=0.;
               //---
               if(myDealInfo.InfoDouble(DEAL_VOLUME,deal_vol))
                  if(myDealInfo.InfoString(DEAL_SYMBOL,deal_symbol))
                    {
                     //--- position
                     CPositionInfo myPos;
                     double pos_vol=WRONG_VALUE;
                     //---
                     if(myPos.Select(deal_symbol))
                        pos_vol=myPos.Volume();

                     //--- if the market was entered
                     if(deal_entry==DEAL_ENTRY_IN)
                       {
                        //--- 1) opening of a position
                        if(deal_vol==pos_vol)
                           PrintFormat("\n%s: new position opened",deal_symbol);

                        //--- 2) addition to the open position
                        else if(deal_vol<pos_vol)
                           PrintFormat("\n%s: addition to the current position",deal_symbol);
                       }

                     //--- if the market was exited
                     else if(deal_entry==DEAL_ENTRY_OUT)
                       {
                        if(deal_vol>0.0)
                          {
                           //--- 1) closure of a position
                           if(pos_vol==WRONG_VALUE)
                              PrintFormat("\n%s: position closed",deal_symbol);

                           //--- 2) partial closure of the open position
                           else if(pos_vol>0.0)
                              PrintFormat("\n%s: partial closing of the current position",deal_symbol);
                          }
                       }

                     //--- if position was reversed
                     else if(deal_entry==DEAL_ENTRY_INOUT)
                       {
                        if(deal_vol>0.0)
                           if(pos_vol>0.0)
                              PrintFormat("\n%s: position reversal",deal_symbol);
                       }
                    }

               //--- order activation
               if(trade_obj==TRADE_OBJ_ORDER)
                  PrintFormat("Pending order activation: %d",order);
              }
           }

         //---
         break;
        }
```

Transaction type TRADE\_TRANSACTION\_POSITION is unique and is processed only when a position gets modified:

```
//--- 6) if it is a modification of a position
      case TRADE_TRANSACTION_POSITION:
        {
         is_to_reset_cnt=true;
         //---
         PrintFormat("Modification of a position: %s",deal_symbol);
         //---
         if(InpIsLogging)
           {
            PrintFormat("New price of stop loss: %0."+
                        IntegerToString(_Digits)+"f",trans.price_sl);
            PrintFormat("New price of take profit: %0."+
                        IntegerToString(_Digits)+"f",trans.price_tp);
           }

         //---
         break;
        }
```

The last case-module gets enabled at processing the TRADE\_TRANSACTION\_ORDER\_UPDATE type.

This type appears only for work with a pending order. It launches at triggering of any trade operation, concerning pending orders, however the stage may vary.

```
//--- 7) if it is a modification of an open order
      case TRADE_TRANSACTION_ORDER_UPDATE:
        {

         //--- if it was the first pass
         if(gTransCnt==0)
           {
            trade_obj=TRADE_OBJ_ORDER;
            PrintFormat("Canceling the order: #%d",trans.order);
           }
         //--- if it was the second pass
         if(gTransCnt==1)
           {
            //--- if it is an order modification
            if(last_action==TRADE_ACTION_MODIFY)
              {
               PrintFormat("Pending order modified: #%d",trans.order);
               //--- clear counter
               is_to_reset_cnt=true;
              }
            //--- if it is deletion of the order
            if(last_action==TRADE_ACTION_REMOVE)
              {
               PrintFormat("Delete pending order: #%d",trans.order);

              }
           }
         //--- if it was the third pass
         if(gTransCnt==2)
           {
            PrintFormat("A new pending order was placed: #%d, "+
                        EnumToString(trans.order_type),trans.order);
            //--- clear counter
            is_to_reset_cnt=true;
           }

         //---
         break;
        }
```

Summing up, if this type appeared at the first triggering of OnTradeTransaction(), then the order was either canceled or executed.

If the type appeared at the second launch of the event handler, then the order was either deleted or modified. To find out what exactly the order resulted in, refer to the static variable last\_action which contains the data on the last trading operation.

The third launch of the event handler is the last case when this type can appear. The third launch completes the procedure of placing a pending order.

A boolean variable is\_to\_reset\_cnt is also used in the code. It has a role of a flag for clearing a counter of the passes of the OnTradeTransaction() handle.

That is pretty much all about processing of the [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event. I would also add a pause in the beginning of the call for handler. It will minimize a chance of a deal or an order being delayed in getting to the history.

### Conclusion

In this article I tried to illustrate how different trade operations can be worked with and how one can retrieve information about things happening in the terminal.

The greatest advantage of this approach is that the program can receive information about phased implementation of a trade operation. In my opinion, such an approach can be used for copying deals from one terminal to another.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1111](https://www.mql5.com/ru/articles/1111)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1111.zip "Download all attachments in the single ZIP archive")

[tradeprocessor.mq5](https://www.mql5.com/en/articles/download/1111/tradeprocessor.mq5 "Download tradeprocessor.mq5")(31.35 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Cookbook — Macroeconomic events database](https://www.mql5.com/en/articles/11977)
- [MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)
- [MQL5 Cookbook – Economic Calendar](https://www.mql5.com/en/articles/9874)
- [MQL5 Cookbook: Trading strategy stress testing using custom symbols](https://www.mql5.com/en/articles/7166)
- [MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)
- [MQL5 Cookbook - Pivot trading signals](https://www.mql5.com/en/articles/2853)
- [MQL5 Cookbook - Trading signals of moving channels](https://www.mql5.com/en/articles/1863)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/36799)**
(17)


![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
27 Oct 2023 at 10:58

Denis, thank you for the article! I read it with interest, but I have not yet fully realised what I have read. I also downloaded your TradeProcessor Expert Advisor and ran it on my terminal, where I currently have an [open position](https://www.metatrader5.com/en/terminal/help/trading/performing_deals#position_manage "MetaTrader 5 Help: Opening and Closing Positions in the MetaTrader 5 Trading Terminal"). I looked at the printouts and immediately some questions appeared, to which I want to find answers on my own. Well, if I can't do it, I will have to bother you. ))

Regards, Vladimir.

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
27 Oct 2023 at 11:13

_**MrBrooklin open position. I looked at the printouts and immediately some questions appeared, to which I want to find answers on my own. Well, if I can't do it, I will have to bother you. ))**_

_Regards, Vladimir._

Thank you for your opinion. You are welcome, MrBrooklin! ))

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
27 Oct 2023 at 16:31

**Denis Kirichenko [#](https://www.mql5.com/ru/forum/36025#comment_50178922):**

Thank you for your opinion. You are welcome, MrBrooklin! ))

Denis, sorry, but without your help my mind "explodes". Here is the result of one of the passes of your EA (by the way, a very cool EA!!!):

```
2023.10.27 17:11:02.514 TradeProcessor (EURUSDrfd,D1)   Проход : #100
2023.10.27 17:11:02.514 TradeProcessor (EURUSDrfd,D1)   Поступил запрос: изменить параметры отложенного ордера
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   ---===Транзакция===---
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тикет сделки: 0
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тип сделки: DEAL_TYPE_BUY
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тикет ордера: 1030195768
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Состояние ордера: ORDER_STATE_PLACED
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тип ордера: ORDER_TYPE_SELL_STOP
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Цена: 1.05853
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Уровень Stop Loss: 1.(скрыл значения стоп-лосса)
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Уровень Take Profit: 1.05803
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Цена срабатывания стоп-лимитного ордера: 0.00000
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Торговый инструмент: EURUSDrfd
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Срок истечения отложенного ордера: 2023.10.27 00:00
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тип ордера по времени действия: ORDER_TIME_DAY
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тип торговой транзакции: TRADE_TRANSACTION_ORDER_UPDATE
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Тикет позиции: 0
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Объём в лотах: 0.04
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)
2023.10.27 17:11:02.733 TradeProcessor (EURUSDrfd,D1)   Проход : #101
```

What I don't understand:

1. my EA placed a pending order SELL\_STOP;
2. Your EA writes - trade type DEAL\_TYPE\_BY (highlighted in yellow). Just in case, I look in the MQL5 Reference Guide. It says that DEAL\_TYPE\_BY is:

ENUM\_DEAL\_TYPE

| Identifier | Description |
| --- | --- |
| DEAL\_TYPE\_BUY | Buy |

Question - how when modifying a pending order SELL\_STOP the transaction type is determined to **BUY**????? ))

Regards, Vladimir.

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
27 Oct 2023 at 16:53

And there is no transaction. Placing a pending order and its processing does not entail any transaction. Trade ticket = 0, type = 0, where 0 is equivalent to DEAL\_TYPE\_BUY  for the enumeration ENUM\_DEAL\_TYPE. That is, in the **MqlTradeTransaction trans**structure , some fields are populated and some are not. The unfilled fields are usually nulled.

More details: in the [Documentation about pending orders](https://www.mql5.com/en/docs/constants/structures/mqltradetransaction).

The deal field will be populated for a transaction of this type:

_TRADE\_TRANSACTION\_DEAL\_\*_

_The following fields are populated in the MqlTradeTransaction structure for trade transactions related to transaction processing (TRADE\_TRANSACTION\_DEAL\_ADD, TRADE\_TRANSACTION\_DEAL\_UPDATE and TRADE\_TRANSACTION\_DEAL\_DELETE):_

- _deal - the ticket of the trade;_
- _order - ticket of the order, on the basis of which the deal was made;_
- _symbol - name of the financial instrument in the deal;_
- _type - type of trade transaction;_
- _deal\_type - type of the deal;_
- _price - price at which the deal was made;_
- _price\_sl - Stop Loss price (it is filled in if it is specified in the order on the basis of which the deal was made);_
- _price\_tp - Take Profit price (it is filled in, if it is specified in the order, on the basis of which the deal was made);_
- _volume - volume of the deal in lots._
- _position - a ticket of a position opened, changed or closed as a result of a deal execution._
- _position\_by - a ticket of a counter position. It is filled in only for trades for closing a position by counter (out by)._

Only 3 types of transactions belong to "deal" types: TRADE\_TRANSACTION\_DEAL\_ADD, TRADE\_TRANSACTION\_DEAL\_UPDATE, TRADE\_TRANSACTION\_DEAL\_DELETE.

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
27 Oct 2023 at 16:56

**Denis Kirichenko [#](https://www.mql5.com/ru/forum/36025#comment_50184491):**

And there is no transaction. Placing a pending order does not entail any transaction. Trade ticket = 0, type = 0, where 0 is equivalent to DEAL\_TYPE\_BUY  for the enumeration ENUM\_DEAL\_TYPE. That is, in the **MqlTradeTransaction trans**structure , some fields are populated and some are not. The unfilled fields are usually zeroed.

More details: in the [Documentation about pending orders](https://www.mql5.com/en/docs/constants/structures/mqltradetransaction).

Ahhhh, that's it!!! Man, it broke my head! ))

Thank you!!!

Regards, Vladimir.

![MQL5 Cookbook: Handling Typical Chart Events](https://c.mql5.com/2/11/OnChartEvent_MetaTrader5.png)[MQL5 Cookbook: Handling Typical Chart Events](https://www.mql5.com/en/articles/689)

This article considers typical chart events and includes examples of their processing. We will focus on mouse events, keystrokes, creation/modification/removal of a graphical object, mouse click on a chart and on a graphical object, moving a graphical object with a mouse, finish editing of text in a text field, as well as on chart modification events. A sample of an MQL5 program is provided for each type of event considered.

![How to Prepare a Trading Account for Migration to Virtual Hosting](https://c.mql5.com/2/11/VHC_start.png)[How to Prepare a Trading Account for Migration to Virtual Hosting](https://www.mql5.com/en/articles/994)

MetaTrader client terminal is perfect for automating trading strategies. It has all tools necessary for trading robot developers ‒ powerful C++ based MQL4/MQL5 programming language, convenient MetaEditor development environment and multi-threaded strategy tester that supports distributed computing in MQL5 Cloud Network. In this article, you will find out how to move your client terminal to the virtual environment with all custom elements.

![MQL5 Cookbook: Handling Custom Chart Events](https://c.mql5.com/2/11/avatar.png)[MQL5 Cookbook: Handling Custom Chart Events](https://www.mql5.com/en/articles/1163)

This article considers aspects of design and development of custom chart events system in the MQL5 environment. An example of an approach to the events classification can also be found here, as well as a program code for a class of events and a class of custom events handler.

![Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://c.mql5.com/2/11/fundamental_analysis_statistica_MQL5_MetaTrader5.png)[Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://www.mql5.com/en/articles/1087)

This article considers the application of multiple regression analysis to macroeconomic statistics. It also gives an insight into the evaluation of the statistics impact on the currency exchange rate fluctuation based on the example of the currency pair EURUSD. Such evaluation allows automating the fundamental analysis which becomes available to even novice traders.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=hwwqrefcigkkqciczfpbizaldlricswf&ssn=1769181585706513263&ssn_dr=0&ssn_sr=0&fv_date=1769181585&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1111&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Cookbook%3A%20Processing%20of%20the%20TradeTransaction%20Event%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918158549253620&fz_uniq=5069347842528576401&sv=2552)

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