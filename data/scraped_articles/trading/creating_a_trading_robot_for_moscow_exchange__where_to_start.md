---
title: Creating a trading robot for Moscow Exchange. Where to start?
url: https://www.mql5.com/en/articles/2513
categories: Trading, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:29:12.598772
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/2513&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062502446342775638)

MetaTrader 5 / Examples


Many traders on [Moscow Exchange](https://www.mql5.com/go?link=https://www.moex.com/ "http://moex.com/") would like to automate their trading algorithms, but they do not know where to start. The MQL5 language offers a huge range of trading functions, and it additionally provides ready classes that help users to make their first steps in algo trading. In this article we will discuss native tools provided by the MQL5 trading strategy language, from which MOEX traders can benefit.

### Two Types of Trade Orders on Moscow Exchange (MOEX)

Moscow Exchange supports two types of trade orders: market and limit.

- **Market** orders are immediately delivered to the Exchange and are executed at the best available price. These orders can be sent using MQL5 [market orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) of the ORDER\_TYPE\_BUY and ORDER\_TYPE\_SELL types.
- **Limit** orders are stored on the Exchange server and can be executed as soon as a suitable opposite order appears on the Exchange. In the MQL5 language, these orders are represented by two types: ORDER\_TYPE\_BUY\_LIMIT and ORDER\_TYPE\_SELL\_LIMIT.

A market order can guarantee trade execution (provided there is enough liquidity), but it cannot guarantee the price. This means that the price of the executed trade may significantly differ from the current offer. A limit order can guarantee the price at which a trade will be executed, but it does not guarantee the execution of the trade. As a result, your order can fail to be executed.

All other types of orders available for brokers on Moscow Exchange are part of the software system, through which traders interact with the Exchange. In other words, all other orders are algorithmic. These orders are stored and processed outside of Moscow Exchange, and are sent to the Exchange in the form of market or limit orders as a result of internal processing.

The MetaTrader 5 platform provides the following types of trading orders, which traders can use for trading on Moscow Exchange:

|     |     |     |
| --- | --- | --- |
| **ID** | **Description** | **Storage and execution** |
| ORDER\_TYPE\_BUY | A market Buy order | The order is sent to the Exchange in the form of a market Buy order at the best selling price available |
| ORDER\_TYPE\_SELL | A market Sell order | The order is sent to the Exchange in the form of a market Sell order at the best buying price available |
| ORDER\_TYPE\_BUY\_LIMIT | A pending Buy Limit order | The order is sent to the Exchange in the form of a limit Buy order. The order will be executed as soon as a Sell offer with the specified or better price appears in the market. |
| ORDER\_TYPE\_SELL\_LIMIT | A pending Sell Limit order | The order is sent to the Exchange in the form of a limit Sell order. The order will be executed as soon as a Buy offer with the specified or better price appears in the market. |
| ORDER\_TYPE\_BUY\_STOP | A pending Buy Stop order | The order is stored on the MetaTrader 5 server. Once the order triggers, it is sent to the Exchange:<br>- for the FX and Securities Markets, it is sent as a market Buy order<br>- for FORTS, the order is sent as a limit Buy order at the worst channel threshold price |
| ORDER\_TYPE\_SELL\_STOP | A pending Sell Stop order | The order is stored on the MetaTrader 5 server. Once the order triggers, it is sent to the Exchange:<br>- for the FX and Securities Markets, it is sent as a market Sell order<br>- for FORTS, the order is sent as a limit Sell order at the worst channel threshold price |
| ORDER\_TYPE\_BUY\_STOP\_LIMIT | Pending BUY STOP LIMIT | The order is stored on the MetaTrader 5 server. Once the order triggers, it is sent to the Exchange in the form of a limit Buy order |
| ORDER\_TYPE\_SELL\_STOP\_LIMIT | Pending SELL STOP LIMIT | The order is stored on the MetaTrader 5 server. Once the order triggers, it is sent to the Exchange in the form of a limit Sell order |

The MetaTrader 5 platform allows setting TakeProfit and StopLoss levels for open positions. The levels are stored on the MetaTrader 5 trade server and trigger automatically, even if the trading account is not connected:

- The TakeProfit level sets a price to close the position in the profit direction. Once a TakeProfit triggers, a limit order at the TakeProfit price is sent to the Exchange;

- The StopLoss level allows implementing a protective stop order if the position is making loss. Once a StopLoss triggers, a market order at the StopLoss price is sent to the Exchange when trading in the FX and Securities markets, and a limit order at the worst channel threshold price is sent when trading FORTS;

In addition, the MetaTrader 5 platform allows traders to set and modify StopLoss/TakeProfit levels for pending orders, as well as to modify trigger levels of all pending orders.

### Trading Operations in MetaTrader 5

MetaTrader 5 provides basic types of trading operations, which you may want to use in your trading robot:

- buying/selling at the current price
- placing a pending order to buy/sell under a certain condition
- modifying/deleting a pending order
- closing/increasing/reducing/reversing a position

All trading operations in MQL5 are implemented using the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function that returns control to the program as soon as a trade order is successfully sent to Moscow Exchange. The [order state](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state) then changes to ORDER\_STATE\_PLACED, which however does not mean that your order will be successfully executed (that order state will change to ORDER\_STATE\_FILLED or ORDER\_STATE\_PARTIAL). The result of your trade order execution depends on the current market, and the order may be rejected by the Exchange (state ORDER\_STATE\_REJECTED) for different reasons.

An asynchronous version of the function is also available — [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync), which performs much faster than OrderSend(), because it does not wait for the order to be sent to the trading system of the Exchange. A response to this function is sent immediately, as soon as the MetaTrader 5 terminal sends the order outside. This means that your trade order has passed the basic verification in the terminal, and is now sent for processing to the MetaTrader 5 trade server. The time required to add your order to the queue on the Exchange, as well as when the order will be executed or rejected depend on how busy the Exchange is and how fast your Internet connection is.

The wide variety of trade operations is described in the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure, which contains the description of a trade request. Therefore the main task here is to correctly fill in the MqlTradeRequest structure and process the request execution result.

In accordance with the rules of your trading system, you can buy or sell at the market price (BUY or SELL), or place a pending order to buy/to sell at a distance from the current price:

- BUY STOP, SELL STOP — buying or selling once the price breaks through the specified level (worse than the current price). Orders of this type are stored on the MetaTrader 5 trade server and are sent to Moscow Exchange as soon as the order triggers. They are sent in the form of a market (for FX and Securities) or a limit (for FORTS) order.

- BUY LIMIT, SELL LIMIT — buying or selling when the price reaches the specified level (best than the current price). Such orders are sent to Moscow Exchange immediately in the form of a limit order. On Moscow Exchange, it is possible to specify a level inside spread or on the other side of spread. This helps to limit slippage during trade execution.

- BUY STOP LIMIT, SELL STOP LIMIT — placing BUY LIMIT or SELL LIMIT once the specified price is reached. Orders of this type are stored on the MetaTrader 5 trade server. Once the specified conditions trigger, a limit order is sent to Moscow Exchange. The opening level of such a limit order can be either higher or lower than the order trigger level.


The below figure shows the principle of using BUY STOP, SELL STOP and BUY LIMIT, SELL LIMIT, as well as it suggests how to place these order from [Market Depth](https://www.metatrader5.com/en/terminal/help/trading/depth_of_market "https://www.metatrader5.com/en/terminal/help/trading/depth_of_market").

![](https://c.mql5.com/2/23/Orders-en__2.png)

In addition, you may need to modify or even delete a pending order. This can also be done by using the OrderSend()/OrderSendAsync() functions. Managing an open position is also easy, because management is performed through trading operations.

In this article we'll show you how to program buying and selling in MQL5, and we will demonstrate how to work with a trading account and symbol properties. This can be done by using [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) of the Standard Library.

### Exchange Trading Using a Robot is Easy

The MQL5 language natively supports all trading capabilities of the MetaTrader 5 platform: it contains many [trading functions](https://www.mql5.com/en/docs/trading) for working with orders, positions and trade requests. Operations are the same no mater what securities you are trading: futures, securities, options, bonds, etc.

Using MQL5 opportunities, you can create a [trade order](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) and send it to a server using the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) or [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) function, you can also receive the [order execution result](https://www.mql5.com/en/docs/constants/structures/mqltraderesult), view trading history, check out security [contract specification](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants), handle a [trade event](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) and access other useful information.

An important note for developers of trading robots: any trading operation, including opening of a position, setting of StopLoss or TakeProfit, or closure of a position by an opposite one, always consists of a set of transactions performed on the MetaTrader 5 server and on Moscow Exchange. You can view the process by running the _TradeTransactionListener.mql5_ Expert Advisor on your account. The EA listens to [TradeTransaction](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) events and displays brief information about the events:

```
//+------------------------------------------------------------------+
//|                                     TradeTransactionListener.mq5 |
//|                        Copyright 2016, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   PrintFormat("LAST PING=%.f ms",
               TerminalInfoInteger(TERMINAL_PING_LAST)/1000.);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
//---
   static int counter=0;   // Counter of OnTradeTransaction() calls
   static uint lasttime=0; // Time of the last call of OnTradeTransaction()
//---
   uint time=GetTickCount();
//--- If the last transaction was performed more than 1 seconds ago,
   if(time-lasttime>1000)
     {
      counter=0; // then this is a new trade operation, an the counter can be reset
      if(IS_DEBUG_MODE)
         Print(" New trade operation");
     }
   lasttime=time;
   counter++;
   Print(counter,". ",__FUNCTION__);
//--- Result of trade request execution
   ulong            lastOrderID   =trans.order;
   ENUM_ORDER_TYPE  lastOrderType =trans.order_type;
   ENUM_ORDER_STATE lastOrderState=trans.order_state;
//--- The name of the symbol for which a transaction was performed
   string trans_symbol=trans.symbol;
//--- type of transaction
   ENUM_TRADE_TRANSACTION_TYPE  trans_type=trans.type;
   switch(trans.type)
     {
      case  TRADE_TRANSACTION_POSITION:   // Position modification
        {
         ulong pos_ID=trans.position;
         PrintFormat("MqlTradeTransaction: Position #%I64u %s modified: SL=%.5f TP=%.5f",
                     pos_ID,trans_symbol,trans.price_sl,trans.price_tp);
        }
      break;
      case TRADE_TRANSACTION_REQUEST:     // Sending a trade request
         PrintFormat("MqlTradeTransaction: TRADE_TRANSACTION_REQUEST");
         break;
      case TRADE_TRANSACTION_DEAL_ADD:    // Adding a trade
        {
         ulong          lastDealID   =trans.deal;
         ENUM_DEAL_TYPE lastDealType =trans.deal_type;
         double        lastDealVolume=trans.volume;
         //--- Trade ID in an internal system - a ticket assigned by Moscow Exchange
         string Exchange_ticket="";
         if(HistoryDealSelect(lastDealID))
            Exchange_ticket=HistoryDealGetString(lastDealID,DEAL_EXTERNAL_ID);
         if(Exchange_ticket!="")
            Exchange_ticket=StringFormat("(MOEX deal=%s)",Exchange_ticket);

         PrintFormat("MqlTradeTransaction: %s deal #%I64u %s %s %.2f lot   %s",EnumToString(trans_type),
                     lastDealID,EnumToString(lastDealType),trans_symbol,lastDealVolume,Exchange_ticket);
        }
      break;
      case TRADE_TRANSACTION_HISTORY_ADD: // Adding an order to the history
        {
         //--- Order ID in an internal system - a ticket assigned by Moscow Exchange
         string Exchange_ticket="";
         if(lastOrderState==ORDER_STATE_FILLED)
           {
            if(HistoryOrderSelect(lastOrderID))
               Exchange_ticket=HistoryOrderGetString(lastOrderID,ORDER_EXTERNAL_ID);
            if(Exchange_ticket!="")
               Exchange_ticket=StringFormat("(MOEX ticket=%s)",Exchange_ticket);
           }
         PrintFormat("MqlTradeTransaction: %s order #%I64u %s %s %s   %s",EnumToString(trans_type),
                     lastOrderID,EnumToString(lastOrderType),trans_symbol,EnumToString(lastOrderState),Exchange_ticket);
        }
      break;
      default: // other transactions
        {
         //--- Order ID in an internal system - a ticket assigned by Moscow Exchange
         string Exchange_ticket="";
         if(lastOrderState==ORDER_STATE_PLACED)
           {
            if(OrderSelect(lastOrderID))
               Exchange_ticket=OrderGetString(ORDER_EXTERNAL_ID);
            if(Exchange_ticket!="")
               Exchange_ticket=StringFormat("MOEX ticket=%s",Exchange_ticket);
           }
         PrintFormat("MqlTradeTransaction: %s order #%I64u %s %s   %s",EnumToString(trans_type),
                     lastOrderID,EnumToString(lastOrderType),EnumToString(lastOrderState),Exchange_ticket);
        }
      break;
     }
//--- order ticket
   ulong orderID_result=result.order;
   string retcode_result=GetRetcodeID(result.retcode);
   if(orderID_result!=0)
      PrintFormat("MqlTradeResult: order #%d retcode=%s ",orderID_result,retcode_result);
//---
  }
//+------------------------------------------------------------------+
//| Converts numeric response codes to string mnemonics              |
//+------------------------------------------------------------------+
string GetRetcodeID(int retcode)
  {
   switch(retcode)
     {
      case 10004: return("TRADE_RETCODE_REQUOTE");             break;
      case 10006: return("TRADE_RETCODE_REJECT");              break;
      case 10007: return("TRADE_RETCODE_CANCEL");              break;
      case 10008: return("TRADE_RETCODE_PLACED");              break;
      case 10009: return("TRADE_RETCODE_DONE");                break;
      case 10010: return("TRADE_RETCODE_DONE_PARTIAL");        break;
      case 10011: return("TRADE_RETCODE_ERROR");               break;
      case 10012: return("TRADE_RETCODE_TIMEOUT");             break;
      case 10013: return("TRADE_RETCODE_INVALID");             break;
      case 10014: return("TRADE_RETCODE_INVALID_VOLUME");      break;
      case 10015: return("TRADE_RETCODE_INVALID_PRICE");       break;
      case 10016: return("TRADE_RETCODE_INVALID_STOPS");       break;
      case 10017: return("TRADE_RETCODE_TRADE_DISABLED");      break;
      case 10018: return("TRADE_RETCODE_MARKET_CLOSED");       break;
      case 10019: return("TRADE_RETCODE_NO_MONEY");            break;
      case 10020: return("TRADE_RETCODE_PRICE_CHANGED");       break;
      case 10021: return("TRADE_RETCODE_PRICE_OFF");           break;
      case 10022: return("TRADE_RETCODE_INVALID_EXPIRATION");  break;
      case 10023: return("TRADE_RETCODE_ORDER_CHANGED");       break;
      case 10024: return("TRADE_RETCODE_TOO_MANY_REQUESTS");   break;
      case 10025: return("TRADE_RETCODE_NO_CHANGES");          break;
      case 10026: return("TRADE_RETCODE_SERVER_DISABLES_AT");  break;
      case 10027: return("TRADE_RETCODE_CLIENT_DISABLES_AT");  break;
      case 10028: return("TRADE_RETCODE_LOCKED");              break;
      case 10029: return("TRADE_RETCODE_FROZEN");              break;
      case 10030: return("TRADE_RETCODE_INVALID_FILL");        break;
      case 10031: return("TRADE_RETCODE_CONNECTION");          break;
      case 10032: return("TRADE_RETCODE_ONLY_REAL");           break;
      case 10033: return("TRADE_RETCODE_LIMIT_ORDERS");        break;
      case 10034: return("TRADE_RETCODE_LIMIT_VOLUME");        break;
      case 10035: return("TRADE_RETCODE_INVALID_ORDER");       break;
      case 10036: return("TRADE_RETCODE_POSITION_CLOSED");     break;
      default:
         return("TRADE_RETCODE_UNKNOWN="+IntegerToString(retcode));
         break;
     }
//---
  }
//+------------------------------------------------------------------+
```

Example of the Listener operation:

```
2016.06.09 14:51:19.763 TradeTransactionListener (Si-6.16,M15)  LAST PING=14 ms
Buying
2016.06.09 14:51:24.856 TradeTransactionListener (Si-6.16,M15)  1. OnTradeTransaction
2016.06.09 14:51:24.856 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_ADD order #49118594 ORDER_TYPE_BUY ORDER_STATE_STARTED
2016.06.09 14:51:24.859 TradeTransactionListener (Si-6.16,M15)  2. OnTradeTransaction
2016.06.09 14:51:24.859 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_REQUEST
2016.06.09 14:51:24.859 TradeTransactionListener (Si-6.16,M15)  MqlTradeResult: order #49118594 retcode=TRADE_RETCODE_PLACED
2016.06.09 14:51:24.859 TradeTransactionListener (Si-6.16,M15)  3. OnTradeTransaction
2016.06.09 14:51:24.859 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_UPDATE order #49118594 ORDER_TYPE_BUY ORDER_STATE_REQUEST_ADD
2016.06.09 14:51:24.881 TradeTransactionListener (Si-6.16,M15)  4. OnTradeTransaction
2016.06.09 14:51:24.881 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_UPDATE order #49118594 ORDER_TYPE_BUY ORDER_STATE_PLACED
2016.06.09 14:51:24.881 TradeTransactionListener (Si-6.16,M15)  5. OnTradeTransaction
2016.06.09 14:51:24.881 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_DELETE order #49118594 ORDER_TYPE_BUY ORDER_STATE_PLACED
2016.06.09 14:51:24.884 TradeTransactionListener (Si-6.16,M15)  6. OnTradeTransaction
2016.06.09 14:51:24.884 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_HISTORY_ADD order #49118594 ORDER_TYPE_BUY Si-6.16 ORDER_STATE_FILLED   (MOEX ticket=3377179723)
2016.06.09 14:51:24.884 TradeTransactionListener (Si-6.16,M15)  7. OnTradeTransaction
2016.06.09 14:51:24.885 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_DEAL_ADD deal #6945344 DEAL_TYPE_BUY Si-6.16 1.00 lot   (MOEX deal=185290434)
Setting SL/TP
2016.06.09 14:51:50.872 TradeTransactionListener (Si-6.16,M15)  1. OnTradeTransaction
2016.06.09 14:51:50.872 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_REQUEST
2016.06.09 14:51:50.872 TradeTransactionListener (Si-6.16,M15)  2. OnTradeTransaction
2016.06.09 14:51:50.872 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: Position  #0 Si-6.16 modified: SL=62000.00000 TP=67000.00000
Closing the position (selling)
2016.06.09 14:52:24.063 TradeTransactionListener (Si-6.16,M15)  1. OnTradeTransaction
2016.06.09 14:52:24.063 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_ADD order #49118750 ORDER_TYPE_SELL ORDER_STATE_STARTED
2016.06.09 14:52:24.067 TradeTransactionListener (Si-6.16,M15)  2. OnTradeTransaction
2016.06.09 14:52:24.067 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_REQUEST
2016.06.09 14:52:24.067 TradeTransactionListener (Si-6.16,M15)  MqlTradeResult: order #49118750 retcode=TRADE_RETCODE_PLACED
2016.06.09 14:52:24.067 TradeTransactionListener (Si-6.16,M15)  3. OnTradeTransaction
2016.06.09 14:52:24.067 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_UPDATE order #49118750 ORDER_TYPE_SELL ORDER_STATE_REQUEST_ADD
2016.06.09 14:52:24.071 TradeTransactionListener (Si-6.16,M15)  4. OnTradeTransaction
2016.06.09 14:52:24.071 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_UPDATE order #49118750 ORDER_TYPE_SELL ORDER_STATE_PLACED
2016.06.09 14:52:24.073 TradeTransactionListener (Si-6.16,M15)  5. OnTradeTransaction
2016.06.09 14:52:24.073 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_DEAL_ADD deal #6945378 DEAL_TYPE_SELL Si-6.16 1.00 lot   (MOEX deal=185290646)
2016.06.09 14:52:24.075 TradeTransactionListener (Si-6.16,M15)  6. OnTradeTransaction
2016.06.09 14:52:24.075 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_ORDER_DELETE order #49118750 ORDER_TYPE_SELL ORDER_STATE_PLACED
2016.06.09 14:52:24.077 TradeTransactionListener (Si-6.16,M15)  7. OnTradeTransaction
2016.06.09 14:52:24.077 TradeTransactionListener (Si-6.16,M15)  MqlTradeTransaction: TRADE_TRANSACTION_HISTORY_ADD order #49118750 ORDER_TYPE_SELL Si-6.16 ORDER_STATE_FILLED   (MOEX ticket=3377182821)
```

Now it is time to review source code examples.

### Working with a Trading Account

At the very start of the trading robot, it is necessary to obtain information about the trading account, on which the robot is to trade.

Account details can be accessed using the special [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) class. Let's include the use of the AccountInfo.mqh file into our code and declare the **account** variable of this class:

```
#include <Trade\AccountInfo.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- An object for working with the account
CAccountInfo account;
//--- Get the number of the account the EA is running on
   long login=account.Login();
   Print("Login=",login);
//--- Print the account currency
   Print("Account currency: ",account.Currency());
//--- Print balance and current5 profit on the account
   Print("Balance=",account.Balance(),"  Profit=",account.Profit(),"   Equity=",account.Equity());
//--- Print account type
   Print("Account type: ",account.TradeModeDescription());
//--- Find out whether trading is allowed on this account
   if(account.TradeAllowed())
      Print("Trading on the account is allowed");
   else
      Print("Trading on the account is not allowed: probably connected in with an investor password");
//--- Margin calculation mode
   Print("Margin calculation mode: ",account.MarginModeDescription());
//--- Check if trading using Expert Advisors is allowed on the account
   if(account.TradeExpert())
      Print("Automate trading on the account is allowed");
   else
      Print("Automated trading using Expert Advisors or scripts is not allowed");
//--- Is the maximum number of orders specified or not
   int orders_limit=account.LimitOrders();
   if(orders_limit!=0)Print("The maximum allowable number of actual pending orders: ",orders_limit);
//--- Print the name of the company and the name of the server
   Print(account.Company(),": server ",account.Server());
   Print(__FUNCTION__,"  completed"); //---
  }
```

As can be seen from the above code, by using the **account** variable in the OnInit() function, you can obtain a lot of useful information. You can add this code into your Expert Advisor, and it will be much easier to parse the logs while analyzing the EA operation.

The below figure shows the result of running the script.

![](https://c.mql5.com/2/23/accountinfo__2-en__1.png)

### Obtaining Properties of a Financial Instrument

We have obtained information about the account. However, in order to be able to perform trading operations, we also need to know the properties of the financial asset we are going to trade. For this purpose, we will use another convenient class [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) with a large number of methods. The below examples features some of them.

```
#include<Trade\SymbolInfo.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- An object for receiving symbol properties
CSymbolInfo symbol_info;
//--- Setting the name of the symbol, for which we want to obtain information
   symbol_info.Name(_Symbol);
//--- Receiving current quotes and printing them
   symbol_info.RefreshRates();
   Print(symbol_info.Name()," (",symbol_info.Description(),")",
         "  Bid=",symbol_info.Bid(),"   Ask=",symbol_info.Ask());
//--- Receiving the number of decimal places and the point value
   Print("Digits=",symbol_info.Digits(),
         ", Point=",DoubleToString(symbol_info.Point(),symbol_info.Digits()));
//--- Requesting order execution type, check restrictions
   Print("Restrictions on trading operations: ",EnumToString(symbol_info.TradeMode()),
         " (",symbol_info.TradeModeDescription(),")");
//--- Checking trade execution modes
   Print("Trade execution mode: ",EnumToString(symbol_info.TradeExecution()),
         " (",symbol_info.TradeExecutionDescription(),")");
//--- Finding out hoe the value of contracts is calculated
   Print("Calculating contract value: ",EnumToString(symbol_info.TradeCalcMode()),
         " (",symbol_info.TradeCalcModeDescription(),")");
//--- Contract size
   Print("Standard contract size: ",symbol_info.ContractSize());
//--- Value of initial margin per 1 contract
   Print("Initial margin for a standard contract: ",symbol_info.MarginInitial()," ",symbol_info.CurrencyBase());
//--- Minimum and maximum volume size in trading operations
   Print("Volume info: LotsMin=",symbol_info.LotsMin(),"  LotsMax=",symbol_info.LotsMax(),
         "  LotsStep=",symbol_info.LotsStep());
//---
   Print(__FUNCTION__,"  completed");
  }
```

The below figure shows the properties of symbol Si-6.16 traded on the FORTS market of Moscow Exchange. Now you are ready to move on to trading.

![](https://c.mql5.com/2/23/SymbolInfo__2.png)

### Programming Trading Operations

Trading orders can be sent using two MQL5 functions: OrderSend() and OrderSendAsync(). In fact, these are two implementations of the same function. OrderSend() sends a trade order and waits for its execution result, while the asynchronous OrderSendAsync() function only sends a request and allows the program to continue running without waiting for the trade server response. Thus it is really easy to trade with MQL5, where you only need to use one function for all trading operations.

Both functions receive the [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure as the first parameter, which contains over a dozen fields. The set of required fields depends on the [trading operation type](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions), therefore not all fields need to be filled in. If a value is incorrect or a required field is missing, the request will fail to pass verification in the terminal and will not be sent to the server. In five of the fields, you will need to correctly specify values from predefined enumerations.

The trade request contains so many fields, because you need to describe a lot of properties of an order, which may change depending on the fill policy, expiry and other parameters. No need to memorize all these details, simply use the ready [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class. Here is how the use of the class inside a trading robot may look like:

```
#include<Trade\Trade.mqh>
//--- An object for performing trading operations
CTrade  trade;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Setting MagicNumber to identify EA's orders
   int MagicNumber=123456;
   trade.SetExpertMagicNumber(MagicNumber);
//--- Setting allowable slippage in points for buying/selling
   int deviation=10;
   trade.SetDeviationInPoints(deviation);
//--- Order filling mode, use the mode that is allowed by the server
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
//--- Logging mode: it is advisable not to call this method as the class will set the optimal mode by itself
   trade.LogLevel(1);
//--- the function to be used for trading: true - OrderSendAsync(), false - OrderSend()
   trade.SetAsyncMode(true);
//---
   return(0);
  }
```

As a rule, the [ORDER\_FILLING\_RETURN](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling) mode is used for Exchange trading. Description from the language reference:

This mode is only used in the Market and Exchange Execution [modes](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution): for market (ORDER\_TYPE\_BUY and ORDER\_TYPE\_SELL), limit and stop-limit orders (ORDER\_TYPE\_BUY\_LIMIT, ORDER\_TYPE\_SELL\_LIMIT, ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT). In case of partial filling, a market or limit order with remaining volume is not canceled but processed further.

During activation of the ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT orders, an appropriate limit order ORDER\_TYPE\_BUY\_LIMIT/ORDER\_TYPE\_SELL\_LIMIT with the ORDER\_FILLING\_RETURN type is created.

Now it is time to discuss how CTrade helps in trading operations.

**Buying/Selling at the Current Price**

Often in trading strategies we need to buy or to sell an asset immediately at the current price. CTrade is familiar with this situation, and it only needs to know the required volume of the trading operation. All other parameters, including the Open price, symbol name, Stop Loss and Take Profit levels, as well as the comment to the order can be omitted.

```
//--- 1. An example of buying the current symbol
   if(!trade.Buy(1))
     {
      //--- Report the failure
      Print("The Buy() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The Buy() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

By default, if the symbol name is not specified, CTrade will use the name of the symbol, on whose chart it is running. This is great for simple strategies. If an Expert Advisor trades multiple symbols at the same time, the symbol of a trade operation should be specified for each operation.

```
//--- 2. An example of buying the specified symbol
   if(!trade.Buy(1,"Si-6.16"))
     {
      //--- Report the failure
      Print("The Buy() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The Buy() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

You can specify all the parameters of an order: Stop Loss/Take Profit, open price and comment.

```
//--- 3. An example of buying the specified symbol with the preset SL and TP
   double volume=1;           // Specifying the volume of the trading operation
   string symbol="Si-6.16";   // Specifying the symbol of the trading operation
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // Number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // Point
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);             // Current price to close LONG
   double SL=bid-100*point;                                    // Unnormalized value of SL
   SL=NormalizeDouble(SL,digits);                              // Normalizing Stop Loss
   double TP=bid+100*point;                                    // Unnormalized value of TP
   TP=NormalizeDouble(TP,digits);                              // Normalizing Take Profit
//--- Getting the current open price for LONG positions
   double open_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
   string comment=StringFormat("Buy %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(open_price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
   if(!trade.Buy(volume,symbol,open_price,SL,TP,comment))
     {
      //--- Report the failure
      Print("The Buy() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The Buy() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

MagicNumber and allowable slippage were specified during the initialization of the CTrade instance, so we do not need to set them now. Although they can also be set directly before each trading operation, if necessary.

**Placing a Limit Order**

A limit order can be sent using the appropriate method of the class: [BuyLimit()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit) or [SellLimit()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit). A brief version, where you only specify the open price and the volume, will be enough in most situations. The open price for BuyLimit must be below the current price, and it must be above the current price for SellLimit. These orders are used to enter the market at a better price, for example, in strategies that expect price to roll back from the support level. The symbol on which the EA is running is used in this case:

```
//--- 1. An example of placing a pending BuyLimit order
   string symbol="Si-6.16";                                    // Specifying the symbol of the order
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // Number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // Point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // Current buying price
   double price=ask-100*point;                                 // Unnormalized opening price
   price=NormalizeDouble(price,digits);                        // Normalizing the opening price
//--- Everything is ready, sending the pending Buy Limit order to the server
   if(!trade.BuyLimit(1,price))
     {
      //--- Report the failure
      Print("The BuyLimit() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The BuyLimit() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

You can also use a more detailed version, where you need to specify all parameters: SL/TP, expiry, the name of the symbol and a comment to the order.

```
//--- 2. An example of placing a pending BuyLimit order with all parameters
   double volume=1;
   string symbol="Si-6.16";                                    // Specifying the symbol of the order
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // Number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // Point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // Current buying price
   double price=ask-100*point;                                 // Unnormalized opening price
   price=NormalizeDouble(price,digits);                        // Normalizing the opening price
   int SL_pips=100;                                            // Stop Loss in points
   int TP_pips=100;                                            // Take Profit in points
   double SL=price-SL_pips*point;                              // Unnormalized value of SL
   SL=NormalizeDouble(SL,digits);                              // Normalizing Stop Loss
   double TP=price+TP_pips*point;                              // Unnormalized value of TP
   TP=NormalizeDouble(TP,digits);                              // Normalizing Take Profit
   datetime expiration=TimeTradeServer()+PeriodSeconds(PERIOD_D1);
   string comment=StringFormat("Buy Limit %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
//--- Everything is ready, sending the pending Buy Limit order to the server
   if(!trade.BuyLimit(volume,price,symbol,SL,TP,ORDER_TIME_DAY,expiration,comment))
     {
      //--- Report the failure
      Print("The BuyLimit() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The BuyLimit() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

In the second variant it is necessary to correctly specify the SL and TP levels. Note that when you buy, Take Profit must be above the open price, and Stop Loss must be below the open price. Mirrored for SellLimit orders. If a level is set incorrectly, you will easily find it out while backtesting your robot, because the CTrade class **automatically** prints logs in such cases (if you do not call the [LogLevel](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeloglevel) function).

**Placing a Stop Order**

Stop orders can be sent using [BuyStop()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop) and [SellStop()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) methods. The open price for Buy Stop must be above the current price, and below for SellStop. Stop orders are used in strategies that enter the market once the price breaks a resistance level, as well as they can help to limit losses. Simple version:

```
//--- 1. An example of placing a pending Buy Stop order
   string symbol="RTS-6.16";    // Specifying the symbol of the order
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // Number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // Point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // Current buying price
   double price=ask+100*point;                                 // Unnormalized opening price
   price=NormalizeDouble(price,digits);                        // Normalizing the opening price
//--- Everything is ready, sending the pending Buy Stop order to the server
   if(!trade.BuyStop(1,price))
     {
      //--- Report the failure
      Print("The BuyStop() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The BuyStop() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

Here is a more detailed version when you need to specify more parameters for the pending BuyStop order:

```
//--- 2. An example of placing a pending Buy Stop order with all parameters
   double volume=1;
   string symbol="RTS-6.16";    // Specifying the symbol of the order
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // Number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // Point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // Current buying price
   double price=ask+100*point;                                 // Unnormalized opening price
   price=NormalizeDouble(price,digits);                        // Normalizing the opening price
   int SL_pips=100;                                            // Stop Loss in points
   int TP_pips=100;                                            // Take Profit in points
   double SL=price-SL_pips*point;                              // Unnormalized value of SL
   SL=NormalizeDouble(SL,digits);                              // Normalizing Stop Loss
   double TP=price+TP_pips*point;                              // Unnormalized value of TP
   TP=NormalizeDouble(TP,digits);                              // Normalizing Take Profit
   datetime expiration=TimeTradeServer()+PeriodSeconds(PERIOD_D1);
   string comment=StringFormat("Buy Stop %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
//--- Everything is ready, sending the pending Buy Stop order to the server
   if(!trade.BuyStop(volume,price,symbol,SL,TP,ORDER_TIME_DAY,expiration,comment))
     {
      //--- Report the failure
      Print("The BuyStop() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The BuyStop() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

A SellStop order can be sent using the appropriate method of the CTrade class. In this case it is important to correctly specify the price.

**Working with a Position**

Instead of using the Buy() and Sell() methods, you can use methods for opening a position. However, in this case you will need to specify more details:

```
//--- Number of decimal places
   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
//--- Point value
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
//--- Getting the buying price
   double price=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
//--- Calculating and normalizing the SL and TP levels
   double SL=NormalizeDouble(price-100*point,digits);
   double TP=NormalizeDouble(price+100*point,digits);
//--- Adding a comment
   string comment="Buy "+_Symbol+" 1 at "+DoubleToString(price,digits);
//--- Everything is ready, trying to open a Buy position
   if(!trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,1,price,SL,TP,comment))
     {
      //--- Report the failure
      Print("The PositionOpen() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The PositionOpen() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

In order to close a position, we only need to specify the name of the symbol, and the CTrade class will do the rest.

```
//--- Closing the position on the current symbol
   if(!trade.PositionClose(_Symbol))
     {
      //--- Report the failure
      Print("The PositionClose() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The PositionClose() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

It is possible to change the StopLoss and TakeProfit levels of an open position. This can be done using the ModifyPosition() method.

```
//--- Number of decimal places
   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
//--- Point value
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
//--- Getting the current Bid price
   double price=SymbolInfoDouble(_Symbol,SYMBOL_BID);
//--- Calculating and normalizing the SL and TP levels
   double SL=NormalizeDouble(price-100*point,digits);
   double TP=NormalizeDouble(price+100*point,digits);
//--- Everything is ready, trying to modify a Buy position
   if(!trade.PositionModify(_Symbol,SL,TP))
     {
      //--- Report the failure
      Print("The PositionModify() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The PositionModify() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

**Modifying and Deleting an Order**

For modifying parameters of a pending order, the CTrade class provides the [OrderModify()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeordermodify) method, to which the required parameters should be passed.

```
//--- Checking if the order exists
   if(!OrderSelect(ticket))
     {
      Print("Order #",ticket," not found");
      return;

     }
//--- Symbol
   string symbol=OrderGetString(ORDER_SYMBOL);
//--- Number of decimal places
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
//--- Point value
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
//--- Getting the opening price
   double price=OrderGetDouble(ORDER_PRICE_OPEN);
//--- Calculating and normalizing the SL and TP levels
   double SL=NormalizeDouble(price-200*point,digits);
   double TP=NormalizeDouble(price+200*point,digits);
//--- Everything is ready, trying to modify the order
   if(!trade.OrderModify(ticket,price,SL,TP,ORDER_TIME_DAY,0))
     {
      //--- Report the failure
      Print("The OrderModify() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The OrderModify() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

You need to obtain the ticket of the order that you want to modify, and to specify correct StopLoss and TakeProfit levels depending on the order type. In addition, the new opening price must also be correct in relation to the current price.

If you want to delete an order, you only need to know its ticket:

```
//--- Checking if the order exists
   if(!OrderSelect(ticket))
     {
      Print("Order #",ticket," not found");
      return;
     }
//--- Everything is ready, trying to delete the order
   if(!trade.OrderDelete(ticket))
     {
      //--- Report the failure
      Print("The OrderDelete() method has failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("The OrderDelete() method has been successfully performed. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

The class also contains the universal [OrderOpen()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) method, which can place pending orders of any type. Unlike the specialized [BuyLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit), [BuyStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop), [SellLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit) and [SellStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) methods, OrderOpen() contains more required parameters. So you may choose what method to use.

### What's More in Trade Classes

In this article we have analyzed simple techniques for programming trading Buy and Sell operations, as well as for programming operations with pending orders. However, the Trade Classes section contains some more convenient assistants for MQL5 robot developers:

- [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) — for operations with orders;

- [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) — for operations with historic orders;

- [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) — for operations with positions;

- [CDealInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo) — for operations with trades;
- [CTerminalInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo) — for obtaining information about the terminal.


Use of these classes will help you focus on the trading side of your strategy while maximally solving technical issues. In addition, the CTrade class can be used to study trade requests, for example using the [debugger](https://www.metatrader5.com/en/metaeditor/help/development/debug "https://www.metatrader5.com/en/metaeditor/help/development/debug"). Then after some time you will be able to create your own classes based on CTrade, and implement the required logic of processing trade request execution results in this class.

### Start your journey to algorithmic trading with simple scripts

The methods of MQL5 robot development described in this article are mainly intended for beginners, although experienced developers may also find something new and useful here. Try to execute the simple scripts described in this article, and you will understand that creating a trading robot is easier than you might have thought.

If you want to study the topic further, here are two more related articles:

1. [Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)
2. [How to Secure Yourself and Your Expert Advisor While Trading on the Moscow Exchange](https://www.mql5.com/en/articles/1683)


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2513](https://www.mql5.com/ru/articles/2513)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2513.zip "Download all attachments in the single ZIP archive")

[accountinfo.mq5](https://www.mql5.com/en/articles/download/2513/accountinfo.mq5 "Download accountinfo.mq5")(4.6 KB)

[symbolinfo.mq5](https://www.mql5.com/en/articles/download/2513/symbolinfo.mq5 "Download symbolinfo.mq5")(4.91 KB)

[expert\_sample.mq5](https://www.mql5.com/en/articles/download/2513/expert_sample.mq5 "Download expert_sample.mq5")(37.09 KB)

[tradetransactionlistener.mq5](https://www.mql5.com/en/articles/download/2513/tradetransactionlistener.mq5 "Download tradetransactionlistener.mq5")(16.13 KB)

[limit\_sample.mq5](https://www.mql5.com/en/articles/download/2513/limit_sample.mq5 "Download limit_sample.mq5")(12.86 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/89849)**
(16)


![Denis Sartakov](https://c.mql5.com/avatar/2020/8/5F49477B-C850.jpg)

**[Denis Sartakov](https://www.mql5.com/en/users/denissergeev)**
\|
17 Sep 2016 at 14:30

**Renat Fatkhullin:**

On MOEX instruments, sessions come from the exchange and are prescribed a day in advance.

I'm sorry, but there are two schedules,

this one:

10.00 - 14.00 Main trading session (daily Settlement period)

14.00 - 14.05 Day clearing session (intermediate clearing)

14.05 - 18.45 Main Trading Session (evening Clearing Period)

18.45 - 19.00 Evening Clearing Session (Main Clearing)

19.00 - 23.50 Evening Additional Trading Session

and the schedule that this function produces:

but **[SymbolInfoSessionTrade](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote "MQL5 Documentation: Obtaining Market Information") (....**) it gives a different schedule....

**Suppose for symbol RTS-12.16 this function gives the following interval**

**for trading session 15:45 - 22:15, does it mean that trade orders for this symbol outside this interval**

**the server will not execute ? and what about clearing ? clearing, as I understand, will be executed in any case ?**

![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
17 Sep 2016 at 16:09

**Denis Sartakov:**

I'm sorry, but there are two schedules,

this one:

10.00 - 14.00 Main Trading Session (Day Clearing Session)

14.00 - 14.05 Day clearing session (intermediate clearing)

14.05 - 18.45 Main Trading Session (evening Clearing Period)

18.45 - 19.00 Evening Clearing Session (Main Clearing)

19.00 - 23.50 Evening Additional Trading Session

and the schedule that this function produces:

but **[SymbolInfoSessionTrade](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote "MQL5 Documentation: Obtaining Market Information") (....**) it gives a different schedule....

**Suppose for symbol RTS-12.16 this function gives the following interval**

**for trading session 15:45 - 22:15, does it mean that trade orders for this symbol outside this interval**

**the server will not execute ? and what about clearing ? clearing, as I understand, will be executed in any case ?**

Please provide a screenshot of the symbol properties with sessions, the full format of the request and the result of the request.


![Denis Sartakov](https://c.mql5.com/avatar/2020/8/5F49477B-C850.jpg)

**[Denis Sartakov](https://www.mql5.com/en/users/denissergeev)**
\|
17 Sep 2016 at 17:16

**Renat Fatkhullin:**

Please provide a screenshot of the symbol properties with sessions, the full query format and the query result.

Unfortunately, I only have a demo account with Otkritie broker,

and everyone here shouts that demo and real are heaven and earth, that's why I only work out on demo

trading sets, I pay little attention to the other moments, I'm not sure, but it seems that the demo-server

doesn't celebrate trading sessions given by **[SymbolInfoSessionTrade](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote "MQL5 Documentation: Obtaining Market Information")** function(....),

i.e. it does not pay attention to the contract specification....

I am preparing for a real account, that's why I ask such questions....

![Denis Sartakov](https://c.mql5.com/avatar/2020/8/5F49477B-C850.jpg)

**[Denis Sartakov](https://www.mql5.com/en/users/denissergeev)**
\|
17 Sep 2016 at 20:10

**Rashid Umarov:**

The function outputs exactly what is specified on the trade server in the contract specification.

It's clear, but I don't understand what schedule to trade on.

on the schedule that this function produces?

but your Script is fine, "you can feel the hand of the master..."

![JRandomTrader](https://c.mql5.com/avatar/avatar_na2.png)

**[JRandomTrader](https://www.mql5.com/en/users/jrandomtrader)**
\|
12 Dec 2021 at 09:12

**Denis Sartakov [#](https://www.mql5.com/ru/forum/88499/page2#comment_2822342):**

it's clear, it's just not clear what the timetable is for trading.

according to the schedule given by this function ?

but your Script is fine, "you can feel the hand of the master..."

In Open, this function, as well as the specification, gives the average temperature in the hospital.

[![GMKN Spec.](https://c.mql5.com/3/375/MT5-GMKN_Spec__1.png)](https://c.mql5.com/3/375/MT5-GMKN_Spec.png "https://c.mql5.com/3/375/MT5-GMKN_Spec.png")

![Regular expressions for traders](https://c.mql5.com/2/23/ava.png)[Regular expressions for traders](https://www.mql5.com/en/articles/2432)

A regular expression is a special language for handling texts by applying a specified rule, also called a regex or regexp for short. In this article, we are going to show how to handle a trade report with the RegularExpressions library for MQL5, and will also demonstrate the optimization results after using it.

![How to create bots for Telegram in MQL5](https://c.mql5.com/2/22/telegram-avatar.png)[How to create bots for Telegram in MQL5](https://www.mql5.com/en/articles/2355)

This article contains step-by-step instructions for creating bots for Telegram in MQL5. This information may prove useful for users who wish to synchronize their trading robot with a mobile device. There are samples of bots in the article that provide trading signals, search for information on websites, send information about the account balance, quotes and screenshots of charts to you smart phone.

![Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://c.mql5.com/2/23/avatar__3.png)[Using text files for storing input parameters of Expert Advisors, indicators and scripts](https://www.mql5.com/en/articles/2564)

The article describes the application of text files for storing dynamic objects, arrays and other variables used as properties of Expert Advisors, indicators and scripts. The files serve as a convenient addition to the functionality of standard tools offered by MQL languages.

![Creating an assistant in manual trading](https://c.mql5.com/2/23/panel__1.png)[Creating an assistant in manual trading](https://www.mql5.com/en/articles/2281)

The number of trading robots used on the currency markets has significantly increased recently. They employ various concepts and strategies, however, none of them has yet succeeded to create a win-win sample of artificial intelligence. Therefore, many traders remain committed to manual trading. But even for such specialists, robotic assistants or, so called, trading panels, are created. This article is yet another example of creating a trading panel from scratch.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/2513&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062502446342775638)

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