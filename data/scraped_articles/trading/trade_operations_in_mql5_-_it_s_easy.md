---
title: Trade Operations in MQL5 - It's Easy
url: https://www.mql5.com/en/articles/481
categories: Trading, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:21:57.670738
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/481&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069378139227882515)

MetaTrader 5 / Examples


Almost all traders come to market to make money but some traders also enjoy the process itself. However, it is not only manual trading that can provide you with an exciting experience. Automated trading systems development can also be quite absorbing. Creating a trading robot can be as interesting as reading a good mystery novel.

When developing a trading algorithm, we have to deal with plenty of technical issues including the most important ones:

1. What to trade?
2. When to trade?
3. How to trade?


We need to answer the first question to choose the most suitable symbol. Our choice can be affected by many factors including the ability to automate our trading system for the market. The second question involves elaboration of the trading rules clearly indicating deals' direction, as well as entry and exit points. The third question seems to be relatively simple: how to buy and sell using some definite programming language?

In this article we will consider how to implement trade operations in algorithmic trading using MQL5 language.

![](https://c.mql5.com/2/4/eggs__1.jpg)

### MQL5 Features for Algo Trading

MQL5 is a trading strategies' programming language having plenty of [trade functions](https://www.mql5.com/en/docs/trading) for working with orders, positions and trade requests. Thus, making algo trading robots in MQL5 is the least labor-consuming task for a developer.

MQL5 features allow you to make a [trade request](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) and send it to a server using [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) or [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) functions, receive [its processing result](https://www.mql5.com/en/docs/constants/structures/mqltraderesult), view the trading history, examine [contract specification](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants) for a symbol, handle a [trade event](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction), as well as receive other necessary data.

Besides, MQL5 can be used for writing custom technical indicators and applying already implemented ones, drawing any marks and objects on a chart, developing custom user interface, etc. Implementation examples can be seen in multiple [articles](https://www.mql5.com/en/articles).

### Trade Operations: Easy As ABC!

There are several basic types of trade operations that can be necessary for your trading robot:

1. buying/selling at the current price,
2. placing a pending order for buying/selling according to a certain condition,
3. modifying/deleting a pending order,
4. closing/adding to/reducing/reversing a position.


All these operations are performed using OrderSend() function. There is also an asynchronous version called OrderSendAsync(). All variety of trade operations is described by [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure containing a trade request description. Therefore, only correct filling of MqlTradeRequest  structure and handling request execution results can be challenging when dealing with trade operations.

According to your trading system, you can buy or sell at market price (BUY or SELL), as well as place a pending buy/sell order at some distance from the current market price:

- BUY STOP, SELL STOP - buying or selling in case of a specified level breakout (worse than the current price);

- BUY LIMIT, SELL LIMIT - buying or selling in case a specified level is reached (better than the current price);
- BUY STOP LIMIT, SELL STOP LIMIT - setting BUY LIMIT or SELL LIMIT in case a specified price is reached.


These standard order types correspond to [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) enumeration.

![](https://c.mql5.com/2/4/Charts-EN.png)

You may need to modify or delete a pending order. This can also be done using OrderSend()/OrderSendAsync() functions. Modifying an open position is also quite an easy process, as it is performed using the same trade operations.

If you think trade operations to be complex and intricate, it is about time that you change your mind. We will show not only how to code buys and sells in MQL5 quickly and easily but also how to work with a trade account and symbols' properties. [Trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) will help us in this undertaking.

### Check Your Trading Account with CAccountInfo

The first thing you need to know when launching your trading robot is what trading account will be used for its operation. Since we are writing a training code, we will implement a check for the case when Expert Advisor has been launched on a real account.

[CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) class is used for working with an account. We will add AccountInfo.mqh file inclusion and declare the variable of the class - **account**:

```
#include <Trade\AccountInfo.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- object for working with the account
   CAccountInfo account;
//--- receiving the account number, the Expert Advisor is launched at
   long login=account.Login();
   Print("Login=",login);
//--- clarifying account type
   ENUM_ACCOUNT_TRADE_MODE account_type=account.TradeMode();
//--- if the account is real, the Expert Advisor is stopped immediately!
   if(account_type==ACCOUNT_TRADE_MODE_REAL)
     {
      MessageBox("Trading on a real account is forbidden, disabling","The Expert Advisor has been launched on a real account!");
      return(-1);
     }
//--- displaying the account type
   Print("Account type: ",EnumToString(account_type));
//--- clarifying if we can trade on this account
   if(account.TradeAllowed())
      Print("Trading on this account is allowed");
   else
      Print("Trading on this account is forbidden: you may have entered using the Investor password");
//--- clarifying if we can use an Expert Advisor on this account
   if(account.TradeExpert())
      Print("Automated trading on this account is allowed");
   else
      Print("Automated trading using Expert Advisors and scripts on this account is forbidden");
//--- clarifying if the permissible number of orders has been set
   int orders_limit=account.LimitOrders();
   if(orders_limit!=0)Print("Maximum permissible amount of active pending orders: ",orders_limit);
//--- displaying company and server names
   Print(account.Company(),": server ",account.Server());
//--- displaying balance and current profit on the account in the end
   Print("Balance=",account.Balance(),"  Profit=",account.Profit(),"   Equity=",account.Equity());
   Print(__FUNCTION__,"  completed"); //---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
```

As we can see from the above code, plenty of useful data can be received using **account** variable in OnInit() function. You can add this code to your Expert Advisor to examine the logs easily when analyzing its operation.

Results of an Expert Advisor launched on the Automated Trading Championship 2012 account are shown below.

![](https://c.mql5.com/2/4/CAccountInfo__1.png)

### Receiving Symbol Settings with CSymbolInfo

We now have the data on the account but we also need to know the properties of the symbol we are going to trade before performing the necessary operations. [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class with great number of methods is designed for these purposes. We will show only a small part of the methods in the below example.

```
#include<Trade\SymbolInfo.mqh>
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- object for receiving symbol settings
   CSymbolInfo symbol_info;
//--- set the name for the appropriate symbol
   symbol_info.Name(_Symbol);
//--- receive current rates and display
   symbol_info.RefreshRates();
   Print(symbol_info.Name()," (",symbol_info.Description(),")",
         "  Bid=",symbol_info.Bid(),"   Ask=",symbol_info.Ask());
//--- receive minimum freeze levels for trade operations
   Print("StopsLevel=",symbol_info.StopsLevel()," pips, FreezeLevel=",
         symbol_info.FreezeLevel()," pips");
//--- receive the number of decimal places and point size
   Print("Digits=",symbol_info.Digits(),
         ", Point=",DoubleToString(symbol_info.Point(),symbol_info.Digits()));
//--- spread info
   Print("SpreadFloat=",symbol_info.SpreadFloat(),", Spread(current)=",
         symbol_info.Spread()," pips");
//--- request order execution type for limitations
   Print("Limitations for trade operations: ",EnumToString(symbol_info.TradeMode()),
         " (",symbol_info.TradeModeDescription(),")");
//--- clarifying trades execution mode
   Print("Trades execution mode: ",EnumToString(symbol_info.TradeExecution()),
         " (",symbol_info.TradeExecutionDescription(),")");
//--- clarifying contracts price calculation method
   Print("Contract price calculation: ",EnumToString(symbol_info.TradeCalcMode()),
         " (",symbol_info.TradeCalcModeDescription(),")");
//--- sizes of contracts
   Print("Standard contract size: ",symbol_info.ContractSize(),
         " (",symbol_info.CurrencyBase(),")");
//--- minimum and maximum volumes in trade operations
   Print("Volume info: LotsMin=",symbol_info.LotsMin(),"  LotsMax=",symbol_info.LotsMax(),
         "  LotsStep=",symbol_info.LotsStep());
//---
   Print(__FUNCTION__,"  completed");
//---
   return(0);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

  }
```

EURUSD properties at the Automated Trading Championship are shown below. Now we are ready to perform trade operations.

![](https://c.mql5.com/2/4/CSymbolInfo__1.png)

### CTrade - Convenient Class for Trade Operations

Trading in MQL5 is performed only by two functions - OrderSend() and OrderSendAsync(). In fact, these are two implementations of one function. OrderSend() sends a trade request and waits for its execution result, while asynchronous OrderSendAsync() just sends a request allowing the application to continue its operation without waiting for a trading server's response. Thus, it is really easy to trade in MQL5, as you use only one function for all trade operations.

So, what is the challenge? Both functions receive [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure containing more than a dozen of fields as the first parameter. Not all fields should be necessarily filled. The set of necessary ones depends on a [trade operation type](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions). Incorrect value or blank field that is necessary to be filled will result in an error and the request will not be sent to a server. 5 of these fields require correct values from predefined enumerations.

Such a large number of fields is necessary for describing lots of order properties in one trade request. The orders may change depending on execution policy, expiration time and some other parameters. But you do not have to learn all these subtleties. Just use ready-made [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class. That is how the class can be used in your trading robot:

```
#include<Trade\Trade.mqh>
//--- object for performing trade operations
CTrade  trade;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- set MagicNumber for your orders identification
   int MagicNumber=123456;
   trade.SetExpertMagicNumber(MagicNumber);
//--- set available slippage in points when buying/selling
   int deviation=10;
   trade.SetDeviationInPoints(deviation);
//--- order filling mode, the mode allowed by the server should be used
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
//--- logging mode: it would be better not to declare this method at all, the class will set the best mode on its own
   trade.LogLevel(1);
//--- what function is to be used for trading: true - OrderSendAsync(), false - OrderSend()
   trade.SetAsyncMode(true);
//---
   return(0);
  }
```

Now let's see how CTrade helps in trade operations.

**Buying/selling at the current price**

Trading strategies often provide the possibility to buy or sell at the current price right now. In this case, CTrade asks only to specify a necessary trade operation volume. All other parameters (open price and symbol name, Stop Loss and Take Profit levels, order comments) are optional.

```
//--- 1. example of buying at the current symbol
   if(!trade.Buy(0.1))
     {
      //--- failure message
      Print("Buy() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

By default, CTrade will use the symbol name of the chart it has been launched on if the symbol name is not specified. It is convenient for simple strategies. For multicurrency strategies you should always explicitly specify the symbol, for which the trade operation will be performed.

```
//--- 2. example of buying at the specified symbol
   if(!trade.Buy(0.1,"GBPUSD"))
     {
      //--- failure message
      Print("Buy() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

All order parameters may be specified: Stop Loss/Take Profit levels, open price and comments.

```
//--- 3. example of buying at the specified symbol with specified SL and TP
   double volume=0.1;         // specify a trade operation volume
   string symbol="GBPUSD";    //specify the symbol, for which the operation is performed
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double bid=SymbolInfoDouble(symbol,SYMBOL_BID);             // current price for closing LONG
   double SL=bid-1000*point;                                   // unnormalized SL value
   SL=NormalizeDouble(SL,digits);                              // normalizing Stop Loss
   double TP=bid+1000*point;                                   // unnormalized TP value
   TP=NormalizeDouble(TP,digits);                              // normalizing Take Profit
//--- receive the current open price for LONG positions
   double open_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
   string comment=StringFormat("Buy %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(open_price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
   if(!trade.Buy(volume,symbol,open_price,SL,TP,comment))
     {
      //--- failure message
      Print("Buy() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

As we have already said, Magic Number and permissible slippage were set when initializing Ctrade copy. Therefore, they are not required. However, they can also be set before each trade operation if necessary.

**Placing a limit order**

The appropriate [BuyLimit()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit) or [SellLimit()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit) class method is used for sending a limit order. The shortened version (when only an open price and a volume are specified) will be appropriate in most cases. Open price for Buy Limit should be lower than the current price, while it should be higher for Sell Limit. These orders are used to enter the market at the best price and are usually most suitable for the strategies expecting the price bounce from the support line. The symbol, at which an Expert Advisor has been launched, is used in that case:

```
//--- 1. example of placing a Buy Limit pending order
   string symbol="GBPUSD";    // specify the symbol, at which the order is placed
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // current buy price
   double price=1000*point;                                   // unnormalized open price
   price=NormalizeDouble(price,digits);                       // normalizing open price
//--- everything is ready, sending a Buy Limit pending order to the server
   if(!trade.BuyLimit(0.1,price))
     {
      //--- failure message
      Print("BuyLimit() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("BuyLimit() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

More detailed version with specifying all parameters can also be used: SL/TP levels, expiration time, symbol name and comments to the order.

```
//--- 2. example of placing a Buy Limit pending order with all parameters
   double volume=0.1;
   string symbol="GBPUSD";    // specify the symbol, at which the order is placed
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // current buy price
   double price=1000*point;                                 // unnormalized open price
   price=NormalizeDouble(price,digits);                      // normalizing open price
   int SL_pips=300;                                         // Stop Loss in points
   int TP_pips=500;                                         // Take Profit in points
   double SL=price-SL_pips*point;                           // unnormalized SL value
   SL=NormalizeDouble(SL,digits);                            // normalizing Stop Loss
   double TP=price+TP_pips*point;                           // unnormalized TP value
   TP=NormalizeDouble(TP,digits);                            // normalizing Take Profit
   datetime expiration=TimeTradeServer()+PeriodSeconds(PERIOD_D1);
   string comment=StringFormat("Buy Limit %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
//--- everything is ready, sending a Buy Limit pending order to the server
   if(!trade.BuyLimit(volume,price,symbol,SL,TP,ORDER_TIME_GTC,expiration,comment))
     {
      //--- failure message
      Print("BuyLimit() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("BuyLimit() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

Your task in the second version is to indicate SL and TP levels correctly. It should be noted that Take Profit level must be higher than open price when buying, while Stop Loss level must be lower than open price. The reverse situation is for Sell Limit orders. You can easily know about your error when testing an Expert Advisor on historical data. In such cases CTrade class **automatically** displays messages (unless you have called [LogLevel](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeloglevel) function).

**Placing a stop order**

Similar [BuyStop()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop) and [SellStop()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) methods are used to send a stop order. Open price for Buy Stop should be higher than the current price, while it should be lower for Sell Stop. Stop orders are used in strategies that enter the market during a resistance level breakout, as well as for cutting losses. Simple version:

```
//--- 1. example of placing a Buy Stop pending order
   string symbol="USDJPY";    // specify the symbol, at which the order is placed
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // current buy price
   double price=1000*point;                                    // unnormalized open price
   price=NormalizeDouble(price,digits);                        // normalizing open price
//--- everything is ready, sending a Buy Stop pending order to the server
   if(!trade.BuyStop(0.1,price))
     {
      //--- failure message
      Print("BuyStop() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("BuyStop() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

More detailed version when the maximum amount of parameters for Buy Stop pending order should be specified:

```
//--- 2. example of placing a Buy Stop pending order with all parameters
   double volume=0.1;
   string symbol="USDJPY";    // specify the symbol, at which the order is placed
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
   double ask=SymbolInfoDouble(symbol,SYMBOL_ASK);             // current buy price
   double price=1000*point;                                   // unnormalized open price
   price=NormalizeDouble(price,digits);                       // normalizing open price
   int SL_pips=300;                                          // Stop Loss in points
   int TP_pips=500;                                          // Take Profit in points
   double SL=price-SL_pips*point;                            // unnormalized SL value
   SL=NormalizeDouble(SL,digits);                             // normalizing Stop Loss
   double TP=price+TP_pips*point;                            // unnormalized TP value
   TP=NormalizeDouble(TP,digits);                             // normalizing Take Profit
   datetime expiration=TimeTradeServer()+PeriodSeconds(PERIOD_D1);
   string comment=StringFormat("Buy Stop %s %G lots at %s, SL=%s TP=%s",
                               symbol,volume,
                               DoubleToString(price,digits),
                               DoubleToString(SL,digits),
                               DoubleToString(TP,digits));
//--- everything is ready, sending a Buy Stop pending order to the server
   if(!trade.BuyStop(volume,price,symbol,SL,TP,ORDER_TIME_GTC,expiration,comment))
     {
      //--- failure message
      Print("BuyStop() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("BuyStop() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

The appropriate CTrade class method is used to send Sell Stop order. Specifying the prices correctly is of critical importance here.

**Working with positions**

You can use position opening methods instead of Buy() and Sell() ones but you will have to specify more details in this case:

```
//--- number of decimal places
   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
//--- point value
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
//--- receiving a buy price
   double price=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
//--- calculate and normalize SL and TP levels
   double SL=NormalizeDouble(price-1000*point,digits);
   double TP=NormalizeDouble(price+1000*point,digits);
//--- filling comments
   string comment="Buy "+_Symbol+" 0.1 at "+DoubleToString(price,digits);
//--- everything is ready, trying to open a buy position
   if(!trade.PositionOpen(_Symbol,ORDER_TYPE_BUY,0.1,price,SL,TP,comment))
     {
      //--- failure message
      Print("PositionOpen() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("PositionOpen() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

You need to specify only a symbol name, the rest will be done by CTrade class.

```
//--- closing a position at the current symbol
   if(!trade.PositionClose(_Symbol))
     {
      //--- failure message
      Print("PositionClose() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("PositionClose() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

Only Stop Loss and Take Profit levels are available for modifying an open position. This is done using PositionModify() method

```
//--- number of decimal places
   int    digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
//--- point value
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
//--- receiving the current Bid price
   double price=SymbolInfoDouble(_Symbol,SYMBOL_BID);
//--- calculate and normalize SL and TP levels
   double SL=NormalizeDouble(price-1000*point,digits);
   double TP=NormalizeDouble(price+1000*point,digits);
//--- everything is ready, trying to modify the buy position
   if(!trade.PositionModify(_Symbol,SL,TP))
     {
      //--- failure message
      Print("Метод PositionModify() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("PositionModify() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

**Modifying and d** **eleting an order**

[OrderModify()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeordermodify) method has been implemented in CTrade class to change pending order's parameters. All required parameters should be submitted to this method.

```
//--- this is a sample order ticket, it should be received
   ulong ticket=1234556;
//--- this is a sample symbol, it should be received
   string symbol="EURUSD";
//--- number of decimal places
   int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS);
//--- point value
   double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
//--- receiving a buy price
   double price=SymbolInfoDouble(symbol,SYMBOL_ASK);
//--- calculate and normalize SL and TP levels
//--- they should be calculated based on the order type
   double SL=NormalizeDouble(price-1000*point,digits);
   double TP=NormalizeDouble(price+1000*point,digits);
   //--- setting one day as a lifetime
   datetime expiration=TimeTradeServer()+PeriodSeconds(PERIOD_D1);
//--- everything is ready, trying to modify the order
   if(!trade.OrderModify(ticket,price,SL,TP,ORDER_TIME_GTC,expiration))
     {
      //--- failure message
      Print("OrderModify() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("OrderModify() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

You should receive the ticket of the order that should be changed. Correct Stop Loss and Take Profit levels should be specified depending on its type. Besides, the new open price shpuld also be correct relative to the current price.

You should know a ticket of an order to delete it:

```
//--- this is a sample order ticket, it should be received
   ulong ticket=1234556;
//--- everyrging is ready, trying to modify a buy position
   if(!trade.OrderDelete(ticket))
     {
      //--- failure message
      Print("OrderDelete() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("OrderDelete() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
```

The class also contains the multipurpose [OrderOpen()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) method, which can set pending orders of any type. Unlike specialized [BuyLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuylimit), [BuyStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuystop), [SellLimit](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeselllimit) and [SellStop](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesellstop) methods, it requires to specify more essential parameters. Perhaps, you will find it more convenient.

### What Else Should Be Solved?

So, we have answered two out of three questions. You have chosen the symbol for your strategy and we have shown you how to code buy and sell operations, as well as pending orders in a trading robot easily. But Trade Classes section has some more useful tools for MQL5 developers:

- [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) \- for working with orders;

- [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) \- for working with executed orders in trading history;

- [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) \- for working with positions;

- [CDealInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo) \- for working with deals;
- [CTerminalInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo) \- to receive data on the terminal (this one is very interesting).


With these classes, you can focus your attention on the trading side of your strategy minimizing all technical issues. Besides, CTrade class can be used to examine trade requests. After some practice you will be able to use it to create your custom classes with necessary logics of handling trade requests execution results.

The last question is how to receive trading signals and how to code that in MQL5. Most newcomers in algo trading start from studying simple standard trading systems, for example, the ones based on moving averages' crossing. To do this, you should first learn to work with technical indicators creating and using them in your trading robot.

We recommend that you read the articles from [Indicators](https://www.mql5.com/en/articles/indicators) and [Examples->Indicators](https://www.mql5.com/en/articles/examples_indicators) sections beginning from the earliest ones. That will allow you to move from the most simple to the most complex matters. If you want to quickly receive an idea about how to use indicators, see [MQL5 for Newbies: Guide to Using Technical Indicators in Expert Advisors](https://www.mql5.com/en/articles/31).

![Trade Operations in MQL5](https://c.mql5.com/2/33/easy_for_article.jpg)

### Make Complicated Things Simple

In any undertaking the first difficulties gradually turn into the most simple issues you have to deal with. The methods of trading robots' development offered here are meant mainly for newcomers though many experienced developers may also find something new and useful.

MQL5 language provides not only limitless opportunities for algo trading but also allows everyone to implement them in the most simple and fast way. Use [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) from the Standard Library to save time for more important things, for example, for searching the answer to the eternal question of all traders - what is a trend and how can it be found in real time.

Soon you will see that developing a trading robot in MQL5 is much easier than learning a foreign language or following a trend!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/481](https://www.mql5.com/ru/articles/481)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/481.zip "Download all attachments in the single ZIP archive")

[demo\_caccountinfo.mq5](https://www.mql5.com/en/articles/download/481/demo_caccountinfo.mq5 "Download demo_caccountinfo.mq5")(3.03 KB)

[demo\_ctrade.mq5](https://www.mql5.com/en/articles/download/481/demo_ctrade.mq5 "Download demo_ctrade.mq5")(1.98 KB)

[ctrade\_sample\_ea.mq5](https://www.mql5.com/en/articles/download/481/ctrade_sample_ea.mq5 "Download ctrade_sample_ea.mq5")(17.97 KB)

[demo\_csymbolinfo.mq5](https://www.mql5.com/en/articles/download/481/demo_csymbolinfo.mq5 "Download demo_csymbolinfo.mq5")(3.12 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/7552)**
(39)


![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
16 Sep 2020 at 11:42

Good day to all!

I continue to study the [MQL5 programming language](https://www.mql5.com/en/docs "MQL5 Programming Language Reference"). I have searched almost the whole site in search of information that is useful for me. Most of the information found on the site is intended for people who already have a basic understanding of programming.

And here! I found one more **GREAT** article, which helped me to understand a lot of things and refine my EA! It's a pity that the author did not continue writing this series of articles and limited himself to 2012 only. But all the same, I express **GREAT respect to** this man and say to him the same **GREAT** THANK YOU on behalf of all beginners!

With respect, Vladimir.

![Keith Watford](https://c.mql5.com/avatar/avatar_na2.png)

**[Keith Watford](https://www.mql5.com/en/users/forexample)**
\|
22 Jan 2021 at 13:20

Comments that do not relate to this topic, have been moved to " [Off Topic Posts](https://www.mql5.com/en/forum/339471)".

![Huong Huong](https://c.mql5.com/avatar/2021/2/6027E595-0246.jpg)

**[Huong Huong](https://www.mql5.com/en/users/huonghuong)**
\|
15 Feb 2021 at 20:24

**pdev:**

Hi, Thanks for this very helpful post and please help me resolve this. I am new to MT5 and am learning to create EAs so I copied sample code to execute Ctrade.Buy but the backtest failed. Here's more info:

1) Account: Its a live account with base currency as NZD

2) MetaEditor settings for backtest:

3) Code: Copied from https://www.mql5.com/en/articles/481:

//+------------------------------------------------------------------+

//\|                                                         demo.mq5 \|

//\|                        Copyright 2017, MetaQuotes Software Corp. \|

//\|                                             https://www.mql5.com \|

//+------------------------------------------------------------------+

#property copyright "Copyright 2017, MetaQuotes Software Corp."

#property link      "https://www.mql5.com"

#property version   "1.00"

#include<Trade\\Trade.mqh>

//\-\-\- object for performing trade operations

CTrade  trade;

//+------------------------------------------------------------------+

//\| Expert initialization function                                   \|

//+------------------------------------------------------------------+

int OnInit()

{

//\-\-\- set MagicNumber for your orders identification

int MagicNumber=123456;

trade.SetExpertMagicNumber(MagicNumber);

//\-\-\- set available slippage in points when buying/selling

int deviation=10;

trade.SetDeviationInPoints(deviation);

//\-\-\- order execution mode

trade.SetTypeFilling(ORDER\_FILLING\_RETURN);

//\-\-\- logging mode: it would be better not to declare this method at all, the class will set the best mode on its own

trade.LogLevel(1);

//\-\-\- what function is to be used for trading: true - OrderSendAsync(), false - OrderSend()

trade.SetAsyncMode(true);

//---

return(0);

}

//+------------------------------------------------------------------+

//\| Expert deinitialization function                                 \|

//+------------------------------------------------------------------+

void OnDeinit(const int reason)

{

//---

}

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

void OnTick()

{

BuySample1();

}

//\-\-\- Buy sample

//+------------------------------------------------------------------+

//\|  Buying a specified volume at the current symbol                 \|

//+------------------------------------------------------------------+

void BuySample1()

{

//\-\-\- 1\. example of buying at the current symbol

if(!trade.Buy(0.1))

     {

      //\-\-\- failure message

      Print("Buy() method failed. Return code=",trade.ResultRetcode(),

            ". Code description: ",trade.ResultRetcodeDescription());

     }

else

     {

      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),

            " (",trade.ResultRetcodeDescription(),")");

     }

//---

}

4) Error log (Please note that I am testing only on EUR/USD):

GJ 0 19:36:44.410 127.0.0.1 login (build 1730)

HH 0 19:36:44.420 Network 38520 bytes of account info loaded

JO 0 19:36:44.420 Network 1482 bytes of tester parameters loaded

QE 0 19:36:44.420 Network 188 bytes of input parameters loaded

FR 0 19:36:44.421 Network 443 bytes of symbols list loaded

IF 0 19:36:44.421 Tester expert file added: Experts\\demo.ex5. 46684 bytes loaded

QH 0 19:36:44.433 Tester initial deposit 10000.00 NZD, leverage 1:100

JN 0 19:36:44.437 Tester successfully initialized

ES 0 19:36:44.437 Network 46 Kb of total initialization data received

PP 0 19:36:44.437 Tester Intel Core i7-4510U  @ 2.00GHz, 8103 MB

RJ 0 19:36:44.799 Symbols EURUSD: symbol to be synchronized

HR 0 19:36:44.800 Symbols EURUSD: symbol synchronized, 3624 bytes of symbol info received

NJ 0 19:36:44.800 History EURUSD: history synchronization started

GO 0 19:36:44.856 History EURUSD: load 27 bytes of history data to synchronize in 0:00:00.000

RQ 0 19:36:44.856 History EURUSD: history synchronized from 2012.01.01 to 2017.11.15

EF 0 19:36:44.993 History EURUSD,Daily: history cache allocated for 1010 bars and contains 312 bars from 2014.01.01 00:00 to 2014.12.31 00:00

ND 0 19:36:44.993 History EURUSD,Daily: history begins from 2014.01.01 00:00

OL 0 19:36:44.996 Tester EURUSD,Daily (HalifaxPlus-Live): every tick generating

GN 0 19:36:44.996 Tester EURUSD,Daily: testing of Experts\\demo.ex5 from 2015.01.01 00:00 to 2017.11.15 00:00 started

CK 0 19:36:56.288 Symbols NZDUSD: symbol to be synchronized

IS 0 19:36:56.288 Symbols NZDUSD: symbol synchronized, 3624 bytes of symbol info received

JL 0 19:36:56.288 History NZDUSD: history synchronization started

HJ 0 19:36:56.575 History NZDUSD: load 14 Kb of history data to synchronize in 0:00:00.078

LS 0 19:36:56.575 History NZDUSD: history synchronized from 2013.01.01 to 2017.11.15

CO 0 19:36:56.579 Symbols EURNZD: symbol to be synchronized

OJ 0 19:36:56.580 Symbols EURNZD: symbol synchronized, 3624 bytes of symbol info received

DL 0 19:36:56.580 History EURNZD: history synchronization started

MK 0 19:36:56.656 History EURNZD: load 27 bytes of history data to synchronize in 0:00:00.000

OD 0 19:36:56.656 History EURNZD: history synchronized from 2013.01.01 to 2017.11.15

IN 0 19:36:56.665 Trade 2015.01.02 03:00:00   market buy 0.10 EURUSD (1.20538 / 1.20549 / 1.20538)

PE 0 19:36:56.665 Trades 2015.01.02 03:00:00   deal #2 buy 0.10 EURUSD at 1.20549 done ( [based on order](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties "MQL5 documentation: Deal Properties") #2)

FH 0 19:36:56.666 Trade 2015.01.02 03:00:00   deal performed \[#2 buy 0.10 EURUSD at 1.20549\]

OG 0 19:36:56.666 Trade 2015.01.02 03:00:00   order performed buy 0.10 at 1.20549 \[#2 buy 0.10 EURUSD at 1.20549\]

**FO 0 19:36:56.670 demo (EURUSD,D1) 2015.01.02 03:00:00   Buy() method executed successfully. Return code=10009 (done at 1.20549)**

**NM 2 19:37:15.823 History NZDUSD 2016.09.21 23:01:00: corrupted history detected (s:-73370, o:73433, h:+48, l:-123, c:-117 -- tv:63, rv:11250111)**

**JF 2 19:37:15.823 History NZDUSD 2016.09.21, bad container found, must be resynchronized**

**LQ 2 19:37:16.106 Tester history error 9 in undefined function**

**OH 2 19:37:16.106 Tester stopped on 0% of testing interval with error '20 NZDUSD'**

Please tell me whats wrong and how do I resolve this?

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
6 May 2021 at 10:24

**pdev:**

Hi, Thanks for this very helpful post and please help me resolve this. I am new to MT5 and am learning to create EAs so I copied sample code to execute Ctrade.Buy but the backtest failed. Here's more info:

1) Account: Its a live account with base currency as NZD

2) MetaEditor settings for backtest:

3) Code: Copied from https://www.mql5.com/en/articles/481:

//+------------------------------------------------------------------+

//\|                                                         demo.mq5 \|

//\|                        Copyright 2017, MetaQuotes Software Corp. \|

//\|                                             https://www.mql5.com \|

//+------------------------------------------------------------------+

#property copyright "Copyright 2017, MetaQuotes Software Corp."

#property link      "https://www.mql5.com"

#property version   "1.00"

#include<Trade\\Trade.mqh>

//\-\-\- object for performing trade operations

CTrade  trade;

//+------------------------------------------------------------------+

//\| Expert initialization function                                   \|

//+------------------------------------------------------------------+

int OnInit()

{

//\-\-\- set MagicNumber for your orders identification

int MagicNumber=123456;

trade.SetExpertMagicNumber(MagicNumber);

//\-\-\- set available slippage in points when buying/selling

int deviation=10;

trade.SetDeviationInPoints(deviation);

//\-\-\- order execution mode

trade.SetTypeFilling(ORDER\_FILLING\_RETURN);

//\-\-\- logging mode: it would be better not to declare this method at all, the class will set the best mode on its own

trade.LogLevel(1);

//\-\-\- what function is to be used for trading: true - OrderSendAsync(), false - OrderSend()

trade.SetAsyncMode(true);

//---

return(0);

}

//+------------------------------------------------------------------+

//\| Expert deinitialization function                                 \|

//+------------------------------------------------------------------+

void OnDeinit(const int reason)

{

//---

}

//+------------------------------------------------------------------+

//\| Expert tick function                                             \|

//+------------------------------------------------------------------+

void OnTick()

{

BuySample1();

}

//\-\-\- Buy sample

//+------------------------------------------------------------------+

//\|  Buying a specified volume at the current symbol                 \|

//+------------------------------------------------------------------+

void BuySample1()

{

//\-\-\- 1\. example of buying at the current symbol

if(!trade.Buy(0.1))

     {

      //\-\-\- failure message

      Print("Buy() method failed. Return code=",trade.ResultRetcode(),

            ". Code description: ",trade.ResultRetcodeDescription());

     }

else

     {

      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),

            " (",trade.ResultRetcodeDescription(),")");

     }

//---

}

4) Error log (Please note that I am testing only on EUR/USD):

GJ 0 19:36:44.410 127.0.0.1 login (build 1730)

HH 0 19:36:44.420 Network 38520 bytes of account info loaded

JO 0 19:36:44.420 Network 1482 bytes of tester parameters loaded

QE 0 19:36:44.420 Network 188 bytes of input parameters loaded

FR 0 19:36:44.421 Network 443 bytes of symbols list loaded

IF 0 19:36:44.421 Tester expert file added: Experts\\demo.ex5. 46684 bytes loaded

QH 0 19:36:44.433 Tester initial deposit 10000.00 NZD, leverage 1:100

JN 0 19:36:44.437 Tester successfully initialized

ES 0 19:36:44.437 Network 46 Kb of total initialization data received

PP 0 19:36:44.437 Tester Intel Core i7-4510U  @ 2.00GHz, 8103 MB

RJ 0 19:36:44.799 Symbols EURUSD: symbol to be synchronized

HR 0 19:36:44.800 Symbols EURUSD: symbol synchronized, 3624 bytes of symbol info received

NJ 0 19:36:44.800 History EURUSD: history synchronization started

GO 0 19:36:44.856 History EURUSD: load 27 bytes of history data to synchronize in 0:00:00.000

RQ 0 19:36:44.856 History EURUSD: history synchronized from 2012.01.01 to 2017.11.15

EF 0 19:36:44.993 History EURUSD,Daily: history cache allocated for 1010 bars and contains 312 bars from 2014.01.01 00:00 to 2014.12.31 00:00

ND 0 19:36:44.993 History EURUSD,Daily: history begins from 2014.01.01 00:00

OL 0 19:36:44.996 Tester EURUSD,Daily (HalifaxPlus-Live): every tick generating

GN 0 19:36:44.996 Tester EURUSD,Daily: testing of Experts\\demo.ex5 from 2015.01.01 00:00 to 2017.11.15 00:00 started

CK 0 19:36:56.288 Symbols NZDUSD: symbol to be synchronized

IS 0 19:36:56.288 Symbols NZDUSD: symbol synchronized, 3624 bytes of symbol info received

JL 0 19:36:56.288 History NZDUSD: history synchronization started

HJ 0 19:36:56.575 History NZDUSD: load 14 Kb of history data to synchronize in 0:00:00.078

LS 0 19:36:56.575 History NZDUSD: history synchronized from 2013.01.01 to 2017.11.15

CO 0 19:36:56.579 Symbols EURNZD: symbol to be synchronized

OJ 0 19:36:56.580 Symbols EURNZD: symbol synchronized, 3624 bytes of symbol info received

DL 0 19:36:56.580 History EURNZD: history synchronization started

MK 0 19:36:56.656 History EURNZD: load 27 bytes of history data to synchronize in 0:00:00.000

OD 0 19:36:56.656 History EURNZD: history synchronized from 2013.01.01 to 2017.11.15

IN 0 19:36:56.665 Trade 2015.01.02 03:00:00   market buy 0.10 EURUSD (1.20538 / 1.20549 / 1.20538)

PE 0 19:36:56.665 Trades 2015.01.02 03:00:00   deal #2 buy 0.10 EURUSD at 1.20549 done ( [based on order](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties "MQL5 documentation: Deal Properties") #2)

FH 0 19:36:56.666 Trade 2015.01.02 03:00:00   deal performed \[#2 buy 0.10 EURUSD at 1.20549\]

OG 0 19:36:56.666 Trade 2015.01.02 03:00:00   order performed buy 0.10 at 1.20549 \[#2 buy 0.10 EURUSD at 1.20549\]

**FO 0 19:36:56.670 demo (EURUSD,D1) 2015.01.02 03:00:00   Buy() method executed successfully. Return code=10009 (done at 1.20549)**

**NM 2 19:37:15.823 History NZDUSD 2016.09.21 23:01:00: corrupted history detected (s:-73370, o:73433, h:+48, l:-123, c:-117 -- tv:63, rv:11250111)**

**JF 2 19:37:15.823 History NZDUSD 2016.09.21, bad container found, must be resynchronized**

**LQ 2 19:37:16.106 Tester history error 9 in undefined function**

**OH 2 19:37:16.106 Tester stopped on 0% of testing interval with error '20 NZDUSD'**

Please tell me whats wrong and how do I resolve this?

![Dmitrii Troshin](https://c.mql5.com/avatar/2020/3/5E5D0467-98B7.png)

**[Dmitrii Troshin](https://www.mql5.com/en/users/orangetree)**
\|
29 Jun 2021 at 06:21

In the article when opening limit and stop orders everywhere

```
double price=1000*point;
```

I wonder if those who write that the article helped them [insert the code](https://www.mql5.com/en/articles/24#insert-code "Article: MQL5 Community - User's Guide: Inserting Code") from the article?

![MetaQuotes ID in MetaTrader Mobile Terminal](https://c.mql5.com/2/0/MetaQuotes-ID.png)[MetaQuotes ID in MetaTrader Mobile Terminal](https://www.mql5.com/en/articles/476)

Android and iOS powered devices offer us many features we do not even know about. One of these features is push notifications allowing us to receive personal messages, regardless of our phone number or mobile network operator. MetaTrader mobile terminal already can receive such messages right from your trading robot. You should only know MetaQuotes ID of your device. More than 9 000 000 mobile terminals have already received it.

![Interview with Irina Korobeinikova (irishka.rf)](https://c.mql5.com/2/0/zh0ku.png)[Interview with Irina Korobeinikova (irishka.rf)](https://www.mql5.com/en/articles/465)

Having a female member on the MQL5.community is rare. This interview was inspired by a one of a kind case. Irina Korobeinikova (irishka.rf) is a fifteen-year-old programmer from Izhevsk. She is currently the only girl who actively participates in the "Jobs" service and is featured on the Top Developers list.

![How to purchase a trading robot from the MetaTrader Market and to install it?](https://c.mql5.com/2/0/MQL5_market__1.png)[How to purchase a trading robot from the MetaTrader Market and to install it?](https://www.mql5.com/en/articles/498)

A product from the MetaTrader Market can be purchased on the MQL5.com website or straight from the MetaTrader 4 and MetaTrader 5 trading platforms. Choose a desired product that suits your trading style, pay for it using your preferred payment method, and activate the product.

![Application of the Eigen-Coordinates Method to Structural Analysis of Nonextensive Statistical Distributions](https://c.mql5.com/2/0/Eigencoordinates_Nonextensive_Statistical_Distributions_MQL5.png)[Application of the Eigen-Coordinates Method to Structural Analysis of Nonextensive Statistical Distributions](https://www.mql5.com/en/articles/412)

The major problem of applied statistics is the problem of accepting statistical hypotheses. It was long considered impossible to be solved. The situation has changed with the emergence of the eigen-coordinates method. It is a fine and powerful tool for a structural study of a signal allowing to see more than what is possible using methods of modern applied statistics. The article focuses on practical use of this method and sets forth programs in MQL5. It also deals with the problem of function identification using as an example the distribution introduced by Hilhorst and Schehr.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/481&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069378139227882515)

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