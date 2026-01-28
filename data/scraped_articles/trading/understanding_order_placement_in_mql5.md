---
title: Understanding order placement in MQL5
url: https://www.mql5.com/en/articles/13229
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:06:04.146450
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qdgegfloiklvdfqomamyllwgcddnjaxv&ssn=1769180762750823435&ssn_dr=0&ssn_sr=0&fv_date=1769180762&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F13229&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Understanding%20order%20placement%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918076286384044&fz_uniq=5069084754306859098&sv=2552)

MetaTrader 5 / Trading


### Introduction

In any trading system, we need to deal with orders and their operations such as opening positions, placing stop-loss and profit-taking, and modifying orders. Therefore, it is very important to understand how to handle order operations in mql5 when creating a trading system for MetaTrader5. The objective of this article is to provide you a simple guidance for most of the order and position operations to be able to deal with everything about this topic effectively. We will cover in this article the following topics:

- [Orders, positions, and deals terms](https://www.mql5.com/en/articles/13229#terms)
- [OrderSend()](https://www.mql5.com/en/articles/13229#ordersend)
- [OrderSend() application](https://www.mql5.com/en/articles/13229#application1)
- [CTrade class](https://www.mql5.com/en/articles/13229#class)
- [CTrade class application](https://www.mql5.com/en/articles/13229#application2)
- [Conclusion](https://www.mql5.com/en/articles/13229#conclusion)

I hope that you will find this article useful and valuable to develop your MetaTrader5 trading system smoothly regarding order placement. All applications in this article, you must test them before using to make sure that they are profitable or suitable for your trading because the main purpose of mentioning them is only to give an example to create a trading system using two different methods regarding working with order, deal, and position operations.

Disclaimer: All information is provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Orders, Positions, and Deals terms

In this part, we will talk about important terms to understand how to deal with orders effectively. We'll understand the differences between three related terms to orders in the MetaTrader 5. These terms are order, deal, and position, we can consider these terms as steps to execute the trade.

Order: is a request received by the trading server to open a buy or sell trade with a specific lot or volume at a specific price. There are two types of orders, the market order and the pending order.

- Market order: an order that can be executed immediately at the current market price.
- Pending order: an order that executes the trade at predetermined conditions concerning the price to execute the trade at its level and the time to execute the trade.

These pending orders can be one of the following:

- Buy stop: Place a buy pending order at a specific price that is above the current price in the market.
- Buy limit: Place a buy pending order at a specific price that is below the current price in the market.
- Sell stop: Place a sell pending order at a specific price that is below the current price in the market.
- Sell limit: Place a sell pending order at a specific price that is above the current price in the market.

Once the order placed, regardless of whether it is a market or pending order, it can be found in the Trade tab of the Toolbox in the MetaTrader 5. The following is an example:

![1- trade tab](https://c.mql5.com/2/57/1-_trade_tab.png)

When the order is closed or canceled without execution, we can find them in the History tab of the Toolbox.

![2- history tab](https://c.mql5.com/2/57/2-_history_tab__1.png)

When we want to modify a current position by MQL5, we need to deal with these orders the same as we will see later when handling order modifying.

Deal: It is the result when the trade order is executed or filled. We can find it as in and out actions based on the execution of the trade. Let’s say that there is a buy order of one lot executed. After that, we closed a partial of the position 0.5 lots, then closed the remaining 0.5 lots. So, deals will be the same as the following:

- 1 buy in
- 0.5 sell out
- 0.5 sell out

We can find these deals in the History tab of the Toolbox of the MetaTrader 5 by appearing deals data by right-clicking and choosing them.

![3- deals](https://c.mql5.com/2/57/3-_deals__1.png)

Position: It is the net of deals of long or short based on buying or selling the financial asset. We can find it in the Trade tab as an active trade or the History tab if we choose positions to appear.

![4- positions](https://c.mql5.com/2/57/4-_positions__1.png)

It is good to mention that the executable price of the buy order is the Ask price, and the executable price when closing is the Bid price. On the other hand, the executable price of the sell order is the Bid price and the executable price when closing is the Ask price.

OrderSend()

After understanding important terms about trade execution in MetaTrader 5, we need to learn how we should execute orders automatically in MQL5. Firstly, you can check about [Trade Functions](https://www.mql5.com/en/docs/trading) in the MQL5 reference for more information about functions intended for managing activities.

We will discuss the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function, which can be used to execute trade operations by sending requests to a trade server. In other words, it can be used to place, modify, and close orders.

The following is the format of this function:

```
bool  OrderSend(
   MqlTradeRequest&  request,
   MqlTradeResult&   result
   );
```

As we can see the function has two parameters:

- MqlTradeRequest structure: It contains the order parameters and all the necessary fields or variables to perform trade deals. Objects passed by reference as per the ampersands. The [TradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) is the method for interaction between the client terminal and the trade server to execute orders.
- MqlTradeResult structure: The [TradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) returns the results of the order request. Objects passed by reference also.

**The MqlTradeRequest structure:**

The structure is a set of related data of different types. We can find the MqlTradeRequest structure definition the same as the following:

```
struct MqlTradeRequest
  {
   ENUM_TRADE_REQUEST_ACTIONS    action;           // Trade operation type
   ulong                         magic;            // Expert Advisor ID (magic number)
   ulong                         order;            // Order ticket
   string                        symbol;           // Trade symbol
   double                        volume;           // Requested volume for a deal in lots
   double                        price;            // Price
   double                        stoplimit;        // StopLimit level of the order
   double                        sl;               // Stop Loss level of the order
   double                        tp;               // Take Profit level of the order
   ulong                         deviation;        // Maximal possible deviation from the requested price
   ENUM_ORDER_TYPE               type;             // Order type
   ENUM_ORDER_TYPE_FILLING       type_filling;     // Order execution type
   ENUM_ORDER_TYPE_TIME          type_time;        // Order expiration type
   datetime                      expiration;       // Order expiration time (for the orders of ORDER_TIME_SPECIFIED type)
   string                        comment;          // Order comment
   ulong                         position;         // Position ticket
   ulong                         position_by;      // The ticket of an opposite position
  };
```

If we want to declare an MqlTradeRequest object, we can do so through the following line of code:

```
MqlTradeRequest request;
```

Then, we can assign the trade parameters to the variables of the created request object by adding a dot (.) after the object, as shown in the following example:

```
request.symbol = _Symbol;
request.volume = 0.01;
request.type = ORDER_TYPE_BUY;
```

The following is a list of all members or variables of the MqlTradeRequest structure, along with their accepted values for assignment.

| Variable | Description | Accepted value for assignment |
| --- | --- | --- |
| action | It represents the type of trade operation | One of the [ENUM\_TRADE\_REQUEST\_ACTIONS](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) (TRADE\_ACTION\_DEAL, TRADE\_ACTION\_PENDING, TRADE\_ACTION\_SLTP, TRADE\_ACTION\_MODIFY, TRADE\_ACTION\_REMOVE, TRADE\_ACTION\_CLOSE\_BY) |
| magic | It represents the expert advisor ID for identifying orders placed by a certain expert advisor | Any ulong value |
| order | It represents the order ticket. It is required when using TRADE\_ACTION\_MODIFY or TRADE\_ACTION\_REMOVE in the action variable | Any ulong value |
| symbol | The specified symbol or the instrument to trade.the (\_SYMBOL) refers to the current one | Any string symbol |
| volume | It specify trade volume or lots | Any allowed double value |
| price | The opening price | Double value |
| stoplimit | The opening price of limit pending order. The action must be TRADE\_ACTION\_PENDING and the type variable must be ORDER\_TYPE\_BUY\_STOP\_LIMIT or ORDER\_TYPE\_SELL\_STOP\_LIMIT and it is required for stop limit orders | Double valuue |
| sl | The stop loss price of the trade | Double value |
| tp | The take profit price of the trade | Double value |
| deviation | The maximal price deviation in points | ulong value |
| type | The order type | One of the [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) (ORDER\_TYPE\_BUY, ORDER\_TYPE\_SELL, ORDER\_TYPE\_BUY\_STOP, ORDER\_TYPE\_SELL\_STOP, <br>ORDER\_TYPE\_BUY\_LIMIT, ORDER\_TYPE\_SELL\_LIMIT,  ORDER\_TYPE\_BUY\_STOP\_LIMIT, ORDER\_TYPE\_SELL\_STOP\_LIMIT) |
| type\_filling | Order execution type or the filling policy of the order | One of the [ENUM\_ORDER\_TYPE\_FILLING](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling) (ORDER\_FILLING\_FOK, ORDER\_FILLING\_IOC, ORDER\_FILLING\_BOC, ORDER\_FILLING\_RETURN) |
| type\_time | The type of the pending order expiration | One of the [ENUM\_ORDER\_TYPE\_TIME](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_time) (ORDER\_TIME\_GTC, ORDER\_TIME\_DAY, ORDER\_TIME\_SPECIFIED, ORDER\_TIME\_SPECIFIED\_DAY) |
| expiration | The pending order expiration time. It is required when the type\_time is ORDER\_TIME\_SPECIFIED | datetime value |
| comment | The order comment | string value |
| position | The ticket of a position. It is required when a position is modified or closed to identify it | ulong value |
| position\_by | The ticket of an opposite position. It is used when the position is closed by an opposite open one in the opposite direction for the same symbol | ulong value |

Now, we will mention some important actions that we need to use when placing orders in MQL5.

- Market order
- Adding stop-loss and take-profit
- Pending order
- Pending order modifying
- Remove pending order

**Market Order:**

In this type of action, we need to place a market order which means the order will be placed at the current market price, we will use the ( [TRADE\_ACTION\_DEAL](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_deal)) action. If the order is a buy order, the order will be placed at the current ask price or if the order is a sell order, the order will be placed at the current bid price. Here is how to do that:

After declaring MqlTradeRequest and MqlTradeResult objects, we can assign the following values to following variables:

```
request.action = TRADE_ACTION_DEAL;                    &nbsthe p;   //market order palcement
request.type = ORDER_TYPE_BUY;                             //type of order is buy
request.symbol = _Symbol;                                  //applied for the cuurrent symbol
request.volume = 0.1;                                      //the lot size
request.type_filling = ORDER_FILLING_FOK;                  //the filling policy fill or kill
request.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);      //the price is the current ask for buy
request.sl = 0;                                            //stop loss is 0
request.tp = 0;                                            //take profit is 0
request.deviation = 50;                                    //slippage is 50
OrderSend(request, result);                                //calling the OrderSend function
```

As we can see in the previous code, we did not add stop loss and take profit values. We can add their values to the code along with the other variables, or we can add them through another action, as we will see in the next point.

**Adding stop-loss and take-profit:**

When we need to add stop-loss and take-profit, we assign the [TRADE\_ACTION\_SLTP](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_sltp) to the variable action as shown in the following example.

```
request.action = TRADE_ACTION_SLTP;          //adding sl and tp
request.symbol = _Symbol;                    //applied for the cuurrent symbol
request.sl = 1.07000;                        //sl price
request.sl = 1.09000;                        //tp price
OrderSend(request, result);                  //calling the OrderSend function
```

**Pending order:**

If we want to place a pending order, we can use another action ( [TRADE\_ACTION\_PENDING](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_pending)). Then, we determine the type of order. We can set the time if we need an expiration time for the pending order.

```
request.action = TRADE_ACTION_PENDING;          //pending order placement
request.type = ORDER_TYPE_BUY_STOP;             //type of order is buy stop
request.symbol = _Symbol;                       //applied for the cuurrent symbol
request.volume = 0.1;                           //the lot size
request.price = 1.07000;                        //opening price
request.sl = 1.06950;                           //stop loss
request.tp = 1.07100;                           //take profit
request.type_time = ORDER_TIME_SPECIFIED;       //to set an expiration time
request.expiration = D'2023.08.31 00.00';       //expiration time - datetime constant
request.type_filling = ORDER_FILLING_FOK;       //the filling policy fill or kill
request.stoplimit = 0;                          //for stoplimit order only
OrderSend(request, result);                     //calling the OrderSend function
```

**Pending order modifying:**

If we need to modify the pending order, we need to get the order ticket number of the pending order that we need to modify. We can use the [OrderGetTicket](https://www.mql5.com/en/docs/trading/ordergetticket) function to get the order ticket. This function returns the ticket of the corresponding order, which we can use to select the order and work with it.

```
ulong  OrderGetTicket(
   int  index      // Number in the list of orders
   );
```

Let's consider that we have a variable named (ticket) that holds the order ticket number. We can use this variable to assign the order ticket number to the order variable of the request object. We can then modify the order using the action ( [TRADE\_ACTION\_MODIFY](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_modify)), the same as the following example.

```
request.action = TRADE_ACTION_MODIFY;           //pending order modyfying
request.order = ticket;                         //ticket variable that holds the pending order ticket to modify
request.price = 1.07050;                        //new opening price
request.sl = 1.07000;                           //new stop loss
request.tp = 1.07150;                           //new take profit
request.type_time = ORDER_TIME_SPECIFIED;       //to set an expiration time
request.expiration = D'2023.09.01 00.00';       //new expiration time - datetime constant
OrderSend(request, result);                     //calling the OrderSend function
```

**Remove Pending Order:**

If we need to remove the pending order, we can do that using the action of ( [TRADE\_ACTION\_REMOVE](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_remove)). We also need the ticket number of the pending order that we need to remove. We can use the variable ticket assuming that it holds the needed ticket number.

```
request.action = TRADE_ACTION_REMOVE;           //pending order remove
request.order = ticket;                         //ticket variable that holds the pending order ticket to remove
OrderSend(request, result);                     //calling the OrderSend function
```

**The MqlTradeResult structure:**

The MqlTradeResult structure returns the result of whether or not the order was successful once an order has been placed by the OrderSend() function. It contains trade information from the trade server, such as the ticket number, the volume, and the price.

The following is the definition of the MqlTradeResult structure:

```
struct MqlTradeResult
  {
   uint     retcode;          // Operation return code
   ulong    deal;             // Deal ticket, if it is performed
   ulong    order;            // Order ticket, if it is placed
   double   volume;           // Deal volume, confirmed by broker
   double   price;            // Deal price, confirmed by broker
   double   bid;              // Current Bid price
   double   ask;              // Current Ask price
   string   comment;          // Broker comment to operation (by default it is filled by description of trade server return code)
   uint     request_id;       // Request ID set by the terminal during the dispatch
   int      retcode_external; // Return code of an external trading system
  };
```

We can declare an object named result to pass it as a second parameter after the first one (request) to the OrderSend() function call.

```
MqlTradeResult result;
```

As we can see in the variables of the MqlTradeResult structure, the retcode variable is very important because it returns the code from the trade server to indicate whether or not the request was successful.

If the trade is not placed, the return code indicates an error state. You can check the list of [return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes) from the MQL5 reference for more information. It is crucial to include a code in our trading system to report if there is an error returned or not, such as the following example:

```
   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
     {
      Print("Trade Placed Successfully");
     }
   else
     {
      Print("Trade Not Placed, Error ", result.retcode);
     }
```

As we can see in the previous code, it will print a message with the result after the request and fill in the result variables. If the trade was placed, it'll send us a message with that, or if there is a problem, it will print a message that there was an error and return the error code to help navigate the issue.

### OrderSend() Application

We need to create a simple trading system that can execute trades by using the OrderSend() function. The trading system that we need to create is a simple moving average crossover and the action is placing a market order only.

The objective here is to understand the differences when creating the same trading system using the OrderSend() and the CTrade class. The following are steps to create this MA crossover using the OrderSend() function.

In the global scope create two integer variables (simpleMA, barsTotal) without assignment and they will be assigned later in the OnInit part.

```
int simpleMA;
int barsTotal;
```

The simpleMA will be assigned to the iMA function that returns the handle of the moving average technical indicator. Its parameters are:

- symbol: To specify the symbol name, the \_Symbol refers to the current instrument.
- period: To specify the time frame, we will use PERIOD\_H1 which refers to the one-hour time frame.
- ma\_period: the period of the moving average, we will 50.
- ma\_shift: to specify the needed horizontal shift.
- ma\_method: To specify the moving average type, we will specify the simple one.
- applied\_price: To specify the type of price that will be used in the MA calculation, we will set the closing price.

The barsTotal will be assigned to the iBars function that returns the number of bars. Its parameters are:

- symbol: The symbol name
- timeframe: the time frame period

```
   simpleMA = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_SMA, PRICE_CLOSE);
   barsTotal=iBars(_Symbol,PERIOD_H1);
```

In the OnTick part, we will create two arrays one for price using the MqlRates type which stores information about price, volume, and spread and the other for the moving average using double variable type

```
   MqlRates priceArray[];
   double mySMAArray[];
```

Defining the ask and bid prices

```
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
```

Creating two objects for the OrderSend() function one is for request using the MqlTradeReuest and the other for the result using the MqlTradeResult, then resetting the request variable by reference

```
ZeroMemory(request);
```

Setting the AS\_SERIES flag to these two created arrays priceArray and mySMAArray by using the ArraySetAsSeries function. Its parameters:

- array\[\]: To specify the needed array
- flag: Array indexing direction.

```
   ArraySetAsSeries(priceArray,true);
   ArraySetAsSeries(mySMAArray,true);
```

Getting historical data of MqlRates by using the CopyRates function. Its parameters are:

- symbol: the symbol name or the \_Symbol for the current symbol or instrument.
- time frame: time frame period or \_period for the current time frame.
- start position: the position to start from.
- count: data count to copy.
- rates\_array: the target array to copy.

```
int Data=CopyRates(_Symbol,_Period,0,3,priceArray);
```

Getting the indicator data buffer by using the CopyBuffer function. Its parameters:

- indicator\_handle: To specify the indicator handle, the MA.
- buffer\_num: To specify the indicator buffer number.
- start\_pos: the starting position to count from.
- count: amount to copy.
- buffer\[\]: the target array.

```
CopyBuffer(simpleMA,0,0,3,mySMAArray);
```

Defining last, previous closing prices, last, and previous SMA values

```
   double lastClose=(priceArray[1].close);
   double prevClose=(priceArray[2].close);
   double SMAVal = NormalizeDouble(mySMAArray[1],_Digits);
   double prevSMAVal = NormalizeDouble(mySMAArray[2],_Digits);
```

Create an integer bars variable and assign it to the iBars

```
int bars=iBars(_Symbol,PERIOD_H1);
```

Checking bars, if the barsTotal is not equal to the bars variable

```
if(barsTotal != bars)
```

Updating barsTotal value to equal to the bars if they are not equal to each other

```
barsTotal=bars;
```

Checking the condition of the strategy to open a trade, if the last close is greater than the last SMA value after the previous close was less than the previous SMA value

```
if(prevClose<prevSMAVal && lastClose>SMAVal)
```

Opening a market buy order using suitable variables related to the MqlTradeRequest function

```
         request.action = TRADE_ACTION_DEAL;
         request.type = ORDER_TYPE_BUY;
         request.symbol = _Symbol;
         request.volume = 0.1;
         request.type_filling = ORDER_FILLING_FOK;
         request.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
         request.sl = Ask-(500*_Point);
         request.tp = Ask+(1000*_Point);
         request.deviation = 50;
         OrderSend(request, result);
```

Checking the condition of the strategy to open a sell trade, if the last close is less than the last SMA value after the previous close was greater than the previous SMA value

```
if(prevClose>prevSMAVal && lastClose<SMAVal)
```

Opening a market sell order using suitable variables related to the MqlTradeRequest function

```
         request.type = ORDER_TYPE_SELL;
         request.symbol = _Symbol;
         request.volume = 0.1;
         request.type_filling = ORDER_FILLING_FOK;
         request.price = SymbolInfoDouble(_Symbol,SYMBOL_BID);
         request.sl = Bid+(500*_Point);
         request.tp = Bid-(1000*_Point);
         request.deviation = 50;
         OrderSend(request, result);
```

The following is the full code in one block to create this type of trading system using the OrderSend() function

```
//+------------------------------------------------------------------+
//|                                     OrderSend_Trading_system.mq5 |
//+------------------------------------------------------------------+
int simpleMA;
int barsTotal;
int OnInit()
  {
   simpleMA = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_SMA, PRICE_CLOSE);
   barsTotal=iBars(_Symbol,PERIOD_H1);
   return(INIT_SUCCEEDED);
  }
void OnTick()
  {
   MqlRates priceArray[];
   double mySMAArray[];
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ArraySetAsSeries(priceArray,true);
   ArraySetAsSeries(mySMAArray,true);
   int Data=CopyRates(_Symbol,_Period,0,3,priceArray);
   CopyBuffer(simpleMA,0,0,3,mySMAArray);
   double lastClose=(priceArray[1].close);
   double prevClose=(priceArray[2].close);
   double SMAVal = NormalizeDouble(mySMAArray[1],_Digits);
   double prevSMAVal = NormalizeDouble(mySMAArray[2],_Digits);
   int bars=iBars(_Symbol,PERIOD_H1);
   if(barsTotal != bars)
     {
      barsTotal=bars;
      if(prevClose<prevSMAVal && lastClose>SMAVal)
        {
         request.action = TRADE_ACTION_DEAL;
         request.type = ORDER_TYPE_BUY;
         request.symbol = _Symbol;
         request.volume = 0.1;
         request.type_filling = ORDER_FILLING_FOK;
         request.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
         request.sl = Ask-(500*_Point);
         request.tp = Ask+(1000*_Point);
         request.deviation = 50;
         OrderSend(request, result);
        }
      if(prevClose>prevSMAVal && lastClose<SMAVal)
        {
         request.action = TRADE_ACTION_DEAL;
         request.type = ORDER_TYPE_SELL;
         request.symbol = _Symbol;
         request.volume = 0.1;
         request.type_filling = ORDER_FILLING_FOK;
         request.price = SymbolInfoDouble(_Symbol,SYMBOL_BID);
         request.sl = Bid+(500*_Point);
         request.tp = Bid-(1000*_Point);
         request.deviation = 50;
         OrderSend(request, result);
        }
     }
  }
```

### CTrade class

After learning how we can place orders by OrderSend() function, we have an easy method to access trade functions. We can use the CTrade ready-made class provided by MQL5, or we can create our own class for more personal preferences. Now, we will mention how we can use the CTrade ready-made class in the MQL5. You can check all about [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class from MQL5 reference for more information.

First, we can find the include file of the CTrade class in the Trade folder in the include folder in the MetaTrader 5 installation files. All we need is to include this file in the expert advisor for calling and using all trade functions the same as the following:

```
#include <Trade\Trade.mqh>
```

Create an object from the CTrade class

```
CTrade trade;
```

After that we'll be able to use all trade functions in the CTrade class by using trade before the dot(.) and the desired function. We can use all functions of operations with orders, operations with positions, access to the last request parameters, access to the last request checking results, access to the last request execution results, and others.

We will mention some important functions the same as we did with OrderSend() to understand the differences between the two methods. We will understand how to do the following:

- Market order
- Adding stop-loss and take-profit
- Pending order
- Pending order modifying
- Remove pending order

**Market order:**

After creating our trade object we can use the [PositionOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionopen) function to place a market order, its parameters are:

- symbol: to specify the needed symbol
- order\_type: to specify the order type to open a position
- volume: to specify the lot size
- price: to specify the price of the position opening
- sl: to specify the stop loss price
- tp: to specify the ttake profit price
- comment: to specify a comment or NULL

The following is an example of that:

```
trade.PositionOpen(
   _Symbol,             //to be applied for the current symbol
   ORDER_TYPE_BUY,      //to place buy order
   0.1,                 //lot size or volume
   Ask,                 //opening price of the order - current ask
   Ask-(500*_Point),    //sl
   Ask+(1000*_Point),   //tp
   NULL                 //NULL
);
```

We can also use the additional methods in the CTrade like [Buy](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradebuy), [Sell](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradesell) instead of PositionOpen to open a market order.

**Adding stop-loss and take-profit:**

We can modify position by symbol or ticket number using the [PositionModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradepositionmodify) function. Its parameters are:

- symbol or ticket: to specify the position we need to modify, if modifying by symbol we will specify the symbol name or if modifying by ticket we will specify the ticket number.
- sl: the new stop loss price
- tp: the take profit price

Example for modifying by symbol:

```
trade.PositionModify(
   EURUSD,       //the symbol name
   1.06950,      //the new sl
   1.07100,      //the new tp
);
```

Example for modifying by ticket:

```
trade.PositionModify(
   ticket,       //the ticket variable that holds the needed ticket number to modify
   1.06950,      //the new sl
   1.07100,      //the new tp
);
```

**Pending order:**

If we need to place a pending order using the CTrade class we can use the [OrderOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) function. Its parameters are:

- symbol: the determine the symbol name
- order\_type: to determine the pending order type
- volume: to specify the lot size
- limit\_price: to specify the stop limit price
- price: to specify the execution price of the pending order
- sl: to specify the stop loss
- tp: to specify the take Profit price
- type\_time: to specify the type by expiration
- expiration: to specify the expiration datetime variable
- comment: to specify a comment if needed

The following example is for placing a buy limit pending order using the OrderOpen function

```
         trade.OrderOpen(
            "EURUSD",                 // symbol
            ORDER_TYPE_BUY_LIMIT,     // order type
            0.1,                      // order volume
            0,                        // StopLimit price
            1.07000,                  // execution price
            1.06950,                  // Stop Loss price
            1.07100,                  // Take Profit price
            ORDER_TIME_SPECIFIED,     // type by expiration
            D'2023.08.31 00.00',      // expiration
            ""                        // comment
         );
```

**Pending order modifying:**

If we need to modify the placed pending order we can do that using the CTrade class by the [OrderModify](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeordermodify) function. Its parameters are:

- ticket: to specify the pending order ticket to modify
- price: the new execution price
- sl: the new stop Loss price
- tp: the new take Profit price
- type\_time: to specify the type by expiration
- expiration: to specify the expiration datatime variable
- stoplimit : to determine the limit order price

The following is an example of pending order modifying

```
         trade.OrderModify(
            ticket,                   // ticket number of the pending order to modify
            1.07050,                  // execution price
            1.07000,                  // Stop Loss price
            1.07150,                  // Take Profit price
            ORDER_TIME_SPECIFIED,     // type by expiration
            D'2023.08.31 00.00',      // expiration
            0,                        // StopLimit price
         );
```

**Delete pending order:**

If you need to delete a pending order you can do that by using the [OrderDelete](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderdelete) function and it needs the ticket number of the pending order to be deleted. The following is an example of that function.

```
         trade.OrderDelete(
            ticket,                 // tick number of the pending order to delete
         );
```

### CTrade class Application

We need to create the same trading system that we created using the OrderSend(). We will use the CTrade class to understand the differences in how functions work in both of them.

The following are the only different steps to create this trading system using the CTrade class, and the remaining steps are the same as we did before.

Including the Trade include file using the preprocessor #include

```
#include <Trade\Trade.mqh>
```

Create the trade object from the CTrade class

```
CTrade trade;
```

When the buy condition is met we can use the PositionOpen and the order type will be ORDER\_TYPE\_BUY or the additional buy method the same as the following code

```
trade.Buy(0.1,_Symbol,Ask,Ask-(500*_Point),Ask+(1000*_Point),NULL);
```

When the sell condition is met, we can use the PositionOpen along with the order type ORDER\_TYPE\_SELL or the additional sell method the same as the following code

```
trade.Sell(0.1,_Symbol,Bid,Bid+(500*_Point),Bid-(1000*_Point),NULL);
```

The following is the full code to create the trading system using the CTrade class:

```
//+------------------------------------------------------------------+
//|                                        CTrade_Trading_System.mq5 |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
int simpleMA;
int barsTotal;
CTrade trade;
int OnInit()
  {
   simpleMA = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_SMA, PRICE_CLOSE);
   barsTotal=iBars(_Symbol,PERIOD_H1);
   return(INIT_SUCCEEDED);
  }
void OnTick()
  {
   MqlRates priceArray[];
   double mySMAArray[];
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
   ArraySetAsSeries(priceArray,true);
   ArraySetAsSeries(mySMAArray,true);
   int Data=CopyRates(_Symbol,_Period,0,3,priceArray);
   CopyBuffer(simpleMA,0,0,3,mySMAArray);
   double lastClose=(priceArray[1].close);
   double prevClose=(priceArray[2].close);
   double SMAVal = NormalizeDouble(mySMAArray[1],_Digits);
   double prevSMAVal = NormalizeDouble(mySMAArray[2],_Digits);
   int bars=iBars(_Symbol,PERIOD_H1);
   if(barsTotal != bars)
     {
      barsTotal=bars;
      if(prevClose<prevSMAVal && lastClose>SMAVal)
        {
         trade.Buy(0.1,_Symbol,Ask,Ask-(500*_Point),Ask+(1000*_Point),NULL);
        }
      if(prevClose>prevSMAVal && lastClose<SMAVal)
        {
         trade.Sell(0.1,_Symbol,Bid,Bid+(500*_Point),Bid-(1000*_Point),NULL);
        }
     }
  }
```

### Conclusion

If you have completed reading this article, it is supposed that understand how orders, positions, and deals work in MQL5. In addition to understanding how we can create a trading system smoothly using the two methods of trade operations, the OrderSend method and the CTrade method.

We identified how we place market orders and pending orders, add stop-loss and take-profit, modify pending orders, and remove or delete pending orders using the two methods.

It is supposed that you understood how to apply all of the previous to create an application because we provided two simple applications to create the same moving average crossover trading system using the two methods.

- OpenSend\_Trading\_System
- CTrade\_Trading\_System

The objective of mentioned applications is only to understand the differences between them practically so, take care, you must test them before using them in any live account to make sure they are profitable and suitable for your trading.

I believe that you released how easy it is to use the ready-made CTrade class when working with trade operations because it saves a lot of time and effort. In addition to other features when using classes in programming generally.

If you want to learn more about that and the object-oriented programming in MQL5 you can read my previous article [Understanding MQL5 Object-Oriented Programming (OOP)](https://www.mql5.com/en/articles/12813), I hope that you find it useful. MQL5 made a lot of appreciated work and effort to provide us with tools to develop and create trading software smoothly and easily.

I hope that you found this article useful and got many useful insights to understand and do your work easily after these insights. If you want to read more articles for me, you can check my [publication](https://www.mql5.com/en/users/m.aboud/publications) for many articles about how to create trading systems using the most popular technical indicators and others and I hope that you will find them useful.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13229.zip "Download all attachments in the single ZIP archive")

[OrderSend\_Trading\_system.mq5](https://www.mql5.com/en/articles/download/13229/ordersend_trading_system.mq5 "Download OrderSend_Trading_system.mq5")(2.17 KB)

[CTrade\_Trading\_System.mq5](https://www.mql5.com/en/articles/download/13229/ctrade_trading_system.mq5 "Download CTrade_Trading_System.mq5")(1.45 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/453741)**
(6)


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
24 Oct 2023 at 01:11

**gunther64 [#](https://www.mql5.com/en/forum/453741#comment_50106579):**

hello,

great text, tyvm.

I have found 1 small bug:

in the box under  [TRADE\_ACTION\_SLTP](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions#trade_action_sltp) you have written twice request.sl, the second one should be request.tp, as indicated in the comment at the end of the line.

Best Regards,

Gunther

Hello,

Thanks for your kind comment. You are correct, it is a mistake and it will be considered.

Regards,

![rurubest](https://c.mql5.com/avatar/avatar_na2.png)

**[rurubest](https://www.mql5.com/en/users/rurubest)**
\|
8 Apr 2024 at 12:45

hello!

Using your code I am writing a simple order placement for an instrument!

double price = 94500;

double stopLoss = price - (500 \* \_Point);

double takeProfit = price + (1000 \* \_Point);

ulong ticket = trade.OrderOpen(

"SiM4", // symbol

[ORDER\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type "MQL5 Documentation: Order Properties"), // order type

1.0, // order volume

price, // StopLimit price

stopLoss, // execution price

takeProfit, // Stop Loss price

NULL

);

returns GetLastError() = 0 and the order is not placed and there is no deal

Is it necessary to specify a special access token when placing an order from the broker?

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
8 Apr 2024 at 13:12

**rurubest [#](https://www.mql5.com/ru/forum/460483#comment_52979290):**

returns GetLastError() = 0 and the order is not placed and there is no transaction

Is it necessary to specify a special access token when placing an order from the broker?

Look at the logs and analyse the result of OrderSend execution.


![Ahmad Juniar](https://c.mql5.com/avatar/2023/8/64D6118C-B729.jpg)

**[Ahmad Juniar](https://www.mql5.com/en/users/ahmadjuniar)**
\|
8 Nov 2024 at 10:53

Hello Abdel Maaboud,

thank you for your kindness wrote this tutorial.

This tutorial only give order placement. Is there any code for close position ( [take profit](https://www.mql5.com/en/articles/7113 "Article: Scratching Profits to the Last Pip ") or cut loss) in your article ?

Best Regards,

Ahmad Juniar

![2020_fusaroli.it](https://c.mql5.com/avatar/avatar_na2.png)

**[2020\_fusaroli.it](https://www.mql5.com/en/users/2020_fusaroli.it)**
\|
19 Nov 2024 at 15:25

Thank you sooooo soooo much for your precious and detailed tutorial. You saved me tons of time and research. Thank you again!


![Elastic net regression using coordinate descent in MQL5](https://c.mql5.com/2/58/Elastic_net_regression_using_coordinate_descent_in_MQL5_AVATAR.png)[Elastic net regression using coordinate descent in MQL5](https://www.mql5.com/en/articles/11350)

In this article we explore the practical implementation of elastic net regression to minimize overfitting and at the same time automatically separate useful predictors from those that have little prognostic power.

![Category Theory in MQL5 (Part 19): Naturality Square Induction](https://c.mql5.com/2/58/Category-Theory-p19-avatar.png)[Category Theory in MQL5 (Part 19): Naturality Square Induction](https://www.mql5.com/en/articles/13273)

We continue our look at natural transformations by considering naturality square induction. Slight restraints on multicurrency implementation for experts assembled with the MQL5 wizard mean we are showcasing our data classification abilities with a script. Principle applications considered are price change classification and thus its forecasting.

![Neural networks made easy (Part 37): Sparse Attention](https://c.mql5.com/2/53/Avatar_NN_part_37_Sparse_Attention.png)[Neural networks made easy (Part 37): Sparse Attention](https://www.mql5.com/en/articles/12428)

In the previous article, we discussed relational models which use attention mechanisms in their architecture. One of the specific features of these models is the intensive utilization of computing resources. In this article, we will consider one of the mechanisms for reducing the number of computational operations inside the Self-Attention block. This will increase the general performance of the model.

![Data label for time series  mining(Part 1)：Make a dataset with trend markers through the EA operation chart](https://c.mql5.com/2/57/data-label-for-time-series-mining-avatar.png)[Data label for time series mining(Part 1)：Make a dataset with trend markers through the EA operation chart](https://www.mql5.com/en/articles/13225)

This series of articles introduces several time series labeling methods, which can create data that meets most artificial intelligence models, and targeted data labeling according to needs can make the trained artificial intelligence model more in line with the expected design, improve the accuracy of our model, and even help the model make a qualitative leap!

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/13229&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069084754306859098)

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