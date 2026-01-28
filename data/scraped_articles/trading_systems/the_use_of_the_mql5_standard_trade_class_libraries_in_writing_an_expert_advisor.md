---
title: The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor
url: https://www.mql5.com/en/articles/138
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:52:12.659405
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/138&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062779274164873401)

MetaTrader 5 / Trading systems


### Introduction

The new MQL5 program comes with
a lot of built-in [Standard class libraries](https://www.mql5.com/en/docs/standardlibrary) which are meant to make development
of MQL5 Expert Advisors, Indicators and Scripts as easy as possible for traders
and developers.

These class libraries are available in the \\Include\ folder
located within the MQL5 folder in the MetaTrader 5 client terminal folder. The class libraries
are divided into various categories – [Arrays](https://www.mql5.com/en/docs/standardlibrary/datastructures), [ChartObjects](https://www.mql5.com/en/docs/standardlibrary/chart_object_classes), [Charts](https://www.mql5.com/en/docs/standardlibrary/cchart), [Files](https://www.mql5.com/en/docs/standardlibrary/fileoperations),
[Indicators](https://www.mql5.com/en/docs/standardlibrary/technicalindicators), [Strings](https://www.mql5.com/en/docs/standardlibrary/stringoperations) and [Trade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) classes.

In this article, we will describe in
detail how we can use the built-in [Trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) to write an Expert Advisor.
The Expert Advisor will be based on a strategy that will include closing and
modifying of opened positions when a stipulated condition is met.

If you already have an idea of
what classes are and how they can be used, then you are welcome to another
world of opportunity which the new MQL5 language has to offer.

If, on the other
hand, you are completely new to MQL5; then I suggest you read these two
articles for a start [Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://www.mql5.com/en/articles/100), [Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116) or any other article that will give you an introduction to the new MQL5 language. There are a lot of [articles](https://www.mql5.com/en/articles) that have been written that will give you the required knowledge.

### **1.The Trade Classes**

The trade classes folder
consists of different classes which are meant to make life easier for traders
who which to develop an EA for personal use or for programmers who will not
have to re-invent the wheel when developing their [Expert Advisors](https://www.mql5.com/en/docs/runtime) (EA).

In using a [class](https://www.mql5.com/en/docs/basis/types/classes#class), you do not have to know the
internal workings of the class (that is, how it accomplishes what the developer
says it does), all you need to concentrate on is how the class can be used to
solve your problem. This is why using a built in class library makes things
pretty easy for anyone who wants to use them. In this article we will be
looking at the major classes that will be needed in the course of developing an
Expert Advisor.

In discussing the classes, we will not bother ourselves with
the internal details of the classes, but we will discuss in details what the
class can do and how we can use it to accomplish our mission in developing a
very profitable EA. Let us discuss them one after the other.

**1.1The СAccountInfo**
**Class**

The [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) is a
class that makes it easy for the user to have access to all the account
properties or information for the current opened trade account in the client terminal.

To
understand better, we will look at the major member functions of this class
that we may likely use in our EA. Before we can use a class, we
must first of all create an object of that class, so to use the CAccountInfo
class, we must create an object of the class.

Let’s call it **myaccount**:

```
//--- The AccountInfo Class Object
CAccountInfo myaccount;
```

Remember that to create an
object of a class, you will use the class name followed by the name you wish to
give the object.

We can now use our **myaccount** object to access the public
member functions of the [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo)class.

| Method | Description | Example of use |
| --- | --- | --- |
| **myaccount.Login()** | This<br>function is used when you want to get the account number for the current opened<br>trade in the terminal. | ```<br>// returns account number, example 7770<br>long accountno = myaccount.Login()<br>``` |
| **myaccount.TradeModeDescription()** | This function is used to get the description of the trade mode for the currently active account on the terminal. | ```<br>// returns Demo trading account, <br>// or Real trading account or Contest trading account<br>string  acc_trading_mode = myaccount.TradeModeDescription();<br>``` |
| **myaccount.Leverage()** | This<br>function is used to get the description of the trade mode for the currently<br>active account on the terminal. | ```<br>// returns leverage given to the active account<br>long acc_leverage = myaccount.Leverage(); <br>``` |
| **myaccount.TradeAllowed()** | This<br>function is used to check if trade is allowed on the active account on the<br>terminal. If trade is not allowed, the account cannot trade. | ```<br>if (myaccount.TradeAllowed())<br>{<br>    // trade is allowed<br>}<br>else<br>{<br>  // trade is not allowed<br>}<br>``` |
| **myaccount.TradeExpert()** | This function is used to check if Expert Advisors are allowed to trade for the currently active account in the terminal. | ```<br>if (myaccount.TradeExpert())<br>{<br>   // Expert Advisor trade is allowed<br>}<br>else<br>{<br>   // Expert Advisor trade is not allowed<br>}<br>``` |
| **myaccount.Balance()** | This<br>function gives the account balance for the active account on the terminal. | ```<br>// returns account balance in the deposit currency<br>double acс_balance =  myaccount.Balance(); <br>``` |
| **myaccount.Profit()** | This<br>function is used to obtain the current profit of the active account on the<br>terminal. | ```<br>// returns account profit in deposit currency<br>double acс_profit =  myaccount.Profit();<br>``` |
| **myaccount.FreeMargin()** | This<br>function is used to get the free margin of the active account on the terminal. | ```<br>// returns free margin for active account<br>double acс_free_margin = myaccount.FreeMargin();<br>``` |
| **myaccount.Currency()** | This<br>function is used to get the deposit currency for the active account on the<br>terminal. | ```<br>string acс_currency = myaccount.Currency();<br>``` |
| **myaccount.OrderProfitCheck(const _string_ symbol, _ENUM\_ORDER\_TYPE_  trade\_operation, _double_ volume, _double_ price\_open, _double_ price\_close)** | This function gets the evaluated profit, based on the parameters passed. The input parameters are: symbol, trade operation type, volume and open/close prices. | ```<br>double op_profit=myaccount.OrderProfitCheck(_Symbol,ORDER_TYPE_BUY,<br>1.0,1.2950,1.3235);<br>Print("The amount of Profit for deal buy EURUSD",<br>      "at 1.2950 and sell at 1.3235 is: ",op_profit);<br>``` |
| **myaccount.MarginCheck(const _string_ symbol, _ENUM\_ORDER\_TYPE_ trade\_operation, _double_**<br>**volume,** **_double_ price** **)** | This<br>function is used to get the margin required to open an order. This function<br>has four input parameters which are : the symbol (currency-pair), order type, the lots (or volume) to trade and the order price. This function is very important when<br>placing a trade. | ```<br>// depending on the type of position to open - in our case buy<br>double price=SymbolInfoDouble(_Symbol,SYMBOL_ASK); <br>double margin_req=myaccount.MarginCheck(_Symbol,ORDER_TYPE_BUY,LOT,price);<br>``` |
| **myaccount.FreeMarginCheck(const**<br>**_string_ symbol, _ENUM\_ORDER\_TYPE_ trade\_operation, _double_ volume,** **_double_ price** **)** | This<br>function is used to obtain the amount of free margin left in that active<br>account when an order<br>is placed. It has four input parameters which<br>are : the symbol (currency-pair), order type, lots (or volume) to trade and the order price. | ```<br>double acс_fm=myaccount.FreeMarginCheck(_Symbol,ORDER_TYPE_BUY,LOT,price);<br>``` |
| **myaccount.MaxLotCheck(const**<br>**_string_ symbol, _ENUM\_ORDER\_TYPE_ trade\_operation, _double_ price)** | This<br>function is used to get the maximum lot<br>possible to placing an order for the active account on the terminal. It has three input parameters<br>which are : the symbol, the order type and the order open price. | ```<br>double max_lot=myaccount.MaxLotCheck(_Symbol,ORDER_TYPE_BUY,price);<br>``` |

**1.2 The СSymbolInfo Class**

The [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo)
class makes it very easy for the user to quickly have access to all the properties of
the current symbol.

To use the class, we must create an object of the class, in
this case we will call it **mysymbol**.

```
// the CSymbolInfo Class object
CSymbolInfo mysymbol;
```

Let us have a look at most of the functions of this class that
may be used in the process of writing our Expert Advisor:

| Method | Description | Example of use |
| --- | --- | --- |
| **mysymbol.Name( _string_**<br>**name)** | This<br>function is used to set the symbol for the class object. It takes the symbol<br>name as input parameter. | ```<br>// set the symbol name for our CSymbolInfo class Object<br>mysymbol.Name(_Symbol);<br>``` |
| **mysymbol.Refresh()** | This<br>function is used to refresh all the symbol data. It is also called<br>automatically when you set a new symbol name for the class. | ```<br>mysymbol.Refresh();<br>``` |
| **mysmbol.RefreshRates()** | This<br>function is used to check the latest quotes data. It returns true on success<br>and false on failure. This is a useful function you cannot do without. | ```<br>//--- Get the last price quote using the CSymbolInfo <br>// class object function<br>   if (!mysymbol.RefreshRates())<br>   {<br>      // error getting latest price quotes <br>   }<br>``` |
| **mysymbol.IsSynchronized()** | This<br>function is used to check if the current data of the set symbol on the terminal<br>is synchronized with the data on the server. It returns true if data are<br>synchronized and false if not. | ```<br>// check if symbol data are synchronized with server<br>  if (!mysymbol.IsSynchronized())<br>   {<br>     // error! Symbol data aren't synchronized with server<br>   }<br>``` |
| **mysymbol.VolumeHigh()** | This<br>function is used to get the maximum volume of the day for the set symbol. | ```<br>long max_vol = mysymbol.VolumeHigh();<br>``` |
| **mysymbol.VolumeLow()** | This<br>function is used to get the minimum volume of the day for the set symbol. | ```<br>long min_vol = mysymbol.VolumeLow();<br>``` |
| **mysymbol.Time()** | This<br>function is used to get the time of the last price quote for the set symbol. | ```<br>datetime qtime = mysymbol.Time();<br>``` |
| **mysymbol.Spread()** | This function is used to get the current spread value (in points) for the set symbol. | ```<br>int spread = mysymbol.Spread();<br>``` |
| **mysymbol.StopsLevel()** | This function is used to get the minimal level (in points) to the current close price for which stop loss can be placed for the set symbol. A very useful function for use if you are considering using Trailing Stop or order /position modification. | ```<br>int stp_level = mysymbol.StopsLevel();<br>``` |
| **mysymbol.FreezeLevel()** | This function is used to get the distance (in points) of freezing trade operation for the set symbol | ```<br>int frz_level = mysymbol.FreezeLevel();<br>``` |
| **mysymbol.Bid()** | This<br>function is used to get the current BID price for the set symbol. | ```<br>double bid =  mysymbol.Bid();<br>``` |
| **mysymbol.BidHigh()** | This<br>function is used to get the maximum/highest BID price for the day. | ```<br>double max_bid = mysymbol.BidHigh();<br>``` |
| **mysymbol.BidLow()** | This<br>function is used to get the minimum/lowest BID price for the day for the set<br>symbol. | ```<br>double min_bid = mysymbol.BidLow();<br>``` |
| **msymbol.Ask()** | This<br>function is used to get the current ASK price for the set symbol. | ```<br>double ask = mysymbol.Ask();<br>``` |
| **mysymbol.AskHigh()** | This<br>function is used to get the maximum/highest ASK price for the day for the set<br>symbol. | ```<br>double max_ask = mysymbol.AskHigh();<br>``` |
| **mysymbol.AskLow()** | This<br>function is used to get the minimum/lowest ASK price for the day. | ```<br>double min_ask = mysymbol.AskLow();<br>``` |
| **mysymbol.CurrencyBase()** | This<br>function is used to get the base currency for the set symbol. | ```<br>// returns "USD" for USDJPY or USDCAD<br>string base_currency = mysymbol.CurrencyBase();<br>``` |
| **mysymbol.ContractSize()** | This<br>function is used to get the amount for the contract size for trading the set<br>symbol. | ```<br>double cont_size =  mysymbol.ContractSize();<br>``` |
| **mysymbol.Digits()** | This<br>function is used to get the number of digits after the decimal point for the<br>set symbol. | ```<br>int s_digits = mysymbol.Digits();<br>``` |
| **mysymbol.Point()** | This<br>function is used to get the value of one point for the set symbol. | ```<br>double s_point =  mysymbol.Point();<br>``` |
| **mysymbol.LotsMin()** | This<br>function is used to obtain the minimum volume required to close a deal for the<br>symbol. | ```<br>double min_lot =  mysymbol.LotsMin();<br>``` |
| **mysymbol.LotsMax()** | This<br>function is used to obtain the maximum volume required to close a deal for the<br>symbol. | ```<br>double max_lot =  mysymbol.LotsMax();<br>``` |
| **mysymbol.LotsStep()** | This<br>function is used to obtain the minimum step of volume change to close a deal<br>for the symbol. | ```<br>double lot_step = mysymbol.LotsStep();<br>``` |
| **mysymbol.NormalizePrice( _double_**<br>**price)** | This<br>function is used to get a normalized price to the correct digits of the set<br>symbol. | ```<br>// A normalized current Ask price<br>double n_price = mysymbol.NormalizePrice(mysymbol.Ask()); <br>``` |
| **mysymbol.Select()** | This<br>function is used to determine if a symbol has been selected in the market watch<br>window. It returns true if symbol has been selected otherwise it returns false. | ```<br>if (mysymbol.Select())<br>{<br>  //Symbol successfully selected<br>}<br>else<br>{<br>  // Symbol could not be selected<br>}<br>``` |
| **mysymbol.Select( _bool_**<br>**select)** | This<br>function is used to select a symbol in the Market watch window or to remove a<br>symbol in the market watch window. It should be noted that removing a symbol<br>from the market watch window when the chart is opened of when it already has a<br>position opened will return false. | ```<br>if (!mysymbol.Select())<br>{<br>   //Symbol not selected, Select the symbol<br>    mysymbol.Select(true);<br>}<br>else<br>{<br> // Symbol already selected, <br> // remove Symbol from market watch window<br>    mysymbol.Select(false);<br>}<br>``` |
| **mysymbol.MarginInitial()** | This<br>function is used to get the amount required for opening a position with volume<br>of one lot in the margin currency. | ```<br>double init_margin = mysymbol.MarginInitial() ; <br>``` |
| **mysymbol.TradeMode()** | This function<br>is used to obtain the order execution type allowed for the symbol. | ```<br>if (mysymbol.TradeMode() == SYMBOL_TRADE_MODE_FULL)<br>{<br> // Full trade allowed for this symbol,<br> // no trade restrictions <br>}<br>``` |
| **mysymbol.TradeModeDescription()** | This<br>function is used to obtain the description of the order execution type allowed<br>for the symbol. | ```<br>Print("The trade mode for this symbol is",<br>       mysymbol.TradeModeDescription());<br>``` |

**1.3 The СHistoryOrderInfo Class**

The [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) is another
class that makes it very easy to handle order history properties.

Once we create
an object of this class we can then use the object to access the important
public member functions that we need to solve an immediate problem.

Let us name the object of the class **myhistory**.

```
// The CHistoryOrderInfo Class object
CHistoryOrderInfo myhistory;
```

Let us look at some of the
major functions of this class.

In using this class to get the
details of orders in history, we need to first of all get the total orders in
history and then pass the order ticket to our class object, **myhistory**.

```
//Select all history orders within a time period
if (HistorySelect(0,TimeCurrent()))  // get all history orders
{
// Get total orders in history
int tot_hist_orders = HistoryOrdersTotal();
```

We will now iterate through the
total history orders available and get the details of each history orders with
our class object.

```
ulong h_ticket; // Order ticket

for (int j=0; j<tot_hist_orders; j++)
{
  h_ticket = HistoryOrderGetTicket(j));

  if (h_ticket>0)
  {
    // First thing is to now set the order Ticket to work with by our class object
```

| Method | Description | Example of use |
| --- | --- | --- |
| **myhistory.Ticket( _ulong_ ticket)** | This<br>function is used to select the order ticket for which we want to obtain its<br>properties or details. | ```<br>myhistory.Ticket(h_ticket);<br>``` |
| **myhistory.Ticket()** | This function is used to obtain the order ticket for an order. | ```<br>ulong o_ticket = myhistory.Ticket();<br>``` |
| **myhistory.TimeSetup()** | This<br>function is used to obtain the Time the order was carried out or setup. | ```<br>datetime os_time = myhistory.TimeSetup();<br>``` |
| **myhistory.OrderType()** | This<br>function is used to obtain the order type (ORDER\_TYPE\_BUY, etc). | ```<br>if (myhistory.OrderType() == ORDER_TYPE_BUY)<br>{<br>// This is a buy order<br>}<br>``` |
| **myhistory.State()** | This<br>function is used to obtain the current state of the order. <br>If the order has<br>been cancelled, accepted, rejected or placed, etc. | ```<br>if(myhistory.State() == ORDER_STATE_REJECTED)<br>{<br>// order was rejected, not placed.<br>}<br>``` |
| **myhistory.TimeDone()** | This<br>function is used to obtain the time the order was placed, cancelled or rejected. | ```<br>datetime ot_done =  myhistory.TimeDone();<br>``` |
| **myhistory.Magic()** | This<br>function is used to get the Expert Advisor id that initiated the order. | ```<br>long o_magic = myhistory.Magic();<br>``` |
| **myhistory.PositionId()** | This<br>function is used to get the id of position to which the order was included when<br>placed. | ```<br>long o_posid = myhistory.PositionId();<br>``` |
| **myhistory.PriceOpen()** | This<br>function is used to get the Order open price. | ```<br>double o_price =  myhistory.PriceOpen();<br>``` |
| **myhistory.Symbol()** | This<br>function is used to get the symbol property (currency pair) of the order. | ```<br>string o_symbol =  myhistory.Symbol();<br>``` |

Don’t forget that we used these
functions within a loop of total orders in history.

**1.4 The СOrderInfo Class**

This [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) is a
class that provides easy access to all the pending order properties. Once an
object of this class has been created, it can be used to public member
functions of this class.

The usage of this class is somewhat similar to the
[CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) class discussed above.

Let us create an object of the
class, we will call it **myorder**.

```
// The OrderInfo Class object
COrderInfo myorder;
```

To be able to use this class to
obtain the details of a pending order, we need to first of all get the total
available orders and then select them by the order ticket.

```
// Select all history orders within a time period
if (HistorySelect(0,TimeCurrent()))   // get all history orders
   {
     // get total orders
     int o_total = OrdersTotal();
```

We will loop through the
total orders and obtain their corresponding properties using the object we have
created.

```
for (int j=0; j<o_total; j++)
{
 // we must confirm if the order is available for us to get its details using the Select function of the COrderInfo Class
```

| Method | Description | Example of use |
| --- | --- | --- |
| **myorder.Select( _ulong_ ticket)** | This fucntion is used to select an order by ticket number so that the order can easily be manipulated. | ```<br>if (myorder.Select(OrderGetTicket(j)) <br>   { // order has been selected and can now be manipulated.<br>   }<br>``` |
| **myorder.Ticket()** | This function is used to get the order ticket for the selected order. | ```<br>ulong o_ticket = myorder.Ticket();<br>``` |
| **myorder.TimeSetup()** | This<br>function is used to get the time this order was setup. | ```<br>datetime o_setup = myorder.TimeSetup();<br>``` |
| **myorder.Type()** | This<br>function is used to get the order type like ORDER\_TYPE\_BUY\_STOP, etc. | ```<br>if (myorder.Type() == ORDER_TYPE_BUY_LIMIT)<br>{<br>// This is a Buy Limit order, etc<br>}<br>``` |
| **myorder.State()** | This<br>function is used to get the state of the order. <br>If the order has been<br>cancelled, accepted, rejected or placed, etc. | ```<br>if (myorder.State() ==ORDER_STATE_STARTED)<br>{<br>// order has been checked <br>// and may soon be treated by the broker<br>}<br>``` |
| **myorder.TimeDone()** | This<br>function is used to get the time the order was placed, rejectedor cancelled. | ```<br>datetime ot_done = myorder.TimeDone();<br>``` |
| **myorder.Magic()** | This<br>function is used to get the id of the Expert Advisor that initiated the order. | ```<br>long o_magic =  myorder.Magic();<br>``` |
| **myorder.PositionId()** | This<br>function is used to obtain the id of the position to which the order is<br>included when placed. | ```<br>long o_posid = myorder.PositionId();<br>``` |
| **myorder.PriceOpen()** | This<br>function is used to get the open price for the order. | ```<br>double o_price = myorder.PriceOpen();<br>``` |
| **myorder.StopLoss()** | This<br>function is used to obtain the Stoploss<br>of the order. | ```<br>double  s_loss = myorder.StopLoss();<br>``` |
| **myorder.TakeProfit()** | This<br>function is used to obtain the Take Profit of the order. | ```<br>double t_profit = myorder.TakeProfit();<br>``` |
| **myorder.PriceCurrent()** | This<br>function is used to get the current price of the symbol in which the order was<br>placed. | ```<br>double cur_price =  myorder.PriceCurrent();<br>``` |
| **myorder.Symbol()** | This<br>function is used to get the name of the symbol in which the order was placed. | ```<br>string o_symbol = myorder.Symbol();<br>``` |
| **myorder.StoreState()** | This<br>function is used to save or store the current detail of the order so that we<br>will able to compare if anything has changed later. | ```<br>myorder.StoreState();<br>``` |
| **myorder.CheckState()** | This<br>function is used to check if the detail of the order that was saved or stored<br>has changed. | ```<br>if (myorder.CheckState() == true)<br>{<br>// Our order status or details have changed<br>}<br>``` |

**1.5 The CDealInfo Class**

The [CDealInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo) class provide access to
all the history of deal properties or information.  Once we have created an
object of this class, we will then use it to get every information about deals
in history, in a similar way to the [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) class.

So, the first thing
we want to do is to create an object of this class and name it **mydeal.**

```
// The DealInfo Class object
CDealInfo myinfo;
```

We will start by getting the
total deals in history

```
if (HistorySelect(0,TimeCurrent()))
   {
    // Get total deals in history
    int tot_deals = HistoryDealsTotal();
```

We will now iterate through the
total history orders available and get the details of each history orders with
our class object.

```
ulong d_ticket; // deal ticket
for (int j=0; j<tot_deals; j++)
    {
     d_ticket = HistoryDealGetTicket(j);
     if (d_ticket>0)
     {
      // First thing is to now set the deal Ticket to work with by our class object
```

| Method | Description | Example of use |
| --- | --- | --- |
| **mydeal.Ticket( _ulong_ ticket)** | This<br>function is used to set the deal ticket for further use by the object we have<br>created | ```<br>mydeal.Ticket(d_ticket);<br>``` |
| **mydeal.Ticket()** | This function is used to get the deal ticket | ```<br>ulong deal_ticket = mydeal.Ticket();<br>``` |
| **mydeal.Order()** | This<br>function is used to get the order ticket for the order in which the deal was<br>executed | ```<br>long deal_order_no =  mydeal.Order();<br>``` |
| **mydeal.Time()** | This<br>function is used to get the time the deal was executed | ```<br>datetime d_time = mydeal.Time();<br>``` |
| **mydeal.Type()** | This<br>function is used to obtain the deal type, whether it was a [DEAL\_TYPE\_SELL](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties), etc | ```<br>if (mydeal.Type() == DEAL_TYPE_BUY)<br>{<br>// This deal was executed as a buy deal type<br>}<br>``` |
| **mydeal.Entry()** | This<br>function is used to obtain the direction of the deal, whether it is<br>[DEAL\_ENTRY\_IN](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) or [DEAL\_ENTRY\_OUT](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties), etc. | ```<br>if (mydeal.Entry() == DEAL_ENTRY_IN)<br>{<br>// This was an IN entry deal<br>}<br>``` |
| **mydeal.Magic()** | This<br>function is used to obtain the id of the Expert Advisor that executed the deal. | ```<br>long d_magic = mydeal.Magic();<br>``` |
| **mydeal.PositionId()** | This<br>function is used to get the unique position identifier for the position in<br>which the deal was part of. | ```<br>long d_post_id = mydeal.PositionId();<br>``` |
| **mydeal.Price()** | This<br>function is used to get the price at which the deal was executed | ```<br>double d_price = mydeal.Price();<br>``` |
| **mydeal.Volume()** | This<br>function is used to get the volume (lot) of the deal | ```<br>double d_vol = mydeal.Volume();<br>``` |
| **mydeal.Symbol()** | This<br>function is used to obtain the symbol (currency-pair) for which the deal was<br>executed | ```<br>string d_symbol = mydeal.Symbol();<br>``` |

**1.6 The CPositionInfo** **Class**

The[CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo)class provides easy access
to the current position properties. We must create an object of this class to
be able to use it to get the position properties.

Let us create an object of this
class and call it **myposition.**

```
// The object of the CPositionInfo class
CPositionInfo myposition;
```

We will now use this object to
get open positions details. We will start by getting the total open positions
available:

```
int pos_total = PositionsTotal();
```

It is
now time to go through all open positions to get their details.

```
for (int j=0; j<pos_total; j++)
    {
```

| Method | Description | Example of use |
| --- | --- | --- |
| **myposition.Select(const**<br>**_string_ symbol)** | This<br>function is used to select the symbol corresponding to the current open<br>position so that it can be worked on. | ```<br>if (myposition.Select(PositionGetSymbol(j)))<br>{<br> // symbol successfully selected, we can now work <br> // on the current open position for this symbol <br>}<br>OR<br>// when dealing with the current symbol/chart only<br>if (myposition.Select(_Symbol)) <br>{<br> // symbol successfully selected, we can now work <br> // on the current open position for this symbol <br>}<br>``` |
| **myposition.Time()** | This<br>function is used to get the time the position was opened. | ```<br>datetime pos_time = myposition.Time();<br>``` |
| **myposition.Type()** | This<br>function is used to get the type of position opened. | ```<br>if (myposition.Type() == POSITION_TYPE_BUY)<br>{<br>// This is a buy position<br>}<br>``` |
| **myposition.Magic()** | This<br>function is used to obtain the id of the Expert Advisor that opened the position. | ```<br>long pos_magic = myposition.Magic();<br>``` |
| **myposition.Volume()** | This<br>function is used to get the volume (lots) of the open position. | ```<br>double pos_vol = myposition.Volume(); // Lots<br>``` |
| **myposition.PriceOpen()** | This<br>function is used to get the price at which the position was opened – the<br>position open price. | ```<br>double pos_op_price = myposition.PriceOpen();<br>``` |
| **myposition.StopLoss()** | This<br>function is used to get the Stop Loss price for the open position. | ```<br>double pos_stoploss = myposition.StopLoss();<br>``` |
| **myposition.TakeProfit()** | This<br>function is used to get the Take Profit price for the open position. | ```<br>double pos_takeprofit = myposition.TakeProfit();<br>``` |
| **myposition.StoreState()** | This function is<br>used to store the current state of the position. | ```<br>// stores the current state of the position<br>myposition.StoreState();<br>``` |
| **myposition.CheckState()** | This<br>function is used to check if the state of the open position has changed. | ```<br>if (!myposition.CheckState())<br>{<br>  // position status has not changed yet<br>}<br>``` |
| **myposition.Symbol()** | This<br>function is used to get the name of the symbol in which the position was<br>opened. | ```<br>string pos_symbol = myposition.Symbol();<br>``` |

**1.7 The СTrade Class**

The [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class provides easy access
to the trade operations in MQL5. To use this class, we have to create an object of
the class and then use it to perform the necessary trade operations.

We will create an object of
this class and name it **mytrade:**

```
//An object of the CTrade class
CTrade mytrade;
```

The first step is to set most
of the parameters that the object will use in making trade operation.

| Method | Description | Example of use |
| --- | --- | --- |
| **mytrade.SetExpertMagicNumber( _ulong_**<br>**magic)** | This<br>function is used to set the expert id (magic number) the class will use for<br>trade operations. | ```<br>ulong Magic_No=12345;<br>mytrade.SetExpertMagicNumber(Magic_No);<br>``` |
| **mytrade.SetDeviationInPoints( _ulong_**<br>**deviation)** | This<br>function is also used to set the value of deviation (in points) to be used when<br>placing a trade. | ```<br>ulong Deviation=20;<br>mytrade.SetDeviationInPoints(Deviation); <br>``` |
| **mytrade.****OrderOpen(const** _**string**_ **symbol,** _**ENUM\_ORDER\_TYPE**_ **order\_type,** _**double**_ **volume,**_**double**_ **limit\_price,** _**double**_ **price,** _**double**_ **sl,** _**double**_ **tp,** _**ENUM\_ORDER\_TYPE\_TIME**_ **type\_time,**_**datetime**_ **expiration,const** _**string**_ **comment="")** | This<br>function is used to place a pending order. Touse this function, the parameters must first of all be prepared and then<br>passed to this function. | ```<br>// define the input parameters<br>double Lots = 0.1;<br>double SL = 0;<br>double TP = 0;<br>// latest Bid price using CSymbolInfo class object<br>double Oprice = mysymbol.Bid()-_Point*550;<br>// place (BuyStop) pending order<br>mytrade.OrderOpen(_Symbol,ORDER_TYPE_SELLSTOP,Lots,0.0,Oprice,<br>                  SL,TP,ORDER_TIME_GTC,0);<br>``` |
| **mytrade.OrderModify( _ulong_**<br>**ticket, _double_ price, _double_ sl, _double_ tp,_ENUM\_ORDER\_TYPE\_TIME_ type\_time, _datetime_ expiration)** | This<br>function is used to modify an existing pending order. | ```<br>// Select total orders in history and get total pending orders <br>// (as shown within the COrderInfo class section). <br>// Use the CSymbolInfo class object to get the current ASK/BID price<br>int Stoploss = 400;<br>int Takeprofit = 550;<br>for(int j=0; j<OrdersTotal(); j++)<br>{<br>  ulong o_ticket = OrderGetTicket(j);<br>  if(o_ticket != 0)<br>  {<br>   // Stoploss must have been defined<br>   double SL = mysymbol.Bid() + Stoploss*_Point;   <br>   // Takeprofit must have been defined  <br>   double TP = mysymbol.Bid() - Takeprofit*_Point; <br>   // lastest ask price using CSymbolInfo class object<br>   double Oprice = mysymbol.Bid();                 <br>   // modify pending BuyStop order<br>   mytrade.OrderModify(o_ticket,Oprice,SL,TP,ORDER_TIME_GTC,0);<br>  }<br>}<br>``` |
| **mytrade.OrderDelete( _ulong_**<br>**ticket)** | This<br>function is used to delete a pending order. | ```<br>// Select total orders in history and get total pending orders<br>// (as shown within the COrderInfo class section). <br>int o_total=OrdersTotal();<br>for(int j=o_total-1; j>=0; j--)<br>{<br>   ulong o_ticket = OrderGetTicket(j);<br>   if(o_ticket != 0)<br>   {<br>    // delete the pending Sell Stop order<br>    mytrade.OrderDelete(o_ticket);<br>   }<br>}<br>``` |
| **mytrade.PositionOpen(const**<br>**_string_ symbol, _ENUM\_ORDER\_TYPE_ order\_type, _double_ volume,_double_**<br>**price, _double_ sl, _double_ tp,const _string_ comment="")** | This<br>function is used to open a BUY or a SELL position. To use this function, all<br>the required parameters must first of all be prepared and then passed to this<br>function. | ```<br>// define the input parameters and use the CSymbolInfo class<br>// object to get the current ASK/BID price<br>double Lots = 0.1;<br>// Stoploss must have been defined <br>double SL = mysymbol.Ask() – Stoploss*_Point;   <br>//Takeprofit must have been defined <br>double TP = mysymbol.Ask() + Takeprofit*_Point; <br>// latest ask price using CSymbolInfo class object<br>double Oprice = mysymbol.Ask();<br>// open a buy trade<br>mytrade.PositionOpen(_Symbol,ORDER_TYPE_BUY,Lots,<br>                     Oprice,SL,TP,"Test Buy");<br>``` |
| **mytrade.PositionModify(const**<br>**_string_ symbol, _double_ sl, _double_ tp)** | This<br>function is used to modify the StopLoss and/or TakeProfit for an existing open<br>position. To use this function, we must first of all select the position to be<br>modified using the CPositionInfo Class object, use the CSymbolInfo class object<br>to get current BID/ASK price. | ```<br>if (myposition.Select(_Symbol))<br>{<br>  int newStoploss = 250;<br>  int newTakeprofit = 500;<br>  double SL = mysymbol.Ask() – newStoploss*_Point;    <br>  double TP = mysymbol.Ask() + newTakeprofit*_Point;  <br>  //modify the open position for this symbol<br> mytrade.PositionModify(_Symbol,SL,TP);<br>}<br>``` |
| **mytrade.PositionClose(const**<br>**_string_ symbol, _ulong_ deviation=ULONG\_MAX)** | This<br>function is used to close an existing open position. | ```<br>if (myposition.Select(_Symbol))<br>{<br> //close the open position for this symbol<br> mytrade.PositionClose(_Symbol);  <br>}<br>``` |
| **mytrade.Buy( _double_**<br>**volume,const _string_ symbol=NULL, _double_ price=0.0, _double_ sl=0.0, _double_ tp=0.0,const _string_ comment="")** | This<br>function is used to open a buy trade. It is recommended that you set the **volume**<br>(or lots ) to trade when using this function. While the **tp** (take profit) and **sl**<br>(stop loss) can be set later by modify the opened position, it uses the current<br>**Ask** price to open the trade. | ```<br>double Lots = 0.1;<br>// Stoploss must have been defined <br>double SL = mysymbol.Ask() – Stoploss*_Point; <br>//Takeprofit must have been defined<br>double TP = mysymbol.Ask() +Takeprofit*_Point;<br>// latest ask price using CSymbolInfo class object<br>double Oprice = mysymbol.Ask();<br>// open a buy trade<br>mytrade.Buy(Lots,NULL,Oprice,SL,TP,“Buy Trade”);<br>//OR<br>mytrade.Buy(Lots,NULL,0.0,0.0,0.0,“Buy Trade”);<br>// modify position later<br>``` |
| **mytrade.Sell( _double_**<br>**volume,const _string_ symbol=NULL, _double_ price=0.0, _double_ sl=0.0, _double_ tp=0.0,const _string_ comment="")** | This<br>function is used to open a Sell trade. It is recommended that you set the **volume**<br>(or lots ) to trade when using this function. While the **tp** (take profit) and **sl** <br>(stop loss) can be set later by modify the opened position, it uses the current<br>**Bid** price to open the trade. | ```<br>double Lots = 0.1;<br>// Stoploss must have been defined <br>double SL = mysymbol.Bid() + Stoploss*_Point;<br>//Takeprofit must have been defined<br>double TP = mysymbol.Bid() - Takeprofit*_Point; <br>// latest bid price using CSymbolInfo class object<br>double Oprice = mysymbol.Bid();<br>// open a Sell trade<br>mytrade.Sell(Lots,NULL,Oprice,SL,TP,“Sell Trade”); <br>//OR<br>mytrade.Sell(Lots,NULL,0.0,0.0,0.0,“Sell Trade”); <br>//(modify position later)<br>``` |
| **mytrade.BuyStop( _double_ volume, _double_ price,const _string_ symbol=NULL, _double_ sl=0.0, _double_ tp=0.0,**<br>**_ENUM\_ORDER\_TYPE\_TIME_ type\_time=ORDER\_TIME\_GTC, _datetime_ expiration=0,const _string_ comment="")** | This function is used to place a BuyStop pending order. The default <br>Order type time is ORDER\_TIME\_GTC, and expiration is 0. There is no need<br> to specify these two variables if you have the same order type time in <br>mind. | ```<br> double Lot = 0.1;<br>//Buy price = bar 1 High + 2 pip + spread<br> int sprd=mysymbol.Spread();<br> double bprice =mrate[1].high + 2*_Point + sprd*_Point;<br>//--- Buy price<br> double mprice=NormalizeDouble(bprice,_Digits); <br>//--- Stop Loss<br> double stloss = NormalizeDouble(bprice - STP*_Point,_Digits);<br>//--- Take Profit<br> double tprofit = NormalizeDouble(bprice+ TKP*_Point,_Digits);<br>//--- open BuyStop order<br> mytrade.BuyStop(Lot,mprice,_Symbol,stloss,tprofit); <br>``` |
| **mytrade.SellStop( _double_**<br>**volume, _double_ price,const _string_ symbol=NULL, _double_ sl=0.0, _double_ tp=0.0,_ENUM\_ORDER\_TYPE\_TIME_**<br>**type\_time=ORDER\_TIME\_GTC, _datetime_ expiration=0,const _string_**<br>**comment="")** | This<br>function is used to place a SellStop Pending order with the set parameters. The default Order type time is ORDER\_TIME\_GTC, and expiration is 0. There is no need to specify these two variables if you have the same order type time in mind. | ```<br> double Lot = 0.1;<br>//--- Sell price = bar 1 Low - 2 pip <br>//--- MqlRates mrate already declared<br> double sprice=mrate[1].low-2*_Point;<br>//--- SellStop price<br> double slprice=NormalizeDouble(sprice,_Digits);<br>//--- Stop Loss<br> double ssloss=NormalizeDouble(sprice+STP*_Point,_Digits);<br>//--- Take Profit<br> double stprofit=NormalizeDouble(sprice-TKP*_Point,_Digits);<br>//--- Open SellStop Order<br> mytrade.SellStop(Lot,slprice,_Symbol,ssloss,stprofit);<br>``` |
| **mytrade.BuyLimit( _double_**<br>**volume, _double_ price,const _string_ symbol=NULL,double sl=0.0, _double_ tp=0.0,_ENUM\_ORDER\_TYPE\_TIME_**<br>**type\_time=ORDER\_TIME\_GTC, _datetime_ expiration=0,const _string_**<br>**comment="")** | This<br>function is used to place a BuyLimit order with the set parameters. | ```<br>Usage:<br>//--- Buy price = bar 1 Open  - 5 pip + spread<br>double Lot = 0.1;<br>int sprd=mysymbol.Spread();<br>//--- symbol spread<br>double bprice = mrate[1].open - 5*_Point + sprd*_Point;<br>//--- MqlRates mrate already declared<br>double mprice=NormalizeDouble(bprice,_Digits);<br>//--- BuyLimit price<br>//--- place buyLimit order, modify stoploss and takeprofit later<br>mytrade.BuyLimit(Lot,mprice,_Symbol); <br>``` |
| **mytrade.SellLimit( _double_**<br>**volume, _double_ price,const _string_ symbol=NULL,double sl=0.0, _double_ tp=0.0, _ENUM\_ORDER\_TYPE\_TIME_**<br>**type\_time=ORDER\_TIME\_GTC, _datetime_ expiration=0,const _string_ comment="")** | This function is used to place a Sell Limit order with the set parameters. | ```<br>//--- Sell Limit price = bar 1 Open  + 5 pip<br>double Lot = 0.1;<br>//--- MqlRates mrate already declared<br>double sprice = mrate[1].open + 5*_Point;<br>//--- SellLimit<br>double slprice=NormalizeDouble(sprice,_Digits);<br>//place SellLimit order, modify stoploss and takeprofit later<br>mytrade.SellLimit(Lot,slprice,_Symbol);<br>``` |
| **TRADE RESULT FUNCTIONS** |  |  |
| **mytrade.ResultRetcode()** | This<br>function is used to get the result code for a trade operation. | ```<br>// a trade operation has just been carried out<br>int return_code = mytrade.ResultRetcode();<br>``` |
| **mytrade.ResultRetcodeDescription()** | This<br>function is used to get the full description or interpretation of the returned<br>code of a trade operation. | ```<br>string ret_message =  ResultRetcodeDescription();<br>// display it<br>Alert("Error code - " , mytrade.ResultRetcode() ,<br>      "Error message - ", ret_message);<br>``` |
| **mytrade.ResultDeal()** | This<br>function is used to get the deal ticket for the open position. | ```<br>long dl_ticket = mytrade.ResultDeal();<br>``` |
| **mytrade.ResultOrder()** | This<br>function is used to get the order ticket for the opened position. | ```<br>long o_ticket = mytrade.ResultOrder();<br>``` |
| **mytrade.ResultVolume()** | This<br>function is used to get the volume (Lots) of order for the opened position. | ```<br>double o_volume = mytrade.ResultVolume();<br>``` |
| **mytrade.ResultPrice()** | This<br>function is used to get the deal price for the opened position. | ```<br>double r_price = mytrade.ResultPrice();<br>``` |
| **mytrade.ResultBid()** | This<br>function is used to get the current market BID price (re-quote price). | ```<br>double rq_bid = mytrade.ResultBid;<br>``` |
| **mytrade.ResultAsk()** | This<br>function is used to get the current market ASK price (re-quote price). | ```<br>double rq_ask = mytrade.ResultAsk;<br>``` |
| **mytrade.PrintRequest()** / **mytrade.PrintResult()** | These<br>two functions can be used to print, to the Journal Tab, the trade request<br>parameters and the result parameters respectively. | ```<br>// after a trade operation<br>// prints the trade request parameters<br>mytrade.PrintRequest(); <br>//prints the trade results<br>mytrade.PrintResult();  <br>``` |
| **TRADE REQUEST FUNCTIONS** |  |  |
| **mytrade.RequestAction()** | This function is used to obtain the Trade Operation type for the last Trade request that has just been sent. | ```<br>//determine the Trade operation type for the last Trade request<br>if (mytrade.RequestAction() == TRADE_ACTION_DEAL)<br>{<br>  // this is a market order for an immediate execution<br>}<br>else if (mytrade.RequestAction() == TRADE_ACTION_PENDING)<br>{<br>  // this is a pending order.<br>}  <br>``` |
| **mytrade.RequestMagic()** | This<br>function is used to obtain the Expert Magic number that was used in the last<br>request. | ```<br>ulong mag_no = mytrade. RequestMagic();<br>``` |
| **mytrade.RequestOrder()** | This<br>function is used to obtain the order ticket that was used in the last request.<br>This relates mainly to modification of pending orders. | ```<br>ulong po_ticket =  mytrade.RequestOrder();<br>``` |
| **mytrade.RequestSymbol()** | This<br>function is used to obtain the symbol or currency pair that was used in the<br>last request. | ```<br>string symb = mytrade.RequestSymbol(); <br>``` |
| **mytrade.RequestVolume()** | This<br>function is used to obtain the volume of trade (in lots) placed in the last<br>request. | ```<br>double Lot = mytrade.RequestVolume();  <br>``` |
| **mytrade.RequestPrice()** | This<br>function is used to obtain the order price usedin the last request. | ```<br>double oprice = mytrade.RequestPrice();   <br>``` |
| **mytrade.RequestStopLimit()** | This<br>function is used to obtain the Stop Lossprice usedin the last request. | ```<br>double limitprice = mytrade.RequestStopLimit(); <br>``` |
| **mytrade.RequestSL()** | This<br>function is used to obtain the Stop Lossprice usedin the last request. | ```<br>double sloss = mytrade.RequestSL();  <br>``` |
| **mytrade.RequestTP()** | This<br>function is used to obtain the Take Profit price usedin the last request. | ```<br>double tprofit = mytrade.RequestTP();   <br>``` |
| **mytrade.RequestDeviation()** | This<br>function is used to obtain the Deviation usedin the last request. | ```<br>ulong dev = mytrade.RequestDeviation();  <br>``` |
| **mytrade.RequestType()** | This<br>function is used to obtain the type of order that was placed in the last<br>request. | ```<br>if (mytrade.RequestType() == ORDER_TYPE_BUY)<br> {<br>  // market order Buy was placed in the last request.<br> } <br>``` |
| **mytrade.RequestTypeDescription()** | This<br>function is used to get the description of the type of order placed in the last<br>request. | ```<br>Print("The type of order placed in the last request is :",<br>      mytrade.RequestTypeDescription());  <br>``` |
| **mytrade.RequestActionDescription()** | This<br>function is used to get the description of the request action used in the last<br>request. | ```<br>Print("The request action used in the last request is :", <br>      mytrade.RequestTypeDescription());<br>``` |
| **mytrade.RequestTypeFillingDescription()** | This<br>function is used to get the type of order filling policy used in the last<br>request. | ```<br>Print("The type of order filling policy used",<br>      " in the last request is :",<br>      RequestTypeFillingDescription()); <br>``` |

The Trade Class  Request functions are very useful when identifying errors associated with placing of orders. There are times when we get some error messages when placing an order and it becomes a bit confusing when we can not immediately identify what went wrong. By using the Trade Class request functions, we can be able to identify what we did wrong by printing our some of the request parameters that was sent to the trade server. An example of such usage will be similar as the code below:

```
 //--- open Buy position and check the result
         if(mytrade.Buy(Lot,_Symbol,mprice,stloss,tprofit))
         //if(mytrade.PositionOpen(_Symbol,ORDER_TYPE_BUY,Lot,mprice,stloss,tprofit))
           { //--- Request is completed or order placed
             Alert("A Buy order at price:", mytrade.ResultPrice() , ", vol:",mytrade.ResultVolume(),
                  " has been successfully placed with deal Ticket#:",mytrade.ResultDeal(),"!!");
            mytrade.PrintResult();
           }
         else
           {
            Alert("The Buy order request at vol:",mytrade.RequestVolume(), ", sl:", mytrade.RequestSL(),
                 ", tp:",mytrade.RequestTP(), ", price:", mytrade.RequestPrice(),
                 " could not be completed -error:",mytrade.ResultRetcodeDescription());
            mytrade.PrintRequest();
            return;
           }
```

In the above code, we have tried to be able to identify some of the the parameters sent in our request in case there was an error. For example, if we did not specify the correct Stop Loss price, we may get Invalid Stops error and by printing out the value of the Stop Loss using the **mytrade.RequestSL(),** we will be able to know what the problem is with our specified Stop Loss price.

Having taken time to show how
each of the classes can be used, it is now time to
put into practice some of the functionalities we have described.

Please note
that all the functionalities we are going to use in the Expert Advisor has already been described above, it will be a good idea to
always refer to the descriptions once you see any of the functions in the codes we are going to write.

### 2.Using the Trade Classes' functionalities

In order to demonstrate how to
use these [trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses)' functionalities, we are going to write an Expert
Advisor that will perform the following tasks.

- It
will check for a Buy or Sell condition, and if the condition is met, it will
place a Buy or Sell order depending on the condition that was met.
- If
a position has been opened and the trade continues to go in our direction,
we will modify the take profit or stop loss of the position. However, if
the trade is going against us and our profit target has not been hit, we
will close the position.
- Our
EA will be used to trade on the daily chart on any of the following
currencies – GBPUSD, AUDUSD, EURUSD, etc.

**2.1 Writing the Expert Advisor**

To begin, start a new MQL5 document
and select **Expert Advisor (template)** and click the Next button:

![Staring a new MQL5 Document](https://c.mql5.com/2/2/new-mql-doc.png)

Figure 1. Starting a new MQL5 document

Type the name for the Expert
Advisor and click the Finish button. We will define the input parameters
manually later.

![Give a name to the new document](https://c.mql5.com/2/2/new_expert-name.png)

Figure 2. Naming the Expert Advisor

The created new document should
look similar like below.

![The expert code skeleton](https://c.mql5.com/2/2/new_expert-code.png)

Just immediately after the
[#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) version line, we will include all the [Trade classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) we are going to
use.

```
//+------------------------------------------------------------------+
//|  Include ALL classes that will be used                           |
//+------------------------------------------------------------------+
//--- The Trade Class
#include <Trade\Trade.mqh>
//--- The PositionInfo Class
#include <Trade\PositionInfo.mqh>
//--- The AccountInfo Class
#include <Trade\AccountInfo.mqh>
//--- The SymbolInfo Class
#include <Trade\SymbolInfo.mqh>
```

Next we will define our input
parameters:

```
//+------------------------------------------------------------------+
//|  INPUT PARAMETERS                                              |
//+------------------------------------------------------------------+
input int      StopLoss=100;     // Stop Loss
input int      TakeProfit=240;   // Take Profit
input int      ADX_Period=15;    // ADX Period
input int      MA_Period=15;     // Moving Average Period
input ulong    EA_Magic=99977;   // EA Magic Number
input double   Adx_Min=24.0;     // Minimum ADX Value
input double   Lot=0.1;          // Lots to Trade
input ulong    dev=100;          // Deviation
input long     Trail_point=32;   // Points to increase TP/SL
input int      Min_Bars = 20;    // Minimum bars required for Expert Advisor to trade
input double   TradePct = 25;    // Percentage of Account Free Margin to trade
```

We will also specify other
parameters that will be used in this Expert Advisor code:

```
//+------------------------------------------------------------------+
//|  OTHER USEFUL PARAMETERS                                         |
//+------------------------------------------------------------------+
int adxHandle;                     // handle for our ADX indicator
int maHandle;                    // handle for our Moving Average indicator
double plsDI[],minDI[],adxVal[]; // Dynamic arrays to hold the values of +DI, -DI and ADX values for each bars
double maVal[];                  // Dynamic array to hold the values of Moving Average for each bars
double p_close;                    // Variable to store the close value of a bar
int STP, TKP;                   // To be used for Stop Loss, Take Profit
double TPC;                        // To be used for Trade percent
```

Let us now create an object of
each of the classes we have included:

```
//+------------------------------------------------------------------+
//|  CREATE CLASS OBJECTS                                            |
//+------------------------------------------------------------------+
//--- The Trade Class Object
CTrade mytrade;
//--- The PositionInfo Class Object
CPositionInfo myposition;
//--- The AccountInfo Class Object
CAccountInfo myaccount;
//--- The SymbolInfo Class Object
CSymbolInfo mysymbol;
```

The next thing we want to do
now, is to define some functions we are going to use to make our work very
easy.

Once we have defined these functions, we will just be calling them within
necessary sections in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) and [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) functions.

**2.1.1 The checkTrading function**

This function is going to be
used to perform all initial checks to see if our Expert Advisor can trade or
not. If this function returns true, our EA will proceed, otherwise the EA will
not perform any trade.

```
//+------------------------------------------------------------------+
//|  Checks if our Expert Advisor can go ahead and perform trading   |
//+------------------------------------------------------------------+
bool checkTrading()
{
  bool can_trade = false;
  // check if terminal is syncronized with server, etc
  if (myaccount.TradeAllowed() && myaccount.TradeExpert() && mysymbol.IsSynchronized())
  {
    // do we have enough bars?
    int mbars = Bars(_Symbol,_Period);
    if(mbars >Min_Bars)
    {
      can_trade = true;
    }
  }
  return(can_trade);
}
```

We declared a bool data type
_can\_trade_ and make it false. We used the
object of the [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) class to check if trade is allowed and also if
Expert Advisors are allowed to trade on this account. We also use an object of the
[CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class to check if the terminal is synchronized with the trade
server.

Once these three conditions are satisfied, we then check if the total
number of current bars is greater than the minimum required bars for our EA to
trade. If this function returns true, then our EA will perform trade
activities, otherwise, our EA will not engage in any trade activity until the
conditions in this function is satisfied.

As you have seen, we have decided to
include all the necessary trade check activities in this function, using the
necessary objects of the standard trade class libraries.

**2.1.2 The ConfirmMargin function**

```
//+------------------------------------------------------------------+
//|  Confirms if margin is enough to open an order
//+------------------------------------------------------------------+
bool ConfirmMargin(ENUM_ORDER_TYPE otype,double price)
  {
   bool confirm = false;
   double lot_price = myaccount.MarginCheck(_Symbol,otype,Lot,price); // Lot price/ Margin
   double act_f_mag = myaccount.FreeMargin();                        // Account free margin
   // Check if margin required is okay based on setting
   if(MathFloor(act_f_mag*TPC)>MathFloor(lot_price))
     {
      confirm =true;
     }
    return(confirm);
  }
```

We use the object of the
[CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo) class to confirm if there is enough margin to place a trade based
on the setting that we will only use a certain percentage of our account free
margin to place an order.

If the required percentage of the account free margin
is greater that the margin required for the order, then this function returns true, otherwise, it returns false.  By this, we only want to place an order if
the function returns true. This function takes the order type as input
parameter.

**2.1.3 The checkBuy function**

```
//+------------------------------------------------------------------+
//|  Checks for a Buy trade Condition                                |
//+------------------------------------------------------------------+
bool checkBuy()
{
  bool dobuy = false;
  if ((maVal[0]>maVal[1]) && (maVal[1]>maVal[2]) &&(p_close > maVal[1]))
  {
    // MA increases upwards and previous price closed above MA
    if ((adxVal[1]>Adx_Min)&& (plsDI[1]>minDI[1]))
    {
      // ADX is greater than minimum and +DI is greater tha -DI for ADX
      dobuy = true;
    }
  }
  return(dobuy);
}
```

We have decided to wrap up the
conditions for opening a buy trade in this function. We did not use any of the
Class object functionalities here. We are checking for condition where the
values of the Moving Average indicator is increasing upwards and the close price
of the previous bar is higher than the value of Moving average at that point.

We also want a situation where the value of ADX indicator is greater than the
required minimum set in the input parameters and the value of positive DI of
ADX indicator is greater than the negative DI value. Once these conditions are
met, then we will want our EA to open a BUY order.

**2.1.4 The checkSell function**

```
//+------------------------------------------------------------------+
//|  Checks for a Sell trade Condition                               |
//+------------------------------------------------------------------+
bool checkSell()
{
  bool dosell = false;
  if ((maVal[0]<maVal[1]) && (maVal[1]<maVal[2]) &&(p_close < maVal[1]))
  {
    // MA decreases downwards and previuos price closed below MA
    if ((adxVal[1]>Adx_Min)&& (minDI[1]>plsDI[1]))
    {
      // ADX is greater than minimum and -DI is greater tha +DI for ADX
      dosell = true;
    }
  }
  return(dosell);
}
```

This function checks exactly
the opposite of the CheckBuy function. Also we did not use any of the class
objects in this function. This function checks for a condition where the values
of the Moving Average indicator is decreasing downwards and the close price of
the previous bar is lower than the value of Moving average at that point.

We
also want a situation where the value of ADX indicator is greater than the
required minimum set in the input parameters and the value of negative DI of
ADX indicator is greater than the positive DI value. Once these conditions are
met, then we will want our EA to open a SELL order.

**2.1.5 The checkClosePos function**

```
//+------------------------------------------------------------------+
//|  Checks if an Open position can be closed                        |
//+------------------------------------------------------------------+
bool checkClosePos(string ptype, double Closeprice)
{
   bool mark = false;
   if (ptype=="BUY")
   {
      // Can we close this position
     if (Closeprice < maVal[1]) // Previous price close below MA
      {
         mark = true;
      }
   }
   if (ptype=="SELL")
   {
      // Can we close this position
      if (Closeprice > maVal[1]) // Previous price close above MA
      {
         mark = true;
      }
   }
   return(mark);
}
```

This function is used to check
if the present open position can be closed. This function is used to monitor
the if the close price of the previous bar is higher or lower than the value of
the Moving Average indicator at that point (depending on the direction of the
trade).

If any of the condition is met, this function returns true and then we
will expect our EA to close the position. This function has two input
parameters, the type of order (this time the name – BUY or SELL) and the close
price of the previous bar.

**2.1.6 The ClosePosition function**

```
//+------------------------------------------------------------------+
//| Checks and closes an open position                               |
//+------------------------------------------------------------------+
bool ClosePosition(string ptype,double clp)
  {
   bool marker=false;

      if(myposition.Select(_Symbol)==true)
        {
         if(myposition.Magic()==EA_Magic && myposition.Symbol()==_Symbol)
           {
            //--- Check if we can close this position
            if(checkClosePos(ptype,clp)==true)
              {
               //--- close this position and check if we close position successfully?
               if(mytrade.PositionClose(_Symbol)) //--- Request successfully completed
                 {
                  Alert("An opened position has been successfully closed!!");
                  marker=true;
                 }
               else
                 {
                  Alert("The position close request could not be completed - error: ",
                       mytrade.ResultRetcodeDescription());
                 }
              }
           }
        }
      return(marker);
     }
```

This is the function that
actually uses the above function ( **checkclosepos**). It makes use of the objects
of the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) and the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) classes. This function uses the object of
the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) class to check the available open positions for the position
that was opened by our EA and for the current symbol. If any position is found,
it checks if it can be closed using the **checkclosepos** function.

If the
**checkclosepos** function returns true, this function uses the object of the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class to close the position and displays the results for the position
close operation. If the position was closed successfully, this function returns true, otherwise, it returns false.

The function takes two input parameters (the _position name_ , BUY or SELL and the _previous bar close price_). These parameters were actually passed to the **checkclosepos** function which uses them.

**2.1.7 The CheckModify function**

```
//+------------------------------------------------------------------+
//|  Checks if we can modify an open position                        |
//+------------------------------------------------------------------+
bool CheckModify(string otype,double cprc)
{
   bool check=false;
   if (otype=="BUY")
   {
      if ((maVal[2]<maVal[1]) && (maVal[1]<maVal[0]) && (cprc>maVal[1]) && (adxVal[1]>Adx_Min))
      {
         check=true;
      }
   }
   else if (otype=="SELL")
   {
      if ((maVal[2]>maVal[1]) && (maVal[1]>maVal[0]) && (cprc<maVal[1]) && (adxVal[1]>Adx_Min))
      {
         check=true;
      }
   }
   return(check);
}
```

This function is used to check
for a condition that confirms if an opened position can be modified or not. It
uses the order type name and the previous bar close price as input parameters.

What this function does is to check if the Moving average is still increasing
upwards and the previous bar close price is still higher than the Moving
average value at that point and the value of ADX is also greater that the
required minimum (for a BUY position) while it checks if the Moving average is
still decreasing downwards and the close price of the previous bar is lower
that the value of moving average at that point (for a SELL position). Depending
on the type of position we have, if any of the condition is met, the EA will
consider modifying the position.

The function takes tow input parameters (the _position name_, BUY or SELL, and the _previous bar close price_).

**2.1.8 The Modify function**

```
//+------------------------------------------------------------------+
//| Modifies an open position                                        |
//+------------------------------------------------------------------+
   void Modify(string ptype,double stpl,double tkpf)
     {
       //--- New Stop Loss, new Take profit, Bid price, Ask Price
      double ntp,nsl,pbid,pask;
      long tsp=Trail_point;
       //--- adjust for 5 & 3 digit prices
      if(_Digits==5 || _Digits==3) tsp=tsp*10;
       //--- Stops Level
      long stplevel= mysymbol.StopsLevel();
       //--- Trail point must not be less than stops level
      if(tsp<stplevel) tsp=stplevel;
      if(ptype=="BUY")
        {
          //--- current bid price
         pbid=mysymbol.Bid();
         if(tkpf-pbid<=stplevel*_Point)
           {
            //--- distance to takeprofit less or equal to Stops level? increase takeprofit
            ntp = pbid + tsp*_Point;
            nsl = pbid - tsp*_Point;
           }
         else
           {
            //--- distance to takeprofit higher than Stops level? dont touch takeprofit
            ntp = tkpf;
            nsl = pbid - tsp*_Point;
           }
        }
      else //--- this is SELL
        {
          //--- current ask price
         pask=mysymbol.Ask();
         if(pask-tkpf<=stplevel*_Point)
           {
            ntp = pask - tsp*_Point;
            nsl = pask + tsp*_Point;
           }
         else
           {
            ntp = tkpf;
            nsl = pask + tsp*_Point;
           }
        }
      //--- modify and check result
      if(mytrade.PositionModify(_Symbol,nsl,ntp))
        {
          //--- Request successfully completed
         Alert("An opened position has been successfully modified!!");
         return;
        }
      else
        {
         Alert("The position modify request could not be completed - error: ",
               mytrade.ResultRetcodeDescription());
         return;
        }

     }
```

This function makes use of the
above function ( **checkmodify**) to do its job. It uses the objects of the
[CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) and [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) classes. First of all, we declared four double data
types to hold the new take profit, stop loss, bid price and ask price. Then we
declared a new long data type **tsp** to hold the **Trail\_point** value set at the
input parameters section.

The trail point value ( **tsp**) was then adjusted for 5
and 3 digit prices. We then used the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) object to get the _stops level_
and make sure that the trail point we want to add is not less than the required
stop level. If it is less than stops level, then we will use the stops level
value.

Depending on the position type,
we use the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class object to get the current BID or ASK price as the
case may be. If the difference between the current BID or ASK price and the
initial take profit price is less or equal to the stops level, we decide to
adjust both the stop loss and take profit prices otherwise, we only adjust the
stop loss value.

We then use the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class
object to modify the Stop loss and the take profit for the position. Based on
the trade result return code, a success or failure message is also displayed.

We have finished defining some
user defined functions that will make our job easier. Let us now go ahead to
the EA codes section.

**2.1.9 The OnInit Section**

```
//--- set the symbol name for our SymbolInfo Object
   mysymbol.Name(_Symbol);
// Set Expert Advisor Magic No using our Trade Class Object
   mytrade.SetExpertMagicNumber(EA_Magic);
// Set Maximum Deviation using our Trade class object
   mytrade.SetDeviationInPoints(dev);
//--- Get handle for ADX indicator
   adxHandle=iADX(NULL,0,ADX_Period);
//--- Get the handle for Moving Average indicator
   maHandle=iMA(_Symbol,Period(),MA_Period,0,MODE_EMA,PRICE_CLOSE);
//--- What if handle returns Invalid Handle
   if(adxHandle<0 || maHandle<0)
     {
      Alert("Error Creating Handles for MA, ADX indicators - error: ",GetLastError(),"!!");
      return(1);
     }
   STP = StopLoss;
   TKP = TakeProfit;
//--- Let us handle brokers that offers 5 or 3 digit prices instead of 4
   if(_Digits==5 || _Digits==3)
     {
      STP = STP*10;
      TKP = TKP*10;
     }

//--- Set trade percent
    TPC = TradePct;
    TPC = TPC/100;
//---
```

We decide to set the
current symbol for the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class object. We also set the Expert Advisor magic
number and the deviation (in points) using the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class object. After this
we decide to get the handles for our indicators and display an error if getting
of handles failed.

Next, we decide to adjust the
stop loss and take profit for 3 and 5 digit prices and we also convert the
percentage free account margin to use for trade into percentage.

**2.1.10 The OnDeinit Section**

```
//--- Release our indicator handles
   IndicatorRelease(adxHandle);
   IndicatorRelease(maHandle);
```

Here we decide to release all
the indicator handles.

**2.1.11. The OnTick Section**

```
//--- check if EA can trade
    if (checkTrading() == false)
   {
      Alert("EA cannot trade because certain trade requirements are not meant");
      return;
   }
//--- Define the MQL5 MqlRates Structure we will use for our trade
   MqlRates mrate[];          // To be used to store the prices, volumes and spread of each bar
/*
     Let's make sure our arrays values for the Rates, ADX Values and MA values
     is store serially similar to the timeseries array
*/
// the rates arrays
   ArraySetAsSeries(mrate,true);
// the ADX values arrays
   ArraySetAsSeries(adxVal,true);
// the MA values arrays
   ArraySetAsSeries(maVal,true);
// the minDI values array
   ArraySetAsSeries(minDI,true);
// the plsDI values array
   ArraySetAsSeries(plsDI,true);
```

The first thing we do here is to check and be sure if our EA should trade or not. If the **checktrade**
function returns false, EA will wait for the next tick and make the check
again.

After this we declared a MQL5 [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates)
Structure to get the prices of each bar and then we use the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries)
function to set all the required arrays.

```
//--- Get the last price quote using the SymbolInfo class object function
   if (!mysymbol.RefreshRates())
     {
      Alert("Error getting the latest price quote - error:",GetLastError(),"!!");
      return;
     }

//--- Get the details of the latest 3 bars
   if(CopyRates(_Symbol,_Period,0,3,mrate)<0)
     {
      Alert("Error copying rates/history data - error:",GetLastError(),"!!");
      return;
     }

//--- EA should only check for new trade if we have a new bar
// lets declare a static datetime variable
   static datetime Prev_time;
// lest get the start time for the current bar (Bar 0)
   datetime Bar_time[1];
   //copy the current bar time
   Bar_time[0] = mrate[0].time;
// We don't have a new bar when both times are the same
   if(Prev_time==Bar_time[0])
     {
      return;
     }
//Save time into static varaiable,
   Prev_time = Bar_time[0];
```

We use the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class
object to get the current price quotes and then copy the current bar prices to
the mrates array. Immediately after this we decide to check for the presence of
a new bar.

If we have a new bar, then our EA will proceed to check if a BUY or
SELL condition has been met, otherwise it will wait until we have a new bar.

```
//--- Copy the new values of our indicators to buffers (arrays) using the handle
   if(CopyBuffer(adxHandle,0,0,3,adxVal)<3 || CopyBuffer(adxHandle,1,0,3,plsDI)<3
      || CopyBuffer(adxHandle,2,0,3,minDI)<3)
     {
      Alert("Error copying ADX indicator Buffers - error:",GetLastError(),"!!");
      return;
     }
   if(CopyBuffer(maHandle,0,0,3,maVal)<3)
     {
      Alert("Error copying Moving Average indicator buffer - error:",GetLastError());
      return;
     }
//--- we have no errors, so continue
// Copy the bar close price for the previous bar prior to the current bar, that is Bar 1

   p_close=mrate[1].close;  // bar 1 close price
```

Here, we used the [CopyBuffer](https://www.mql5.com/en/docs/series/copybuffer)
functions to get the buffers of our indicators into arrays and if error occurs
in the process, it will be displayed. The previous bar close price was copied.

```
//--- Do we have positions opened already?
  bool Buy_opened = false, Sell_opened=false;
   if (myposition.Select(_Symbol) ==true)  // we have an opened position
    {
      if (myposition.Type()== POSITION_TYPE_BUY)
       {
            Buy_opened = true;  //It is a Buy
          // Get Position StopLoss and Take Profit
           double buysl = myposition.StopLoss();      // Buy position Stop Loss
           double buytp = myposition.TakeProfit();    // Buy position Take Profit
           // Check if we can close/modify position
           if (ClosePosition("BUY",p_close)==true)
             {
                Buy_opened = false;   // position has been closed
                return; // wait for new bar
             }
           else
           {
              if (CheckModify("BUY",p_close)==true) // We can modify position
              {
                  Modify("BUY",buysl,buytp);
                  return; // wait for new bar
              }
           }
       }
      else if(myposition.Type() == POSITION_TYPE_SELL)
       {
            Sell_opened = true; // It is a Sell
            // Get Position StopLoss and Take Profit
            double sellsl = myposition.StopLoss();    // Sell position Stop Loss
            double selltp = myposition.TakeProfit();  // Sell position Take Profit
             if (ClosePosition("SELL",p_close)==true)
             {
               Sell_opened = false;  // position has been closed
               return;   // wait for new bar
             }
             else
             {
                 if (CheckModify("SELL",p_close)==true) // We can modify position
                 {
                     Modify("SELL",sellsl,selltp);
                     return;  //wait for new bar
                 }
             }
       }
    }
```

We use the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) class
object to select and check if we have an open position for the current symbol.
If a position exists, and it is a BUY, we set **Buy\_opened** to be true and then
use the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) class object to get the _**stop loss**_ and _**take profit**_ of the
position. Using a function we had defined earlier, **ClosePosition**, we checked if
the position can be close. If the function returns true, then the position has
been closed, so we set **Buy\_opened** to false we the initial BUY position has just
been closed. The EA will now wait for a new tick.

However, if the function
returns false, then the position has not been closed. It is now time to check
if we can modify the position. This we achieved by using the function **CheckModify** which we had earlier defined.
If the function returns true, then it means the position can be modified, so we
use the **Modify** function to modify the position.

If, on the other hand, a
position exists and it is a SELL, we set **Sell\_opened** to be trueand use the [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo) class object to get
the stop loss and take profit of the position. We repeated the same step as we
did for the BUY position in order to see if the position can be closed or
modified.

```
      if(checkBuy()==true)
        {
         //--- any opened Buy position?
         if(Buy_opened)
           {
            Alert("We already have a Buy position!!!");
            return;    //--- Don't open a new Sell Position
           }

         double mprice=NormalizeDouble(mysymbol.Ask(),_Digits);                //--- latest ask price
         double stloss = NormalizeDouble(mysymbol.Ask() - STP*_Point,_Digits); //--- Stop Loss
         double tprofit = NormalizeDouble(mysymbol.Ask()+ TKP*_Point,_Digits); //--- Take Profit
         //--- check margin
         if(ConfirmMargin(ORDER_TYPE_BUY,mprice)==false)
           {
            Alert("You do not have enough money to place this trade based on your setting");
            return;
           }
         //--- open Buy position and check the result
         if(mytrade.Buy(Lot,_Symbol,mprice,stloss,tprofit))
         //if(mytrade.PositionOpen(_Symbol,ORDER_TYPE_BUY,Lot,mprice,stloss,tprofit))
           {
               //--- Request is completed or order placed
             Alert("A Buy order has been successfully placed with deal Ticket#:",
                  mytrade.ResultDeal(),"!!");
           }
         else
           {
            Alert("The Buy order request at vol:",mytrade.RequestVolume(),
                  ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                  ", price:", mytrade.RequestPrice(),
                     " could not be completed -error:",mytrade.ResultRetcodeDescription());
            return;
           }
        }
```

Or we can use the PositionOpen function

```
      if(checkBuy()==true)
        {
         //--- any opened Buy position?
         if(Buy_opened)
           {
            Alert("We already have a Buy position!!!");
            return;    //--- Don't open a new Sell Position
           }

         double mprice=NormalizeDouble(mysymbol.Ask(),_Digits);               //--- latest Ask price
         double stloss = NormalizeDouble(mysymbol.Ask() - STP*_Point,_Digits); //--- Stop Loss
         double tprofit = NormalizeDouble(mysymbol.Ask()+ TKP*_Point,_Digits); //--- Take Profit
         //--- check margin
         if(ConfirmMargin(ORDER_TYPE_BUY,mprice)==false)
           {
            Alert("You do not have enough money to place this trade based on your setting");
            return;
           }
         //--- open Buy position and check the result
         //if(mytrade.Buy(Lot,_Symbol,mprice,stloss,tprofit))
         if(mytrade.PositionOpen(_Symbol,ORDER_TYPE_BUY,Lot,mprice,stloss,tprofit))
           {
              //--- Request is completed or order placed
              Alert("A Buy order has been successfully placed with deal Ticket#:",
            mytrade.ResultDeal(),"!!");
           }
         else
           {
            Alert("The Buy order request at vol:",mytrade.RequestVolume(),
                    ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                    ", price:", mytrade.RequestPrice(),
                    " could not be completed -error:",mytrade.ResultRetcodeDescription());
            return;
           }
        }
```

Here, we use the function
**checkbuy** to checkfor a buy setup and if
it returns true, then our BUY trade conditions have been met. If we already
have a BUY position, we don't want to place a new order. We then used the [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo) class object to get the current ASK price and
calculated the Stop loss and Take profit as required.

We also use the **ConfirmMargin** function to check
if the the percentage of the account allowed for placing an order is greater
than the required margin for placing this order. If the function returns true, then we go ahead and place the trade otherwise, we will not place the trade.

Using the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class object, we
placed our order and used the same calls object to get the trade operation
return code. Based on the result of the trade, a message is displayed.

```
      if(checkSell()==true)
        {
         //--- any opened Sell position?
         if(Sell_opened)
           {
            Alert("We already have a Sell position!!!");
            return;    //--- Wait for a new bar
           }

         double sprice=NormalizeDouble(mysymbol.Bid(),_Digits);             //--- latest Bid price
         double ssloss=NormalizeDouble(mysymbol.Bid()+STP*_Point,_Digits);   //--- Stop Loss
         double stprofit=NormalizeDouble(mysymbol.Bid()-TKP*_Point,_Digits); //--- Take Profit
         //--- check margin
         if(ConfirmMargin(ORDER_TYPE_SELL,sprice)==false)
           {
            Alert("You do not have enough money to place this trade based on your setting");
            return;
           }
         //--- Open Sell position and check the result
         if(mytrade.Sell(Lot,_Symbol,sprice,ssloss,stprofit))
         //if(mytrade.PositionOpen(_Symbol,ORDER_TYPE_SELL,Lot,sprice,ssloss,stprofit))
           {
               //---Request is completed or order placed
               Alert("A Sell order has been successfully placed with deal Ticket#:",mytrade.ResultDeal(),"!!");
           }
         else
           {
            Alert("The Sell order request at Vol:",mytrade.RequestVolume(),
                    ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                    ", price:", mytrade.RequestPrice(),
                    " could not be completed -error:",mytrade.ResultRetcodeDescription());
            return;
           }

        }
```

Or we can also used the PositionOpen fucntion:

```
      if(checkSell()==true)
        {
         //--- any opened Sell position?
         if(Sell_opened)
           {
            Alert("We already have a Sell position!!!");
            return;    //--- Wait for a new bar
           }

         double sprice=NormalizeDouble(mysymbol.Bid(),_Digits);             //--- latest Bid price
         double ssloss=NormalizeDouble(mysymbol.Bid()+STP*_Point,_Digits);   //--- Stop Loss
         double stprofit=NormalizeDouble(mysymbol.Bid()-TKP*_Point,_Digits); //--- Take Profit
         //--- check margin
         if(ConfirmMargin(ORDER_TYPE_SELL,sprice)==false)
           {
            Alert("You do not have enough money to place this trade based on your setting");
            return;
           }
         //--- Open Sell position and check the result
         //if(mytrade.Sell(Lot,_Symbol,sprice,ssloss,stprofit))
         if(mytrade.PositionOpen(_Symbol,ORDER_TYPE_SELL,Lot,sprice,ssloss,stprofit))
           {
             //---Request is completed or order placed
             Alert("A Sell order has been successfully placed with deal Ticket#:",mytrade.ResultDeal(),"!!");
           }
         else
           {
            Alert("The Sell order request at Vol:",mytrade.RequestVolume(),
                 ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                 ", price:", mytrade.RequestPrice(),
                   " could not be completed -error:",mytrade.ResultRetcodeDescription());
            return;
           }
        }
```

Just as we did for the BUY, we
used the **Checksell** function to check for a sell setup. If it returns true and
we do not have an already open sell position, we used the **ConfirmMargin**
function to check if we have enough money to open the order. If **ConfirmMargin**
returns true, the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class object is used to place the order and based on
the response from the trade server, the result of the trade is displayed using
the **CTrade** class object functions.

So far we have looked at how we
can use the [Trade class](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) libraries in writing an Expert Advisor. The next thing
is to test our Expert Advisor with the strategy tester and see its performance.

Compile the EA code and then load it in the Strategy Tester.

![Compile report for the EA](https://c.mql5.com/2/2/compile-result.png)

         Figure 3. Expert Advisor compile report

On the GBPUSD Daily chart using
the default settings: Take Profit - 270, Stop Loss - 100 and Trails Point (TP/SL)
\- 32, we have the following results:

![](https://c.mql5.com/2/2/GBPUSD-results.png)

Figure 4. Expert Advisor test report - GBPUSD daily chart

![](https://c.mql5.com/2/2/GBPUSDgraph.png)

Figure 5. Expert Advisor test graph result - GBPUSD daily chart

![](https://c.mql5.com/2/2/GBPUSDpositionmodify.png)

Figure 6. Expert Advisor test report shows modification of open positions - GBPUSD daily chart

![](https://c.mql5.com/2/2/GBPUSDDaily.png)

Figure 7.  Expert Advisor test chart report for GBPUSD daily chart

You are free to test the EA on
any other symbol daily chart with different settings of the Take profit, Stop
loss and the Trail point setting and see what you get.

However, you should understand that this Expert Advisor has been written for test purposes only...

Let us now see how we can use the other classes ( [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo), [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo),
and [CDealInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo)) to get order/deal details.

**2.2 Opening/Deleting**
**a Pending Order**

In this example, we will write a simple Expert Advisor which will place a pending order (BuyStop or SellStop) when we have a buy or Sell setup conditions met respectively.

**2.2.1 Include The Required Classes**

```
//+------------------------------------------------------------------+
//|  Include ALL classes that will be used                           |
//+------------------------------------------------------------------+
//--- The Trade Class
#include <Trade\Trade.mqh>
//--- The PositionInfo Class
#include <Trade\PositionInfo.mqh>
//--- The SymbolInfo Class
#include <Trade\SymbolInfo.mqh>
//--- The OrderInfo Class
#include <Trade\OrderInfo.mqh>
```

We have included the four classes we will be using in this simple Expert Advisor. They have been explained in the examples above.

I will not explain every section of this Expert Advisor as it is similar to the one explained above, however, I will go through the essential part of the Expert Advisor that explains what we want to discuss in this section.

The only thing that is different is that we have decided to declare [MqlRates](https://www.mql5.com/en/docs/constants/structures/mqlrates) mrate\[\] on a global scope.

```
//--- Define the MQL5 MqlRates Structure we will use for our trade
   MqlRates mrate[];     // To be used to store the prices, volumes and spread of each bar
```

Once we have included the [classes](https://www.mql5.com/en/docs/basis/types/classes#class), we must also remember to create objects of each class:

```
//+------------------------------------------------------------------+
//|  CREATE CLASS OBJECTS                                            |
//+------------------------------------------------------------------+
//--- The CTrade Class Object
CTrade mytrade;
//--- The CPositionInfo Class Object
CPositionInfo myposition;
//--- The CSymbolInfo Class Object
CSymbolInfo mysymbol;
//--- The COrderInfo Class Object
COrderInfo myorder;
```

The **CheckBuy()** and **CheckSell()** functions is the same as in the Expert Advisor explained before.

What we want to do here is to place a BUYSTOP order when we have a buy setup and a SELLSTOP order when we have a sell setup.

Let us now go through some of the funсtions we have created to make things easy for us.

**2.2.2 The CountOrders function**

```
//+------------------------------------------------------------------+
//|  Count Total Orders for this expert/symbol                             |
//+------------------------------------------------------------------+
int CountOrders()
  {
   int mark=0;

   for(int i=OrdersTotal()-1; i>=0; i--)
     {
      if(myorder.Select(OrderGetTicket(i)))
        {
         if(myorder.Magic()==EA_Magic && myorder.Symbol()==_Symbol) mark++;
        }
     }
   return(mark);
  }
```

This function is used to get the total pending orders available at a point in time.

We used the object of our class [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) to check the details of order if it is successfully selected with the [myorder.Select()](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo/corderinfoselect) function.

If the Magic return by our class object and the symbol returned is what we are looking for, then the order was placed by our Expert Advisor, so it is counted and stored in the variable **mark.**

**2.2.3 The DeletePending function**

```
//+------------------------------------------------------------------+
//| Checks and Deletes a pending order                                |
//+------------------------------------------------------------------+
bool DeletePending()
  {
   bool marker=false;
//--- check all pending orders
   for(int i=OrdersTotal()-1; i>=0; i--)
     {
      if(myorder.Select(OrderGetTicket(i)))
        {
         if(myorder.Magic()==EA_Magic && myorder.Symbol()==_Symbol)
           {
            //--- check if order has stayed more than two bars time
            if(myorder.TimeSetup()<mrate[2].time)
              {
               //--- delete this pending order and check if we deleted this order successfully?
                if(mytrade.OrderDelete(myorder.Ticket())) //Request successfully completed
                  {
                    Alert("A pending order with ticket #", myorder.Ticket(), " has been successfully deleted!!");
                    marker=true;
                  }
                 else
                  {
                    Alert("The pending order # ",myorder.Ticket(),
                             " delete request could not be completed - error: ",mytrade.ResultRetcodeDescription());
                  }

              }
           }
        }
     }
   return(marker);
  }
```

Just like the countorder
function, this function also makes use of the [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo) class functions to get
the order properties. The function checks for any pending order that has was
setup three bars before (the pending order setup time is less than
mrate\[2\].time) and has not yet been triggered.

If any order falls into that
category, the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class function [OrderDelete](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderdelete) is used to delete the order. This function returns true on success and false if otherwise.

The above two functions are
used immediately after a new bar is formed, before checking for a new trade
setup. We want to be sure we don't have more than three pending orders placed
at every point in time.  To do this we use the following code:

```
// do we have more than 3 already placed pending orders
if (CountOrders()>3)
  {
     DeletePending();
     return;
  }
```

**2.2.4 Placing a Pending Order**

```
   if(checkBuy()==true)
     {
      Alert("Total Pending Orders now is :",CountOrders(),"!!");
      //--- any opened Buy position?
      if(Buy_opened)
        {
         Alert("We already have a Buy position!!!");
         return;    //--- Don't open a new Sell Position
        }
      //Buy price = bar 1 High + 2 pip + spread
      int sprd=mysymbol.Spread();
      double bprice =mrate[1].high + 10*_Point + sprd*_Point;
      double mprice=NormalizeDouble(bprice,_Digits);               //--- Buy price
      double stloss = NormalizeDouble(bprice - STP*_Point,_Digits); //--- Stop Loss
      double tprofit = NormalizeDouble(bprice+ TKP*_Point,_Digits); //--- Take Profit
      //--- open BuyStop order
      if(mytrade.BuyStop(Lot,mprice,_Symbol,stloss,tprofit))
      //if(mytrade.OrderOpen(_Symbol,ORDER_TYPE_BUY_STOP,Lot,0.0,bprice,stloss,tprofit,ORDER_TIME_GTC,0))
        {
         //--- Request is completed or order placed
         Alert("A BuyStop order has been successfully placed with Ticket#:",mytrade.ResultOrder(),"!!");
         return;
        }
      else
        {
         Alert("The BuyStop order request at vol:",mytrade.RequestVolume(),
                 ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
               ", price:", mytrade.RequestPrice(),
                 " could not be completed -error:",mytrade.ResultRetcodeDescription());
         return;
        }
     }
```

Or  we can also use the OrderOpen function to place the BUYSTOP order

```
   if(checkBuy()==true)
     {
      Alert("Total Pending Orders now is :",CountOrders(),"!!");
      //--- any opened Buy position?
      if(Buy_opened)
        {
         Alert("We already have a Buy position!!!");
         return;    //--- Don't open a new Sell Position
        }
      //Buy price = bar 1 High + 2 pip + spread
      int sprd=mysymbol.Spread();
      double bprice =mrate[1].high + 10*_Point + sprd*_Point;
      double mprice=NormalizeDouble(bprice,_Digits);               //--- Buy price
      double stloss = NormalizeDouble(bprice - STP*_Point,_Digits); //--- Stop Loss
      double tprofit = NormalizeDouble(bprice+ TKP*_Point,_Digits); //--- Take Profit
      //--- open BuyStop order
      //if(mytrade.BuyStop(Lot,mprice,_Symbol,stloss,tprofit))
      if(mytrade.OrderOpen(_Symbol,ORDER_TYPE_BUY_STOP,Lot,0.0,bprice,stloss,tprofit,ORDER_TIME_GTC,0))
        {
         //--- Request is completed or order placed
         Alert("A BuyStop order has been successfully placed with Ticket#:",mytrade.ResultOrder(),"!!");
         return;
        }
      else
        {
         Alert("The BuyStop order request at vol:",mytrade.RequestVolume(),
              ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
              ", price:", mytrade.RequestPrice(),
                " could not be completed -error:",mytrade.ResultRetcodeDescription());
         return;
        }
     }
```

In placing our BUYSTOP order,
the open price is the **Bar 1** High + 2pip + spread.

Remember that the price
displayed on the chart is the BID price and in placing long/buy orders you need
the ASK price, that is why we decide to add the spread to the Bar 1 High such
that what we now have is the corresponding Ask price + 2pip. The stop loss and
Take profit has already been defined in the input parameters.

Once we have prepared all the
necessary parameters, we use the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class function **BuyStop** or [OrderOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) to place our
order. The Order Type here is [ORDER\_TYPE\_BUY\_STOP](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) (Buy Stop order). We use the
same price for the limit price but this is not a BuyLimit order. We also set
the order validity time to ORDER\_TIME\_GTC which means orders remains valid
until it is cancelled.

If you use [ORDER\_TIME\_GTC](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_time) or [ORDER\_TIME\_DAY](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_time), there is no
need to specify expiration time, that is why we set the expiration time to 0.

```
   if(checkSell()==true)
     {
      Alert("Total Pending Orders now is :",CountOrders(),"!!");
      //--- any opened Sell position?
      if(Sell_opened)
        {
         Alert("We already have a Sell position!!!");
         return;    //--- Wait for a new bar
        }
      //--- Sell price = bar 1 Low - 2 pip
      double sprice=mrate[1].low-10*_Point;
      double slprice=NormalizeDouble(sprice,_Digits);            //--- Sell price
      double ssloss=NormalizeDouble(sprice+STP*_Point,_Digits);   //--- Stop Loss
      double stprofit=NormalizeDouble(sprice-TKP*_Point,_Digits); //--- Take Profit
      //--- Open SellStop Order
      if(mytrade.SellStop(Lot,slprice,_Symbol,ssloss,stprofit))
      //if(mytrade.OrderOpen(_Symbol,ORDER_TYPE_SELL_STOP,Lot,0.0,slprice,ssloss,stprofit,ORDER_TIME_GTC,0))
        {
         //--- Request is completed or order placed
         Alert("A SellStop order has been successfully placed with Ticket#:",mytrade.ResultOrder(),"!!");
         return;
        }
      else
        {
         Alert("The SellStop order request at Vol:",mytrade.RequestVolume(),
              ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                ", price:", mytrade.RequestPrice(),
                " could not be completed -error:",mytrade.ResultRetcodeDescription());
         return;
        }
     }
```

Or we can also use the OrderOpen function to place the order:

```
   if(checkSell()==true)
     {
      Alert("Total Pending Orders now is :",CountOrders(),"!!");
      //--- any opened Sell position?
      if(Sell_opened)
        {
         Alert("We already have a Sell position!!!");
         return;    //--- Wait for a new bar
        }
      //--- Sell price = bar 1 Low - 2 pip
      double sprice=mrate[1].low-10*_Point;
      double slprice=NormalizeDouble(sprice,_Digits);            //--- Sell price
      double ssloss=NormalizeDouble(sprice+STP*_Point,_Digits);   //--- Stop Loss
      double stprofit=NormalizeDouble(sprice-TKP*_Point,_Digits); //--- Take Profit
      //--- Open SellStop Order
      //if(mytrade.SellStop(Lot,slprice,_Symbol,ssloss,stprofit))
      if(mytrade.OrderOpen(_Symbol,ORDER_TYPE_SELL_STOP,Lot,0.0,slprice,ssloss,stprofit,ORDER_TIME_GTC,0))
        {
         //--- Request is completed or order placed
         Alert("A SellStop order has been successfully placed with Ticket#:",mytrade.ResultOrder(),"!!");
         return;
        }
      else
        {
         Alert("The SellStop order request at Vol:",mytrade.RequestVolume(),
                ", sl:", mytrade.RequestSL(),", tp:",mytrade.RequestTP(),
                ", price:", mytrade.RequestPrice(),
              " could not be completed -error:",mytrade.ResultRetcodeDescription());
         return;
        }
     }
```

Just like the BuyStop order,
the open price is **Bar 1** low + 2pip. Here we don't need to add
spread, since, ordinarily, we need the BID price to place a short/sell orders.

We also use the same [OrderOpen](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade/ctradeorderopen) function or the **SellStop** function to place the SellStop Order. The order
type here is [ORDER\_TYPE\_SELL\_STOP](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type) (Sell Stop order).

Below are the results of our simple Expert Advisor.

![](https://c.mql5.com/2/2/EURUSD-results.png)

Figure 8. The test report for the pending order EA

![](https://c.mql5.com/2/2/EURUSD-graph.png)

Figure 9 - The graph report for the EA

![](https://c.mql5.com/2/2/EURUSDH2chart.png)

Figure 10. The Chart report for the EA

**2.3 Getting Order/Deal**
**Details**

In this example, we will show
how we can get the details of an order once it has been triggered.

At this
stage, it is no more a pending order because it has been triggered and
transformed into a deal.

To fully understand this
procedure, let us look at a journal detail from one of our trades:

![Order processing procedure](https://c.mql5.com/2/2/order_process.png)

Figure 11. Order processing procedure

- **Step 1:** A pending order is placed
waiting for conditions to be met (Pending order)
- **Step 2:** Condition is met, pending order
is triggered, it becomes a deal (Pending order now in history)
- **Step 3:** Deal is performed and we have a
position opened. (Deal is now in history)

**2.3.1 Obtaining Order Properties (History)**

```
//+------------------------------------------------------------------+
//|  Include ALL classes that will be used                           |
//+------------------------------------------------------------------+
//--- The Trade Class
#include <Trade\HistoryOrderInfo.mqh>
//+------------------------------------------------------------------+
//|  CREATE CLASS OBJECT                                             |
//+------------------------------------------------------------------+
//--- The HistoryOrderInfo Class Object
CHistoryOrderInfo myhistory;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Get all orders in History and get their details
   int buystop=0;
   int sellstop=0;
   int buylimit=0;
   int selllimit=0;
   int buystoplimit=0;
   int sellstoplimit=0;
   int buy=0;
   int sell=0;

   int s_started=0;
   int s_placed=0;
   int s_cancelled=0;
   int s_partial=0;
   int s_filled=0;
   int s_rejected=0;
   int s_expired=0;

   ulong o_ticket;
// Get all history records
   if(HistorySelect(0,TimeCurrent())) // get all history orders
     {
      // Get total orders in history
      for(int j=HistoryOrdersTotal(); j>0; j--)
        {
         // select order by ticket
         o_ticket=HistoryOrderGetTicket(j);
         if(o_ticket>0)
           {
            // Set order Ticket to work with
            myhistory.Ticket(o_ticket);
            Print("Order index ",j," Order Ticket is: ",myhistory.Ticket()," !");
            Print("Order index ",j," Order Setup Time is: ",TimeToString(myhistory.TimeSetup())," !");
            Print("Order index ",j," Order Open Price is: ",myhistory.PriceOpen()," !");
            Print("Order index ",j," Order Symbol is: ",myhistory.Symbol() ," !");
            Print("Order index ",j," Order Type is: ", myhistory.Type() ," !");
            Print("Order index ",j," Order Type Description is: ",myhistory.TypeDescription()," !");
            Print("Order index ",j," Order Magic is: ",myhistory.Magic()," !");
            Print("Order index ",j," Order Time Done is: ",myhistory.TimeDone()," !");
            Print("Order index ",j," Order Initial Volume is: ",myhistory.VolumeInitial()," !");
            //
            //
            if(myhistory.Type() == ORDER_TYPE_BUY_STOP) buystop++;
            if(myhistory.Type() == ORDER_TYPE_SELL_STOP) sellstop++;
            if(myhistory.Type() == ORDER_TYPE_BUY) buy++;
            if(myhistory.Type() == ORDER_TYPE_SELL) sell++;
            if(myhistory.Type() == ORDER_TYPE_BUY_LIMIT) buylimit++;
            if(myhistory.Type() == ORDER_TYPE_SELL_LIMIT) selllimit++;
            if(myhistory.Type() == ORDER_TYPE_BUY_STOP_LIMIT) buystoplimit++;
            if(myhistory.Type() == ORDER_TYPE_SELL_STOP_LIMIT) sellstoplimit++;

            if(myhistory.State() == ORDER_STATE_STARTED) s_started++;
            if(myhistory.State() == ORDER_STATE_PLACED) s_placed++;
            if(myhistory.State() == ORDER_STATE_CANCELED) s_cancelled++;
            if(myhistory.State() == ORDER_STATE_PARTIAL) s_partial++;
            if(myhistory.State() == ORDER_STATE_FILLED) s_filled++;
            if(myhistory.State() == ORDER_STATE_REJECTED) s_rejected++;
            if(myhistory.State() == ORDER_STATE_EXPIRED) s_expired++;
           }
        }
     }
// Print summary
   Print("Buy Stop Pending Orders : ",buystop);
   Print("Sell Stop Pending Orders: ",sellstop);
   Print("Buy Orders : ",buy);
   Print("Sell Orders: ",sell);
   Print("Total Orders in History is :",HistoryOrdersTotal()," !");

   Print("Orders type summary");
   Print("Market Buy Orders: ",buy);
   Print("Market Sell Orders: ",sell);
   Print("Pending Buy Stop: ",buystop);
   Print("Pending Sell Stop: ",sellstop);
   Print("Pending Buy Limit: ",buylimit);
   Print("Pending Sell Limit: ",selllimit);
   Print("Pending Buy Stop Limit: ",buystoplimit);
   Print("Pending Sell Stop Limit: ",sellstoplimit);
   Print("Total orders:",HistoryOrdersTotal()," !");

   Print("Orders state summary");
   Print("Checked, but not yet accepted by broker: ",s_started);
   Print("Accepted: ",s_placed);
   Print("Canceled by client: ",s_cancelled);
   Print("Partially executed: ",s_partial);
   Print("Fully executed: ",s_filled);
   Print("Rejected: ",s_rejected);
   Print("Expired: ",s_expired);
  }
```

This is just a simple script
that shows how to obtain the details of orders in our History records. We included the [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo) class and created an object of the class.

We now use the object to get the details of the orders.

![The result for the history-order script](https://c.mql5.com/2/2/history-order-result.png)

Figure 12. History order script result

**2.3.2 Obtaining Deal Properties (History)**

```
//+------------------------------------------------------------------+
//|  Include ALL classes that will be used                           |
//+------------------------------------------------------------------+
//--- The CDealInfo Class
#include <Trade\DealInfo.mqh>
//+------------------------------------------------------------------+
//|  Create class object                                             |
//+------------------------------------------------------------------+
//--- The CDealInfo Class Object
CDealInfo mydeal;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- Get all deals in History and get their details
    int buy=0;
    int sell=0;
    int deal_in=0;
    int deal_out=0;
    ulong d_ticket;
    // Get all history records
    if (HistorySelect(0,TimeCurrent()))
    {
      // Get total deals in history
      for (int j=HistoryDealsTotal(); j>0; j--)
      {
         // select deals by ticket
         if (d_ticket = HistoryDealGetTicket(j))
         {
          // Set Deal Ticket to work with
          mydeal.Ticket(d_ticket);
          Print("Deal index ", j ," Deal Ticket is: ", mydeal.Ticket() ," !");
          Print("Deal index ", j ," Deal Execution Time is: ", TimeToString(mydeal.Time()) ," !");
          Print("Deal index ", j ," Deal Price is: ", mydeal.Price() ," !");
          Print("Deal index ", j ," Deal Symbol is: ", mydeal.Symbol() ," !");
          Print("Deal index ", j ," Deal Type Description is: ", mydeal.TypeDescription() ," !");
          Print("Deal index ", j ," Deal Magic is: ", mydeal.Magic() ," !");
          Print("Deal index ", j ," Deal Time is: ", mydeal.Time() ," !");
          Print("Deal index ", j ," Deal Initial Volume is: ", mydeal.Volume() ," !");
          Print("Deal index ", j ," Deal Entry Type Description is: ", mydeal.EntryDescription() ," !");
          Print("Deal index ", j ," Deal Profit is: ", mydeal.Profit() ," !");
          //
          if (mydeal.Entry() == DEAL_ENTRY_IN) deal_in++;
          if (mydeal.Entry() == DEAL_ENTRY_OUT) deal_out++;
          if (mydeal.Type() == DEAL_TYPE_BUY) buy++;
          if (mydeal.Type() == DEAL_TYPE_SELL) sell++;
         }
      }
    }
    // Print Summary
    Print("Total Deals in History is :", HistoryDealsTotal(), " !");
    Print("Total Deal Entry IN is : ", deal_in);
    Print("Total Deal Entry OUT is: ", deal_out);
    Print("Total Buy Deal is : ", buy);
    Print("Total Sell Deal is: ", sell);
  }
```

This is also a simple script
that shows how to obtain the details of our deals records.

![The result for the deal script](https://c.mql5.com/2/2/history-deal-result.png)

Figure 13. The history deal script result

### Conclusion

In this article, we have been
able to look at the main functions of the Standard [Trade Class libraries](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) and have demonstrated how some of these functionalities can be used in writing
Expert Advisors which implements position modifying, pending order placing and
deletion and verifying of Margin before placing a trade.

We have also
demonstrated how they can be used to obtain order and deal details. There are some of these functions we did not use in the course
of writing our Expert Advisor, depending on the type of trading strategy you
employ, you may use more or less than we have used in this example.

It will be a
good idea to revise the description section for the various functions and see
how you can make use of them in writing your own Expert Advisor.

The [Standard class\\
libraries](https://www.mql5.com/en/docs/standardlibrary) are meant to make life easier for both traders and developers, so
make sure you use them.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/138.zip "Download all attachments in the single ZIP archive")

[trade\_classes\_test.mq5](https://www.mql5.com/en/articles/download/138/trade_classes_test.mq5 "Download trade_classes_test.mq5")(20.12 KB)

[history\_deal.mq5](https://www.mql5.com/en/articles/download/138/history_deal.mq5 "Download history_deal.mq5")(3.26 KB)

[history\_order.mq5](https://www.mql5.com/en/articles/download/138/history_order.mq5 "Download history_order.mq5")(4.98 KB)

[mql5\_standardclass\_ea.mq5](https://www.mql5.com/en/articles/download/138/mql5_standardclass_ea.mq5 "Download mql5_standardclass_ea.mq5")(20.08 KB)

[mql5\_standardclass\_ea\_stoporders.mq5](https://www.mql5.com/en/articles/download/138/mql5_standardclass_ea_stoporders.mq5 "Download mql5_standardclass_ea_stoporders.mq5")(13.71 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Guide to Testing and Optimizing of Expert Advisors in MQL5](https://www.mql5.com/en/articles/156)
- [Writing an Expert Advisor Using the MQL5 Object-Oriented Programming Approach](https://www.mql5.com/en/articles/116)
- [Step-By-Step Guide to writing an Expert Advisor in MQL5 for Beginners](https://www.mql5.com/en/articles/100)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2546)**
(38)


![Hady Candra](https://c.mql5.com/avatar/2013/1/5109639E-8C5B.png)

**[Hady Candra](https://www.mql5.com/en/users/dasimplify)**
\|
21 Mar 2019 at 13:28

great article.

Thank you for writing this great article.

let me correct this code.....

this code was typo:

[![typo](https://c.mql5.com/3/273/SS-2019-03-21_192537__1.jpg)](https://c.mql5.com/3/273/SS-2019-03-21_192537.jpg "https://c.mql5.com/3/273/SS-2019-03-21_192537.jpg")

the correct one is:

### **myposition.PositionType()**

![Samuel Olowoyo](https://c.mql5.com/avatar/2010/10/4CCB0815-4DCE.jpg)

**[Samuel Olowoyo](https://www.mql5.com/en/users/olowsam)**
\|
4 Jun 2019 at 14:35

**Hady Candra:**

great article.

Thank you for writing this great article.

let me correct this code.....

this code was typo:

the correct one is:

### **myposition.PositionType()**

Hello Hady,

Thanks for your observation. The article is being updated. It will reflect the corrected version once the update is completed.

Thanks once again.

![Pierre Rougier](https://c.mql5.com/avatar/2018/4/5AE5ECFE-3BFA.jpg)

**[Pierre Rougier](https://www.mql5.com/en/users/pierre8r)**
\|
1 Aug 2019 at 11:57

Hello,

All comments in the [source codes](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development") of this article are in Cyrillic characters, incomprehensible to me.

Is it possible to have the same source codes with comments in English?

Thank you.

Pierre


![mindful FX UG (haftungsbeschraenkt)](https://c.mql5.com/avatar/2016/6/576D8CA5-4235.jpg)

**[Cristof Ensslin](https://www.mql5.com/en/users/mindfulfx)**
\|
3 Nov 2020 at 23:49

Samuel: Thank you for this wonderful article. It helped me a great deal!


![Harold Moody Campbell](https://c.mql5.com/avatar/2024/6/6669c56a-a713.jpg)

**[Harold Moody Campbell](https://www.mql5.com/en/users/hcampbell)**
\|
26 Mar 2025 at 14:52

Hi Samuel,

I found this article very interesting. I will pin the Tab in my browser to use as a reference. Thanks

![Technical Analysis: How Do We Analyze?](https://c.mql5.com/2/0/analysis_charts.png)[Technical Analysis: How Do We Analyze?](https://www.mql5.com/en/articles/174)

This article briefly describes the author's opinion on redrawing indicators, multi-timeframe indicators and displaying of quotes with Japanese candlesticks. The article contain no programming specifics and is of a general character.

![Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://c.mql5.com/2/0/anufrenko_ava.png)[Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://www.mql5.com/en/articles/535)

The risky development of Alexander Anufrenko (Anufrenko321) had been featured among the top three of the Championship for three weeks. Having suffered a catastrophic Stop Loss last week, his Expert Advisor lost about $60,000, but now once again he is approaching the leaders. In this interview the author of this interesting EA is describing the operating principles and characteristics of his application.

![Designing and implementing new GUI widgets based on CChartObject class](https://c.mql5.com/2/0/Design_Widgets_MQL5.png)[Designing and implementing new GUI widgets based on CChartObject class](https://www.mql5.com/en/articles/196)

After I wrote a previous article on semi-automatic Expert Advisor with GUI interface it turned out that it would be desirable to enhance interface with some new functionalities for more complex indicators and Expert Advisors. After getting acquainted with MQL5 standard library classes I implemented new widgets. This article describes a process of designing and implementing new MQL5 GUI widgets that can be used in indicators and Expert Advisors. The widgets presented in the article are CChartObjectSpinner, CChartObjectProgressBar and CChartObjectEditTable.

![Building interactive semi-automatic drag-and-drop Expert Advisor based on predefined risk and R/R ratio](https://c.mql5.com/2/0/z5491084X.png)[Building interactive semi-automatic drag-and-drop Expert Advisor based on predefined risk and R/R ratio](https://www.mql5.com/en/articles/192)

Some traders execute all their trades automatically, and some mix automatic and manual trades based on the output of several indicators. Being a member of the latter group I needed an interactive tool to asses dynamically risk and reward price levels directly from the chart. This article will present a way to implement an interactive semi-automatic Expert Advisor with predefined equity risk and R/R ratio. The Expert Advisor risk, R/R and lot size parameters can be changed during runtime on the EA panel.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/138&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062779274164873401)

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