---
title: Orders, Positions and Deals in MetaTrader 5
url: https://www.mql5.com/en/articles/211
categories: Trading, Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T18:23:06.684947
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/211&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069394210995504216)

MetaTrader 5 / Trading


### Trading terms

The ultimate goal of a trader is to extract profits through the means of trading operations on the financial markets. This article describes the terms and processes of the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading platform, the knowledge of which is necessary for a proper understanding of the work of [trade functions](https://www.mql5.com/en/docs/trading) of the MQL5 language.

- [Orders](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties) — are the [trade operation](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) [requests](https://www.mql5.com/en/docs/constants/structures/mqltraderequest), **received** by the trading server, formulated in compliance to the MetaTrader 5 platform requirements. If the request is incorrect, it will not appear on the trading platform in the form of an order. Orders can be of immediate execution, such as to buy or sell a certain volume, at the current market price on a specified financial instrument. Another type of orders - are pending orders, which contain an order to commit a [trading operation](https://www.mql5.com/en/docs/constants/tradingconstants/enum_trade_request_actions) under the presence of certain condition. Pending orders may also contain a time restriction on their actions - the order expiration date.

![Orders and positions in the MetaTrader 5 terminal](https://c.mql5.com/2/2/1.gif)

The placed (pending) orders, that are waiting for the conditions of their execution or cancellation, are shown on the " [Trade"](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface") tab in the terminal. These orders can be modified or canceled. The placing, cancellation, and modification of orders is done using the [OrderSend(](https://www.mql5.com/en/docs/trading/ordersend) [)](https://www.mql5.com/en/docs/trading/ordersend) function. If the order was canceled or reached the order expiration time, or if the order was executed, it moves into the orders history. Executed and canceled orders are shown in the " [History"](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface") tab of the client terminal. Orders from the history are not available for modification.

- [Deals](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties) \- are the result of an execution of an order (a command to commit the deal). Each deal is based on a one particular order, but a single order can generate a set of deals. For example, an order to buy 10 lots may be carried out by several successive deals with partial filling. The deals are always stored in the trading history and cannot be modified. In the terminal, the deals are displayed in the ["History"](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface") tab.

![Transactions in the MetaTrader 5 terminal](https://c.mql5.com/2/2/2.gif)

- [Positions](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties) are the contracts, bought or sold on a financial instrument. A long position (Long) is formed as a result of buys in anticipation of a price increase, a short position (Short) is the result of the sale of asset, in the anticipation of future price decrease. For each account, for every financial instrument, there can be only one position. For each symbol, at any given time, there can be only one open position - long or short.

![Historical orders in the MetaTrader 5 terminal](https://c.mql5.com/2/2/3.gif)

The position volume may increase as a result of a new trading operation in the same direction. Meaning that the volume of the long positions will be increased after the new buy (Buy deal), and will be decreased after the sale (Sell deal). The position is closed, if the volume of commitments became equal to zero as a result of trading operation. Such operation is called closing the position.


**Note:** Active orders and positions are always displayed on the " [Trade](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface")" tab, and the deals and orders from the history are always reflected in the "History" tab. The active order from the "Trade" tab should not be confused with the historical orders from the "History" tab.

### How the terminal receives and stores the trade information from the server

The terminal stores the trading history in a special base, and receives only the missing history of deals and completed orders on the trading account, at each [connection to the trading server](https://www.metatrader5.com/en/terminal/help/startworking/authorization "https://www.metatrader5.com/en/terminal/help/startworking/authorization"). This is done to save up on traffic. When closing the MetaTrader 5 client terminal or changing the current active account, the entire history is recorded on the hard disk and read from at the time of the next launch of the terminal.

All databases are recorded to the disk in an encrypted form and the encryption key is dependent to the computer on which the terminal is installed. This protects the user of the terminal against unauthorized access to his data, in case of copying.

During the connection to the account, the terminal loads the saved account base, with the history of the account, and sends the trading server a request to synchronize its own database of history with the history of the account on the trading server. Further, after a successful connection to the account, the trading server sends the terminal a report about the ongoing trading events, related to this account.

Trading event are the following changes in the account:

- withdrawal and balance operations.
- charge of commissions, swaps and taxes.
- placing, deletion and modification of orders.
- the execution of deals based on orders.
- opening and closing of positions.
- change in the volume and direction of positions.

In the case of termination of the connection with the trading server, the terminal periodically attempts to reconnect. After the reconnection with the server, the terminal requests all of the recent changes in the trading history to maintain the integrity of data in its own database of the history.

The trading history, displayed in the "History" tab of the terminal, is taken from the base of the terminal history, and the changes of the period, displayed in the terminal of history can only increase the range of the history, stored in this database. Decreasing the period of the displayed history does not lead to a physical removal of the history from the base of the terminal.

![Installation of the interval of the displayed trading history](https://c.mql5.com/2/2/2010-11-19_16_26_47.png)

This means that the installation of a shorter interval of the displayed history, does not reduce the depth of the stored trading history. But if we specify a wider interval range, for the display in the "History" tab, then such an action could lead to a request, from the trading server, of a more profound history, if the terminal's own base does not yet have the requested data for that period.

The general scheme of interaction between the terminal and the MetaTrader 5 trading server is demonstrated at following Figure:

![](https://c.mql5.com/2/2/5__1.gif)

The client terminal sends a synchronization request to its own trading history base, during the start of the terminal, during the reconnection with the server after a connection failure, during a switch from one account to another, and during the direct request for the missing trading history.

In its turn, the trading server independently, without any requests from the terminal, sends a client messages about the trading events, taking place on the account: the changes of the state of orders and positions, the conduction of deals based on orders, the charging of commissions, the balance and withdrawal of money, and so on.

### Access to the trading history from the MQL5-program

The terminal can operate simultaneously with a number of indicators, scripts, and EAs, and all of these programs can request the information they need about trading: orders, deals, and positions. The direct work of the mql5-program with the database of the terminal is excluded, due to the considerations of overall stability, security and performance.

Each mql5-program **by request** receives for its work a "model" of the trading environment in its cache. A cache is a special area of memory for a fast access to data. For example, before beginning processing the order, the order needed must be obtained in the cache of the mql5-program. All of the further work, when referring to the order, will be made with the cached copy of that order.

The work with the positions, deals, and orders from the history is carried out in a similar way. The general scheme of obtaining the trading information from the MQL5 program is shown at figure:

![](https://c.mql5.com/2/2/6__1.gif)

Before the data about the trading history becomes available for the processing of mql5-program, they must be requested from the terminal database. After the request, the obtained data will be placed in its own cache of the mql5-program.

**Note:** the data in the cache is not automatically synchronized with the terminal database, and therefore, it must be constantly updated to maintain an appropriate status of the data in the cache.

There is a possibility of consequences if the cache is used improperly.

- if the data requested could not be obtained, the cache will be empty and will not contain the necessary data.
- if the data in the cache required updates, but the updating was not requested, then working with such data can lead to unpredictable results. For example, the data on the current position has not been updated, and the program does not know anything about the open position for the given symbol and about the growing loss for it.

### The function for working with the cache

The trading history may contain thousands of executed orders and deals that are not needed for current work of the mql5-program. Therefore, the work with cache is built on the principle of requests the cache always contains the information that was uploaded at the last connection to the database of the terminal. If you need to obtain the whole history of orders and deals, you need to explicitly request it by specifying the desired interval.

For each type of information, an independent cache is formed. The data about the orders is stored in the order's cache, the information about the positions is stored in the position's cache, the data on deals and orders is stored in the respective instances of the cache's history.

Before requesting the information from the cache, it needs to be filled.

**Note:** Any request for filling the cache previously clears it, regardless of the result of the requests' execution.

![](https://c.mql5.com/2/2/7.gif)

The trading functions can be separated into two categories: the functions for filling the cache and the functions for reading the information from the cache.

### The function for filling the cache

For processing the trading history, it must first be obtained and located in the appropriate cache. Functions that form a cache can be divided into two subgroups.

The function for filling the trading cache (active orders and positions):

- The [OrderSelect(ticket)](https://www.mql5.com/en/docs/trading/orderselect) \- copies the active order **by its ticket** (from the terminal base) into the cache of the current order for the further request of its properties using the OrderGetDouble(), OrderGetInteger() and OrderGetString() functions.
- The [OrderGetTicket(index)](https://www.mql5.com/en/docs/trading/ordergetticket) \- copies, from the terminal base of the active order, **by its index in the orders list of orders** terminal base into the cache of the current orders for further request to the properties using the OrderGetDouble(), OrderGetInteger() and OrderGetString() functions. The total number of orders in the base of the terminal can be obtained using the [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal) function.
- The [PositionSelect(symbol)](https://www.mql5.com/en/docs/trading/positionselect) \- copies the open position **by the symbol name** (from the base of the terminal) into the cache for the further request of its properties using the PositionGetDouble(), PositionGetInteger() and PositionGetString() functions.
- The [PositionGetSymbol(index)](https://www.mql5.com/en/docs/trading/positiongetsymbol) \- copies the open position **by its index in the position list** (from the base of the terminal) of the terminal base into the cache for the further request of its properties using the PositionGetDouble(), PositionGetInteger() and PositionGetString() functions. The total number of positions in the base of the terminal can be obtained by the [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) function.

The function of filling the history cache:

- The [HistoryOrderSelect(ticket)](https://www.mql5.com/en/docs/trading/historyorderselect) \- copies the history order **by its ticket** into the cache of the history orders (from the base of the terminal) for the further calls to its properties by the HistoryOrderGetDouble(), HistoryOrderGetInteger() and HistoryOrderGetString() functions.
- The [HistoryDealSelect(ticket)](https://www.mql5.com/en/docs/trading/historydealselect) \- copies the deal **by its ticket** into the deals cache (from the base of the terminal) for the further calls to its properties by the HistoryDealGetDouble(), HistoryDealGetInteger() and HistoryDealGetString() functions .

We need to separately consider the two functions, which affect the available, in the cache, **trading history in general**:

- The [HistorySelect(start, end)](https://www.mql5.com/en/docs/trading/historyselect) \- fills the history cache with deals and orders for the specified interval of the server's time. From the results of the execution of this function, depends the values, that are returned from HistoryDealsTotal() and HistoryOrdersTotal().
- The [HistorySelectByPosition (position\_ID)](https://www.mql5.com/en/docs/trading/historyselectbyposition) \- fills the history cache with deals and orders, having the specified identifier position. The result of the execution of this function, also affects the HistoryDealsTotal() and HistoryOrdersTotal().

**OrderSelect and OrderGetTicket**

The [OrderSelect(ticket)](https://www.mql5.com/en/docs/trading/orderselect) and [OrderGetTicket()](https://www.mql5.com/en/docs/trading/ordergetticket) general functions work in the same way, - they fill the cache of active orders with one single order. The [OrderSelect(ticket)](https://www.mql5.com/en/docs/trading/orderselect) is intended for the case where a ticket order is known in advance. The OrderGetTicket(), in conjunction with OrdersTotal() allows for the examination of all of the available orders in the base terminal of orders.

After a call to any of these functions, the cache of the active orders contains the information of only one order, if the order is successfully selected. Otherwise, there is nothing in the cache of active orders. The result of the execution of the function [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal) does not change - it always returns the actual number of active orders in the base of the terminal, regardless of whether the cache is full.

**PositionSelect and PositionGetSymbol**

Just like for the orders, these two functions also work in the same way for positions - they fill the cache of positions with a single position. The [PositionGetSymbol(index)](https://www.mql5.com/en/docs/trading/positiongetsymbol) requires the number in the list of the positions base, as a parameter, and the [PositionSelect(symbol)](https://www.mql5.com/en/docs/trading/positionselect) fills the cache based on the symbol name, on which the position is opened. The name of the symbol, in turn, can be obtained by the PositionGetSymbol(index) function.

After performing any of these functions, the cache of positions contains data only on one position, if the function is executed successfully. Otherwise, there is nothing in the cache of positions. The result of the execution of the [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) function does not depend on whether the cache is filled, - it always returns the actual number of open positions in the base terminal for all symbols.

**HistoryOrderSelect**

The [HistoryOrderSelect(ticket)](https://www.mql5.com/en/docs/trading/historyorderselect) chooses into the cache the historical order from the base of the terminal by its ticket. The function is intended for being used when the ticket of the required order is known in advance.

If the execution is successful, the cache will contain a single order, and the [HistoryOrdersTotal()](https://www.mql5.com/en/docs/trading/historyorderstotal) function return a single unit. Otherwise, the cache of historical orders will be empty and the HistoryOrdersTotal() function will return a zero.

**HistoryDealSelect**

The [HistoryDealSelect(ticket)](https://www.mql5.com/en/docs/trading/historydealselect) selects the deal from the base terminal based by its ticket. The function is intended for being used when the ticket of the deal is known in advance.

If the execution is successful, the cache will contain a single deal, and the [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) function will return 1. Otherwise, the cache of deal will be empty and the HistoryDealsTotal() function will return a zero.

### The function for obtaining information from the cache

Before requesting information about the properties of the position, deal or order, it is necessary to update the corresponding cache of the mql5-program. This is due to the fact that the requested information may have already been updated, and this means that the copy, stored in the cache, is already outdated.

- **Orders**

In order to obtain information on active orders, it must first be copied into the cache of active orders of one of the two functions: OrderGetTicket() or OrderSelect(). It is for the order, which is stored in the cache, that the property values will be given out, when the corresponding functions are called:

1. OrderGetDouble(type\_property)
2. OrderGetInteger(type\_property)
3. OrderGetString(type\_property)

These functions obtain all of the data from the cache, therefore, in order to guarantee the obtainment of accurate data for the order, it is recommended to call the function that fills the cache.

- **Positions**

For obtaining the information about a position, it must be previously selected and copied into the cache, using one of the two functions: PositionGetSymbol or PositionSelect. It is from this cache, that the property values of position will be given out, when the corresponding functions are called:

1. PositionGetDouble(type\_property)
2. PositionGetInteger(type\_property)
3. PositionGetString(type\_property)

Since these functions receive all of their data from the cache, then in order to guarantee the obtainment of accurate data for the position, it is recommended to call the function that fills the cache of positions.

- **Historical orders**

For obtaining information about an order from the history, it is required to first create the cache of historical orders, using one of the three functions: HistorySelect(start, end), HistorySelectByPosition() or HistoryOrderSelect(ticket). If the implementation is successful, the cache will store the number of orders, returned by the [HistoryOrdersTotal()](https://www.mql5.com/en/docs/trading/historyorderstotal) function. The access to the properties of these orders is carried out by each element on the ticket, using the appropriate function:

1. HistoryOrderGetDouble(ticket\_order, type\_property)
2. HistoryOrderGetInteger(ticket\_order, type\_property)
3. HistoryOrderGetString(ticket\_order, type\_property)

The ticket of the historical order can be found out using the [HistoryOrderGetTicket(index)](https://www.mql5.com/en/docs/trading/historyordergetticket) function, by its index in the cache of historical orders. In order to have a guaranteed receipt of accurate data on the order, it is recommended to call the function that fills the cache of historical orders.

- **Deals**

For obtaining information about a specific deal in the history, it is needed to first create the deals cache, using one of the three functions: HistorySelect (start, end), HistorySelectByPosition() or HistoryDealSelect (ticket). If the function implementation is successful, the cache will store the deal in the amount returned by the function [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal). Access to the properties of these deals is carried out, based on the ticket, using the appropriate functions:

1. HistoryDealGetDouble(ticket\_deals, type\_property)
2. HistoryDealGetInteger(ticket\_deals, type\_property)
3. HistoryDealGetString(ticket\_deals, type\_property)

The ticket of the deals can be obtained, using the [HistoryDealGetTicket(index)](https://www.mql5.com/en/docs/trading/historydealgetticket) function, by its index in the cache of deals. In order to have a guaranteed receipt of accurate data about the deal, it is recommended to call the function that fills the deals cache.

**The function for obtaining the ticket from the cache history**

The [HistoryOrderGetTicket (index)](https://www.mql5.com/en/docs/trading/historyordergetticket) return the ticket of the historical order, by its index **from the cache** of the historical orders (not from the terminal base!). The obtained ticket can be used in the [HistoryOrderSelect (ticket)](https://www.mql5.com/en/docs/trading/historyorderselect) function, which **clears** the cache and **re-fill** it with only **one** order, in the case of success. Recall that the value, returned from HistoryOrdersTotal() depends on the number of orders in the cache.

The [HistoryDealGetTicket(index)](https://www.mql5.com/en/docs/trading/historydealgetticket) returns the ticket of the deal by its index **from the cache** of deals. The ticket of the deal can be used by the function [HistoryDealSelect(ticket)](https://www.mql5.com/en/docs/trading/historydealselect), which **clears** the cache, and **re-fills** the cache with only one deal, in the case of success. The value, returned by the HistoryDealsTotal() function depends on the number of deals in the cache.

**Note:** Before calling the HistoryOrderGetTicket (index) and HistoryDealGetTicket (index) functions, you need to fill the history cache with historical orders and deals in a **sufficient** volume. To do this, use one of the functions: HistorySelect (start, end), HistorySelectByPosition (position\_ID), HistoryOrderSelect (ticket), and HistoryDealSelect (ticket).

### Obtaining information using active orders

Checking the current active orders is a standard procedure. If it is needed to obtain information about some specific order, then, **knowing its ticket**, this can be done using the function [OrderSelect(ticket)](https://www.mql5.com/en/docs/trading/orderselect).

```
bool selected=OrderSelect(ticket);
if(selected)
  {
   double price_open=OrderGetDouble(ORDER_PRICE_OPEN);
   datetime time_setup=OrderGetInteger(ORDER_TIME_SETUP);
   string symbol=OrderGetString(ORDER_SYMBOL);
   PrintFormat("Ордер #%d for %s was set at %s",ticket,symbol,TimeToString(time_setup));
  }
else
  {
   PrintFormat("Error selecting order with ticket %d. Error %d",ticket, GetLastError());
  }
```

In the above example, it is assumed that the ticket of the order is known in advance, for example, it is obtained from [global variable](https://www.mql5.com/en/docs/basis/variables/global). In general cases, however, the ticket information is absent, and thus we need to turn to the help of the [OrderGetTicket(index)](https://www.mql5.com/en/docs/trading/ordergetticket) function, which also selects one order and places it into the cache, but only the order number, in the list of current orders, needs to be specified as the parameter.

The overall algorithm for working with orders (analogous with deals and positions) is the following:

1. Obtain the total number of orders, using the [OrdersTotal()](https://www.mql5.com/en/docs/trading/orderstotal) function.
2. Organize the loop through a search of all of the orders, by their indexes in the list.
3. Copy, one by one, each order into the cache, using the OrderGetTicket() function.
4. Obtain the correct order data from the cache, using the OrderGetDouble(), OrderGetInteger(), and OrderGetString() functions. If needed, analyze the obtained data and take the appropriate actions.

Here is a brief example of such an algorithm:

```
input long my_magic=555;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- obtain the total number of orders
   int orders=OrdersTotal();
//--- scan the list of orders
   for(int i=0;i<orders;i++)
     {
      ResetLastError();
      //--- copy into the cache, the order by its number in the list
      ulong ticket=OrderGetTicket(i);
      if(ticket!=0)// if the order was successfully copied into the cache, work with it
        {
         double price_open  =OrderGetDouble(ORDER_PRICE_OPEN);
         datetime time_setup=OrderGetInteger(ORDER_TIME_SETUP);
         string symbol      =OrderGetString(ORDER_SYMBOL);
         long magic_number  =OrderGetInteger(ORDER_MAGIC);
         if(magic_number==my_magic)
           {
            //  process the order with the specified ORDER_MAGIC
           }
         PrintFormat("Order #%d for %s was set out %s, ORDER_MAGIC=%d",ticket,symbol,TimeToString(time_setup),magic_number);
        }
      else         // call OrderGetTicket() was completed unsuccessfully
        {
         PrintFormat("Error when obtaining an order from the list to the cache. Error code: %d",GetLastError());
        }
     }
  }
```

### Obtaining the information on open positions

The constant monitoring of open positions is not just a standard procedure, but it should certainly be implemented in each instance. For obtaining information on specific positions, it is sufficient enough to know the name of the instrument by which it is open. To do this, use the function [PositionSelect(symbol)](https://www.mql5.com/en/docs/trading/positionselect). For those cases, in which the EA is working on only one symbol (on the symbol of the chart, to which it is attached), the name of the symbol can be obtained by a [Symbol()](https://www.mql5.com/en/docs/check/symbol) function or from a predefined variable [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol).

```
//--- we will look for the position by the symbol of the chart, on which the EA is working
   string symbol=Symbol();
//--- attempt to get the position
   bool selected=PositionSelect(symbol);
   if(selected) // if the position is selected
     {
      long pos_id            =PositionGetInteger(POSITION_IDENTIFIER);
      double price           =PositionGetDouble(POSITION_PRICE_OPEN);
      ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      long pos_magic         =PositionGetInteger(POSITION_MAGIC);
      string comment         =PositionGetString(POSITION_COMMENT);
      PrintFormat("Position #%d by %s: POSITION_MAGIC=%d, price=%G, type=%s, commentary=%s",
                 pos_id, symbol, pos_magic, price,EnumToString(type), comment);
     }

   else        // if selecting the position was unsuccessful
     {
      PrintFormat("Unsuccessful selection of the position by the symbol %s. Error",symbol,GetLastError());
     }
  }
```

In a general case, the information on the symbol can be obtained, using the [PositionGetSymbol (index)](https://www.mql5.com/en/docs/trading/positiongetsymbol) function, which selects one position and places it into the cache. As a parameter, it is necessary to specify the index of position in the list of open positions. This is best done through a search of all positions in the loop.

The overall algorithm for working with positions:

1. Obtain the total number of positions, using the [PositionsTotal()](https://www.mql5.com/en/docs/trading/positionstotal) function.
2. Organize the loop through searching all of the positions by their indexes in the list.
3. Copy, one by one, each position into the cache, using the PositionGetSymbol() function.
4. Obtain the required position data from the cache using the PositionGetDouble(), PositionGetInteger(), and PositionGetString() functions. If needed, analyze the obtained data and take the appropriate actions.

An example of such an algorithm:

```
#property script_show_inputs

input long my_magic=555;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- obtain the total number of positions
   int positions=PositionsTotal();
//--- scan the list of orders
   for(int i=0;i<positions;i++)
     {
      ResetLastError();
      //--- copy into the cache, the position by its number in the list
      string symbol=PositionGetSymbol(i); //  obtain the name of the symbol by which the position was opened
      if(symbol!="") // the position was copied into the cache, work with it
        {
         long pos_id            =PositionGetInteger(POSITION_IDENTIFIER);
         double price           =PositionGetDouble(POSITION_PRICE_OPEN);
         ENUM_POSITION_TYPE type=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         long pos_magic         =PositionGetInteger(POSITION_MAGIC);
         string comment         =PositionGetString(POSITION_COMMENT);
         if(pos_magic==my_magic)
           {
           //  process the position with a specified POSITION_MAGIC
           }
         PrintFormat("Position #%d by %s: POSITION_MAGIC=%d, price=%G, type=%s, commentary=%s",
                     pos_id,symbol,pos_magic,price,EnumToString(type),comment);
        }
      else           // call to PositionGetSymbol() was unsuccessful
        {
         PrintFormat("Error when receiving into the cache the position with index %d."+
                     " Error code: %d", i, GetLastError());
        }
     }
  }
```

### Rules for working with the history cache

Often, the code for working with the history cache is written by the programmer in such a way, that it works smoothly only if the history contains 5-10 deals and orders. A typical example of a **wrong** approach - loading the entire trading history into the cache, and processing it in a loop, searching through all of the orders and trades:

```
//---
   datetime start=0;           // initial time set to 1970 year
   datetime end=TimeCurrent();  // the ending time set to the current server time
//--- request into the cache of the program the entire trading history
   HistorySelect(start,end);
//--- obtain the number of all of the orders in the history
   int history_orders=HistoryOrdersTotal();
//--- now scan through all of the orders
   for(int i=0;i<history_orders;i++)
     {
     //  processing each order in the history
     }

    ...

//--- obtain the number of all deals in the history
   int deals=HistoryDealsTotal();
//--- now scan through all of the deals
   for(int i=0;i<deals;i++)
     {
     //  process each deal in the history
     }
```

The attempt to handle all of the trading history, in the majority of cases, is wrong. When the number of processed deals/orders becomes around thousands and tens of thousands, the work of the program drastically slows down.

**Note:** always cautiously refer to all of the cases of calling the HistorySelect() function! Unthoughtful and excessive loading of all of the available trading history into the cache of the mql5-program, degrades its performance.

This is primarily important for testing - the user discovers that the tester suddenly becomes thoughtful, and begins looking for the reasons for this in the client terminal. Therefore, firstly always think about optimizing the code of the program MQL5 (EA and indicators, which are called from the EA). Do not rely on the fact that the computer is made of iron and has many kernels.

For the proper work of the EA and the indicator online, this is just as important. A non-optimal code of the program can paralyze the work of even the most powerful computer.

**The correct algorithm for working with the trading history**:

1. Determine the need for requesting the trading history into the cache. If this is not necessary, then do not perform the following actions.
2. Determine the final date of the trading history (perhaps the history up to the moment is not necessary).
3. Calculate the initial date of the trading history, starting from the ending date. Usually, the EAs require the trading history, no deeper than a single day or week.
4. Obtain the tickets of deals and historical orders for obtaining the properties, by the known tickets:
   - HistoryOrderGetDouble()
   - HistoryOrderGetInteger()
   - HistoryOrderGetString()
   - HistoryDealGetDouble()
   - HistoryDealGetInteger()
   - HistoryDealGetString()
5. If the tickets are not known, and if it is necessary, organize a cycle through sorting.
6. In the loop, obtain the ticket for each deal/order from the cache of the trading history, by the index ( [HistoryOrderGetTicket](https://www.mql5.com/en/docs/trading/historyordergetticket)(Index) and [HistoryDealGetTicket](https://www.mql5.com/en/docs/trading/historydealgetticket)(Index)).
7. Obtain the necessary properties of orders and deals by the known ticket (see point 4).

An example of a code for this algorithm:

```
//--- the variable, which is set in true only during the change in the trading history
   bool TradeHistoryChanged=false;
//--- here we check for the changes in the history and put out the TradeHistoryChanged=true if needed
//... the needed code

//--- check  if there are changes in the trading history or not
   if(!TradeHistoryChanged) return;

//--- if the history has changed, then it makes sense to load it into the cache
//--- the ending time set for the current server time
   datetime end=TimeCurrent();
//--- the beginning time is set to 3 days ago
   datetime start=end-3*PeriodSeconds(PERIOD_D1);
//--- request in the cache of the program, the trading history for the last 3 days
   HistorySelect(start,end);
//--- obtain the number of orders in the cache of the history
   int history_orders=HistoryOrdersTotal();
//--- now scan through the orders
   for(int i=0;i<history_orders;i++)
     {
      //--- obtain the ticket of the historical order
      ulong ticket=HistoryOrderGetTicket(i);
      //--- work with this order - receive its problems
      long order_magic=HistoryOrderGetInteger(ticket,ORDER_MAGIC);
      // obtain the rest of the properties for the order by the ticket
      // ...
     }
```

The basic idea presented by this example - is that first you must **verify the fact of changes taking place** in the trading history. One of the options is to, inside the [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) function, set for the global variable TradeHistoryChanged, the value of true, since the [Trade](https://www.mql5.com/en/docs/runtime/event_fire#trade) event always returns with any type of trading event.

If the trading history has not changed, then there is no need to upload the trading history into the cache again, and waste resources of the CPU. This is logical and does not require any explanation. If the trading history has changed, then we upload only the necessary part of it, and go through each deal/order only once. Avoid unnecessary repeat cycles.

**Note:** each request to the cache of the entire trading history, done by the function HistorySelect(), and each cycle of processing deals and orders from the history, have to be grounded. Otherwise, your computer's resources will be spent inefficiently.

Examples of correct and incorrect work with the trading history are attached to this article, as files WrongWorkWithHistory.mq5 and RightWorkWithHistory.mq5.

### Obtaining information by orders from the history

Working with historical orders is almost no different from working with the active orders, with but one exception. If the number of active orders in the cache of the mql5-program can not be more than one, then the result [HistoryOrdersTotal()](https://www.mql5.com/en/docs/trading/historyorderstotal), and the number of historical orders in the cache, depends on how much trading history has been loaded by the HistorySelect(start, end), HistorySelectByPosition() or HistoryOrderSelection() function.

**Note:** If the trading history has not been loaded into the cache of the mql5-program by one of the functions HistorySelect(), HistorySelectByPosition() or HistoryOrderSelect(), then working with historical orders and deals is impossible. Be sure to request the required history of deals and orders before receiving the data on trading history.

For example, we provide a script, which searches for the last order of the last day, and displays information for it.

```
// --- determining the time intervals of the required trading history
   datetime end=TimeCurrent();                // current server time
   datetime start=end-PeriodSeconds(PERIOD_D1);// set the beginning for 24 hours ago
//--- request in the cache of the program the trading history for a day
   HistorySelect(start,end);
//--- receive the number of orders in the history
   int history_orders=HistoryOrdersTotal();
//--- obtain the ticket of the order, which has the last index in the list, from the history
   ulong order_ticket=HistoryOrderGetTicket(history_orders-1);
   if(order_ticket>0) // obtain in the cache the historical order, work with it
     {
      //--- order status
      ENUM_ORDER_STATE state=(ENUM_ORDER_STATE)HistoryOrderGetInteger(order_ticket,ORDER_STATE);
      long order_magic      =HistoryOrderGetInteger(order_ticket,ORDER_MAGIC);
      long pos_ID           =HistoryOrderGetInteger(order_ticket,ORDER_POSITION_ID);
      PrintFormat("Order #%d: ORDER_MAGIC=#%d, ORDER_STATE=%d, ORDER_POSITION_ID=%d",
                  order_ticket,order_magic,EnumToString(state),pos_ID);
     }
   else              // unsuccessful attempt to obtain the order
     {
      PrintFormat("In total, in the history of %d orders, we couldn't select the order"+
                  " with the index %d. Error %d",history_orders,history_orders-1,GetLastError());
     }
```

In more general cases, it is needed to sort through the orders in the loop **from the cache**, and analyze them. The general algorithm will be as follows:

1. Determine the time ranges of the **sufficient** history, if the history is loaded by the function [HistorySelect()](https://www.mql5.com/en/docs/trading/historyselect) \- it is not recommended to load the entire trading history into the cache.
2. **Load into the cache of the program, the trading history** HistorySelect(), HistorySelectByPosition() or HistoryOrderSelect (ticket) functions.
3. Obtain the total number of orders in the cache, using the HistoryOrdersTotal().
4. Organize the cycle by a search through all of the orders by their indexes in the list.
5. Obtain a ticket of the orders in the cache, using the HistoryOrderGetTicket() function .
6. Obtain the data of the order from the cache, by using the HistoryOrderGetDouble(), HistoryOrderGetInteger(), and HistoryOrderGetString() functions. If needed, analyze the obtained data and take the appropriate actions.

An example of such an algorithm:

```
#property script_show_inputs

input long my_magic=999;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
// --- setting the time intervals of the required trading history
   datetime end=TimeCurrent();                // the current server time
   datetime start=end-PeriodSeconds(PERIOD_D1);// set the beginning for 24 hours ago
//--- request into the cache of the program the needed interval of the trading history
   HistorySelect(start,end);
//--- obtain the number of orders in history
   int history_orders=HistoryOrdersTotal();
//--- now scroll through all of the orders
   for(int i=0;i<history_orders;i++)
     {
      //--- obtain the ticket of the order by its number in the list
      ulong order_ticket=HistoryOrderGetTicket(i);
      if(order_ticket>0) //  obtain in the cache, the historical order, and work with it
        {
         //--- time of execution
         datetime time_done=HistoryOrderGetInteger(order_ticket,ORDER_TIME_DONE);
         long order_magic  =HistoryOrderGetInteger(order_ticket,ORDER_MAGIC);
         long pos_ID       =HistoryOrderGetInteger(order_ticket,ORDER_POSITION_ID);
         if(order_magic==my_magic)
           {
           //  process the position with the set ORDER_MAGIC
           }
         PrintFormat("Order #%d: ORDER_MAGIC=#%d, time_done %s, ORDER_POSITION_ID=%d",
                     order_ticket,order_magic,TimeToString(time_done),pos_ID);
        }
      else               // unsuccessful attempt to obtain the order from the history
        {
         PrintFormat("we were not able to select the order with the index %d. Error %d",
                     i,GetLastError());
        }
     }
  }
```

**Note:** always cautiously refer to all of the cases of calling the function HistorySelect()! Unthoughtful and excessive loading of all of the available trading history into the cache of the mql5-program, degrades its performance.

### Obtaining information on deals from the history

Processing of deals has the same features as the processing of historical orders. The number of deals in the trading history and the result of the execution of [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal), depends on how much of the trading history has been loaded into the cache by the [HistorySelect(start, end)](https://www.mql5.com/en/docs/trading/historyselect) or [HistorySelectByPosition()](https://www.mql5.com/en/docs/trading/historyselectbyposition) function.

To fill the cache with only one deal by its ticket, use the [HistoryDealSelect(ticket)](https://www.mql5.com/en/docs/trading/historydealselect) function.

```
// --- determining the time intervals of the required trading history
   datetime end=TimeCurrent();                // current sever time
   datetime start=end-PeriodSeconds(PERIOD_D1);// set the beginning for 24 hours ago
//--- request in the cache of the program the needed interval of the trading history
   HistorySelect(start,end);
//--- obtain the number of deals in history
   int deals=HistoryDealsTotal();
//--- obtain the ticket for the deal, which has the last index in the list
   ulong deal_ticket=HistoryDealGetTicket(deals-1);
   if(deal_ticket>0) // we obtained in the cache of the deal, and work with it
     {
      //--- the ticket order, based on which the deal was made
      ulong order     =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
      long order_magic=HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
      long pos_ID     =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
      PrintFormat("Deal #%d for the order #%d with the ORDER_MAGIC=%d  that participated in the position",
                  deals-1,order,order_magic,pos_ID);
     }
   else              // unsuccessful attempt of obtaining a deal
     {
      PrintFormat("In total, in the history %d of deals, we couldn't select a deal"+
                  " with the index %d. Error %d",deals,deals-1,GetLastError());
     }
```

In more general cases, it is needed to search in the deal loop **from the cache**, and make analyze them. The general algorithm will be as follows:

1. Determine the boundaries of the **sufficient** history, if history loads by the [HistorySelect(start, end)](https://www.mql5.com/en/docs/trading/historyselect) function - then it is not recommended to load the entire history of trade into the cache.
2. **Load into the cache of the program, the trading history of the** functions HistorySelect() or HistorySelectByPosition().
3. Obtain the total number of deals in the history, using the [HistoryDealsTotal()](https://www.mql5.com/en/docs/trading/historydealstotal) function.
4. Organize the cycle by searching through all of the deals, by their numbers in the list.
5. Determine the ticket of the next deal in the cache, by using the HistoryDealGetTicket().
6. Obtain the information about the deal from the cache, by using the functions HistoryDealGetDouble(), HistoryDealGetInteger(), and HistoryDealGetString(). If needed, analyze the obtained data and take the appropriate actions.

An example of such an algorithm for calculating the profits and losses:

```
input long my_magic=111;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
// --- determine the time intervals of the required trading history
   datetime end=TimeCurrent();                 // current server time
   datetime start=end-PeriodSeconds(PERIOD_D1);// set the beginning time to 24 hours ago

//--- request in the cache of the program the needed interval of the trading history
   HistorySelect(start,end);
//--- obtain the number of deals in the history
   int deals=HistoryDealsTotal();

   int returns=0;
   double profit=0;
   double loss=0;
//--- scan through all of the deals in the history
   for(int i=0;i<deals;i++)
     {
      //--- obtain the ticket of the deals by its index in the list
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket>0) // obtain into the cache the deal, and work with it
        {
         string symbol             =HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         datetime time             =HistoryDealGetInteger(deal_ticket,DEAL_TIME);
         ulong order               =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
         long order_magic          =HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
         long pos_ID               =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
         ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);

         //--- process the deals with the indicated DEAL_MAGIC
         if(order_magic==my_magic)
           {
            //... necessary actions
           }

         //--- calculate the losses and profits with a fixed results
         if(entry_type==DEAL_ENTRY_OUT)
          {
            //--- increase the number of deals
            returns++;
            //--- result of fixation
            double result=HistoryDealGetDouble(deal_ticket,DEAL_PROFIT);
            //--- input the positive results into the summarized profit
            if(result>0) profit+=result;
            //--- input the negative results into the summarized losses
            if(result<0) loss+=result;
           }
        }
      else // unsuccessful attempt to obtain a deal
        {
         PrintFormat("We couldn't select a deal, with the index %d. Error %d",
                     i,GetLastError());
        }
     }
   //--- output the results of the calculations
   PrintFormat("The total number of %d deals with a financial result. Profit=%.2f , Loss= %.2f",
               returns,profit,loss);
  }
```

**Note:** always cautiously refer to all of the cases of calling the function HistorySelect()! Unthoughtful and excessive loading of all of the available trading history into the cache of the mql5-program, degrades its performance.

### Obtaining in the cache of the history by the identifier of the position (POSITION\_IDENTIFIER)

The [HistorySelectByPosition (position\_ID)](https://www.mql5.com/en/docs/trading/historyselectbyposition) function just like the [HistorySelect (start, end)](https://www.mql5.com/en/docs/trading/historyselect) function fills the cache with deals and orders from the history, but only under one condition - they must have the specified identifier of the position ( [POSITION\_IDENTIFIER](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties)). The identifier of the position - is a unique number, which is automatically assigned to each re-opened position, and does not change throughout its life. Meanwhile. it must be kept in mind, that the change of the position (shift of the type of the position from POSITION\_TYPE\_BUY to POSITION\_TYPE\_SELL) does not change the identifier of the position.

Each open position is the result of one or more deals on that instrument. Therefore, to analyze the position changes, during its lifetime, each deal and order, based on which the deal was done, is assigned an identifier to the position, in which this deal participated. Thus, knowing the identifier of the current open positions, we can reconstruct the entire history - find all of the orders and deals that have changed it.

The [HistorySelectByPosition(position\_ID)](https://www.mql5.com/en/docs/trading/historyselectbyposition) function serves to spare the programmer from having to write their own code for iterating through the entire trading history in search of such information. A typical algorithm for working with this function:

1. Obtain the right position identifier.
2. Obtain, using the function HistorySelectByPosition(), into the cache of the trading history, all of the orders and deals, the identifier of which, equals the identifier of the current position.
3. Process the trading history according to the algorithm.

### Conclusion

The entire trading subsystem platform [MetaTrader 5](https://www.metatrader5.com/ru "https://www.metatrader5.com/ru") is well thought out and user-friendly More-ever, the abundance of [trading functions](https://www.mql5.com/en/docs/trading), allows us to solve each specific problem in the most efficient way.

But even despite the fact that the specialized [trading classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) from the standard library, allow us not to worry about too many nuances, and write programs on a high level, without going into implementation, the understanding of the basics, will allow us to create more reliable and efficient trading EAs.

All of the given examples can be found in the files, attached to this article.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/211](https://www.mql5.com/ru/articles/211)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/211.zip "Download all attachments in the single ZIP archive")

[demo\_positiongetsymbolbyindex.mq5](https://www.mql5.com/en/articles/download/211/demo_positiongetsymbolbyindex.mq5 "Download demo_positiongetsymbolbyindex.mq5")(2.08 KB)

[demo\_positionselectbysymbol.mq5](https://www.mql5.com/en/articles/download/211/demo_positionselectbysymbol.mq5 "Download demo_positionselectbysymbol.mq5")(1.65 KB)

[demo\_ordergetticketbyindex.mq5](https://www.mql5.com/en/articles/download/211/demo_ordergetticketbyindex.mq5 "Download demo_ordergetticketbyindex.mq5")(1.87 KB)

[demo\_orderselectbyticket.mq5](https://www.mql5.com/en/articles/download/211/demo_orderselectbyticket.mq5 "Download demo_orderselectbyticket.mq5")(1.31 KB)

[demo\_historydealselectbyindex.mq5](https://www.mql5.com/en/articles/download/211/demo_historydealselectbyindex.mq5 "Download demo_historydealselectbyindex.mq5")(2.92 KB)

[demo\_historydealselectbyticket.mq5](https://www.mql5.com/en/articles/download/211/demo_historydealselectbyticket.mq5 "Download demo_historydealselectbyticket.mq5")(1.96 KB)

[demo\_historyorderselectbyindex.mq5](https://www.mql5.com/en/articles/download/211/demo_historyorderselectbyindex.mq5 "Download demo_historyorderselectbyindex.mq5")(2.22 KB)

[demo\_historyorderselectbyticket.mq5](https://www.mql5.com/en/articles/download/211/demo_historyorderselectbyticket.mq5 "Download demo_historyorderselectbyticket.mq5")(2.02 KB)

[rightworkwithhistory.mq5](https://www.mql5.com/en/articles/download/211/rightworkwithhistory.mq5 "Download rightworkwithhistory.mq5")(3.85 KB)

[wrongworkwithhistory.mq5](https://www.mql5.com/en/articles/download/211/wrongworkwithhistory.mq5 "Download wrongworkwithhistory.mq5")(1.42 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/3093)**
(50)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
30 Jun 2021 at 18:11

```
#define  PRINT(A) Print(#A + " = " + (string)(A))

void OnStart()
{
  PRINT(TerminalInfoInteger(TERMINAL_MEMORY_USED));

  if (HistorySelect(0, INT_MAX))
  {
    PRINT(HistoryDealsTotal());
    PRINT(HistoryOrdersTotal());

    PRINT(MQLInfoInteger(MQL_MEMORY_USED));
    PRINT(TerminalInfoInteger(TERMINAL_MEMORY_USED));
  }
}
```

Run result on Terminal with one M1 chart, 5000 bars, one symbol, no resources and no graphics.

```
TerminalInfoInteger(TERMINAL_MEMORY_USED) = 426
HistoryDealsTotal() = 134502
HistoryOrdersTotal() = 218740
MQLInfoInteger(MQL_MEMORY_USED) = 1
TerminalInfoInteger(TERMINAL_MEMORY_USED) = 789
```

It's a lot. 10 synchronous (OrderSend) EAs eats 4 gigs. Two options:

1. Open a new account, transfer funds to it and continue trading on it. Unfortunately, it is not always possible.
2. Combine all the bots into one through asynchrony [(OrderSendAsync](https://www.mql5.com/en/docs/trading/ordersendasync "MQL5 documentation: OrderSendAsync function")). This is a very hard variant of catching bugs in case of super-active trading.

In the second point, it is still necessary to write a manager (GUI and so on) of bots embedded in a single Expert Advisor.


![mktr8591](https://c.mql5.com/avatar/avatar_na2.png)

**[mktr8591](https://www.mql5.com/en/users/mktr8591)**
\|
30 Jun 2021 at 18:43

**fxsaber:**

2. Combine all bots into one through asynchrony [(OrderSendAsync](https://www.mql5.com/en/docs/trading/ordersendasync "MQL5 documentation: OrderSendAsync function")). Very heavy variant of catching bugs at super-active trading.

There is no other way. (unless, of course, you cut off the old history and redo the whole algorithm of working with history, but this is only if MQ do not return the old sorting).

![Carlos Camargo](https://c.mql5.com/avatar/avatar_na2.png)

**[Carlos Camargo](https://www.mql5.com/en/users/camargo.cr)**
\|
29 Sep 2021 at 17:15

Hi, folks!

It would be helpful that [@MetaQuotes](https://www.mql5.com/en/users/metaquotes) upgrade this article with [Trade Classes](https://www.mql5.com/en/docs/standardlibrary/tradeclasses) ( [CAccountInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/caccountinfo), [CSymbolInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo), [COrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/corderinfo), [CHistoryOrderInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/chistoryorderinfo), [CPositionInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cpositioninfo), [CDealInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cdealinfo), [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade), [CTerminalInfo](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/cterminalinfo)). Develop EA under Object-Oriented paradigm could modify (and simplify) this operations of synchronize cache and to get data over symbols, orders, positions, deals, trades, etc.

Am I right?

![Ahmed Elnashar](https://c.mql5.com/avatar/2022/8/630AC10F-F21F.jpg)

**[Ahmed Elnashar](https://www.mql5.com/en/users/diamondiptv)**
\|
16 Oct 2021 at 12:16

if you please how to calculate order commision with profit to be like this

" Profit += profit + swap + commision "

![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
5 Oct 2025 at 15:37

Please help with the answer to the question!

Floating position indicators "Market Value" and "Profit" in MT5 terminal are calculated by the terminal itself on the basis of translated quotes and symbols specification, or they are translated by MT5 server and cached on disc?

If they are cached, is it likely to catch unsynchronisation between the received quotes and the current "Market Value" and "Profit" indicators ?

![Exposing C# code to MQL5 using unmanaged exports](https://c.mql5.com/2/0/logo__5.png)[Exposing C# code to MQL5 using unmanaged exports](https://www.mql5.com/en/articles/249)

In this article I presented different methods of interaction between MQL5 code and managed C# code. I also provided several examples on how to marshal MQL5 structures against C# and how to invoke exported DLL functions in MQL5 scripts. I believe that the provided examples may serve as a basis for future research in writing DLLs in managed code. This article also open doors for MetaTrader to use many libraries that are already implemented in C#.

![How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://c.mql5.com/2/0/MetaTrader5_MetaTrader4_MQL5.png)[How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://www.mql5.com/en/articles/189)

Is it possible to trade on a real MetaTrader 5 account today? How to organize such trading? The article contains the theory of these questions and the working codes used for copying trades from the MetaTrader 5 terminal to MetaTrader 4. The article will be useful both for the developers of Expert Advisors and for practicing traders.

![MQL5 Wizard: How to Create a Risk and Money Management Module](https://c.mql5.com/2/0/CMoney_MQL5.png)[MQL5 Wizard: How to Create a Risk and Money Management Module](https://www.mql5.com/en/articles/230)

The generator of trading strategies of the MQL5 Wizard greatly simplifies testing of trading ideas. The article describes how to develop a custom risk and money management module and enable it in the MQL5 Wizard. As an example we've considered a money management algorithm, in which the size of the trade volume is determined by the results of the previous deal. The structure and format of description of the created class for the MQL5 Wizard are also discussed in the article.

![Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://c.mql5.com/2/0/MQL5_Mini-Max_Indicator.png)[Moving Mini-Max: a New Indicator for Technical Analysis and Its Implementation in MQL5](https://www.mql5.com/en/articles/238)

In the following article I am describing a process of implementing Moving Mini-Max indicator based on a paper by Z.G.Silagadze 'Moving Mini-max: a new indicator for technical analysis'. The idea of the indicator is based on simulation of quantum tunneling phenomena, proposed by G. Gamov in the theory of alpha decay.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/211&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069394210995504216)

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