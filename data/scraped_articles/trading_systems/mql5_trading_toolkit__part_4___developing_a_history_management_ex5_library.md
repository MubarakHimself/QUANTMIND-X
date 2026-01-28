---
title: MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library
url: https://www.mql5.com/en/articles/16528
categories: Trading Systems, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:37:30.473244
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=hrwurvsfvpkuiotmdzqrravyqhnvuirr&ssn=1769157448632237783&ssn_dr=0&ssn_sr=0&fv_date=1769157448&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16528&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%204)%3A%20Developing%20a%20History%20Management%20EX5%20Library%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915744816740222&fz_uniq=5062599705877194055&sv=2552)

MetaTrader 5 / Examples


### Introduction

In this engaging article series, we have developed two comprehensive EX5 libraries: [_PositionsManager.ex5_](https://www.mql5.com/en/articles/15224), which processes and manages positions, and [_PendingOrdersManager.ex5_](https://www.mql5.com/en/articles/15888), which handles pending orders. Alongside these, we created practical examples, including some with graphical user interfaces, to demonstrate the implementation of these libraries effectively.

In this article, we will introduce another essential _EX5 library_ designed to retrieve and process the history of completed orders, deals, and position transactions. Additionally, we will develop analytical modules to generate trade reports that assess the performance of trading systems, Expert Advisors, or specific symbols based on different flexible criteria.

This article serves as a practical guide for beginner MQL5 developers who may find working with positions, orders, and deal histories challenging. It will also be a valuable resource for any MQL5 programmer seeking a library to streamline and enhance the efficiency of handling trading history.

To get started, we will address a few critical questions that many MQL5 programmers, especially those new to processing trade histories in MetaTrader 5, often find challenging to understand.

### What Is The Lifecycle Of A Trade Transaction In MQL5?

In MQL5, the lifecycle of a trade transaction begins with the execution of an order. This order can be classified into two main types: a **_direct market order_** or a **_pending order_**.

**Direct Market Order Entry**

A direct market order is a real-time request to buy or sell an asset at the current market price ( _Ask_ or _Bid_). We previously covered how to process these orders in the [first](https://www.mql5.com/en/articles/14822) and [second](https://www.mql5.com/en/articles/15888) articles while developing the Positions Manager library.

![Direct Market Entry Order](https://c.mql5.com/2/103/Direct_Market_Entry_Order.png)

A direct market order is executed immediately, making it ideal for manual and automated trading strategies. Once executed, the order transitions into an active open position and is assigned a unique **_ticket number_** and a separate **_position identifier_ ( _POSITION\_ID_)**, which is more reliable for tracking and managing the various stages of the position throughout its lifecycle.

**Pending Order Entry**

A pending order ( _BUY STOP_, _BUY LIMIT_, _SELL STOP_, _SELL LIMIT_, _BUY STOP LIMIT_, and _SELL STOP LIMIT_), by contrast, is a delayed order triggered when a specified price level is reached. An in-depth guide on processing these types of orders is covered in the [third article](https://www.mql5.com/en/articles/15888) of this series, where we develop the Pending Orders Manager library.

![Pending Order Entry](https://c.mql5.com/2/103/Pending_Order_Entry.png)

Until the market price aligns with the predefined pending order trigger price, the pending order remains inactive. Once triggered, it converts into a **market order** and is executed, receiving a unique _ticket number_ and _position identifier (POSITION\_ID)_ similar to a direct market order.

### How Can A Position's Status Change During Its Lifetime?

Throughout a position's lifespan, its status can change due to various factors:

- **Partial Closure**: If a portion of the position is closed, a corresponding _exit (out) deal_ is recorded in the trade history.
- **Position Reversal**: A position reversal, such as a " _close by_" transaction, is also logged as an _exit deal_.
- **Full Closure**: When the entire position is closed, either manually or automatically through a _Take Profit_, _Stop Loss_, or _Stop Out_ event due to a margin call, a final _exit deal_ is recorded in the trade history.

Understanding the lifecycle of a trade operation in MQL5 is crucial. Every trade begins as an order sent to the trade server, whether it’s a request to open a pending order, execute a direct market order to buy or sell, or partially close an existing position. Regardless of the type, all trade operations are first recorded as orders.

If the order is successfully executed, it transitions into the next stage and is saved in the history database as a deal. Using the various properties and functions available for orders and deals, you can trace each deal back to its originating order and link it to its corresponding position. This creates a clear and systematic trail of a trade's lifecycle.

This "breadcrumb" approach allows you to track the origin, progression, and outcome of every trade or transaction in the MQL5 environment. It provides a detailed audit trail, including the order that initiated the transaction, the exact time of execution, any modifications made along the way, and the final result of the trade (position). Such tracking not only enhances transparency but will also empower you as an MQL5 programmer to develop algorithms for analyzing trading strategies, identifying areas for improvement, and optimizing performance.

What Is The Difference Between a Position and a Trade In MQL5?

In MQL5, a position represents an **active ongoing** trade ( _position_) that you currently hold in the market. It is in an **open state** that reflects either a _buy_ or _sell_ position on a specific _symbol_. A trade, on the other hand, refers to a completed transaction—when a position has been **fully closed**. Active open positions and pending orders are displayed in MetaTrader 5’s _Toolbox_ window under the " **_Trade_**" tab.

![MetaTrader 5 Toolbox Trades Tab](https://c.mql5.com/2/103/MT5_Toolbox_Trades_Tab.png)

Closed _positions_( _trades_), along with _orders_ and _deals_, are shown in the " **_History_**" tab of the Toolbox window.

![MetaTrader 5 Toolbox History Tab](https://c.mql5.com/2/103/MT5_Toolbox_History_Tab.png)

To access the complete history of positions, you can use the platform’s menu options and select the " **_Positions_**" menu item. Orders and deals history can also be accessed using the same menu options.

![MetaTrader 5 Toolbox History Tab - Positions History Selecting](https://c.mql5.com/2/103/MT5_Toolbox_History_Tab_-_Positions_History_Selecting.png)

The distinction between positions and trades can be confusing for beginner MQL5 programmers, particularly when using the platform’s standard history functions. This article, along with the detailed code in the library we are about to create, will provide you with a clear understanding of how positions and trades are categorized and tracked in MQL5. If you're short on time and need a ready-to-use history library, you can simply follow the comprehensive documentation in the next proceeding article on how to implement it directly into your project.

### Create The History Manager Library Source Code File (.mq5)

To get started, open your _MetaEditor IDE_ and access the _MQL Wizard_ by selecting **'** _New_' from the menu. In the wizard, choose to create a new _Library_ source file, which we will name _HistoryManager.mq5_. This file will be the foundation for our core functions, which are dedicated to managing and analyzing the accounts' trade history. When creating the new _HistoryManager.mq5_ library, save it in the _Libraries\\Toolkit_ folder that we established in the first article. By storing this new file in the same directory as the _Positions Manager_ and _Pending Orders Manager_ EX5 libraries, we maintain a clear and consistent organizational structure for our project. This approach will make it easier to locate and manage each component as our toolkit expands.

![MetaEditor Navigator Library Toolkit Directory](https://c.mql5.com/2/103/MetaEditor_Navigator_Library_Toolkit_Directory.png)

Here’s what our newly created _HistoryManager.mq5_ source file looks like. Begin by deleting the " _My function_" comments located below the _property_ directives. The _copyright_ and _link_ property directives in your file might differ, but this won’t impact the behavior or performance of the code. You can customize the _copyright_ and _link_ directives with any information you prefer, but ensure that the _library_ property directive remains unchanged.

```
//+------------------------------------------------------------------+
//|                                               HistoryManager.mq5 |
//|                          Copyright 2024, Wanateki Solutions Ltd. |
//|                                         https://www.wanateki.com |
//+------------------------------------------------------------------+
#property library
#property copyright "Copyright 2024, Wanateki Solutions Ltd."
#property link      "https://www.wanateki.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| My function                                                      |
//+------------------------------------------------------------------+
// int MyCalculator(int value,int value2) export
//   {
//    return(value+value2);
//   }
//+------------------------------------------------------------------+
```

### Data Structures, Preprocessor Directives and Global Variables

In our newly created _HistoryManager.mq5_ library source file, we will start by defining the following components:

- **Preprocessor Directives**: These will assist in sorting and querying various types of trade history.
- **Data Structures**: These will store history data for orders, deals, positions, and pending orders.
- **Global Dynamic Structure Arrays**: These will hold all the relevant history data in the library.

Defining these components in the global scope ensures they are accessible across the entire library and can be utilized by all the different modules or functions within it.

**Preprocessor Directives**

Since our history management library will handle various types of requests, it’s crucial to design it in a way that retrieves only the specific history data needed for each request. This modular and targeted approach will enhance our library’s performance while maintaining flexibility for various use cases.

To achieve this, we will define integer constants that serve as identifiers for specific types of historical data. These constants will allow the library to target only the required data, ensuring minimal resource consumption and faster processing.

We will organize the history data into five major categories:

1. Orders History.
2. Deals History.
3. Positions History.
4. Pending Orders History.
5. All History Data.

By using these constants, functions within the library can specify the type of history they want to process. The main history-fetching function will query and return only the requested data, saving time and computational resources. Let us begin by defining these integer constants by placing them directly below the last _**#property**_ directives in our code.

```
#define GET_ORDERS_HISTORY_DATA 1001
#define GET_DEALS_HISTORY_DATA 1002
#define GET_POSITIONS_HISTORY_DATA 1003
#define GET_PENDING_ORDERS_HISTORY_DATA 1004
#define GET_ALL_HISTORY_DATA 1005
```

**Data Structures**

Our EX5 library will store various history data in globally declared data structures. These structures will efficiently hold the deals, orders, positions, and pending orders history whenever they are queried.

```
//- Data structure to store deal properties
struct DealData
  {
   ulong             ticket;
   ulong             magic;
   ENUM_DEAL_ENTRY   entry;
   ENUM_DEAL_TYPE    type;
   ENUM_DEAL_REASON  reason;
   ulong             positionId;
   ulong             order;
   string            symbol;
   string            comment;
   double            volume;
   double            price;
   datetime          time;
   double            tpPrice;
   double            slPrice;
   double            commission;
   double            swap;
   double            profit;
  };

//- Data structure to store order properties
struct OrderData
  {
   datetime                timeSetup;
   datetime                timeDone;
   datetime                expirationTime;
   ulong                   ticket;
   ulong                   magic;
   ENUM_ORDER_REASON       reason;
   ENUM_ORDER_TYPE         type;
   ENUM_ORDER_TYPE_FILLING typeFilling;
   ENUM_ORDER_STATE        state;
   ENUM_ORDER_TYPE_TIME    typeTime;
   ulong                   positionId;
   ulong                   positionById;
   string                  symbol;
   string                  comment;
   double                  volumeInitial;
   double                  priceOpen;
   double                  priceStopLimit;
   double                  tpPrice;
   double                  slPrice;
  };

//- Data structure to store closed position/trade properties
struct PositionData
  {
   ENUM_POSITION_TYPE type;
   ulong              ticket;
   ENUM_ORDER_TYPE    initiatingOrderType;
   ulong              positionId;
   bool               initiatedByPendingOrder;
   ulong              openingOrderTicket;
   ulong              openingDealTicket;
   ulong              closingDealTicket;
   string             symbol;
   double             volume;
   double             openPrice;
   double             closePrice;
   datetime           openTime;
   datetime           closeTime;
   long               duration;
   double             commission;
   double             swap;
   double             profit;
   double             tpPrice;
   double             slPrice;
   int                tpPips;
   int                slPips;
   int                pipProfit;
   double             netProfit;
   ulong              magic;
   string             comment;
  };

//- Data structure to store executed or canceled pending order properties
struct PendingOrderData
  {
   string                  symbol;
   ENUM_ORDER_TYPE         type;
   ENUM_ORDER_STATE        state;
   double                  priceOpen;
   double                  tpPrice;
   double                  slPrice;
   int                     tpPips;
   int                     slPips;
   ulong                   positionId;
   ulong                   ticket;
   datetime                timeSetup;
   datetime                expirationTime;
   datetime                timeDone;
   ENUM_ORDER_TYPE_TIME    typeTime;
   ulong                   magic;
   ENUM_ORDER_REASON       reason;
   ENUM_ORDER_TYPE_FILLING typeFilling;
   string                  comment;
   double                  volumeInitial;
   double                  priceStopLimit;
  };
```

**Global Dynamic Structure Arrays**

The final declarations in the global scope will consist of the dynamic data structure arrays of the structures we defined earlier. These arrays will serve as the primary storage for all the core data managed by our library.

```
OrderData orderInfo[];
DealData dealInfo[];
PositionData positionInfo[];
PendingOrderData pendingOrderInfo[];
```

### **Get History Data Function**

The _GetHistoryDataFunction()_ will serve as the core of our EX5 library, forming the backbone of its functionality. Most other functions in the library will depend on it to retrieve trading history based on specified periods and history types. Since this function is meant for internal use only, it will not be defined as exportable.

This function is designed to fetch the requested history data for a given period and history type. It is a _bool_ type function, meaning it will return _true_ if the history is successfully retrieved and _false_ if the operation fails.

The _GetHistoryDataFunction()_ accepts three input parameters:

1. Two _datetime_ variables, _fromDateTime_ and _toDateTime_, specifying the start and end of the desired period.
2. An unsigned integer, _dataToGet_, which corresponds to one of the predefined constants at the top of the file.

By combining these inputs, the function can efficiently query and process the required history data. Let us begin by defining the function.

```
bool GetHistoryData(datetime fromDateTime, datetime toDateTime, uint dataToGet)
  {
   return(true);

//-- Our function's code will go here

  }
```

The first task for our function will be to verify whether the provided date range is valid. Since the _datetime_ data type in MQL5 is essentially a _long_ integer representing time in the _Unix epoch_ format (i.e., _the number of seconds elapsed since January 1, 1970, 00:00:00 UTC_), we can directly compare these values to ensure correctness. Also, note that when requesting history data in MQL5, the time is based on the trade server’s time, not your local machine’s.

To validate the date range, we will check that the _fromDateTime_ value is less than the _toDateTime_ value. If _fromDateTime_ is greater than or equal to _toDateTime_, it indicates an invalid period, as the start date cannot be later than or equal to the end date. If the provided period fails validation, we will return _false_ and exit the function.

```
if(fromDateTime >= toDateTime)
     {
      //- Invalid time period selected
      Print("Invalid time period provided. Can't load history!");
      return(false);
     }
```

Once the dates and period are verified, we will reset MQL5's error cache to ensure accurate error codes if any issues arise. Next, we will call the _HistorySelect()_ function within an _if-else_ statement, passing the validated _datetime_ values to retrieve the history of _deals_ and _orders_ for the specified period. Since _HistorySelect()_ returns a _boolean_, it will return _true_ if it successfully finds history to process, or _false_ if it encounters an error or fails to retrieve the data.

```
ResetLastError();
if(HistorySelect(fromDateTime, toDateTime)) //- History selected ok
  {
//-- Code to process the history data will go here
  }
else //- History selecting failed
  {
   Print("Selecting the history failed. Error code = ", GetLastError());
   return(false);
  }
```

Inside the _else_ part of the _if-else_ statement, we've added code to log a message indicating that history selection failed, along with the error code, before exiting the function and returning a _boolean_ value of _false_. In the _if_ part, we will use a _switch_ statement to call the appropriate functions for processing the loaded trading history data, based on the value of _dataToGet_.

```
switch(dataToGet)
  {
   case GET_DEALS_HISTORY_DATA: //- Get and save only the deals history data
      SaveDealsData();
      break;

   case GET_ORDERS_HISTORY_DATA: //- Get and save only the orders history data
      SaveOrdersData();
      break;

   case GET_POSITIONS_HISTORY_DATA: //- Get and save only the positions history data
      SaveDealsData();  //- Needed to generate the positions history data
      SaveOrdersData(); //- Needed to generate the positions history data
      SavePositionsData();
      break;

   case GET_PENDING_ORDERS_HISTORY_DATA: //- Get and save only the pending orders history data
      SaveOrdersData(); //- Needed to generate the pending orders history data
      SavePendingOrdersData();
      break;

   case GET_ALL_HISTORY_DATA: //- Get and save all the history data
      SaveDealsData();
      SaveOrdersData();
      SavePositionsData();
      SavePendingOrdersData();
      break;

   default: //-- Unknown entry
      Print("-----------------------------------------------------------------------------------------");
      Print(__FUNCTION__, ": Can't fetch the historical data you need.");
      Print("*** Please specify the historical data you need in the (dataToGet) parameter.");
      break;
  }
```

Here is the complete _GetHistoryDataFunction()_ with all the code segments included.

```
bool GetHistoryData(datetime fromDateTime, datetime toDateTime, uint dataToGet)
  {
//- Check if the provided period of dates are valid
   if(fromDateTime >= toDateTime)
     {
      //- Invalid time period selected
      Print("Invalid time period provided. Can't load history!");
      return(false);
     }

//- Reset last error and get the history
   ResetLastError();
   if(HistorySelect(fromDateTime, toDateTime)) //- History selected ok
     {
      //- Get the history data
      switch(dataToGet)
        {
         case GET_DEALS_HISTORY_DATA: //- Get and save only the deals history data
            SaveDealsData();
            break;

         case GET_ORDERS_HISTORY_DATA: //- Get and save only the orders history data
            SaveOrdersData();
            break;

         case GET_POSITIONS_HISTORY_DATA: //- Get and save only the positions history data
            SaveDealsData();  //- Needed to generate the positions history data
            SaveOrdersData(); //- Needed to generate the positions history data
            SavePositionsData();
            break;

         case GET_PENDING_ORDERS_HISTORY_DATA: //- Get and save only the pending orders history data
            SaveOrdersData(); //- Needed to generate the pending orders history data
            SavePendingOrdersData();
            break;

         case GET_ALL_HISTORY_DATA: //- Get and save all the history data
            SaveDealsData();
            SaveOrdersData();
            SavePositionsData();
            SavePendingOrdersData();
            break;

         default: //-- Unknown entry
            Print("-----------------------------------------------------------------------------------------");
            Print(__FUNCTION__, ": Can't fetch the historical data you need.");
            Print("*** Please specify the historical data you need in the (dataToGet) parameter.");
            break;
        }
     }
   else
     {
      Print(__FUNCTION__, ": Selecting the history failed. Error code = ", GetLastError());
      return(false);
     }
   return(true);
  }
```

If you save and attempt to compile our source code file at this stage, you will encounter numerous compile errors and warnings. This is because many of the functions referenced in the code have not yet been created. Since we are still in the early stages of developing our EX5 library, once all the missing functions are implemented, the EX5 library file will compile without any errors or warnings.

### Save Deals Data Function

The _SaveDealsData()_ function will be responsible for retrieving and saving all deal history currently available in the trading history cache for the different periods that will be requested by different functions in the library. It will not return any data and is not defined as exportable because it is called internally within the library, specifically from the _GetHistoryData()_ function. This function will utilize MQL5's _HistoryDealGet..._ standard functions to fetch the various deal properties and store them in the _dealInfo_ dynamic data structure array.

First, let us begin by creating the function definition or signature.

```
void SaveDealsData()
  {

//-- Our function's code will go here

  }
```

Since _SaveDealsData()_ is called within the _GetHistoryData()_ function, there is no need to call _HistorySelect()_ again before processing the trading history. The first step in the _SaveDealsData()_ function will be to check if there is any deal history to process. We will achieve this using the _HistoryDealsTotal()_ function, which returns the total number of deals available in the history cache. For efficiency, we will create an integer and call it _totalDeals_ to store the total history deals and an unsigned long called _dealTicket_ to store the deal tickets identifiers.

```
int totalDeals = HistoryDealsTotal();
ulong dealTicket;
```

If no deals are available or found ( _totalDeals is 0 or less_), we will log a message to indicate this, and then exit the function early to avoid unnecessary processing.

```
if(totalDeals > 0)
  {
//-- Code to process deal goes here
  }
else
  {
   Print(__FUNCTION__, ": No deals available to be processed, totalDeals = ", totalDeals);
  }
```

If deal history exists, the next step will be to prepare an array to store the fetched data. We will use the _dealInfo_ dynamic array for this task and begin by resizing its size to match the total number of deals using the _ArrayResize()_ function, ensuring it has enough capacity to store all relevant deal properties.

```
ArrayResize(dealInfo, totalDeals);
```

We will then iterate through the deals in reverse order, starting from the most recent, using a for loop. For each deal, we will use the _HistoryDealGetTicket()_ function to retrieve the unique ticket associated with the deal. If the ticket retrieval is successful, we will fetch and save the various deal properties. We will store each property into its corresponding field in the _dealInfo_ array at the index corresponding to the current loop iteration.

If the _HistoryDealGetTicket()_ function fails to retrieve a valid ticket for any deal, we will log an error message, including the error code, for debugging purposes. This will ensure transparency in the event of unexpected issues during the property retrieval process.

```
for(int x = totalDeals - 1; x >= 0; x--)
  {
   ResetLastError();
   dealTicket = HistoryDealGetTicket(x);
   if(dealTicket > 0)
     {
      //- Deal ticket selected ok, we can now save the deals properties
      dealInfo[x].ticket = dealTicket;
      dealInfo[x].entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
      dealInfo[x].type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
      dealInfo[x].magic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
      dealInfo[x].positionId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
      dealInfo[x].order = HistoryDealGetInteger(dealTicket, DEAL_ORDER);
      dealInfo[x].symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
      dealInfo[x].comment = HistoryDealGetString(dealTicket, DEAL_COMMENT);
      dealInfo[x].volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
      dealInfo[x].price = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
      dealInfo[x].time = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
      dealInfo[x].tpPrice = HistoryDealGetDouble(dealTicket, DEAL_TP);
      dealInfo[x].slPrice = HistoryDealGetDouble(dealTicket, DEAL_SL);
      dealInfo[x].commission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
      dealInfo[x].swap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
      dealInfo[x].reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(dealTicket, DEAL_REASON);
      dealInfo[x].profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
     }
   else
     {
      Print(
         __FUNCTION__, " HistoryDealGetTicket(", x, ") failed. (dealTicket = ", dealTicket,
         ") *** Error Code: ", GetLastError()
      );
     }
  }
```

Here is the complete _SaveDealsData()_ function with all the code segments in full.

```
void SaveDealsData()
  {
//- Get the number of loaded history deals
   int totalDeals = HistoryDealsTotal();
   ulong dealTicket;
//-
//- Check if we have any deals to be worked on
   if(totalDeals > 0)
     {
      //- Resize the dynamic array that stores the deals
      ArrayResize(dealInfo, totalDeals);

      //- Let us loop through the deals and save them one by one
      for(int x = totalDeals - 1; x >= 0; x--)
        {
         ResetLastError();
         dealTicket = HistoryDealGetTicket(x);
         if(dealTicket > 0)
           {
            //- Deal ticket selected ok, we can now save the deals properties
            dealInfo[x].ticket = dealTicket;
            dealInfo[x].entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
            dealInfo[x].type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);
            dealInfo[x].magic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
            dealInfo[x].positionId = HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
            dealInfo[x].order = HistoryDealGetInteger(dealTicket, DEAL_ORDER);
            dealInfo[x].symbol = HistoryDealGetString(dealTicket, DEAL_SYMBOL);
            dealInfo[x].comment = HistoryDealGetString(dealTicket, DEAL_COMMENT);
            dealInfo[x].volume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
            dealInfo[x].price = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
            dealInfo[x].time = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
            dealInfo[x].tpPrice = HistoryDealGetDouble(dealTicket, DEAL_TP);
            dealInfo[x].slPrice = HistoryDealGetDouble(dealTicket, DEAL_SL);
            dealInfo[x].commission = HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
            dealInfo[x].swap = HistoryDealGetDouble(dealTicket, DEAL_SWAP);
            dealInfo[x].reason = (ENUM_DEAL_REASON)HistoryDealGetInteger(dealTicket, DEAL_REASON);
            dealInfo[x].profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
           }
         else
           {
            Print(
               __FUNCTION__, " HistoryDealGetTicket(", x, ") failed. (dealTicket = ", dealTicket,
               ") *** Error Code: ", GetLastError()
            );
           }
        }
     }
   else
     {
      Print(__FUNCTION__, ": No deals available to be processed, totalDeals = ", totalDeals);
     }
  }
```

### Print Deals History Function

The _PrintDealsHistory()_ function is designed to retrieve and display the historical deal data for a specified period. This function will be useful in situations where you need to examine a range of trade data within a given timeframe. It does not return any data but instead outputs the deal information to MetaTrader 5's log for review. This function can be called externally to provide users with insights into past trades by utilizing the _GetHistoryData()_ function to fetch the relevant data.

We begin by defining the _PrintDealsHistory()_ function. The function requires two parameters, _fromDateTime_, and _toDateTime_, which represent the start and end times for the period we want to search through. The function will fetch deals that were executed within this timeframe. Notice that the function is marked as _export_, meaning it can be called from other programs or libraries, making it readily available for external use.

```
void PrintDealsHistory(datetime fromDateTime, datetime toDateTime) export
  {
//-- Our function's code will go here
  }
```

Next, we call the _GetHistoryData()_ function, passing the _fromDateTime_, _toDateTime_, and an additional constant _GET\_DEALS\_HISTORY\_DATA_. This tells the function to pull the relevant trade data between the specified start and end times. This function call ensures that the deal information for the desired period is fetched and stored in the _dealInfo_ array.

```
GetHistoryData(fromDateTime, toDateTime, GET_DEALS_HISTORY_DATA);
```

Once the deal data is fetched, we need to check whether any data is available. We use the _ArraySize()_ function to get the total number of deals stored in the _dealInfo_ array. If no deals are found (i.e., _totalDeals is 0_), we log a message to inform the user and exit the function. If there are no deals to display, the function terminates early, saving time and preventing unnecessary operations.

```
int totalDeals = ArraySize(dealInfo);
if(totalDeals <= 0)
  {
   Print("");
   Print(__FUNCTION__, ": No deals history found for the specified period.");
   return; //-- Exit the function
  }
```

If deal data is found, we proceed to print out the details. The first step is to print a summary message showing the total number of deals found and the date range for which they were executed.

```
Print("");
Print(__FUNCTION__, "-------------------------------------------------------------------------------");
Print(
   "Found a total of ", totalDeals,
   " deals executed between (", fromDateTime, ") and (", toDateTime, ")."
);
```

Next, we use a _for_ loop to iterate over all the deals in the _dealInfo_ array. For each deal, we print the relevant details, such as the deal's symbol, ticket number, position ID, entry type, price, stop loss (SL), take profit (TP) levels, swap, commission, profit, and more. Each deal's details are neatly printed with descriptive labels, making it easy for the user to understand the transaction history.

```
for(int r = 0; r < totalDeals; r++)
  {
   Print("---------------------------------------------------------------------------------------------------");
   Print("Deal #", (r + 1));
   Print("Symbol: ", dealInfo[r].symbol);
   Print("Time Executed: ", dealInfo[r].time);
   Print("Ticket: ", dealInfo[r].ticket);
   Print("Position ID: ", dealInfo[r].positionId);
   Print("Order Ticket: ", dealInfo[r].order);
   Print("Type: ", EnumToString(dealInfo[r].type));
   Print("Entry: ", EnumToString(dealInfo[r].entry));
   Print("Reason: ", EnumToString(dealInfo[r].reason));
   Print("Volume: ", dealInfo[r].volume);
   Print("Price: ", dealInfo[r].price);
   Print("SL Price: ", dealInfo[r].slPrice);
   Print("TP Price: ", dealInfo[r].tpPrice);
   Print("Swap: ", dealInfo[r].swap, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Commission: ", dealInfo[r].commission, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Profit: ", dealInfo[r].profit, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Comment: ", dealInfo[r].comment);
   Print("Magic: ", dealInfo[r].magic);
   Print("");
  }
```

Here is the complete _PrintDealsHistory()_ function with all the code segments integrated.

```
void PrintDealsHistory(datetime fromDateTime, datetime toDateTime) export
  {
//- Get and save the deals history for the specified period
   GetHistoryData(fromDateTime, toDateTime, GET_DEALS_HISTORY_DATA);
   int totalDeals = ArraySize(dealInfo);
   if(totalDeals <= 0)
     {
      Print("");
      Print(__FUNCTION__, ": No deals history found for the specified period.");
      return; //-- Exit the function
     }

   Print("");
   Print(__FUNCTION__, "-------------------------------------------------------------------------------");
   Print(
      "Found a total of ", totalDeals,
      " deals executed between (", fromDateTime, ") and (", toDateTime, ")."
   );

   for(int r = 0; r < totalDeals; r++)
     {
      Print("---------------------------------------------------------------------------------------------------");
      Print("Deal #", (r + 1));
      Print("Symbol: ", dealInfo[r].symbol);
      Print("Time Executed: ", dealInfo[r].time);
      Print("Ticket: ", dealInfo[r].ticket);
      Print("Position ID: ", dealInfo[r].positionId);
      Print("Order Ticket: ", dealInfo[r].order);
      Print("Type: ", EnumToString(dealInfo[r].type));
      Print("Entry: ", EnumToString(dealInfo[r].entry));
      Print("Reason: ", EnumToString(dealInfo[r].reason));
      Print("Volume: ", dealInfo[r].volume);
      Print("Price: ", dealInfo[r].price);
      Print("SL Price: ", dealInfo[r].slPrice);
      Print("TP Price: ", dealInfo[r].tpPrice);
      Print("Swap: ", dealInfo[r].swap, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Commission: ", dealInfo[r].commission, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Profit: ", dealInfo[r].profit, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Comment: ", dealInfo[r].comment);
      Print("Magic: ", dealInfo[r].magic);
      Print("");
     }
  }
```

### Save Orders Data Function

The _SaveOrdersData()_ function will be responsible for retrieving and storing the historical order data available in the trading history cache. This function processes orders one by one, extracts their key properties using MQL5's _HistoryOrderGet..._ functions, and stores them in a dynamic array called _orderInfo_. This array will then be used by other parts of the library to analyze and manipulate the data as needed. This function will not return any data, will not be defined as exportable since it is used internally within the library, will handle errors gracefully, and log any issues for debugging purposes.

Let us begin by defining the function signature.

```
void SaveOrdersData()
  {
//-- Our function's code will go here
  }
```

Next, we determine how many historical orders are available. This is achieved using the _HistoryOrdersTotal()_ function, which returns the total count of historical orders in the cache. The result is stored in a variable named _totalOrdersHistory_. In addition, we declare an unsigned long variable, _orderTicket_, to hold the ticket of each order as we process them.

```
int totalOrdersHistory = HistoryOrdersTotal();
ulong orderTicket;
```

If there are no historical orders ( _totalOrdersHistory <= 0_), the function logs a message indicating this and exits early to avoid unnecessary processing.

```
if(totalOrdersHistory > 0)
  {
   //-- Code to process orders goes here
  }
else
  {
   Print(__FUNCTION__, ": No order history available to be processed, totalOrdersHistory = ", totalOrdersHistory);
   return;
  }
```

When historical orders are available, we prepare the _orderInfo_ array to store the retrieved data. This is done by resizing the array using the _ArrayResize()_ function to match the total number of historical orders.

```
ArrayResize(orderInfo, totalOrdersHistory);
```

We loop through the orders in reverse order ( _starting with the most recent_) using a for loop. For each order. We begin by retrieving the order ticket using the _HistoryOrderGetTicket()_ function. If the ticket retrieval is successful, we will extract the various properties of the order using _HistoryOrderGet..._ functions and store them in the corresponding fields of the _orderInfo_ array. If the ticket retrieval fails, the function logs an error message along with the error code for debugging.

```
for(int x = totalOrdersHistory - 1; x >= 0; x--)
  {
   ResetLastError();
   orderTicket = HistoryOrderGetTicket(x);
   if(orderTicket > 0)
     {
      //- Order ticket selected ok, we can now save the order properties
      orderInfo[x].ticket = orderTicket;
      orderInfo[x].timeSetup = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_SETUP);
      orderInfo[x].timeDone = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_DONE);
      orderInfo[x].expirationTime = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_EXPIRATION);
      orderInfo[x].typeTime = (ENUM_ORDER_TYPE_TIME)HistoryOrderGetInteger(orderTicket, ORDER_TYPE_TIME);
      orderInfo[x].magic = HistoryOrderGetInteger(orderTicket, ORDER_MAGIC);
      orderInfo[x].reason = (ENUM_ORDER_REASON)HistoryOrderGetInteger(orderTicket, ORDER_REASON);
      orderInfo[x].type = (ENUM_ORDER_TYPE)HistoryOrderGetInteger(orderTicket, ORDER_TYPE);
      orderInfo[x].state = (ENUM_ORDER_STATE)HistoryOrderGetInteger(orderTicket, ORDER_STATE);
      orderInfo[x].typeFilling = (ENUM_ORDER_TYPE_FILLING)HistoryOrderGetInteger(orderTicket, ORDER_TYPE_FILLING);
      orderInfo[x].positionId = HistoryOrderGetInteger(orderTicket, ORDER_POSITION_ID);
      orderInfo[x].positionById = HistoryOrderGetInteger(orderTicket, ORDER_POSITION_BY_ID);
      orderInfo[x].symbol = HistoryOrderGetString(orderTicket, ORDER_SYMBOL);
      orderInfo[x].comment = HistoryOrderGetString(orderTicket, ORDER_COMMENT);
      orderInfo[x].volumeInitial = HistoryOrderGetDouble(orderTicket, ORDER_VOLUME_INITIAL);
      orderInfo[x].priceOpen = HistoryOrderGetDouble(orderTicket, ORDER_PRICE_OPEN);
      orderInfo[x].priceStopLimit = HistoryOrderGetDouble(orderTicket, ORDER_PRICE_STOPLIMIT);
      orderInfo[x].tpPrice = HistoryOrderGetDouble(orderTicket, ORDER_TP);
      orderInfo[x].slPrice = HistoryOrderGetDouble(orderTicket, ORDER_SL);
     }
   else
     {
      Print(
         __FUNCTION__, " HistoryOrderGetTicket(", x, ") failed. (orderTicket = ", orderTicket,
         ") *** Error Code: ", GetLastError()
      );
     }
  }
```

After processing all orders, the function exits gracefully. Here is the full implementation, of the _SaveOrdersData()_ function, with all the code segments included.

```
void SaveOrdersData()
  {
//- Get the number of loaded history orders
   int totalOrdersHistory = HistoryOrdersTotal();
   ulong orderTicket;
//-
//- Check if we have any orders in the history to be worked on
   if(totalOrdersHistory > 0)
     {
      //- Resize the dynamic array that stores the history orders
      ArrayResize(orderInfo, totalOrdersHistory);

      //- Let us loop through the order history and save them one by one
      for(int x = totalOrdersHistory - 1; x >= 0; x--)
        {
         ResetLastError();
         orderTicket = HistoryOrderGetTicket(x);
         if(orderTicket > 0)
           {
            //- Order ticket selected ok, we can now save the order properties
            orderInfo[x].ticket = orderTicket;
            orderInfo[x].timeSetup = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_SETUP);
            orderInfo[x].timeDone = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_DONE);
            orderInfo[x].expirationTime = (datetime)HistoryOrderGetInteger(orderTicket, ORDER_TIME_EXPIRATION);
            orderInfo[x].typeTime = (ENUM_ORDER_TYPE_TIME)HistoryOrderGetInteger(orderTicket, ORDER_TYPE_TIME);
            orderInfo[x].magic = HistoryOrderGetInteger(orderTicket, ORDER_MAGIC);
            orderInfo[x].reason = (ENUM_ORDER_REASON)HistoryOrderGetInteger(orderTicket, ORDER_REASON);
            orderInfo[x].type = (ENUM_ORDER_TYPE)HistoryOrderGetInteger(orderTicket, ORDER_TYPE);
            orderInfo[x].state = (ENUM_ORDER_STATE)HistoryOrderGetInteger(orderTicket, ORDER_STATE);
            orderInfo[x].typeFilling = (ENUM_ORDER_TYPE_FILLING)HistoryOrderGetInteger(orderTicket, ORDER_TYPE_FILLING);
            orderInfo[x].positionId = HistoryOrderGetInteger(orderTicket, ORDER_POSITION_ID);
            orderInfo[x].positionById = HistoryOrderGetInteger(orderTicket, ORDER_POSITION_BY_ID);
            orderInfo[x].symbol = HistoryOrderGetString(orderTicket, ORDER_SYMBOL);
            orderInfo[x].comment = HistoryOrderGetString(orderTicket, ORDER_COMMENT);
            orderInfo[x].volumeInitial = HistoryOrderGetDouble(orderTicket, ORDER_VOLUME_INITIAL);
            orderInfo[x].priceOpen = HistoryOrderGetDouble(orderTicket, ORDER_PRICE_OPEN);
            orderInfo[x].priceStopLimit = HistoryOrderGetDouble(orderTicket, ORDER_PRICE_STOPLIMIT);
            orderInfo[x].tpPrice = HistoryOrderGetDouble(orderTicket, ORDER_TP);
            orderInfo[x].slPrice = HistoryOrderGetDouble(orderTicket, ORDER_SL);
           }
         else
           {
            Print(
               __FUNCTION__, " HistoryOrderGetTicket(", x, ") failed. (orderTicket = ", orderTicket,
               ") *** Error Code: ", GetLastError()
            );
           }
        }
     }
   else
     {
      Print(__FUNCTION__, ": No order history available to be processed, totalOrdersHistory = ", totalOrdersHistory);
     }
  }
```

### Print Orders History Function

The _PrintOrdersHistory()_ function provides an essential feature for displaying the order history details within a specified period. It queries the previously saved data from the _orderInfo_ array and prints all the relevant details of the orders. This function is defined as export because it is intended to be accessible for external modules or MQL5 apps utilizing this library. It follows a similar approach to the _PrintDealsHistory()_ function. Here is the full implementation, of the _PrintOrdersHistory()_ function, with explanatory comments to help you better understand how each part of the code operates.

```
void PrintOrdersHistory(datetime fromDateTime, datetime toDateTime) export
  {
//- Get and save the orders history for the specified period
   GetHistoryData(fromDateTime, toDateTime, GET_ORDERS_HISTORY_DATA);
   int totalOrders = ArraySize(orderInfo);
   if(totalOrders <= 0)
     {
      Print("");
      Print(__FUNCTION__, ": No orders history found for the specified period.");
      return; //-- Exit the function
     }

   Print("");
   Print(__FUNCTION__, "-------------------------------------------------------------------------------");
   Print(
      "Found a total of ", totalOrders,
      " orders filled or cancelled between (", fromDateTime, ") and (", toDateTime, ")."
   );

   for(int r = 0; r < totalOrders; r++)
     {
      Print("---------------------------------------------------------------------------------------------------");
      Print("Order #", (r + 1));
      Print("Symbol: ", orderInfo[r].symbol);
      Print("Time Setup: ", orderInfo[r].timeSetup);
      Print("Type: ", EnumToString(orderInfo[r].type));
      Print("Ticket: ", orderInfo[r].ticket);
      Print("Position ID: ", orderInfo[r].positionId);
      Print("State: ", EnumToString(orderInfo[r].state));
      Print("Type Filling: ", EnumToString(orderInfo[r].typeFilling));
      Print("Type Time: ", EnumToString(orderInfo[r].typeTime));
      Print("Reason: ", EnumToString(orderInfo[r].reason));
      Print("Volume Initial: ", orderInfo[r].volumeInitial);
      Print("Price Open: ", orderInfo[r].priceOpen);
      Print("Price Stop Limit: ", orderInfo[r].priceStopLimit);
      Print("SL Price: ", orderInfo[r].slPrice);
      Print("TP Price: ", orderInfo[r].tpPrice);
      Print("Time Done: ", orderInfo[r].timeDone);
      Print("Expiration Time: ", orderInfo[r].expirationTime);
      Print("Comment: ", orderInfo[r].comment);
      Print("Magic: ", orderInfo[r].magic);
      Print("");
     }
  }
```

### Save Positions Data Function

The _SavePositionsData()_ function organizes the deals and order history to reconstruct the lifecycle of each position, playing a central role in creating position history by synthesizing information from the available data. In the MQL5 documentation, you'll notice that there are no standard functions (s _uch as HistoryPositionSelect() or HistoryPositionsTotal()_) to directly access historical position data. Therefore, we need to create a custom function that combines orders and deals data, using the _Position ID_ as the connecting key to link deals with their originating orders.

We will start by examining the _deals_ to identify all _exit deals_, which indicate that a position has been closed. From there, we will trace back to the corresponding _entry deal_ to collect details about the position's opening. Finally, we will use the order history to enrich the position history information with additional context, such as the originating order type or whether the position was initiated by a pending order. This step-by-step process will ensure that each position's lifecycle—from opening to closing—will be accurately reconstructed, providing a straightforward audit trail.

Let us begin by defining the function signature. Since this function will only be used internally by the EX5 library core modules, it will not be exportable.

```
void SavePositionsData()
  {
//-- Our function's code will go here
  }
```

Next, we will calculate the total number of deals in the _dealInfo_ array, which contains all deal data. Afterward, we'll resize the _positionInfo_ array, which we will use to save all the position history data and prepare it to accommodate the expected number of positions.

```
int totalDealInfo = ArraySize(dealInfo);
ArrayResize(positionInfo, totalDealInfo);
int totalPositionsFound = 0, posIndex = 0;
```

If there are no deals available in the _dealInfo_ array ( _i.e., totalDealInfo == 0_), we exit the function early since there’s no data to process.

```
if(totalDealInfo == 0)
  {
   return;
  }
```

Next, we loop through the deals in reverse order ( _starting with the most recent deal_) to ensure we can map the exit deals to their corresponding entry deals. We check if the current deal is an _exit deal_ by evaluating its entry property. ( _dealInfo\[x\].entry == DEAL\_ENTRY\_OUT_). It's crucial to start by searching for exit deals, as this confirms that a position has been closed and is no longer active. We only want to record historical, closed positions, not active ones.

```
for(int x = totalDealInfo - 1; x >= 0; x--)
  {
   if(dealInfo[x].entry == DEAL_ENTRY_OUT)
     {
      // Process exit deal
     }
  }
```

If an _exit deal_ is found, we search for its corresponding _entry deal_ by matching the _POSITION\_ID_. When an _entry deal_ is found, we begin saving its relevant information into the _positionInfo_ array.

```
for(int k = ArraySize(dealInfo) - 1; k >= 0; k--)
  {
   if(dealInfo[k].positionId == positionId)
     {
      if(dealInfo[k].entry == DEAL_ENTRY_IN)
        {
         exitDealFound = true;
         totalPositionsFound++;
         posIndex = totalPositionsFound - 1;

         // Save the entry deal data
         positionInfo[posIndex].openingDealTicket = dealInfo[k].ticket;
         positionInfo[posIndex].openTime = dealInfo[k].time;
         positionInfo[posIndex].openPrice = dealInfo[k].price;
         positionInfo[posIndex].volume = dealInfo[k].volume;
         positionInfo[posIndex].magic = dealInfo[k].magic;
         positionInfo[posIndex].comment = dealInfo[k].comment;
        }
     }
  }
```

Once the exit deal has been matched with an entry deal, we proceed to save the exit deal's properties, such as the closing price, closing time, profit, swap, and commission. We also calculate the trade's duration and net profit by considering the swap and commission.

```
if(exitDealFound)
  {
   if(dealInfo[x].type == DEAL_TYPE_BUY)
     {
      positionInfo[posIndex].type = POSITION_TYPE_SELL;
     }
   else
     {
      positionInfo[posIndex].type = POSITION_TYPE_BUY;
     }

   positionInfo[posIndex].positionId = dealInfo[x].positionId;
   positionInfo[posIndex].symbol = dealInfo[x].symbol;
   positionInfo[posIndex].profit = dealInfo[x].profit;
   positionInfo[posIndex].closingDealTicket = dealInfo[x].ticket;
   positionInfo[posIndex].closePrice = dealInfo[x].price;
   positionInfo[posIndex].closeTime = dealInfo[x].time;
   positionInfo[posIndex].swap = dealInfo[x].swap;
   positionInfo[posIndex].commission = dealInfo[x].commission;

   positionInfo[posIndex].duration = MathAbs((long)positionInfo[posIndex].closeTime -
                                     (long)positionInfo[posIndex].openTime);
   positionInfo[posIndex].netProfit = positionInfo[posIndex].profit + positionInfo[posIndex].swap -
                                      positionInfo[posIndex].commission;
  }
```

For each position, we calculate the _pip_ values for the _stop loss (SL)_ and _take profit (TP)_ levels, depending on whether the position is a _buy_ or _sell_. We use the _symbol's point_ value to determine the number of pips.

```
if(positionInfo[posIndex].type == POSITION_TYPE_BUY)
  {
// Calculate TP and SL pip values for buy position
   if(positionInfo[posIndex].tpPrice > 0)
     {
      double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
      positionInfo[posIndex].tpPips = int((positionInfo[posIndex].tpPrice -
                                           positionInfo[posIndex].openPrice) / symbolPoint);
     }
   if(positionInfo[posIndex].slPrice > 0)
     {
      double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
      positionInfo[posIndex].slPips = int((positionInfo[posIndex].openPrice -
                                           positionInfo[posIndex].slPrice) / symbolPoint);
     }
// Calculate pip profit for buy position
   double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
   positionInfo[posIndex].pipProfit = int((positionInfo[posIndex].closePrice -
                                           positionInfo[posIndex].openPrice) / symbolPoint);
  }
else
  {
// Calculate TP and SL pip values for sell position
   if(positionInfo[posIndex].tpPrice > 0)
     {
      double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
      positionInfo[posIndex].tpPips = int((positionInfo[posIndex].openPrice -
                                           positionInfo[posIndex].tpPrice) / symbolPoint);
     }
   if(positionInfo[posIndex].slPrice > 0)
     {
      double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
      positionInfo[posIndex].slPips = int((positionInfo[posIndex].slPrice -
                                           positionInfo[posIndex].openPrice) / symbolPoint);
     }
// Calculate pip profit for sell position
   double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
   positionInfo[posIndex].pipProfit = int((positionInfo[posIndex].openPrice -
                                           positionInfo[posIndex].closePrice) / symbolPoint);
  }
```

Finally, we look through the _orderInfo_ array to find the order that initiated the position. We match the _POSITION\_ID_ and ensure the order is in the _ORDER\_STATE\_FILLED_ state. Once found, we store the opening order's _ticket_ and _type_, which will help determine whether the position was initiated by a _pending order_ or a _direct market entry_.

```
for(int k = 0; k < ArraySize(orderInfo); k++)
  {
   if(
      orderInfo[k].positionId == positionInfo[posIndex].positionId &&
      orderInfo[k].state == ORDER_STATE_FILLED
   )
     {
      positionInfo[posIndex].openingOrderTicket = orderInfo[k].ticket;
      positionInfo[posIndex].ticket = positionInfo[posIndex].openingOrderTicket;

      //- Determine if the position was initiated by a pending order or direct market entry
      switch(orderInfo[k].type)
        {
         case ORDER_TYPE_BUY_LIMIT:
         case ORDER_TYPE_BUY_STOP:
         case ORDER_TYPE_SELL_LIMIT:
         case ORDER_TYPE_SELL_STOP:
         case ORDER_TYPE_BUY_STOP_LIMIT:
         case ORDER_TYPE_SELL_STOP_LIMIT:
            positionInfo[posIndex].initiatedByPendingOrder = true;
            positionInfo[posIndex].initiatingOrderType = orderInfo[k].type;
            break;
         default:
            positionInfo[posIndex].initiatedByPendingOrder = false;
            positionInfo[posIndex].initiatingOrderType = orderInfo[k].type;
            break;
        }

      break; //- Exit the orderInfo loop once the required data is found
     }
  }
```

Finally, to clean up the _positionInfo_ array, we resize it to remove any empty or unused elements after all positions have been processed.

```
ArrayResize(positionInfo, totalPositionsFound);
```

Here is the full implementation, of the _SavePositionsData()_ function, with all the code segments included.

```
void SavePositionsData()
  {
//- Since every transaction is recorded as a deal, we will begin by scanning the deals and link them
//- to different orders and generate the positions data using the POSITION_ID as the primary and foreign key
   int totalDealInfo = ArraySize(dealInfo);
   ArrayResize(positionInfo, totalDealInfo); //- Resize the position array to match the deals array
   int totalPositionsFound = 0, posIndex = 0;
   if(totalDealInfo == 0) //- Check if we have any deal history available for processing
     {
      return; //- No deal data to process found, we can't go on. exit the function
     }
//- Let us loop through the deals array
   for(int x = totalDealInfo - 1; x >= 0; x--)
     {
      //- First we check if it is an exit deal to close a position
      if(dealInfo[x].entry == DEAL_ENTRY_OUT)
        {
         //- We begin by saving the position id
         ulong positionId = dealInfo[x].positionId;
         bool exitDealFound = false;

         //- Now we check if we have an exit deal from this position and save it's properties
         for(int k = ArraySize(dealInfo) - 1; k >= 0; k--)
           {
            if(dealInfo[k].positionId == positionId)
              {
               if(dealInfo[k].entry == DEAL_ENTRY_IN)
                 {
                  exitDealFound = true;

                  totalPositionsFound++;
                  posIndex = totalPositionsFound - 1;

                  positionInfo[posIndex].openingDealTicket = dealInfo[k].ticket;
                  positionInfo[posIndex].openTime = dealInfo[k].time;
                  positionInfo[posIndex].openPrice = dealInfo[k].price;
                  positionInfo[posIndex].volume = dealInfo[k].volume;
                  positionInfo[posIndex].magic = dealInfo[k].magic;
                  positionInfo[posIndex].comment = dealInfo[k].comment;
                 }
              }
           }

         if(exitDealFound) //- Continue saving the exit deal data
           {
            //- Save the position type
            if(dealInfo[x].type == DEAL_TYPE_BUY)
              {
               //- If the exit deal is a buy, then the position was a sell trade
               positionInfo[posIndex].type = POSITION_TYPE_SELL;
              }
            else
              {
               //- If the exit deal is a sell, then the position was a buy trade
               positionInfo[posIndex].type = POSITION_TYPE_BUY;
              }

            positionInfo[posIndex].positionId = dealInfo[x].positionId;
            positionInfo[posIndex].symbol = dealInfo[x].symbol;
            positionInfo[posIndex].profit = dealInfo[x].profit;
            positionInfo[posIndex].closingDealTicket = dealInfo[x].ticket;
            positionInfo[posIndex].closePrice = dealInfo[x].price;
            positionInfo[posIndex].closeTime = dealInfo[x].time;
            positionInfo[posIndex].swap = dealInfo[x].swap;
            positionInfo[posIndex].commission = dealInfo[x].commission;
            positionInfo[posIndex].tpPrice = dealInfo[x].tpPrice;
            positionInfo[posIndex].tpPips = 0;
            positionInfo[posIndex].slPrice = dealInfo[x].slPrice;
            positionInfo[posIndex].slPips = 0;

            //- Calculate the trade duration in seconds
            positionInfo[posIndex].duration = MathAbs((long)positionInfo[posIndex].closeTime - (long)positionInfo[posIndex].openTime);

            //- Calculate the net profit after swap and commission
            positionInfo[posIndex].netProfit =
               positionInfo[posIndex].profit + positionInfo[posIndex].swap - positionInfo[posIndex].commission;

            //- Get pip values for the position
            if(positionInfo[posIndex].type == POSITION_TYPE_BUY) //- Buy position
              {
               //- Get sl and tp pip values
               if(positionInfo[posIndex].tpPrice > 0)
                 {
                  double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
                  positionInfo[posIndex].tpPips =
                     int((positionInfo[posIndex].tpPrice - positionInfo[posIndex].openPrice) / symbolPoint);
                 }
               if(positionInfo[posIndex].slPrice > 0)
                 {
                  double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
                  positionInfo[posIndex].slPips =
                     int((positionInfo[posIndex].openPrice - positionInfo[posIndex].slPrice) / symbolPoint);
                 }

               //- Get the buy profit in pip value
               double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
               positionInfo[posIndex].pipProfit =
                  int((positionInfo[posIndex].closePrice - positionInfo[posIndex].openPrice) / symbolPoint);
              }
            else //- Sell position
              {
               //- Get sl and tp pip values
               if(positionInfo[posIndex].tpPrice > 0)
                 {
                  double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
                  positionInfo[posIndex].tpPips =
                     int((positionInfo[posIndex].openPrice - positionInfo[posIndex].tpPrice) / symbolPoint);
                 }
               if(positionInfo[posIndex].slPrice > 0)
                 {
                  double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
                  positionInfo[posIndex].slPips =
                     int((positionInfo[posIndex].slPrice - positionInfo[posIndex].openPrice) / symbolPoint);
                 }

               //- Get the sell profit in pip value
               double symbolPoint = SymbolInfoDouble(positionInfo[posIndex].symbol, SYMBOL_POINT);
               positionInfo[posIndex].pipProfit =
                  int((positionInfo[posIndex].openPrice - positionInfo[posIndex].closePrice) / symbolPoint);
              }

            //- Now we scan and get the opening order ticket in the orderInfo array
            for(int k = 0; k < ArraySize(orderInfo); k++) //- Search from the oldest to newest order
              {
               if(
                  orderInfo[k].positionId == positionInfo[posIndex].positionId &&
                  orderInfo[k].state == ORDER_STATE_FILLED
               )
                 {
                  //- Save the order ticket that intiated the position
                  positionInfo[posIndex].openingOrderTicket = orderInfo[k].ticket;
                  positionInfo[posIndex].ticket = positionInfo[posIndex].openingOrderTicket;

                  //- Determine if the position was initiated by a pending order or direct market entry
                  switch(orderInfo[k].type)
                    {
                     //- Pending order entry
                     case ORDER_TYPE_BUY_LIMIT:
                     case ORDER_TYPE_BUY_STOP:
                     case ORDER_TYPE_SELL_LIMIT:
                     case ORDER_TYPE_SELL_STOP:
                     case ORDER_TYPE_BUY_STOP_LIMIT:
                     case ORDER_TYPE_SELL_STOP_LIMIT:
                        positionInfo[posIndex].initiatedByPendingOrder = true;
                        positionInfo[posIndex].initiatingOrderType = orderInfo[k].type;
                        break;

                     //- Direct market entry
                     default:
                        positionInfo[posIndex].initiatedByPendingOrder = false;
                        positionInfo[posIndex].initiatingOrderType = orderInfo[k].type;
                        break;
                    }

                  break; //--- We have everything we need, exit the orderInfo loop
                 }
              }
           }
        }
      else //--- Position id not found
        {
         continue;//- skip to the next iteration
        }
     }
//- Resize the positionInfo array and delete all the indexes that have zero values
   ArrayResize(positionInfo, totalPositionsFound);
  }
```

### Print Positions History Function

The _PrintPositionsHistory()_ function is designed to display a detailed history of closed positions within a specified timeframe. It accesses previously saved data from the _positionInfo_ array and prints relevant details for each position. This function is exportable, making it accessible to external modules or MQL5 apps that utilize this library. Its implementation will follow a similar structure to other print functions we have developed. Here is the full implementation, with detailed comments for clarity.

```
void PrintPositionsHistory(datetime fromDateTime, datetime toDateTime) export
  {
//- Get and save the deals, orders, positions history for the specified period
   GetHistoryData(fromDateTime, toDateTime, GET_POSITIONS_HISTORY_DATA);
   int totalPositionsClosed = ArraySize(positionInfo);
   if(totalPositionsClosed <= 0)
     {
      Print("");
      Print(__FUNCTION__, ": No position history found for the specified period.");
      return; //- Exit the function
     }

   Print("");
   Print(__FUNCTION__, "-------------------------------------------------------------------------------");
   Print(
      "Found a total of ", totalPositionsClosed,
      " positions closed between (", fromDateTime, ") and (", toDateTime, ")."
   );

   for(int r = 0; r < totalPositionsClosed; r++)
     {
      Print("---------------------------------------------------------------------------------------------------");
      Print("Position #", (r + 1));
      Print("Symbol: ", positionInfo[r].symbol);
      Print("Time Open: ", positionInfo[r].openTime);
      Print("Ticket: ", positionInfo[r].ticket);
      Print("Type: ", EnumToString(positionInfo[r].type));
      Print("Volume: ", positionInfo[r].volume);
      Print("0pen Price: ", positionInfo[r].openPrice);
      Print("SL Price: ", positionInfo[r].slPrice, " (slPips: ", positionInfo[r].slPips, ")");
      Print("TP Price: ", positionInfo[r].tpPrice, " (tpPips: ", positionInfo[r].tpPips, ")");
      Print("Close Price: ", positionInfo[r].closePrice);
      Print("Close Time: ", positionInfo[r].closeTime);
      Print("Trade Duration: ", positionInfo[r].duration);
      Print("Swap: ", positionInfo[r].swap, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Commission: ", positionInfo[r].commission, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Profit: ", positionInfo[r].profit, " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("Net profit: ", DoubleToString(positionInfo[r].netProfit, 2), " ", AccountInfoString(ACCOUNT_CURRENCY));
      Print("pipProfit: ", positionInfo[r].pipProfit);
      Print("Initiating Order Type: ", EnumToString(positionInfo[r].initiatingOrderType));
      Print("Initiated By Pending Order: ", positionInfo[r].initiatedByPendingOrder);
      Print("Comment: ", positionInfo[r].comment);
      Print("Magic: ", positionInfo[r].magic);
      Print("");
     }
  }
```

### Save Pending Orders Data Function

The _SavePendingOrdersData()_ function processes the data from the order history to generate and save the pending order history. This function essentially filters pending orders from the order history, stores key details, and calculates specific values such as the number of _pips_ for _take profit (TP)_ and _stop loss (SL)_ levels. It plays a crucial role in tracking the lifecycle of pending orders, helping to generate an accurate order history, and augmenting the system with data on how each pending order was structured and executed.

MQL5 doesn't currently have standard functions like _HistoryPendingOrderSelect()_ or _HistoryPendingOrdersTotal()_ for directly accessing historical pending order data. As a result, we must create a custom function to scan the history of the orders and build a data source containing all filled or canceled pending orders within a given historical timeframe.

Let us begin by defining the function signature. Since this function will only be used internally by the EX5 library core modules, it will not be exportable.

```
void SavePendingOrdersData()
  {
//-- Function's code will go here
  }
```

Next, we calculate the total number of orders in the _orderInfo_ array, which holds the details of all orders. We'll resize the _pendingOrderInfo_ array to accommodate the total number of orders initially, ensuring enough space to store the filtered pending orders.

```
int totalOrderInfo = ArraySize(orderInfo);
ArrayResize(pendingOrderInfo, totalOrderInfo);
int totalPendingOrdersFound = 0, pendingIndex = 0;
```

If there are no orders to process ( _i.e., totalOrderInfo == 0_), we immediately exit the function as there is no pending order data to handle.

```
if(totalOrderInfo == 0)
  {
   return;
  }
```

Now, we loop through the orders in reverse order to ensure that we are processing the most recent orders first. Inside the loop, we check if the current order is a pending order by evaluating its type. The saved orders history will include pending orders ( _like buy limits, sell stops, etc._) that were either executed ( _filled_) and converted into positions or canceled without becoming positions.

```
for(int x = totalOrderInfo - 1; x >= 0; x--)
  {
   if(
      orderInfo[x].type == ORDER_TYPE_BUY_LIMIT || orderInfo[x].type == ORDER_TYPE_BUY_STOP ||
      orderInfo[x].type == ORDER_TYPE_SELL_LIMIT || orderInfo[x].type == ORDER_TYPE_SELL_STOP ||
      orderInfo[x].type == ORDER_TYPE_BUY_STOP_LIMIT || orderInfo[x].type == ORDER_TYPE_SELL_STOP_LIMIT
   )
     {
      totalPendingOrdersFound++;
      pendingIndex = totalPendingOrdersFound - 1;

      //-- Save the pending order properties into the pendingOrderInfo array

     }
```

If the order is a pending order, we save its properties ( _e.g., type, state, position ID, ticket, symbol, time, and more_) into the _pendingOrderInfo_ array.

```
pendingOrderInfo[pendingIndex].type = orderInfo[x].type;
pendingOrderInfo[pendingIndex].state = orderInfo[x].state;
pendingOrderInfo[pendingIndex].positionId = orderInfo[x].positionId;
pendingOrderInfo[pendingIndex].ticket = orderInfo[x].ticket;
pendingOrderInfo[pendingIndex].symbol = orderInfo[x].symbol;
pendingOrderInfo[pendingIndex].timeSetup = orderInfo[x].timeSetup;
pendingOrderInfo[pendingIndex].expirationTime = orderInfo[x].expirationTime;
pendingOrderInfo[pendingIndex].timeDone = orderInfo[x].timeDone;
pendingOrderInfo[pendingIndex].typeTime = orderInfo[x].typeTime;
pendingOrderInfo[pendingIndex].priceOpen = orderInfo[x].priceOpen;
pendingOrderInfo[pendingIndex].tpPrice = orderInfo[x].tpPrice;
pendingOrderInfo[pendingIndex].slPrice = orderInfo[x].slPrice;
```

We then calculate the number of _pips_ for both the _take profit (TP)_ and _stop loss (SL)_ levels, if they are specified. To do this, we use the _symbol's point_ value to determine the number of pips.

```
if(pendingOrderInfo[pendingIndex].tpPrice > 0)
  {
   double symbolPoint = SymbolInfoDouble(pendingOrderInfo[pendingIndex].symbol, SYMBOL_POINT);
   pendingOrderInfo[pendingIndex].tpPips =
      (int)MathAbs((pendingOrderInfo[pendingIndex].tpPrice - pendingOrderInfo[pendingIndex].priceOpen) / symbolPoint);
  }
if(pendingOrderInfo[pendingIndex].slPrice > 0)
  {
   double symbolPoint = SymbolInfoDouble(pendingOrderInfo[pendingIndex].symbol, SYMBOL_POINT);
   pendingOrderInfo[pendingIndex].slPips =
      (int)MathAbs((pendingOrderInfo[pendingIndex].slPrice - pendingOrderInfo[pendingIndex].priceOpen) / symbolPoint);
  }
```

We also save additional properties such as the order's _magic number, reason, filling type, comment, initial volume,_ and _stop limit_ price.

```
pendingOrderInfo[pendingIndex].magic = orderInfo[x].magic;
pendingOrderInfo[pendingIndex].reason = orderInfo[x].reason;
pendingOrderInfo[pendingIndex].typeFilling = orderInfo[x].typeFilling;
pendingOrderInfo[pendingIndex].comment = orderInfo[x].comment;
pendingOrderInfo[pendingIndex].volumeInitial = orderInfo[x].volumeInitial;
pendingOrderInfo[pendingIndex].priceStopLimit = orderInfo[x].priceStopLimit;
```

Once we have processed all the orders, we resize the _pendingOrderInfo_ array to remove any empty or unused elements, ensuring that the array only contains the relevant pending order data.

```
ArrayResize(pendingOrderInfo, totalPendingOrdersFound);
```

Here is the full implementation, of the _SavePendingOrdersData()_ function, with all the code segments included.

```
void SavePendingOrdersData()
  {
//- Let us begin by scanning the orders and link them to different deals
   int totalOrderInfo = ArraySize(orderInfo);
   ArrayResize(pendingOrderInfo, totalOrderInfo);
   int totalPendingOrdersFound = 0, pendingIndex = 0;
   if(totalOrderInfo == 0)
     {
      return; //- No order data to process found, we can't go on. exit the function
     }

   for(int x = totalOrderInfo - 1; x >= 0; x--)
     {
      //- Check if it is a pending order and save its properties
      if(
         orderInfo[x].type == ORDER_TYPE_BUY_LIMIT || orderInfo[x].type == ORDER_TYPE_BUY_STOP ||
         orderInfo[x].type == ORDER_TYPE_SELL_LIMIT || orderInfo[x].type == ORDER_TYPE_SELL_STOP ||
         orderInfo[x].type == ORDER_TYPE_BUY_STOP_LIMIT || orderInfo[x].type == ORDER_TYPE_SELL_STOP_LIMIT
      )
        {
         totalPendingOrdersFound++;
         pendingIndex = totalPendingOrdersFound - 1;

         pendingOrderInfo[pendingIndex].type = orderInfo[x].type;
         pendingOrderInfo[pendingIndex].state = orderInfo[x].state;
         pendingOrderInfo[pendingIndex].positionId = orderInfo[x].positionId;
         pendingOrderInfo[pendingIndex].ticket = orderInfo[x].ticket;
         pendingOrderInfo[pendingIndex].symbol = orderInfo[x].symbol;
         pendingOrderInfo[pendingIndex].timeSetup = orderInfo[x].timeSetup;
         pendingOrderInfo[pendingIndex].expirationTime = orderInfo[x].expirationTime;
         pendingOrderInfo[pendingIndex].timeDone = orderInfo[x].timeDone;
         pendingOrderInfo[pendingIndex].typeTime = orderInfo[x].typeTime;
         pendingOrderInfo[pendingIndex].priceOpen = orderInfo[x].priceOpen;
         pendingOrderInfo[pendingIndex].tpPrice = orderInfo[x].tpPrice;
         pendingOrderInfo[pendingIndex].slPrice = orderInfo[x].slPrice;

         if(pendingOrderInfo[pendingIndex].tpPrice > 0)
           {
            double symbolPoint = SymbolInfoDouble(pendingOrderInfo[pendingIndex].symbol, SYMBOL_POINT);
            pendingOrderInfo[pendingIndex].tpPips =
               (int)MathAbs((pendingOrderInfo[pendingIndex].tpPrice - pendingOrderInfo[pendingIndex].priceOpen) / symbolPoint);
           }
         if(pendingOrderInfo[pendingIndex].slPrice > 0)
           {
            double symbolPoint = SymbolInfoDouble(pendingOrderInfo[pendingIndex].symbol, SYMBOL_POINT);
            pendingOrderInfo[pendingIndex].slPips =
               (int)MathAbs((pendingOrderInfo[pendingIndex].slPrice - pendingOrderInfo[pendingIndex].priceOpen) / symbolPoint);
           }

         pendingOrderInfo[pendingIndex].magic = orderInfo[x].magic;
         pendingOrderInfo[pendingIndex].reason = orderInfo[x].reason;
         pendingOrderInfo[pendingIndex].typeFilling = orderInfo[x].typeFilling;
         pendingOrderInfo[pendingIndex].comment = orderInfo[x].comment;
         pendingOrderInfo[pendingIndex].volumeInitial = orderInfo[x].volumeInitial;
         pendingOrderInfo[pendingIndex].priceStopLimit = orderInfo[x].priceStopLimit;

        }
     }
//--Resize the pendingOrderInfo array and delete all the indexes that have zero values
   ArrayResize(pendingOrderInfo, totalPendingOrdersFound);
  }
```

### Print Pending Orders History Function

The _PrintPendingOrdersHistory()_ function is designed to display a detailed history of filled or canceled pending orders within a specified timeframe. It accesses previously saved data from the _pendingOrderInfo_ array and prints relevant details for each pending order. This function is exportable, making it accessible to external modules or MQL5 apps that utilize this EX5 library. Its implementation will follow a similar structure to other print functions we have developed. Here is the full, implementation with detailed comments for clarity.

```
void PrintPendingOrdersHistory(datetime fromDateTime, datetime toDateTime) export
  {
//- Get and save the pending orders history for the specified period
   GetHistoryData(fromDateTime, toDateTime, GET_PENDING_ORDERS_HISTORY_DATA);
   int totalPendingOrders = ArraySize(pendingOrderInfo);
   if(totalPendingOrders <= 0)
     {
      Print("");
      Print(__FUNCTION__, ": No pending orders history found for the specified period.");
      return; //- Exit the function
     }

   Print("");
   Print(__FUNCTION__, "-------------------------------------------------------------------------------");
   Print(
      "Found a total of ", totalPendingOrders,
      " pending orders filled or cancelled between (", fromDateTime, ") and (", toDateTime, ")."
   );

   for(int r = 0; r < totalPendingOrders; r++)
     {
      Print("---------------------------------------------------------------------------------------------------");
      Print("Pending Order #", (r + 1));
      Print("Symbol: ", pendingOrderInfo[r].symbol);
      Print("Time Setup: ", pendingOrderInfo[r].timeSetup);
      Print("Type: ", EnumToString(pendingOrderInfo[r].type));
      Print("Ticket: ", pendingOrderInfo[r].ticket);
      Print("State: ", EnumToString(pendingOrderInfo[r].state));
      Print("Time Done: ", pendingOrderInfo[r].timeDone);
      Print("Volume Initial: ", pendingOrderInfo[r].volumeInitial);
      Print("Price Open: ", pendingOrderInfo[r].priceOpen);
      Print("SL Price: ", pendingOrderInfo[r].slPrice, " (slPips: ", pendingOrderInfo[r].slPips, ")");
      Print("TP Price: ", pendingOrderInfo[r].tpPrice, " (slPips: ", pendingOrderInfo[r].slPips, ")");
      Print("Expiration Time: ", pendingOrderInfo[r].expirationTime);
      Print("Position ID: ", pendingOrderInfo[r].positionId);
      Print("Price Stop Limit: ", pendingOrderInfo[r].priceStopLimit);
      Print("Type Filling: ", EnumToString(pendingOrderInfo[r].typeFilling));
      Print("Type Time: ", EnumToString(pendingOrderInfo[r].typeTime));
      Print("Reason: ", EnumToString(pendingOrderInfo[r].reason));
      Print("Comment: ", pendingOrderInfo[r].comment);
      Print("Magic: ", pendingOrderInfo[r].magic);
      Print("");
     }
  }
```

### Conclusion

In this article, we explored how to use MQL5 to retrieve transaction history data for _orders_ and _deals_. You learned how to leverage this data to generate the history of _closed positions_ and _pending orders_, complete with an audit trail that tracks the lifecycle of each closed position. This includes its source, how it was closed, and other valuable details such as _net profit, pip profit, pip value_ for _stop loss_ and _take profit, trade duration,_ and more.

We also developed the core functions of the _History Manager EX5_ library, enabling us to _query, save,_ and _categorize_ different types of historical data. These foundational functions form part of the library engine that handles its inner workings. However, there is still more to be done. Most of the functions we created in this article are preparatory, setting the stage for a more user-oriented library.

In the next article, we will expand the _History Manager EX5_ library by introducing exportable functions designed to sort and analyze historical data based on common user requirements. For instance, you will be able to retrieve properties of the most recently closed positions, analyze the last filled or canceled pending orders, check the last closed position for a specific symbol, calculate the current day's closed profit, and determine weekly pip profits, among other functionalities.

Additionally, we will include advanced sorting and analytics modules to generate detailed trade reports similar to those produced by the _MetaTrader 5 Strategy Tester_. These reports will analyze real trading history data, offering insights into the performance of an Expert Advisor or trading strategy. You will also be able to programmatically filter and sort this data by parameters such as symbols or magic numbers.

To make implementation seamless, we will provide comprehensive documentation for the _History Manager EX5_ library, along with practical use-case examples. These examples will demonstrate how to integrate the library into your projects and perform effective trade analysis. Furthermore, we will include simple Expert Advisor examples and step-by-step demonstrations to help you optimize your trading strategies and take full advantage of the library's capabilities.

You can find the attached _HistoryManager.mq5_ source code file at the end of this article. Thank you for following along, and I wish you great success in your trading and MQL5 programming journey!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16528.zip "Download all attachments in the single ZIP archive")

[HistoryManager.mq5](https://www.mql5.com/en/articles/download/16528/historymanager.mq5 "Download HistoryManager.mq5")(33.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**[Go to discussion](https://www.mql5.com/en/forum/477937)**

![Creating a Trading Administrator Panel in MQL5 (Part VIII): Analytics Panel](https://c.mql5.com/2/104/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_VIII____LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part VIII): Analytics Panel](https://www.mql5.com/en/articles/16356)

Today, we delve into incorporating useful trading metrics within a specialized window integrated into the Admin Panel EA. This discussion focuses on the implementation of MQL5 to develop an Analytics Panel and highlights the value of the data it provides to trading administrators. The impact is largely educational, as valuable lessons are drawn from the development process, benefiting both upcoming and experienced developers. This feature demonstrates the limitless opportunities this development series offers in equipping trade managers with advanced software tools. Additionally, we'll explore the implementation of the PieChart and ChartCanvas classes as part of the continued expansion of the Trading Administrator panel’s capabilities.

![Utilizing CatBoost Machine Learning model as a Filter for Trend-Following Strategies](https://c.mql5.com/2/104/yandex_catboost_2__1.png)[Utilizing CatBoost Machine Learning model as a Filter for Trend-Following Strategies](https://www.mql5.com/en/articles/16487)

CatBoost is a powerful tree-based machine learning model that specializes in decision-making based on stationary features. Other tree-based models like XGBoost and Random Forest share similar traits in terms of their robustness, ability to handle complex patterns, and interpretability. These models have a wide range of uses, from feature analysis to risk management. In this article, we're going to walk through the procedure of utilizing a trained CatBoost model as a filter for a classic moving average cross trend-following strategy.

![Neural Networks Made Easy (Part 95): Reducing Memory Consumption in Transformer Models](https://c.mql5.com/2/81/Neural_networks_are_easy_Part_95_LOGO.png)[Neural Networks Made Easy (Part 95): Reducing Memory Consumption in Transformer Models](https://www.mql5.com/en/articles/15117)

Transformer architecture-based models demonstrate high efficiency, but their use is complicated by high resource costs both at the training stage and during operation. In this article, I propose to get acquainted with algorithms that allow to reduce memory usage of such models.

![Trading with the MQL5 Economic Calendar (Part 5): Enhancing the Dashboard with Responsive Controls and Filter Buttons](https://c.mql5.com/2/104/Trading_with_the_MQL5_Economic_Calendar_Part_5___LOGO.png)[Trading with the MQL5 Economic Calendar (Part 5): Enhancing the Dashboard with Responsive Controls and Filter Buttons](https://www.mql5.com/en/articles/16404)

In this article, we create buttons for currency pair filters, importance levels, time filters, and a cancel option to improve dashboard control. These buttons are programmed to respond dynamically to user actions, allowing seamless interaction. We also automate their behavior to reflect real-time changes on the dashboard. This enhances the overall functionality, mobility, and responsiveness of the panel.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/16528&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062599705877194055)

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