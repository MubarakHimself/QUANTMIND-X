---
title: MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions
url: https://www.mql5.com/en/articles/16742
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:43:09.694856
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/16742&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069684486360205578)

MetaTrader 5 / Examples


### Introduction

Accessing the details of the most recently filled pending order is particularly valuable in scenarios where your trading logic depends on the type of the last filled pending order. For instance, you can leverage this data to refine your trading strategies based on whether the most recent filled order was a _Buy Limit, Sell Stop, Buy Stop, Sell Limit, Buy Stop Limit_, or _Sell Stop Limit_. Understanding the type of order can provide insights into market conditions and inform adjustments to your approach, such as adapting entry or exit points.

This information is also crucial for gathering and analyzing historical trading data to optimize your trading systems or gathering data on how fast your broker fills or executes pending orders when they are triggered and activated. By examining details like _slippage_, the time elapsed from order placement to execution, and the conditions under which the order was filled, you can identify patterns and areas for improvement in your strategy. Additionally, this data allows you to monitor execution quality, ensuring that your orders are being filled as expected and helping you address potential inefficiencies in your trading operations. Such detailed analysis can enhance decision-making and lead to more effective and robust trading strategies over time.

This _history management EX5_ library simplifies the process of retrieving details and properties of the most recently filled pending order. By invoking a single function, you can access this data without the additional effort of specifying a _period_ for the trade history search—the EX5 library seamlessly manages that for you. All you need to do is provide the relevant history-querying function with a variable to store the specific pending order property as input. The function will then save the retrieved details into the referenced variable you supply.

If the function successfully retrieves the requested data, it returns _true_, indicating success. If the specified pending order data is unavailable, it returns _false_. This streamlined approach eliminates unnecessary complexity, allowing you to focus on analyzing or integrating the retrieved data into your trading strategies, ensuring both efficiency and accuracy in your trading operations.

To get started, we will open the **_HistoryManager.mq5_** file from the [previous article](https://www.mql5.com/en/articles/16681) where we created the position history retrieval functions and begin by creating the _GetLastFilledPendingOrderData()_ function. Make sure you have downloaded the **_HistoryManager.mq5_** source file, which you will find attached at the end of the [previous article](https://www.mql5.com/en/articles/16681). We will continue adding new code below the _LastClosedPositionDuration()_ function, which is where we left off previously.

### Get Last Filled Pending Order Data Function

The _GetLastFilledPendingOrderData()_ function is responsible for retrieving the properties of the last filled pending order from the available trading history. It systematically checks the history data for any pending orders that have been filled, and if such an order is found, it saves the relevant data in the referenced _getLastFilledPendingOrderData_ variable.

This function works in tandem with the _FetchHistoryByCriteria()_ function to fetch the necessary trading history data. It ensures that only the filled pending orders are considered and updates the provided structure with the data of the last filled pending order.

Let us begin by defining the _GetLastFilledPendingOrderData()_ function signature. Since this function is marked as _export_, it is accessible to any MQL5 source code file that imports the compiled EX5 library. The function references the _getLastFilledPendingOrderData_ variable, where the retrieved data will be stored.

```
bool GetLastFilledPendingOrderData(PendingOrderData &getLastFilledPendingOrderData) export
  {
//-- Function logic will be implemented here
  }
```

Next, we will use the _FetchHistoryByCriteria()_ function and pass the GET\_PENDING\_ORDERS\_HISTORY\_DATA constant as the function parameter to ensure we have the required pending order history data. If no data is found, we will print a message and return _false_.

```
if(!FetchHistoryByCriteria(GET_PENDING_ORDERS_HISTORY_DATA))
  {
   Print(__FUNCTION__, ": No trading history available. Last filled pending order can't be retrieved.");
   return(false);
  }
```

If we successfully retrieve the data, we proceed to find the last filled pending order. We get the total number of pending orders from the _GetTotalDataInfoSize()_ function, then loop through the _pendingOrderInfo_ array to find a pending order that has been filled. Once found, we assign its data to the _getLastFilledPendingOrderData_ structure. Finally, the function returns _true_ if a filled pending order is found and its data is saved. If no filled pending order is found, the function will exit and return _false_.

```
int totalPendingOrderInfo = GetTotalDataInfoSize(GET_PENDING_ORDERS_HISTORY_DATA);
for(int x = 0; x < totalPendingOrderInfo; x++)
  {
   if(pendingOrderInfo[x].state == ORDER_STATE_FILLED)
     {
      getLastFilledPendingOrderData = pendingOrderInfo[x];
      break;
     }
  }
  return(true);
```

Here is the full implementation of the _GetLastFilledPendingOrderData()_ function with all the code segments in their correct sequence.

```
bool GetLastFilledPendingOrderData(PendingOrderData &getLastFilledPendingOrderData) export
  {
   if(!FetchHistoryByCriteria(GET_PENDING_ORDERS_HISTORY_DATA))
     {
      Print(__FUNCTION__, ": No trading history available. Last filled pending order can't be retrieved.");
      return(false);
     }

//-- Save the last filled pending order data in the referenced getLastFilledPendingOrderData variable
   int totalPendingOrderInfo = GetTotalDataInfoSize(GET_PENDING_ORDERS_HISTORY_DATA);
   for(int x = 0; x < totalPendingOrderInfo; x++)
     {
      if(pendingOrderInfo[x].state == ORDER_STATE_FILLED)
        {
         getLastFilledPendingOrderData = pendingOrderInfo[x];
         break;
        }
     }
   return(true);
  }
```

### Last Filled Pending Order Type Function

The _LastFilledPendingOrderType()_ function is responsible for determining the type of the most recently filled pending order. It stores this type in the referenced variable _lastFilledPendingOrderType._ This capability is essential for applications that need to analyze order execution types or track the performance of specific order categories.

Let us begin by defining the function signature. Since this function is marked as export, it will be accessible for use in any MQL5 source file that imports the EX5 library. The function accepts the referenced _lastFilledPendingOrderType_ variable as an input, where the type of the last filled pending order will be stored.

```
bool LastFilledPendingOrderType(ENUM_ORDER_TYPE &lastFilledPendingOrderType) export
  {
//-- Function logic will be implemented here
  }
```

We will start by declaring a variable of type _PendingOrderData_ named _lastFilledPendingOrderInfo_. This variable will temporarily store the details of the most recently filled pending order.

```
PendingOrderData lastFilledPendingOrderInfo;
```

Next, we will use the _GetLastFilledPendingOrderData()_ function to retrieve the details of the last filled pending order. If the operation is successful, we will extract the _type_ field from the _lastFilledPendingOrderInfo_ structure, store it in the referenced _lastFilledPendingOrderType_ variable, and return _true_. If the retrieval fails, we will skip updating the referenced variable and return _false_ to indicate that no data was found.

```
if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
  {
   lastFilledPendingOrderType = lastFilledPendingOrderInfo.type;
   return(true);
  }
return(false);
```

Here is the full implementation of the _LastFilledPendingOrderType()_ function.

```
bool LastFilledPendingOrderType(ENUM_ORDER_TYPE &lastFilledPendingOrderType) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderType = lastFilledPendingOrderInfo.type;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Symbol Function

The _LastFilledPendingOrderSymbol()_ function retrieves the symbol of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderSymbol_ variable. It accepts the referenced _lastFilledPendingOrderSymbol_ variable as input to store the _symbol_.

First, we declare a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, to temporarily hold the details of the last filled pending order. We then use the _GetLastFilledPendingOrderData()_ function to retrieve the order details. If successful, we extract the symbol from _lastFilledPendingOrderInfo_, store it in _lastFilledPendingOrderSymbol_, and return _true_. If the retrieval fails, we return _false_ without updating the referenced variable.

Here’s the implementation of the _LastFilledPendingOrderSymbol()_ function.

```
bool LastFilledPendingOrderSymbol(string &lastFilledPendingOrderSymbol) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderSymbol = lastFilledPendingOrderInfo.symbol;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Ticket Function

The _LastFilledPendingOrderTicket()_ function retrieves the _ticket_ number of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderTicket_ variable. It accepts the _lastFilledPendingOrderTicket_ variable as input, where the _ticket_ number will be saved.

We start by defining a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which serves as a temporary container for the details of the most recently filled pending order. Next, we call the _GetLastFilledPendingOrderData()_ function to fetch the order information. Upon successful retrieval, we extract the _ticket_ from _lastFilledPendingOrderInfo_, save it into the _lastFilledPendingOrderTicket_ variable, and indicate success by returning _true_. If the retrieval is unsuccessful, the function returns _false_, leaving the referenced variable unchanged.

Here’s the implementation of the _LastFilledPendingOrderTicket()_ function.

```
bool LastFilledPendingOrderTicket(ulong &lastFilledPendingOrderTicket) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTicket = lastFilledPendingOrderInfo.ticket;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Price Open Function

The _LastFilledPendingOrderPriceOpen()_ function retrieves the _opening price_ of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderPriceOpen_ variable. It accepts the _lastFilledPendingOrderPriceOpen_ variable as input to store the opening price.

First, we declare a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, to temporarily hold the details of the last filled pending order. We then use the _GetLastFilledPendingOrderData()_ function to retrieve the order details. If successful, we extract the _priceOpen_ from _lastFilledPendingOrderInfo_, store it in _lastFilledPendingOrderPriceOpen_, and return _true_. If the retrieval fails, we return _false_ without updating the referenced variable.

Here’s the implementation of the _LastFilledPendingOrderPriceOpen()_ function.

```
bool LastFilledPendingOrderPriceOpen(double &lastFilledPendingOrderPriceOpen) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderPriceOpen = lastFilledPendingOrderInfo.priceOpen;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Stop Loss Price Function

The _LastFilledPendingOrderSlPrice()_ function retrieves the _stop loss price_ of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderSlPrice_ variable. It accepts the _lastFilledPendingOrderSlPrice_ variable as input to store the retrieved stop loss price. This function is particularly useful for scenarios where tracking or analyzing stop loss levels of recently filled pending orders is required.

Let us begin by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_ to temporarily hold the details of the last filled pending order. We then use the _GetLastFilledPendingOrderData()_ function to fetch the details of the order.

If the data retrieval is successful, the _slPrice_ field from _lastFilledPendingOrderInfo_ is extracted and saved in the referenced _lastFilledPendingOrderSlPrice_ variable. The function then returns _true_ to indicate success. If the retrieval fails, the function exits without updating the referenced variable and returns _false_ to signal that no data was found.

Here’s the full implementation of the _LastFilledPendingOrderSlPrice()_ function:

```
bool LastFilledPendingOrderSlPrice(double &lastFilledPendingOrderSlPrice) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderSlPrice = lastFilledPendingOrderInfo.slPrice;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Take Profit Price Function

The _LastFilledPendingOrderTpPrice()_ function retrieves the _take profit price_ of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderTpPrice_ variable. This function is essential for applications that analyze or manage take profit levels of filled pending orders. It accepts the _lastFilledPendingOrderTpPrice_ variable as input to store the take profit price.

We start by declaring a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, to temporarily hold the details of the last filled pending order. We then use the _GetLastFilledPendingOrderData()_ function to retrieve the order details. If the retrieval is successful, we extract the _tpPrice_ from _lastFilledPendingOrderInfo_, store it in the _lastFilledPendingOrderTpPrice_ variable, and return _true_. If the retrieval fails, we return _false_ without updating the referenced variable.

Here’s the implementation of the _LastFilledPendingOrderTpPrice()_ function.

```
bool LastFilledPendingOrderTpPrice(double &lastFilledPendingOrderTpPrice) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTpPrice = lastFilledPendingOrderInfo.tpPrice;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Stop Loss Pips Function

The _LastFilledPendingOrderSlPips()_ function retrieves the _stop-loss pips_ value of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderSlPips_ variable. This function is vital for analyzing the risk parameters associated with executed pending orders.

First, we declare a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_. This variable will temporarily hold the details of the last filled pending order. We then use the _GetLastFilledPendingOrderData()_ function to fetch the order details. If the operation is successful, we extract the _slPips_ field from the _lastFilledPendingOrderInfo_ structure and store it in the referenced _lastFilledPendingOrderSlPips_ variable. The function then returns _true_ to indicate success. If the retrieval process fails, the referenced variable remains unchanged, and the function returns _false_ to signal that no data was found.

Here’s the complete implementation of the _LastFilledPendingOrderSlPips()_ function,

```
bool LastFilledPendingOrderSlPips(int &lastFilledPendingOrderSlPips) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderSlPips = lastFilledPendingOrderInfo.slPips;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Take Profit Pips Function

The _LastFilledPendingOrderTpPips()_ function retrieves the _take profit pips_ value of the most recently filled pending order and saves it in the referenced _lastFilledPendingOrderTpPips_ variable. This functionality is valuable for tracking the take profit levels of executed pending orders and can be used for performance analysis or strategy adjustments.

To implement this, we first declare a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_. This variable is used to temporarily hold the details of the most recently filled pending order. We then call the _GetLastFilledPendingOrderData()_ function to retrieve the order's details. If the retrieval is successful, the _tpPips_ value is extracted from _lastFilledPendingOrderInfo_ and stored in the _lastFilledPendingOrderTpPips_ variable. The function then returns _true_. If the retrieval fails, the referenced variable remains unchanged, and the function returns _false_ to indicate the failure.

Here is the complete implementation of the _LastFilledPendingOrderTpPips()_ function.

```
bool LastFilledPendingOrderTpPips(int &lastFilledPendingOrderTpPips) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTpPips = lastFilledPendingOrderInfo.tpPips;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Time Setup Function

The _LastFilledPendingOrderTimeSetup()_ function retrieves the _time_ when the most recently filled pending order was set up. It stores this time in the referenced _lastFilledPendingOrderTimeSetup_ variable. This function is essential for tracking when specific pending orders were initiated, enabling time-based analysis of order activity.

The _LastFilledPendingOrderTimeSetup()_ function accepts a referenced variable, _lastFilledPendingOrderTimeSetup_, as input. This variable will hold the _setup time_ of the last filled pending order after the function executes. To achieve this, the function begins by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_. This variable is used to temporarily store the details of the last filled pending order.

Next, the _GetLastFilledPendingOrderData()_ function is called to retrieve the details of the last filled pending order. If the retrieval is successful, the _timeSetup_ field from the _lastFilledPendingOrderInfo_ structure is extracted and stored in the _lastFilledPendingOrderTimeSetup_ variable. The function then returns _true_ to indicate success. If the retrieval fails, the function returns _false_ without updating the referenced variable.

Here is the full implementation of the _LastFilledPendingOrderTimeSetup()_ function.

```
bool LastFilledPendingOrderTimeSetup(datetime &lastFilledPendingOrderTimeSetup) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTimeSetup = lastFilledPendingOrderInfo.timeSetup;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Time Done Function

The _LastFilledPendingOrderTimeDone()_ function retrieves the time at which the most recently filled pending order was triggered ( _time done_) and saves it in the referenced _lastFilledPendingOrderTimeDone_ variable. This function is essential for tracking the execution times of filled pending orders, allowing you to analyze order timing or create detailed reports.

The function takes the referenced _lastFilledPendingOrderTimeDone_ variable as input to store the _time done_. It begins by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which is used to temporarily store the details of the most recently filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the operation succeeds, the function extracts the _timeDone_ field from the _lastFilledPendingOrderInfo_ structure and saves it in the referenced _lastFilledPendingOrderTimeDone_ variable. It then returns _true_ to indicate success.

If the retrieval fails, the function does not update the referenced variable and instead returns _false_ to signal that no valid data was found.

Here’s the full implementation of the _LastFilledPendingOrderTimeDone()_ function.

```
bool LastFilledPendingOrderTimeDone(datetime &lastFilledPendingOrderTimeDone) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTimeDone = lastFilledPendingOrderInfo.timeDone;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Expiration Time Function

The _LastFilledPendingOrderExpirationTime()_ function retrieves the _expiration time_ of the most recently filled pending order and saves it in the referenced _lastFilledPendingOrderExpirationTime_ variable. This function is useful for managing and analyzing the lifespan of pending orders, particularly when monitoring their validity period.

The function accepts the referenced _lastFilledPendingOrderExpirationTime_ variable as input to store the expiration time. It starts by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which serves as a temporary container for the details of the most recently filled pending order.

The function then invokes _GetLastFilledPendingOrderData()_ to fetch the data for the last filled pending order. If the operation is successful, it extracts the _expirationTime_ field from the _lastFilledPendingOrderInfo_ structure and saves it in the referenced _lastFilledPendingOrderExpirationTime_ variable. It returns _true_ to indicate the successful retrieval of the data.

If the retrieval fails, the function leaves the referenced variable unchanged and returns _false_ to signal the absence of valid data.

Here’s the full implementation of the _LastFilledPendingOrderExpirationTime()_ function.

```
bool LastFilledPendingOrderExpirationTime(datetime &lastFilledPendingOrderExpirationTime) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderExpirationTime = lastFilledPendingOrderInfo.expirationTime;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Position ID Function

The _LastFilledPendingOrderPositionId()_ function retrieves the _position ID_ of the most recently filled pending order and saves it in the referenced _lastFilledPendingOrderPositionId_ variable. This function is particularly useful for associating pending orders with their corresponding positions, enabling better tracking and management of trading activity.

The function takes the referenced _lastFilledPendingOrderPositionId_ variable as input to store the _position ID_. It begins by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which serves as a temporary container for the details of the most recently filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to fetch the data for the last filled pending order. If the data retrieval is successful, it extracts the _positionId_ field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderPositionId_ variable. It then returns _true_ to indicate that the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to signal the failure.

Here’s the complete implementation of the _LastFilledPendingOrderPositionId()_ function.

```
bool LastFilledPendingOrderPositionId(ulong &lastFilledPendingOrderPositionId) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderPositionId = lastFilledPendingOrderInfo.positionId;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Magic Function

The _LastFilledPendingOrderMagic()_ function retrieves the _magic number_ of the most recently filled pending order and saves it in the referenced _lastFilledPendingOrderMagic_ variable. This function is essential for identifying the unique identifier associated with a specific order, especially when managing multiple strategies or systems within the same trading account.

The function takes the referenced _lastFilledPendingOrderMagic_ variable as input to store the _magic number_. It begins by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which temporarily holds the details of the most recently filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the data retrieval is successful, it extracts the _magic_ field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderMagic_ variable. It then returns _true_ to indicate that the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to indicate the failure.

Here’s the complete implementation of the _LastFilledPendingOrderMagic()_ function.

```
bool LastFilledPendingOrderMagic(ulong &lastFilledPendingOrderMagic) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderMagic = lastFilledPendingOrderInfo.magic;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Reason Function

The _LastFilledPendingOrderReason()_ function retrieves the _reason_ for the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderReason_ variable. This function is useful for tracking the specific reason behind the execution of an order. It helps identify whether the order was executed from a _mobile_, _web_, or _desktop application_, triggered by an _Expert Advisor_ or _script_, or _activated by a stop loss, take profit,_ or _stop-out_ event. This information is crucial for analysis and debugging, providing insights into how and why an order was filled.

The function takes the referenced _lastFilledPendingOrderReason_ variable as input to store the _reason_. It starts by declaring a _PendingOrderData_ variable named _lastFilledPendingOrderInfo_, which temporarily holds the details of the last filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the data retrieval is successful, it extracts the _reason_ field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderReason_ variable. The function then returns _true_ to indicate the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to indicate the failure.

Here’s the complete implementation of the _LastFilledPendingOrderReason()_ function.

```
bool LastFilledPendingOrderReason(ENUM_ORDER_REASON &lastFilledPendingOrderReason) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderReason = lastFilledPendingOrderInfo.reason;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Type Filling Function

The _LastFilledPendingOrderTypeFilling()_ function retrieves the _filling type_ of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderTypeFilling_ variable. This function is important for determining how the pending order was filled, which could be through a _Fill or Kill, Immediate or Cancel, Passive (Book or Cancel)_, or _Return_ policies.

The function takes the referenced _lastFilledPendingOrderTypeFilling_ variable as input to store the _filling type_. It begins by declaring a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, which temporarily holds the details of the last filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the data retrieval is successful, it extracts the _typeFilling_ field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderTypeFilling_ variable. The function then returns _true_ to indicate the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to indicate the failure.

Here’s the complete implementation of the _LastFilledPendingOrderTypeFilling()_ function.

```
bool LastFilledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastFilledPendingOrderTypeFilling) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTypeFilling = lastFilledPendingOrderInfo.typeFilling;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Type Time Function

The _LastFilledPendingOrderTypeTime()_ function retrieves the _type time_ of the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderTypeTime_ variable. This function helps determine the lifetime of the pending order.

The function takes the referenced _lastFilledPendingOrderTypeTime_ variable as input to store the _type time_. It begins by declaring a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, to temporarily hold the details of the last filled pending order.

The function then calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the data retrieval is successful, it extracts the _typeTime_ field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderTypeTime_ variable. The function then returns _true_ to indicate the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to indicate the failure.

Here’s the complete implementation of the _LastFilledPendingOrderTypeTime()_ function.

```
bool LastFilledPendingOrderTypeTime(datetime &lastFilledPendingOrderTypeTime) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderTypeTime = lastFilledPendingOrderInfo.typeTime;
      return(true);
     }
   return(false);
  }
```

### Last Filled Pending Order Comment Function

The _LastFilledPendingOrderComment()_ function retrieves the _comment_ associated with the most recently filled pending order and stores it in the referenced _lastFilledPendingOrderComment_ variable. This function is useful for capturing any additional notes or annotations that may have been added to the pending order.

The function accepts the referenced _lastFilledPendingOrderComment_ variable as input, where it will store the _comment_. It begins by declaring a _PendingOrderData_ variable, _lastFilledPendingOrderInfo_, to temporarily hold the details of the last filled pending order.

Next, the function calls _GetLastFilledPendingOrderData()_ to retrieve the data for the last filled pending order. If the data retrieval is successful, it extracts the comment field from the _lastFilledPendingOrderInfo_ structure and assigns it to the referenced _lastFilledPendingOrderComment_ variable. The function then returns _true_ to indicate the operation was successful.

If the data retrieval fails, the function does not modify the referenced variable and returns _false_ to indicate the failure.

Here’s the complete implementation of the _LastFilledPendingOrderComment()_ function.

```
bool LastFilledPendingOrderComment(string &lastFilledPendingOrderComment) export
  {
   PendingOrderData lastFilledPendingOrderInfo;
   if(GetLastFilledPendingOrderData(lastFilledPendingOrderInfo))
     {
      lastFilledPendingOrderComment = lastFilledPendingOrderInfo.comment;
      return(true);
     }
   return(false);
  }
```

### Conclusion

We have successfully developed the exportable functions to retrieve and store the properties of the most recently filled pending order, enhancing the resourcefulness of the _History Management EX5_ library. These new capabilities provide seamless access to critical data about filled pending orders, enabling more straightforward and effective trade analysis and strategy optimization, while strengthening the library’s ability to handle essential pending order history operations.

In the next article, we will expand the library further by adding functions to fetch and store the properties of the most recently canceled pending order. This addition will enhance the library’s ability to manage and analyze trade history, offering even more valuable tools for your trading toolkit.

The updated **_HistoryManager.mq5_** library source code, including all the functions created in this and previous articles, is available at the end of this article. Thank you for following along, and I look forward to continuing this journey with you in the next article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16742.zip "Download all attachments in the single ZIP archive")

[HistoryManager.mq5](https://www.mql5.com/en/articles/download/16742/historymanager.mq5 "Download HistoryManager.mq5")(68 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**[Go to discussion](https://www.mql5.com/en/forum/479700)**

![Developing a Replay System (Part 56): Adapting the Modules](https://c.mql5.com/2/83/Desenvolvendo_um_sistema_de_Replay_Parte_56__LOGO_3_.png)[Developing a Replay System (Part 56): Adapting the Modules](https://www.mql5.com/en/articles/12000)

Although the modules already interact with each other properly, an error occurs when trying to use the mouse pointer in the replay service. We need to fix this before moving on to the next step. Additionally, we will fix an issue in the mouse indicator code. So this version will be finally stable and properly polished.

![MetaTrader 5 on macOS](https://c.mql5.com/2/0/1045_13.png)[MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)

We provide a special installer for the MetaTrader 5 trading platform on macOS. It is a full-fledged wizard that allows you to install the application natively. The installer performs all the required steps: it identifies your system, downloads and installs the latest Wine version, configures it, and then installs MetaTrader within it. All steps are completed in the automated mode, and you can start using the platform immediately after installation.

![Mastering Log Records (Part 2): Formatting Logs](https://c.mql5.com/2/108/logify60x60.png)[Mastering Log Records (Part 2): Formatting Logs](https://www.mql5.com/en/articles/16833)

In this article, we will explore how to create and apply log formatters in the library. We will see everything from the basic structure of a formatter to practical implementation examples. By the end, you will have the necessary knowledge to format logs within the library, and understand how everything works behind the scenes.

![Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://c.mql5.com/2/82/Neural_networks_are_simple_Piecewise_linear_representation_of_time_series__LOGO.png)[Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://www.mql5.com/en/articles/15217)

This article is somewhat different from my earlier publications. In this article, we will talk about an alternative representation of time series. Piecewise linear representation of time series is a method of approximating a time series using linear functions over small intervals.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lgzodbzoauhxztqcwicuethvwbagvxcw&ssn=1769182987746873873&ssn_dr=0&ssn_sr=0&fv_date=1769182987&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16742&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%206)%3A%20Expanding%20the%20History%20Management%20EX5%20Library%20with%20the%20Last%20Filled%20Pending%20Order%20Functions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918298759893823&fz_uniq=5069684486360205578&sv=2552)

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