---
title: MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions
url: https://www.mql5.com/en/articles/16906
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:42:37.554894
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=tmcnjtpgngasfbjqpzntougueqxjodqt&ssn=1769182956585208210&ssn_dr=0&ssn_sr=0&fv_date=1769182956&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16906&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Trading%20Toolkit%20(Part%207)%3A%20Expanding%20the%20History%20Management%20EX5%20Library%20with%20the%20Last%20Canceled%20Pending%20Order%20Functions%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918295611719072&fz_uniq=5069676940102666475&sv=2552)

MetaTrader 5 / Examples


### Introduction

Throughout this article series, we have dedicated significant effort to building a comprehensive suite of trading EX5 libraries. These libraries are designed to streamline the development process for MQL5 applications, drastically reducing the time and effort required to handle trade operations and manage the historical data of orders, deals, and positions. By offering well-structured and intuitive functions, these libraries simplify complex tasks, allowing developers to seamlessly process trading history with precision and efficiency.

In this article, we will finalize the development of the last module in our _History Management EX5_ library, specifically designed to handle the retrieval and storage of properties associated with the most recently canceled pending order. This module addresses a key limitation in the MQL5 language, which lacks straightforward, single-line functions to access and manage such historical data. By bridging this gap, our library offers developers a streamlined solution to efficiently work with canceled pending order information.

The focus of this module is to provide a simple yet effective way to retrieve and store critical details of canceled pending orders, such as their symbol, opening price, pip-based stop loss, take profit, time-based durations, and other relevant attributes. By encapsulating this functionality into easy-to-use functions, the library allows developers to access the required data with minimal effort and complexity. This makes it an indispensable tool for anyone looking to build MQL5 applications that rely on precise and accessible trading history data. For those of you who are interested in analyzing trading performance, this module simplifies the process, enabling you to focus on the bigger picture.

To get started, open the _**HistoryManager.mq5**_ file from the [previous article](https://www.mql5.com/en/articles/16742), where we developed functions for handling the most recently filled pending order. In this section, we will begin implementing the **_GetLastCanceledPendingOrderData()_** function, which is central to processing canceled pending orders. Before proceeding, ensure you have downloaded the **_HistoryManager.mq5_** source file, which is attached at the end of the [previous article](https://www.mql5.com/en/articles/16742).

Once the file is ready, locate the section where we concluded in the last article. Specifically, we will continue adding new code just below the **_LastFilledPendingOrderComment()_** function. This placement ensures that the new functionality is logically organized alongside related functions, making the library easier to navigate and extend in the future if the need arises.

### Main Content

01. [Get The Last Canceled Pending Order Data](https://www.mql5.com/en/articles/16906/#para2)
02. [Get The Last Canceled Pending Order Type](https://www.mql5.com/en/articles/16906/#para3)
03. [Get The Last Canceled Pending Order Symbol](https://www.mql5.com/en/articles/16906/#para4)
04. [Get The Last Canceled Pending Order Ticket](https://www.mql5.com/en/articles/16906/#para5)
05. [Get The Last Canceled Pending Order Price Open](https://www.mql5.com/en/articles/16906/#para6)
06. [Get The Last Canceled Pending Order Stop Loss Price](https://www.mql5.com/en/articles/16906/#para7)
07. [Get The Last Canceled Pending Order Take Profit Price](https://www.mql5.com/en/articles/16906/#para8)
08. [Get The Last Canceled Pending Order Stop Loss Pips](https://www.mql5.com/en/articles/16906/#para9)
09. [Get The Last Canceled Pending Order Take Profit Pips](https://www.mql5.com/en/articles/16906/#para10)
10. [Get The Last Canceled Pending Order Time Setup](https://www.mql5.com/en/articles/16906/#para11)
11. [Get The Last Canceled Pending Order Time Done](https://www.mql5.com/en/articles/16906/#para12)
12. [Get The Last Canceled Pending Order Expiration Time](https://www.mql5.com/en/articles/16906/#para13)
13. [Get The Last Canceled Pending Order Position ID](https://www.mql5.com/en/articles/16906/#para14)
14. [Get The Last Canceled Pending Order Magic](https://www.mql5.com/en/articles/16906/#para15)
15. [Get The Last Canceled Pending Order Reason](https://www.mql5.com/en/articles/16906/#para16)
16. [Get The Last Canceled Pending Order Type Filling](https://www.mql5.com/en/articles/16906/#para17)
17. [Get The Last Canceled Pending Order Type Time](https://www.mql5.com/en/articles/16906/#para18)
18. [Get The Last Canceled Pending Order Comment](https://www.mql5.com/en/articles/16906/#para19)
19. [Conclusion](https://www.mql5.com/en/articles/16906/#para20)

### Get Last Canceled Pending Order Data Function

The _GetLastCanceledPendingOrderData()_ function retrieves the details of the most recently canceled pending order and saves this information in the specified _getLastCanceledPendingOrderData_ reference. It relies on the _FetchHistoryByCriteria()_ function to access historical trading data, identifying the last canceled pending order by analyzing the relevant dataset. If no such data is found, the function logs an error message and returns _false_. When successful, it stores the retrieved information in the provided reference and returns _true_.

Let us begin by creating the function definition or signature. The _GetLastCanceledPendingOrderData()_ function should be accessible to importing MQL5 programs for external use, which is why we have defined it as _exported_.

```
bool GetLastCanceledPendingOrderData(PendingOrderData &getLastCanceledPendingOrderData) export
  {
//-- Function implementation will be explained step by step below.
  }
```

We will go ahead and attempt to fetch the history of pending orders using the _FetchHistoryByCriteria()_ function with the _GET\_PENDING\_ORDERS\_HISTORY\_DATA_ argument. This ensures that the function has access to the required data for canceled pending orders.

If the _FetchHistoryByCriteria()_ function returns _false_, it indicates that there is no trading history available. In this case, we log an error message using the _Print()_ function to aid in debugging. The function then returns _false_, signifying the failure to retrieve the data.

```
if(!FetchHistoryByCriteria(GET_PENDING_ORDERS_HISTORY_DATA))
  {
   Print(__FUNCTION__, ": No trading history available. Last canceled pending order can't be retrieved.");
   return(false);
  }
```

Once the historical data is available, we will calculate the total number of pending orders using the _GetTotalDataInfoSize()_ function. This value will help us loop through the _pendingOrderInfo_ array to locate the most recent canceled pending order.

Next, we will loop through the _pendingOrderInfo_ array to find an order with a state of _ORDER\_STATE\_CANCELED_. The function will save the first matching order to the _getLastCanceledPendingOrderData_ variable. The loop ensures that we check each order's state, and once a canceled order is found, it is saved in the reference variable, after which the loop exits. After storing the data for the last canceled pending order, the function returns _true_, indicating that the operation was successful.

```
int totalPendingOrderInfo = GetTotalDataInfoSize(GET_PENDING_ORDERS_HISTORY_DATA);
for(int x = 0; x < totalPendingOrderInfo; x++)
  {
   if(pendingOrderInfo[x].state == ORDER_STATE_CANCELED)
     {
      getLastCanceledPendingOrderData = pendingOrderInfo[x];
      break;
     }
  }
return(true);
```

If no historical data is available, an error message is logged using the _Print()_ function, including the function name ( _\_\_FUNCTION\_\__) for clarity. This message helps identify the issue during debugging. The function ensures that the provided reference variable remains unchanged if the operation fails, maintaining the integrity of the input.

Here is the complete implementation of the _GetLastCanceledPendingOrderData()_ function.

```
bool GetLastCanceledPendingOrderData(PendingOrderData &getLastCanceledPendingOrderData) export
  {
   if(!FetchHistoryByCriteria(GET_PENDING_ORDERS_HISTORY_DATA))
     {
      Print(__FUNCTION__, ": No trading history available. Last canceled pending order can't be retrieved.");
      return(false);
     }

//-- Save the last canceled pending order data in the referenced getLastCanceledPendingOrderData variable
   int totalPendingOrderInfo = GetTotalDataInfoSize(GET_PENDING_ORDERS_HISTORY_DATA);
   for(int x = 0; x < totalPendingOrderInfo; x++)
     {
      if(pendingOrderInfo[x].state == ORDER_STATE_CANCELED)
        {
         getLastCanceledPendingOrderData = pendingOrderInfo[x];
         break;
        }
     }
   return(true);
  }
```

### Last Canceled Pending Order Type Function

The _LastCanceledPendingOrderType()_ function is responsible for retrieving the _order type_ of the most recently canceled pending order and saving it in the referenced variable, _lastCanceledPendingOrderType_. This variable is passed to the function as input, where it stores the retrieved order type.

To accomplish this, a temporary _PendingOrderData_ variable, _lastCanceledPendingOrderInfo_, is declared to hold the details of the last canceled pending order. The function then calls _GetLastCanceledPendingOrderData()_ to obtain the required order information.

If the retrieval is successful, the _type_ field from _lastCanceledPendingOrderInfo_ is extracted and assigned to _lastCanceledPendingOrderType_. The function then returns _true_ to confirm the successful operation. Conversely, if the retrieval fails, the function returns _false_, leaving the _lastCanceledPendingOrderType_ variable unchanged.

Below is the full implementation of the _LastCanceledPendingOrderType()_ function.

```
bool LastCanceledPendingOrderType(ENUM_ORDER_TYPE &lastCanceledPendingOrderType) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderType = lastCanceledPendingOrderInfo.type;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Symbol Function

The _LastCanceledPendingOrderSymbol()_ function is designed to fetch the trading _symbol_ linked to the most recently canceled pending order. This _symbol_ is stored in the provided _lastCanceledPendingOrderSymbol_ variable, offering a straightforward way to access this specific property. The function relies on the _GetLastCanceledPendingOrderData()_ utility to obtain the necessary order details.

The process begins by invoking _GetLastCanceledPendingOrderData()_, which retrieves the details of the last canceled pending order. If the retrieval is successful, the symbol field from the fetched data is assigned to the referenced _lastCanceledPendingOrderSymbol_ variable, and the function returns _true_ to confirm the operation's success.

In cases where the data retrieval fails—such as when there is no relevant order history—the function leaves the referenced variable unchanged and returns _false_ to indicate the failure.

Below is the full implementation of the _LastCanceledPendingOrderSymbol()_ function.

```
bool LastCanceledPendingOrderSymbol(string &lastCanceledPendingOrderSymbol) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderSymbol = lastCanceledPendingOrderInfo.symbol;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Ticket Function

The _LastCanceledPendingOrderTicket()_ function retrieves the _ticket_ number of the most recently canceled pending order and stores it in the referenced _lastCanceledPendingOrderTicket_ variable. It calls the _GetLastCanceledPendingOrderData()_ function to fetch the order details.

If the data retrieval is successful, the _ticket_ number is stored in the referenced variable, and the function returns _true_. If the process fails, the function returns _false_, leaving the variable unchanged.

Below is the complete code for the _LastCanceledPendingOrderTicket()_ function.

```
bool LastCanceledPendingOrderTicket(ulong &lastCanceledPendingOrderTicket) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTicket = lastCanceledPendingOrderInfo.ticket;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Price Open Function

The _LastCanceledPendingOrderPriceOpen()_ function fetches the _opening price_ of the most recently canceled pending order and holds it in the referenced _lastCanceledPendingOrderPriceOpen_ variable. It calls _GetLastCanceledPendingOrderData()_ to gather the order details.

Upon successful retrieval, the _opening price_ is saved in the provided variable, and the function returns _true_. If the retrieval fails, the function returns _false_ without altering the variable.

Here is the full implementation of the _LastCanceledPendingOrderPriceOpen()_ function.

```
bool LastCanceledPendingOrderPriceOpen(double &lastCanceledPendingOrderPriceOpen) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderPriceOpen = lastCanceledPendingOrderInfo.priceOpen;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Stop Loss Price Function

The _LastCanceledPendingOrderSlPrice()_ function gets the _stop loss price_ of the most recently canceled pending order and saves it in the referenced _lastCanceledPendingOrderSlPrice_ variable. It uses the _GetLastCanceledPendingOrderData()_ function to fetch the order details.

If the retrieval is successful, the _stop loss price_ is stored in the referenced variable, and the function returns _true_. If the process fails, the function returns _false_ without altering the variable.

Here’s how the _LastCanceledPendingOrderSlPrice()_ function is fully implemented.

```
bool LastCanceledPendingOrderSlPrice(double &lastCanceledPendingOrderSlPrice) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderSlPrice = lastCanceledPendingOrderInfo.slPrice;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Take Profit Price Function

The _LastCanceledPendingOrderTpPrice()_ function retrieves the _take profit price_ of the most recently canceled pending order and holds it in the referenced _lastCanceledPendingOrderTpPrice_ variable. It utilizes the _GetLastCanceledPendingOrderData()_ function to fetch the order details.

If the data retrieval is successful, the _take profit price_ is saved in the referenced variable, and the function returns _true_. If the retrieval fails, the function returns _false_ without modifying the variable.

Below is the complete implementation of the _LastCanceledPendingOrderTpPrice()_ function.

```
bool LastCanceledPendingOrderTpPrice(double &lastCanceledPendingOrderTpPrice) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTpPrice = lastCanceledPendingOrderInfo.tpPrice;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Stop Loss Pips Function

The _LastCanceledPendingOrderSlPips()_ function retrieves the _stop loss pips_ value of the most recently canceled pending order and stores it in the referenced _lastCanceledPendingOrderSlPips_ variable. By utilizing the _GetLastCanceledPendingOrderData()_ function, it accesses the relevant order details to extract this specific value.

The function begins by invoking _GetLastCanceledPendingOrderData()_ to obtain the data of the last canceled pending order. If the retrieval is successful, the _stop loss pips_ value is extracted from the fetched data and assigned to the _lastCanceledPendingOrderSlPips_ variable. The function then indicates success by returning _true_.

On the other hand, if the function fails to retrieve the required data—perhaps due to a lack of historical information—it refrains from altering the _lastCanceledPendingOrderSlPips_ variable and instead returns _false_.

Here is the full implementation of the _LastCanceledPendingOrderSlPips()_ function.

```
bool LastCanceledPendingOrderSlPips(int &lastCanceledPendingOrderSlPips) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderSlPips = lastCanceledPendingOrderInfo.slPips;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Take Profit Pips Function

The _LastCanceledPendingOrderTpPips()_ function retrieves the take _profit pips_ value associated with the most recently canceled pending order. The retrieved value is saved in the referenced variable, _lastCanceledPendingOrderTpPips_. To achieve this, the function relies on the _GetLastCanceledPendingOrderData()_ function, which fetches the necessary order details.

Initially, a _PendingOrderData_ variable is declared to temporarily store the details of the last canceled pending order. The _GetLastCanceledPendingOrderData()_ function is then called to populate this variable with relevant data. If the operation succeeds, the _take profit pips_ value is extracted from the variable and stored in the _lastCanceledPendingOrderTpPips_ variable. The function then returns _true_, indicating success.

However, if the data retrieval fails, the function returns _false_, leaving the referenced variable unchanged.

Here is the complete implementation of the _LastCanceledPendingOrderTpPips()_ function.

```
bool LastCanceledPendingOrderTpPips(int &lastCanceledPendingOrderTpPips) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTpPips = lastCanceledPendingOrderInfo.tpPips;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Time Setup Function

The _LastCanceledPendingOrderTimeSetup()_ function retrieves the _time setup_ of the most recently canceled pending order and stores it in the referenced _lastCanceledPendingOrderTimeSetup_ variable. To perform this operation, the function uses the _GetLastCanceledPendingOrderData()_ function to access the necessary details about the canceled order.

We begin by declaring a _PendingOrderData_ variable to temporarily hold the data for the last canceled pending order. The _GetLastCanceledPendingOrderData()_ function is then invoked to populate this variable with the relevant order details. If the retrieval is successful, the _time setup_ value is extracted and stored in the _lastCanceledPendingOrderTimeSetup_ variable. The function subsequently returns _true_, confirming the success of the operation.

On the other hand, if the data retrieval fails, the function returns _false_, ensuring that the _lastCanceledPendingOrderTimeSetup_ variable remains unchanged.

Below is the full implementation of the _LastCanceledPendingOrderTimeSetup()_ function.

```
bool LastCanceledPendingOrderTimeSetup(datetime &lastCanceledPendingOrderTimeSetup) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTimeSetup = lastCanceledPendingOrderInfo.timeSetup;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Time Done Function

The _LastCanceledPendingOrderTimeDone()_ function is designed to retrieve the _time_ when the most recently canceled pending order was executed. This _time_ is stored in the referenced _lastCanceledPendingOrderTimeDone_ variable, providing a straightforward way to access this specific piece of data. The function relies on the _GetLastCanceledPendingOrderData()_ utility function to gather the required details of the canceled order.

The function begins by calling _GetLastCanceledPendingOrderData()_, which extracts the details of the most recently canceled pending order. If the retrieval operation is successful, the _time_ field from the fetched data is assigned to the _lastCanceledPendingOrderTimeDone_ variable, and the function returns _true_, confirming the successful operation.

However, if the retrieval process fails—such as in the absence of relevant order history—the function does not alter the referenced variable and instead returns _false_, signaling the failure to fetch the desired information.

For tasks requiring the calculation of the _total time_ a pending order was _open_ before being _canceled_, this function can be used alongside the _LastCanceledPendingOrderTimeSetup()_ function. Together, they enable you to determine the _duration_ between the order's _creation_ and its _cancellation_.

Below is the complete implementation of the _LastCanceledPendingOrderTimeDone()_ function.

```
bool LastCanceledPendingOrderTimeDone(datetime &lastCanceledPendingOrderTimeDone) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTimeDone = lastCanceledPendingOrderInfo.timeDone;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Expiration Time Function

The _LastCanceledPendingOrderExpirationTime()_ function fetches the _expiration time_ of the most recently canceled pending order and assigns it to the referenced _lastCanceledPendingOrderExpirationTime_ variable. It calls the _GetLastCanceledPendingOrderData()_ function to obtain the order details.

If the data retrieval is successful, the _expiration time_ is saved into the referenced variable and the function returns _true_. If the retrieval fails, the function returns _false_ without altering the variable.

Below is the complete implementation of the _LastCanceledPendingOrderExpirationTime()_ function.

```
bool LastCanceledPendingOrderExpirationTime(datetime &lastCanceledPendingOrderExpirationTime) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderExpirationTime = lastCanceledPendingOrderInfo.expirationTime;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Position ID Function

The _LastCanceledPendingOrderPositionId()_ function extracts the _position ID_ of the most recently canceled pending order and updates the referenced _lastCanceledPendingOrderPositionId_ variable with this value. To access the necessary data, it invokes the _GetLastCanceledPendingOrderData()_ function.

If the operation is successful, the _position ID_ is placed into the referenced variable, and the function returns _true_. In case the retrieval is unsuccessful, the function returns _false_, leaving the variable unchanged.

Here is the full implementation of the _LastCanceledPendingOrderPositionId()_ function.

```
bool LastCanceledPendingOrderPositionId(ulong &lastCanceledPendingOrderPositionId) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderPositionId = lastCanceledPendingOrderInfo.positionId;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Magic Function

The _LastCanceledPendingOrderMagic()_ function obtains the _magic number_ associated with the most recently canceled pending order and assigns it to the referenced _lastCanceledPendingOrderMagic_ variable. This process relies on the _GetLastCanceledPendingOrderData()_ function to fetch the required details.

If the data retrieval succeeds, the _magic number_ is transferred to the referenced variable, and the function returns _true_. Should the operation fail, the function returns _false_, leaving the variable unmodified.

Here is the complete implementation of the _LastCanceledPendingOrderMagic()_ function.

```
bool LastCanceledPendingOrderMagic(ulong &lastCanceledPendingOrderMagic) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderMagic = lastCanceledPendingOrderInfo.magic;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Reason Function

The _LastCanceledPendingOrderReason()_ function extracts the _reason code_ for the most recently canceled pending order and stores it in the referenced _lastCanceledPendingOrderReason_ variable. It uses the _GetLastCanceledPendingOrderData()_ utility function to retrieve the order details.

The _reason code_ indicates how the order was placed or why it was triggered. For instance, _ORDER\_REASON\_CLIENT_ shows that the order was placed manually from a desktop terminal, while _ORDER\_REASON\_EXPERT_ indicates the order was placed by an Expert Advisor, and _ORDER\_REASON\_WEB_ indicates that the order was placed from a web platform. Other possible reasons include activation due to _Stop Loss_ or T _ake Profit_, or as a result of a _Stop Out_ event.

If the data retrieval is successful, the function stores the _reason code_ in the provided variable and returns _true_. If it fails, the function returns _false_ without modifying the variable.

Below is the full implementation of the _LastCanceledPendingOrderReason()_ function.

```
bool LastCanceledPendingOrderReason(ENUM_ORDER_REASON &lastCanceledPendingOrderReason) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderReason = lastCanceledPendingOrderInfo.reason;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Type Filling Function

The _LastCanceledPendingOrderTypeFilling()_ function determines the _filling type_ of the most recently canceled pending order and assigns it to the referenced _lastCanceledPendingOrderTypeFilling_ variable. It achieves this by calling the _GetLastCanceledPendingOrderData()_ function to gather the necessary order details.

The _filling type_ provides essential information about how the order was intended to be executed. If the type is " _Fill or Kill (FOK)_," the order must be fully _filled_ at the requested _price_, and if this is not possible, it is _canceled_. For " _Immediate or Cancel (IOC)_," the order is executed immediately for the available _volume_ at the requested _price_, and any unfilled portion is discarded. The " _Return_" type allows the order to be partially _filled_ if the full _volume_ is unavailable, with the remaining unfilled _volume_ converted into a _limit order_. This limit order remains active in the market until it is either _filled_ or _canceled_ manually or by an Expert Advisor.

If the data retrieval operation is successful, the _filling type_ is stored in the provided variable, and the function returns _true_. If the retrieval fails, the function returns _false_ without modifying the referenced variable.

Below is the full implementation of the _LastCanceledPendingOrderTypeFilling()_ function.

```
bool LastCanceledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastCanceledPendingOrderTypeFilling) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTypeFilling = lastCanceledPendingOrderInfo.typeFilling;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Type Time Function

The _LastCanceledPendingOrderTypeTime()_ function extracts the _type time_ of the most recently canceled pending order and updates the referenced _lastCanceledPendingOrderTypeTime_ variable with this value. It uses the _GetLastCanceledPendingOrderData()_ function to retrieve the necessary order details.

The _type time_ of an order indicates how long the order remains valid. There are several types of time associated with orders: _Good Till Cancel (GTC)_, where the order remains active until it is manually _canceled_; _Good Till Current Trade Day_, where the order _expires_ at the end of the trading day; _Good Till Expired_, where the order _expires_ after a specific date or time; and _Good Till Specified Day_, where the order is valid until _23:59:59_ of the specified day, with the added condition that if the time falls outside a trading session, the order _expires_ at the nearest trading time.

If the retrieval is successful, the function assigns the _type time_ to the referenced variable and returns _true_. If the retrieval fails, it leaves the variable unchanged and returns _false_.

Here is the complete implementation of the _LastCanceledPendingOrderTypeTime()_ function.

```
bool LastCanceledPendingOrderTypeTime(datetime &lastCanceledPendingOrderTypeTime) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderTypeTime = lastCanceledPendingOrderInfo.typeTime;
      return(true);
     }
   return(false);
  }
```

### Last Canceled Pending Order Comment Function

The _LastCanceledPendingOrderComment()_ function acquires the _comment_ associated with the most recently canceled pending order and places it in the referenced _lastCanceledPendingOrderComment_ variable. It utilizes the _GetLastCanceledPendingOrderData()_ function to fetch the required order information.

If the order data is successfully obtained, the _comment_ is assigned to the referenced variable, and the function returns _true_. If the operation fails, the referenced variable remains unmodified, and the function returns _false_.

Below is the full implementation of the _LastCanceledPendingOrderComment()_ function.

```
bool LastCanceledPendingOrderComment(string &lastCanceledPendingOrderComment) export
  {
   PendingOrderData lastCanceledPendingOrderInfo;
   if(GetLastCanceledPendingOrderData(lastCanceledPendingOrderInfo))
     {
      lastCanceledPendingOrderComment = lastCanceledPendingOrderInfo.comment;
      return(true);
     }
   return(false);
  }
```

### Conclusion

We have now developed a comprehensive history management library that is capable of querying, retrieving, categorizing, and storing the trading history of filled and canceled pending orders, as well as deals and positions. This marks a significant milestone in simplifying the management of historical trading data in MQL5, equipping developers with a versatile and efficient tool set to handle complex data requirements.

What sets this library apart is its robust and organized framework for managing trading data. By providing structured and intuitive functions, it transforms the often cumbersome task of handling trading history into a seamless process. This approach enhances accessibility to critical data while also ensures it can be effectively applied in various real-world scenarios, such as creating performance analysis tools, optimizing trading strategies, or conducting in-depth historical reviews.

The subsequent step in this series will involve creating the necessary header files that will allow end users to seamlessly import and integrate the library into their projects. Once the headers are complete, I will demonstrate how to implement the library, ensuring developers can easily incorporate its functionality into their codebase.

For your convenience, the updated _**HistoryManager.mq5**_ library source code, which includes all the functions created in this and previous articles, plus the compiled EX5 executable binary file **_HistoryManager.ex5_** are attached at the end of this article. In the next article, we will wrap up the history management library by providing complete implementation documentation and practical examples to guide you in using this EX5 library effectively. Thank you for following along, I look forward to connecting with you again as we bring this project to its final stage.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16906.zip "Download all attachments in the single ZIP archive")

[HistoryManager.mq5](https://www.mql5.com/en/articles/download/16906/historymanager.mq5 "Download HistoryManager.mq5")(81.42 KB)

[HistoryManager.ex5](https://www.mql5.com/en/articles/download/16906/historymanager.ex5 "Download HistoryManager.ex5")(33.63 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase](https://www.mql5.com/en/articles/17015)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**[Go to discussion](https://www.mql5.com/en/forum/480185)**

![Master MQL5 from Beginner to Pro (Part III): Complex Data Types and Include Files](https://c.mql5.com/2/84/Learning_MQL5_-_from_beginner_to_pro_Part_III___LOGO.png)[Master MQL5 from Beginner to Pro (Part III): Complex Data Types and Include Files](https://www.mql5.com/en/articles/14354)

This is the third article in a series describing the main aspects of MQL5 programming. This article covers complex data types that were not discussed in the previous article. These include structures, unions, classes, and the 'function' data type. It also explains how to add modularity to your program using the #include preprocessor directive.

![Price Action Analysis Toolkit Development (Part 8): Metrics Board](https://c.mql5.com/2/112/Price_Action_Analysis_Toolkit_Development_Part_8___LOGO2.png)[Price Action Analysis Toolkit Development (Part 8): Metrics Board](https://www.mql5.com/en/articles/16584)

As one of the most powerful Price Action analysis toolkits, the Metrics Board is designed to streamline market analysis by instantly providing essential market metrics with just a click of a button. Each button serves a specific function, whether it’s analyzing high/low trends, volume, or other key indicators. This tool delivers accurate, real-time data when you need it most. Let’s dive deeper into its features in this article.

![Monitoring trading with push notifications — example of a MetaTrader 5 service](https://c.mql5.com/2/85/Monitoring_Trade_Using_Push_Notifications___LOGO.png)[Monitoring trading with push notifications — example of a MetaTrader 5 service](https://www.mql5.com/en/articles/15346)

In this article, we will look at creating a service app for sending notifications to a smartphone about trading results. We will learn how to handle lists of Standard Library objects to organize a selection of objects by required properties.

![Developing a Calendar-Based News Event Breakout Expert Advisor in MQL5](https://c.mql5.com/2/107/News_logo.png)[Developing a Calendar-Based News Event Breakout Expert Advisor in MQL5](https://www.mql5.com/en/articles/16752)

Volatility tends to peak around high-impact news events, creating significant breakout opportunities. In this article, we will outline the implementation process of a calendar-based breakout strategy. We'll cover everything from creating a class to interpret and store calendar data, developing realistic backtests using this data, and finally, implementing execution code for live trading.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/16906&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069676940102666475)

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