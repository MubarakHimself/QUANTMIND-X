---
title: MQL5 Trading Toolkit (Part 8): How to Implement and Use the History Manager EX5 Library in Your Codebase
url: https://www.mql5.com/en/articles/17015
categories: Trading, Trading Systems
relevance_score: 6
scraped_at: 2026-01-22T17:57:58.378529
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/17015&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049520199020948723)

MetaTrader 5 / Trading


### Introduction

In previous articles, I walked you through developing a robust and comprehensive History Management EX5 library designed to streamline how you interact with trade histories in MetaTrader 5. This powerful library enables you to effortlessly scan, retrieve, sort, categorize, and filter all types of trade histories—whether it’s closed positions, pending orders, or deal histories—directly from your MetaTrader 5 account. Building on that foundation, this article takes the final step by showing you how to efficiently integrate the History Manager library into your MQL5 projects.

I will provide a detailed breakdown of the functions available in the EX5 library, complete with clear explanations and practical examples. These real-world use cases will not only help you understand how each function works but also demonstrate how to apply them in your trading strategies or analysis tools. Whether you’re a beginner looking to enhance your coding skills or an experienced developer seeking to optimize your workflow, this guide will equip you with the knowledge and tools to harness the full potential of the History Manager library. By the end of this article, you’ll be able to implement this versatile library with confidence and start leveraging its capabilities to manage trade histories more efficiently than ever before.

**Benefits of Using the History Manager in MQL5**

Below are the key advantages of integrating the History Manager into your MQL5 projects:

- _Effortless Data Retrieval:_ The library provides intuitive functions to retrieve trade history data, such as deals, orders, positions, and pending orders, with minimal code. This eliminates the need for complex and repetitive coding.
- _Unified Interface:_ All trade history data is accessible through a single, consistent interface, making it easier to manage and analyze different types of trade activities.
- _Time-Saving Functions:_ With one-line function calls, you can quickly access critical trade data, reducing development time and allowing you to focus on strategy implementation.
- _Automated Data Processing:_ The library automates the retrieval, sorting, and filtering of trade histories, streamlining workflows and improving overall efficiency.
- _Error-Free Data Handling:_ The library ensures accurate retrieval of trade history data, minimizing the risk of errors that can occur with manual coding.
- _Comprehensive Data Coverage:_ From closed positions to pending orders, the library provides access to all types of trade histories, ensuring no data is overlooked.
- _Detailed Insights:_ Functions like _GetLastClosedPositionData(), GetLastFilledPendingOrderData(),_ and _GetLastClosedProfitablePositionData()_ among many others, provide granular details about trades, enabling more in-depth analysis and better decision-making.
- _Customizable Filters:_ You can filter trade histories by _symbol, magic number,_ or _time range_, allowing for targeted analysis tailored to your specific needs.
- _Easy Implementation:_ The library is designed to integrate seamlessly into existing MQL5 projects, requiring minimal setup and configuration.
- _Scalability:_ Whether you’re working on a small script or a large-scale trading system, the History Manager scales effortlessly to meet your requirements.
- _Adaptable to Your Needs:_ The library’s modular design allows you to use only the functions you need, making it highly adaptable to various trading styles and strategies.
- _Extendable Functionality:_ Developers can build on the library’s existing functions to create custom solutions for unique trading requirements.
- _Pre-Built Solutions:_ By leveraging the library’s pre-built functions, you can significantly reduce the time and cost associated with developing custom trade history management tools.
- _Open to All Skill Levels:_ Whether you’re a beginner or an experienced developer, the library’s straightforward design makes it accessible to users of all skill levels.
- _Data-Driven Decisions:_ Access to comprehensive trade histories enables you to make informed decisions when developing or refining trading strategies.
- _Historical Back testing:_ Use the library to retrieve historical trade data for back testing, ensuring your strategies are robust and reliable.
- _Clean and Readable Code:_ The library’s functions are designed to produce clean, readable code, making it easier to maintain and update your projects.

### Importing and Setting Up the History Manager EX5 Library

Before you can start using History Manager EX5 Library's features, you need to properly import and set up the library in your MQL5 project. This section will guide you through the process step by step, ensuring a smooth and hassle-free setup.

To use the _HistoryManager.ex5_ library file in your MQL5 projects, it needs to be downloaded and placed in the correct directory within your MetaTrader 5 installation folder. Launch MetaEditor from your MetaTrader 5 terminal by navigating to _Tools > MetaQuotes Language Editor_ or pressing _F4_ and follow the steps below:

- **Step 1.** Create the Necessary Directories:

Inside the Libraries' folder located in the root MQL5 directory, create the following subdirectories if they do not already exist: " _\\Wanateki\\Toolkit\\HistoryManager_".

The full path in your MQL5 installation folder should look like this: " _MQL5\\Libraries\\Wanateki\\Toolkit\\HistoryManager_".

- **Step 2.** Download the HistoryManager.ex5 Binary File:

Scroll to the bottom or end of this article, locate the attached _HistoryManager.ex5_ file, and download it. Place the file into the _HistoryManager_ folder you created in _Step 1_.

Your _Libraries'_ folder directory structure should now resemble the image below:

![Directory to save the HistoryManager.ex5 executable library file](https://c.mql5.com/2/116/Place_The_HistoryManager.ex5_In_The_Libraries_Folder.png)

- **Step 3.** Create the Header Library File:

1\. In _MetaEditor IDE_, launch _MQL Wizard_ using the _New_ menu item button.

![MQL5 Wizard New File](https://c.mql5.com/2/116/MQL5_Wizard_NewFile.png)

2. Select the _Include (\*.mqh)_ option and click _Next_.

![MQL5 Wizard New Include File Set Up](https://c.mql5.com/2/116/MQL5_Wizard_New_Include_File_Set_Up.png)

3\. In the _General properties of the include file_ window, locate the _Name_: input box. Clear all the text and enter the following path to specify the directory and name for the include file: _Include\\Wanateki\\Toolkit\\HistoryManager\\HistoryManager.mqh_

**_![MQL5 Wizard New Include File Directory Set Up](https://c.mql5.com/2/116/MQL5_Wizard_New_Include_File_Directory_Set_Up.png)_**

4\. Press the Finish button to generate the library header file.

### Coding The History Manager Header File (HistoryManager.mqh)

Now that we have successfully created a blank _include_ file for the _HistoryManager.mqh_ library header, the next step is to populate it with the necessary components that will define its functionality. This involves adding data structures to manage and store information efficiently, incorporating macro directives to simplify repetitive tasks and improve code readability, and importing, and declaring the library functions that will serve as the core interfaces for interacting with the HistoryManager.ex5 functionality.

Under the _#property link_, we will begin by declaring the _DealData, OrderData, PositionData,_ and _PendingOrderData_ data _structures_ in the global scope. These structures will be responsible for storing various trade history properties and will be accessible from any part of our codebase.

```
struct DealData
  {
  // Add all the DealData members here
  }

struct OrderData
  {
  // Add all the OrderData members here
  }

struct PositionData
  {
  // Add all the PositionData members here
  }

struct PendingOrderData
  {
  // Add all the PendingOrderData members here
  }
```

Next, we will _define_ several _constants_ or _macros_ to represent key time periods in _seconds_. These include _NOW, ONE\_DAY_, _ONE\_WEEK_, _ONE\_MONTH_, _ONE\_YEAR_, _EPOCH, TODAY, THIS\_WEEK, THIS\_MONTH,_ and _THIS\_YEAR_ which will simplify time-related calculations and improve code readability.

```
#define NOW datetime(TimeCurrent())
#define ONE_DAY datetime(TimeCurrent() - PeriodSeconds(PERIOD_D1))
#define ONE_WEEK datetime(TimeCurrent() - PeriodSeconds(PERIOD_W1))
#define ONE_MONTH datetime(TimeCurrent() - PeriodSeconds(PERIOD_MN1))
#define ONE_YEAR datetime(TimeCurrent() - (PeriodSeconds(PERIOD_MN1) * 12))
#define EPOCH 0 // 1st Jan 1970
//--
#define TODAY 12
#define THIS_WEEK 13
#define THIS_MONTH 14
#define THIS_YEAR 15
```

We will also create the _ALL\_SYMBOLS_ string macro to store an empty string. This will serve as a signal to indicate that we want to process the specified history for all _symbols_. Additionally, we will define the _ALL\_POSITIONS_, _ALL\_ORDERS_, and _ALL\_DEALS_ integer macros. These will allow us to specify the types of _orders_, _deals_, and _position_ history we wish to process.

```
#define ALL_SYMBOLS ""
#define ALL_POSITIONS 1110
#define ALL_ORDERS 1111
#define ALL_DEALS 1112
```

Now that we have created all the necessary _data structures_ and defined the essential _preprocessor directives_ or macros, it’s time to import the _HistoryManager.ex5_ library. To do this, we will use the _#import_ directive followed by the folder path where the EX5 library is located.

```
#import "Wanateki/Toolkit/HistoryManager/HistoryManager.ex5"
```

After the _#import_ directive, we will add the library function declarations and conclude by inserting the closing _#import_ directive as the final line of our _HistoryManager.mqh_ header file.

```
#import "Wanateki/Toolkit/HistoryManager/HistoryManager.ex5"
//--
void PrintDealsHistory(datetime fromDateTime, datetime toDateTime);
void PrintOrdersHistory(datetime fromDateTime, datetime toDateTime);
void PrintPositionsHistory(datetime fromDateTime, datetime toDateTime);
void PrintPendingOrdersHistory(datetime fromDateTime, datetime toDateTime);

//--
bool GetDealsData(DealData &dealsData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetOrdersData(OrderData &ordersData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetPositionsData(PositionData &positionsData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetPendingOrdersData(PendingOrderData &pendingOrdersData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
//--

// Add all the other function declarations here...

#import
```

Below are the essential code segments that form the _HistoryManager.mqh_ include file. This header file provides all the necessary resources for importing and integrating the _History Manager EX5_ library into your MQL5 projects. By _including_ it at the beginning of your source file, you gain access to all the data structures and library functions, making them readily available for immediate use. The complete _HistoryManager.mqh_ source file is also attached at the end of this article for you to download.

```
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
//--
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
//--
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
//--
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
//--
#define NOW datetime(TimeCurrent())
#define ONE_DAY datetime(TimeCurrent() - PeriodSeconds(PERIOD_D1))
#define ONE_WEEK datetime(TimeCurrent() - PeriodSeconds(PERIOD_W1))
#define ONE_MONTH datetime(TimeCurrent() - PeriodSeconds(PERIOD_MN1))
#define ONE_YEAR datetime(TimeCurrent() - (PeriodSeconds(PERIOD_MN1) * 12))
#define EPOCH 0 // 1st Jan 1970
//--
#define TODAY 12
#define THIS_WEEK 13
#define THIS_MONTH 14
#define THIS_YEAR 15
//--
#define ALL_SYMBOLS ""
#define ALL_POSITIONS 1110
#define ALL_ORDERS 1111
#define ALL_DEALS 1112
//--

#import "Wanateki/Toolkit/HistoryManager/HistoryManager.ex5"
//--
void PrintDealsHistory(datetime fromDateTime, datetime toDateTime);
void PrintOrdersHistory(datetime fromDateTime, datetime toDateTime);
void PrintPositionsHistory(datetime fromDateTime, datetime toDateTime);
void PrintPendingOrdersHistory(datetime fromDateTime, datetime toDateTime);

//--
bool GetDealsData(DealData &dealsData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetOrdersData(OrderData &ordersData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetPositionsData(PositionData &positionsData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
bool GetPendingOrdersData(PendingOrderData &pendingOrdersData[], datetime fromDateTime, datetime toDateTime, string symbol, ulong magic);
//--

bool GetDealsData(DealData &dealsData[], datetime fromDateTime, datetime toDateTime);
bool GetOrdersData(OrderData &ordersData[], datetime fromDateTime, datetime toDateTime);
bool GetPositionsData(PositionData &positionsData[], datetime fromDateTime, datetime toDateTime);
bool GetPendingOrdersData(PendingOrderData &pendingOrdersData[], datetime fromDateTime, datetime toDateTime);

//--
bool GetAllDealsData(DealData &dealsData[]);
bool GetAllOrdersData(OrderData &ordersData[]);
bool GetAllPositionsData(PositionData &positionsData[]);
bool GetAllPendingOrdersData(PendingOrderData &pendingOrdersData[]);
//--
bool GetAllDealsData(DealData &dealsData[], string symbol, ulong magic);
bool GetAllOrdersData(OrderData &ordersData[], string symbol, ulong magic);
bool GetAllPositionsData(PositionData &positionsData[], string symbol, ulong magic);
bool GetAllPendingOrdersData(PendingOrderData &pendingOrdersData[], string symbol, ulong magic);

//--
bool GetLastClosedPositionData(PositionData &lastClosedPositionInfo);
bool LastClosedPositionType(ENUM_POSITION_TYPE &lastClosedPositionType);
bool LastClosedPositionVolume(double &lastClosedPositionVolume);
bool LastClosedPositionSymbol(string &lastClosedPositionSymbol);
bool LastClosedPositionTicket(ulong &lastClosedPositionTicket);
bool LastClosedPositionProfit(double &lastClosedPositionProfit);
bool LastClosedPositionNetProfit(double &lastClosedPositionNetProfit);
bool LastClosedPositionPipProfit(int &lastClosedPositionPipProfit);
bool LastClosedPositionClosePrice(double &lastClosedPositionClosePrice);
bool LastClosedPositionOpenPrice(double &lastClosedPositionOpenPrice);
bool LastClosedPositionSlPrice(double &lastClosedPositionSlPrice);
bool LastClosedPositionTpPrice(double &lastClosedPositionTpPrice);
bool LastClosedPositionSlPips(int &lastClosedPositionSlPips);
bool LastClosedPositionTpPips(int &lastClosedPositionTpPips);
bool LastClosedPositionOpenTime(datetime &lastClosedPositionOpenTime);
bool LastClosedPositionCloseTime(datetime &lastClosedPositionCloseTime);
bool LastClosedPositionSwap(double &lastClosedPositionSwap);
bool LastClosedPositionCommission(double &lastClosedPositionCommission);
bool LastClosedPositionInitiatedByPendingOrder(bool &lastClosedPositionInitiatedByPendingOrder);
bool LastClosedPositionInitiatingOrderType(ENUM_ORDER_TYPE &lastClosedPositionInitiatingOrderType);
bool LastClosedPositionId(ulong &lastClosedPositionId);
bool LastClosedPositionOpeningOrderTicket(ulong &lastClosedPositionOpeningOrderTicket);
bool LastClosedPositionOpeningDealTicket(ulong &lastClosedPositionOpeningDealTicket);
bool LastClosedPositionClosingDealTicket(ulong &lastClosedPositionClosingDealTicket);
bool LastClosedPositionMagic(ulong &lastClosedPositionMagic);
bool LastClosedPositionComment(string &lastClosedPositionComment);
bool LastClosedPositionDuration(long &lastClosedPositionDuration);

//--
bool GetLastClosedProfitablePositionData(PositionData &lastClosedProfitablePositionInfo);
bool GetLastClosedLossPositionData(PositionData &lastClosedLossPositionData);

//--
bool GetLastFilledPendingOrderData(PendingOrderData &getLastFilledPendingOrderData);
bool LastFilledPendingOrderType(ENUM_ORDER_TYPE &lastFilledPendingOrderType);
bool LastFilledPendingOrderSymbol(string &lastFilledPendingOrderSymbol);
bool LastFilledPendingOrderTicket(ulong &lastFilledPendingOrderTicket);
bool LastFilledPendingOrderPriceOpen(double &lastFilledPendingOrderPriceOpen);
bool LastFilledPendingOrderSlPrice(double &lastFilledPendingOrderSlPrice);
bool LastFilledPendingOrderTpPrice(double &lastFilledPendingOrderTpPrice);
bool LastFilledPendingOrderSlPips(int &lastFilledPendingOrderSlPips);
bool LastFilledPendingOrderTpPips(int &lastFilledPendingOrderTpPips);
bool LastFilledPendingOrderTimeSetup(datetime &lastFilledPendingOrderTimeSetup);
bool LastFilledPendingOrderTimeDone(datetime &lastFilledPendingOrderTimeDone);
bool LastFilledPendingOrderExpirationTime(datetime &lastFilledPendingOrderExpirationTime);
bool LastFilledPendingOrderPositionId(ulong &lastFilledPendingOrderPositionId);
bool LastFilledPendingOrderMagic(ulong &lastFilledPendingOrderMagic);
bool LastFilledPendingOrderReason(ENUM_ORDER_REASON &lastFilledPendingOrderReason);
bool LastFilledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastFilledPendingOrderTypeFilling);
bool LastFilledPendingOrderTypeTime(datetime &lastFilledPendingOrderTypeTime);
bool LastFilledPendingOrderComment(string &lastFilledPendingOrderComment);

//--
bool GetLastCanceledPendingOrderData(PendingOrderData &getLastCanceledPendingOrderData);
bool LastCanceledPendingOrderType(ENUM_ORDER_TYPE &lastCanceledPendingOrderType);
bool LastCanceledPendingOrderSymbol(string &lastCanceledPendingOrderSymbol);
bool LastCanceledPendingOrderTicket(ulong &lastCanceledPendingOrderTicket);
bool LastCanceledPendingOrderPriceOpen(double &lastCanceledPendingOrderPriceOpen);
bool LastCanceledPendingOrderSlPrice(double &lastCanceledPendingOrderSlPrice);
bool LastCanceledPendingOrderTpPrice(double &lastCanceledPendingOrderTpPrice);
bool LastCanceledPendingOrderSlPips(int &lastCanceledPendingOrderSlPips);
bool LastCanceledPendingOrderTpPips(int &lastCanceledPendingOrderTpPips);
bool LastCanceledPendingOrderTimeSetup(datetime &lastCanceledPendingOrderTimeSetup);
bool LastCanceledPendingOrderTimeDone(datetime &lastCanceledPendingOrderTimeDone);
bool LastCanceledPendingOrderExpirationTime(datetime &lastCanceledPendingOrderExpirationTime);
bool LastCanceledPendingOrderPositionId(ulong &lastCanceledPendingOrderPositionId);
bool LastCanceledPendingOrderMagic(ulong &lastCanceledPendingOrderMagic);
bool LastCanceledPendingOrderReason(ENUM_ORDER_REASON &lastCanceledPendingOrderReason);
bool LastCanceledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastCanceledPendingOrderTypeFilling);
bool LastCanceledPendingOrderTypeTime(datetime &lastCanceledPendingOrderTypeTime);
bool LastCanceledPendingOrderComment(string &lastCanceledPendingOrderComment);

//*
//--
bool GetLastClosedPositionData(PositionData &lastClosedPositionInfo, string symbol, ulong magic);
bool LastClosedPositionType(ENUM_POSITION_TYPE &lastClosedPositionType, string symbol, ulong magic);
bool LastClosedPositionVolume(double &lastClosedPositionVolume, string symbol, ulong magic);
bool LastClosedPositionSymbol(string &lastClosedPositionSymbol, string symbol, ulong magic);
bool LastClosedPositionTicket(ulong &lastClosedPositionTicket, string symbol, ulong magic);
bool LastClosedPositionProfit(double &lastClosedPositionProfit, string symbol, ulong magic);
bool LastClosedPositionNetProfit(double &lastClosedPositionNetProfit, string symbol, ulong magic);
bool LastClosedPositionPipProfit(int &lastClosedPositionPipProfit, string symbol, ulong magic);
bool LastClosedPositionClosePrice(double &lastClosedPositionClosePrice, string symbol, ulong magic);
bool LastClosedPositionOpenPrice(double &lastClosedPositionOpenPrice, string symbol, ulong magic);
bool LastClosedPositionSlPrice(double &lastClosedPositionSlPrice, string symbol, ulong magic);
bool LastClosedPositionTpPrice(double &lastClosedPositionTpPrice, string symbol, ulong magic);
bool LastClosedPositionSlPips(int &lastClosedPositionSlPips, string symbol, ulong magic);
bool LastClosedPositionTpPips(int &lastClosedPositionTpPips, string symbol, ulong magic);
bool LastClosedPositionOpenTime(datetime &lastClosedPositionOpenTime, string symbol, ulong magic);
bool LastClosedPositionCloseTime(datetime &lastClosedPositionCloseTime, string symbol, ulong magic);
bool LastClosedPositionSwap(double &lastClosedPositionSwap, string symbol, ulong magic);
bool LastClosedPositionCommission(double &lastClosedPositionCommission, string symbol, ulong magic);
bool LastClosedPositionInitiatingOrderType(ENUM_ORDER_TYPE &lastClosedPositionInitiatingOrderType, string symbol, ulong magic);
bool LastClosedPositionId(ulong &lastClosedPositionId, string symbol, ulong magic);
bool LastClosedPositionInitiatedByPendingOrder(bool &lastClosedPositionInitiatedByPendingOrder, string symbol, ulong magic);
bool LastClosedPositionOpeningOrderTicket(ulong &lastClosedPositionOpeningOrderTicket, string symbol, ulong magic);
bool LastClosedPositionOpeningDealTicket(ulong &lastClosedPositionOpeningDealTicket, string symbol, ulong magic);
bool LastClosedPositionClosingDealTicket(ulong &lastClosedPositionClosingDealTicket, string symbol, ulong magic);
bool LastClosedPositionMagic(ulong &lastClosedPositionMagic, string symbol, ulong magic);
bool LastClosedPositionComment(string &lastClosedPositionComment, string symbol, ulong magic);
bool LastClosedPositionDuration(long &lastClosedPositionDuration, string symbol, ulong magic);

//--
bool GetLastClosedProfitablePositionData(PositionData &lastClosedProfitablePositionInfo, string symbol, ulong magic);
bool GetLastClosedLossPositionData(PositionData &lastClosedLossPositionData, string symbol, ulong magic);

//--
bool GetLastFilledPendingOrderData(PendingOrderData &lastFilledPendingOrderData, string symbol, ulong magic);
bool LastFilledPendingOrderType(ENUM_ORDER_TYPE &lastFilledPendingOrderType, string symbol, ulong magic);
bool LastFilledPendingOrderSymbol(string &lastFilledPendingOrderSymbol, string symbol, ulong magic);
bool LastFilledPendingOrderTicket(ulong &lastFilledPendingOrderTicket, string symbol, ulong magic);
bool LastFilledPendingOrderPriceOpen(double &lastFilledPendingOrderPriceOpen, string symbol, ulong magic);
bool LastFilledPendingOrderSlPrice(double &lastFilledPendingOrderSlPrice, string symbol, ulong magic);
bool LastFilledPendingOrderTpPrice(double &lastFilledPendingOrderTpPrice, string symbol, ulong magic);
bool LastFilledPendingOrderSlPips(int &lastFilledPendingOrderSlPips, string symbol, ulong magic);
bool LastFilledPendingOrderTpPips(int &lastFilledPendingOrderTpPips, string symbol, ulong magic);
bool LastFilledPendingOrderTimeSetup(datetime &lastFilledPendingOrderTimeSetup, string symbol, ulong magic);
bool LastFilledPendingOrderTimeDone(datetime &lastFilledPendingOrderTimeDone, string symbol, ulong magic);
bool LastFilledPendingOrderExpirationTime(datetime &lastFilledPendingOrderExpirationTime, string symbol, ulong magic);
bool LastFilledPendingOrderPositionId(ulong &lastFilledPendingOrderPositionId, string symbol, ulong magic);
bool LastFilledPendingOrderMagic(ulong &lastFilledPendingOrderMagic, string symbol, ulong magic);
bool LastFilledPendingOrderReason(ENUM_ORDER_REASON &lastFilledPendingOrderReason, string symbol, ulong magic);
bool LastFilledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastFilledPendingOrderTypeFilling, string symbol, ulong magic);
bool LastFilledPendingOrderTypeTime(datetime &lastFilledPendingOrderTypeTime, string symbol, ulong magic);
bool LastFilledPendingOrderComment(string &lastFilledPendingOrderComment, string symbol, ulong magic);

//--
bool GetLastCanceledPendingOrderData(PendingOrderData &lastCanceledPendingOrderData, string symbol, ulong magic);
bool LastCanceledPendingOrderType(ENUM_ORDER_TYPE &lastCanceledPendingOrderType, string symbol, ulong magic);
bool LastCanceledPendingOrderSymbol(string &lastCanceledPendingOrderSymbol, string symbol, ulong magic);
bool LastCanceledPendingOrderTicket(ulong &lastCanceledPendingOrderTicket, string symbol, ulong magic);
bool LastCanceledPendingOrderPriceOpen(double &lastCanceledPendingOrderPriceOpen, string symbol, ulong magic);
bool LastCanceledPendingOrderSlPrice(double &lastCanceledPendingOrderSlPrice, string symbol, ulong magic);
bool LastCanceledPendingOrderTpPrice(double &lastCanceledPendingOrderTpPrice, string symbol, ulong magic);
bool LastCanceledPendingOrderSlPips(int &lastCanceledPendingOrderSlPips, string symbol, ulong magic);
bool LastCanceledPendingOrderTpPips(int &lastCanceledPendingOrderTpPips, string symbol, ulong magic);
bool LastCanceledPendingOrderTimeSetup(datetime &lastCanceledPendingOrderTimeSetup, string symbol, ulong magic);
bool LastCanceledPendingOrderTimeDone(datetime &lastCanceledPendingOrderTimeDone, string symbol, ulong magic);
bool LastCanceledPendingOrderExpirationTime(datetime &lastCanceledPendingOrderExpirationTime, string symbol, ulong magic);
bool LastCanceledPendingOrderPositionId(ulong &lastCanceledPendingOrderPositionId, string symbol, ulong magic);
bool LastCanceledPendingOrderMagic(ulong &lastCanceledPendingOrderMagic, string symbol, ulong magic);
bool LastCanceledPendingOrderReason(ENUM_ORDER_REASON &lastCanceledPendingOrderReason, string symbol, ulong magic);
bool LastCanceledPendingOrderTypeFilling(ENUM_ORDER_TYPE_FILLING &lastCanceledPendingOrderTypeFilling, string symbol, ulong magic);
bool LastCanceledPendingOrderTypeTime(datetime &lastCanceledPendingOrderTypeTime, string symbol, ulong magic);
bool LastCanceledPendingOrderComment(string &lastCanceledPendingOrderComment, string symbol, ulong magic);

//--
string BoolToString(bool boolVariable);
datetime GetPeriodStart(int periodType);

#import
//+------------------------------------------------------------------+
```

### Practical Implementation and Overview of the History Manager EX5 Library

We will now examine the library functions, organizing them into their respective categories for better clarity and usability. Along the way, we will describe their roles and provide simple use cases to demonstrate how they can be used to accomplish various tasks.

**1\. Printing Trade Histories for a Specified Time Range**

These functions enable you to print trade history data directly to the MetaTrader 5 terminals' log, making it convenient for quick reference and debugging. They are of type _void_, meaning they do not _return_ any values, and accept two parameters of the _datetime_ type, which are used to specify the time range of the history to be processed:

| Function Prototype Definition | Description | Example Use Case |
| --- | --- | --- |
| ```<br>void PrintDealsHistory(<br>   datetime fromDateTime,<br>   datetime toDateTime<br>);<br>``` | Prints details of all deals within a specified time range. | ```<br>// Print the deals history for the last 24 hours (1 day)<br>PrintDealsHistory(ONE_DAY, NOW);<br>``` |
| ```<br>void PrintOrdersHistory(<br>   datetime fromDateTime,<br>   datetime toDateTime<br>);<br>``` | Prints details of all orders within a specified time range. | ```<br>// Print the orders history for the last 24 hours (1 day)<br>PrintOrdersHistory(ONE_DAY, NOW);<br>``` |
| ```<br>void PrintPositionsHistory(<br>   datetime fromDateTime,<br>   datetime toDateTime<br>);<br>``` | Prints details of all closed positions within a specified time range. | ```<br>// Print the positions history for the last 24 hours (1 day)<br>PrintPositionsHistory(ONE_DAY, NOW);<br>``` |
| ```<br>void PrintPendingOrdersHistory(<br>   datetime fromDateTime,<br>   datetime toDateTime<br>);<br>``` | Prints details of all pending orders within a specified time range. | ```<br>// Print the pending orders history for the last 24 hours (1 day)<br>PrintPendingOrdersHistory(ONE_DAY, NOW);<br>``` |

**2\. Retrieving Trade History Data for a Specified Time Range**

These functions allow you to programmatically retrieve trade history data, filtered by _symbol_ and _magic_ number, for a specified period. The retrieved data is stored in data structure arrays, enabling further analysis or processing:

| Function Prototype Definition | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool GetDealsData(<br>   DealData &dealsData[], // [out]<br>   datetime fromDateTime,<br>   datetime toDateTime,<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function searches for and retrieves deal data within a specified time range. Additionally, it offers the option to filter the retrieved data by symbol and magic number, allowing for more targeted processing. | ```<br>// Print the total account net profit for the last 7 days<br>DealData dealsData[];<br>if(<br>   GetDealsData(<br>      dealsData,<br>      ONE_WEEK, NOW,<br>      ALL_SYMBOLS, 0<br>   ) &&<br>   ArraySize(dealsData) > 0<br>)<br>  {<br>   double totalGrossProfit = 0.0,<br>          totalSwap = 0.0,<br>          totalCommission = 0.0;<br>   int totalDeals = ArraySize(dealsData);<br>   for(int k = 0; k < totalDeals; k++)<br>     {<br>      if(dealsData[k].entry == DEAL_ENTRY_OUT)<br>        {<br>         totalGrossProfit += dealsData[k].profit;<br>         totalSwap += dealsData[k].swap;<br>         totalCommission += dealsData[k].commission;<br>        }<br>     }<br>   double totalExpenses = totalSwap + totalCommission;<br>   double totalNetProfit = totalGrossProfit - MathAbs(totalExpenses);<br>   Print("-------------------------------------------------");<br>   Print(<br>      "Account No: ", AccountInfoInteger(ACCOUNT_LOGIN),<br>      " [ 7 DAYS NET PROFIT ]"<br>   );<br>   Print(<br>      "Total Gross Profit: ",<br>      DoubleToString(totalGrossProfit, 2),<br>      " ", AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print(<br>      "Total Swap: ", DoubleToString(totalSwap, 2),<br>      " ", AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print(<br>      "Total Commission: ", DoubleToString(totalCommission, 2),<br>      " ", AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print(<br>      "Total Net Profit: ",<br>      DoubleToString(totalNetProfit, 2), " ",<br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>  }<br>``` |
| ```<br>bool GetOrdersData(<br>   OrderData &ordersData[], // [out]<br>   datetime fromDateTime,<br>   datetime toDateTime,<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function queries for and retrieves order data within a specified time range. Additionally, it offers the option to filter the retrieved data by symbol and magic number. | ```<br>// Print the total BUY Orders filled in the last 7 days<br>OrderData ordersData[];<br>if(<br>   GetOrdersData(ordersData, ONE_WEEK, NOW) &&<br>   ArraySize(ordersData) > 0<br>)<br>  {<br>   int totalBuyOrdersFilled = 0,<br>       totalOrders = ArraySize(ordersData);<br>   for(int w = 0; w < totalOrders; w++)<br>     {<br>      if(ordersData[w].type == ORDER_TYPE_BUY)<br>         ++totalBuyOrdersFilled;<br>     }<br>   Print("");<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN));<br>   Print(totalBuyOrdersFilled, " BUY Orders Filled in the last 7 days!");<br>  }<br>``` |
| ```<br>bool GetPositionsData(<br>   PositionData &positionsData[], // [out]<br>   datetime fromDateTime,<br>   datetime toDateTime,<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function searches and retrieves closed position history data within a specified time range. You have the option of filtering the retrieved data by symbol and magic number values. | ```<br>// Print total pips earned in last 24hrs for specified symbol and magic<br>string symbol = _Symbol;<br>long magic = 0;<br>PositionData positionsData[];<br>if(<br>   GetPositionsData(positionsData, ONE_DAY, NOW, symbol, magic) &&<br>   ArraySize(positionsData) > 0<br>)<br>  {<br>   int totalPipsEarned = 0,<br>       totalPositions = ArraySize(positionsData);<br>   for(int k = 0; k < totalPositions; k++)<br>     {<br>      totalPipsEarned += positionsData[k].pipProfit;<br>     }<br>   Print("");<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN));<br>   Print(<br>      totalPipsEarned, " pips earned in the last 24hrs for ", symbol,<br>      " with magic no. ", magic<br>   );<br>  }<br>``` |
| ```<br>bool GetPendingOrdersData(<br>   PendingOrderData &pendingOrdersData[], // [out]<br>   datetime fromDateTime,<br>   datetime toDateTime,<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function gets the pending orders data history within a specified time range. It also gives you the option of filtering the pending orders history data by symbol and magic number. | ```<br>// Print total number of buy and sell stops filled for symbol and magic<br>// in the last 7 days<br>string symbol = _Symbol;<br>long magic = 0;<br>PendingOrderData pendingOrdersData[];<br>if(<br>   GetPendingOrdersData(pendingOrdersData, ONE_WEEK, NOW, symbol, magic) &&<br>   ArraySize(pendingOrdersData) > 0<br>)<br>  {<br>   int totalBuyStopsFilled = 0, totalSellStopsFilled = 0,<br>       totalPendingOrders = ArraySize(pendingOrdersData);<br>   for(int k = 0; k < totalPendingOrders; k++)<br>     {<br>      if(pendingOrdersData[k].type == ORDER_TYPE_BUY_STOP)<br>         ++totalBuyStopsFilled;<br>      if(pendingOrdersData[k].type == ORDER_TYPE_SELL_STOP)<br>         ++totalSellStopsFilled;<br>     }<br>   Print("");<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN), ", Magic No = ", magic);<br>   Print(<br>      symbol, " --> Total Filled - (Buy Stops = ", totalBuyStopsFilled,<br>      ") (Sell Stops = ", totalSellStopsFilled, ") in the last 7 days."<br>   );<br>  }<br>``` |

**3\. Retrieving All Historical Trade**

These functions provide a comprehensive way to fetch all available trade history data in the account, with optional filters for _symbol_ and _magic_ number, without the requirement of inputting a specific time range:

| Function Prototype Definition | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool GetAllDealsData(<br>   DealData &dealsData[], // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | The function retrieves all deal data, with the option of filtering the deals history data by symbol and magic number. | ```<br>// Find and list total deposited funds in the account<br>DealData dealsData[];<br>if(GetAllDealsData(dealsData) && ArraySize(dealsData) > 0)<br>  {<br>   double totalDeposits = 0.0;<br>   int totalDeals = ArraySize(dealsData);<br>   Print("");<br>   for(int k = 0; k < totalDeals; k++)<br>     {<br>      if(dealsData[k].type == DEAL_TYPE_BALANCE)<br>        {<br>         totalDeposits += dealsData[k].profit;<br>         Print(<br>            dealsData[k].profit, " ", AccountInfoString(ACCOUNT_CURRENCY),<br>            " --> Cash deposit on: ", dealsData[k].time<br>         );<br>        }<br>     }<br>   Print("-------------------------------------------------");<br>   Print(<br>      "Account No: ", AccountInfoInteger(ACCOUNT_LOGIN),<br>      " Total Cash Deposits: ", totalDeposits, " ", <br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>  }<br>``` |
| ```<br>bool GetAllOrdersData(<br>   OrderData &ordersData[], // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function fetches all order data while providing the option to filter the orders history by symbol and magic number. | ```<br>// Find if the account has ever gotten a Stop Out/Margin Call<br>OrderData ordersData[];<br>if(GetAllOrdersData(ordersData) && ArraySize(ordersData) > 0)<br>  {<br>   int totalStopOuts = 0;<br>   int totalOrders = ArraySize(ordersData);<br>   Print("");<br>   for(int k = 0; k < totalOrders; k++)<br>     {<br>      if(ordersData[k].reason == ORDER_REASON_SO)<br>        {<br>         ++totalStopOuts;<br>         Print(<br>            EnumToString(ordersData[k].type),<br>            " --> on: ", ordersData[k].timeDone<br>         );<br>        }<br>     }<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN));<br>   Print("Total STOP OUT events: ", totalStopOuts);<br>  }<br>``` |
| ```<br>bool GetAllPositionsData(<br>   PositionData &positionsData[], // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | The function obtains all position history data, with the option of filtering the closed positions data by symbol and magic number. | ```<br>// Find the average trade duration in seconds<br>PositionData positionsData[];<br>if(GetAllPositionsData(positionsData) && ArraySize(positionsData) > 0)<br>  {<br>   long totalTradesDuration = 0; <br>   int totalPositions = ArraySize(positionsData);<br>   Print("");<br>   for(int k = 0; k < totalPositions; k++)<br>     {<br>      totalTradesDuration += positionsData[k].duration;<br>     }<br>   long averageTradesDuration = totalTradesDuration / totalPositions;<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN));<br>   Print("Average trade duration: ", averageTradesDuration, " seconds.");<br>  }<br>``` |
| ```<br>bool GetAllPendingOrdersData(<br>   PendingOrderData &pendingOrdersData[], // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function retrieves all pending order data, with the option of filtering the pending orders history data by symbol and magic number. | ```<br>// Find the total expired pending orders in the account<br>PendingOrderData pendingOrdersData[];<br>if(GetAllPendingOrdersData(pendingOrdersData) && ArraySize(pendingOrdersData) > 0)<br>  {<br>   int totalExpiredPendingOrders = 0;<br>   int totalPendingOrders = ArraySize(pendingOrdersData);<br>   Print("");<br>   Print("-- EXPIRED PENDING ORDERS --");<br>   for(int k = 0; k < totalPendingOrders; k++)<br>     {<br>      if(pendingOrdersData[k].state == ORDER_STATE_EXPIRED)<br>        {<br>         ++totalExpiredPendingOrders;<br>         Print("Symbol = ", pendingOrdersData[k].symbol);<br>         Print("Time Setup = ", pendingOrdersData[k].timeSetup);<br>         Print("Ticket = ", pendingOrdersData[k].ticket);<br>         Print("Price Open = ", pendingOrdersData[k].priceOpen);<br>         Print(<br>            "SL Price = ", pendingOrdersData[k].slPrice,<br>            ", TP Price = ", pendingOrdersData[k].tpPrice<br>         );<br>         Print("Expiration Time = ", pendingOrdersData[k].expirationTime);<br>         Print("");<br>        }<br>     }<br>   Print("-------------------------------------------------");<br>   Print("Account No: ", AccountInfoInteger(ACCOUNT_LOGIN));<br>   Print("Total Expired Pending Orders: ", totalExpiredPendingOrders);<br>  }<br>``` |

**4\. Analyzing Last Closed Positions**

These functions are designed to obtain comprehensive details about the most recently closed positions, providing insights into key aspects such as profit or loss, trade volume, and the timing of each position's closure. Each function serves a unique purpose, ensuring you have access to every critical aspect of your last trade. The functions also give you the option of filtering the position history data to be processed by _symbol_ and _magic_ number:

| Function Prototype Definition | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool GetLastClosedPositionData(<br>   PositionData &lastClosedPositionInfo, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | The function fetches a complete snapshot of the final closed position, including all relevant details. You have the option of filtering the result by symbol and magic number. | ```<br>// Get the last closed position in the account<br>PositionData lastClosedPositionInfo;<br>if(<br>   GetLastClosedPositionData(lastClosedPositionInfo) &&<br>   lastClosedPositionInfo.ticket > 0<br>)<br>  {<br>   // Process the last closed position data<br>   Print("---LAST CLOSED POSITION--");<br>   Print("Symbol: ", lastClosedPositionInfo.symbol);<br>   Print("Type: ", EnumToString(lastClosedPositionInfo.type));<br>   Print("Open Time: ", lastClosedPositionInfo.openTime);<br>   Print("Close Time: ", lastClosedPositionInfo.closeTime);<br>   Print("Profit: ", lastClosedPositionInfo.profit);<br>   // Place more position properties analysis code....<br>  }<br>//-<br>// Get the last closed position for GBPUSD and magic 0<br>PositionData lastClosedPositionInfo;<br>string symbol = "GBPUSD";<br>if(<br>   GetLastClosedPositionData(lastClosedPositionInfo, symbol, 0) &&<br>   lastClosedPositionInfo.ticket > 0<br>)<br>  {<br>   // Process the last closed position data<br>   Print("---LAST CLOSED POSITION FOR ", symbol, " --");<br>   Print("Symbol: ", lastClosedPositionInfo.symbol);<br>   Print("Type: ", EnumToString(lastClosedPositionInfo.type));<br>   Print("Open Time: ", lastClosedPositionInfo.openTime);<br>   Print("Close Time: ", lastClosedPositionInfo.closeTime);<br>   Print("Profit: ", lastClosedPositionInfo.profit);<br>   // Place more position properties analysis code....<br>  }<br>``` |
| ```<br>bool LastClosedPositionType(<br>   ENUM_POSITION_TYPE &lastClosedPositionType, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function reveals whether the last position was a buy or sell trade. It has the option of filtering the data by symbol and magic number. | ```<br>// Get the last closed position type in the account<br>ENUM_POSITION_TYPE lastClosedPositionType;<br>LastClosedPositionType(lastClosedPositionType);<br>Print(<br>   "Account's last closed position type: ",<br>   EnumToString(lastClosedPositionType)<br>);<br>//--<br>// Get the last closed position type for EURUSD and magic 0<br>ENUM_POSITION_TYPE lastClosedPositionType;<br>LastClosedPositionType(lastClosedPositionType, "EURUSD", 0);<br>Print(<br>   "EURUSD: last closed position type: ",<br>   EnumToString(lastClosedPositionType)<br>);<br>``` |
| ```<br>bool LastClosedPositionVolume(<br>   double &lastClosedPositionVolume, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function provides the trade volume of the most recently closed position. The function also gives you the option of filtering the data by symbol and magic number. | ```<br>// Get the last closed position volume in the account<br>double lastClosedPositionVolume;<br>LastClosedPositionVolume(lastClosedPositionVolume);<br>Print(<br>   "Account's last closed position volume: ",<br>   lastClosedPositionVolume<br>);<br>//--<br>// Get the last closed position volume for GBPUSD and magic 0<br>double lastClosedPositionVolume;<br>LastClosedPositionVolume(lastClosedPositionVolume, "GBPUSD", 0);<br>Print(<br>   "GBPUSD: last closed position volume: ",<br>   lastClosedPositionVolume<br>);<br>``` |
| ```<br>bool LastClosedPositionSymbol(<br>   string &lastClosedPositionSymbol, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | The function obtains the symbol of the most recently closed position. Gives you the option of filtering the result by symbol and magic number. | ```<br>// Get the last closed position's symbol in the account<br>string lastClosedPositionSymbol;<br>LastClosedPositionSymbol(lastClosedPositionSymbol);<br>Print(<br>   lastClosedPositionSymbol, " is the last closed position symbol ", <br>   "in the account"<br>);<br>//--<br>// Get the last closed position's symbol for magic 0<br>string lastClosedPositionSymbol;<br>LastClosedPositionSymbol(lastClosedPositionSymbol, ALL_SYMBOLS, 0);<br>Print(<br>   lastClosedPositionSymbol, " is the last closed position symbol ",<br>   "for magic no: 0."<br>);<br>``` |
| ```<br>bool LastClosedPositionTicket(<br>   ulong &lastClosedPositionTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function retrieves the ticket number of the most recently closed position. The function hast the option of filtering the result by symbol and magic number. | ```<br>// Get the last closed position's ticket in the account<br>long lastClosedPositionTicket;<br>LastClosedPositionTicket(lastClosedPositionTicket);<br>Print(<br>   "Account's last closed position's ticket: ",<br>   lastClosedPositionTicket<br>);<br>//--<br>// Get the last closed position's ticket for EURUSD and magic 0<br>long lastClosedPositionTicket;<br>LastClosedPositionTicket(lastClosedPositionTicket, "EURUSD", 0);<br>Print(<br>   "EURUSD: last closed position's ticket: ",<br>   lastClosedPositionTicket<br>);<br>``` |
| ```<br>bool LastClosedPositionProfit(<br>   double &lastClosedPositionProfit, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function delivers the gross profit generated by the last trade. You also get the option of filtering the data by symbol and magic number. | ```<br>// Get the last closed position's profit in the account<br>double lastClosedPositionProfit;<br>LastClosedPositionProfit(lastClosedPositionProfit);<br>Print(<br>   "Account's last closed position's profit: ",<br>   lastClosedPositionProfit, " ", AccountInfoString(ACCOUNT_CURRENCY)<br>);<br>//--<br>// Get the last closed position's profit for EURUSD and magic 0<br>//double lastClosedPositionProfit;<br>LastClosedPositionProfit(lastClosedPositionProfit, "EURUSD", 0);<br>Print(<br>   "EURUSD: last closed position's profit: ",<br>   lastClosedPositionProfit, " ", AccountInfoString(ACCOUNT_CURRENCY)<br>);<br>``` |

Here are the remaining library functions responsible for retrieving the properties of the last closed position. For implementation details, refer to the code examples above, including functions like _LastClosedPositionVolume()_, _LastClosedPositionType()_, and others, as they all follow a similar approach.

| Function Prototype Definition | Description |
| --- | --- |
| ```<br>bool LastClosedPositionNetProfit(<br>   double &lastClosedPositionNetProfit, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Calculates the net profit of the last trade, factoring in swaps and commissions. The function has the option of filtering the data by symbol and magic number. |
| ```<br>bool LastClosedPositionPipProfit(<br>   int &lastClosedPositionPipProfit, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Obtains the profit measured in pips for the final closed position. Data can be filtered by symbol and magic number. |
| ```<br>bool LastClosedPositionClosePrice(<br>   double &lastClosedPositionClosePrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the price at which the position was closed. The function provides the option of filtering the result by symbol and magic number. |
| ```<br>bool LastClosedPositionOpenPrice(<br>   double &lastClosedPositionOpenPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Reveals the price at which the position was initially opened. You can optionally filter the data by symbol and magic number. |
| ```<br>bool LastClosedPositionSlPrice(<br>   double &lastClosedPositionSlPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches the stop-loss price set for the last closed position. The function gives you the option of filtering the result by symbol and magic number. |
| ```<br>bool LastClosedPositionTpPrice(<br>   double &lastClosedPositionTpPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Provides the take-profit price assigned to the last closed trade. An option to filter the data by symbol and magic number is also provided. |
| ```<br>bool LastClosedPositionSlPips(<br>   int &lastClosedPositionSlPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Indicates the stop-loss distance in pips for the last closed position. Data filtering by symbol and magic number is also supported. |
| ```<br>bool LastClosedPositionTpPips(<br>   int &lastClosedPositionTpPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Shows the take-profit distance in pips for the most recently closed trade. The function can also optionally filter the result by symbol and magic number. |
| ```<br>bool LastClosedPositionOpenTime(<br>   datetime &lastClosedPositionOpenTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the timestamp when the last closed position was opened. Data filtering by symbol and magic number is also provided. |
| ```<br>bool LastClosedPositionCloseTime(<br>   datetime &lastClosedPositionCloseTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Provides the exact time when the most recently closed position was closed. You can also optionally filter the result by symbol and magic number. |
| ```<br>bool LastClosedPositionSwap(<br>   double &lastClosedPositionSwap, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Reveals the swap value associated with the last closed position. The information can also be optionally filtered by symbol and magic number values. |
| ```<br>bool LastClosedPositionCommission(<br>   double &lastClosedPositionCommission, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches the commission charged for the last closed trade. Filtering by symbol and magic number is also supported. |
| ```<br>bool LastClosedPositionInitiatingOrderType(<br>   ENUM_ORDER_TYPE &lastClosedPositionInitiatingOrderType, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Identifies the type of order that initiated the last closed position. You can optionally filter the result by symbol and magic number. |
| ```<br>bool LastClosedPositionId(<br>   ulong &lastClosedPositionId, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the unique identifier for the most recent closed position. The symbol and magic parameters are optional and only required when you need to filter the result. |
| ```<br>bool LastClosedPositionInitiatedByPendingOrder(<br>   bool &lastClosedPositionInitiatedByPendingOrder, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Checks if the last closed position was triggered or initiated by a pending order. You can optionally filter the data by symbol and magic number. |
| ```<br>bool LastClosedPositionOpeningOrderTicket(<br>   ulong &lastClosedPositionOpeningOrderTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Provides the ticket number of the order that opened the position. The function can also optionally filter the data by symbol and magic number. |
| ```<br>bool LastClosedPositionOpeningDealTicket(<br>   ulong &lastClosedPositionOpeningDealTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches the ticket number of the deal that initiated the position.  You have the option of filtering the result by symbol and magic number. |
| ```<br>bool LastClosedPositionClosingDealTicket(<br>   ulong &lastClosedPositionClosingDealTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the ticket number of the deal that closed the position. Filtering the data by symbol and magic is also an option. |
| ```<br>bool LastClosedPositionMagic(<br>   ulong &lastClosedPositionMagic, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Reveals the magic number associated with the last closed position. You have the option of filtering the data by symbol and magic number. |
| ```<br>bool LastClosedPositionComment(<br>   string &lastClosedPositionComment, // [Out]<br>   string symbol,  // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves any comments or notes attached to the most recently closed trade.  Filtering the result by symbol and magic number is also a provided option. |
| ```<br>bool LastClosedPositionDuration(<br>   long &lastClosedPositionDuration, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Calculates the total duration (in seconds) that the position remained open. You have the option of filtering the result by symbol and magic number. |

**5\. Analyzing The Last Closed Profitable and Loss-Making Positions**

When you need to retrieve the properties of the last closed profitable or loss-making position, the History Manager library provides two dedicated functions for this task. These functions allow you to quickly access all relevant details with a single function call, enabling you to analyze the most recent closed position based on its profitability. By providing essential data, they help in evaluating trading performance and refining strategies. Additionally, these functions offer the flexibility to filter results by a specific _symbol_ and _magic_ number, allowing you to focus on particular trades and assess the performance of individual strategies or assets more effectively.

| Function Prototype Definition | Description | Example Use Case |
| --- | --- | --- |
| ```<br>bool GetLastClosedProfitablePositionData(<br>   PositionData &lastClosedProfitablePositionInfo, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the details and properties of the most recently closed position that closed with a profit. The function also gives you the option of filtering the results by symbol and magic number. | ```<br>// Get the symbol, ticket, and net and pip profit of<br>// the last closed profitable position<br>PositionData lastClosedProfitablePosition;<br>if(<br>   GetLastClosedProfitablePositionData(lastClosedProfitablePosition) &&<br>   lastClosedProfitablePosition.ticket > 0<br>)<br>  {<br>   Print("-------------------------------------------------");<br>   Print(<br>      lastClosedProfitablePosition.symbol,<br>      " --> LAST CLOSED PROFITABLE POSITION"<br>   );<br>   Print("Ticket = ", lastClosedProfitablePosition.ticket);<br>   Print(<br>      "Net Profit = ", lastClosedProfitablePosition.netProfit, " ",<br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print("Pip Profit = ", lastClosedProfitablePosition.pipProfit);<br>  }<br>//--<br>// Get the ticket, and net and pip profit of<br>// the last closed profitable position for EURUSD and magic 0<br>PositionData lastClosedProfitablePosition;<br>if(<br>   GetLastClosedProfitablePositionData(<br>      lastClosedProfitablePosition, "EURUSD", 0<br>   ) &&<br>   lastClosedProfitablePosition.ticket > 0<br>)<br>  {<br>   Print("-------------------------------------------------");<br>   Print("EURUSD --> LAST CLOSED PROFITABLE POSITION");<br>   Print("Ticket = ", lastClosedProfitablePosition.ticket);<br>   Print(<br>      "Net Profit = ", lastClosedProfitablePosition.netProfit, " ",<br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print("Pip Profit = ", lastClosedProfitablePosition.pipProfit);<br>  }<br>``` |
| ```<br>bool GetLastClosedLossPositionData(<br>   PositionData &lastClosedLossPositionData,  // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the details and properties of the most recently closed position that closed or resulted in a loss. The function also enables you to optionally filter the results by symbol and magic number. | ```<br>// Get the symbol, ticket, and net and pip profit of<br>// the last closed loss position<br>PositionData lastClosedLossPosition;<br>if(<br>   GetLastClosedLossPositionData(lastClosedLossPosition) &&<br>   lastClosedLossPosition.ticket > 0<br>)<br>  {<br>   Print("-------------------------------------------------");<br>   Print(<br>      lastClosedLossPosition.symbol,<br>      " --> LAST CLOSED LOSS POSITION"<br>   );<br>   Print("Ticket = ", lastClosedLossPosition.ticket);<br>   Print(<br>      "Net Profit = ", lastClosedLossPosition.netProfit, " ",<br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print("Pip Profit = ", lastClosedLossPosition.pipProfit);<br>  }<br>//--<br>// Get the ticket, and net and pip profit of<br>// the last closed loss position for GBPUSD and magic 0<br>PositionData lastClosedLossPosition;<br>if(<br>   GetLastClosedLossPositionData(<br>      lastClosedLossPosition, "GBPUSD", 0<br>   ) &&<br>   lastClosedLossPosition.ticket > 0<br>)<br>  {<br>   Print("-------------------------------------------------");<br>   Print("GBPUSD --> LAST CLOSED LOSS POSITION");<br>   Print("Ticket = ", lastClosedLossPosition.ticket);<br>   Print(<br>      "Net Profit = ", lastClosedLossPosition.netProfit, " ",<br>      AccountInfoString(ACCOUNT_CURRENCY)<br>   );<br>   Print("Pip Profit = ", lastClosedLossPosition.pipProfit);<br>  }<br>``` |

**6\. Analyzing Last Filled and Canceled Pending Orders**

To retrieve detailed information about the most recently filled or canceled pending order, the History Manager library provides specialized functions for this purpose. These functions allow you to fetch all relevant order details with a single function call, making it easier to analyze the execution or cancellation of pending orders. By providing key data, they help in assessing order handling, execution efficiency, and trading strategy adjustments. Additionally, these functions offer the flexibility to filter results by a specific _symbol_ and _magic_ number, enabling you to focus on particular orders and refine your approach based on precise historical data.

Since the _last filled and canceled pending order_ functions are imported and implemented using the same approach as those that process the _last closed position_ history data, only their _function prototypes_ and brief descriptions will be provided. If you need an example of how to implement these functions, refer to the " _Analyzing Last Closed Positions_" section above.

To obtain complete details of the last filled pending order, the _GetLastFilledPendingOrderData()_ function retrieves all relevant properties, offering a broad view of the order's execution. If you only need specific attributes, the following dedicated functions will allow you to extract individual details or properties:

| Function Prototype Definition | Description |
| --- | --- |
| ```<br>bool GetLastFilledPendingOrderData(<br>   PendingOrderData &lastFilledPendingOrderData, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function retrieves all pertinent details of the most recent pending order that has been filled, giving a complete picture of how the order was executed. |
| ```<br>bool LastFilledPendingOrderType(<br>   ENUM_ORDER_TYPE &lastFilledPendingOrderType, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches the order type, helping you determine whether it was a buy stop, sell limit, or another pending order variant. |
| ```<br>bool LastFilledPendingOrderSymbol(<br>   string &lastFilledPendingOrderSymbol, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Identifies the trading instrument, allowing you to focus on orders associated with a particular asset. |
| ```<br>bool LastFilledPendingOrderTicket(<br>   ulong &lastFilledPendingOrderTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Provides the unique order ticket, essential for referencing or tracking a specific executed order. |
| ```<br>bool LastFilledPendingOrderPriceOpen(<br>   double &lastFilledPendingOrderPriceOpen, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the price at which the order was triggered, crucial for analyzing entry conditions. |
| ```<br>bool LastFilledPendingOrderSlPrice(<br>   double &lastFilledPendingOrderSlPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastFilledPendingOrderTpPrice(<br>   double &lastFilledPendingOrderTpPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | These functions return the stop-loss and take-profit levels, respectively, ensuring you can assess risk management settings. |
| ```<br>bool LastFilledPendingOrderSlPips(<br>   int &lastFilledPendingOrderSlPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastFilledPendingOrderTpPips(<br>   int &lastFilledPendingOrderTpPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | These functions express stop-loss and take-profit distances in pips, giving a clearer view of the trade’s risk-to-reward ratio. |
| ```<br>bool LastFilledPendingOrderTimeSetup(<br>   datetime &lastFilledPendingOrderTimeSetup, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastFilledPendingOrderTimeDone(<br>   datetime &lastFilledPendingOrderTimeDone, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastFilledPendingOrderTimeSetup()_ records when the order was placed, while _LastFilledPendingOrderTimeDone()_ logs when it was executed. This helps in analyzing the time duration between when the pending order was set up and when it was triggered or filled. |
| ```<br>bool LastFilledPendingOrderExpirationTime(<br>   datetime &lastFilledPendingOrderExpirationTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Checks and provides the expiration time, useful for time-sensitive trading strategies. |
| ```<br>bool LastFilledPendingOrderPositionId(<br>   ulong &lastFilledPendingOrderPositionId, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the position ID which is critical for linking the order to its corresponding position, ensuring accurate tracking. |
| ```<br>bool LastFilledPendingOrderMagic(<br>   ulong &lastFilledPendingOrderMagic, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the magic number, letting you filter orders based on automated strategies or trading bots. |
| ```<br>bool LastFilledPendingOrderReason(<br>   ENUM_ORDER_REASON &lastFilledPendingOrderReason, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Captures the reason for execution, offering insight into system-initiated or manual order processing. |
| ```<br>bool LastFilledPendingOrderTypeFilling(<br>   ENUM_ORDER_TYPE_FILLING &lastFilledPendingOrderTypeFilling, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastFilledPendingOrderTypeTime(<br>   datetime &lastFilledPendingOrderTypeTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastFilledPendingOrderTypeFilling()_ and _LastFilledPendingOrderTypeTime()_ define the order filling method and time model, respectively, aiding in execution analysis. |
| ```<br>bool LastFilledPendingOrderComment(<br>   string &lastFilledPendingOrderComment, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches any associated comments of the last filled pending order, which can be useful for labeling and categorizing trades. |

Similarly, if you need to examine the last canceled pending order, the _GetLastCanceledPendingOrderData()_ function retrieves all details in one call. Alternatively, you can extract individual properties using the following specialized functions:

| Function Prototype Definition | Description |
| --- | --- |
| ```<br>bool GetLastCanceledPendingOrderData(<br>   PendingOrderData &lastCanceledPendingOrderData, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | This function provides a comprehensive record of the most recently canceled pending order, including all its properties and attributes. |
| ```<br>bool LastCanceledPendingOrderType(<br>   ENUM_ORDER_TYPE &lastCanceledPendingOrderType, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves and specifies the type of the last canceled pending order. |
| ```<br>bool LastCanceledPendingOrderSymbol(<br>   string &lastCanceledPendingOrderSymbol, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Gets and identifies the trading symbol for the most recently canceled pending order, allowing for targeted analysis. |
| ```<br>bool LastCanceledPendingOrderTicket(<br>   ulong &lastCanceledPendingOrderTicket, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Provides the last canceled pending order’s unique ticket number for reference |
| ```<br>bool LastCanceledPendingOrderPriceOpen(<br>   double &lastCanceledPendingOrderPriceOpen, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Gets and saves the original intended entry price of the last canceled pending order. |
| ```<br>bool LastCanceledPendingOrderSlPrice(<br>   double &lastCanceledPendingOrderSlPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastCanceledPendingOrderTpPrice(<br>   double &lastCanceledPendingOrderTpPrice, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastCanceledPendingOrderSlPrice()_ and _LastCanceledPendingOrderTpPrice()_ retrieve the mostly recently canceled pending order’s stop-loss and take-profit price levels. |
| ```<br>bool LastCanceledPendingOrderSlPips(<br>   int &lastCanceledPendingOrderSlPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastCanceledPendingOrderTpPips(<br>   int &lastCanceledPendingOrderTpPips, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastCanceledPendingOrderSlPips()_ and _LastCanceledPendingOrderTpPips()_ express the last canceled pending order's stop-loss and take-profit distances in pips. |
| ```<br>bool LastCanceledPendingOrderTimeSetup(<br>   datetime &lastCanceledPendingOrderTimeSetup, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastCanceledPendingOrderTimeDone(<br>   datetime &lastCanceledPendingOrderTimeDone, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastCanceledPendingOrderTimeSetup()_ provides the time when the last canceled pending order was placed, while _LastCanceledPendingOrderTimeDone()_ records the time when it was removed by cancellation. |
| ```<br>bool LastCanceledPendingOrderExpirationTime(<br>   datetime &lastCanceledPendingOrderExpirationTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves and saves the time when the canceled pending order was set to expire. |
| ```<br>bool LastCanceledPendingOrderMagic(<br>   ulong &lastCanceledPendingOrderMagic, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Gets and stores the magic number associated with the last canceled pending order allowing for filtering based on automated strategy identifiers. |
| ```<br>bool LastCanceledPendingOrderReason(<br>   ENUM_ORDER_REASON &lastCanceledPendingOrderReason, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Retrieves the order reason of the last canceled pending order and allows us to identify on which platform the order was initiated from. |
| ```<br>bool LastCanceledPendingOrderTypeFilling(<br>   ENUM_ORDER_TYPE_FILLING &lastCanceledPendingOrderTypeFilling, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>bool LastCanceledPendingOrderTypeTime(<br>   datetime &lastCanceledPendingOrderTypeTime, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | _LastCanceledPendingOrderTypeFilling()_ and _LastCanceledPendingOrderTypeTime()_ retrieve and save the filling and time type settings for the last canceled pending order. |
| ```<br>bool LastCanceledPendingOrderComment(<br>   string &lastCanceledPendingOrderComment, // [Out]<br>   string symbol, // Optional parameter<br>   ulong magic // Optional parameter<br>);<br>``` | Fetches any user-defined comments saved when opening the last canceled pending order for tracking purposes. |

**7\. Utility Functions**

| Function Prototype Definition | Description |
| --- | --- |
| ```<br>string BoolToString(bool boolVariable);<br>``` | The utility function converts a _boolean_ value to a string (" _TRUE_" or " _FALSE_") for easier readability and logging. |
| ```<br>datetime GetPeriodStart(int periodType);<br>``` | This utility function returns the _datetime_ value for the _start of the year, month, week (starting on Sunday),_ or _day_, depending on the input type. It accepts predefined _macros_ from the header file as input, namely _TODAY_, _THIS\_WEEK_, _THIS\_MONTH_, and _THIS\_YEAR_. This function simplifies the process of obtaining the datetime value for these periods, allowing you to do so with a single line of code. |

In the next section, we will explore practical examples demonstrating how the History Management library can be used to generate insightful trading analytics. Additionally, we will develop an Expert Advisor that leverages the history library to determine the optimal volume or lot and position direction for executing new trades. This will highlight how historical data can be effectively incorporated into automated trading systems, improving both strategy execution and decision-making.

### How To Calculate an Expert Advisor's or Symbol's Profit Factor, Gross Profit/Loss, and Average Gross Profit and Loss Per Trade

When evaluating the performance of an Expert Advisor or a trading strategy in MetaTrader 5, one of the most important metrics to consider is the Profit Factor. The Profit Factor is a ratio that measures the gross profit against the gross loss over a specific period. It provides a clear indication of how profitable a trading strategy is by comparing the total gains to the total losses.

A high Profit Factor indicates that the strategy is generating more profits relative to its losses, which is a desirable trait for any trading system. Conversely, a low Profit Factor suggests that the strategy may not be as effective and might need further optimization.

The Profit Factor is calculated using the following formula:

```
Profit Factor = Gross Profit / Gross Loss
```

Where:

- _Gross Profit_ is the total amount of money earned from all profitable trades.
- _Gross Loss_ is the total amount of money lost from all losing trades.

A Profit Factor greater than 1 indicates that the strategy is profitable, as the gross profit exceeds the gross loss. A Profit Factor of 1 means that the gross profit and gross loss are equal, suggesting a break-even scenario. A Profit Factor less than 1 indicates that the strategy is losing money, as the gross loss exceeds the gross profit.

Understanding and calculating the Profit Factor is crucial for traders and developers to make informed decisions about the viability and performance of their trading strategies.

In this example, I will use the _HistoryManager.ex5_ library to demonstrate how easy it is to implement this calculation in your MQL5 code.

Start by creating a new Expert Advisor and naming it _GetProfitFactor.mq5_. Save it in the following folder path: _Experts\\Wanateki\\HistoryManager\\GetProfitFactor.mq5_. Note that the directory where you save the Expert Advisor is important, as it will influence the path used to include the header file containing the library function prototype definitions.

The _GetProfitFactor.mq5_ Expert Advisor will help you analyze the performance of your trading strategies by calculating key metrics such as gross profit, gross loss, and the profit factor itself.

First, we need to include the _HistoryManager.ex5_ library and define an _enumeration_ that we will use to input the _symbol_ string value in our Expert Advisor. The _symbolName_ enumeration will allow us to specify whether we want to analyze the current chart symbol or _all symbols_ in the account. This enumeration together with the _magic_ number input will allow us to filter the results to suit our targeted symbols and Expert Advisors.

```
#include <Wanateki/Toolkit/HistoryManager/HistoryManager.mqh>
//--
enum symbolName
  {
   CURRENT_CHART_SYMBOL,
   ALL_ACCOUNT_SYMBOLS,
  };
```

Next, we define the _input_ parameters the Expert Advisor user will use to filter the output.

```
//--- input parameters
input ulong  magicNo = 0;          //Magic Number (0 to disable)
input symbolName getSymbolName = ALL_ACCOUNT_SYMBOLS;
```

We initialize the variables that will be used to store our calculations and display the results.

```
string currency = " " + AccountInfoString(ACCOUNT_CURRENCY);
double totalGrossProfit = 0.0, totalGrossLoss = 0.0,
       averageGrossProfit = 0.0, averageGrossLoss = 0.0,
       averageProfitOrLossPerTrade = 0.0,
       profitFactor = 0.0;
int totalTrades = 0,
    totalLossPositions = 0, totalProfitPositions = 0;
string symbol, printedSymbol, profitFactorSummery,
       commentString;
```

In the _OnInit()_ function, we will use a _switch_ statement to determine the _symbol_ to analyze. Once the _symbol_ is identified, we will call the _CalcProfitFactor()_ function, which is responsible for calculating the Profit Factor. As the first task when the Expert Advisor loads, we will then display all the results on the chart using the _Comment()_ function.

```
int OnInit()
  {
//---
   switch(getSymbolName)
     {
      case CURRENT_CHART_SYMBOL:
         symbol = _Symbol;
         break;
      case ALL_ACCOUNT_SYMBOLS:
         symbol = ALL_SYMBOLS;
         break;
      default:
         symbol = ALL_SYMBOLS;
     }
   printedSymbol = symbol;
   if(symbol == "")
      printedSymbol = "ALL_SYMBOLS";
//--
   CalcProfitFactor();
   Comment(commentString);
//---
   return(INIT_SUCCEEDED);
  }
```

Before we can proceed, let us code the _CalcProfitFactor()_ function, as it contains most of the code that runs the _GetProfitFactor_ Expert Advisor.

We will begin the _CalcProfitFactor()_ function by coding the function signature, and then proceed by resetting all variables to their initial values. This ensures that any previous calculations do not interfere with the current analysis.

```
   totalGrossProfit = 0.0;
   totalGrossLoss = 0.0;
   averageProfitOrLossPerTrade = 0.0;
   averageGrossProfit = 0.0;
   averageGrossLoss = 0.0;
   profitFactor = 0.0;
   totalTrades = 0;
   totalLossPositions = 0;
   totalProfitPositions = 0;
```

Next, we use the _GetAllPositionsData()_ function to retrieve historical trade data. This function populates the _positionsData_ array with all the trade positions based on the specified _symbol_ and _magic_ number. Likewise, we check if the function returns _true_ and if the array size exceeds zero to ensure that we have valid data to process. This step is essential for accessing the necessary trade information to perform our calculations.

```
if(GetAllPositionsData(positionsData, symbol, magicNo) && ArraySize(positionsData) > 0)
     {
      //--
     }
```

We then loop through each trade in the _positionsData_ array to calculate the _total gross profit and loss_. For each trade, we check if the profit exceeds zero to determine if it is a profitable trade. If it is, we increment the _totalProfitPositions_ counter and add the profit to _totalGrossProfit_. If the profit is less than or equal to _zero_, we increment the _totalLossPositions_ counter and add the _absolute_ value of the profit to _totalGrossLoss_. This loop helps us accumulate the total gains and losses from all trades.

```
      totalTrades = ArraySize(positionsData);
      for(int r = 0; r < totalTrades; r++)
        {
         if(positionsData[r].profit > 0) // profitable trade
           {
            ++totalProfitPositions;
            totalGrossProfit += positionsData[r].profit;
           }
         else  // loss trade
           {
            ++totalLossPositions;
            totalGrossLoss += MathAbs(positionsData[r].profit);
           }
        }
      // Calculate the profit factor and other data
      if(totalGrossProfit > 0 || totalGrossLoss > 0)
         averageProfitOrLossPerTrade = NormalizeDouble(
                                          (totalGrossProfit + totalGrossLoss) / totalTrades, 2
                                       );
      if(totalGrossProfit > 0)
         averageGrossProfit = NormalizeDouble(
                                 (totalGrossProfit / totalProfitPositions), 2
                              );
      if(totalGrossLoss > 0)
         averageGrossLoss = NormalizeDouble(
                               (totalGrossLoss / totalLossPositions), 2
                            );
      //--
      if(totalGrossLoss == 0.0)
         profitFactor = 0.0; // Avoid division by zero, indicating no losses
      profitFactor = totalGrossProfit / totalGrossLoss;
     }
```

After accumulating the _total gross profit and loss_, we proceed to calculate various metrics. First, we compute the average profit or loss per trade by dividing the sum of total gross profit and loss by the total number of trades. Furthermore, we then calculate the average gross profit by dividing the total gross profit by the number of profitable trades. Similarly, we calculate the average gross loss by dividing the total gross loss by the number of loss trades. Finally, we calculate the profit factor by dividing the total gross profit by the total gross loss. These metrics provide a comprehensive view of the trading strategy's performance.

```
   profitFactorSummery = "Profit Factor = " + DoubleToString(profitFactor, 2);
   if(profitFactor > 2.0)
     {
      profitFactorSummery = profitFactorSummery +
                            "\nThe trading strategy is HIGHLY PROFITABLE and efficient.";
     }
   else
      if(profitFactor > 1.5)
        {
         profitFactorSummery = profitFactorSummery +
                               "\nThe trading strategy is profitable and well-balanced.";
        }
      else
         if(profitFactor > 1.0)
           {
            profitFactorSummery = profitFactorSummery +
                                  "\nThe trading strategy is slightly profitable but may need improvement.";
           }
         else
            if(profitFactor == 1.0)
              {
               profitFactorSummery = profitFactorSummery +
                                     "\nThe strategy is break-even with no net gain or loss.";
              }
            else
              {
               profitFactorSummery = profitFactorSummery +
                                     "\nThe trading strategy is unprofitable and needs optimization.";
              }
```

We analyze the profit factor to generate a summary string that describes the trading strategy's performance. Here’s how we interpret the profit factor:

- Profit Factor > 2.0: The strategy is highly profitable and efficient.
- Profit Factor between 1.0 and 1.5: The strategy is slightly profitable but may need improvement.
- Profit Factor = 1.0: The strategy breaks even, with no net gain or loss.
- Profit Factor < 1.0: The strategy is unprofitable and requires optimization.

This profit factor analysis provides a quick understanding of the overall effectiveness of the trading strategy.

Lastly, we generate a detailed _comment_ string that includes all the calculated metrics and the profit factor summary. This string is formatted to display the _symbol_ being analyzed, the _magic_ number, the total number of trades, the number of profitable and loss trades, the total gross profit and loss, the average profit or loss per trade, the average gross profit and loss, and the profit factor summary.

Here is the full _CalcProfitFactor()_ function with all the code segments in their place.

```
void CalcProfitFactor()
  {
   totalGrossProfit = 0.0;
   totalGrossLoss = 0.0;
   averageProfitOrLossPerTrade = 0.0;
   averageGrossProfit = 0.0;
   averageGrossLoss = 0.0;
   profitFactor = 0.0;
   totalTrades = 0;
   totalLossPositions = 0;
   totalProfitPositions = 0;
//--
   PositionData positionsData[];
   if(GetAllPositionsData(positionsData, symbol, magicNo) && ArraySize(positionsData) > 0)
     {
      totalTrades = ArraySize(positionsData);
      for(int r = 0; r < totalTrades; r++)
        {
         if(positionsData[r].profit > 0) // profitable trade
           {
            ++totalProfitPositions;
            totalGrossProfit += positionsData[r].profit;
           }
         else  // loss trade
           {
            ++totalLossPositions;
            totalGrossLoss += MathAbs(positionsData[r].profit);
           }
        }
      // Calculate the profit factor and other data
      if(totalGrossProfit > 0 || totalGrossLoss > 0)
         averageProfitOrLossPerTrade = NormalizeDouble(
                                          (totalGrossProfit + totalGrossLoss) / totalTrades, 2
                                       );
      if(totalGrossProfit > 0)
         averageGrossProfit = NormalizeDouble(
                                 (totalGrossProfit / totalProfitPositions), 2
                              );
      if(totalGrossLoss > 0)
         averageGrossLoss = NormalizeDouble(
                               (totalGrossLoss / totalLossPositions), 2
                            );
      //--
      if(totalGrossLoss == 0.0)
         profitFactor = 0.0; // Avoid division by zero, indicating no losses
      profitFactor = totalGrossProfit / totalGrossLoss;
     }

// Analyze the Profit Factor result
   profitFactorSummery = "Profit Factor = " + DoubleToString(profitFactor, 2);
   if(profitFactor > 2.0)
     {
      profitFactorSummery = profitFactorSummery +
                            "\nThe trading strategy is HIGHLY PROFITABLE and efficient.";
     }
   else
      if(profitFactor > 1.5)
        {
         profitFactorSummery = profitFactorSummery +
                               "\nThe trading strategy is profitable and well-balanced.";
        }
      else
         if(profitFactor > 1.0)
           {
            profitFactorSummery = profitFactorSummery +
                                  "\nThe trading strategy is slightly profitable but may need improvement.";
           }
         else
            if(profitFactor == 1.0)
              {
               profitFactorSummery = profitFactorSummery +
                                     "\nThe strategy is break-even with no net gain or loss.";
              }
            else
              {
               profitFactorSummery = profitFactorSummery +
                                     "\nThe trading strategy is unprofitable and needs optimization.";
              }

   commentString =
      "\n\n-----------------------------------------------------------------------------------------------------" +
      "\n  HistoryManager.ex5 --- PROFIT FACTOR ANALYTICS ---" +
      "\n-----------------------------------------------------------------------------------------------------" +
      "\n   -> Symbol   = " + printedSymbol +
      "\n   -> Magic No = " + IntegerToString(magicNo) +
      "\n-----------------------------------------------------------------------------------------------------" +
      "\n-----------------------------------------------------------------------------------------------------" +
      "\n" + profitFactorSummery +
      "\n-----------------------------------------------------------------------------------------------------" +
      "\n-----------------------------------------------------------------------------------------------------" +
      "\n   -> Total Trades Analysed   = " + IntegerToString(totalTrades) +
      "\n   -> Total Profitable Trades = " + IntegerToString(totalProfitPositions) +
      "\n   -> Total Loss Trades       = " + IntegerToString(totalLossPositions) +
      "\n   --------------------------------------------------------" +
      "\n   -> Total Gross Profit = " + DoubleToString(totalGrossProfit, 2) + currency +
      "\n   -> Total Gross Loss   = -" + DoubleToString(totalGrossLoss, 2) + currency +
      "\n   ----------------------------------" +
      "\n   -> Total Gross (Profit - Loss) = " + DoubleToString(totalGrossProfit - totalGrossLoss, 2) + currency +
      "\n   --------------------------------------------------------" +
      "\n   -> Average Profit or Loss Per Trade = (-/+)" + DoubleToString(averageProfitOrLossPerTrade, 2) + currency +
      "\n   --------------------------------------------------------" +
      "\n   -> Average Gross Profit = " + DoubleToString(averageGrossProfit, 2) + currency +
      "\n   -> Average Gross Loss   = -" + DoubleToString(averageGrossLoss, 2) + currency +
      "\n   --------------------------------------------------------" +
      "\n-----------------------------------------------------------------------------------------------------";
//--
  }
```

We also place the function inside the _OnTrade()_ standard function to execute it and re-calculate the Profit Factor after every trade operation.

```
void OnTrade()
  {
//---
   CalcProfitFactor();
  }
```

To make sure that we are printing updated data on the chart, we will use the _Comment()_ function to print the _commentString_ that holds all the information about the currently updated Profit Factor and print it after every incoming new tick on the _OnTick()_ standard function.

```
void OnTick()
  {
//---
   Comment(commentString);
  }
```

Here is the output of the _GetProfitFactor_ Expert Advisor on the MetaTrader 5 chart window.

![GetProfitFactor EA powered by the HistoryManager.ex5](https://c.mql5.com/2/120/ProfitFactor_EA_powered_by_the_HistoryManager.ex5.png)

The _GetProfitFactor.mq5_ source file is attached at the end of this article for your convenience.

### How To Calculate the Current Week's Monetary Gross and Net Profit for a Specific Expert Advisor or Symbol

In this example, we are going to create an MQL5 script that will provide a comprehensive overview of the financial performance for the current week, including gross profit, swaps, commissions, and net profit. This will help demonstrate how you can use the History Manager library to quickly assess the weekly performance of the entire account, a specific symbol, or an Expert Advisor's magic number directly within the MetaTrader 5 terminal.

We start by creating a new script named _GetNetProfitThisWeek.mq5_ and save it in this folder: ( _Scripts\\Wanateki\\HistoryManager\\GetNetProfitThisWeek.mq5_). In the new script file, begin by including the _HistoryManager.mqh_ library, which will provide us with access to all the library functions.

```
#include <Wanateki/Toolkit/HistoryManager/HistoryManager.mqh>
```

Next, we will put all our code inside the _OnStart()_ function to keep things simple and clear. The first segment of the script will be to initialize a string variable to store the account currency. This will be used to format the output of financial metrics.

```
string currency = " " + AccountInfoString(ACCOUNT_CURRENCY);
```

Before we can begin any calculations, we need to first calculate the _start time_ of the _current week_. This has been made simple as the _HistoryManager.ex5_ library contains the utility function _GetPeriodStart()_ which we will use to get the _datetime_ value of when the current week starts. To get this value, we just need to invoke the G _etPeriodStart()_ function with the _THIS\_WEEK_ parameter as the input. This will help us filter deals that occurred within the current week.

```
datetime thisWeekStartTime = GetPeriodStart(THIS_WEEK);
```

We declare an array to store the deal data and use the _GetDealsData()_ function to populate it with deals that occurred between the start of the current week and the current time ( _NOW_). Likewise, we filter deals for all _symbols_ with a _magic_ number of 0.

```
if(
      GetDealsData(
         dealsData,
         thisWeekStartTime, NOW,
         ALL_SYMBOLS, 0
      ) &&
      ArraySize(dealsData) > 0
   )
```

We initialize variables to store the total _gross profit, total swap,_ and _total commission_. These variables will be used to accumulate the respective values from the deal data.

```
double totalGrossProfit = 0.0,
             totalSwap = 0.0,
             totalCommission = 0.0;
```

We loop through each _deal_ in the _dealsData_ array. For each _deal_, we check if it is a closing deal ( _DEAL\_ENTRY\_OUT_). If it is, we add the _profit_, _swap_, and _commission_ to their respective total variables.

```
int totalDeals = ArraySize(dealsData);
for(int k = 0; k < totalDeals; k++)
  {
   if(dealsData[k].entry == DEAL_ENTRY_OUT)
     {
      totalGrossProfit += dealsData[k].profit;
      totalSwap += dealsData[k].swap;
      totalCommission += dealsData[k].commission;
     }
  }
```

We calculate the total expenses by summing the total _swap_ and _commission_. Furthermore, we then calculate the _total net profit_ by subtracting the _total expenses_ from the _total gross profit_.

```
double totalExpenses = totalSwap + totalCommission;
double totalNetProfit = totalGrossProfit - MathAbs(totalExpenses);
```

We use the _Print()_ function to output the results to the MetaTrader 5 log. This includes the total gross profit, total swaps, total commission, and total net profit for the current week. Additionally, we also use the _Comment()_ function to display the results directly in the chart window. This includes a detailed summary of the financial metrics for the current week.

![HistoryManager.ex5 - Get the Total Net Profit This Week](https://c.mql5.com/2/121/HistoryManager.ex5_Total_Net_Profit_This_Week.png)

Here is the full _OnStart()_ function with all the code segments in their proper sequence and in full.

```
void OnStart()
  {
//---
   string currency = " " + AccountInfoString(ACCOUNT_CURRENCY);
   datetime thisWeekStartTime = GetPeriodStart(THIS_WEEK);
   DealData dealsData[];
   if(
      GetDealsData(
         dealsData,
         thisWeekStartTime, NOW,
         ALL_SYMBOLS, 0
      ) &&
      ArraySize(dealsData) > 0
   )
     {
      double totalGrossProfit = 0.0,
             totalSwap = 0.0,
             totalCommission = 0.0;
      int totalDeals = ArraySize(dealsData);
      for(int k = 0; k < totalDeals; k++)
        {
         if(dealsData[k].entry == DEAL_ENTRY_OUT)
           {
            totalGrossProfit += dealsData[k].profit;
            totalSwap += dealsData[k].swap;
            totalCommission += dealsData[k].commission;
           }
        }
      double totalExpenses = totalSwap + totalCommission;
      double totalNetProfit = totalGrossProfit - MathAbs(totalExpenses);

      Print("-------------------------------------------------");
      Print(
         "Account No: ", AccountInfoInteger(ACCOUNT_LOGIN),
         " [ THIS WEEK'S NET PROFIT ]"
      );
      Print(
         "Total Gross Profit This Week: ",
         DoubleToString(totalGrossProfit, 2),
         " ", currency
      );
      Print(
         "Total Swaps This Week: ", DoubleToString(totalSwap, 2),
         " ", currency
      );
      Print(
         "Total Commission This Week: ", DoubleToString(totalCommission, 2),
         " ", currency
      );
      Print(
         "Total Net Profit This Week: ",
         DoubleToString(totalNetProfit, 2), " ",
         currency
      );

     //--
     Comment(
         "\n\n-----------------------------------------------------------------------------------------------------" +
         "-------------------------------------------------------------------" +
         "\n  HistoryManager.ex5 --- TOTAL NET PROFIT THIS WEEK ---" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "------------------------------------------------------" +
         "\n DATE = ( From: " + TimeToString(thisWeekStartTime) + ", to: " + TimeToString(NOW) + " )" +
         "\n Account No  = " + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) +
         "\n------------------------------------------------------" +
         "\n   -> Total Gross Profit =  " + DoubleToString(totalGrossProfit, 2) +
         currency +
         "\n   -> Total Swaps        =  " + DoubleToString(totalSwap, 2) +
         currency +
         "\n   -> Total Commission   =  " + DoubleToString(totalCommission, 2) +
         currency +
         "\n-----------------------------------------------------------------------------------------------------" +
         "------------------------------------------------------" +
         "\n   -> TOTAL NET PROFIT   =  " + DoubleToString(totalNetProfit, 2) +
         currency +
         "\n-----------------------------------------------------------------------------------------------------" +
         "------------------------------------------------------" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "-------------------------------------------------------------------"
      );
     }
  }
```

For the full script source code, please download the _**GetNetProfitThisWeek.mq5**_ source file attached at the end of this article.

### How To Calculate the Profit to Loss Ratio in Pips for a Specific Symbol or Expert Advisor

In this example, we will explore how to calculate the Profit to Loss Ratio in Pips for a specific trading symbol or an entire Expert Advisor using MQL5. This calculation is a crucial tool for evaluating the performance of a trading strategy, as it provides a clear and quantifiable metric to determine whether the strategy is profitable, break even, or unprofitable based on the pips gained or lost. The provided code is designed to analyze historical trade data, compute the ratio, and present the results in a user-friendly and easily interpretable format.

We will name the Expert Advisor _GetSymbolPipsProfitToLossRatio.mq5_, save it in the appropriate folder, and begin by including the _HistoryManager.mqh_ library as the first and most important line in our source file. Next, we will create an _enum_ to store the _symbol_ name and define the input parameters ( _symbol_ and _magic_ number). These parameters will allow the user to filter the data results based on their preferences: whether they want to analyze all closed positions for a specific symbol, closed positions for a specific symbol that includes a particular magic number, or all positions with a specific magic number, regardless of the symbol.

In the _OnInit()_ function, we will retrieve the account currency and determine the _symbol(s)_ to analyze based on the user's input. If the user selects _CURRENT\_CHART\_SYMBOL_, our Expert Advisor will focus on the _symbol_ of the _current chart_. If they select _ALL\_ACCOUNT\_SYMBOLS_, it will analyze all _symbols_ traded on the _account_.

We will use the _GetAllPositionsData()_ function to fetch historical trade data for the specified symbol(s) and magic number. This data will be stored in an array of _PositionData_ structures. Likewise, we will then iterate through the array to categorize trades as either profitable or loss-making. During this process, we will calculate the total pips gained from profitable trades ( _totalPipsProfit_) and the total pips lost from losing trades ( _totalPipsLoss_).

To calculate the _pip profit-to-loss ratio_, we will divide the total pips gained by the total pips lost. Futhermore, we will use the absolute value to ensure the ratio is always positive. If there are no losing trades ( _totalPipsLoss == 0_), the ratio will be _undefined_, and the Expert Advisor will provide a message indicating that the strategy has no losing trades. The Expert Advisor will interpret the ratio as follows:

- Ratio > 1.0: The strategy is profitable, as it gains more pips than it loses.
- Ratio == 1.0: The strategy breaks even in terms of pips.
- Ratio < 1.0: The strategy is unprofitable, as it loses more pips than it gains.

We will be using the _Comment()_ function to display the analysis results directly on the chart. The output will include: The symbol(s) analyzed, the magic number used (if any), the total number of trades analyzed, the number of profitable and loss-making trades, along with the total pips gained and lost, and the calculated Profit to Loss Ratio in pips and its interpretation.

![HistoryManager.ex5 Pips Profit to Loss Ratio EA](https://c.mql5.com/2/122/HistoryManager.ex5_Pips_Profit_to_Loss_Ratio_EA.png)

In the _OnDeinit()_ function, we will perform a simple clean-up by clearing the chart comments when the Expert Advisor is removed or deinitialized, ensuring a clean workspace. We will leave the _OnTick()_ function empty, as the analysis is performed only once during initialization. However, you can expand this function to perform real-time calculations or updates if needed.

Here are all the code segments presented in their correct sequence:

```
//--
#include <Wanateki/Toolkit/HistoryManager/HistoryManager.mqh>
//--
enum symbolName
  {
   CURRENT_CHART_SYMBOL,
   ALL_ACCOUNT_SYMBOLS,
  };
//--- input parameters
input ulong  magicNo = 0;          //Magic Number (0 to disable)
input symbolName getSymbolName = CURRENT_CHART_SYMBOL;
```

```
int OnInit()
  {
//---
   string currency = " " + AccountInfoString(ACCOUNT_CURRENCY);
   string symbol, printedSymbol;

   switch(getSymbolName)
     {
      case CURRENT_CHART_SYMBOL:
         symbol = _Symbol;
         break;
      case ALL_ACCOUNT_SYMBOLS:
         symbol = ALL_SYMBOLS;
         break;
      default:
         symbol = ALL_SYMBOLS;
     }
   printedSymbol = symbol;
   if(symbol == "")
      printedSymbol = "ALL_SYMBOLS";
//--
   int totalTrades = 0;
   int totalLossPositions = 0;
   int totalProfitPositions = 0;
   double totalPipsProfit = 0;
   double totalPipsLoss = 0;
   string interpretation;
   double pipsProfitToLossRatio = 0;
//--
   PositionData positionsData[];
   if(GetAllPositionsData(positionsData, symbol, magicNo) && ArraySize(positionsData) > 0)
     {
      totalTrades = ArraySize(positionsData);
      for(int r = 0; r < totalTrades; r++)
        {
         if(positionsData[r].profit > 0) // profitable trade
           {
            ++totalProfitPositions;
            totalPipsProfit += positionsData[r].pipProfit;
           }
         else  // loss trade
           {
            ++totalLossPositions;
            totalPipsLoss += positionsData[r].pipProfit;
           }
        }
      // Calculate the pip profit loss ratioInterpretation
      if(totalPipsLoss == 0)
        {
         interpretation = "Pips Profit-to-Loss Ratio: Undefined (Total pips loss is zero)." +
                          "The strategy has no losing trades.";
        }
      else
        {
         pipsProfitToLossRatio = fabs(totalPipsProfit / totalPipsLoss);

         switch(pipsProfitToLossRatio > 1.0 ? 1 : pipsProfitToLossRatio == 1.0 ? 0 : -1)
           {
            case 1:
               interpretation = "Pips Profit-to-Loss Ratio: " + DoubleToString(pipsProfitToLossRatio, 2) +
                                ". The strategy is profitable as it gains more pips than it loses.";
               break;
            case 0:
               interpretation = "Pips Profit-to-Loss Ratio: " + DoubleToString(pipsProfitToLossRatio, 2) +
                                ". The strategy breaks even in terms of pips.";
               break;
            case -1:
               interpretation = "Pips Profit-to-Loss Ratio: " + DoubleToString(pipsProfitToLossRatio, 2) +
                                ". The strategy is unprofitable as it loses more pips than it gains.";
               break;
           }
        }

      Comment(
         "\n\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n  HistoryManager.ex5 --- PIPS PROFIT TO LOSS RATIO ---" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n   -> Symbol   = " + printedSymbol +
         "\n   -> Magic No = " + IntegerToString(magicNo) +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n" + interpretation +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------" +
         "\n   -> Total Trades Analysed     = " + IntegerToString(totalTrades) +
         "\n   -> Total Profitable Trades   = " + IntegerToString(totalProfitPositions) +
         " ( " + DoubleToString(totalPipsProfit, 0) + " Pips )" +
         "\n   -> Total Loss Trades         = " + IntegerToString(totalLossPositions) +
         " ( " + DoubleToString(totalPipsLoss, 0) + " Pips )" +
         "\n   --------------------------------------------------------------------------" +
         "\n   -> PIPS PROFIT TO LOSS RATIO = " + DoubleToString(pipsProfitToLossRatio, 2) +
         "\n   --------------------------------------------------------------------------" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------"
      );
     }

//---
   return(INIT_SUCCEEDED);
  }
```

You can download the full _**GetSymbolPipsProfitToLossRatio.mq5**_ source file at the bottom of this article.

### How To Get the Total Cash Value of Account Deposits

In this section, we will create a straightforward MQL5 script named _GetTotalDeposits.mq5_ that leverages the _GetAllDealsData()_ function from our library to retrieve and analyze deal data to help you get a clear overview of your account's funding history. This is a valuable tool to have in your toolkit for when you algorithmically want to track the total funds deposited into your trading account, audit your account's funding history for record-keeping or tax purposes, or simply verify deposit transactions to ensure accuracy.

We will use the _GetAllDealsData()_ function to fetch all historical deal data for the account. This data will be stored in an array of _DealData_ structures. Next, we will initialize a variable to store the _total deposit_ value and iterate through the deal data. Identify deposit transactions by checking if the deal type is _DEAL\_TYPE\_BALANCE_. Sum up the profit values of these deals to calculate the total deposits.

After we have calculated and saved all the targeted data, we will print the total deposit value to the terminal and display it on the chart using the _Comment()_ function. This provides a user-friendly summary of the account's deposit history.

```
#include <Wanateki/Toolkit/HistoryManager/HistoryManager.mqh>
void OnStart()
  {
//---
// Find and list total deposited funds in the account
   DealData dealsData[];
   if(GetAllDealsData(dealsData) && ArraySize(dealsData) > 0)
     {
      double totalDeposits = 0.0;
      int totalDeals = ArraySize(dealsData);
      Print("");
      for(int k = 0; k < totalDeals; k++)
        {
         if(dealsData[k].type == DEAL_TYPE_BALANCE)
           {
            totalDeposits += dealsData[k].profit;
            Print(
               dealsData[k].profit, " ", AccountInfoString(ACCOUNT_CURRENCY),
               " --> Cash deposit on: ", dealsData[k].time
            );
           }
        }
      Print("-------------------------------------------------");
      Print(
         "Account No: ", AccountInfoInteger(ACCOUNT_LOGIN),
         " Total Cash Deposits: ", totalDeposits, " ",
         AccountInfoString(ACCOUNT_CURRENCY)
      );

      Comment(
         "\n\n-----------------------------------------------------------------------------------------------------" +
         "---------------------------------------------------------" +
         "\n  HistoryManager.ex5 --- TOTAL ACCOUNT DEPOSITS ---" +
         "\n-----------------------------------------------------------------------------------------------------" +
         "------------------------------------------------------" +
         "\n   -> Account No  = " + IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)) +
         "\n   -> Total Cash Deposits =  " + DoubleToString(totalDeposits, 2) +
         AccountInfoString(ACCOUNT_CURRENCY) +
         "\n-----------------------------------------------------------------------------------------------------" +
         "------------------------------------------------------"
      );
     }
  }
```

You can download the full _**GetTotalDeposits.mq5**_ source file at the bottom of this article.

### How To Create a Price-Data Driven Expert Advisor Powered by the History Manager EX5 Library

For the final example in this article, we will create a simple, yet powerful price-data-driven Expert Advisor named _PriceTrader\_EA_, powered by the _PositionsManager.ex5_ and _HistoryManager.ex5_ libraries. This Expert Advisor has significant profit-making potential, especially when used on a large account and after thorough back testing and optimization. You will notice that the source code for _PriceTrader\_EA_ is concise and efficient, thanks to the use of our pre-developed _PositionsManager.ex5_ and _HistoryManager.ex5_ libraries. These libraries allow us to minimize the code while maintaining reliability and consistency.

_PriceTrader\_EA_ is designed to make trading decisions based on price data and historical trade performance. It incorporates the following key features:

- **Dynamic Lot Sizing:** The Expert Advisor adjusts lot sizes based on the outcome of previous trades, doubling the lot size after a losing trade to recover losses ( _within predefined limits_).
- **Trade Direction Based on Price Action:** _PriceTrader\_EA_ opens trades in the direction of the prevailing trend, as determined by the price action on the _H1 timeframe_.
- **Risk Management**: Stop-loss ( _SL_) and take-profit ( _TP_) levels are calculated dynamically based on the current _spread_, ensuring adaptability to changing market conditions.

We will begin by including the _HistoryManager.mqh_ and _PositionsManager.mqh_ library header files in the header section of our source file.

```
#include <Wanateki/Toolkit/HistoryManager/HistoryManager.mqh>
#include <Wanateki/Toolkit/PositionsManager/PositionsManager.mqh>
```

Next, we will define the _input_ parameters for _PriceTrader\_EA_. These parameters allow us to configure the _magic_ number, _spread multipliers_ for _TP_ and _SL_, and the _maximum lot_ _size_ increase.

```
input ulong magicNo = 101010;
input int tpSpreadMulti = 70;
input int slSpreadMulti = 90;
input int maxLotIncrease = 1000;
```

We will initialize key variables to store information such as the current _spread, lot size,_ and the number of open positions. These variables will be used throughout the Expert Advisor’s logic.

```
bool eaJustLoaded = true;
double spread;
int spreadPips;
long minSLTP = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
long freezeLevel = SymbolInfoInteger(Symbol(), SYMBOL_TRADE_FREEZE_LEVEL);
double lotSize;
int sl, tp;
int totalOpenPositions, totalBuyPositionsOpen, totalSellPositionsOpen;
//--
PositionData lastClosedPositionInfo;
```

In the _OnInit()_ function, we will calculate the current _spread_ and initialize the _TP_ and _SL_ levels based on the _spread multipliers_. We will also play a sound to indicate that _PriceTrader\_EA_ has been successfully loaded.

```
int OnInit()
  {
//---
   spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
   spread = NormalizeDouble(spread, _Digits);
   spreadPips = int(spread / _Point);
   tp = spreadPips * tpSpreadMulti;
   sl = spreadPips * slSpreadMulti;
//--
   PlaySound("connect.wav");
//---
   return(INIT_SUCCEEDED);
  }
```

In the _OnDeinit()_ function, we will clear the chart comments and play a sound to indicate that _PriceTrader\_EA_ has been unloaded and removed from the chart.

```
void OnDeinit(const int reason)
  {
//---
   Comment("");
   PlaySound("disconnect.wav");
  }
```

The _OnTick()_ function contains the core logic of _PriceTrader\_EA_. Here’s what we will do in this function:

- Retrieve the current lot size and the number of open positions.
- Check if _PriceTrader\_EA_ has just been loaded and open an initial trade based on the H1 price action.
- Adjust the lot size and trade direction based on the outcome of the last closed trade.
- Open consecutive trades to capitalize on profitable trends or recover losses.

```
void OnTick()
  {
//---
   lotSize = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
//--
   totalOpenPositions = SymbolPositionsTotal(_Symbol, magicNo);
   totalBuyPositionsOpen = SymbolBuyPositionsTotal(_Symbol, magicNo);
   totalSellPositionsOpen = SymbolSellPositionsTotal(_Symbol, magicNo);

   if(eaJustLoaded && totalOpenPositions == 0)
     {
      //--
      GetLastClosedPositionData(lastClosedPositionInfo, _Symbol, magicNo);
      if(lastClosedPositionInfo.ticket > 0 && lastClosedPositionInfo.profit < 0)
        {
         if(lastClosedPositionInfo.volume * 2 < lotSize * maxLotIncrease)
            lotSize = lastClosedPositionInfo.volume * 2; // double lot size
        }
      //--
      if(iOpen(_Symbol, PERIOD_H1, 0) < iClose(_Symbol, PERIOD_H1, 0))
        {
         OpenBuyPosition(magicNo, _Symbol, lotSize, sl, tp, "Initial_Position");
        }
      else
        {
         OpenSellPosition(magicNo, _Symbol, lotSize, sl, tp, "Initial_Position");
        }
      if(totalOpenPositions > 0)
         eaJustLoaded = false;
     }
   else
     {
      eaJustLoaded = false;
     }

   if(totalOpenPositions == 0 && !eaJustLoaded)
     {
      if(GetLastClosedPositionData(lastClosedPositionInfo, _Symbol, magicNo))
        {
         if(lastClosedPositionInfo.profit > 0) // PROFITABLE TRADE
           {
            if(lastClosedPositionInfo.type == POSITION_TYPE_BUY)
              {
               OpenBuyPosition(magicNo, _Symbol, lotSize, sl, tp, "Consecutive Profit");
              }
            else  // SELL POSITION
              {
               OpenSellPosition(magicNo, _Symbol, lotSize, sl, tp, "Consecutive Profit");
              }
           }
         else   // LOSS TRADE
           {
            if(lastClosedPositionInfo.volume * 2 < lotSize * maxLotIncrease)
               lotSize = lastClosedPositionInfo.volume * 2; // double lot size
            //--
            if(lastClosedPositionInfo.type == POSITION_TYPE_BUY)
              {
               // Reverse trade direction
               OpenSellPosition(magicNo, _Symbol, lotSize, sl, tp, "Loss Recovery");
              }
            else  // SELL POSITION
              {
               OpenBuyPosition(magicNo, _Symbol, lotSize, sl, tp, "Loss Recovery");
              }
           }
        }
     }
```

_PriceTrader\_EA_ is ideal for you if you are the kind of trader who favors or requires a simple yet effective automated trend-following strategy with dynamic risk management. This EA is designed to recover losses quickly through adaptive lot sizing, ensuring that your trading strategy remains robust even during unfavorable market conditions. Additionally, you will notice that _PriceTrader\_EA_ is cleverly coded to make back testing on MetaTrader 5 as simple and efficient as possible. The user inputs for _lot size, stop loss,_ and _take profit_ are automatically adjusted based on the loaded _symbol_, eliminating the need for manual adjustments and ensuring optimal performance across different instruments.

You will find the complete **_PriceTrader\_EA._** mq5 source file at the bottom of this article, as well as the **_PositionsManager.ex5_** library.

### Back Testing the Price Trader Expert Advisor

Let us run a back test in the MetaTrader 5 strategy tester to see how this simple trading strategy performs over the last fourteen months.

Here are the settings we will apply in the strategy tester:

- Broker: Deriv

- Server: Deriv-Demo

- Symbol: Volatility 50 (1s) Index

- Timeframe: Daily

- Testing Period (Date): 1 year, 2 months (Jan 2024 to Feb 2025)

- Modeling: Every tick based on real ticks

- Deposit: 5,000 USD

- Leverage: 1:1000


![Wanateki PriceTrader_EA Backtest Settings](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_Settings.png)

Inputs settings:

![Wanateki PriceTrader_EA Backtest Inputs](https://c.mql5.com/2/122/Wanateki_PriceTrader_EA_Backtest_results_-_inputs.png)

Here are the back testing results for the _PriceTrader\_EA_:

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki-PriceTrader_EA-Backtest-results-report-1.png)

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_results_-_report_1_.png)

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_results_-_report_2.png)

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_results_-_report_2_.png)

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_results_-_report_3.png)

![Wanateki PriceTrader_EA Backtest Report](https://c.mql5.com/2/123/Wanateki_PriceTrader_EA_Backtest_results_-_report_3_.png)

Reviewing our back testing results, _PriceTrader\_EA_ delivered an impressive profit of over 129% return while maintaining a low equity drawdown of just 29%. This simple yet effective strategy demonstrates significant potential and can be further refined and optimized to achieve even better results. Since _PriceTrader\_EA_ dynamically adjusts its inputs—such as _lot size, stop loss,_ and _take profit_—based on the _symbol_ or asset it is trading, you can easily test it on a demo account. Simply load it onto a chart and observe its performance over a day or more to see if it generates profits, just as it did during our strategy tester evaluations. This flexibility makes it an excellent tool for both testing and live trading.

### Conclusion

As demonstrated in this final article of the series, the _HistoryManager.ex5_ Library is a powerful and efficient tool that simplifies the processing of trade histories in MetaTrader 5. With its comprehensive range of functions, this library enables you to effortlessly access and manage data related to deals, orders, positions, and pending orders—all through simple, one-line function calls. This streamlined approach saves time and effort, allowing you to focus on developing and optimizing your trading strategies.

Throughout this article, I have provided practical code examples to help you harness the full potential of the HistoryManager.ex5 Library. These examples, combined with the knowledge shared in this series, have equipped you with the tools and resources needed to algorithmically process any type of historical data generated from your trading activities in MetaTrader 5 using MQL5. As a parting gift to all the readers who have followed along, I have created _PriceTrader\_EA_, a basic yet effective Expert Advisor that showcases the practical application of a few of these concepts.

Thank you for joining me on this journey through MQL5 development. Your dedication to learning and exploring these tools is a testament to your commitment to mastering the art of algorithmic trading. As always, I wish you the very best in your quest to unravel the complexities of the markets and achieve success through your MQL5 developments. Happy coding, and may your strategies always thrive!

### Resources and Source Files

All the code referenced in this article is provided below for your convenience. The table included here outlines the accompanying EX5 libraries and source code files, making it easy for you to access, implement, and explore the examples discussed.

| File Name | Description |
| --- | --- |
| HistoryManager.ex5 | EX5 library designed to process and manage trade histories. |
| PositionsManager.ex5 | EX5 library for managing and processing positions and orders. |
| HistoryManager.mqh | Header file used to import data structures and prototype functions from the HistoryManager.ex5 library into your source files. |
| PositionsManager.mqh | Header file used to import prototype functions from the PositionsManager.ex5 library into your source files. |
| GetProfitFactor.mq5 | Expert Advisor that analyzes the performance of your trading strategies by calculating key metrics such as gross profit, gross loss, and the profit factor. |
| GetNetProfitThisWeek.mq5 | Script that calculates the net profit for the current week. |
| GetSymbolPipsProfitToLossRatio.mq5 | Expert Advisor that calculates the Profit to Loss Ratio in Pips for a specific trading symbol or an entire Expert Advisor. |
| GetTotalDeposits.mq5 | Script that retrieves and analyzes deal data to provide a clear overview of your account's funding or cash deposit history. |
| PriceTrader\_EA.mq5 | A price-data-driven Expert Advisor that uses trade history data to detect price direction and recover from losses. Powered by the PositionsManager.ex5 and HistoryManager.ex5 libraries. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17015.zip "Download all attachments in the single ZIP archive")

[HistoryManager.ex5](https://www.mql5.com/en/articles/download/17015/historymanager.ex5 "Download HistoryManager.ex5")(51.85 KB)

[PositionsManager.ex5](https://www.mql5.com/en/articles/download/17015/positionsmanager.ex5 "Download PositionsManager.ex5")(44.74 KB)

[HistoryManager.mqh](https://www.mql5.com/en/articles/download/17015/historymanager.mqh "Download HistoryManager.mqh")(17.75 KB)

[PositionsManager.mqh](https://www.mql5.com/en/articles/download/17015/positionsmanager.mqh "Download PositionsManager.mqh")(3.56 KB)

[GetProfitFactor.mq5](https://www.mql5.com/en/articles/download/17015/getprofitfactor.mq5 "Download GetProfitFactor.mq5")(8.07 KB)

[GetNetProfitThisWeek.mq5](https://www.mql5.com/en/articles/download/17015/getnetprofitthisweek.mq5 "Download GetNetProfitThisWeek.mq5")(4.16 KB)

[GetSymbolPipsProfitToLossRatio.mq5](https://www.mql5.com/en/articles/download/17015/getsymbolpipsprofittolossratio.mq5 "Download GetSymbolPipsProfitToLossRatio.mq5")(6.09 KB)

[GetTotalDeposits.mq5](https://www.mql5.com/en/articles/download/17015/gettotaldeposits.mq5 "Download GetTotalDeposits.mq5")(2.38 KB)

[PriceTrader\_EA.mq5](https://www.mql5.com/en/articles/download/17015/pricetrader_ea.mq5 "Download PriceTrader_EA.mq5")(5.12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Toolkit (Part 7): Expanding the History Management EX5 Library with the Last Canceled Pending Order Functions](https://www.mql5.com/en/articles/16906)
- [MQL5 Trading Toolkit (Part 6): Expanding the History Management EX5 Library with the Last Filled Pending Order Functions](https://www.mql5.com/en/articles/16742)
- [MQL5 Trading Toolkit (Part 5): Expanding the History Management EX5 Library with Position Functions](https://www.mql5.com/en/articles/16681)
- [MQL5 Trading Toolkit (Part 4): Developing a History Management EX5 Library](https://www.mql5.com/en/articles/16528)
- [MQL5 Trading Toolkit (Part 3): Developing a Pending Orders Management EX5 Library](https://www.mql5.com/en/articles/15888)
- [MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/482559)**
(3)


![hini](https://c.mql5.com/avatar/2024/3/65e98921-0708.jpg)

**[hini](https://www.mql5.com/en/users/hini)**
\|
5 Jul 2025 at 15:31

Hello author, thank you for your code. Have you tested its performance? How fast is it under a large [number of orders](https://www.mql5.com/en/docs/trading/orderstotal "MQL5 documentation: OrdersTotal function")?


![Wanateki Solutions LTD](https://c.mql5.com/avatar/2025/1/677afe52-ae53.png)

**[Kelvin Muturi Muigua](https://www.mql5.com/en/users/kelmut)**
\|
6 Jul 2025 at 23:03

Hello hini, thanks for your interest in the article. It's fast as it returns the queried results in milliseconds even when working with a large number of historical orders, but that depends on how fast your computer is.


![hini](https://c.mql5.com/avatar/2024/3/65e98921-0708.jpg)

**[hini](https://www.mql5.com/en/users/hini)**
\|
7 Jul 2025 at 00:18

**Kelvin Muturi Muigua [#](https://www.mql5.com/en/forum/482559#comment_57427765):**

Hello hini, thanks for your interest in the article. It's fast as it returns the queried results in milliseconds even when working with a large number of historical orders, but that depends on how fast your computer is.

thanks!


![Price Action Analysis Toolkit Development (Part 16): Introducing Quarters Theory (II) — Intrusion Detector EA](https://c.mql5.com/2/123/Price_Action_Analysis_Toolkit_Development_Part_16__V2___LOGO.png)[Price Action Analysis Toolkit Development (Part 16): Introducing Quarters Theory (II) — Intrusion Detector EA](https://www.mql5.com/en/articles/17321)

In our previous article, we introduced a simple script called "The Quarters Drawer." Building on that foundation, we are now taking the next step by creating a monitor Expert Advisor (EA) to track these quarters and provide oversight regarding potential market reactions at these levels. Join us as we explore the process of developing a zone detection tool in this article.

![William Gann methods (Part III): Does Astrology Work?](https://c.mql5.com/2/91/William_Ganns_Methods_Part_3__LOGO.png)[William Gann methods (Part III): Does Astrology Work?](https://www.mql5.com/en/articles/15625)

Do the positions of planets and stars affect financial markets? Let's arm ourselves with statistics and big data, and embark on an exciting journey into the world where stars and stock charts intersect.

![Neural Networks in Trading: State Space Models](https://c.mql5.com/2/88/logo-neuronetworks_in_trading_15546_388_3728.png)[Neural Networks in Trading: State Space Models](https://www.mql5.com/en/articles/15546)

A large number of the models we have reviewed so far are based on the Transformer architecture. However, they may be inefficient when dealing with long sequences. And in this article, we will get acquainted with an alternative direction of time series forecasting based on state space models.

![Multiple Symbol Analysis With Python And MQL5 (Part 3): Triangular Exchange Rates](https://c.mql5.com/2/122/Multiple_Symbol_Analysis_With_Python_And_MQL5_Part_3__LOGO.png)[Multiple Symbol Analysis With Python And MQL5 (Part 3): Triangular Exchange Rates](https://www.mql5.com/en/articles/17258)

Traders often face drawdowns from false signals, while waiting for confirmation can lead to missed opportunities. This article introduces a triangular trading strategy using Silver’s pricing in Dollars (XAGUSD) and Euros (XAGEUR), along with the EURUSD exchange rate, to filter out noise. By leveraging cross-market relationships, traders can uncover hidden sentiment and refine their entries in real time.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/17015&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049520199020948723)

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