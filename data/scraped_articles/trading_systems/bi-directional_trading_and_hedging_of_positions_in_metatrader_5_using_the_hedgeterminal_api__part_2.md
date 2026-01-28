---
title: Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal API, Part 2
url: https://www.mql5.com/en/articles/1316
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:51:03.538916
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1316&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062764624031426676)

MetaTrader 5 / Trading systems


### Table Of Contents

- [INTRODUCTION](https://www.mql5.com/en/articles/1316#intro)
- [CHAPTER 1. INTERACTION OF EXPERT ADVISORS WITH HEDGE TERMINAL API AND ITS PANEL](https://www.mql5.com/en/articles/1316#chapter1)

  - [1.1. Installation of HedgeTermianlAPI. The First Start of the Library](https://www.mql5.com/en/articles/1316#c1_1)

  - [1.2. Integration of Expert Advisors with the HedgeTerminal Panel](https://www.mql5.com/en/articles/1316#c1_2)

  - [1.3. General Principles of HedgeTerminalAPI](https://www.mql5.com/en/articles/1316#c1_3) [Operation](https://www.mql5.com/en/articles/1316#c1_3)

  - [1.4. Selecting Transactions](https://www.mql5.com/en/articles/1316#c1_4)

  - [1.5. Getting Error Codes Using GetHedgeError()](https://www.mql5.com/en/articles/1316#c1_5)

  - [1.6. Detailed Analysis of Trading and Identification of Errors Using TotalActionsTask() and GetActionResult()](https://www.mql5.com/en/articles/1316#c1_6)

  - [1.7. Tracking Task Execution](https://www.mql5.com/en/articles/1316#c1_7) [Status](https://www.mql5.com/en/articles/1316#c1_7)

  - [1.8. Modifying and Closing Bi-Directional Positions](https://www.mql5.com/en/articles/1316#c1_8)

  - [1.9. Setting HedgeTerminal Properties from an Expert Advisor](https://www.mql5.com/en/articles/1316#c1_9)

  - [1.10. Synchronous and Asynchronous Modes of Operation](https://www.mql5.com/en/articles/1316#c1_10)

  - [1.11. Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)

  - [1.12. The SendTradeRequest Function and the HedgeTradeRequest Structure through the Example of Chaos II EA](https://www.mql5.com/en/articles/1316#c1_12)

  - [1.13. On "Duplicate Symbols" and Virtualization by Broker](https://www.mql5.com/en/articles/1316#c1_13)

- [CHAPTER 2. HEDGE TERMINAL API MANUAL](https://www.mql5.com/en/articles/1316#chapter2)

  - [2.1. Transaction Selection Functions](https://www.mql5.com/en/articles/1316#c2_1)

  - [Function TransactionsTotal()](https://www.mql5.com/en/articles/1316#TransactionsTotal)

  - [Function TransactionType()](https://www.mql5.com/en/articles/1316#TransactionType)

  - [Function TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect)

  - [Function HedgeOrderSelect()](https://www.mql5.com/en/articles/1316#HedgeOrderSelect)

  - [Function HedgeDealSelect()](https://www.mql5.com/en/articles/1316#HedgeDealSelect)

  - [2.2. Functions for Getting Properties of a Selected Transaction](https://www.mql5.com/en/articles/1316#c2_2)

  - [Function HedgePositionGetInteger()](https://www.mql5.com/en/articles/1316#HedgePositionGetInteger)

  - [Function HedgePositionGetDouble()](https://www.mql5.com/en/articles/1316#HedgePositionGetDouble)

  - [Function HedgePositionGetString()](https://www.mql5.com/en/articles/1316#HedgePositionGetString)

  - [Function HedgeOrderGetInteger()](https://www.mql5.com/en/articles/1316#HedgeOrderGetInteger)

  - [Function HedgeOrderGetDouble()](https://www.mql5.com/en/articles/1316#HedgeOrderGetDouble)

  - [Function HedgeDealGetInteger()](https://www.mql5.com/en/articles/1316#HedgeDealGetInteger)

  - [Function HedgeDealGetDouble()](https://www.mql5.com/en/articles/1316#HedgeDealGetDouble)

  - [2.3. Functions for Setting and Getting HedgeTerminal Properties from Expert Advisors](https://www.mql5.com/en/articles/1316#c2_3)

  - [Function HedgePropertySetInteger()](https://www.mql5.com/en/articles/1316#HedgePropertySetInteger)

  - [Function HedgePropertyGetInteger()](https://www.mql5.com/en/articles/1316#HedgePropertyGetInteger)

  - [2.4. Functions for Getting and Handling Error Codes](https://www.mql5.com/en/articles/1316#c2_4)

  - [Function GetHedgeError()](https://www.mql5.com/en/articles/1316#GetHedgeError)

  - [Function ResetHedgeError()](https://www.mql5.com/en/articles/1316#ResetHedgeError)

  - [Function TotalActionsTask()](https://www.mql5.com/en/articles/1316#TotalActionsTask)

  - [Function GetActionResult()](https://www.mql5.com/en/articles/1316#GetActionResult)

  - [2.5. Trading](https://www.mql5.com/en/articles/1316#c2_5)

  - [Function SendTradeRequest()](https://www.mql5.com/en/articles/1316#SendTradeRequest)

  - [Trade Request Structure HedgeTradeRequest](https://www.mql5.com/en/articles/1316#HedgeTradeRequest)

  - [2.6. Enumerations for Working with Transaction Selection Functions](https://www.mql5.com/en/articles/1316#c2_6)

  - [ENUM\_TRANS\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_TRANS_TYPE)

  - [ENUM\_MODE\_SELECT](https://www.mql5.com/en/articles/1316#ENUM_MODE_SELECT)

  - [ENUM\_MODE\_TRADES](https://www.mql5.com/en/articles/1316#ENUM_MODE_TRADES)

  - [2.7. Enumerations for Working with the Functions that Get Transaction Properties](https://www.mql5.com/en/articles/1316#c2_7)

  - [ENUM\_TRANS\_DIRECTION](https://www.mql5.com/en/articles/1316#ENUM_TRANS_DIRECTION)

  - [ENUM\_HEDGE\_POSITION\_STATUS](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_STATUS)

  - [ENUM\_HEDGE\_POSITION\_STATE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_STATE)

  - [ENUM\_HEDGE\_POSITION\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_INTEGER)

  - [ENUM\_HEDGE\_POSITION\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_DOUBLE)

  - [ENUM\_HEDGE\_POSITION\_PROP\_STRING](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_STRING)

  - [ENUM\_HEDGE\_ORDER\_STATUS](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_STATUS)

  - [ENUM\_HEDGE\_ORDER\_SELECTED\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_SELECTED_TYPE)

  - [ENUM\_HEDGE\_ORDER\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER)

  - [ENUM\_HEDGE\_ORDER\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_DOUBLE)

  - [ENUM\_HEDGE\_DEAL\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_INTEGER)

  - [ENUM\_HEDGE\_DEAL\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_DOUBLE)

  - [2.8. Enumerations for Setting and Getting HedgeTerminal Properties](https://www.mql5.com/en/articles/1316#c2_8)

  - [ENUM\_HEDGE\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_INTEGER)

  - [2.9. Enumerations for Working with Error Codes Handling Functions](https://www.mql5.com/en/articles/1316#c2_9)

  - [ENUM\_TASK\_STATUS](https://www.mql5.com/en/articles/1316#ENUM_TASK_STATUS)

  - [ENUM\_HEDGE\_ERR](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ERR)

  - [ENUM\_TARGET\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_TARGET_TYPE)

  - [2.10. Enumerations for Working with Trade Requests](https://www.mql5.com/en/articles/1316#c2_10)

  - [ENUM\_REQUEST\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_REQUEST_TYPE)

  - [ENUM\_CLOSE\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_CLOSE_TYPE)

- [CHAPTER 3. THE FUNDAMENTALS OF ASYNCHRONOUS TRADING](https://www.mql5.com/en/articles/1316#chapter3)

  - [3.1. Organization and Scheme of Sending a Synchronous Trading Order](https://www.mql5.com/en/articles/1316#c3_1)

  - [3.2. Organization and Scheme of Sending an Asynchronous Trading Order](https://www.mql5.com/en/articles/1316#c3_2)

  - [3.3. Asynchronous Order Execution Speed](https://www.mql5.com/en/articles/1316#c3_3)

- [CHAPTER 4. THE FUNDAMENTALS OF MULTI-THREADED PROGRAMMING IN THE METATRADER 5 IDE](https://www.mql5.com/en/articles/1316#chapter4)

  - [4.1. Multi-Threaded Programming through the Example of Quote Collector UnitedExchangeQuotes](https://www.mql5.com/en/articles/1316#c4_1)

  - [4.2. Use of Multi-Threaded Interaction between Expert Advisors](https://www.mql5.com/en/articles/1316#c4_2)

- [ATTACHMENT DESCRIPTION](https://www.mql5.com/en/articles/1316#description)
- [CONCLUSION](https://www.mql5.com/en/articles/1316#summary)

### Introduction

This article is a continuation of the first part " [Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://www.mql5.com/en/articles/1297)". In the second part, we will discuss integration of Expert Advisors and other MQL5 programs with the HedgeTerminalAPI library. Read this article to learn how to work with the library. It will help you create bi-directional trading Expert Advisors while still working in a comfortable and simple trading environment.

In addition to the library description, the article touches on the fundamentals of asynchronous trading and multi-threaded programming. These descriptions are given in the third and fourth sections of this article. Therefore, this material will be useful for the traders who are not interested in bi-directional trading, but who would like to find out something new about asynchronous and multi-threaded programming.

The material presented below is intended for experienced algorithmic traders who know the MQL5 programming language. If you don't know MQL5, please read the first part of the article, which contains simple diagrams and drawings explaining the general principle of the library and the HedgeTerminal panel.

### Chapter 1. Interaction of Expert Advisors with HedgeTerminal and its panel

**1.1. Installation of HedgeTermianlAPI. The First Start of the Library**

The process of HedgeTerminalAPI installation differs from the installation of the HT visual panel, since the library can't run alone in MetaTrader 5. Instead, you will need to develop a special Expert Advisor to call the HedgeTerminalInstall() function from the library. This function will set a special header file Prototypes.mqh describing the functions available in HedgeTerminalAPI.

A library is installed on a computer in three steps:

**_Step 1._** Download the HedgeTerminalAPI library on your computer. Library location relative to your terminal: \\MQL5\\Experts\\Market\\HedgeTerminalApi.ex5.

_**Step 2.**_ Create a new Expert Advisor in the MQL5 Wizard using a standard template. The MQL5 Wizard generates the following source code:

```
//+------------------------------------------------------------------+
//|                                   InstallHedgeTerminalExpert.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//---
   return(INIT_SUCCEEDED);
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

**_Step 3._** You will need only one function of the resulting Expert Advisor - OnInit(), and the export directive describing the special installer function HedgeTerminalInstall() exported by the HedgeTerminalApi library. Run this function right in the OnInit() function. The source code marked in yellow performs these operations:

```
//+------------------------------------------------------------------+
//|                                   InstallHedgeTerminalExpert.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"

#import "HedgeTerminalAPI.ex5"
   void HedgeTerminalInstall(void);
#import

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   HedgeTerminalInstall();
   ExpertRemove();
//---
   return(INIT_SUCCEEDED);
  }
```

**_Step 4._** Your further actions depend on whether you have purchased the library. If you have bought it, you can run the Expert Advisor in real-time directly on the chart. This will start the standard installer of the whole product line of HedgeTerminal. You can easily complete this step by following the instructions described in sections 2.1 and 2.2 of the article " [Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://www.mql5.com/en/articles/1297)". The installation wizard installs all the required files including the header file and the test Expert Advisor on your computer.

If you have not purchased the library and only wish to test it, then EA operation in real time will not be available to you, but you can test API by running the EA in the strategy tester. The installer will not run in this case. In the test mode, HedgeTermianalAPI works in a single-user mode, so it does not need the files installed in the normal mode. It means you do not need to configure anything else.

As soon as EA testing is done, folder \HedgeTerminal is created in the common folder of the terminal. The normal path to the common directory of the MetaTrader terminals is c:\\Users\\<Username>\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\HedgeTerminal\, where <Username> is the name of your current computer account. The \\HedgeTerminal folder already contains files \\MQL5\\Include\\Prototypes.mqh and \\MQL5\\Experts\\Chaos2.mq5. Copy these files into the same directories of your terminal: file Prototypes.mqh to \\MetaTrader5\\MQL5\\Include, and file Chaos2.mq5 to \\MetaTrader5\\MQL5\\Experts.

File Prototypes.mqh is a header file containing description of the functions exported from the HedgeTerminalAPI library. Their purpose and descriptions are contained in comments to them.

File Chaos2.mq5 contains a sample EA described in section ["The SendTradeRequest Function and the HedgeTradeRequest Structure through the Example of Chaos II EA"](https://www.mql5.com/en/articles/1316#c1_12). This way you can visually understand how HedgeTerminalAPI works and how to develop an Expert Advisor utilizing the virtualization technology of HedgeTerminal.

The copied files are available for your EAs. So you only need to include the header file in the Expert Advisor source code to start using the library. Here is an example:

```
#include <Prototypes.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   int transTotal = TransactionsTotal();
   printf((string)transTotal);
  }
```

For example, the above code gets the total number of active positions and displays the number in the "Experts" tab of the MetaTrader 5 terminal.

_It is important to understand that HedgeTerminal is actually initialized at the first call of one of its functions. This initialization is called [lazy](https://en.wikipedia.org/wiki/Lazy_initialization "https://en.wikipedia.org/wiki/Lazy_initialization"). Therefore, the first call of one of its functions can take a long time. If you want a fast response during the first run, you must initialize HT in advance, for example, you can call the TransactionTotal() function in the block of OnInit()._

With lazy initialization you can omit the explicit initialization of the Expert Advisor. This greatly simplifies working with HedgeTerminal and makes it unnecessary to pre-configure it.

**1.2. Integration of Expert Advisors with the HedgeTerminal Panel**

If you have the HedgeTerminal visual panel and the full-featured version of the library, which can be run in real time, you can integrate your Expert Advisors with the panel so that all trade operations performed by them will also appear in the panel. In general, the integration is not visible. If you use HedgeTermianalAPI functions, actions performed by the robots are automatically displayed on the panel. However, you can expand the visuality by indicating the EA name in each committed transaction. You can do it by uncommenting the below line in the file Settings.xml:

```
<Column ID="Magic" Name="Magic" Width="100"/>
```

This tag is in sections <Active-Position> and <History-Position>.

Now, comments are removed, and the tags are included into processing. After panel restart, new column of _"Magic"_ will appear in the tables of active and history positions. The column contains the magic number of the Expert Advisor, to which the position belongs.

If you want to show the EA name instead of its magic number, add the name to the alias file ExpertAliases.xml. For example, an EA's magic number is _123847_, and you want to displays its name, like _"ExPro 1.1"_, add the following tag to the file:

```
<Expert Magic="123847" Name="ExPro 1.1"></Expert>
```

If it's done correctly, the EA name will be displayed instead of its magic in the appropriate column:

![Fig. 1.  Displaying EA name instead of Magic](https://c.mql5.com/2/12/ExPro.png)

Fig. 1. Displaying EA name instead of Magic

Note that the panel and Expert Advisors communicate in real time. This means that if you close an EA's position directly on the panel, the EA will know about this with the next call of the TransactionsTotal() function. And vice versa: after the Expert Advisor closes its position, it immediately disappears from the active positions tab.

**1.3. General Principles of HedgeTerminalAPI** **Operation**

In addition to bi-directional positions, HedgeTerminal works also with other trading types, such as pending orders, deals and broker's operations. HedgeTerminal treats all these types as a single group of _**transactions**_. A deal, a pending order, a be-directional position - all of them are transactions. However, a transaction cannot exist alone. In terms of object-oriented programming, a transaction can be introduced as an abstract base class, from which all possible trading entities, such as deals and be-directional positions are inherited. In this regard all functions of HedgeTerminalAPI can be divided into several groups:

1. Transaction search and selection functions. The common signature of the functions and the way they work almost completely coincide with the functions [OrderSend()](https://docs.mql4.com/en/trading/ordersend "https://docs.mql4.com/en/trading/ordersend") and [OrderSelect()](https://docs.mql4.com/en/trading/orderselect "https://docs.mql4.com/en/trading/orderselect") in MetaTrader 4;
2. Functions for getting properties of a selected transaction. Every transaction has a specific set of properties and specific functions for property selection. The common signature of the functions and the way they work resemble MetaTrader 5 system functions in how they access positions, deals and orders (such as [OrderGetDouble()](https://www.mql5.com/en/docs/trading/ordergetdouble) or [HistoryDealGetInteger()](https://www.mql5.com/en/docs/trading/historydealgetinteger));
3. HedgeTerminalAPI uses only one trading function: **SendTradeRequest()**. This function allows closing a bi-directional position or part of it. The same function is used for modifying stop loss, take profit or the outgoing comment. Working with the function is similar to [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) in MetaTrader 5;
4. The function for getting common errors **GetHedgeError()**, functions for detailed analysis of HedgeTerminal trading actions: **TotalActionsTask()** and **GetActionResult()**. Also used for error detection. There are no analogues in MetaTrader 4 or MetaTrader 5.

Working with almost all the functions is similar to using MetaTrader 4 and MetaTrader 5 system functions. As a rule, the function input is some identifier (enumeration value), and the function returns a value that corresponds to it.

Specific enumerations are available for each function. The common call signature is the following:

```
<value> = Function(<identifier>);
```

Let's consider an example of getting a unique position identifier. This is how it looks line in MetaTrader 5:

```
ulong id = PositionGetInteger(POSITION_IDENTIFIER);
```

In HedgeTerminal, receiving a unique identifier of a bi-directional position is as follows:

```
ulong id = HedgePositionGetInteger(HEDGE_POSITION_ENTRY_ORDER_ID)
```

The general principles of the functions are the same. Only types of enumerations differ.

**1.4. Selecting Transactions**

Selecting a transaction is going through the list of transactions, which is similar to search for orders in MetaTrader 4. However, in MetaTrader 4 only orders are searched for, while in HedgeTerminal anything can be found as a transaction - such as a pending order or a hedging position. Therefore, each transaction should first be selected using the TransactionSelect() function, and then its type should be identified through TransactionType().

Two lists of transactions are available to date: active and history transactions. The list to be applied is defined based on the ENUM\_MODE\_TRADES modifier. It is similar to the MODE\_TRADES modifier in MetaTrader 4.

Transaction search and selection algorithm is as follows:

```
1: for(int i=TransactionsTotal(MODE_TRADES)-1; i>=0; i--)
2:     {
3:      if(!TransactionSelect(i,SELECT_BY_POS,MODE_TRADES))continue;
4:      if(TransactionType()!=TRANS_HEDGE_POSITION)continue;
5:      if(HedgePositionGetInteger(HEDGE_POSITION_MAGIC) != Magic)continue;
6:      if(HedgePositionGetString(HEDGE_POSITION_SYMBOL) != Symbol())continue;
7:      if(HedgePositionGetInteger(HEDGE_POSITION_STATE) == POSITION_STATE_FROZEN)continue;
8:      ulong id = HedgePositionGetInteger(HEDGE_POSITION_ENTRY_ORDER_ID)
9:     }
```

The code loops through the list of active transactions in the cycle for (line 1). Before you proceed with the transaction, select it using TransactionSelect() (line 3). Only bi-directional positions are selected from these transactions (line 4). If the magic number of the position and its symbol do not match the magic number of the current EA and the symbol it is running on, HT moves on to the next position (lines 5 and 6). Then it defines the unique position identifier (line 8).

Special attention should be paid to line 7. The selected positions should be checked in terms of modification possibility. If the position is already in the process of modification, it cannot be changed in the current thread, although you can get one of its properties. If the position is locked, better wait until it's released to access its properties or retry to modify it. Property HEDGE\_POSITION\_STATE is used to find out whether position modification is possible.

The POSITION\_STATE\_FROZEN modifier denotes that the position is "frozen" and cannot be changed. The POSITION\_STATE\_ACTIVE modifier shows that a position is active and can be changed. These modifiers are listed in the ENUM\_HEDGE\_POSITION\_STATE enumeration, which is documented in the appropriate [section](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_STATE).

If a search through historical transactions is needed, the MODE\_TRADES modifier in functions TransactionTotal() and TransactionSelect() must be replaced by MODE\_HISTORY.

One transaction in HedgeTerminal can be nested into another. This is very much different from the concept of MetaTrader 5 where there is no nesting. For example, the historical bi-directional position in HedgeTerminal consists of two orders, each of which includes an arbitrary set of deals. Nesting can be represented as follows:

![Fig. 2. Nested transactions](https://c.mql5.com/2/12/l4l2hfok2na_ik4v39uvr3.png)

Fig. 2. Nested transactions

Nesting of transactions is clearly seen in the HedgeTerminal visual panel.

The below screenshot shows the details of a position of MagicEx 1.3:

![Fig. 3. Nested transactions in the HedgeTerminal panel](https://c.mql5.com/2/12/fnseib_l_5gyqd1.png)

Fig. 3. Nested transactions in the HedgeTerminal panel

You can access the properties of a particular order or a deal in the bi-directional position.

To do this:

1. Select a historical transaction and make sure that it is a bi-directional position;
2. Select one of the orders of this position using HedgeOrderSelect();
3. Get one of the properties of the selected order: the number of deals that it contains;
4. Select one of the deals belonging to the order by searching through all deals;
5. Get the required deal property.

Note that after the transaction has been selected, its specific properties become available to it. For example, if the transaction is an order, then after selection through HedgeOrderSelect(), you can find the number of deals for it (HedgeOrderGetInter(HEDGE\_ORDER\_DEALS\_TOTAL)) or the weighted average entry price (HedgeDealGetDouble(HEDGE\_DEAL\_PRICE\_EXECUTED)).

Let's find out the price of deal _#1197610_, which is marked red on the screenshot. This deal is part of the bi-directional position of the MagicEx 1.3 EA.

Through the below code, the EA can access its position and this deal:

```
#include <Prototypes.mqh>

ulong Magic=5760655; // MagicEx 1.3.

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   for(int i=TransactionsTotal(MODE_HISTORY)-1; i>=0; i--)
    {
      if(!TransactionSelect(i,SELECT_BY_POS,MODE_HISTORY))continue;        // Select transaction #i;
      if(TransactionType()!=TRANS_HEDGE_POSITION)continue;                 // If transaction is not position - continue;
      if(HedgePositionGetInteger(HEDGE_POSITION_MAGIC) != Magic)continue;  // If position is not main - continue;
      ulong id = HedgePositionGetInteger(HEDGE_POSITION_ENTRY_ORDER_ID);   // Get id for closed order;
      if(id!=5917888)continue;                                             // If id of position != 5917888 - continue;
      printf("1: -> Select position #"+(string)id);                        // Print position id;
      if(!HedgeOrderSelect(ORDER_SELECTED_CLOSED))continue;                // Select closed order or continue;
      ulong order_id = HedgeOrderGetInteger(HEDGE_ORDER_ID);               // Get id closed order;
      printf("2: ----> Select order #" + (string)order_id);                // Print id closed order;
      int deals_total = (int)HedgeOrderGetInteger(HEDGE_ORDER_DEALS_TOTAL);// Get deals total in selected order;
      for(int deal_index = deals_total-1; deal_index >= 0; deal_index--)   // Search deal #1197610...
        {
         if(!HedgeDealSelect(deal_index))continue;                         // Select deal by index or continue;
         ulong deal_id = HedgeDealGetInteger(HEDGE_DEAL_ID);               // Get id for current deal;
         if(deal_id != 1197610)continue;                                   // Select deal #1197610;
         double price = HedgeDealGetDouble(HEDGE_DEAL_PRICE_EXECUTED);     // Get price executed;
         printf("3: --------> Select deal #"+(string)deal_id+              // Print price excecuted;
              ". Executed price = "+DoubleToString(price,0));
        }
     }
  }
```

After code execution, the following entry will be created in the Experts tab of the MetaTrader 5 terminal:

```
2014.10.21 14:46:37.545 MagicEx1.3 (VTBR-12.14,D1)      3: --------> Select deal #1197610. Executed price = 4735
2014.10.21 14:46:37.545 MagicEx1.3 (VTBR-12.14,D1)      2: ----> Select order #6389111
2014.10.21 14:46:37.545 MagicEx1.3 (VTBR-12.14,D1)      1: -> Select position #5917888
```

The EA first selects position #5917888, and then selects order #6389111 inside the position. Once the order is selected, the EA starts searching for the deal number 1197610. When the deal is found, the EA gets its execution price and adds the price in the journal.

**1.5. How to Get Error Codes Using GetHedgeError()**

Errors and unforeseen situations may occur while working with the HedgeTerminal environment. Error getting and analyzing functions are used in these cases.

The simplest case when you get an error is when you forget to select a transaction using the TransactionSelect() function. The TransactionType() function will return modifier TRANS\_NOT\_DEFINED in this case.

To understand where the problem lies, we need to get the modifier of the last error. The modifier will tell us that now transaction has been selected. The following code does this:

```
for(int i=TransactionsTotal(MODE_HISTORY)-1; i>=0; i--)
  {
   //if(!TransactionSelect(i,SELECT_BY_POS,MODE_HISTORY))continue;        // forgot to select;
   ENUM_TRANS_TYPE type = TransactionType();
   if(type == TRANS_NOT_DEFINED)
   {
      ENUM_HEDGE_ERR error = GetHedgeError();
      printf("Error, transaction type not defined. Reason: " + EnumToString(error));
   }
  }
```

This is the resulting message:

```
Error, transaction type not defined. Reason: HEDGE_ERR_TRANS_NOTSELECTED
```

The error ID suggests that we have forgotten to select a transaction before trying to get its type.

All possible errors are listed in the ENUM\_HEDGE\_ERR structure.

**1.6. Detailed Analysis of Trading and Identification of Errors Using TotalActionsTask() and GetActionResult()**

In addition to the errors occurring in the process of working with the HedgeTerminal environment, trade errors may occur as a result of SendTradeRequest() call. These types of errors are more difficult to deal with. One task performed by SendTradeRequest() can contain multiple trading activities (subtasks). For example, to change the outgoing comment in an active position protected by a stop loss level, you must make two trading actions:

1. Cancel the pending stop order representing the stop level;
2. Place a new pending stop order with a new comment in the place of the previous order.


If the new stop order triggers, then its comment will be displayed as a comment closing the position, which is a correct way.

However, the task can be executed in part. Suppose, the pending order is successfully canceled, but placing new order fails for whatever reason. In this case, the position will be left without the stop loss level. In order to be able to handle this error, the EA will need to call a special task log and search in it to find the subtask that failed.

This is done using two functions: **TotalActionsTask()** returns the total number of trading actions (subtasks) within this task; and **GetActionResult()** accepts the subtasks index and returns its type and its execution result. Since all trading operations are performed using standard MetaTrader 5 tools, the result of their performance corresponds to the return code of the trade server.

In general, the algorithm of search for the failure reason is as follows:

1. Getting the total number of sub-tasks in the task using TotalActionsTask();
2. Searching through all subtasks in the _for_ loop. Determining the type of each subtask and its result.

Suppose, the stop order with a new comment could not be placed because the order execution price was too close to the current price level.

The below example code shows how the EA could find the reason for this failure:

```
#include <Prototypes.mqh>

ulong Magic=5760655; // MagicEx 1.3.

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//detect active position
   for(int i=TransactionsTotal(MODE_HISTORY)-1; i>=0; i--)
     {
      if(!TransactionSelect(i,SELECT_BY_POS,MODE_HISTORY))continue;
      ENUM_TRANS_TYPE type=TransactionType();
      if(type==TRANS_NOT_DEFINED)
        {
         ENUM_HEDGE_ERR error=GetHedgeError();
         printf("Error, transaction not defined. Reason: "+EnumToString(error));
        }
      if(TransactionType()!=TRANS_HEDGE_POSITION)continue;
      if(HedgePositionGetInteger(HEDGE_POSITION_MAGIC) != Magic)continue;
      if(HedgePositionGetString(HEDGE_POSITION_SYMBOL) != Symbol())continue;
      HedgeTradeRequest request;
      request.action=REQUEST_MODIFY_COMMENT;
      request.exit_comment="My new comment";
      if(!SendTradeRequest(request)) // Is error?
        {
         for(uint action=0; action < TotalActionsTask(); action++)
           {
            ENUM_TARGET_TYPE typeAction;
            int retcode=0;
            GetActionResult(action, typeAction, retcode);
            printf("Action#" + (string)action + ": " + EnumToString(type) +(string)retcode);
           }
        }
     }
  }
```

The following message will appear after the code execution:

```
Action #0 TARGET_DELETE_PENDING_ORDER 10009 (TRADE_RETCODE_PLACED)
Action #1 TARGET_SET_PENDING_ORDER 10015 (TRADE_RETCODE_INVALID_PRICE)
```

By comparing the numbers with standard modifiers of trade server return codes, we find out that the pending order was successfully removed, but placing of a new order failed. The trade server returned an error 10015 (incorrect price), which may mean that the current price is too close to the stop level.

Knowing this, the EA can take control over the stop levels. To do so, the EA will only have to close this position using the same SendTradeRequest() function.

**1.7. Tracking the Status of Trade Task Execution**

Every trading task can consist of any number of subtasks that should be performed sequentially.

In the asynchronous mode, one task can be performed in several passes of the code. There may also be cases when the task can "freeze". Therefore EA's control over task execution is required. When calling the HedgePositionGetInteger() function with the HEDGE\_POSITION\_TASK\_STATUS modifier, it returns the ENUM\_TASK\_STATUS type enumeration containing the status of the current position task.

For example, if something goes wrong after sending an order to close a position, due to which the position is not closed, then you need to get the status of the task.

The following example shows the code that an asynchronous Expert Advisor can execute to analyze the status of the task for the position:

```
ENUM_TASK_STATUS status=HedgePositionGetInteger(HEDGE_POSITION_TASK_STATUS);
switch(status)
  {
   case TASK_STATUS_COMPLETE:
      printf("Task complete!");
      break;
   case TASK_STATUS_EXECUTING:
      printf("Task executing. Waiting...");
      Sleep(200);
      break;
   case TASK_STATUS_FAILED:
      printf("Filed executing task. Print logs...");
      for(int i=0; i<TotalActionsTask(); i++)
        {
         ENUM_TARGET_TYPE type;
         uint retcode;
         GetActionResult(i,type,retcode);
         printf("#"+i+" "+EnumToString(type)+" "+retcode);
        }
      break;
   case TASK_STATUS_WAITING:
      printf("task will soon start.");
      break;
  }
```

Note that execution of some complex tasks requires multiple iterations.

In the asynchronous mode, coming events that signal changes in trading environment start a new iteration. Thus, all the iterations are performed without delay, one after another, following responses received from the trading server. The task execution differs in the synchronous mode.

The synchronous method uses the _**synchronous operation emulator**_, due to which users can perform even composite tasks in a single pass. The emulator uses time lags. For example, after execution of a subtask starts, the emulator does not return the execution thread to the EA. Instead, it waits for some time expecting the trading environment to change. After that, it rereads the trading environment again. If it understands that the subtask has been successfully completed, starts the next subtasks.

This process somewhat reduces the overall performance, as it takes some time to wait. But it turns execution of even complex tasks into quite a simple sequential operation performed in a single function call. Therefore, you almost never need to analyze the task execution log in the synchronous method.

**1.8. How to Modify and Close Bi-Directional Positions**

Bi-directional positions are modified and closed using the **SendTradeRequest()** function. Only three options can be applied to an active position:

1. A position can be fully or partially closed;
2. Position stop loss and take profit can be modified;
3. The outgoing comment of a position can be changed.

Historical position cannot be change. Similar to the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function in MetaTrader 5, SendTradeRequest() uses a pre-compiled query in the form of a trading structure **HedgeTraderRequest**. Read the documentation for further details on the SendTradeRequest() function and the HedgeTraderRequest structure. The example showing position modification and closure is available in the section on Chaos II EA.

**1.9. How to Set HedgeTerminal Properties from an Expert Advisor**

HedgeTerminal possesses a set of properties, such as refresh frequency, the number of seconds to wait for a response from the server and others.

All of these properties are defined in Settings.xml. When an EA is running in real time, the library reads properties from the file and sets appropriate internal parameters. When the EA is tested on a chart, file Settings.xml is not used. However, in some situations you may need to individually modify the EA properties regardless of whether it is running on a chart or in the strategy tester.

This is done through the special set of functions _**HedgePropertySet…**_ The current version features only one prototype from this set:

```
enum ENUM_HEDGE_PROP_INTEGER
{
   HEDGE_PROP_TIMEOUT,
};

bool HedgePropertySetInteger(ENUM_HEDGE_PROP_INTEGER property, int value)
```

For example, to set the timeout for the library to wait for a server response, write the following:

```
bool res = HedgePropertySetInteger(HEDGE_PROP_TIMEOUT, 30);
```

If the server response is not received within 30 seconds after you send an asynchronous request, the locked position will be released.

**1.10. Synchronous and Asynchronous Modes of Operation**

HedgeTerminal and its API perform trading activities completely asynchronously.

However, this mode requires more complex logic of EAs. To hide this complexity, HedgeTerminalAPI includes a special emulator of synchronous operation, allowing EAs developed in a conventional synchronous method to communicate with asynchronous algorithms of HedgeTerminalAPI. This interaction is revealed at the time of bi-directional position modification and closure through SendTradeRequest(). This function allows executing a trade task either in synchronous or asynchronous mode. By default, all trading actions are executed synchronously through the synchronous operation emulator. However, if a trade request (HedgeTradeRequest structure) contains an explicitly specified flag asynch\_mode = true, the trade task will be performed in an asynchronous mode.

In the asynchronous mode, tasks are performed independently of the main thread. Implementation of interaction between an asynchronous EA and asynchronous algorithms of HedgeTerminal is not complete yet.

The synchronous emulator is very simple. It starts the subtasks sequentially, and then waits for some time until the trade environment in MetaTrader 5 changes. The emulator analyzes these changes and determines the status of the current task. If the task execution is successful, the emulator moves on to the next one.

Synchronous emulator causes minor delays in the execution of trading orders. This is due to the fact that the trading environment in the MetaTrader 5 takes some time to reflect executed trading activities. The necessity to access to the environment is primarily connected with the fact that HedgeTermianlAPI cannot access events coming to the [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) handler in the synchronous thread emulation mode.

The problems of interaction between asynchronous threads, as well as between the asynchronous and synchronous threads through emulation are too complicated and have no obvious solutions.

**1.11. Management of Bi-Directional Position Properties through the Example of a Script**

In the script below, the TransactionSelect() function searches through all available transactions in the list of active transactions.

Each transaction is selected from the list. If the transaction is a position, some of its properties are accessed and then printed. In addition to the properties of the positions, properties of orders and deals inside the position are also printed. An order and a deal are first selected using HedgeOrderSelect() and HedgeDealSelect() respectively.

All properties of the position, its orders and deals are combined and printed as a single line using the system function printf.

```
//+------------------------------------------------------------------+
//|                                           sample_using_htapi.mq5 |
//|         Copyright 2014, Vasiliy Sokolov, Russia, St.-Petersburg. |
//|                              https://login.mql5.com/ru/users/c-4 |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, Vasiliy Sokolov."
#property link      "https://login.mql5.com/ru/users/c-4"
#property version   "1.00"

// Include prototypes function of HedgeTerminalAPI library.
#include <Prototypes.mqh>

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnStart()
  {
   // Search all transaction in list transaction...
   for(int i=TransactionsTotal(); i>=0; i--)
     {
      if(!TransactionSelect(i,SELECT_BY_POS,MODE_TRADES))                           // Selecting from active transactions
        {
         ENUM_HEDGE_ERR error=GetHedgeError();                                      // Get reason if selecting has failed
         printf("Error selecting transaction # "+(string)i+". Reason: "+            // Print reason
                EnumToString(error));
         ResetHedgeError();                                                         // Reset error
         continue;                                                                  // Go to next transaction
        }
      // Only for hedge positions
      if(TransactionType()==TRANS_HEDGE_POSITION)
        {
         // --- Position captions --- //
         ENUM_TRANS_DIRECTION direction=(ENUM_TRANS_DIRECTION)                      // Get direction caption
                              HedgePositionGetInteger(HEDGE_POSITION_DIRECTION);
         double price_entry = HedgeOrderGetDouble(HEDGE_ORDER_PRICE_EXECUTED);      // Get volume of positions
         string symbol = HedgePositionGetString(HEDGE_POSITION_SYMBOL);             // Get symbol of position
         // --- Order captions --- //
         if(!HedgeOrderSelect(ORDER_SELECTED_INIT))continue;                        // Selecting init order in position
         double slippage = HedgeOrderGetDouble(HEDGE_ORDER_SLIPPAGE);               // Get some slippage was
         uint deals_total = (uint)HedgeOrderGetInteger(HEDGE_ORDER_DEALS_TOTAL);    // Get deals total
         // --- Deals captions --- //
         double commissions=0.0;
         ulong deal_id=0;
         //Search all deals in list deals...
         for(uint d_index=0; d_index<deals_total; d_index++)
           {
            if(!HedgeDealSelect(d_index))continue;                                  // Selecting deal by its index
            deal_id = HedgeDealGetInteger(HEDGE_DEAL_ID);                           // Get deal id
            commissions += HedgeDealGetDouble(HEDGE_DEAL_COMMISSION);               // Count commissions
           }
         int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
         printf("Position #" + (string)i + ": DIR " + EnumToString(direction) +     // Print result line
         "; PRICE ENTRY " + DoubleToString(price_entry, digits) +
         "; INIT SLIPPAGE " + DoubleToString(slippage, 2) + "; LAST DEAL ID " +
         (string)deal_id + "; COMMISSIONS SUM " + DoubleToString(commissions, 2));
        }
     }
  }
```

**1.12. The SendTradeRequest() Function and the HedgeTradeRequest Structure through the Example of Chaos II EA**

As an example, let's develop a trading robot based on the trading tactics proposed by Bill Williams in his book [Trading Chaos. Second Edition](https://www.mql5.com/go?link=https://www.amazon.com/dp/0471463086/ref=rdr_ext_tmb "http://www.amazon.com/dp/0471463086/ref=rdr_ext_tmb").

We will not follow all his recommendations, but simplify the scheme by omitting the [Alligator](https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/alligator "https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/alligator") indicator and some other conditions. The choice of this strategy stems from several considerations. The main one is that this strategy includes composite position maintenance tactics. Sometimes you need to close a part of a position and move the stop loss to breakeven.

When moved to breakeven, the stop level should be trailed following the price. The second consideration is that this tactic is known enough and indicators developed for it are included in the standard MetaTrader 5 delivery pack. Let's slightly modify and simplify the rules, to prevent the complex logic of the Expert Advisor from hindering its primary objectives: to show an example of EA interaction with the HedgeTerminalAPI library. The EA's logic uses most of trading functions of HedgeTerminalAPI. This is a good test for the library.

Let's start on the _**reversal bar**_. A bullish reversal bar is a bar with the close price in its upper third, whose Low is the lowest one for the last N bars. A bearish reversal bar is a bar with the close price in its lower third, whose High is the highest one for the last N bars. N is a randomly chosen parameter, it can be set during start of the Expert Advisor. This differs from the classic strategy "Chaos 2".

After the reversal bar is defined, two pending orders are placed. For a bullish bar, the orders are placed above its high, a bearish bar - just below its low. If these two orders do not trigger during the _OldPending_ bars, the signal is considered obsolete, and orders are canceled. The values of OldPending and N are set by the user before launching the EA on the chart.

The orders trigger and turn into two bi-directional positions. The EA distinguishes between them by numbers in the comments, "# 1" and "# 2", respectively. This is not a very elegant solution, but it's fine for demonstration purposes. Once the orders trigger, a stop loss is placed at the high (for a bearish bar) or the low (if the bar bullish) of the reversal bar.

The first position has tight targets. Its take profit is set so that in case of triggering, the position profit would be equal to the absolute loss of a triggered stop loss. For example, if a long position is opened at a price of 1.0000, and its stop loss is at the level of 0.9000, the level of take profit would be _1.0000 + (1.0000 - 0.9000) = 1.1000_. The EA exits the position at stop loss or take profit.

The second position is a long term one. Its stop loss is trailed following the price. The stop moves after the newly formed Bill Williams' fractal. For a long position, the stop is moved according to the lower fractals, and upper fractals are used for a short position. The EA exits the position only at stop loss.

The following chart illustrates this strategy:

![Fig. 4. The representation of bi-directional positions of the Chaos 2 EA on the price chart](https://c.mql5.com/2/12/Chaos2.png)

Fig. 4. The representation of bi-directional positions of the Chaos 2 EA on the price chart

Reversal bars are marked with a red frame. The N period on this chart is equal to 2. The most opportune moment is chosen for this strategy. Short positions are shown as a blue dotted line, long positions are represented by green. As can be seen, long and short positions can exist simultaneously even in a relatively simple strategy. Pay attention to the period from 5 to 8 January, 2014.

This is a turning point for the AUDCAD downtrend. January 4 a signal was received from the bullish reversal bar, and January 5 two long positions were opened. At the same time, there were still three short positions whose stops were trailed following the trend (dashed red line). Then, on January 7, stop triggered for the short positions, so only long positions were left in the market.

Changes would be hard to monitor on a net position, since the net volume would not take into account the number of positions actually maintained by the EA. HedgeTerminal allows the EAs to monitor their individual positions regardless of the current net position, making it possible to get these charts and develop similar strategies.

Below is the code implementing this strategy.

I intentionally did not use object oriented programming, adapting the code for beginners:

```
//+------------------------------------------------------------------+
//|                                                       Chaos2.mq5 |
//|     Copyright 2014, Vasiliy Sokolov specially for HedgeTerminal. |
//|                                          St.-Petersburg, Russia. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, Vasiliy Sokolov."
#property link      "https://login.mql5.com/ru/users/c-4"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Include files                                                    |
//+------------------------------------------------------------------+
#include <Prototypes.mqh>           // Include prototypes function of HedgeTerminalAPI library

//+------------------------------------------------------------------+
//| Input parameters.                                                |
//+------------------------------------------------------------------+
input uint N=2;                     // Period of extermum/minimum
input uint OldPending=3;            // Old pending

//+------------------------------------------------------------------+
//| Private variables of expert advisor.                             |
//+------------------------------------------------------------------+
ulong Magic = 2314;                 // Magic number of expert
datetime lastTime = 0;              // Remembered last time for function DetectNewBar
int hFractals = INVALID_HANDLE;     // Handle of indicator 'Fractals'. See: 'https://www.mql5.com/en/docs/indicators/ifractals'
//+------------------------------------------------------------------+
//| Type of bar by Bill Wiallams strategy.                           |
//+------------------------------------------------------------------+
enum ENUM_BAR_TYPE
  {
   BAR_TYPE_ORDINARY,               // Ordinary bar.
   BAR_TYPE_BEARISH,                // This bar close in the upper third and it's minimum is lowest at N period
   BAR_TYPE_BULLISH,                // This bar close in the lower third and it's maximum is highest at N period
  };
//+------------------------------------------------------------------+
//| Type of Extremum.                                                |
//+------------------------------------------------------------------+
enum ENUM_TYPE_EXTREMUM
  {
   TYPE_EXTREMUM_HIGHEST,           // Extremum from highest prices
   TYPE_EXTREMUM_LOWEST             // Extremum from lowest prices
  };
//+------------------------------------------------------------------+
//| Type of position.                                                |
//+------------------------------------------------------------------+
enum ENUM_ENTRY_TYPE
  {
   ENTRY_BUY1,                      // Buy position with short target
   ENTRY_BUY2,                      // Buy position with long target
   ENTRY_SELL1,                     // Sell position with short target
   ENTRY_SELL2,                     // Sell position with long target
   ENTRY_BAD_COMMENT                // My position, but wrong comment
  };
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Create indicator 'Fractals' ---//
   hFractals=iFractals(Symbol(),NULL);
   if(hFractals==INVALID_HANDLE)
      printf("Warning! Indicator 'Fractals' not does not create. Reason: "+
             (string)GetLastError());
//--- Corection magic by timeframe ---//
   int minPeriod=PeriodSeconds()/60;
   string strMagic=(string)Magic+(string)minPeriod;
   Magic=StringToInteger(strMagic);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- Delete indicator 'Fractals' ---//
   if(hFractals!=INVALID_HANDLE)
      IndicatorRelease(hFractals);
//---
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Run logic only open new bar. ---//
   int totals=SupportPositions();
   if(NewBarDetect()==true)
     {
      MqlRates rates[];
      CopyRates(Symbol(),NULL,1,1,rates);
      MqlRates prevBar=rates[0];
      //--- Set new pendings order ---//
      double closeRate=GetCloseRate(prevBar);
      if(closeRate<=30 && BarIsExtremum(1,N,TYPE_EXTREMUM_HIGHEST))
        {
         DeleteOldPendingOrders(0);
         SetNewPendingOrder(1,BAR_TYPE_BEARISH);
        }
      else if(closeRate>=70 && BarIsExtremum(1,N,TYPE_EXTREMUM_LOWEST))
        {
         DeleteOldPendingOrders(0);
         SetNewPendingOrder(1,BAR_TYPE_BULLISH);
        }
      DeleteOldPendingOrders(OldPending);
     }
//---
  }
//+------------------------------------------------------------------+
//| Analyze open positions and modify it if needed.                  |
//+------------------------------------------------------------------+
int SupportPositions()
  {
//---
   int count=0;
   //--- Analize active positions... ---//
   for(int i=0; i<TransactionsTotal(); i++) // Get total positions.
     {
      //--- Select main active positions ---//
      if(!TransactionSelect(i, SELECT_BY_POS, MODE_TRADES))continue;             // Select active transactions
      if(TransactionType() != TRANS_HEDGE_POSITION)continue;                     // Select hedge positions only
      if(HedgePositionGetInteger(HEDGE_POSITION_MAGIC) != Magic)                 // Select main positions by magic
      if(HedgePositionGetInteger(HEDGE_POSITION_STATE) == POSITION_STATE_FROZEN) // If position is frozen - continue
         continue;                                                               // Let's try to get access to positions later
      count++;
      //--- What position do we choose?... ---//
      ENUM_ENTRY_TYPE type=IdentifySelectPosition();
      bool modify=false;
      double sl = 0.0;
      double tp = 0.0;
      switch(type)
        {
         case ENTRY_BUY1:
         case ENTRY_SELL1:
           {
            //--- Check sl, tp levels and modify it if need. ---//
            double currentStop=HedgePositionGetDouble(HEDGE_POSITION_SL);
            sl=GetStopLossLevel();
            if(!DoubleEquals(sl,currentStop))
               modify=true;
            tp=GetTakeProfitLevel();
            double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
            double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
            //--- Close by take-profit if price more tp level
            bool isBuyTp=tp<bid && !DoubleEquals(tp,0.0) && type==ENTRY_BUY1;
            bool isSellTp=tp>ask && type==ENTRY_SELL1;
            if(isBuyTp || isSellTp)
              {
               HedgeTradeRequest request;
               request.action=REQUEST_CLOSE_POSITION;
               request.exit_comment="Close by TP from expert";
               request.close_type=CLOSE_AS_TAKE_PROFIT;
               if(!SendTradeRequest(request))
                 {
                  ENUM_HEDGE_ERR error=GetHedgeError();
                  string logs=error==HEDGE_ERR_TASK_FAILED ? ". Print logs..." : "";
                  printf("Close position by tp failed. Reason: "+EnumToString(error)+" "+logs);
                  if(error==HEDGE_ERR_TASK_FAILED)
                     PrintTaskLog();
                  ResetHedgeError();
                 }
               else break;
              }
            double currentTakeProfit=HedgePositionGetDouble(HEDGE_POSITION_TP);
            if(!DoubleEquals(tp,currentTakeProfit))
               modify=true;
            break;
           }
         case ENTRY_BUY2:
           {
            //--- Check sl level and set modify flag. ---//
            sl=GetStopLossLevel();
            double currentStop=HedgePositionGetDouble(HEDGE_POSITION_SL);
            if(sl>currentStop)
               modify=true;
            break;
           }
         case ENTRY_SELL2:
           {
            //--- Check sl level and set modify flag. ---//
            sl=GetStopLossLevel();
            double currentStop=HedgePositionGetDouble(HEDGE_POSITION_SL);
            bool usingSL=HedgePositionGetInteger(HEDGE_POSITION_USING_SL);
            if(sl<currentStop || !usingSL)
               modify=true;
            break;
           }
        }
      //--- if  need modify sl, tp levels - modify it. ---//
      if(modify)
        {
         HedgeTradeRequest request;
         request.action=REQUEST_MODIFY_SLTP;
         request.sl = sl;
         request.tp = tp;
         if(type==ENTRY_BUY1 || type==ENTRY_SELL1)
            request.exit_comment="Exit by T/P level";
         else
            request.exit_comment="Exit by trailing S/L";
         if(!SendTradeRequest(request))
           {
            ENUM_HEDGE_ERR error=GetHedgeError();
            string logs=error==HEDGE_ERR_TASK_FAILED ? ". Print logs..." : "";
            printf("Modify stop-loss or take-profit failed. Reason: "+EnumToString(error)+" "+logs);
            if(error==HEDGE_ERR_TASK_FAILED)
               PrintTaskLog();
            ResetHedgeError();
           }
         else break;
        }
     }
   return count;
//---
  }
//+------------------------------------------------------------------+
//| Return stop-loss level for selected position.                    |
//| RESULT                                                           |
//|   Stop-loss level                                                |
//+------------------------------------------------------------------+
double GetStopLossLevel()
  {
//---
   double point=SymbolInfoDouble(Symbol(),SYMBOL_TRADE_TICK_SIZE)*3;
   double fractals[];
   double sl=0.0;
   MqlRates ReversalBar;

   if(!LoadReversalBar(ReversalBar))
     {
      printf("Reversal bar load failed.");
      return sl;
     }
   //--- What position do we choose?... ---//
   switch(IdentifySelectPosition())
     {
      case ENTRY_SELL2:
        {
         if(HedgePositionGetInteger(HEDGE_POSITION_USING_SL))
           {
            sl=NormalizeDouble(HedgePositionGetDouble(HEDGE_POSITION_SL),Digits());
            CopyBuffer(hFractals,UPPER_LINE,ReversalBar.time,TimeCurrent(),fractals);
            for(int i=ArraySize(fractals)-4; i>=0; i--)
              {
               if(DoubleEquals(fractals[i],DBL_MAX))continue;
               if(DoubleEquals(fractals[i],sl))continue;
               if(fractals[i]<sl)
                 {
                  double price= SymbolInfoDouble(Symbol(),SYMBOL_ASK);
                  int ifreeze =(int)SymbolInfoInteger(Symbol(),SYMBOL_TRADE_FREEZE_LEVEL);
                  double freeze=SymbolInfoDouble(Symbol(),SYMBOL_TRADE_TICK_SIZE)*ifreeze;
                  if(fractals[i]>price+freeze)
                     sl=NormalizeDouble(fractals[i]+point,Digits());
                 }
              }
            break;
           }
        }
      case ENTRY_SELL1:
         sl=ReversalBar.high+point;
         break;
      case ENTRY_BUY2:
         if(HedgePositionGetInteger(HEDGE_POSITION_USING_SL))
           {
            sl=NormalizeDouble(HedgePositionGetDouble(HEDGE_POSITION_SL),Digits());
            CopyBuffer(hFractals,LOWER_LINE,ReversalBar.time,TimeCurrent(),fractals);
            for(int i=ArraySize(fractals)-4; i>=0; i--)
              {
               if(DoubleEquals(fractals[i],DBL_MAX))continue;
               if(DoubleEquals(fractals[i],sl))continue;
               if(fractals[i]>sl)
                 {
                  double price= SymbolInfoDouble(Symbol(),SYMBOL_BID);
                  int ifreeze =(int)SymbolInfoInteger(Symbol(),SYMBOL_TRADE_FREEZE_LEVEL);
                  double freeze=SymbolInfoDouble(Symbol(),SYMBOL_TRADE_TICK_SIZE)*ifreeze;
                  if(fractals[i]<price-freeze)
                     sl=NormalizeDouble(fractals[i]-point,Digits());
                 }
              }
            break;
           }
      case ENTRY_BUY1:
         sl=ReversalBar.low-point;
     }
   sl=NormalizeDouble(sl,Digits());
   return sl;
//---
  }
//+------------------------------------------------------------------+
//| Return Take-Profit level for selected position.                  |
//| RESULT                                                           |
//|   Take-profit level                                              |
//+------------------------------------------------------------------+
double GetTakeProfitLevel()
  {
//---
   double point=SymbolInfoDouble(Symbol(),SYMBOL_TRADE_TICK_SIZE)*3;
   ENUM_ENTRY_TYPE type=IdentifySelectPosition();
   double tp=0.0;
   if(type==ENTRY_BUY1 || type==ENTRY_SELL1)
     {
      if(!HedgePositionGetInteger(HEDGE_POSITION_USING_SL))
         return tp;
      double sl=HedgePositionGetDouble(HEDGE_POSITION_SL);
      double openPrice=HedgePositionGetDouble(HEDGE_POSITION_PRICE_OPEN);
      double deltaStopLoss=MathAbs(NormalizeDouble(openPrice-sl,Digits()));
      if(type==ENTRY_BUY1)
         tp=openPrice+deltaStopLoss;
      if(type==ENTRY_SELL1)
         tp=openPrice-deltaStopLoss;
      return tp;
     }
   else
      return 0.0;
//---
  }
//+------------------------------------------------------------------+
//| Identify what position type is select.                           |
//| RESULT                                                           |
//|   Return type position. See ENUM_ENTRY_TYPE                      |
//+------------------------------------------------------------------+
ENUM_ENTRY_TYPE IdentifySelectPosition()
  {
//---
   string comment=HedgePositionGetString(HEDGE_POSITION_ENTRY_COMMENT);
   int pos=StringLen(comment)-2;
   string subStr=StringSubstr(comment,pos);
   ENUM_TRANS_DIRECTION posDir=(ENUM_TRANS_DIRECTION)HedgePositionGetInteger(HEDGE_POSITION_DIRECTION);
   if(subStr=="#0")
     {
      if(posDir==TRANS_LONG)
         return ENTRY_BUY1;
      if(posDir==TRANS_SHORT)
         return ENTRY_SELL1;
     }
   else if(subStr=="#1")
     {
      if(posDir==TRANS_LONG)
         return ENTRY_BUY2;
      if(posDir==TRANS_SHORT)
         return ENTRY_SELL2;
     }
   return ENTRY_BAD_COMMENT;
//---
  }
//+------------------------------------------------------------------+
//| Set pending orders under or over bar by index_bar.               |
//| INPUT PARAMETERS                                                 |
//|   index_bar - index of bar.                                      |
//|   barType - type of bar. See enum ENUM_BAR_TYPE.                 |
//| RESULT                                                           |
//|   True if new order successfully set, othewise false.            |
//+------------------------------------------------------------------+
bool SetNewPendingOrder(int index_bar,ENUM_BAR_TYPE barType)
  {
//---
   MqlRates rates[1];
   CopyRates(Symbol(),NULL,index_bar,1,rates);
   MqlTradeRequest request={0};
   request.volume=SymbolInfoDouble(Symbol(),SYMBOL_VOLUME_MIN);
   double vol=request.volume;
   request.symbol = Symbol();
   request.action = TRADE_ACTION_PENDING;
   request.type_filling=ORDER_FILLING_FOK;
   request.type_time=ORDER_TIME_GTC;
   request.magic=Magic;
   double point=SymbolInfoDouble(Symbol(),SYMBOL_TRADE_TICK_SIZE)*3;
   string comment="";
   if(barType==BAR_TYPE_BEARISH)
     {
      request.price=rates[0].low-point;
      comment="Entry sell by bearish bar";
      request.type=ORDER_TYPE_SELL_STOP;
     }
   else if(barType==BAR_TYPE_BULLISH)
     {
      request.price=rates[0].high+point;
      comment="Entry buy by bullish bar";
      request.type=ORDER_TYPE_BUY_STOP;
     }
   MqlTradeResult result={0};
//--- Send pending order twice...
   for(int i=0; i<2; i++)
     {
      request.comment=comment+" #"+(string)i;       // Detect order by comment;
      if(!OrderSend(request,result))
        {
         printf("Trade error #"+(string)result.retcode+" "+
                result.comment);
         return false;
        }
     }
   return true;
//---
  }
//+------------------------------------------------------------------+
//| Delete old pending orders. If pending order set older that       |
//| n_bars ago pending orders will be removed.                       |
//| INPUT PARAMETERS                                                 |
//|   period - count bar.                                            |
//+------------------------------------------------------------------+
void DeleteOldPendingOrders(int n_bars)
  {
//---
   for(int i=0; i<OrdersTotal(); i++)
     {
      ulong ticket = OrderGetTicket(i);            // Get ticket of order by index.
      if(!OrderSelect(ticket))                     // Continue if not selected.
         continue;
      if(Magic!=OrderGetInteger(ORDER_MAGIC))      // Continue if magic is not main.
         continue;
      if(OrderGetString(ORDER_SYMBOL)!=Symbol())   // Continue if symbol is not main.
         continue;
      //--- Count time elipsed ---//
      datetime timeSetup=(datetime)OrderGetInteger(ORDER_TIME_SETUP);
      int secElapsed=(int)(TimeCurrent()-timeSetup);
      //--- delete old pending order ---//
      if(secElapsed>=PeriodSeconds() *n_bars)
        {
         MqlTradeRequest request={0};
         MqlTradeResult result={0};
         request.action= TRADE_ACTION_REMOVE;
         request.order = ticket;
         if(!OrderSend(request,result))
            printf("Delete pending order failed. Reason #"+(string)result.retcode+" "+result.comment);
        }
     }
//---
  }
//+------------------------------------------------------------------+
//| Detect new bar.                                                  |
//+------------------------------------------------------------------+
bool NewBarDetect(void)
  {
//---
   datetime timeArray[1];
   CopyTime(Symbol(),NULL,0,1,timeArray);
   if(lastTime!=timeArray[0])
     {
      lastTime=timeArray[0];
      return true;
     }
   return false;
//---
  }
//+------------------------------------------------------------------+
//| Get close rate. Type bar defined in trade chaos strategy         |
//| and equal enum 'ENUM_TYPE_BAR'.                                  |
//| INPUT PARAMETERS                                                 |
//|   index - index of bars series. for example:                     |
//|   '0' - is current bar. 1 - previous bar.                        |
//| RESULT                                                           |
//|   Type of ENUM_TYPE_BAR.                                         |
//+------------------------------------------------------------------+
double GetCloseRate(const MqlRates &bar)
  {
//---
   double highLowDelta = bar.high-bar.low;      // Calculate diaposon bar.
   double lowCloseDelta = bar.close - bar.low;  // Calculate Close - Low delta.
   double percentClose=0.0;
   if(!DoubleEquals(lowCloseDelta, 0.0))                    // Division by zero protected.
      percentClose = lowCloseDelta/highLowDelta*100.0;      // Calculate percent 'lowCloseDelta' of 'highLowDelta'.
   return percentClose;
//---
  }
//+------------------------------------------------------------------+
//| If bar by index is extremum - return true, otherwise             |
//| return false.                                                    |
//| INPUT PARAMETERS                                                 |
//|   index - index of bar.                                          |
//|   period - Number of bars prior to the extremum.                 |
//|   type - Type of extremum. See ENUM_TYPE_EXTREMUM TYPE enum.     |
//| RESULT                                                           |
//|   True - if bar is extremum, otherwise false.                    |
//+------------------------------------------------------------------+
bool BarIsExtremum(const int index,const int period,ENUM_TYPE_EXTREMUM type)
  {
//--- Copy rates --- //
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   CopyRates(Symbol(),NULL,index,N+1,rates);
//--- Search extremum --- //
   for(int i=1; i<ArraySize(rates); i++)
     {
      //--- Reset comment if you want include volume analize. ---//
      //if(rates[0].tick_volume<rates[i].tick_volume)
      //   return false;
      if(type==TYPE_EXTREMUM_HIGHEST &&
         rates[0].high<rates[i].high)
         return false;
      if(type==TYPE_EXTREMUM_LOWEST &&
         rates[0].low>rates[i].low)
         return false;
     }
   return true;
//---
  }
//+------------------------------------------------------------------+
//| Print current error and reset it.                                |
//+------------------------------------------------------------------+
void PrintTaskLog()
  {
//---
   uint totals=(uint)HedgePositionGetInteger(HEDGE_POSITION_ACTIONS_TOTAL);
   for(uint i = 0; i<totals; i++)
     {
      uint retcode=0;
      ENUM_TARGET_TYPE type;
      GetActionResult(i,type,retcode);
      printf("---> Action #"+(string)i+"; "+EnumToString(type)+"; RETCODE: "+(string)retcode);
     }
//---
  }
//+------------------------------------------------------------------+
//| Load reversal bar. The current position must be selected.        |
//| OUTPUT PARAMETERS                                                |
//|   bar - MqlRates bar.
//+------------------------------------------------------------------+
bool LoadReversalBar(MqlRates &bar)
  {
//---
   datetime time=(datetime)(HedgePositionGetInteger(HEDGE_POSITION_ENTRY_TIME_SETUP_MSC)/1000+1);
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   CopyRates(Symbol(),NULL,time,2,rates);
   int size=ArraySize(rates);
   if(size==0)return false;
   bar=rates[size-1];
   return true;
//---
  }
//+------------------------------------------------------------------+
//| Compares two double numbers.                                     |
//| RESULT                                                           |
//|   True if two double numbers equal, otherwise false.             |
//+------------------------------------------------------------------+
bool DoubleEquals(const double a,const double b)
  {
//---
   return(fabs(a-b)<=16*DBL_EPSILON*fmax(fabs(a),fabs(b)));
//---
  }
```

Below is a brief description of how this code works. The EA is called on every tick. It analyzes the previous bar using the BarIsExtremum() function: if it is bearish or bullish, it places two pending orders (function SetNewPendingOrder()). Once activated, pending orders are converted into positions. The EA sets stop loss and take profit for the positions then.

Unfortunately, these levels cannot be placed together with pending orders, because there is no real position yet. The levels are set through the SupportPositions() function. To operate properly, we need to know the position for which take profit should be placed, and the position that should be trailed following fractals. This definition of positions is done by the IdentifySelectPosition() function. It analyzes the initiating position comment, and if it contains the string "#1", a tight target is set for it; if it contains "# 2", trailing stop is applied.

To modify an open bi-directional position, or to close it, a special trade request is created, which is then sent to the SendTradeRequest() function for execution:

```
...
if(modify)
  {
   HedgeTradeRequest request;
   request.action=REQUEST_MODIFY_SLTP;
   request.sl = sl;
   request.tp = tp;
   if(type==ENTRY_BUY1 || type==ENTRY_SELL1)
      request.exit_comment="Exit by T/P level";
   else
      request.exit_comment="Exit by trailing S/L";
   if(!SendTradeRequest(request))
     {
      ENUM_HEDGE_ERR error=GetHedgeError();
      string logs=error==HEDGE_ERR_TASK_FAILED ? ". Print logs..." : "";
      printf("Modify stop-loss or take-profit failed. Reason: "+EnumToString(error)+" "+logs);
      if(error==HEDGE_ERR_TASK_FAILED)
         PrintTaskLog();
      ResetHedgeError();
     }
   else break;
  }
...
```

Pay attention to error handling.

If sending fails and the function returns false, we need to get the last error code using the GetHedgeError() function. In some cases the execution of a trading order will not even start. If position has not been pre-selected, then the query is made incorrectly, and its execution is impossible.

If an order is not executed, it is pointless to analyze the log of its implementation, getting an error code is enough.

However, if the query is correct, but the order has not been executed for some reason, error HEDGE\_ERR\_TASK\_FAILED will be returned. In this case, it is necessary to analyze the order execution log by searching through the log. This is done through the special function PrintTaskLog():

```
//+------------------------------------------------------------------+
//| Print current error and reset it.                                |
//+------------------------------------------------------------------+
void PrintTaskLog()
  {
//---
   uint totals=(uint)HedgePositionGetInteger(HEDGE_POSITION_ACTIONS_TOTAL);
   for(uint i = 0; i<totals; i++)
     {
      uint retcode=0;
      ENUM_TARGET_TYPE type;
      GetActionResult(i,type,retcode);
      printf("---> Action #"+(string)i+"; "+EnumToString(type)+"; RETCODE: "+(string)retcode);
     }
//---
  }
```

These messages allow to identify the reason for the failure and fix it.

Let us now illustrate the display of the Chaos2 EA and its positions in HedgeTerminal in real time. The EA is running on the M1 chart:

![Fig. 5. The representation of bi-directional positions of the Chaos 2 EA in the HedgeTerminal panel](https://c.mql5.com/2/12/Chaos2TH.png)

Fig. 5. The representation of bi-directional positions of the Chaos 2 EA in the HedgeTerminal panel

As can be seen, even bi-directional positions of one EA can perfectly coexist.

**1.13. On "Duplicate Symbols" and Virtualization by Broker**

When MetaTrader 5 was released, some brokers started to provide the so-called duplicate symbols. Their quotes are equal to the original instruments, but they have a postfix as a rule, like _"\_m"_ or _"\_1"_. They were introduced to allow traders to have bi-directional positions on virtually the same symbol.

However, such symbols are almost useless for algorithmic traders using robots. And here's why. Suppose we would need to write the "Chaos II" EA without the HedgeTerminalAPI library. Instead, we would have some duplicate symbols. How would we do that? Assume all sell operations were opened on a single instrument, such as EURUSD, and all buy operations on the other one, for example EURUSD\_m1.

But what would happen if, at the moment of position opening one of the symbols were already traded by another robot or by the trader? Even if such symbols were always free, the problem would not be solved for this robot, which could simultaneously have multiple positions in the same direction.

The screenshot above shows three sell positions, and there can be even more of them. The positions have different protective stop levels, that is why they cannot be combined into a single net position. The solution is to open a new position for a new duplicate symbol. But there can be not enough such symbols, because one robot needs six duplicate instruments (three in each trade direction). If two robots run on different timeframes, 12 symbols are required.

None of the brokers provide so many duplicate symbols. But even if there were an unlimited amount of such symbols, and they were always free, a complex decomposition of the algorithm would be required. The robot would have to go through all the available symbols searching for duplicates and its own positions. This would create more problems than it could solve.

There are even more difficulties with the duplicate symbols. Here is a brief list of additional problems arising from their use:

- You pay for each duplicate symbol in the form of negative swaps, because swaps for locking or partial locking are always negative, and this is the case when you keep two bi-directional positions on two different instruments.
- Not all brokers provide duplicate symbols. A strategy developed for a broker who provides duplicate symbols will not work with a broker providing only one instrument. The difference in the symbol names is another potential source of problems.
- Creating a duplicate symbol is not always possible. On transparent markets subject to strict regulations, any transaction is a financial document. The net position is the de facto standard in such markets and therefore creation of individual symbols is far from possible there. For example, no broker providing duplicate symbols can ever appear on Moscow Exchange MOEX. In less strict regulated markets brokers can create any symbols for their clients.
- Duplicate instruments are ineffective when trading using robots. The reasons for their ineffectiveness have been disclosed in the above example of the Chaos 2 EA.

A duplicate symbol is essentially a virtualization on the broker side. HedgeTerminal uses virtualization on the client side.

In both cases we use virtualization as such. It changes the actual representation of trader's obligations. With virtualization, one position can turn into two positions. There is no problem when it occurs on the client side, because clients can represent whatever they want. But if virtualization is done by the broker, regulatory and licensing organizations may have questions about how the information provided relates to the actual information. The second difficulty is that this requires having two APIs in one: a set of functions and modifiers for use in the net mode, and another one for the bi-directional mode.

Many algorithmic traders have found their own way to bind trades into a single position. Many of these methods work well, and there are articles describing these methods. However, virtualization of positions is a more complicated procedure than it might seem. In HedgeTerminal, the algorithms associated with virtualization of positions take about 20,000 lines of the source code. Moreover, HedgeTerminal implements only basic functions. Creating a similar amount of code in your EA only to accompany bi-directional positions would be too much resource consuming.

### Chapter 2. HedgeTerminal API Manual

**2.1. Transaction Selection Functions**

**Function TransactionsTotal()**

The function returns the total number of transactions in the list of transactions. This is the basic function for searching though available transactions (see the example in section [1.4](https://www.mql5.com/en/articles/1316#c1_4) and [1.11](https://www.mql5.com/en/articles/1316#c1_11) of this article).

```
int TransactionsTotal(ENUM_MODE_TRADES pool = MODE_TRADES);
```

_**Parameters**_

- \[in\] _pool=MODE\_TRADES_  – Specifies the identifier of the data source for selection. It can be one of the values of the [ENUM\_MODE\_TRADES](https://www.mql5.com/en/articles/1316#ENUM_MODE_TRADES) enumeration.

_**Return Value**_

The function returns the total number of transactions in the list of transactions.

**Function TransactionType()**

The function returns the type of a selected transaction.

```
ENUM_TRANS_TYPE TransactionType(void);
```

_**Return Value**_

Return type. The value can be one of the [ENUM\_TRANS\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_TRANS_TYPE) values.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function TransactionSelect()**

This function selects a transaction for further manipulations. The function selects a transaction by its index or unique identifier in the list of transactions.

```
bool TransactionSelect(int index,
     ENUM_MODE_SELECT select = SELECT_BY_POS,
     ENUM_MODE_TRADES pool=MODE_TRADES
     );
```

_**Parameters**_

- \[in\] _index_ – The index of the order in the list of orders or a unique identifier of the transaction depending on the 'select' parameter.
- \[in\] _select=SELECT\_BY\_POS_ – Identifier of the 'index' type of the parameter. The value can be one of the [ENUM\_MODE\_SELECT](https://www.mql5.com/en/articles/1316#ENUM_MODE_SELECT) values.
- \[in\] _pool=MODE\_TRADES_ – Specifies the identifier of the data source for selection. It can be one of the values of the [ENUM\_MODE\_TRADES](https://www.mql5.com/en/articles/1316#ENUM_MODE_TRADES) enumeration.

_**Return Value**_

Returns true if a transaction has been successfully selected or false otherwise. To get error details, call [GetHedgeError()](https://www.mql5.com/en/articles/1316#GetHedgeError).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

_**Note**_

If a transaction is selected based on its index, the complexity of the operation corresponds to _O(1)_. If a transaction is selected based on its unique identifier, the complexity of the operation asymptotically tends to _O(log2(n))_.

**Function HedgeOrderSelect()**

The function selects one of the orders included in the bi-directional position. The bi-directional position, which includes the required order, must be pre-selected using [TransactionSelect ()](https://www.mql5.com/en/articles/1316#TransactionSelect).

```
bool HedgeOrderSelect(ENUM_HEDGE_ORDER_SELECTED_TYPE type);
```

**_Parameters_**

- \[in\] _type_ – the identifier of the order to be selected. The value can be one of the [ENUM\_HEDGE\_ORDER\_SELECTED\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_SELECTED_TYPE) enumeration values.

_**Return Value**_

Returns true if an order has been successfully selected or false otherwise. To get error details, call [GetHedgeError()](https://www.mql5.com/en/articles/1316#GetHedgeError).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgeDealSelect()**

The function selects one of the deals that have executed the order. The order the part of which is the selected deal must be pre-selected using the [HedgeOrderSelect()](https://www.mql5.com/en/articles/1316#HedgeOrderSelect) function.

```
bool HedgeDealSelect(int index);
```

_**Parameters**_

- \[in\] _index_ – The index of the deal to be selected from the list of deals that executed the order. To find out the total number of deals inside one order, call the appropriate order property using the [HedgeOrderGetInteger()](https://www.mql5.com/en/articles/1316#HedgeOrderGetInteger) function. For the parameter, use the [ENUM\_HEDGE\_ORDER\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) modifier equal to the [HEDGE\_ORDER\_DEALS\_TOTAL](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) value.

_**Return Value**_

Returns true if a deal has been successfully selected or false otherwise. To get error details, call [GetHedgeError()](https://www.mql5.com/en/articles/1316#GetHedgeError).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**2.2. Functions for Getting Properties of a Selected Transaction**

**Function HedgePositionGetInteger()**

The function returns the property of a selected bi-directional position. The property can be of type [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int), [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long), [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) or [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst) depending on the type of the requested property. The bi-directional position must be pre-selected using the [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect) function.

```
ulong HedgePositionGetInteger(ENUM_HEDGE_POSITION_PROP_INTEGER property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the bi-directional position. The value can be one of the [ENUM\_HEDGE\_DEAL\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_INTEGER) enumeration values.

_**Return Value**_

Value of the [ulong](https://www.mql5.com/en/docs/basis/types/integer/integertypes#ulong) type. For further use of the value, its type should be [explicitly cast](https://www.mql5.com/en/docs/basis/types/casting) to the type of the requested property.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgePositionGetDouble()**

The function returns the property of a selected bi-directional position. The type of the return property is [double](https://www.mql5.com/en/docs/basis/types/double). Property type is specified through the [ENUM\_HEDGE\_POSITION\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_DOUBLE) enumeration. The bi-directional position must be pre-selected using the [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect).

```
ulong HedgePositionGetDouble(ENUM_HEDGE_POSITION_PROP_DOUBLE property);
```

_**Parameters**_

- \[in\] _property_– Identifier of the property of the bi-directional position. The value can be one of the [ENUM\_HEDGE\_DEAL\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_DOUBLE) enumeration values.

_**Return Value**_

A value of type [double](https://www.mql5.com/en/docs/basis/types/double).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgePositionGetString()**

The function returns the property of a selected bi-directional position. The property is of the [string](https://www.mql5.com/en/docs/basis/types/stringconst) type. Property type is specified through the [ENUM\_HEDGE\_POSITION\_PROP\_STRING](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_STRING) enumeration. The bi-directional position must be pre-selected using the [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect).

```
ulong HedgePositionGetString(ENUM_HEDGE_POSITION_PROP_STRING property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the bi-directional position. The value can be one of the [ENUM\_HEDGE\_POSITION\_PROP\_STRING](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_PROP_STRING) enumeration values.

**_Return Value_**

A value of the [string](https://www.mql5.com/en/docs/basis/types/stringconst) type.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgeOrderGetInteger()**

The function returns the property of the selected order, which is part of the bi-directional position. The property can be of type [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int), [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long), [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) or [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst). Property type is specified through the [ENUM\_HEDGE\_ORDER\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) enumeration. The order must be pre-selected using the [HedgeOrderSelect()](https://www.mql5.com/en/articles/1316#HedgeOrderSelect) function.

```
ulong HedgeOrderGetInteger(ENUM_HEDGE_ORDER_PROP_INTEGER property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the order, which is part of the bi-directional position. The value can be one of the [ENUM\_HEDGE\_ORDER\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) enumeration values.

_**Return Value**_

Value of the [ulong](https://www.mql5.com/en/docs/basis/types/integer/integertypes#ulong) type. For further use of the value, its type must be [explicitly cast](https://www.mql5.com/en/docs/basis/types/casting) to the type of the requested property.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgeOrderGetDouble()**

The function returns the property of the selected order, which is part of the bi-directional position. The requested property is of the [double](https://www.mql5.com/en/docs/basis/types/double) type. Property type is specified through the [ENUM\_HEDGE\_ORDER\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) enumeration. The order must be pre-selected using the [HedgeOrderSelect()](https://www.mql5.com/en/articles/1316#HedgeOrderSelect) function.

```
double HedgeOrderGetDouble(ENUM_HEDGE_ORDER_PROP_DOUBLE property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the order, which is part of the bi-directional position. The value can be any of the [ENUM\_HEDGE\_ORDER\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ORDER_PROP_INTEGER) enumeration values.

_**Return Value**_

A value of type [double](https://www.mql5.com/en/docs/basis/types/double).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgeDealGetInteger()**

The function returns the property of the selected deal, which is part of the executed order. The property can be of type [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int), [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long), [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) or [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst). Property type is specified through the [ENUM\_HEDGE\_DEAL\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_INTEGER) enumeration. The deal must be pre-selected using the [HedgeDealSelect()](https://www.mql5.com/en/articles/1316#HedgeDealSelect) function.

```
ulong HedgeOrderGetInteger(ENUM_HEDGE_DEAL_PROP_INTEGER property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the selected deal included in the executed order. The value can be one of the [ENUM\_HEDGE\_DEAL\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_INTEGER) enumeration values.

_**Return Value**_

Value of the [ulong](https://www.mql5.com/en/docs/basis/types/integer/integertypes#ulong) type. For further use of the value, its type should be explicitly cast to the type of the requested property.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function HedgeDealGetDouble()**

The function returns the property of the selected deal, which is part of the executed order. The property can be of type [double](https://www.mql5.com/en/docs/basis/types/double). Property type is specified through the [ENUM\_HEDGE\_DEAL\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_DOUBLE) enumeration. The deal must be pre-selected using the [HedgeDealSelect()](https://www.mql5.com/en/articles/1316#HedgeDealSelect) function.

```
ulong HedgeOrderGetDouble(ENUM_HEDGE_DEAL_PROP_DOUBLE property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property of the selected deal included in the executed order. The value can be one of the [ENUM\_HEDGE\_DEAL\_PROP\_DOUBLE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_DEAL_PROP_DOUBLE) enumeration values.

_**Return Value**_

A value of type [double](https://www.mql5.com/en/docs/basis/types/double).

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**2.3. Functions for Setting and Getting HedgeTerminal Properties from Expert Advisors**

**Function HedgePropertySetInteger()**

The function sets one of the HedgeTerminal properties. The property can be of type [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int), [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long), [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) or [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst). Property type is specified through the [ENUM\_HEDGE\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_INTEGER) enumeration.

```
bool HedgePropertySetInteger(ENUM_HEDGE_PROP_INTEGER property, long value);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property that should be set for HedgeTerminal. The value can be one of the [ENUM\_HEDGE\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_INTEGER) enumeration values.

_**Return Value**_

A value of type [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst). If the property has been successfully set, the function returns true, otherwise it returns false.

_**Example of Use**_

In the example, the function is used to set the position locking time while sending an asynchronous request. If the server response is not received within 30 seconds after you send an asynchronous request, the blocked position will be unblocked.

```
void SetTimeOut()
  {
   bool res=HedgePropertySetInteger(HEDGE_PROP_TIMEOUT,30);
   if(res)
      printf("The property is set successfully");
   else
      printf("Property is not set");
  }
```

**Function HedgePropertyGetInteger()**

The function gets one of the HedgeTerminal properties. The property can be of type [int](https://www.mql5.com/en/docs/basis/types/integer/integertypes#int), [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long), [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime) or [bool](https://www.mql5.com/en/docs/basis/types/integer/boolconst). Property type is specified through the [ENUM\_HEDGE\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_INTEGER) enumeration.

```
long HedgePropertyGetInteger(ENUM_HEDGE_PROP_INTEGER property);
```

_**Parameters**_

- \[in\] _property_ – Identifier of the property that should be received from HedgeTerminal. The value can be one of the [ENUM\_HEDGE\_PROP\_INTEGER](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_PROP_INTEGER) enumeration values.

_**Return Value**_

A value of type [long](https://www.mql5.com/en/docs/basis/types/integer/integertypes#long).

_**Example of Use**_

The function receives the position blocking time while sending an asynchronous request and shows it in the terminal.

```
void GetTimeOut()
  {
   int seconds=HedgePropertyGetInteger(HEDGE_PROP_TIMEOUT);
   printf("Timeout is "+(string) seconds);
  }
```

**2.4. Functions for Getting and Handling Error Codes**

**Function GetHedgeError()**

The function returns the identifier of the error, which was obtained from the last action. The error identifier corresponds to the [ENUM\_HEDGE\_ERR](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ERR) enumeration.

```
ENUM_HEDGE_ERR GetHedgeError(void);
```

_**Return Value**_

Position ID. The value can be any of the [ENUM\_HEDGE\_ERR](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ERR) enumeration type.

_**Note**_

After the call, the GetHedgeError() function does not reset the error ID. To reset the error ID use the [ResetHedgeError()](https://www.mql5.com/en/articles/1316#ResetHedgeError) function.

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function ResetHedgeError()**

The function resets the identifier of the last received error. After its call, the [ENUM\_HEDGE\_ERR](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ERR) identifier returned by GetHedgeError() will be equal to [HEDGE\_ERR\_NOT\_ERROR](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_ERR).

```
void ResetHedgeError(void);
```

_**Example of Use**_

See the example of function use in section 1.11 of this article: " [Management of Bi-Directional Position Properties through the Example of a Script](https://www.mql5.com/en/articles/1316#c1_11)".

**Function TotalActionsTask()**

Once the position is selected using the [HedgePositionSelect()](https://www.mql5.com/en/articles/1316#HedgePositionSelect) function, it can be modified using the [SendTradeRequest()](https://www.mql5.com/en/articles/1316#SendTradeRequest) function. For example, it can be closed, or its outgoing comment can be changed. This modification is performed by a special _trading task_. Each task can consist of several trading activities (subtasks). A task may fail. In this case you may need to analyze the result of all the subtasks included in the task to see what kind of the subtasks failed.

The TotalActionTask() function returns the number of subtasks contained in the last trading task being executed for the selected position. Knowing the total number of subtasks, you can search though all the subtasks by their index, and analyze their execution results using the [GetActionResult()](https://www.mql5.com/en/articles/1316#GetActionResult) function and thereby find out the circumstances of the failure.

```
uint TotalActionsTask(void);
```

_**Return Value**_

Returns the total number of subtasks within the task.

**_Example of Use_**

See the example of use in section 1.6 of this article: " [Detailed Analysis of Trading and Identification of Errors Using TotalActionsTask() and GetActionResult()](https://www.mql5.com/en/articles/1316#c1_6)".

**Function GetActionResult()**

The function takes the index of the subtask within the task (see [TotalActionTask()](https://www.mql5.com/en/articles/1316#TotalActionsTask)). Returns the type of the subtask and its execution results through reference parameters. The type of the subtask is defined by the [ENUM\_TARGET\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_TARGET_TYPE) enumeration. The subtask execution result corresponds to the MetaTrader 5 [trade server return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes).

```
void GetActionResult(uint index, ENUM_TARGET_TYPE& target_type, uint& retcode);
```

_**Parameters**_

- \[in\] _index_ – The index of the subtask in the list of subtasks.
- \[out\] _target\_type_ – Type of the subtask. The value can be one of the [ENUM\_TARGET\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_TARGET_TYPE) enumeration values.
- \[out\] _retcode_ – Trade server return code received upon the subtask execution.

_**Example of Use**_

See the example of use in section 1.6 of this article: " [Detailed Analysis of Trading and Identification of Errors Using TotalActionsTask() and GetActionResult()](https://www.mql5.com/en/articles/1316#c1_6)".

**2.5. Trading**

**Function SendTradeRequest()**

The function sends a request to change the selected bi-directional position in HedgeTerminalAPI. The function execution result is one of the three actions:

1. Closing a position or a part of its volume;
2. Modification of stop loss and take profit levels;
3. Modification of the outgoing comment.

The action type and its parameters are specified in the [HedgeTradeRequest](https://www.mql5.com/en/articles/1316#HedgeTradeRequest) structure, which is passed by reference as a parameter. Before the function call, the bi-directional position must be pre-selected using the [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect) function.

```
bool SendTradeRequest(HedgeTradeRequest& request);
```

_**Parameters**_

\[in\] _request_ – The structure of the request to modify the bi-directional position. Please see the structure description and the explanation of its fields in the description of the [HedgeTradeRequest](https://www.mql5.com/en/articles/1316#HedgeTradeRequest) structure.

_**Return Value**_

Returns true, if the request for position modification has been successfully executed. Returns false otherwise. In case of request execution failure, use functions [TotalActionsTask()](https://www.mql5.com/en/articles/1316#TotalActionsTask) and [GetActionResult()](https://www.mql5.com/en/articles/1316#GetActionResult) to find the failure and its reasons.

_**Note**_

In the asynchronous mode of request sending, the return flag contains true if a task has been successfully placed and started. However, we must remember that even in the case of the successful start of a task, its execution cannot be guaranteed. Therefore this flag cannot be used to control task completion in the asynchronous mode. In the synchronous mode, a task is started and executed in a single thread, so in the synchronous mode, you can control the trade request execution result using this flag.

**Trade Request Structure HedgeTradeRequest()**

Bi-directional positions in HedgeTerminal are closed and modified through a call of the [SendTradeRequest()](https://www.mql5.com/en/articles/1316#SendTradeRequest) function, in which the trade request is used as an argument. The request is represented by a special predefined structure HedgeTradeRequest, which contains all the fields necessary to close or modify the selected position:

```
struct HedgeTradeRequest
  {
   ENUM_REQUEST_TYPE action;             // type of action
   double            volume;             // volume of position
   ENUM_CLOSE_TYPE   close_type;         // Marker of closing order
   double            sl;                 // stop-loss level
   double            tp;                 // take-profit level
   string            exit_comment;       // outgoing comment
   uint              retcode;            // last retcode in executed operation
   bool              asynch_mode;        // true if the closure is performed asynchronously, otherwise false
   ulong             deviation;          // deviation in step price
                     HedgeTradeRequest() // default params
     {
      action=REQUEST_CLOSE_POSITION;
      asynch_mode=false;
      volume=0.0;
      sl = 0.0;
      tp = 0.0;
      retcode=0;
      deviation=3;
     }
  };
```

_**Fields description**_

| Field | Description |
| --- | --- |
| action | The type of the required action with the position. The value can be any of the ENUM\_REQUEST\_TYPE enumeration values |
| volume | The volume to close. Can be less than the volume of the currently active position. If the volume is zero, the active position will be closed completely. |
| sl | The stop loss level to be placed for the active position. |
| tp | The take profit level to be placed for the active position. |
| exit\_comment | The outgoing comment for the active position. |
| retcode | Result code of the last executed operation. |
| asynch\_mode | True if the asynchronous mode for sending requests is used, false otherwise. |
| deviation | Maximum deviation from the used price. |

**2.6. Enumerations for Working with Transaction Selection Functions**

**ENUM\_TRANS\_TYPE**

All transactions available for the analysis, including pending orders and bi-directional positions, are on the list of active and historical transactions.

The ENUM\_TRANS\_TYPE enumeration contains the type of each selected transaction. This enumeration is returned by the [TransactionType()](https://www.mql5.com/en/articles/1316#TransactionType) function. Below are the enumeration fields and their descriptions:

| Field | Description |
| --- | --- |
| TRANS\_NOT\_DEFINED | The transaction is not selected by the TransactionSelect() function or its type is undefined. |
| TRANS\_HEDGE\_POSITION | The transaction is a bi-directional position. |
| TRANS\_BROKERAGE\_DEAL | The transaction is a broker's deal (account operation). For example, adding money to the account or correction. |
| TRANS\_PENDING\_ORDER | The transaction is a pending order. |
| TRANS\_SWAP\_POS | The transaction is a swap charged for a net position. |

**ENUM\_MODE\_SELECT**

The enumeration defines the type of the index parameter set in the [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect) function.

| Field | Description |
| --- | --- |
| SELECT\_BY\_POS | The index parameter is used to pass the index of the transaction in the list. |
| SELECT\_BY\_TICKET | The ticket number is passed in the index parameter. |

**ENUM\_MODE\_TRADES**

The enum defines the data source, from which a transaction is selected using [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect).

| Field | Description |
| --- | --- |
| MODE\_TRADES | The transaction is selected from active transactions. |
| MODE\_HISTORY | The transaction is selected from historical transactions. |

**2.7. Enumerations for Working with the Functions that Get Transaction Properties**

****Enumeration**ENUM\_TRANS\_DIRECTION**

Every transaction, no matter whether it's a deal or a bi-directional position, has a market direction.

This market direction is defined by the ENUM\_TRANS\_DIRECTION enumeration. Below are its fields and their descriptions:

| Field | Description |
| --- | --- |
| TRANS\_NDEF | The direction of a transaction is undefined. For example, broker's transactions on the account do not have market direction and come with this modifier. |
| TRANS\_LONG | Indicates that the transaction (order or bi-directional position) is a Buy transaction. |
| TRANS\_SHORT | Indicates that the transaction (order or bi-directional position) is a Sell transaction. |

****Enumeration**ENUM\_HEDGE\_POSITION\_STATUS**

The enumeration contains the status of a bi-directional position.

| Field | Description |
| --- | --- |
| HEDGE\_POSITION\_ACTIVE | An active position. Active positions appear on the Active tab of the HedgeTerminal panel. |
| HEDGE\_POSITION\_HISTORY | A historical position. Historical positions appear on the History tab of the HedgeTerminal panel. |

****Enumeration**ENUM\_HEDGE\_POSITION\_STATE**

The enumeration contains the state of a bi-directional position.

| Field | Description |
| --- | --- |
| POSITION\_STATE\_ACTIVE | The selected position is active and can be modified using HedgeTradeRequest. |
| POSITION\_STATE\_FROZEN | The selected position is locked and cannot be modified. If this modifier is received, one should wait until the position is unlocked. |

****Enumeration**ENUM\_HEDGE\_POSITION\_PROP\_INTEGER**

The enumeration sets the type of the property returned by [HedgePositionGetInteger()](https://www.mql5.com/en/articles/1316#HedgePositionGetInteger).

| Field | Description |
| --- | --- |
| HEDGE\_POSITION\_ENTRY\_TIME\_SETUP\_MSC | The time in milliseconds since 01.01.1970, when the order initiating the bi-directional position was placed. |
| HEDGE\_POSITION\_ENTRY\_TIME\_EXECUTED\_MSC | The time in milliseconds since 01.01.1970, when the order initiating the bi-directional position was executed (position opening time). |
| HEDGE\_POSITION\_EXIT\_TIME\_SETUP\_MSC | The time in milliseconds since 01.01.1970, when the order to close the bi-directional position was placed. |
| HEDGE\_POSITION\_EXIT\_TIME\_EXECUTED\_MSC | The time in milliseconds since 01.01.1970, when the order to close the bi-directional position was executed (position closing time). |
| HEDGE\_POSITION\_TYPE | The type of the bi-directional position. Equal to the type of the initiating order. Contains one of the values of the system enumeration [ENUM\_ORDER\_TYPE](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties). |
| HEDGE\_POSITION\_DIRECTION | Position direction. Defined by the enumeration ENUM\_TRANS\_DIRECTION. |
| HEDGE\_POSITION\_MAGIC | The magic number of the Expert Advisor to which the selected position belongs. A value of zero indicates that the position was opened manually. |
| HEDGE\_POSITION\_CLOSE\_TYPE | The marker of the order closing the position. Defined by [ENUM\_CLOSE\_TYPE](https://www.mql5.com/en/articles/1316#ENUM_CLOSE_TYPE). |
| HEDGE\_POSITION\_ID | Position ID. Equal to the identifier of the initiating order. |
| HEDGE\_POSITION\_ENTRY\_ORDER\_ID | The identifier of the initiating order. |
| HEDGE\_POSITION\_EXIT\_ORDER\_ID | The identifier of a closing order for a historical position. |
| HEDGE\_POSITION\_STATUS | Position status. Defined by [ENUM\_HEDGE\_POSITION\_STATUS](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_STATUS). |
| HEDGE\_POSITION\_STATE | Position state. Defined by [ENUM\_HEDGE\_POSITION\_STATE](https://www.mql5.com/en/articles/1316#ENUM_HEDGE_POSITION_STATE). |
| HEDGE\_POSITION\_USING\_SL | The flag of stop loss use. If a stop loss is used, then the HedgePositionGetInteger() function returns true, otherwise false. |
| HEDGE\_POSITION\_USING\_TP | The flag of a used take profit level. If a take profit is used, [HedgePositionGetInteger()](https://www.mql5.com/en/articles/1316#HedgePositionGetInteger) returns true, otherwise it returns false. |
| HEDGE\_POSITION\_TASK\_STATUS | The status of the task that is being performed for the selected position. The position can be under modification. This modifier is used to track changes in this position. The position status is defined by [ENUM\_TASK\_STATUS](https://www.mql5.com/en/articles/1316#ENUM_TASK_STATUS). |
| HEDGE\_POSITION\_ACTIONS\_TOTAL | Returns the total number of subtasks started to change this position. |

****Enumeration**ENUM\_HEDGE\_POSITION\_PROP\_DOUBLE**

The enumeration sets the type of the property returned by the [HedgePositionGetDouble()](https://www.mql5.com/en/articles/1316#HedgePositionGetDouble) function.

| Field | Description |
| --- | --- |
| HEDGE\_POSITION\_VOLUME | The volume of the bi-directional position. |
| HEDGE\_POSITION\_PRICE\_OPEN | The weighted average open price of a position. |
| HEDGE\_POSITION\_PRICE\_CLOSED | The weighted average close price of a position. |
| HEDGE\_POSITION\_PRICE\_CURRENT | The current price of an active position. For a historical position, this modifier returns the position close price. |
| HEDGE\_POSITION\_SL | The stop loss level. Zero if stop loss is not used. |
| HEDGE\_POSITION\_TP | The take profit level. Zero if take profit is not used. |
| HEDGE\_POSITION\_COMMISSION | The amount of commission paid for the position. |
| HEDGE\_POSITION\_SLIPPAGE | Slippage in points. |
| HEDGE\_POSITION\_PROFIT\_CURRENCY | Profit or loss of the position. The value is specified in the deposit currency. |
| HEDGE\_POSITION\_PROFIT\_POINTS | Profit or loss of the position. The value is specified in the points of the financial symbol of the position. |

_**Note**_

The slippage HEDGE\_POSITION\_SLIPPAGE is calculated as the difference in points between the best position entry deal and the average weighted entry price.

****Enumeration**ENUM\_HEDGE\_POSITION\_PROP\_STRING**

The enumeration sets the type of the property returned by the [HedgePositionGetString()](https://www.mql5.com/en/articles/1316#HedgePositionGetString) function.

| Field | Description |
| --- | --- |
| HEDGE\_POSITION\_SYMBOL | The symbol of the current position. |
| HEDGE\_POSITION\_ENTRY\_COMMENT | The incoming comment of a position. |
| HEDGE\_POSITION\_EXIT\_COMMENT | The outgoing comment of a position. |

**Enumeration ENUM\_HEDGE\_ORDER\_STATUS**

The enumeration contains order type.

| Field | Description |
| --- | --- |
| HEDGE\_ORDER\_PENDING | The order is pending and is available in the Trade tab of MetaTrader 5. |
| HEDGE\_ORDER\_HISTORY | The order is historical and is available in the history of orders in MetaTrader 5. |

****Enumeration**ENUM\_HEDGE\_ORDER\_SELECTED\_TYPE**

The enumeration defines the type of the order selected by function [HedgeOrderSelect()](https://www.mql5.com/en/articles/1316#HedgeOrderSelect()).

| Field | Value |
| --- | --- |
| ORDER\_SELECTED\_INIT | The order initiates a bi-directional position. |
| ORDER\_SELECTED\_CLOSED | The order closes a bi-directional position. |
| ORDER\_SELECTED\_SL | The order acts as a stop loss level. |

****Enumeration**ENUM\_HEDGE\_ORDER\_PROP\_INTEGER**

The enumeration sets the type of the property returned by the [HedgePositionGetInteger()](https://www.mql5.com/en/articles/1316#HedgeOrderGetInteger) function.

| Field | Description |
| --- | --- |
| HEDGE\_ORDER\_ID | A unique order identifier. |
| HEDGE\_ORDER\_STATUS | Order status. The value can be one of the values of the ENUM\_HEDGE\_ORDER\_STATUS enumeration. |
| HEDGE\_ORDER\_DEALS\_TOTAL | The total number of deals that have filled the order. The value is zero for pending orders. |
| HEDGE\_ORDER\_TIME\_SETUP\_MSC | Pending order placing time in milliseconds since 01.01.1970. |
| HEDGE\_ORDER\_TIME\_EXECUTED\_MSC | The execution time of an executed order in milliseconds since 01.01.1970. |
| HEDGE\_ORDER\_TIME\_CANCELED\_MSC | The time of cancellation of an executed order in milliseconds since 01.01.1970. |

_**Note**_

Order execution time HEDGE\_ORDER\_TIME\_EXECUTED\_MSC is equal to the time of its latest deal.

****Enumeration**ENUM\_HEDGE\_ORDER\_PROP\_DOUBLE**

The enumeration sets the type of the property returned by the [HedgeOrderGetDouble()](https://www.mql5.com/en/articles/1316#HedgeOrderGetDouble) function.

| Field | Description |
| --- | --- |
| HEDGE\_ORDER\_VOLUME\_SETUP | The volume of the order specified in the order. |
| HEDGE\_ORDER\_VOLUME\_EXECUTED | Executed volume of the order. If an order is pending, the executed volume is zero. |
| HEDGE\_ORDER\_VOLUME\_REJECTED | The volume of the order that could not be executed. Equal to the difference between the initial volume and the executed volume. |
| HEDGE\_ORDER\_PRICE\_SETUP | Order placing price. |
| HEDGE\_ORDER\_PRICE\_EXECUTED | The average weighted execution price of an order. |
| HEDGE\_ORDER\_COMMISSION | The amount of commission paid to the broker for order execution. Specified in the deposit currency. |
| HEDGE\_ORDER\_SLIPPAGE | Order slippage. |

_**Note**_

The slippage HEDGE\_ORDER\_SLIPPAGE is calculated as the difference in points between the best executed deal and the weighted average entry price of the order.

**Enumeration ENUM\_HEDGE\_DEAL\_PROP\_INTEGER**

The enumeration sets the type of the property returned by [HedgeDealGetInteger()](https://www.mql5.com/en/articles/1316#HedgeDealGetInteger).

| Field | Description |
| --- | --- |
| HEDGE\_DEAL\_ID | A unique deal identifier. |
| HEDGE\_DEAL\_TIME\_EXECUTED\_MSC | Deal execution time in milliseconds since 01.01.1970 |

**Enumeration ENUM\_HEDGE\_DEAL\_PROP\_DOUBLE**

The enumeration sets the type of the property returned by [HedgeDealGetDouble()](https://www.mql5.com/en/articles/1316#HedgeDealGetDouble).

| Field | Description |
| --- | --- |
| HEDGE\_DEAL\_VOLUME\_EXECUTED | Volume of a deal. |
| HEDGE\_DEAL\_PRICE\_EXECUTED | Deal execution price. |
| HEDGE\_DEAL\_COMMISSION | The amount of commission paid to the broker for deal execution. Specified in the deposit currency. |

**2.8. Enumerations for Setting and Getting HedgeTerminal Properties**

**Enumeration ENUM\_HEDGE\_PROP\_INTEGER**

The enumeration sets the type of the property that you want to get or set in HedgeTerminal.

| Field | Description |
| --- | --- |
| HEDGE\_PROP\_TIMEOUT | Time, in seconds, during which HedgeTerminal will wait for a response from the server before unlocking a position being modified. |

**2.9. Enumerations for Working with Error Codes Handling Functions**

**Enumeration ENUM\_TASK\_STATUS**

Every bi-directional position can be under modification. A position is modified through a trade task.

Every running trading task has its execution status defined in ENUM\_TASK\_STATUS. Below are its fields and their descriptions:

| Field | Description |
| --- | --- |
| TASK\_STATUS\_WAITING | No current task, or the task is waiting. |
| TASK\_STATUS\_EXECUTING | The trading task is currently being executed. |
| TASK\_STATUS\_COMPLETE | The trading task for the position has completed successfully. |
| TASK\_STATUS\_FAILED | The trading task for the position has failed. |

**Enumeration ENUM\_HEDGE\_ERR**

The enumeration contains the ID of the error that can be returned by [GetHedgeError()](https://www.mql5.com/en/articles/1316#GetHedgeError).

| Field | Description |
| --- | --- |
| HEDGE\_ERR\_NOT\_ERROR | No error. |
| HEDGE\_ERR\_TASK\_FAILED | The task for the selected position has failed. |
| HEDGE\_ERR\_TRANS\_NOTFIND | Transaction not found. |
| HEDGE\_ERR\_WRONG\_INDEX | Incorrect index. |
| HEDGE\_ERR\_WRONG\_VOLUME | Incorrect volume. |
| HEDGE\_ERR\_TRANS\_NOTSELECTED | Transaction has not been preselected using [TransactionSelect()](https://www.mql5.com/en/articles/1316#TransactionSelect). |
| HEDGE\_ERR\_WRONG\_PARAMETER | One of the passed parameters is incorrect. |
| HEDGE\_ERR\_POS\_FROZEN | The bi-directional position is currently under modification and is not available for new changes. Wait till the position is released. |
| HEDGE\_ERR\_POS\_NO\_CHANGES | The trade request has no changes. |

**Enumeration ENUM\_TARGET\_TYPE**

The enumeration defines the type of the task selected by the [GetActionResult()](https://www.mql5.com/en/articles/1316#GetActionResult) function.

| Field | Description |
| --- | --- |
| TARGET\_NDEF | The subtask is undefined. |
| TARGET\_CREATE\_TASK | The subtask is being created now. This type is used in the internal logics of HedgeTerminalAPI. |
| TARGET\_DELETE\_PENDING\_ORDER | Deleting a pending order. |
| TARGET\_SET\_PENDING\_ORDER | Placing of a pending order. |
| TARGET\_MODIFY\_PENDING\_ORDER | Modification of the pending order price. |
| TARGET\_TRADE\_BY\_MARKET | Making trading operations. |

**2.10. Enumerations for Working with Error Codes Handling Functions**

**Enumeration ENUM\_REQUEST\_TYPE**

The enumeration describes the action of HedgeTerminal applied to the bi-directional position.

| Field | Description |
| --- | --- |
| REQUEST\_CLOSE\_POSITION | Closes the position. If the volume field of the HedgeTradeRequest structure contains a volume below the current one, only a part of the position will be closed. In this case, the part of the closed position corresponds to the value of the volume field. |
| REQUEST\_MODIFY\_SLTP | Sets or modifies the existing levels of stop loss and take profit. |
| REQUEST\_MODIFY\_COMMENT | Modifies the outgoing comment of an active position. |

**Enumeration ENUM\_CLOSE\_TYPE**

The enumeration defines a special marker for the order closing the bi-directional position. The marker indicates the reason for position closing. It can be one of the following reasons:

- The position has reached the maximum loss level or stop loss;
- The position has reached a certain profit level or take profit;
- Position closed by market. The stop loss and take profit levels were not placed or reached.

| Field | Description |
| --- | --- |
| CLOSE\_AS\_MARKET | Indicates that the position is closed by market. The stop loss and take profit levels were not placed or reached. |
| CLOSE\_AS\_STOP\_LOSS | Indicates that the position is closed due to reaching the stop loss level. |
| CLOSE\_AS\_TAKE\_PROFIT | Indicates that the position is closed due to reaching the take profit level. |

### Chapter 3. The Fundamentals of Asynchronous Trading

The subject of asynchronous operations is complex and requires a separate detailed article. However, due to the fact that HedgeTerminal actively uses asynchronous operations, it is appropriate to briefly describe the principles of organization of Expert Advisors using this type of request submission. In addition, there are almost no materials about the subject.

**3.1. Organization and Scheme of Sending a Synchronous Trading Order**

MetaTrader 5 provides two functions for sending trade requests to the server:

- [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend);
- [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync).

The [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function accepts a request as a filled [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) structure and performs basic verification of the structure correctness. If the basic verification is successful, it sends the request to a server, waits for its result, and then returns the result to the custom thread through the [MqlTradeResult](https://www.mql5.com/en/docs/constants/structures/mqltraderesult) structure and the return flag. If the basic verification fails, the function returns a negative value.

The reason why the request could not be verified is also included in MqlTradeResult.

The below scheme features the execution of the thread of a custom MQL5 program with the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function:

![Fig. 6. The scheme of organization and sending of a synchronous trade request](https://c.mql5.com/2/12/OrderSend.png)

Fig. 6. The scheme of organization and sending of a synchronous trade request.

As seen from the scheme, the thread of the MQL5 program cannot be separated from the common system thread sending a request to the server and executing trade operations on the exchange.

That is why, after completion of [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend), we can analyze the actual result of the trade request. The custom thread is marked by red arrows. It is executed almost instantly. Most of the time is taken to perform trading operations on the exchange. Since the two threads are connected, considerable amount of time passes between the beginning and end of the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function. Due to the fact that the trading operations are executed in a single thread, the logic of MQL5-programs can be sequential.

**3.2. Organization and Scheme of Sending an Asynchronous Trading Order**

The [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) function is different. Like [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend), it accepts the trade request [MqlTradeRequest](https://www.mql5.com/en/docs/constants/structures/mqltraderequest) and returns a flag indicating its result.

However, unlike the first example, it does not wait for the trade request to be executed by the server, but returns the values ​​obtained only from the module of basic verification of trade request values (Basic verification inside the terminal). The below scheme shows the procedure of custom thread execution when using the [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) function:

![Fig. 7. The scheme of organization and sending of an asynchronous trade request.](https://c.mql5.com/2/12/OrderSendAsynch.png)

Fig. 7. The scheme of organization and sending of an asynchronous trade request.

Once a trade request is successfully verified, it is sent to the trading server parallel to the main thread. Passing of a trade request over the network as well as its execution on the exchange takes some time, like in the first case. But the custom thread will get almost an instant result from the [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) function.

The above scheme shows that [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) actually forms a new parallel thread which is executed by a trade server, and its execution result gets into the [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) or [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade) function. These functions begin a new custom thread. The result of sending a trade request should be processed in this new thread. This greatly complicates the logic of the Expert Advisor, because with asynchronous order sending, it is impossible to organize sending of the request and its checking in a single thread. For example, you cannot sequentially place the code for sending and checking a request in [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick).

Let's write a simple test EA to illustrate the above:

```
//+------------------------------------------------------------------+
//|                                                    AsynchExp.mq5 |
//|                           Copyright 2014, Vasiliy Sokolov (C-4). |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2014, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
input bool UsingAsynchMode=true;
bool sendFlag=false;
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(sendFlag)return;
   printf("Formation of order and send to the server...");
   MqlTradeRequest request={0};
   request.magic=12345;
   request.symbol = Symbol();
   request.volume = 0.1;
   request.type=ORDER_TYPE_BUY;
   request.comment= "asynch test";
   request.action = TRADE_ACTION_DEAL;
   request.type_filling=ORDER_FILLING_FOK;
   MqlTradeResult result;
   uint tiks= GetTickCount();
   bool res = false;
   if(UsingAsynchMode)
      res=OrderSendAsync(request,result);
   else
      res=OrderSend(request,result);
   uint delta=GetTickCount()-tiks;
   if(OrderSendAsync(request,result))
     {
      printf("The order has been successfully"+
             "sent to the server.");
     }
  else
     {
     printf("The order is not shipped."+
             " Reason: "+(string)result.retcode);
     }
   printf("Time to send a trade request: "+(string)delta);
   sendFlag=true;
//---
  }
```

Let's make sure that the EA works by starting it with UsingAsynchMode = false.

The EA opens a long 0.1-lot position. The trade request is performed synchronously using the [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) function. Here is its sample log:

```
2014.11.06 17:49:28.442 AsynchExp (AUDCAD,H1)   Time to send a trade request: 94
2014.11.06 17:49:28.442 AsynchExp (AUDCAD,H1)   The order has been successfullysent to the server.
2014.11.06 17:49:28.345 AsynchExp (AUDCAD,H1)   Formation of order and send to the server...
```

The trade request was completed within 94 milliseconds. This time tells us that the request passed the basic verification, was sent to the server, and then was filled.

Now we modify the EA code by changing the transaction volume to the maximum possible value [DBL\_MAX](https://www.mql5.com/en/docs/constants/namedconstants/typeconstants):

```
request.volume = DBL_MAX;
```

Obviously, this value is outside the actual range. Let's try to execute this request in the synchronous mode:

```
2014.11.06 17:54:15.373 AsynchExp (AUDCAD,H1)   Time to send a trade request: 0
2014.11.06 17:54:15.373 AsynchExp (AUDCAD,H1)   The order is not shipped. Reason: 10014
2014.11.06 17:54:15.373 AsynchExp (AUDCAD,H1)   Formation of order and send to the server...
```

Sending a request failed. The reason for the failure is [error 10014](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes) (Invalid requested volume). The request failed during the basic verification and was not even sent to the server, as clear from the request execution time of 0 milliseconds.

Again, let's change the request. This time we specified a large enough volume, but not an extreme value – 15 lots. For the account of $1,000 where the EA is tested, it is too much. Such a position cannot be opened on this account.

Let's see what [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend) returns:

```
2014.11.06 17:59:22.643 AsynchExp (AUDCAD,H1)   Time to send a trade request: 78
2014.11.06 17:59:22.643 AsynchExp (AUDCAD,H1)   The order is not shipped. Reason: 10019
2014.11.06 17:59:22.550 AsynchExp (AUDCAD,H1)   Formation of order and send to the server...
```

The error is different this time: 10019 (Insufficient funds for request execution, which is true). Note that the request execution time is now _79 milliseconds_. It indicates that the request was sent to the server, and that the server returned an error.

Let's now send the same request with the 15-lot volume using the [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) function. Like with [OrderSend()](https://www.mql5.com/en/docs/trading/ordersend), no position is opened. But let's analyze the log:

```
2014.11.06 18:03:58.106 AsynchExp (AUDCAD,H1)   Time to send a trade request: 0
2014.11.06 18:03:58.106 AsynchExp (AUDCAD,H1)   The order has been successfully sent to the server.
2014.11.06 18:03:58.104 AsynchExp (AUDCAD,H1)   Formation of order and send to the server...
```

The log tells there is no error! Since the error 10019 is detected by the trading server, it is not available for the current thread in the asynchronous order sending mode. The return value only indicates that the request passed the basic verification. In order to get the actual error 10019, we need to analyze the results in a new custom thread, in the [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) system function, which should be added to our EA:

```
void  OnTradeTransaction(const MqlTradeTransaction    &trans,
                         const MqlTradeRequest        &request,
                         const MqlTradeResult         &result)
  {
   uint delta = GetTickCount() - tiks;
   printf("Server answer: " + (string)result.retcode + "; Time: " + (string)delta);
  }
```

Let's run the EA again and see logs:

```
2014.11.06 18:17:00.943 AsynchExp (AUDCAD,H1)   Server answer: 10019; Time: 94
2014.11.06 18:17:00.854 AsynchExp (AUDCAD,H1)   Time to send a trade request: 0
2014.11.06 18:17:00.854 AsynchExp (AUDCAD,H1)   The order has been successfully sent to the server.
2014.11.06 18:17:00.851 AsynchExp (AUDCAD,H1)   Formation of order and send to the server...
```

Error 10019 was received, but not immediately after sending. It was received in the new custom thread running in [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction).

**3.3. Asynchronous Order Execution Speed**

Traders mistakenly believe that the execution speed of an asynchronous request is close to zero.

It stems from the observation of [OrderSendAsync()](https://www.mql5.com/en/docs/trading/ordersendasync), which is completed as a rule in less than one millisecond. In reality, as has been shown above, the actual time of execution of the trade transaction should be measured when a response from the serve is received inside the functions [OnTradeTransaction()](https://www.mql5.com/en/docs/basis/function/events#ontradetransaction) or [OnTrade()](https://www.mql5.com/en/docs/basis/function/events#ontrade). This measurement shows the real speed, which is equal to the speed of a synchronous execution for a single order. Real advantages in execution time are perceptible when sending a group of transactions. There are at least three situations where you need to send multiple requests:

- The required time between two successive requests is so small that there is no possibility to check the request result before sending the next one. When the next request is sent, it is hoped that the previous one has been executed. Similar tactics are used in high frequency trading;
- You need to open multiple positions for multiple symbols at a time. For example, arbitrage strategies and composite synthetic positions require the simultaneous opening of positions for various instruments at current prices. The gradual formation of positions is undesirable in such tactics;
- It is required to complete the thread as soon as possible and wait for further events and user commands. This requirement is important for multi-threaded and infrastructure solutions. This is the main reason why HedgeTerminal uses asynchronous requests. If HT used synchronous sending of requests, it would constantly freeze for 1-2 seconds each time the user closes or modifies the position, which it unacceptable.

Remember to take into account the limit for sending requests when placing multiple orders.

In MetaTrader 5 build 1010 and higher, the limit is 64 transactions, 4 of which are reserved for users, and others are available to Expert Advisors. The limit is aimed at protecting novice traders from serious errors in their programs, as well as reducing the spam load on a trading server.

This means that at the same time, for example in the loop for, you can send up to 60 trade orders by calling [SendOrderAsync()](https://www.mql5.com/en/docs/trading/ordersendasync) with an appropriate trade request. After all the 60 transactions are sent, the transaction buffer will be full. We need to wait for confirmation from the server that one of the transactions has been processed by the server.

After being handled, the place of a transaction in the buffer of transactions is released, and a new trade request can take it. Once the buffer is full, the space for new transactions is released slowly, because a trade server needs time to process each transaction, and the [TradeTransaction()](https://www.mql5.com/en/docs/runtime/event_fire#tradetransaction) event notifying of the start of processing is passed over the network, which causes additional delays.

Thus, the time required to send requests will grow nonlinearly compared to the growth of the number of requests. The table below features the estimated rates of order sending in the asynchronous mode. Tests were carried out several times, and the shown rate is the mean value:

| Number of requests | Time, milliseconds |
| --- | --- |
| 50 | 50 |
| 100 | 180 |
| 200 | 2100 |
| 500 | 9000 |
| 1000 | 23000 |

In the case where the number of requests is less than 60, the script does not wait for the server response, that's why the time is so small. It is approximately equal to the time it takes to send a single request. In fact, to get an approximate real execution time, add the average request execution time to the request placing time specified in the table.

### Chapter 4. The Fundamentals of Multi-Threaded Programming in the MetaTrader 5 IDE

MQL5 programmers know that threads cannot be controlled directly from MQL-programs. This restriction is for the good of novice programmers, because the use of threads greatly complicates the program algorithms. However, in some situations, two or more EAs must communicate with each other, for example, they must create and read global data.

HedgeTerminal is one of such EAs. To inform every EA using the HedgeTerminalAPI library about the actions of other Expert Advisors, the HT organizes data exchange through multi-threaded reading and writing of the ActivePositions.xml file. This solution is non-trivial and is rarely used by MQL programmers. Therefore, we will create a multi-threaded EA with the algorithm similar to HedgeTerminal. This will help to better understand the multi-threaded programming, and thus better understand how HedgeTerminal works.

**4.1. Multithreaded Programming through the Example of Quote Collector UnitedExchangeQuotes**

We will learn the basics of multi-threaded programming through a specific example: we'll write a collector of quotes from different providers (brokers).

The idea is this: suppose we have 6-7 brokers that provide quotes for the same instrument. Naturally, quotes from different brokers may vary slightly. The analysis of these differences opens the way to arbitrage strategies. In addition, comparison of quotes dynamics will help identify the best and worst provider. For example, if a broker is providing better prices, we'd rather select this broker to trade with. We are not hunting for the practical value of the results, instead we only describe the mechanism by which these results can be achieved.

Here's a screenshot of the EA that we will have to write by the end of this chapter:

![Fig. 8. The appearance of the quote collector UnitedExhangesQuotes.](https://c.mql5.com/2/12/UnitedExchangeQuotes.png)

Fig. 8. The appearance of the quote collector UnitedExhangesQuotes.

The Expert Advisor displays the results in a simple table consisting of four columns and an unlimited number of rows.

Each row represents a broker providing symbol quotes (in this case, EURUSD). Ask and Bid are the best offer and demand of the broker. The screenshot shows that prices slightly differ. The difference between the offer of the current broker and another one appears in the D-ASK (Delta Ask) column. Similarly, the difference between the demand values is displayed in D-BID (Delta Bid). For example, at the time the screenshot was taken, the best Ask was provided by "Alpari Limited", and the most expensive was that of "Bank VTB 24".

MQL programs cannot access the environment of other MetaTrader terminals. In other words, if a program is running on one terminal, it cannot receive data from another one. However, all MQL programs can communicate through the files in the shared directory of the MetaTrader terminals. If any program writes information, e.g. the current quote, to a file, the MQL program from another terminal can read it. MQL does not have any other means without external DLLs. Therefore, we will use this method.

The greatest difficulty is to organize such access. On the one hand, the EA has to read quotes from other providers, and on the other - to write the quote of its provider to the same file. Another problem is the fact that at the time of reading quotes, another EA can be writing a new quote to this file. The result of such parallel work is unpredictable. At best, this will be followed by a crash and program interruption, and at worst this will lead to occasional emergence of strange subtle errors associated with the display of quotes.

To eliminate these errors, or at least minimize the probability of their occurrence, we will develop a clear plan.

Firstly, all the information will be stored in the XML format. This format has replaced the clumsy ini file. XML allows flexible deployment of its nodes into complex data structures, such as classes. Next, let's determine the general algorithm of reading and writing. There are two basic operations: reading data and writing data. When any MQL program is reading or writing, no other program can access this file. Thus we eliminate a situation where one program reads the data, and the second one changes them. Due to this, access to data will not always be possible.

Let's create a special class CQuoteList that will contain the XML access algorithms, as well as data of all quotes from this file.

One of the functions of this class is TryGetHandle(), it tries to access the file and returns its handle in case of success. Here is the implementation of the function:

```
int CQuoteList::TryGetHandle(void)
{
   int attempts = 10;
   int handle = INVALID_HANDLE;
   // We try to open 'attemps' times
   for(att = 0; att < attempts; att++)
   {
      handle = FileOpen("Quotes.xml", FILE_WRITE|FILE_READ|FILE_BIN|FILE_COMMON);
      if(handle == INVALID_HANDLE)
      {
         Sleep(15);
         continue;
      }
      break;
   }
   return handle;
}
```

It makes several attempts to open the file in a combined read/write mode. The default number of attempts is ten.

If an attempt is not successful, the function freezes for 15 milliseconds and retries to open the file, thus making up to 10 attempts.

Once the file is opened, its handle is passed to the LoadQuotes() function. A complete listing of this function, and the CQuoteList class are available as an attachment to the article. So here we describe only the sequence of the actions in the function:

1. TryGetHandle() opens the file to read and write;
2. The XML document is uploaded to the EA memory using the [XML Parser](https://www.mql5.com/en/code/712) library;
3. Based on the uploaded XML document, a new array of quotes is formed storing the required information;
4. The created array contains a quote belonging to the current EA. Its values ​​are updated;
5. The array of quotes is converted back into an XML document. The contents of the open XML file are replaced with this XML document;
6. The XML file of quotes is closed.

The LoadQuotes() function does a great job, but in most cases it takes less than 1 millisecond.

Data reading and data updating with their further saving are combined into one block. This is done on purpose, so as not to lose control of access to the file between the operations of reading and writing.

Once quotes are loaded and are inside a class, they can be accessed just like any other data in MetaTrader 5. This is done through a special MetaTrader 5 -style program interface implemented in the CQuotesList class.

Function call and data rendering are performed inside the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) block. Here are the contents:

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!AccountInfoInteger(ACCOUNT_TRADE_EXPERT))
      return;
   if(!QuotesList.LoadQuotes())
      return;
   PrintQuote quote = {0};
   Panel.DrawAccess(QuotesList.CountAccess());
   double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
   double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
   string brokerName = AccountInfoString(ACCOUNT_COMPANY);
   for(int i = 0; i < QuotesList.BrokersTotal(); i++)
   {
      if(!QuotesList.BrokerSelect(i))
         continue;
      if(!QuotesList.SymbolSelect(Symbol()))
         continue;
      quote.ask = QuotesList.QuoteInfoDouble(QUOTE_ASK);
      quote.bid = QuotesList.QuoteInfoDouble(QUOTE_BID);
      quote.delta_ask = ask - quote.ask;
      quote.delta_bid = quote.bid - bid;
      quote.broker_name = QuotesList.BrokerName();
      quote.index = i;
      Panel.DrawBroker(quote);
   }
  }
```

It is noteworthy that the sample code works both in MetaTrader 4 and in MetaTrader 5 without any additional modifications!

There are only minor cosmetic differences in the way the panels are displayed in different versions of the terminal. Undoubtedly, this is a remarkable fact that facilitates code porting between platforms.

The operation of the EA is best observed in dynamics. The below video shows the EA operation on different accounts:

UnitedEchamgeQuotes 1 - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1316)

MQL5.community

1.91K subscribers

[UnitedEchamgeQuotes 1](https://www.youtube.com/watch?v=3mnoXaS4sPw)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

You're signed out

Videos you watch may be added to the TV's watch history and influence TV recommendations. To avoid this, cancel and sign in to YouTube on your computer.

CancelConfirm

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=3mnoXaS4sPw&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1316)

0:00

0:00 / 1:46

•Live

•

Reading and writing to a file have significant advantages, but there are also some disadvantages.

The main advantages are:

1. Flexibility. You can store and load any data, even whole classes;
2. Relatively high speed. The entire cycle of reading and rewriting almost always takes mo more than 1 millisecond, which is a good time as compared to relatively slow trading operations that take 80 - 150 milliseconds or sometimes even more;
3. Based on the standard tools of the MQL5 language without calling DLL.

The main drawback of such a solution is a serious load on the storage system. When there is one quote and two brokers, the number of rewrite operations is relatively small, but with a heavy stream of quotes and a large number of brokers/symbols, the number of rewrite operations becomes very large. In less than one hour, the demo EA produced over 90,000 Quotes.xml file rewriting operations. These statistics is shown at the top of the EA panel: "I/O Rewrite" shows the total number of file rewrites, "fps" indicates the rate between the last two rewrite operations, and "Avrg" displays the average speed of rewrites per second.

If you store files on SSD or HDD, these operation will have a negative impact on the disk lifetime. Therefore it is better to use a virtual RAM disk for such a data exchange.

Unlike the above example, HedgeTerminal sparingly uses ActivePositions.xml, writing only significant position changes that are inaccessible through the global context. So it produces much fewer read/write operations than the above example, and therefore does not require any special conditions, such as RAM disks.

**4.2. Use of Multi-Threaded Interaction between Expert Advisors**

Real time interaction between independent MQL programs is a complicated but interesting subject. The article contains only a very brief description of it, but it is worthy of a separate article. In most cases the multi-threaded interaction between Expert Advisors is not required. However, here is a list of tasks and the variety of programs, for which the organization of such interaction is required:

- **Trade copier.** Any trade copier involves simultaneous launch of at least two EAs, one of which provides trades, the other one copies them. In this case, it is necessary to organize multi-threaded reading/writing of a common data file for providing and copying trades;
- **Organization of global data exchange between EAs, global variables.** Standard global variables in MetaTrader 5 are available to Expert Advisors only at the level of one terminal. A global variable declared in one terminal is not available in the other. However, through the use of common data, you can organize complex global variables that might be available for all terminals even of different versions;
- **Arbitrage strategies. Analyzers of quotes from different liquidity providers.** If the difference between prices provided by different brokers is significant, traders can benefit from this by creating arbitrage strategies. Analyzers also allow gathering statistics of the best prices and objectively identify the best liquidity provider.

### Description of Attachments

Here is a brief description of files attached to the article, as well as of the compilation procedure.

**Prototypes.mqh** is a file with the description of the HedgeTerminalAPI library functions. This file contains the description and prototypes of the functions from the [HedgeTerminalAPI](https://www.mql5.com/en/market/product/5096#) library. It lets your EA know what functions and modifiers are available in the library, how to call the functions, and what values they return.

Save this file to C:\\Program Files\\MetaTrader 5\\MQL5\\Include, where " _C:\\Program Files\\MetaTrader 5_\\" is the name of the directory where your MetaTrader 5 terminal is installed. Once the file is copied to the correct directory, you can _refer_ to it in your MQL program. This should be done whenever you need to use the HedgeTerminalAPI library. To refer to the Prototypes.mqh file, add the special file include directory into your code:

```
#include <Prototypes.mqh>
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   //...
   // Here is the content of your program.
   //...
  }
```

In the above example, this directive is marked in yellow and is called " _#include <Ptototypes.mqh>_". Now the script above can refer to the library functions and use their functionality.

Please note that in the process of development of the HedgeTerminalAPI library, the file of the prototypes may undergo minor changes. Often, with the update of the library version, you will need to update the prototype file, which will describe the changes. Please take this uncomfortable factor with understanding. In any case, the latest version of the prototype file can always be installed manually from the library (the installation procedure is described in section [1.1](https://www.mql5.com/en/articles/1316#c1_1)) or downloaded from the attachment to this article (regular updates for the attachments are expected).

**Chaos2.mqh** is a source code of the Chaos2 EA. Its operation is described in section 1.12: " [The example of the SendTradeRequest function and the HedgeTradeRequest function through the example of Chaos II EA](https://www.mql5.com/en/articles/1316#c1_12). To successfully compile the code, save the file of function prototypes to the corresponding directory \\Include and save the HedgeTerminalAPI library to: C:\\Program Files\\MetaTrader 5\\MQL5\\Market\\hedgeterminalapi.ex5. Where " _C:_\ _Program Files_\ _MetaTrader 5_\\" is the name of the directory (terminal data folder) where your MetaTrader 5 terminal is installed.

**The UnitedExchangeQuotes source code** is the special zip archive (unitedexchangequotes.zip) that contains the project described in detail in chapter 4: " [The fundamentals of multi-threaded programming in the MetaTrader 5 IDE](https://www.mql5.com/en/articles/1316#chapter4)". This zip contains the following files:

- **UnitedExchangeQuotes.mq5** \- the central file of the EA. Save it to the experts folder: \\MetaTrader 5\\MQL5\\Experts. Compile this file in MetaEditor.
- **MultiThreadXML.mqh** is the main file containing algorithms of multi-threaded access to the XML file. It organizes the information exchange between independent threads. Save to \\MetaTrader 5\\MQL5\\Include. The algorithms in this file are based on the special library developed by [ya-sha](https://www.mql5.com/en/users/yu-sha), available in [CodeBase](https://www.mql5.com/en/code/97). However, it has been slightly modified for multi-threaded operation. The attachment contains this modified version. It consists of the following files:

  - XmlBase.mqh;
  - XmlDocument.mqh;
  - XmlAttribute.mqh;
  - XmlElement.mqh.

Save these files to the \\Include folder.
- **Panel.mqh** contains the panel class described in the example. Save this file into the same directory where you save UnitedEchangesQuotes.mqh, i.e. to the \\Experts folder.

All files in the archive contain relative paths. For example, file UnitedExchangeQuotes.mq5 is located in folder \\MQL5\\Experts. This means that it should be placed in the same subdirectory of the MetaTrader 5 terminal data folder, such as C:\\Program Files\\MetaTrader 5\\MQL5\\Experts\\UnitedExchangeQuotes.mq5.

### Conclusion

We have considered the details of working with the HedgeTerminal program interface.

It has been shown that the principles of this library are very much similar to MetaTrader 4 API. Like in MetaTrader 4 API, before you start working with a transaction (an analogue of the "order" concept in MetaTrader 4), you should first select it using the TransactionSelect(). A transaction in Hedge Terminal is a bi-directional position as a rule. Once a position is selected, you can get its properties or apply a trading action to it, for example, set a stop loss level or close it. This sequence of actions is almost identical to the algorithm of working with orders in MetaTrader 4.

In addition to basic information about the number of bi-directional positions and their properties, HedgeTerminal provides access to the values ​​that are not available directly in MetaTrader 5 and require complex analytical calculations. For example, you can see the amount of slippage of each bi-directional position by requesting only one of its properties. You can check the number of deals inside a selected position. All such calculations and the required matching of deals are performed "behind the scenes" during start of HedgeTerminal. It is convenient because the trading Expert Advisor does not need to calculate anything. All the necessary information has been already calculated and is available through a simple and intuitive API.

The use of the common algorithms by the HedgeTerminal API and the panel enables unified data presentation. Therefore, you can control EAs from the HedgeTerminal panel, while changes made by the EA will be displayed straight on the panel.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1316](https://www.mql5.com/ru/articles/1316)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1316.zip "Download all attachments in the single ZIP archive")

[Prototypes.mqh](https://www.mql5.com/en/articles/download/1316/prototypes.mqh "Download Prototypes.mqh")(15.3 KB)

[Chaos2.mq5](https://www.mql5.com/en/articles/download/1316/chaos2.mq5 "Download Chaos2.mq5")(23.56 KB)

[unitedexchangequotes.zip](https://www.mql5.com/en/articles/download/1316/unitedexchangequotes.zip "Download unitedexchangequotes.zip")(14.09 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing graphical interfaces based on .Net Framework and C# (part 2): Additional graphical elements](https://www.mql5.com/en/articles/6549)
- [Developing graphical interfaces for Expert Advisors and indicators based on .Net Framework and C#](https://www.mql5.com/en/articles/5563)
- [Custom Strategy Tester based on fast mathematical calculations](https://www.mql5.com/en/articles/4226)
- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)
- [Universal Expert Advisor: CUnIndicator and Use of Pending Orders (Part 9)](https://www.mql5.com/en/articles/2653)
- [Implementing a Scalping Market Depth Using the CGraphic Library](https://www.mql5.com/en/articles/3336)
- [Universal Expert Advisor: Accessing Symbol Properties (Part 8)](https://www.mql5.com/en/articles/3270)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/42842)**
(6)


![Little---Prince](https://c.mql5.com/avatar/avatar_na2.png)

**[Little---Prince](https://www.mql5.com/en/users/little---prince)**
\|
16 Nov 2015 at 08:23

WeChat free experience shouting, another profit more than a thousand points of QQ group: 375124107, plus group please note "77", thank you for cooperation!

![fxmatte](https://c.mql5.com/avatar/avatar_na2.png)

**[fxmatte](https://www.mql5.com/en/users/fxmatte)**
\|
15 Oct 2018 at 20:43

Hi there,

thank you for the article and the hard work you put in. Quick question: Is the code maintained and are you continue developiing the framework?

Cheers,

Matt

![Kira27](https://c.mql5.com/avatar/avatar_na2.png)

**[Kira27](https://www.mql5.com/en/users/kira27)**
\|
7 Feb 2021 at 15:36

I'm surprised this article didn't resonate)

![Kira27](https://c.mql5.com/avatar/avatar_na2.png)

**[Kira27](https://www.mql5.com/en/users/kira27)**
\|
7 Feb 2021 at 15:37

Thanks for it!!! it's awesome written!!!

![Kira27](https://c.mql5.com/avatar/avatar_na2.png)

**[Kira27](https://www.mql5.com/en/users/kira27)**
\|
7 Feb 2021 at 17:10

**Bongioanni:**

What is the point of a fifth metatrader if there is a fourth? What's its advantage?

And classes? You can't program them in 4, can you?

![Trading Ideas Based on Prices Direction and Movement Speed](https://c.mql5.com/2/18/zbm4cy.png)[Trading Ideas Based on Prices Direction and Movement Speed](https://www.mql5.com/en/articles/1747)

The article provides a review of an idea based on the analysis of prices' movement direction and their speed. We have performed its formalization in the MQL4 language presented as an expert advisor to explore viability of the strategy being under consideration. We also determine the best parameters via check, examination and optimization of an example given in the article.

![Studying the CCanvas Class. How to Draw Transparent Objects](https://c.mql5.com/2/17/CCanvas_class_Standard_library_MetaTrader5.png)[Studying the CCanvas Class. How to Draw Transparent Objects](https://www.mql5.com/en/articles/1341)

Do you need more than awkward graphics of moving averages? Do you want to draw something more beautiful than a simple filled rectangle in your terminal? Attractive graphics can be drawn in the terminal. This can be implemented through the CСanvas class, which is used for creating custom graphics. With this class you can implement transparency, blend colors and produce the illusion of transparency by means of overlapping and blending colors.

![Plotting trend lines based on fractals using MQL4 and MQL5](https://c.mql5.com/2/18/TrendLines_Fractals_Based.png)[Plotting trend lines based on fractals using MQL4 and MQL5](https://www.mql5.com/en/articles/1201)

The article describes the automation of trend lines plotting based on the Fractals indicator using MQL4 and MQL5. The article structure provides a comparative view of the solution for two languages. Trend lines are plotted using two last known fractals.

![Optimization. A Few Simple Ideas](https://c.mql5.com/2/10/DSCI2306_p28-640-480.png)[Optimization. A Few Simple Ideas](https://www.mql5.com/en/articles/1052)

The optimization process can require significant resources of your computer or even of the MQL5 Cloud Network test agents. This article comprises some simple ideas that I use for work facilitation and improvement of the MetaTrader 5 Strategy Tester. I got these ideas from the documentation, forum and articles.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1316&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062764624031426676)

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