---
title: MQL5 Cookbook: Implementing Your Own Depth of Market
url: https://www.mql5.com/en/articles/1793
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-23T11:29:22.845377
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1793&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062504241639105376)

MetaTrader 5 / Examples


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/1793#intro)
- [Chapter 1. Standard DOM in MetaTrader 5 and methods of using it](https://www.mql5.com/en/articles/1793#chapter1)
  - [1.1. Standard Depth of Market in MetaTrader 5](https://www.mql5.com/en/articles/1793#c1_1)
  - [1.2. Event model for working with Depth of Market](https://www.mql5.com/en/articles/1793#c1_2)
  - [1.3. Receiving second level quotes with MarketBookGet functions and MqlBookInfo structure](https://www.mql5.com/en/articles/1793#c1_3)
- [Chapter 2. CMarketBook class for easy access and operation with Depth of Market](https://www.mql5.com/en/articles/1793#chapter2)
  - [2.1. Designing CMarketInfoBook class](https://www.mql5.com/en/articles/1793#c2_1)
  - [2.2. Index calculation of most frequently used Depth of Market levels](https://www.mql5.com/en/articles/1793#c2_2)
  - [2.3. Predicting indexes of best Bid and Ask prices based on previous values of these indexes](https://www.mql5.com/en/articles/1793#c2_3)
  - [2.4. Determining maximum slippage with GetDeviationByVol method](https://www.mql5.com/en/articles/1793#c2_4)
  - [2.5. Examples for operating with CMarketBook class](https://www.mql5.com/en/articles/1793#c2_5)
- [Chapter 3. Writing your own Depth of Market as a panel indicator](https://www.mql5.com/en/articles/1793#chapter3)
  - [3.1. General principles of designing the Depth of Market panel. Creating an indicator](https://www.mql5.com/en/articles/1793#c3_1)
  - [3.2. Processing of clicking and creating events for Depth of Market](https://www.mql5.com/en/articles/1793#c3_2)
  - [3.3. Depth of Market cells](https://www.mql5.com/en/articles/1793#c3_3)
  - [3.4. Displaying volume histogram in Depth of Market](https://www.mql5.com/en/articles/1793#c3_4)
  - [3.5. Quick calculation of maximum volumes in Depth of Market, optimization of iteration](https://www.mql5.com/en/articles/1793#c3_5)
  - [3.6. Finishing touches: a volume histogram and a dividing line](https://www.mql5.com/en/articles/1793#c3_6)
  - [3.7. Adding properties of DOM with information about the total number of limit orders for trading instrument](https://www.mql5.com/en/articles/1793#c3_7)
- [Chapter 4. Documentation for CMarketBook class](https://www.mql5.com/en/articles/1793#chapter4)
  - [4.1. Methods of obtaining basic information from Depth of Market and operation with it](https://www.mql5.com/en/articles/1793#c4_1)
    - [Refresh() method](https://www.mql5.com/en/articles/1793#Refresh)
    - [InfoGetInteger() method](https://www.mql5.com/en/articles/1793#InfoGetInteger)
    - [InfoGetDouble() method](https://www.mql5.com/en/articles/1793#InfoGetDouble)
    - [GetDeviationByVol method](https://www.mql5.com/en/articles/1793#GetDeviationByVol)
  - [4.2. Enumerations and modifiers of CMarketBook class](https://www.mql5.com/en/articles/1793#c4_2)
    - [Enumeration of ENUM\_MBOOK\_SIDE](https://www.mql5.com/en/articles/1793#ENUM_MBOOK_SIDE)
    - [Enumeration of ENUM\_MBOOK\_INFO\_INTEGER](https://www.mql5.com/en/articles/1793#ENUM_MBOOK_INFO_INTEGER)
    - [Enumeration of ENUM\_MBOOK\_INFO\_DOUBLE](https://www.mql5.com/en/articles/1793#ENUM_MBOOK_INFO_DOUBLE)
  - [4.3. Example for using CMarketBook class](https://www.mql5.com/en/articles/1793#c4_3)
- [Conclusion](https://www.mql5.com/en/articles/1793#exit)

### Introduction

MQL5 language is constantly evolving and offering more opportunities for operation with exchange information every year. One of such exchange data types is information about Depth of Market. It is a special table showing price levels and volumes of limit orders. MetaTrader 5 has a built-in Depth of Market for displaying limit orders, but it is not always sufficient. First of all, your Expert Advisor has to be given a simple and convenient access to Depth of Market. Certainly, MQL5 language has few special features for working with such information, but they are low-level features that require additional mathematical calculations.

However, all intermediate calculations can be avoided. All you have to do is to write a special class for working with Depth of Market. All complex calculations will be carried out within Depth of Market, and the class itself will provide convenient ways for operation with DOM prices and levels. This class will enable an easy creation of the efficient panel in a form of an indicator, which will be promptly reflecting the current state of prices in Depth of Market:

![](https://c.mql5.com/2/19/1_3ez3s6_s43.png)

Fig. 1. Depth of Market displayed as a panel

This article demonstrates users how to utilize Depth of Market (DOM) programmatically and describes the operation principle of **CMarketBook** class, that can expand the Standard Library of MQL5 classes and offer convenient methods of using DOM.

After reading the first chapter of this article, it will become clear that the regular Depth of Market offered by MetaTrader 5 has impressive capabilities. We will not try to duplicate all these multiple opportunities in our indicator, as our task will be completely different. With a practical example of creating user-friendly Depth of Market trading panel we will show that the principles of object-oriented programming allow relatively easy handling of complex data structures. We will ensure that it won't be difficult to gain access to Depth of Market directly from your Expert Advisor with MQL5 and, consequently, to visualize its representation as it is convenient for us.

### Chapter 1. Standard Depth of Market in MetaTrader 5 and methods of using it

**1.1. Standard Depth of Market in MetaTrader 5**

MetaTrader 5 supports trading on centralized exchanges and provides standard tools for operation with Depth of Market. First of all, it is, certainly, a table of limit orders, that recently has been given an advanced mode of representation. To open Depth of Market it is necessary to connect to one of the exchanges that supports MetaTrader 5 and to select in the context menu "View" --> "Depth of Market" --> "Name of instrument." A separate window that combines a tick price chart and a table of limit orders will appear:

![](https://c.mql5.com/2/20/2._maywnc92a0_2e2xzj_7vh.png)

Fig. 2. Standard Depth of Market in MetaTrader 5

Standard Depth of Market in MetaTrader 5 can boast a rich functionality. In particular, it allows to display the following:

- Buy and Sell limit orders, their price levels and volume (standard form of classical DOM);
- current spread level and price levels occupied by limit orders (advanced mode);
- tick chart and visualized Bid, Ask and last trade volumes;
- total level of Buy and Sell orders (displayed as two lines at the tick chart's top and bottom, respectively).

The list provided confirms that Depth of Market features are more than impressive. Let's find out how to operate with data by getting access to it programmatically. First of all, you need to get an idea about how is Depth of Market established ​​and what is the key to its data organization. For more information read the article " [Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)" in chapter " [13\. Matching Sellers and Buyers. Exchange Depth of Market](https://www.mql5.com/en/articles/1284#c1_3)". We will not be spending much time on this table description, assuming that the reader has already sufficient knowledge of the subject.

**1.2. Event model for working with Depth of Market**

Depth of Market has a very dynamic data table. On fast dynamic markets the table of limit orders may change dozens of times per second. Therefore, you must try to process only information that is actually needed for processing, otherwise the amount of transferred data and the central processor's loading for processing this data may exceed all reasonable limits. This is the reason why MetaTrader 5 requires a special event model that prevents acquisition and processing of data that will not be actually used. Let's have a thorough examination of this model.

Any event occurring on the market, such as the arrival of a new tick or the execution of a trading transaction, can be processed by calling the corresponding function associated with it. For example, with the arrival of a new tick in MQL5, a special OnTick() event handler function is called. Resizing the chart or its position calls the OnChartEvent() function. This event model also applies to Depth of Market changes. For example, if someone places Sell or Buy limit orders in Depth of Market, its status will change and call a special OnBookEvent() function.

Since there are tens or even hundreds of different symbols available on the terminal with their own Depth of Market, the number of calls of OnBookEvent function can be enormous and resource intensive. In order to avoid this, the terminal must be notified in advance when running an indicator or an Expert Advisor from which instruments in particular it is required to obtain information about second level quotations (information provided by Depth of Market is also called this way). A special system function MarketBookAdd is used for such purposes. For example, if we want to obtain information about Depth of Marked based on instrument Si-9.15 (futures contract for USD/RUR with expiration in September 2015), we need to write the following code in our Expert Advisor or an indicator of OnInit function:

```
void OnInit()
{
   MarketBookAdd("Si-9.15");
}
```

With this function we have created the so-called "subscription" and notified the terminal, that the Expert Advisor or indicator has to be informed in the event of Depth of Market change based on the instrument Si-9.15. The Depth of Market changes for other instruments will be unavailable to us, which will considerably reduce the resources consumed by the program.

The opposite function of MarketBookAdd is MarketBookRelease function. On the contrary, it "unsubscribes" us from the notifications of Depth of Market changes. It is advisable for a programmer to unsubscribe in OnDeinit section, thus closing access to data for closing the Expert Advisor or indicator:

```
void OnDeinit(const int reason)
{
   MarketBookRelease("Si-9.15");
}
```

The call of MarketBookAdd function basically implies that at the moment of change in Depth of Market for a required instrument, a special event handling OnBookEvent() function will be called. This way a simple Expert Advisor or an indicator operating with Depth of Market will contain three system functions:

- OnInit - Expert Advisor or indicator initialization function for subscribing to receive an update in case of DOM change for the required instrument.
- OnDeinit - deinitialization of Expert Advisor or indicator function for unsubscribing from receiving information in the event of Depth of Market change for the required instrument.
- OnBookEvent - function called after a change in Depth of Market to signal that Depth of Market has changed.

Our first simple Expert Advisor will contain these three functions. As seen from the example below, for every Depth of Market change the following message will be typed: "Depth of Market for Si-9.15 was changed":

```
//+------------------------------------------------------------------+
//|                                                       Expert.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
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
   MarketBookAdd("Si-9.15");
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   MarketBookRelease("Si-9.15");
  }
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
//---
   printf("Depth of Market for " + symbol +  " was changed");
  }
//+------------------------------------------------------------------+
```

Key instructions in the code are highlighted with yellow marker.

**1.3. Receiving second level quotes with MarketBookGet functions and MqlBookInfo structure**

Now that we have learned to receive notifications for Depth of Market changes, it's time to learn how to access DOM information by using the MarketBookGet function. Let's examine its prototype and usage.

As already mentioned, Depth of Market is presented with a special table consisting of two parts to display Sell and Buy limit orders. Like any other table, it is the easiest way to display Depth of Market in the form of array, where the array index is the table's line number, and the array value is a certain line or sequence of data that includes volume, price and application type. Let's also imagine Depth of Market as a table showing the index of each line:

| Line index | Order type | Volume | Price |
| --- | --- | --- | --- |
| 0 | Sell Limit | 18 | 56844 |
| 1 | Sell Limit | 1 | 56843 |
| 2 | Sell Limit | 21 | 56842 |
| 3 | Buy Limit | 9 | 56836 |
| 4 | Buy Limit | 5 | 56835 |
| 5 | Buy Limit | 15 | 56834 |

Table 1. Depth of Market presented in a table

To make it easier to navigate across the table, Sell Orders are marked with pink, and Buy Orders with blue colors. The Depth of Market table is basically a two-dimensional array. The first dimension indicates a line number and the second dimension - one of the three table factors (order type - 0, order volume - 1 and order price - 2). However, to avoid working with multidimensional arrays, a special MqlBookInfo structure is used in MQL5. It includes all the necessary values. This way every Depth of Market index contains a MqlBookInfo structure, which in turn carries information about the order type, its volume and price. Let's give a definition of this structure:

```
struct MqlBookInfo
  {
   ENUM_BOOK_TYPE   type;       // order type from ENUM_BOOK_TYPE enumeration
   double           price;      // order price
   long             volume;     // order volume
  };
```

Now the method of working with Depth of Market should be clear to us. MarketBookGet function returns an array of MqlBookInfo structures. Array index indicates the line of price table, and the index structure contains information about volume, price and order type. Knowing this, we will try to get access to the first DOM order, and for this reason we will slightly modify the OnBookEvent function of our Expert Advisor from the previous example:

```
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
//---
   //printf("Depth of Market " + symbol +  " changed");
   MqlBookInfo book[];
   MarketBookGet(symbol, book);
   if(ArraySize(book) == 0)
   {
      printf("Failed load market book price. Reason: " + (string)GetLastError());
      return;
   }
   string line = "Price: " + DoubleToString(book[0].price, Digits()) + "; ";
   line += "Volume: " + (string)book[0].volume + "; ";
   line += "Type: " + EnumToString(book[0].type);
   printf(line);
  }
```

When running an Expert Advisor on any of the charts, we will receive the reports about the first DOM and its parameters:

```
2015.06.05 15:54:17.189 Expert (Si-9.15,H1)     Price: 56464; Volume: 56; Type: BOOK_TYPE_SELL
2015.06.05 15:54:17.078 Expert (Si-9.15,H1)     Price: 56464; Volume: 56; Type: BOOK_TYPE_SELL
2015.06.05 15:54:17.061 Expert (Si-9.15,H1)     Price: 56464; Volume: 56; Type: BOOK_TYPE_SELL
...
```

Looking at our previous table, it is easy to guess that the level available for zero index of DOM corresponds to the worst Ask price (BOOK\_TYPE\_SELL). And on the contrary, the lowest Bid price takes the last index in the obtained array. Best Ask and Bid prices are positioned approximately in the middle of Depth of Market. The first disadvantage of the obtained prices is that the calculation in Depth of Market is normally performed using the best prices that are usually placed in the middle of the table. The lowest Ask and Bid prices have a secondary importance. In the future, when analyzing the CMarketBook class we will solve this issue by providing specific indexers convenient for working with Depth of Market.

### Chapter 2. CMarketBook class for easy access and operation with Depth of Market

**2.1. Designing CMarketInfoBook class**

In the first chapter we got acquainted with the system functions for operating with Depth of Market and found out the particular features of the event model for arranging access to the second level quotations. In this chapter we will create a special class **CMarketBook** for a convenient use of a standard Depth of Market. Based on the knowledge gained from the first chapter, we can discuss the properties our class should have to work with this type of data.

So the first thing that must be considered in designing this class is a resource intensity of received data. Depth of Market can be updated dozens of times per second, in addition to that, it contains dozens of MqlBookInfo type elements. Therefore, our class must initially work only with one instrument. When processing several Depth of Market from different instruments, it is sufficient to create multiple copies of our class with a specific instrument indication:

```
CMarketBook("Si-9.15");            // Depth of Market for Si-9.15
CMarketBook("ED-9.15");            // Depth of Market for ED-9.15
CMarketBook("SBRF-9.15");          // Depth of Market for SBRF-9.15
CMarketBook("GAZP-9.15");          // Depth of Market for GAZP-9.15
```

The second aspect we need to take care of is the organization of an easy access to data. Since the limit orders generate a high-frequency stream of price levels, it is impossible to copy Depth of Market to a secure object-oriented table. Therefore, our class will provide a direct, albeit not as secure, access to the MqlBookInfo array provided by MarketBookGet system function. For example, to access a zero index of DOM with our class, the following has to be written:

```
CMarketBook BookOnSi("Si-9.15");
...
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
//---
   MqlBookInfo info = BookOnSi.MarketBook[0];
  }
//+------------------------------------------------------------------+
```

MarketBook is an array that is directly obtained using the MarketBookGet function. However, the convenience of using our class will be primarily based on the fact that besides the direct access to an array of limit orders, our class will allow targeted access to the most commonly used Depth of Market prices. For example, to obtain the best Ask price, it is enough to write the following:

```
double best_ask = BookOnSi.InfoGetDouble(MBOOK_BEST_ASK_PRICE);
```

It is more convenient than calculating the best Ask index in your expert and then getting the price value based on this index. From the above code it is obvious that CMarketBook, as well as many other MQL5 system functions such as SymbolInfoDouble or OrderHistoryInteger, use their own set of modifiers and InfoGetInteger and InfoGetDouble methods to access integer and double values respectively. To obtain the required properties we must specify a certain modifier of this property. Let's thoroughly describe the modifiers of these properties:

```
//+------------------------------------------------------------------+
//| Specifies modifiers for integer type properties                  |
//| of DOM.                                                          |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_INFO_INTEGER
{
   MBOOK_BEST_ASK_INDEX,         // Index of best Ask price
   MBOOK_BEST_BID_INDEX,         // Index of best Bid price
   MBOOK_LAST_ASK_INDEX,         // Index of worst Ask price
   MBOOK_LAST_BID_INDEX,         // Index of worst Bid price
   MBOOK_DEPTH_ASK,              // Number of Sell levels
   MBOOK_DEPTH_BID,              // Number of Buy levels
   MBOOK_DEPTH_TOTAL             // Total number of DOM levels
};
//+------------------------------------------------------------------+
//| Specifies modifiers for double type properties                   |
//| of DOM.                                                          |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_INFO_DOUBLE
{
   MBOOK_BEST_ASK_PRICE,         // Best Ask price
   MBOOK_BEST_BID_PRICE,         // Best Bid price
   MBOOK_LAST_ASK_PRICE,         // Worst Ask price
   MBOOK_LAST_BID_PRICE,         // Worst Bid price
   MBOOK_AVERAGE_SPREAD          // Average spread between Ask and Bid
};
```

Certainly, in addition to the methods of InfoGet... group, our class will contain Refresh method triggering the Depth of Market update. Due to the fact that our class requires information update by calling Refresh() method, we will use a resource intensive class update only when it is required.

**2.2. Index calculation of most frequently used Depth of Market levels**

CMarketBook class is a wrapper for the MqlBookInfo array. Its main purpose is to provide quick and convenient access to the most frequently requested information from this array. Consequently, there are only two basic resource-intensive operations enabled by the class:

- copying of MqlBookInfo array with MarketBookGet system function;
- index calculation of most commonly used prices.

We can't speed ​​up the operation of your MarketBookGet system function, but it is not necessary, since all system functions of MQL5 language have maximum optimization. But we can produce the fastest calculation of the required indexes. Once again we will refer to ENUM\_MBOOK\_INFO\_INTEGER and ENUM\_MBOOK\_INFO\_DOUBLE property modifiers. As you can see, almost all available properties are based on the calculation of four indexes:

- index of best Ask price;
- index of best Bid price;
- index of worst Bid price;
- index of worst Ask price.

Also three integer-valued properties are used:

- number of Sell price levels or Depth of Market for Sell (Ask depth);
- number of Buy price levels or Depth of Market for Buy (Bid depth);
- total Depth of Market equal to the total number of elements in DOM.

It is obvious that the index of the worst Ask price will always be zero, since the array produced using MarketBookGet function begins with the worst Ask price. Finding the worst Bid price index is trivial - it will always have the last index in the obtained MqlInfoBook array (we would like to remind you, that the last element index in the array is less than the total number of elements of this array per unit):

_index of worst Ask price = 0_

_index of worst Bid price = total number of elements in Depth of Market - 1_

Indexes of integer-valued properties are also easy to calculate. Thus, the total Depth of Market always equals the number of elements in the MqlBookInfo array. Depth of Market from the Ask side is the following:

_Ask depth = index of best Ask price - index of worst Ask price + 1_

We always add one, as the numbering in the array starts with zero and to determine the number of elements it is required to add one to a better index. However, by adding one to the index of best Ask price we already obtain the index of best Bid price. For example, if in the Table 1 we add one to the line with the second number, we will move from the best Sell Limit order at the price 56 842 to the best Buy Limit order at the price 56 836. We have also found out that the index of worst Ask price is always zero. Therefore, we can reduce the formula to find the Ask depth accordingly:

_Ask depth = index of best Bid price_

Calculation of Bid depth is somewhat different. It is obvious that the number of Buy orders equals the total number of orders minus the number of Sell orders or Ask depth. Since in the previous formula we have learned that the Ask depth equals the index of best Bid price, it won't be difficult to make the formula to determine the number of Buy orders or the Bid depth:

_Bid depth = total Depth of Market - index of best Ask price_

Total Depth of Market always equals the total of Bid and Ask depth, and therefore equals the total number of elements in the Depth of Market:

_total Depth of Market = total number of elements in Depth of Market_

Analytically, we have found almost all frequently used indexes. Using the mathematical reductions, we have replaced the index calculation with the direct indexation. This is very important in CMarketBook class, since the fastest possible access to Depth of Market properties and indexes is required.

Apart from the actual indexes, it is often needed to know the average spread level for the current instrument. Spread is a difference between the best Bid and Ask prices. CMarketBook class allows to obtain the average value of this parameter using the InfoGetDouble method by calling it with the MBOOK\_AVERAGE\_SPREAD modifier. CMarketBook calculates the current spread in the Refresh method as well as its average value by memorizing the number of this method's calls.

However, we haven't yet found the main indexes of the best Bid and Ask prices, so we move on to the next section.

**2.3. Predicting indexes of best Bid and Ask prices based on previous values of these indexes**

Calculation of the best Ask and Bid prices is a more difficult task that we have yet to complete. For example, in Table 1 the index of the best Ask price will be index with number 2, and the index of the best Bid price will be index number 3. A simplified Depth of Market consisting from only 6 levels is presented in this table. In reality, Depth of Market may be considerably bigger and contain up to 64 Sell and Buy levels. This is a significant value, considering that the DOM update can occur several times per second.

The easiest solution would be to use a "divide in two" method here. Indeed, if we take the total number of levels in Table 1, which is 6, and divide it in two, the obtained figure (3) is the index of the best Bid price. Therefore, its preceding index is the index of the best Ask price (2). However, this method only works if the number of Sell levels equals the number of Buy levels in DOM. This normally occurs on the liquid markets, however, on the illiquid markets Depth of Market may be partially full, and on one of the sides there may be no levels at all.

Our CMarketBook class has to operate on any markets and with any Depth of Market, so the method of dividing in two is not suitable for us. To illustrate the situation when the dividing in two method may not work, we refer to the following figure:

![](https://c.mql5.com/2/20/3._vwi6ma_12s_18_kmtdw9_blkji7.png)

Fig. 3. The number of Bid price levels is not always equal to the number of Ask price levels

Two Depth of Market are shown on Figure 3. The first of them is a futures contract DOM for two-year federal loan bonds (OFZ2-9.15). The second is a EUR/USD futures contract (ED-9.15). It shows that for OFZ2-9.15 the number of Buy price levels is four, while the number of Sell price levels is eight. On a more liquid ED-9.15 market the amount of both Buy and Sell levels is 12 for each of the parties. In case with ED-9.15 the method of determining indexes by diving in two would have worked, and with OFZ2 - it wouldn't.

A more reliable way to find the index would be to use the DOM iteration until the first order occurrence with BOOK\_TYPE\_BUY type. The previous index would automatically become the index of the best Ask price. This is the method that a CMarketBook class has. We will refer to it for illustration of the above:

```
void CMarketBook::SetBestAskAndBidIndex(void)
{
   if(!FindBestBid())
   {
      //Find best ask by slow full search
      int bookSize = ArraySize(MarketBook);
      for(int i = 0; i < bookSize; i++)
      {
         if((MarketBook[i].type == BOOK_TYPE_BUY) || (MarketBook[i].type == BOOK_TYPE_BUY_MARKET))
         {
            m_best_ask_index = i-1;
            FindBestBid();
            break;
         }
      }
   }
}
```

The main purpose of this method lies in the iteration of DOM by for operator. Once a first order with the BOOK\_TYPE\_BUY type is encountered in Depth of Market, the indexes of best Bid and Ask prices are set, and the iteration is interrupted. Full Depth of Market iteration for each update would be an extremely resource-intensive solution.

Instead of having iteration for every update, there is an option to _remember earlier obtained indexes of the best Bid and Ask prices._ Indeed, Depth of Market normally contains a fixed amount of Buy and Sell levels. Therefore, it is not required to iterate Depth of Market each time in order to find new indexes. It is sufficient to refer to the previously found indexes and to understand whether they are still indexes of the best Bid and Ask prices. FindBestBid private method is used to solve such issue. Let's see its content:

```
//+------------------------------------------------------------------+
//| Fast find best bid by best ask                                   |
//+------------------------------------------------------------------+
bool CMarketBook::FindBestBid(void)
{
   m_best_bid_index = -1;
   bool isBestAsk = m_best_ask_index >= 0 && m_best_ask_index < m_depth_total &&
                    (MarketBook[m_best_ask_index].type == BOOK_TYPE_SELL ||
                    MarketBook[m_best_ask_index].type == BOOK_TYPE_SELL_MARKET);
   if(!isBestAsk)return false;
   int bestBid = m_best_ask_index+1;
   bool isBestBid = bestBid >= 0 && bestBid < m_depth_total &&
                    (MarketBook[bestBid].type == BOOK_TYPE_BUY ||
                    MarketBook[bestBid].type == BOOK_TYPE_BUY_MARKET);
   if(isBestBid)
   {
      m_best_bid_index = bestBid;
      return true;
   }
   return false;
}
```

It is easy to operate it. Firstly, the method confirms that the current index of the best Ask price still complies with the index of Ask price. It then resets the index of the best Bid price and tries to find it again, referring to the element that follows the index of the best Ask price:

```
int bestBid = m_best_ask_index+1;
```

If the found element is indeed the best index of Bid price, then the previous state of DOM had the same number of Buy and Sell levels as the present state. Because of that, DOM iteration can be avoided, since in the SetBestAskAndBidIndex method a FindBestBid method is called before iteration. Thus, DOM iteration is executed only at the first function calling, as well as in case of changing the number of Sell and/or Buy levels.

Although the resulting source code came out bigger and more complex than a simple Depth of Market, in fact it will operate faster. The benefit in performance will be especially noticeable on big DOM of liquid markets. Simple instructions to check the conditions are fulfilled very quickly, and the quantity of these checks is much smaller than the loops of for operator. Therefore, the performance of this method aimed at finding best price indexes will be higher than the normal iteration.

**2.4. Determining maximum slippage with GetDeviationByVol method**

Often Depth of Market is used by traders to determine the current market liquidity, _i.e. Depth of Market is used as an additional instrument to control risks_. If market liquidity is low, entering market through market orders can cause high slippage. Slippage always implies additional losses that may be of significant value.

To avoid such situations, the additional methods for controlling the market entry have to be used. For more information please read the article " [How to Secure Your Expert Advisor While Trading on the Moscow Exchange](https://www.mql5.com/en/articles/1683)". Therefore, we will not describe these methods in detail and will only mention, that by getting access to Depth of Market we can estimate the value of a potential slippage before entering a market. The size of a slippage depends on two factors:

- current liquidity of Bid side (sell orders) and Ask side (buy orders);
- volume of a deal.

Having access to Depth of Market allows us to see at what volume and prices our order will be executed. If we know the volume of our order, we can calculate the weighted average price for market entry. The difference between this price and the best Bid or Ask price (depending on the entrance direction) will be our slippage.

It is impossible to calculate the weighted average of the entry price manually, as it is required to produce a large volume of calculations in a very short space of time (let me remind you, that the state of DOM may change several times per second). Thus, it is natural to delegate this task to an Expert Advisor or indicator.

CMarketBook class includes a special method to calculate this characteristic - GetDeviationByVol. Since the deal's volume effects the slippage size, it is required to pass the volume expected to be executed on the market to the method. Since this method uses integer-valued arithmetic of volumes, as adopted on the Moscow Exchange futures market, the method takes volume as a long type value. In addition to that, the method needs to know for which side of liquidity the calculation has to be performed, therefore a special ENUM\_MBOOK\_SIDE enumeration is used:

```
//+------------------------------------------------------------------+
//| Side of MarketBook.                                              |
//+------------------------------------------------------------------+
enum ENUM_MBOOK_SIDE
{
   MBOOK_ASK,                    // Ask side
   MBOOK_BID                     // Bid side
};
```

Now let's introduce a source code of the GetDeviationByVol method:

```
//+------------------------------------------------------------------+
//| Get deviation value by volume. Return -1.0 if deviation is       |
//| infinity (insufficient liquidity)                                |
//+------------------------------------------------------------------+
double CMarketBook::GetDeviationByVol(long vol, ENUM_MBOOK_SIDE side)
{
   int best_ask = InfoGetInteger(MBOOK_BEST_ASK_INDEX);
   int last_ask = InfoGetInteger(MBOOK_LAST_ASK_INDEX);
   int best_bid = InfoGetInteger(MBOOK_BEST_BID_INDEX);
   int last_bid = InfoGetInteger(MBOOK_LAST_BID_INDEX);
   double avrg_price = 0.0;
   long volume_exe = vol;
   if(side == MBOOK_ASK)
   {
      for(int i = best_ask; i >= last_ask; i--)
      {
         long currVol = MarketBook[i].volume < volume_exe ?
                        MarketBook[i].volume : volume_exe ;
         avrg_price += currVol * MarketBook[i].price;
         volume_exe -= MarketBook[i].volume;
         if(volume_exe <= 0)break;
      }
   }
   else
   {
      for(int i = best_bid; i <= last_bid; i++)
      {
         long currVol = MarketBook[i].volume < volume_exe ?
                        MarketBook[i].volume : volume_exe ;
         avrg_price += currVol * MarketBook[i].price;
         volume_exe -= MarketBook[i].volume;
         if(volume_exe <= 0)break;
      }
   }
   if(volume_exe > 0)
      return -1.0;
   avrg_price/= (double)vol;
   double deviation = 0.0;
   if(side == MBOOK_ASK)
      deviation = avrg_price - MarketBook[best_ask].price;
   else
      deviation = MarketBook[best_bid].price - avrg_price;
   return deviation;
}
```

As you can see, the code has a significant volume, but the principle of its calculation is not complex at all. Firstly, Depth of Market iteration is performed from the best towards the worst price. The iteration is executed to the relevant side for each direction. During iteration the current volume is added to the total volume. If the total volume is called and it matches the required volume, then the exit from the loop occurs. Then the average entry price for a given volume is calculated. And finally, the difference between the average entry price and the best Bid/Ask prices are calculated. The absolute difference will be an estimated deviation.

This method is required to calculate the direct iteration of DOM. Although only one out of two DOM parts is iterated, and most of the times only partially, nevertheless, such calculation needs more time than calculation of frequently used indexes of DOM. Therefore, this calculation is implemented directly in a separate method and is produced on demand, i.e. only in those cases when it is required to obtain this information in a clear form.

**2.5. Examples for operating with CMarketBook class**

So we covered the basic methods of CMarketBook class, and it's time to practice a little using it. Our test example is fairly simple and understandable even for beginners to programming. Let's write a test Expert Advisor for executing one-off information output based on DOM. Certainly, for these purposes a script would fit better, but the access to Depth of Market through a script is not possible, and either an Expert Advisor or an indicator has to be used. The source code for our EA is given below:

```
//+------------------------------------------------------------------+
//|                                               TestMarketBook.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Trade\MarketBook.mqh>     // Include CMarketBook class
CMarketBook Book(Symbol());         // Initialize class with current instrument

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   PrintMbookInfo();
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---

  }
//+------------------------------------------------------------------+
//| Print MarketBook Info                                            |
//+------------------------------------------------------------------+
void PrintMbookInfo()
  {
   Book.Refresh();                  // Update Depth of Market status.
   /* Obtain basic statistics */
   int total=Book.InfoGetInteger(MBOOK_DEPTH_TOTAL);
   int total_ask = Book.InfoGetInteger(MBOOK_DEPTH_ASK);
   int total_bid = Book.InfoGetInteger(MBOOK_DEPTH_BID);
   int best_ask = Book.InfoGetInteger(MBOOK_BEST_ASK_INDEX);
   int best_bid = Book.InfoGetInteger(MBOOK_BEST_BID_INDEX);

   printf("TOTAL DEPTH OF MARKET: "+(string)total);
   printf("NUMBER OF PRICE LEVELS FOR SELL: "+(string)total_ask);
   printf("NUMBER OF PRICE LEVELS FOR BUY: "+(string)total_bid);
   printf("INDEX OF BEST ASK PRICE: "+(string)best_ask);
   printf(INDEX OF BEST BID: "+(string)best_bid);

   double best_ask_price = Book.InfoGetDouble(MBOOK_BEST_ASK_PRICE);
   double best_bid_price = Book.InfoGetDouble(MBOOK_BEST_BID_PRICE);
   double last_ask = Book.InfoGetDouble(MBOOK_LAST_ASK_PRICE);
   double last_bid = Book.InfoGetDouble(MBOOK_LAST_BID_PRICE);
   double avrg_spread = Book.InfoGetDouble(MBOOK_AVERAGE_SPREAD);

   printf("BEST ASK PRICE: " + DoubleToString(best_ask_price, Digits()));
   printf("BEST BID PRICE: " + DoubleToString(best_bid_price, Digits()));
   printf("WORST ASK PRICE: " + DoubleToString(last_ask, Digits()));
   printf("WORST BID PRICE: " + DoubleToString(last_bid, Digits()));
   printf("AVERAGE SPREAD: " + DoubleToString(avrg_spread, Digits()));
  }
//+------------------------------------------------------------------+
```

When running this test Expert Advisor on OFZ2 chart we get the following report:

```
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   AVERAGE SPREAD: 70
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   WORST BID PRICE: 9831
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   WORST ASK PRICE: 9999
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   BEST BID PRICE: 9840
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   BEST ASK PRICE: 9910
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   BEST BID INDEX: 7
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   BEST ASK INDEX: 6
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   NUMBER OF PRICE LEVELS FOR BUY: 2
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   NUMBER OF PRICE LEVELS FOR SELL: 7
2015.06.16 17:13:23.482 TestMarketBook (OFZ2-9.15,D1)   TOTAL DEPTH OF MARKET: 9
```

Let's compare the obtained report with DOM screenshot for this instrument:

![](https://c.mql5.com/2/20/4._OFZ2-9.15.png)

Fig. 4. Depth of Market for OFZ2 when running a test report

We have confirmed that the received indexes and prices fully comply with the current Depth of Market.

### Chapter 3. Writing your own Depth of Market as a panel indicator

**3.1. General principles of designing the Depth of Market panel. Creating an indicator**

With access to the CMarketBook class, it is relatively simple to create a special panel that displays current Depth of Market directly on the chart. We will create our panel on the basis of the user's indicator. Using an indicator as a base was selected due to the fact that just one Expert Advisor can be located on every chart, while there can be an unlimited number of indicators. If we take an Expert Advisor as a base of the panel, then trading with an expert on the same chart would no longer be possible, and that would be inconvenient.

We will provide our Depth of Market with the ability to appear and hide on the chart, just as it is implemented in a standard trading panel for each chart. We will use the same button for showing or hiding it:

![](https://c.mql5.com/2/19/5._fua16uh65cr_3z4dp6yv_yvx002_o_VN5.png)

Fig. 5. Standard trading panel in MetaTrader 5

Our Depth of Market will be placed in the upper left corner of the chart, in the same place where the trading panel is located. This is due to the fact that on the chart with Depth of Market indicator a trading Expert Advisor has been launched. In order not to block its icon located in the top right corner, we will move our panel to the left.

When creating an indicator it is required to use one out of two system functions OnCalculate. Since our panel does not use the information received from these functions, we will leave these methods empty. Also, the indicator will not use any graphic series, so indicator\_plots property in this case will equal zero.

OnBookEvent system function will be the main function that our indicator will use, thus we will need to sign the current chart symbol in order to receive information about Depth of Market changes. We will subscribe using MarketBookAdd function already known to us.

Depth of Market panel will be implemented in a form of a special **CBookPanel** class. Now, without going into further details about this class, we will provide the code of the most starting indicator file:

```
//+------------------------------------------------------------------+
//|                                                   MarketBook.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0
#include <Trade\MarketBook.mqh>
#include "MBookPanel.mqh"

CBookPanel Panel;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   MarketBookAdd(Symbol());
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| MarketBook change event                                          |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
  {
   Panel.Refresh();
  }
//+------------------------------------------------------------------+
//| Chart events                                                     |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,         // event identifier
                  const long& lparam,   // event parameter of long type
                  const double& dparam, // event parameter of double type
                  const string& sparam) // event parameter of string type
  {
   Panel.Event(id,lparam,dparam,sparam);
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
  {
//---

//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Now CBookPanel class contains only the basic elements for its operation: the arrow which you click for Depth of Market to appear, and the "MarketBook" label for signing our future DOM. When running our indicator, the chart will look the following way:

![](https://c.mql5.com/2/19/6._jzj3acb7qekn_dhzgz_0pu0l0.png)

Fig. 6. Location of future MarketBook panel on the chart

Each element of this class is also an independent class derived from a basic CNode class. This class contains basic methods, such as Show and Hide, which can be overridden in descendant classes. CNode class also generates a unique name for each instance, which makes it more convenient to use standard functions for creating graphical objects and setting their properties.

**3.2. Processing of click events and creating Depth of Market form**

Currently our indicator does not react to clicking the arrow, so we will continue to work on it. The first thing we should do for our panel is to enter the OnChartEvent event handler. We will call this method as Event. It will take the parameters obtained from OnChartEvent. Also, we will extend the CNode basic class providing it with CArrayObj array, which will contain other graphical elements of CNode type. In the future it will help us to create many same type elements - Depth of Market cells.

Now we are going to provide a source code of the CBookPanel class and its parent class CNode:

```
//+------------------------------------------------------------------+
//|                                                   MBookPanel.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include <Trade\MarketBook.mqh>
#include "Node.mqh"
#include "MBookText.mqh"
#include "MBookFon.mqh"
//+------------------------------------------------------------------+
//| CBookPanel class                                                 |
//+------------------------------------------------------------------+
class CBookPanel : CNode
  {
private:
   CMarketBook       m_book;
   bool              m_showed;
   CBookText         m_text;
public:
   CBookPanel();
   ~CBookPanel();
   void              Refresh();
   virtual void Event(int id, long lparam, double dparam, string sparam);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CBookPanel::CBookPanel()
{
   m_elements.Add(new CBookFon(GetPointer(m_book)));
   ObjectCreate(ChartID(), m_name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XDISTANCE, 70);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, -3);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_COLOR, clrBlack);
   ObjectSetString(ChartID(), m_name, OBJPROP_FONT, "Webdings");
   ObjectSetString(ChartID(), m_name, OBJPROP_TEXT, CharToString(0x36));
}
CBookPanel::~CBookPanel(void)
{
   OnHide();
   m_text.Hide();
   ObjectDelete(ChartID(), m_name);
}

CBookPanel::Refresh(void)
{

}

CBookPanel::Event(int id, long lparam, double dparam, string sparam)
{
   switch(id)
   {
      case CHARTEVENT_OBJECT_CLICK:
      {
         if(sparam != m_name)return;
         if(!m_showed)OnShow();
         else OnHide();
         m_showed = !m_showed;
      }
   }
}
//+------------------------------------------------------------------+
```

Refresh method that updates the state of DOM is not yet full. We will create it a bit later. Current functionality already enables to show the first prototype of our Depth of Market. So far when clicking the arrow, only the standard gray canvas is displayed. When clicked again, it disappears:

![](https://c.mql5.com/2/19/7._5d2_rrcobtv1_wqecj84_ikl.png)

Fig. 7. Appearance of future Depth of Market

Depth of Market doesn't look very convincing yet, but we will continue to improve it.

**3.3. Depth of Market cells.**

Cells create the base for Depth of Market. Each cell is a table's element that contains information about volume or price. Also, cells can be distinguished by color: for Buy limit orders it is painted blue, for Sell limit orders - pink. The number of cells may be different for each Depth of Market, therefore, all cells need to be created _dynamically_ on demand and stored in a special data container _CArrayObj_. Since all cells, regardless of what they show, have the same size and type, the class that implements various types of cells will be the same for all types of cells.

For the cells showing volume and for the cells showing price, a special **CBookCeil** class will be used. The cell type will be specified when creating an object of this class, so each class instance will know what kind of information from Depth of Market it will be necessary to show, and what color should the background be painted. CBookCeil will use two graphics primitives: text label OBJ\_TEXT\_LABEL and rectangular label OBJ\_RECTANBLE\_LABEL. The first will display the text, the second - the actual Depth of Market cell.

Here is a source code of the CBookCeil class:

```
//+------------------------------------------------------------------+
//|                                                   MBookPanel.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#include "Node.mqh"
#include <Trade\MarketBook.mqh>
#include "Node.mqh"
#include "MBookText.mqh"

#define BOOK_PRICE 0
#define BOOK_VOLUME 1

class CBookCeil : public CNode
{
private:
   long  m_ydist;
   long  m_xdist;
   int   m_index;
   int m_ceil_type;
   CBookText m_text;
   CMarketBook* m_book;
public:
   CBookCeil(int type, long x_dist, long y_dist, int index_mbook, CMarketBook* book);
   virtual void Show();
   virtual void Hide();
   virtual void Refresh();

};

CBookCeil::CBookCeil(int type, long x_dist, long y_dist, int index_mbook, CMarketBook* book)
{
   m_ydist = y_dist;
   m_xdist = x_dist;
   m_index = index_mbook;
   m_book = book;
   m_ceil_type = type;
}

void CBookCeil::Show()
{
   ObjectCreate(ChartID(), m_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XDISTANCE, m_xdist);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, m_ydist);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_FONTSIZE, 9);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   m_text.Show();
   m_text.SetXDist(m_xdist+10);
   m_text.SetYDist(m_ydist+2);
   Refresh();
}

void CBookCeil::Refresh(void)
{
   ENUM_BOOK_TYPE type = m_book.MarketBook[m_index].type;
   if(type == BOOK_TYPE_BUY || type == BOOK_TYPE_BUY_MARKET)
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrCornflowerBlue);
   else if(type == BOOK_TYPE_SELL || type == BOOK_TYPE_SELL_MARKET)
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrPink);
   else
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrWhite);
   MqlBookInfo info = m_book.MarketBook[m_index];
   if(m_ceil_type == BOOK_PRICE)
      m_text.SetText(DoubleToString(info.price, Digits()));
   else if(m_ceil_type == BOOK_VOLUME)
      m_text.SetText((string)info.volume);
}

void CBookCeil::Hide(void)
{
   OnHide();
   m_text.Hide();
   ObjectDelete(ChartID(),m_name);
}
```

The main operation of this class is performed using methods Show and Refresh. The latter, depending on the transmitted cell type, colors it in a relevant color and displays the volume or price in it. To create a cell, you must specify its type, location on the X axis, location on the Y axis, DOM index that corresponds to this cell, and Depth of Market from where the cell will receive the information.

A special private method CreateCeils will create cells in a class that implements a substrate of DOM. Here is the source code:

```
void CBookFon::CreateCeils()
{
   int total = m_book.InfoGetInteger(MBOOK_DEPTH_TOTAL);
   for(int i = 0; i < total; i++)
   {
      CBookCeil* Ceil = new CBookCeil(0, 12, i*15+20, i, m_book);
      CBookCeil* CeilVol = new CBookCeil(1, 63, i*15+20, i, m_book);
      m_elements.Add(Ceil);
      m_elements.Add(CeilVol);
      Ceil.Show();
      CeilVol.Show();
   }
}
```

It will be called by clicking the arrow that expands Depth of Market.

Now everything is ready to create our new version of Depth of Market. After making changes and compiling the project, our indicator has acquired a new form:

![](https://c.mql5.com/2/19/1_5oiw2j_gln.png)

Fig. 8. The first version of Depth of Market as an indicator

**3.4. Displaying volume histogram in Depth of Market**

The obtained Depth of Market already performs the basic function - it shows trading levels, volume and prices of Sell and Buy limit orders. Thus, every change of values ​​in Depth of Market also changes the values ​​in the corresponding cells. However, visually it is not easy to keep track of volumes in the obtained table. For example, in a standard MetaTrader 5 DOM volume is displayed on the histogram's background, which shows the relative magnitude of the current volume in regards to the maximum volume in Depth of Market. Also, it wouldn't hurt to implement similar functionality to our Depth of Market.

There are different ways to solve this issue. The easiest solution would be to make all the necessary calculations directly in the CBookCeil class. Therefore it is required to write the following in its Refresh method:

```
void CBookCeil::Refresh(void)
{
   ...
   MqlBookInfo info = m_book.MarketBook[m_index];
   ...
   //Update Depth of Market histogram
   int begin = m_book.InfoGetInteger(MBOOK_LAST_ASK_INDEX);
   int end = m_book.InfoGetInteger(MBOOK_BEST_ASK_INDEX);
   long max_volume = 0;
   if(m_ceil_type != BOOK_VOLUME)return;
   for(int i = begin; i < end; i++)
   {
      if(m_book.MarketBook[i].volume > max_volume)
         max_volume = m_book.MarketBook[i].volume;
   }
   double delta = 1.0;
   if(max_volume > 0)
      delta = (info.volume/(double)max_volume);
   long size = (long)(delta * 50.0);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XSIZE, size);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, m_ydist);
}
```

In the complete DOM iteration method there is a maximum volume of DOM, then the current volume is divided by the maximum. The obtained share is multiplied by the maximum width of a volume table cell (it is represented by a constant of 50 pixels). The obtained width of a canvas will be a required histogram:

![](https://c.mql5.com/2/19/8._o1j2wk2hutu_d8rblq5.png)

Fig. 9. Depth of Market with histogram volumes

However, the problem with this code is that the DOM iteration is done in each cell every time Refresh is being called. For Depth of Market that is 40 elements deep it means 800 iterations of for loop per each Depth of Market update. Each cell iterates Depth of Market only for its own side, therefore iteration within each cell consists of twenty iterations (Depth of Market divided in half). Although modern computers handle this task, it is a very inefficient operation method, especially given the fact that it is necessary working with Depth of Market using the quickest and most efficient algorithms.

**3.5. Quick calculation of maximum volumes in Depth of Market, optimization of iteration**

Unfortunately, it is impossible to eliminate full iteration of Depth of Market. After each Depth of Market update the maximum volume and its price level can drastically change. However, we can try to minimize the number of iterations. For this purpose you should learn to make DOM iteration no more than one time between two Refresh calls. The second thing you have to do is to minimize the number of full iteration calls. For this purpose you must use the deferred calculation or, in other words, to carry out this calculation only at the explicit demand. We will transfer all calculations directly to the Depth of Market CMarketBook class, and write a special calculation subclass **CBookCalculation** located inside the CMarketBook. Please find its source code below:

```
class CMarketBook;

class CBookCalculation
{
private:
   int m_max_ask_index;         // Index of maximum Ask volume
   long m_max_ask_volume;       // Volume of maximum Ask price

   int m_max_bid_index;         // Index of maximum Bid volume
   long m_max_bid_volume;       // Volume of maximum Bid price

   long m_sum_ask_volume;       // Total volume of Ask price in DOM
   long m_sum_bid_volume;       // Total volume of Bid price in DOM.

   bool m_calculation;          // flag indicating that all calculations are executed
   CMarketBook* m_book;         // Depth of market indicator

   void Calculation(void)
   {
      // FOR ASK SIDE
      int begin = (int)m_book.InfoGetInteger(MBOOK_LAST_ASK_INDEX);
      int end = (int)m_book.InfoGetInteger(MBOOK_BEST_ASK_INDEX);
      for(int i = begin; i < end; i++)
      {
         if(m_book.MarketBook[i].volume > m_max_ask_volume)
         {
            m_max_ask_index = i;
            m_max_ask_volume = m_book.MarketBook[i].volume;
         }
         m_sum_ask_volume += m_book.MarketBook[i].volume;
      }
      // FOR BID SIDE
      begin = (int)m_book.InfoGetInteger(MBOOK_BEST_BID_INDEX);
      end = (int)m_book.InfoGetInteger(MBOOK_LAST_BID_INDEX);
      for(int i = begin; i < end; i++)
      {
         if(m_book.MarketBook[i].volume > m_max_bid_volume)
         {
            m_max_bid_index = i;
            m_max_bid_volume = m_book.MarketBook[i].volume;
         }
         m_sum_bid_volume += m_book.MarketBook[i].volume;
      }
      m_calculation = true;
   }

public:
   CBookCalculation(CMarketBook* book)
   {
      Reset();
      m_book = book;
   }

   void Reset()
   {
      m_max_ask_volume = 0.0;
      m_max_bid_volume = 0.0;
      m_max_ask_index = -1;
      m_max_bid_index = -1;
      m_sum_ask_volume = 0;
      m_sum_bid_volume = 0;
      m_calculation = false;
   }
   int GetMaxVolAskIndex()
   {
      if(!m_calculation)
         Calculation();
      return m_max_ask_index;
   }

   long GetMaxVolAsk()
   {
      if(!m_calculation)
         Calculation();
      return m_max_ask_volume;
   }
   int GetMaxVolBidIndex()
   {
      if(!m_calculation)
         Calculation();
      return m_max_bid_index;
   }

   long GetMaxVolBid()
   {
      if(!m_calculation)
         Calculation();
      return m_max_bid_volume;
   }
   long GetAskVolTotal()
   {
      if(!m_calculation)
         Calculation();
      return m_sum_ask_volume;
   }
   long GetBidVolTotal()
   {
      if(!m_calculation)
         Calculation();
      return m_sum_bid_volume;
   }
};
```

All Depth of Market iteration and resource-intensive calculations are hidden inside the Calculate private method. It is called only if the calculation flag m\_calculate is reset to false condition. Resetting this flag happens only in the Reset method. Since this class is designed exclusively to operate within CMarketBook class, only this class has access to it.

After updating Depth of Market Refresh method CMarketBook class resets the condition of the calculating module by calling its Reset method. Due to this the complete iteration of Depth of Market occurs not more than once between its two upgrades. Also a pending execution is used. In other words, the Calculate method of CBookCalcultae class is called only when there is a clear call from one out of six publicly available methods.

In addition to finding volume, the class that performs a full iteration of Depth of Market had fields added containing the total amount of Sell and Buy limit orders. Additional time to calculate these parameters is not required, since the total cycle of the array is calculated.

Now, instead of a constant Depth of Market iteration a smart iteration on demand is used. This greatly reduces the resources used, making operation with Depth of Market extremely efficient and fast.

**3.6. Finishing touches: a volume histogram and a dividing line**

We have almost completed our task of creating an indicator. The practical need to find the maximum volume has helped us create an effective method of economical calculation of the required indicators. If in the future we will want to add new calculation parameters to our Depth of Market, it will be easy to do so. For this purpose it will be sufficient to extend our CBookCalculate class with relevant methods and enter the appropriate modifiers to ENUM\_MBOOK\_INFO\_INTEGER and ENUM\_MBOOK\_INFO\_DOUBLE enumerations.

Now we have to make use of the work we have done and to rewrite the Refresh method for each cell:

```
void CBookCeil::Refresh(void)
{
   ENUM_BOOK_TYPE type = m_book.MarketBook[m_index].type;
   long max_volume = 0;
   if(type == BOOK_TYPE_BUY || type == BOOK_TYPE_BUY_MARKET)
   {
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrCornflowerBlue);
      max_volume = m_book.InfoGetInteger(MBOOK_MAX_BID_VOLUME);
   }
   else if(type == BOOK_TYPE_SELL || type == BOOK_TYPE_SELL_MARKET)
   {
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrPink);
      max_volume = m_book.InfoGetInteger(MBOOK_MAX_ASK_VOLUME); //The volume has been previously calculated, reoccurring iteration doesn't occur
   }
   else
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrWhite);
   MqlBookInfo info = m_book.MarketBook[m_index];
   if(m_ceil_type == BOOK_PRICE)
      m_text.SetText(DoubleToString(info.price, Digits()));
   else if(m_ceil_type == BOOK_VOLUME)
      m_text.SetText((string)info.volume);
   if(m_ceil_type != BOOK_VOLUME)return;
   double delta = 1.0;
   if(max_volume > 0)
      delta = (info.volume/(double)max_volume);
   long size = (long)(delta * 50.0);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XSIZE, size);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, m_ydist);
}
```

Visually our panel indicator works the same as in the preceding version, but in reality the histogram's calculation speed has significantly increased. This is what the art of programming is about - creating effective and easy-to-use algorithms, hiding the complexity of their implementation within private methods of the corresponding modules (classes).

With the appearance of volume histogram it became very vague where the dividing line between volumes of Bid and Ask prices is. Therefore, we will create such line adding to CBookPanel class the **CBookLine** special subclass implementing this feature:

```
class CBookLine : public CNode
{
private:
   long m_ydist;
public:
   CBookLine(long y){m_ydist = y;}
   virtual void Show()
   {
      ObjectCreate(ChartID(),     m_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, m_ydist);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_XDISTANCE, 13);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_YSIZE, 3);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_XSIZE, 108);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_COLOR, clrBlack);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrBlack);
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   }
};
```

This is a very simple class which, basically, just determines its position. The line position on Y-axis must be calculated at the time of its creation in the Show method. Knowing the index of best Ask price it is relatively easy to do this:

```
long best_bid = m_book.InfoGetInteger(MBOOK_BEST_BID_INDEX);
long y = best_bid*15+19;
```

In this case, the index of best\_bid cell is multiplied by the width of each cell (15 pixels), and an additional constant of 19 pixels is added to it.

Our Depth of Market has finally acquired that minimum appearance and functionality to make the experience of using it enjoyable. Certainly, much more could still be done. If desired, our indicator could be made much closer in functionality to the standard MetaTrader 5 Depth of Market.

But the main purpose of this article is not about this. The Depth of Market panel was created with a sole purpose of showing the possibilities of CMarketBook class. It helped to make this class faster, better and more functional, and therefore has fully achieved its goal. We will show you a short video, which will reveal all our work that we have done so far. Below is our DOM panel in dynamics:

Stakan - YouTube

Tap to unmute

[Stakan](https://www.youtube.com/watch?v=o5AhL7uyerA) [MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ)

MQL5.community1.91K subscribers

[Watch on](https://www.youtube.com/watch?v=o5AhL7uyerA)

**3.7. Adding properties of DOM with information about the total number of limit orders for trading instrument**

One distinguishing feature of the Moscow Exchange is a transmission of information about the total number of limit orders in real time. This article highlights the operation of Depth of Market as such, without being addressed to any specific market. However, this information despite being specific (peculiar to a specific trading platform), is available at the terminal's system level. Moreover, it expands data provided by Depth of Market. It was therefore decided to extend the enumeration of property modifiers and to include the support of these properties directly to the Depth of Market CMarketBook class.

The Moscow Stock Exchange provides the following real-time information:

- number of Sell limit orders placed for instrument at the current time;
- number of all Buy limit orders placed for instrument at the current time;
- total volume of all Sell limit orders placed for instrument at the current time;
- total amount of all Buy limit orders placed for instrument at the current time;
- number of open positions and open interest (only for futures markets).

While open interest is not directly related to the number of limit orders on the market (i.e. with its current liquidity), nevertheless, this information is often required in conjunction with the information about the limit orders, so the access to it through the CMarketBook class also looks appropriate. To access this information you must use SymbolInfoInteger and SymbolInfoDouble functions. However, in order for data to be accessible from a single location, we will expand our Depth of Market class by introducing additional enumeration and changes in InfoGetInteger and InfoGetDouble functions:

```
long CMarketBook::InfoGetInteger(ENUM_MBOOK_INFO_INTEGER property)
{
   switch(property)
   {
      ...
      case MBOOK_BUY_ORDERS:
         return SymbolInfoInteger(m_symbol, SYMBOL_SESSION_BUY_ORDERS);
      case MBOOK_SELL_ORDERS:
         return SymbolInfoInteger(m_symbol, SYMBOL_SESSION_SELL_ORDERS);
      ...
   }
   return 0;
}
```

```
double CMarketBook::InfoGetDouble(ENUM_MBOOK_INFO_DOUBLE property)
{
   switch(property)
   {
      ...
      case MBOOK_BUY_ORDERS_VOLUME:
         return SymbolInfoDouble(m_symbol, SYMBOL_SESSION_BUY_ORDERS_VOLUME);
      case MBOOK_SELL_ORDERS_VOLUME:
         return SymbolInfoDouble(m_symbol, SYMBOL_SESSION_SELL_ORDERS_VOLUME);
      case MBOOK_OPEN_INTEREST:
         return SymbolInfoDouble(m_symbol, SYMBOL_SESSION_INTEREST);
   }
   return 0.0;
}
```

As you can see, the code is quite simple. In fact, it duplicates the standard functionality of MQL. But the point of adding it to the CMarketBook class is to provide users with a convenient and _centralized_ module for accessing information on limit orders and their price levels.

### Chapter 4. Documentation for CMarketBook class

We have completed the description and creation of the class for operation with CMarketBook Depth of Market. The fourth chapter contains documentation for its public methods. Using these documents the class operation becomes simple and straightforward, even for entry-level programmers. Also, this chapter is convenient for being used as a small guidebook for working with the class.

**4.1. Methods of obtaining basic information from Depth of Market and operation with it**

**Refresh() method**

It updates Depth of Market condition. For every call of OnBookEvent system event (Depth of Market has changed), it is also required to call this method.

```
void        Refresh(void);
```

_**Use**_

Find the example of usage in the relevant section of the fourth chapter.

**InfoGetInteger() method**

Returns one of the Depth of Market properties corresponding to the ENUM\_MBOOK\_INFO\_INTEGER modifier. A complete list of supported features can be found in the Description of the [ENUM\_MBOOK\_INFO\_INTEGER](https://www.mql5.com/en/articles/1793#ENUM_MBOOK_INFO_INTEGER) listing.

```
long        InfoGetInteger(ENUM_MBOOK_INFO_INTEGER property);
```

_**Returned value**_

Integer-valued property of Depth of Market long type. In case of failure returns -1.

_**Use**_

Find the example of usage in the relevant section of the fourth chapter.

**InfoGetDouble() method**

It returns one of the Depth of Market properties corresponding to the ENUM\_MBOOK\_INFO\_DOUBLE modifier. A complete list of supported features can be found in the description of the [ENUM\_MBOOK\_INFO\_DOUBLE](https://www.mql5.com/en/articles/1793#ENUM_MBOOK_INFO_DOUBLE) listing.

```
double      InfoGetDouble(ENUM_MBOOK_INFO_DOUBLE property);
```

_**Returned value**_

Property of Depth of Market double type. In case of failure, it returns -1.0.

_**Use**_

Find the example of usage in the relevant section of the fourth chapter.

**IsAvailable() Method**

Returns true, if the information on Depth of Market is available, and false if otherwise. This method must be called before operating with Depth of Market class for checking the possibility of working with this type of information.

```
bool        IsAvailable(void);
```

_**Returned value**_

True, if Depth of Market is available for further operation, and false if otherwise.

**SetMarketBookSymbol() method**

Sets the symbol, whose Depth of Market we will be required to work with. It is also possible to set a symbol of Depth of Market when creating an instance of CMarketBook class, clearly stating the name of used symbol in the constructor.

```
bool        SetMarketBookSymbol(string symbol);
```

_**Returned value**_

True, if 'symbol' is available for trading, false if otherwise.

**GetMarketBookSymbol() Method**

It returns the symbol of an instrument, for whose operation with Depth of Market the current instance of the class is set.

```
string      GetMarketBookSymbol(void);
```

_**Returned value**_

Name of the instrument whose Depth of Market displays the current instance of the class. NULL, if an instrument is not selected or not available.

**GetDeviationByVol() Method**

Returns the value of a potential slippage at the market entrance with a market order. This value has an estimated character and the obtained slippage may differ from what has been calculated by this function, if the state of Depth of Market at the point of entry has changed. However, this feature provides a fairly accurate assessment of slippage, which will take place at the market entry, and can be used as a source of additional information.

The method takes two parameters: the amount of the proposed deal and iteration indicating the type of liquidity used at closing a deal. For example, the liquidity of Sell limit orders will be used to Buy and, in this case the type of MBOOK\_ASK will have to be specified as _side_. To Sell, on the contrary, MBOOK\_BID will have to be indicated. For more information on the DOM side, please read a description of ENUM\_BOOK\_SIDE enumeration.

```
double     GetDeviationByVol(long vol, ENUM_MBOOK_SIDE side);
```

_**Parameters:**_

- \[in\] vol — volume of the proposed deal;
- \[in\] side — Depth of Market side which will be used for making a deal.

_**Returned value**_

The amount of potential slippage in the instrument's points.

**4.2. Enumerations and modifiers of CMarketBook class**

**Enumeration of ENUM\_MBOOK\_SIDE**

ENUM\_BOOK\_SIDE enumeration contains modifiers that indicate the type of liquidity. Enumeration and description fields are listed below:

| Field | Description |
| --- | --- |
| MBOOK\_ASK | Indicates the liquidity provided by Sell limit orders. |
| MBOOK\_BID | Indicates the liquidity provided by Buy limit orders. |

_**Note**_

Every market order can be executed by limit orders. Depending on the direction of the order, Buy or Sell limit orders are used. The downside of the Buy deal will be one or few Sell limit orders. The downside of the Sell deal will be one or few Buy limit orders. This way the modifier indicates one out of two Depth of Market parts: Buy side or Sell side. The modifier is used by the GetDeviationByVol function, for operation of which you are required to know which side of liquidity will be used by the expected market deal.

**Enumeration of ENUM\_MBOOK\_INFO\_INTEGER**

Enumeration ENUM\_MBOOK\_INFO\_INTEGER contains property modifiers, which have to be obtained using InfoGetInteger method. Enumeration and description fields are listed below:

| Field | Description |
| --- | --- |
| MBOOK\_BEST\_ASK\_INDEX | Index of best Ask price |
| MBOOK\_BEST\_BID\_INDEX | Index of best Bid price |
| MBOOK\_LAST\_ASK\_INDEX | Index of worst or last Ask price |
| MBOOK\_LAST\_BID\_INDEX | Index of worst or last Bid price |
| MBOOK\_DEPTH\_ASK | Depth of Market from the Ask side or its total number of trade levels |
| MBOOK\_DEPTH\_BID | Depth of Market from the Bid side or its total number of trade levels |
| MBOOK\_DEPTH\_TOTAL | Total Depth of Market or the number of Buy and Sell trading levels |
| MBOOK\_MAX\_ASK\_VOLUME | Maximum Ask volume |
| MBOOK\_MAX\_ASK\_VOLUME\_INDEX | Level index of maximum Ask volume |
| MBOOK\_MAX\_BID\_VOLUME | Maximum Bid volume |
| MBOOK\_MAX\_BID\_VOLUME\_INDEX | Level index of maximum Bid level |
| MBOOK\_ASK\_VOLUME\_TOTAL | Total volume of Sell limit orders available in the current Depth of Market |
| MBOOK\_BID\_VOLUME\_TOTAL | Total volume of Buy limit orders available in the current Depth of Market |
| MBOOK\_BUY\_ORDERS | Total volume of Buy limit orders currently available on the stock market |
| MBOOK\_SELL\_ORDERS | Total volume of Sell limit orders currently available on the stock market |

**Enumeration of ENUM\_MBOOK\_INFO\_DOUBLE**

Enumeration ENUM\_MBOOK\_INFO\_DOUBLE contains property modifiers, which have to be obtained using InfoGetDouble method. Enumeration and description fields are listed below:

| Field | Description |
| --- | --- |
| MBOOK\_BEST\_ASK\_PRICE | Best Ask price |
| MBOOK\_BEST\_BID\_PRICE | Best Bid price |
| MBOOK\_LAST\_ASK\_PRICE | Worst or last Ask price |
| MBOOK\_LAST\_BID\_PRICE | Worst or last Bid price |
| MBOOK\_AVERAGE\_SPREAD | Average difference between the best Bid and Ask, or a spread |
| MBOOK\_OPEN\_INTEREST | Open interest |
| MBOOK\_BUY\_ORDERS\_VOLUME | Number of Buy orders |
| MBOOK\_SELL\_ORDERS\_VOLUME | Number of Sell orders |

**4.3. Example for using CMarketBook class**

This example contains a source code in the form of an Expert Advisor that displays basic Depth of Market information at the point of starting:

```
//+------------------------------------------------------------------+
//|                                               TestMarketBook.mq5 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Trade\MarketBook.mqh>     // Include CMarketBook class
CMarketBook Book(Symbol());         // Initialize class with current instrument

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   PrintMbookInfo();
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---

  }
//+------------------------------------------------------------------+
//| Print MarketBook Info                                            |
//+------------------------------------------------------------------+
void PrintMbookInfo()
  {
   Book.Refresh();                                                   // Update Depth of Market status.
//--- Get main integer-value statistics
   int total=(int)Book.InfoGetInteger(MBOOK_DEPTH_TOTAL);            // Get total Depth of Market
   int total_ask = (int)Book.InfoGetInteger(MBOOK_DEPTH_ASK);        // Get the amount of Sell price levels
   int total_bid = (int)Book.InfoGetInteger(MBOOK_DEPTH_BID);        // Get the amount of Buy price levels
   int best_ask = (int)Book.InfoGetInteger(MBOOK_BEST_ASK_INDEX);    // Get index of best Ask price
   int best_bid = (int)Book.InfoGetInteger(MBOOK_BEST_BID_INDEX);    // Get index of best Bid price

//--- Displaу basic statistics
   printf("TOTAL DEPTH OF MARKET: "+(string)total);
   printf("NUMBER OF PRICE LEVELS FOR SELL: "+(string)total_ask);
   printf("NUMBER OF PRICE LEVELS FOR BUY: "+(string)total_bid);
   printf("INDEX OF BEST ASK PRICE: "+(string)best_ask);
   printf(INDEX OF BEST BID: "+(string)best_bid);

//--- Get main statistics of double
   double best_ask_price = Book.InfoGetDouble(MBOOK_BEST_ASK_PRICE); // Get best Ask price
   double best_bid_price = Book.InfoGetDouble(MBOOK_BEST_BID_PRICE); // Get best Bid price
   double last_ask = Book.InfoGetDouble(MBOOK_LAST_ASK_PRICE);       // Get worst Ask price
   double last_bid = Book.InfoGetDouble(MBOOK_LAST_BID_PRICE);       // Get worst Bid price
   double avrg_spread = Book.InfoGetDouble(MBOOK_AVERAGE_SPREAD);    // Get the average spread during Depth of Market operation

//--- Display prices and spread
   printf("BEST ASK PRICE: " + DoubleToString(best_ask_price, Digits()));
   printf("BEST BID PRICE: " + DoubleToString(best_bid_price, Digits()));
   printf("WORST ASK PRICE: " + DoubleToString(last_ask, Digits()));
   printf("WORST BID PRICE: " + DoubleToString(last_bid, Digits()));
   printf("AVERAGE SPREAD: " + DoubleToString(avrg_spread, Digits()));
  }
//+------------------------------------------------------------------+
```

### Conclusion

The article turned out to be rather dynamic. We have analyzed Depth of Market from a technical perspective and proposed a high-performance class-container to work with it. As an example, we have created a Depth of Market indicator based on this class-container, which can be compactly displayed on the instrument's price chart.

Our Depth of Market indicator is very basic and it still lacks a number of things. However, the main objective was achieved - we made sure that with a CMarketBook class that we have created we can relatively quickly build complex Expert Advisors and indicators for analyzing current liquidity with the instrument. When designing the CMarketBook class a lot of attention has been paid to performance, since Depth of Market has a very dynamic table that changes hundreds of times per minute.

The class described in the article can become a solid base for your scalper or a high-frequency system. Feel free to add functionality specific to your system. To do this, simply create your Depth of Market class derived from CMarketBook, and write extension methods you will need. We do hope, however, that even those basic properties provided by Depth of Market will make your work easier and more reliable.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1793](https://www.mql5.com/ru/articles/1793)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1793.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/1793/mql5.zip "Download MQL5.zip")(202.33 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/65168)**
(44)


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
27 Oct 2025 at 15:52

**Rashid Umarov [#](https://www.mql5.com/ru/forum/60996/page4#comment_58370444):**

I wanted to replace the files in the article, but it turned out that the stastav files are very different.

Maybe it's a different version. I'll check it now.


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
27 Oct 2025 at 16:02

Yes, different version. This 2015 article, contains the code from the second 2017 follow-up article [https://www.mql5.com/en/articles/3336](https://www.mql5.com/en/articles/3336). Unfortunately, it is not compiled. For training purposes you can use the original code base from 2015, in it the price glass works without a chart. I have published the code above. As for the new version, it will take some time to fix it.


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
27 Oct 2025 at 18:11

**Vasiliy Sokolov [#](https://www.mql5.com/ru/forum/60996/page5#comment_58371101):**

Unfortunately, it does not compile.

There are only a couple of errors in the code attached to this article.

In the **MBookFon.mqh** file, the method implementation is not defined correctly - void is missing:

```
void  CBookFon::Show(void)
{
   ObjectCreate(ChartID(), m_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, 13);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XDISTANCE, 6);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XSIZE, 116);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrWhite);
   int total = (int)m_book.InfoGetInteger(MBOOK_DEPTH_TOTAL);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YSIZE, total*15+16);
   CreateCeils();

   OnShow();
}
```

In the **MBookCeil.mqh** file, there is no check for array size and array overrun:

```
void CBookCeil::Refresh(void)
{
   int total = (int)m_book.InfoGetInteger(MBOOK_DEPTH_TOTAL);
   if(total == 0 || m_index < 0 || m_index > total-1)
      return;
   ENUM_BOOK_TYPE type = m_book.MarketBook[m_index].type;
   long max_volume = 0;
   if(type == BOOK_TYPE_BUY || type == BOOK_TYPE_BUY_MARKET)
   {
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrCornflowerBlue);
      max_volume = m_book.InfoGetInteger(MBOOK_MAX_BID_VOLUME);
   }
   else if(type == BOOK_TYPE_SELL || type == BOOK_TYPE_SELL_MARKET)
   {
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrPink);
      max_volume = m_book.InfoGetInteger(MBOOK_MAX_ASK_VOLUME);
   }
   else
      ObjectSetInteger(ChartID(), m_name, OBJPROP_BGCOLOR, clrWhite);
   MqlBookInfo info = m_book.MarketBook[m_index];
   if(m_ceil_type == BOOK_PRICE)
      m_text.SetText(DoubleToString(info.price, Digits()));
   else if(m_ceil_type == BOOK_VOLUME)
      m_text.SetText((string)info.volume);
   if(m_ceil_type != BOOK_VOLUME)return;
   double delta = 1.0;
   if(max_volume > 0)
      delta = (info.volume/(double)max_volume);
   if(delta > 1.0)delta = 1.0;
   long size = (long)(delta * 50.0);
   if(size == 0)size = 1;
   ObjectSetInteger(ChartID(), m_name, OBJPROP_XSIZE, size);
   ObjectSetInteger(ChartID(), m_name, OBJPROP_YDISTANCE, m_ydist);
}
```

At the first run, when the price glass is not connected yet, the array size is zero and, accordingly, the critical error crashes.

![startatrix](https://c.mql5.com/avatar/avatar_na2.png)

**[startatrix](https://www.mql5.com/en/users/startatrix)**
\|
4 Nov 2025 at 22:23

I compiled the files, however there are still errors. Can someone share a [working version](https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development") please


![Artyom Trishkin](https://c.mql5.com/avatar/2022/7/62C4775C-ABD6.jpg)

**[Artyom Trishkin](https://www.mql5.com/en/users/artmedia70)**
\|
5 Nov 2025 at 04:34

**startatrix [#](https://www.mql5.com/ru/forum/60996/page5#comment_58438918):**

I have compiled the files but the errors remain. Can anyone share a working version please.

If compiled, there are no errors. If failed to compile, what are the errors?


![Indicator for Spindles Charting](https://c.mql5.com/2/19/LOGO__2.png)[Indicator for Spindles Charting](https://www.mql5.com/en/articles/1844)

The article regards spindle chart plotting and its usage in trading strategies and experts. First let's discuss the chart's appearance, plotting and connection with japanese candlestick chart. Next we analyze the indicator's implementation in the source code in the MQL5 language. Let's test the expert based on indicator and formulate the trading strategy.

![How to Secure Your Expert Advisor While Trading on the Moscow Exchange](https://c.mql5.com/2/18/MOEX.png)[How to Secure Your Expert Advisor While Trading on the Moscow Exchange](https://www.mql5.com/en/articles/1683)

The article delves into the trading methods ensuring the security of trading operations at the stock and low-liquidity markets through the example of Moscow Exchange's Derivatives Market. It brings practical approach to the trading theory described in the article "Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market".

![Handling ZIP Archives in Pure MQL5](https://c.mql5.com/2/19/Icon3.png)[Handling ZIP Archives in Pure MQL5](https://www.mql5.com/en/articles/1971)

The MQL5 language keeps evolving, and its new features for working with data are constantly being added. Due to innovation it has recently become possible to operate with ZIP archives using regular MQL5 tools without getting third party DLL libraries involved. This article focuses on how this is done and provides the CZip class, which is a universal tool for reading, creating and modifying ZIP archives, as an example.

![Drawing Resistance and Support Levels Using MQL5](https://c.mql5.com/2/19/avatar__1.png)[Drawing Resistance and Support Levels Using MQL5](https://www.mql5.com/en/articles/1742)

This article describes a method of finding four extremum points for drawing support and resistance levels based on them. In order to find extremums on a chart of a currency pair, RSI indicator is used. To give an example, we have provided an indicator code that displays support and resistance levels.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1793&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062504241639105376)

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