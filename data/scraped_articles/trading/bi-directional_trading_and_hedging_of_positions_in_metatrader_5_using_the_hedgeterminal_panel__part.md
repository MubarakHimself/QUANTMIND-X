---
title: Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1
url: https://www.mql5.com/en/articles/1297
categories: Trading, Integration
relevance_score: 3
scraped_at: 2026-01-23T18:19:08.399232
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/1297&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069338750082810729)

MetaTrader 5 / Trading


### Table of Contents

- [INTRODUCTION](https://www.mql5.com/en/articles/1297#intro)
  - [What is This Article About?](https://www.mql5.com/en/articles/1297#i1)
  - [How to Read this Article](https://www.mql5.com/en/articles/1297#i2)
- [CHAPTER 1. ORGANIZATION THEORY OF BI-DIRECTIONAL TRADING](https://www.mql5.com/en/articles/1297#chapter1)
  - [1.1. MetaTrader 5 Opportunities in Organizing Bi-Directional Trading](https://www.mql5.com/en/articles/1297#c1_1)
  - [1.2. Pairing orders - Basis of Hedging and Statistics](https://www.mql5.com/en/articles/1297#c1_2)
  - [1.3. Relationship of the MetaTrader 5 Net Positions and the HedgeTerminal Positions](https://www.mql5.com/en/articles/1297#c1_3)
  - [1.4. Requirements of the Algorithms Implementing Bi-Directional Trading](https://www.mql5.com/en/articles/1297#c1_4)
- [CHAPTER 2. INSTALLATION OF HEDGE TERMINAL, FIRST LAUNCH](https://www.mql5.com/en/articles/1297#chapter2)
  - [2.1. Installation of HedgeTerminal](https://www.mql5.com/en/articles/1297#c2_1)
  - [2.2. Three Step Installation. Installation Diagram and Solution to Possible Problems](https://www.mql5.com/en/articles/1297#c2_2)
  - [2.3. Getting Started with HedgeTerminal, First Launch](https://www.mql5.com/en/articles/1297#c2_3)
  - [2.4. Hedging and Calculation of Financial Operations](https://www.mql5.com/en/articles/1297#c2_4)
  - [2.5. One Click Trading](https://www.mql5.com/en/articles/1297#c2_5)
  - [2.6. Placing StopLoss and TakeProfit, Trailing Stop](https://www.mql5.com/en/articles/1297#c2_6)
  - [2.7. Report Generation](https://www.mql5.com/en/articles/1297#c2_7)
  - [2.8. Currency Swap Presentation](https://www.mql5.com/en/articles/1297#c2_8)
  - [2.9. Bottom Row](https://www.mql5.com/en/articles/1297#c2_9)
  - [2.10. Changing the Appearance of HedgeTerminal Tables](https://www.mql5.com/en/articles/1297#c2_10)
  - [2.11. Planned Yet Unimplemented Features](https://www.mql5.com/en/articles/1297#c2_11)
- [CHAPTER 3. UNDER THE BONNET OF HEDGE TERMINAL. SPECIFICATION AND PRINCIPLES OF OPERATIONS](https://www.mql5.com/en/articles/1297#chapter3)
  - [3.1. Global and Local Contours. Context, Transfer and Storage of Information](https://www.mql5.com/en/articles/1297#c3_1)
  - [3.2. Storing Global and Local Information](https://www.mql5.com/en/articles/1297#c3_2)
  - [3.3. Stop Loss and Take Profit Levels. Order System Issues and OCO Orders](https://www.mql5.com/en/articles/1297#c3_3)
  - [3.4. Can OCO Orders Resolve Issues with Protection of Bi-Directional Positions?](https://www.mql5.com/en/articles/1297#c3_4)
  - [3.5. Storing Links to Initializing Orders](https://www.mql5.com/en/articles/1297#c3_5)
  - [3.6. Limitations at Work with HedgeTerminal](https://www.mql5.com/en/articles/1297#c3_6)
  - [3.7. Mechanism of Pairing Orders and Determinism of Actions](https://www.mql5.com/en/articles/1297#c3_7)
  - [3.8. Splitting and Connecting Deals - the Basis of the Order Arithmetics](https://www.mql5.com/en/articles/1297#c3_8)
  - [3.9. Order and Deal Virtualization](https://www.mql5.com/en/articles/1297#c3_9)
  - [3.10. Mechanism of Hiding Orders](https://www.mql5.com/en/articles/1297#c3_10)
  - [3.11. Adaptation Mechanisms](https://www.mql5.com/en/articles/1297#c3_11)
  - [3.12. Performance and Memory Usage](https://www.mql5.com/en/articles/1297#c3_12)
- [CONCLUSION](https://www.mql5.com/en/articles/1297#summary)

### Introduction

Within the last 18 months MetaQuotes have conducted extensive work on consolidating the MetaTrader 4 and MetaTrader 5 platforms into a unified trading ecosystem. Now both platforms share a common market of program solutions - [Market](https://www.mql5.com/en/market), offering different products from external developers. The compilers for both platforms were united as well. As a result both platforms have a common compiler based on MQL5 and one programming language - MQL with a different function set depending on the platform in use. All publicly available source codes located in the [Code Base](https://www.mql5.com/en/code) were also revised and some of them were adjusted to be compatible with the new compiler.

This major unification of the platforms left aside the unification of their trading parts. The trading models of MetaTrader 4 and MetaTrader 5 are still fundamentally incompatible despite the fact that the major part of the trading environment is shared. MetaTrader 4 facilitates individual management of trading positions through the system of orders - special program entities making the bi-directional trading in this terminal simple and easy. MetaTrader 5 is intended for the exchange trade where the main representation of a trader's obligations is their aggregate net position. Orders in MetaTrader 5 are simply instructions to buy or sell a financial instrument.

The difference between the trading performance of these two platforms caused a lot of heated discussions and debates. However, discussions remained discussions. Unfortunately, since the release of MetaTrader 5, not a single working solution was published that could enable presenting a trader's obligations as bi-directed positions like in MetaTrader 4. Although numerous published articles suggest various solutions, they are not flexible enough to be conveniently used on a large scale. In addition, none of those decisions are suitable for the exchange trading, which involves a lot of nuances that have to be taken into consideration.

This article should settle the controversies between the fans of the fourth and the fifth versions of the MetaTrader platform. This is going to provide a universal solution in the form of thorough program specification and the exact program solution implemented by this specification. This article discusses a visual panel and the virtualization library _**HedgeTerminal**_, which enable presenting a trader's obligation as bi-directional positions like in MetaTrader 4. At the same time, the model underlying HedgeTerminal takes into account the character of the trading orders execution. That means that it can be successfully implemented both at the over-the-counter market FOREX and centralized exchanges like, for example, trading derivative securities in the derivatives section of [Moscow Exchange](https://www.mql5.com/go?link=http://www.moex.com/ "http://moex.com/").

HedgeTerminal is a fully featured trading terminal inside the MetaTrader 5 terminal. Through the mechanism of _**virtualization**_ it changes the representation of the current positions so a trader or a trading robot can manage their individual trading positions. Neither the number of positions, nor their direction have any importance. I would like to place emphasis on the fact that we are talking about virtualization - a specific mechanism transforming the presentation of a trader's obligations but not their qualitative characteristics.

This is not about distorting the results of the trader's financial activity but about transforming representation of this activity. HedgeTerminal is based on the MetaTrader 5 trading environment and the MQL5 programming language. It does not bring new trading information to the terminal, it simply makes the current trading environment seen from a different angle. That means that HedgeTerminal is essentially MetaTrader 5 and is its native application. Even though, we could put an identity sign between them, an inclusion sign is more appropriate as HedgeTerminal is just a small and one of many MetaTrader 5 applications.

The virtualization possibility and the HedgeTerminal existence are based on three paradigms:

1. _Conceptually, a complete and guaranteed convertibility of the position net representation into individual bi-directional trading transactions is possible_. This statement proves the fact that in some external trading platforms including the ones designed for exchange trading, there is a means to manage bi-directional positions;
2. _The trading model of MetaTrader 5 at the user's level allows creating one way links of orders to each other_. Calculations have shown that links designed in a certain way will be resistant to collisions. Moreover, even in the case of a corrupted history or any force majeure situations, bi-directional transactions could be retrospectively corrected and brought down to a financial result calculated at net representation;
3. _Advanced API in MQL5 allows to put the identity sign between MQL5 and the MetaTrader 5 terminal_. In other words, nearly everything in MetaTrader 5 is accessible through the program interface and the MQL5 programming language. For instance, your own version of the terminal can be written inside the MetaTrader 5 terminal like HedgeTerminal.

This article discusses the underlying algorithms and how HedgeTerminal works. The specifications and algorithms ensuring consistent bi-directional trading are discussed in this article in detail. Irrespective of whether you decide to use HedgeTerminal or to create your own library to manage your own trading algorithms, you will find useful information in this article and its continuation.

This article is not targeting programmers specifically. If you do not have any experience in programming, you are still going to understand it. The MQL source code is not included in this article deliberately. All source code was replaced with more illustrative diagrams, tables and pictures schematically representing the operational principle and the data organization. I can tell from experience that even having a good grasp of programming principles, it is a lot easier to see a common pattern of the code than to analyze it.

In the second part of this article, we are going to talk about the integration of Expert Advisors with the HedgeTerminalAPI visualization library and then programming will be involved. However, even in that case, everything was done to simplify the perception of the code especially for novice programmers. For instance, object-oriented constructions like classes are not used though HedgeTerminal is an object oriented application.

**How to Read this Article**

This article is rather lengthy. On the one hand, this is good as the answer to virtually any question regarding this topic can be found here. On the other hand, many users would prefer to read only the most important information returning to the relevant section whenever necessity arises. This article covers in full a consistent presentation of the material. Similar to books, we are going to give a short summary of every chapter so you can see whether you need to read it or not.

- _**Part 1, Chapter 1. Theory of the bi-directional trading organization.**_ This chapter contains the main ideas of HedgeTerminal. If you do not want in-depth information about organization of bi-directional trading, this chapter is sufficient to get an idea of the general principles of HedgeTerminal operation. This chapter is recommended for all readers.

- _**Part 1, Chapter 2. Installation of HedgeTerminal, first launch.**_ This chapter describes the launch and setting up the visual panel of HedgeTerminal. If you are not going to use a visual panel of HedgeTerminal, then you can skip this chapter. If you, however, are going to use the HedgeTerminalAPI library, you will need to go through sections 2.1 and 2.2 of this chapter. They are dedicated to the HedgeTerminal installer. The installer is the common component for all the HedgeTerminal products.

- _**Part 1, Chapter 3. Under the bonnet of HedgeTerminal. Specification and principle of operations.**_ This chapter highlights the internal arrangement of HedgeTerminal, its algorithm and internal data organization. Those who are interested in bi-directional trading may find this chapter informative. Professional algorithmic traders developing their own virtualization libraries and using a lot of robots simultaneously can find this chapter useful too.

- _**Part 2, Chapter 1. Communication of Expert Advisors with HedgeTerminal and its panel**_. This chapter will be appreciated by those who are just exploring this aspect and those who are in algorithmic trading professionally. It describes common principles of work with the HedgeTerminal library and the architecture of the trading robot.

- _**Part 2, Chapter 2. Documentation to the API HedgeTerminal.**_ Contains documentation on using the functions of the HedgeTerminalAPI library. The chapter is written as a list of documentation that could be referred to from time to time. There are no unnecessary discussions and excessive text in it. This chapter contains only prototypes of function, structures and enumerations as well as brief examples of their usage.

- _**Part 2, Chapter 3. Basics of asynchronous trading operations.**_ HedgeTerminal uses asynchronous operations in its work. This chapter comprises the experience of using them. This chapter was included in this article so any reader could find some useful information irrespective of their plans to use the HedgeTerminal in their work.

- _**Part 2, Chapter 4. Basics of multi-threaded programming in the MetaTrader environment.**_ This chapter explains what multi-threading is, how to arrange it and what patterns of data organization can be used. Like Chapter 3 it shares the experience of multi-threaded application development with all MetaTrader users.

I hope that this article is interesting enough so you read it up to the end.

### Chapter 1. Organization theory of bi-directional trading

**1.1. MetaTrader 5 Opportunities in Organizing Bi-Directional Trading**

The article ["Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market"](https://www.mql5.com/en/articles/1284) carefully describes the nuances of the exchange price formation and methods of financial result calculation of the market players. It outlines that pricing and calculation on the Moscow Exchange significantly differ from concepts and calculation methods accepted in the FOREX trading.

On the whole, formation of the exchange price is more complex and contains a lot of significant details hidden when trading Forex and in the MetaTrader 4 terminal.

For example, in MetaTrader 4 the deals that executed the trader's order are hidden whereas in MetaTrader 5 such information is available. On the other hand, detailed trade information in MetaTrader 5 is not always required. It may make work difficult for an inexperienced user or a novice programmer and cause misunderstanding. For example, in MetaTrader 4, to find out the price of the executed order, you simple need to look up the correspondent value in the "Price" column. In the MQL4 programming language it is enough to call the [OrderOpenPrice()](https://docs.mql4.com/en/trading/orderopenprice "https://docs.mql4.com/en/trading/orderopenprice") function. In MetaTrader 5 it is required to find all the deals that executed the order and then go over them and calculate their weighted average price. This very price is the price of the order execution.

There are other situations when extended representation of the trading environment in MetaTrader 5 requires additional efforts to analyze this information. That prompts logical questions:

_Is there a way of making the trading process in MetaTrader 5 as simple and clear as in MetaTrader 4 and keep a convenient access to all required trade details? If there is a way to arrange bi-directional exchange trading using MetaTrader 5 the same simple way as MetaTrader 4? - The answer to those questions is: "Yes, there is"!_

Let us refer to the diagram of capabilities of the MetaTrader 4 and MetaTrader 5 terminals to understand how this is possible:

![Fig. 1 Capabilities of MetaTrader 4 and MetaTrader 5](https://c.mql5.com/2/18/ez7rt297ywn_s245.png)

Fig. 1 Capabilities of MetaTrader 4 and MetaTrader 5

As we can see, the "MetaTrader 5" set includes the "MetaTrader 4" subset. It means that everything possible in MetaTrader 4 can be done in MetaTrader 5 though the converse is false. New capabilities of MetaTrader 5 inevitably increase the amount and the complexity of the trade information presentation. This difficulty can be delegated to special assistant programs working in the MetaTrader 5 environment. These programs can process the complexity, leaving the terminal capabilities at the same level. One of such programs _**HedgeTerminal**_ is the focus of this chapter.

HedgeTerminal is a full-fledged trading terminal inside the MetaTrader 5 terminal. It uses the MetaTrader 5 trading environment, transforms it with the MQL5 language and represents it as a convenient graphical interface - panel _**HedgeTerminalUltimate**_ and special interface _**HedgeTerminalAPI**_ for interaction with independent algorithms (Expert Advisors, scripts and indicators).

MetaTrader 4 features a possibility to use _**bi-directional positions**_ or **_locking orders_**. In MetaTrader 5 there is such capability too but it is not explicit. This can be enabled using a specific add-on program, which HedgeTerminal essentially is. HedgeTerminal is built in MetaTrader 5 and uses its environment, gathering information by deals and orders into integrated positions looking very similar to orders in MetaTrader 4 and having all the capabilities of MetaTrader 5.

Such positions can be in a complete or partial locking order (when active long and short positions exist at the same time). The opportunity of maintaining such positions is not a goal in itself for HedgeTerminal. Its main objective is to unite trade information in unified groups (positions) that would be easy to analyze, manage and get access to. Bi-directional positions can exist in HedgeTerminal only because this is very convenient. In the case several traders are trading on one account, or more than one strategy is used, splitting trading actions must be arranged.

In addition, a number of nuances have to be taken into consideration at exchange trading such as partial order execution, position rollover, variation margin calculation, statistics and many others. HedgeTerminal was developed for meeting those challenges. It provides the user or an Expert Advisor a regular high level interface resembling MetaTrader 4 and at the same time working correctly in the exchange environment.

**1.2. Pairing orders - Basis of Hedging and Statistics**

To be able to manage trading techniques and algorithms consistently, it is required to know _with certainty_ what trading action belongs to what algorithm.  I highlighted the words "with certainty" because if there is even the smallest probability of failure, crash of position management is inevitable sooner or later. In its turn, it will result in damage of statistics and undermining the idea of managing different algorithms on one account.

A reliable separation of trade activities is based on two fundamental possibilities:

1. The possibility of _**uniting**_ or " _**pairing**_" two trading orders together so it can always be defined which of the two orders is opening the individual (virtual) position and which of them is closing this position;
2. The algorithm, analyzing orders for their pairing must be completely _**deterministic**_ and unified for all program modules.


The second requirement for the algorithm determinism will be considered in detail below. Now we are going to focus on the first one.

An order is an instruction to sell or buy. An order is a defined entity, which includes many more information "fields" in addition to the main information like magic number and order number, required price and opening conditions.

One of such fields in MetaTrader 5 is called " _**Order Magic**_". This is a specified field used for a trading robot or an Expert Advisor to be able to mark this order with its individual unique number also called the "magic number". This field is not used in manual trade though it is very important for trading algorithms because when a trading algorithm analyses the values of this field, it can always see if the order in question was placed by it or any other algorithm.

Let us take a look at an example. Let us assume that we need to open a classical long position and then, in some time, close it. For that we have to place two orders. The first order will open this position and the second order will close it:

![Fig. 2. Orders forming a historical net position](https://c.mql5.com/2/12/m09ismo_1.png)

Fig. 2. Orders forming a historical net position

What if we write the magic number of the first order in the field "Order Magic" of the second order at the moment of sending it to the market?

Later, this order's field can be read and if the value is equal to the number of the first order, then we can definitely say that the second order is related to the first one and is opposite to it, i.e. a closing order.

On a diagram this pairing would look like:

![Fig. 3. Pairing orders](https://c.mql5.com/2/12/jjvd4m8.png)

Fig. 3. Pairing orders

Nominally such orders can be called _**paired**_ as the second order contains a link to the first one. The first order opening a new position can be called an _**initiating**_ or _**opening**_ order. The second order can be called _**closing**_.

A pair of such orders can be called _**position**_. To avoid confusion with the concept of "position" in MetaTrader 5, we are going to call such paired positions **_bi-directional,_ _**hedging**_ or _**the HedgeTerminal**_ positions. Positions in MetaTrader 5 will be called _**net positions**_ or the _**MetaTrader 5**_ classical positions.**

Apparently, the number of HedgeTerminal positions and their directions, unlike the classical ones, can be any. What if there is an executed order that is not referred to by any other order? Such an order can be presented as an active bi-directional position. In fact, if an opposite order that contains a link to this order is placed, it will become an order that closes the first order. Such orders will become paired and make a _**closed**_ bi-directional position as the volume of two orders is going to be equal and their direction is opposite.

So, let us define what is meant in HedgeTerminal by position:

_If an executed order is not referred to by any other order, then HedgeTerminal treats such an order as_ _**an active bi-directional position**_.

_If one executed order is referred to by another executed order, then two such orders make a pair and are treated by HedgeTerminal as one unified **historical** or **closed** bi-directional position._

In reality, pairing orders in HedgeTerminal is more complicated, as every order generates a minimum one deal and in exchange trade there can be many such deals. In general, the trade process can be presented as follows: a trader places an order to open a new position through the MetaTrader 5 terminal. The exchange executes this order through one or several deals.

Deals, similar to orders, contain fields for additional information. One of such fields contains the order id, on the basis of which the deal was executed. This field contains information about the order a certain deal belongs to. The converse is false. The order itself does not know what deals belong to it. It happens because at the time of placing the order it is not clear what deals will execute the order and if the order is going to be executed at all.

This way, causality or determinism of actions is observed. Deals are referring to orders and orders are referring to each other. Such a structure can be presented as a [singly linked list](https://en.wikipedia.org/wiki/Linked_list "https://en.wikipedia.org/wiki/Linked_list").

Executed deals of the opening order generate a classical position in MetaTrader 5 and deals that belong to a closing order, on the contrary, close this position. These pairs are presented in the figure below:

![Fig. 4. Diagram of a relationship between orders, deals and exchange](https://c.mql5.com/2/12/xndqaclzh9_9pwydy5_q_vd75ya.png)

Fig. 4. Diagram of a relationship between orders, deals and exchange

We are going to get back to the detailed analysis of this diagram as the direction of the relationships is very important for building a strictly determined system of the trader's actions recording, which is HedgeTerminal.

**1.3. Relationship of the MetaTrader 5 Net Positions and the HedgeTerminal Positions**

From the point of view of HedgeTerminal, two opposite orders with the same volumes can be two different positions. In this case their net position will be zero. That is why HedgeTerminal does not use information about the factual net positions opened in MetaTrader 5. Therefore, _**positions in HedgeTerminal are not connected with positions in MetaTrader 5**_. The only time when the current net positions get verified is the moment of the HedgeTerminal launch. The total volumes of the opposite active positions must be identical to the values of the factual net positions.

If this is not the case, then a warning exclamation mark in a frame will appear on the HedgeTerminal, ![Positions are not equal](https://c.mql5.com/2/12/lty9y4gm97w1640_30pk.png)indicating that positions in HedgeTerminal and positions in MetaTrader 5 are not equal. This asymmetry does not affect the efficiency of HedgeTerminal, though for further correct work it has to be eliminated.

In the majority of cases, it can appear when users make errors editing the files of excluded orders _ExcludeOrders.xml_, though corruption of the order and deal history on the server can also cause this sign to emerge. In any case, these discrepancies can be removed by the exclusion mechanism implemented through the _ExcludeOrders.xml_ file.

**1.4. Requirements of the Algorithms Implementing Bi-Directional Trading**

Rather strict requirements are imposed on the algorithms implementing bi-directional trading. They must be met when developing HedgeTerminal. Otherwise, HedgeTerminal would quickly turn into a program working stochastically. The program would be "yet another solution that could operate or fail to do so with the same probability"

Below are some requirements specified for its development:

1. The representation of traders' bi-directional positions must be reliable. Any idea implemented in HedgeTerminal must not lead to ambiguity or potential errors in the business logic. If some property or opportunity does not meet these requirements, then this idea won't be used in spite of its convenience and the demand for it;
2. All algorithms must be based on the MetaTrader 5 trading environment as much as possible. Storing additional information in the files is permitted in the cases when this is strictly necessary. The virtualization algorithms must receive the major part of information from the trading environment. This property enhances the total level of reliability as the majority of changes are conveyed through the server and therefore are accessible from any point in the world;
3. All actions on the trading environment transformation must be performed behind the scenes. Complex configuration and initialization of the API HedgeTerminal library should not be required. A user should be able to launch the application "out of the box" and receive _the expected_ result;
4. The HedgeTerminal visual panel must fit the general MetaTrader 5 interface as seamlessly as possible, giving simple and understandable visual tools for work with bi-directional positions familiar to all the users of MetaTrader 4 and 5. In other words, the visual panel must be _intuitively clear and simple_ for all users;
5. The visual panel of HedgeTerminal must be designed taking into consideration algorithmic traders' high requirements. For instance, the panel must be configurable, the user must be able to change its appearance and even add custom modules;
6. It must provide the Intuitive and simple program interface (API) of interactions with external Experts. The program part of interactions between external Experts and the HedgeTerminal algorithms must fit seamlessly the existing standard of the program interaction of the custom Experts with the MetaTrader 4/5 system functions. Actually, the HedgeTerminal API appears to be a hybrid of the MetaTrader 4 and MetaTrader 5 APIs;
7. HedgeTerminal must ensure reliable work in the exchange environment, taking into account all the nuances of the exchange orders execution. HedgeTerminal is written based on the canonical article [Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)". Initially, this article was a part of a long article about HedgeTerminal, which later was divided into several independent articles because of the volume. One can say that HedgeTerminal is a program implementation of the ideas discussed in this article.

Many of these ideas are not quite connected with each other. For instance, the abundance of the exchange information that is to be reflected in the exchange trading is difficult to connect with the simplicity of this information representation.

Another difficulty was to create a panel that could be easy to use for novice traders and at the same time provide extensive opportunities for professional algorithmic traders. Nevertheless, assessing the result, it is clear that these mutually exclusive properties were successfully implemented in HedgeTerminal.

### Chapter 2. Installation of HedgeTerminal, First Launch

**2.1. Installation of HedgeTerminal**

We know that all executed orders in HedgeTerminal are considered as positions. Positions can consist either of two paired orders, which make a closed historical position or an unbound order, which makes an open or active position.

If before HedgeTerminal was installed, there were some actions on the trading account and the trading history contains a lot of executed orders, then from the point of view of HedgeTerminal all these orders will be open positions as the links between them were not created. It does not matter if the account history contains 2-3 executed orders. If there are thousands of them, HedgeTerminal will generate thousands of open positions. What can one do with them? These orders can be "closed" by placing opposite orders containing links to the initiating orders through HedgeTerminal. There is a down side though. If by the time of installing HedgeTerminal there are too many of them, it can ruin the trader as the brokerage fee and slippage expenses are to be paid.

To avoid this, HedgeTerminal launches a dedicated installation wizard at the beginning of its installation, where different solutions of this problem are suggested. Let us launch HedgeTerminal, call this wizard and describe its work in detail. For that download and install [HedgeTerminalDemo](https://www.mql5.com/en/market/product/5084) from the MetaTrader 5 Market.

Similar to [HedgeTerminalUltimate](https://www.mql5.com/en/market/product/5011), it has a form of an Expert Advisor and all it takes to launch it is dragging its icon available in the "Navigator" folder to a free chart.

Dragging this icon will bring up a standard window suggesting to launch the Expert Advisor on the chart:

![](https://c.mql5.com/2/17/HedgeTerminalDemo.png)

Fig. 5. The HedgeTerminal window before the launch

At this stage it is enough to set the flag "Allow AutoTrading", allowing the EA to perform trade actions. HedgeTerminal will follow your orders as it does not have its own trading logic but for executing trades it will still need your permission.

For any Expert Advisor to start trading, the general permission for trading through Expert Advisors must be enabled on the panel in addition to the personal permission to trade in MetaTrader 5.

![Fig. 6. Enabling automated trading](https://c.mql5.com/2/12/AutoTrading.png)

Fig. 6. Enabling automated trading

HedgeTerminal was designed so the user could avoid long and complex configuration.

That is why all available settings are included in a special XML text file. The only explicit parameter for HedgeTerminal is the actual name of these settings file:

![Fig. 7. Settings window of the HedgeTerminal panel](https://c.mql5.com/2/12/HT_-_Settings.png)

Fig. 7. Settings window of the HedgeTerminal panel

The nature of these settings and the way to change them will be discussed later.

After pressing "OK", the HedgeTerminal installation wizard launches suggesting to start the installation process. The process of installation goes down to creating a few files in the shared directory for the terminals MetaTrader 4 and MetaTrader 5.

HedgeTerminal requests permission to install such files:

![](https://c.mql5.com/2/17/HTSetup_-_InstallBeginDialog2.png)

Fig. 8. Dialog of the installation start

If you do not want some files to be installed on your computer, press "Cancel". In this case HedgeTerminal will finish work. To continue installation press "OK".

The appearance of the following dialog will depend on the account you launch HedgeTerminal with. If no trades were executed on the trading account, then HedgeTerminal will complete its work.

If some trades have already been executed, HedgeTerminal will display the following dialog:

![](https://c.mql5.com/2/17/HTSetup_-_YesNoCancel.png)

Fig. 9. Dialog detecting the first launch of HedgeTerminal

On the figure above, HedgeTerminal identified 5 active orders. In your case their number will be different (it is likely to be big and equal to the total number of executed orders of the account lifetime). These orders do not make a pair with a closing order as from the point of view of HedgeTerminal, they are active positions.

HedgeTerminal suggests several options.

1. Exclude these orders from HedgeTerminal: _"You can hide them in HedgeTerminal... To hide these orders, click 'YES' and go into the next step"_. If you choose this option and press _"YES"_, HedgeTerminal will put them into a specified list and from then will stop accounting their contribution to the aggregate net position as well as their financial result. These orders can be placed in the list only if there is no currently open net position. If you have an open position on one or several symbols, HedgeTerminal will call an additional dialog suggesting to close existing positions.
2. Leave orders as they are and close them later if required: _"You can… close manually later... Click 'No' if you want close these orders manually later. In this case, you have to perform 5 trades of opposite direction"_. In this case, if you press _"No"_, HedgeTerminal will reflect after launch all these orders in the _"Active"_ tab (i.e. as active positions). Later these orders can be closed with other orders through the HedgeTerminal panel. After that they will turn into closed positions and will be transferred to the _"History"_ tab (historical positions). If these orders are great in number, then it is better to hide them than to close all executed orders again and pay brokerage fees.
3. You can stop the installation: _"If you are not ready continue press Cancel. In this case HedgeTerminal complete its work"_. If you choose this option and press "Cancel", HedgeTerminal will stop its work.

If there are no active positions by the time HedgeTerminal is installed, installation will be stopped at this stage.

If you have selected the second option and you currently have one or several opened positions, HedgeTerminal will call an additional dialog suggesting to close them:

![](https://c.mql5.com/2/17/HTSetup_-_AutoClosePositions.png)

Fig. 10. Dialog suggesting to close net positions automatically

HedgeTerminal requires closing all existing positions as all executed orders are to be put into the list of excluded orders. If there are no net positions, then any following order initializes a new net position. The direction and volume in this case are the same as in HedgeTerminal and that ensures avoiding unsynchronization of net positions with the total positions in HedgeTerminal.

HedgeTerminal can automatically close all net positions for you: _"The HedgeTerminal can automatically close all active positions"_. If this is acceptable to you, press _"OK"_. In this case it will try to close positions and if successful will finish work. If positions cannot be closed for some reason, it will move on to the dialog of manual position closing. If you select the manual position closing, _"Click 'Cancel' if you want to close a position manually"_, press _"Cancel"_.

The dialog of manual position closing is called either when manual closure type was chosen in the previous dialog window or when automatic position closure is impossible.

![](https://c.mql5.com/2/17/HTSetup_-_ManualClosePositions.png)

Fig. 11. Dialog suggesting to close net positions manually

At this point all active positions have to be closed manually through MetaTrader 5 or installation should be cancelled by pressing _"Cancel"_. After all positions are closed, press _"Retry"_.

**2.2. Three Step Installation. Installation Diagram and Solution to Possible Problems**

If we simplify the installation process as much as possible, it can be narrowed down to three steps:

1. Before installing HedgeTerminal, close all currently active net positions in the MetaTrader 5 terminal;
2. Launch HedgeTerminal on the chart and press _"Yes"_ in the appeared window of the installation wizard to start installation. HedgeTerminal in this case will install all the files required for its operation;
3. In the next window if this appears, select the second option and press _"Yes"_. In this case active positions won't appear when HedgeTerminal is launched and the information about previously executed orders will be transferred to the _ExcludeOrders.xml_ file automatically as there are no active positions requiring closure.

_The simplest way to describe it is as follows: close all positions before launching Hedge Terminal and then press **"Yes"** two times in the HedgeTerminal installation wizard._

Complete pattern of the installation wizard is represented on the diagram below. It will help to answer the questions and perform installation correctly:

![](https://c.mql5.com/2/17/8e5sk_7m10z8d_yyetc8fql.png)

Fig. 12. Installation wizard

_**What is the course of action if installation was not performed correctly or HedgeTerminal is required to be deleted from the computer?**_ In the case installation was incorrect, simply delete all installed files. For that, go to the folder where programs for MetaTrader store shared information (as a rule this is located at: c:\\Users\\<your\_user\_name>\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\). If you want to delete information about the HedgeTerminal installation from all accounts, find the "HedgeTerminal" folder in this directory and delete it. If you want to delete information about the HedgeTerminal installation only for a certain account, go to the \\HedgeTerminal\\Brokers directory and select the folder, containing the name of your broker and the account number that looking like " _Broker's name - account number_". Delete this folder. Next time the installation wizard will launch again when HedgeTerminal is started on this account.

_**Installation on the terminal connected to the account already working with HedgeTerminal.**_ It may happen that that HedgeTerminal is required to be installed on the terminal connected to the account where HedgeTerminal is already working. As you already know, the installation process consists of creating and configuring system files. If all these files are already created on another computer and properly configured, then there is no need to install HedgeTerminal. As a rule, these files are stored at c:\\Users\\<your\_user\_name>\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\HedgeTerminal. Simply carry over this folder to the same place on your computer. After copying the directory, launch HedgeTerminal again. The installation wizard will not get called as all files are present and configured. In this case active positions will be similar to the display on other terminals and computers.

_**Installation of HedgeTerminalAPI and using the library in test mode.**_ HedgeTerminal is implemented both as a visual panel and a library of program functions - HedgeTerminalAPI. The library contains a similar installation wizard called at the first launch of HedgeTerminalAPI in real time. Installing files when using the library is similar. At the first call of any function, the Expert will bring up a relevant MessageBox suggesting to start installation. In that case the user has to do the same thing - close all positions before calling HedgeTerminalAPI and perform installation in three steps.

It is not required to install HedgeTerminal when launching the library in test mode. As in test mode the Expert starts every time working with a new virtual account, there is no need to hide previously executed order and install files. The settings stored in the file Settings.xml _,_ in the library can be defined grammatically through calling relevant functions.

**2.3. Getting Started with HedgeTerminal, First Launch**

After HedgeTerminal has installed all the files required for its work, it will launch on the chart and display its panel:

![Fig. 13. First launch of HedgeTerminal, appearance](https://c.mql5.com/2/12/HT_-_vta_mqqzu7.png)

Fig. 13. First launch of HedgeTerminal, appearance

As all existing orders are hidden, there are no displayed active positions. The upper panel contains a button of the HedgeTerminal menu on the left hand side and dedicated icons displaying the state of the panel. The buttons have the following functions:

- ![](https://c.mql5.com/2/12/Demo.png) - Indicates that the panel demo version has been launched. This does not support the trade on real accounts and displays positions only on the AUDCAD and VTBR\* symbols. In this mode, the history of positions is also limited by the last 10 closed positions.
- ![](https://c.mql5.com/2/12/b54uakolke47i8w_ngfp.png) \- Indicates that positions in HedgeTerminal are not equal to the total net position in MetaTrader 5. This is not a critical error but it is required to be eliminated. If the HedgeTerminal installation is correct, refer to the section of this article describing how to eliminate the mistake.
- ![](https://c.mql5.com/2/12/NotAlloved.png) \- The icon signifies that trade is impossible. Hover the mouse over this icon and find out from the pop-up tip what the matter is. Possible reasons: no connection with the server; the Expert Advisor is prohibited to trade; trading with Expert Advisors is not allowed in MetaTrader 5.
- ![](https://c.mql5.com/2/12/TradeAlovedTrue.png) \- An icon indicating that the trade is permitted; HedgeTerminal can perform any trade action.

Now, when HedgeTerminal is launched and ready for work, carry out several trades and see how they are displayed. If HedgeTerminal is launched on Forex, trading actions are to be executed on the AUDCAD symbol. There should not be any open net positions as installation of HedgeTerminal could be successful only in that case. If this is not the case for some reason, close all active positions.

If HedgeTerminal is launched on an account connected to the Moscow Exchange, trading should be executed on one of the futures of the VTBR\* group, for example VTBR-13.15 or VTBR-06.15. As an illustration, buy _0.4_ lot of AUDCAD by the current price through the standard window "NewOrder". In a few moments, the order and the deal that executed it will appear in the order history of the terminal:

![Fig. 14. Historical order and its deal in MetaTrader 5](https://c.mql5.com/2/12/kk79s_u_ao8_ph5p3z_s_ry5.png)

Fig. 14. Historical order and its deal in MetaTrader 5

The active positions tab will contain a correspondent long position of 0.4 lot:

![Fig. 15. Active net position in MetaTrader 5](https://c.mql5.com/2/12/dc5hs8vu_gbm6erh.png)

Fig. 15. Active net position in MetaTrader 5

At the same time, HedgeTerminal will display the historical order as an active position:

![Fig. 16. Active bi-directional position in HedgeTerminal and its deal](https://c.mql5.com/2/12/89m8tb9w_9eiwj96_HT.png)

Fig. 16. Active bi-directional position in HedgeTerminal and its deal

As we can see, the results are the same, as one position in HedgeTerminal is correspondent to 1 net position in MetaTrader 5.

Please pay attention that in addition to the order, the position in HedgeTerminal contains a deal that executed this position (to see it, maximize the position string by pressing ![](https://c.mql5.com/2/12/d5g912_rncodn8cqnx0m_lacgihg.png))

A part of an active position can be hidden. For that simply enter a new volume value in the field "Vol." (Volume). The new volume should be less than the current one. The difference between the current and the new volumes will be covered by the new order that forms a historical position and is displayed in the correspondent tab of historical positions. Enter the value of 0.2 in this field and then press Enter. In some time the volume of the active position is going to be 0.2 and the historical positions tab will feature the first historical transaction with the volume of 0.2 lot ( _0.4 - 0.2 = 0.2_):

![Fig. 17. Historical bi-directional position in HedgeTerminal](https://c.mql5.com/2/12/gzkqscz3yc0z_453bgcc_HT.png)

Fig. 17. Historical bi-directional position in HedgeTerminal

In other words, we closed half of our active position. Common financial result of the historical and active position will be identical to the result in the terminal.

Now we are going to close the rest of the active position. For that press the button of closing a position in ![](https://c.mql5.com/2/12/sp0ffz_f71iutrh_0l58f2i.png)HedgeTerminal:

![Fig. 18. Close button of a bi-directional position in HedgeTerminal](https://c.mql5.com/2/12/bs2a8a9v_ebz7rg0_e_HT.png)

Fig. 18. Close button of a bi-directional position in HedgeTerminal

After this button has been pressed, the remaining position must be closed by the opposite order and transferred to the historical position tab. This way active positions (both in MetaTrader 5 and in HedgeTerminal) will be closed.

**2.4. Hedging and Calculation of Financial Operations**

In this section we are going to describe methods of work with HedgeTerminal using several bi-directional positions as an example.

So currently we do not have any active positions. Open a new long position with the volume of 0.2 lot on AUDCAD. For that it is enough to open the "new order" dialog window and place a correspondent order. After the position has been opened, _lock_ it by the opposite position - sell 0.2 lot through the MetaTrader 5 terminal.

This time the trade result in HedgeTerminal and in the MetaTrader 5 terminal window are different. HedgeTerminal displays two positions:

![Fig. 19. Opposite bi-directional positions in HedgeTerminal (lock)](https://c.mql5.com/2/12/2_4__cmywusd0j4cez7qyc_zgvqmi7_6_HT_5l4d3.png)

Fig. 19. Opposite bi-directional positions in HedgeTerminal (lock)

In MetaTrader 5 they are not present at all:

![Fig. 20. Absence of an active net position in MetaTrader 5](https://c.mql5.com/2/12/rkfpc8kndgu_jkhll3f_n_j05.png)

Fig. 20. Absence of an active net position in MetaTrader 5

Let us assess the result. 4 deals have been executed. They are in the red frame on the screenshot below:

![Fig. 21. Result of executed deals in MetaTrader 5](https://c.mql5.com/2/12/pl9zri_r_MetaTrader5.png)

Fig. 21. Result of executed deals in MetaTrader 5

Put these deals in a table:

| Type | Direction | Price | Volume | Profit, pips |
| --- | --- | --- | --- | --- |
| buy | in | 0,98088 | 0,2 |  |
| sell | out | 0,98089 | 0,2 | 1 |
| buy | in | 0,98207 | 0,2 |  |
| sell | out | 0,98208 | 0,2 | 1 |

Table 1. Displaying deals in the MetaTrader 5 terminal

It is obvious that 2 points were earned on 4 deals. The value of every point is roughly 0,887 dollar when trading 1 lot. So, the financial result is 0.18 dollar per position (0,887\*0,2\*1) or 34 cents for both positions. The "Profit" column shows that this result is correct. The net profit is **_34 cents_.**

Now look at the result in HedgeTerminal:

![Fig. 22. Result of historical bi-directional positions in HedgeTerminal](https://c.mql5.com/2/12/4tkiev_g_HT.png)

Fig. 22. Result of historical bi-directional positions in HedgeTerminal

At first they seem to differ significantly. Let us calculate the result for our two bi-directional positions:

| Direction | Entry price | Exit price | Profit, pips | Profit, $ |
| --- | --- | --- | --- | --- |
| buy | 0,98088 | 0,98208 | 0.00120 | 21,28 |
| sell | 0,98089 | 0,98207 | -0.00118 | -20,92 |

Table 2. Displaying positions in HedgeTerminal

The difference between them is _**0.34$ (21,28$ - 20,92$)**_, which is exactly the same as the result obtained in net trading.

**2.5. One Click Trading**

HedgeTerminal employs a peculiar management system. This is based on the direct entry of the required values to the active position field. There is no alternative management.

For instance, to place StopLoss for an active position, simply enter the required value in the StopLoss field and then press Enter:

![Fig. 23. Entry of the StopLoss level directly to the table](https://c.mql5.com/2/12/SL.png)

Fig. 23. Entry of the StopLoss level directly to the table

Similarly, this way a trader can modify the volume, place TakeProfit or modify its level, change the _**initial comment**_.

In general, the position line in HedgeTerminal is similar to the position display in MetaTrader 4/5. The columns of the table with positions have identical names and values used in the MetaTrader terminals. There are differences too. So, a position in HedgeTerminal has two comments whereas in MetaTrader a position has only one comment. This means that a position in HedgeTerminal is formed by two orders and each of them has its own comment field. As HedgeTerminal does not use these comments for storing technical information, it is possible to create a position with an opening and closing comment.

HedgeTerminal features an opportunity to add columns that are not displayed by default. For example, a column reflecting the name of the Expert Advisor that opened the position. The section dedicated to configuring the Settings.xml file describes the way to do it.

**2.6. Placing StopLoss and TakeProfit, Trailing Stop**

HedgeTerminal allows to close positions by the StopLoss or TakeProfit levels.

To see how it happens, introduce the StopLoss and TakeProfit levels in the active position and then wait till one of those levels triggers. In our case TakeProfit triggered. It closed the position and moved it to the list of historical transactions:

![Fig. 24. Historical bi-directional position and its StopLoss and TakeProfit levels](https://c.mql5.com/2/12/TP.png)

Fig. 24. Historical bi-directional position and its StopLoss and TakeProfit levels

The green mark in the TakeProfit level indicates that TakeProfit triggered and the position was closed by that level. In addition to TakeProfit, a position contains information about the StopLoss level which was also used when the position was active. Similarly, if the StopLoss level triggered, then the StopLoss cell would be highlighted pink and TakeProfit would not be colored.

HedgeTerminal supports TrailingStop and future versions will allow to write a specific custom module containing the logic of carrying over the StopLoss. Currently the work with Trailing Stop in HedgeTerminal is different from the same Trailing Stop in the MetaTrader terminal. To enable it, it is required to enter the StopLoss level in the StopLoss cell of the correspondent position. When StopLoss is placed, following the price option can be flagged in the cell marked with a sign ![](https://c.mql5.com/2/12/wh1etp4d_wg1w_y1i7bd.png):

![Fig. 25. Placing Trailing Stop in HedgeTerminal](https://c.mql5.com/2/12/j7xkl17h_0nvs.png)

Fig. 25. Placing Trailing Stop in HedgeTerminal

After flagging, HedgeTerminal fixes the distance between the current price and the StopLoss level. If it increases, StopLoss will follow the price so the distance stays the same. Trailing Stop works only when HedgeTerminal is working and gets cleared upon the exit.

In HedgeTerminal, StopLoss is implemented with the BuyStop and SellStop orders. Every time a StopLoss is placed, a correspondent order with a magic number connected with the active position is placed too. If this pending order gets deleted in the MetaTrader terminal, then the StopLoss level in HedgeTerminal will disappear. Changing the price of the pending order will trigger the change of the StopLoss in HedgeTerminal.

TakeProfit works a little differently. This is _virtual_ and it hides its trigger level from the broker. It means that HedgeTerminal closes a position by TakeProfit autonomously. Therefore, if you want your TakeProfit to trigger, you need to keep your HedgeTerminal in a working order.

**2.7. Report Generation**

HedgeTerminal allows to save the information about its bi-directional positions to designated files, which can be loaded to third party statistical programs of analysis, for instance to Microsoft Excel.

Since HedgeTerminal currently does not have a system of analysis and statistics gathering, this feature is especially important. At the moment, only one format of report saving is supported - [CSV](https://en.wikipedia.org/wiki/Comma-separated_values "https://en.wikipedia.org/wiki/Comma-separated_values") (Comma-Separated Values). Let us see how this works. To save the bi-directional positions in the CSV file, select the _"Save CSV Report"_ option in the menu after launching HedgeTerminal:

![](https://c.mql5.com/2/12/SaveCSV.png)

Fig. 26. Saving a report through the "Save CSV Report" menu

After selecting those points in the menu, HedgeTerminal will generate two files in the CSV format. One of them will comprise the information about active positions and another one about historical, completed ones. The reason why two different files are generated is because each of them has a different set of columns. Besides, the number of active positions can constantly vary, therefore it is more convenient to analyze them separately. The generated files are available by the relative path _._\ _HedgeTerminal_\ _Brokers_\ _<Broker's name - account number>_\ _._ There must be two files at the end: History.csv and Active.csv. The first one contains information about historical positions and the second one about active positions.

The files can be loaded into the statistical programs for analysis. Let us look at this procedure using Microsoft Excel as an example. Launch this program and select _"Data" --> "From the text"_. In the appeared data export wizard select the mode with separators. In the following window, select semicolon as the separator. Press _"Next"_. In the following window change the general format of floating point number representation. To do that press _"Details"_. In the appeared window select the period character as a separator of the integer and fractional parts:

![Fig. 27. Export the CSV report to Microsoft Excel](https://c.mql5.com/2/12/ExcelExport.png)

Fig. 27. Export the CSV report to Microsoft Excel

Press "ОК", and then "Finish". The appeared table will contain information about active positions:

![Fig. 28. Information about active positions exported to Excel](https://c.mql5.com/2/12/Excel.png)

Fig. 28. Information about active positions exported to Excel

Data on historical positions can be loaded in the next sheet the same way. The number of columns will be correspondent to the factual number of columns in HedgeTerminal. If one of the columns gets deleted from the HedgeTerminal panel, it will not be included into the report. This is a convenient way to form a report based only on the required data.

Later versions of the program will allow to save a report in the XML and HTML formats. Added to that, a report saved in HTML will be similar to a standard HTML report in MetaTrader 5, though it will consist of completed bi-directional positions.

**2.8. Currency Swap Presentation**

[Currency swap](https://en.wikipedia.org/wiki/Currency_swap "https://en.wikipedia.org/wiki/Currency_swap") is a combination of two bi-directional conversion deals for the same sum with different valuation dates.

In simple words, _swap is a derivative operation on accruing the difference of the interest rates paid for holding two currencies forming a net position_.

There is no history of net positions in MetaTrader 5. There are historical orders and deals that formed and closed historical net positions instead. As there are no historical positions, the swap is assigned to the order that closes a historical position:

![Fig. 29. Currency swap as an independent transaction](https://c.mql5.com/2/12/bhn0aytkl7_xc0wmd_4Swapy.png)

Fig. 29. Currency swap as an independent transaction

It would be more accurate to present the swap as an independent derivative operation appearing as a result of a net position prolongation.

The current version of HedgeTerminal does not support swap representation though the later versions will feature such an option. There a swap will be shown as a separate historical position in the historical position tab. The swap identifier will be correspondent to the identifier of the net position that it was accrued on. The swap will include the name of the currency pair and its financial result.

**2.9. Bottom Row**

Similar to MetaTrader, HedgeTerminal displays the final row with common characteristics for the whole account in the table of deals. For the table of active and historical positions these characteristics and their values are listed below:

**_Balance_** – Total balance. Equal to the similar value " _Profit_" in the window of active positions in MetaTrader. This does not take into account the floating profit or loss from open positions.

_**Floating P/L**_ – Floating profit/loss. This is equal to the sum of the profit of all currently active positions.

_**Margin**_ – Contains a share of pledge funds from the account balance in percent. Can vary from 0% to 100%. Indicates a degree of the load on deposit.

_**Total P/L**_– Contains a sum of all profits and losses for closed positions.

_**Pos.:#**_\- Displays the number of historical positions.

For correct display of the bottom row, it is required to add the _**"Arial Rounded MT Bold"**_ font to the system.

**2.10. Changing the Appearance of HedgeTerminal Tables**

As mentioned before, HedgeTerminal has only one explicit parameter in the window of the Expert launch. This is the name if the settings file. This is so because the HedgeTerminal settings cannot be placed in the window of the Expert settings. There are many of them and they are too specific for such a window. That is why all settings are placed to the designated configuration file Settings.xml. For every launched version of HedgeTerminal, there can be its own settings file. That gives an opportunity to configure every launched version of HedgeTerminal individually.

HedgeTerminal currently does not allow editing this file through the visual windows of settings. Therefore, the only way to change the behavior of HedgeTerminal is to edit the file manually. The settings file is configured in the optimal way by default and therefore most of the time there is no need to edit the content. There may be situations when such editing may be required. Such editing is necessary for fine tuning and customizing the panel appearance. For instance, when a file has been edited, unnecessary columns can be deleted or, on the contrary, the columns not displayed by default can be added.

Let us see the content of this file and see how to customize it. To do that, find this file in the directory where HedgeTerminal has installed its files. This directory, as was mentioned before, can be located at the address c:\\Users\\<your user name>\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\HedgeTerminal. The Settings.xml file should be located inside. Open it with the help of any text editor:

```
<!--This settings valid for HedgeTerminal only. For settings HedgeTerminalAPI using API function and special parameters.-->
<Hedge-Terminal-Settings>
        <!--This section defines what columns will be showed on the panel.-->
        <Show-Columns>
                <!--You can change the order of columns or comment not using columns.-->
                <!--You can change the value of 'Width' for a better scaling of visual table.-->
                <!--You can change the value of 'Name' to install your column name.-->

                <!--This columns for tab 'Active'-->
                <Active-Position>
                        <Column ID="CollapsePosition" Name="CollapsePos." Width="20"/>
                        <!-- Unset this comment if you want auto trading and want see name of your expert:
                        <Column ID="Magic" Name="Magic" Width="100"/>-->
                        <Column ID="Symbol" Name="Symbol" Width="70"/>
                        <Column ID="EntryID" Name="Entry ID" Width="80"/>
                        <Column ID="EntryDate" Name="Entry Date" Width="110"/>
                        <Column ID="Type" Name="Type" Width="80"/>
                        <Column ID="Volume" Name="Vol." Width="30"/>
                        <Column ID="EntryPrice" Name="EntryPrice" Width="50"/>
                        <Column ID="TralStopLoss" Name="TralSL" Width="20"/>
                        <Column ID="StopLoss" Name="S/L" Width="50"/>
                        <Column ID="TakeProfit" Name="T/P" Width="50"/>
                        <Column ID="CurrentPrice" Name="Price" Width="50"/>
                        <Column ID="Commission" Name="Comm." Width="40"/>
                        <Column ID="Profit" Name="Profit" Width="60"/>
                        <Column ID="EntryComment" Name="Entry Comment" Width="100"/>
                        <Column ID="ExitComment" Name="Exit Comment" Width="100"/>
                </Active-Position>
                <!--This columns for tab 'History'-->
                <History-Position>
                        <Column ID="CollapsePosition" Name="CollapsePos." Width="20"/>
                        <!-- Unset this comment if you want auto trading and want see name of your expert:
                        <Column ID="Magic" Name="Magic" Width="100"/>-->
                        <Column ID="Symbol" Name="Symbol" Width="70"/>
                        <Column ID="EntryID" Name="Entry ID" Width="80"/>
                        <Column ID="EntryDate" Name="Entry Date" Width="110"/>
                        <Column ID="Type" Name="Type" Width="40"/>
                        <Column ID="EntryPrice" Name="Entry Price" Width="50"/>
                        <Column ID="Volume" Name="Vol." Width="30"/>
                        <Column ID="ExitPrice" Name="Exit Price" Width="50"/>
                        <Column ID="ExitDate" Name="Exit Date" Width="110"/>
                        <Column ID="ExitID" Name="Exit ID" Width="80"/>
                        <Column ID="StopLoss" Name="S/L" Width="50"/>
                        <Column ID="TakeProfit" Name="T/P" Width="50"/>
                        <Column ID="Commission" Name="Comm." Width="40"/>
                        <Column ID="Profit" Name="Profit" Width="50"/>
                        <Column ID="EntryComment" Name="Entry Comment" Width="90"/>
                        <Column ID="ExitComment" Name="Exit Comment" Width="90"/>
                </History-Position>
        </Show-Columns>
        <Other-Settings>
                <Deviation Value="30"/>
                <Timeout Seconds="180"/>
                <!-- If your computer is not fast enough - set a value 'Milliseconds' greater than 200, such as 500 or 1000. -->
                <RefreshRates Milliseconds="200"/>
        </Other-Settings>
</Hedge-Terminal-Settings>
```

This file contains a special document written in [the xml markup language](https://en.wikipedia.org/wiki/XML "https://en.wikipedia.org/wiki/XML"), describing the behavior and the appearance of HedgeTerminal.

The main thing to do before the start of editing is to create a backup copy in case the file gets accidentally corrupted. The file has a hierarchical structure and consists of several sections enclosed into each other. The main section is called <Hedge-Terminal-Settings>and it contains two main subsections: <Show-Columns> and <Other-Settings>. As follows from their names, the first section adjusts the displayed columns and their conditional width and the second section has the settings for the panel itself and the HedgeTerminalAPI library.

_**Changing the size and the name of columns**_. Let us refer to the <Show-Columns> section and consider its structure. Essentially, this section contains two sets of columns: one for the table of active positions in the "Active" tab and the other one for the table of historical positions in the "History" tab. Every column in one of the sets looks like:

```
<Column ID="Symbol" Name="Symbol" Width="70"/>
```

The column must contain an identifier ( _ID="Symbol"_), name of the column ( _Name="Symbol"_) and conditional width ( _Width="70"_).

The identifier contains a unique internal name of the column. It prompts HedgeTerminal how to display this column and what value to place in it. The identifier cannot be changed and it should be supported by HedgeTerminal.

The name of the column defines what name the column will be displayed with in the table. If the name is changed for another one, like: _Name="_ _Currency Pair_ _"_, then at the next launch of HedgeTerminal, the name of the column displaying the symbol will change:

![](https://c.mql5.com/2/17/Currency_Pair.png)

Fig. 30. Change of the column names in the HedgeTerminal panel

This very property allows creating a localized version of the panel where the name of every column will be displayed in preferred language, for example in Russian:

![Fig. 31. The HedgeTerminal panel localization](https://c.mql5.com/2/17/5ardwg5reob.png)

Fig. 31. The HedgeTerminal panel localization

The next tag contains the conditional width of the column. The thing is that a graphics engine automatically aligns the table size with the width of the window where the panel is launched. The wider the window is the wider each column. It is impossible to set up the width of the column precisely but the proportions can be specified by setting up the basic width of the column. The basic width is the width of the column in pixels when the width of the window is 1280 pixels.

If your display is wider, then the factual size of the column is greater. To see how uniform scaling works, set up the width of the Symbol column as 240 conditional pixels: Width="240". Then save the file and reset the panel:

![Fig. 32. The width of the Symbol column is set up for 240 pixels.](https://c.mql5.com/2/17/z7cz2rp_s3zd4n_Symbol.png)

Fig. 32. The width of the Symbol column is set up for 240 pixels.

The width of the column with symbols has increased significantly. However, it should be noted that its width was gained at the expense of the width of other columns (" _Entry ID"_ and _"Entry Date"_).

Other columns are now looking worse, last values do not fit in the allowed width. The size of the columns should be changed carefully, finding the best size for each of them. The standard settings have precise values for majority of displays and changing their size is normally not required.

_**Deleting columns.**_ In addition to changing column sizes and their headers, the columns can be added or deleted too. For example, let us try and delete the _"Symbol"_ column from the list. To do that, simply comment out the tag of the column in the Settings.xml file (highlighted in yellow):

```
<Column ID="CollapsePosition" Name="CollapsePos." Width="20"/>
<!-- Unset this comment if you want auto trading and want see name of your expert:
<Column ID="Magic" Name="Magic" Width="100"/>-->
<!--<Column ID="Symbol" Name="Symbol" Width="70"/>-->
<Column ID="EntryID" Name="Entry ID" Width="80"/>
```

In xml a specified tag **<!--** _content of the comment_ **-->** has a role of the comment. The commented out column must be located in the tag. After changes in the file have been saved and HedgeTerminal was relaunched, its set of columns for historical positions will change. It won't contain a column displaying the symbol:

![Fig. 33. Deleting the Symbol column from the table of active positions](https://c.mql5.com/2/17/zt6wn7nz_ot99xw9_dh_0po6we.png)

Fig. 33. Deleting the Symbol column from the table of active positions

Several columns can be deleted at once.

Let us comment out all columns except the symbol name and the profit size and then relaunch HedgeTerminal on the chart:

![Fig. 34. Deleting main columns from the HedgeTerminal panel](https://c.mql5.com/2/12/c8pap4pe_lpnn_8ydlnwh.png)

Fig. 34. Deleting main columns from the HedgeTerminal panel

As we can see, the HedgeTerminal table has significantly reduced. One should be careful with deleting columns as in some cases they contain elements of position management. In our case, the column that allows reversing positions and view their deals got deleted. Even the "Profit" column from the table of active positions can be deleted. This column, however, contains a button of closing position. In this case a position won't get closed as the close button is missing.

**_Adding columns and displaying the aliases of Experts._** Columns can be deleted by editing and added, if HedgeTerminal was programmed to work with them. In the future, a user can create their own columns and calculate their own parameters. Such columns can be connected via the interface of the external indicator iCustom and the manifest of interaction with extension modules, written in xml.

Currently there is no such a functionality and only the columns supported by HedgeTerminal are available. The only column supported by the panel but not displayed in the settings by default is the column that contains the Expert's magic number.

Position can be open not only manually but also using a robot. In this case, the identifier of the robot, which opened a position can be displayed for every position. The commented out column tag including the Expert Advisor's number is already present in the settings file. Let us uncomment it:

```
<Column ID="Magic" Name="Magic" Width="100"/>
```

Save the changes and relaunch the panel:

![Fig. 35. Adding the Magic column to the HedgeTerminal table](https://c.mql5.com/2/17/Magic.png)

Fig. 35. Adding the Magic column to the HedgeTerminal table

As we can see on the picture above, a column called _"Magic"_ appeared. This is the identifier of the Expert Advisor that opened a position. The position was opened manually and therefore the identifier is equal to zero. A position opened by a robot as a rule contains an identifier different from zero. In that case a correspondent magic number of the Expert will be displayed in the column.

It is more convenient though to see the names of the Expert Advisors instead of their numbers. HedgeTerminal has this option. It takes the aliases (names) of the EAs taken from the ExpertAliases.xml file and displays them instead of the correspondent magic numbers.

For instance, if you have an Expert called _"ExPro 1.1"_ and trading under the magic number _123847_, it is enough to place the following tag in the ExpertsAliases.xml file in the <Expert-Aliases> section:

```
<Expert Magic="123847" Name="ExPro 1.1"></Expert>
```

From now on, when your robot opens a new position, it will have its name:

![Fig. 36. Displaying the name of the Expert in the column Magic](https://c.mql5.com/2/12/phkm6ka9ls.png)

Fig. 36. Displaying the name of the Expert in the column Magic

The number of robots and their names is not limited. The only condition here is for their numbers to differ.

_**Other settings of HedgeTerminal**_. We have described all the settings of the visual panel the currently available. The next section <Other-Settings> contains the settings defining internal work of the terminal. Please note, that the program call library HedgeTerminalAPI uses these settings as default values. The Expert Advisor, however, can change the current values of these settings using the special _Set..._ functions. To get the values of those settings, the Expert only needs to call special _Get..._ functions. We are going to describe tags with settings in more detail.

```
<Deviation Value="30"/>
```

Contains the value of extreme deviation from the required price in units of minimal price change. So, for the pair EURUSD quoted to the fifth decimal place, it will be 0,0003 point. For the RTS index future this value will be 300 points as the minimum step of the price changing is 10 points. If the price deviation is greater, then the order to close or change the price will not be executed. This is in place for protection from unfavorable slippage.

```
<Timeout Seconds="180"/>
```

This tag contains the value of the maximum permissible response from the server. HedgeTerminal works in the asynchronous mode. That means that HedgeTerminal sends a trading signal to the server without waiting to receive a response from the latter. Instead, HedgeTerminal controls the sequence of events pointing at the fact that the order was placed successfully. The event may not take place at all. This is the reason why it is so important to set the wait timeout period, during which HedgeTerminal will be waiting for the server response. If there is no response during that time, the task will be cancelled, HedgeTerminal will unblock the position and carry on its work.

The wait timeout period can be shortened or extended. By default this is 180 seconds.

It is important to understand that in real trading, this limitation is never reached. Responses to trading signals come back quickly, the majority of them are executed within 150 – 200 milliseconds.

More details about the work of HedgeTerminal and about the peculiarities of work in the asynchronous mode can be found in the third section of this article: ["Under the Bonnet of HedgeTerminal"](https://www.mql5.com/en/articles/1297#chapter3).

```
<RefreshRates Milliseconds="200"/>
```

This tag sets up the frequency of the panel updates. It contains the time in milliseconds between two consequent panel updates. The less this value is, the higher the panel update frequency is and the more CP resources are used. If your processor is not very powerful and it cannot cope with the update frequency of 200 milliseconds (default value), you can increase it up to 500 or even to 1000 milliseconds by editing the tag.

In this case the CP load will drop significantly. Values less than 100 milliseconds are not recommended. Increasing the update frequency, the CP load will increase nonlinearly. It is important to understand that the update frequency defines discreteness of the terminal. Some of its actions are defined by this timer and happen with a certain speed.

**2.11. Planned Yet Unimplemented Features**

HedgeTerminal has a flexible and nontrivial architecture due to which new nontrivial capabilities of this program become feasible. Today these capabilities have not been implemented, though they may appear in the future if there is demand. Below are the main ones:

_**Using color schemes and skins.**_ HedgeTerminal uses its own graphics engine. This is based on the graphic primitives like a rectangular label or a usual text. Graphics based on pictures is not used at all. This gives an opportunity to change the color, size and font of all the elements displayed in HedgeTerminal. This way it is easy to create a description of fonts and the color scheme in the form of skin and load it at the launch of HedgeTerminal, changing its appearance.

_**Connecting custom Trailing Stop modules.**_ Every Expert, and HedgeTerminal is essentially an Expert, can initiate a calculation of an arbitrary indicator through a specific program interface (function [iCustom()](https://www.mql5.com/en/docs/indicators/icustom)). It allows to call the indicator calculation, which depends on the set of arbitrary parameters. Trailing Stop is an algorithm that places a new or keeps the old price level depending on the current price. This algorithm and its price levels can be implemented as an indicator. If the passed parameters are agreed on, HedgeTerminal can call such an indicator and calculate the required price level. HedgeTerminal can take care of the mechanics of transferring the Trailing Stop. This way, any HedgeTerminal user can write their own (even the most unusual one) module of managing Trailing Stop.

_**Adding new columns for the table of positions. That includes the custom columns.**_ The HedgeTerminal tables are designed the way that they allow adding new columns. Support of new columns can be programmed inside HedgeTerminal and implemented in a familiar way through the [iCustom()](https://www.mql5.com/en/docs/indicators/icustom) interface. Every cell in the line of a position represents a parameter (for example, opening price or Take Profit level). This parameter can be calculated by an indicator and that means that an unlimited number of indicators can be written so each of them calculates some parameter for a position. If passing parameters is coordinated for such indicators, it will become possible to add an unlimited number of custom columns to a table in HedgeTerminal.

_**Connecting extension modules.**_ Other calculation algorithms can be calculated through the mechanism of the custom indicator call. For example, the report system includes a lot of calculation parameters such as expected payoff and Sharpe ratio. Many of those parameters can be received by moving their calculation block to the custom indicator.

_**Copying deals, receiving and transmission of deals from other accounts.**_ HedgeTerminal is essentially a position manager. This can easily be a base for a deal copier as the main functionality have already been implemented. Such a copier will have nontrivial capabilities to copy bi-directional MetaTrader 4 positions to the MetaTrader 5 terminal. The copier will display those positions as bi-directional like in MetaTrader 4 with a possibility to manage every position individually.

_**Reports and statistics. The Equity chart and the “Summary" tab.**_ The counting of bi-directional positions allows analyzing the contribution to the result of every strategy or trader. Alongside with statistics, a portfolio analysis can be carried out. This report can notably differ from the report in MetaTrader 5 and add to it. In the Summary tab report, in addition to the generally accepted parameters like expected payoff, maximum drawdown, profit factor, etc, other parameters will be included too. The names and the values of the latter can be obtained from the custom extension modules. Instead of the familiar balance chart on a picture, the Equity chart in the form of a custom candlestick exchange chart can be used.

_**Sorting columns.**_ Usually, when pressing on the table header, the rows get sorted in the ascending or descending order (second pressing). Current version of HedgeTerminal does not support this option as sorting is connected with the deal filter and the custom set of columns, which are not available at the moment. Later versions will feature this option.

_**Sending a trade report via email, ftp. Push notifications.**_ HedgeTerminal can use the system functions to send emails, ftp files and even push notifications. For instance it can form an html report once a day and send it to the list of users. Since HedgeTerminal is a manager of Expert Advisors and it knows all trading actions of other EAs, it can notify users about trading actions of other Expert Advisors. For instance, if one of the EAs opens a new position, HedgeTerminal can send a push notification informing users that a certain Expert entered a new position. HedgeTerminal will also specify the direction of the entry, its date, time and volume. The EA itself won't need to be configured, it will be enough to add its name into the alias file.

_**Filtering positions with the regular expressions console.**_ This is the most powerful of the planned developments. It will bring up the work of HedgeTerminal to a completely new level. Regular expressions can be used for filtering historical and active positions so only those meeting the requirements of this filter are displayed. Regular expressions can be combined and entered in a dedicated console above the table of active and historical positions. This is how this console may look like in the future versions of HedgeTerminal:

![Fig. 37. Request console in the future version of HedgeTerminal](https://c.mql5.com/2/17/Hedgeterminal_future.png)

Fig. 37. Request console in the future version of HedgeTerminal

Regular expressions can form very flexible conditions for filtering positions followed by statistic calculation on them. For example, to display historical positions only by the AUDCAD symbol, it is enough to enter the "AUDCAD" symbol in the cell "Symbol". Conditions can be combined. For example, positions by AUDCAD executed by a particular robot can be displayed for the period from 01.09.2014 till 01.10.2014. All you need to do for that is to enter conditions to the correspondent cells. After the filter displays the results, the report from the "Summary" tab will change in accordance with the new filter conditions.

Regular expressions will consist of a small number of simple operators. However, used together, they will allow to create very flexible filters.

The presence of the below operators is necessary and sufficient in console:

_**Operator =**_ \- Strict equality. If the word AUDСAD gets entered in the "Symbol" field, then all the symbols containing this substring, for example AUDCAD\_m1 or AUDCAD\_1, will be found. That means that an implicit insertion operator will be used. Strict equality "=" requires a complete match of the expression and therefore all symbols except AUDCAD will be excluded. AUDCAD\_m1 or EURUSD will be excluded.

_**Operator >**_ \- Displays only the values greater than the specified one.

_**Operator <**_ \- Displays only the values less than the specified one.

_**Operator !**_ – Logical negation. Reflects only the values that are not equal to the specified one.

**_Operator_ \|** \- Logical _"OR"_. Allows specifying two or more conditions in one row at the same time. At the same time, to meet the criterion, it is enough to fulfill at least one stipulation. For instance, the expression _"\> 10106825_ **\|** _=10106833"_ entered in the cell _"Entry Order"_ of the expression console will show all positions with the incoming order identifier greater than 10106825 or equal to 10106833.

_**Operator &**_ \- Logical _"AND"._ Allows to specify two or more conditions in one row at the same time and each of them has to be fulfilled. The expression " _>10106825 **&** <10105939_" entered in the cell _"Entry Order"_ shows all positions with the incoming identifier greater than 10106825 or less than 10105939. Positions with identifiers between these two numbers, for example 10106320, will be filtered.

_**Correcting and managing positions using special commands.**_ Additional symbols can be entered into the cells reflecting the volume or StopLoss and TakeProfit levels. This makes a more complex position management possible. For instance, to close a half of the current position volume, enter the value "50%" correspondent to the active position in the _"Volume"_ field.

Instead of placing StopLoss and TakeProfit levels, a value, for example "1%", can be entered into those cells. HedgeTerminal will automatically calculate the StopLoss and TakeProfit level so they will be placed at a distance of 1% from the entry price. Numbers with the postfix "p" can be entered in these cells. For instance, "200p" will mean the order: " _place the StopLoss and TakeProfit levels 200 points away from the position entry price_". In the future, if a minus is put before the volume in the column _"Volume",_ the volume will close by the value specified after that sign. For instance, if you have a position with the volume 1.0 and we want to close a part of the volume (for example 0.3), then it is enough to enter "-0.3" in the volume cell.

### Chapter 3. Under the Bonnet of HedgeTerminal. Specification and Principle of Operations

**3.1. Global and Local Contours. Context, Transfer and Storage of Information**

The appearance of HedgeTerminal and trading process resemble the familiar MetaTrader 4. This is possible due to the _virtualization_ and the transformation of data display, when trading information available through MetaTrader 5 gets displayed by the panel in a more convenient way. This chapter describes the mechanisms that allow to create such virtualization and the mechanisms of group data processing.

As you know, HedgeTerminal is represented in several different products. The main of them are the visual panel and the library with program interface. The latter allows implementing management of bi-directional positions in any external Expert Advisor and it also integrates it into the HedgeTerminal visual panel. For example, an active position of an EA can be closed straight from the panel. The Expert Advisor will get the information and process it accordingly.

Apparently, such a structure requires a group interaction between the Expert Advisors. The panel, which is essentially an Expert Advisor, must know about all trading actions that take place. Conversely, every Expert Advisor using the HedgeTerminal library must know about trading actions executed manually (by third party programs or using the HedgeTerminal panel).

In general, the information about trading actions can be received from the trading environment. For instance, when the user opens a new position, the number of orders changes. The last order can tell what symbol the position was opened for and what volume it had. The information about orders and deals is stored on the server. That is why it is available for any terminal connected to the trading account. This information can be called _**global**_ as this is available for everyone and is distributed through the global channel of communication. Communication with the trading server has the "Request - response" format.

Therefore such communication can be presented as a _global contour_. **_"Contour"_** is a concept from the graph theory. In simple words, a contour is a closed line with a few nodes interacting with each other. This definition may not be sufficient but we will leave the precision for mathematicians. The important thing for us is to present the trading process as some closed sequence of actions.

Not all the required information can be passed through the global contour. A part of information cannot be passed as MetaTrader 5 does not support such passing besides this information does not exist explicitly. The sequence of trading actions is:

1. A trader places an order to buy.
2. The order gets executed in some time.
3. Trading environment changes. The executed order gets into the list of historical orders. The aggregate position of the symbol changes.
4. The trader or an Expert Advisor detects the change in the trading environment and makes next decision.

Some time passes between the first and the fourth actions. It can be significant. If only one trader trades on the account, they know what actions are taken and wait for the appropriate response from the server. If there are a few traders or Expert Advisors trading on the account at the same time, there can be management errors.

For example, the second trader can place an order to close an existing position in the time between the first and the fourth step. Can you see a potential problem? The second order is placed when the first one is being executed. In other words, an order to close a position will be sent twice.

This example seems to be far fetched and unlikely if the trader is trading on the account alone and manually and uses synchronous methods of placing orders. When there are several robots trading on the account and executing independent trading operations, there is a high probability of such errors.

As HedgeTerminal is a position manager, working in the asynchronous mode and ensuring simultaneous parallel work of a number of Experts, it can be very close to those mistakes. To avoid this, HedgeTerminal synchronizes actions between all its launched copies (no matter if this is the HedgeTerminalAPI library or the visual panel) through the mechanism of the _**local contour**_ implemented as a multi-threaded reading and changing of the _**ActivePositions.xml**_ file. Interaction with the ActivePositions.xml file is the core of the local contour, the most important part of HedgeTerminal. This mechanism is described below.

In a simplistic way, the work of HedgeTerminal goes down to the cooperation with a local and global contour like on the figure below:

![Fig. 38. A simplified scheme of information exchange between the global and local contour](https://c.mql5.com/2/12/lmnoygy5oh_f6v8w_6imasteavxz_a_ftb9gokfut_fas8wuo.png)

Fig. 38. A simplified scheme of information exchange between the global and local contour

Any trading action (the label _**start**_ on the diagram) in HedgeTerminal starts with writing of a special tag in ActivePositions.xml, which blocks further changes of the position being modified.

After the position block has been set and the local contour has been successfully passed, HedgeTerminal sends a trading order to the server. For instance, a counter order to close the position. In some time, the order gets executed and the trading environment gets changed. HedgeTerminal processes this change and detects that the global contour is passed successfully and the order is executed. It unblocks the position and returns to the initial state (label _**finish**_ on the diagram).

There can be a situation when an order cannot be executed. In this case HedgeTerminal also unblocks the position and makes a record about the reason of the failure in the log of the MetaTrader 5 terminal.

In reality the pattern of the communication of information is more complex. As we already mentioned, in one MetaTrader 5 terminal several HedgeTerminal copies can be running. They can be the library or the visual panel. Every copy can act as a _**listener**_ and a _**writer**_. When HedgeTerminal performs a trading action, this is a writer because it blocks the position for changes using the records of the designated xml tag. All other copies of HedgeTerminal are listeners as they are reading the file ActivePositions.xml with a certain periodicity and block a position for changes having come across the blocking tag.

This mechanism ensures spreading information between independent threads. It facilitates parallel work between several panels and Experts using the HedgeTerminalAPI library.

A realistic diagram, showing the work of HedgeTerminal in the conditions of multi-threaded cooperation:

![Fig. 39. Near-natural pattern of the information exchange between copies of HedgeTerminal](https://c.mql5.com/2/12/0cwq4_hzoc0qlrprk_e_3kxg0sfz5o_z64gkpf.png)

Fig. 39. Near-natural pattern of the information exchange between copies of HedgeTerminal

The efficiency of such data organization is very high. Operations on the reading and writing the ActivePositions.xml file normally take less than 1 millisecond, whereas passing the global contour with sending and executing the order can take up to 150-200 milliseconds. To see the difference between the scale of these values, take a look at the diagram.

The width of the green rectangle shows the time of passing the local contour and the width of the blue one is the time of the trading order execution:

![Fig. 40. Time scale of making a record to the file and the time required for the order execution](https://c.mql5.com/2/12/7ffz607y_ye60qgin7m.png)

Fig. 40. Time scale of making a record to the file and the time required for the order execution

As you can see, the green rectangle looks more like a vertical line. In reality the difference between scales is even greater.

Quick operations of reading and recording can facilitate a complex data exchange between Experts and creating of distributed high frequency trading systems.

**3.2. Storing Global and Local Information**

Conventionally, all trading information used by HedgeTerminal can be divided into two parts:

- _**Local information.**_ This is stored in the computer files and is passed exclusively through the local contour;
- _**Global information.**_ This information is stored on the trading server. It is passed through the global contour and is available from any terminal connected to the account.

Orders, deals and the information about the account belong to global information. This information is available through specific functions of MetaTrader 5, like [HistoryOrderGetInteger()](https://www.mql5.com/en/docs/trading/historyordergetinteger) or [AccountInfoInteger()](https://www.mql5.com/en/docs/account/accountinfointeger). HedgeTerminal mainly uses this information. Due to that, HedgeTerminal displays the following data:

- Active and historical positions of HedgeTerminal;
- StopLoss of active and historical orders;
- Triggered TakeProfit levels of historical positions;
- Incoming comment of an active position, incoming and outgoing comments of historical positions;
- All other properties of active and historical positions not included into the list of local information.

As all these properties are global, their display is unique for all copies of HedgeTerminal, running on different computers. For example, if one bi-directional position gets closed in one of the terminals, this position will get closed in the other HedgeTerminal even if this is launched on another computer.

In addition to the global data, HedgeTerminal uses local information in its work. Local information is available only within one computer. This information is the basis of the following properties:

- TakeProfit levels of active positions;
- TakeProfit levels of historical positions that did not trigger;
- Outgoing comment of an active position;
- Service flag, blocking a change of a bi-directional position.

Take profit levels that did not trigger are stored in the HistoryPositions.xml file. The rest of the local information is stored in the ActivePositions.xml file.

Local data storage means that if you place a TakeProfit, it will be visible only in the HedgeTerminal copies running on your computer. No one except you will know this level.

**3.3. Stop Loss and Take Profit Levels. Order System Issues and OCO Orders**

In MetaTrader 5 as well as in MetaTrader 4, there is a concept of StopLoss and TakeProfit levels. These are protective stops. Similar to MetaTrader 4, they close a position in the case when it reaches the certain level of loss (StopLoss) or profit (TakeProfit). In MetaTrader 4, though, such stops are active for each open order individually. In MetaTrader 5 these stops work for the whole aggregate net position.

The issue here is that a net position is not connected with bi-directional positions of Experts and particularly HedgeTerminal. That means that regular StopLoss and TakeProfit levels cannot be used for bi-directional positions, however, these levels can be presented as separate pending orders. If you have a long position open, then two pending orders will emulate the work of TakeProfit and StopLoss respectively. One of them is SellLimit placed above the opening price of this position and another one is SellStop placed below this price.

In fact, if the price after opening reaches SellLimit of the order, this order will close the position with profit and if the price reaches the SellStop order, it will close the position with a loss. The only drawback is that after one order triggers, the second order will not stop existing and if the price then changes direction, the second order can trigger after the first one.

As there is no position at that point, triggering of the second order will open a new net position instead of closing the previous one.

The figure below illustrates this problem on the example of using protective stops in a long position:

![Fig. 41.  Emulating StopLoss and TakeProfit using the orders SellStop and SellLimit](https://c.mql5.com/2/12/Classic_pending_orders.png)

Fig. 41. Emulating StopLoss and TakeProfit using the orders SellStop and SellLimit

To avoid such inconsistency, in exchange trading the _OCO orders_(" _One Cancels the Other_") are used.

These are two pending orders connected with each other so triggering of one order cancels the other. In our example, after one order triggered, the second order is cancelled by the trading server and therefore new positions won't be opened and that is exactly what we are looking for. The work of this type of orders is presented on the diagram below:

![Fig. 42. OCO orders as StopLoss and TakeProfit](https://c.mql5.com/2/12/OCO-orders_as_StopLoss_and_TakeProfit_levels.png)

Fig. 42. OCO orders as StopLoss and TakeProfit

MetaTrader 5 does not support OCO orders. As a bundle of three orders cannot be used, orders simultaneously acting as StopLoss and TakeProfit are not suitable. _A pair of two orders can be used!_ So, it can be either StopLoss or TakeProfit.

In fact, if a pending order connected with the executed order initializing a new bi-directional position was placed (for example StopLoss), then such a construction is safe. There won't be a second pending order and it won't be able to open a new bi-directional position. Factually, there will be only three scenarios:

- A pending order will be cancelled for some reason by a broker or a user;
- A pending order will trigger;
- A pending order won't trigger.

There are no other scenarios. If a pending order is cancelled, it will be equal to canceling StopLoss or TakeProfit. If a pending order triggers, it will close the position. If the order does not trigger, then a bi-directional position will stay active with a placed StopLoss.

Even if HedgeTerminal is disabled when the pending order triggers, then later, when it launches, it will be able to process its triggering and understand that the position was closed by this order. It is possible to work out if the position was closed by StopLoss or TakeProfit if the field with the magic number contains special service information indicating whether the closing order is a regular order, TakeProfit or StopLoss. The way the link and the service information are stored in the field with the magic number is explained in detail in the next section.

Since two real protective stops cannot be used at the same time, a compromise was made when HedgeTerminal was designed:

_Bi-directional positions in HedgeTerminal are protected by the real orders of BuyStop and SellStop playing a role of StopLoss. TakeProfit levels are virtual and supported at the level of HedgeTerminal copies run on one computer and unavailable at the global level._

The StopLoss levels are chosen, as these levels are the ones required to have a high level of triggering reliability. If TakeProfit does not work, it won't be catastrophic and the account won't be closed by margin call, whereas StopLoss that did not trigger can lead to the account bankruptcy.

Nevertheless, there is an algorithmic opportunity to choose a way of trailing a position. You can choose between a real StopLoss and virtual TakeProfit or a virtual StopLoss and a real TakeProfit. The StopLoss and TakeProfit levels can also be virtual. At the moment this feature has not been implemented but if it is in demand, it may appear.

The virtualization of the TakeProfit level lowers its general reliability, though not significantly. The TakeProfit levels are distributed locally and are available to every copy of HedgeTerminal. It is sufficient to have at least one running copy of HedgeTerminal as an Expert Advisor that uses the library or the HedgeTerminal panel, for TakeProfit to be executed. When there are several copies of HedgeTerminal running, only one of them will execute TakeProfit. It will be the first one putting a blocking tag on the bi-directional position. In this sense, instances of HedgeTerminal are competing against each other in the multi-thread mode of writing and reading of the data.

For a user, trading manually through the HedgeTerminal panel or using the virtualization library in the EAs, work with TakeProfit does not differ from work with StopLoss. All factual differences between these levels are hidden behind the scene of HedgeTerminal. It is sufficient for the trader to enter "TakeProfit" and "StopLoss". These levels will be present simultaneously and have the same color indication warning about triggering of one of the levels.

**3.4. Can OCO Orders Resolve Issues with Protection of Bi-Directional Positions?**

OCO orders make it possible to use real StopLoss and TakeProfit levels simultaneously. Are they really that versatile in organizing bi-directional trade? Below are their characteristics. We already know that OCO orders allow canceling one order when the other triggers.

It seems that it will protect our bi-directional position from both sides as in that case Stop-Loss and Take-Profit can be real orders not requiring HedgeTerminal be running on the computer. The thing is that in the exchange order execution it is necessary to take into account _[partial order execution](https://www.mql5.com/en/articles/1284#c1_10)_. This property can destroy the business logic of the application. Let us consider a simple example:

1. A long position with the volume of 10 contracts gets open. Two linked OCO orders implementing StopLoss and TakeProfit levels are placed;
2. The SellLimit order gets partially executed when TakeProfit level gets reached. 7 out of 10 contracts were closed by it and the remaining 3 stayed open as a long position;
3. SellStop order, implementing StopLoss level, will be cancelled as the SellLimit order connected with it was executed though only partially.
4. Position in three contracts does not have a protective stop any more.

This scenario is presented on the figure below:

![Fig. 43. Partial execution of protective stops](https://c.mql5.com/2/12/OCO-orders_with_partial_executed.png)

Fig. 43. Partial execution of protective stops

There may be an objection that OCO orders can be designed so they account for a partial execution and that will allow to avoid such clearing of protective stops. Volumes of two OCO orders can be interconnected. In this case partial execution will definitely be foreseen. However, this will complicate complex enough logics of the order system used in net trading.

The main issue here is that OCO orders cannot provide the same opportunities as MetaTrader 4, even accounting for partial execution. For instance, placing a pending order with TakeProfit and StopLoss levels will be difficult. The reason is that two interconnected orders cannot take into account triggering of the initiating order.

To write a truly versatile algorithm allowing to manage positions similar to MetaTrader 4, OCO orders must have the following characteristics:

1. Every linked order must adjust its volume depending on the execution degree of the order linked to it. For instance, if TakeProfit executed 7 out of 10 contacts, the StopLoss linked to it must change its volume from 10 to 3 (10 - 7 = 3);
2. Each of the linked orders must take into account the volume of the initializing order execution. In the case a trader places a pending order of the BuyLimit type and protects it with StopLoss and TakeProfit in the form of orders, it does not necessarily mean that BuyLimit is guaranteed to execute the whole volume. These cases must also be catered for by the paired orders.
3. In addition to the condition for cancellation, a paired order must have an additional condition for triggering. It can trigger only when an additional order connected with it triggers. That means that an OCO order must have links to two orders. The first link is to the order which triggering will activate the current order. The second link is the order which triggering will cancel the current order. Such a mechanism will allow to create a position with a pending initializing order;

Such mechanisms, even if they appear, will be very complex for a user with limited experience. Their suitability is doubtful. Virtualization on the client's side, like the one currently used in HedgeTerminal, is easier to use.

As an alternative to OCO orders, MetaQuotes could consider a possibility of introducing specific algorithmic TakeProfit and StopLoss levels ensuring protection of a certain trading order. Surely, this is only a theory though it has rational kernel. Such algorithmic levels can hide the major part of the implementation and configuration on the trading server side offering to the end users a simple ready-to-use protection mechanism.

Summing up, our little discourse about the perspectives of integrating OCO orders in the MetaTrader 5 platform:

_OCO orders are not effective when orders are executed partially, they are not reliable enough and too complex for a regular user of the platform._

**3.5. Storing Links to Initializing Orders**

This section considers a detailed description of the internal storage of the links to other orders and mechanism of binding between them. As was mentioned before, the field "Order Magic" contains the identifier of the Expert Advisor that placed an order. This means that any trader can enter any integer value in this field using the MQL5 programming language. In such cases a _**collision**_ is possible, when the order initializing a new position will contain the identifier of an Expert Advisor matching the identifier of an existing order. In this case a wrong link to the order will appear.

If a trader uses identifier for the EAs close to zeros, like _"1"_, _"7"_ or _"100"_, and the order numbers are significantly greater than those numbers, like _"10002384732"_, these collisions can be avoided. It would be rather naive to believe that traders will keep that in mind. That is why HedgeTerminal stores links to orders in a special way so the probability of collisions is very low and its algorithms do not allow ambiguousness and eliminate collisions automatically if they appear.

The _"Order Magic"_ field storing the link takes 64 bit. Due to its width, this field can take a very long number. In reality, the working range of order identifiers is a lot smaller. That allows Hedge Terminal to use higher digits of this field for its needs safely, forming a link to the order in a special way. Let us refer to the scheme showing how HedgeTerminal stores a link to the initiating order:

![Fig. 44. A pattern of storing a link to the initializing order in HedgeTerminal](https://c.mql5.com/2/12/8jjdw_22zhvumf_xq6t7j_t_zyph_OrderMagic.png)

Fig. 44. A pattern of storing a link to the initializing order in HedgeTerminal

The top digit of the field (63) is always marked as 1. This enables a very quick iteration over all orders. Clearly, if the next digit is not equal to 1, then the order cannot contain a link to another order and it can be skipped. Besides, assigning the value of 1 to the highest digit makes the magic number very large and that increases the distance between the working range of the order identifiers and the working range of the links in HedgeTerminal, which minimizes a probability of a collision.

HedgeTerminal can fill the following three bits with service information. Unlike the order identifier, HedgeTerminal stores order identifiers in this field inside out. At first it fills higher digits and then smaller ones, which is pointed at by a blue arrow _direction SI_(service information) at Fig. 44. This way of storage makes the ranges of service information and order identifiers to meet half way. If necessity arises, their composition can be changed. The size of service information can be increased through the digits for storing the order identifier. This information allows to identify the _**type of the closing order**_.

The point is that active orders in MetaTrader 4 can be closed by TakeProfit or StopLoss. These are certain price levels at which orders get closed with fixed profit or loss. In MetaTrader 5, TakeProfit and StopLoss can be applied only to net positions and they are not suitable for paired orders. Only ordinary pending orders can play a role of TakeProfit and StopLoss orders.

In HedgeTerminal such orders are assigned special identifiers specifying whether this is a TakeProfit or StopLoss order. As magic number is stored on the trading server, the service information becomes available for all the traders who have access to the trading account. This way, even several HedgeTerminal copies running on different computers will have the information about the type of the triggered orders and will display the information on closed positions correctly.

The referenced information about the order identifier is stored in the range from _0_ to _59_ digit. This is stored in the standard direction, using digits from right to left, which is indicated by a blue arrow _direction order id._ To evaluate the allocated storage size, let us calculate the amount required for storing the range of all orders sent to Moscow Exchange during a year.

My broker's report dated 13.06.2013 contains an order with the _10 789 965 471_ identifier. This number uses _33.3297 bits_ ( _log2(10 789 965 471)_) or _34_  out of _64_ digits. The identifier of one of the orders placed by 25.09.2014 is _13 400 775 716_. This number uses _33.6416_ bits. Although _2.6 billion orders_ were placed in one year and four months, the size of the identifier increased by only 0.31263 bit, which is less than even one order. I do not claim that the Sun will die before the size of the order identifier will come to the 59th digit, but I am pretty confident that this will happen no earlier than in a couple of million years.

HedgeTerminal does not store a link openly. For that it encrypts the field OrderMagic, leaving the senior digit untouched. The code of HedgeTerminal is based on the reversible rearrangement of digits implemented by the special encrypting function working on the variable length key. After such an encryption, the service information and order identifier get mixed with each other and hidden under a special mask so that on the exit they represent an even distribution of ones and zeros. The encryption takes place for two reasons. First of all, the digit rearrangement decreases the probability of links and order identifiers overlapping and then it protects the internal algorithms of HedgeTerminal from the external deliberate or random impact.

This procedure is absolutely safe, reversible and is not prone to collisions. That warrants that irrespective to the actions a trader performs using the Expert Advisors, they do not affect the reliability of the internal HedgeTerminal algorithms. This is very important as in a real situation managing these links is impossible without complex dedicated algorithms.

At the same time, if we confine ourselves with only controlling the links, the failure of the business logic is inevitable. The following sections will explain why as they are dedicated to the detailed description of these algorithms. Encryption is also used to avoid this failure. The reverse of the coin of this restriction is that there is no other way to manage the HedgeTerminal positions but to use this.

**3.6. Limitations at Work with HedgeTerminal**

The peculiarity of the structure of link storage prompts that HedgeTerminal stores links to the orders evenly using numbers from _9223372036854775808_ to _18446744073709551615_. If a 64-digit field is represented as a number with the sign of the long type, then these will be negative values. Here arise three limitations for work with HedgeTerminal.

The first limitation concerns the trading robot, working with HedgeTerminal. This is not strict and it can be treated as a recommendation:

_A trading robot or an Expert Advisor working with HedgeTerminal must have an identifier (Expert's magic number) not exceeding the value_ _9223372036854775808_.

In reality, usual Expert Advisors will never come across this restriction as identifiers exceeding 5-6 digits are used very seldom. The most common identifier for an Expert is "12345" or something of the kind). This restriction may be applicable for the robots storing service information like links to other orders in their magic numbers. HedgeTerminal is not compatible with such Experts and cannot work in conjunction with them.

If for some reason mentioned limit is exceeded, then appears a zero probability of collision. The latter is very small as coincidence in such a range is very unlikely. Even in this case HedgeTerminal will solve this collision using its algorithms. This will slow its work down as deciphering a link, comparing this link with the existing orders and analyzing this order for suitability for pairing with another order will take extra time. So, to avoid this, it is better not to use long numbers with a negative sign.

The second limitation is hard but it concerns only the broker and the exchange HedgeTerminal is going to be connected to:

_Order identifiers must use numbers from 0 to 2^59 or 576 460 752 303 423 488._

This is obvious as only 59 digits are used for storing order identifiers instead of 64. If your broker uses order identifiers greater than this value, you cannot use Hedge Terminal in your work.

The third limitation follows from the way of position representation:

_HedgeTerminal is not compatible with any other position management system. This is not compatible with external trading panels featuring the function of closing positions and cannot be used together with them._

**3.7. Mechanism of Pairing Orders and Determinism of Actions**

We have considered in detail position representation in HedgeTerminal and the structure of its links. Not we are going to discuss the algorithm description that manage bound orders. For instance, we have two orders chosen from a number of other orders and we have to link them. If we do not rely on the rule that a link to the order must be unique and be possessed only by one order, all possible situations can be divided into three groups:

1. The order is not referenced by any other orders;
2. The order is referenced by one order;
3. The order is referenced by two or more orders.

The first group does not cause difficulties and such an order is considered an open position. Things are very straight forward with the third group too as a pair of orders makes a closed position. What can we do with the cases from the third group? What if an order is referenced to by two other orders? Which of them is supposed to be linked with the first one and what will happen to the second order? It is easy to answer this question if we present the process of pairing as a sequential one:

1. We come across an order that does not contain a link to another order. It gets transferred into the section of active orders (positions) and after that the iteration over orders continues;
2. Then we come across another order that has a reference to the first order. Then the referenced order is being searched in the section of the active orders. If the referenced order is in this section, then it gets paired with the current order and they both get transferred to the section of complete transactions in the form of historical positions;
3. During further iteration, we come across one more order that contains a link to the order described in point 1. Then the referenced order is being searched in the section of the active orders. This time the search will be unsuccessful as the sought orders was carried over from this section to the section of completed transactions. Since the order referenced to by the current order was not found, then the current order despite its link is an active order and is carried to the section with the active positions.

Since the order identifiers are filled in a consistent manner and their list is sorted by time, then their sequential iteration can be performed. So, the initializing order will be paired with the very first order containing a link to it irrespective to the number of other orders containing the same links. All other orders containing links to the initializing order will be included in the list of active initiating positions.

As we already mentioned, the algorithm executing such an iteration must be completely deterministic and consistent. HedgeTerminal uses such an iteration algorithm in its work. Factually, this is not simply an iteration but a repetition of all trading actions that have been performed from the moment of the first deal. Effectively, at every launch HedgeTerminal consistently builds a chain of trading actions from the very beginning to the current moment. Thanks to that, its current position representation is the result of its retrospective trading at the moment of launch.

Since the iteration over all orders in history is carried out sequentially, it is required to perform trading actions sequentially too. That determines the way of the order execution in HedgeTerminal. For instance, an active bi-directional HedgeTerminal position protected by StopLoss is required to be closed. We know that such a bi-directional position essentially consists of two orders: executed order initiating an active position and a pending order buy-stop or sell-stop, acting as a stop-loss order. To close such a position, it is required to delete the pending order and close an active bi-directional position by a counter order with the same volume. So, two trading actions have to be performed. HedgeTerminal executes all trading orders in an asynchronously.

This way orders can be executed simultaneously i.e. the first order can be placed to cancel the pending order and the second order to execute the counter order, which will close your position. This however will upset the determinism of actions and HedgeTerminal cannot perform that. If for some reason, the pending order is not cancelled and the position does not get closed by the counter order, there will be ambiguity as position will be closed and its StopLoss will still exist. In the end, the order that implements StopLoss can trigger and generate a new bi-directional position. That should not happen. That is why HedgeTerminal will cancel StopLoss and after successful cancellation will place a counter order. There is a possibility that the counter order will not be executed or executed partially at the second step. In that case though, ambiguity will be eliminated as even a partially executed counter order will close a part of the active position. This is an ordinary situation.

There is a more complex sequence of actions implemented by HedgeTerminal. Let us use an example similar to the previous one but this time we shall close a part of the position. That means that HedgeTerminal will have to carry out three actions:

1. Delete the pending order that acted as the StopLoss order;
2. Execute the counter order closing a part of the volume of a bi-directional position;
3. Place a new StopLoss with a new volume protecting the remaining part of the active position.

All these actions will be executed consistently to avoid disrupting the determinism of actions. The natural thing to expect is the increase of the order execution speed due to the simultaneous order placement though in that case a sequential execution of trading operations cannot be guaranteed and ambiguities are possible. A sequential order processing does not imply any ambiguity.

**3.8. Splitting and Connecting Deals - the Basis of the Order Arithmetics**

A sequential order handling is not sufficient. Here two things should be noted:

- One order can include several deals. The number of those deals can be random;
- It is also important to take into account a partial order execution and a more common case when a volume of a closing order may be not equal to the volume of the initiating order;

The order can be executed by several deals at the same time and it can be executed partially. If you do not know why it can happen, please refer to the article dedicated to the description of the exchange pricing " [Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)". The question "Why can the volume of the closing order be not equal to the volume of the opening one?" requires an answer. What can we do in this case?

In fact, their volumes can be not equal if we presume an opportunity of a _partial_ closure of an active position. If we open an active position with the volume of 50 contracts and close a part of positions by the counter order with the volume of 20 contracts, then the active position will be split into two parts. The first part will make a new historical position with the volume of 20 orders with the counter order and the second part will still open though its volume will decrease and will be 30 contracts.

This algorithm of a partial position closure is implemented in HedgeTerminal. If a new volume value is entered in the "Volume" field of HedgeTerminal panel, then a partial closure will take place. In the "History" folder a new position will appear and the volume of the current position will be equal to the new value. HedgeTerminal can process the situation even when the volume of the closing order is greater than the volume of the initiating one! Such a situation can take place if HedgeTerminal for some reason places an incorrect volume of the closing order or the broker backhandedly cancels a several deals included into the initiating order, and that will change the executed volume for a smaller one.

To take into account a potential difference in volumes, it is required to use a universal algorithm based on the calculation of the total volume of all deals related to the order. In this case, it is deals, not orders that determine everything. The volume of an order is the volume of its deals. The price of an executed order is the average price of its deals.

HedgeTerminal uses the algorithm that links orders with each other or divides them. It is based on the addition and subtraction of the deals. Its work is based on uniting the deals between the initializing and the closing order with the following formation of a historical position.

To understand how this algorithm works, let us assume that there are two orders required to be paired to make a historical position. Both orders have the same number of deals and their executed volume is the same. Numbers of deals will be three digit for simplicity:

| Order #1 (in order) | Volume (10/10) |
| --- | --- |
| deal #283 | 3 |
| deal #288 | 2 |
| deal #294 | 5 |

Table 3. Order №1 and its deals

| Order #2 (out order) | Volume (10/10) |
| --- | --- |
| deal #871 | 1 |
| deal #882 | 3 |
| deal #921 | 6 |

Table 4. Order №2 and its deals

Let us put their deals with volumes together:

![Table 5. Putting orders together](https://c.mql5.com/2/12/InOut.png)

Table 5. Putting orders together

Select two last orders from each column: №294 and №921. In general they cannot be at the same level with each other (like in this example).

Select a deal with the least volume. This is deal №294 with the volume 5. Divide the opposite deal №921 into two. The first deal is equal to the volume of deal №294 (5 contracts) and the second deal contains the remaining volume of 1 contract (6 contracts – 5 contracts = 1 contract). Unite deal #294 with the volume 5 with the first part of the volume №921 with the similar volume:

![Table 6. Subtraction of volumes](https://c.mql5.com/2/12/3_4.png)

Table 6. Subtraction of volumes

Transfer the united part to a new column containing the deals of the historic position.

This is highlighted in _**green**_. Leave the remaining part of the deal #921 with volume 1 in the initial column of active position. This is highlighted in _**gray**_:

![Table 7. Split and carrying over deals](https://c.mql5.com/2/12/3_5.png)

Table 7. Split and carrying over deals

We have made the first step in uniting deals of two orders and carrying them over to historical position. Let us reflect the sequence of actions in a brief form:

![Table 8. Split and carrying over deals. Step 1](https://c.mql5.com/2/12/3_6_.png)

Table 8. Split and carrying over deals. Step 1

The volume of the deal №294 was carried over in full to the section of historical positions. The deal was _annihilated_ completely. So the deal was _split_ and carried over to the section of historical orders. That is why at the following step we can proceed to the following deal №288 with volume 2. Deal №921 is still present and its volume is equal to the remaining second part of the deal. At the following step this volume will interact with the volume of deal №288.

At the second step repeat the procedure with deals №288 and №921. This time the remaining volume of deal №921 (1 contract) is united with the volume of deal №288 after getting to the column of historical orders. The remaining volume of deal №288 is equal to 1 contract and will remain in the column of the active position:

![Table 9. Split and carrying over deals. Steps 1-2](https://c.mql5.com/2/12/3_7_.png)

Table 9. Split and carrying over deals. Steps 1-2

Repeat the same actions with deals №288 and №882:

![Table 10. Split and carrying over deals. Steps 1-3](https://c.mql5.com/2/12/3_8_.png)

Table 10. Split and carrying over deals. Steps 1-3

Execute steps IV and V the same way:

![Table 11. Split and carrying over deals. Steps 1-5](https://c.mql5.com/2/12/3_9_.png)

Table 11. Split and carrying over deals. Steps 1-5

After step V, the volume of the closing order will be absolutely the same as the volume of the opening order deals. Deals carried over to the column of historical positions make a complete historical transaction. The remaining deals of the active position make an active position. In this case there are no deals left in the column of the active position. This means that after such a unity, the active position will cease to exist. A new historical position will appear and will include all the deals of the active position.

After deals were carried over, many of them will be split into parts and will occupy several lines. To avoid that, an option on collecting deals can be added to the algorithm of uniting/splitting of the deals. Simply unite the volume of the deals with the same identifiers into :

![Table 12. Uniting deals in one level](https://c.mql5.com/2/12/3_10_.png)

Table 12. Uniting deals in one level

After uniting, the number of deals and their volume completely match the initial deals and volumes. This happens only because their volumes initially matched. If their volumes were different, then after the procedure of uniting, the deals would have different volumes.

This algorithm is universal as it does not require the same number of deals for the initiating and closing order. There is another example based on this property we are going to look at:

![Table 13. Uniting orders with different number of deals](https://c.mql5.com/2/12/3_11_.png)

Table 13. Uniting orders with different number of deals

As we can see, uniting orders into one historical position was successful. In spite of a different number of deals, their volume matched again.

Now, imagine that the aggregate volume of the closing order (12 contracts) is less than the volume of the deals of the initiating order (22 contracts). How will uniting go in this case?

![Table 14. Uniting orders with different volumes](https://c.mql5.com/2/12/3_12_.png)

Table 14. Uniting orders with different volumes

As we can see, at the second step, the volume of the deals of the closing order is equal to zero whereas the initiating order contains two more deals №321 with volume 4 and №344 with volume 6. There is an excess of the active order deals. This excess will exist as an active bi-directional position. There is, however, a new historical position with the deals carried over to the green column. Its initial and exit volumes of 12 contracts matched again.

In the case when the volume of the closing order is greater than the initiating, there is also an excess though this time it is on the side of the closing order:

![Table 15. Uniting orders with different volumes](https://c.mql5.com/2/12/3_13_.png)

Table 15. Uniting orders with different volumes

As we can see, the initial order with volume 4 and closing order with volume 6 make two positions. The first one is a historical position with volume 4 and deals №625 of the initial order and №719, №720 of the closing order. The second position is an excess of pairing deals of these orders. It contains deal №719 with volume 2. This deal and the order make an active position in the "Active" tab of the HedgeTerminal panel.

Deals and orders can be divided by the algorithm into historical and active positions. Volumes can be different. The main thing is that the algorithm allows to bring together the volumes of the initiating and closing orders by forming a historical position with equal volumes of entry and exit. It insures the impossibility of the situation when these volumes are not equal and therefore of the errors of position representation causing an asymmetry of the position.

Let us assume that in the first example the broker cancelled a deal included into the closing order:

![Table 16. Simulating of deleting a deal from the history](https://c.mql5.com/2/12/3_14_.png)

Table 16. Simulating of deleting a deal from the history

A new net position with volume of 6 contracts. It will be equal to the volume of cancelled deals. Let us see how the algorithm of Hedge Terminal will work in this case:

![Table 17. Restoring the representation integrity](https://c.mql5.com/2/12/3_15.png)

Table 17. Restoring the representation integrity

As we can see, on step two there is an excess of 6 contracts (3+2+1). This excess will turn into an active position and thus the volume and direction of the bi-directional position is equal to that of the net position.

_Summing up, we can say that the algorithm of uniting deals guarantees the equality of the volumes of the initiating and closing orders in the historical positions due to excessive unbound deals. This excess of deals makes an active bi-directional position, which makes the net position in MetaTrader 5 equal to the net position of all active positions in HedgeTerminal._

This mechanism works both retrospectively and in the real time mode and that means that it will make the net positions in Hedge Terminal even with the net position in MetaTrader 5 irrespective to the broker's action on canceling trading actions. The combinations on the change of a net position and deal cancellation can be as they do not cause an asymmetry between the net position of the terminal and the net position of HedgeTerminal.

The mechanism of partial position closure is based on the capability of this algorithm to bring together different volumes. This algorithm is one of the most important parts of HedgeTerminal ensuring its stable work as a self-adapting system with no setting required.

**3.9. Order and Deal Virtualization**

Every order and deal in HedgeTerminal have a real prototype with a correspondent identifier. Nevertheless, from the point of view of HedgeTerminal one order can make an active position and at the same time be a part of a historical one. Deals and orders in MetaTrader 5 are indivisible entities. An order in the platform can be either pending or executed. A deal also has a constant volume and is always an executed transaction.

In HedgeTerminal same orders and deals can be present in different bi-directional positions. The same deal or order in it can make an active and a historical position. In other words, orders and deals in HedgeTerminal are divided into several virtual ones. This data representation greatly differs from data representation in MetaTrader 5. However, this representation allows to be flexible and adapt to the retrospective changes of trading information and bring orders and deals together.

**3.10. Mechanism of Hiding Orders**

We mentioned the mechanism of hiding orders in the section describing the installation of HedgeTerminal. HedgeTerminal can ignore some orders. For an order or a deal to stop existing for HedgeTerminal, it is enough to enter its identifier to the designated file ExcludeOrders.xml _._

Let us assume that we have several executed sell or buy orders on one symbol. The aggregate volume of sell orders is equal to the aggregate volume of buy orders. This way, no matter how many orders we have, their total position is equal to zero. If identifiers of these orders are not in the ExcludeOrders.xml file, HedgeTerminal will display each of them as a bi-directional position. Their total position though will be zero. Therefore, if a net position in MetaTrader 5 is zero, then the contribution of this order set to the net position can be simply ignored.

Now, let us assume that at the moment _t_ we have a zero net position on the _**S**_ symbol and a set of orders _**N**_ executed on this symbol by this time. As there is no position on this symbol, the total volume of the set of orders _**N**_ is insignificant. In fact, the number of orders and their volume are irrelevant as _since there is no position, these orders do not contribute to the total net position_. This means that such orders can simply be ignored and there is no need to represent them as bi-directional positions.

This is the very mechanism that HedgeTerminal uses at the moment of its installation. At the moment of installation _**t**_ if there is no a net position, HedgeTerminal includes the set of orders _**N**_ to the list of exception. Their total volume and number are irrelevant as there is no net position. If there is a net position at the moment of the HedgeTerminal installation, then it simply won't be installed until the net position is closed. After the installation, new orders will change the state of the net position. This state though will be synchronized with the net volume of the bi-directional positions in HedgeTerminal.

Seemingly, we could get away without putting orders executed by the time of HedgeTerminal to the list of exceptions. At the same time, it may happen that there are a lot of orders by the time of installation and from the point of view of HedgeTerminal all of them will become bi-directional positions and their total volume cab differ from the net volume in MetaTrader 5.

The mechanism of hiding orders can be effective against corruption of the account history. This is how it works. Let us assume that there is some order history. When HedgeTerminal is launched on this account, it will iterate all orders and will build bi-directional position based on them. If all orders are available from the time of the opening and data about these orders are not corrupted, the net position of these orders will be correspondent to the net position in MetaTrader 5. Apparently, net position in HedgeTerminal will be equal to the sum of these orders. This is illustrated on the diagram below:

![Fig. 45. Diagram of integral history](https://c.mql5.com/2/12/GoodHistory.png)

Fig. 45. Diagram of integral history

A part of history can be missing or the information about orders be incorrect.

It is not important if only one order is missing or several orders and it is irrelevant if information is missing in the beginning of the account history or in the middle. Net position in HedgeTerminal running on such an account is equal to the net position of all available orders. The net position of orders will not be equal to the factual net position of the terminal because of the missing part of the history. This situation is presented on the diagram **B**:

![Fig. 46. Partially corrupted history](https://c.mql5.com/2/12/BadHistory.png)

Fig. 46. Partially corrupted history

To synchronize a net position in HedgeTerminal with the factual net position in MetaTrader 5, we do not need to know what orders disappeared or corrupted. All we need is to calculate the difference between these net positions. In the above example, a position to buy with the volume of 5 contracts is open in the MetaTrader 5 terminal. In HedgeTerminal this will be correspondent to a long total position for 8 contracts. The difference between these two positions will be 3 contracts to buy because _**8 BUY – 5 BUY = 3 BUY**_.

After the contract difference has been calculated, it is required to place a correspondent buy or sell order with the volume equal to the difference. In our example it is required to place an order of buying 3 contracts. When the order is executed, HedgeTerminal will display it in the tab of active positions and its total net position will increase by 3 contracts and will become equal to 11 contracts to buy.

The position in MetaTrader 5 will also increase and will make 8 contracts. The identifier of this order has to be entered in the list of ExcludeOrders.xml, and then the terminal has to be restarted. So, if the identifier of our order is equal to 101162513, then the following tag is supposed to be written in the file:

```
<Orders-Exclude>
        ...
        <Order AccountID="10052699" ID="101162513"></Order>
</Orders-Exclude>
```

After restarting HedgeTerminal, this bi-directional position will disappear. This way a net position in MetaTrader 5 will match a net position in HedgeTerminal and will make 5 contracts to buy. The described sequence of actions is presented on the diagram below:

![Fig. 47. Diagram of restoring data integrity](https://c.mql5.com/2/12/Recovery_State.png)

Fig. 47. Diagram of restoring data integrity

Financial result of the disappeared position will not be recorder in the HedgeTerminal statistics. Unfortunately, hidden bi-directional positions do not take part in the trading statistics.

In reality, a situation when a part of history is unavailable or corrupted is highly unlikely. Such a mechanism must be in place though the majority of MetaTrader 5 users will never come across a situation when they need to use it. After all a possibility of program errors of HedgeTerminal are not excluded. Even in this case there should be a reliable instrument of resolving these errors.

**3.11. Adaptation Mechanisms**

We have considered mechanisms allowing HedgeTerminal to represent bi-directional positions in the net environment of MetaTrader 5. There are three of these mechanisms:

1. Sequential iteration of deals;
2. Mechanism of splitting and bringing together deals;
3. Mechanism of hiding orders.

Each of these mechanisms is addressing their issues and allows to avoid ambiguities and errors at their levels. Let us bring these problems and their solutions to the table. The first column contains the problems and possible errors of binding and the second column the mechanisms that solve them:

| Problems, errors, ambiguities arising at the bi-directional trading organization | Mechanisms of error correction |
| --- | --- |
| Errors in the links to the opening orders; Link collision; Long magic numbers in Experts; deleting orders from the history; Errors of trading operation execution. | Sequential iteration over orders. Determinism of actions. |
| Volume errors; Matching orders with different volumes; partial order execution; deleting deals from history; Covering a smaller volume with a greater one. | Mechanism of splitting and matching deals. |
| Installing HedgeTerminal on the account with a great number of executed orders; Corrupted history; Bugs of HedgeTerminal in the work with orders. | Mechanism of hiding orders. |

Table 18. Possible errors and mechanisms of their elimination

All three mechanisms together with the system of storing links ensure stable data representation. Essentially, these mechanisms cover all possible cases of unforeseen failures and guarantee matching of the net position in HedgeTerminal with the net position in MetaTrader 5.

**3.12. Performance and Memory Usage**

HedgeTerminal is essentially an object oriented application. Due to the OOP principles underlying its architecture, the terminal has high efficiency and low requirements to the storage. The only resource consuming task is extracting the history of orders and deals in the computer memory at the moment of launching the terminal on a chart.

After all required transactions were extracted in the memory, the terminal will print a message informing about the time spent on this operation and also memory usage. Launching the HedgeTerminal panel on the account containing more than 20,000 deals on the computer with the processor Intel i7 took less than 30 seconds and it required 118 Mb of RAM:

```
2014.11.20 16:26:19.785 hedgeterminalultimate (EURUSD,H1)       We begin. Parsing of history deals (22156) and orders (22237) completed for 28.080 sec. 118MB RAM used.
```

The HedgeTerminalAPI library works even faster and takes a lot less memory as graphical representation of transactions is not required. Below is the result of the launch on the same account:

```
2014.11.20 16:21:46.183 TestHedgeTerminalAPI (EURUSD,H1)        We are begin. Parsing of history deals (22156) and orders (22237) completed for 22.792 sec. 44MB RAM used.
```

A simple calculation shows that extracting one position takes from 1 to 1.26 milliseconds depending on the program type. Storing one transaction takes: (22 156 deals + 22237 orders) / 44 Mb = 1 Kb of RAM. Storing additional graphic representation of one transaction takes approximately: (118 Mb – 44 Mb) \* 1024 / (22 156 deals + 22237 orders) = 1.71 Kb of memory.

The code profiling shows that the major part of the time is taken by the only block of the order analysis. The major part of the latter is the system function call. In the future versions this block is going to be optimized and that will allow to boost efficiency at the launch by 10-15%.

### Conclusion

We have considered the key points in working with the visual panel HedgeTerminal. It was an example of creating a new class of panels with a possibility of flexible configuration.

Schemes and specifications gave an in-depth idea of the organization principles of bi-directional trading. If you are creating your own virtualization libraries, the second chapter of this article will help you to design such a library.

The character of the exchange execution requires to take into account the key points in the process of deal and order representation as bi-directional positions. This article showed that such a representation was impossible without the virtualization of deals and orders. Virtualization is a mechanism of "breaking" the volume of executed deals and cloning real orders so one order can be a part of several transactions.

These manipulations with the trading environment are rather brave however the major part of the information required at the virtualization is stored on the trading server, such representation can be treated as reliable.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1297](https://www.mql5.com/ru/articles/1297)

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
**[Go to discussion](https://www.mql5.com/en/forum/41237)**
(68)


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
25 Jan 2015 at 09:20

**Wahoo:**

1 :100 leverage. Account 1000$. We open an order on USDCAD for 1 lot. I.e. we use 100% margin. StopOut will not occur until the position goes down by 100 pips.

That's correct. However, Magin will not be converted a la MT for the time being, as the vast majority of HT owners remain neutral.

**komposter:**

95% of users won't even adjust the speakers if it can't be done "just with the mouse".

It's the 21st century and a consumerist society )

And you don't have to. Speaker customisation is an option for enthusiasts and technically advanced users. The default settings are typical and comfortable for most people.

**komposter:**

By the way, the basic settings could be displayed in inputs. The window stretches, lists-enumerations allow convenient selection.

And write the made choice to a file, and use it until the user changes something else.

Now there are almost no such settings that could be put in the inputs window, for example, in the form of lists. And in the future it will not be difficult to write your own graphical window of settings.

**zaskok:**

...It is necessary not to follow the established imposed misconceptions, but to proceed from the definition of the concepts themselves.

...

Swap is a financial transaction calculated for the current net position _._ In reality, a swap is an independent brokerage transaction, not some attribute of a trade transaction with _out_ direction, as it is presented by MT users. Therefore, swap in HT will be displayed as a separate historical transaction, but additionally a classic swap column will be added.

**zaskok:**

Vasily, fully support! On swaps - at least in per cent, not in points, as it is common in all terminals. ....

...

For now there will be a classical representation in the [deposit currency](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_string "MQL5 Documentation: Account Information"). Then all possible variations - through custom column mechanism.


![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
25 Jan 2015 at 09:45

**komposter:**

It would be better to load and draw a panel (empty, a little shaded), and show loading in the centre with a progress\_bar (there are ready-made ones in the base).

Or at least just centre the progress\_bar, so you don't have to look for a small comment.

I'll make a simple small window, with a status bar in the middle. It's simple and much prettier than archaic Comment.

Andrei, don't forget that the mechanism of [history loading](https://www.mql5.com/en/articles/239 "Article \"Testing Basics in MetaTrader 5\"") is the _same_ for both the panel and API, where there is no graphical library at all. Therefore, it is fundamentally impossible to display the panel at the moment of loading.

![Vasiliy Sokolov](https://c.mql5.com/avatar/2017/9/59C3C7E4-C9E1.png)

**[Vasiliy Sokolov](https://www.mql5.com/en/users/c-4)**
\|
28 Jan 2015 at 14:54

**HedgeTerminal 1.03 update:**

All products\*\*:

\- ProgressBar added. Load status is now visible in a small graphical window:

![](https://c.mql5.com/3/56/ProgressBar.png)

This progress bar is displayed for both the panel and the HedgeTerminalAPI

PANEL:

\- The chart is now specially prepared for the table before loading. Once the EA is unloaded, the state of the chart is restored to its original state. Now the panel looks much prettier:

![](https://c.mql5.com/3/56/ndli1rzzgh6jz_0mv0b6.png)

\- Fixed Margin reading. Now it really shows the level of margin loading from 0 to 100%:

![](https://c.mql5.com/3/56/Margin.png)

\- Added seconds to Entry Date and Exit Date columns:

![](https://c.mql5.com/3/56/Seconds.png)

API, article:

\- The second part of the article describing software interaction with the HedgeTerminalAPI library has been released: ["Multidirectional trading and hedging positions in MetaTrader 5 using HedgeTerminal API, part 2](https://www.mql5.com/en/articles/1316)".

In addition to the description of interaction with the library, the article contains information about the basics of multithreaded programming and reveals the specifics of asynchronous trading operations.

The new update will be available after the appropriate moderator's check.


![TipMyPip](https://c.mql5.com/avatar/avatar_na2.png)

**[TipMyPip](https://www.mql5.com/en/users/pcwalker)**
\|
2 Mar 2015 at 18:28

**MetaQuotes:**

New article [Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1](https://www.mql5.com/en/articles/1297) has been published:

Author: [Vasiliy Sokolov](https://www.mql5.com/en/users/C-4 "C-4")

Thank you very much for the Article.


![Rodrigo Malacarne](https://c.mql5.com/avatar/2024/10/67017000-067f.png)

**[Rodrigo Malacarne](https://www.mql5.com/en/users/malacarne)**
\|
8 Jun 2015 at 12:47

Very interesting article. Thanks for sharing!


![Optimization. A Few Simple Ideas](https://c.mql5.com/2/10/DSCI2306_p28-640-480.png)[Optimization. A Few Simple Ideas](https://www.mql5.com/en/articles/1052)

The optimization process can require significant resources of your computer or even of the MQL5 Cloud Network test agents. This article comprises some simple ideas that I use for work facilitation and improvement of the MetaTrader 5 Strategy Tester. I got these ideas from the documentation, forum and articles.

![Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://c.mql5.com/2/12/MOEX.png)[Principles of Exchange Pricing through the Example of Moscow Exchange's Derivatives Market](https://www.mql5.com/en/articles/1284)

This article describes the theory of exchange pricing and clearing specifics of Moscow Exchange's Derivatives Market. This is a comprehensive article for beginners who want to get their first exchange experience on derivatives trading, as well as for experienced forex traders who are considering trading on a centralized exchange platform.

![Studying the CCanvas Class. How to Draw Transparent Objects](https://c.mql5.com/2/17/CCanvas_class_Standard_library_MetaTrader5.png)[Studying the CCanvas Class. How to Draw Transparent Objects](https://www.mql5.com/en/articles/1341)

Do you need more than awkward graphics of moving averages? Do you want to draw something more beautiful than a simple filled rectangle in your terminal? Attractive graphics can be drawn in the terminal. This can be implemented through the CСanvas class, which is used for creating custom graphics. With this class you can implement transparency, blend colors and produce the illusion of transparency by means of overlapping and blending colors.

![MQL5 Cookbook: ОСО Orders](https://c.mql5.com/2/17/OCO-Orders-MetaTrader5.png)[MQL5 Cookbook: ОСО Orders](https://www.mql5.com/en/articles/1582)

Any trader's trading activity involves various mechanisms and interrelationships including relations among orders. This article suggests a solution of OCO orders processing. Standard library classes are extensively involved, as well as new data types are created herein.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1297&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069338750082810729)

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