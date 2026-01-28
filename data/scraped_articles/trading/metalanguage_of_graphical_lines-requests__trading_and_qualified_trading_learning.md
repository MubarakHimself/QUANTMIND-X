---
title: Metalanguage of Graphical Lines-Requests. Trading and Qualified Trading Learning
url: https://www.mql5.com/en/articles/1524
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:41:42.945960
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1524&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083046035128849381)

MetaTrader 4 / Trading


### Introduction

The article dwells on the phenomenon of difficulties experienced by practicing traders when using the set of Technical Analysis means. The offered problem solution implies a half-automated use of the Graphical Analysis in trading in the form of a metalanguage of graphical lines-trade orders. Below will be shown that this is the metalanguage of trade graphical orders. Evaluation of the offered metalanguage of graphical objects for the real trading and training is available in the Expert Advisor GTerminal created by the author of the article.

The article also contains some recommendations for using in trading learning. The attached file is the source file of GTerminal of MQL4 formate. The EA has been tested on a live deposit.

## The Main Technical Analysis Tool — Graphical Analysis

Before entering the market, a trader is believed to necessarily draw a security chart in a trade terminal or in a special analytical program. This preliminary work is called Graphical Analysis.

The necessity of Graphical Analysis is fundamental for trading learning. The ability of a trainee to conduct Graphical Analysis is the sign of his or her skills. Particularly, after a theoretical course a student has practical training, when a trader-to-be does not yet trade, but every day sends the graphical analysis of market events to his teacher.

However after trading learning is over beginning traders rarely use a lined chart. Here are some of traders' explanations, why they do not use it:

\- allegedly, technical analysis is now out of date and cannot be applied in my work;

\- I am using Elliot wave analysis, I don't need technical analysis;

\- I understand nothing in technical analysis, I trade based on an indicator;

\- graphical analysis is not used in my trading system;

\- I studied, but cannot remember figures of technical analysis;

\- I don't have time for drawing charts, etc.

Only after a large period of accumulating their own reading experience, usually quite sad, a trader little by little gets the ability to see the chart. For a beginning trader technical analysis has a formal character and is hardly connected with the trading process.

The reason is psychological.

Graphical analysis denotes lines, spacial reasoning in a certain duration. However, when switching form the analysis to the implementation of a trading plan, a trader has to wait for the market moment for entering/exiting. Here the reasoning is different, implying the attempt to identify an entry point, i.e. actually a "point" reasoning, which does not match with graphical lines.

The reason is technical.

Graphical analysis is not executed by the client terminal. There is no computer support between the graphical analysis and the market, there is no automated trading. We need a simple, available, functionally obvious language of graphical trade orders compatible with the traditional technical analysis.

Strange as it may seem, but there should be nothing new in the wanted language of graphical trade orders. Trading is a conservative occupation. If trade requests are executed on the basis of graphical analysis lines, the basis of graphical management must be trade request lines.

There are two types of requests - breakout and pivot. Correspondingly, there are two ready notions: stop order and limit order. All the necessary information for these orders must be contained directly in a graphical object, i.e. in an object's text properties: in its name and description text.

## Metalanguage of Graphical Trading Management

Below is a certain implementation of the metalanguage of graphical trading management in the GTerminal EA with the following limitations:

\- only one order up/down can be opened;

\- lot size common for all orders is set in the EA properties.

These limitations are set here for training financial discipline in deposit management.

Metalanguage has been rested in live trading in Forex, CFD and in the visual mode of a strategy tester.

Trading request lines are, strictly speaking, lines of the expected breakout or pivot, plus trade request = line name + parameters of Take Profit and Stop Loss orders.

Lines-requests "to open order" have the following names:

BuyStoptp=x sl=x

BuyLimittp=x sl=x

SellStoptp=x sl=x

SellLimittp=x sl=x

Here x is a numeric value in points. For example, BuyStoptp=40 sl=120

A request name is written in one word, delimiters are spaces, no space between "=" and figures. Capital/small letters are not differentiated. Take Profit and Stop Loss are located in any order, can be absent, default value=0.

Example of creating a request line name:

![](https://c.mql5.com/2/16/1.png)

GTerminal accepted for execution the SellLimit line, recorded in "Properties/Description" of the line numerical values for verification.

To see the verification data, it is not necessary to enter line properties, you can simply move the mouse cursor to a line.

![](https://c.mql5.com/2/16/2.png)

Lines-requests "to close order":

SlBuy

TpBuy

SlSell

TpSell

In the TpSell line properties the description "O.k." appeared after it was accepted for execution.

![](https://c.mql5.com/2/16/3.png)

Lines-requests "to close all orders of the specified type":

SlAllBuy

TpAllBuy

SlAllSell

TpAllSell

Orders of other symbols, as well as pending orders are not closed. Convenient to use for orders opened manually. They are equivalent to trailing (except for the cases of server connection failure).

Suppose we expect price to exit the TA figure "triangle", correspondingly we place the line-request BuyStop above the figure and SellStop below it. Take Profit will be placed upon TpBuy and TpSell lines, and the trailing Stop Loss will be set by SlBuyand SlSell lines. For the case of a long connection failure let us set order Take Profit and Stop Loss. Lines-request are lines of MetaTrader 4 terminal with the name set in the metalanguage of trade requests. I.e. place a line on a chart, enter the line properties and in the "name" field write the metalanguage request.

Suppose a price channel with obvious sides. Let's place lines-requests BuyLimit and SellLimit. On the contrary channel side let's place TpBuy and TpSell lines. Caring about the possibility of price exiting the channel, let's place SlSell and SlBuy lilnes on the chart. Length of trade requests lines is the scope of our forecasting. When the forecast is no more effective let's finish the line.

Use of lines-requests does not require any additional training, because all they correspond to common trade operations.

On the chart there are two pairs of graphical orders accepted for execution:

SellLimitis the upper brown line, TpSell is the lower brown line,BuyLimit is the upper blue line,TpBuyis the lower blue line. Colors of graphical orders are set by an EA. Line names are shown when pointing a mouse cursor at a line.

![](https://c.mql5.com/2/16/oxrrb.gif)

After a graphical request to open an order triggers, the EA changes the line name and its graphical attribute. A brown dotted line remains instead of the executed upper SellLimit line. Note, the newly opened Sell contract has already set Stop Loss and Take Profit, which were earlier placed by value in the line-request name. The profit of this opened contract will be registered by the EA when the price reaches the lower brown line TpSell.

![](https://c.mql5.com/2/16/jrwjtpjthfxmcsgntuwjjsmy.gif)

The name of the executed line-request "SellLimittp=180 sl=60" has been changed by the EA: an order ticker is added to the left, date and time are added to the right.

![](https://c.mql5.com/2/16/4.png)

Implementation Peculiarities of the Graphical Management Metalanguage

In the graphical management metalanguage a trading event happens at the intersection of price/line-request.

In GTerminal properties of crossing are adjustable.

The position of crossing is set: the 'start' bar number, on which the crossing is checked. By default it is the zero bar.

Condition of crossing calculation is set:

\- stringent condition, close prices of two bars Close\[start\] and Close\[start+1\] are on the opposite sides of the line;

\- mild condition (by default), close price of one bar Close\[start\] has crossed the line.

Trade request are not effective beyond the line ends.

Graphical position of the line-request is checked at each tick. That is why lines can be shifted during operation, thus trading can be managed. For example, the level of profit/loss fixing can be shifted to cross the price line and thus open/close an order.

MT4 Client Terminal does not allow graphical lines with the same names. However, in the offered metalanguage names are trade requests consisting of a constant left part - 'command' and the right part variable 'parameters'. That is why MT4 does not control introducing to a chart lines of the same command but with different parameters. GTerminal solves this problem the following way:

\- To check the price/line-request crossing the closest to the price line-request is chosen from the list of lines having the same names-commands.

\- In lines-requests the name-command must be necessarily placed in the left part, after it any information can be written separated by a space.

This implementation feature allows the preliminary preparation of several lines-request that differ only in the optional record in the right part (obligatory through a space). Then in the operation process the needed lines-requests can be shifted closer to a price.

The metalanguage can be substantially extended. However, this is out of the scope of the article purposes, namely - the solution of a problem of a half-automated graphical Analysis use in trading.

Conclusion for Graphical Management Metalanguage

As for their functions lines-requests are lines of graphical orders. In terms of programming it is convenient to use the name "graphical line-request" to avoid confusion with the often-repeated word 'order'. However in practical trading it is more convenient to call graphical management lines 'graphical orders'. The set of graphical orders is the trading plan to be executed by the EA GTerminal.

## Psychological Concept of GTerminal – Mild Start

One of the main disadvantages of the Technical Analysis is the difficulty to learn it. Many practicing traders ignore TA, because they are not used to applying even support/resistance levels, but are guided by their inner feeling. TA ignoring is visible in forums and article headings. The doubts about TA application in the present market are expressed publicly. The reason for these doubts is actually very simple: TA is said to be unsuitable for application only because the personal training is not enough for using it.

Accordingly, training of Technical Analysis tools without using computer means is rather boring. TA methods are not bound with trading and therefore are not popular in learning. The proper learning of Technical Analysis in theory and on trading accounts requires thousands of lessons, long calender months. This is not modern and inefficient.

That is why GTerminal Was initially designed as a trainer that sets a necessary reasoning through graphical lines as is common for the classical TA. The live computer implementation impels a user to learn the wealth of TA means.

Here is one more psychological aspect. It is known that many MT4 users are still afraid of a PC. The reason is they never used a computer before starting to trade in MT4. This fear of PC is especially obvious in the order management.

Conducting lessons with beginners we discovered that they easily understand trading principles explained on a chart, but when order sending is explained, we have to learn once again what has already been learned. The difficulty for beginners here is that after the image thinking on a chart they have to switch to Parameter tabs, fill in them, check, amend, send and then observe the server answer. It is all quite difficult for a person that is not used to computer execution.

But if trading parameters are contained on a chart like in the offered metalanguage, there will be no stress and it will be easier to explain trading principles.

Fear of computer also results in the fear of automated trading systems. In this connection a trading Expert Advisor that offers automation services but does not interfere with decision making and the EA that can be used from the very beginning of trading learning provides a mild switch, helps to start using other programs for trading automation.

## Trading Learning

Graphical management of GTerminal allows to trade in the mode of visual MT4 testing.

Let's draw some examples of its usage in learning.

Animated training material.GTerminalis a training Expert Advisor in terms of the display of trading processes on a screen. Starting from the very first lesson when the answer to "what is the idea of trading" is explained, GTerminal provides wide illustration possibilities.

Training/testing/demonstrating a trading strategy. The forecast of a tested trading strategy is shown in graphical order lines and the result is expected.

Manual repetition of manual trading on the same part of historic data.

The value of such a repetition consists in the following. Trading on one and the same part of history repeatedly, a trader again and again observes price behavior and remembers some of TA figures, starts reacting to candlestick combinations, indicator signals. Experience shows that a trader needs up to 8 trade repetitions on the same history part for getting a satisfactory result. The result of the repeated trading will be better, which creates a positive motivation to training.

Multiple repetition of graphical analysis in the same calendar period.

It should be noted that MT4 Tester allows using any indicators. Press \|\| and a chart will stop. Then draw graphical analysis, take into account indicators, place lines-requests of a trading plan - graphical orders of GTerminal. Then press >\> and observe the execution of the trading plan. Get the trading assessment. All the disadvantages and mistakes of the analysis will be visible. Then analyze trading results. After that repeat the same on the same part until the satisfactory result of forecasting using Technical Analysis is achieved.

Cultivation of the betting feeling.GTerminal can be applied outside any TA for developing perceptibility in scalping and pips trading.

Competition.GTerminal in MT4 strategy tester allows the competition of several participants.

## Conclusion

Graphical trade interface is so convenient that probably in course of time the metalanguage of graphical lines- trade requests or the metalanguage of graphical orders will become a built-in terminal language.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1524](https://www.mql5.com/ru/articles/1524)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1524.zip "Download all attachments in the single ZIP archive")

[GTerminal.mq4](https://www.mql5.com/en/articles/download/1524/GTerminal.mq4 "Download GTerminal.mq4")(28.4 KB)

[GTerminal\_V5a\_en.mq4](https://www.mql5.com/en/articles/download/1524/GTerminal_V5a_en.mq4 "Download GTerminal_V5a_en.mq4")(35.56 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39439)**
(3)


![tradelover](https://c.mql5.com/avatar/avatar_na2.png)

**[tradelover](https://www.mql5.com/en/users/tradelover)**
\|
16 May 2008 at 07:35

Very nice tool, indeed! Thanks for publishing it. For a more complex (free) tool for graphical trading and strategy testing, see my blog at [vamist.com](https://www.mql5.com/go?link=http://forum.vamist.com/blog/tradelover/index.php?showentry=100 "http://forum.vamist.com/blog/tradelover/index.php?showentry=100")

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
6 Jun 2008 at 16:49

That's exactly the kind of feature I dreamt of... Great!

![Bluesky](https://c.mql5.com/avatar/avatar_na2.png)

**[Bluesky](https://www.mql5.com/en/users/bluesky)**
\|
21 Nov 2009 at 14:51

Great man! Been using for a while for testing.

Only problem: cannot open more than 1 buy or [sell order](https://www.mql5.com/en/docs/constants/tradingconstants/enum_book_type "MQL5 documentation: Trade Orders in Depth Of Market"). Why?

Thanks

DAvid

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization](https://c.mql5.com/2/15/575_71.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization](https://www.mql5.com/en/articles/1516)

This article dwells on implementation algorithm of simplest trading systems. The article will be useful for beginning traders and EA writers.

![Easy Way to Publish a Video at MQL4.Community](https://c.mql5.com/2/15/582_26.jpg)[Easy Way to Publish a Video at MQL4.Community](https://www.mql5.com/en/articles/1520)

It is usually easier to show, than to explain. We offer a simple and free way to create a video clip using CamStudio for publishing it in MQL.community forums.

![Comparative Analysis of 30 Indicators and Oscillators](https://c.mql5.com/2/15/577_13.gif)[Comparative Analysis of 30 Indicators and Oscillators](https://www.mql5.com/en/articles/1518)

The article describes an Expert Advisor that allows conducting the comparative analysis of 30 indicators and oscillators aiming at the formation of an effective package of indexes for trading.

![An Expert Advisor Made to Order. Manual for a Trader](https://c.mql5.com/2/117/robot__2.png)[An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)

Not all traders are programmers. And not all of the programmers are really good ones. So, what should be done, if you need to automate your system by do not have time and desire to study MQL4?

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1524&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083046035128849381)

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