---
title: Non-standard Automated Trading
url: https://www.mql5.com/en/articles/1485
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:03:22.704592
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1485&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071539169202743785)

MetaTrader 4 / Trading systems


### Introduction

Successful and comfortable trading using MT4 platform without detailed market analysis - is it possible? Can such trading be implemented in practice?

I suppose, yes.

Especially in terms of the automated trading! MQL4 allow doing this. The further described automated trading system is characterized by good repeatability. And it can be easily implemented even by newbies, who are just starting to get acquainted with the basics of writing Expert Advisors.

The system itself is actually a reflection of the outworld. The harmoniously developing life dictates its laws. When children, we all observed the following scene: ants hold a straw from different sides and pull it into an ant-hill. And each ant pulls the straw its own way. Nevertheless, finally the stray is carried in the direction of the ant-hill! A secret of nature? Let us try to simulate the situation on a trading platform.

### Non-standard Automated System

Suppose we have some automated trading system of a profit kind. The system meets the following requirements:

1. Entry signals are practically random.
2. The system is constantly in the market, i.e. it operates with counter-directed positions instead of stops - it is important.
3. Using several static parameters, located at the entrance, we optimize the system, in order to get a maximal profit with a reasonable drawdown.


Practically this system will give annual profit +3000 points even without Money Management block. After that we enable this system in a reverse mode. In other words now it will operate the following way:

1. If we were buying in a direct mode, now we will be selling and vice versa. After that we once again optimize the system using static parameters. And at the exit of the reverse version we get the maximal profit with reasonable drawdown. It should not cause any problems, because our automated trading system is originally built on random entries.

2. After that we simultaneously start both versions - direct and reverse. The simultaneous operation of both versions is a very important, key moment of the described trading system!


Let us see what we have.

At a simultaneous start of both systems - direct and reverse, we make a double profit. However, the systems operate in counter modes, i.e. opposite each other, consciously, not blindly! Besides we have reduced total losses - current relative drawdown. And why do systems operate consciously, not blindly?

The reason is, that because of other static parameters, already in three-four trades after the start the reverse system will trade with some shift in time and price, as compared with the direct system. Still, the entrance algorithm in the reverse mode is the same.

But the total profit will progressively increase! I think it is obvious, while both systems - direct and reverse, are optimized for a profit operation. Moreover, current losses of one version will be almost always covered by the current profit of another version! Consequently we make the maximal profit and minimal drawdown.

Some disadvantage is the increasing of margin requirements. But can this fact be called a disadvantage? Actually here are two operating independent trading systems - direct and reverse. And, naturally, the margin requirement will be double. And as far as risks are concerned, they will be sufficiently reduced! This is the main idea - not to increase profit, but to maximally reduce drawdown. But one thing inflicts another. And, once again, the total profit in such a trading will progressively increase. Consequently, now we can enable the block Money Management.

The site [MQL4.community](https://www.mql5.com/en/code/mt4) contains a wide choice of different Expert Advisors, including those meeting the above listed requirements. The market itself makes us put new tasks and find different, non-standard solutions! We can also find possibilities for further search.

For the realization of the idea and further experiments, we used an Expert Advisor of Yury Reshetov "Artificial Intelligence" as a basis, described earlier in the same part (see the article by Y. Reshetov " [How to Develop a Profitable Trading Strategy](https://www.mql5.com/en/articles/1447)". It was used with some amendments and additions, in particular, it includes an option of calling another basic indicator for the operation of Perceptron. It has also some additional conditions for opening and further tracking of positions.

Here are some results of the experiment, gained during the testing of the pair GBPUSD on the timeframe H1. The start-up deposit is 10000 units. 2.5 year history - from January 2005 till May 2007.

The direct version contained 250 trades within this period. The reverse one - 360 trades. The number of trades differs, because the stoploss level during optimization was different for each version. The net profit in both cases is approximately +10000. During operation with 0.1-lot, not enabling the block Money Management, profitable and loss trades are in the ratio 3:2 in both versions.

Here are the examples of the balance/equity graphs of the direct and reverse versions on the history:

![](https://c.mql5.com/2/14/p_r.gif)

You can see, that in most cases losing trades of the direct version are hedged by profitable trades of the reverse version. And vice versa - losses of the reverse version are hedged by the profits of the direct version.

Moreover, where one chart contains flat, another one contains uptrend! Finally we get the maximal total profit with minimal risks, i.e. with minimal drawdown. It will not cause any problem to gather the history of all trades in Excel and draw a resulting chart, visualising the idea described.

The next step is enabling the block Money Management. We can expect, that enabling this block during such a trading will significantly improve the final results in terms of profit, drawdown and the convenience of trading. In the direct and reverse versions I included calling the library of lots' calculation "b-lots" after I. Kim. It is easily included into the source code and operates quite well: [https://www.mql5.com/en/code/8048](https://www.mql5.com/en/code/8048)

During testing I used proportionate method of calculating lots ( LotsWayChoice=2 ), which gives a reasonable minimal drawdown with relatively good profits (Ryan Jones, "The Trading Game: Playing by the Numbers to Make Millions"). The results are quite good, as compared with other methods of calculating lots, which very often show large profits with large drawdown.

Let's view the same history, from January 2005 till May 2007. The results of testing with the same parameters, as earlier:

Reverse version:

- Net profit +79864
- Maximal drawdown 16969 (24%)
- Relative drawdown 33% (3511)

Direct version:

- Net profit +196520
- Maximal drawdown 25801 (12.3%)
- Relative drawdown 18.14% (6972)

Here are the balance charts:

![](https://c.mql5.com/2/14/mm__1.gif)

These charts also very vividly display, how the current losses are covered by the reverse-version profits. I think, the convenience of such trading is evident.

### Practical Use

As an example and for further primary experiments, here is a code of a reverse version, which works contrary to the direct version of the author's variant of the Expert Advisor "AI" by Y. Reshetov. A direct version and its description are located at: [https://www.mql5.com/ru/code/10289](https://www.mql5.com/en/code/10289)

Beside this, the attached file contains the indicator Perceptron (the author - NoName from Ukraine, Kremenchug), which enables to control visually the current work of Expert Advisors (direct and reverse), and know beforehand, in which direction a new position will be opened. Here set the values of weight coefficients X1-X4 of the indicator Perceptron equal to the corresponding EA values. Here is an example of the reverse version:

![](https://c.mql5.com/2/14/4444_4.gif)

For those, who only start working with MQL4, I tried to give maximum comments on the EA operation. The perceptron outputs of this reverse version's operation are displayed on the chart in the upper left corner:

```
//+------------------------------------------------------------------+
//|                                 ArtificialIntelligenceRevers.mq4 |
//|                               Copyright й 2006, Yury V. Reshetov |
//|                                Modifed by   Leonid553            |
//|                                  http://www.tradersforum.net.ru/ |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright й 2006, Yury V. Reshetov ICQ:282715499"
#property link      "http://reshetov.xnet.uz/"
//---- input parameters
extern int    x1 = 88;
extern int    x2 = 172;
extern int    x3 = 39;
extern int    x4 = 172;
// StopLoss level
extern double sl = 50;
extern double lots = 0.1;
extern int    MagicNumber = 808;
static int prevtime = 0;
static int spread = 3;
//+------------------------------------------------------------------+
//| expert initialization function                                   |
//+------------------------------------------------------------------+
int init()
  {
//----
   return(0);
  }
//+------------------------------------------------------------------+
//| expert deinitialization function                                 |
//+------------------------------------------------------------------+
int deinit()
  {
//----
   return(0);
  }
//+------------------------------------------------------------------+
//| expert start function                                            |
//+------------------------------------------------------------------+
int start()
  {
     Comment(perceptron());
// Wait for the formation of a new candlestick
// If a new candlestick appears, check for the possibility of a trade
   if(Time[0] == prevtime) return(0);
   prevtime = Time[0];
//----
   if(IsTradeAllowed())
     {
       spread = MarketInfo(Symbol(), MODE_SPREAD);
     }
   else
     {
       prevtime = Time[1];
       return(0);
     }
   int ticket = -1;
   // check for opened position
   int total = OrdersTotal();
   for(int i = 0; i < total; i++)
     {
       OrderSelect(i, SELECT_BY_POS, MODE_TRADES);
       // check for symbol & magic number
       if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         {
           int prevticket = OrderTicket();
           // long position is opened
           if(OrderType() == OP_BUY)
              // if a long position is opened and ...
             {
               // check profit
               // the current profit is larger than the value of =(stoploss + spread) and ...
               if(Bid > (OrderStopLoss() + (sl * 2  + spread) * Point))
                 {
                   if(perceptron() > 0)
                     {
                       // perceptron is more than zero, then turn to Sell
                       // reverse
                       ticket = OrderSend(Symbol(), OP_SELL, lots * 2, Bid, 3,
                         Ask + sl * Point, 0, "AI", MagicNumber,
                                          0, Red);
                       Sleep(30000);
                       if(ticket < 0)
                         {
                           prevtime = Time[1];
                         }
                       else
                         {
                           OrderCloseBy(ticket, prevticket, Blue);
                         }
                     }
                   else
//if perceptron is less than zero, trail the stoploss to the distance =sl
//from the current price
  {
                       // trailing stop
                       if(!OrderModify(OrderTicket(), OrderOpenPrice(),
                          Bid - sl * Point, 0, 0, Blue))
                         {
                           Sleep(30000);
                           prevtime = Time[1];
                         }
                     }
                 }
               // short position is opened
             }
           else
             {
               // if a short position is opened and ...
               // check profit
              if(Ask < (OrderStopLoss() - (sl * 2 + spread) * Point))
                 {
                  // the current profit is larger than the value of =(stoploss + spread) and ...
                   if(perceptron() < 0)
                     {
                       // perceptron is less than zero, then turn to Buy
                       // reverse
                       ticket = OrderSend(Symbol(), OP_BUY, lots * 2, Ask, 3,
                           Bid - sl * Point, 0, "AI", MagicNumber,
                                          0, Blue);
                       Sleep(30000);
                       if(ticket < 0)
                         {
                           prevtime = Time[1];
                         }
                       else
                         {
                           OrderCloseBy(ticket, prevticket, Blue);
                         }
                     }
                   else
//if perceptron is more than zero, trail the stoploss to the distance =sl
//from the current price
                   {
                       // trailing stop
                       if(!OrderModify(OrderTicket(), OrderOpenPrice(),
                          Ask + sl * Point, 0, 0, Blue))
                         {
                           Sleep(30000);
                           prevtime = Time[1];
                         }
                     }
                 }
             }
           // exit
           return(0);
         }
     }
//********************************************************************
   // check for long or short position possibility
   // initial entrance to the market:

   if(perceptron() < 0)
     {
       // if the perceptron is less than zero, open a long position :
       // long
       ticket = OrderSend(Symbol(), OP_BUY, lots, Ask, 3, Bid - sl * Point, 0,
                      "AI", MagicNumber, 0, Blue);
       if(ticket < 0)
         {
           Sleep(30000);
           prevtime = Time[1];
         }
     }
   else
     // if the perceptron is more than zero, open a short position:
     {
       // short
       ticket = OrderSend(Symbol(), OP_SELL, lots, Bid, 3, Ask + sl * Point, 0,
                      "AI", MagicNumber, 0, Red);
       if(ticket < 0)
         {
           Sleep(30000);
           prevtime = Time[1];
         }
     }
//--- exit
   return(0);
  }
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;
   double a1 = iAC(Symbol(), 0, 0);
   double a2 = iAC(Symbol(), 0, 7);
   double a3 = iAC(Symbol(), 0, 14);
   double a4 = iAC(Symbol(), 0, 21);
   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
//+------------------------------------------------------------------+}
```

It should be noted, though, that the main meaning of the described trading tactic is not in this particular Expert Advisor. But on the contrary, it is in the cooperation of the direct and reverse versions of any proper Expert Advisor, as explained in the beginning of the article. And here are different variants possible: you can have several reverse versions - created and optimized upon different criteria, in accordance with the initial algorithm of the direct version. The choice here is quite wide.

Working further on this idea, I would offer drawing support and resistance lines. Or attach the MA indicator or any other proper indicator to the balance chart. In this case, upon the indicator signals we could to some extent handle the prohibition of trades on each version. I think it is programmable.

But it is in prospect.

### Conclusion

A can presuppose the objections of sceptics - and what if both versions start working at a loss?

Well, this may happen in rear, exceptional cases. But no more than that - "nothing is ideal in this world". However, at practically accidental entries, both versions work on the signals of one indicator and in counter directions - one against the other!

Combining both versions into one Expert Advisor, we get an efficient tool for further management of a portfolio trading. And for further experiments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1485](https://www.mql5.com/ru/articles/1485)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1485.zip "Download all attachments in the single ZIP archive")

[AI\_REVERS.mq4](https://www.mql5.com/en/articles/download/1485/AI_REVERS.mq4 "Download AI_REVERS.mq4")(6.98 KB)

[PerceptronIndicator\_AC.mq4](https://www.mql5.com/en/articles/download/1485/PerceptronIndicator_AC.mq4 "Download PerceptronIndicator_AC.mq4")(2.57 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39339)**
(6)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Aug 2007 at 09:33

How did you backtest your EAs? With downloaded charts and simulated ticks?

I have experimented with direct and [reverse strategies](https://www.mql5.com/en/blogs/tags/trading-strategies "Trading Strategies") a lot, and I learnt the
following: hedge trades don't really work on Forex. If you open a 1 lot buy and
a 1 lot buy, you did nothing, but paid for spread the two will cancel each other
out. If you have an opened buy and you opened a sell, it is the same as closing
your buy. If you have an opened buy and an opened sell and you close your sell
earlier than your buy, it is the same as opening a buy at that time. What matters
is the pricemovement under which there was an imbalance in the number of opposing
trades. What I want to say here: everything that can be with two or more opposing
trades can be made with one trade too, which way spread will be half. So I am a
bit unbelieving here.


![LesioS](https://c.mql5.com/avatar/avatar_na2.png)

**[LesioS](https://www.mql5.com/en/users/lesios)**
\|
30 Aug 2007 at 10:01

As I properly understood the main idea, it's not hegding. Direct and revers EA works
on **different** magic numbers, so they don't interfere each other. They both work on their own
orders and buy & sell indepenently.


![Leonid Borsky](https://c.mql5.com/avatar/avatar_na2.png)

**[Leonid Borsky](https://www.mql5.com/en/users/leonid553)**
\|
21 Sep 2007 at 12:23

For LesiosS :

['Статья: Нестандартная автоматическая торговля'](https://www.mql5.com/ru/forum/103968/page4)

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
17 Oct 2007 at 00:21

Ok the

\- Reverse version is posted above.

\- The direct version is at ['AI'](https://www.mql5.com/ru/code/10289)

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
10 Dec 2010 at 11:21

I like the idea of direct and reverse. The reason being that no stop loss is used.

But to acheive that I assume that you use two currency pairs. Not one.

That is, one currency pair for direct and a different pair for reverse.

So first you open a direct buy and when in -50 pips(let's say) you open a sell on the other pair.

When do you [take profit](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders")? When the total profit of both is $150(lets say)?

Please reply

Regards

Andreas

![Testing Visualization: Account State Charts](https://c.mql5.com/2/14/415_13.jpg)[Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)

Enjoy the process of testing with charts, displaying the balance - now all the necessary information is always in view!

![Tester in the Terminal MetaTrader 4: It Should Be Known](https://c.mql5.com/2/14/424_77.gif)[Tester in the Terminal MetaTrader 4: It Should Be Known](https://www.mql5.com/en/articles/1490)

The elaborate interface of the terminal MetaTrader 4 is a forefront, but beside this the terminal includes a deep-laid tester of strategies. And while the worth of MetaTrader 4 as a trading terminal is obvious, the quality of the tester's strategy testing can be assessed only in practice. This article shows the advantages and conveniences of testing in MetaTrader 4.

![Strings: Table of ASCII Symbols and Its Use](https://c.mql5.com/2/14/457_10.png)[Strings: Table of ASCII Symbols and Its Use](https://www.mql5.com/en/articles/1474)

In this article we will analyze the table of ASCII symbols and the ways it can be used. We will also deal with some new functions, the principle of operation of which is based on the peculiarities of the ASCII table, and then we will create a new library, which will include these functions. They are quite popular in other programming languages, but they are not included into the list of built-in functions. Besides, we will examine in details the basics of working with strings. So, I think you will certainly learn something new about this useful type of data.

![Interaction between MetaTrader 4 and Matlab via CSV Files](https://c.mql5.com/2/14/421_29.gif)[Interaction between MetaTrader 4 and Matlab via CSV Files](https://www.mql5.com/en/articles/1489)

Step-by-step instructions of how to organize data arrays exchange between MetaTrader 4 and Matlab via CSV files.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1485&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071539169202743785)

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