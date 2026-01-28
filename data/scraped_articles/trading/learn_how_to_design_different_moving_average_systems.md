---
title: Learn how to design different Moving Average systems
url: https://www.mql5.com/en/articles/3040
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:15:09.197893
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/3040&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069279793066738299)

MetaTrader 5 / Trading


### Introduction

I think that whatever your experience period in the trading filed you must hear that quote “Trend is your friend”. And if you didn’t hear about it, here what is meaning as all know that we have different types of market directions or trends of price movement.

- Uptrend: we can see it when prices make higher lows and higher highs
- Downtrend: we can see it when prices make the opposite of Uptrend, Lower highs and lower lows.
- Sideway: is every moving except uptrend or downtrend.

And the following figures can show uptrend and downtrend as a line chart and any movement except them is a sideway:

![Uptrend](https://c.mql5.com/2/44/Uptrend.png)

![Downtrend](https://c.mql5.com/2/44/Downtrend.png)

Now, we identified the trend but why the trend is my friend?

We will talk here about trends, i.e. uptrend and downtrend, as they have a clear movement either up or down. We can see the control of one particular market participant. During the uptrend, we can see that the buyer is the controller as he pushes the prices to up whatever the supply is and here, we call this market as a bull market. And vice versa, during the downtrend, we can see that the seller is the controller as he pushes the prices to down and here. We call this market as a bear market.

But in some cases, we may experience what we call whipsaws or false breakouts. And these false breakouts may harm our trading results, how is that? We will see here ho they can, but first let us identify what are whipsaws or false breakouts.

**False breakouts** are these signals that give us a trigger to take a decision then and after taking the decision the market goes against that decision. And for sure these kinds of decisions will harm our trading results. These false breakouts or whipsaws can be reduced by filtering them by many strategies. One of the best strategies that can do that for us is using a moving average and this is the subject of this article to learn how we can use some of moving average strategies and how to design an algorithmic trading system for them and make using of them accurate, easy, and systematic.

In this article, we will go through the following topics:

- [Moving Average Definition](https://www.mql5.com/en/articles/3040#par1)
- [Types of Moving Averages](https://www.mql5.com/en/articles/3040#par2)
- [Strategy 1: One Moving Average Crossover](https://www.mql5.com/en/articles/3040#par3)
- [Strategy2: Two Moving Averages Crossover](https://www.mql5.com/en/articles/3040#par4)
- [Strategy3: Three Moving Averages Crossover](https://www.mql5.com/en/articles/3040#par5)

**Disclaimer:** All content of this article is made for the purpose of education only not for anything else. So, any action will be taken based on the content of this article, it will be your responsibility as the content of this article will not guarantee any kind of results. All strategies may need prior testing and optimizations to give better results as like what I mention the main objective from this article is an educational purpose only.

All codes will be written by MQL5 and will be tested on MT5.

Note that there are many strategies that can be used to filter generated signals based on any strategy, even by using the moving average itself which is the subject of this article. So, the objective of this article to share with you some of Moving Average Strategies and how to design an algorithmic trading system to let you know and open your eyes on what you can do and how could you develop your trading strategy. Now, let us go through this subject “Learn How to Design Different Moving Average Systems” as I am very excited to share it with you…

### Moving Average Definition

Moving Average is an indicator that is commonly used in technical analysis and it is calculated to give us an average from prices during a specific period of time or in other words it helps us to smooth prices data to mitigate random or fluctuation of short-term movements on the chart.

There are many types of Moving Averages and the differences between them are related to different calculations for each, like what I will mention in details in the next topic. But here what I want you to know till further details is that there are many types and the difference between them because of the calculation to get the best result.

Moving Average is a trend following indicator as it is calculated by the prices. Therefore, if the price moves in a specific trend, the moving average will follow the same trend.

Moving Average is a lagging indicator as it moves after the price, and this is normal as it is calculated by the price so the price must move first. Or, in other words, we must have prices data first, and then we can find the moving average data by make calculations on these data of prices.

According to the nature of moving average, we can understand that the moving average:

- It gives the Mean.
- It works better with trending markets.
- It will confirm that there is a trend or it will help us to identify the trend.
- It will clear noise or random movement from the prices.
- It will help us to avoid whipsaws or false breakouts because of random movements as it will be eliminated.
- And much more…...

### Types of Moving Averages

There are many types of Moving Averages that can be used and the most common used are:

- Simple Moving Average.
- Weighted Moving Average.
- Exponential Moving Average.

The main difference between them is connected with different calculations to get a better result the same as I mention before.

So, we will identify now these types of moving averages and will explore the differences between them. We will also see how these moving averages can be used in MQL5 or how we can call them if we want to use a specific kind of them.

**Simple Moving Average:**

This type is the simplest form or type of the moving average, its commonly used shortcut is SMA.

It is calculated by taking the arithmetic mean of a given set of values over a specific period of time. Or a set of numbers of prices are added together then divided by the number of prices in the set.

![SMA](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.11.51_PM.png)

Example: We have 5 trading days and the closing prices during these 5 days was as the following:

- Day1 = 10
- Day2 = 15
- Day3 = 20
- Day4 = 30
- Day5 = 35

If we want to get SMA for these 5 Days, it will be calculated as the following:

![SMA example](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.12.04_PM.png)

**Weighted Moving Average**

The Weighted Moving Average (WMA) as it is commonly used, it gives an average for prices during a specific period of time but the difference here as it gives more weight for recent data. This means that if we calculate 10 trading days, this kind of average will not give the weight for all 10 trading days but it will give more weight to the recent data.

It can be calculated the same as the following formula:

![WMA](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.12.18_PM.png)

Example:

If we have the same 5 trading days and the closing prices during these 5 days was as the following:

- Day1 = 10
- Day2 = 15
- Day3 = 20
- Day4 = 30
- Day5 = 35

If we want to get WMA for these 5 Days, it will be calculated as the following:

![WMA example](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.12.34_PM.png)

![WMA example1](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.12.46_PM.png)

**Exponential Moving Average**

The Exponential Moving Average or EMA as it is commonly used. It gives an average for prices and the difference here is EMA does not cancel the period before the calculated period at MA, this means that if we need to calculate 10 MA, the period before those 10 trading days will be considered.

It can be calculated the same as the following:

First, we will calculate “K” to refer to exponent to calculate the weight of an interval, and “n” will refer to number of intervals:

![EMA](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.29.50_PM.png)

Then,

![EMA](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.30.10_PM.png)

Example:

If we have the same 5 trading days and the closing prices during these 5 days was as the following:

- Day1 = 10
- Day2 = 15
- Day3 = 20
- Day4 = 30
- Day5 = 35
- Day6 = 40

If we want to get EMA for these 5 Days, it will be calculated as the following:

![EMA example](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.30.28_PM.png)

![EMA example](https://c.mql5.com/2/44/Screen_Shot_2022-01-28_at_10.30.40_PM.png)

Note that here EMA of yesterday will be the first time to calculate EMA at Day6, so we will use SMA of 5 days (22).

Now, after identifying what is Moving Average and types of moving averages, we will go through the most interesting part at this article. We will talk about Moving Average strategies and how we can design algorithmic trading systems for them.

### Strategy1: One Moving Average Crossover

In this article, we will choose Simple Moving Average. However, you can use any desired Moving Average type in your code.

According to the strategy, prices and SMA will be checked at every tick:

- Price > SMA: signal will be to buy, and we need this signal to be appear as a comment on the chart.
- Price < SMA: signal will be to sell, and we need this signal to be appear as a comment on the chart.
- If anything else, do nothing.

The following screenshot shows One Simple Moving Average Blueprint that we want to design:

![1MA Blueprint](https://c.mql5.com/2/44/1MA_Blueprint.png)

The following is the code to design this strategy:

```
//+------------------------------------------------------------------+
//|                                            One SMA crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //create an array for price
   double myMovingAverageArray1[];

   //define Ask, Bid
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   //define the properties of  MAs - simple MA 24
   int movingAverage1 = iMA(_Symbol, _Period, 24, 0, MODE_SMA, PRICE_CLOSE);

   //sort the price arrays current candle
   ArraySetAsSeries(myMovingAverageArray1,true);

   //Defined MA - one line - currentcandle, 3 candles - store result
   CopyBuffer(movingAverage1,0,0,3,myMovingAverageArray1);

   //Check if we have a buy entry signal
   if (
      (Ask>myMovingAverageArray1[0])
      )
         {
         Comment("BUY");
         }

   //check if we have a sell entry signal
   if (
      (Bid<myMovingAverageArray1[0])
      )
         {
         Comment("SELL");
         }
  }
//+------------------------------------------------------------------+
```

The following screenshots show the generated signals after executing the code:

![1MA - buy signal](https://c.mql5.com/2/44/1MA_-_buy_signal.png)

![1MA - sell signal](https://c.mql5.com/2/44/1MA_-_sell_signal.png)

### Strategy2: Two Moving Averages Crossover

In this strategy, we will use two simple moving averages. The shorter simple moving average period is 24 and the longer one period is 50.

According to this strategy, we need the two simple moving averages to be checked at every tick:

- If 24 SMA > 50 SMA: signal will be to buy, and we need this signal to be appeared as a comment on the chart.
- If 24 SMA < 50 SMA: signal will be to sell, and we need this signal to be appeared as a comment on the chart.
- If anything, else do nothing.

The following screenshot shows the Two Simple Moving Averages Blueprint that we want to design:

![2MA Blueprint](https://c.mql5.com/2/44/2MA_Blueprint.png)

The following is the code to design this strategy:

```
//+------------------------------------------------------------------+
//|                                            Two SMA crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //create an array for several prices
   double myMovingAverageArray1[], myMovingAverageArray2[];

   //define the properties of  MAs - simple MA, 1st 24 / 2nd 50
   int movingAverage1 = iMA(_Symbol, _Period, 24, 0, MODE_SMA, PRICE_CLOSE);
   int movingAverage2 = iMA(_Symbol,_Period,50,0,MODE_SMA,PRICE_CLOSE);

   //sort the price arrays 1, 2 from current candle
   ArraySetAsSeries(myMovingAverageArray1,true);
   ArraySetAsSeries(myMovingAverageArray2,true);

   //Defined MA1, MA2 - one line - currentcandle, 3 candles - store result
   CopyBuffer(movingAverage1,0,0,3,myMovingAverageArray1);
   CopyBuffer(movingAverage2,0,0,3,myMovingAverageArray2);

   //Check if we have a buy entry signal
   if (
      (myMovingAverageArray1[0]>myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]<myMovingAverageArray2[1])
      )
         {
         Comment("BUY");
         }

   //check if we have a sell entry signal
   if (
      (myMovingAverageArray1[0]<myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]>myMovingAverageArray2[1])
      )
         {
         Comment("SELL");
         }
  }
//+------------------------------------------------------------------+
```

The following screenshots demonstrates the generated signals after executing the code:

### ![2MA - buy signal](https://c.mql5.com/2/44/2MA_-_buy_signal.png)          ![2MA - sell signal](https://c.mql5.com/2/44/2MA_-_sell_signal.png)

### Strategy3: Three Moving Averages Crossover

In this strategy, we will use three simple moving averages: the shorter simple moving average period is 10, the longer one period is 48, and in between a period of 24.

According to the strategy, we need the three simple moving averages to be checked at every tick:

- If 10 SMA > 24 SMA, 10 SMA > 48 SMA, and 24 SMA > 48 SMA: the signal will be to buy and we need to be appeared as a comment on the chart.
- If 10 SMA < 24 SMA, 10 SMA < 48 SMA, and 24 SMA < 48 SMA: the signal will be to sell and we need to be appeared as a comment on the chart.
- If anything, else do nothing.

The following screenshot demonstrates the Three Simple Moving Averages Blueprint that we want to design:

![3MA Blueprint](https://c.mql5.com/2/44/3MA_Blueprint.png)

The following is the code to design this strategy:

```
//+------------------------------------------------------------------+
//|                                          Three SMA crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //create an array for several prices
   double myMovingAverageArray1[], myMovingAverageArray2[],myMovingAverageArray3[];

   //define the properties of  MAs - simple MA, 1st 10 / 2nd 24, 3rd 48
   int movingAverage1 = iMA(_Symbol, _Period, 10, 0, MODE_SMA, PRICE_CLOSE);
   int movingAverage2 = iMA(_Symbol,_Period,24,0,MODE_SMA,PRICE_CLOSE);
   int movingAverage3 = iMA(_Symbol,_Period,48,0,MODE_SMA,PRICE_CLOSE);

   //sort the price arrays 1, 2, 3 from current candle
   ArraySetAsSeries(myMovingAverageArray1,true);
   ArraySetAsSeries(myMovingAverageArray2,true);
   ArraySetAsSeries(myMovingAverageArray3,true);

   //Defined MA1, MA2, MA3 - one line - currentcandle, 3 candles - store result
   CopyBuffer(movingAverage1,0,0,3,myMovingAverageArray1);
   CopyBuffer(movingAverage2,0,0,3,myMovingAverageArray2);
   CopyBuffer(movingAverage3,0,0,3,myMovingAverageArray3);

   //Check if we have a buy entry signal
   if (
      (myMovingAverageArray1[0]>myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]<myMovingAverageArray2[1])
   && (myMovingAverageArray1[0]>myMovingAverageArray3[0])
   && (myMovingAverageArray1[1]<myMovingAverageArray3[1])
   && (myMovingAverageArray2[0]>myMovingAverageArray3[0])
   && (myMovingAverageArray2[1]<myMovingAverageArray3[1])
      )
         {
         Comment("BUY");
         }

   //check if we have a sell entry signal
   if (
      (myMovingAverageArray1[0]<myMovingAverageArray2[0])
   && (myMovingAverageArray1[1]>myMovingAverageArray2[1])
   && (myMovingAverageArray1[0]<myMovingAverageArray3[0])
   && (myMovingAverageArray1[1]>myMovingAverageArray3[1])
   && (myMovingAverageArray2[0]<myMovingAverageArray3[0])
   && (myMovingAverageArray2[1]>myMovingAverageArray3[1])
      )
         {
         Comment("SELL");
         }
  }
//+------------------------------------------------------------------+
```

The following screenshots demonstrate the generated signals after executing the code:

### ![3MA - buy signal](https://c.mql5.com/2/44/3MA_-_buy_signal.png)          ![3MA - sell signal](https://c.mql5.com/2/44/3MA_-_sell_signal.png)

### Conclusion

In this article, I mentioned one of the most commonly used and important indicators by sharing three different strategies. I also tried to show how we can use these indicators and how we can design algorithmic trading systems based n each one of them. These strategies can be used accurately and effectively, of course after prior testing and optimizations.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3040.zip "Download all attachments in the single ZIP archive")

[One\_Moving\_Average\_System.mq5](https://www.mql5.com/en/articles/download/3040/one_moving_average_system.mq5 "Download One_Moving_Average_System.mq5")(1.74 KB)

[Two\_Moving\_Averages\_System.mq5](https://www.mql5.com/en/articles/download/3040/two_moving_averages_system.mq5 "Download Two_Moving_Averages_System.mq5")(1.92 KB)

[Three\_Moving\_Averages\_System.mq5](https://www.mql5.com/en/articles/download/3040/three_moving_averages_system.mq5 "Download Three_Moving_Averages_System.mq5")(2.6 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/388218)**
(3)


![Tobias Johannes Zimmer](https://c.mql5.com/avatar/2022/3/6233327A-D1E7.JPG)

**[Tobias Johannes Zimmer](https://www.mql5.com/en/users/pennyhunter)**
\|
1 Mar 2022 at 12:36

On one hand thank you for making this beginner series. On the other hand I am surprised that you haven't got any comments about why you are not supposed to initialise indicator handles inside OnTick yet.


![Gunnar Forsgren](https://c.mql5.com/avatar/avatar_na2.png)

**[Gunnar Forsgren](https://www.mql5.com/en/users/gunnarforsgren)**
\|
8 Mar 2024 at 19:06

Very easy to comprehend reasoning. I was at first worrying about some possible jitter around the point of crossover; some risk of the position moving back and forth between a [buy/sell signal](https://www.mql5.com/en/articles/522 "Article: MetaTrader 5 Added Trading Signals - Better Than PAMM Accounts! ") resulting in lots of short-lived loss trades. That some margin would be needed in the calculation for a hysteresis effect. But margins also affects reponsiveness. But I see different prices are used in the buy/sell cases (ask/bid) and this makes for some margin.

The Strategy Tester will be useful. By parameterizing uncertain behavior the Strategy tester can reveal unexpected findings that help improve on a strategy.


![Ronald Alarcon](https://c.mql5.com/avatar/avatar_na2.png)

**[Ronald Alarcon](https://www.mql5.com/en/users/ronaldalarconlepez)**
\|
4 Sep 2025 at 17:50

Excellent [analysis](https://www.mql5.com/en/articles/5638 "Article: MQL analysis using MQL ") for this article, very clear for those who are starting like me in this world, it will serve me a lot, thank you.


![Graphics in DoEasy library (Part 90): Standard graphical object events. Basic functionality](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__2.png)[Graphics in DoEasy library (Part 90): Standard graphical object events. Basic functionality](https://www.mql5.com/en/articles/10139)

In this article, I will implement the basic functionality for tracking standard graphical object events. I will start from a double click event on a graphical object.

![Combinatorics and probability for trading (Part IV): Bernoulli Logic](https://c.mql5.com/2/44/ieu9.png)[Combinatorics and probability for trading (Part IV): Bernoulli Logic](https://www.mql5.com/en/articles/10063)

In this article, I decided to highlight the well-known Bernoulli scheme and to show how it can be used to describe trading-related data arrays. All this will then be used to create a self-adapting trading system. We will also look for a more generic algorithm, a special case of which is the Bernoulli formula, and will find an application for it.

![Combinatorics and probability for trading (Part V): Curve analysis](https://c.mql5.com/2/43/bvmb.png)[Combinatorics and probability for trading (Part V): Curve analysis](https://www.mql5.com/en/articles/10071)

In this article, I decided to conduct a study related to the possibility of reducing multiple states to double-state systems. The main purpose of the article is to analyze and to come to useful conclusions that may help in the further development of scalable trading algorithms based on the probability theory. Of course, this topic involves mathematics. However, given the experience of previous articles, I see that generalized information is more useful than details.

![Advanced EA constructor for MetaTrader - botbrains.app](https://c.mql5.com/2/43/avatar.png)[Advanced EA constructor for MetaTrader - botbrains.app](https://www.mql5.com/en/articles/9998)

In this article, we demonstrate features of botbrains.app - a no-code platform for trading robots development. To create a trading robot you don't need to write any code - just drag and drop the necessary blocks onto the scheme, set their parameters, and establish connections between them.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=yazjdufeugdfuqvattxqtcwxtflkiouz&ssn=1769181307506411373&ssn_dr=0&ssn_sr=0&fv_date=1769181307&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F3040&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Learn%20how%20to%20design%20different%20Moving%20Average%20systems%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918130751590510&fz_uniq=5069279793066738299&sv=2552)

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