---
title: The All or Nothing Forex Strategy
url: https://www.mql5.com/en/articles/336
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:55:44.946498
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/336&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083218632684607279)

MetaTrader 5 / Trading systems


The purpose of this article is to create the most simple trading strategy that implements the "All or Nothing" gaming principle. That is an example of an Expert Advisor that implements the ForEx market lottery. The main goal of the lottery Expert Advisor is to increase the initial deposit several times with the highest possible probability. _Profitability_, i.e. increasing the deposit in average, is _not required_ from the lottery Expert Advisor. In contrast to conventional lottery, which is played out by selling thousands of tickets, the lottery Expert Advisor is playing the ForEx lottery, using ForEx as a source of money in case of winning.

### Introduction

The objectives of ForEx trading can be divided into three groups: **Earn**, **Save** and **Multiply**. Let's consider each group separately.

1. **EARN**. This is a standard goal in the ForEx market. It sounds like this: "I've got a capital and I want to increase it by trading on the market. Give me an Expert Advisor that will reliably make $101 out of $100 in a day." In goals of this type _**a guaranteed average increase of capital is required**_. The problem of "earning" is solved by an enormous amount of competitors. They are well equipped both with information and technical means. Therefore, this task is very difficult though not hopeless. Studies show that exchange rates are different from purely random. Searching for your own trading strategy is similar to searching for gold during the gold rush.

2. **SAVE**. It sounds like this: "I've got $1,000. I want to spend them on vacation next year. If I put them in the bank, then I will lose about 10 percent due to inflation. Give me an Expert Advisor that will save my money. I don't want to earn or lose money." In fact, this problem is to exchange one currency into another and back again at appropriate moments. In goals of the "SAVE" type _**a guaranteed average preservation of capital is required**_. "Preservation" of capital is solved by vast majority of our citizens. This is indicated by a large number of street currency exchange centers and multi-currency bank deposits. The task of "saving" is not mathematically sophisticated. Even if you choose the market entry and exit points accidentally, you can quite successfully win back the average inflation rate.

3. **MULTIPLY**. Goals of this type are formulated as follows: "Lottery. I have $100. To buy a car I need one million dollars more. Give me an Expert Advisor that will make $1,000,000 out of $100. I understand that on average I will lose money. The probability of winning a million is fewer than a 100/1,000,000. But it suits me."

Less aggressive formulation is as follows: "I have $1,000. To organize a party I need $10,000. Give me an Expert Advisor that will make $10,000 out of $1,000. I understand that I will lose $1,000 with a probability of slightly more than 0.9, but I will have a party with a probability of slightly less than 0.1."

Another task demanded: "In my electronic purse there are a few cents. With these money I can't neither buy anything nor withdraw cash. Give me an Expert Advisor that will even with a small probability but turn them into a meaningful amount."

In goals of this type _**a guaranteed average loss of capital is achieved**_. But this is acceptable for all. The probability of winning the lottery, of course, should be as high as possible. On the ForEx market very few are meeting this challenge consciously. However, judging by the number of lottery tickets in different offices, this problem is demanded in society. Mathematically, the "multiply" goal has long been solved. In this article we will consider the MQL5 implementation of this type of Expert Advisor.


Our classification does not include tasks such as "Game of chance - I want adrenaline", "Pretty smart toy - I want to play at my spare time" and many other delicious task.

Thus, the **problem definition**: we need to implement a lottery on the ForEx market. That is to increase the capital several times with some probability, or go bankrupt. The average increase of capital is not required. The probability of winning should be maximal if possible. ForEx is needed as a source of money in case of winning. ForEx is also used here as a random number generator that everyone has free access to.

### 1\. Algorithm Idea

Proposed is the following trivial algorithm to solve the problem.

1. Enter the market in a random direction.
2. Wait for a given time T.
3. Exit the market.
4. Check your account. If we have won the lottery or have gone bankrupt, then finish the trade, otherwise - go back to Step 1.

This algorithm assumes that the exchange rate is a pure random walk (see the [Random Walk and the Trend Indicator](https://www.mql5.com/en/articles/248 "Random Walk and the Trend Indicator") article). This market model is obviously wrong, but it is enough to create a lottery. Of course, the more adequate market models will provide a more efficient algorithm.

Before writing an Expert Advisor in [MQL5](https://www.mql5.com/en/docs "Automated Trading Language Documentation") programming language, we have to detail the algorithm. We need to solve the following questions:

1. Leverage
2. Bet size
3. Take Profit and Stop Loss levels
4. Time of waiting T
5. Selecting a currency pair

### 2\. Bet and Leverage

Since we're guessing with a probability of 50/50 and we need to pay for spreads, the number of deals should be as few as possible. On each trade we lose one spread. Therefore, the amount of leverage and the bet size should be maximized in order to get the result (to win the lottery or go bankrupt) with minimum number of trades.

Generally, the maximum volume of deal on one currency pair is limited by a broker. This limits the size of the winning and minimal time of the lottery.

Calculate how long the lottery time will be stretched out. Let's assume that the maximum volume of deal is 5 lots. This means that we have $500,000 at our disposal for trading. When the trade is "lucky enough" we can win $500,000 \* 0.02 = $10,000.

The 0.02 coefficient has the dimension of "profit dollars/dollar of capital in a day." This coefficient is an experimental constant for the ForEx market. It does not depend on the timeframe we are trading on (excluding spreads and swaps) and the currency pair. It can be measured based on the relative average size of bar, and knowing the drunken sailor theorem (see the the Choosing Currency Pair section and Maximal Yield Indicator graph below). The numerical value of this coefficient is approximate (may differ 2-3 times).

If we trade 100 days, the daily profit of $10,000 should be multiplied not by 100, but by the square root of 100, which is 10, so we are trading using a random walk. And in 100 days of "quite lucky" trade, we will win $100,000. And in 400 days of "quite lucky" trade, we will win $200,000. If leverage was 1:100, this means that the initial deposit was no less than $5,000 ($500,000/100).

All in all, in 100 days we have increased the initial deposit 20 times, and in 400 days - 40 times. Unfortunately, with this maximal volume of deal and initial deposit we won't get the higher speed of increasing our deposit.

If the initial deposit is small and is not enough for the maximal volume of bet, the growth speed can be much higher, up to exponential rate. But we must still find a broker who works with small deposits, and see its trading conditions.

To outwit the restriction of maximal volume, you can try to play on several currency pairs. If currency pairs are independent, we get the average and the growth speed will be less than on one currency pair. If currency pairs are correlated, such as EURUSD and EURCHF, it is possible that this restriction will be outwitted. However, the correlation of rates is not always observed.

Thus, we still can create a lottery to multiply a sufficiently big initial capital by 10. We can not solve task about electronic purse and about car for $100. At the very least, [MetaTrader 5 Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "Strategy Tester in the MetaTrader 5 Trading Platform") won't allow us to do it.

### 3\. Selecting Take Profit and Stop Loss

Take Profits and Stop Losses based on [random walk](https://www.mql5.com/en/articles/248 "Random Walk and the Trend Indicator") only increase the frequency of trades. The closer Take Profit and Stop Loss are to the opening price, the more frequently they trigger, and the greater is frequency of trades. Take Profit and Stop Loss does not directly affect the probability of winning using random walk. Because we want to make trades as few as possible, we do not place these orders.

In reality, Stop Loss still exists - this is Stop Out. Usually it triggers at 50 percent and the trading is forcibly finished. Because there are still some money on the deposit after Stop Out, it means that we have not used all the chances and could continue trading. Therefore, the Expert Advisor has to warn about the Stop Out situation. Deposit must be fully exhausted up to the minimum bet, and ideally - up to zero.

It makes sense to place Take Profit. The idea is that the real exchange rate - is not a random walk. Sometimes it has abnormally large jumps, for example, after the news. Abnormal jumps more likely have the form of spire, not the stair. You can play on this.

![Figure 1. Sharp spike on the EURUSD, M1](https://c.mql5.com/2/3/shpil.GIF)

Figure 1. Sharp spike on the EURUSD, M1

Our algorithm can simply miss such jumps. If the spike does not happen in the direction of our last deal, and we miss it, that's good. Although the Stop Out may trigger. But if the spike happen in our direction, we grudge loosing it. To spot it, we place the Take Profit.

Take Profit can be placed in several ways: the **direct, second and third** ways.

**Direct Way** \- keep track of the current price and compare it with the history prices. This is a very difficult way, even on an algorithmic level. In fact, it is necessary to identify and cut off the fat tails of the distribution of price changes.

**Second Way** \- keep track of your current profit and compare it with the profits of previous deals. Once the profit is much larger than the average profit of previous trades - take your profit. This way is easier, but when Expert Advisor starts there is simply no history of ideals. In addition, we want to make trades as few as possible, which means that the history will be short.

**Second Way (another variant)** \- keep track of your current profit and compare it with the expected calculated profit. Once the profit is larger than the expected calculated profit - take your profit. The expected profit can be calculated based on the price history, but it is as difficult as with the direct way.

**Third Way** \- keep track the balance/equity ratio and compare it with the constants. Constants are determined in advance during Expert Advisor optimization. These constants of course will depend on trade conditions - leverage, maximal volume of transaction, etc. And Expert Advisor will be optimized for some specific trade conditions, that are about the same for all the brokers. Let's take the typical ones. And most importantly - this way is as simple as possible:

- If equity is greater than balance 2 (or 3) times - take your profit.
- If equity is greater than balance for $10,000 (or $30,000) - take your profit.

The specific numbers 2, 3, 10000, 30000 ... will be determined after optimization.

### 4\. Time of Waiting T

If we are to enter and exit the market very often (for example, every minute), the rate will change little and profit we will be too little, but we still have to pay the fixed spread. The total spread will override all the profits, even if we will guess pretty well.

On the other hand, if you make deals very seldom (for example, once a year or once a month), then the spread will be negligible compared to the profit per trade. But you will have to trade for a very long time. Also, keeping a position opened for a long time is unprofitable due to swaps.

Therefore, there is some optimal frequency of deals. It depends on volatility of exchange rate and trade conditions, such as floating spread. Therefore, it is impossible to accurately calculate the optimal frequency of deals in advance.

However, it can be estimated. Without going into mathematical explanations and surveying, I will provide the following graph:

![Figure 2. The boundaries and the center of the equity probability distribution function, when trading for one day with different times of T](https://c.mql5.com/2/3/Figure1_Profit_Distribution.png)

Figure 2. The boundaries and the center of the equity probability distribution function, when trading for one day with different times of T

On the graph in Figure 2 the abscissa axis shows time T - time of one trade of our trivial algorithm. The ordinates axis shows how many dollars of profit we would have from one dollar of capital by trading one day every T minutes, with no leverage and capitalization of profit on the EURUSD pair. Mathematically speaking, these are the boundaries of the probability distribution of our trivial strategy for different times T. The blue curve - for absolute guessing, red - for absolute not guessing, orange and teal - for "quite successful/unsuccessful guessing".

For example, entering and exiting the market every minute (M1), with $1 dollar of capital per day we could win maximum $0.5 and lose maximum $1.3. Most likely we would have lost $0.3. During the day we could make 1440 trades , paying $0.0002 of spread per trade. The total spread for all trades per day will be $0.288. The average size of EURUSD M1 bar is $0.00056. Winning with absolute guessing is $0.00056 \* 1440 = $0.8064. Subtract the spread from of winning: $0.8064 - $0.288 = $0.51 profit from one dollar per day. Place point (М1, 0.51) on the graph.

We are interested in "fairly lucky" guess - the orange curve. Let's draw it in a bigger scale:

![Figure 3. Profit of trivial trading strategy with sufficiently successful guessing at different times of T](https://c.mql5.com/2/3/Figure3_Probability_Distribution_Orange.png)

Figure 3. Profit of trivial trading strategy with sufficiently successful guessing at different times of T

Looking at Figure 3, we see that it is not profitable to trade more frequently than every 30 minutes - the spread devours all the profit. The optimal time T of trade for us lies in the range 1 hour - 1 week. Let's stop on it for now. Later on, when our EA will be finished, we will specify the optimal time using optimization. If someone has trading ideas of ​​predicting the rate better than 50/50, then the trivial algorithm can be improved. The optimal time and optimal bet will decrease as well.

By selecting the time T, we have actually chosen the timeframe of chart we will be working on. Strictly speaking, when the time T is given, you can can choose any timeframe - EA will work the same way, but drawing on the wrong timeframe will be uncomfortable.

### 5\. Selecting Currency Pair

Since we consider rates of all the currency pairs as random walk, among all the rates we need to choose one with the largest average relative size of bar. Then with smaller number of trades we will achieve the result (win the lottery or go bankrupt).

To do this we need to go through all the available currency pair rates and for each one of them calculate the average relative size of bar. In order not to do this manually, we will write the Maximal Yield Indicator - YieldClose.mq5.

![Figure 4. Maximal Yield Indicator](https://c.mql5.com/2/3/Figure4_en.png)

Figure 4. Maximal Yield Indicator - YieldClose.mq5 (EURUSD, D1. averaging by 10 bars. The indicator oscillates in the range of 2-3 times)

After writing this article, I have accidentally discovered that the volatility indicator (Kaufman Volatility from the [Smarter Trading: Improving Performance in Changing Markets](https://www.mql5.com/go?link=https://www.amazon.com/Smarter-Trading-Improving-Performance-Changing/dp/0070340021 "http://www.amazon.com/Smarter-Trading-Improving-Performance-Changing/dp/0070340021") book by Perry Kaufman) included into standard delivery of [MetaTrader 5](https://www.metatrader5.com/ "The MetaTrader 5 Trading Platform") client terminal is virtually the same as the Maximal Yield Indicator. When the scope of intellect is not enough, you have to reinvent the wheel. Yes, it's hard to comprehend hundreds of indicators and Expert Advisors from the standard set! Unfortunately, there is no general textbook at this point.

It turns out that the average relative size of bar oscillates in range of 2-3 times for a single currency pair. Within these 2-3 times the average relative size of bar is the same for all the currencies. In fact, the Maximal Yield Indicator shows trading activity.

When entering the market, among all the currency pairs we need to choose one with highest trading activity, i.e. one with maximal values shown by indicator. In addition, it is better to trade during the day, when the activity is higher, and to wait out the night. Purely day trading will increase your chances of winning the lottery, but will stretch the working time of Expert Advisor almost two times. Which is better - greater chances of winning or shorter time - is up to user to decide.

As discussed above, you can trade virtually every minute, but in so doing the chances of winning will become very scanty. On the other hand, we also can't trade for years in order to maximize the probability of winning. The "time of trade/probability of winning" ratio has to be detailed at time of problem definition, but who knew that it could be so hard?

This is a typical example of how difficult is to compose requirements specification for writing an Expert Advisor. So far, in order not to complicate our EA let's focus on continuous single-currency trading on the well-known EURUSD pair.

At the same time note two interesting properties of the indicator.

1. On the H1 timeframe indicator shows daily oscillations of trading activity (volatility) (see Figure 5).
2. The indicator maxima correspond to the end/beginning of trends or flat (see Figure 6).

![Figure 5. Daily oscillations of activity](https://c.mql5.com/2/3/Figure5_en.png)

Figure 5. Maximal Yield Indicator shows daily oscillations of trading activity (EURCHF, H1, averaging by 10 bars)

![The Maximal Yield Indicator maxima correspond to the beginning/end of trend/flat](https://c.mql5.com/2/3/Figure6_en.png)

Figure 6. The Maximal Yield Indicator maxima correspond to the beginning/end of trend/flat (USDCAD, M5, averaging by 10 bars)

An idea for future use: if indicator (market activity) begins to increase above its average value, then close profitable positions and leave the unprofitable ones - the market is changing. If indicator falls below its average, then leave profitable positions and close the unprofitable ones - in nearest future market won't change. But this idea requires a separate study.

Implementation of well developed algorithm in the MQL5 language requires technical skills. You can find the EA code with comments in attachment to this article (lottery.mq5).

### 6\. Expert Advisor Optimization

EA has to be optimized for specific trading conditions available in Strategy Tester: initial deposit - $5,000, leverage - 1:100, date - 1 year, lottery winning - $100,000, maximal bet - 5 lots, currency pair - EURUSD, Stop Out level - 50%.

Optimization of Expert Advisors, proposed in the MetaTrader 5 client terminal, does not suit us. Indeed, during optimization we need to maximize the probability of winning the lottery. To do this, we must run EA on 1000 different pieces of history and calculate the winnings/losses ratio. Running EA on a single piece of history makes no sense: it will give us either winning or loss, the state of balance is known in advance - either $0 or $100,000.

Running EA manually on 1000 pieces of history is boring, so we will go another way. To determine direction of entering the market, our EA uses the random number generator that creates a random buy/sell sequence. Let's run EA with 1000 different buy/sell sequences on one piece of history. Of course, this is not the same as 1000 different pieces of history, but very similar.

In order to optimize some parameter, such as the time T, for each value of T we are running 1000 different buy/sell sequences and determine the probability of winning. To do this, select the [Slow complete algorithm](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types "Optimization Types") of optimization by two parameters: the time T and the number of lucky ticket, i.e. number of random sequence.

Export the optimization results to Excel and draw the graph:

![Figure 7. Probability of winning the lottery, depending on the time T. The abscissa axis - trivial strategy timeout, i.e. time of one trade. The ordinate axis - probability of winning with such time T.](https://c.mql5.com/2/3/Figure7_en.png)

Figure 7. Probability of winning the lottery, depending on the time T. The abscissa axis - trivial strategy timeout, i.e. time of one trade. The ordinate axis - probability of winning with such time T.

Looking at Figure 7, we determine the optimal time T. The maximum probability of winning corresponds to the approximate time T = 350 000 seconds. The graph is similar to the theoretical estimation above on the Figure 3 - with small values ​​of T the probability of winning is virtually zero. The form of the graph depends on the history period and length. The graph always falls down to big values of time about 500 000 seconds.

To determine the optimal values ​​of Take Profit we observe the graph of balance and equity, trying Take Profit to trigger only on enormously large emissions of equity. Optimization of the Take Profit constants by the maximal balance makes no sense: big emissions happen very seldom, maybe once per all the time of EA functioning, and even more rarely. If we run optimization by the maximal balance, it will simply adjust to this given piece of history.

### 7\. Checking Expert Advisor

To determine the EA quality, we will run it with 10 000 different buy/sell sequences. Open the table with optimization results in Excel, then calculate and draw the winning/loosing ratio.

From the results of measurements, our Expert Advisor wins the lottery (gaining more than $100,000) with probability of 0.045 and the theoretical limit of 0.05. Expert Advisor loses the lottery (gains less than $150) with probability of 0.88. The remaining probability of 0.075 corresponds to the balance values ​​between $150 and $100,000. With probability of 0.1 Expert Advisor gains more equity than the initial deposit of $5000.

![Figure 8. Probability of winning and losing depending on the number of trades](https://c.mql5.com/2/3/Figure8_en.png)

Figure 8. Lottery time. The abscissa axis - number of trades. The ordinate axis - probability of win the lottery for a given number of trades.

Figure 8 shows the curves that demonstrate the probability of winning and losing, depending on the number of trades. The blue curve - the number of trades in the general case, the red curve - the number of trades in case of winning. In general, the lottery ends up losing in 20 trades (2 months, 1 trade = 350 000 seconds). Lottery may take up to six months or more (60-70 trades). Winnings are the most likely during 3-5 month of lottery (30-50 trades, red curve).

### Conclusion

We have created the lottery Expert Advisor optimized for specific trading conditions. EA is written in the simplest way of all the possible options. Pros and cons of the lottery Expert Advisor are obvious.

Pros:

- You can play the lottery alone. No need to sell millions of tickets.
- You can select the ratio of ticket price (initial deposit) and winning.
- The probability of winning is known in advance and is close to the theoretical limit.
- Win results can be checked for integrity on ForEx history available for free.

Cons:

- Very long time of the lottery - a few months. Time is limited by trading conditions.
- Possible "ticket price"/"winning" ratio is little - about 1:10.
- Big initial deposit is required.

Despite the best efforts of developers, the implementation of even a trivial algorithm requires a non-trivial ingenuity, knowledge of mathematics and MQL5 language. But, thanks to the efforts of developers, the implementation is still possible.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/336](https://www.mql5.com/ru/articles/336)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/336.zip "Download all attachments in the single ZIP archive")

[lottery.mq5](https://www.mql5.com/en/articles/download/336/lottery.mq5 "Download lottery.mq5")(8.99 KB)

[yieldclose.mq5](https://www.mql5.com/en/articles/download/336/yieldclose.mq5 "Download yieldclose.mq5")(3.01 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Money-Making Algorithms Employing Trailing Stop](https://www.mql5.com/en/articles/442)
- [Random Walk and the Trend Indicator](https://www.mql5.com/en/articles/248)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6189)**
(17)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
26 Mar 2016 at 05:16

Brilliant article.


![Roberto Jacobs](https://c.mql5.com/avatar/2015/6/55706E50-4CAE.jpg)

**[Roberto Jacobs](https://www.mql5.com/en/users/3rjfx)**
\|
26 Mar 2016 at 05:30

Thanks a lot for interesting article ![](https://c.mql5.com/3/92/b0212-smile.gif)

![Shengjie Lv](https://c.mql5.com/avatar/2017/2/58A2BDA4-4CCB.jpg)

**[Shengjie Lv](https://www.mql5.com/en/users/285858315)**
\|
18 Feb 2017 at 11:09

How come the [backtest](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") is not billed?


![qianxun002](https://c.mql5.com/avatar/avatar_na2.png)

**[qianxun002](https://www.mql5.com/en/users/qianxun002)**
\|
11 Apr 2019 at 17:46

That suits a lot of people.


![Edwin Luk](https://c.mql5.com/avatar/2020/7/5F1EA1EF-9284.jpg)

**[Edwin Luk](https://www.mql5.com/en/users/edwinluk)**
\|
13 Jan 2021 at 10:55

Amazing discussions. I, particularly, love your discussions on different timeframes.

Спасибо большое

![Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://c.mql5.com/2/0/MQL5_protection_methods.png)[Securing MQL5 code: Password Protection, Key Generators, Time-limits, Remote Licenses and Advanced EA License Key Encryption Techniques](https://www.mql5.com/en/articles/359)

Most developers need to have their code secured. This article will present a few different ways to protect MQL5 software - it presents methods to provide licensing capabilities to MQL5 Scripts, Expert Advisors and Indicators. It covers password protection, key generators, account license, time-limit evaluation and remote protection using MQL5-RPC calls.

![Using Discriminant Analysis to Develop Trading Systems](https://c.mql5.com/2/0/Discriminant_Analysis_MQL5.png)[Using Discriminant Analysis to Develop Trading Systems](https://www.mql5.com/en/articles/335)

When developing a trading system, there usually arises a problem of selecting the best combination of indicators and their signals. Discriminant analysis is one of the methods to find such combinations. The article gives an example of developing an EA for market data collection and illustrates the use of the discriminant analysis for building prognostic models for the FOREX market in Statistica software.

![Promote Your Development Projects Using EX5 Libraries](https://c.mql5.com/2/0/Use_ex5_libraries.png)[Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)

Hiding of the implementation details of classes/functions in an .ex5 file will enable you to share your know-how algorithms with other developers, set up common projects and promote them in the Web. And while the MetaQuotes team spares no effort to bring about the possibility of direct inheritance of ex5 library classes, we are going to implement it right now.

![Speed Up Calculations with the MQL5 Cloud Network](https://c.mql5.com/2/0/speed_network.png)[Speed Up Calculations with the MQL5 Cloud Network](https://www.mql5.com/en/articles/341)

How many cores do you have on your home computer? How many computers can you use to optimize a trading strategy? We show here how to use the MQL5 Cloud Network to accelerate calculations by receiving the computing power across the globe with the click of a mouse. The phrase "Time is money" becomes even more topical with each passing year, and we cannot afford to wait for important computations for tens of hours or even days.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dpzhlakupmbmzojyvmzbiicyzyfopkfx&ssn=1769252144187732974&ssn_dr=0&ssn_sr=0&fv_date=1769252144&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F336&back_ref=https%3A%2F%2Fwww.google.com%2F&title=The%20All%20or%20Nothing%20Forex%20Strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925214412850419&fz_uniq=5083218632684607279&sv=2552)

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