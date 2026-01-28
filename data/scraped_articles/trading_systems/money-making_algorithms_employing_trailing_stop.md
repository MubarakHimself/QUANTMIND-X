---
title: Money-Making Algorithms Employing Trailing Stop
url: https://www.mql5.com/en/articles/442
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:00:57.886698
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/442&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071507644142791000)

MetaTrader 5 / Trading systems


### Introduction

One of the algorithms featuring a random entry and timed exit was already reviewed in the article [The "All or Nothing" Forex Strategy](https://www.mql5.com/en/articles/336). That algorithm did not take into account market events and the direction in which the market should be entered. Only market volatility that is more or less constant was of importance to the algorithm. Basically, the algorithm did not yield any profit or loss but it proved very useful when playing the lottery.

The main point we currently need to take from that article is that the duration of one trade could not be set to less than an hour or more than a week. The values of less than an hour resulted in a very fast loss due to spread while the values of more than a week dragged out the lottery for years.

Let us begin our study of the algorithms with the familiar EURUSD currency pair and an exit using trailing stop.

### 1\. Algorithm Featuring a Random Entry and Exit Using Trailing Stop

1. Enter the market in a random direction;
2. Set the trailing stop equal to TS;
3. Wait until the trailing stop is triggered;
4. Go back to point 1 or stop the trade.

You can expect to get some profit using this algorithm as the market situation is monitored by the trailing stop. Clearly, if the price movement was a random walk, this algorithm would not gain anything. But the real price movement is far from being chaotic so there is hope for some profit.

Instead of guessing how the algorithm will perform, let us develop an EA and test it. A lot has already been said about the ways of setting a trailing stop in a program. We are not going to move from algorithmic level down to programming in this article, otherwise we will simply not get to the end. Since the EA is going to be developed for study purposes, we take the maximum deposit of USD 100.000 and a minimum lot of 0.1. This will allow us to see more action before a stop out.

Let us here agree that the trailing stop will be equal to TS and all other stops in this article will be expressed as a percentage of the average body size of the last five candlesticks (high-low) in the current chart. The time frame we are going to use to display the current chart is D1. You can take any other number of candlesticks, not necessarily five, as it is not going to have any significant effect on the reasoning. It is important that having chosen this measuring scale, we do not depend on current volatility, or on the chosen currency, or currency pair.

For a test run, set TS=100% on D1. Having set such TS value, the duration of one trade will be around one day. As already mentioned above, a shorter trade duration and a smaller TS value cannot be set because of quick loss due to spread. Setting greater values will drag out the algorithm runtime.

![Fig. 1. Balance obtained by the algorithm featuring a random entry and exit using trailing stop, where TS=100](https://c.mql5.com/2/4/EN_EURUSD100__1.png)

Fig. 1. Balance obtained by the algorithm featuring a random entry and exit using trailing stop, where TS=100

Figure 1 suggests that the algorithm brings profit and now that the EA has been developed, we could finish the article here, as is usually the case.

However, after reading articles of this type, there are three questions that leave you feeling deeply unsatisfied:

1. Is the demonstrated profit a result of fitting the parameters to historical data?
2. Why was the EURUSD currency pair chosen? What will happen if other currency pairs are used?
3. Why was this very part of historical data selected? Besides, the algorithm has a random entry, and the profit yielded could well be of a merely random nature.

We are going to answer these questions one by one.

The algorithm featuring a random entry and exit using trailing stop has only one parameter TS which was selected based on purely general considerations - loss due to spread and trade duration. The algorithm should nevertheless be optimized.

For optimization, we are going to use almost all available historical data from 1990 through 2012. Since the algorithm has a random entry, we will take 100 different random sequences of entries for every TS value. Thus, we will eliminate randomness in the algorithm and avoid fitting to historical data.

![Fig. 2. Optimization of the trailing stop TS value for EURUSD, D1 (optimal TS=500)](https://c.mql5.com/2/4/optimisation.png)

Fig. 2. Optimization of the trailing stop TS value for EURUSD, D1 (optimal TS=500)

Optimization was performed in the Tester's "Opening prices only" mode which explains loose calculations, especially for smaller TS values that nevertheless render the general idea correctly.

As can be seen in Fig. 2, the optimization does not give a clear maximum. Smaller trailing stop, where TS is equal to 50 and 100, leads to losses. Further, where TS lies between 150 and 850, the algorithm, on average, yields profit. Where TS is within the range of 900 to 1500, the algorithm starts losing again.

TS values of more than 1500 should not be considered. Where TS=1500, the algorithm executes around 25 trades over 22 years which is on the verge of reasonable. Since we have not identified a clear maximum, we are going to take the center of the profitable range of 150 to 850, i.e. TS=500 (130 trades over 22 years).

Let us now consider the balance obtained by the algorithm for different currency pairs. Again, to avoid fitting to historical data, we are not going to consider the balance of a single pass but rather the average balance of 100 passes with different random entries.

![Fig 3. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for EURUSD, averaged over 100 random entries](https://c.mql5.com/2/4/EURUSD.PNG)

Fig 3. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for EURUSD, averaged over 100 random entries

![Fig 4. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for GBPUSD, averaged over 100 random entries](https://c.mql5.com/2/4/GBPUSD.PNG)

Fig 4. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for GBPUSD, averaged over 100 random entries

![Fig 5. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for USDJPY, averaged over 100 random entries](https://c.mql5.com/2/4/USDJPY__1.PNG)

Fig 5. Balance obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for USDJPY, averaged over 100 random entries

The balances obtained by the algorithm featuring a random entry and exit using trailing stop TS=500 for EURUSD, GBPUSD and USDJPY, averaged over 100 random entries, are shown in Figures 3-5. The averaged balances are displayed along the Y-axes, time is displayed along the X-axes. Let us have a closer look at the averaged balances.

The first thing that can be pointed out for all currency pairs is sharp vertical upturns and long flat downturns. I suggest calling algorithms that result in balances of this type "pseudo-losing" algorithms. Indeed, if we take a random 3-month period, it will most likely not contain a sharp upturn and all we will see will be a graceful, steady, consistent loss. That said, this loss will be much faster than a regular loss due to spread. At the same time, we can see that the pseudo-losing algorithm can be both winning and losing over a period of 20 years.

Second. The number of vertical upturns varies for different currency pairs. E.g. there are not so many upturns in the GBPUSD chart as in the USDJPY chart. In addition, the upturns are not at all chaotic in nature relative to time. A significant upward movement of 2009 can be observed for all three currency pairs: EURUSD, GBPUSD and USDJPY. The currency charts suggest that the upward movement of 2009 is a result of the crisis of December 2008. The upturns are therefore indications of crises. From this moment on, I am going to refer to all significant upward movements as crises.

The crises shown in the averaged balance charts can reflect real historical events but can as well happen on their own, i.e. can be false. Such crises could also be observed in an ideally chaotic random walk, however all of them would be false. Crises would never be in sync with each other in chaotic random walks. The number of crises identified on the averaged balance curve determines whether the real price movement is of crisis or crisis-free nature.

Third. Having examined the balances and the algorithm performance using different currencies, we gradually get to understand the operation of the algorithm. While the market is quiet, the price movement is similar to a chaotic random walk and the algorithm, on average, loses due to spread. Periods of consistent loss alternate with false crises - upward movements - resulting, on average, in loss due to spread.

Since the trades in the algorithm are of a very long duration, where the trade duration may be about two months, the losses are very slow and can be neglected. Upon approaching a real crisis, the price movement stops being chaotic. It gets predictable and trend-driven, i.e. prone to development of trends. Further development of the crisis may lead to avalanche-like price movement being totally different from a chaotic random walk.

The algorithm behaves in crisis as follows: having guessed the trend direction, the algorithm waits for the crisis peak drawing the trailing stop closer to it which then triggers once the peak is reached. If the algorithm made a wrong guess regarding the trend direction, the trailing stop kicks in at the beginning of the trend and the algorithm reverses the position with a 50/50 probability. Thus, the algorithm successfully handles about three quarters of all real crises which is basically how it makes money.

As can be seen, to make profit, the algorithm needs crisis trend movements. It fully relates to the EURUSD price movement out of the currency pairs shown above. Whereas the GBPUSD price movement is trendless and crisis-free. Why is there such a difference in price behavior? All I know from fundamental analysis is that trade wars always break out between Europe and the United States while the UK and the U.S. have a very friendly relationship. We can see what this friendship is worth in the averaged balance charts plotted using the algorithm.

The price behavior can also be determined by financial regulators. The crisis of 2008 was suppressed by financial injections reflected by spikes in the averaged balance charts of almost all currency pairs. We are well aware of the 2008 crisis from the news reports back then. The United States have undertaken two rounds of quantitative easing since then. Where, when, in what direction and how the money was injected was for some reason, as usual, not reported.

The averaged balance for EURUSD suggests crises in 2010 and mid 2010. Can those be the rounds of quantitative easing? Regulators try to intervene quietly without informing the market. It is not always easy to identify a crisis in the price chart with the naked eye - price highs or lows do not always indicate crises. The averaged balance of the price movement serves as an indicator helping to reveal crises. Which of the displayed crises are false and which are true is a separate complicated subject.

Let us say a few words about the stability of price movements. For example, the GBPUSD balance is steadily going down for 22 years while the EURUSD balance shows a fairly steady growth. The USDJPY movement has very little stability in it. The stability of the price movement is very important as it is the only guarantee of the future algorithm profitability. There can be no other guarantee of algorithm profitability in technical analysis.

So we have figured out how and what price movements are used by the algorithm to bring profit. It is high time to look into profit. For study purposes, the lot size used above was 0.1. This lot size is not optimal in terms of profit. Let us calculate the optimal lot size. The profit grows in proportion to the lot size. The risk of a stop out is also proportional to the size of the lot. For the EURUSD balance, the maximal drawdown of USD 200 was observed in 1991.

If the lot size was 25 instead of 0.1, the drawdown could reach USD 50.000, or 50%. In other words, if the lot size was 25, we would definitely not be able to avoid the stop out. Thus, the optimal lot size lies in the range of 0.1 and 25. Feel free to make a more accurate calculation of the lot size, if you want; I would simply take the average of 0.1 and 25 and get a rough 10. So the optimal lot size is 10.

The algorithm made USD 1400 using the lot size of 0.1 (see the EURUSD balance). If the lot was 10, the algorithm would bring profit of USD 140.000. The deposit was USD 100.000. Consequently, over 22 years, our profit would be 140%, or around 6% per annum. From a down-to-earth point of view, it is not much but more than an interest rate offered by many banks on deposits held in foreign currency.

**Algorithm Variations**

The above discussed algorithm featured a random entry and exit using trailing stop. A random entry was necessary to obtain the averaged balances free of fitting to historical data. It was of great help when trying to analyze and understand of how the trailing stop works. But the algorithm was not at its best dealing with a crisis and trend-driven nature of price movements.

As we could see above, the algorithm could successfully handle only three quarters of crises. Development of an algorithm that would optimally handle the trend-driven behavior of prices is one of the general tasks of technical analysis which we are not going to tackle this time. For now, we are just going to try to improve the algorithm with a random entry.

One of the simplest ideas is an algorithm with a reverse entry. We could see above that upon unsuccessful entry into the trend and triggering the trailing stop, the algorithm could guess the trend direction with a 50/50 probability. Let us stop guessing and enter right in the direction opposite to the previous trade.

### 2\. Algorithm Featuring a Reverse Entry and Exit Using Trailing Stop

1. Enter the market in the direction opposite to the previous trade. First, we enter, say, in the buy direction.
2. Set the trailing stop equal to TS
3. Wait until the trailing stop is triggered
4. Go back to point 1 or stop the trade.

Developing an EA based on the algorithm is a routine task. Without getting preoccupied with the optimization of the TS value, we take the previously identified optimal value of 500. (If you choose to perform the optimization, you will still get TS=500). The deposit is, as before, USD 100.000, the lot size is 0.1 and the time frame is D1.

![Fig. 6. Balance obtained by the algorithm with a reverse entry (first entry - buy)](https://c.mql5.com/2/4/EN_TSreverse-buy.png)

Fig. 6. Balance obtained by the algorithm with a reverse entry (first entry - buy)

![Fig. 6. Balance obtained by the algorithm with a reverse entry (first entry - sell)](https://c.mql5.com/2/4/EN_TSreverse-sell.png)

Fig. 6. Balance obtained by the algorithm with a reverse entry (first entry - sell)

Figures 5-6 show the balances obtained by the algorithm with a reverse entry. We can see that the direction of the first entry only matters till the first crisis. After the first crisis, the balances get parallel.

As before, this is a pseudo-losing algorithm that makes money on crises, especially on the crisis of 2008. The drawdown values shown by the algorithms with a random and reverse entry are about the same, however the profit yielded using the reverse entry algorithm is USD 9.000 as opposed to USD 1400 before. And consequently, the profitability is no longer 6% but instead 6\*9000/1400=38% per annum. And 38% per annum from a down-to-earth point of view is not bad indeed.

The algorithm featuring an exit using trailing stop can further be improved in different directions. You can use different entries to predict the trend direction or make use of periodic nature of crises or enable/disable the algorithm based on fundamental analysis. And many other things. I will leave it as a special treat for the enthusiasts.

The most difficult part of these algorithms is not the development or even correct optimization of an EA, but rather the obtaining of a long-term stable price behavior (meaning the averaged balance) from the algorithmic standpoint. And the stability of the price will be generously repaid in profit. The obtaining of the price behavior in terms of a certain algorithm will require modification of algorithm parameters which is another separate complicated procedure.

Finally, I cannot but take advantage of the steady losing behavior of the GBPUSD prices shown in the Figure above. The most primitive thing we can do is to trade in the direction opposite to the trades of the algorithm featuring a random entry and exit using trailing stop. However, it will not be so great. A better solution would be to use trailing take or trailing profit - I am not sure which is the best way to call it.

Trailing take is, in essence, very similar to trailing stop but instead of the stop loss level the algorithm constantly trails the take profit level. If the current price moved away from the take profit level by a value greater than TP, the take profit level is moved towards the price. The stop loss level remains unset, i.e. at the stop out level.

To avoid repeating the random entry algorithm, I will immediately set forth the algorithm featuring a reverse entry and exit using trailing take.

### 3\. Algorithm Featuring a Reverse Entry and Exit Using Trailing Take

1. Enter in the direction opposite to the previous trade;
2. Set the trailing take equal to TP;
3. Wait until the trailing take is triggered;
4. Go back to point 1.

We are working with GBPUSD, D1, the deposit is USD 100.000, the lot size for study purposes is 0.1, TP=500.

![Fig. 7. Balance obtained by the algorithm featuring a reverse entry and exit using trailing take. GBPUSD, D1](https://c.mql5.com/2/4/EN_TP.png)

Fig. 7. Balance obtained by the algorithm featuring a reverse entry and exit using trailing take. GBPUSD, D1

The algorithm works on crisis-free, trendless price movements.

The algorithm operating mechanism is as follows: A trendless price movement tends to break any trend and transform into a horizontal channel. If the take profit is triggered in the horizontal channel, it means that we are already next to the channel wall and we should enter in the direction opposite to the previous trade which is exactly what we do.

The Figure suggests that the drawdown values demonstrated by this algorithm are a bit smaller than those of the previous ones, while the profit yielded over 19 years using the minimum lot size of 0.1 adopted for study purposes is USD 7.000. The profitability achieved using the optimal lot can be roughly estimated at 30% per annum.

I think I will, in the usual way, finish this article with this balance chart, quite vigorously moving upward. Hopefully, the currency pair and the historical data chosen, as well as the algorithm optimization provided in this article do not leave you feeling deeply unsatisfied.

### Conclusion

The article has considered three algorithms featuring random and reverse entries into trade and exits using trailing stop. It has demonstrated the EURUSD, USDJPY and GBPUSD price behavior in terms of the algorithm featuring a random entry and exit using trailing stop.

Based on the exhibited stability of price movements, it has been proposed to use the algorithm featuring a reverse entry and exit using trailing stop as profitable with estimated profitability of 6% per annum. According to stability of the price movements and understanding of the operation of the random entry algorithm, two reverse entry algorithms have been proposed that can achieve profitability of 30% per annum. The algorithm operating mechanisms have been considered and the codes of the relevant EAs provided.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/442](https://www.mql5.com/ru/articles/442)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/442.zip "Download all attachments in the single ZIP archive")

[trailingstop\_en.mq5](https://www.mql5.com/en/articles/download/442/trailingstop_en.mq5 "Download trailingstop_en.mq5")(10.77 KB)

[trailingstopvariation\_en.mq5](https://www.mql5.com/en/articles/download/442/trailingstopvariation_en.mq5 "Download trailingstopvariation_en.mq5")(10.68 KB)

[trailingprofitvariation\_en.mq5](https://www.mql5.com/en/articles/download/442/trailingprofitvariation_en.mq5 "Download trailingprofitvariation_en.mq5")(11.13 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [The All or Nothing Forex Strategy](https://www.mql5.com/en/articles/336)
- [Random Walk and the Trend Indicator](https://www.mql5.com/en/articles/248)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/7168)**
(52)


![enbo lu](https://c.mql5.com/avatar/2013/9/52326F50-93D0.jpg)

**[enbo lu](https://www.mql5.com/en/users/luenbo)**
\|
24 Sep 2013 at 15:38

reverse entries into trade: reversing entries?

I don't understand what this means at all, do I?

In the original text, it should mean reverse entries into trade (as opposed to the previous trade).

![Jose](https://c.mql5.com/avatar/2013/10/524FCB40-0FED.jpg)

**[Jose](https://www.mql5.com/en/users/jlwarrior)**
\|
21 Oct 2013 at 22:13

Basically what I'm inferring from these articles is that money management is far more [important](https://www.mql5.com/en/economic-calendar/united-states/imports "US Economic Calendar: Imports") than choosing the right entry point

![tao zemin.](https://c.mql5.com/avatar/avatar_na2.png)

**[tao zemin.](https://www.mql5.com/en/users/taozemin)**
\|
17 Apr 2014 at 04:00

**luenbo:**

reverse entries into trade: reversing entries?

I don't understand what this means at all, do I?

Looking at the original text, it should mean reverse entries into trade (relative to the previous trade).

I feel that the core of this algorithm is: 1) Random entry; 2) Trailing stop loss 3) Determine the price level of the trailing stop loss through optimisation, and constantly determine the [optimal parameter](https://www.mql5.com/en/articles/341 "Article: Speed Up Calculations with the MQL5 Cloud Network ") values. The author is very good at giving the test money curve from 2006~2009, this seems to have some difficulty - your TS parameters make the Jan-June curve look good, but June-December, can it still be so smooth?

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
26 Mar 2016 at 05:48

I wonder what could result from having both a **trailing take** based on an initial (wide) **take profit** value and a **trailing stop** based on an initial **stop loss** value...

Has anybody tried this? What results did you achieve?


![Feng Xia Zhang](https://c.mql5.com/avatar/2021/1/6000032F-22FB.gif)

**[Feng Xia Zhang](https://www.mql5.com/en/users/longtum)**
\|
14 Jan 2021 at 16:40

**MetaQuotes:**

New article [Money-making algorithm using trailing stops](https://www.mql5.com/en/articles/442) has been published:

By [Гребенев Вячеслав](https://www.mql5.com/en/users/Virty)

Trading on the Forex market All completed trades Long and short as long as they are profitable are good

![Introduction to the Empirical Mode Decomposition Method](https://c.mql5.com/2/0/Empirical_Mode_Decomposition_MQL5.png)[Introduction to the Empirical Mode Decomposition Method](https://www.mql5.com/en/articles/439)

This article serves to familiarize the reader with the empirical mode decomposition (EMD) method. It is the fundamental part of the Hilbert–Huang transform and is intended for analyzing data from nonstationary and nonlinear processes. This article also features a possible software implementation of this method along with a brief consideration of its peculiarities and gives some simple examples of its use.

![The Most Active MQL5.community Members Have Been Awarded iPhones!](https://c.mql5.com/2/0/win_iPhone.png)[The Most Active MQL5.community Members Have Been Awarded iPhones!](https://www.mql5.com/en/articles/451)

After we decided to reward the most outstanding MQL5.com participants, we have selected the key criteria to determine each participant's contribution to the Community development. As a result, we have the following champions who published the greatest amount of articles on the website - investeo (11 articles) and victorg (10 articles), and who submitted their programs to Code Base – GODZILLA (340 programs), Integer (61 programs) and abolk (21 programs).

![How to Make a Trading Robot in No Time](https://c.mql5.com/2/0/development.png)[How to Make a Trading Robot in No Time](https://www.mql5.com/en/articles/443)

Trading on financial markets involves many risks including the most critical one - the risk of making a wrong trading decision. The dream of every trader is to find a trading robot, which is always in good shape and not subject to human weaknesses - fear, greed and impatience.

![Why Is MQL5 Market the Best Place for Selling Trading Strategies and Technical Indicators](https://c.mql5.com/2/0/mql5-market.png)[Why Is MQL5 Market the Best Place for Selling Trading Strategies and Technical Indicators](https://www.mql5.com/en/articles/401)

MQL5.community Market provides Expert Advisors developers with the already formed market consisting of thousands of potential customers. This is the best place for selling trading robots and technical indicators!

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/442&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071507644142791000)

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