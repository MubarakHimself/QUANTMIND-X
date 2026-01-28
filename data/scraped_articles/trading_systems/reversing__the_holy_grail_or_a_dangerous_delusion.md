---
title: Reversing: The holy grail or a dangerous delusion?
url: https://www.mql5.com/en/articles/5008
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:43:35.728122
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/5008&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070511860270176053)

MetaTrader 5 / Trading systems


### Table of contents

- [Introduction](https://www.mql5.com/en/articles/5008#para1)
- [Basics of proper reversing](https://www.mql5.com/en/articles/5008#para2)
- [The Expert Advisor that trades with and without reversing](https://www.mql5.com/en/articles/5008#para3)
- [Testing symbols and parameters](https://www.mql5.com/en/articles/5008#para4)
- [How reversing is performed](https://www.mql5.com/en/articles/5008#para5)
- [Reversing as an independent trading strategy](https://www.mql5.com/en/articles/5008#para6)
- [Entry once a day](https://www.mql5.com/en/articles/5008#para7)
- [Standard indicators](https://www.mql5.com/en/articles/5008#para8)
- [Summary table of results](https://www.mql5.com/en/articles/5008#para9)
- [Conclusion](https://www.mql5.com/en/articles/5008#para10)

### Introduction

Reversing is a type of martingale. Such systems suggest lot doubling after a losing trade, so that profit could cover previous trade's losses in case of success.

The main difference of the reverse martingale from the classical one is that the deal following the losing one is opened in the opposite direction. Let us review an example:

- You open a 0.1-lot Long trade #1;
- If the trade closes by Take Profit, the next trade will also have the volume of 0.1 lot;
- If trade #1 closes by Stop Loss, trade #2 should be opened immediately after the Stop Loss. This new trade is opened in the opposite direction with a doubled lot: Short, 0.2 lots;
- If trade #2 closes with a profit, this profit covers the loss of trade #1 and additionally brings profit;
- If trade #2 closes by Stop Loss, trade #3 is opened immediately after the Stop Loss. This new trade is opened in the opposite direction: Long, 0.4 lots;
- If trade #3 closes with a profit, this profit covers the losses of trades #1 and #2, and additionally brings profit of 0.1 lots (0.4 – 0.2 – 0.1 = 0.1 lots);
- If not, you again open a trade in the opposite direction with a double lot ...

Do you think such a strategy can be profitable? Why not, when used in a trending market? This strategy helps you follow the trend. If you go wrong with the trend in your first trade, you simply close it and open a new one in the trend direction. The most important thing is to use a large enough stop loss, so that it is not hit by noise market fluctuations. We will talk about this in the next section of the article.

### Basics of proper reversing

Having performed many tests while optimizing the below Expert Advisor, I have prepared some rules that need to be followed if you want to minimize the risk of losing your deposit. These rules are not the ultimate truth, but worked for all tested symbols and period.

**Rule 1:** Take Profit should be greater than Stop Loss.

Articles relating to martingale and reverse martingale techniques, suggest that Stop Loss should be equal to Take Profit. But tests proved this opinion to be wrong. In all tests conducted in all markets, the use of equal Stop Loss and Take Profit caused the deposit to be lost. Your only chance to obtain profit is to set Take Profit at least twice as large as Stop Loss.

The optimal ratio between these two parameters depends on the particular financial instrument. For example, the ratio for GBPUSD is approximately 3:1 or 4:1; for EURUSD it can be both 2:1 and 4:1.

The ratio of 2:1 means that the Take Profit level should be twice as large as Stop Loss. I.e. if SL is 40 points, TP should be equal to 80 points.

**Rule 2:** Stop Loss should be large enough.

In a ranging market, such as Forex, the combination of small Stop Loss levels with a reversing strategy causes deposit loss. All performed tests confirmed this rule.

In the below EA, we will use Stop Loss of 40-90 points. Please keep in mind that since currency symbols have 5 or 3 decimal places, the selected SL and TP are multiplied by 10 (I don't know why, but this is done in all Expert Advisors published under the [Articles](https://www.mql5.com/en/articles) section). That is, the real Stop Loss is 400-900 points. That can be compared to the average daily movement of the financial instrument.

Stop Loss values below 400 points cause complete deposit loss regardless of TP. The only exception to this rule will be provided below, in the description of the strategy involving entries once a day.

**Rule 3:** The timeframe should not be too small.

The use of large SL and TP values reduces strategy dependence on the timeframe. Positions are held open for quite a long time - from one day to one or more weeks. But it was noticed that profit on timeframes below M15 was lower. The same was observed on timeframes above H1, which is because the number of trades decreases.

**Rule 4:** Stop at the proper time.

In theory, if you keep opening new positions after a losing one, you will ultimately obtain profit. But in practice, the endless opening of positions with a doubled volume causes losses or a complete deposit loss.

Tests have shown that after a certain number of reversals the profit begins to decline, until the deposit is completely lost. This can be easily explained. A trade following the Nth loss would be so large, that if it also closes with a loss, the deposit will either be completely lost or so badly damaged, that recovery will take too much time.

The experimentally obtained optimal number of reversals is 8. For some symbols, this number can be greater when a small Stop Loss is used.

We will use 8 reverse trades in our tests, since it is the most suitable one for the most cases. Although, a value of 7 or 9 might be more appropriate for some indicators.

### The Expert Advisor that trades with and without reversing

In this article we will not consider the process of developing an Expert Advisor which uses the reverse martingale method. Suppose, we already have one. The source code of this Expert Advisor is attached below.

The use of certain indicators can be enabled in the Expert Advisor settings. To do this, set the period of the desired indicator to a nonzero value:

[https://c.mql5.com/2/34/en_parameters__of_expert.png](https://c.mql5.com/2/34/en_parameters__of_expert.png "https://c.mql5.com/2/34/en_parameters__of_expert.png")![Expert Advisor Parameters](https://c.mql5.com/2/34/en_parameters__of_expert__3.png)

During testing, we will enable each of the indicators one by one to see their behavior and to find out which indicator is the most suitable one for the reversing technique. If you conduct your own tests, do not forget to turn off the previous indicator so that it does not affect testing results. Alternatively, you may download the SET file of the desired indicator and symbol. SET files are available in the archive attached to the article.

Please keep in mind that the purpose of the article is not to develop an Expert Advisor with maximum profitability. Therefore, we optimized only indicator parameters and the point values ​​of the Stop Loss and Take Profit levels. Indicator parameters were optimized separately from Stop Loss and Take Profit. Therefore, the found results may not be the most optimal ones. But they are quite suitable for our purposes.

Also, the best direction of the first trade was selected during optimization (long, short or any). All other Expert Advisor settings were not changed.

### Testing symbols and parameters

We will test the EA using two probably the most popular symbols: EURUSD and GBPUSD.

Testing timeframe: M15.

Time interval: the maximum possible, from 1998 to July 2018, which is equal to about 20 years.

First, we will test the operation of indicators combined with the reversing technique. Then we will perform tests with a fixed lot, without reversing. We will only test one symbol without reversing — EURUSD. Because the main purpose of this article is reversing.

For each of the variant, we will select maximum profitable parameters of the indicator, as well as Stop Loss and Take Profit points. This will be followed by a chart of the most profitable (least unprofitable) variant. Full testing reports are available in the attachment.

All tests will be performed in the M1 OHLC mode. Since we are working on the M15 timeframe, this will not affect testing much, but will significantly increase its speed.

I tried to follow certain rules in the process of testing, in order to provide similar conditions for all the indicators. In particular:

- Initial balance: $ 10,000; if any optimization variant leads to deposit loss, then it is not taken into account when choosing the best variant, even if its profitability is much higher than that of other optimization variants; i.e. with an initial deposit of $15,000 or more, more profitable EA parameters could possibly be found;
- If the found variant of EA parameters optimization has a slight drawdown, then the parameter _"Size of add. lot in reversing_" is set to 0.01, which allows to increase the EA's profitability by 1.5-2 times.

Let us view some of the Expert Advisor parameters:

- _Lot size_. The lot value of the initial trade.
- _Size of add. lot in reversing_. It allows to increase the volume of opened reverse positions by the specified number of lots.

For example, if the parameter value is 0.01, while the lot size is 0.01, then the reverse position at the second step will be opened with the volume of 0.03 lots, and not 0.02. I.e. 0.01\*2+0.01.

The volume at the third step will be 0.05 lots. I.e. (0.01\*2)\*2+0.01.

Increase of additional lot allows increasing system profitability. However, this may also increase drawdowns.
- _Stop Loss type_. For our purposes, Stop Loss type should always be in points.
- _Stop Loss_. Allows setting the Stop Loss size in points. The SL value for currencies is multiplied by 10. It means that 40 points make a Stop Loss equal to 400 points.
- _Take Profit type_. Allows selecting TP in points or setting the calculation based on the Stop Loss and a multiplier. In the second case, if the "Take Profit" parameter is equal to 4, the Take Profit level will be equal to 4 Stop Losses. For example, if Stop Loss is 40 points, Take Profit will be: 40\*4=160 points.
- _Take Profit_. The Take Profit value in points or a multiplier. If Take Profit is specified in points, the actual value for Forex symbols is multiplied by 10.
- _Open long positions_. Many of the strategies discussed further are profitable only in one of the directions. This parameter allows disabling the opening of Long positions, if this direction is unprofitable for the strategy.
- _Open short positions_. Allows disabling opening of Short positions.
- _Use ORDER\_FILLING\_RETURN instead of ORDER\_FILLING\_FOK_. If attempt to open position using this EA causes the same error with your broker, try to set this parameter to true.
- _Action in case of Stop Loss_. This parameter allows determining the usage of the reversing technique.
- _Max. lot multiplier in reversing_. If the EA uses the reversing method, this parameter allows limiting the maximum allowed number of reversals. In all tests, this parameter will be set to 8. If you set 0, reversing will be disabled.

Other Expert Advisor parameters which are not related to indicators should better be used as is.

### How reversing is performed

Now let us consider how the reversing method is executed in this Expert Advisor. In fact it's quite simple:

- At the moment the first trade is opened, an oppositely directed Buy Stop or Sell Stop order with a doubled lot is placed at the Stop Loss level of the open position;
- At the opening if a new bar, a check is performed whether our position exists in the system, as well as whether there is a Buy Stop or Sell Stop order. If the position exists, but there is no opposite order, it means that the order has triggered, so we need to place one more double-lot order in the opposite position;
- If there is no position, while the order still exists, then the position has been closed by Take Profit, so the EA deletes the existing order.

Since the presence of a position and an order is checked only at the opening of a new bar, i.e. once every 15 minutes, there is a small chance that the order can trigger within the 15 minutes and can be closed by Stop Loss. That would interrupt the reversing series, since the Expert Advisor would fail to create a new pending order.

Well, we can accept this, because testing results have shown, that the check at a new bar opening proved to be more profitable than checking and creating a new order right at the next tick.

### Reversing as an independent trading strategy

Now, let's move on to reversing. This is an interesting situation: in reverse martingale trading, the direction and time of the first trade do not matter. Even if this first trade is unsuccessful, the next trade in the opposite direction will fix this mistake.

So, we do not need to search for good entry points, and we can enter whenever we want. In theory, the reversing technique can be considered an independent trading system. Let's check it.

In our case, we will enter every 15 minutes, if there is no position for the symbol.

The entry direction does not matter... Well, it would not matter if it were not for the testing results.

So, here is the GBPUSD balance graph, with SL 85 and TP 190:

![GBPUSD M15: reversing without indicators](https://c.mql5.com/2/33/gbpusd_m15_plain.png)

There are 2647 trades in total. The EA made the profit of 115,115 for 20 years. That is equal to 57% per year.

Here is the EURUSD balance graph, with SL 45 and TP 175:

![EURUSD M15: reversing without indicators](https://c.mql5.com/2/33/eurusd_plain.png)

There are 3693 trades in total. Profit: 117,521. This is 58% per year.

Both for GBPUSD and EURUSD, we open only Short positions. If Long positions are enabled, profit becomes less. But still, the strategy remains profitable. It means that reversing can be used as a self-sufficient trading system. We managed to achieve profitability on two symbols over the period of twenty years.

Though, the balance graphs are not very beautiful. In particular, GBPUSD didn't have any profit growth in the period from 2004 to 2008. However, there are no large drawdowns.

The EURUSD balance graph is better, but it still has flat periods that are 1-2 years long.

Nevertheless, the profitability of reversing is obvious. I think there is no point in showing balance graphs without reversing. Logically, they would be losing.

### Entry once a day

There is one interesting trading system: It does not use any indicators, but applies some other ideas instead. The idea is as follows: if you keep entering at the same time, while using a small Stop Loss and a huge Take Profit, most probably you will profit.

It seems to be absurd. However, I might believe that the idea could work for trading indices. For example, on Dow Jones, if you enter at 16:25 or 16:30, i.e., before the opening of the stock market. Usually, strong movements happen at this time, and a trend may begin.

Actually, it does work with indices and shows quite a good profitability, even without reversing. But will the strategy show good results in the Forex market?

Surprisingly, there are two time periods for GBPUSD, during which this trading system really works. These are 15:15 and 19:45. I do not know why this time. But have a look at the balance graph (entries at 19:45, SL 20, TP 750):

![GBPUSD M5: entries once a day](https://c.mql5.com/2/33/gbpusd_openmarket.png)

There are 1788 trades in total. Profit: 215,834. This is equal to 107% per year.

Profitability of about 100% per year for 20 years is impressive. None of the indicators discussed further will provide such profitability.

Of course, the system has long losing periods, but it uses a Stop Loss of 20 points and a Take Profit of 750, so that one profitable trade may cover hundreds of losing deals.

Note that I additionally made a few changes to the system. The above profitability was obtained with no entries on Fridays and in the summer months - June, July and August (I disabled appropriate entries). If entries are allowed on any day and month, then the balance graph and total profit become a little worse, although the strategy still remains profitable, showing very good results.

Moreover, exactly the same rule (disabling trading on Fridays and in the summer months) also works for the EURUSD symbol. However, it shows better results if trading is disabled only in August instead of all summer months. Even if we additionally disable June and July, this would cut profitability by as little as 1,000 dollars.

Here is the EURUSD graph, with SL 20, TP 680:

![EURUSD M5: entries once a day](https://c.mql5.com/2/33/eurusd_openmarket.png)

There are 1379 trades in total. Profit: 256,280. This is 128% per year.

A very interesting fact: disabling of trading on Friday leads to an increase in profits in both cases. Why Friday? The triple swap in the Forex market is charged on Wednesdays. So Friday is an ordinary day. Probably, volatility increases before weekend, thus the Stop Loss is hit.

Now let us view the EURUSD graph (SL 20, TP 900), entries are performed once a day and the fixed lot is used. I.e. no reversing is used:

![EURUSD M5: entries once a day, without reversing](https://c.mql5.com/2/33/norevert_openmarket.png)

There are 830 trades. Profit: 69. In this case, profit is obtained rather randomly.

### Standard indicators

Let's try to improve trading results by applying standard indicators.

**Bollinger Bands**

Long entry conditions:

- if the current bar is above the moving average, skip it;
- if the previous bar is falling, skip it;
- if the bar before the previous one is not beyond the lower Bollinger border, skip;
- if the bar before the previous one did not close above the lower Bollinger border, skip;

Here is the GBPUSD balance graph (SL 44, TP 167):

![GBPUSD M15: Bollinger Bands](https://c.mql5.com/2/33/gbpusd_bb.png)

4158 trades. Profit: 81,894. This is equal to 40% per year.

The graph has not improved much. Profit is even less than when trading without indicators.

Now, here is the EURUSD balance graph, with SL 45 and TP 160:

![EURUSD M15: Bollinger Bands](https://c.mql5.com/2/33/eurusd_bb.png)

3135 trades. Profit: 77,090. This is 38% per year.

The graph got even worse. Profits have declined substantially.

The EURUSD balance graph without reversing (SL 80, TP 180) looks bad:

![EURUSD M15: Bollinger Bands without reversing](https://c.mql5.com/2/33/norevert_bb.png)

Trades: 1358. Profit: 74.

**Average Directional Movement Index (ADX)**

The indicator will be used as follows: if ADX value is greater than the parameter value, and DI+ is above DI-, no entry should be performed. In all other cases open a Long position.

GBPUSD balance graph, SL 45, TP 140:

![GBPUSD M15: the ADX indicator](https://c.mql5.com/2/33/gbpusd_adx.png)

Trades: 5795. Profit: 120,043. This is 60% per year.

This result is much better. The use of ADX caused an increase in profits, compared to trading without indicators. The increase is not very significant. The number of trades also increased twice due to a lower Stop Loss value. There are still flat periods: 1998-2004, 2005-2008, 2015-2017.

The EURUSD balance graph does not differ much from trading without indicators (SL 45, TP 140):

![EURUSD M15: the ADX indicator](https://c.mql5.com/2/33/eurusd_adx.png)

Trades: 3903. Profit: 116,791. This is 58% per year.

EURUSD balance graph without reversing looks much worse, however there is a small upward trend (SL 45, TP 140):

![EURUSD M15: the ADX indicator, without reversing](https://c.mql5.com/2/33/norevert_adx.png)

Trades: 3613. Profit: 283.

**Commodity Channel Index (CCI)**

Long entry condition: the indicator value exceeds the appropriate parameter. Short entries are basically unprofitable, so we only enable Long entries.

GBPUSD balance graph (SL 40, TP 135):

![GBPUSD M15: the CCI indicator](https://c.mql5.com/2/33/gbpusd_cci.png)

Trades: 5285. Profit: 132,990. This is equal to 66% per year.

The result is even better than when using the ADX indicator. However, profit is not much higher than when trading without indicators. And there are still flat periods. Drawdowns in such periods are not as large as with ADX, but they are larger than when trading without indicators.

EURUSD balance graph, SL 45, TP 230:

![EURUSD M15: the CCI indicator](https://c.mql5.com/2/33/eurusd_cci.png)

Trades: 2381. Profit: 78,913 or 39% per year. The result is much worse than when using no indicators at all.

EURUSD balance graph without reversing has a clear uptrend only until 2005 (SL 90, TP 120):

![EURUSD M15: the CCI indicator, without reversing](https://c.mql5.com/2/33/norevert_cci.png)

Trades: 1687. Profit: 522.

**Momentum**

The indicator is used similarly to previous ones: Long entries are performed is the indicator value exceeds the appropriate parameter.

GBPUSD balance graph, SL 45, TP 170:

![GBPUSD M15: the Momentum indicator](https://c.mql5.com/2/33/gbpusd_momentum.png)

Trades: 4775. Profit: 81,398. This is equal to 40% per year. The result is not very impressive.

EURUSD balance graph, SL 40, TP 185:

![EURUSD M15: the Momentum indicator](https://c.mql5.com/2/33/eurusd_momentum.png)

Trades: 3625. Profit: 30,896 or 15% per year. The result is also not very good.

The EURUSD balance graph without reversing (SL 60, TP 140):

![EURUSD M15: the Momentum indicator, without reversing](https://c.mql5.com/2/33/norevert_momentum.png)

Trades: 258. Profit: 296.

**Moving Average (MA)**

Long entry conditions:

- the previous bar Open price is less than the value of the moving average, while the previous bar's Close price is above the Moving Average,
- the Moving Average of the current bar is less than the Moving Average of the previous bar;
- the difference between Moving Averages of the current and previous bars should be greater than the appropriate parameter value.

Short entry conditions are opposite to Long ones.

GBPUSD balance graph (SL 46, TP 174):

![GBPUSD M15: the Moving Average indicator](https://c.mql5.com/2/33/gbpusd_ma.png)

Trades: 2788. Profit: 73,020. i.e. 36% per year. This is much worse than trading without indicators.

EURUSD balance graph, SL 95, TP 175:

![EURUSD M15: the Moving Average indicator](https://c.mql5.com/2/33/eurusd_ma.png)

Trades: 829. Profit: 24,397 (12% per year). It is a very bad result.

EURUSD balance graph without reversing looks well, compared to those considered earlier (SL 40, TP 145):

![EURUSD M15: Moving Average, without reversing](https://c.mql5.com/2/33/norevert_ma.png)

Trades: 610. Profit: 493.

**Moving Averages Convergence-Divergence (MACD)**

If the MACD histogram is below the line, enter a Long trade. If the histogram is above the line, enter a Short trade.

GBPUSD balance graph, SL 45, TP 170:

![GBPUSD M15: the MACD indicator](https://c.mql5.com/2/33/gbpusd_macd.png)

Trades: 4394. Profit: 29,392 (14% per year). Trading without indicators showed much better results.

EURUSD balance graph, SL 50, TP 165:

![EURUSD M15: the MACD indicator](https://c.mql5.com/2/33/eurusd_macd.png)

Trades: 2895. Profit: 56,692. This is 28% per year. This is also much worse than trading without an indicator.

The EURUSD balance graph without reversing (SL 80, TP 175):

![EURUSD M15: the MACD indicator, without reversing](https://c.mql5.com/2/33/norevert_macd.png)

Trades: 1620. Profit: -255.

This is the only indicator, which caused losses when used without reversing.

**Williams' Percent Range (WPR)**

If the WPR value of the current bar is greater than the value specified in the EA settings, and the WPR of the previous bar is lower than the value specified in the EA settings, enter a Long position.

GBPUSD balance graph, SL 45, TP 170:

![GBPUSD M15: the WPR indicator](https://c.mql5.com/2/33/gbpusd_wpr__1.png)

Trades: 3577. Profit: 110,077. This is equal to 55% per year. It is a good result. At least, it is better than when trading without indicators.

EURUSD balance graph (SL 90, TP 185):

![EURUSD M15: the WPR indicator](https://c.mql5.com/2/33/eurusd_wpr__1.png)

Trades: 1180. Profit: 72,469 (37% per year). This is not the best result.

EURUSD balance graph without reversing shows an uptrend up to the year of 2008, after which is has a flat period (SL 95, TP 195):

![EURUSD M15: the WPR indicator, without reversing](https://c.mql5.com/2/33/norevert_wpr__1.png)

Trades: 626. Profit: 573.

### Summary table of results

Now, let's represent all the results in one table, and select the most profitable indicators and trading systems for the reversing technique.

Here is the table for GBPUSD:

| Strategy | Net profit | Profit factor | Trades | Max. drawdown | % of profitable trades | Stop Loss | Take Profit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Without indicators | 115,115 | 1.24 | 2647 | 14,269 (47.61%) | 31.92 | 85 | 190 |
| Once a day | 215,834 | 1.68 | 1788 | 54,900 (53.64%) | 3.02 | 20 | 750 |
| Bollinger Bands | 81,894 | 1.21 | 4158 | 26,170 (36.14%) | 20.9 | 44 | 167 |
| ADX | 120,043 | 1.28 | 5795 | 16,823 (62.52%) | 24.31 | 45 | 140 |
| CCI | 132,990 | 1.36 | 5285 | 15,171 (14.81%) | 23.77 | 40 | 135 |
| Momentum | 81,398 | 1.2 | 4775 | 23,996 (24.84%) | 21.97 | 45 | 170 |
| Moving Average | 73,020 | 1.26 | 2788 | 17,261 (29.87%) | 20.48 | 46 | 174 |
| MACD | 29,392 | 1.14 | 4394 | 11,309 (77.69%) | 20.3 | 45 | 170 |
| WPR | 110,077 | 1.33 | 3577 | 17,923 (62.78%) | 21.44 | 45 | 170 |

The summary table for EURUSD:

| Strategy | Net profit | Profit factor | Trades | Max. drawdown | % of profitable trades | Stop Loss | Take Profit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Without indicators | 117,521 | 1.37 | 3693 | 13,063 (10.24%) | 21.04 | 45 | 175 |
| Once a day | 256,280 | 2.02 | 1379 | 39,447 (44.84%) | 3.92 | 20 | 680 |
| Bollinger Bands | 77,090 | 1.25 | 3135 | 16,616 (20.88%) | 21.15 | 45 | 160 |
| ADX | 116,791 | 1.39 | 3903 | 18,356 (71.11%) | 24.49 | 45 | 140 |
| CCI | 78,913 | 1.26 | 2381 | 25,817 (22.97) | 17.01 | 45 | 230 |
| Momentum | 30,896 | 1.17 | 3625 | 10,663 (16.43%) | 18.23 | 40 | 185 |
| Moving Average | 24,397 | 1.35 | 829 | 11,586 (76.72%) | 35.34 | 95 | 175 |
| MACD | 56,692 | 1.21 | 2895 | 14,388 (47.37%) | 23.11 | 50 | 165 |
| WPR | 72,469 | 1.75 | 1180 | 9,987 (23.57%) | 32.03 | 85 | 195 |

We will not compare entering once a day with other systems. It is better than all other cases.

As for other strategies, the profitability of systems without indicators is not much worse, than those using indicator signals.

Moreover, EURUSD trading without indicators shows a better profitability. This pure strategy can only be compared with the use of the ADX indicator. But it shows much worse performance in terms of maximum drawdown, and is very similar to trading without indicators, both in terms of profit and the number of trades.

For GBPUSD trading, the ADX and CCI indicators can help improve the results. But the improvement is not significant, while the profit chart has larger drawdowns, and the number of trades is doubled. Let's have a look at these charts. The ADX indicator:

![GBPUSD M15: the ADX indicator](https://c.mql5.com/2/33/gbpusd_adx__1.png)

The CCI indicator:

![GBPUSD M15: the CCI indicator](https://c.mql5.com/2/33/gbpusd_cci__1.png)

The CCI chart looks better than that of ADX. It rises more smoothly, has less losing periods and larger profit.

Finally, let's look at the EURUSD results without reversing:

| Strategy | Net profit | Profit factor | Trades | Max. drawdown | % of profitable trades | Stop Loss | Take Profit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Once a day | 69 | 1.04 | 830 | 375 (3.75%) | 2.41 | 20 | 900 |
| Bollinger Bands | 74 | 1.01 | 1358 | 385 (3.7%) | 31.96 | 80 | 180 |
| ADX | 283 | 1.02 | 3613 | 434.01 (4.11%) | 25.27 | 45 | 140 |
| CCI | 522 | 1.06 | 1687 | 307.43 (2.87%) | 45.29 | 90 | 120 |
| Momentum | 296 | 1.3 | 258 | 113.32 (1.12%) | 36.43 | 60 | 140 |
| Moving Average | 493 | 1.27 | 610 | 86.99 (0.83%) | 26.56 | 40 | 145 |
| MACD | -255 | 0.97 | 1620 | 502.68 (4.93%) | 31.73 | 80 | 175 |
| WPR | 573 | 1.14 | 626 | 232.3 (2.21%) | 38.18 | 100 | 195 |

The most interesting results were obtained when using a simple moving average. The strategy shows rather low maximum drawdown and a good profit, compared to other indicators. We can increase the lot size for the Moving Average strategy, so that the maximum drawdown becomes similar to that of the reversing strategies, then the profit will be somewhat equal to those strategies. With the maximum drawdown of 16,000, the profit will be around 90,000.

Other indicators, except Momentum, show very low profitability. It means that in real trading they can become generally unprofitable.

### Conclusion

Now it's time to draw some conclusions.

**Conclusion 1.** Virtually none of the standard indicators can provide stable and constant profits when used alone, at least without a proper money management. The only exception is the Moving Average. Although this is one of the oldest technical indicators, it generates a smooth graph which moves upwards for the period of 20 years:

![EURUSD M15: the MA indicator, without reversing](https://c.mql5.com/2/33/norevert_ma__1.png)

The strategy can be further improved by using a time-based filter for entries.

**Conclusion 2.** Reversing can be used as a self-sufficient trading system. Use of additional indicators affects its performance negatively. Perhaps such results are connected with large Stop Loss levels and huge Take Profit levels - the indicators cannot predict such large movements. Another reason for that may be the incorrect use of the indicators. They might have performed better with lower Stop levels or larger indicator periods.

**Conclusion 3.** There is an opinion that the reverse martingale system is a very risky and obviously losing strategy. I hope the series of tests provided herein have proved that this opinion, at least the statement "obviously losing" is wrong.

With a proper use of reversing, none of the trading systems using one of the indicators became unprofitable. Moreover, the strategies that made loss without reversing, became profitable with the reversing technique. Furthermore, the use of reversing does not make a strategy more losing.

**Conclusion 4.** Definitely, the reversing strategy cannot earn a million dollars out of one dollar a month. The obtained profit (in comparison with the initial deposit) is not very impressive. However, such a strategy can teach you to enjoy losses — the more losses precede a profitable trade, the greater profit is obtained.

This seems to be a very strange statement. In theory, the finally obtained profit should always be equal to the initial deal value. The rest of the profit covers those previous losses. This is true, if using equal Take Profit and Stop Loss values. Since our Take Profit is 2-5 times larger than Stop Loss, in addition to that profit equal to the initial trade lot, we also receive profit for the entire volume of the final trade, which is equal to the point distance between the Stop Loss and Take Profit.

When using such a large Take Profit, it is not necessary to increase the volume of each reversing trade. Alternatively, we can increase every second or third trade. This would probably cut drawdowns and increase the number of reversals, although profit would be lower in this case.

**Conclusion 5.** Reversing is really a high-risk strategy. Check the maximum drawdown: In all the above results, the drawdown is greater than the initial deposit.

**Conclusion 6.** So what is reversing?

Is it a Holy Grail? Obviously, it is not. Reverse martingale does not eliminate losses, it may only delay them for some time. With reversing, one cannot expect huge profits.

Is this a delusion then? No, it is not. Reversing allows you to make profit, which is small, but quite stable. The key point here is to have a large initial deposit.

So, what is reversing? Apparently, this is a common trading strategy.

It is up to you to decide whether to apply reversing or not. One thing can be said for sure: reversing requires a large deposit. In order to sustain one maximum drawdown from one losing series of reverse trades (even if you use the minimum initial lot) you need to have a deposit of no less than $3,000. It is not easy to risk 3,000 dollars while trying to earn 100,000.

However, it is much easier to invest 30-100 dollars and risk this amount targeting to earn 1,000 dollars.

But the minimum required deposit is $3,000, so how to use the reversing technique with $30? Of course, on cent accounts. Thus, 30 dollars turn into 3,000.

Let's end the article here. The Expert Advisor used for testing the above strategies is available in the attachment below. Save the EA's MQL5 and EX5 files to the _Experts_ folder. The _Strategies_ folder should be saved to _Include_. Also, the archive contains SET-files with Expert Advisor parameters for different symbols and indicators. All Strategy Tester reports in the HTML format are also provided in the attachment.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5008](https://www.mql5.com/ru/articles/5008)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5008.zip "Download all attachments in the single ZIP archive")

[SET\_files.zip](https://www.mql5.com/en/articles/download/5008/set_files.zip "Download SET_files.zip")(59.73 KB)

[html\_reports.zip](https://www.mql5.com/en/articles/download/5008/html_reports.zip "Download html_reports.zip")(10475.18 KB)

[mql5.zip](https://www.mql5.com/en/articles/download/5008/mql5.zip "Download mql5.zip")(117.84 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a cross-platform grid EA: testing a multi-currency EA](https://www.mql5.com/en/articles/7777)
- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)
- [Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)
- [Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)
- [Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)
- [Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)
- [Developing a cross-platform grider EA](https://www.mql5.com/en/articles/5596)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/284743)**
(19)


![ELDAR133](https://c.mql5.com/avatar/2023/3/64159824-782c.jpg)

**[ELDAR133](https://www.mql5.com/en/users/eldar133)**
\|
13 Jul 2019 at 09:14

Thank you for the article! Very informative, clear and lucid.


![Aleksei Kuznetsov](https://c.mql5.com/avatar/2013/10/52601B64-7C6E.jpg)

**[Aleksei Kuznetsov](https://www.mql5.com/en/users/elibrarius)**
\|
30 Aug 2019 at 18:29

Two years have passed. You can do a test on the new 2017-2019 data for your chosen parameters. This will turn out to be a forward.

Usually the best forward models change the trend from rising to falling.

I wonder if this will be the same with your methodology? Or is the fit to history still avoided?

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
26 Nov 2019 at 04:03

**elibrarius:**

Two years have passed. You can do a test on the new 2017-2019 data for your chosen parameters. This will turn out to be a forward.

Usually the best forward models change the trend from rising to falling.

I wonder if this will be the same with your methodology? Or is the fit to history still avoided?

Yeah. That's an interesting topic, by the way. I don't know how things are now - you can look at it, but there was a published signal - the account on it was merged, due to a HUGE, in IMHO, well or among others, difference - there 10 times - it is a difference between SL and TR, conditionally, SL 90 TR 970 - it's nonsense - you can't do that.... If you put it that way, then it is necessary, for example, to hook the trawl from the [profile](https://www.metatrader5.com/en/metaeditor/help/development/profiling "MetaEditor User Guide: Code profiling")....

I want to test it myself... and yes - it will be forward.

I'll post what will work - here...

![behtilb](https://c.mql5.com/avatar/avatar_na2.png)

**[behtilb](https://www.mql5.com/en/users/behtilb)**
\|
29 Aug 2022 at 17:08

Is there no version for MT4?


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
30 Aug 2022 at 01:19

**behtilb [#](https://www.mql5.com/ru/forum/276563/page2#comment_41705897):**

Is there no version for MT4?

Only analogues in code base - here for example

[https://www.mql5.com/en/code/13532](https://www.mql5.com/en/code/13532 "https://www.mql5.com/en/code/13532")

I'm using it myself now in real trading. I've added a trawl from the profit.

![Modeling time series using custom symbols according to specified distribution laws](https://c.mql5.com/2/33/Custom_series_modelling.png)[Modeling time series using custom symbols according to specified distribution laws](https://www.mql5.com/en/articles/4566)

The article provides an overview of the terminal's capabilities for creating and working with custom symbols, offers options for simulating a trading history using custom symbols, trend and various chart patterns.

![Using indicators for optimizing Expert Advisors in real time](https://c.mql5.com/2/34/indicator_RealTime_optimaze.png)[Using indicators for optimizing Expert Advisors in real time](https://www.mql5.com/en/articles/5061)

Efficiency of any trading robot depends on the correct selection of its parameters (optimization). However, parameters that are considered optimal for one time interval may not retain their effectiveness in another period of trading history. Besides, EAs showing profit during tests turn out to be loss-making in real time. The issue of continuous optimization comes to the fore here. When facing plenty of routine work, humans always look for ways to automate it. In this article, I propose a non-standard approach to solving this issue.

![100 best optimization passes (part 1). Developing optimization analyzer](https://c.mql5.com/2/34/TOP100passes.png)[100 best optimization passes (part 1). Developing optimization analyzer](https://www.mql5.com/en/articles/5214)

The article dwells on the development of an application for selecting the best optimization passes using several possible options. The application is able to sort out the optimization results by a variety of factors. Optimization passes are always written to a database, therefore you can always select new robot parameters without re-optimization. Besides, you are able to see all optimization passes on a single chart, calculate parametric VaR ratios and build the graph of the normal distribution of passes and trading results of a certain ratio set. Besides, the graphs of some calculated ratios are built dynamically beginning with the optimization start (or from a selected date to another selected date).

![MQL5 Cookbook: Getting properties of an open hedge position](https://c.mql5.com/2/34/position.png)[MQL5 Cookbook: Getting properties of an open hedge position](https://www.mql5.com/en/articles/4830)

MetaTrader 5 is a multi-asset platform. Moreover, it supports different position management systems. Such opportunities provide significantly expanded options for the implementation and formalization of trading ideas. In this article, we discuss methods of handling and accounting of position properties in the hedging mode. The article features a derived class, as well as examples showing how to get and process the properties of a hedge position.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nyioiuxqkspvurquqkalihwisardheme&ssn=1769186614009707884&ssn_dr=0&ssn_sr=0&fv_date=1769186614&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5008&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Reversing%3A%20The%20holy%20grail%20or%20a%20dangerous%20delusion%3F%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918661432817515&fz_uniq=5070511860270176053&sv=2552)

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