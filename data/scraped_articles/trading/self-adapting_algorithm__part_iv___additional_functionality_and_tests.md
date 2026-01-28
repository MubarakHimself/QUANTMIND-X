---
title: Self-adapting algorithm (Part IV): Additional functionality and tests
url: https://www.mql5.com/en/articles/8859
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:15:39.488004
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/8859&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6429648776416133785)

MetaTrader 5 / Trading


### Introduction

Before starting, I recommend that you read the previous article in the series " [Self-adapting algorithm (Part III): Abandoning optimization"](https://www.mql5.com/en/articles/8807). This is necessary for understanding the current article.

### Working within the main series

Trading is performed by position series. The funds allocated for opening a position by a trading instrument is divided into several parts to make the entry point blurry. The algorithm predicts the probability of a reversal in a certain area, however it cannot determine exactly when the price reverses. Thus, to compensate errors occurring when opening a position, the volume is accumulated by a series of positions as needed.

In the previous article, I demonstrated how the algorithm generates a signal for opening a position and analyzes several scales simultaneously for defining the maximum trend scale. The basic operation algorithm was described. The price series chart does not consist of one scale. The trend can be present on several scales at the same time, while there can be flat on other scales. This feature should be used to make a profit.

Here, a trend section is a segment, on which the trend continuation probability exceeds 50%, while a flat segment is one, on which the trend reversal probability exceeds 50%. In other words, if the previous block was growing, then in the trend section, the new block will also grow with a probability higher than 50%. On a flat chart, a growing block is most probably followed by a falling block. I described the proposed definition in detail in the article " [What is a trend and is the market structure based on trend or flat](https://www.mql5.com/en/articles/8184)?"

![trend-flat](https://c.mql5.com/2/41/59syjl6yr1.PNG)

Figure 1. Trend and flat on different scales

Figure 1 shows a clearly visible bearish trend on 32 blocks of 0.00061. The trend is almost absent on 32 blocks with the scale of 0.00131. In most cases, there are simultaneously the scales featuring both trend and flat.

The algorithm always starts the analysis on the smallest blocks. However, the size of blocks may increase significantly along the way up to the point when the block of a larger scale is able to contain enough blocks of the initial size to be able to generate the new series start signal. The example is shown in Figure 2.

![inside from block](https://c.mql5.com/2/41/ggkabv_n3ndv.PNG)

Figure 2. Movements within the large-scale block

Figure 2 shows what is happening inside one of the blocks. We can see a growing block on large-scale blocks, but if we look inside, we are able to see how events developed. If the algorithm has switched to a large working scale, then the processes occurring inside the blocks of this scale help obtaining an additional profit, which is to be used later to increase the algorithm stability.

The work is carried out by logically connected series of positions. But if a block scale is greatly increased and the initiated series is not complete, start a new position series independent of the previous one but logically connected with it.

We need criteria for starting a new series with an incomplete previous one. If we simply start a new series when a block size of the main series exceeds the threshold value, the result will be poor. The algorithm in such cases will not be adaptive because the new series start step is set manually. With this approach, the algorithm periodically starts new series on a long trend section increasing a load on a deposit.

![long trend](https://c.mql5.com/2/42/long_trend2.PNG)

Figure 3. Smaller scale oscillations on a large trend segment

Figure 3 shows a long bearish trend on GBPUSD in 2008. The rollback on this trend up to the series closure point took 275 days. Longer terms are also quite common. It would be wrong if the algorithm did not trade all this time waiting for a price rollback to complete the previously started position series instead. The red ellipse shows the area where we cannot start a new series, while the blue circle shows where additional series will bring profit.

It is necessary to introduce the basic timeframe concept. In my work, I consider a timeframe to be a block size. The initial data for analysis is taken from the static M1 timeframe. The basic timeframe is a timeframe currently used by the algorithm to open positions. The algorithm starts the market analysis from the first timeframe and increases it up to the necessary one over the course of its work depending on the trend movement size. If the block size of the first timeframe is equal to 10 points, the block size of the second timeframe turns out to be equal to 10\*KTF. Here, KTF is a block size multiplication ratio for obtaining blocks of the next timeframe. If KTF=1.1 and the first timeframe block size is 10, the block sizes are to be as follows:

TF1=10 points, TF2=10\*1.1=11 points, TF3=11\*1.1=12.1 points, etc.

The main task is to prevent the start of new series during a long movement without rollbacks. If we use the example from Fig. 3, the algorithm scans all scales and immediately moves on to the maximum scale or a scale close to the maximum one after starting the second series in the middle of the trend. In this case, the block size of the second series basic timeframe is roughly equal to the block size of the first series basic timeframe, which does not make sense.

For the second series signal to remain meaningful, the basic timeframe block size should exceed the threshold one, besides there should be a flat movement of sufficient amplitude on the basic timeframe of the second series. The mechanism shown in Figure 4 has been developed.

![confirmed series](https://c.mql5.com/2/42/confirmed_series__1.PNG)

Figure 4. The mechanism of confirming the start of the second series

The mechanism has been implemented the following way. The start of the second series becomes possible after the basic timeframe of the first series exceeds the threshold one. After that, the search for the excess of falling or growing blocks starts on the first timeframe of the second series for forming the second series start signal. This is done the same way as for the first series. After the second series start signal has been found (a trend area of sufficient magnitude has been found), the ability to start the second series should be confirmed. To achieve this, we need to find the flat area on the basic timeframe of the second series. This allows us to make sure that the found trend area is a part of the flat area, rather than of a large trend used for the first series operation.

If 10 blocks are analyzed for the series start and searching for a trend movement (like in Figure 4), then how many blocks should be used to search for the flat segment? If the value is set rigidly, the algorithm loses its adaptability. This means the number of blocks should somehow depend on what is happening in the market.

To define the series confirmation blocks, we need to set the range, within which the search for the flat segment is performed. The range is set using two values:

- Bmin — minimum number of blocks of the second series confirmation range;
- Bmax — maximum number of blocks of the second series confirmation range;
- NPb(s2) — number of prevailing blocks of the second series;
- %PV — specified percentage for defining the flat.

Bmin is simply defined as **Bmin=NPb(s2)/(%PV/100).**

To define the upper limit of the Bmax range, we need to calculate the number of basic timeframe blocks of the second series fitting into the two last formed blocks of the basic timeframe of the first series and the blocks of the basic timeframe of the second series, for which the big block is not formed yet. This is how the upper limit of the Bmax range is calculated.

Let's consider an example on Figure 4. Let %PV=50, the number of prevailing blocks is NPb(S2)=9. In this case, Bmin=9/(50/100)=18. The minimum number of blocks where the flat can be found is 18. To define Bmax, calculate the number of small blocks in the last two big ones. On Fig. 4, these are 14 blocks. Next, calculate the number of small blocks to the right of closing the last big one. This makes for 5 blocks. Next, define the sum 14+5=19. The resulting range is Bmin=18, Bmax=19.

Next, search for the flat segment in the range from 18 to 19 blocks of the basic timeframe of the second series. The segment, on which the number of falling blocks is equal to the number of growing ones, is considered flat. The flat segment criterion can be adjusted in the settings because 50% is a too narrow range. We can set 50-55% and assume that if the number of prevailing blocks is from 50% to 55%, then this is a flat.

During the calculations, Bmax may turn out to be quite large, while Bmin is a floating value. Therefore, it would be wrong to define a flat segment using a fixed percentage for different number of blocks. It would be more reasonable to set the value as the probability of falling into the range and convert it into the percentage afterwards. The table 1 below does just that.

![series confirmation table](https://c.mql5.com/2/42/sbuw6lf_ttflttzm9ivcg_c3axd.PNG)

Table 1. Table for calculating the percentage of the number of blocks for confirming the series

Let 56.25% of blocks of the same direction out of 16 blocks serve as a flat segment criterion. This percentage corresponds to 80.36% of all events falling into the range from 2 to 16 vertical blocks. This means the process passes from 2 to 16 blocks vertically within 16 blocks in 80.36% of cases. Using the table, it is possible to recalculate the series confirmation percentage for each unique number of blocks considering that the confirmation percentage should be so that 80.36% of all events fall within the amplitude range.

We are able to use the table to calculate a relevant series confirmation using the table for each number of blocks from the Bmin-Bmax range.

Such a mechanism allows starting new position series when the previous one is not closed yet and considerably decreases the number of instances, in which the block size of the second series becomes equal to the block size of the first series, and the second series become meaningless.

At its start, the second series works similarly to the first one. In other words, the algorithm contains all mechanisms of tracking trends and adjusting the current trading scale.

As soon as the second series starts, the algorithm starts tracking the block size of its basic timeframe. When it exceeds the threshold value, the third series starts according to the conditions described above. This allows creating as many additional series as needed.

> **How it works**

As an example, let's consider the section of the chart displayed in Figure 5. This is the same section displayed in Figure 3. For the algorithm, this section is quite complicated as it features a long wait for a rollback for closing positions.

![opening positions](https://c.mql5.com/2/42/5rnqfi_1.PNG)

Figure 5. Creating several series simultaneously

Figure 5 shows the algorithm behavior on a long trend. After the main series starts with Buy positions, the basic timeframe block has become too big allowing the algorithm to create the second and third series without waiting for the basic one to close. Such an approach allows using price movements on lesser scales and receive an additional profit smoothing out fluctuations on a deposit.

Let's consider two sections with the same price and evaluate the drawdown by funds at these two points in time. The first point is on 11.13.2008 with the price of 1.45554, Buy positions of the first series are open and the price goes against open positions. The funds comprise $8265.16. Then the price continues to fall and returns to 1.45554 on 04.10.2009. But now the funds have increased by $1677.57 comprising $9942.48. If the price goes against our position, the obtained drawdown is not too big compared to the one we can get without using the additional series mechanism.

It is possible to considerably decrease the risks arising when trading on a large scale using asset price fluctuations occurring on a smaller scale. Auto scaling allows us to always trade on a relevant scale, while the additional series confirmation mechanism is a function of adapting to the current conditions.

### Compensating loss-making positions and decision making errors

The trend defining and auto scaling methods work well but they are not perfect. During its work, the algorithm opens positions out of time. Positions may open at the very start of an extended trend possibly leading to a loss in the position series closure point. Position closure point cannot be shifted since it is calculated based on the price series parameters. Trend rollback calculation has been described in the previous article " [Self-adapting algorithm (Part III): Abandoning optimization](https://www.mql5.com/en/articles/8807)". However, it is possible to detect and correct erroneous positions.

**Correcting erroneous positions using the profit of additional series**

One of the objectives of the additional series algorithm lies in the correction of erroneous positions. In the previous article, I showed that the average price movement amplitude is increased non-linearly depending on the number of steps taken. Accordingly, the average loss on an erroneously open position increases non-linearly from the number of steps taken.

![average loss](https://c.mql5.com/2/42/average_loss.PNG)

Figure 6. The law of increasing loss depending on the number of passed steps

Figure 6 shows an approximate law defining the increase of an average loss on an open position. The parameters slightly differ for each instrument but, most importantly, the curve form remains more or less the same meaning this parameter can be measured. The law allows us to predict the average loss of an incorrectly opened position and shows the possible efficiency of using the profit from additional series to close incorrectly opened positions.

A part of the profit obtained from the additional series will be used to close the erroneously opened positions. I made this parameter customizable in the settings for myself. Some of the gained profit is to be spent on supporting the algorithm reliability. The better the algorithm, the lesser profit percentage should be spent to let the algorithm remain stable. Now I use 80% of the additional series profit to compensate for losing positions.

A simple method is applied to detect erroneous positions. If the series receives a loss in a series closure reference point, it definitely has erroneous positions. Out of all positions of the series, we find the most losing one and evaluate the current loss on it. If the position belongs to the first series, then 80% of the profit obtained from the second series should cover the current loss on the position. If the condition is fulfilled, the position is closed.

The funds are accumulated during the work. They are to be used to compensate for loss-making positions. When the necessity arises, these funds are used to compensate for a loss on an open position and the loss from closing a position is subtracted from the reserve. The algorithm goes on accumulating reserves for the next compensation. This continues till all erroneous positions are closed or the main series ends. After the main series completes, all unused funds from the reserve are set to zero.

Erroneously opened positions of the first series remain in the market until the profit of the second series allows them to be closed with a conditional profit equal to zero. It works similarly for the rest of the series. The profit of the third series is used to compensate for the erroneous positions of the second series, and so on.

Figure 5 shows that some Buy positions of the first series were closed at a loss without waiting for the general closing point of the series. This happened due to the loss compensation algorithm activation.

Thus, the fluctuations exploited for profit in the second series are always less than the amplitude of the first series trend. The trend amplitude of the third series is even smaller. The additional series do not lead to a significant increase in equity drawdown, but bring additional profit, which is used to maintain stability. The number of simultaneously open additional series depends solely on the current state of the market and is not configurable by traders. This is another adaptive function. The algorithm itself has a fractal structure. It copies itself at different scales. At the same time, it adjusts its work for each scale individually.

**The correction of the series closure point based on smaller scales fluctuations**

As I wrote earlier, the closure point of the series (rollback from the main trend) depends on the rollback speed. The speed is measured in the number of steps the price needed to make a rollback. The higher the number of steps, the lower the rollback from a trend movement. This feature is associated with the fundamental reasons for the emergence of trends. Trends create deal volumes exceeding the current liquidity. In other words, opening or closing positions for large amounts. A conditional position that gave rise to a trend movement should be closed. This requires liquidity. If it is closed instantly, then the rollback will be 100% of the previous trend, that is, exactly the same trend movement in the opposite direction will appear. But the slower the position closes, the less the rollback. In this case, it does not matter whether the movement was caused by a single or multiple market participants. The mechanism is the same.

The algorithm adjusts the closure point based on calculating the number of basic timeframe blocks. On large scales, this is insufficient. Blocks of great scales feature their own price fluctuations, which should also be considered for adjusting the closure point since blocks are only a conditional representation of price movements created for convenience.

The most correct thing to do is to calculate everything and consider all fluctuations. But I will do it differently for the sake of simplicity (the algorithm is pretty complex already). I am going to use the profit of the additional series for an instrument to correct the closure point. This makes for 80% of the profit obtained from closing additional series. We calculate the amount of profit in the deposit currency to be obtained by the series in case all positions are closed in the reference closure point. The currently available profit of the additional series is added to the current profit of the series. If the obtained value exceeds or is equal to the calculated profit in the closing point, the main series is complete. The rollback is considered to be over and the series is complete.

![Close point](https://c.mql5.com/2/42/close_point1.PNG)

Figure 7. Correcting the closure points based on the additional series profit

Figure 7 displays correcting the closure point based on the additional series profit. Open positions should be closed in the upper point at 1.69846 with the profit of $915.66. By the time the price reaches 1.60887, the current profit level is $109.35. However, due to large fluctuations, additional series are able to earn $1009. 80% of that profit is used to close the series earlier. This means $807.2 is available. The inequality of 109.35+807.2=916.55>915.66 is checked and the series is completed early.

This allows us to define the closure point more accurately and reduce the number of cases when a rollback is already over, while a series is not closed yet.

**Correcting erroneous positions using the profit obtained from trading other assets**

The algorithm described above allows reducing the assets volatility. However, the system is meant for working on 28 trading instruments simultaneously. Certain trading instruments can be in a state of excessive upward or downward trend for a long time. We need to use the profit gained from closing each first series on other instruments to reduce the profitability chart volatility even further.

This approach works if the trend on some trading instruments is accompanied by a flat on others. Figure 6 shows an approximate law explaining the increase of the average amplitude of fluctuations on a single instrument depending on the number of steps. Knowing the law, it is possible to predict the same developments on 28 trading instruments. In other words, the average distance passed by the price vertically is approximately proportional to the step size multiplied by the number of steps to the power of 0.5 (the power is different for different instruments and markets). For more clarity, it is better to take 28 trading instruments and calculate the distance they are able to pass within n steps on average displaying this on a graph.

So, how can we describe 28 independent trading instruments on a single graph? Each of them has its own volatility and different fluctuation range. It is well-known that the behavior of each trading instrument is not similar to another. However, they are not similar only in the form, in which we see them on the graph. If we recalculate the size of the candles in a single currency (for example, USD) rather than in points and assume that we make transactions for a fixed amount in USD, the scale and volatility of each instrument becomes approximately the same. Within a trading system, the decisive factor is the amount of USD passed by the price vertically within n steps when trading a fixed sum.

We should move on from the graph in points to the graph of profits. Here I am talking about a fixed sum for trading rather than a fixed lot. These are completely different things. Let the trading be carried out in currency pairs in the amount of $1000 for each instrument. In this case, we have four equation types depending on a currency pair format. Let's recalculate the candle size from points to USD according to the following equations:

- ![formula EURUSD](https://c.mql5.com/2/42/45q83ut_EURUSD.PNG);

- ![formula USDCAD](https://c.mql5.com/2/42/9b11otn_USDCAD.PNG);

- ![formula eurgbp](https://c.mql5.com/2/42/v3136nn_EURGBP.PNG);

- ![formula cadchf](https://c.mql5.com/2/42/1z5v2u0_CADCHF.PNG).

In the equations:

- EURUSD, USDCAD, EURGBP, CADCHF — candle size in USD for the corresponding currency pair when trading on $1000;
- EURUSD(-1), USDCAD(-1), EURGBP(-1), CADCHF(-1) — previous candle close price;
- EURUSD(0), USDCAD(0), EURGBP(0), CADCHF(0) — close price of the last formed candle;

- 1000 — deal amount in USD.

Let's consider the first equation for EURUSD and all currency pairs, in which the deal is performed in the amount of the currency coming second. Deals are made for the sum of $1000. In this case, the logic of defining the candle size is as follows: Sell $1000 **-->>** buy EUR at the rate of the previous candle (-1 candle) **-->>** on $1000, a reverse deal is performed on EUR at the rate of the last closed candle (0 candle) - **->>** from the number of EUR bought for $1000 at the rate of (-1) candle, subtract the number of EUR sold for $1000 at the zero candle rate **-->>** the obtained profit/loss in EUR is converted to USD at the current rate of EURUSD - **->>** profit/loss in USD is obtained. This is how we define the candle size in USD.

Next, in order to build a graph with candle size in USD, the size of each subsequent candle is added to the size of the previous one.

For USDCAD and other currency pairs, in which USD takes the first place, the calculation is simpler. The transaction is performed in USD already. For cross rates, the calculation is similar to the EURUSD equation, except that in EURGBP pair, $1000 should be converted into GBP, then a trading operation should be performed, and then the resulting profit in EUR should be converted back into USD. For CADCHF, the idea is similar to the previous options, but the calculation is performed via the pair having USD taking the first place, so the equation is modified in the appropriate way.

In any case, the idea behind the operations is to calculate the amount of USD it is possible to earn/lose trading the sum of $1000 for the candle of a given amplitude for each currency pair.

In this algorithm, 80% of the entire profit obtained from closing each first series of any trading instrument is used to compensate for erroneously opened positions of an instrument, for which such a compensation is required. Thus, the entire algorithm works to close "excessive positions" on time on any trading instrument. The objective is to close erroneous positions as quickly as possible so that protracted trend sections do not lead to significant equity drawdowns. This should reduce the volatility of the profitability graph.

![28 currency pairs](https://c.mql5.com/2/42/28_mrlmv_m4zhiy4lxr.PNG)

Figure 8. The path of 28 currency pairs within 10,000 steps

Figure 8 shows the profit graphs for 28 currency pairs simultaneously calculated using the equations shown above. The graphs are built for 10,000 H4 candles. The analyzed period is from 2004 to mid-2010. The Excel file is attached below so that you are able to insert your data and see the equations used to build the charts.

Figure 8 demonstrates that a flat is detected on some currency pairs during a trend on others. This confirms the possibility of reducing the profitability graph volatility using the compensation of erroneously opened positions by the profit obtained from trading on all instruments simultaneously. There is always a trading instrument, on which the algorithm is currently able to earn.

This behavior is typical not only for the foreign exchange market, but also for the stock market with some reservations. The foreign exchange market is often symmetrical for price rises and falls, while the stock market is asymmetric. The developments on the forex and stock markets will approximately be the same with the only difference that the stock market will show the upward bias and the greater correlation between assets. But if we take trading instruments from exchanges of different countries, then the correlation can be significantly reduced down to a negative correlation between the assets.

Besides, I have changed the average candle size (in USD) of all trading instruments. It comprised $2.26. Knowing the number of steps and the average step size, it is possible to evaluate the average vertical movement of all instruments. $2.26\*10,000^0.5=$226. In reality, all trading instruments moved $216.41 on average within 10,000 steps, which is close to the theoretical value.

In this algorithm, 80% of the entire profit obtained from closing each first series of any trading instrument is used to compensate for erroneously opened positions of an instrument, for which such a compensation is required. Thus, the entire algorithm works to close "excessive positions" on time on any trading instrument. The objective is to close erroneous positions as quickly as possible so that protracted trend sections do not lead to significant equity drawdowns. This should reduce the volatility of the profitability graph.

Figure 9 shows how this works. Non-correlating instruments are selected for the test. The test itself is conducted on M1 for the year of 2020. It should be noted that the settings were not optimized and set similar for all instruments. The algorithm adjusts its work in real time.

![GBPUSD+AUDNZD](https://c.mql5.com/2/42/GBPUSDcAUDNZD.PNG)

![EURCAD+CHFJPY](https://c.mql5.com/2/42/EURCADdCHFJPY.PNG)

![GBPUSD+AUDNZD+EURCAD+CHFJPY](https://c.mql5.com/2/42/GBPUSDuAUDNZDlEURCADqCHFJPY.PNG)

Figure 9. Multi-instrument compensation function

At first, the test is conducted on separate currency pairs to evaluate the profitability graph of each pair. Then the test is conducted simultaneously on four currency pairs with the compensation modes enabled.

We can see that the maximum drawdown by funds was on GBPUSD comprising $4311. The maximum profitability is also detected on GBPUSD and is equal to $1555. At the same time, AUDNZD demonstrates a loss of $109 due to stopping the test. If it had been continued, the trading would have been profitable.

The multi-instrument compensation algorithm allows decreasing the drawdown by funds during the test on four currency pairs simultaneously from $4311 to $3517, i.e. by 18%. The total profitability has become $2015.21. The resulting total number of completed transactions during separate trading is 1446 decreasing down to 1409 during joint trading suggesting that the compensation algorithms practically do not affect the course of trading, but improve the stability of work.

Thus, the more trading instruments are used for trading, the more smooth the profitability graph can be obtained (up to a linear one) when trading 1000 trading instruments from various world exchanges and markets simultaneously. We are able to trade stocks, currencies and even cryptocurrencies simultaneously balancing transaction volumes for each trading instrument in a proper way.

### Tests

The algorithm should work on any trading instrument. So first, I will show how it works on AAPL (Apple) stocks. At the moment, the EA is unable to collect statistical parameters of an instrument. This function is in development. This is why I will configure two parameters manually. These are the percentages for opening and closing a series. However, the parameter values are measured using the indicator to be integrated to the EA rather than by using the optimization and fitting. The EA will adjust these values on its own in the future.

![AAPL Buy](https://c.mql5.com/2/42/AAPL_BUY_tester_chart.PNG)

![AAPL Buy tester report](https://c.mql5.com/2/42/AAPL_Buy_tester_report.PNG)

Figure 10. AAPL (long positions only)

The tests are performed for Buy and Sell positions separately since the statistical characteristics of the instrument are not symmetrical for rising and falling. For Buy positions, the parameters are more aggressive, while for Sell ones, they are more conservative. Only two parameters change.

Figure 10 shows the result of the backtest from 06.24.2012 to 12.28.2020 (7.5 years). The test was performed on М1. The profitability graph is smooth, the maximum drawdown by equity is $858, while the profit is $1704. The total number of open positions is 176, while the profit factor is 18.13, which is a very good sign.

Some may say that a complex algorithm is unnecessary if we want to make profit from a growing AAPL stock by opening Buy positions only. And they are right. Let's see how the algorithm behaves if only Sell positions are allowed. Opening short positions on Apple stocks seems like a suicidal idea but the algorithm is supposed to be adaptive. So let's see if it survives.

![AAPL sell](https://c.mql5.com/2/42/AAPL_Sell_tester_chart.PNG)

![AAPL sell report](https://c.mql5.com/2/42/AAPL_Sell_tester_report.PNG)

Figure 11. AAPL (short positions only)

Figure 11 shows the algorithm results on AAPL shares from 06.24.2012 to 12.28.2020, M1. They are not as good as in the case of Buy positions but the algorithm still manages to make a profit. Here, the maximum equity drawdown is $10,533, while the profitability is $2234. The profit factor is considerably lower and is equal to 1.4. The total number of open positions is 944. If it was able to to both buy and sell, the result would be better due to the erroneous position compensation algorithms reducing the profitability graph volatility. The results will become even better when trading 28 trading instruments simultaneously. The algorithm's ability to pass 7.5 years in almost autonomous mode without fitting the parameters to the history is a good result.

Now it is important that it is able to adapt to difficult conditions. I decided to perform the test on AAPL for a reason since this is obviously a difficult trading instrument for making short positions. Let's have a look at the instrument chart during tests.

![AAPL chart](https://c.mql5.com/2/42/AAPL_chart.PNG)

Figure 12. AAPL chart from 06.24.2012 to 12.28.2020

Figure 12 shows the strong growth of AAPL stock. Over the last 7.5 years, the asset has grown 6.3 times. Making profit on such a chart using auto trading only with no optimization is a pretty challenging task.

The algorithm is meant for trading 28 instruments simultaneously. Therefore, the tests are performed on 28 trading instruments. First, let's see the results for working on FOREX. The following currency pairs were used in the test: GBPUSD, EURUSD, NZDUSD, AUDUSD, USDCAD, USDCHF, USDJPY, EURGBP, EURAUD, EURNZD, EURCAD, EURCHF, EURJPY, GBPAUD, GBPNZD, GBPCAD, GBPCHF, GBPJPY, AUDNZD, AUDCAD, AUDCHF, AUDJPY, NZDCAD, NZDCHF, NZDJPY, CADCHF, CADJPY and CHFJPY. All pairs have completely identical settings. The algorithm adapts to all changes in real time.

The test was performed from 01.01.2008 to 01.13.2021 (13 years) on М1. The EA consumes multiple resources and the test of one year takes about 12 days. Therefore, the entire test interval is divided into sections of 1 - 2.5 years to parallelize the tests.

![2008-2009](https://c.mql5.com/2/42/2008.PNG)

![2009-2010](https://c.mql5.com/2/42/2009.PNG)

![2010-2011](https://c.mql5.com/2/42/2010.PNG)

![24.06.2010 -24.06.2012](https://c.mql5.com/2/42/2010-1.PNG)

![24.06.2012 - 24.06.2014](https://c.mql5.com/2/42/2012.PNG)

![24.06.2014-24.06.216](https://c.mql5.com/2/42/2014-2016.PNG)

![24.06.2016 - 24.06.2018](https://c.mql5.com/2/42/2016-2018.PNG)

![24.06.2018-13.01.2021](https://c.mql5.com/2/42/2018-2021.PNG)

Figure 13. Tests of 28 currency pairs from 01.01.2008 to 01.13.2021

Figure 13 displays the tests on 28 currency pairs simultaneously. As we can see, trading on almost all test intervals ended in profit except for the interval from 06.24.2012 to 06.24.2014. Here, the test completes with a small loss of $123. This is because the test has stopped. If the test continued 13 years ceaselessly, the EA would come out of the drawdown. The total number of open positions is 14,415. This statistically significant value suggests that the obtained results were not random. The equity drawdown has an adequate value. However, the future algorithm versions should feature improved profitability and stability parameters.

Since the EA is versatile, let's see how it works on stocks of Russian companies traded on MOEX. The settings are the same as the ones applied for the currencies. However, only long positions are allowed. The current EA version is not yet able to adjust the parameters for short and long positions separately. This function is currently under development. The test was performed from 02.01.2013 to 09.20.2018 (5 years and 7 months). The specified leverage in the tester is 1:5. The portfolio contains the most liquid instruments: GAZP, CHMF, ALRS, HYDR, LKOH, MAGN, MGNT, MTSS, NLMK, NVTK, ROSN, RTKM, SBER, SNGS, SNGSP, TATN, VTBR, SBERP, TATNP, AFLT, FEES, GMKN, RSTI, SIBN, UPRO, MSNG, MTLR and PLZL.

![moex](https://c.mql5.com/2/42/moex_5qo4du_w6zrzrtxbk.PNG)

Figure 14. The test performed on 28 shares of Russian companies from 02.01.2013 to 09.20.2018

Figure 14 shows the test results for 28 shares of Russian companies. The algorithm earns quite consistently and evenly. Just like in the previous case, the profitability has turned out to be low, but most importantly, it remains profitable. The total number of open positions is 5485. This suggests that the result is not random in its nature.

### Sources of profit

Developing a working algorithm or model is insufficient. Another important component is understanding why it works and where the profit comes from. In this case, the profit is obtained from a non-obvious source, namely from increasing entropy. I am talking about the entropy determined by the Boltzmann's principle and characterized by the number of possible system states and the probability of a particular state.

In other words, the entropy in the market is approximately constant. The higher the liquidity of an instrument, the higher the entropy of the price chart of this instrument. With an increase in the number of trading operations performed on a trading instrument, the number of possible states also increases and the system strives to occupy the most probable state. The effect of increasing the entropy arises when the entropy itself is approximately constant, but the number of possible states of the system grows.

The highest entropy can be seen in the foreign exchange market, while the lowest one is detected on low-liquid assets. The entropy of each specific trading instrument tends to become the highest one, but the inflow of funds prevents this from happening. The inflow of funds arises for two reasons: investment attractiveness of the asset and the funds emission. The entropy of each trading instrument eventually grows.

### Summary

The profitability is quite low but the articles demonstrate the model of the fully automated profitable trading on completely different instruments traded on fundamentally different markets. The tests on 57 trading instruments are displayed. Since they have identical parameters, this is equivalent to 520 years of tests for a single trading instrument.

- I have developed the basic algorithm capable of reliable work and having a room for improvements. Each described mechanism should be improved to considerably increase profitability and decrease fund drawdowns.
- In the previous article, I wrote about the two methods of tracking the current trend. They are described in the "position opening delay" paragraph. The tests for 28 trading instruments are conducted using the "trend segment delay" method. This is not the best method as "delay based on the instrument statistical characteristics" works better, but the algorithm is not finished yet, so I have used the working functionality only. In the future versions of the algorithm, this function will be significantly improved.
- The additional analysis of data for the preliminary selection of the minimum block size is required to decrease the number of erroneously opened positions.
- The algorithm does not consider differences in asset prices leading to differences in block prices based on these assets. This significantly reduces the profitability of exchange instruments. We need to add balancing a lot size by a portfolio depending on a block price for each specific asset.
- It is also necessary to implement the separate management of the decision making algorithms for long and short positions, as well as add the analysis of statistics parameters and asymmetry of each certain trading instrument.
- 80% of the entire profit is spent on maintaining the algorithm stability. It is necessary to both improve the mechanism of compensating for loss-making positions and the decision-making mechanisms. This improves the profitability considerably.
- The algorithm is unable to collect statistical parameters of each trading instrument and currently applies the adaptation mechanisms only. In the next versions, the parameters are to be adjusted based on statistical characteristics of a trading instrument fully automatically improving the profitability.
- The differences between price charts of currency pairs and stocks are not used yet. The model should be improved by adding the already existing knowledge.

### Previous articles on this topic

- [Self-adapting algorithm (Part III): Abandoning optimization](https://www.mql5.com/en/articles/8807) – the requirements specification for the described EA is attached to this article. The version tested on 28 FOREX and MOEX instruments almost completely corresponds to the requirements specification.
- [Developing a self-adapting algorithm (Part II): Improving efficiency](https://www.mql5.com/en/articles/8767)
- [Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)
- [What is a trend and is the market structure based on trend or flat?](https://www.mql5.com/en/articles/8184)
- [Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136) – in this article, I explain why I decided to analyze block charts instead of candle ones.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8859](https://www.mql5.com/ru/articles/8859)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8859.zip "Download all attachments in the single ZIP archive")

[chart\_in\_dollars.zip](https://www.mql5.com/en/articles/download/8859/chart_in_dollars.zip "Download chart_in_dollars.zip")(16374.28 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-adapting algorithm (Part III): Abandoning optimization](https://www.mql5.com/en/articles/8807)
- [Developing a self-adapting algorithm (Part II): Improving efficiency](https://www.mql5.com/en/articles/8767)
- [Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)
- [A scientific approach to the development of trading algorithms](https://www.mql5.com/en/articles/8231)
- [What is a trend and is the market structure based on trend or flat?](https://www.mql5.com/en/articles/8184)
- [Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/366708)**
(81)


![Maxim Romanov](https://c.mql5.com/avatar/2025/11/691b0e3f-04d1.png)

**[Maxim Romanov](https://www.mql5.com/en/users/223231)**
\|
21 May 2021 at 20:18

**Gerardo Castano:**

V I see that Infinitely Just is based on the EAs in the article "Developing a self-adapting algorithm (Part I and II). Are you planning to release the version that uses blocks instead of bars and that you describe in part III and IV of your Article?

You have done a good job and it seems only fair that you try to make a profit on your EA.

I did not plan to sell the robot from Articles III and IV.  It is too complicated for a simple user who buys robots in the market.  And it will cost too much for a mass user.  The second article contains a much improved version of "Just", which I traded on [real accounts](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties") for 2 years.  But if you want, then you can discuss the sale of the robot from the third and fourth articles.


![Gerardo Castano](https://c.mql5.com/avatar/2014/4/5352EB16-2D8C.jpg)

**[Gerardo Castano](https://www.mql5.com/en/users/geriflu)**
\|
26 May 2021 at 20:57

**Maxim Romanov:**

I did not plan to sell the robot from Articles III and IV.  It is too complicated for a simple user who buys robots in the market.  And it will cost too much for a mass user.  The second article contains a much improved version of "Just", which I traded on [real accounts](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 documentation: Account Properties") for 2 years.  But if you want, then you can discuss the sale of the robot from the third and fourth articles.

I'm sorry that you don't put your work up for sale, although at least I, as a programmer, have greater personal satisfaction seeing that an EA of mine works well, than selling an EA that is not very worth it.

Following your indications from the Word robot 50% file - V3.docx I am working on an EA for a single symbol on MT4. It has been difficult for me to understand all the complexities of the code, but I already have a version working, although without all the options that you mention in the code. The next thing I have to add is the opening of secondary series and being able to close series by progressively decreasing profits. Little by little I am moving forward. Thanks for your work and your explanations.

If you want, you can write to me at gerardocastanyo@gmail.com and tell me the price of your MT5 version, assuming it is already a stable version. Greetings

![Zhongquan Jiang](https://c.mql5.com/avatar/avatar_na2.png)

**[Zhongquan Jiang](https://www.mql5.com/en/users/coolsnake)**
\|
17 Nov 2021 at 03:40

looking foward to the new article.


![buniollo](https://c.mql5.com/avatar/avatar_na2.png)

**[buniollo](https://www.mql5.com/en/users/buniollo)**
\|
12 Feb 2023 at 15:28

This seems like an article i ve seen long time ago about wavelet fokker planck komolgorov smirnov and robbins monro. Intresting.


![Mauro Andrade](https://c.mql5.com/avatar/avatar_na2.png)

**[Mauro Andrade](https://www.mql5.com/en/users/maurosandrade)**
\|
9 Apr 2024 at 13:02

Hi Maxim, first of all, congratulations on the fantastic work.

it’s fascinating.

Would like to discuss purchasing the bot for articles 3&4.

Thanks in advance.

a

![Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://c.mql5.com/2/42/yandex_catboost__3.png)[Machine learning in Grid and Martingale trading systems. Would you bet on it?](https://www.mql5.com/en/articles/8826)

This article describes the machine learning technique applied to grid and martingale trading. Surprisingly, this approach has little to no coverage in the global network. After reading the article, you will be able to create your own trading bots.

![Useful and exotic techniques for automated trading](https://c.mql5.com/2/42/exotic.png)[Useful and exotic techniques for automated trading](https://www.mql5.com/en/articles/8793)

In this article I will demonstrate some very interesting and useful techniques for automated trading. Some of them may be familiar to you. I will try to cover the most interesting methods and will explain why they are worth using. Furthermore, I will show what these techniques are apt to in practice. We will create Expert Advisors and test all the described techniques using historic quotes.

![Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__2.png)[Prices in DoEasy library (Part 64): Depth of Market, classes of DOM snapshot and snapshot series objects](https://www.mql5.com/en/articles/9044)

In this article, I will create two classes (the class of DOM snapshot object and the class of DOM snapshot series object) and test creation of the DOM data series.

![Prices in DoEasy library (part 63): Depth of Market and its abstract request class](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__1.png)[Prices in DoEasy library (part 63): Depth of Market and its abstract request class](https://www.mql5.com/en/articles/9010)

In the article, I will start developing the functionality for working with the Depth of Market. I will also create the class of the Depth of Market abstract order object and its descendants.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/8859&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6429648776416133785)

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