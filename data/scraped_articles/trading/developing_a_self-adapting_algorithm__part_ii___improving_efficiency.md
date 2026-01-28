---
title: Developing a self-adapting algorithm (Part II): Improving efficiency
url: https://www.mql5.com/en/articles/8767
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:27:31.443941
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/8767&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6397511516120064752)

MetaTrader 5 / Trading


### Introduction

Before reading this article, I recommend that you study the first article " [Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)". This is not necessary, since the main point will still be clear, but reading will be more interesting.

In the previous article, I detected a simple pattern and developed a very simple algorithm that exploits it. But the algorithm has no flexible structure, and it makes no sense to expect any outstanding results from it.

We need to greatly improve it so that it becomes more flexible and adjusts its operation parameters depending on the market situation so that it is possible to achieve better results and stability.

### Analyzing drawbacks

Let's start the development of the new algorithm by analyzing the drawbacks of the previous version. I have highlighted the following drawbacks:

- Signals to open a series are too rare. Improving the signal quality greatly reduces the number of entry signals and the overall profit;
- A fixed sampling window is taken for analysis. While samples are set by a range, analyzing one sample of a fixed size is not a very efficient solution. The market is not a sine wave and "tails" affect the current signal quality. Sample boundaries should be fuzzy and they should influence the final decision. Roughly speaking, the previous version took 100 candles and analyzed the preponderance of bearish and bullish ones. If the excess passed the threshold, an entry signal was generated. The sample should be dynamic rather than fixed. It is necessary to be aware of both the situation in the analysis window and beyond;
- The issue of a fixed analysis window is not common to all well-known methods;
- The threshold percentage has a fixed value for each sample, regardless of the number of candles in it. This solution is ineffective, because the probability of a 75% excess in a sample of 30 candles and in a sample of 200 candles is far from the same. It decreases non-linearly with an increase in the number of candles in the sample;
- Opening positions on each candle is too costly and leads to increased drawdowns. Too many positions are often opened making it necessary to reduce the number of open positions in the series. This improves the capital use efficiency;
- Closing positions with a fixed profit per lot reduces stability or profitability. We need to find a compromise between stability and profitability in an ever-changing market. Without adjusting the parameter, the robot will soon suffer a loss simply because it will miss the optimal closing point for a position series;
- Severe limitation of the number of instruments for simultaneous trading reduces the overall system profitability. Series opening signals on different instruments slightly correlate with each other. Therefore, receiving a loss on one instrument may correlate with receiving a loss on another one. It is necessary to develop measures to reduce the correlation of signals, so that the number of simultaneously traded instruments can be significantly increased without a significant increase in the drawdown value.

### Simplified work algorithm

Let me remind you how the first version of the algorithm worked. In the new version, the work remains the same, but each step is to be revised and improved.

- scan a window of N candles;
- check which candles are prevailing - bearish or bullish;
- if the prevalence exceeds the threshold value, the signal to start a series of positions is activated;
- bearish candles are prevalent = Buy signal, bullish candles are prevalent = Sell;
- calculate the lot;
- open a new position on each subsequent candle till the series closure condition is triggered;
- the series closure condition is triggered;
- close all positions;
- search for a new signal.

### Modernization

The robot was developed in 2016 for MetaTrader 4. Its code  is attached below.

During the development, I am going to eliminate all identified drawbacks, so I will divide the entire development into separate tasks.

1. Dynamic threshold series start percentage

The algorithm of the first version became more stable with an increase in the number of candles in the analysis window or with an increase in the threshold percentage of the excess of bearish or bullish candles. I had to make a compromise and set a larger sample size for analysis or a larger percentage of the prevailing candle excess. The tests showed that the optimal excess percentage can be adjusted on almost any number of candles, but it should decrease for each larger number of candles.

In case of a fixed threshold percentage, an increase in the number of candles in the analysis window decreases the likelihood of such a combination. Therefore, we can set any maximum number of candles in the analysis window as this has almost no effect on the result simply because the probability of encountering such a combination is rapidly falling. To increase the number of series start signals, we need to reduce the threshold excess percentage so that the probability of the combination remains approximately the same with an increase in the number of candles in the sample.

I make an assumption that the probability of a given excess percentage on a given number of candles can be calculated using combinatorics.

![CP](https://c.mql5.com/2/41/CPP.PNG),

where

- С \- number of combinations
- n - number of candles in the sample
- k - number of bullish candles
- P - event probability
- P2 - doubled event probability

The event probability P should be multiplied by 2 because P is calculated for the case when there are more candles of the same direction - bearish or bullish. We are interested in the overall probability, regardless of the direction. There is no need to multiply the probability by 2 only if the number of bearish candles = number of bullish ones.

As an example, let's calculate the probability of an event when out of 30 candles there are 24 candles in one direction and 6 in another. To achieve this, I have prepared the table shown in Figure 1 below.

![probability table](https://c.mql5.com/2/41/i1rc9ws_zhfvhrlx0ntb_eng1.PNG)

Figure 1. Probability table

24 candles of one direction out of 30 ones correspond to the excess of 80%. The probability of such a combination is 0.11%. Now let's refer to the table to see the necessary excess percentage for 100 candles in the sample so that the probability of its occurrence is 0.11%. We can see that such a combination probability cannot be found for 100 candles. There are probabilities of 0.172% and 0.091%. I will choose a rarer option. It corresponds to the ratio of one type of candles to another one of 66/34, or 66% of candles in one direction.

Clearly, the combinations for 30 candles with the bullish/bearish candle excess of 80% occur as often as combinations of 100 candles with the excess of 66%. The dependence of the excess percentage on the number of candles is non-linear. Therefore, a non-linear function should be applied to adjust the percentage with an increase in the number of candles. I have developed such a function:

![Nonliner proc](https://c.mql5.com/2/41/Nonliner_proc.PNG)

where:

- Nb — number of candles in the sample.
- Koef\_NonLiner\_proc — ratio of the settings for adjusting the threshold percentage.

The Figure 2 graph shows how the excess percentage decreases (with a static probability of a combination occurrence) with an increase in the number of candles using the combinatorial method. It also allows evaluating the suitability of the developed function.

![Chart function](https://c.mql5.com/2/41/Chart_function.PNG)

Figure 2

The purple graph shows a decrease in the excess percentage while increasing the number of candles in the sample with a fixed probability of the combination occurrence. The red graph belongs to the excess percentage decrease function based on the number of candles in the sample. Both functions are non-linear, but the developed function decays more slowly. This is done deliberately because the more candles in the sample, the more positions in the series can be opened when returning to 50%. More positions mean greater load on the deposit and more drawdown.

It has been arranged so that the signal appears less often on a larger number of candles. Moreover, it happens only if the excess percentage is really high for a given number of candles.

> 2\. Improving signal quality

To improve the signal quality, the sample should not consist of a rigidly fixed number of candles. We can see that the excess of bullish candles is 65% on 30 candles, but how can we define whether it is a lot or a little? If there is also some kind of excess of bullish candles on 100 candles, then this fact should make the signal stronger, but if there is no excess or, conversely, there is an excess of bearish candles, then the signal should be weakened. I have developed two mechanisms of enhancing or weakening a signal.

> a) Applying a weighted average percentage. Find the weighted average percentage for all samples from the range set in the Min\_diap\_bar and Max\_diap\_bar parameters. Samples should be taken with some step. I prefer an even step of 2. The weighted average percentage will be determined for the type of candles that is larger in the first sample. If the first sample has more bullish candles, we should calculate the percentage of bullish candles in all other samples. The largest or the smallest sample can be made the first one. To do this, the Bigin\_pereb\_bar switch has been added to the settings. Weight ratios can be made using any function, but I have made them proportional to the number of candles in the sample.
>
> If the first sample has the smallest number of candles, then its weight ratio is W1, for the second sample, the weight is W2, and so on up to Wn.
>
> ![weighted average](https://c.mql5.com/2/41/ck87my983lk1n0jm.PNG)

> - W1 - weight ratio of the first sample;
> - W2 - weight ratio of the second sample;
> - Wn - weight ratio of the n th sample;
> - Nb1 - number of candles of the first sample;
> - Nbn - number of candles of the n th sample;
> - Pw - weighted average percentage of candle excess;
> - P1 - percentage of bearish/bullish candles of the first sample;
> - Pn - percentage of bearish/bullish candles of the n th sample;
> - Nbm - number of candles in the largest sample.

> If the first sample has the largest number of candles, weight ratios should be reversed
>
> ![ inverted weights](https://c.mql5.com/2/41/3p1rvqj4v3z5_h27v_h58.PNG)

> To get the series start signal, the weighted average percentage should be compared with a separate value specified in the Porog\_Weighted\_proc setting.

> b) Using multiple sampling. Several samples with an excess of one type of candles over the threshold value should be selected from the range. Here we will assume that the larger the number of samples, where the excess percentage is greater than the threshold value, the higher the signal quality. A non-linear percentage is to be used here as a threshold percentage. In other words, a separate excess threshold percentage should be applied for each sample with its own number of candles.

> The Kol\_vibor parameter sets the minimum number of samples, on which the excess should exceed the threshold value. The type of candles to define the excess percentage for depends on what candle type prevails in the first sample. A sample, on which an excess higher than the threshold value is detected, is considered the first one. I made it possible to iterate over the range from smallest to largest and vice versa to compare their work.

> This allows considering the signal in a wider window without being tied to a fixed analysis window. For example, the analysis window range is set from 30 to 300 candles with the step of 2. To form the signal, we need to collect at least 5 samples from the range with an excess greater than the threshold value. The excess may be formed on 30, 52, 100, 101 and 200 candles. Each sample is compared with its threshold percentage by the usage of a non-linear percentage. This allows us to evaluate the developments in the wide candle range more efficiently without binding to a fixed value.

> ![Price Chart](https://c.mql5.com/2/41/Price_Chart1.PNG)

> Figure 3
>
> An example is provided in Figure 3. We can see the area, in which bullish candles prevail and there is a signal to open Sell positions. This is not a clear number of candles, but an area.

> In the first article, I wrote that the number of candles (and, accordingly, the number of positions in the series), on which the resulting excess will theoretically be compensated, is calculated based on the number of candles, on which it appeared. Thanks to this approach, the first steps are taken towards the self-adaptation. The algorithm is still far from self-adapting, but the use of the current market parameters for a small adjustment of the operating parameters is already bringing it closer to the goal.

> A perfect algorithm should not contain configurable parameters. Each parameter should be accurately calculated based on the current market parameters. We need to know exactly what parameters should be set at each moment in time for the algorithm to remain profitable.

> 3\. Reducing the number of opened positions in the series

The first version of the algorithm had an issue: when the series started, the EA opened positions on each candle. This approach is not very efficient as it leads to large drawdowns and increases the requirements for the deposit. There were often situations when positions were opened, but the excess percentage in the sample continued to grow or fall not quickly enough. This unaccounted factor led to a loss of stability. We need to consider this factor and improve the algorithm stability.

I have developed two measures decreasing the number of positions in the series.

> a) In case of Sell positions, new positions are opened only on bullish candles. In case of Buy positions, positions are opened only on bearish candles. Such a mechanism allows us to more effectively earn profits from the fact that one type of candles prevails over another.

> Opening Sell positions on bearish candles leads to a decrease in the average position open price and to an additional load on the deposit if the price rises. At the same time, "extra positions" add almost no profit and do not accelerate closing of the series with a profit. The logic is similar for Buy positions.
>
> b) Opening with an excess percentage control. This is an additional filter for opening positions in a series. It is necessary to check the excess percentage on the entire sample with the appearance of each new candle. The sample size is increased by one when a new candle appears. A position can only be opened if the excess percentage increases when a new candle appears.
>
> This works almost like (a) but in a slightly different way. The idea is similar to to the one stated in (а): preventing a decrease of the average open price of a series for Sell positions and preventing an increase of the average open price of a series for Buy positions.
>
> If a weighted average is used as a threshold percentage, the method can adjust opening additional positions comparing the current weighted percentage value with the value on the previous candle.

These two items can be used both together and separately. They reduce profitability but significantly increase stability, which is more important to me. Thus, one unaccounted factor has been eliminated increasing the stability.

![open positions before](https://c.mql5.com/2/41/open_positions_befor.PNG)

![open positions after](https://c.mql5.com/2/41/open_positions_after.PNG)

Figure 4. Decreasing the number of positions in the series

In Figure 4, positions are opened without limitations, while in Figure below they are opened using the algorithm from (b). As we can see, fewer positions were opened. The comparative results for this deal are listed in the table below.

|  | Profit | Maximum loss | Number of positions | Profit factor |
| --- | --- | --- | --- | --- |
| Opening on each candle (upper figure) | +$71.99 | -$463.96 | 92 | 1.56 |
| Opening with the excess percentage control (lower figure) | +$42.26 | -$263.86 | 48 | 1.68 |

According to the table, the maximum drawdown has decreased and so has the profitability. Most importantly, the number of positions has decreased 1.91 times, which ultimately has a positive effect on the stability of the algorithm at the moments when the market parameters deviate from their typical values.

Other methods of decreasing the number of positions in a series were also developed. They are described in the requirements specification. I have shown the ones that seem the most efficient to me.

> 4\. Improving closing positions in profit

The first version of the algorithm introduces the profit per lot concept. This value was set in the deposit currency for the lot equal to one. Afterwards, the number of lots was calculated by open positions of the series, and the value from the settings was multiplied by the current total lot by open positions of the series. Suppose that the settings instruct closing a position series when the profit exceeds $15 per lot. Currently, we have 11 positions of 0.01 lot, which means the total position is 0.01\*11=0.11. Further on, the profit per lot is multiplied by the obtained number of lots $15\*0.11=$1.65. If the total profit on open positions reaches $1.65, the series of positions should be completed.

Setting a fixed amount of profit per lot in the deposit currency is not the most effective solution. When an instrument volatility decreases, the risk of missing the correct closing point increases. Conversely, when volatility rises, the robot loses profit. As a result, optimization yields an average profit per lot, which is not efficient enough.

The simplest solution is make the "profit per lot" parameter adjust based on the current volatility. The higher the volatility, the more profit per lot and vice versa. Applying the average volatility is not a perfect solution as well, but it is better than a fixed value.

This is another small self-adaptation function, the parameter is not rigidly configured, but the dependence on the current market state is set. It is known that the profit in a series directly depends on the size of the candlesticks, and this dependence should be used to increase stability.

I decided not to manage profit in the deposit currency because the point price is not constant for currency pairs like USD/XXX and changes from the current price value. The profit is to be managed in points. To do this, take the current value of the ATR indicator (in points) and multiply it by the Koef\_multi\_ATR value. The result is the number of points to be closed with profit. Next, the number of points passed from the open price to the current value is calculated and the average value for all positions is found. The obtained average value is compared with the number of points for closing with profit. If the average number of points exceeds or is equal to the number of points for closing, the series is complete. If not, the procedure is repeated on the next candle.

The current profit should be monitored on each tick or by the timer. To avoid excessive calculations, it is reasonable to check the profit by timer once per second or even once per 10 seconds for a system of this kind.

> 5\. Working with multiple trading instruments

The previous EA version was able to trade 10 instruments simultaneously. This was insufficient, so several EA instances had to be launched at once. The new version can simultaneously trade 28 instruments for better control of trading.

As I mentioned earlier, the series start signals on different instruments slightly correlate. This leads to synchronization of drawdowns for different instruments, an increase in deposit requirements and a decrease in relative profitability.

Ideally, the correlation between the series start signals on different trading instruments should be negative. The large current drawdown on one instrument should coincide with small profitable deals on any other instrument. Disabling simultaneous opening of series on more than one instrument is also not the best solution since it decreases the overall profitability.

Correlation of the new series start signals occurs due to the fact that one of the currencies may start to fall or rise in relation to others. A signal to buy or sell a currency in relation to other currencies may appear on several pairs at once. So, we should protect ourselves from such a situation.

To reduce the correlation between the series start signals on different instruments to the minimum possible degree, we need to divide currency pairs into separate currencies and assume that positions are opened in separate currencies. If a Sell position is opened on EURUSD, it is divided into a Sell position for EUR and a Buy one for USD. Before the start of each new series, we need to check the presence of positions on currencies forming the pair. If there is a Sell position on EUR, we should disable launching any series where EUR is sold. However, we should not disable signals that require buying EUR.

![](https://c.mql5.com/2/41/kepaek.PNG)

Figure 5

Dividing currencies into pairs is shown in Figure 5. Figure 6 below shows what positions can be opened if Buy EURUSD is opened. For other options, all works the same way.

![Currencies 2](https://c.mql5.com/2/41/ompgfo2.PNG)

Figure 6

With this approach, it is no longer necessary to limit the maximum number of instruments for simultaneous opening of positions. Nevertheless, I have left the function in the settings.

> 6\. Lot correction based on the current volatility

In the second version of the algorithm, I abandoned auto refinancing, therefore the lot does not change when changing the deposit size. I have developed this EA for real trading, and I do not need this function. Instead, I have developed another lot correction function. Profit and loss of such an algorithm depend on the candle size or volatility. It would be logical to make the profitability graph as flat as possible, without dips and spikes.

To stabilize the profitability graph, we need to change the lot size based on the current volatility. The stronger it is, the less we need to use the lot, the lower the volatility, the larger may be the lot. The lot changes in proportion to the current volatility. To achieve this, the normal lot and volatility are defined in the settings.

To calculate the currnt lot, let's take the current ATR value and divide it by Norm\_ATR from the settings. After that, divide Lot from the settings by the resulting ratio. Round off the obtained value to the correct one. This is how the lot decreases with the growth of candles, while the profitability graph remains as stable as possible.

After the start of the series, the volatility may change, therefore I have introduced two options. In the first one, the lot is defined before the series start and remains stable till the series is over.

In the second option, the lot changes from the current volatility after the series start. In this case, when calculating the current profit, positions affect the total profit in the deposit currency either improving or worsening the results. The function is experimental. If you like how it works, I can modify it in the next version.

### Tests

The article describes only the most basic and interesting modifications and operating modes. In reality, much more has been implemented, and all modes can be combined with each other. The requirements specification for the algorithm with all the details is attached below.

I am going to run the tests on the same currency pairs I used to test the first version of the algorithm in order to visually highlight the differences. Like the first version, this algorithm works by closing candles, so you can safely test it in the "control points" mode. Inside the candle, it only controls the current profit and makes sure that the current funds do not fall below the threshold value defined in the settings. As before, the tests will be carried out with an overestimated spread. I will set a spread of 40 for GBPUSD.

Like in the first algorithm version, we can trade and optimize any timeframe. The minimum timeframe is limited by the size of candles relative to spreads and commissions. The lesser the timeframe, the higher the requirements to the signal quality and expected payoff. As the timeframe increases, the size of the candles grows and, accordingly, the drawdown level rises as well. Therefore, the maximum timeframe is limited by trading style preferences.

I have performed optimization in 2017 when using the robot for trading on real accounts. Therefore, I simply took the previous settings without performing a new optimization.

![GBPUSD 2000 tester chart](https://c.mql5.com/2/41/GBPUSD_2000_Tester_Chart.PNG)

![GBPUSD 2000 Tester report](https://c.mql5.com/2/41/GBPUSD_2000_Tester_report.PNG)

Figure 7. GBPUSD H1 2000.01.01 - 2020.12.08, static lot

Figure 7 displays the test of GBPUSD H1 for the period of almost 21 years since 2000.01.01 up to 2020.12.08. The multiple sampling from the range is used here. The analysis range is 68-200 candles. 15 samples are used.

If the first algorithm version passed the test only from 2001, the second one easily passes it since 2000. Compared to the first version, the number of positions has increased 3 times, while the profit has increased 1.9 times. The profit factor has decreased from 7.5 to 2.74, but still remains at a decent level. The series start signals are generated more often, while the average number of positions in the series has decreased. Probably, we can find better settings, but I have taken the ones I used for trading.

The lot adjustment function from the current ATR value has been developed previously. Figure 8 shows the same test as in Figure 7 but with a dynamic lot based on volatility. Since the lot of 0.01 was used for $3000, I have increased the lot adjustment algorithm up to $30,000 increasing the lot up to 0.1.

![GBPUSD 2000 tester chart dyn lot](https://c.mql5.com/2/41/GBPUSD_2000_Tester_Chart_dyn_lot.PNG)

![GBPUSD 2000 Tester report dyn lot](https://c.mql5.com/2/41/GBPUSD_2000_Tester_report_dyn_lot.PNG)

Figure 8. GBPUSD H1 2000.01.01 - 2020.12.08, dynamic lot depending on ATR

As we can see on Figure 8, the profitability graph has become significantly more linear, as expected during development. At the same time, the profitability has slightly dropped, while the maximum drawdown has increased a bit. The mode has turned out to be interesting. It will be useful for obtaining the maximum predicted profitability with high Sharpe ratio values.

We need to check how the stability of the parameters I used in the above tests. To achieve this, I will launch the test on GBPUSD M15 with the same settings I used for H1. Since the volatility unevenness is much greater on smaller timeframes, the profit ratio should be slightly reduced. Since I know why this should be done, this parameter should be made self-adapting. However, the current EA version has no such feature, so I adjust it manually.

![GBPUSD M15 tester chart](https://c.mql5.com/2/41/GBPUSD_2009_m15_Tester_chart.PNG)

![GBPUSD 2009 m15 Tester report](https://c.mql5.com/2/41/GBPUSD_2009_m15_Tester_report.PNG)

Figure 9. GBPUSD M15 2009.01.01 - 2020.12.08

Figure 9 shows the test of GBPUSD M15 2009.01.01 - 2020.12.08 with the H1 settings. The test has been passed consistently since 2009. The result is good considering that the timeframe settings were not specifically optimized for M15. To find the optimal settings for M15, it is enough to optimize the most difficult section for 1.5 years from 2008.01.01 to 2009.06.01. If we optimize the parameters for this section, the EA will pass the test for 21 years without any problems.

For the first version, I showed the tests for EURUSD H1. Let's compare the results. Figure 10 shows the result for EURUSD H1 for the period from 2000.01.01 to 2020.12.08.

![EURUSD 2000 tester Chart](https://c.mql5.com/2/41/EURUSD_2000_tester_Chart.PNG)

![EURUSD 2000 tester report](https://c.mql5.com/2/41/EURUSD_2000_tester_report.PNG)

Figure 10. EURUSD H1 2000.01.01 - 2020.12.08

The first version passed the backtest only since 2007, the new version passes the test since 2000 already. The stability has improved significantly. At the same time, the maximum drawdown has decreased 1.2 times and the profit has increased 3 times. The number of deals has increased 2.3 times.

Since we want to figure out how much better the second version has become, let's see the test on GBPJPY H1. The first version was tested since 2009.

![GBPJPY 2000 tester chart](https://c.mql5.com/2/41/GBPJPY_2000_tester_chart.PNG)

![GBPJPY 2000 tester report](https://c.mql5.com/2/41/GBPJPY_2000_tester_report.PNG)

Figure 11. GBPJPY H1 2000.01.01 - 2020.12.08

As we can see in Figure 11, the algorithm now passes backtests since 2000. The drawdown has decreased 1.8 times, while the profit factor has increased up to 3.3.

I have optimized 28 currency pairs consisting of eight major currencies USD, GBP, EUR, CHF, CAD, AUD, NZD and JPY. Some currency pairs show the best result, while some show the worst, but all of them are optimized well and usually pass the tests since 2004. Some currency pairs pass the test since 2007.

MetaTrader 4 is unable to perform tests on several instruments at once. Therefore, I conducted tests on each of 28 symbols separately and used a third-party software to combine the reports. Since I did it in 2017, the combined report covers the period from 2010 to 2017.

![Full report](https://c.mql5.com/2/41/small_report.PNG)

Figure 12. Combined backtest for 28 currency pairs from 2010 to 2017

Figure 12 shows that the profitability graph is fairly flat with a yield of about 100% per annum. The result is quite impressive, but in reality, the profitability is lower due to limitation of simultaneous trading on several currency pairs. I used this robot for trading on real accounts for two years, and the real profitability was at 56% per annum.

The real profitability is lower than the calculated one because the test was conducted with inflated spreads. The algorithm works in such a way that the higher the spread, the higher the profitability. This can be explained by the fact that the higher the spread, the lower the stability. Besides, in reality, the effect of a decrease in the number of signals due to the prohibition of unidirectional deals is also evident.

While trading, I used the most conservative settings. However, it is possible to make the algorithm act much more aggressively.

### Additional functions

I have added the ability to sort series start signals by other timeframes. For example, trading may be performed on H1, but signals are additionally checked on all standard timeframes. The series start signal is formed only if M1, M5, M15, M30, H1, H4 and D1 timeframes also have such a signal. The series confirmation signal using additional timeframes is generated using the same rules and in the same mode as on the main timeframe. Additional timeframes can be enabled or disabled independently of each other.

Unfortunately, I was unable to check the mode as there were issues with the correct function operation.

The EA features many additional functions I have not considered because they are not as interesting as the basic mechanics. These are the following:

- The ability to set take profit and stop loss. The values are calculated according to a specific algorithm and can duplicate real closing points or work independently.
- Added the current funds management function. If funds fall below the set threshold, the EA closes positions and stops trading
- Added the ability to limit the maximum number of positions for a single symbol or all symbols.
- Added the ability to manage the current loss on each trading symbol. This is an additional virtual stop loss
- We can set a pause between two adjacent series in the number of candles so that the series does not open immediately after the previous one is closed. This function allows starting series on other instruments during a pause.
- We are able to set the minimum ATR value, at which new series can start on an instrument. If the volatility is low, then trading may not be profitable.
- Non-linear profit may be used to close positions. The more positions are opened in the series, the more desirable it is to complete it as fast as possible. Therefore, the threshold profit falls in proportion to the square root of the number of open positions.
- The screen displays all the necessary information for each traded instrument, so that you can follow the EA logic.
- The algorithm handles PC and terminal resets, picks up its open positions and continues trading according to the current situation using global variables for that.

The EA features 2337 settings for 28 trading instruments so all users are able to find an operation mode they are interested in.

### Conclusions and further development

- I have used a simple but understandable pattern as a basis significantly improving the algorithm characteristics during the development. This has become possible because the pattern used is very clear and you can study what exactly affects profit and loss.
- The algorithm is optimized well for a variety of instruments and is quite stable over long periods of time.
- The operation parameters have become more flexible and can already be slightly adjusted depending on the current market parameters. This feature has made it possible to significantly increase stability. The algorithm can be improved further improving its operation quality.
- There is a significant potential for improving the algorithm parameters. The logic can be inverted for both trend and counter-trend trading.
- Each instrument requires optimization. This is a drawback indicating the lack of a full-fledged theoretical model showing what parameters are needed at a certain time for a certain instrument and why.
- The need for optimization for each instrument means we do not know the differences between price series of different trading instruments. Any uncertainty gives rise to unaccounted factors.
- If we do not know how the parameters of the price series of one instrument differ from another, then there is no way to control the presence of a pattern on the current instrument. We cannot say exactly which parameters of the price series affect the profitability and stability of the algorithm.
- It is not clear why the optimized settings are stable on one instrument and unstable on another. It is possible to choose settings that are stable on all trading instruments, but this leads to a significant drop in profitability.
- We cannot get rid of unaccounted factors while optimizing parameters on historical data. It is the number of unaccounted factors that affects stability.
- To achieve real stability, it is necessary to significantly revise the theory and abandon optimization in the next algorithm versions.
- Each parameter within the settings should depend on something. There should not be a single parameter that is set simply because EA works better with it. We need a detailed explanation of why the parameter is used now and when it should change its value.


In the next article, I will continue the algorithm development and significantly revise the theoretical model. The new algorithm is to be designed for MetaTrader 5 since it is a more powerful, flexible and efficient platform.

The EA code, requirements specification, test settings, an example of settings for 28 currency pairs and the Excel table for calculating the event probability are attached below.

The author of the idea and requirements specification is[Maxim Romanov](https://www.mql5.com/en/users/223231). The code has been written by [Vyacheslav Ivanov](https://www.mql5.com/en/users/viper70).

### Previous articles on the topic:

[Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8767](https://www.mql5.com/ru/articles/8767)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8767.zip "Download all attachments in the single ZIP archive")

[robot.zip](https://www.mql5.com/en/articles/download/8767/robot.zip "Download robot.zip")(43.21 KB)

[Set.zip](https://www.mql5.com/en/articles/download/8767/set.zip "Download Set.zip")(152.63 KB)

[technical\_task.zip](https://www.mql5.com/en/articles/download/8767/technical_task.zip "Download technical_task.zip")(67.45 KB)

[probability\_table.zip](https://www.mql5.com/en/articles/download/8767/probability_table.zip "Download probability_table.zip")(15.13 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-adapting algorithm (Part IV): Additional functionality and tests](https://www.mql5.com/en/articles/8859)
- [Self-adapting algorithm (Part III): Abandoning optimization](https://www.mql5.com/en/articles/8807)
- [Developing a self-adapting algorithm (Part I): Finding a basic pattern](https://www.mql5.com/en/articles/8616)
- [A scientific approach to the development of trading algorithms](https://www.mql5.com/en/articles/8231)
- [What is a trend and is the market structure based on trend or flat?](https://www.mql5.com/en/articles/8184)
- [Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/363155)**
(110)


![Mahir Zukic](https://c.mql5.com/avatar/2017/9/59AEB033-FD0C.jpg)

**[Mahir Zukic](https://www.mql5.com/en/users/mahirzukic2)**
\|
14 Apr 2021 at 00:19

**SysFX:**

This topic is certainly interesting, and it's clear that a lot of time has gone into the [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly"), but the code presented generates a large number of warnings.

Having reviewed the code, there appeared to be two specific bugs which have now been corrected. However, it may be necessary to re-optimise as the behaviour of the EA will have changed slightly.

The attached files now compile with zero warnings.

I am having issues with running both the original version of the code as well as your with fixed bugs.

I have tried a simple EURUSD with the default parameter values, I have only set the \`Only\_one\_symbol\` to true. Afterwards, I have ran it as false. The results were the same. No trades at all.

I used the 2021.01.01 to 2021.03.16 time interval for all tries. I even tried extending it to 2020.01.01 to 2021.03.16, and still got the same result, no trades at all, it just took way longer.

Do you know how should this be run? And how to do optimization for example on a specific pair, e.g. NZDCAD?

![Maxim Romanov](https://c.mql5.com/avatar/2025/11/691b0e3f-04d1.png)

**[Maxim Romanov](https://www.mql5.com/en/users/223231)**
\|
14 Apr 2021 at 11:44

**Mahir Zukic:**

I am having issues with running both the original version of the code as well as your with fixed bugs.

I have tried a simple EURUSD with the default parameter values, I have only set the \`Only\_one\_symbol\` to true. Afterwards, I have ran it as false. The results were the same. No trades at all.

I used the 2021.01.01 to 2021.03.16 time interval for all tries. I even tried extending it to 2020.01.01 to 2021.03.16, and still got the same result, no trades at all, it just took way longer.

Do you know how should this be run? And how to do optimization for example on a specific pair, e.g. NZDCAD?

Download the files attached to the article. There is a file for EURUSD, apply it to the EA and install the EA for the [EURUSD pair](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: techical analysis"). Everything should work now. To use another pair in the tester, you need to specify its first one in the settings. The steam in the tester must match the one indicated in the settings. For example, NZDUSD in the tester, in this case NZDUSD should be in the settings. This is for MT4.

```
If the advisor did not make deals on the specified dates, increase the testing period, perhaps there were no signals
```

![Mahir Zukic](https://c.mql5.com/avatar/2017/9/59AEB033-FD0C.jpg)

**[Mahir Zukic](https://www.mql5.com/en/users/mahirzukic2)**
\|
17 Apr 2021 at 02:02

**Michele Catanzaro:**

Ok thanks a lot for your reply, for now I have commented the other 27 currency pairs to test it just on 1.

Hey Michele. Can you share the EA which you converted into MT5? I have tried to do it via \`mq4.mqh\` file to bridge that gap by still using the MT4 code and only changing minor stuff to get rid of compilation errors, but when I run the EA, nothing happens. No trades.

I would really appreciate it.

EDIT: after having a single run finish after some time (about 10 minutes) for 2018 - 2021 period for EURUSD for the same settings as used in MT4, I get about 20 or so trades in MT5, whereas I would get 1000 - 2000 trades in MT4. Also, MT5 is orders of magnitude slower for the same period, which takes about 10 minutes, while MT4 about 20 - 30 seconds. All other settings were the same, both for the EA as well as backtesting, period (2018 - 2021), timeframe (H1), symbol (EURUSD) and modelling ( [Open prices only](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ")).


![Mahir Zukic](https://c.mql5.com/avatar/2017/9/59AEB033-FD0C.jpg)

**[Mahir Zukic](https://www.mql5.com/en/users/mahirzukic2)**
\|
22 Apr 2021 at 15:13

Anybody? With the MT5 version?


![Tommy Jake](https://c.mql5.com/avatar/avatar_na2.png)

**[Tommy Jake](https://www.mql5.com/en/users/tommyjake)**
\|
31 May 2021 at 10:55

Very informative


![Neural networks made easy (Part 10): Multi-Head Attention](https://c.mql5.com/2/48/Neural_networks_made_easy_0110.png)[Neural networks made easy (Part 10): Multi-Head Attention](https://www.mql5.com/en/articles/8909)

We have previously considered the mechanism of self-attention in neural networks. In practice, modern neural network architectures use several parallel self-attention threads to find various dependencies between the elements of a sequence. Let us consider the implementation of such an approach and evaluate its impact on the overall network performance.

![Brute force approach to pattern search (Part III): New horizons](https://c.mql5.com/2/41/0964a7dfa2f4664c34e6be839022c67a.png)[Brute force approach to pattern search (Part III): New horizons](https://www.mql5.com/en/articles/8661)

This article provides a continuation to the brute force topic, and it introduces new opportunities for market analysis into the program algorithm, thereby accelerating the speed of analysis and improving the quality of results. New additions enable the highest-quality view of global patterns within this approach.

![Practical application of neural networks in trading (Part 2). Computer vision](https://c.mql5.com/2/42/neural_DLL.png)[Practical application of neural networks in trading (Part 2). Computer vision](https://www.mql5.com/en/articles/8668)

The use of computer vision allows training neural networks on the visual representation of the price chart and indicators. This method enables wider operations with the whole complex of technical indicators, since there is no need to feed them digitally into the neural network.

![Finding seasonal patterns in the forex market using the CatBoost algorithm](https://c.mql5.com/2/41/yandex_catboost__3.png)[Finding seasonal patterns in the forex market using the CatBoost algorithm](https://www.mql5.com/en/articles/8863)

The article considers the creation of machine learning models with time filters and discusses the effectiveness of this approach. The human factor can be eliminated now by simply instructing the model to trade at a certain hour of a certain day of the week. Pattern search can be provided by a separate algorithm.

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/8767&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6397511516120064752)

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