---
title: Developing a self-adapting algorithm (Part I): Finding a basic pattern
url: https://www.mql5.com/en/articles/8616
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:48:15.677994
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=cgsaxndtwgxjtxpdpcrecdhwpwhrrmcp&ssn=1769093294040543443&ssn_dr=0&ssn_sr=0&fv_date=1769093294&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8616&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20self-adapting%20algorithm%20(Part%20I)%3A%20Finding%20a%20basic%20pattern%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909329444772285&fz_uniq=5049391053649324717&sv=2552)

MetaTrader 5 / Trading


### Introduction

Any trading algorithm is generally a tool which may bring profit to an experienced trader or instantly destroy a deposit of an inexperienced one. The issue of creating a profitable and reliable algorithm is that we cannot understand what needs to be done in order to earn money and what methods are used by "successful traders". While HFT, arbitrage, option strategies and calendar spread-based systems boast a solid theoretical basis clearly stating what needs to be done to make profit, the algorithms based on price analysis and fundamental data are much more ambiguous. This area has no full-fledged theoretical basis that would describe pricing making it extremely difficult to create a stable trading algorithm. Trading turns into art here, while science helps systematizing everything.

But is it possible to create a fully automated trading algorithm based only on the analysis of price changes working on any trading instrument without optimization and with no need to manually adjust parameters for each trading instrument separately? Is there an algorithm you can simply apply to a necessary trading instrument chart so that it immediately defines profitable parameters for it?

Is it possible for an algorithm not to lose regardless of market changes? I believe, it is possible, albeit difficult to achieve. There are two ways to solve this issue:

1. Use neural networks with machine learning and get a "black box" analyzing the market and making profit following its own criteria. Despite the apparent obviousness and attractiveness of this approach, it has its own complexities. If it were so easy to do, then such companies as Google and Yandex, with their vast experience in using machine learning and large budgets, would have solved this problem long ago, or at least they would have been among the leaders in the analysis of market data and the quality of forecasts. I believe, it is possible to create such an algorithm. It should be constantly refined as technology develops in order to remain competitive. After all, you will have to compete with tech giants.
2. Study the pricing patterns, develop a theoretical model of price formation and create an algorithm creating as much factors affecting pricing as possible. In my opinion, this method is even more complicated than machine learning, but I think it is real as well. Such an algorithm will improve along with the development of the theoretical background and developer's knowledge. It will also require constant modernization.

I decided to use the second method, namely create a full-fledged theory and develop an algorithm based on it. I am going to develop the algorithm in a series of articles moving from a small pattern to a large project. Besides, I will show how to find and study patterns while improving work efficiency.

The first algorithm will be developed only for FOREX and MetaTrader 4 terminal. Along the way, it is to evolve into a universal MetaTrader 5 algorithm working both in FOREX and exchanges.

### Basic pattern

Let's start from searching for the basic pattern that will form the basis of the algorithm. This pattern should be fundamental and inherent in most markets in order to easily scale it to any market in the future. The scalability for any market suggests that the algorithm is truly self-adapting.

To begin with, let's try to define why it is difficult to earn consistently in the market. It is all about direct competition. The task of some traders is to conclude a profitable deal with others, but the problem is that each of the parties wants to conclude a profitable deal and someone definitely gets a loss. Therefore, the market should not have stable simple patterns that everyone can notice.

In my first article ["Price series discretization, random component and "noises"](https://www.mql5.com/en/articles/8136), I wrote about a random component in the market and the fact that the distribution of increments for a price series is very similar to the distribution of increments for a random walk. At the first glance, the price series seems to have no patterns, everything happens randomly and it is impossible to make stable money on this.

Therefore, as a basic rule, I took that there are exactly the same number of rising candles for any instrument as there are falling ones. This is a very simple pattern, and any deviation from it leads to stable earnings for everyone noticing it. I think, many will notice it since it is so simple. But as we know, it is impossible for everyone to make money, so there can be no significant alteration of the number of falling candles from the number of growing ones in the long term. This is implicitly stated in the mentioned article. Theoretically, there can be deviations from this rule, because I did not check all the tools, but only about a hundred of them. However, it is not so important at this stage.

For example, let's take 100,000 GBPUSD H1 candles for any period and calculate how many of them are rising and how many are falling. I took the period of 100,000 candles from 2020.10.27 to the past. The period contains 50,007 bullish and 48,633 bearish candles. For the remaining candles, the opening price is equal to the closing one. In other words, GBPUSD features 50.7% bullish and 49.3% bearish candles, the remaining candles have Open = Close. The values are roughly equal. If we take a larger sample, the result will tend to 50%.

I will not measure every instrument within this article. I observed hundreds of instruments and the results were similar everywhere. But to be convincing, let's take an instrument from a completely different market GAZP (Gazprom) M15 and measure the number of falling and growing candles on 99,914 candles. The result is 48,237 bullish and 48,831 bearish ones. The remaining ones are neutral. As you can see, the number of bearish and bullish candles is approximately equal.

This pattern is not only present on a large number of instruments, but also remains on any timeframe. To make sure of this, let's analyze GBPUSD M1 and take the steadily growing AAPL stock on M15. I combined all the data into a table.

| Symbol | Timeframe | Total candles | Total bullish candles | Total bearish candles |
| --- | --- | --- | --- | --- |
| GBPUSD | M1 | 100,000 | 50.25% | 49.74% |
| GBPUSD | H1 | 100,125 | 50.69% | 49.30% |
| GAZP | M15 | 99,914 | 49.69% | 50.30% |
| AAPL | M15 | 57,645 | 50.60% | 49.40% |

According to the table, the ratio of rising to falling candlesticks tends to 50% on the considered instruments and is valid for all considered timeframes. The XLSX file containing data used in my measurements, as well as the results, is attached below.

The results shown in the table confirm the conclusion that the statistical parameters of the price series over a large number of samples are very similar to the random walk parameters. Moreover, on a small number of samples, there will be deviations from the reference distribution. The algorithm will be based on the assumption that the distribution of the increments for the price series is only superficially similar to normal.

On a small number of samples, the distribution shape differs from the reference one, while with an increase in the number of samples, the shape tends to the reference one. Most importantly, the speed, with which the distribution tends to the reference one, is individual for each instrument and is quite stable.

![distribution](https://c.mql5.com/2/41/37_h7s0i.PNG)

Figure 1

Figure 1 visualizes all that has been described above. The reference distribution of increments is shown in red, while the distribution of increments for a price series with a large number of samples is shown in black. The distribution of price series increments on a small number of samples is shown in violet, green and blue.

The fundamental regularity inherent in a large number of trading instruments has been found and an assumption where to look for a profit has been made. The number of bearish candles = the number of bullish ones, and when a deviation from the equilibrium occurs, the rate of return to the equilibrium for each instrument is individual, stable and fluctuates within certain limits. I will use this pattern to make a profit.

### Developing a test algorithm

Now we need to develop the simplest algorithm making use of this pattern and test it to understand whether it is worth developing the theory further. The local deviation of the number of bearish and bullish candles from 50% is to be used for trading. On average, the number of bearish and bullish candles is approximately the same. However, the preponderance of one type of candles over another one can differ from 50% greatly tending to equilibrium afterwards. The algorithm works as follows:

General simplified algorithm

- scan a window of N candles;
- check which candles are prevailing - bearish or bullish;
- if the prevalence exceeds the threshold value, the signal to start a series of positions is activated;
- bearish candles are prevalent = Buy signal, bullish candles are prevalent = Sell;
- calculate the lot;
- open a new position on each subsequent candle till the series closure condition is triggered;
- the series closure condition is triggered;
- close all positions;
- search for a new signal.

The algorithm applies market orders for simplicity. The algorithm uses close prices only, and all actions are performed after a new candle is formed. There is no need to analyze anything inside the candle. We only need to control the current profit and funds. Any other calculations are unnecessary.

Position open signal

1. Analysis window. Set the number of candles (analysis window) to check the current percentage of bearish and bullish candles. Analyzing a fixed number of candles is not a good idea, because there may be longer or shorter sections with deviations of 50% or more. Set the analysis window from the minimum to the maximum number of candles with the priority of a larger number of candles since a small area can be part of a large one, but you need to capture the entire range. The analysis window is set by the MinBars and MaxBars parameters. The algorithm searches for the preponderance of bearish or bullish candles in the range from MinBars to MaxBars. As soon as it is found, the algorithm selects the maximum number of candles with an excess of bearish or bullish candles above the threshold.
2. The threshold excess percentage for opening a position. We need to define what excess percentage of bearish or bullish candles can be considered a threshold for signal generation. This parameter can be selected by optimization. The settings feature the OpenPerc parameter. It is set in percentage and is the percentage value for opening. If the percentage of bearish candles in the sample is greater than or equal to the opening percentage, open Buy. If the percentage of bullish candles in the sample is greater than or equal to the opening percentage, open Sell.
3. Range enumeration. We need to analyze candles from the range with some step. An odd step is not suitable since it inevitably features a deviation from 50% providing false entry signal, therefore the step for checking the range can be found in the settings (Step parameter). The range enumeration moves from the smallest to the largest.

Maximum number of positions in a series

During trading, a position is filled with a series of orders, so we need to limit the maximum number of positions. To do this, we need to calculate the number of candles, at which the excess can be straightened up to 50%. We use the sample that triggered the start of the series (consisting of N candles), calculate the number of dominant candles, subtract the remaining candles from it, add it to the number of candles in the initial sample and multiply by the К adjusting factor. The К ratio is indirectly linked to the statistical characteristics of the instrument.

This is how we obtain the expected number of candles, at which the deviation from 50% should disappear.

```
R=(N+NB-NM)*K
N - number of candles in the sample featuring an excess above the threshold
NB - number of dominant candles
NM - number of remaining candles
K - adjusting factor defined in the settings
```

Next, we need to calculate the maximum number of positions the robot is to open. To achieve this, we need to subtract the number of candles in the initial N sample from the total number of candles R. The result is the maximum number of positions E the robot should not exceed in the series.

```
E=R-N
Е - maximum number of positions in the series
R - total number of candles, at which the return to 50% is planned
```

This algorithm is not self-adapting yet, but the first steps are taken and the number of positions is adjusted from the number of candles in the initial sample.

Lot selection

The algorithm has been created to check the pattern. Therefore, we need to add several settings to determine the lot. Suppose that the lot size depends on the expected number of positions in the series. The more positions in the series, the smaller the lot. To do this, add two settings: Depo and RiskPerc. For simplicity, let's assume that $1000 = lot ($500=0.5 lot). Take the value from the Depo setting and multiply it by RiskPerc. Divide the obtained value by the maximum number of positions E and round it to the nearest correct value. This is how we get the lot size of one position.

We need to add the lot auto increase function if the deposit grows. In that case, if Depo = 0, the current deposit value is multiplied by RiskPerc and the lot for opening positions is found using the obtained value.

Closing positions

Closure occurs in several cases:

> 1) The total profit on open positions reaches the values set in CloseProfit. Since the number of positions opened by the EA is not constant and the lot size may change, it would be incorrect to set the fixed profit in USD. We need the USD profit depend on the current number of open positions and the total lot in the market. To do this, we need a concept of "profit per lot".

The CloseProfit setting sets the profit for the total position of 1 lot, and the robot re-calculates the profit for the number of lots it opened. If we have 10 open positions at 0.01 lot each, the total lot is 0.1. So, in order to get the profit value for closing a position, Profit = CloseProfit \* 0.1. Adjust the profit to close positions each time the number of positions changes. As a result, when the total profit becomes greater than or equal to the calculated Profit value, the EA closes positions.

> 2) When the current excess percentage becomes less than or equal to ClosePerc. ClosePerc sets the excess percentage of prevailing candles to generate a signal for closing the series of positions. When the first position was opened, the excess was found on the N number of candles. Now, with each new candle, the number of candles increases becoming N+1; N+2; N+3 etc... With each new candle, we need to check the current excess percentage of prevailing blocks on a sample with the number of candles increasing by 1. Complete the series after the condition is met.

> 3) Upon reaching MinEquity. If the current funds have fallen below the set value, you need to complete the open series and avoid opening new ones until the funds increase. This is the Stop loss function protecting the deposit from losses.

### Test

The algorithm is quite primitive. It is unable to adapt to changing market conditions and is used only to check the idea viability. Therefore, I will select the settings using optimization. The optimization is to be performed by enumerating all options with no genetic algorithm. The EA was developed back in 2014 and made for MetaTrader 4. I will first test it on GBPUSD H1. I will set the artificially high spread of 40, so that the optimization is performed in non-perfect conditions to ensure a certain margin of stability in the future. An increased spread is needed because the EA controls the current profit on open positions and is influenced by the spread.

I will optimize only three conditions: the minimum number of candles MinBars, percentage for opening positions OpenPerc and profit per lot CloseProfit. Presumably, the larger the minimum number of candles for analysis and the higher the opening percentage, the more reliable, but less frequent, the signals will be. CloseProfit indirectly depends on volatility. Therefore, before optimization, you need to look at the volatility of the instrument and set an adequate range. The optimization is to be performed for the year from 2019.11.25 to 2020.11.25 on H1.

![optimization GBPUSD](https://c.mql5.com/2/41/3kzkh7o455h_GBPUSD_neb6uzi7l.PNG)

Figure 2. Optimization results

The brute force optimization is necessary to see how the parameters affect the result and how the results converge with the assumptions about how the algorithm should work in theory. Figure 2 shows some of the results. We need to enable sorting by maximum profit and select the settings having the most adequate indicators of drawdown and profitability. The highlighted settings seem to be fine.

The further analysis of the optimization results shows that an increase in MinBars and OpenPercent reduces profitability and drawdown. Next, I took the highlighted parameters, tested them within a year, obtained results and defined what changes if MinBars and OpenPercent are increased/decreased. My conclusion is that the number of deals decreases and the signal reliability increases along with an increase of MinBars and OpenPercent. This means, we need to find the balance between profitability and drawdown.

For trading, I took deliberately more conservative parameters to ensure a margin of safety for changing market conditions. Figure 3 below shows how the EA opens positions.

![trade](https://c.mql5.com/2/41/trade_gbpusd__1.PNG)

Figure 3. Opening positions

Figure 3 shows that a position is made of several deals if necessary. The entry signal is stretched over time. The first version of the algorithm features a rough signal considerably stretched in time at some points. It looks more like a fuzzy input rather than a standard averaging. This is an area with fuzzy boundaries, in which bearish candles are more likely than bullish ones. It is possible to make profit from this probability.

The algorithm remains reliable at MinBars = 70, but I set it to 80 so that there is a margin for fluctuations in the trading instrument characteristics. The logic is similar when selecting the CloseProfit parameter. In the example, it is equal to 150. In case of a lesser value, the algorithm becomes more stable but the profit is reduced. If increased up to 168, the algorithm is no longer reliable, therefore I will stick to 150. As a result, we have obtained the profitability graph for a year as in Figure 4. CloseProfit is nothing more than the average volatility converted into USD.

![GBPUSD 2019 Chart](https://c.mql5.com/2/41/GBPUSD_80_64_150.PNG)

![GBPUSD 2019 report](https://c.mql5.com/2/41/GBPUSD_80_64_150_report.PNG)

Figure 4. GBPUSD H1, 2019.11.25 - 2020.11.25

The deposit is set to $10,000 for optimization. After completing the research, we can set the sum back to the optimal one. The test and optimization were performed in the reference point mode because the algorithm works on close prices, so the events within a candle are of no importance to it. Figure 5 shows the test for the same period in the "Every tick" mode.

![GBPUSD tick 2019 Chart](https://c.mql5.com/2/41/GBPUSD_tick_2019_Chart.PNG)

![GBPUSD tick 2019 report](https://c.mql5.com/2/41/GBPUSD_tick_2019_report.PNG)

Figure 5. GBPUSD H1, 2019.11.25 - 2020.11.25 "Every tick" mode

According to Figure 5, the profit has even slightly increased in the "Every tick" mode because the test has become more accurate. The profitability graph remains almost completely the same. Both tests have yielded an excellent profit factor of just over five.

It would be wrong to draw conclusions about the pattern reliability based on tests for one optimized year, so let's see how many years the algorithm is able to last with these parameters in the past. Figure 6 shows the test result outside the optimized period.

![GBPUSD 2001 Chart](https://c.mql5.com/2/41/GBPUSD_80_64_150_2001.PNG)

![GBPUSD report 2001](https://c.mql5.com/2/41/GBPUSD_80_64_150_report_2001.PNG)

Figure 6. GBPUSD H1, 2001.01.01 - 2020.11.25

Figure 6 shows the backtest since 2001 with the same parameters that were obtained during the one-year optimization from 2019.11.25 to 2020.11.25. The test shows that the drawdown has increased by only a couple of dollars on such a large interval, while the profit increased significantly and the profit factor rose to 7.5. The test was carried out for the deposit of $3000 with no refinancing in order to understand how the pattern behaves over long periods of time.

The fact that the optimization has been carried out in a year, while the algorithm shows a stable result for 20 years, indicates that the pattern is quite stable and the parameters are not adjusted to the history. For some reason, the GBPUSD pair does not deviate greatly from its inherent statistical characteristics.

It would be wrong to draw conclusions based on the test of a single currency pair and timeframe. Therefore, let's consider the tests for EURUSD. As in the previous case, the optimization was carried out in a year from 2019.11.25 to 2020.11.25 on H1. I approached the choice of parameters in the same way as in the previous case. The results are shown in Figure 7.

![EURUSD 2019 Chart](https://c.mql5.com/2/41/EURUSD_Chart_2019.PNG)

![EURUSD 2019 report](https://c.mql5.com/2/41/EURUSD_report_2019.PNG)

Figure 7. EURUSD H1, 2019.11.25 - 2020.11.25

As we can see in Figure 7, the profitability on EURUSD is lower than on GBPUSD, while the drawdown is slightly larger. The profitability graph shows that there was a segment with multiple opened positions. After slightly tightening the MinBars and OpenPercent parameters, we are able to reduce the number of positions and, consequently, drawdown. Let's move on to the long-period test. Figure 8 displays testing EURUSD from 2007.01.01 to 2020.11.25.

![EURUSD 2007 chart](https://c.mql5.com/2/41/EURUSD_2007_chart.PNG)

![EURUSD 2007 report](https://c.mql5.com/2/41/EURUSD_2007_report.PNG)

Figure 8. EURUSD H1, 2007.01.01 - 2020.11.25

Trading EURUSD is not so stable compared to GBPUSD. The period of stable work using the data that is further back in time has turned out to be six years less. The result is still good. Optimization of parameters was carried out in a year, and stable work lasted almost 14 years. This fact again suggests that the parameters were not simply adjusted to the history, and the trading instrument features a fairly stable pattern.

Next, you need to check how the algorithm behaves on other timeframes. Theoretically, with a decrease in the timeframe, stability should decrease, because the size of candles on a lower timeframe decreases significantly relative to the spread. Consequently, there will be less profit and more trading costs. Besides, on smaller timeframes, the entry point may become even more stretched in time leading to more open positions and, accordingly, greater drawdowns.

Let's conduct a test on GBPUSD M15. Like in the previous case, the optimization is to be performed for the year from 2019.11.25 to 2020.11.25. But I will not display the optimization graph for a year. Instead, I will display the largest possible interval the algorithm is able to pass smoothly.

![GBPUSD M15 chart](https://c.mql5.com/2/41/GBPUSD_m15_2000_chart.PNG)

![](https://c.mql5.com/2/41/GBPUSD_m15_2000_report.PNG)

Figure 9. GBPUSD m15 2000.01.01-2020.11.25

Figure 9 shows the test of GBPUSD M15 conducted from the year of 2000. But the number of entry signals and positions is small. As I wrote above, the lower timeframe demonstrates less stability, and the settings turn out to be very conservative. The entry signal is rarely generated, the profitability is not high, but is adequate relative to the drawdown.

Next, let's conduct the test on the higher timeframe GBPUSD H4. H4 has few candles for optimization. Therefore, I will perform optimization on a two-year segment from 2018.11.25 to 2020.11.25. The result will be shown on the maximum interval.

![GBPUSD H4](https://c.mql5.com/2/41/GBPUSD_H4_2000_Chart.PNG)

![GBPUSD H4 Report](https://c.mql5.com/2/41/GBPUSD_H4_2000_report.PNG)

Figure 10. GBPUSD H4 2000.01.01-2020.11.25

H4 shows a stable result within almost 20 years from 2000.01.01 to 2020.11.25. Like in the previous cases, the entire optimization boils down to finding the balance between profit and reliability. The conservative settings from M15 work reliably on both H1 and H4. But there is no point in them there due to very rare signals and a small number of deals.

You can also test any other trading instrument. Depending on a symbol, the algorithm works better or worse. But the trend continues — one-year optimization allows for a stable work for several years. Below are the results for GBPJPY H1. Optimization was carried out for a year, the result is shown in Figure 11.

![GBPJPY chart](https://c.mql5.com/2/41/GBPJPY_2009_Chart.PNG)

![GBPGPY report](https://c.mql5.com/2/41/GBPJPY_2009_Report.PNG)

Figure 11. GBPJPY 2009.01.01 - 2020.11.25

GBPJPY shows a stable backtest since 2009. The result is not as impressive as in the case of GBPUSD, but it works.

The EA features the ability to re-invest the earned funds. You need to use it. Until now, I showed the tests with conservative settings. But what if we set very aggressive settings and enable a lot increase? I am not a fan of high risks, but let's see what the algorithm is capable of. I will perform the test on GBPUSD from 2006.01.01 to 2020.11.25 in the "Every tick" mode. Of course, it is possible to test another symbol. The spread is reduced to 20. This is slightly above average. Figure 12 shows the backtest result for almost 15 years.

![GBPUSD max risk](https://c.mql5.com/2/41/GBPUSD_max_risk_tick_chart_combo.png)

Figure 12. GBPUSD from 2006.01.01 to 2020.11.25, aggressive settings

As you may remember, the algorithm uses close prices. Therefore, this result is not a "test grail". In addition, the adequate spread of 20 is set. The algorithm's trading result on the real market usually coincides with the one obtained in the tester. I have never used it to trade with such aggressive settings. Besides, it is impossible to take into account real spreads in MetaTrader 4, so I will not argue that it would have passed this period as well in real trading.

### Analyzing the results

The EA is simple and easy to optimize. Optimization for a short period of time and the use of non-optimal parameters in advance allow it to remain stable for several years, and sometimes dozens of years. This suggests that the found pattern is not accidental and is really present on the market. However, the algorithm is quite "rigid" as it has few degrees of freedom and operates with rigidly specified parameters. If a price series goes beyond the capabilities of the algorithm, then it immediately loses its stability.

Most importantly, it can be used on any timeframe starting with the one having the candle size considerably exceeding the spread and commission. As the timeframe increases, the stability grows up to the daily timeframe. However, the requirements for the deposit also increase since the candle size becomes significant causing large drawdowns.

We can conclude that the price series is only superficially similar to a random walk as I described in my first article " [Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136)". The price series contains at least one hidden pattern.

On the other hand, I know and thoroughly understand the pattern that allows making profit, which means it is time for some serious modernization.

### Usage in trading

The big drawback of the current algorithm is its low return relative to risks. We need to be honest with ourselves and understand that there are risks (just like anywhere else). But while the development of improvements is in progress, we may use the EA for trading. To do this, we need to raise its profitability and maintain the drawdown level. It is necessary to use the algorithm features. It rarely generates signals for trading instruments and, as a rule, they are significantly spaced in time for different trading instruments. Most of the time there is no trading on the instrument. We need to fill these "downtime".

Generally, there is no stable correlation of signals for different instruments. We should use this feature to increase profit while maintaining the drawdown level.

To increase profitability, we need to analyze several instruments simultaneously. When a signal appears for one of the instruments, open positions on it. If positions are opened for one instrument, ignore the series start signals from other symbols. This allows using time as efficiently as possible, increase the number of signals, and, accordingly, raise profit proportionally. The maximum drawdown remains at one instrument level.

Series start signals are weakly correlated not only between independent symbols but also between different timeframes of the same symbol. Therefore, it is possible to use time even more efficiently and achieve a situation where there are almost always open positions in the market by using three timeframes of the same symbol in trading. The drawdown level remains at the same level, while the profit grows significantly.

Several modifications were made for working on real accounts:

> 1) Add nine new independent trading instruments, each with its own parameters. This allows the algorithm analyze and trade 10 trading instruments at once. It is possible to optimize parameters for any instrument to one degree or another, so the decision seems logical.

> 2) Limited simultaneous opening of positions for several trading instruments. The MaxSeries parameter is added for that. If set to 1, trading is performed only on one instrument. If 2, positions can be opened simultaneously on two instruments, and so on. This will make it possible for the EA to generate position open signals more often, using its "free time". The profit is increased proportionally, while the maximum drawdown remains at the level of one instrument.

MetaTrader 4 tester is unable to perform multi-currency tests. But if more than one symbol is traded, the drawdown probably increases proportionally to the square root of the number of simultaneously traded instruments, provided that there is no correlation between signals of different instruments. If we expect a drawdown of $1000, then, trading three instruments simultaneously, we can expect an increase in the drawdown to the level of $1000\*root(3)=$1732.

> 3) Added limitation on the minimum number of "MinEquity" funds. Upon reaching this value, trading stops and positions are closed. This is necessary in order to plan the risks and adhere to them.

> 4) The EA can be used on several timeframes simultaneously to increase profitability. According to my tests, position open signals usually have a correlation between different timeframes, although it is far from 100%.

I used the EA to trade on 25 currency pairs, as well as on H1 and H4 timeframes simultaneously with conservative settings and managed to maintain the profitability at 10% per month. On another account, with slightly more aggressive settings, I managed to achieve a profitability of 15% per month. There were a lot of deals.

### Conclusions and further development

- A non-conventional trading method has been found;

- The selected basic pattern has turned out to be universal and is present on tested trading instruments;

- The pattern is simple and clear. We need to study the reasons for its occurrence and develop ways to improve the signal quality;

- The algorithm is perfectly optimized and remains stable for a long time;

- The entry point is strongly stretched in time. We need to work on better localization of the position open signal;

- Unaccounted factors lead to a loss of the algorithm stability over time;

- It is necessary to continue the development of the theoretical model to reduce the number of unaccounted factors with each new version and, as a result, take into account all possible factors influencing the pattern;

- The next version of the algorithm should become more flexible and should already begin to slightly adjust its parameters for market changes.

The next article is to be devoted to mechanisms I have developed to improve the algorithm reliability and flexibility.

The EA code and the appropriate requirements specification are attached below.

The author of the idea and requirements specification is [Maxim Romanov](https://www.mql5.com/en/users/223231). The code has been written by [Vyacheslav Ivanov](https://www.mql5.com/en/users/viper70).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8616](https://www.mql5.com/ru/articles/8616)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8616.zip "Download all attachments in the single ZIP archive")

[50p\_V1.mq4](https://www.mql5.com/en/articles/download/8616/50p_v1.mq4 "Download 50p_V1.mq4")(100.93 KB)

[Technical\_task.zip](https://www.mql5.com/en/articles/download/8616/technical_task.zip "Download Technical_task.zip")(30.59 KB)

[AAPL\_GAZP\_statistics.zip](https://www.mql5.com/en/articles/download/8616/aapl_gazp_statistics.zip "Download AAPL_GAZP_statistics.zip")(12989.18 KB)

[GBPUSD\_statistics.zip](https://www.mql5.com/en/articles/download/8616/gbpusd_statistics.zip "Download GBPUSD_statistics.zip")(14471.42 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-adapting algorithm (Part IV): Additional functionality and tests](https://www.mql5.com/en/articles/8859)
- [Self-adapting algorithm (Part III): Abandoning optimization](https://www.mql5.com/en/articles/8807)
- [Developing a self-adapting algorithm (Part II): Improving efficiency](https://www.mql5.com/en/articles/8767)
- [A scientific approach to the development of trading algorithms](https://www.mql5.com/en/articles/8231)
- [What is a trend and is the market structure based on trend or flat?](https://www.mql5.com/en/articles/8184)
- [Price series discretization, random component and noise](https://www.mql5.com/en/articles/8136)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/362352)**
(47)


![Eric Pedron](https://c.mql5.com/avatar/2018/4/5AD72FFE-6406.jpg)

**[Eric Pedron](https://www.mql5.com/en/users/dojisan)**
\|
28 Feb 2021 at 01:31

Since you are using the second method, maybe it's good to change the topic because the algorithm given is not really self-adapting. The strategy looks interesting though.


![Maxim Romanov](https://c.mql5.com/avatar/2025/11/691b0e3f-04d1.png)

**[Maxim Romanov](https://www.mql5.com/en/users/223231)**
\|
28 Feb 2021 at 07:28

**Eric Pedron:**

Since you are using the second method, maybe it's good to change the topic because the algorithm given is not really self-adapting. The strategy looks interesting though.

yes, this algorithm is not self-adapting, it is the first step towards developing an idea.  There will be 4 articles in total and in the last two I will show you fully adaptive.


![Maxim Romanov](https://c.mql5.com/avatar/2025/11/691b0e3f-04d1.png)

**[Maxim Romanov](https://www.mql5.com/en/users/223231)**
\|
19 Mar 2021 at 10:08

**Aleksey\_Kryukov:**

_General simplified algorithm of operation_

- _a window of N candlesticks is scanned;_
- _the window of N candlesticks is scanned; it is checked whether there are more falling or rising candlesticks;_
- _if the preponderance is greater than the threshold, the signal to start a series of positions;_
- _more [falling candlesticks](https://www.mql5.com/en/articles/8767 "Article: Development of a self-adaptive algorithm (Part II): Improving efficiency ") = Buy signal, more rising candlesticks = Sell;_
- _lot is calculated;_
- _a new position is opened at each next candlestick until the series closing condition is triggered;_
- _series closing condition is triggered;_
- _all positions are closed;_
- _search for a new signal._

The author invented the RSI indicator. Bravo!

Relative Strength Index

RSI= 100 - (100 / (1 + U / D))

Where:

U - average value of positive price changes;

D\- average value of negative price changes.

not the RSI indicator is analysed, but the number of falling and rising candles, this is a different meaning. Yes then, when positions start to open, there is an analogy with the indicator. With the help of a set of positions, the formula becomes similar. And not exactly, but only similar. Plus, the indicator has a period, and positions are opened on the period on which it is necessary.

There can be many criticisms of this system, it is far from perfect, but the fact that it is RSI is overkill. The system is as similar to RSI as an aeroplane is to a car. Both have wheels and both burn fuel.

![Luis Leal](https://c.mql5.com/avatar/2013/12/52A1FC3B-E443.jpg)

**[Luis Leal](https://www.mql5.com/en/users/firstdimension)**
\|
6 Dec 2021 at 13:23

You are searching for the efficiency, not for the equilibrium... For this work, are you sure that only the number of the candles is the unique place to see and if you will see too the average of the amplitudes between open and close could be more accurate and we can see other things...? You are treating the candles as a simple bet, like red and black in roulette, but the roulette is a closed circuit with almost 100% of efficiency and do not have opposite forces, only the 0s...

To have a self-adapting approach, we will be need an engine to see first the start point to counting and the duration time to count... The time and amplitudes of the the movements, averages, discards, the cycles... etc...

![Maxim Romanov](https://c.mql5.com/avatar/2025/11/691b0e3f-04d1.png)

**[Maxim Romanov](https://www.mql5.com/en/users/223231)**
\|
6 Dec 2021 at 14:58

**Luis Leal [#](https://www.mql5.com/en/forum/362352#comment_26313010):**

You are searching for the efficiency, not for the equilibrium... For this work, are you sure that only the number of the candles is the unique place to see and if you will see too the average of the amplitudes between open and close could be more accurate and we can see other things...? You are treating the candles as a simple bet, like red and black in roulette, but the roulette is a closed circuit with almost 100% of efficiency and do not have opposite forces, only the 0s...

To have a self-adapting approach, we will be need an engine to see first the start point to counting and the duration time to count... The time and amplitudes of the the movements, averages, discards, the cycles... etc...

I wrote 3 more articles on this topic, here are the links to them in order, read how the model developed and what I came to in my articles.

2 - [https://www.mql5.com/en/articles/8767](https://www.mql5.com/en/articles/8767 "https://www.mql5.com/en/articles/8767")

3 - [https://www.mql5.com/en/articles/8807](https://www.mql5.com/en/articles/8807 "https://www.mql5.com/en/articles/8807")

4 - [https://www.mql5.com/en/articles/8859](https://www.mql5.com/en/articles/8859 "https://www.mql5.com/en/articles/8859")

I also have articles preceding this one, in which I bring them to the topic.

Naturally, I did not stop and continued to develop the theoretical model. Now I can already tell how the price series differ from a random walk, how to find these differences and what are the reasons for these differences. In the articles I have not described this yet, but read my next works, you may be interested.

![Neural networks made easy (Part 9): Documenting the work](https://c.mql5.com/2/48/Neural_networks_made_easy_009.png)[Neural networks made easy (Part 9): Documenting the work](https://www.mql5.com/en/articles/8819)

We have already passed a long way and the code in our library is becoming bigger and bigger. This makes it difficult to keep track of all connections and dependencies. Therefore, I suggest creating documentation for the earlier created code and to keep it updating with each new step. Properly prepared documentation will help us see the integrity of our work.

![Neural networks made easy (Part 8): Attention mechanisms](https://c.mql5.com/2/48/Neural_networks_made_easy_008.png)[Neural networks made easy (Part 8): Attention mechanisms](https://www.mql5.com/en/articles/8765)

In previous articles, we have already tested various options for organizing neural networks. We also considered convolutional networks borrowed from image processing algorithms. In this article, I suggest considering Attention Mechanisms, the appearance of which gave impetus to the development of language models.

![The market and the physics of its global patterns](https://c.mql5.com/2/40/5a55ed9f370f2c15608b457b.png)[The market and the physics of its global patterns](https://www.mql5.com/en/articles/8411)

In this article, I will try to test the assumption that any system with even a small understanding of the market can operate on a global scale. I will not invent any theories or patterns, but I will only use known facts, gradually translating these facts into the language of mathematical analysis.

![Prices in DoEasy library (part 59): Object to store data of one tick](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__3.png)[Prices in DoEasy library (part 59): Object to store data of one tick](https://www.mql5.com/en/articles/8818)

From this article on, start creating library functionality to work with price data. Today, create an object class which will store all price data which arrived with yet another tick.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=gtlgeccspprttohknhqdacvipavhgrck&ssn=1769093294040543443&ssn_dr=0&ssn_sr=0&fv_date=1769093294&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8616&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Developing%20a%20self-adapting%20algorithm%20(Part%20I)%3A%20Finding%20a%20basic%20pattern%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909329444737845&fz_uniq=5049391053649324717&sv=2552)

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