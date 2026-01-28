---
title: Trend criteria in trading
url: https://www.mql5.com/en/articles/16678
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:32:48.035983
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/16678&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071914897236766926)

MetaTrader 5 / Trading


### Introduction

All traders know the phrase "The trend is your friend". Indeed, trending price movements can bring quite large profits. Trend trading is based on the assumption that price movement will continue in the same direction. The main problem with this type of trading is to determine the start and end time of the trend with sufficient accuracy.

Today, there are many approaches to defining and calculating trend parameters. In this article, we will look at the most interesting of them and try to apply them in practice.

### Smoothing and trends

Price movement can be represented using a simple model. There is some deterministic component that depends on time. Some random component is added to it, which does not depend on anything and behaves unpredictably. One of the tasks facing a trader is to somehow reduce the influence of this component.

One of the simplest filters is the simple moving average. But this indicator has one serious drawback - it lags. Let's simulate the trend and apply [SMA](https://www.mql5.com/en/docs/indicators/ima) to it with the period of 3.

| Trend | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| SMA | - | - | 1 | 2 | 3 | 4 |

Now let's try to get rid of the delay. The SMA equation looks like this:

![](https://c.mql5.com/2/165/0__23.png)

Add a correction to it, which will be equal to the average price change for one bar:

![](https://c.mql5.com/2/165/0__24.png)

Let's see how the ratios of our new indicator change:

![](https://c.mql5.com/2/165/0__25.png)

The indicator with such ratios will accurately hit the trend. Let's generalize this indicator to any number of prices handled. The calculation of an indicator with a period of 4 will look like this.

First, find the SMA value:

![](https://c.mql5.com/2/165/0__26.png)

There are now two corrections for the average price change:

![](https://c.mql5.com/2/165/0__27.png)

![](https://c.mql5.com/2/165/0__28.png)

In other words, we calculate the average price change relative to the SMA center. Then, the indicator equation will be as follows:

![](https://c.mql5.com/2/165/0__29.png)

The difference between this indicator and SMA is especially noticeable with short periods.

![](https://c.mql5.com/2/165/1__3.png)

The main disadvantage of this approach is that there are no criteria to be used to select the indicator period. Traders should choose it arbitrarily, based on their own considerations.

Let's try to build an indicator that will not depend on the period. Suppose we have the V time series that we would like to forecast one step ahead. For the forecast we will use the simplest method - previous actions define the future ones. For example, if we have the initial value of V, then the one-step-ahead forecast will be equal to this value:

![](https://c.mql5.com/2/165/0__30.png)

Once the new value of V appears, we find the half-sum between the forecast and V. The resulting value will be the forecast for the next step:

![](https://c.mql5.com/2/165/0__31.png)

In other words, the forecast will be adjusted as new values of the time series appear. This method of forecasting leads to [exponential smoothing](https://ru.wikipedia.org/wiki/%D0%AD%D0%BA%D1%81%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5_%D1%81%D0%B3%D0%BB%D0%B0%D0%B6%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5 "https://ru.wikipedia.org/wiki/%D0%AD%D0%BA%D1%81%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5_%D1%81%D0%B3%D0%BB%D0%B0%D0%B6%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5") with the ratio of 0.5.

Now let's change the approach to forecasting a bit. Let's assume that the forecast change occurs with the same intensity:

![](https://c.mql5.com/2/165/0__32.png)

In other words, all forecast values belong to the linear trend. The indicator equation built on this principle will look like this:

![](https://c.mql5.com/2/165/0__33.png)

We have a recursive indicator - to calculate the current value of the indicator, its past values are used. But in this form it will be unstable - one of the ratios is equal to 1. To overcome this problem, we will apply the same [recursion](https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%BA%D1%83%D1%80%D1%81%D0%B8%D1%8F "https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%BA%D1%83%D1%80%D1%81%D0%B8%D1%8F") to Indicator\[i+1\]:

![](https://c.mql5.com/2/165/0__34.png)

The indicator now consists of two parts - SMA with a period of 2 and half the speed of the indicator in previous readings. Let's make one more change to make the indicator more robust:

![](https://c.mql5.com/2/165/0__35.png)

This is what our new indicator looks like compared to a similar EMA with a period of 3.

![](https://c.mql5.com/2/165/2__3.png)

Unfortunately, suppressing noise completely is a nearly impossible task. But it is entirely possible to identify the trend component in price movement. Both indicators are sensitive to changes in trend parameters. At the same time, their delay is reduced to the minimum possible. These indicators can be used both independently, for price smoothing, and as a source of data for other indicators.

### Trend criteria

It is more correct to call trend criteria randomness criteria. The essence of their application is very simple. The criterion allows us to check how random the price series is. If the criterion shows us that the series is not random, then we can say that it is a trend. Let's look at possible criteria and how they can be applied in technical analysis.

**_Abbe criterion_**. This criterion is based on calculating the variance in two different ways. Normal [sample variance](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F_%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D0%BE%D0%B9_%D0%B2%D0%B5%D0%BB%D0%B8%D1%87%D0%B8%D0%BD%D1%8B "https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F_%D1%81%D0%BB%D1%83%D1%87%D0%B0%D0%B9%D0%BD%D0%BE%D0%B9_%D0%B2%D0%B5%D0%BB%D0%B8%D1%87%D0%B8%D0%BD%D1%8B") is sensitive to trend. In the [Allan variance](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F_%D0%90%D0%BB%D0%BB%D0%B0%D0%BD%D0%B0 "https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F_%D0%90%D0%BB%D0%BB%D0%B0%D0%BD%D0%B0"), the influence of the trend tends to zero. By comparing these dispersions, one can estimate the contribution of the trend component to the price change.

I will slightly change the calculation of this criterion for use in trading:

![](https://c.mql5.com/2/165/0__36.png)

This criterion only indicates the presence of a trend. The direction of the trend needs to be determined in other ways.

![](https://c.mql5.com/2/165/3__3.png)

This criterion can help in identifying moments of trend changes.

**_Criterion of signs of first differences_**. This criterion is very simple and intuitive. We take N prices. We divide them into consecutive pairs, of which there will be N-1. We apply the sign function to each pair:

![](https://c.mql5.com/2/165/0__37.png)

After that, obtain the criterion value:

![](https://c.mql5.com/2/165/0__38.png)

In [RSI](https://www.mql5.com/en/docs/indicators/irsi), the same approach is used, but the price movements up and down are calculated separately, without taking into account the signs.

This statistic has one drawback - it does not distinguish the order in which the signs appear. In other words, we can swap the prices and get the same result. To get rid of this drawback, we can assign weights to each sign, depending on the moment of its appearance. Then the criterion calculation will look like this:

![](https://c.mql5.com/2/165/0__39.png)

In this case, the sequence of signs becomes unique, and the same result is possible only if the two time series are similar.

Let's see how this criterion can be used for technical analysis. The values that can be obtained using this criterion lie within strictly limited bounds. Intuition suggests that values close to zero may be more common. But, it is better to perform the necessary checks.

![](https://c.mql5.com/2/165/4__3.png)

The assumption turned out to be correct. In strict trading language: we have obtained an empirical probability function. What can be learned from this feature? We can get overbought/oversold levels. I set this level to 33% - the indicator sorts out a third of the lowest and highest values. The third that remains in the middle is flat. The indicator itself looks like this:

![](https://c.mql5.com/2/165/5__3.png)

The sign criterion is non-parametric. The main advantage of such criteria is their stability and insensitivity to sudden price changes.

**_Kendall's criterion_**. This criterion is also based on the function of signs. But this function is applied a little differently.

The calculation of this criterion can be carried out in two stages. First, for each price we find the sum of the signs with all the prices preceding it:

![](https://c.mql5.com/2/165/0__40.png)

After that, we find the total sum of these values:

![](https://c.mql5.com/2/165/0__41.png)

With this criterion, we compare the number of price movements up and down for all pairwise price combinations. Thanks to this, we can assess the direction and strength of the trend more accurately.

![](https://c.mql5.com/2/165/6__1.png)

**_Wald-Wolfowitz rank test_**. To calculate this criterion, we need to know the rank of each price. Rank is the number of prices that are below the current one. For example, I will take 5 price values:

| Index | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Price | 1.05702 | 1.05910 | 1.05783 | 1.05761 | 1.05657 |
| Rank | 1 | 4 | 3 | 2 | 0 |

The price with index 0 is higher than a single price with index 4. This means that the rank of this price is 1. The price with index 1 is higher than all the others, and its rank is 4. The ranks of all other prices are calculated the same way.

The essence of this criterion is very simple - if prices form a trend, then their ranks will also be organized. Conversely, if the ranks are mixed up in some incomprehensible way, then something incomprehensible happens with the prices as well. In this example, prices are partially ordered, which may indicate the presence of a trend.

The value of this criterion is calculated using the equation:

![](https://c.mql5.com/2/165/0__42.png)

In essence, this criterion is a robust variant of the [auto correlation](https://en.wikipedia.org/wiki/Autocorrelation "https://en.wikipedia.org/wiki/Autocorrelation") function. The indicator built on its basis looks like this:

![](https://c.mql5.com/2/165/7__1.png)

**_Foster-Stewart criterion_**. This criterion allows us to simultaneously assess the presence of a trend in the means and variances. It is based on counting the number of top and bottom records. For each price reading, we determine the value of two variables H and L.

The variable H is equal to 1 if the current price is higher than all previous ones. The variable L is equal to 1 if the price is lower than all previous ones. In all other cases, these variables are equal to zero. The criterion parameters are calculated as follows:

![](https://c.mql5.com/2/165/0__43.png)

![](https://c.mql5.com/2/165/0__44.png)

The T parameter shows the strength and direction of the trend. The D parameter is similar to the Abbe criterion and indicates only the presence of a trend. These parameters can be used either separately or in combination with each other.

![](https://c.mql5.com/2/165/8__1.png)

This criterion can be modernized to take into account both parameters simultaneously. In the classical version, a trend can be considered established if the value of T or D is large enough (of course, you need to take the absolute value for T). Thus, during a trend, the product of these parameters deviates maximally from zero. This approach allows us to identify the strongest trends.

Now let's look at how these criteria can be used in trading.

### Trading strategies

Unfortunately, trend criteria do not tell us anything about the beginning of a trend. They only indicate that the trend has taken place. It is this property that can be used in trading - after a strong trend, the direction of price movement may change to the opposite.

We can use this assumption to create a simple strategy. If the trend criterion has reached some minimum value, then you need to open a buy position. To open sell positions, the criterion should reach its maximum. In other words, trend criteria are used to determine overbought/oversold conditions. This approach looks promising.

![](https://c.mql5.com/2/165/9__2.png)

Using additional filters can improve the performance of the strategy. The Abbe criterion can be used as such filter. Let me remind you that this criterion only determines the presence of a trend. The direction of the trend needs to be determined in other ways. For example, I will determine the trend using the average speed of price movement:

![](https://c.mql5.com/2/165/0__45.png)

If the obtained value is above or below a certain level, then I consider the trend to be established. To confirm this assumption, I use the Abbe criterion. If its value is above a certain level, then the trend assumption is true. I leave the opening and closing of positions as in the previous example. The result of such a strategy looks like this:

![](https://c.mql5.com/2/165/10__2.png)

The Foster-Stewart criterion allows one to immediately assess both the presence of a trend and its direction. In other words, this criterion can simultaneously serve as both a signal and a filter. The application of this criterion can give the following results:

![](https://c.mql5.com/2/165/11__2.png)

A modernized version of this criterion allows us to obtain different results.

![](https://c.mql5.com/2/165/TesterGraphReport2024.12.16__2.png)

The use of trend criteria is entirely justified, but is associated with some difficulties:

- First, it is necessary to use additional filters to reduce the number of false signals.
- Secondly, separate rules are needed for closing positions to reduce risks and the burden on the deposit.
- Third, trend criteria may be sensitive to the number of prices processed. Therefore, their application may require preliminary smoothing of the time series. For small periods, smoothing should be mandatory.

### Conclusion

Currently, there are several dozen trend criteria. The application of these criteria can be useful both in analyzing market situations and in trading.

The following programs were used when writing this article.

| Name | Type | Description |
| --- | --- | --- |
| tSMA | indicator | Trend analogue of SMA<br>- **iPeriod** \- indicator period, minimum value 3 |
| tEMA | indicator | Trend analogue of EMA with period 3 |
| Abbe criterion | indicator | Abbe criterion |
| Criterion Signs | indicator | Criterion of signs of first differences |
| scr Criterion Signs | script | The script allows evaluating the distribution of the sign criterion values<br>- **iPeriod** \- criterion period<br>- **_ScreenShot_** \- allows saving the image in the Files folder |
| Kendall's criterion | indicator | Kendall's criterion |
| Foster-Stewart criterion | indicator | Foster-Stewart criterion<br>- **_Type_** \- parameter selection<br>- **_iPeriod_** \- criterion period |
| Foster-Stewart criterion I | indicator | A modernized version of the Foster-Stewart criterion |
| Wald-Wolfowitz criterion | indicator | Wald-Wolfowitz criterion |
| EA 3 criterions | EA | The EA trades according to 3 criteria<br>- **_Type_** \- criterion selection<br>- **_Period Criterion_** \- period<br>- **_Criterion level_** \- response level, permissible values 1 - 49 |
| EA Abbe criterion | EA | - **_Average speed_**<br>- **_Abbe level_** \- criterion level, acceptable values 1 - 99 |
| EA Foster-Stewart criterion | EA | - **_T level_** \- trend parameter level, acceptable values 1 - 49<br>- **_D level_** \- dispersion parameter level, acceptable values 1 - 99 |
| EA Foster-Stewart criterion I | EA | - _Level_ \- criterion level, acceptable values 1 - 49 |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16678](https://www.mql5.com/ru/articles/16678)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16678.zip "Download all attachments in the single ZIP archive")

[tSMA.mq5](https://www.mql5.com/en/articles/download/16678/tSMA.mq5 "Download tSMA.mq5")(5.56 KB)

[tEMA.mq5](https://www.mql5.com/en/articles/download/16678/tEMA.mq5 "Download tEMA.mq5")(4.72 KB)

[Abbe\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/Abbe_criterion.mq5 "Download Abbe_criterion.mq5")(5.66 KB)

[Criterion\_Signs.mq5](https://www.mql5.com/en/articles/download/16678/Criterion_Signs.mq5 "Download Criterion_Signs.mq5")(7.52 KB)

[scr\_Criterion\_Signs.mq5](https://www.mql5.com/en/articles/download/16678/scr_Criterion_Signs.mq5 "Download scr_Criterion_Signs.mq5")(5.26 KB)

[Kendallws\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/Kendallws_criterion.mq5 "Download Kendallws_criterion.mq5")(6.96 KB)

[Foster-Stewart\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/Foster-Stewart_criterion.mq5 "Download Foster-Stewart_criterion.mq5")(7.59 KB)

[Foster-Stewart\_criterion\_I.mq5](https://www.mql5.com/en/articles/download/16678/Foster-Stewart_criterion_I.mq5 "Download Foster-Stewart_criterion_I.mq5")(7.08 KB)

[Wald-Wolfowitz\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/Wald-Wolfowitz_criterion.mq5 "Download Wald-Wolfowitz_criterion.mq5")(6.77 KB)

[EA\_3\_criterions.mq5](https://www.mql5.com/en/articles/download/16678/EA_3_criterions.mq5 "Download EA_3_criterions.mq5")(7.89 KB)

[EA\_Abbe\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/EA_Abbe_criterion.mq5 "Download EA_Abbe_criterion.mq5")(7.86 KB)

[EA\_Foster-Stewart\_criterion.mq5](https://www.mql5.com/en/articles/download/16678/EA_Foster-Stewart_criterion.mq5 "Download EA_Foster-Stewart_criterion.mq5")(7.54 KB)

[EA\_Foster-Stewart\_criterion\_I.mq5](https://www.mql5.com/en/articles/download/16678/EA_Foster-Stewart_criterion_I.mq5 "Download EA_Foster-Stewart_criterion_I.mq5")(7.05 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Polynomial models in trading](https://www.mql5.com/en/articles/16779)
- [Cycles and trading](https://www.mql5.com/en/articles/16494)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/494100)**
(7)


![Aleksandr Grigorev](https://c.mql5.com/avatar/2024/7/6691a636-f80e.png)

**[Aleksandr Grigorev](https://www.mql5.com/en/users/metatradebot)**
\|
28 Dec 2024 at 16:39

**Aleksej Poljakov [#](https://www.mql5.com/ru/forum/478473#comment_55484613):**

I'll try... there are some other interesting criteria, including the definition of a pivot point. I need to figure out how to explain them in a simpler way.

I don't know what the problem is with the tester. I'm getting this.

Thanks, I'll give it a try. And where do you look for explanations of the criteria and mathematical mechanisms for detecting them? Looks like scientific sources ...

![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
28 Dec 2024 at 17:35

**Aleksandr Grigorev [#](https://www.mql5.com/ru/forum/478473#comment_55484838):**

Thanks, I'll give it a try. And where do you look for explanations of the criteria and mathematical mechanisms for their detection? Looks like scientific sources ...

there is a lot of literature, but all the criteria are mostly scattered here and there. Here's a good selection of different criteria

![Maksim Galichev](https://c.mql5.com/avatar/2025/1/6799b538-ed3a.jpg)

**[Maksim Galichev](https://www.mql5.com/en/users/freetrader.ru)**
\|
29 Mar 2025 at 15:52

I am working on the practical application of the Wald-Wolfowitz trend criterion described in your article. As I understand it, the Wald-Wolfowitz criterion tests the hypothesis of randomness/stationarity of data. In the code of trading Expert Advisors, it is important to understand what exactly the indicator returns?

Do I understand correctly that the [Indicator calculates](https://www.mql5.com/en/docs/basis/function/events#oncalculate "MQL5 Documentation: function OnCalculate()") the probability (in per cent) that the sequence of prices (in this case - open values) is random on the basis of the Wald-Wolfowitz criterion.

The result is stored in buffer buffer\[0\] and represents the percentage probability (from 0 to 100).

The closer the value is to 100%, the higher the probability of randomness (no trend).

The closer to 0%, the higher the probability of non-randomness (presence of trend or clustering)?

**Calculation logic**:

The indicator ranks the open values for a selected period ( iPeriod ), then calculates statistics based on the ranks and converts it to a percentage value via CDF (empirical distribution function):

```
buffer[i] = 100. * cdf / cnt; // Percentage probability
```

**Levels in the graph:**

indicator\_level1 = 33 and indicator\_level2 = 67 are benchmarks for interpretation:

<33% - strong non-randomness (trend possible).

>67% - high randomness (flat).

Do I understand the interpretation of the indicator presented in your article correctly?

![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
29 Mar 2025 at 17:24

**Maksim Galichev [#](https://www.mql5.com/ru/forum/478473#comment_56300036):**

I am working on the practical application of the Wald-Wolfowitz trend criterion described in your article. As I understand it, the Wald-Wolfowitz criterion tests the hypothesis of randomness/stationarity of data. In the code of trading Expert Advisors it is important to understand what exactly the indicator returns?

Do I understand correctly that the Indicator calculates the probability (in per cent) that the sequence of prices (in this case - open values) is random on the basis of the Wald-Wolfowitz criterion.

The result is stored in buffer buffer\[0\] and represents the percentage probability (from 0 to 100).

The closer the value is to 100%, the higher the probability of randomness (no trend).

The closer the value is to 0%, the higher the probability of non-randomness (presence of a trend or clustering)?

**Calculation logic**:

The indicator ranks open values for a selected period ( iPeriod ), then calculates statistics based on the ranks and converts it to a percentage value via CDF (empirical distribution function):

**Levels in the graph:**

indicator\_level1 = 33 and indicator\_level2 = 67 are benchmarks for interpretation:

<33% - strong non-randomness (possible trend).

>67% - high randomness (flat).

Do I understand the interpretation of the indicator presented in your article correctly?

Yes, you understand everything correctly. The only thing is that I set the levels 33 and 67 just because I needed some levels. You can set other levels, for example, 25 and 80.

![Maksim Galichev](https://c.mql5.com/avatar/2025/1/6799b538-ed3a.jpg)

**[Maksim Galichev](https://www.mql5.com/en/users/freetrader.ru)**
\|
29 Mar 2025 at 17:51

**Aleksej Poljakov [#](https://www.mql5.com/ru/forum/478473#comment_56300432):**

Yes, you understand everything correctly. The only thing is that I set levels 33 and 67 simply because I needed some levels. You can set other levels, for example, 25 and 80.

Thank you for your reply.

![Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://c.mql5.com/2/165/19285-simplifying-databases-in-mql5-logo__2.png)[Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)

We explore how to manipulate databases in MQL5 using the language's native functions. We cover everything from table creation, insertion, updating, and deletion to data import and export, all with sample code. The content serves as a solid foundation for understanding the internal mechanics of data access, paving the way for the discussion of ORM, where we'll build one in MQL5.

![Getting Started with MQL5 Algo Forge](https://c.mql5.com/2/152/18518-kak-nachat-rabotu-s-mql5-algo-logo.png)[Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)

We are introducing MQL5 Algo Forge — a dedicated portal for algorithmic trading developers. It combines the power of Git with an intuitive interface for managing and organizing projects within the MQL5 ecosystem. Here, you can follow interesting authors, form teams, and collaborate on algorithmic trading projects.

![Developing a Replay System (Part 77): New Chart Trade (IV)](https://c.mql5.com/2/104/Desenvolvendo_um_sistema_de_Replay_Parte_77___LOGO.png)[Developing a Replay System (Part 77): New Chart Trade (IV)](https://www.mql5.com/en/articles/12476)

In this article, we will cover some of the measures and precautions to consider when creating a communication protocol. These are pretty simple and straightforward things, so we won't go into too much detail in this article. But to understand what will happen, you need to understand the content of the article.

![Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/165/19105-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19105)

In this article, we develop a Bat Pattern system in MQL5 that identifies bullish and bearish Bat harmonic patterns using pivot points and Fibonacci ratios, triggering trades with precise entry, stop loss, and take-profit levels, enhanced with visual feedback through chart objects

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/16678&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071914897236766926)

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