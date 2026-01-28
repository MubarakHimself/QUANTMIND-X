---
title: Pair trading
url: https://www.mql5.com/en/articles/13338
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:29:42.536616
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/13338&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082902604695998741)

MetaTrader 5 / Trading


### Introduction

Currently, there are a huge number of trading strategies for every taste. All these strategies are aimed at making a profit. But making a profit is in one way or another connected with the risks - the greater the expected profit, the higher the risks. A logical question arises: is it possible to reduce trading risks to a minimum, while receiving small but stable profits? These conditions are met by [pair trading](https://en.wikipedia.org/wiki/Pairs_trade "https://en.wikipedia.org/wiki/Pairs_trade").

Pair trading is a variery of [statistical arbitrage](https://en.wikipedia.org/wiki/Statistical_arbitrage "https://en.wikipedia.org/wiki/Statistical_arbitrage") first proposed by Jerry Bamberger in the 1980s. This trading strategy is market neutral, allowing traders to profit in almost any market condition. Pair trading is based on the assumption that the characteristics of interrelated financial instruments will return to their historical averages after a temporary deviation. Thus, pair trading comes down to a few simple operations:

- identify discrepancies in the statistical relationship between two financial instruments;

- open positions on them;

- close positions when the characteristics of the instruments return to the average.

Despite its apparent simplicity, pair trading is not an easy or risk-free way to make a profit. The market is constantly changing and statistical relationships may change as well. Besides, any unlikely price movement could result in significant losses. Dealing with such adverse situations requires strict adherence to the trading strategy and risk management rules.

### Correlation

Pair trading strategies are most often based on the [correlation](https://en.wikipedia.org/wiki/Correlation "https://en.wikipedia.org/wiki/Correlation") of two financial instruments. Changes in the prices of several currency pairs can be interrelated. For example, the price of one symbol moves in the same direction as the price of another symbol. In this case, there is a positive correlation between these symbols. In case of a negative correlation, prices move in opposite directions.

The correlation based pair trading strategy is very simple. First, traders should select two financial instruments with a strong correlation. Then they needs to analyze the change in correlation using historical data. Based on this analysis, traders can make an informed decision about entering a trade.

For trading, the most interesting currency pairs are those with negative correlation. For example, this is what the movement of EURUSD and USDCHF looks like.

> ![](https://c.mql5.com/2/58/1.png)

[Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient") is the most commonly used method for estimating correlation. This coefficient is calculated by the equation:

> ![](https://c.mql5.com/2/58/2.png)

This calculation always yields a biased estimate. In small samples, the resulting estimate of r may be very different from the exact correlation value. To reduce this error, we can use [Olkin-Pratt adjustment](https://www.mql5.com/go?link=https://www.researchgate.net/publication/38366915_Unbiased_Estimation_of_Certain_Correlation_Coefficients "https://www.researchgate.net/publication/38366915_Unbiased_Estimation_of_Certain_Correlation_Coefficients"):

> ![](https://c.mql5.com/2/58/01.png)

Let's try to develop rules for a trading strategy based on correlation.

First, we need to select two suitable currency pairs. At the same time, the average correlation value of these pairs in history should be negative. Less is better.

Next, we need to collect statistics with sample correlation values on the history of these currency pairs. These statistics will be needed to calculate the signals.

The next step is to set the trigger level. If the current correlation reaches this level, the EA can open positions. This level can be set explicitly. For example, -0.95, -0.9 etc. There is also an alternative approach. We can take the historical correlation values and sort them in ascending order. As the response level, we can take the limit of 10% of the lowest values.

Before opening positions, we need to determine their type. If the current price of a currency pair is below the moving average, then a Buy position is opened for this symbol. Conversely, if the price is above the average, then a Sell position is opened. In this case, the positions opened should be multidirectional. This condition must be met, otherwise opening positions is prohibited.

In addition, the volumes of positions for different instruments should be interrelated. Suppose that **_PointValue_** is a price of one point in the deposit currency. Then the position volumes should be such that equality is satisfied.

> ![](https://c.mql5.com/2/58/01__1.png)

In this case, the price movement by the same number of points will give approximately the same result for each of the instruments.

In addition, I added two more levels to the EA. Crossing the first level indicates the need to transfer positions to breakeven. Its value is 33%. Crossing the second level leads to the closure of all positions. The closing level is 67%, but not more than zero. Changing these levels can greatly affect the EA profitability.

Let's test an EA following these rules. This is what the balance change looks like for EURUSD and USDCHF from 2021.01.01 to 2023.06.30.

> ![](https://c.mql5.com/2/58/3.png)

Not bad. But the Pearson correlation coefficient has several features. Its use is justified only if the time series values have a normal distribution. Also, this coefficient is significantly influenced by spikes. Besides, Pearson correlation can only recognize linear relationships. To illustrate these features, it is best to use [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet "https://en.wikipedia.org/wiki/Anscombe%27s_quartet").

> ![](https://c.mql5.com/2/58/4.png)

The first graph shows a linear correlation without any peculiarities. The second set of data features a nonlinear relationship, the strength of which the Pearson coefficient could not reveal. In the third set, the correlation coefficient is influenced by a strong spike. There is no correlation in the fourth graph, but nevertheless, even one value is enough for a fairly strong correlation to appear.

[Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient "https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient") is free from these shortcomings. It captures the constantly increasing or decreasing dependence of two time series pretty well. For Spearman correlation, it does not matter according to which law the original data are distributed. The Pearson coefficient only works well with data that is normally distributed. Conversely, the Spearman coefficient can easily cope with any other distribution or their combination.

Also, the Spearman correlation coefficient can reveal nonlinear relationships. For example, one time series has a linear trend, while another has an exponential one. The Spearman coefficient can easily handle this situation, while the Pearson coefficient will not be able to fully reveal the strength of the relationship between these series.

We can calculate the Spearman rank correlation coefficient as follows. First we need to create two arrays. In each array, we will write the price value and bar index for both symbols.

| index | EURUSD | USDCHF |
| --- | --- | --- |
| 0 | 1.06994 | 0.89312 |
| 1 | 1.06980 | 0.89342 |
| 2 | 1.07058 | 0.89277 |
| 3 | 1.07045 | 0.89294 |
| 4 | 1.07089 | 0.89283 |

Now we need to sort both arrays in ascending order. After sorting, the price values are of no interest to us. We only care about the index values that were before sorting and the current ones. The numbers in brackets are the price indices that were before the arrays sorting.

| cur. index | EURUSD | USDCHF |
| --- | --- | --- |
| 0 | 1.06980   (1) | 0.89277   (2) |
| 1 | 1.06994   (0) | 0.89283   (4) |
| 2 | 1.07045  (3) | 0.89294   (3) |
| 3 | 1.07058   (2) | 0.89312   (0) |
| 4 | 1.07089   (4) | 0.89342   (1) |

Now we need to find the differences between the current indices of prices with the same indices before sorting. For example, let's find the difference D0. First, let's find the price index, which was equal to zero. These are 1.06994 EURUSD and 0.89312 USDCHF. The current indices of these prices are 1 and 3. Then, the difference D0 = 1 – 3 = -2.

Next, find the difference D1. The current price index of 1.06980 EURUSD is 0, and the price index of 0.89342 USDCHF is 4. D1 = 0 – 4 = -4.

The remaining differences are calculated in the same way.

After we have calculated all the differences, we can calculate the Spearman rank correlation coefficient:

> ![](https://c.mql5.com/2/58/01__4.png)

At first glance, the difference between the Pearson and Spearman coefficients is small.

> ![](https://c.mql5.com/2/58/5.png)

But it can have a significant impact on trading results. Testing the EA with the same parameters showed a better result compared to the Pearson coefficient.

> ![](https://c.mql5.com/2/58/6.png)

It should be remembered that the trading strategy used can be significantly improved. For example, we can use a trailing stop instead of strictly transferring positions to breakeven, while the use of stop loss and take profit helps reducing the load on the deposit.

Much attention should be paid to the choice of the correlation period. The trading style depends on it. A short correlation period indicates a scalping nature of trading, and a large period indicates a trend-following one.

[https://c.mql5.com/2/58/01__5.png](https://c.mql5.com/2/58/01__5.png "https://c.mql5.com/2/58/01__5.png")

### Cointegration

In 1980s, [Clive Granger](https://en.wikipedia.org/wiki/Clive_Granger) came up with the concept of time series [cointegration](https://en.wikipedia.org/wiki/Cointegration "https://en.wikipedia.org/wiki/Cointegration"). Since there is cointegration, then there should be integration first. Let's see what it is.

Let's assume that we have a time series whose values change according to the following law:

> ![](https://c.mql5.com/2/58/01__6.png)

where **_c_** is a constant, while **_rand_** is a random number. The equation looks simple, but it can be used to create interesting motion trajectories. To generate random numbers, we will use the [Statistics](https://www.mql5.com/en/docs/standardlibrary/mathematics/stat) library. This library has all the necessary distributions allowing us to generate integrated time series.

For example, this is what a movement looks like, in which the random component is subject to a uniform distribution.

> ![](https://c.mql5.com/2/58/7.png)

Does it look like a price chart? Now let's replace the uniform distribution with a normal one. We will get a chart more similar to the price movement.

> ![](https://c.mql5.com/2/58/8.png)

But still something is missing. Price charts often feature gaps. Let's take the sum of the normal distribution and the Cauchy distribution as a random variable. It is this distribution that is responsible for black swans, white crows and other surprises. As a result, we get the following time series.

> ![](https://c.mql5.com/2/58/9.png)

Is it possible to somehow use all this integration in trading? Let us assume that we have two integrated series, the random increments of which obey the same law, albeit with different parameters. If we find the difference between these series, then we can expect that the random components of both series will cancel each other out. Then we will be able to identify long-term relationships between these series, and the series themselves will be cointegrated.

In practice, the behavior of cointegrated currency pairs can be monitored using the difference:

> ![](https://c.mql5.com/2/58/01__7.png)

**_k_** and **_m_** ratios should be selected in such a way that the **_d\[i\]_** values deviate from zero as little as possible. Their values can be estimated using the least squares method using the equations:

> ![](https://c.mql5.com/2/58/01__8.png)

> ![](https://c.mql5.com/2/58/01__9.png)

This is what the change in the difference between USDCHF and USDCAD looks like.

> ![](https://c.mql5.com/2/58/10.png)

The value of this difference is not limited in any way either above or below. Its behavior in history is the main criterion when choosing cointegrated pairs. This difference should fluctuate around zero and change sign. The more such changes in signs in history, the better.

The trading strategy for cointegrated currency pairs is simple, and in many ways resembles the correlation strategy. The opening of two oppositely directed positions occurs when the difference between the two instruments reaches a certain maximum or minimum value. These positions must be closed when the difference becomes zero.

The EA working on USDCHF and USDCAD showed the following change in the trade balance for the period from 2021.01.01 to 2023.06.30.

> ![](https://c.mql5.com/2/58/11.png)

To improve the quality of EA trading, the same recommendations apply as for the correlation-based EA.

### Conclusion

As you can see, pair trading strategies are quite usable. However, they require careful study and refinement for practical application.

The following programs were used when writing the article:

| Name | Type | Features |
| --- | --- | --- |
| sPearson | script | **_iPeriod_** \- correlation period<br>Analyzes historical correlations for all symbols available in Market Watch. Upon completion, saves the average correlation values in the Files folder |
| iPearson | indicator | **_SecSymbol_** \- second symbol<br>**_iPeriod_** \- correlation period<br>Shows the current Pearson correlation coefficient |
| sSpearman | script | Analyzes historical Spearman correlation |
| iSpearman | indicator | Shows the current Spearman correlation |
| EA Correlation | EA | EA based on Pearson and Spearman correlations |
| Integrated Series | script | The script shows the capabilities of constructing integrated time series. It is possible to use different distributions |
| sCointegration | script | The script evaluates possible cointegration of currency pairs |
| iCointegration | indicator | The indicator shows the cointegration difference between two currency pairs |
| EA Cointegration | EA | The EA applying cointegration of currency pairs for trading |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13338](https://www.mql5.com/ru/articles/13338)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13338.zip "Download all attachments in the single ZIP archive")

[sPearson.mq5](https://www.mql5.com/en/articles/download/13338/spearson.mq5 "Download sPearson.mq5")(2.33 KB)

[iPearson.mq5](https://www.mql5.com/en/articles/download/13338/ipearson.mq5 "Download iPearson.mq5")(3.14 KB)

[sSpearman.mq5](https://www.mql5.com/en/articles/download/13338/sspearman.mq5 "Download sSpearman.mq5")(2.43 KB)

[iSpearman.mq5](https://www.mql5.com/en/articles/download/13338/ispearman.mq5 "Download iSpearman.mq5")(3.3 KB)

[EA\_Correlation.mq5](https://www.mql5.com/en/articles/download/13338/ea_correlation.mq5 "Download EA_Correlation.mq5")(21.48 KB)

[Integrated\_Series.mq5](https://www.mql5.com/en/articles/download/13338/integrated_series.mq5 "Download Integrated_Series.mq5")(7.36 KB)

[sCointegration.mq5](https://www.mql5.com/en/articles/download/13338/scointegration.mq5 "Download sCointegration.mq5")(2.29 KB)

[iCointegration.mq5](https://www.mql5.com/en/articles/download/13338/icointegration.mq5 "Download iCointegration.mq5")(3.03 KB)

[EA\_Cointegration.mq5](https://www.mql5.com/en/articles/download/13338/ea_cointegration.mq5 "Download EA_Cointegration.mq5")(19.21 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Polynomial models in trading](https://www.mql5.com/en/articles/16779)
- [Trend criteria in trading](https://www.mql5.com/en/articles/16678)
- [Cycles and trading](https://www.mql5.com/en/articles/16494)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/461456)**
(19)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
14 Sep 2023 at 19:54

**Dmitry Fedoseev [#](https://www.mql5.com/ru/forum/454000#comment_49327193):**

Theoretically. But how do you practically calculate this APR?

If the question is about the ATR period, you can take the average period of holding a position on the TS and multiply it by N>1 (for the reserve). It is important that the ATR should be taken into account, because without it, the drain is assured. Practically. The only exception is coincidence when ATR1 is approximately equal to ATR2.

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
14 Sep 2023 at 20:37

**Dmitry Fedoseev [#](https://www.mql5.com/ru/forum/454000/page2#comment_49333709):**

Therefore, we consider the correlation with the sloping lines. It is necessary to count for two periods - long and short. The correlation for the long period will give an understanding, whether the symbols go in one direction or in different directions.

Suddenly - correlation of a long uptrend with a downward sloping line can easily give a positive value greater than 0.5....

For example, the trend was formed by impulses, and the main time the course was corrected downwards.

![Dmitry Fedoseev](https://c.mql5.com/avatar/2014/9/54056F23-4E95.png)

**[Dmitry Fedoseev](https://www.mql5.com/en/users/integer)**
\|
14 Sep 2023 at 23:50

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/454000/page2#comment_49346708):**

suddenly - correlation of a long uptrend with downward sloping lines, can quietly give a positive value greater than 0.5.

For example, the trend was formed by impulses, and the main time the course was corrected downwards.

That's sensational! Show me a numerical series that gives such an effect.

![marker](https://c.mql5.com/avatar/2011/1/4D29A107-873B.jpg)

**[marker](https://www.mql5.com/en/users/marker)**
\|
16 Sep 2023 at 10:14

[https://www.mql5.com/en/signals/2020078?source=Site+Signals+Favourites](https://www.mql5.com/en/signals/2020078?source=Site "https://www.mql5.com/en/signals/2020078?source=Site")

this signal seems to be based on arbitrage (divergence) of eurusd and usdchf pairs.

P/S not an advertisement, just caught my eye and judging by the history of trades it is so.

![Jean Francois Le Bas](https://c.mql5.com/avatar/avatar_na2.png)

**[Jean Francois Le Bas](https://www.mql5.com/en/users/ionone)**
\|
30 Jan 2024 at 18:27

if you get a "Filling" Order add

```
request.type_filling=ORDER_FILLING_IOC;
```

with every request you find

![DRAKON visual programming language — communication tool for MQL developers and customers](https://c.mql5.com/2/58/visual_programming_language_drakon_avatar.png)[DRAKON visual programming language — communication tool for MQL developers and customers](https://www.mql5.com/en/articles/13324)

DRAKON is a visual programming language designed to simplify interaction between specialists from different fields (biologists, physicists, engineers...) with programmers in Russian space projects (for example, in the Buran reusable spacecraft project). In this article, I will talk about how DRAKON makes the creation of algorithms accessible and intuitive, even if you have never encountered code, and also how it is easier for customers to explain their thoughts when ordering trading robots, and for programmers to make fewer mistakes in complex functions.

![Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://c.mql5.com/2/58/Volume_Bill_Williams_indicators_avatar.png)[Ready-made templates for including indicators to Expert Advisors (Part 2): Volume and Bill Williams indicators](https://www.mql5.com/en/articles/13277)

In this article, we will look at standard indicators of the Volume and Bill Williams' indicators category. We will create ready-to-use templates for indicator use in EAs - declaring and setting parameters, indicator initialization and deinitialization, as well as receiving data and signals from indicator buffers in EAs.

![Population optimization algorithms: Shuffled Frog-Leaping algorithm (SFL)](https://c.mql5.com/2/58/Shuffled_Frog_Leaping_SFL_Avatar.png)[Population optimization algorithms: Shuffled Frog-Leaping algorithm (SFL)](https://www.mql5.com/en/articles/13366)

The article presents a detailed description of the shuffled frog-leaping (SFL) algorithm and its capabilities in solving optimization problems. The SFL algorithm is inspired by the behavior of frogs in their natural environment and offers a new approach to function optimization. The SFL algorithm is an efficient and flexible tool capable of processing a variety of data types and achieving optimal solutions.

![Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://c.mql5.com/2/65/Data_Science_and_Machine_Learning_4Part_19y_Supercharge_Your_AI_models_with_AdaBoost___LOGO.png)[Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://www.mql5.com/en/articles/14034)

AdaBoost, a powerful boosting algorithm designed to elevate the performance of your AI models. AdaBoost, short for Adaptive Boosting, is a sophisticated ensemble learning technique that seamlessly integrates weak learners, enhancing their collective predictive strength.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/13338&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082902604695998741)

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