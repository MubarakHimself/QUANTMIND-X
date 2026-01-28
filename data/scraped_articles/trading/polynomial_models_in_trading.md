---
title: Polynomial models in trading
url: https://www.mql5.com/en/articles/16779
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:27:20.801313
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/16779&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082866711654305927)

MetaTrader 5 / Trading


### Introduction

Trading efficiency largely depends on the methods of analyzing market data. One such method is orthogonal polynomials. These polynomials are mathematical functions that can be used to solve a number of problems related to trading.

The most famous orthogonal polynomials are the Legendre, Chebyshev, Laguerre and Hermite polynomials. Each of these polynomials has unique properties that allow them to be used to solve different problems. Here are some of the main ways to use them:

- **Simulating time series.** Orthogonal polynomials can be used to describe time series. Their use can help in identifying trends and other patterns.
- **Regression.** Orthogonal polynomials can be applied in regression analysis. Their use allows us to improve the quality of the model and make it more interpretable.
- **Forecasting.** Orthogonal polynomials can be used to make predictions about what the price will be if current trends continue.

Let's see how orthogonal polynomials can be applied in practice.

### Orthogonal polynomials and indicators

The whole point of technical analysis comes down to identifying patterns in price movements. But financial time series typically contain noise that obscures these patterns. Let's see how orthogonal polynomials can be applied in a market setting.

The basic idea is that these polynomials can be used to decompose complex signals into simpler components. This decomposition allows us to sort out noise and identify trends.

As an example, I will use [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials "https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%87%D0%BB%D0%B5%D0%BD%D1%8B_%D0%9B%D0%B5%D0%B6%D0%B0%D0%BD%D0%B4%D1%80%D0%B0") to make a smoothing indicator based on them. The general equation of these polynomials and their definition domain are given by the following expressions:

![](https://c.mql5.com/2/168/0__17.png)

I will use polynomials up to the 9th degree. This is quite sufficient for smoothing, and using higher degree polynomials can add noise and hide the main trends in price movement. The polynomial equations used here are given in the table.

| n | Legendre polynomials |
| --- | --- |
| 0 | 1 |
| 1 | x |
| 2 | (3\*x^2-1)/2 |
| 3 | (5\*x^3-3\*x)/2 |
| 4 | (35\*x^4-30\*x^2+3)/8 |
| 5 | (63\*x^5-70\*x^3+15\*x)/8 |
| 6 | (231\*x^6-315\*x^4+105\*x^2-5)/16 |
| 7 | (429\*x^7-693\*x^5+315\*x^3-35\*x)/16 |
| 8 | (6435\*x^8-12012\*x^6+6930\*x^4-1260\*x^2+35)/128 |
| 9 | (12155\*x^9-25740\*x^7+18018\*x^5-4620\*x^3+315\*x)/128 |

First of all, I need to convert the price indices into the domain of these polynomials. To do this, I apply the shift function for each i index:

![](https://c.mql5.com/2/168/0__18.png)

After that, I calculate the values of all the polynomials I am interested in for each value of x\[i\]. For example, I will calculate the value of a 2nd degree polynomial with the period of 3:

![](https://c.mql5.com/2/168/0__19.png)

![](https://c.mql5.com/2/168/0__20.png)

![](https://c.mql5.com/2/168/0__21.png)

Now I need to introduce a correction for discreteness. The sum of the values of a polynomial of degrees 1 and higher should be equal to zero. To fulfill this condition, I need to calculate the correction:

![](https://c.mql5.com/2/168/0__22.png)

With this correction, I adjust the values of the polynomial:

![](https://c.mql5.com/2/168/0__23.png)

Now the most interesting part begins. Any time series can be decomposed into a sum of polynomials taken with a certain weight:

![](https://c.mql5.com/2/168/0__24.png)

The weights themselves can be calculated as follows:

![](https://c.mql5.com/2/168/0__25.png)

It looks strange and a little scary. In fact, everything is simple. Let's take a polynomial of degree 0 with period N. Its values at all points are equal to 1. And its weight will be:

![](https://c.mql5.com/2/168/0__26.png)

This is the [SMA](https://www.mql5.com/en/docs/indicators/ima) equation. The weights of higher-order polynomials are equivalent to some oscillators with specially selected ratios. In other words, orthogonal polynomials are SMA with some cleverly calculated additions. Legendre polynomials are very robust to noise and perform excellently in sorting applications. For example, the Legendre polynomial of the first degree looks like this.

![](https://c.mql5.com/2/168/1__10.png)

[Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials "https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%87%D0%BB%D0%B5%D0%BD%D1%8B_%D0%A7%D0%B5%D0%B1%D1%8B%D1%88%D1%91%D0%B2%D0%B0") can also be used for trend detection and smoothing in addition to Legendre polynomials. There are two types of such polynomials. They differ from each other in their behavior at the edges. The main advantage of these polynomials is their sensitivity to sudden price changes. The equation for these polynomials is very simple:

![](https://c.mql5.com/2/168/0__27.png)

| n | polynomials of the first kind | polynomials of the second kind |
| --- | --- | --- |
| 0 | 1 | 1 |
| 1 | x | 2\*x |
| 2 | 2\*x^2-1 | 4\*x^2-1 |
| 3 | 4\*x^3-3\*x | 8\*x^3-4\*x |
| 4 | 8\*x^4-8\*x^2+1 | 16\*x^4-12\*x^2+1 |
| 5 | 16\*x^5-20\*x^3+5\*x | 32\*x^5-32\*x^3+6\*x |
| 6 | 32\*x^6-48\*x^4+18\*x^2-1 | 64\*x^6-80\*x^4+24\*x^2-1 |
| 7 | 64\*x^7-112\*x^5+56\*x^3-7\*x | 128\*x^7-192\*x^5+80\*x^3-8\*x |
| 8 | 128\*x^8-256\*x^6+160\*x^4-32\*x^2+1 | 256\*x^8-448\*x^6+240\*x^4-40\*x^2+1 |
| 9 | 256\*x^9-576\*x^7+432\*x^5-120\*x^3+9\*x | 512\*x^9-1024\*x^7+672\*x^5-160\*x^3+10\*x |

This is what smoothing with the Chebyshev polynomial of the 3rd degree looks like.

![](https://c.mql5.com/2/168/2__10.png)

So far we have considered polynomials with the definition domain of +/-1, but there are polynomials with other domains. For example, [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials "https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%87%D0%BB%D0%B5%D0%BD%D1%8B_%D0%9B%D0%B0%D0%B3%D0%B5%D1%80%D1%80%D0%B0") is defined for all non-negative values of the argument. Its equation looks like this:

![](https://c.mql5.com/2/168/0__28.png)

| n | Laguerre polynomials |
| --- | --- |
| 0 | 1 |
| 1 | -x+1 |
| 2 | (x^2-4\*x+2)/2 |
| 3 | (-x^3+9\*x^2-18\*x+6)/6 |
| 4 | (x^4-16\*x^3+72\*x^2-96\*x+24)/24 |
| 5 | (-x^5+25\*x^4-200\*x^3+600\*x^2-600\*x+120)/120 |
| 6 | (x^6-36\*x^5+450\*x^4-2400\*x^3+5400\*x^2-4320\*x+720)/720 |
| 7 | (-x^7+49\*x^6-882\*x^5+7350\*x^4-29400\*x^3+52920\*x^2-35280\*x+5040)/5040 |
| 8 | (x^8-64\*x^7+1568\*x^6-18816\*x^5+117600\*x^4-376320\*x^3+564480\*x^2-322560\*x+40320)/40320 |
| 9 | (-x^9+81\*x^8-2592\*x^7+42336\*x^6-381024\*x^5+1905120\*x^4-5080320\*x^3+6531840\*x^2-3265920\*x+362880)/362880 |

And the shift function for its argument looks like this:

![](https://c.mql5.com/2/168/0__29.png)

This change leads to the fact that the behavior of the Laguerre polynomial depends not only on the degree, but also on its period. This polynomial is sensitive to recent price changes. For example, this is what the Laguerre polynomial of the fifth degree looks like on a graph:

![](https://c.mql5.com/2/168/3__10.png)

Another interesting example of orthogonality is [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials "https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%87%D0%BB%D0%B5%D0%BD%D1%8B_%D0%AD%D1%80%D0%BC%D0%B8%D1%82%D0%B0"). These polynomials are used in many areas of math and physics. These polynomials are defined for any values of the argument, and their equation looks like this:

![](https://c.mql5.com/2/168/0__30.png)

| n | Hermite polynomials |
| --- | --- |
| 0 | 1 |
| 1 | x |
| 2 | x^2-1 |
| 3 | x^3-3\*x |
| 4 | x^4-6\*x^2+3 |
| 5 | x^5-10\*x^3+15\*x |
| 6 | x^6-15\*x^4+45\*x^2-15 |
| 7 | x^7-21\*x^5+105\*x^3-105\*x |
| 8 | x^8-28\*x^6+210\*x^4-420\*x^2+105 |
| 9 | x^9-36\*x^7+378\*x^5-1260\*x^3+945\*x |

The shift function centers the price values:

![](https://c.mql5.com/2/168/0__31.png)

As a result, we obtained a smoothing filter, the efficiency of which depends on the degree of the polynomial and its period.

![](https://c.mql5.com/2/168/4__10.png)

We have considered the basic classical orthogonal polynomials. But it is quite possible to create your own version of such polynomials. For example, a polynomial that combines the advantages of the Chebyshev and Hermite polynomials is given by the equation:

![](https://c.mql5.com/2/168/0__32.png)

| n | Chebyshev-Hermite polynomials |
| --- | --- |
| 0 | 1 |
| 1 | 2\*x |
| 2 | 4\*x^2-2 |
| 3 | 8\*x^3-12\*x |
| 4 | 16\*x^4-48\*x^2+12 |
| 5 | 32\*x^5-160\*x^3+120\*x |
| 6 | 64\*x^6-480\*x^4+720\*x^2-120 |
| 7 | 128\*x^7-1344\*x^5+3360\*x^3-1680\*x |
| 8 | 256\*x^8-3584\*x^6+13440\*x^4-13440\*x^2+1680 |
| 9 | 512\*x^9-9216\*x^7+48384\*x^5-80640\*x^3+30240\*x |

This polynomial is sensitive to nonlinear trends - quadratic, cubic, etc. This is what the Chebyshev-Hermite polynomial of the 9th degree looks like on a graph:

![](https://c.mql5.com/2/168/5__9.png)

The use of orthogonal polynomials provides a number of advantages.

- **_Stability and de-correlation._** The orthogonality of polynomials ensures their stability to changes in model parameters. Each polynomial is independent of the others, which allows each component of the time series to be modeled and studied separately.
- **_Interpretability._** Each orthogonal polynomial corresponds to its own model of price behavior. The weighting ratios of polynomials allow us to identify the most important models and focus on them.
- **_Adaptability and efficiency._** Orthogonal polynomials are adjusted to specific values of the time series. The emergence of new prices leads to a change in the weighting ratios, due to which the polynomial model adapts to the current state of the market. And the use of nonlinear polynomials makes this adaptation efficient.

The only feature of orthogonal polynomials that can be considered a disadvantage is that these polynomials handle all prices inside the polynomial. In other words, an indicator built on such polynomials "draws". In my opinion, this feature is not a disadvantage - the indicator simply finds the best approximation when new data arrives.

The use of orthogonal polynomials imposes some limitations. The indicator period should be greater than the degree of the polynomial. For practical purposes, we can limit ourselves to the 3rd degree of the polynomial. This indicator combines SMA, linear trend and two parabolas - quadratic and cubic. This is enough to smooth out quite complex situations. But, by increasing the indicator period, we can also increase the degree of the polynomial.

Now, let's look at some examples of the application of orthogonal polynomials in trading.

### Trading strategies

Based on orthogonal polynomials, traders can create various trading strategies. For example, using polynomial regression based on orthogonal polynomials, a trader can not only analyze past price movements, but also create a model that can adapt to current changes in the market.

As an example, let's look at several trading strategies based on deviations from the mean.

Let's take the simplest strategy - positions are opened when the price crosses the SMA. We will replace the moving average with an orthogonal polynomial leaving the rules for opening and closing positions unchanged:

- The price crosses the polynomial from bottom to top - open a buy position, close a sell position.
- The price crosses the polynomial from top to bottom - open a sell position, close a buy position.

Despite its simplicity, this strategy is quite efficient.

![](https://c.mql5.com/2/168/6__6.png)

Let's complicate this strategy a little and replace the price with the value of some polynomial. In other words, we will get an analogue of the strategy with the intersection of two SMAs. The result is somewhat predictable: of all possible options, the strategy optimizer preferred orthogonal polynomials.

![](https://c.mql5.com/2/168/7__2.png)

Polynomial models can also be used in more complex strategies. If the strategy uses any indicators, you can try to replace them with orthogonal polynomials. Such a replacement could improve the strategy.

I will use the [Commodity channel index (CCI)](https://en.wikipedia.org/wiki/Commodity_channel_index "https://ru.wikipedia.org/wiki/%D0%98%D0%BD%D0%B4%D0%B5%D0%BA%D1%81_%D1%82%D0%BE%D0%B2%D0%B0%D1%80%D0%BD%D0%BE%D0%B3%D0%BE_%D0%BA%D0%B0%D0%BD%D0%B0%D0%BB%D0%B0") indicator as an example. Its classic equation looks like this:

![](https://c.mql5.com/2/168/0__33.png)

I will make some minor changes to this indicator. As I already said: SMA is a polynomial of degree 0. Instead of SMA, I will use an orthogonal polynomial of any degree. Accordingly, I will calculate the mean absolute deviation relative to this polynomial. The resulting indicator is as follows:

![](https://c.mql5.com/2/168/8__1.png)

The trading strategy will be similar to the classic one:

- open a buy position if the indicator is below the specified level and continues to decline;
- open a sell position if the indicator is above the specified level and continues to rise;
- close positions on the opposite signal.

The result of testing such a strategy with a 3rd degree polynomial:

![](https://c.mql5.com/2/168/9__2.png)

Other indicators can also be constructed based on orthogonal polynomials. For example, I want to construct an equivalent of [RSI](https://www.mql5.com/en/docs/indicators/irsi). The essence is simple: first I build a polynomial, and then I count how many prices were located above this polynomial. Based on this number, I conclude that the price is overbought/oversold.

I use the indicator to create a simple strategy:

- open a position if the indicator has reached the overbought or oversold level;
- close positions if the indicator is in the center.

![](https://c.mql5.com/2/168/10__2.png)

Moreover, orthogonal polynomials can be integrated into various machine learning algorithms. The flexibility of polynomial functions enables machine learning algorithms to detect complex relationships in data.

For example, polynomials can be used to generate new features from source data. Orthogonal polynomials provide independent estimates of the different components of a time series. Due to this property, they can reduce overfitting. This can improve the quality of training data and improve the performance of models.

Take any [neural network](https://www.mql5.com/en/neurobook) and feed the values of the weighting ratios of the orthogonal polynomials to it. But in this case, it will be necessary to construct a complete system of polynomials: starting with a polynomial of the 1st degree, and ending with N-1, where N is the number of prices processed.

### Conclusion

Orthogonal polynomials are a powerful tool for analyzing financial time series. They provide a number of benefits and allow us to assess changes in the market. Using orthogonal polynomials in trading strategies can improve their efficiency and enhance trading results.

The following programs were used when writing this article.

| Name | Type | Description |
| --- | --- | --- |
| Orthogonal polynomials | indicator | simulates orthogonal polynomials on a graph<br>- **_Type_** \- polynomial type<br>- **_iPeriod_** \- polynomial period (not less than 2)<br>- **_N_** \- polynomial degree, acceptable values 0 - 9<br>- **_Shift_** \- indicator shift |
| EA Orthogonal polynomials | EA | implements a trading strategy based on the intersection of price and polynomial |
| EA Orthogonal polynomials 2 | EA | implements a strategy at the intersection of two polynomials |
| Orthogonal CCI | indicator | CCI, in which orthogonal polynomials can be used instead of SMA |
| EA Orthogonal CCI | EA | implements a strategy based on the orthogonal variant of CCI |
| Orthogonal RSI | indicator | determines overbought/oversold conditions using orthogonal polynomials |
| EA Orthogonal RSI | EA | implements a strategy based on the orthogonal version of RSI |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16779](https://www.mql5.com/ru/articles/16779)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16779.zip "Download all attachments in the single ZIP archive")

[Orthogonal\_polynomials.mq5](https://www.mql5.com/en/articles/download/16779/Orthogonal_polynomials.mq5 "Download Orthogonal_polynomials.mq5")(8.89 KB)

[EA\_Orthogonal\_polynomials.mq5](https://www.mql5.com/en/articles/download/16779/EA_Orthogonal_polynomials.mq5 "Download EA_Orthogonal_polynomials.mq5")(3.64 KB)

[EA\_Orthogonal\_polynomials\_2.mq5](https://www.mql5.com/en/articles/download/16779/EA_Orthogonal_polynomials_2.mq5 "Download EA_Orthogonal_polynomials_2.mq5")(3.94 KB)

[Orthogonal\_CCI.mq5](https://www.mql5.com/en/articles/download/16779/Orthogonal_CCI.mq5 "Download Orthogonal_CCI.mq5")(9.08 KB)

[EA\_Orthogonal\_CCI.mq5](https://www.mql5.com/en/articles/download/16779/EA_Orthogonal_CCI.mq5 "Download EA_Orthogonal_CCI.mq5")(3.61 KB)

[Orthogonal\_RSI.mq5](https://www.mql5.com/en/articles/download/16779/Orthogonal_RSI.mq5 "Download Orthogonal_RSI.mq5")(9.15 KB)

[EA\_Orthogonal\_RSI.mq5](https://www.mql5.com/en/articles/download/16779/EA_Orthogonal_RSI.mq5 "Download EA_Orthogonal_RSI.mq5")(3.72 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Trend criteria in trading](https://www.mql5.com/en/articles/16678)
- [Cycles and trading](https://www.mql5.com/en/articles/16494)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/495035)**
(2)


![Tretyakov Rostyslav](https://c.mql5.com/avatar/2020/1/5E2C375E-C6C8.jpg)

**[Tretyakov Rostyslav](https://www.mql5.com/en/users/makarfx)**
\|
10 Jan 2025 at 14:08

**MetaQuotes:**

The article [Polynomial models in trading](https://www.mql5.com/en/articles/16779) was published:

Author: [Aleksej Poljakov](https://www.mql5.com/en/users/Aleksej1966 "Aleksej1966")

Thank you. As always, a lot of food for thought.


![Chike Kene Assuzu](https://c.mql5.com/avatar/avatar_na2.png)

**[Chike Kene Assuzu](https://www.mql5.com/en/users/mercuryassuzu)**
\|
12 Sep 2025 at 23:47

a very impressive concept


![Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://c.mql5.com/2/168/19365-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 39): Automating BOS and ChoCH Detection in MQL5](https://www.mql5.com/en/articles/19365)

This article presents Fractal Reaction System, a compact MQL5 system that converts fractal pivots into actionable market-structure signals. Using closed-bar logic to avoid repainting, the EA detects Change-of-Character (ChoCH) warnings and confirms Breaks-of-Structure (BOS), draws persistent chart objects, and logs/alerts every confirmed event (desktop, mobile and sound). Read on for the algorithm design, implementation notes, testing results and the full EA code so you can compile, test and deploy the detector yourself.

![Market Simulation (Part 01): Cross Orders (I)](https://c.mql5.com/2/107/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 01): Cross Orders (I)](https://www.mql5.com/en/articles/12536)

Today we will begin the second stage, where we will look at the market replay/simulation system. First, we will show a possible solution for cross orders. I will show you the solution, but it is not final yet. It will be a possible solution to a problem that we will need to solve in the near future.

![From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://c.mql5.com/2/168/19299-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://www.mql5.com/en/articles/19299)

Today we will develop a multi-chart view system using chart objects. The goal is to enhance news trading by applying MQL5 algorithms that help reduce trader reaction time during periods of high volatility, such as major news releases. In this case, we provide traders with an integrated way to monitor multiple major symbols within a single all-in-one news trading tool. Our work is continuously advancing with the News Headline EA, which now features a growing set of functions that add real value both for traders using fully automated systems and for those who prefer manual trading assisted by algorithms. Explore more knowledge, insights, and practical ideas by clicking through and joining this discussion.

![Big Bang - Big Crunch (BBBC) algorithm](https://c.mql5.com/2/108/16701-logo.png)[Big Bang - Big Crunch (BBBC) algorithm](https://www.mql5.com/en/articles/16701)

The article presents the Big Bang - Big Crunch method, which has two key phases: cyclic generation of random points and their compression to the optimal solution. This approach combines exploration and refinement, allowing us to gradually find better solutions and open up new optimization opportunities.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fsscsemfzhukljpvkpotnxcajwnwbkim&ssn=1769250439948099601&ssn_dr=0&ssn_sr=0&fv_date=1769250439&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16779&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Polynomial%20models%20in%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925043997282954&fz_uniq=5082866711654305927&sv=2552)

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