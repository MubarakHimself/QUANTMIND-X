---
title: Market Theory
url: https://www.mql5.com/en/articles/1825
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:38:04.244116
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/1825&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083003278729417483)

MetaTrader 5 / Trading


### Introduction

**Market** is a mechanism of commodity-money relations that operates according to relevant laws, connects buyers (demand representatives) and sellers (supply representatives), and forms buy/sell prices. The price acts as a main landmark for the market relations. It holds (in accordance with the labour theory of value) the monetary value of goods as a product of labour \[1\].

In general, three types of markets can be distinguished:

1. Monopolistic competition with open pricing based on the price competition, the so-called competitive market. In this market seller's income is decreased with the increase of goods' sales price, and vice versa. The income elasticity coefficient from the sales price is always negative, and the profit varies with the change of sales price following the complex pattern. This market type is common for various goods and services markets.

2. Perfect competition where sellers are not in position to change the goods' sales price due to a high competition among sellers. Any attempts to change the price force the participant to exit the market. This type is most common for wholesale markets.

3. Monopolistic market where a seller or a group of sellers have an opportunity to set the goods' sales price that is most profitable for them. Their revenue and profit is increased according to the increase of goods' sales prices, and, therefore, the elasticity coefficient in such markets is always positive.


The researchers felt that the price has a paramount role in the market management mechanism, which is known as 3 axioms of Dow Theory \[2\]. It still remains unknown, how the price manages to control the market mechanism and form its sole correct value as a market price level that shapes the market in a wonderful and mysterious manner that is not yet understood.

**The goals of this article are:**

1. Discover the role of price in the market management mechanism based on the principle of balancing interests of its participants to ensure market stability.

2. Identify the types, formation nature, reasons of appearance and the extent of interaction between different levels of real and virtual market prices.

3. Identify the market type by the nature of changing the current price and by learning the mechanism of their interconversion.

4. Establish a common pattern between the market of services and goods and the Forex market.

5. Find out the reasons of formation and change of trends.

6. Generate signals for market entry and exit.

7. Establish the principle of the Forex market operation.

8. Study the relevance and potential of applying the market theory conclusions in order to create and use various indicators and Expert Advisors for trading.


We are going to begin the development of the market theory by identifying the interests of its participants. It is natural to assume that the market participants are driven by the opportunity to gain profit or other benefits to satisfy their material and spiritual needs. Therefore, we will choose it as a factor for maintaining the balance of interests for all sides of the market — both sellers and buyers. Since in the market conditions profit mainly depends on the goods' sales price, first we are going to identify this dependence.

### **Dependence of return on the goods' sales price**

Generally, in order to calculate the return (R), all types of costs (C) (variable (Cv) and fixed costs (Cf)) are deducted from total sales (S):

![](https://c.mql5.com/2/19/formulae_html_362da2b2.gif)

As simple and obvious this formula may be, it also includes the dependence from the goods' sales price (Ps) in the implicit form, which complicates the analysis, if Ps changes. Let's try to obtain the dependence of return on the sales price in the explicit form.

We know that in the market conditions the dependence of the quantity of sold goods (Q) on the sales price (Ps) can be expressed with the hyperbole equation that reflects the law of supply and demand:

![](https://c.mql5.com/2/19/formulae_html_13c238d2.gif)

By multiplying both parts of the equation by Ps, we obtain the dependence of total sales (St) on sales price (Ps), and since QPs = S, then:

![](https://c.mql5.com/2/19/formulae_html_aa5898c.gif)

Therefore, in the market conditions the dependence of total sales (St) on the sales price (Ps) is linear, and now the coefficients in equations (2) and (3) acquire a clear physical meaning, namely, the S coefficient is the maximum virtual sale achieved by the unlimited decrease of Ps to its insignificant values, and it indirectly reveals the potential market demand for these goods. We will mark Y coefficient as the elasticity of total sales from the goods' sales price, which numerically reveals the change of total sales when the goods' sales price changes by unit in the payment currency of goods' price.

Obviously, if there is a competition in market conditions, the elasticity always holds a negative value, i.e. Y ≤ 0 and the increase of sales price (Ps) by unit leads to the decrease of total sales by Y units in the payment currency of goods' price and, vice versa, the decrease of sales price (Ps) by unit leads to the increase of total return by Y units.

If there is an actual data array consisting of n sets of Ps and St values, numerical values of Y elasticity and virtual S sales are defined using the method of least squares:

![](https://c.mql5.com/2/19/formulae_html_bf90e6d.gif)

![](https://c.mql5.com/2/19/formulae_html_m1a565fdc.gif)

Now, if we zero out the equation (3), we will find the limit realization price (Pl) above which goods can't be sold on this market because it exceeds the allowable value, and goods lose their attractiveness to buyers:

![](https://c.mql5.com/2/19/formulae_html_5c4db679.gif)

We will introduce the concept of the competition level (C) on the analyzed market and determine its numerical value as the ratio of difference between the sales price and the original price (Po) (its buy price and the production cost) towards the difference between the limit realization price (Pl) and the original price (Po):

![](https://c.mql5.com/2/19/formulae_html_514926f7.gif)

For example, the current or actual level of competition Cs on the market at the sales price (Ps) is calculated accordingly:

![](https://c.mql5.com/2/19/formulae_html_d7bc3cb.gif)

Let's introduce the degree of excess Ps sales price (ds) the original price (Po) and the degree of limit realization price (dl):

![](https://c.mql5.com/2/19/formulae_html_m23660f79.gif)

![](https://c.mql5.com/2/19/formulae_html_m34788282.gif)

![](https://c.mql5.com/2/19/formulae_html_m2ceae613.gif)

![](https://c.mql5.com/2/19/formulae_html_1ccfe053.gif)

Variable costs (Cv) consist of expenses for buying or producing merchantable goods (Cg) and other variable costs (Co) that depend on the total sales (St). We will define them through the fraction of costs for obtaining goods in the total sales _L_ (level of competition) at the current sales price (Ps) and the fraction of other variable costs (v) in the total sales (St):

![](https://c.mql5.com/2/19/formulae_html_2f356cbd.gif)

![](https://c.mql5.com/2/19/formulae_html_m38c7b547.gif)

![](https://c.mql5.com/2/19/formulae_html_m2877f0a9.gif)

where:

![](https://c.mql5.com/2/21/formulae_html_694318e2.gif)

![](https://c.mql5.com/2/21/formulae_html_68b290a3.gif)

Net revenue Sp will be presented through its h fraction in the total sales (St):

![](https://c.mql5.com/2/19/formulae_html_cbb0081.gif)

where:

![](https://c.mql5.com/2/19/formulae_html_23a7096e.gif)

The impact of fixed costs (Cf) on the return will be considered with the coefficient of fixed expenses (df) calculated accordingly:

![](https://c.mql5.com/2/19/formulae_html_7f2a86e9.gif)

![](https://c.mql5.com/2/21/formulae_html_314e0067.gif)

Presence of other variable costs (Co) from their v fractions leads to the increase of prices for goods, obtained at the Po price, to the Pv value:

![](https://c.mql5.com/2/19/formulae_html_9dee13c.gif)

where:

![](https://c.mql5.com/2/19/formulae_html_23b8af65.gif)

Let's introduce the market price (Pm), which will be defined numerically using the dm market coefficient as follows:

![](https://c.mql5.com/2/19/formulae_html_m481a2d76.gif)

where:

![](https://c.mql5.com/2/19/formulae_html_m3b64937c.gif)

Now, if we introduce the concept of optimal realization price (Popt) that guarantees the maximum return (Rmax), the equation (1) becomes a formula to determine the return (R) at any values of sales price (Ps):

![](https://c.mql5.com/2/19/formulae_html_m4be812ce.gif)

If we zero out the first derivative from (25) by Ps, we will obtain the ratio to determine Popt:

![](https://c.mql5.com/2/19/formulae_html_m57b234f6.gif)

The concept of an optimal level of exceeding the realization price (dopt) is entered here:

![](https://c.mql5.com/2/19/formulae_html_m4892d2cd.gif)

By placing the Ps=Popt in (25), we will get the form to define the maximum return:

![](https://c.mql5.com/2/19/formulae_html_29f82cab.gif)

If we zero out the return formula (25), we will obtain the ratio to determine two break-even points P1 and P2 as solutions to the obtained quadric equation:

![](https://c.mql5.com/2/19/formulae_html_221a05f6.gif)

![](https://c.mql5.com/2/19/formulae_html_69877753.gif)

or:

![](https://c.mql5.com/2/19/formulae_html_m7b8c38c.gif)

![](https://c.mql5.com/2/19/formulae_html_m23756c5b.gif)

![](https://c.mql5.com/2/19/formulae_html_m4cc0632e.gif)

![](https://c.mql5.com/2/19/formulae_html_501826bd.gif)

![](https://c.mql5.com/2/19/formulae_html_m46c3c7ac.gif)

Now the equation (25) can also be presented as:

![](https://c.mql5.com/2/19/formulae_html_64183612.gif)

![](https://c.mql5.com/2/19/formulae_html_m47c98535.gif)

By comparing (25) and (35) we get:

![](https://c.mql5.com/2/19/formulae_html_624e42ce.gif)

![](https://c.mql5.com/2/19/formulae_html_m3d1d358.gif)

Therefore, the market price (Pm) is an arithmetic average, and the optimal realization price (Popt) is a geometric average price value corresponding to two break-even points.

A French mathematician Louis Cauchy (1789-1857) proved that the arithmetic average of two non-negative numbers is no less than their geometric average, and equality can only be achieved when P1=P2:

![](https://c.mql5.com/2/19/formulae_html_10b91ea2.gif)

Therefore, the optimal realization price of goods (Popt) is always less than the established market price (Pm) depending on the existing competition level, actual values of fixed and variable costs based on the ratio (24):

![](https://c.mql5.com/2/19/formulae_html_m5489632.gif)

We are going to name it the Cauchy difference![](https://c.mql5.com/2/19/formulae_html_7c969947111.gif):

![](https://c.mql5.com/2/19/formulae_html_7c969947.gif)

Now the ratio for defining the maximum return (28) becomes:

![](https://c.mql5.com/2/19/formulae_html_m1e83d9b9.gif)

In general, the price hierarchy on the market is built as follows:

![](https://c.mql5.com/2/19/formulae_html_7ddb2f70.gif)

The general break-even point (Pg) occurs when meeting the conditions:

![](https://c.mql5.com/2/19/formulae_html_m6e93c86c.gif)

Based on (23), (24), (26), (27) and (39) we conclude that the following inequation is always valid on the market:

![](https://c.mql5.com/2/19/formulae_html_m38dc582.gif)

By solving the inequation (45) towards Po we come to conclusion that to get the return in the current market conditions, the buy price or the regular price (Pr) of goods shouldn't exceed its maximum value Pol:

![](https://c.mql5.com/2/19/formulae_html_694c0fe4.gif)

![](https://c.mql5.com/2/19/formulae_html_m152f8730.gif)

![](https://c.mql5.com/2/19/formulae_html_m112533da.gif)

Similarly, by solving the inequation (45) towards v, we find that the proportion of other variable expenses is limited with the ratio:

![](https://c.mql5.com/2/19/formulae_html_7553206c.gif)

By solving the inequation (28) towards the fixed costs (Cf), we discover that they are limited by the ratio:

![](https://c.mql5.com/2/19/formulae_html_32ca8082.gif)

### **Real and virtual price levels on the market**

According to the analyzed theory, multiple real and virtual price levels are formed and operate on the market. Let's list them and give a short description to all the price levels.

Real price levels:

1. Po — buy price or production price (cost price) of goods;

2. Pol — maximum buy price — when exceeded, the return from reselling it on the market can't be obtained, defined by ratios (46) - (48);

3. Pv — buy price provided variable expenses are present, defined by (21) and (22);

4. Pfc = df\*Po — price of fixed costs provided variable expenses are present, defined by (20);

5. Ps — sales price of goods.


Virtual price levels:

1. P1 — first break-even or non-profit level, calculated as (30-34);

2. Popt — optimal price level that allows to obtain a maximum profit when selling goods at this price, in some circumstances may become a global break-even level, for example, when reaching maximum variable value and/or fixed costs, calculated as (26);

3. Pm — average market price, calculated as (23-24);

4. P2 — second break-even or non-profit level, calculated as (30-34);

5. Pl — maximum sell price — when exceeded, goods on the market can't be sold, calculated as (6).


### **Approbation of theory conclusions using the example of selling goods on the market**

We will use values of price levels and matching of actual and estimated return values as an example, and make sure how accurately the algorithm, which currently analyses the Forex market, describes and analyses the situation of the trading process on the real market of goods and services based on the example of the single entrepreneur analysis.

Supposedly, the entrepreneur has purchased the goods at the price Po=100 dollars per unit in order to resell them on the market (in the shop, supermarket etc.). During the first day of trading, while selling goods at the price Ps1=112 dollars per unit, he gained profit of St1=59300 dollars. On the second day he increased the sales price to Ps2=118 dollars and his return amounted to only St2=8800 dollars. Variable costs, including taxes, account for 10% from the income, and fixed costs are located at the level of Cf=200 dollars daily.

We have to determine the entrepreneur's return (R) for two trading days, break-even points P1 and P2, maximum values of goods' buy price (Pol), variables (Cv) and fixed costs (Cf) that after being exceeded cease profit gaining, and also to analyze the market in order to determine the optimal sales price of goods (Popt) that guarantees maximum profit (Pmax).

This is how the problem is solved by the algorithm, which after being analyzed will leave no doubts that the real market theory has been actually discovered:

| Po | Price | Ps | St | Cg | v | Cv | Cf | R | P1 | Popt | Pm | P2 | Pl | R(Ps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | Ps1 | 112 | 59200 | 52857 | 0.1 | 5920 | 200 | 222.86 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | 222.86 |
| 100 | Ps2 | 118 | 8800 | 7458 | 0.1 | 880 | 200 | 262.37 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | 262.37 |
| 100 | P1 | 111.5020 | 63383 | 56845 | 0.1 | 633.83 | 200 | 0 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | 0 |
| 100 | Popt | 111.0109 | 33908 | 29483 | 0.1 | 339.08 | 200 | 834.79 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | 834.79 |
| 100 | P2 | 118.6302 | 3506 | 2955 | 0.1 | 350.6 | 200 | 0 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | 0 |
| 100 | Pl | 119.0476 | 0 | 0 | 0.1 | 0 | 200 | -200 | 111.5020 | 115.0109 | 115.0661 | 118.6302 | 119.0476 | -200 |
| Pol=103.9723 | Popt | 117.2730 | 14907 | 13216 | 0.1 | 149.07 | 200 | 0 | 117.2730 | 117.2730 | 117.2730 | 117.2730 | 119.0476 | 0 |
| 100 | Popt | 120.9134 | -15672 | -12961 | 0.1857 | -2911 | 200 | 0 | 120.9134 | 120.9134 | 120.9134 | 120.9134 | 119.0476 | 0 |
| 100 | Popt | 115.0109 | 33908 | 29483 | 0.1 | 339.08 | 1034.79 | 0 | 115.0109 | 115.0109 | 115.0109 | 115.0109 | 119.0476 | 0 |

![](https://c.mql5.com/2/21/10_06_15_2_3fydc.png)

![](https://c.mql5.com/2/21/10_06_15_5_qq975.png)

### **Testing the theory on real Forex market data**

Let's look at the market theories that are most well-known among traders who try to use them in order to gain a statistic advantage when organizing profitable trading and creating profitable trading strategies on the Forex market based on them. There are three main theories:

1. [Gann Theory](https://www.mql5.com/go?link=http://www.scribd.com/doc/18040260/Gann-Theory-Overview "http://www.fxguild.info/content/view/446/36/") is a product of practical study of the model, price and time ratios and their influence on the market.

2. [Elliott Wave](https://www.mql5.com/go?link=http://www.investopedia.com/articles/technical/111401.asp "http://www.finam.ru/investor/library0003C000C2/?material=462") — through practical research Mr. Elliott came to the conclusion that any trend consists of the same repeated basic models (sections) that are divided into two types:

a) impulse section ("Impulse") that consists of 5 segments and acts as a moving section with a trend development;

    b) corrective section ("Correction") that consists of 3 segments and compensates for the previous impulse movement.

3. Strategy based on the [Ichimoku Indicator](https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D "https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D") — associated with "Ichimoku cloud" which is a product of the author's practical research of 30 years.


They all have a common feature — the absence of a strong theoretic base showing the real connection to the process of real trading with goods and services. These theories are the result of practical investigations and assumptions of their authors. Furthermore, they are connected with a united idea, namely, the understanding that along the price movement there are some levels and powers affecting its pattern, and the authors have devoted their lives to the frantic search for regularities of forming the indicated levels and forces.

Applying indicated theories in the trading practice led to variable success, however, due to the lack of more reliable theories, the researchers aimed only at the positive results, regardless the convention of their application and the actual money loss, explaining this as the "wrong" interpretation on the trader's side or presenting it as the disadvantages of one or the other theory.

I am trying to convey the essence of the new market theory that doesn't have any disadvantages listed above. This theory is based on a strong theoretical foundation, with equal elegance it describes the process of real trading of goods and Forex trading on the basis of price interaction between three virtual price levels, that the brightest men dedicated their lives to find, but, unfortunately, never succeeded.

These are the levels:

1. Current price level that can become bullish or bearish, depending on the situation. When the market is bullish, the price becomes bearish, and vice versa.

2. Virtual price level is formed by the market, and, like the current price, may turn bullish and bearish, depending on the circumstances.

3. Virtual managing level of the optimal market price — lion level.

4. Virtual managing level of the average market price — leopard level.


The figure shows real and virtual price levels based on the considered market theory over the years 2010 and 2011:

![](https://c.mql5.com/2/21/30_07_2.png)

### **Indicator concept based on the new market theory and principles of market entry and exit**

To create an indicator you need to bear in mind that it must clearly show the market condition at any time, namely the bear market and the bull market, the pattern of market price and line level movement, the change in trend, whether peaceful or showing strife between bulls and bears. The indicator must clearly indicate the moments and levels of the market entry and exit based on the principle:

- bulls manage the market — buy;
- bears manage the market — sell;
- lions manages the market — beyond market.


The future indicator will look according to the charts below.

Market condition on 11.06.2015 00:00 Moscow, EURUSD; recommendation to buy, since bulls are leading the market:

![](https://c.mql5.com/2/21/30_07_3__1.png)

Levels of real and virtual market prices at this moment:

| P = P2 (Bull) | Popt (Lion) | Pm (Leopard) | Pvmp = P1 (Bear) |
| --- | --- | --- | --- |
| 1,13229 | 1,12255439 | 1,1225962 | 1,11290249 |

GBPUSD. All levels will be able to gather at the lion level shortly. Until the situation is cleared, it is recommended to buy:

![](https://c.mql5.com/2/21/30_07_4__1.png)

| P (P2, Bull) | Popt (Lion) | Pm (Leopard) | Pvmp (P1, Bear) |
| --- | --- | --- | --- |
| 1,557067 | 1,5546163 | 1,5546183 | 1,55217 |

### **Conclusion**

The proposed market theory is based on the pattern analysis for gaining profit, depending on the sales price of goods. It was discovered that by analyzing the market condition, we can estimate the maximum virtual return that indirectly reflects the market demand in these goods, and the elasticity of return towards the sales price which allows to evaluate the maximum level of goods' sales price on the market, named the "market level".

As established, a virtual optimal level of goods' sales price, presented as a geometric value of two break-even levels that allows to obtain the maximum profit when setting the sales price at this estimated level, is formed on the market. Furthermore, it was revealed that the income formation is also influenced by the virtual level of the average market price, which is formed on the market and represents the average value of two break-even levels. This level was called the "leopard level".

Based on the market condition analysis from the perspective of necessity to organize the trading process around the specific price levels on the market, the following was revealed:

1. The best results on the real market of goods and services are achieved through organizing trading around the optimal level of the goods' sales price, named the "lion level", which guarantees the maximum return. Commercial organizations may use this article as a guideline for optimizing the trading process.

2. Apart from the real level of actual price, 3 more virtual levels — market, lion and leopard levels — are formed on the Forex market. Price and market levels oppose each other and can take turns to become bullish and bearish.

3. When trading on the Forex market is organized around the first lower break-even level, the price turns bearish and aims to take this level — the market downtrend occurs, and the market level becomes bullish, and vice versa, when it is convenient to organize trading around the second upper break-even level, the price becomes bullish in the attempt to stay at this level, thus the uptrend is formed, and the market level turns bearish. These metamorphoses are organized and managed by the virtual lion and leopard levels.

4. Trading on the Forex market can be organized around the third global break-even level — the lion level, only in exceptional cases. This situation is accompanied with a severe flat market.

5. Normally, there are two ways the trend changes:

a) at any moment and any price level by a sharp intervention of the managing levels of lion and leopard in the process of trading with the consequent transfer of the price management to the opposite side and/or the consequent struggle of the opposing sides: current price formed by sellers and buyers against the unpredictable market;

b) voluntary transfer of the price management from bulls to bears on the lion level.


### **References**

1. [Market and types of markets (in Russian)](https://www.mql5.com/go?link=http://pidruchniki.com/18421120/politekonomiya/rynok_ego_vidy "/go?link=http://pidruchniki.com/18421120/politekonomiya/rynok_ego_vidy") [.](https://www.mql5.com/go?link=http://pidruchniki.com/18421120/politekonomiya/rynok_ego_vidy "http://pidruchniki.com/18421120/politekonomiya/rynok_ego_vidy")

2. [Universal Regression Model for Market Price Prediction](https://www.mql5.com/en/articles/250).


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1825](https://www.mql5.com/ru/articles/1825)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Universal regression model for market price prediction (Part 2): Natural, technological and social transient functions](https://www.mql5.com/en/articles/9868)
- [Universal Regression Model for Market Price Prediction](https://www.mql5.com/en/articles/250)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/70854)**
(29)


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
23 Sep 2025 at 15:44

Yusuf - good afternoon! What about the bidding! Any results?

Are you alive? In the market?

![Vitaly Muzichenko](https://c.mql5.com/avatar/2025/11/691d3a3a-b70b.png)

**[Vitaly Muzichenko](https://www.mql5.com/en/users/mvs)**
\|
23 Sep 2025 at 16:42

**Roman Shiredchenko [#](https://www.mql5.com/ru/forum/63050/page3#comment_58101603):**

Yusuf - good afternoon! What about the bidding! Any results?

Are you alive? In the market?

Take the course

![](https://c.mql5.com/3/475/6364333823033.png)

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
23 Sep 2025 at 19:15

**Vitaly Muzichenko [#](https://www.mql5.com/ru/forum/63050/page3#comment_58102036):**

Take the course

sometimes it seems like you're not the one writing....)

with such "stupid" entries.... )

download the theme!!!

![Vitaly Muzichenko](https://c.mql5.com/avatar/2025/11/691d3a3a-b70b.png)

**[Vitaly Muzichenko](https://www.mql5.com/en/users/mvs)**
\|
23 Sep 2025 at 20:35

**Roman Shiredchenko [#](https://www.mql5.com/ru/forum/63050/page3#comment_58103107):**

Sometimes it seems like you're not the one writing....)

with such "dumb" entries.... )

download the theme!!!

Bring up all the useless threads on the forum ...

You have nothing to do, a lot of extra time and lack of attention? Get married, and you will not be deprived of attention and excess time.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
24 Sep 2025 at 00:01

**Vitaly Muzichenko [#](https://www.mql5.com/ru/forum/63050/page3#comment_58103731):**

Bring up all the pointless threads on the forum ...

You have nothing to do, a lot of extra time and not enough attention? Get married, and you will not be deprived of attention and excess time.

it seemed that the topic is not finished!!! )

I laughed! )

thanks. )

PS already divorced (or divorced )), exhaled, paying alimony, thanks again - so far I've had enough!

![Graphical Interfaces I: Preparation of the Library Structure (Chapter 1)](https://c.mql5.com/2/21/Graphic-interface.png)[Graphical Interfaces I: Preparation of the Library Structure (Chapter 1)](https://www.mql5.com/en/articles/2125)

This article is the beginning of another series concerning development of graphical interfaces. Currently, there is not a single code library that would allow quick and easy creation of high quality graphical interfaces within MQL applications. By that, I mean the graphical interfaces that we are used to in familiar operating systems.

![MQL5 for beginners: Anti-vandal protection of graphic objects](https://c.mql5.com/2/20/ava.png)[MQL5 for beginners: Anti-vandal protection of graphic objects](https://www.mql5.com/en/articles/1979)

What should your program do, if graphic control panels have been removed or modified by someone else? In this article we will show you how not to have "ownerless" objects on the chart, and how not to lose control over them in cases of renaming or deleting programmatically created objects after the application is deleted.

![Graphical Interfaces I: Form for Controls (Chapter 2)](https://c.mql5.com/2/21/Graphic-interface__1.png)[Graphical Interfaces I: Form for Controls (Chapter 2)](https://www.mql5.com/en/articles/2126)

In this article we will create the first and main element of the graphical interface - a form for controls. Multiple controls can be attached to this form anywhere and in any combination.

![Using Assertions in MQL5 Programs](https://c.mql5.com/2/19/avatar_OoPs.png)[Using Assertions in MQL5 Programs](https://www.mql5.com/en/articles/1977)

This article covers the use of assertions in MQL5 language. It provides two examples of the assertion mechanism and some general guidance for implementing assertions.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dbozqvlacautqsoytpaujjnglsukckuw&ssn=1769251083726390043&ssn_dr=0&ssn_sr=0&fv_date=1769251083&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1825&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Market%20Theory%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925108336810990&fz_uniq=5083003278729417483&sv=2552)

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