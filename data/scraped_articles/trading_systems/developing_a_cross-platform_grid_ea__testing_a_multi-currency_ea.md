---
title: Developing a cross-platform grid EA: testing a multi-currency EA
url: https://www.mql5.com/en/articles/7777
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:48:17.749351
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/7777&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062731840546056142)

MetaTrader 5 / Trading systems


### Introduction

This article is a kind of postscript for a series of articles devoted to grid Expert Advisors:

- [Developing a cross-platform grid EA](https://www.mql5.com/en/articles/5596)

- [Developing a cross-platform grid EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)

- [Developing a cross-platform grid EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)

- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/ru/articles/7219)

We will not create or improve Expert Advisors in this article. Our multi-currency EA has already been created. Its new version will be attached below, along with testing reports and the SET file that was used for testing.

The main purpose of the article is to test the averaging and martingale-based EA in the market, for which it has not been initially intended. Can an Expert Advisor trading only long positions survive the S&P 500 drop from USD 3400 to USD 2200? Thus, the drop was more than 30 percent.

### Basic rules of the trading system

The trading system implemented in the Expert Advisor consists of the following rules.

**Market**. The EA trades only the US exchange stocks. The broker provides access to a few dozens of the most popular stocks. Stocks to test the multi-currency EA will be selected from the available set.

**Entry direction**. All positions are only opened in the Long direction.

The selection of this direction is obvious if you look at long-term price charts of most of the stocks. Especially, if you analyze a period from 2010 to early 2020. For example, take a look at the weekly chart for Microsoft stock:

![Microsoft stock, weekly chart](https://c.mql5.com/2/38/MSFT_W1__1.png)

Stocks were steadily rising before the market drop.

**Entry point**. The position will be opened based on the value of the RSI indicator. Readings will be interpreted in different ways depending on the asset. On some of the assets, entry will be performed when RSI falls below 30. On other assets we will enter when RSI rises above 70. Thus, it depends purely on the asset.

**The number of averaging steps**. It is extremely dangerous to use averaging with an unlimited number of steps. If the asset price moves in one direction for a long time, your losses will increase with every new positions explosively.

Therefore, the grid length for each symbol will be limited to four open positions (steps).

I.e. if a grid for a certain symbols consists of a maximum of 3 steps and the symbol price reaches a level where step 4 should be opened, the system will close all existing positions at this level instead of opening the fourth one. This provides the use of stop losses in the averaging algorithm.

**Averaging method**. An existing position can be averaged (by opening an additional position at a lower price) using a fixed lot or increasing the lot for each newly opened position.

If lot is increased using a martingale method, then loss in case of Stop Loss would be much higher than the profit received from Take Profit. That is why using martingale can be psychologically difficult. When you see that with a Stop Loss you lost the entire profit that the Expert Advisor had earned for the last two weeks, this can be frustrating. So better not to have Stop Losses at all.

Let us start with the selection of instruments which show good results in fixed-lot averaging.

If we need to use martingale, lot will increase not exponentially, but it will increase by the initial lot value at each new step.

The below table shows calculations for different averaging types.

| Averaging Type | Number of Chain Steps | Losses at Stop Loss |
| --- | --- | --- |
| Fixed | 2 | 3 |
| Fixed | 3 | 6 |
| Fixed | 4 | 10 |
| Increase by initial lot value | 2 | 4 |
| Increase by initial lot value | 3 | 10 |
| Increase by initial lot value | 4 | 20 |

Losses in case of Stop Loss at the appropriate step are specified in the number of steps. That is, if the step size is 7 points, then the losses with a fixed averaging at step 4 will equal to 10\*7 = 70 points.

Thus, if the lot size is increased by the initial lot size, the chain cannot be longer than three steps. This is because losses at step 4 would be too large.

**Grid size and Take Profit size**. Both the grid size (distance in points between two open positions) and the Take Profit size will be selected based on optimization. To limit losses in case of Stop Loss, the parameters will be set so that the loss can be equal to 1-3 profits from Take Profit.

**Trading and testing period**. As can be seen from most stock charts, price is steadily growing over a long period of time. But the key word here is "long period". This trading system is designed for a period of no less than one year. The optimal period is 4 years. If you decide to use this trading system, I recommend considering the four-year period.

Testing and optimization were also performed on the 4-year period, From January 2016 to January 2020. The found set of trading instruments will be additionally tested in the period up to April 2020, which is the period when stock markets dropped.

**Trading lot**. Testing and optimization will be performed with a fixed lot. Individual lot size will be used for each of the symbols.

The resulting multi-currency EA will be additionally tested with a gradually increasing trading lot. In this case, if the balance doubles the lot will also double. This increment will be repeated until the working lot reaches the value equal to 25 initial lots.

Lot is increased via input parameters. These parameters are used for specifying the deposit size, upon reaching which the working lot should be increased (see the Figure below).

![Input parameters for increasing the working lot](https://c.mql5.com/2/38/LotType__1.png)

**Conclusion**. These are all the rules according to which our trading system operates. Let us test the system profitability on single instruments.

### Optimization parameters on individual instruments

I will not provide the details of each individual optimization because the idea of the article is different.

Optimizations were performed in mode _Every tick based on real ticks_.

Optimization results were sorted by the Recovery Factor. If a test pass showed profit, the recovery factor value is returned. If a pass is losing, a percent by which the account balance has decreased (with a negative sign) is shown instead of the Recovery Factor. If the EA executed less than 30 trades during testing, 0 is returned because it is a very small number of trades which is insufficient for obtaining objective statistics.

This sorting method is not available among standard options. It is implemented in the Expert Advisor. To use it, open the _Settings_ tab in the Strategy Tester and selected _Custom max_ (see the Figure below).

![The highest value of the custom criterion](https://c.mql5.com/2/38/optimization.png)

Also, forward tests were used during single symbol optimization and testing. The optimal forward period for our 4-year interval is _1/4_. It means that the first three years are used for backtests, and the last year is used for a forward test.

### Symbols selected for the multi-currency Expert Advisor

Now, let us consider testing results. Here are the balance charts of the instruments which have been eventually included in our multi-currency Expert Advisor. There will be 11 such instruments. Firstly, the EA is not designed for trading more instruments. Secondly, it is very difficult to pick up symbols positions for which would be uncorrelated.

AAPL:

![AAPL, 2016-2020](https://c.mql5.com/2/38/AAPL__1.png)

BRK.B:

![BRK.B, 2016-2020](https://c.mql5.com/2/38/BRKB__1.png)

PEP:

![PEP, 2016-2020](https://c.mql5.com/2/38/PEP__1.png)

WMT:

![WMT, 2016-2020](https://c.mql5.com/2/38/WMT__1.png)

CVX:

![CVX, 2016-2020](https://c.mql5.com/2/38/CVX__1.png)

EBAY:

![EBAY, 2016-2020](https://c.mql5.com/2/38/EBAY__1.png)

MSFT:

![MSFT, 2016-2020](https://c.mql5.com/2/38/MSFT__1.png)

DIS:

![DIS, 2016-2020](https://c.mql5.com/2/38/DIS__1.png)

JPM:

![JPM, 2016-2020](https://c.mql5.com/2/38/JPM__1.png)

JNJ:

![JNJ, 2016-2020](https://c.mql5.com/2/38/JNJ__1.png)

S&P500:

![S&P 500, 2016-2020](https://c.mql5.com/2/38/SP500__1.png)

A big drop closer to chart end is not actually a drop. This place shows the end of backtesting and the beginning of a forward one. This is because a forward test start with the initial deposit amount, and not with the backtest balance.

Here is a table with the testing results of the selected trading instruments:

| Symbol | Recovery Factor (back/forward) | Profit Factor (back/forward) | Max drawdown (back/forward) | Trades (back/forward) |
| --- | --- | --- | --- | --- |
| AAPL | 7.25 / 11.04 | 3.93 / 37.99 | 49.41 / 30.36 | 134 / 58 |
| BRK.B | 7.41 / 1.79 | 3.11 / 2.01 | 15.06 / 14.96 | 70 / 29 |
| PEP | 5.2 / 3.26 | 2.49 / 5.42 | 13.96 / 10.42 | 55 / 15 |
| WMT | 5.9 / 3.19 | 2.51 / 2.56 | 25.52 / 20.7 | 67 / 25 |
| CVX | 6.51 / 3.25 | 3.03 / 4.26 | 19.17 / 14.82 | 78 / 24 |
| EBAY | 4.57 / 1.95 | 8.87 / 8.85 | 20.7 / 12.96 | 43 / 12 |
| MSFT | 7.41 / 3.13 | 6.69 / 5.26 | 16 / 20.93 | 72 / 44 |
| DIS | 3.97 / 1.19 | 2.32 / 1.84 | 26.97 / 32.02 | 101 / 49 |
| JPM | 4.34 / 3.07 | 1.75 / 2.81 | 12.69 / 10.86 | 164 / 54 |
| JNJ | 6.24 / 1.23 | 5.66 / 2.31 | 28.94 / 44.36 | 68 / 29 |
| S&P500 | 2.55 / 1.98 | 1.65 / 1.57 | 17.81 / 21.18 | 85 / 91 |

All Strategy Tester reports are attached below, at the end of this article.

It is believed that forward testing allows obtaining more accurate results and also allows avoiding over-optimization. I do not agree with this point of view. Nevertheless, I also used this type of testing when optimizing.

In my opinion, there is one advantage of a forward test. It allows testing an EA using two different starting points. This allows you to save time.

As for the rest, interpreting of results can be more complicated with forward testing.

For example, you cannot receive the total recovery factor over the entire testing period by simply adding results from back and forward tests. This is because maximum drawdown is usually different in these periods. But we need this common maximum drawdown in order to determine the initial deal volume or the minimum deposit. Thus, we have to use the highest darwdown value as the maximum value for the entire period. In this case, the recovery factor calculated in the period with a lower drawdown will not be accurate.

Moreover, if you can view the balance chart, you will not over-optimize the Expert Advisor event without a forward test. If the balance chart is steadily growing throughout the entire testing period, it means that you have selected suitable parameters. If the main growth was registered at the beginning of the balance graph, while the funds level stands still or falls in the right part of the graph, the parameters are unsuitable.

### Multi-currency EA testing before market drop

What will be the effect of combining the selected trading instruments into one Expert Advisor? Will the results be better than those when trading each symbol separately? Here is the balance graph and testing results when trading a fixed lot:

![Balance graph when trading 11 instruments at once (fixed lot)](https://c.mql5.com/2/38/multi_fixed__1.png)

Even with the naked eye, you see how much the balance graph has smoothed out compared to single instrument trading. Let us additionally check the table with testing results.

| Recovery Factor | Profit factor | Net profit | Max drawdown | Trades |
| --- | --- | --- | --- | --- |
| 20.56 | 2.94 | 1 992 | 96.91 | 1 308 |
| --- | --- | --- | --- | --- |

The effect of the diversification is also demonstrated in this table.

The initial deposit was USD 200. After four years, we have USD 1,992. It means that the profit for four years is almost 900 percent when trading the fixed lot.

This seems to be a very good result. However, things are a little worse in reality. first, the initial deposit of USD 200 is too small.

The maximum drawdown was USD 97. But this does not mean that a deposit of USD 97 would be enough to trade this system. You also need to take into account the margin size reserved to maintain open positions.

After testing, the EA adds information about the maximum drawdown registered during testing, in the _Journal_ tab. In our case, the maximum drawdown was USD 190. It means that comfortable trading requires having at least USD 97 + USD 190 = USD 280. Let's round up to USD 300. This means that the profit for 4 years is almost 600 percent.

This profit was achieved form fixed-lot trading. What if we increase lot each time the balance grows by the initial deposit value? The Expert Advisor allows increasing the initial lot by a maximum of 25 times. I think this limitation is pretty enough for our tests:

![Balance graph when trading 11 instruments at once (increasing lot)](https://c.mql5.com/2/38/multi_lot25__1.png)

This time the graph is not so smooth. But the final profit is impressive! Deposit increased from USD 200 (or the required USD 300) to USD 20,701. This equals to almost 7,000 % for 4 years!

In this case, the EA started using the maximum lot with a deposit of USD 7,100. It means that the working lot could further be increased if we did not use the limitation.

### Multi-currency EA testing including the market drop period

The results were very good. But how would the EA behave during the recent market drop?

This part concerns the main topic of our article. The next testing period is from 2016.01.01 to 2020.04.01.

Multi-currency EA testing results with the same settings, trading the fixed lot:

![Balance graph when trading 11 instruments at once (fixed lot)](https://c.mql5.com/2/38/multi_fixed_fall__1.png)

At first glance, the drop is almost imperceptible. Let us now take a look at testing results:

| Recovery Factor | Profit factor | Net profit | Max drawdown | Trades |
| --- | --- | --- | --- | --- |
| 13.91 | 2.54 | 1 971 | 141.69 | 1 400 |
| --- | --- | --- | --- | --- |

Now you see the difference. The recovery factor decreased by 7. The maximum drawdown increased by 1.5 times. Profit remained practically unchanged. We actually lost what we had earned from January till the end of February. This is more than a 30% drop.

Let us now take a look at testing results using the lot size that gradually increases up to 25 time:

![Balance graph when trading 11 instruments at once (increasing lot)](https://c.mql5.com/2/38/multi_lot25_fall__1.png)

Even in this case the fall is not so significant. The profit is USD 20,180, which is only USD 600 less than it was at the beginning of 2020.

### Let's sum up.

In general, I hope I have proven that even trading averaging and martingale utilizing systems are not that dangerous. Of course, you should set responsible limits for averaging. It means that Stop Loss should also be used with averaging and martingale.

### Conclusion

More than 10 years have passed since the previous large market crash in 2008. Markets have been steadily growing since them. Who knows, probably a similar 10-year growth may start after the 2020 crash. This would be the perfect time for trading systems like the one described in this article.

As for me, I will definitely launch a robot that traded based on the described system on a real account. The Expert Advisor is attached below, so you may also use it.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7777](https://www.mql5.com/ru/articles/7777)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/7777.zip "Download all attachments in the single ZIP archive")

[article\_files.zip](https://www.mql5.com/en/articles/download/7777/article_files.zip "Download article_files.zip")(4164.36 KB)

[griderKatMultiEA.ex4](https://www.mql5.com/en/articles/download/7777/griderkatmultiea.ex4 "Download griderKatMultiEA.ex4")(482.52 KB)

[griderKatMultiEA.ex5](https://www.mql5.com/en/articles/download/7777/griderkatmultiea.ex5 "Download griderKatMultiEA.ex5")(615.21 KB)

[griderKatMultiEA.mq4](https://www.mql5.com/en/articles/download/7777/griderkatmultiea.mq4 "Download griderKatMultiEA.mq4")(410.15 KB)

[griderKatMultiEA.mq5](https://www.mql5.com/en/articles/download/7777/griderkatmultiea.mq5 "Download griderKatMultiEA.mq5")(410.15 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a cross-platform grid EA (Last part): Diversification as a way to increase profitability](https://www.mql5.com/en/articles/7219)
- [Developing a cross-platform grider EA (part III): Correction-based grid with martingale](https://www.mql5.com/en/articles/7013)
- [Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)
- [Developing a cross-platform grider EA (part II): Range-based grid in trend direction](https://www.mql5.com/en/articles/6954)
- [Selection and navigation utility in MQL5 and MQL4: Adding data to charts](https://www.mql5.com/en/articles/5614)
- [Developing a cross-platform grider EA](https://www.mql5.com/en/articles/5596)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/347824)**
(9)


![פינחס פוגל](https://c.mql5.com/avatar/2020/6/5EDD752D-B6D9.jpg)

**[פינחס פוגל](https://www.mql5.com/en/users/a089557176)**
\|
6 Dec 2020 at 11:55

I wonder if it is possible to create such an algorithm for Forex, is there a significant difference between Forex and stocks in this regard?


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
7 Jan 2022 at 17:13

Yes, everything is running - but trades are not opening - history is loaded - set file is loaded into the tester - does not open trades.

[![](https://c.mql5.com/3/377/5918654201503__1.png)](https://c.mql5.com/3/377/5918654201503.png "https://c.mql5.com/3/377/5918654201503.png")

the second set with 25 symbols. The first tester also does not open trades....

Please help with settings...

[![](https://c.mql5.com/3/377/736638674659__1.png)](https://c.mql5.com/3/377/736638674659.png "https://c.mql5.com/3/377/736638674659.png")

does not make trades in the tester:

[![](https://c.mql5.com/3/377/6221170261615__1.png)](https://c.mql5.com/3/377/6221170261615.png "https://c.mql5.com/3/377/6221170261615.png")

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
7 Jan 2022 at 17:48

**noteca [#](https://www.mql5.com/ru/forum/336841#comment_16046684):**

Good afternoon, everyone.

Has anyone figured out the work of the Expert Advisor ?

I tried it with the standard set file, the account stood for a couple of days, nothing happened.

I looked at the differences in the settings in the tester report and in the settings of the set file - the difference in marketwatch false-true and **allSymb\_max\_prices=200.**

I don't really understand what they are responsible for.

I'm looking into it now, he has tickers there with the prefix \*.m - this is apparently a mini on roboforex - I'm making corrections now.

[![](https://c.mql5.com/3/377/5113283995132__1.png)](https://c.mql5.com/3/377/5113283995132.png "https://c.mql5.com/3/377/5113283995132.png")

Here I have a ticker - highlighted - with the prefix "m".

[![](https://c.mql5.com/3/377/1605793412224__1.png)](https://c.mql5.com/3/377/1605793412224.png "https://c.mql5.com/3/377/1605793412224.png")

[![](https://c.mql5.com/3/377/3149947209144__1.png)](https://c.mql5.com/3/377/3149947209144.png "https://c.mql5.com/3/377/3149947209144.png")

[![](https://c.mql5.com/3/377/4426327927098__1.png)](https://c.mql5.com/3/377/4426327927098.png "https://c.mql5.com/3/377/4426327927098.png")

[![](https://c.mql5.com/3/377/6369746712564__1.png)](https://c.mql5.com/3/377/6369746712564.png "https://c.mql5.com/3/377/6369746712564.png")

Here.

I've uploaded the history.

The curveballs have gone...

[![](https://c.mql5.com/3/377/1641989689004__1.png)](https://c.mql5.com/3/377/1641989689004.png "https://c.mql5.com/3/377/1641989689004.png")

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
7 Jan 2022 at 17:56

****SERGEI TABALENKOV [#](https://www.mql5.com/ru/forum/336841#comment_15846354):****

**Сложно проверить Ваши выводы. Советник не запускается ни в какую. И ошибок не выдает.**

**пока в тестере запустился исправно. Сеты выложу. И робота тоже.**

**noteca [#](https://www.mql5.com/ru/forum/336841#comment_16046684)**
**:**

Good afternoon, everyone.

Has anyone figured out the work of the Expert Advisor ?

I tried it with the standard set file, the account stood for a couple of days, nothing happened.

I looked at the differences in the settings in the tester report and in the settings of the set file - the difference in marketwatch false-true and **allSymb\_max\_prices=200.**

I don't really understand what they are responsible for.

I figured it out - everything is working so far - these parameters don't seem to affect....

**SERGEI TABALENKOV [#](https://www.mql5.com/ru/forum/336841#comment_15846354):**

It is difficult to check your conclusions. The Expert Advisor does not start in any way. And it does not generate any errors.

exclude \*.m in the ticker name - write it as you have in your broker.

**Valery Pavlushev [#](https://www.mql5.com/ru/forum/336841#comment_15904602):**

So much work has been done, but I cannot see it.

I downloaded your set file from the archive. I get an error in the left corner of the window: Timer NOT SET

Can you upload a working set file for GbpUsd?

exclude \*.m in the ticker name - write it as in your broker.

In general, while testing, if there will be more "bugs" - in general, about the names of symbols here, of course, it is necessary to specify the author of the article, IMHO, if for people ... :-)

[![](https://c.mql5.com/3/377/5371420909358__1.png)](https://c.mql5.com/3/377/5371420909358.png "https://c.mql5.com/3/377/5371420909358.png")

the test is going on - the flight is normal, if there will be any more "bugs" - I will write here:

[![](https://c.mql5.com/3/377/4345661399725__1.png)](https://c.mql5.com/3/377/4345661399725.png "https://c.mql5.com/3/377/4345661399725.png")

in general, everything is set up normally in the tester, it is being tested. Robot (without MT5 edits) and set will be posted after the test, the second set with MM.

**Note: When starting a test or putting on trades, check the name of the symbols being traded - those symbols should be entered in the set.**

[![](https://c.mql5.com/3/377/4485770099746__1.png)](https://c.mql5.com/3/377/4485770099746.png "https://c.mql5.com/3/377/4485770099746.png")

For a long time during the last year "went" to review these similar cycles of articles, there is time to understand... :-)

here is the test from the article:

[![](https://c.mql5.com/3/377/914156139354__1.png)](https://c.mql5.com/3/377/914156139354.png "https://c.mql5.com/3/377/914156139354.png")

I will post after the test on the same parameters - on the same values. There is something to work on in terms of optimisation...

here is basically the forward from year 20:

in general, here are the intermediate layouts (forward essentially from 2020) - the chatter is about 10 000,00 - it will be necessary to reoptimise.

[![](https://c.mql5.com/3/377/5536587151716__1.png)](https://c.mql5.com/3/377/5536587151716.png "https://c.mql5.com/3/377/5536587151716.png")

here is the closing at exceeding the number of averages - more than 3, as the author of the article reported:

[![](https://c.mql5.com/3/377/2606892969114__1.png)](https://c.mql5.com/3/377/2606892969114.png "https://c.mql5.com/3/377/2606892969114.png")

in general, it is unambiguous to reoptimise - it is impossible to trade on such values:

(the idea of limiting the number of averages is a good one).

[![](https://c.mql5.com/3/377/1313566858710__1.png)](https://c.mql5.com/3/377/1313566858710.png "https://c.mql5.com/3/377/1313566858710.png")

in general downward slope - it is impossible to trade on such values now:

[![](https://c.mql5.com/3/377/4029453228187__1.png)](https://c.mql5.com/3/377/4029453228187.png "https://c.mql5.com/3/377/4029453228187.png")

[![](https://c.mql5.com/3/377/2239218481450__1.png)](https://c.mql5.com/3/377/2239218481450.png "https://c.mql5.com/3/377/2239218481450.png")

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
8 Jan 2022 at 09:15

where without increasing the lotness of the set - the first one - it is also perfectly clear that it is impossible to trade at such values - you need overoptimisation:

[![](https://c.mql5.com/3/377/103522560837__1.png)](https://c.mql5.com/3/377/103522560837.png "https://c.mql5.com/3/377/103522560837.png")

![Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://c.mql5.com/2/38/MQL5-avatar-doeasy-library__5.png)[Timeseries in DoEasy library (part 40): Library-based indicators - updating data in real time](https://www.mql5.com/en/articles/7771)

The article considers the development of a simple multi-period indicator based on the DoEasy library. Let's improve the timeseries classes to receive data from any timeframes to display it on the current chart period.

![Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__3.png)[Continuous Walk-Forward Optimization (Part 6): Auto optimizer's logical part and structure](https://www.mql5.com/en/articles/7718)

We have previously considered the creation of automatic walk-forward optimization. This time, we will proceed to the internal structure of the auto optimizer tool. The article will be useful for all those who wish to further work with the created project and to modify it, as well as for those who wish to understand the program logic. The current article contains UML diagrams which present the internal structure of the project and the relationships between objects. It also describes the process of optimization start, but it does not contain the description of the optimizer implementation process.

![Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://c.mql5.com/2/38/MQL5-avatar-continuous_optimization__4.png)[Continuous Walk-Forward Optimization (Part 7): Binding Auto Optimizer's logical part with graphics and controlling graphics from the program](https://www.mql5.com/en/articles/7747)

This article describes the connection of the graphical part of the auto optimizer program with its logical part. It considers the optimization launch process, from a button click to task redirection to the optimization manager.

![MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://c.mql5.com/2/38/MQL5-avatar-dialog_form.png)[MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)

This paper continues checking the new conception to describe the window interface of MQL programs, using the structures of MQL. Automatically creating GUI based on the MQL markup provides additional functionality for caching and dynamically generating the elements and controlling the styles and new schemes for processing the events. Attached is an enhanced version of the standard library of controls.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/7777&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062731840546056142)

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