---
title: Reversing: Reducing maximum drawdown and testing other markets
url: https://www.mql5.com/en/articles/5111
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:43:25.600094
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/5111&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070509691311691563)

MetaTrader 5 / Trading systems


### Introduction

[In the previous article](https://www.mql5.com/en/articles/5008), we analyzed the reversing strategy. We tested the strategy on two Forex instruments. We also tried to use different indicators to improve the system performance.

As a result, we found out that the reversing strategy can work with an average annual yield of about 50%. But this is a high-risk strategy, because maximum drawdowns could exceed the initial deposit amount. With the initial deposit of 10,000 dollars, the maximum drawdowns on the analyzed financial instruments reached 12,000-15,000 dollars with any indicators. Can this variable be improved? How will the taken measures affect the strategy profitability? This will be the subject of the first part of this article.

After dealing with this issue, we will move on to the second subject — we will try to trade various financial instruments in addition to Forex symbols. We will try to find out, which market is the most optimal for this trading strategy. Are there any considerable differences in reversing trading in different markets.

In all tests within the article, we will use the M15 timeframe and the maximum number of steps in a chain set to 8. The reasons for choosing these parameter values were described [in the previous article](https://www.mql5.com/en/articles/5008).

In addition, we will not use any indicators in all tests, except GBPUSD and XAGUSD. The strategy performs an entry in the fixed direction, once the previous trade chain is closed. For GBPUSD and XAGUSD, entry depends on the CCI indicator values. Testing has shown that CCI can increase profitability for the above symbols.

An archive attached below contains all SET files with proper Expert Advisor settings for each symbol considered in this article. These settings were used for tests, the resulting profit charts of which are presented in the article.

### Changes in tests

Stricter testing and optimization will be performed in this article.

**First**, all tests will be performed in the _Every tick based on real ticks_ mode.

**Second**, optimization will be performed not just for the maximum balance, but for the maximum balance plus the minimum drawdown.

This change is quite understandable. We set the Take Profit level twice as large as Stop Loss. This causes a strong imbalance in profit, depending on which reversing step resulted in Take Profit triggering. For example, Take Profit at the very first step can bring $1, while at the maximum step the net profit will be equal to $10 (i.e. the profit after covering losses of all unprofitable steps in the chain).

For this reason, using profit for optimization criterion is not always appropriate for finding the best parameters. Often, the opposite happens: with the found parameters, Take Profit can be rarely achieved at the first step.

**Third**, testing will be performed on multiple broker accounts.

Each broker provides specific spreads, swaps, slippages and units. Therefore, the results can vary greatly depending on your broker. Let's check it out. We will test three different brokers.

### Changes in the Expert Advisor

A new ReverseEA version is attached below. Differences from the previous published version:

- fixed Expert Advisor crash, which could occasionally occur when sending an order;
- in addition to doubling exponentially, it is now possible to double trade volume after a step or two;
- the number of the reversing step in a chain is now written in the order comment — this is used for enabling volume increased in one or two trades;
- a new checkbox is added in settings " _Do not open the first deal (only manage)"_;
- added settings restricting entries by time: to avoid entries in certain hour, day of the week or month;
- a new Expert Advisor operation mode has been added: the EA will close all open positions and orders and will open a new order in the direction specified by you and add an appropriate comment, after that the EA will complete operation;
- added RSI operation options;
- added new options for other indicators: CCI and Momentum.

Here are more details about some of the new features.

**New operation mode**. This new EA operation mode is used when you want to close the current position and immediately open a new one with the reduced volume, in any direction, which may differ from the EA's default direction.

For example, if it is the sixth reversing or more, the price goes in your direction, the entire reversing chain is in profit and you are afraid that the price will turn against you. In this case you can take the profit using this mode and immediately start a new chain in a selected direction, with the initial volume.

To use this mode, set the number of the reversing step to start the mode in the parameter _"Open Long trade with this comment and exit_" or " _Open Short trade with this comment and exit_".

**Only deal management**. In comments to the previous article it was mentioned that trading using technical analysis would be much better in terms of profitability than the analyzed trading method with standard indicators or without any of them. This is absolutely true. Entry from a specific level in the direction you select would be far less risky than entering in the fixed direction at any time, regardless of where the price is now.

But technical analysis implies manual work, and it is quite difficult to program it properly. However, if you prefer to trade manually using technical analysis, then this new EA feature will be useful for you.

New checkbox _"Do not open the first deal (only manage)_" allows you to manually open the first deal and let the EA manage it. If you enable the checkbox, the EA will manage only open deals.

The first deal can be opened using the new mode ( _"Open Long trade with this comment and exit_" or " _Open Short trade with this comment and exit_").

Alternatively, you may use my tool [Creating orders with a fixed stop in dollars](https://www.mql5.com/en/market/product/29801). The tool allows opening positions with a fixed risk in dollars or deposit percent by specifying a required comment (1 — to open the first deal in the chain) and Magic number (must match the RevertEA Magic).

You may use your own Expert Advisors, which allow specifying Magic and a comment when opening a deal.

Please note that if you select the deal management mode, the method for determining the take profit target for pending orders within the chain also changes. In a normal mode, the Take Profit price for new orders in the chain is determined based on the EA's _Take Profit_ setting. In the management mode, the Take Profit value is determined as a difference between current position's take profit and open price, to which the current spread value is added.

Therefore, the number of profit points in case of Take Profit can vary. The value may vary by the difference between the current spread and the spread as of the position opening moment. If the spread during first position opening or during previous order placing is equal to the spread as of the current order placing time, then there is no difference between these two calculation methods. If the current spread is greater than the prior value, then Take Profit will be larger by the difference between the current and previous spread values. Otherwise, the Take Profit size will be less by this difference.

**RSI operation options**. Now, the RSI filter can operate in three modes:

- if the current RSI value is less than or equal to _rsiValMax_, enter a Long position, otherwise skip; if current RSI is greater than or equal to _rsiValMin_, perform a Short entry, otherwise skip;
- if the current RSI value is greater than _rsiValMax_, enter a Long position, otherwise skip; if current RSI is less than _rsiValMin_, perform a Short entry, otherwise skip;
- if the current RSI value is greater than _rsiValMax_, enter a Short position, otherwise skip; if current RSI is less than _rsiValMin_, perform a Long entry, otherwise skip;

Please note that in the first mode, if both Long and Short deals are allowed, two deals at a time can be then for some RSI values.

### Reducing risks when trading Forex symbols

How can the maximum drawdown be reduced?

We already trade starting with the minimum lot. So, initial lot cannot be reduced.

In addition to the minimum lot, we increase the lot value in further reversal steps. The maximum drawdown can be reduced almost twice if we do not increase the lot. But this is not enough. This allows reducing the maximum drawdown to $6000 - 8000. While the acceptable drawdown value is about $3000 - 5000, which is 30-50% of the initial deposit. How can this be done?

The answer to this question was found in the previous article. Our Take Profit is almost always at least twice as large as the stop loss value. This means that we do not necessarily need to double the lot at each step, in order to be profitable. Alternatively, we can increase the lot every second or third trade.

This will have the following consequences:

- lot size at the largest reversing step will be significantly reduced, so risks will also be reduced;
- the number of reversing steps can be increased, if necessary;
- profit can become more even, while also the initial lot size can be increased, which may increase the profit in the first steps;
- the balance chart will become smoother.

Now, let's move on to practicing.

In the previous article, we analyzed only two financial instruments: EURUSD and GBPUSD. There are also other symbols, which produce acceptable results: USDJPY, AUDJPY, GBPJPY, XAUUSD and XAGUSD. Let's add them to our basket.

**EURUSD**. For comparison, we will first analyze the profit chart for lot doubling at each reversing step. The chart is slightly worse than that presented in the previous article, because the testing mode has changed: now we use _Every tick based on real ticks_:

![EURUSD, broker №2, lot doubling at each step](https://c.mql5.com/2/34/eurusd_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 3 392 | 1.45 | 13 865 (16%) | 137 466 | 22 | 45 | 175 |

The same symbol with the lot doubling applied at every second step:

![EURUSD, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/eurusd_even2_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 3 392 | 1.26 | 3 574 (14%) | 24 569 | 22 | 45 | 175 | 0.02 |

Both the profit and profitability decreased significantly as well as the maximum drawdown. That's what we wanted to achieve.

**GBPUSD**. Profit chart for lot doubling at each reversing step:

![GBPUSD, broker #2, lot doubling at each step](https://c.mql5.com/2/34/gbpusd_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GBPUSD | 4 218 | 1.52 | 13 632 (12%) | 153 384 | 21 | 40 | 135 |

Lot doubling applied at every second step:

![EURUSD, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/gbpusd_even2_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBPUSD | 4 161 | 1.24 | 3 620 (20%) | 36 540 | 22 | 40 | 135 | 0.04 |

**GBPJPY** lot doubling at every second step:

![GBPJPY, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/gbpjpy_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBPJPY | 1 559 | 1.25 | 4 892 (22%) | 17 534 | 14 | 160 | 380 | 0.01 |

**USDJPY** lot doubling at every second step:

![USDJPY, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/usdjpy_even2_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| USDJPY | 1 787 | 1.22 | 3 066 (13%) | 13 559 | 25 | 60 | 230 | 0.02 |

**AUDJPY** lot doubling at each step:

![AUDJPY, broker #2, lot doubling at each step](https://c.mql5.com/2/34/audjpy_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDJPY | 2 661 | 1.21 | 5 275 (18%) | 15 902 | 15 | 80 | 130 | 0.01 |

In this case, Take Profit is less than twice the Stop Loss, so we cannot use doubling at every second step. But we don’t need it here, because the maximum drawdown with the minimum lot does not exceed $6000.

**XAUUSD** lot doubling at every second step:

![XAUUSD, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/xauusd_even2_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XAUUSD | 841 | 1.4 | 3 263 (13%) | 21 250 | 8 | 1 700 | 3 600 | 0.03 |

Note that in this case the maximum consecutive loss series does not exceed 8, which means that all our chains for 15 years, sooner or later ended in profit. However, the chart shows that profit was not always enough to cover swap losses.

**XAGUSD** lot doubling at every second step:

![XAGUSD, broker #2, lot doubling at every second step](https://c.mql5.com/2/34/xagusd_even2_broker2.png)

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XAGUSD | 401 | 1.56 | 5 087 (27%) | 19 718 | 22 | 4 500 | 17 500 | 0.01 |

As you can see, silver is much more volatile than gold, and it often hits Stop Losses. Even in this case, the probability of making a profit is enough for having the deposit grow.

Let's summarize and draw up a general table for all the tested symbols.

**Lot doubling at every second step:**

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP | Lot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 3 392 | 1.26 | 3 574 (14%) | 24 569 | 22 | 45 | 175 | 0.02 |
| GBPUSD | 4 161 | 1.24 | 3 620 (20%) | 36 540 | 22 | 40 | 135 | 0.04 |
| GBPJPY | 1 559 | 1.25 | 4 892 (22%) | 17 534 | 14 | 160 | 380 | 0.01 |
| USDJPY | 1 787 | 1.22 | 3 066 (13%) | 13 559 | 25 | 60 | 230 | 0.02 |
| \\* AUDJPY | 2 661 | 1.21 | 5 275 (18%) | 15 902 | 15 | 80 | 130 | 0.01 |
| XAUUSD | 841 | 1.4 | 3 263 (13%) | 21 250 | 8 | 1 700 | 3 600 | 0.03 |
| XAGUSD | 401 | 1.56 | 5 087 (27%) | 19 718 | 22 | 4 500 | 17 500 | 0.01 |

\\* lot doubling at each step

We managed to reduce the maximum drawdown to the acceptable value. But at the same time, we lost a large share of profits. Moreover, the charts of EURUSD and GBPUSD look smoother with lot doubled at each step.

The attached archive contains testing results of all the above symbols both when doubling at each step and when doubling at every second step. So, you can take a closer look at the results of the trading strategy.

### Testing with a different broker

In comments to the previous article, it was mentioned that testing results could greatly depend on the broker. If testing with one broker shows good results, much worse results can be obtained with another broker. Let's try to check it and perform tests with a different broker.

Here are general testing results. The main conclusion is as follows: the broker really influences testing results. The influence is so strong, that found parameters may not be suitable for a different broker.

However, by performing another optimization for the new broker, it is possible to find parameters, with which trading would be profitable enough. Here are the parameters:

**New broker:**

| Symbol | Trades | Profit factor | Max. drawdown | Profit Column | Max. consecutive losses | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 1 224 | 1.77 | 2 635 (17%) | 10 116 | 9 | 115 | 205 |
| GBPUSD | 3 382 | 1.25 | 1 101 (7%) | 6 063 | 11 | 90 | 145 |
| GBPJPY | 2 179 | 1.32 | 900 (6%) | 3 437 | 9 | 200 | 215 |
| USDJPY | 1 507 | 1.21 | 12 358 (45%) | 10 222 | 12 | 75 | 220 |
| AUDJPY | 1 175 | 1.48 | 4 608 (29%) | 10 731 | 9 | 115 | 230 |
| GOLD | 1006 | 1.75 | 15 615 (26%) | 78 962 | 8 | 1 400 | 2 500 |
| SILVER | 275 | 2.51 | 4 588 (14%) | 28 077 | 8 | 85 | 215 |

Now, here are the charts.

AUDJPY:

![AUDJPY, broker #3, lot doubling at each step](https://c.mql5.com/2/34/audjpy_broker3.png)

EURUSD:

![EURUSD, broker #3, lot doubling at each step](https://c.mql5.com/2/34/eurusd_broker3.png)

GBPJPY:

![GBPJPY, broker #3, lot doubling at each step](https://c.mql5.com/2/34/gbpjpy_broker3.png)

GBPUSD:

![GBPUSD, broker #3, lot doubling at each step](https://c.mql5.com/2/34/gbpusd_broker3.png)

GOLD:

![GOLD, broker #3, lot doubling at each step](https://c.mql5.com/2/34/gold_broker3.png)

Silver:

![Silver, broker #3, lot doubling at each step](https://c.mql5.com/2/34/silver_broker3.png)

Lot doubling at each step was used in all tests. The profit is much lower that that with the first broker. But look at the maximum drawdown and the maximum consecutive loss sequence. In most cases, the values are much worse in the second testing series. In some cases, it is possible to increase the initial lot by 2 or more times with an acceptable increase in the maximum drawdown.

Why does it happen that different sets of parameters may be suitable for different brokers? One of the brokers may have more accurate historic data. But if the second broker offers more accurate data, then the parameters found for that broker would be suitable for the first one. I checked this: parameters found with the second broker produced losing results with the first broker (or much worse results than with native parameters).

So, testing results can really depend on the broker. The question is whether you trust your broker. If you do trust, then you may test EAs using your broker's data, hoping correct data are provided to you. If not, then why are you trading with your broker?

In any case, if you want to use this trading strategy, you should optimize the Expert Advisor parameters and find the most optimal values, instead of using those provided by myself.

### Other Forex symbols

Reversing is not a holy grail which can work well everywhere. To prove this, here are the "best" testing results for other Forex symbols.

AUDUSD:

![AUDUSD, broker #2, lot doubling at each step](https://c.mql5.com/2/34/audusd_broker2.png)

NZDUSD:

![NZDUSD, broker #2, lot doubling at each step](https://c.mql5.com/2/34/nzdusd_broker2.png)

USDCAD:

![USDCAD, broker #2, lot doubling at each step](https://c.mql5.com/2/34/usdcad_broker2.png)

USDCHF:

![USDCHF, broker #2, lot doubling at each step](https://c.mql5.com/2/34/usdchf_broker2.png)

### Stock market

Forex is not the only market in the world. Might this be not the best market for applying reversing techniques? At least the large number of forex symbols, with which reversing does not show good results, can serve as proof for that.

Forex is considered to be a ranging market. Stock market is trending. What can be better for reversing than long trends? Let's check this assumption. Since we are optimistic and we assume that the stock price will rise, the first deal in the chain will always be Long.

We will test the strategy simultaneously with 3 brokers. They provide different trading conditions. Let's see how different conditions affect results.

The first table contains general results for all symbols provided by a broker. After that table, we will make some conclusions. Then we will briefly view balance charts.

All below tables (for all markets) consist of the same columns. Generally, their contents are clear from the column names. Let's additionally consider the contents of some of the columns:

- _Cur. price_ — the security price at the time of article writing. It is provided to enable the comparative analysis of all symbols (the higher the price, the higher the maintenance margin for the symbol, so the lower number of instruments and lots can be traded simultaneously);
- _Annual %_ \- the value is calculated according to the following formula: _((profit/max. drawdown)\*100)/number of years for which the historic symbol data are available_, it is an approximate number and is also provided for the purpose of comparative analysis;
- _Beginning_ — the first data of the period, for which the symbol's history data are available (i.e. the testing period starts with this date and ends with August-September 2018);
- _Max. losses_ — the maximum number of consecutive losses, i.e. how deep into the chain we had to go before reaching Take Profit;
- _Swap L\|S (L%)_ — the rollover commission in symbol currency, which you have to pay when a position is shifted to the next day (or which is payable to you if the swap is positive) for Long and Short positions, as well as percent of the symbol price for Long positions.

The following parameters are highlighted in the tables:

- yellow color in the _TP_ column shows that the Take Profit values for this symbol is twice (or even more) as large as Stop Loss;
- red color in the _TP_ column shows that Take Profit is not much larger than the Stop Loss level, so swap on senior steps can lead to failure to profit or even cause loss;
- red in the _Beginning_ column shows that the testing period does not exceed one year, which is probably not enough to accurately determine suitable parameters;
- red in _Swap L\|S (L%)_ shows daily percent value of more than 0.07% of the instrument price;
- yellow in _Swap L\|S (L%)_ shows daily percent value of less than 0.07% of the instrument price;
- red in _Max. losses_ shows that at least one series of losses for that symbol exceeded the maximum value;
- red in _Annual %_ shows symbols having annual yield less then 15%;
- yellow in _Annual %_ shows symbols having annual yield more then 40%, while the symbol itself has history for more than 5 years;
- red in _Profitability_ shows symbols having profitability less then 1.2;
- red in _Trades, total_ shows symbols, which had no more than 100 trades for the testing period, due to which the results may not be entirely accurate, because we do not have enough statistical data;
- red in _Symbol_ shows loss-making instruments;
- yellow in _Symbol_ shows instruments, having annual profit of more than 100%, while their price does not exceed $100.

Now, let's proceed to testing. We'll start with the broker #1.

**Broker #1**

Compared to other tested brokers, broker 1 has the following features in terms of stock market trading:

- swap for Short operations is positive, i.e. you have additional profit on Swap operations, not loss;
- you are paid dividends for Long positions;
- you lose dividends, if by the payment time you have a Short position for the instrument;
- Long swaps are larger than those provided by other brokers.

Of course, the strategy tester does not take into account the dividends which you could receive for Long positions or which you could be charged for Short ones. As for other aspects, testing results seem to be more reliable, than those with other brokers. This will be explained later.

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Swap L\|S (L%) | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adidas | 295 | 118 | 1.51 | 211 | 534 | 1 873 | 140 % | 6 | \- 0.1 \| 0.01 (0.047%) | 2016.01 | 220 | 465 |
| Adobe | 180 | 72 | 1.23 | 262 | 228 | 220 | 38 % | 5 | \- 0.13 \| 0.02 (0.048%) | 2016.01 | 350 | 460 |
| Aeroflot | 33 | ~65 | 3.13 | 109 | 11 | 42 | ~760 % | 4 | \- 0.1 \| 0.03 (0.095%) | 2018.02 | 310 | 800 |
| Alcoa | 2 319 | 165 | 1.34 | 43 | 455 | 2 620 | 41 % | 8 | \- 0.027 \| 0.004 (0.065%) | 2004.06 | 125 | 190 |
| Amazon | 723 | 289 | 1.32 | 1 929 | 1 577 | 5 969 | 151 % | 6 | \- 0.9 \| 0.14 (0.047%) | 2016.01 | 960 | 1 630 |
| American\_Express | 990 | 70 | 1.17 | 110 | 567 | 1 099 | 13 % | 14 | \- 0.06 \| 0.01 (0.052%) | 2004.06 | 105 | 245 |
| Apple | 1 115 | 82 | 1.55 | 219 | 549 | 5 164 | 69 % | 6 | \- 0.11 \| 0.02 (0.048%) | 2005.01 | 520 | 690 |
| ATT | 210 | 15 | 1.68 | 34 | 269 | 795 | 21 % | 7 | \- 0.02 \| 0.003 (0.065%) | 2004.06 | 120 | 240 |
| Baidu | 350 | 140 | 1.6 | 229 | 327 | 1 597 | 195 % | 6 | \- 0.15 \| 0.02 (0.067%) | 2016.01 | 370 | 610 |
| Banco\_Santander | 49 | 19 | 4.01 | 32 | 183 | 571 | 124 % | 7 | \- 0.02 \| 0.003 (0.061%) | 2016.02 | 50 | 275 |
| Bank\_of\_America | 476 | 32 | 1.27 | 31 | 87 | 282 | 22 % | 5 | \- 0.02 \| 0.003 (0.062%) | 2004.04 | 130 | 175 |
| Bayer | 81 | 54 | 1.72 | 76 | 525 | 454 | 57 % | 7 | \- 0.05 \| 0.006 (0.069%) | 2017.01 | 200 | 355 |
| BMW | 137 | 54 | 1.49 | 85 | 120 | 333 | 111 % | 5 | \- 0.05 \| 0.005 (0.059%) | 2016.01 | 215 | 335 |
| Boeing | 505 | 36 | 0.92 | 370 | 2 068 | -663 | - | 3 | \- 0.21 \| 0.03 (0.057%) | 2004.06 | 350 | 625 |
| Caterpillar | 576 | 41 | 1.27 | 156 | 457 | 956 | 14 % | 6 | \- 0.09 \| 0.01 (0.057%) | 2004.06 | 300 | 400 |
| Celgene | 132 | 88 | 2.43 | 88 | 285 | 1 155 | 270 % | 6 | \- 0.05 \| 0.008 (0.062%) | 2017.01 | 185 | 355 |
| ChinaMobile | 812 | 56 | 1.25 | 48 | 255 | 807 | 22 % | 7 | \- 0.03 \| 0.004 (0.058%) | 2004.04 | 150 | 230 |
| Cisco | 229 | 21 | 1.73 | 48 | 343 | 1 099 | 30 % | 8 | \- 0.03 \| 0.004 (0.055%) | 2008.01 | 75 | 245 |
| Citigroup | 386 | 26 | 2.48 | 74 | 3 258 | 31 752 | 69 % | 9 | \- 0.04 \| 0.007 (0.06%) | 2004.04 | 980 | 1 660 |
| Coca–Cola | 495 | 35 | 1.37 | 46 | 115 | 395 | 24 % | 6 | \- 0.026 \| 0.004 (0.057%) | 2004.06 | 130 | 140 |
| Daimler | 54 | 36 | 1.86 | 57 | 19 | 63 | 221 % | 3 | \- 0.04 \| 0.004 (0.07%) | 2017.01 | 160 | 210 |
| Deutsche\_Bank | 168 | 67 | 1.94 | 10 | 53 | 301 | 227 % | 5 | \- 0.007 \| 0.0008 (0.069%) | 2016.01 | 55 | 95 |
| Disney | 463 | 33 | 1.06 | 110 | 750 | 123 | 1 % | 7 | \- 0.06 \| 0.01 (0.055%) | 2004.06 | 160 | 260 |
| Dropbox | 34 | ~65 | 2.3 | 26 | 35 | 82 | ~468 % | 4 | \- 0.02 \| 0.003 (0.072%) | 2018.04 | 95 | 235 |
| eBay | 84 | 33 | 1.47 | 34 | 36 | 110 | 122 % | 3 | \- 0.025 \| 0.004 (0.077%) | 2016.01 | 110 | 135 |
| ENI | 38 | 25 | 2.18 | 16 | 51 | 83 | 108 % | 6 | \- 0.007 \| 0.0009 (0.046%) | 2017.01 | 30 | 70 |
| EOC | 186 | 74 | 1.16 | 20 | 67 | 40 | 23 % | 6 | \- 0.016 \| 0.002 (0.084%) | 2016.02 | 50 | 85 |
| Estee\_Lauder | 110 | 36 | 1.44 | 143 | 113 | 242 | 71 % | 5 | \- 0.08 \| 0.01 (0.059%) | 2015.10 | 230 | 390 |
| Ethan\_Allen | 222 | 74 | 1.39 | 21 | 55 | 106 | 64 % | 6 | \- 0.014 \| 0.002 (0.068%) | 2015.10 | 80 | 110 |
| Exxon | 1 250 | 89 | 1.39 | 85 | 801 | 2 573 | 22 % | 10 | \- 0.05 \| 0.007 (0.053%) | 2004.06 | 115 | 240 |
| Facebook | 398 | 66 | 1.12 | 164 | 353 | 225 | 10 % | 10 | \- 0.11 \| 0.016 (0.065%) | 2012.05 | 280 | 380 |
| Ferrari | 105 | 35 | 1.98 | 137 | 120 | 481 | 133 % | 5 | \- 0.07 \| 0.01 (0.053%) | 2015.10 | 200 | 550 |
| Ford | 145 | 10 | 1.93 | 9 | 53 | 247 | 30 % | 5 | \- 0.006 \| 0.001 (0.065%) | 2004.04 | 85 | 170 |
| Gazprom | 355 | 236 | 1.3 | 158 | 19 | 41 | 143 % | 6 | \- 0.11 \| 0.03 (0.071%) | 2017.03 | 150 | 180 |
| General\_Electrics | 558 | 39 | 1.48 | 12 | 126 | 468 | 26 % | 6 | \- 0.008 \| 0.001 (0.073%) | 2004.06 | 70 | 115 |
| Goldman\_Sachs | 164 | 65 | 1.6 | 235 | 385 | 1 071 | 111 % | 6 | \- 0.16 \| 0.02 (0.067%) | 2016.01 | 410 | 770 |
| Google | 902 | 200 | 1.1 | 1 184 | 3 743 | 2 818 | 18 % | 7 | \- 0.65 \| 0.1 (0.055%) | 2014.04 | 800 | 1550 |
| Harley\_Davidson | 677 | 56 | 1.29 | 45 | 342 | 906 | 22 % | 7 | \- 0.026 \| 0.004 (0.06%) | 2006.08 | 160 | 220 |
| Hewlett\_Packard | 766 | 54 | 1.28 | 25 | 116 | 412 | 25 % | 6 | \- 0.014 \| 0.002 (0.055%) | 2004.06 | 105 | 130 |
| Home\_Depot | 151 | 10 | 1.04 | 212 | 1 448 | 137 | 1 % | 8 | \- 0.11 \| 0.017 (0.052%) | 2004.06 | 350 | 640 |
| IBM | 949 | 67 | 1.4 | 151 | 882 | 3 571 | 28 % | 9 | \- 0.09 \| 0.014 (0.062%) | 2004.06 | 210 | 430 |
| Inditex | 48 | 32 | 2.24 | 27 | 61 | 184 | 201 % | 6 | \- 0.03 \| 0.004 (0.112%) | 2017.01 | 60 | 180 |
| Intel | 247 | 17 | 1.59 | 46 | 50 | 267 | 38 % | 5 | \- 0.013 \| 0.001 (0.029%) | 2004.06 | 130 | 190 |
| Johnson&Johnson | 512 | 36 | 1.1 | 142 | 676 | 219 | 2 % | 7 | \- 0.08 \| 0.01 (0.055%) | 2004.06 | 190 | 210 |
| JPMorgan | 836 | 59 | 1.25 | 118 | 360 | 818 | 18 % | 7 | \- 0.07 \| 0.01 (0.059%) | 2004.06 | 145 | 220 |
| Lenovo | 15 | 5 | 4.95 | 5 | 51 | 152 | 99 % | 5 | \- 0.002 \| 0.0003 (0.044%) | 2015.11 | 20 | 180 |
| Lukoil | 188 | 125 | 1.26 | 4 732 | 1 845 | 2 094 | 75 % | 7 | \- 3.02 \| 0.9 (0.062%) | 2017.03 | 430 | 830 |
| Mastercard | 544 | 45 | 1.04 | 221 | 455 | 105 | 1 % | 6 | \- 0.11 \| 0.016 (0.048%) | 2006.05 | 170 | 285 |
| McDonald | 431 | 30 | 1.19 | 163 | 467 | 554 | 8 % | 7 | \- 0.09 \| 0.014 (0.055%) | 2004.06 | 180 | 350 |
| Michael\_Kors | 98 | 32 | 1.94 | 72 | 57 | 247 | 144 % | 4 | \- 0.037 \| 0.005 (0.056%) | 2015.10 | 190 | 330 |
| Microsoft | 241 | 17 | 1.48 | 114 | 370 | 1 024 | 14 % | 5 | \- 0.06 \| 0.008 (0.049%) | 2004.06 | 130 | 330 |
| MTS | 60 | ~120 | 1.81 | 273 | 46 | 92 | ~400 % | 5 | \- 0.24 \| 0.07 (0.086%) | 2018.02 | 420 | 895 |
| Netflix | 86 | ~160 | 2.1 | 364 | 80 | 613 | ~1 532 % | 3 | \- 0.18 \| 0.03 (0.047%) | 2018.02 | 570 | 580 |
| Nike | 161 | 11 | 1.16 | 85 | 272 | 160 | 4 % | 7 | \- 0.04 \| 0.006 (0.047%) | 2004.04 | 165 | 320 |
| Nintendo\_US | 318 | 159 | 1.41 | 46 | 18 | 88 | 244 % | 4 | \- 0.034 \| 0.005 (0.075%) | 2016.08 | 55 | 80 |
| Nornickel | 74 | ~140 | 1.4 | 11 904 | 523 | 1 194 | ~456 % | 3 | \- 8.52 \| 2.68 (0.072%) | 2018.02 | 200 | 310 |
| Novatek | 50 | ~100 | 1.66 | 1 115 | 48 | 83 | ~345 % | 3 | \- 0.54 \| 0.17 (0.05%) | 2018.02 | 160 | 210 |
| nVidia | 338 | 135 | 1.12 | 265 | 691 | 431 | 24 % | 6 | \- 0.14 \| 0.02 (0.054%) | 2016.01 | 350 | 560 |
| Oracle | 99 | 39 | 2.39 | 50 | 28 | 160 | 228 % | 3 | \- 0.03 \| 0.004 (0.059%) | 2016.01 | 80 | 150 |
| Petrobras | 336 | 23 | 1.6 | 11 | 86 | 620 | 51 % | 5 | \- 0.008 \| 0.001 (0.075%) | 2004.04 | 240 | 300 |
| PetroChina | 755 | 53 | 1.61 | 77 | 762 | 4 263 | 39 % | 6 | \- 0.04 \| 0.006 (0.053%) | 2004.06 | 310 | 680 |
| Pfizer | 158 | 11 | 1.47 | 44 | 231 | 432 | 13 % | 7 | \- 0.02 \| 0.003 (0.049%) | 2004.06 | 105 | 230 |
| Philip\_Morris | 431 | 41 | 1.04 | 82 | 484 | 88 | 1 % | 7 | \- 0.064 \| 0.01 (0.079%) | 2008.04 | 200 | 260 |
| Procter&Gamble | 568 | 40 | 1.07 | 85 | 467 | 122 | 1 % | 9 | \- 0.047 \| 0.007 (0.057%) | 2004.06 | 150 | 185 |
| PVH | 421 | 140 | 1.07 | 142 | 194 | 114 | 19 % | 6 | \- 0.09 \| 0.013 (0.062%) | 2015.10 | 220 | 230 |
| Ralph\_Lauren | 227 | 75 | 1.71 | 136 | 385 | 1 114 | 96 % | 6 | \- 0.06 \| 0.01 (0.048%) | 2015.10 | 220 | 460 |
| Rosneft | 81 | ~150 | 1.82 | 436 | 36 | 96 | ~532 % | 5 | \- 0.25 \| 0.08 (0.055%) | 2018.02 | 510 | 910 |
| Salesforce | 109 | 43 | 2.19 | 156 | 81 | 413 | 203 % | 4 | \- 0.07 \| 0.01 (0.046%) | 2016.01 | 210 | 480 |
| Sberbank | 202 | 134 | 1.37 | 192 | 45 | 79 | 117 % | 5 | \- 0.034 \| 0.005 (0.017%) | 2017.03 | 420 | 510 |
| Snap | 89 | 59 | 1.97 | 9 | 44 | 222 | 336 % | 4 | \- 0.01 \| 0.001 (0.121%) | 2017.03 | 75 | 135 |
| Spotify | 153 | ~300 | 1.45 | 174 | 34 | 150 | ~882 % | 3 | \- 0.13 \| 0.02 (0.076%) | 2018.04 | 250 | 280 |
| SQM | 151 | 60 | 1.79 | 48 | 44 | 233 | 211 % | 4 | \- 0.03 \| 0.004 (0.061%) | 2016.01 | 125 | 240 |
| Starbucks | 227 | 15 | 1.68 | 57 | 38 | 245 | 46 % | 4 | \- 0.002 \| 0 (0.0001%) | 2004.04 | 160 | 195 |
| Tencent | 209 | 69 | 2.56 | 332 | 46 | 454 | 328 % | 5 | \- 0.24 \| 0.03 (0.074%) | 2015.11 | 450 | 1500 |
| Tesla | 1 731 | 216 | 1.07 | 296 | 2 003 | 1 209 | 7 % | 11 | \- 0.2 \| 0.03 (0.067%) | 2010.07 | 500 | 570 |
| Tiffany | 87 | 29 | 2.48 | 127 | 64 | 347 | 180 % | 4 | \- 0.06 \| 0.01 (0.049%) | 2015.10 | 220 | 500 |
| Toyota | 150 | 60 | 1.49 | 124 | 145 | 250 | 68 % | 5 | \- 0.08 \| 0.012 (0.063%) | 2016.01 | 140 | 330 |
| Travelers | 129 | 11 | 1.22 | 134 | 1 257 | 631 | 4 % | 7 | \- 0.08 \| 0.013 (0.063%) | 2007.03 | 330 | 670 |
| TripAdvisor | 240 | 96 | 1.5 | 49 | 98 | 305 | 124 % | 5 | \- 0.023 \| 0.003 (0.047%) | 2016.01 | 155 | 195 |
| Twitter | 89 | 35 | 2.44 | 29 | 29 | 220 | 303 % | 4 | \- 0.02 \| 0.003 (0.07%) | 2016.01 | 120 | 240 |
| UnitedHealth | 1 131 | 78 | 0.75 | 266 | 1 878 | -1 816 | - | 5 | \- 0.14 \| 0.02 (0.051%) | 2004.04 | 170 | 250 |
| Vale | 62 | 13 | 1.76 | 15 | 36 | 116 | 71 % | 3 | \- 0.008 \| 0.001 (0.055%) | 2014.04 | 85 | 115 |
| Verizon | 275 | 19 | 1.29 | 54 | 115 | 200 | 12 % | 5 | \- 0.03 \| 0.004 (0.054%) | 2004.06 | 145 | 210 |
| VF | 92 | 31 | 2.26 | 92 | 33 | 211 | 213 % | 3 | \- 0.044 \| 0.007 (0.049%) | 2015.10 | 180 | 320 |
| Visa | 97 | 6 | 0.83 | 149 | 269 | -91 | - | 3 | \- 0.07 \| 0.01 (0.049%) | 2004.04 | 290 | 440 |
| Vodafone | 118 | 29 | 1.54 | 22 | 57 | 100 | 43 % | 5 | \- 0.016 \| 0.002 (0.075%) | 2014.06 | 75 | 150 |
| Volkswagen | 397 | 158 | 1.13 | 152 | 1 090 | 610 | 22 % | 12 | \- 0.09 \| 0.01 (0.06%) | 2016.01 | 200 | 400 |
| Wells\_Fargo | 90 | 60 | 1.85 | 54 | 11 | 70 | 424 % | 3 | \- 0.034 \| 0.005 (0.064%) | 2017.01 | 110 | 155 |
| Williams\_Sonoma | 531 | 177 | 1.17 | 66 | 213 | 203 | 31 % | 6 | \- 0.03 \| 0.005 (0.049%) | 2015.10 | 100 | 135 |
| Yandex | 530 | 75 | 1.52 | 33 | 393 | 2 006 | 72 % | 6 | \- 0.024 \| 0.003 (0.074%) | 2011.05 | 95 | 150 |

As it can be seen from the table, the reversing technique is profitable on almost all stock market instruments, even without the entry point search. Only for 3 symbols out of 90 a profitable ratio of Stop Loss to Take Profit could not be found.

Now, let's have a look at the results obtained with other brokers.

There will be no balance charts for broker #1, because it is not possible to analyze all 90 balance charts. If you wish to have a closer look at any of the instruments, you can find the charts in the Strategy Tester report attached below.

However, let's analyze balance charts of symbols marked in yellow, having the history of over 3 years.

Ethan Allen:

![Ethan Allen, broker #1](https://c.mql5.com/2/34/ethan_allen_broker1.png)

Michael Kors:

![Michael Kors, broker #1](https://c.mql5.com/2/34/michael_kors_broker1.png)

Petrobras:

![Petrobras, broker #1](https://c.mql5.com/2/34/petrobras_broker1.png)

Starbucks:

![Starbucks, broker #1](https://c.mql5.com/2/34/starbucks_broker1.png)

Tencent:

![Tencent, broker #1](https://c.mql5.com/2/34/tencent_broker1.png)

Tiffany:

![Tiffany, broker #1](https://c.mql5.com/2/34/tiffany_broker1.png)

Vale:

![Vale, broker #1](https://c.mql5.com/2/34/vale_broker1.png)

VF:

![VF, broker #1](https://c.mql5.com/2/34/vf_broker1.png)

**Broker #2**

The main advantage of this broker over broker 1 is very small swaps. Swap is the same for all stock market symbols. It is equal to 6% of the opening price per year (0.016% per day) for Long positions. For Short positions it is 3% of the opening price per year (0.008% per day).

Compare the swap of 0.016% with the average swap provided by broker 1 = 0.055%. It is at least 3.5 times less.

However, there are certain disadvantages:

- swap for Short trades is negative, so broker 1 pays to you for overnight positions, broker 2 charges commission (which is though two times less than for Long);
- no dividends are paid for Long positions;
- there is also a positive side of that - no dividend is charged for Short positions.

Before we consider the table with the testing results, it should be noted that these results are not entirely accurate. The Strategy Tester practically ignores swaps for this broker. The total loss on swaps usually does not exceed 10 cents over the entire testing period, whereas in real trading broker swaps are much larger. Also note that the annual number of trades is very small for most of the instruments, which means that a swap can take a large share of the profits.

So, you should consider this possibility during your own tests. And use Islamic accounts to trade with such brokers. That is, accounts without swaps, while a commission is paid for each trade regardless of how long you hold the security. However, the commission is also ignored for this broker in the Strategy Tester.

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 477 | 86 | 1.5 | 217 | 132 | 739 | 101 % | 5 | 2013.01 | 250 | 310 |
| ADBE | 224 | 40 | 2.02 | 260 | 102 | 825 | 147 % | 4 | 2013.01 | 340 | 500 |
| AMZN | 642 | 116 | 1.78 | 1 911 | 602 | 8 956 | 270 % | 5 | 2013.01 | 1 440 | 2 040 |
| ATVI | 395 | 71 | 1.29 | 80 | 56 | 197 | 63 % | 5 | 2013.01 | 135 | 140 |
| BA | 980 | 178 | 1.4 | 371 | 229 | 1 308 | 103 % | 6 | 2013.01 | 280 | 310 |
| BAC | 72 | 13 | 1.57 | 30 | 35 | 77 | 40 % | 4 | 2013.01 | 115 | 145 |
| BRK.B | 377 | 68 | 1.58 | 220 | 326 | 1 106 | 61 % | 7 | 2013.01 | 210 | 310 |
| C | 339 | 61 | 1.3 | 73 | 124 | 299 | 43 % | 6 | 2013.01 | 140 | 180 |
| CAT | 398 | 72 | 1.6 | 156 | 117 | 624 | 96 % | 5 | 2013.01 | 260 | 280 |
| CMCSA | 46 | 8 | 2.88 | 37 | 47 | 179 | 69 % | 4 | 2013.01 | 115 | 315 |
| CSCO | 127 | 36 | 1.68 | 48 | 42 | 133 | 90 % | 5 | 2015.01 | 90 | 125 |
| CVX | 732 | 133 | 1.35 | 120 | 296 | 586 | 35 % | 7 | 2013.01 | 165 | 180 |
| DAL | 312 | 56 | 1.67 | 59 | 407 | 997 | 44 % | 11 | 2013.01 | 110 | 245 |
| DIS | 568 | 103 | 1.25 | 110 | 439 | 460 | 19 % | 10 | 2013.01 | 135 | 195 |
| EA | 745 | 135 | 1.48 | 114 | 170 | 961 | 102 % | 6 | 2013.01 | 140 | 200 |
| EBAY | 141 | 25 | 1.25 | 34 | 63 | 72 | 20 % | 6 | 2013.01 | 130 | 145 |
| FB | 384 | 69 | 1.52 | 162 | 129 | 699 | 98 % | 5 | 2013.01 | 310 | 380 |
| FOXA | 265 | 48 | 1.6 | 44 | 47 | 200 | 77 % | 4 | 2013.01 | 90 | 120 |
| GE | 272 | 49 | 1.58 | 12 | 49 | 172 | 63 % | 6 | 2013.01 | 55 | 80 |
| GM | 232 | 42 | 1.67 | 35 | 353 | 552 | 28 % | 7 | 2013.01 | 115 | 150 |
| GOOGL | 715 | 130 | 1.89 | 1 172 | 2 851 | 15 477 | 98 % | 7 | 2013.01 | 1 100 | 1 660 |
| GS | 197 | 35 | 1.52 | 235 | 273 | 819 | 54 % | 5 | 2013.01 | 690 | 750 |
| HPE | 90 | 30 | 1.97 | 16 | 27 | 100 | 123 % | 4 | 2015.11 | 55 | 70 |
| IBM | 239 | 43 | 1.73 | 150 | 332 | 900 | 49 % | 6 | 2013.01 | 350 | 510 |
| INTC | 167 | 30 | 1.93 | 46 | 23 | 208 | 164 % | 4 | 2013.01 | 130 | 180 |
| JNJ | 207 | 37 | 1.78 | 142 | 77 | 364 | 85 % | 4 | 2013.01 | 240 | 280 |
| JPM | 589 | 107 | 1.39 | 117 | 133 | 460 | 62 % | 6 | 2013.01 | 125 | 145 |
| KO | 107 | 19 | 2.13 | 46 | 29 | 151 | 94 % | 4 | 2013.01 | 100 | 160 |
| LLY | 337 | 61 | 1.97 | 106 | 222 | 1 310 | 107 % | 7 | 2013.01 | 120 | 275 |
| MCD | 219 | 39 | 1.71 | 164 | 86 | 350 | 73 % | 4 | 2013.01 | 280 | 290 |
| MMM | 448 | 81 | 1.23 | 216 | 334 | 537 | 29 % | 6 | 2013.01 | 260 | 310 |
| MON | 347 | 63 | 1.56 | 127 | 58 | 393 | 123 % | 4 | 2013.01 | 210 | 230 |
| MSFT | 186 | 33 | 2.09 | 113 | 70 | 476 | 123 % | 5 | 2013.01 | 150 | 275 |
| NEM | 439 | 79 | 1.56 | 31 | 66 | 378 | 104 % | 9 | 2013.01 | 100 | 135 |
| NFLX | 417 | 75 | 1.58 | 360 | 339 | 1 634 | 87 % | 4 | 2013.01 | 510 | 660 |
| NKE | 176 | 32 | 2.24 | 85 | 121 | 741 | 111 % | 6 | 2013.01 | 135 | 275 |
| NVDA | 738 | 133 | 1.58 | 262 | 210 | 1 339 | 115 % | 6 | 2013.01 | 300 | 330 |
| ORCL | 237 | 43 | 1.67 | 50 | 57 | 271 | 86 % | 5 | 2013.01 | 90 | 150 |
| PEP | 369 | 67 | 1.52 | 114 | 94 | 353 | 68 % | 5 | 2013.01 | 140 | 175 |
| PFE | 97 | 17 | 1.75 | 44 | 30 | 122 | 73 % | 4 | 2013.01 | 120 | 165 |
| PG | 340 | 61 | 1.57 | 85 | 108 | 433 | 72 % | 6 | 2013.01 | 115 | 175 |
| PM | 562 | 102 | 1.43 | 83 | 160 | 503 | 57 % | 6 | 2013.01 | 125 | 155 |
| PRU | 690 | 125 | 1.34 | 104 | 159 | 586 | 67 % | 6 | 2013.01 | 150 | 185 |
| PYPL | 141 | 47 | 2.31 | 90 | 69 | 590 | 285 % | 5 | 2015.07 | 135 | 315 |
| SBUX | 266 | 48 | 1.54 | 57 | 55 | 221 | 73 % | 5 | 2013.01 | 130 | 160 |
| TWX | 140 | 25 | 1.85 | 98 | 156 | 408 | 47 % | 5 | 2013.01 | 255 | 345 |
| UPS | 59 | 10 | 2.98 | 118 | 42 | 325 | 140 % | 3 | 2013.01 | 350 | 650 |
| VZ | 182 | 33 | 2.27 | 54 | 79 | 570 | 131 % | 6 | 2013.01 | 95 | 200 |
| WFC | 329 | 59 | 1.45 | 55 | 180 | 293 | 29 % | 7 | 2013.01 | 100 | 130 |
| WMT | 315 | 57 | 1.38 | 95 | 381 | 453 | 21 % | 7 | 2013.01 | 140 | 175 |
| XOM | 523 | 95 | 1.26 | 85 | 603 | 844 | 25 % | 12 | 2013.01 | 105 | 200 |

Unlike broker #1, this broker does not have any losing symbols for our trading strategy. Perhaps this can be due to the absence of swaps and commissions in the Strategy Tester. The other possible reason for that is the testing period, which is only 5 years. Compare with broker #1 providing a 14-year history for some of the instruments.

Now let's view the balance charts for the tested symbols.

AAPL:

![AAPL, broker #2](https://c.mql5.com/2/34/aapl_broker2.png)

ADBE:

![ADBE, broker #2](https://c.mql5.com/2/34/adbe_broker2.png)

AMZN:

![AMZN, broker #2](https://c.mql5.com/2/34/amzn_broker2.png)

ATVI:

![ATVI, broker #2](https://c.mql5.com/2/34/atvi_broker2.png)

BA:

![BA, broker #2](https://c.mql5.com/2/34/ba_broker2.png)

BAC:

![BAC, broker #2](https://c.mql5.com/2/34/bac_broker2.png)

BRKB:

![BRKB, broker #2](https://c.mql5.com/2/34/brkb_broker2.png)

C:

![C, broker #2](https://c.mql5.com/2/34/c_broker2.png)

CAT:

![CAT, broker #2](https://c.mql5.com/2/34/cat_broker2.png)

CMCSA:

![CMCSA, broker #2](https://c.mql5.com/2/34/cmcsa_broker2.png)

CSCO:

![CSCO, broker #2](https://c.mql5.com/2/34/csco_broker2.png)

CVX:

![CVX, broker #2](https://c.mql5.com/2/34/cvx_broker2.png)

DAL:

![DAL, broker #2](https://c.mql5.com/2/34/dal_broker2.png)

DIS:

![DIS, broker #2](https://c.mql5.com/2/34/dis_broker2.png)

EA:

![EA, broker #2](https://c.mql5.com/2/34/ea_broker2.png)

EBAY:

![EBAY, broker #2](https://c.mql5.com/2/34/ebay_broker2.png)

FB:

![FB, broker #2](https://c.mql5.com/2/34/fb_broker2.png)

FOXA:

![FOXA, broker #2](https://c.mql5.com/2/34/foxa_broker2.png)

GE:

![GE, broker #2](https://c.mql5.com/2/34/ge_broker2.png)

GM:

![GM, broker #2](https://c.mql5.com/2/34/gm_broker2.png)

GOOGL:

![GOOGL, broker #2](https://c.mql5.com/2/34/googl_broker2.png)

GS:

![GS, broker #2](https://c.mql5.com/2/34/gs_broker2.png)

HPE:

![HPE, broker #2](https://c.mql5.com/2/34/hpe_broker2.png)

IBM:

![IBM, broker #2](https://c.mql5.com/2/34/ibm_broker2.png)

INTC:

![INTC, broker #2](https://c.mql5.com/2/34/intc_broker2.png)

JNJ:

![JNJ, broker #2](https://c.mql5.com/2/34/jnj_broker2.png)

JPM:

![JPM, broker #2](https://c.mql5.com/2/34/jpm_broker2.png)

KO:

![KO, broker #2](https://c.mql5.com/2/34/ko_broker2.png)

LLY:

![LLY, broker #2](https://c.mql5.com/2/34/lly_broker2.png)

MCD:

![MCD, broker #2](https://c.mql5.com/2/34/mcd_broker2.png)

MMM:

![MMM, broker #2](https://c.mql5.com/2/34/mmm_broker2.png)

MON:

![MON, broker #2](https://c.mql5.com/2/34/mon_broker2.png)

MSFT:

![MSFT, broker #2](https://c.mql5.com/2/34/msft_broker2.png)

NEM:

![NEM, broker #2](https://c.mql5.com/2/34/nem_broker2.png)

NFLX:

![NFLX, broker #2](https://c.mql5.com/2/34/nflx_broker2.png)

NKE:

![NKE, broker #2](https://c.mql5.com/2/34/nke_broker2.png)

NVDA:

![NVDA, broker #2](https://c.mql5.com/2/34/nvda_broker2.png)

ORCL:

![ORCL, broker #2](https://c.mql5.com/2/34/orcl_broker2.png)

PEP:

![PEP, broker #2](https://c.mql5.com/2/34/pep_broker2.png)

PFE:

![PFE, broker #2](https://c.mql5.com/2/34/pfe_broker2.png)

PG:

![PG, broker #2](https://c.mql5.com/2/34/pg_broker2.png)

PM:

![PM, broker #2](https://c.mql5.com/2/34/pm_broker2.png)

PRU:

![PRU, broker #2](https://c.mql5.com/2/34/pru_broker2.png)

PYPL:

![PYPL, broker #2](https://c.mql5.com/2/34/pypl_broker2.png)

SBUX:

![SBUX, broker #2](https://c.mql5.com/2/34/sbux_broker2.png)

TWX:

![TWX, broker #2](https://c.mql5.com/2/34/twx_broker2.png)

UPS:

![UPS, broker #2](https://c.mql5.com/2/34/ups_broker2.png)

VZ:

![VZ, broker #2](https://c.mql5.com/2/34/vz_broker2.png)

WFC:

![WFC, broker #2](https://c.mql5.com/2/34/wfc_broker2.png)

WMT:

![WMT, broker #2](https://c.mql5.com/2/34/wmt_broker2.png)

XOM:

![XOM, broker #2](https://c.mql5.com/2/34/xom_broker2.png)

**Broker #3**

Now let's check the results with the third broker. This broker also provides Islamic accounts. Apparently, that is why the Strategy Tester also ignores the swap. But it takes into account the commission, so testing results in this case are more realistic. Of course, you should also use non-swap accounts for trading.

Without Islamic accounts, the broker swaps are negative for both Long and Short positions. Long swap is 2.5% of the instrument price at the moment of swap calculation per year (0.007% per day). Short swap is 1.5% of the instrument price at the moment of swap calculation per year (0.004% per day).

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #AA | 624 | 56 | 2.4 | 43 | 136 | 1 782 | 119 % | 6 | 2007.07 | 90 | 135 |
| #AIG | 328 | 29 | 1.5 | 54 | 119 | 497 | 37 % | 5 | 2007.07 | 240 | 270 |
| #GE | 354 | 32 | 1.92 | 12 | 91 | 580 | 57 % | 5 | 2007.07 | 60 | 145 |
| #HPQ | 471 | 42 | 1.83 | 25 | 211 | 1 203 | 51 % | 6 | 2007.07 | 90 | 195 |
| #INTC | 300 | 27 | 1.68 | 46 | 137 | 651 | 43 % | 6 | 2007.07 | 90 | 195 |
| #IP | 154 | 14 | 1.91 | 54 | 104 | 488 | 42 % | 5 | 2007.07 | 210 | 350 |
| #KO | 491 | 44 | 1.34 | 46 | 87 | 302 | 31 % | 6 | 2007.07 | 110 | 130 |
| #MO | 160 | 14 | 1.45 | 62 | 72 | 202 | 25 % | 5 | 2007.07 | 185 | 250 |
| #PFE | 245 | 22 | 1.47 | 44 | 32 | 115 | 32 % | 6 | 2007.07 | 100 | 110 |
| #T | 292 | 26 | 1.5 | 33 | 109 | 378 | 31 % | 6 | 2007.07 | 95 | 145 |
| #VZ | 201 | 18 | 1.65 | 54 | 52 | 257 | 44 % | 4 | 2007.07 | 165 | 210 |

As you can see, these results are somewhere between the first and second broker. No losing symbols were found. However, the annual profit percentage is less than that with broker #2.

AA:

![AA, broker #3](https://c.mql5.com/2/34/aa_broker3.png)

AIG:

![AIG, broker #3](https://c.mql5.com/2/34/aig_broker3.png)

GE:

![GE, broker #3](https://c.mql5.com/2/34/ge_broker3.png)

HPQ:

![HPQ, broker #3](https://c.mql5.com/2/34/hpq_broker3.png)

INTC:

![INTC, broker #3](https://c.mql5.com/2/34/intc_broker3.png)

IP:

![IP, broker #3](https://c.mql5.com/2/34/ip_broker3.png)

MO:

![MO, broker #3](https://c.mql5.com/2/34/mo_broker3.png)

PFE:

![PFE, broker #3](https://c.mql5.com/2/34/pfe_broker3.png)

T:

![T, broker #3](https://c.mql5.com/2/34/t_broker3.png)

VZ:

![VZ, broker #3](https://c.mql5.com/2/34/vz_broker3.png)

**Let's sum up.**

As can be seen from Testing Results, the stock market is perfect for applying reversing techniques. Profit can be obtained on almost all financial instruments. There are some differences in the trading strategy behavior in the stock market compared to the Forex market.

The most important difference is that Take Profit does not exceed double Stop Loss value in most cases. Often, the best variant found by the Strategy Tester was the one having Stop Loss greater than Take Profit. However, I did not include them in the tables, since I do not believe such variants - that's because we made the profit because the market was constantly growing. Therefore, take profit was obtained at the very first trade. If the market changes and Take Profit starts triggering at least at the second trade, that profit will not be enough. So the profit of a Take Profit order will be less than the loss of the entire chain.

Before you rush into stock market trading, pay attention to the following regularity. The longer the history available for an instrument, the lower the yearly profit percentage. Almost none of the symbols having 12-year history showed over 50% per year. While almost all symbols with the history less than 2 years show over 200% of profit.

For this reason, you should choose instruments having a long historic period for long-term trading. They produce better results. Whereas symbols with a shorter testing period yet haven't met a period, in which they hit your Stop Loss at every step.

Do not forget about gaps and large swaps due to which your testing profit can turn into losses in live trading. Especially if the Take Profit value is almost equal to Stop Loss. Here is an example of the Spotify symbol, with which I couldn't succeed. In real trading, it had 8 trade chains, each of which was closed by a Take Profit. However the result of all trades was a loss of 3 dollars, due to large gaps, which often happened in the unfavorable direction, and large swaps. Perhaps this was not a good period for this instrument, since none of the chains was completed with a Take Profit on the first step. So, it could possibly cover that loss an bring profit in other periods. Moreover, in the Strategy Tester it showed more than 700% deposit growth per year. However, I had to remove this symbol from the real trading set.

### Agricultural market, commodities, metals

The agricultural, commodity and metal markets can be considered as raging markets. However, they differ from the Forex market at least in that the commodity price is more influenced by season and weather conditions.

Although the markets are ranging, we will start with the Long deal — at least this is a simple solution.

**Broker #1**

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Swap L\|S (L%) | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COFFEE | 305 | 152 | 2.22 | 99 | 268 | 2 842 | 530 % | 5 | \- 0.018 \| - 0.01 (0.019%) | 2016.09 | 120 | 240 |
| CORN | 481 | 240 | 1.36 | 356 | 135 | 397 | 147 % | 6 | \- 0.05 \| - 0.03 (0.015%) | 2016.09 | 250 | 325 |
| SOYBEAN | 277 | 138 | 1.47 | 845 | 801 | 1 856 | 115 % | 7 | \- 0.14 \| - 0.08 (0.017%) | 2016.09 | 950 | 1600 |
| SUGAR | 29 | 14 | 2.78 | 11 | 138 | 541 | 196 % | 3 | \- 0.002 \| - 0.001 (0.02%) | 2016.09 | 70 | 170 |
| WHEAT | 106 | 53 | 1.98 | 518 | 247 | 908 | 183 % | 5 | \- 0.064 \| - 0.036 (0.012%) | 2016.09 | 1 175 | 2 000 |
| COCOA | 75 | 37 | 1.78 | 2 156 | 441 | 873 | 98 % | 17 | \- 0.28 \| - 0.16 (0.013%) | 2016.09 | 64 | 155 |
| CL | 2 677 | 223 | 1.14 | 71 | 13 345 | 19 942 | 12 % | 12 | \- 0.003 \| 0.0001 (0.004%) | 2006.09 | 100 | 190 |
| HO | 4 716 | 410 | 1.17 | 2 | 8 891 | 39 846 | 38 % | 23 | \- 0.0001 \| 0.00002 (0.005%) | 2007.01 | 200 | 460 |
| NG | 235 | 19 | 2.1 | 2 | 14 515 | 61 996 | 36 % | 13 | \- 0.0004 \| - 0.0002 (0.014%) | 2006.11 | 24 | 90 |
| WT | 1 938 | 161 | 1.22 | 71 | 14 730 | 42 437 | 24 % | 15 | \- 0.003 \| 0.0001 (0.004%) | 2006.09 | 105 | 345 |
| BRN | 827 | 75 | 1.59 | 78 | 4 832 | 28 274 | 53 % | 10 | \- 0.003 \| 0.0001 (0.004%) | 2007.08 | 175 | 345 |
| PA | 1 525 | 138 | 1.23 | 1 041 | 4 784 | 19 013 | 36 % | 16 | \- 0.14 \| - 0.08 (0.013%) | 2007.11 | 880 | 2 540 |
| HG | 2 233 | 203 | 1.24 | 2 | 5 589 | 18 067 | 29 % | 8 | \- 0.0005 \| - 0.0003 (0.016%) | 2007.06 | 460 | 650 |

In general, profit is achieved on all instruments. Agrarian market symbols bring much greater profit, but this can be due to the limited testing period. However, this profit is far less than that of the stock market.

Also, symbols associated with oil and gas have positive swaps for Short positions.

COFFEE:

![COFFEE, broker #1](https://c.mql5.com/2/34/coffee_broker1.png)

COCOA:

![COCOA, broker #1](https://c.mql5.com/2/34/cocoa_broker1.png)

CORN:

![CORN, broker #1](https://c.mql5.com/2/34/corn_broker1.png)

SOYBEAN:

![SOYBEAN, broker #1](https://c.mql5.com/2/34/soybean_broker1.png)

WHEAT:

![WHEAT, broker #1](https://c.mql5.com/2/34/wheat_broker1.png)

HO:

![HO, broker #1](https://c.mql5.com/2/34/ho_broker1.png)

HG:

![HG, broker #1](https://c.mql5.com/2/34/hg_broker1.png)

NG:

![NG, broker #1](https://c.mql5.com/2/34/ng_broker1.png)

BRN:

![BRN, broker #1](https://c.mql5.com/2/34/brn_broker1.png)

WT:

![WT, broker #1](https://c.mql5.com/2/34/wt_broker1.png)

**Broker #2**

Broker #2 only supports oil trading, so the table will be small.

Here swaps are negative and are equal to 6% of the position open price per year for Long positions (0.016% per day) and 3% - for Short positions (0.008% per day). Broker #1 offers a much better swap for similar symbols.

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brent | 710 | 142 | 1.54 | 78 | 8 721 | 32 934 | 75 % | 21 | 2013.09 | 70 | 275 |
| WTI | 346 | 173 | 1.61 | 71 | 893 | 3 700 | 207 % | 6 | 2016.07 | 90 | 130 |

This time the Strategy Tester takes swaps into account.

Brent:

![Brent, broker #2](https://c.mql5.com/2/34/brent_broker2.png)

WTI:

![WTI, broker #2](https://c.mql5.com/2/34/wti_broker2.png)

**Let's sum up.**

The following conclusion can be made for these instruments: reversing can also be applied here.

### ETF and indices

An index includes several stocks. It means that they also have a clear trend. And at the same time, their volatility should be much less than that of a single stock, since a large number of stocks are supposed to smooth out strong falls of some of them. Let's see how these features affect testing results.

**Broker #2**

Similar to the stock market, swap is the same for all instruments and is equal to 6% of the position open price per year for Long positions (0.016% per day) and 3% - for Short positions (0.008% per day).

| Symbol | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| US500Cash | 262 | 2.42 | 2926 | 4 276 | 26 996 | 631 % | 14 | 2017.12 | 95 | 400 |
| US30Cash | 624 | 1.53 | 26 713 | 22 633 | 85 694 | 378 % | 7 | 2017.12 | 840 | 1 640 |
| USTECHCash | 69 | 3 | 7 516 | 765 | 6 998 | 914 % | 2 | 2017.12 | 1 000 | 1 420 |

Testing results obtained with this broker look fine. But the results are note quite reliable, since historical data is available for less than a year. Moreover, this year was highly volatile, with large trend movements. So this year can be considered ideal for our trading strategy.

US500Cash:

![US500Cash, broker #2](https://c.mql5.com/2/34/US500Cash_broker2.png)

US30Cash:

![US30Cash, broker #2](https://c.mql5.com/2/34/US30Cash_broker2.png)

USTECHCash:

![USTECHCash, broker #2](https://c.mql5.com/2/34/USTECHCash_broker2.png)

**Broker#1, indices**

| Symbol | Trades, total | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Annual % | Max. losses | Swap L\|S (L%) | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ES | 648 | 51 | 1.5 | 2 939 | 4 192 | 10 832 | 20 % | 10 | \- 0.36 \| - 0.2 (0.012%) | 2006.01 | 1 950 | 4 200 |
| NQ | 1 829 | 146 | 1.33 | 7 585 | 2 793 | 9 014 | 26 % | 8 | \- 0.87 \| - 0.49 (0.011%) | 2006.04 | 3 250 | 4 400 |
| TF | 569 | 51 | 1.2 | 1 721 | 7 603 | 9 385 | 11 % | 13 | \- 0.21 \| - 0.12 (0.012%) | 2007.08 | 185 | 340 |
| YM | 1 334 | 106 | 1.46 | 26 762 | 2 797 | 20 089 | 59 % | 7 | \- 3.18 \| -1.79 (0.012%) | 2006.04 | 155 | 210 |
| FDAX | 1 727 | 138 | 1.18 | 12 402 | 10 631 | 21 110 | 15 % | 9 | \- 1.48 \| - 1.23 (0.012%) | 2006.01 | 680 | 1 820 |
| FESX | 77 | 6 | 1.95 | 3 412 | 10 996 | 33 318 | 24 % | 7 | \- 0.43 \| - 0.35 (0.012%) | 2006.01 | 175 | 430 |
| FTI | 1 405 | 117 | 1.36 | 549 | 5 536 | 23 869 | 35 % | 8 | \- 0.06 \| - 0.05 (0.011%) | 2006.09 | 375 | 780 |
| IBX | 72 | 48 | 1.61 | 9 560 | 2 114 | 3 255 | 102 % | 7 | \- 1.25 \| - 1.03 (0.013%) | 2017.01 | 125 | 245 |
| MIB | 70 | 46 | 1.62 | 21 400 | 583 | 1 154 | 131 % | 4 | \- 2.66 \| - 2.21 (0.012%) | 2017.01 | 380 | 450 |
| Russia50 | 142 | 94 | 1.13 | 1 129 | 360 | 159 | 29 % | 6 | \- 0.16 \| - 0.09 (0.014%) | 2017.03 | 175 | 220 |
| Z | 3 248 | 249 | 1.14 | 7 448 | 7 143 | 10 222 | 11 % | 10 | \- 0.87 \| - 0.78 (0.012%) | 2005.05 | 340 | 550 |
| FCE | 2 540 | 195 | 1.14 | 5 480 | 2 491 | 4 639 | 14 % | 10 | \- 0.62 \| - 0.52 (0.011%) | 2005.12 | 380 | 490 |
| NKD | 635 | 55 | 1.35 | 23 800 | 6 310 | 12 237 | 16 % | 12 | \- 2.82 \| - 1.59 (0.012%) | 2007.03 | 260 | 520 |
| XU | 35 | 23 | 3.26 | 11 765 | 18 | 116 | 429 % | 2 | \- 1.78 \| - 1 (0.015%) | 2017.01 | 3 400 | 5 000 |
| HSI | 319 | 79 | 1.39 | 27 856 | 501 | 749 | 37 % | 7 | \- 3.39 \| - 2.79 (0.012%) | 2014.07 | 360 | 495 |
| TA25 | 747 | - | 0.42 | 1 671 | 774 | -766 | - | 17 | \- 2 \| - 2 (0.12%) | 1992.12 | 300 | 285 |
| CHILE | 4 153 | - | 0 | 27 477 | 2 967 | -2 967 | - | 4 153 | \- 2.4 \| - 2.4 (0.009%) | 1991.01 | 300 | 380 |

The results are significantly worse here. Nevertheless, only two symbols showed loss: the Israeli TA25 index and the South American CHILE index. In both cases, the issues is not connected with our trading strategy, while loss was caused by too large commissions for these symbol trades. The commission for CHILE trades can reach one dollar, while Take Profit makes only a few cents of profit. That is why absolutely all deals for this symbol are losing.

So, the most favorable symbol for the strategy trading is Dow Jones (YM). For 12 years, it shows 60% per annum.

YM:

![YM, broker #1](https://c.mql5.com/2/34/ym_broker1.png)

**Broker #1, ETF**

Broker #1 also supports ETF. This is a new symbol for that broker, that is why it has only a few months of historic data. Therefore we cannot know how the parameters will behave on a larger period.

Also note that the Short swap is positive for this instrument. This is a pleasant addition.

| Symbol | Trades, per year | Profit factor | Cur. price | Max. drawdown | Profit Column | Max. losses | Swap L\|S (L%) | Start | SL | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EWG | 50 | 2.27 | 30 | 2 | 11 | 2 | \- 0.01 \| 0.002 (0.034%) | 2018.06 | 10 | 30 |
| EWU | 12 | 1.56 | 34 | 2 | 2 | 2 | \- 0.01 \| 0.002 (0.034%) | 2018.06 | 50 | 50 |
| EWW | 19 | 1.97 | 50 | 27 | 30 | 4 | \- 0.017 \| 0.003 (0.034%) | 2018.06 | 85 | 185 |
| EWZ | 24 | 2.97 | 33 | 6 | 25 | 3 | \- 0.017 \| 0.002 (0.036%) | 2018.06 | 35 | 135 |
| FXI | 47 | 1.9 | 43 | 5 | 22 | 3 | \- 0.014 \| 0.003 (0.033%) | 2018.06 | 25 | 70 |
| IJH | 49 | 2.12 | 204 | 10 | 53 | 3 | \- 0.066 \| 0.015 (0.033%) | 2018.06 | 65 | 155 |
| ILF | 17 | 2.03 | 31 | 2 | 4 | 1 | \- 0.01 \| 0.002 (0.034%) | 2018.06 | 100 | 45 |
| SPY | 78 | 1.68 | 292 | 13 | 32 | 5 | \- 0.09 \| 0.02 (0.032%) | 2018.06 | 85 | 95 |
| VGK | 17 | 0.29 | 57 | 45 | -30 | 5 | \- 0.019 \| 0.004 (0.034%) | 2018.06 | 45 | 95 |

The found optimal parameters better suit intraday trading rather than medium-term trading. However, this can be connected with the short available history period.

As for ETF of SPY and IJH, trading here is connected with one big issue, which is the large maintenance margin. Even with a minimum lot, the margin is equal to $10. Profit in case of a winning trade is about 1 dollar. And this is only the first trade in the chain. If you are unlucky and have to open the 4th trade in the chain, you will need to have the free margin of 80 dollars. On the 8th step that would be equal to $1280. Does it make sense to freeze $1280 as a margin aiming to earn $1?

I think this is a significant disadvantage of trading ETF with tight Stop levels. It is impossible to test longer Stop levels, because the history is available only for the last 2 months.

Also, EWG testing on a real account was not successful. Each new day opening started with a Stop Loss, because the symbol had huge gaps. Finally, one of the symbol chains reached the 7th step and I immediately closed it, when the profit reached half of the loss. And I forgot about this symbol. However, I later analyzed the chart and realized that I had acted recklessly, since the seventh step would have eventually closed by Take Profit. However, such instruments are not the best choice in terms of the human nervous system.

### Conclusions

As you can see from the charts, the reversing strategy can actually work. Better select new markets for the strategy, as well as markets with good trends, such as stock and index markets. Make sure that your broker provides good conditions allowing medium term trading without additional expenses.

However, it is not recommended to use reversing techniques with very popular instruments traded by big market players, who can move these markets in the desired direction.

The most important aspect you should pay attention to is that entry is performed at any time and not based on a signal. This means that results can differ much from the above testing charts. For example, the eighth step in a chain was closed with a profit. But we could make an entry earlier or later and thus close the entire chain with a loss. So do not fully rely on the above results.

Also, make sure to test and optimize Expert Advisor parameters for your specific broker. As mentioned above, the EA optimal parameters for the same symbols can differ with different brokers. The parameters and screenshots in this article are provided for information purpose only. They are not trading recommendations or instructions for use in the final form.

And the last thing I wanted to mention in this article. The previous article brought its results — I received the author's fee :) Now, you can see how reversing operates on live accounts. To enable this, I published a signal [for the Forex market](https://www.mql5.com/en/signals/465950). It operates on a cent account. The EA trades 7 symbols: EURUSD, GBPUSD, USDJPY, AUDJPY, GBPJPY, XAUUSD and XAGUSD. Lot doubling for the most of the symbols is performed on every second trade within a chain.

Positions are opened and partially managed by the Expert Advisor. However, since I am a human being, I get nervous if the chain reaches the fourth or more step, so I close the chain as soon as the profit exceeds the total chain loss. After that, I open a trade with the initial lot in the direction of the closed position. So the signal profit could be possibly higher if I did not interfere.

### Additional materials

The following zip archives are attached below:

- **RevertEA**. The Expert Advisor, the trading results of which are provided in this article;
- **SETfiles**. SET files with the optimal EA setting for each symbol and each of the considered brokers;
- **TESTfiles\_plain** and **TESTfiles\_cci**. Testing reports on all symbols traded with different brokers;
- **RevertManualEA**. Another reversing Expert Advisor, which only manages trades opened by you.

The RevertManualEA assists in reversing if you prefer to open the first trade manually. Run this EA on one of the charts, after which it will automatically manage all deals, which have a comment with the current step, and which have a matching Magic number.

That is, all you need to do is to manually open the first deal, for example, using the utility [Creating orders with a fixed stop in dollars](https://www.mql5.com/en/market/product/29801). Within a minute (the EA scans for new positions and updates in current positions every 10 seconds) RevertManualEA will create an opposite SELL STOP or BUY STOP trade, and will manage this trade until it hits Take Profit or reaches the maximum step in the chain.

Pay attention to another utility, which can assist in applying this strategy: [Information panel for traders](https://www.mql5.com/en/market/product/30245). The utility shows useful information on any open chart. We are especially interested in the following features of the utility:

- _Show swap info: long; short; 3-day swap_. It displays information in points, symbol currency and percentage of the current symbol price (all the three modes may not be available for all calculation options);
- _Show trade profitability: amount (number)_. Displays the number of trades and the total profit for the symbol for the month/year or the entire history. In addition, the profit amount and the number of trades for the current day are displayed on the chart where the EA is running;
- _Show the amount of loss before the first profitable trade_. Displays how much you have already lost in the current chain (so you don't need to calculate this manually, if you want to close the current chain step once the entire chain result becomes profitable)
- _Notify me of current balance, equity and free funds in_. At a certain time, the utility sends a notification to the specified MetaTrader MetaQuotes ID with the listed information - this helps you to control the amount of available funds on the account, as well as to monitor the performance of your virtual server - you can check if MetaTrader is running and Windows was not restarted to install updates;
- _Show: session closing time (time left) - closing_. This information will be especially useful for intraday traders, who prefer not to leave trades till the next day - this will help to protect your account from swaps and gaps.

I hope this information will help you to improve your trading performance.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/5111](https://www.mql5.com/ru/articles/5111)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/5111.zip "Download all attachments in the single ZIP archive")

[SETfiles.zip](https://www.mql5.com/en/articles/download/5111/setfiles.zip "Download SETfiles.zip")(415.11 KB)

[TESTfiles\_cci.zip](https://www.mql5.com/en/articles/download/5111/testfiles_cci.zip "Download TESTfiles_cci.zip")(3857.98 KB)

[TESTfiles\_plain\_1.zip](https://www.mql5.com/en/articles/download/5111/testfiles_plain_1.zip "Download TESTfiles_plain_1.zip")(13337.29 KB)

[TESTfiles\_plain\_2.zip](https://www.mql5.com/en/articles/download/5111/testfiles_plain_2.zip "Download TESTfiles_plain_2.zip")(2504.23 KB)

[TESTfiles\_plain\_3.zip](https://www.mql5.com/en/articles/download/5111/testfiles_plain_3.zip "Download TESTfiles_plain_3.zip")(13524.94 KB)

[RevertEA.zip](https://www.mql5.com/en/articles/download/5111/revertea.zip "Download RevertEA.zip")(185.58 KB)

[RevertManualEA.zip](https://www.mql5.com/en/articles/download/5111/revertmanualea.zip "Download RevertManualEA.zip")(103.05 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/295660)**
(8)


![Roman Klymenko](https://c.mql5.com/avatar/avatar_na2.png)

**[Roman Klymenko](https://www.mql5.com/en/users/needtome)**
\|
24 Oct 2018 at 14:12

**artem2222:**

Started the Expert Advisor from the previous article. I looked at the result: every 15 minutes the Expert Advisor opens two trades: buy and sell. In general, it does not behave as described in the article. How is this possible?

Please tell me what broker, on what instrument, whether you use SET-file from the ones attached to the article or you have set it up yourself. If you did it yourself, it is desirable to attach the SET-file as well. I will test the work of the [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4") at your broker


![Sebastien Talukder](https://c.mql5.com/avatar/avatar_na2.png)

**[Sebastien Talukder](https://www.mql5.com/en/users/talu)**
\|
21 Dec 2018 at 09:52

Interesting articles thanks for sharing.

What if you keep your loosing trades open instead of having [stop losses](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders")? and your EA close everything once break even reached? I'm not sure but I imagine that it gives you more margin and therefor more attempts.


![talha8877](https://c.mql5.com/avatar/avatar_na2.png)

**[talha8877](https://www.mql5.com/en/users/talha8877)**
\|
27 Dec 2018 at 03:10

Set file: GPBUSD\_15M.set

Broker: ICMARKETS - MT5

Error: Failed market buy. Unsupported filling mode

Why?

![Joao Luiz Sa Marchioro](https://c.mql5.com/avatar/2017/11/5A1389EC-103A.JPG)

**[Joao Luiz Sa Marchioro](https://www.mql5.com/en/users/joaoluiz_sa)**
\|
4 Feb 2019 at 23:01

Very good both articles, give a good light of the simple [functioning of](https://www.mql5.com/en/docs/constants/namedconstants/compilemacros "MQL5 documentation: Predefined macro replacements") the assets by demoralising the need for large complex trading systems. Congratulations. I will study the two articles in more detail.


![Shen Bao Qiu](https://c.mql5.com/avatar/avatar_na2.png)

**[Shen Bao Qiu](https://www.mql5.com/en/users/625507665)**
\|
12 May 2023 at 07:59

You can get an EA with a steady loss and then follow it in reverse.


![Reversing: Formalizing the entry point and developing a manual trading algorithm](https://c.mql5.com/2/34/Reverse_trade.png)[Reversing: Formalizing the entry point and developing a manual trading algorithm](https://www.mql5.com/en/articles/5268)

This is the last article within the series devoted to the Reversing trading strategy. Here we will try to solve the problem, which caused the testing results instability in previous articles. We will also develop and test our own algorithm for manual trading in any market using the reversing strategy.

![Reversal patterns: Testing the Head and Shoulders pattern](https://c.mql5.com/2/34/5358_avatar.png)[Reversal patterns: Testing the Head and Shoulders pattern](https://www.mql5.com/en/articles/5358)

This article is a follow-up to the previous one called "Reversal patterns: Testing the Double top/bottom pattern". Now we will have a look at another well-known reversal pattern called Head and Shoulders, compare the trading efficiency of the two patterns and make an attempt to combine them into a single trading system.

![DIY multi-threaded asynchronous MQL5 WebRequest](https://c.mql5.com/2/34/Multi_WebRequest_MQL5.png)[DIY multi-threaded asynchronous MQL5 WebRequest](https://www.mql5.com/en/articles/5337)

The article describes the library allowing you to increase the efficiency of working with HTTP requests in MQL5. Execution of WebRequest in non-blocking mode is implemented in additional threads that use auxiliary charts and Expert Advisors, exchanging custom events and reading shared resources. The source codes are applied as well.

![Reversal patterns: Testing the Double top/bottom pattern](https://c.mql5.com/2/34/double_top.png)[Reversal patterns: Testing the Double top/bottom pattern](https://www.mql5.com/en/articles/5319)

Traders often look for trend reversal points since the price has the greatest potential for movement at the very beginning of a newly formed trend. Consequently, various reversal patterns are considered in the technical analysis. The Double top/bottom is one of the most well-known and frequently used ones. The article proposes the method of the pattern programmatic detection. It also tests the pattern's profitability on history data.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=zxnponfgftzbmxkmamhlfbclflsvvntz&ssn=1769186603175002931&ssn_dr=0&ssn_sr=0&fv_date=1769186603&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F5111&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Reversing%3A%20Reducing%20maximum%20drawdown%20and%20testing%20other%20markets%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918660363673963&fz_uniq=5070509691311691563&sv=2552)

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