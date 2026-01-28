---
title: Trailing stop in trading
url: https://www.mql5.com/en/articles/14167
categories: Trading, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:29:04.880655
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/14167&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082894109250687215)

MetaTrader 5 / Examples


### Introduction

The main purpose of a trailing stop is to obtain a guaranteed profit with minimal risk. The essence of how a trailing stop works is very simple. The stop loss gradually moves behind the price if it moves in the trader's favor. It remains in place if the price moves in the opposite direction.

Schematically, a trailing stop can be represented as follows. Let's assume that a trader has opened a Buy position on a rising trend. When the price moves up, the stop loss will automatically move behind the price. When the trend changes its direction, the trader can take profit.

> ![](https://c.mql5.com/2/67/1__22.png)

There are a large number of trailing stop options. The [MetaTrader](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading platform has trailing stop options of its own.

> ![](https://c.mql5.com/2/67/2__22.png)

In addition, the trailing stop can be automated and included in the code of trading Expert Advisors. For example, this can be done with [open position support classes](https://www.mql5.com/en/docs/standardlibrary/expertclasses).

The efficiency of a trailing stop largely depends on price volatility and the selection of the stop loss level. A variety of approaches can be used to set a stop loss. For example, if there is a clearly visible trend, we can use the values of price highs or lows. In addition, trailing stop parameters can be determined using technical indicators. Such an approach is described in the article " [How to create your own Trailing Stop](https://www.mql5.com/en/articles/134)". In this article, we will look at the possibility of constructing a trailing stop based on statistical data.

### Simple trailing stop

The trading strategy and trailing stop are independent of each other. The main difference between them is that the strategies open and close positions. A trailing stop is intended only for closing positions.

First, let's look at the restrictions that are placed on the stop loss.

The upper limit of the stop loss is set in the symbol properties - the minimum difference between the closing price of the position and the set stop loss cannot be less than the minimum offset in points from the current closing price of the position.

```
SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)
```

As for the lower limit, we will have to calculate it ourselves. The first requirement for the minimum stop loss is that it must be in the breakeven area. At first glance, everything is simple - the minimum stop loss should be no worse than the opening price of the position. Right? Wrong.

When opening a position, a commission may be charged. In addition, a position may accumulate swap during its existence. These additional costs should be taken into account when calculating the minimum possible stop loss. To do this, we need to find out the cost of one point in the deposit currency.

```
PointValue=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE)*SymbolInfoDouble(_Symbol,SYMBOL_POINT)/SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE)
```

Now we can calculate how many points the price must move to compensate for the commission and swap. In addition, we should not forget about slippage. During trading operations, slippage can work both for the worse and for the better. But we will be careful and assume that slippage will always work against the trader.

Suppose that **_PriceOpen_** is a position open price, while **_Lot_** is its volume. Then the minimum stop loss can be calculated using the following equation:

> ![](https://c.mql5.com/2/67/0__185.png)

The upper signs are used for Buy positions, and the lower ones are for Sell.

The minimum stop loss corresponds to the guaranteed break-even level of the position. In other words, if the position is closed at the minimum stop loss, then its total profit (taking into account the current profit, swap and commission) will be non-negative.

Now we can formulate the basic rules of a trailing stop:

- a new stop loss should be between the minimum and maximum levels;
- a new stop loss must be better than the previous one.

If these requirements are met, the stop loss can be upgraded. But, before upgrading, it is necessary to carry out one more check - the old stop loss must be outside the freeze level (more information about this - " [The checks a trading robot must pass before publication in the Market?](https://www.mql5.com/en/articles/2555)").

We can use different approaches to determine a new stop loss. I will use the levels that are described in the article " [Trader-friendly stop loss and take profit](https://www.mql5.com/en/articles/13737)".

However, I will make small changes to calculate the optimal stop loss. The trailing stop should be triggered on the current bar, so the position holding time will always be equal to 1 bar. In addition, statistics on price movement will be collected on a randomly selected timeframe. This approach makes it possible to adjust the sensitivity of the trailing stop.

A few words about take profit. There may be several different options here.

Let's assume that a trader uses a trading strategy with a given take profit. Then the take profit can remain fixed, and the trailing stop only modifies the stop loss. In this case, a trailing stop takes the position to breakeven.

If the trading strategy does not provide for setting a take profit, or it is possible to use a trailing take profit, then a trailing stop can monitor the levels of both stop loss and take profit. In this case, the profitability of trading may change, since some positions may be closed at more favorable prices.

In addition, it is possible that a trailing stop is used in tick-by-tick mode. Then setting a take profit does not make any sense - the price will never catch up with the take profit. The only possible option for closing a position is only by stop loss.

Let's check how a simple trailing stop works with and without take profit. At the same time, we will introduce an additional condition - setting and modifying the take profit becomes possible only after the trailing stop moves the stop loss to the breakeven area.

I will use EURUSD H1. Test period - from January 1 to December 31, 2023. Trailing stop timeframe M1. The direction and volume of positions are determined randomly. Stop loss and take profit positions are not set.

The balance graph looks as follows.

> ![](https://c.mql5.com/2/79/3__22.png)

Some positions were closed forcibly at the end of the test - this is a drawback of the strategy used. But, the graph gives an idea of how a trailing stop might work. Let's look at the test results.

| PERIOD\_M1 | Total Net Profit | Gross Profit | Gross Loss |
| --- | --- | --- | --- |
| UseTakeProfit=false | 4 758.48 | 10 689.84 | -5 931.36 |
| UseTakeProfit=true | 5 483.94 | 11 297.68 | -5 813.74 |

As we can see, the use of take profit can affect the profitability of trading. Now, let's try to run the test with the same conditions, but set the trailing stop timeframe to M15.

| PERIOD\_M15 | Total Net Profit | Gross Profit | Gross Loss |
| --- | --- | --- | --- |
| UseTakeProfit=false | 16 371.11 | 33 435.31 | -17 064.20 |
| UseTakeProfit=true | 17 038.63 | 34 042.13 | -17 003.50 |

Changing the timeframe had a significant impact on the test results. This is due to changes in the optimal stop loss and take profit levels. We can conclude that small trailing stop timeframes are suitable for scalping, while large timeframes are suitable for trend strategies.

### Moral expectation and trailing stop

[Moral expectation](https://en.wikipedia.org/wiki/Expected_utility_hypothesis "https://en.wikipedia.org/wiki/Expected_utility_hypothesis") is a risk assessment that was first introduced by [Daniel Bernoulli](https://en.wikipedia.org/wiki/Daniel_Bernoulli "https://en.wikipedia.org/wiki/Daniel_Bernoulli") in 1732. Moral expectation allows us to evaluate the usefulness of the game. At the same time, it takes into account the player’s capital, possible wins and losses, as well as their probabilities.

Let's look at the trading process a little differently. Think of each position as a separate player. The position then has a capital - the total profit of the position. This capital may increase if the position is closed at take profit. It may also decrease if the position is closed by stop loss.

Suppose that **_ProfitPoint_** is a position profit in points, while **_p_** is a probability that the position will be closed at take profit. Then the moral expectation of the position can be found as follows:

> ![](https://c.mql5.com/2/67/0__186.png)

> ![](https://c.mql5.com/2/67/0__187.png)

Obviously, the trader needs to select such stop loss and take profit, at which the moral expectation becomes maximum. Let's look at the features of supporting a position with the help of moral expectation.

Stop loss must be strictly less than profit. Only in this case can moral expectation be positive. But this condition is not sufficient. Let's assume that we have found the optimal stop loss and take profit. These levels are ideal for an ideal market. The real market can bring surprises.

Let's assume that the trailing stop has set an optimal stop loss, but due to slippage the position may close at a worse price. In this case, the actual stop loss may be greater than the profit of the position. This cannot be allowed. Thus, the stop loss is subject to a restriction:

> ![](https://c.mql5.com/2/67/0__188.png)

When calculating moral expectation, the take profit value is used. But its use is not mandatory. The features of the trading strategy are more important in this matter.

Such results are shown by a trailing stop based on moral expectation.

| PERIOD\_M1 | Total Net Profit | Gross Profit | Gross Loss |
| --- | --- | --- | --- |
| UseTakeProfit=false | 4 482.35 | 8 175.37 | -3 693.02 |
| UseTakeProfit=true | 4 747.94 | 8 434.11 | -3 686.17 |

The results turned out slightly worse than the ones of a simple trailing stop. This is due to the fact that when positions are supported by moral expectation, the position is transferred to breakeven earlier than with a simple trailing stop. Therefore, such a trailing stop is best used in scalping strategies.

Another serious disadvantage is the large number of operations required. Calculations can be optimized. But even in this case, the trailing stop timeframe by moral expectation should be small.

### Trailing stop for multiple positions

So far, we have applied a trailing stop to each position separately. Is it possible to apply it to several positions at once? Let's consider this possibility. We will assume that positions can be of different types and with different volumes at the same time.

First of all, we need to determine which type of positions has an advantage. To do this, we will find the sum of the volumes of Buy positions and subtract from it the sum of the volumes of Sell positions.

> ![](https://c.mql5.com/2/67/0__189.png)

The result will show which type of position is stronger. A positive number indicates that the Buy position is stronger, a negative number indicates that the Sell position has the upper hand. If the result is zero, then the positions have equal strength.

Next, we need to find the minimum possible closing price for each position. They are calculated in the same way as the minimum stop loss, but do not take slippage into account.

> ![](https://c.mql5.com/2/67/0__190.png)

Now we need to convert these closing prices into Ask/Bid prices and find their weighted average. Position volumes will act as weights.

> ![](https://c.mql5.com/2/67/0__191.png)

> ![](https://c.mql5.com/2/67/0__192.png)

It is more convenient to reduce these two values to one optimal Bid value, which will later be required to track positions.

> ![](https://c.mql5.com/2/67/0__193.png)

Now we have three options:

- If the volumes of Buy and Sell positions are equal to each other, then the optimal Bid value will correspond to the minimum loss. In other words, the trader can close all positions if the actual Bid price matches the optimal one.
- If the volumes of Buy positions are larger, then you need to increase BidOpt by the slippage amount. This will be the minimum stop loss for all positions.
- If the volumes of Sell positions are larger, then the BidOpt value should be reduced by the slippage amount.

Positions are maintained in the same way as with a simple trailing stop. New stop loss levels are calculated and adjusted to the price. The only difference is that this stop loss is virtual. Therefore, we will have to track the price change on each tick.

I have implemented support for several positions based on the principle of a simple trailing stop. This is how the balance graph changes when it is applied.

> ![](https://c.mql5.com/2/79/4__19.png)

Note that the support of all positions and each position individually does not bring conflicts. For example, a simple trailing stop + trailing stop of all positions provide the following results.

|  | Total Net Profit | Gross Profit | Gross Loss |
| --- | --- | --- | --- |
| PERIOD\_M1 | 4 907.90 | 10 425.52 | -5 517.62 |
| PERIOD\_M15 | 16 524.44 | 32 304.01 | -15 779.57 |

Overall, a trailing stop is a useful tool. However, a trader should remember that its use does not guarantee break-even trading. It will not improve an insufficiently developed trading strategy.

### Conclusion

The following program was used when writing this article.

| Name | Type | Features |
| --- | --- | --- |
| Trailing stop | EA | - **_TFTS_** \- trailing stop timeframe<br>- **_Seed_** \- seed number for a series of random numbers. If not 0, then the sequence of positions and their volumes will be repeated<br>- **_Slippage_** \- slippage<br>- **_TypeTS_** \- trailing stop type selection<br>- **_UseTakeProfit_** \- use take profit<br>- **_MultiTS_** \- trailing stop of all open positions |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14167](https://www.mql5.com/ru/articles/14167)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14167.zip "Download all attachments in the single ZIP archive")

[Trailing\_stop.mq5](https://www.mql5.com/en/articles/download/14167/trailing_stop.mq5 "Download Trailing_stop.mq5")(32.67 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/467692)**

![Bill Williams Strategy with and without other indicators and predictions](https://c.mql5.com/2/79/Bill_Williams_Strategy_with_and_without_other_Indicators_and_Predictions__LOGO.png)[Bill Williams Strategy with and without other indicators and predictions](https://www.mql5.com/en/articles/14975)

In this article, we will take a look to one the famous strategies of Bill Williams, and discuss it, and try to improve the strategy with other indicators and with predictions.

![Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://c.mql5.com/2/78/Modified_Grid-Hedge_EA_in_MQL5_yPart_IVq____LOGO.png)[Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)

In this fourth part, we revisit the Simple Hedge and Simple Grid Expert Advisors (EAs) developed earlier. Our focus shifts to refining the Simple Grid EA through mathematical analysis and a brute force approach, aiming for optimal strategy usage. This article delves deep into the mathematical optimization of the strategy, setting the stage for future exploration of coding-based optimization in later installments.

![Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://c.mql5.com/2/79/Integrate_Your_Own_LLM_into_EA__Part_3_-_Training_Your_Own_LLM_with_CPU_____LOGO.png)[Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://www.mql5.com/en/articles/13920)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Neural networks made easy (Part 70): Closed-Form Policy Improvement Operators (CFPI)](https://c.mql5.com/2/63/Neural_Networks_Made_Easy_uPart_70p_CFPI_LOGO.png)[Neural networks made easy (Part 70): Closed-Form Policy Improvement Operators (CFPI)](https://www.mql5.com/en/articles/13982)

In this article, we will get acquainted with an algorithm that uses closed-form policy improvement operators to optimize Agent actions in offline mode.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/14167&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082894109250687215)

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