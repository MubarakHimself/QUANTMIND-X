---
title: Automated grid trading using limit orders on Moscow Exchange (MOEX)
url: https://www.mql5.com/en/articles/10672
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:31:35.972827
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/10672&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082925587065999750)

MetaTrader 5 / Trading


### Introduction

The article thoroughly describes using the grid trading approach while developing the EA in MQL5 language of trading strategies. The EA is to follow a grid strategy while trading on MOEX using MetaTrader 5 terminal. The EA involves closing positions by stop loss and take profit, as well as removing pending orders in case of certain market conditions.

### 1\. Grid trading

The article will consider the application of the grid EA for automating futures contracts trading on MOEX. The EA allows placing orders at specified intervals within a certain price range.

During grid trading, orders are placed above and below a set price, creating a grid of orders with progressively higher and lower prices.

For example, you can place buy orders every RUB 100 below the market price of a futures contract for ordinary shares of VTB Bank (PJSC) — VTBR-6.22 (VBM 2), as well as place sell orders every RUB 100 above its market price. This allows benefiting from trading under various conditions.

Grid trading is perfect for volatile and "flat" markets when prices fluctuate within a predetermined range. This way of trading allows making profit in case of small price changes. The more frequent grid levels, the more often transactions are made. However, this leads to higher costs since the profit of each of the orders becomes lower.

Thus, it is necessary to find a compromise between making a small profit while making a lot of transactions and a strategy with fewer orders but larger profit on each of them.

You can customize the grid settings by setting the upper/lower bounds and the grid frequency. After creating the grid, the EA automatically places buy and sell orders at pre-calculated prices.

Let's see how it works. Suppose that you expect VBM 2 futures contract price to fluctuate between RUB 1,600 to 2,800 in the next 7 days. In this case, you can set up grid trading to trade within the predicted range.

In the grid trading panel, you can set the strategy parameters, including:

- upper and lower limits of the price range;

- number of orders to be placed within the specified price range;

- step between each buy and sell limit order.


![](https://c.mql5.com/2/47/3.1__2.png)

Fig. 1.  Placing an order grid

In this case, as VBM 2 futures price decreases down to RUB 1,600, the grid EA will accumulate positions at a lower price than the market one. When the price starts recovering, the EA will sell at a higher price than the market one. The strategy attempts to capitalize on a price direction reversal.

### 2\. Asset price movement types on financial markets that are most preferable for a trading system

The entry point in the trading strategy is tied to the symbol market movement. This means you can spread the grid (perform a deal entry) at any moment. As you might know, the markets show flag movement most of the time. This EA has been specifically created for such market conditions adhering to the so-called paradigm — buy low, sell high.

The most preferable flat type for grid trading is the one when the price moves in a certain range without a clearly defined direction. The breakdown of the boundaries, as a rule, means the end of the flat and the beginning of a new trend.

A wide flat is formed when the price is squeezed between strong support and resistance levels. Neither bulls, nor bears are strong enough to win, so no new trend is initiated.  All these definitions are strictly individual in terms of symbols and are considered in the context of movements of a particular market symbol.

If we consider this trading approach generally, then, in case of VBM 2 futures contract traded on MOEX for about a month, the total profit exceeds the general loss even in the presence of a unidirectional (general final) movement from the lower border of 1,600 to the upper 2,800, as well as in the presence of price fluctuations (rollbacks from the main movement) and, as a result, order activations and reset limit buy and sell orders.

Price fluctuations of a traded symbol between price levels (inside the range) allow us to increase profits based on trading results obtained using this grid approach within the selected time period.

The grid EA applies the same value for traded contracts/lots in each grid order, so you need to plan your money management based on this trading approach.

Trading a rollback from flat borders assumes continued flat. When using this trading method, it is important to take into account the absence of important news able to provoke a strong movement. In addition, stop losses should be placed outside the range for safety reasons.

### 3\. Features and benefits of grid trading

After setting the grid, the EA does not add new external orders into it till the current grid is closed with profit or loss.

Recommendations based on the trading system: Select a symbol featuring enough volumes and clear movements. Besides, it would be good to exclude position rollovers, so that its market depth is not empty. Also, avoid news and unidirectional movements without rollbacks.

This trading system involves trading from levels using averaging (Nayman E. Master-Trading: Secret Materials, p. 134), when you perform the same operation as the previous one (buying while in a long position or selling while in a short) at a better price.

The main disadvantage of averaging is that you do not know in advance how much the market will go against you. At the same time, averaging requires investing additional margin funds increasing the risk for your position. Most novice traders make a common mistake - in pursuit of high profits, they "overload" their account bringing the amount of leverage to absolutely fantastic values, sometimes even exposing all of their available funds (Nayman E. "Master-Trading: Secret Materials", p. 135).

The main feature of this trading approach is the ability to set the values of the minimum and maximum levels to ensure trading with limit orders within the specified price range of a traded symbol.

By placing stop losses for the upper and lower levels, you can limit the losses on the trading account.

When a stop loss is triggered, the terminal informs of the level breakthrough by the price and its possible movement towards the breakthrough. As a result, it may be necessary to re-analyze new price levels and set their values in the external parameters of the grid trading EA (attached below).

### 4\. Implementing grid trading

**4.1.  Grid trading mechanism**

Grid trading stages:

1. Initial structure - the EA is launched on a selected trading symbol

2. A position is opened when the first limit order of the order grid you placed is triggered

3. The grid is updated when subsequent grid orders (either increasing the initial market position or reducing it) are triggered

4. Stop losses, take profits, trading time limits of the current market position, exit by profit % are at your discretion

5. Control over trading by a grid EA - closing the current profitable grid trading session, re-setting it at the new relevant highs and lows of the traded symbol


This grid strategy starts without an initial market position. The initial position will be activated when the market moves beyond the nearest pending order price after the initial launch.

**Example**

Suppose that you set your strategy parameters as follows for 04.01.2022:

- Contract: futures VBM 2
- Lower price limit: RUB 1,600
- Upper price limit: RUB 2,800
- Number of elements in the grid: 10
- Mode: Equidistant

Suppose that you have set your strategy parameters as follows:

Initial neutral grid buy orders will be placed below the current market price. Meanwhile, sell orders will be placed above the current market price. After triggering the nearest buy or sell limit order, the so-called process will occur - let's call it the trading grid activation. 9 orders will already be used in trading further on.

**The price distribution will be as follows by order types:**

- **Buy Limit:** RUB 1,600, RUB 1,733, RUB 1,866, RUB 1,999
- **Sell Limit:** RUB 2,135, RUB 2,268, RUB 2,401, RUB 2,534, RUB 2,667, RUB 2,800

(you can also empirically consider the optimal option of arranging orders in grids depending on the ATR indicator percentage on D1 to get the optimal profit with a moderate risk).

**The summary of all executed orders.**

Each transaction consists of a pair of corresponding buy and sell orders, transaction type – **FILO** (first in, last out). The profit can be calculated based on each pair of matched buy and sell orders.

The range from the lower border (LOW\_PRICE\_BuyLim) to the upper one (HIGH\_PRICE\_SellLim) is divided by the specified number of orders (Number\_of\_elements\_in\_the\_grid) with equal intervals. It is advisable to independently look at and select the range and number of orders in the grid (beginning from 3), so that the total width between orders is higher than the symbol spread and the price values of the borders are outside the current price.

Moreover, the upper border of the price range HIGH\_PRICE\_SellLim should exceed the lower border of the range LOW\_PRICE\_BuyLim.

**4.2. Updating the grid**

The grid is updated each time when one of the price levels is reached, i.e. when a limit order is executed. The last executed order always remains empty, buy and sell limit orders are then executed at set prices as illustrated in the example below.

For example, in this case, after the activation of the grid strategy by the first limit order to sell and subsequent upward movement of the VBM2 futures contract, as well as after the activation of the next limit order to sell, the lower part of the grid (number of Buy Limit orders)  is increased by 1 BuyLimit order - placed 1 step higher (closer to the price) of the previous BuyLimit order of the grid.

In particular, after activating the grid trading, price upward movement and activation of the first sell limit order since the grid activation, the limit prices for each grid level will be as follows:

| Price, RUB | Order/position direction and type |
| --- | --- |
| 2,534 | Selling by a limit order |
| 2,401 | Selling by a limit order |
| 2201.5 | Market position with the volume of two first sell limit orders |
| 2,132 | Buying with a limit order |
| 1,999 | Buying with a limit order |
| 1,866 | Buying with a limit order |
| 1,733 | Buying with a limit order |
| 1,600 | Buying with a limit order |

This is displayed in the following image:

![](https://c.mql5.com/2/47/3.2.gif)

Fig. 2. Updating the order grid after its activation and triggering the next limit orders

Thus, updating grid orders continues.

Grid parameters:

```
//+------------------------------------------------------------------+
//|                          EXTERNAL GRID VARIABLES
//+------------------------------------------------------------------+
input int Volume = 1;                          //Contract/lot volume
input double HIGH_PRICE_SellLim = 2800.00;     //HIGH_PRICE_SellLim (the upper price of the last SellLim grid order)
input double LOW_PRICE_BuyLim  = 1600.00;      //LOW_PRICE_BuyLim   (the lower price of the last BuyLim grid order) for BuyLim
input double HIGH_PRICE_SL  = 100.00;          // HIGH_PRICE_SL: SL in points from HIGH_PRICE_SellLim (the upper stop loss price) for SellLim
input double LOW_PRICE_SL = 100.00;            // LOW_PRICE_SL:  SL in points from LOW_PRICE_BuyLim (lower stop loss price)  for BuyLim
input int Pending_order_period  = 12;          // Pending_order_period: limit order setting time in months
input int Number_of_elements_in_the_grid = 10; // Number of elements in the grid (number of limit orders)
input double TakeProfit  = 0;                  // TakeProfit from the setting price in order pips
input double Profit_percentage = 0;            // Profit_percentage - grid closure % in profit
                                               // works if > "0"
input  bool Continued_trading = false;         // Continued_trading - whether to continue trading after exiting by the grid closure % with profit

input int    Time_to_restrict_trade  = 0;      // Time_to_restrict_trade - setting the expiration time (in days) for a position with profit
                                               // (exiting the market upon the period end in days)

input int Magic       = 10;                    // Magic Number
```

The names of external variables and their decoding are made as detailed as possible, so they do not require extra comments.

Also, we set the variable

```
HIGH_PRICE_SellLim
```

not lower than the upper and

```
LOW_PRICE_BuyLim
```

not lower than the lower limit of values defined (set) by MOEX:

[https://www.moex.com/en/contract.aspx?code=VTBR-6.22](https://www.mql5.com/go?link=https://www.moex.com/en/contract.aspx?code=VTBR-6.22 "https://www.moex.com/en/contract.aspx?code=VTBR-6.22")

All necessary data is displayed in the MOEX symbol parameters (Fig. 3)

![](https://c.mql5.com/2/47/3.png)

Fig. 3. Symbol parameters - Futures contract for ordinary shares of VTB Bank (PJSC) VTBR-6.22

The trading approach is shown in Fig. 4.2.

As a result, after launching the grid trading EA described here, you should get a detailed information about the robot trading on your selected symbol (Fig. 4.1 and 4.2):

![](https://c.mql5.com/2/47/4.1.png)

Fig. 4.1. Using the grid EA on VTBR-6.22

If stop loss and take profit have other values than zero, their levels will also be displayed in the trading terminal in addition to active buy and sell limit orders:

![](https://c.mql5.com/2/47/5.2.1_15.06.2022.png)

Fig. 4.2. Using the grid EA on VTBR-6.22 with non-zero stop loss and take profit values

As a result, we have a grid of limit orders, which are re-set during activations inside the minimum and maximum price range.

Till the traded symbol price is located inside the price range, the grid EA accumulates profit on the trading account. Even if we consider an obvious drawback of this type of grid trading, namely a strong movement without rollbacks in any direction instead of the expected price movement in a range, I can say the following:

- First, such movements are not "frequent",
- Second, by the time the price leaves the correctly selected range and the number of orders, there will already be enough profit on the account to compensate for the current loss,
- Third, till the price fluctuates (moves) between the maximum and minimum values of the price range making both buy and sell deals, the trading account is in profit.

### Conclusion

Just like the entire grid-based trading approach, the grid EA considered here requires control over whether the price is within the specified range. When the price approaches one of its borders, it is possible to stop grid trading with limit orders and start it on a new range with new volumes.

The grid can be created to profit from trends (this option will be presented in the next part of the article) or ranges (described in the current article). The grid-based trading considers trading using the EA based on constantly reset limit orders, i.e. as long as the price continues to fluctuate sideways executing both sell and buy orders.

The risks of unidirectional price movement are controlled by setting stop loss levels for limit orders. As a result, the EA provided here allows us to avoid a sudden market drawdown (growth) generally followed by corrective price movement. In other words, if we choose the price range, contract values, number of grid orders, as well as take profit and stop loss correctly, we will be able to avoid drawdowns eliminating the need to constantly monitor the market.

**Note**:

The grid strategy discussed here should not be considered as financial or investment advice. Use grid trading at your discretion and at your own risk, which implies control and reasonable approach to trading based on your own financial capabilities.

**References:**

Nayman, E., (2002). _Master-Trading: Secret Materials._ Moscow, Alpina Publisher (in Russian).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/10672](https://www.mql5.com/ru/articles/10672)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10672.zip "Download all attachments in the single ZIP archive")

[GRID\_TRADING\_v.01.mq5](https://www.mql5.com/en/articles/download/10672/grid_trading_v.01.mq5 "Download GRID_TRADING_v.01.mq5")(112.73 KB)

[report\_and\_set\_files.zip](https://www.mql5.com/en/articles/download/10672/report_and_set_files.zip "Download report_and_set_files.zip")(54.59 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex spread trading using seasonality](https://www.mql5.com/en/articles/14035)
- [Benefiting from Forex market seasonality](https://www.mql5.com/en/articles/12996)
- [Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10671)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/429487)**
(30)


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
28 Jan 2025 at 09:30

Sber futures

[![](https://c.mql5.com/3/454/1988952083524__1.png)](https://c.mql5.com/3/454/1988952083524.png "https://c.mql5.com/3/454/1988952083524.png")

Here's a share.

[![](https://c.mql5.com/3/454/1904995798734__1.png)](https://c.mql5.com/3/454/1904995798734.png "https://c.mql5.com/3/454/1904995798734.png")

there are 10 shares in 1 lot - in essence, 100 shares in Sber is 28,000 rubles.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
28 Jan 2025 at 11:04

futures 1 lot 100 shares

shares 100 shares about 28 000 r.

total leverage: 28,000 p / GO (5,500.00 p) which is about **5th leverage** on sber **.**

![JRandomTrader](https://c.mql5.com/avatar/avatar_na2.png)

**[JRandomTrader](https://www.mql5.com/en/users/jrandomtrader)**
\|
28 Jan 2025 at 12:30

**Roman Shiredchenko [#](https://www.mql5.com/ru/forum/427408/page3#comment_55754286):**

futures 1 lot 100 shares

shares 100 shares about 28 000 r.

total leverage: 28,000 p / GO (5,500.00 p) which is about **5th leverage** on sber **.**

The main thing to keep in mind is that futures and stocks move differently. Particularly the issue of divs and divgap in the first place.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
28 Jan 2025 at 14:03

**JRandomTrader [#](https://www.mql5.com/ru/forum/427408/page3#comment_55755274):**

The main thing to remember is that futures and stocks move differently. In particular, the issue of divs and divgap comes first.

there is also an approach of arbitrage type in a personal message. There on yu tuba clips and like do not need a quick connection in the level of hft.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
12 Nov 2025 at 11:44

I plan to continue trading with the grid on [moex](https://www.mql5.com/en/articles/1284 "Article: Fundamentals of exchange pricing on the example of the derivatives section of the Moscow Exchange ") MT5 data for familiarisation I will also provide here...

I do not exclude possibly with the improvement of MT5 trading robots!!!!

![Data Science and Machine Learning (Part 06): Gradient Descent](https://c.mql5.com/2/47/data_science_articles_series__1.png)[Data Science and Machine Learning (Part 06): Gradient Descent](https://www.mql5.com/en/articles/11200)

The gradient descent plays a significant role in training neural networks and many machine learning algorithms. It is a quick and intelligent algorithm despite its impressive work it is still misunderstood by a lot of data scientists let's see what it is all about.

![Neural networks made easy (Part 16): Practical use of clustering](https://c.mql5.com/2/48/Neural_networks_made_easy_016.png)[Neural networks made easy (Part 16): Practical use of clustering](https://www.mql5.com/en/articles/10943)

In the previous article, we have created a class for data clustering. In this article, I want to share variants of the possible application of obtained results in solving practical trading tasks.

![Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://c.mql5.com/2/47/development.png)[Developing a trading Expert Advisor from scratch (Part 17): Accessing data on the web (III)](https://www.mql5.com/en/articles/10447)

In this article we continue considering how to obtain data from the web and to use it in an Expert Advisor. This time we will proceed to developing an alternative system.

![Developing a trading Expert Advisor from scratch (Part 16): Accessing data on the web (II)](https://c.mql5.com/2/46/development__7.png)[Developing a trading Expert Advisor from scratch (Part 16): Accessing data on the web (II)](https://www.mql5.com/en/articles/10442)

Knowing how to input data from the Web into an Expert Advisor is not so obvious. It is not so easy to do without understanding all the possibilities offered by MetaTrader 5.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/10672&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082925587065999750)

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