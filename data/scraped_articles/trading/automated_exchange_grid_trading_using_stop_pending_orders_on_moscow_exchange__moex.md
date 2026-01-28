---
title: Automated exchange grid trading using stop pending orders on Moscow Exchange (MOEX)
url: https://www.mql5.com/en/articles/10671
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:30:47.268331
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/10671&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082914669259133269)

MetaTrader 5 / Trading


### Introduction

The [previous](https://www.mql5.com/en/articles/10672) article delved into a trading approach based on limit pending orders for trading on the Moscow Exchange. In this article, we will focus on using a grid trading approach on stop pending orders using stop loss and/or take profit.

The grid trading method is not described in classic books on trading, perhaps due to its relatively recent appearance.

When trading in the market, one of the simplest strategies is a grid of orders designed to "catch" the market price. Trading systems applying this method usually do not pursue the goal of finding exact entry and exit points - the task is to deploy a grid of positions with deals opening automatically.

**1\. Grid with stop pending orders (Buy Stop, Sell Stop)**

This method is very common in the arsenal of novice traders as it does not require special knowledge regarding the foreign exchange market, and even such common things as technical or fundamental analysis. However, a successful grid strategy requires a rough knowledge of the main trend in the selected time period - up, down, or sideways price movement with a periodic return to the average value.

The general principles of constructing a grid in the market are quite simple:

- Orders are lined up one after another at an equal distance in points

- A trade direction is predetermined

- The grid is set to a specific range


The grid is characterized by the following parameters:

- grid width
- grid step
- take profit
- stop loss


The grid width is the area covered by the placed orders. Grid step is the distance between orders. Grid width and step are calculated in points. Thus, we have reached the definition of the grid method of trading. A trading method, in which the entry into the market is carried out using many orders usually located at the same distance from each other and on both sides of the current price is called a grid.

Whatever direction the market price goes, it will still pass through the grid of positions. Profitable trades can accumulate up to a certain amount, but they can also be closed as soon as the price turns the next order placed on the grid into a market position. Placing next orders in the grid (for example, when the price moves up and triggers buy stop orders, as well as accumulates market position volumes with subsequent placing sell market orders closer to the price (updating the grid)) with subsequent small price downward rollbacks triggering sell stop orders performs the role of the so-called partial closing of the position, which is shown in Fig. 1.

![placing and executing orders](https://c.mql5.com/2/54/0_3wqji6tkisi__j_k37fjp1b.png)

Fig. 1. Placing an order grid and triggering the orders

At the same time, the balance and equity graphs in the strategy tester looks like this (Fig. 2)

![balance graph](https://c.mql5.com/2/54/1_qb6p4ic22dc_z_fe8afx5p_k8rotr.png)

Fig. 2. Strategy tester graph

How could the idea of such an approach to trading come into existence? Presumably, the trader's thought worked in the following direction: first, the trader comprehends the process of opening one trade for a symbol and closing it either by take profit (TP) or stop loss (SL). Over time, the trader learns that it is possible to enter the market in a more sophisticated manner, namely, stepwise.

Entering the market in stages, the trader divides the entire volume of the position into several parts and determines the price levels to open trades at. These levels can be both above and below the current price, and then these entries will be called either "adding positions on a rollback" or "adding positions along a trend". Most traders like to close positions by TP, but it often happens that the price does not reach a predetermined level, at which traders have set their TP, and reverses. Traders lose part (if not all) of their accumulated profits. This fact was pretty disturbing for traders, and therefore, perhaps, some traders had the idea of a "step-by-step exit from the market" similarly to the entry.

This gave way to trading tactics, in which a trader closes part of open positions at predetermined levels. Finally, by combining the step-by-step entry with the step-by-step exit and turning it into a kind of "symmetrical system" that is independent of technical analysis (or only partially dependent on it), the trader’s thought came to the concept of grid trading. I said "symmetrical" because now the entry levels are set at the same distance from each other. The same usually goes for the exit levels.

The grid is characterized by the following parameters: _grid width, grid step, TP, SL._

The grid width is the area covered by the placed orders. Grid step is the distance between orders. Grid width and step are calculated in points.

Thus, we have reached the definition of the grid trading method.  _A trading method, in which the entry into the market is carried out using many orders usually located at the same distance from each other (grid step) and usually on both sides of the current price is called a grid._

The significant advantage of this approach to trading is that it is now unnecessary to carry out hard (and often thankless) work of market forecasting. In fact, the grid method of trading has joined the theory of the efficient market or the theory of random walk of prices.

You may ask, why the grid type of trading can be profitable, or how we can be confident that there can be a profit (using this trading approach often involves no technical analysis).

To understand what this confidence is based on, let's look at Fig. 3 depicting the grid of stop orders and the movement of the futures contract price of PJSC Sberbank ordinary shares price.

The grid of orders in the strategy based on stop orders involves opening pending trades in the direction of a trend. If, however, stop orders are placed on both sides of the current price, then no matter which direction the trend goes, a trader will make a profit (Fig. 3 and 4 present testing an EA with a set take profit value):

![Triggering and replacing orders according to the trend](https://c.mql5.com/2/54/0.1_8678rceb_r_0nlysv3hl1lzw2n_owtey13_2d5_uo.png)

Fig. 3  The principle of grid trading profitability with a take profit less than the grid step width

When closing by take profit less than the grid step, the balance and equity graphs look as follows

![graph](https://c.mql5.com/2/54/0.2_flfxf3_x49ifbsb_1_hk2kjjb20ylzx7t_39i3j0s_t91_x6.png)

Fig. 4. Balance and equity graphs when closing by take profit

The type of this grid is based on the fact that the trader has no preference in the further direction of price movement due to the fact that he or she does not do any analysis at all or believes that further price changes, both upward and downward, are equally likely.

The grid is built from the selected level, which is close to the current price, according to the following principle:

> \- Buy Stop orders are placed above the selected level;
>
> \- Sell Stop orders are placed below the selected level;

The grid step is determined by the trader, but usually ranges from a few pips (above the spread) to the average daily true value of the traded symbol. This value can be determined by the Average True Range (ATR) indicator, which shows how much the price of a symbol has changed over a certain period of time (a measure of volatility). The choice of the step size determines the intensity of grid trading. Naturally, the smaller the grid step, the more aggressive it is and has an increased possible profitability. Therefore, when setting up and maintaining the grid, scripts and Expert Advisors are often used.

A stop loss in this type of grid trading approach is often not set on stop orders (visually controlled by the chart), and a take profit is also determined by the user's preferences (it is also possible to use a value smaller than the grid step).

Images 5 and 5.1 show a test with preset parameters:

> \- grid width 3,500 points;
>
>  \- grid step 184 points;
>
>  \- take profit 100 points (less than the grid step).

![parameter value](https://c.mql5.com/2/54/r0n_1_2t0bxauj_lxtw5s8o0e.png)

Fig. 5. Values of external variables

![TP 100 points step 184](https://c.mql5.com/2/54/5.1._n313l1wu_c_p3_s5_wrnth_c6p46g8.png)

Fig. 5.1. Test with take profit of 100 points, step 184 points

In the presented grid trading approach, it is possible to adjust the grid parameters, its frequency (with a different number of orders in the external parameters of the EA) and set the upper and lower boundaries. Next, the system automatically places orders at an equal distance from each other based on the specified criteria, above and below the specific price of the traded symbol.

**Using this grid trading strategy on stop orders involves the following steps:**

- setting the starting grid structure. It implies defining price levels in external variables, setting and placing stop orders with a rate above and below the market rate, according to the number of stop orders specified by the user.
- opening positions. The grid is activated by a market position when the market price triggers the nearest stop order after the start (above the buy stop price, below the sell stop price).
- updating the grid. The structure is updated every time after reaching one of the set price levels within the price range specified by the user. In other words, after the next stop order is triggered, the stop order is re-placed depending on the direction of orders triggering (price movement). When the price moves down, sell stop orders are triggered, and within the price range specified by the user, additional buy stop orders are placed closer to price.

For example, the following trading principle is presented in the strategy tester - two orders on the far right (buy stop) turn into market positions after being triggered and close by take profit when the symbol price moves up. New sell stop grid orders are placed closer to the price of the tested symbol (Fig. 6):

![stop grid trading principle](https://c.mql5.com/2/54/6_sqimj20_4c93zm1omwgmn1a_y8x41tx.png)

Fig. 6. The principle of triggering and placing new orders according to trading criteria

**2\. Using a trading strategy based on stop pending orders**

Let's consider the principle of working with this type of grid. After closing deals by take profit, new orders are placed instead according to the rule:

- if the current price position is higher than the level of the closed deal, then a pending Sell Stop order is placed;
- if the current price position is lower than the level of the closed deal, then a pending Buy Stop order is placed;

As the grid functions, when the price of the symbol moves, the position is exited by take profit if the take profit is set less than the width of the grid step. This happens both with buy and sell positions. If the take profit is not set or exceeds the grid step, then with the unidirectional movement of the traded symbol, a market position is set when pending orders are triggered. The so-called unloading of the market position occurs during rollbacks from the previous main movement by an amount equal to or greater than the grid step causing opposite pending orders to trigger and increasing the account balance.

Thus, no matter in which direction the price moves, profit is obtained due to the prevalence of trading positions in the direction of the trend.

**The advantage of** **Buy** **Stop and** **Sell** **Stop order grids are:**

1. The ability to trade with minimal skills.
2. Trading is always carried out in the direction of the main trend.


**The disadvantage of** **Buy** **Stop and** **Sell** **Stop order grids is:**

1. In case of a wide flat channel, quotes can immediately activate 5-6 or more stop orders in both directions, and it will take a lot of time and resource of the trading account to get out of the drawdown.

There is no need for careful market forecasting. But in order to increase profits, it is important to consider many factors. As a rule, a good choice for this type of trading is a symbol (the traded rate of which is characterized by frequent and significant, preferably unidirectional rises or falls with no drawdowns).

**Why automated grid trading is so popular:**

> It is convenient. You can figure it out quickly and there are no too complicated calculations that require a lot of experience.
>
> This is effective in terms of risk management.
>
> It is safe. The trading approach has been proven over the years and many traders practice it in a wide variety of markets.
>
> Flexible tactic. It easily adapts to different market conditions.
>
> Finally, grid trading is perfect for automation. It is extremely logical, has a clear structure and does not depend on the market behavior.

This is a grid trading method in which the trader must take the liberty of deciding that some trend will prevail in the traded area. The stronger the trend, the more profit the trader will receive. Stop orders are used here, which are opened in the direction of the trend, i.e. Buy Stop is opened for buying, and Sell Stop for selling. Stop orders are also used in a situation where the price has been in a range for a long time, upon exiting which, in any direction, the so-called additional loading (with stop orders co-directed to the movement) and unloading (with price rollbacks and triggering of pending orders of the opposite direction) will also occur.

For this type of grid trading, strong reverse fluctuations are unprofitable: extreme Buy Stop and Sell Stop orders may require a significant amount of funds on the account when the price returns to its original value.

**3\. Setting the EA parameter values and selecting trading symbols on the Moscow Exchange**

In the code, part of the external variables looks like this:

```
//+------------------------------------------------------------------+
//|                          EXTERNAL GRID VARIABLES
//+------------------------------------------------------------------+
input int Volume = 1;                          //Contract/lot volume
input double HIGH_PRICE_BuyStop = 5500.00;     //HIGH_PRICE_BuyStop (the upper price of the last BuyStop grid order)
input double LOW_PRICE_SellStop = 4000.00;     //LOW_PRICE_SellStop (the lower price of the last SellStop grid order)
input double HIGH_PRICE_TP  = 100.00;          // HIGH_PRICE_TP: TP in points from HIGH_PRICE_BuyStop
input double LOW_PRICE_TP = 100.00;            // LOW_PRICE_TP:  TP in points from LOW_PRICE_SellStop
input double TakeProfit  = 0;                  // TakeProfit from the setting price in order points
input double StopLoss  = 0;                    // StopLoss from the setting price in order points
```

The figure below shows an example from the Settings tab for testing trading by an Expert Advisor in Fig. 7:

![test settings ](https://c.mql5.com/2/54/8_n939e_a04dam1ff.png)

Fig. 7. Example of setting parameter values for testing

### Ways to improve your trading strategy

In addition to a partial exit from the total market position by using opposite pending orders (on price rollbacks from the main movement, which is embedded in the logic of the trading system based on a grid of pending orders) or by stop losses, you can also look at the implementation of the exit through the trailing option.

For example, the Moscow Exchange offers the following options for a futures contract for ordinary shares of PJSC Sberbank (these are the assets the preliminary testing of the grid trading approach was performed on). See Fig. 8 and the link:

[https://www.moex.com/en/contract.aspx?code=SBRF-6.23](https://www.mql5.com/go?link=https://www.moex.com/en/contract.aspx?code=SBRF-6.23 "https://www.moex.com/en/contract.aspx?code=SBRF-6.23")

![PJSC Sberbank specification](https://c.mql5.com/2/55/Image_001.png)

Fig. 8. Parameters of a futures contract for ordinary shares of PJSC Sberbank for placing new orders according to trading criteria

Also, when trading with the grid method, one should not forget about the calculation of the necessary and sufficient margin for the total number of orders and positions in the total volume of contracts.

Figures 9 and 9.1 demonstrate quite good unidirectional movements of futures contract for ordinary shares of PJSC VTB Bank with a relatively small value of margin.

![chart](https://c.mql5.com/2/54/8_hq7nzn65_2bl.png)

Fig. 9. Futures contract for ordinary shares of PJSC VTB Bank

![symbol parameters](https://c.mql5.com/2/55/Image_002.png)

Fig. 9.1. Parameters of futures contract for ordinary shares of PJSC VTB Bank

### Conclusion

In this article, we got acquainted with grid trading and types of trading orders - Buy Stop and Sell Stop. The material presented in the article implies and carries information for the user solely for educational purposes and a general understanding of the functioning and capabilities of the grid method of trading based on stop orders using the EA (attached below). I am also attaching a report on using the strategy on a futures contract for ordinary shares of PJSC Sberbank. Grid trading has capital risks, just like any other margin trading strategy.

Disclaimer: The article is not a trading recommendation. The ideas outlined in the article should only be considered as a possible trading opportunity.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/10671](https://www.mql5.com/ru/articles/10671)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10671.zip "Download all attachments in the single ZIP archive")

[ReportTester\_and\_set.zip](https://www.mql5.com/en/articles/download/10671/reporttester_and_set.zip "Download ReportTester_and_set.zip")(51.97 KB)

[GRID\_TRADING\_v.03\_\_breakdown.mq5](https://www.mql5.com/en/articles/download/10671/grid_trading_v.03__breakdown.mq5 "Download GRID_TRADING_v.03__breakdown.mq5")(119.4 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Forex spread trading using seasonality](https://www.mql5.com/en/articles/14035)
- [Benefiting from Forex market seasonality](https://www.mql5.com/en/articles/12996)
- [Automated grid trading using limit orders on Moscow Exchange (MOEX)](https://www.mql5.com/en/articles/10672)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/449247)**
(14)


![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
14 Jul 2023 at 12:55

**Bohdan Suvorov [#](https://www.mql5.com/ru/forum/446754#comment_47998028):**

...

7\. What is the role and importance of the time factor in grid trading on stop orders? What time intervals or periods are considered when making decisions about placing pending orders and exiting positions?

...

7\. Initially (based on the possible dynamics of the futures (stock) according to statistics) it is assumed that if the trend has started - it goes smoothly upwards, for example, with small pullbacks.

Therefore, it is assumed to trade in the direction of the main price movement of the symbol, regardless of its direction, taking into account the nature of the price movement in the past, it is assumed to use this trading approach from a week and above.

Naturally, with control from the side. I.e. the price has chattered in the range, some number of orders have worked - further, when the price goes beyond the boundaries of the previously selected range, you can close the position yourself, for example, in parts. In fact, the inter-day interval is traded with the transfer of the position cumulative through the night.

Timeframe for calculating the price range: min-max from a week.... month (+ is the upper boundary, - is the lower boundary).

After that, sketch a new range network of orders.

In essence, as trading is implemented here (yes - you don't buy cheaper - but more expensive), if the movement is going on - the position is gained gradually, if the pullbacks are not big and do not attract opposite orders, the equity grows faster, also allowing to gain further by stop orders the cumulative current position.

Further - also outside this robot - it is possible to exit by parts, as I have realised in practice.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
14 Jul 2023 at 13:02

**Bohdan Suvorov pending orders based on the current price position and closed trades? What factors and criteria were taken into account when developing these rules? How do you evaluate the effectiveness of using Buy-Stop and Sell-Stop order grid in different market conditions? What factors or signals do you consider when forecasting the main trend to decide on the direction of pending orders?**
**How do you solve the problem of drawdowns in case of activation of several stop orders in both directions? What strategies or methods are used to get out of such drawdowns and restore the account balance?**
**How does automated grid trading manage risk? What security measures or restrictions are used to minimise potential losses or undesirable situations associated with activating multiple orders in both directions?**
**What factors or signals are used to determine when to exit a take profit position? What criteria or methods are used to set take profit levels?**
**How do you account for flat market conditions and the possibility of long-term price range? What strategies or methods are used to prevent unnecessary activation of stop orders in such situations?**
**What is the role and importance of the time factor in grid trading on stop orders? What time intervals or periods are considered when making decisions about placing pending orders and exiting positions?**
**How do you manage the size of the order grid and the spacing between orders? What factors or techniques are used to determine optimal values for these parameters?**
**What factors or tools are used to evaluate market conditions and make decisions about symbol selection for grid trading? What symbol characteristics are considered most favourable for this type of trading?**
**How do you assess the applicability and effectiveness of grid trading on stop orders for different types of traders? What recommendations or advice can you give to traders who want to use this strategy?**

1\. visually on the chart. You set the range, days - weeks - above, below - levels max, min. Previously on the statistics looking at how the price has moved over the past weeks.

Immediately: For such a trading approach is desirable - directional movements....

2\. "There is no trend ... a trap net of pending limit orders is placed ..... The trap trap has worked out, makes a profit (total on the grid) and closes the whole grid. " +

see on seasonality. Like - summer - flat (not now...) - on limit orders. Autumn - on stops.

3. And how to solve them ... the principle of grid trading is like this ... we wait until either the price movement itself will go to the already opened positions in the desired direction or a sufficient [number of positions](https://www.mql5.com/en/docs/trading/positionstotal "MQL5 documentation: PositionsTotal function") will be opened in the opposite direction to cover the loss from unprofitable positions.

initially put orders based on the size of the balance, taking into account MM and possible triggering of several multidirectional orders, such as their "tipping" in the range ...

4\. The problem of drawdowns and risks is solved by "not roughing up" orders by volume in the grid.

5\. + you close in profit. And further already on the statistics of symbol movement - see the range of min and max prices to set the next network of orders.

6\. The essence of trading on stop orders with re-setting is reduced to the directional movement of the price of the symbol in any direction, so visually analysed the market, price movements, seasonality factor, news and when the absence of"long-term price in the range " is predicted.

...

8\. + visually and statistically on the chart. Week - month up boundary, week - month down boundary. Distance between orders - depends on many factors, including greed....

used in trading the value of about one third of the daily range of futures price movement.

9\. + average value of GO, the symbol's tendency to directional movements.

10\. Grid trading on stop orders is a breakout "theme", i.e. breakdowns are traded, including inter-day breakdowns. It is rather suitable for (so-called "position traders", it took me a long time to remember - I don't know much about the types of traders.... :-))those who carry positions through the night, a - la mid-term... inside the day - often there are "jitters" in the range.

\+ do not forget that even if a position on the directional movement has collected a network of co-directional orders and closed in plus either by you or robot by % of Dep or TR, no one prevents you from assessing the market situation (like rolling up - went out on TR, now - will roll down, and may continue to move up (like early exit)), and also to throw a network on stop orders, evaluating and setting the range: min/max price and with a distance of about 0.3 \* ATR on the days between stop orders - continue to collect profit.

Initially, the idea of the article was formed while trading futures on the Moscow Exchange, often had to either buy manually when the price of the same futures of VTB Bank moved up (often in time (in parts) did not have time, immediately bought "a lot" and when at the computer) or to sell increasing the position when the price moved down, including at the expense of increasing equity of the account, at some point, based on the practice of trading, decided to put this trading approach in the code on mql5, in the form of an expert. Yes, it is not deprived of possible optimisation of the code structure, but it carries and fulfils the semantic load in full. It is embodied in the form of a trading expert for automated trading for the Moscow Exchange on stop orders with their resetting.

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
14 Jul 2023 at 13:52

**Sergei Toroshchin [#](https://www.mql5.com/ru/forum/446754#comment_48008404):**

In my tests the timeframe is only a factor in how often a profit and close is checked .... i.e. let's say we have a short timeframe and a small timeframe, a profit will be caught ... if the timeframe is long, there is a chance that no profit will be caught and the grid will generally go backwards.

In my redesigned version I have two types of TFs set ... one is responsible for checking and closing positions for a profit ... and the second TF is responsible for maintaining the grid of pending orders.

As a result, I also removed the upper and lower limits of the price corridor as useless rubbish ... now it just maintains the minimum number of pending orders placed in each direction ... but no more than the maximum number of orders. I.e. in my case it is minimum 5 and maximum 7 ... which allows not to spam with constant placing and removal of orders and not to worry about the price corridor ... the corridor itself follows the current price

Yes. By the way, you can do it this way too ... It's like throwing a range from the ATR above and below the price and it is constant and orders constantly, until there is no exit (preferably at Takeout :-)).

The task was to describe the trading approach and its implementation in the Expert Advisor for a netting type of account.

![stuartfbs](https://c.mql5.com/avatar/2024/3/66005cee-e663.jpg)

**[stuartfbs](https://www.mql5.com/en/users/stuartfbs)**
\|
2 Oct 2023 at 07:32

Добрый день. Очень интересная и информативная статья. Спасибо! Good day, a very interesting and informative article.Thank you.


![Atiq Pasha](https://c.mql5.com/avatar/2025/12/694b1890-45ea.jpg)

**[Atiq Pasha](https://www.mql5.com/en/users/atiqp118)**
\|
23 Dec 2025 at 22:30

“A well-structured article that clearly explains automated grid trading using stop pending orders on the [Moscow Exchange](https://www.mql5.com/en/articles/1284 "Article: Fundamentals of Exchange Pricing on the Example of the Derivatives Section of the Moscow Exchange "), offering practical insights into performance, execution logic, and risk management.”


![Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://c.mql5.com/2/55/Revolutionize_Your_Trading_Charts_Part_I_avatar.png)[Improve Your Trading Charts With Interactive GUI's in MQL5 (Part I): Movable GUI (I)](https://www.mql5.com/en/articles/12751)

Unleash the power of dynamic data representation in your trading strategies or utilities with our comprehensive guide on creating movable GUI in MQL5. Dive into the core concept of chart events and learn how to design and implement simple and multiple movable GUI on the same chart. This article also explores the process of adding elements to your GUI, enhancing their functionality and aesthetic appeal.

![Creating an EA that works automatically (Part 13): Automation (V)](https://c.mql5.com/2/51/aprendendo_construindo_013_avatar.png)[Creating an EA that works automatically (Part 13): Automation (V)](https://www.mql5.com/en/articles/11310)

Do you know what a flowchart is? Can you use it? Do you think flowcharts are for beginners? I suggest that we proceed to this new article and learn how to work with flowcharts.

![Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://c.mql5.com/2/51/Perceptron_Multicamadas_60x60.png)[Multilayer perceptron and backpropagation algorithm (Part 3): Integration with the Strategy Tester - Overview (I).](https://www.mql5.com/en/articles/9875)

The multilayer perceptron is an evolution of the simple perceptron which can solve non-linear separable problems. Together with the backpropagation algorithm, this neural network can be effectively trained. In Part 3 of the Multilayer Perceptron and Backpropagation series, we'll see how to integrate this technique into the Strategy Tester. This integration will allow the use of complex data analysis aimed at making better decisions to optimize your trading strategies. In this article, we will discuss the advantages and problems of this technique.

![Rebuy algorithm: Math model for increasing efficiency](https://c.mql5.com/2/54/mathematical_model_to_increase_efficiency_Avatar.png)[Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)

In this article, we will use the rebuy algorithm for a deeper understanding of the efficiency of trading systems and start working on the general principles of improving trading efficiency using mathematics and logic, as well as apply the most non-standard methods of increasing efficiency in terms of using absolutely any trading system.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/10671&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082914669259133269)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).