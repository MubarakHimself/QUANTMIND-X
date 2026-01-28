---
title: How to Test a Trading Robot Before Buying
url: https://www.mql5.com/en/articles/586
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:00:19.885477
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/586&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071497392055855410)

MetaTrader 5 / Tester


Buying a trading robot on [MQL5 Market](https://www.mql5.com/en/market "The store for trading robots and technical indicators") has a distinct benefit over all other similar options - an automated system offered can be thoroughly tested directly in the MetaTrader 5 terminal. Before buying, an Expert Advisor **can and should** be carefully run in all unfavorable modes in the built-in [Strategy Tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") to get a complete grasp of the system, seeing that every Expert Advisor offered on MQL5 Market has a demo version available.

**Remember:** it is not only the amount paid that you risk when buying a trading robot, but also potential losses that may arise as a result of using such trading robot to trade on the real account.

Let us have a look at it using as an example a free [Three Moving Averages Expert Advisor](https://www.mql5.com/en/market/product/41) which we are going to [download directly in the MetaTrader 5 terminal](https://www.metatrader5.com/en/terminal/help/startworking/interface "https://www.metatrader5.com/en/terminal/help/startworking/interface"). It is an implementation of a classic trading strategy based on three moving averages.

![Downloading the Expert Advisor from MQL5 Market directly in the MetaTrader 5 terminal](https://c.mql5.com/2/5/fig1.png)

### Expert Advisor Evaluation Methods Based on Test Results

While there is no general method that could give you a 100% guarantee as to performance of the trading robot, there are simple methods that allow you to check the main parameters of any particular trading system in the Strategy Tester of the MetaTrader 5 terminal. The key methods available are as follows:

- Stress Testing in the random delay mode,
- Testing in a different trading environment,
- Testing on a different symbol/time frame,
- Backtesting on bad historical data,

- Backtesting over extended period of history (following the publication of the Expert Advisor in MQL5 Market),
- Forward Testing.

In addition, attention should be given to potentially suspicious factors, such as:

- profit factor that is too high,
- huge profit value on historical data,
- a great number of external parameters in a trading system,
- intricate rules of money management.


Even though all of the above is a fairly easy task, most newbies, as well as many somewhat experienced traders are either not aware of these nuances or not always attentive enough. Let us once again note that any trading robot downloaded from MQL5 Market can be set for testing directly in the Navigator window.

![Starting Expert Advisor using the Navigator menu](https://c.mql5.com/2/5/fig2.png)

Strategy Tester's panel with the Expert Advisor you selected will appear automatically once you press "Test" in the context menu. Everything is at hand to test the downloaded Expert Advisor and we are ready for a detailed review of the evaluation methods pointed out above.

### Stress Testing in the Random Delay Mode

The Strategy Tester is primarily designed to test the trading rules of a system. This means that the Strategy Tester emulates the ideal environment for all processes:

- sending trade requests,
- updating status of open positions and pending orders,
- getting trade events,

- getting price history,
- calculating indicators and many other things.

Everything is aimed at testing and optimizing the trading strategy within minimum time. However, seeing that the operation of a trading robot in real environment is far from being ideal and instantaneous, the Strategy Tester has been enhanced with an additional testing mode that simulates a [random delay](https://www.metatrader5.com/ru/terminal/help/algotrading/testing "https://www.metatrader5.com/ru/terminal/help/algotrading/testing") between sending and execution of a trade order.

![Setting the random delay mode](https://c.mql5.com/2/5/fig3__1.png)

This testing mode accurately detects:

- trading operation handling errors,
- fitting the strategy to certain trading conditions.


Getting markedly different trade results after running a single test of the Expert Advisor in two modes, standard and random delay, should get you thinking. First, take a look at the [Strategy Tester log](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") as numerous trade errors it contains should be a sufficient reason to cross such Expert Advisor off your list. In our case, no errors of that kind have been detected in the course of stress testing in the random delay mode suggesting that the Expert Advisor has successfully passed the first half of the test.

Now, let us see if there is any difference between the [trade results obtained using single tests](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") run in two modes. Significantly decreased number of trades and profit gained in the random delay mode suggest that the strategy is highly dependent on the quality of transmission and execution of trade orders and can only earn under certain ideal conditions. The developer may have done it unintentionally which is very often the case. But such a 'flaw' can turn disastrous to your trading account.

![Comparison of test results in different trade order execution modes](https://c.mql5.com/2/5/fig4_eng.gif)

In our example, switching to a different trade order execution mode has not affected the number of trades and transactions. The test results are just a tiny bit different which can adequately be explained by small price changes present in transactions due to requotes.

**Conclusion:** the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor has passed this test. Stress testing in the random delay mode has not have a substantial effect on the trade results.

### Testing in a Different Trading Environment

Run a test of the trading robot under the conditions as specified in its description on MQL5 Market. Then connect to another broker account and run the test once again. It is somewhat similar to the previous stress testing and allows you to see how small changes in prices and trading conditions (spread, permissible StopLoss/TakeProfit levels, etc.) can affect trade results.

For example, you have the Expert Advisor test results for EURUSD on the **broker A** account. Run the same test on EURUSD, only this time on the **broker B** account. Should the results be very much different, it is a good reason to reconsider the need for such trading robot.

### Another Symbol/Time Frame

The majority of trading robots are developed so as to trade on one particular symbol and some of them even require to be used on a specific time frame. It appears to be quite reasonable as every instrument behaves in its own way. Therefore, symbol and time frame are, as a rule, always specified in the description of a trading robot offered on MQL5 Market.

Download a demo version of the Expert Advisor and start it on a different symbol and/or period. First, you need to make sure that the Expert Advisor is not going to crash with a [critical error](https://www.mql5.com/en/docs/runtime/errors) or fill the log with trade error messages, being used in inappropriate starting conditions. Second, check that a profitable trading strategy has not become extremely loss-making, due to the above changes in the settings - this can happen where curve fitting had taken place.

One of the easiest ways to arrange that kind of test for the Expert Advisor is to optimize it over [all symbols selected in Market Watch](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types "https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types"). We run the optimization of the Expert Advisor in that mode on a quite long time frame H1 with "Every tick" generation and get a fairly quick answer to the second question.

![ Optimization over all symbols selected in Market Watch](https://c.mql5.com/2/5/fig5.png)

Results of such optimization show that the strategy has a right to exist, demonstrating statistically sufficient number of trades on each symbol without yielding really bad results. Mind you, we have tested one strategy on **all 13 symbols** in [Market Watch](https://www.metatrader5.com/en/terminal/help/trading/market_watch "https://www.metatrader5.com/en/terminal/help/trading/market_watch") **with the same parameters** set by default.

![Results of the optimization over all symbols selected in Market Watch](https://c.mql5.com/2/5/fig6.png)

We can certainly not expect that every Expert Advisor would work equally well on any symbol and time frame. Yet it is worth checking it in the Strategy Tester using this method. It will not only reveal possible code errors but can even give new ideas.

**Conclusion**: the behavior of the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor has been normal when tested on a different symbol/time frame. No obvious code errors have been detected during testing.

### Backtesting on Bad Historical Data

We have found out that the Expert Advisor yields best results when working on GBPUSD. But what if this is not a consistent pattern and this behavior is due to the testing interval selected from 2012.01.01 to 2012.09.28 which by a pure fluke turned out to be favorable? To look into this question, we test the Expert Advisor with the same parameters over 2011, taking 2011.01.01-2011.12.31 as an interval. We run the test and see the results.

![Backtesting on bad historical data](https://c.mql5.com/2/5/fig7.png)

The Expert Advisor is no longer profitable and has immediately become much less wowable. Moreover, losses suffered in 2011 significantly exceed profits demonstrated in the Strategy Tester over 2012.01.01-2012.09.28. However, now we are aware of potential losses, even when trading on GBPUSD.

**Conclusion:** the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor requires further development to ensure proper automatic response to changes in the market behavior, or else the right parameters for every interval have to be found through optimization.

### Backtesting Over Extended Period of History

When giving descriptions, developers of trading robots try to show their products at their best and therefore provide reports and test charts with optimum parameters for a particular interval. Since considerable time has usually passed from the date of publishing the trading robot till the date when you get interested in it, we can run a so-called forward test.

Forward testing is testing over a period of history that was not considered when selecting optimum parameters. We are going to continue the analysis of this Expert Advisor on GBPUSD over a little longer testing interval, including historical data after September 28, 2012. The end date is set at 2012.11.26, thus adding nearly two extra months. So, following the test run over the period from 2012.01.01 to 2012.11.26, we get the new testing chart:

![Backtesting over extended period of history](https://c.mql5.com/2/5/fig8.png)

In our case, the results demonstrated by the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor over the additional short interval (Forward) are even better than those achieved over the preceding 10 months. This is however very rare.

**Conclusion:** testing of the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor on GBPUSD over the extended period of history has not shown any weakening of the trade parameters.

### Forward Testing

Forward testing is used to assess stability of the trading system in the changing market behavior. Optimization of parameters in the Strategy Tester allows us to get the parameters at which the trading robot is at its best on historical data within a certain interval. But this does not guarantee that the obtained parameters will be the same best fit even when used for trading in the nearest future.

Traders who develop automated trading systems often confuse such concepts as optimization and curve fitting. The line between a fair optimization and curve fitting is very thin and hard to find. This is where forward testing has proved useful allowing to objectively assess the obtained parameters.

Upon optimization in the MetaTrader 5 Strategy Tester, you can choose to forward test the resulting optimum parameters and set the necessary limits. Let us run forward testing of our trading robot with the settings as shown below.

![Setting the forward optimization mode](https://c.mql5.com/2/5/fig9.png)

Forward is set at 1/4 which means that the specified interval 2012.01.01- 2012.11.26 will be divided into 4 parts. First 3/4 of the history will be used to find the optimum parameters and the best 25% passes (Expert Advisor parameter sets) will be forward tested on the remaining 1/4 of historical data.

Specify the parameters to be optimized - we will select those that are supposed to have impact on the trading logic. Therefore, we will not optimize parameters in charge of money management.

![Parameters to be optimized ](https://c.mql5.com/2/5/fig10.png)

The above combination of the step, as well as start and stop values has resulted in nearly 5 million passes. Under the given circumstances, it is not unreasonable to use genetic algorithm and involve [MQL5 Cloud Network](https://cloud.mql5.com/en "https://cloud.mql5.com/en") in the optimization.

So, let us take a look at the results of the optimization including forward passes that has taken a total of 21 minutes and cost 0.26 credit for more than 4000 passes using the cloud agents. An example of how the costs are calculated can be found in the article [MQL5 Cloud Network: Are You Still Calculating?](https://www.mql5.com/en/forum/8153)

![Chart of forward testing results](https://c.mql5.com/2/5/fig11.png)

At first glance, there seems to be something wrong with it. We check the results and see that the values of the first three optimized parameters are the same throughout all passes. And only the last two parameters Inp\_Signal\_ThreeEMA\_StopLoss and Inp\_Signal\_ThreeEMA\_TakeProfit have varying values.

### ![Table of forward pass results](https://c.mql5.com/2/5/fig12.png)

Considering the above, we can make two assumptions:

- these parameters, specifically StopLoss and TakeProfit values, have in effect no influence on the trading results;
- genetic algorithm failed to get out of local extremum that we hit during the optimization.


Let us check both assumptions by re-optimizing with the same settings and input parameters. This time, the chart of forward testing results looks a little different.

![Another chart of re-optimization on the forward period](https://c.mql5.com/2/5/fig13.png)

As a result of the optimization we can now see three mainstreams. This means that the last two optimized parameters still appear incidental to the given trading robot.

**Conclusion:** optimization of the [Three Moving Averages](https://www.mql5.com/en/market/product/41) Expert Advisor on GBPUSD has shown that the trading logic only depends on three parameters out of seven.

Let us make one last attempt and remove unnecessary parameters from the optimization. We now only have 1650 passes.

![Reduced set of parameters for optimization](https://c.mql5.com/2/5/fig14.png)

Therefore, complete parameter search would make more sense, rather than genetic optimization. MQL5 Cloud Network will in this case provide us with more agents and the time required to complete the process will as a result be significantly reduced.

![Using MQL5 Cloud Network agents upon complete parameter search](https://c.mql5.com/2/5/fig20.png)

The task has been completed in 7 minutes with 2000 cloud agents involved and the forward testing chart looks good.

![Optimization chart](https://c.mql5.com/2/5/fig15.png)

Most passes over the forward period turned out to be profitable, with the number of points above the initial $10.000 being much greater than in the loss-making zone. It looks somewhat hopeful but it does not mean that the resulting parameter sets will also prove profitable in the future.

### Number of Parameters in a Trading System

We have had a chance to see that not all strategy parameters available for setting up a trading robot are equally significant and able to affect trading results. In our case the Inp\_Signal\_ThreeEMA\_StopLoss and Inp\_Signal\_ThreeEMA\_TakeProfit values had virtually no impact on the performance of the Expert Advisor. However, it is more common to come across a trading robot that has a great number of parameter settings.

Numerous parameters allow you to make very accurate settings for a trading robot so as to fit its performance to a certain period of history which is highly likely to be revealed during optimization.

Curve fitting means that the Expert Advisor will probably not show the same level of profitability on data beyond the specified interval used for the optimization as it did on the test data. And worse yet, it may yield quite the opposite results, leading to losses.

It is believed that the less parameter settings a trading system has, the lower the possibility that the identified pattern will vanish in the future. And vice versa - the more parameters in the system, the lower the possibility that the market will keep its characteristics in line with such a fine-tuned Expert Advisor. As a proof of the above, we strongly recommend that you familiarize yourself with the results of the trade analysis provided in the article [Optimization VS Reality: Evidence from ATC 2011](https://championship.mql5.com/2012/en/news/160 "https://championship.mql5.com/2012/en/news/160") which we will turn to further below.

![Correlation between balance and number of parameters](https://c.mql5.com/2/5/fig16_eng.png)

The chart displays trade results of the participants over the [Automated Trading Championship 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en"). Vertical axis shows the account balance as at the end of the Championship and horizontal axis displays the number of EA's external parameters. Expert Advisors are represented by red diamonds. It can clearly be seen that the Expert Advisors with a great number of parameters lost money or, at the best, broke even, when trading over the forward period of the Championship.

The absence of external parameters in a trading robot offered for sale does not say anything about the generality of designed-in trading rules either and cannot be taken as coolness. The developer of the Expert Advisor must have, for some reason, simply threaded the external parameters inside the trading robot.

### Very High Profit Factor

Most traders do not like losing trades and take them as a sign of a faulty operation of a trading system. In fact, they cannot be avoided due to the nature of trading in the financial markets. Any trade upon opening a position can ultimately turn out to be either winning or losing. Trading losses are unavoidable and are seen as a form of naturally occurring pay and inevitable item of expenditure, as in any business.

Many developers of automated trading systems run to extremes, trying to reduce the number of losing trades and gross loss to the minimum. To achieve this and improve results that may be obtained in the Strategy Tester, they add extra filters that allow you to avoid losing trades, thus improving profit factor. Extra filters have their own parameters and settings, adding to the total number of input parameters.

Profit factor is defined as the gross profit divided by the gross loss. Profit factor of profitable systems is always greater than 1. However, if one has tried too hard and over-optimized a trading system in the Strategy Tester, this figure may be much bigger. Let us take a look at yet another chart from the article [Optimization VS Reality: Evidence from ATC 2011](https://championship.mql5.com/2012/en/news/160 "https://championship.mql5.com/2012/en/news/160").

![Very high profit factor as a result of optimization](https://c.mql5.com/2/5/fig17_eng.png)

It is clear that almost all trading robots that had a very high profit factor during testing over historical data were not even close to their backtesting results when tested over the forward period of the [Automated Trading Championship 2011](https://championship.mql5.com/2011/en "https://championship.mql5.com/2011/en") and virtually lost everything. It suggests that a very high profit factor demonstrated in the Strategy Tester was due to fitting the strategy to a certain time period used for the optimization of the trading robot.

### Huge Profit on Historical Data

Another alarming fact can be a huge profit stated in the description of a trading robot. If the attached Strategy Tester reports show a sky-high balance, it most likely has to do with curve fitting. Often developers of such "money printing machines" do not even realize that their system is over-optimized and has too many external parameters. Let us support this assertion by another chart from the above-mentioned report [Optimization VS Reality: Evidence from ATC 2011](https://championship.mql5.com/2012/en/news/160 "https://championship.mql5.com/2012/en/news/160").

![Huge Profit on Historical Data](https://c.mql5.com/2/5/fig18_eng.png)

Buyers of such "Grails" are as a rule inexperienced and easily blinded by huge profits on historical data. In those cases, delusion of profit that such trading robot can earn is genuine and mutual.

### Manipulations with Money Management

Creating special trade manipulation rules that allow you to go through bad historical data in the Strategy Tester with minimum losses and maximize returns on successful transactions is the most complicated and rare approach to the abnormal development of a trading robot. It is far from being what is called money management.

Such fitting can be best detected by testing on data which lies outside the period of history used to obtain the results that are stated by the developer in the description of the trading robot. The more extensive the fitting, the higher the possibility that the trading robot will fail the test.

### Do Not Trust Anyone. Not Even Yourself

Unfortunately, trading robot like any complex program may contain unintentional errors that cannot be detected other than by on-line trading. No developer of trading robots can guarantee that his program is error-free and would correctly handle all non-standard situations. Even the Expert Advisor that was successfully tested can make a trade error or crash due to a [critical error](https://www.mql5.com/en/docs/runtime/errors), when put in unexpected conditions which the developer could not foresee. The only implicit guarantee in this case can be experience and reputation of the trading robot's developer.

And, of course, an Expert Advisor that has demonstrated positive results in the Signals service over a sufficient period of time will be more reliable than the one that has not. Whatever the case, do not get knocked down calculating you future profits and remember two rules that are still valid:

1. do not trust anyone,

2. and no past trading successes can guarantee future profits.

We also recommend following articles dedicated to Market:

- [How to buy a Trading Robot, a Magazine or a Book in MetaTrader Market?](https://www.mql5.com/en/articles/498)
- [How to Post a Product in the Market](https://www.mql5.com/en/articles/385)
- [Tips for an Effective Product Presentation on the Market](https://www.mql5.com/en/articles/999)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/586](https://www.mql5.com/ru/articles/586)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/8712)**
(51)


![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
22 Nov 2025 at 09:13

**PZGBBCQ [#](https://www.mql5.com/en/forum/8712/page5#comment_58573566):**

How do I buy this EA software, can I rent it for years, how do I get in touch with and guide me to use it?

- [TIPS FOR PURCHASING A PRODUCT ON THE MARKET. STEP-BY-STEP GUIDE](https://www.mql5.com/en/articles/1776)
- [The correct way to choose an Expert Advisor from the Market](https://www.mql5.com/en/articles/10212)
- [How to buy, install, test and use a MT4 Expert Advisor](https://www.mql5.com/en/forum/366152)
- [How to buy, install, test and use a MT5 Expert Advisor](https://www.mql5.com/en/forum/366161)

![PZGBBCQ](https://c.mql5.com/avatar/avatar_na2.png)

**[PZGBBCQ](https://www.mql5.com/en/users/pzgbbcq)**
\|
23 Nov 2025 at 01:04

**Alain Verleyen [#](https://www.mql5.com/zh/forum/21576/page10#comment_58147011):**

Please write to the helpdesk.

How can I contact you to purchase EA software?


![PZGBBCQ](https://c.mql5.com/avatar/avatar_na2.png)

**[PZGBBCQ](https://www.mql5.com/en/users/pzgbbcq)**
\|
23 Nov 2025 at 01:10

**Sergey Golubev [#](https://www.mql5.com/zh/forum/21576/page10#comment_58573769):**

- [Tips for buying products in the market. A Step-by-Step Guide](https://www.mql5.com/en/articles/1776)
- [The right way to choose an Intelligent Trading System from the marketplace](https://www.mql5.com/en/articles/10212)
- [How to buy, install, test and use MT4 Expert Advisors](https://www.mql5.com/en/forum/366152)
- [How to buy, install, test and use MT5 Expert Advisors](https://www.mql5.com/en/forum/366161)

How to contact the developer of the EA software


![Hamed Hejav](https://c.mql5.com/avatar/avatar_na2.png)

**[Hamed Hejav](https://www.mql5.com/en/users/hamedhejav5)**
\|
4 Jan 2026 at 00:43

Automatic translation was applied by a moderator. Please post in the language of the forum section you selected.

Oh, I wanted to see when your vacation would end.

![Alexandr Saprykin](https://c.mql5.com/avatar/2017/9/59C03B7B-993D.JPG)

**[Alexandr Saprykin](https://www.mql5.com/en/users/svalex)**
\|
4 Jan 2026 at 03:27

**PZGBBCQ [#](https://www.mql5.com/ru/forum/8711/page10#comment_58576164):**

How to contact the developer of the Expert Advisor

Go to the page of your chosen Expert Advisor and go to the profile of its seller. In the profile click "write a message". It would seem that what could be simpler.


![How to become a Signals Provider for MetaTrader 4 and MetaTrader 5](https://c.mql5.com/2/0/Avatar_How_to_become_a_signal_provider.png)[How to become a Signals Provider for MetaTrader 4 and MetaTrader 5](https://www.mql5.com/en/articles/591)

Do you want to offer your trading signals and make profit? Register on MQL5.com website as a Seller, specify your trading account and offer traders a subscription to copy your trades.

![Interview with Dmitry Terentew (ATC 2012)](https://c.mql5.com/2/0/avatar__20.png)[Interview with Dmitry Terentew (ATC 2012)](https://www.mql5.com/en/articles/588)

Is it really necessary to be a programmer to develop trading robots? Do we need to spend years monitoring price charts to be able to "feel" the market? All these issues have been discussed in our interview with Dmitry Terentew (SAFF), whose trading robot has been occupying the first page of the Championship from the very beginning.

![MetaTrader 4 on Linux](https://c.mql5.com/2/13/1054_12.png)[MetaTrader 4 on Linux](https://www.mql5.com/en/articles/1358)

In this article, we demonstrate an easy way to install MetaTrader 4 on popular Linux versions — Ubuntu and Debian. These systems are widely used on server hardware as well as on traders’ personal computers.

![Interview with Anton Nel (ATC 2012)](https://c.mql5.com/2/0/avatar__17.png)[Interview with Anton Nel (ATC 2012)](https://www.mql5.com/en/articles/583)

Today we talk to Anton Nel (ROMAN5) from South Africa, a professional developer of automated trading systems. Obviously, his Expert Advisor just could not go unnoticed. Breaking into the top ten from the very start of the Championship, it has been holding the first place for more than a week.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/586&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071497392055855410)

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