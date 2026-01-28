---
title: Contest of Expert Advisors inside an Expert Advisor
url: https://www.mql5.com/en/articles/1578
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:02:15.759530
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=pwemafeabysjvqezvogilehqjxtroqnx&ssn=1769191334114653502&ssn_dr=0&ssn_sr=0&fv_date=1769191334&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1578&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Contest%20of%20Expert%20Advisors%20inside%20an%20Expert%20Advisor%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176919133479893724&fz_uniq=5071525163314391462&sv=2552)

MetaTrader 4 / Trading systems


### Introduction

Dear developers and exchange analysts!

Often, many trading systems (TS) are stable for a specific time period, but further, the balance curve starts moving downwards. Trader disappoints in it and starts inventing new grails, optimizing parameters, etc.

I bring to your attention a tool for performing virtual trades - VirtualTrend.mqh. Using the function I suggest, you can open, close and trail virtual trades.

It has the following useful features:

1. The possibility of creating adaptive Expert Advisors that can enable and disable virtual trading depending on the results of former trades.
2. Rating several trading systems (in this example, 5 trading systems are presented) in the percentage terms depending on their profitability. The competition results in a decision of which trading system is the most preferable for further trading at the real market.
3. The possibility of implementing trading strategies, which use several open trades by an instrument, into the MetaTrader 5 platform where cumulative positions are used (this feature is not described in this article, but suggested as a variant of using).

Let's take a look into the first two statements.

Let's take the Competition\_v1.0.mq4 code as an example of adaptive Expert Advisor; there, on the competitive basis, a trade system to be used is determined. The Expert Advisor uses 5 (you can add more if you want) trading systems, which aren't chosen randomly, but on the basis of a more or less stable working on the daily timeframe of EURUSD.

**Important!** You should use just the stable trading systems, which make profit or constantly lose at a certain period of time.

For example, a trading system, which sequence of wins  is "PPPPPPPLLLLLLLLPPLPPPPPPPP", fits the conditions to be used in this Expert Advisor; but if the sequence of wins is "LLPPLPLLPPPLLPLPPLPLPLLLPPLPL", it doesn't fit ("L" – losing trades, "P" – profitable trade).

A trading system where losing trades are more frequent that profitable ones, but the profit trades cover the former loss, cannot be used as well, because the Expert Advisor just doesn't manage to adapt to the profitability.

The stability of a trading system is determined "by sight" on the results of tests conducted for the maximum period (all history) in the strategy tester. The obtained diagram of change of balance allows determining the average weighted number of deals that cause a change of tendency. For example, a trading system performs the following sequence of deals: PPPPPPPPLLLLLLPPPPPPPPPLPLLLLLLLLLLLPPPPPPPLLLLLLLLLLLLLPPPPPPLLLLLLLLLLPLPPPLLPPPPPP. The tendency of balance changes after about 6-8 trades. When calculating the rating, it is recommended to set the period of averaging of the balance equal to a half of those trades. That is 3 or 4.

Enabling and disabling the rating system, which helps the Expert Advisor to adapt to the changes of profitability dynamics, is managed via the RatingON parameter. Using it, you can easily find a period of averaging of balance and set it for the Tх.PeriodRating parameter.

Each trading system can be enabled or disabled at your will using the Tх.Enabled parameter, where х is the number of TS.

All the tests of trading systems represented below are conducted at EURUSD(D1 timeframe) during the period from 1999.01.01 to 2010.06.01.

### Trading System №1

- Condition of entering: when the fast moving average crosses the slow one (see the T1\_SignalOpen() function).

- Condition of exiting: reverse crossing (see the T1\_SignalClose() function).

[![](https://c.mql5.com/2/17/strategy1_small.gif)](https://c.mql5.com/2/17/strategy1.gif)

Figure 1.The test of the trading system №1 without adaptation (RatingON = false)

The result of testing allows drawing a conclusion that after each 18-20 trades this strategy changes its dynamics of profitability.

That is why the T1.PeriodRating parameter is set to 20.

![The test of the trading system №1 with adaptation enabled (RatingON = true)](https://c.mql5.com/2/20/strategy1_rating_small__1.gif)

Figure 2.The test of the trading system №1 with adaptation enabled (RatingON = true)

### Trading System №2

- Condition of entering: when the CCI indicator crosses a certain level from top downwards (See the T2\_SignalOpen() function).
- Condition of exiting: when the signal appears in the opposite direction (See the T2\_SignalClose () function).

During the test for the same period we have found out that the stability is equal to about 10 trades.

[![](https://c.mql5.com/2/17/strategy2_small.gif)](https://c.mql5.com/2/17/strategy2.gif)

Figure 3.The test of the trading system №2 without adaptation

[![](https://c.mql5.com/2/17/strategy2_rating_small.gif)](https://c.mql5.com/2/17/strategy2_rating.gif)

Рисунок 4.The test of the trading system №2 with the adaptation enabled

### Trading System №3

The logic of entering and exiting is the same as the one of TS №1, however the periods of averaging of moving averages are different.

A little digression:you can rewrite the entire Competition\_v1.0.mq4 and combine it from the strategies that consist of only moving averages with different periods.

I haven't conducted this interesting investigation yet, but if I make it, I'll inform you.

[![](https://c.mql5.com/2/17/strategy3_small.gif)](https://c.mql5.com/2/17/strategy3.gif)

Figure 5.The test of the trading system №3 without adaptation

This TS performs few trades during a long period (almost 11 years). I recommend not using this strategy in real trading since the number of trades in the strategy tester is too small.

I've added this strategy for more obviousness of showing my development of the adaptation. The T3.PeriodRating parameter is set to 2 for this strategy.

[![](https://c.mql5.com/2/17/strategy3_rating_small.gif)](https://c.mql5.com/2/17/strategy3_rating.gif)

Figure 6.The test of the trading system №3 with the adaptation enabled

### Trading System №4

I've seen this strategy many times in literature.

It seems attractive even without the adaptation, but using it in trading doesn't require an automated trading, since it's a long-term one.

- It enters the market as three moving averages are arranged in the order and if MACDis higher than the specified level - T4.LimitMACD (see the T4\_SignalOpen() function).
- It exits if the price crosses the second moving average (see the T4\_SignalClose() function).

[![](https://c.mql5.com/2/17/strategy4_small.gif)](https://c.mql5.com/2/17/strategy4.gif)

Figure 7.The test of the trading system №4 without adaptation

This trading strategy has a constant stability, so the period of processing the data for calculation of the rating should be no less than 20 trades. I set T4.PeriodRating=20.

[![](https://c.mql5.com/2/17/strategy4_rating_small.gif)](https://c.mql5.com/2/17/strategy4_rating.gif)

Figure 8.The test of the trading system №4 with the adaptation enabled

### Trading System №5

This TS is developed by me and I want to share it with you in such a sophisticated way.

- We buy if the CCIindicator crosses the specified level T5.LevelCCI from bottom up (see the T5\_SignalOpen() function). Set the level of closing trades MyLevel lower than the level of opening T5.LevelCCI by T5.TralingCCI indicator points. Observe the CCIindicator and pull up the level of closing MyLevel with a specific step, in a way to keep the distance between the current value of the CCIindicator and the MyLevel level not greater than the doubled value of T5.TralingCCI.
- Close the opened buy position if the CCIindicator crosses the MyLevel level from top downwards (see the T5\_SignalClose() function).

[![](https://c.mql5.com/2/17/strategy5_small.gif)](https://c.mql5.com/2/17/strategy5.gif)

Figure 9.The test of the trading system №5 without adaptation

Such chart can be left without the adaptation, but anyway, set T5.PeriodRating=10.

[![](https://c.mql5.com/2/17/strategy5_rating_small.gif)](https://c.mql5.com/2/17/strategy5_rating.gif)

Figure 10.The test of the trading system №5 with the adaptation enabled

### Multisystem Expert Advisor

The working of 5 Expert Advisors with and without the adaptation is demonstrated above.

Now we're going to consider an example of collaborative work of those Expert Advisors:

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Symbol | EURUSD (Euro vs US Dollar) |
| Period | Day (D1) 1999.05.24 00:00 - 2010.07.05 00:00 (1999.01.01 - 2010.07.05) |
| Model | **Control points (a very crude method, the results must not be considered)** |
| Parameters | RatingON=false; FastTest=true; file="virtual.csv"; MinRating=50; \_tmp1\_="---- Trading system 1 ----"; T1.Enabled=1; T1.Magic=101; T1.lot=0.1; T1.Fast=10; T1.Slow=100; T1.TS=7000; T1.PeriodRating=20; \_tmp2\_="---- Trading system 2 ----"; T2.Enabled=1; T2.Magic=102; T2.lot=0.1; T2.PeriodCCI=30; T2.LevelCCI=200; T2.SL=500; T2.PeriodRating=10; \_tmp3\_="---- Trading system 3 ----"; T3.Enabled=1; T3.Magic=103; T3.lot=0.1; T3.Fast=30; T3.Slow=200; T3.TS=5000; T3.PeriodRating=2; \_tmp4\_="---- Trading system 4 ----"; T4.Enabled=1; T4.Magic=104; T4.lot=0.1; T4.SL=5000; T4.TS=5000; T4.LimitMACD=0.002; T4.PeriodRating=60; \_tmp5\_="---- Trading system 5 ----"; T5.Enabled=1; T5.Magic=105; T5.lot=0.1; T5.PeriodCCI=90; T5.LevelCCI=100; T5.TralingCCI=100; T5.SL=5000; T5.TS1=5000; T5.PeriodRating=10; |
|  |  |  |  |  |
| Bars in test | 2994 | Ticks modelled | 219840 | Modelling quality | n/a |
| Initial deposit | 500000.00 |  |  |  |  |
| Total net profit | 617173.70 | Gross profit | 1342671.82 | Gross loss | -725498.13 |
| Profit factor | 1.85 | Expected payoff | 2373.74 |  |  |
| Absolute drawdown | 76798.13 | Maximal drawdown | 172676.05<br>(28.98%) | Relative drawdown | 28.98%<br>(172676.05) |
|  |
| Total trades | 260 | Short positions (won %) | 126 (29.37%) | Long positions (won %) | 134 (33.58%) |
|  | Profit trades (% of total) | 82 (31.54%) | Loss trades (% of total) | 178 (68.46%) |
| Largest | profit trade | 78151.67 | loss trade | -18831.39 |
| Average | profit trade | 16374.05 | loss trade | -4075.83 |
| Maximum | consecutive wins (profit in money) | 6 (89681.19) | consecutive losses (loss in money) | 21<br>(-100325.23) |
| Maximal | consecutive profit (count of wins) | 95057.65 (3) | consecutive loss (count of losses) | -100325.23<br>(21) |
| Average | consecutive wins | 2 | consecutive losses | 4 |

[![](https://c.mql5.com/2/17/mynxgsthkknjdxt_small.gif)](https://c.mql5.com/2/17/mynxgsthkknjdxt.gif)

Figure 11.The test of the multisystem Expert Advisor without adaptation

**The result of combining the trading systems is the increase of profit, drawdown and number of trades.**

Now let's look into the same test but with the RatingON parameter enabled.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Period | Day (D1) 1999.05.24 00:00 - 2010.07.05 00:00 (1999.01.01 - 2010.07.05) |
| Model | **Control points (a very crude method, the results must not be considered)** |
| Parameters | RatingON=true; FastTest=true; file="virtual.csv"; MinRating=1; \_tmp1\_="---- Trading system 1 ----"; T1.Enabled=1; T1.Magic=101; T1.lot=0.1; T1.Fast=10; T1.Slow=100; T1.TS=7000; T1.PeriodRating=20; \_tmp2\_="---- Trading system 2 ----"; T2.Enabled=1; T2.Magic=102; T2.lot=0.1; T2.PeriodCCI=30; T2.LevelCCI=200; T2.SL=500; T2.PeriodRating=10; \_tmp3\_="---- Trading system 3 ----"; T3.Enabled=1; T3.Magic=103; T3.lot=0.1; T3.Fast=30; T3.Slow=200; T3.TS=5000; T3.PeriodRating=2; \_tmp4\_="---- Trading system 4 ----"; T4.Enabled=1; T4.Magic=104; T4.lot=0.1; T4.SL=5000; T4.TS=5000; T4.LimitMACD=0.002; T4.PeriodRating=60; \_tmp5\_="---- Trading system 5 ----"; T5.Enabled=1; T5.Magic=105; T5.lot=0.1; T5.PeriodCCI=90; T5.LevelCCI=100; T5.TralingCCI=100; T5.SL=5000; T5.TS1=5000; T5.PeriodRating=10; |
|  |  |  |  |  |
| Bars in test | 2994 | Ticks modelled | 219840 | Modelling quality | n/a |
| Initial deposit | 500000.00 |  |  |  |  |
| Total net profit | 227123.75 | Gross profit | 388438.79 | Gross loss | -161315.05 |
| Profit factor | 2.41 | Expected payoff | 2341.48 |  |  |
| Absolute drawdown | 10921.17 | Maximal drawdown | 76482.03<br>(12.48%) | Relative drawdown | 12.48%<br>(76482.03) |
|  |
| Total trades | 97 | Short positions (won %) | 50 (40.00%) | Long positions (won %) | 47 (46.81%) |
|  | Profit trades (% of total) | 42 (43.30%) | Loss trades (% of total) | 55 (56.70%) |
| Largest | profit trade | 71192.28 | loss trade | -12680.47 |
| Average | profit trade | 9248.54 | loss trade | -2933.00 |
| Maximum | consecutive wins (profit in money) | 5 (80463.85) | consecutive losses (loss in money) | 13<br>(-50753.48) |
| Maximal | consecutive profit (count of wins) | 80463.85 (5) | consecutive loss (count of losses) | -50753.48<br>(13) |
| Average | consecutive wins | 2 | consecutive losses | 3 |

[![](https://c.mql5.com/2/17/dlqxxxccgbyabuq_jpqjiuosrxztowxbtyvtma1__small.gif)](https://c.mql5.com/2/17/dlqxxxccgbyabuq_jpqjiuosrxztowxbtyvtma1_.gif)

Рис. 12.The test of multisystem Expert Advisor with the adaptation enabled

**The balance line flattened, the number of dramatic drawdowns decreased, the profit decreased 2.71 times, drawdown decreased 2.32 times, the number of trades decreased 2.68 times,the profit factor increased 1.3 times.**

In the fig. 12, unlike in the previous tests, the diagram of volumes has appeared. It occurred as the result of the active rating system.

It acts in the following manner - at first, the profits Tх.PeriodRating of all closed virtual trades are summed up and the obtained value is divided by the number of days those trades were performed for. The obtained value is added by the cumulative profit of all open positions and divided by the number of days of their being active.

If the obtained value is negative, then it's set equal to zero.This operation is performed for each TS.

The rating system tells one trading system from another using magic numbers assigned to active trading strategies in the Tх.Magic initial parameters. It searches for the most profitable TS by the maximum value of profit and set the 100% rating for this system.

All the other TS are assigned with a rating relatively to the rating of leading TS.In the end, a table with three columns is created.

Example:

| Magic | Profit | Rating, % |
| --- | --- | --- |
| 1 | 9 | 40.9 |
| 2 | 0 | 0.0 |
| 3 | 3 | 13.6 |
| 4 | 10 | 45.5 |
| 5 | 0 | 0.0 |
| 6 | 17 | 77.3 |
| 7 | 0 | 0 |
| 8 | 12 | 54.5 |
| 9 | 0 | 0.0 |
| 10 | 22 | 100.0 |

During the real trading, the obtained rating value is multiplied by the bet amount Tх.lot and is divided by 100. In this case, all the trading systems with the rating greater than zero will take part in the real trading.

To make only the leading trading system take part in the real trading, set the MinRating parameter equal to 100 in the Competition\_v1.0 Exper Advisor.

### Conclusion

The represented methodology doesn't give a 100% guarantee of getting a profit; however, I can tell you that smoothing of the balance curve is assured. Namely - decrease of profit, drawdown and number trades performed are guaranteed! Increase of profit factor - probably.

It's up to you to decide if it's good or not.

**The description of the parameters of the Competition\_v1.0.mq4 Expert Advisor:**

- RatingOn – switch of the rating (if it's disabled, then trades in the file must comply with the real trade)
- FastTest – don't change the file at each change of the array of virtual trades during testing.
- file \- file of virtual trading (appears in the TerminalPath()+"\\experts\\files" folder)
- MinRating – minimum rating (in percentage terms) necessary for opening a real position
- Tх.Enabled– TS switch
- Tх.Magic– magic number
- Tх.lot– maximum volume that used for trading when the rating is 100%
- Tх.Fast– period of fast МА
- Tх.Slow– period of slow МА
- Tх.SL – Stop Loss in points
- Tх.TS1 – one-time trailing stop in points (moves the Stop Loss to a break-even level)
- Tх.TS– trailing stop in points
- Tх.PeriodRating \- period of averaging of the rating (number of trades of history)

**Links to similar subjects:** [http://forum.mql4.com/ru/23455](https://www.mql5.com/ru/forum/118249)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1578](https://www.mql5.com/ru/articles/1578)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1578.zip "Download all attachments in the single ZIP archive")

[Competition\_v1\_0.mq4](https://www.mql5.com/en/articles/download/1578/Competition_v1_0.mq4 "Download Competition_v1_0.mq4")(29.88 KB)

[RealTrend.mqh](https://www.mql5.com/en/articles/download/1578/RealTrend.mqh "Download RealTrend.mqh")(8.17 KB)

[VirtualTrend.mqh](https://www.mql5.com/en/articles/download/1578/VirtualTrend.mqh "Download VirtualTrend.mqh")(36.03 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39586)**
(6)


![morken](https://c.mql5.com/avatar/avatar_na2.png)

**[morken](https://www.mql5.com/en/users/morken)**
\|
7 Oct 2010 at 06:51

Thanks for your sharing!

![morken](https://c.mql5.com/avatar/avatar_na2.png)

**[morken](https://www.mql5.com/en/users/morken)**
\|
7 Oct 2010 at 07:31

A newbie question that how to upload the files correctly?

Competition\_v1.0.mq4 upload to // set up / experts, right?

Then where should I put RealTrend.mqh and VirtualTrend.mqh to?

Thanks

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
29 Jun 2011 at 16:51

This is a great concept and resource. I look forward to becoming competent enough to utilize it.

Give the potential, it is to bad that more people aren't getting involved in this.

![Pankaj D Costa](https://c.mql5.com/avatar/2015/8/55DF458F-577F.JPG)

**[Pankaj D Costa](https://www.mql5.com/en/users/nirob76)**
\|
14 Mar 2015 at 20:07

Thanks for wonderful presentation, its really helpful.

Thanks.

![Bahrom Juraev](https://c.mql5.com/avatar/avatar_na2.png)

**[Bahrom Juraev](https://www.mql5.com/en/users/traderprogramer)**
\|
19 Mar 2015 at 16:09

Is the author available yet ?


![Protect Yourselves, Developers!](https://c.mql5.com/2/17/846_12.gif)[Protect Yourselves, Developers!](https://www.mql5.com/en/articles/1572)

Protection of intellectual property is still a big problem. This article describes the basic principles of MQL4-programs protection. Using these principles you can ensure that results of your developments are not stolen by a thief, or at least to complicate his "work" so much that he will just refuse to do it.

![Adaptive Trading Systems and Their Use in the MetaTrader 5 Client Terminal](https://c.mql5.com/2/0/Adaptive_Expert_Advisor_MQL5__2.png)[Adaptive Trading Systems and Their Use in the MetaTrader 5 Client Terminal](https://www.mql5.com/en/articles/143)

This article suggests a variant of an adaptive system that consists of many strategies, each of which performs its own "virtual" trade operations. Real trading is performed in accordance with the signals of a most profitable strategy at the moment. Thanks to using of the object-oriented approach, classes for working with data and trade classes of the Standard library, the architecture of the system appeared to be simple and scalable; now you can easily create and analyze the adaptive systems that include hundreds of trade strategies.

![Controlling the Slope of Balance Curve During Work of an Expert Advisor](https://c.mql5.com/2/0/Balance_Angle_Control_MQL5.png)[Controlling the Slope of Balance Curve During Work of an Expert Advisor](https://www.mql5.com/en/articles/145)

Finding rules for a trade system and programming them in an Expert Advisor is a half of the job. Somehow, you need to correct the operation of the Expert Advisor as it accumulates the results of trading. This article describes one of approaches, which allows improving performance of an Expert Advisor through creation of a feedback that measures slope of the balance curve.

![Several Ways of Finding a Trend in MQL5](https://c.mql5.com/2/0/Determine_Trend_MQL5.png)[Several Ways of Finding a Trend in MQL5](https://www.mql5.com/en/articles/136)

Any trader would give a lot for opportunity to accurately detect a trend at any given time. Perhaps, this is the Holy Grail that everyone is looking for. In this article we will consider several ways to detect a trend. To be more precise - how to program several classical ways to detect a trend by means of MQL5.

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/1578&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071525163314391462)

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