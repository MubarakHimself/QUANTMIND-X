---
title: Trading Strategies
url: https://www.mql5.com/en/articles/1419
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:59:05.520734
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=goawnvsnlcbsgzqofkeymsphsgnqpthm&ssn=1769252343305918363&ssn_dr=0&ssn_sr=0&fv_date=1769252343&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1419&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20Strategies%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925234378179667&fz_uniq=5083256789174065143&sv=2552)

MetaTrader 4 / Trading systems


All categories classifying trading strategies are fully arbitrary.
The classification below is to emphasize the basic differences between
possible approaches to trading.

1. **Following the trend**




The following-the-trend strategy lies in waiting for a certain price
movement followed by opening a position in the same direction. Doing
so, one supposes that the trend will keep moving in the same direction.
When following the trend, one never sells near maximum or buy near
minimum since a significant price movement is needed to signal that the
trend has started. So, using systems of this type, the trader will
always skip the first phase of the price movement and can miss
significant part of profit before the signal to close position comes.
The main issue concerns choice of sensitivity of the trend-following
strategy. A sensitive system that quickly responds to signs of trend
change work more efficiently during strong trends, but generate much
more false signals. A non-sensitive system will have a reverse set of
characteristics.

Many traders try again and again to earn money on every movement of
the market. This results in choosing of faster and faster systems
following the trend. Although on some markets quick systems are usually
more efficient than slow ones, on the most markets it is quite opposite
since minimizing of losing trades and commissions in slower systems
more than pays the reduced profits at good trades. This is why it is
recommended to limit the natural effort to search for more sensitive
systems. In all cases, the choice between quick and slow systems should
be based on experiences and individual intentions of the trader.
There is a great variety of trend-following strategies available.

Below are the main strategies of the kind:


   - **Strategies based on moving average**



     When an up-trend is
     replaced with the down-trend, prices must intersect the moving average
     top-down.
     Similarly, when the down-trend is replaced with the up-trend, prices
     must intersect the moving average bottom-up. In the most moving-average
     systems, these cross points are considered as trade signals.



     ![](https://c.mql5.com/2/14/moving_average_strategy_2.gif)

   - **Break-through strategy**



     The basic conception
     underlying the break-through strategy is rather simple: the market
     ability to reach a new maximum or minimum shows potential trend in the
     breakthrough direction.



     ![](https://c.mql5.com/2/14/break_support_strategy_1.gif)


2. **Against-the-trend strategies**




Against-the-trend strategies are based on waiting for a significant
price movement followed by opening a position in the opposite
direction, assuming that the market will start correction.
Systems working against the trend are often attractive for many traders
since they are aimed at buying at minimum and selling at maximum.
Unfortunately, the solving complexity of this task is inversely
proportional to attraction of such systems. The most important
difference to be remembered is that the trend-following systems are
self-correcting and against-the-trend systems implicate possibility of
unlimited losses. Thus, it is necessary to include protecting stops in
any against-the-trend system. Otherwise, the system may keep a long
position during the entire large-scaled down-trend or a short position
during the entire large-scaled up-trend.



The
prime advantage of against-the-trend systems consists in that they give
a great diversification opportunity when simultaneously used
together with trend-following systems. Related to this, it must be
noted that an against-the-trend system can be desirable if even it
loses money moderately. The reason for this is that, if an
against-the-trend system is oppositely correlated with a
trend-following system, trading with both systems bears fewer risks
than trading with only one of them. Thus, it is highly possible that
combination of these two systems can earn more at the same risk level
if even the against-the-trend system itself loses money.



![](https://c.mql5.com/2/14/versus_trend_strategy.gif)

3. **Model recognition of price behavior**




All systems can, in some sense, be classified as systems of model
recognition. Finally, conditions that give a signal to open position in
or against the trend direction are a kind of price models, too.
Neverthless, this means that the chosen models are not primarily based
on price movements in certain directions as it is in case of
trend-following or against-the-trend systems.

Systems of this type can sometimes use probable models when making
trade decisions. In this case, researchers will try to identify models
that, according to their behavior, were supposed to precede price
increases or decreases. Such behavior models are considered to be used
for assessment of the current probabilities of the market growth or
fall.

It must be noted that the above strategies are not always clearly separated from each other.
Being modified, the systems can be classified as of another type.



![](https://c.mql5.com/2/14/graph_models_strategy.gif)

4. **Trading in channel**




Trading in channel represents trading up and down from
resistance/support levels, lines of which are the channel borders. Such
tactics are good for sideways trends (flats), but are not practically
applicable in up-trends or down-trends. Trading in channel is shown in
a chart below:



![](https://c.mql5.com/2/14/channel_strategy.gif)



Positions should be opened under the following rules:


   - Determine support/resistance levels. A correct computation will
     help to have clear borders of the channel, in which the market moves.

   - As
     soon as the price reaches a border of the channel and jumps back in the
     opposite direction, a buy position should be opened. Short positions
     should be opened if prices reach the resistance level.

   - As
     soon as the price reaches the opposite border, the position should be
     closed. It must be noted that reversal can happen before the price line
     reaches the channel borders, so positions can be closed before the
     price reaches levels of support or resistance.


The advantage of such tactics is possible maximization of profit
through opening and closing of positions several times if the sideways
trend continues. The main disadvantage thereof is that the break
thourgh the channel lines can result in significant and undjustified
losses. To avoid the latter ones, it is necessary to set Stop Loss
correctly that losing positions are closed if the market moves in an
opposite direction compared to the planned one.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1419](https://www.mql5.com/ru/articles/1419)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Ten Basic Errors of a Newcomer in Trading](https://www.mql5.com/en/articles/1418)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39231)**
(5)


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
28 Nov 2006 at 14:11

Could you mind to give a example for each type ?

That will a great Work.


![Vlad Vahnovanu](https://c.mql5.com/avatar/avatar_na2.png)

**[Vlad Vahnovanu](https://www.mql5.com/en/users/vladv)**
\|
30 Nov 2006 at 13:26

That's very interesting!Thanks.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
2 Apr 2007 at 15:48

Hi,

Is there a script for plotting the resistance and support lines on the candlestick
chart to help us with entry and exit points?

Regards

Ben


![Vlad Vahnovanu](https://c.mql5.com/avatar/avatar_na2.png)

**[Vlad Vahnovanu](https://www.mql5.com/en/users/vladv)**
\|
17 Aug 2007 at 11:10

This article is great!Thanks.


![Viraj B](https://c.mql5.com/avatar/2019/1/5C387A74-F5E2.png)

**[Viraj B](https://www.mql5.com/en/users/virajb)**
\|
7 Feb 2019 at 09:09

I agree with the comment to share examples. I'm a learner in Algo Trading, and this would be really helpful to me.

Though I'm still doing a course on Algo Quant [Trading Strategies](https://www.mql5.com/en/articles/3074 "Article: Comparative Analysis of 10 Trending Strategies ") by Quantra, I'm also doing some side by side research to keep learning. Articles like yours make a lot of difference.

Thank you.

![Orders Management - It's Simple](https://c.mql5.com/2/13/126_1.gif)[Orders Management - It's Simple](https://www.mql5.com/en/articles/1404)

The article deals with various ways of how to control open positions and pending orders. It is devoted to simplifying of writing Expert Advisors.

![Ten Basic Errors of a Newcomer in Trading](https://c.mql5.com/2/13/173_1.png)[Ten Basic Errors of a Newcomer in Trading](https://www.mql5.com/en/articles/1418)

There are ten basic errors of a newcomer intrading: trading at market opening, undue hurry in taking profit, adding of lots in a losing position, closing positions starting with the best one, revenge, the most preferable positions, trading by the principle of 'bought for ever', closing of a profitable strategic position on the first day, closing of a position when alerted to open an opposite position, doubts.

![Secrets of MetaTrader 4 Client Terminal: File Library in MetaEditor](https://c.mql5.com/2/14/211_2.gif)[Secrets of MetaTrader 4 Client Terminal: File Library in MetaEditor](https://www.mql5.com/en/articles/1430)

When creating custom programs, code editor is of great importance. The more functions are available in the editor, the faster and more convenient is creation of the program. Many programs are created on basis of an already existing code. Do you use an indicator or a script that does not fully suit your purposes? Download the code of this program from our website and customize it for yourselves.

![Multiple Null Bar Re-Count in Some Indicators](https://c.mql5.com/2/13/139_6.png)[Multiple Null Bar Re-Count in Some Indicators](https://www.mql5.com/en/articles/1411)

The article is concerned with the problem of re-counting of the indicator value in the MetaTrader 4 Client Terminal when the null bar changes. It outlines general idea of how to add to the indicator code some extra program items that allow to restore program code saved before multiple re-counting.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tbbrpcqvxeignbuuxmadrrwrknhkfahf&ssn=1769252343305918363&ssn_dr=0&ssn_sr=0&fv_date=1769252343&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1419&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20Strategies%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925234378157538&fz_uniq=5083256789174065143&sv=2552)

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