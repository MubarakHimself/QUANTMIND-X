---
title: Take a few lessons from Prop Firms (Part 1) — An introduction
url: https://www.mql5.com/en/articles/11850
categories: Trading
relevance_score: -5
scraped_at: 2026-01-24T14:18:00.777907
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/11850&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083474213303491687)

MetaTrader 5 / Trading


### Table of contents

01. [Preamble](https://www.mql5.com/en/articles/11850#preamble)
02. [Challenges](https://www.mql5.com/en/articles/11850#challenges)
03. [Absolute drawdown](https://www.mql5.com/en/articles/11850#absolute_drawdown)
04. [Daily drawdown](https://www.mql5.com/en/articles/11850#daily_drawdown)
05. [Trailing drawdown](https://www.mql5.com/en/articles/11850#trailing_drawdown)
06. [Profit target](https://www.mql5.com/en/articles/11850#profit_target)
07. [Stop-loss](https://www.mql5.com/en/articles/11850#stop_loss)
08. [Risk and volume](https://www.mql5.com/en/articles/11850#risk_volume)
09. [News and weekends](https://www.mql5.com/en/articles/11850#news_weekends)
10. [Martingale, grid and hedging](https://www.mql5.com/en/articles/11850#martingale_grid_hedging)
11. [MT5 vs. MT4](https://www.mql5.com/en/articles/11850#MT5_MT4)
12. [Conclusion](https://www.mql5.com/en/articles/11850#conclusion)

### 1\. Preamble

Many a trader has questioned themselves on whether they can succeed at being a trader in the long run. Often, they are unsure of how to evaluate themselves objectively. One has to profit consistently to succeed, but is that enough? One way to evaluate our metrics is to compare them to that of other successful traders, but that may not be possible, as the truly successful traders earning real money rarely flaunt their results—they keep that information to themselves.

Comparing one’s metrics to the hordes of online signals or pseudo-traders advertising their success on _YouTube_ and the likes is a waste of time, as there is no reliable way to truly validate their results. There is one source of information, however, that one can rely on, even if it is more on the intense end of what makes up successful trading, and that is “prop firm” requirements.

_Proprietary trading firms_ are financial institutions that fund freelance traders while splitting the profits. They offer a great way for good traders to increase their earnings. They also give traders a sense of accomplishment for being recognised when accepted for funding.

They implement a set number of rules and requirements, including minimum metrics for the funded traders, to guarantee continual and consistent returns for both parties. They have a vested interest in evaluating and maintaining a prolonged relationship with successful traders, so their requirements serve as a good comparison to evaluate one’s metrics.

I have researched several well-known prop firms and compiled a summary of the most common requirements imposed by most of them. This initial article will focus on describing these requirements, while future articles will focus on how to implement them in _**MQL**_ programs. For now, use these as guidelines to evaluate your manual or algorithmic trading, and hopefully, they can help you become a consistent and successful trader.

![Prop firm trading](https://c.mql5.com/2/54/Prop_Firm_Trading.jpg)

### 2\. Challenges

Many prop firms require candidate traders to first overcome certain “challenges” which evaluate the candidate’s abilities. One can consider this as one’s ability to first become a proficient trader using a demo account.

It is extremely important that one can succeed at “paper” trading before one can trade with real money. If you can’t succeed with a demo account consistently, then don’t even think of opening up a real money account, or else you will only lose it all.

So learn this lesson from prop firms. Challenge yourself first, and if you can’t make it, don’t trade with real money. Continue developing your skills and improving your knowledge until you do.

### 3\. Absolute drawdown

Every prop firm I researched had this rule—one can’t draw down more than a maximum percentage of your initial capital balance. Breaking this rule immediately expels one from the prop firm program and one has to start all over again with their evaluation process.

Depending on the prop firm and the type of program, this percentage varied between 4-12%. For example, let’s say one starts with a balance of $10000 and the rule is a maximum absolute drawdown of 8%, then your equity can never go below 92% of your starting balance, namely one’s balance can’t drop below $9200.

Use this rule with your trading to protect your invested capital. Define a low watermark and never let it ever go below that. If you fail at this, then you are risking too much with your trades. Managing your risk is the key to your success as a trader. Minimise your risk by adjusting your position size (order volume) so that even if you experience a streak of losing trades, your drawdown stays minimal.

Don’t be reckless. Protect your capital. Without it, you cannot trade. It is your life’s savings. Don’t squander it.

[![Drawdown calculations](https://c.mql5.com/2/54/drawdown.png)](https://www.metatrader5.com/en/terminal/help/trading_advanced/history_report#drawdown "https://www.metatrader5.com/en/terminal/help/trading_advanced/history_report#drawdown")

### 4\. Daily drawdown

Most prop firms I researched had this rule—one can’t draw down more than a static value or a set percentage of the day’s opening balance or equity (whichever is highest). The static value or the set percentage was approximately half of the absolute drawdown (described in the previous section) for many of the prop firms I researched. This means that two consecutive days of high drawdown could violate the absolute drawdown limit. Breaking either rule expels one from the prop firm program and one has to start all over again with their evaluation process.

Unless you are a long-term swing trader, take this rule to heart, especially if you are a day trader. You too can apply it to your daily trading to prevent you from over-trading or going on a “revenge trading” spree. When you reach the set limit, close out all of your trades and stop trading for the day. Take the rest of the day off to calm your mind and relax. Trading while stressed will not benefit you. Allow yourself to start the next trading day with a fresh view of things.

### 5\. Trailing drawdown

A few prop firms simplify things by only having a single easy-to-follow drawdown rule—a trailing drawdown rule based on the highest watermark level of balance (or equity). This is effectively what one usually refers to as the _Maximum Relative Drawdown_.

Apply this to your trading. Define a maximum relative drawdown limit based on the percentages used by prop firms as a guide to restrain your trading activity and to help evaluate your strategy and trading skills. Adjust your risk management and profit targets and aim to never violate this limit.

This rule perfectly fits the needs of successful traders that regularly withdraw a part of their earnings, while steadily maintaining and growing their capital balance.

### 6\. Profit target

Most of the prop firms I researched had profit targets that were on the same level as the mandatory drawdown limits, but never more than twice that. The required monthly profit target for the first phase of the evaluation process was usually higher than the absolute drawdown allowed, but during subsequent phases, they reduced the monthly target, bringing it to the same level in value or percentage as the daily drawdown limit.

Learn a lesson from these prop firms. If you want to succeed, don’t aim for unrealistic returns. A small, slow and steady return is much more beneficial than big flashy and risky gains.

Don’t fall for the trap of thinking you can easily double your capital in just a month. That high-risk mentality can easily cause you to lose all your capital just as easily.

Take it slow, and observe good risk management techniques.

### 7\. Stop-loss

A few of the prop firms I researched had a mandatory stop-loss rule for several of their programs. Even those that did not impose the rule still highly emphasised the need to always use a hard broker-side stop-loss with every trade.

Yes, I know that many _pseudo_ traders will tell you not to use a stop-loss to prevent “stop hunting” by brokers, but then I say—don’t use questionable brokers. Use reputable non-dealing-desk brokers that offer _Straight Through Processing (STP)_ or _Electronic Communication Network (ECN)_ accounts. Check with the regulatory service and verify your broker’s license. Do your research before selecting a broker.

If prop firms can have long-term success stressing this point, then so can you. Don’t use excuses. Use a stop-loss. It’s part of risk management and without it, you can’t properly calculate your order volume or assess your risk and prevent large drawdowns.

Here are a few quotes from prop firms:

- _“Rule: All trades must have a Valid Stop Loss attached to every trade at entry into the market. All professional traders attach a Stop Loss to all their trades. Without a Stop Loss, your whole trading account balance is at risk. It makes no sense to trade without a Stop Loss, and without one, the trading account has a very limited life. “_

- _“It is not mandatory to have a stop-loss on your positions for any of the programs. However, we highly advise you to trade with a stop-loss to manage risk effectively. “_

- _“For \*\*\* funded accounts, if you wish to claim a 50% split upon violation in profits, it must have a stop-loss in every trade. “_

- _“We’re sure you’ve heard of ‘stop-loss hunting’ because the big players know where retail traders have their pending orders set. A few losing trades where the market blew you out exactly at your SL and immediately turned back will have confirmed this assumption for you. In reality, however, it could easily have been a badly set SL. Forex is such an enormous market that manipulating the price is virtually impossible. So while the big players are aware of pending orders, retail traders should instead adjust their SL and TP to the market action and not speculate on market manipulation. “_

- _“… we get quite a few traders who don’t use any stop-loss method. Having reviewed their performance, we can tell you firsthand that they often end up deep, sometimes for days or weeks, in a drawdown before they recover or break even. “_

- _“Having a stop-loss in place is important for any trader. The volatile nature of the markets means they can move quickly and unexpectedly. By placing a stop-loss, you are protecting yourself from large losses should the market turn against your position. “_


![Cut losses](https://c.mql5.com/2/54/cut_losses.png)

### 8\. Risk and volume

Most of the prop firms I researched limited the maximum allowed volume per trade (position size) based on the initial capital balance, either by a set maximum volume (lots) or by the percentage risk of the stop-loss size.

For those using a risk percentage-based limit, it was usually a 1-2% maximum risk per trade, but a few allow for up to 2-5% on the more aggressive account types, which had much lower leverage to compensate for the higher risk allowance.

A few also imposed an aggregate volume limit where the total risk of all open positions could not exceed a set percentage (for example, 3%).

So again, note how prop firms limit their risk and apply this attitude to your trading, learning how to define your position size properly. Limit your stop-loss risk to only 1-2%, especially if you are trading with high leverage. If you are trading multiple symbols simultaneously, lower the risk even more so that your overall exposure does not exceed 2-5%.

Don’t base your position size (volume) only on margin requirements. Consider your stop-loss risk first, and foremost, then reduce the volume if necessary, according to the margin requirement limits you have set.

### 9\. News and weekends

Some of the prop firms I researched did not allow trading during major news events, nor allowed positions to remain open during the weekend. Most allowed positions to remain open overnight but cautioned traders about the costs of swaps and the widening spreads while holding overnight positions.

Some prop firms allowed weekend trading but cautioned against open price gaps that could cause the drawdown rules to be violated. However, they removed these weekend restrictions on cryptocurrencies, since many are 24/7 markets.

The prop firms that allowed unrestricted news trading mostly imposed accounts with lower leverage or lowered the profit-sharing split to compensate for the higher risk that news trading entails.

So, unless you are a long-term swing trader or are trading cryptocurrencies or synthetics, consider closing all your trades before the market closes for the weekend. Avoid unexpected events and the impact weekend news can have on your trading.

Also, during trading days, do your best to be flat during major news events. Even if you believe you know what the results will be, human behaviour can, unfortunately, be very unpredictable and the traders’ reactions to those news events can be the opposite of the expected norm.

Do your best to be out of the market during the weekend and major news events. Close your trades a few minutes before the event, on the possibly affected symbols. Then, a few minutes (or longer) after the event, reevaluate the conditions of your strategy rules to decide if you should open positions again or not. If you are an algorithmic trader, consider adding a news filter to your code.

[![MQL5 Economic Calendar](https://www.metatrader5.com/c/17/0/economic-calendar-fbready__1.png)](https://www.mql5.com/en/economic-calendar)

### 10\. Martingale, grid and hedging

Do we _**really**_ need to discuss these?

If you want to be a successful trader and be “professional” about it, then these amateur methods are out of the question.

Every single prop firm I researched outright banned the use of _Martingale_ and grid-like methods, and for good reason—not only would such methods quickly violate the drawdown rules, but it would also put their entire funding at risk.

As for same-symbol hedging, some allowed it and others not. Those that allowed it cautioned traders of the extra trading costs inherent to such practices.

Here are a few quotes from the prop firms, but they all have similar statements in the rules:

- _“Example Strategies That Violate Our Rules: Grid Trading, Martingale Trading, High-Frequency Trading, Hedging,…”_

- _“Four Trading Strategies You Need To Avoid As A Funded Forex Trader: Martingale, Grid, News, No Strategy. ”_

- _“Martingale strategies can prove some significant results in the short term. However, when you hit a losing streak, you’re going to ruin your trading account balance and most likely lose your funded trading account. Utilise proper risk management, rather than Martingale. “_

- _“The problem with methods such as the Kelly Criterion and the Martingale Method is they often assume unlimited risk capital. “_

- _“… every Martingale trader experiences a time when they are severely stuck, and this is a perfect situation for blowing up an account. “_


![Martingal Graph](https://c.mql5.com/2/54/Martingale_Colour_Graphic.png)

### 11\. MT5 vs. MT4

[_MetaTrader 5_ or _MetaTrader 4_?](https://www.metatrader5.com/en/trading-platform/comparison-mt5-mt4 "https://www.metatrader5.com/en/trading-platform/comparison-mt5-mt4")

That is the question!

Almost all the prop firms I researched supported both _MetaTrader 5_ and _MetaTrader 4_ platforms, while very few only supported _MetaTrader 4_ and not _5_.

The prop firms supporting both platforms were more modern and offered better trading conditions and more market types, while those only supporting the old platform were more _old-school_ and focused mostly only on Forex, but some did offer Indices and Commodities as well.

So, there is no reason to not move up in the world with the newer _MetaTrader 5_ platform which offers much better conditions to test your strategies with real tick data, as well as offering you the opportunity to trade other markets not available via the older platform, which has had no active development done on it for several years now.

### 12\. Conclusion

There is much to learn from _proprietary trading firms._

Their challenges and their trading rules serve as a good basis to evaluate one’s trading on one’s way to a successful outcome as a retail trader.

Take a few lessons from “Prop Firms” and adapt their ways as your own.

Do not falter! Put in the effort and apply discipline to your trading. Be methodical about it. The hard work will pay off, so invest in your knowledge and your skill.

Look forward to the next part, where I delve into the code implementations.

### For now, I wish you good trading!

![Good trading!](https://c.mql5.com/2/54/Good_Trading.wide.jpg)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/445895)**
(46)


![Alexey Volchanskiy](https://c.mql5.com/avatar/2018/8/5B70B603-444A.png)

**[Alexey Volchanskiy](https://www.mql5.com/en/users/vdev)**
\|
3 Feb 2024 at 01:15

I reread the article again. Why does the author confuse balance and equity? Quote:

\- ... The maximum drawdown should not exceed the maximum percentage of the initial balance.

Depending on the company and type of programme, the percentage varies between 4-12%. Let's assume your initial balance is USD 10,000. The maximum absolute drawdown is 8%. In this case, your equity cannot fall below 92% of your initial balance. In other words, your balance  cannot fall below USD 9200.

...Determine a [lower limit](https://www.mql5.com/en/docs/constants/indicatorconstants/lines "MQL5 Documentation: Indicator Lines") and never allow equity to fall below this level.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
3 Feb 2024 at 01:46

**Alexey Volchanskiy lower limit and never allow equity to fall below this level.**

There is no misunderstanding. Many proprietary firms specifically define a maximum drawdown based on the initial balance and ignore any subsequent drawdown based on equity.

Some other proprietary firms have different rules and apply a rolling drawdown based on equity rather than the initial balance.

So no, the author, that is me, didn't get anything wrong.

However, there is a "typographical" error here and it should have read "Namely, your equity or balance cannot fall below $9200."

And by the way, I'm not Brazilian (I'm from Portugal) and the original article was written in English (it's my first language as I lived in an English-speaking country for 20 years in my youth).

(automatic translation from English is attached)

> _There is no misunderstanding. Many prop-firms specifically define the maximum drawdown based on the initial balance and ignore any trailing drawdown based on equity._
>
> _Some other prop-firms have different rules and apply a trailing drawdown based on equity instead of initial balance._
>
> _So no, the author, that is me, confused nothing._
>
> _However, there is a "typographical" error and it should have been "namely one's equity or balance can't drop below $9200"._
>
> _And by the way, I'm not Brazilian (I'm from Portugal) and the original article was written in English (which is my first language as a lived in an English-speaking country for 20 years in my own youth)._

EDIT: Also, ChatGPT did not write this. I take pride in what I do myself, whether it's writing or programming. And I really hate it when people use ChatGPT to do it. If that's not interesting enough for you, then don't read or comment.

> _EDIT: Also, ChatGPT did not write it. I pride myself for doing things myself, be it writing or programming. And I actually hate when people use ChatGPT for that. If it is not interesting enough for you, then don't read or comment on it._

![Artem Chyvelev](https://c.mql5.com/avatar/2023/3/641cc2a5-182f.jpg)

**[Artem Chyvelev](https://www.mql5.com/en/users/chuvels)**
\|
15 Feb 2024 at 11:49

The truth is that 90% (maybe even all 100) of these forex prop companies are fake. They themselves admit that after passing the challenges a trader does not get a [real account](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_trade_mode "MQL5 Documentation: Account Information"), but trades on a demo and if he does not violate the conditions, he gets a reward. They are sort of designed as a training platform, not as a broker or forex brokerage centre. Accordingly, the winnings of traders who fulfil the conditions are paid out of the money of those who did not pass the selection, according to the principle of a financial pyramid.

![Artem Chyvelev](https://c.mql5.com/avatar/2023/3/641cc2a5-182f.jpg)

**[Artem Chyvelev](https://www.mql5.com/en/users/chuvels)**
\|
28 Feb 2024 at 13:26

Chronology of the development of "prop forex companies" in one month, even Metaquotes got involved

[![](https://c.mql5.com/3/430/image__1.jpg)](https://c.mql5.com/3/430/image.jpg "https://c.mql5.com/3/430/image.jpg")

![rohana manjula](https://c.mql5.com/avatar/2024/6/6676f11c-1b2a.jpg)

**[rohana manjula](https://www.mql5.com/en/users/rohanamanjula)**
\|
9 Jun 2024 at 13:58

Great article. Good job!

![Population optimization algorithms: Monkey algorithm (MA)](https://c.mql5.com/2/52/monkey_avatar.png)[Population optimization algorithms: Monkey algorithm (MA)](https://www.mql5.com/en/articles/12212)

In this article, I will consider the Monkey Algorithm (MA) optimization algorithm. The ability of these animals to overcome difficult obstacles and get to the most inaccessible tree tops formed the basis of the idea of the MA algorithm.

![Population optimization algorithms: Harmony Search (HS)](https://c.mql5.com/2/51/Avatar_Harmony_Search.png)[Population optimization algorithms: Harmony Search (HS)](https://www.mql5.com/en/articles/12163)

In the current article, I will study and test the most powerful optimization algorithm - harmonic search (HS) inspired by the process of finding the perfect sound harmony. So what algorithm is now the leader in our rating?

![Neural networks made easy (Part 36): Relational Reinforcement Learning](https://c.mql5.com/2/52/Neural_Networks_Made_036_avatar.png)[Neural networks made easy (Part 36): Relational Reinforcement Learning](https://www.mql5.com/en/articles/11876)

In the reinforcement learning models we discussed in previous article, we used various variants of convolutional networks that are able to identify various objects in the original data. The main advantage of convolutional networks is the ability to identify objects regardless of their location. At the same time, convolutional networks do not always perform well when there are various deformations of objects and noise. These are the issues which the relational model can solve.

![How to detect trends and chart patterns using MQL5](https://c.mql5.com/2/53/detect_trends_chart_patterns_avatar.png)[How to detect trends and chart patterns using MQL5](https://www.mql5.com/en/articles/12479)

In this article, we will provide a method to detect price actions patterns automatically by MQL5, like trends (Uptrend, Downtrend, Sideways), Chart patterns (Double Tops, Double Bottoms).

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11850&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083474213303491687)

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