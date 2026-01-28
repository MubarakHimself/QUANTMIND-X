---
title: How we developed the MetaTrader Signals service and Social Trading
url: https://www.mql5.com/en/articles/1400
categories: Trading
relevance_score: -1
scraped_at: 2026-01-24T14:14:12.421566
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/1400&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083426951483366275)

MetaTrader 4 / Trading


MetaTrader Social Trading appeared in the summer of 2012. That's when
we finally decided to launch a new service for automatic trade copying -
[Trading Signals](https://www.mql5.com/en/signals "MetaTrader Trading Signals Service"). The idea of ​​this project was to make trading a more
widespread phenomenon: the target audience was novice traders with no
experience or special skills. The stringent requirements were set for
the new service: clarity, maximum availability regardless of skills and
knowledge, a transparent operation mechanism and protection of
subscribers.

The deadline for launch was the beginning of October, on the start of
the [Automated Trading Championship 2012](https://championship.mql5.com/2012/en "Automated Trading Championship 2012"). This online competition was
the most ideal venue for introducing and testing the new service,
because it always attracted the attention of a great number of traders.
Indeed, the signals were welcomed by traders: the launch of the service
during the Championship enabled users to copy trading of successful
participants on their own MetaTrader 5 accounts. This was a brand new
feature at the competition.

[![Subscription to the Championship's Signals](https://c.mql5.com/2/13/subscribe_en_small.png)](https://c.mql5.com/2/13/subscribe_en.png)

A month later, in November 2012, support for trading signals appeared in
the MetaTrader 4 platform. Suppliers of signals received access to an
audience of millions of potential investors, and the service began to
gain momentum. At the beginning of November, about 10 signals were added
daily; at the end of the month up to 25 were registered each day.

### Subscriber Security

At the initial stage our task was to create a massive service for
signal distribution, which would protect subscribers from connecting to
unprofitable signals. Trader's protection was the top priority, as we
understood the danger of copy trading, which could lead to loss of
money. So we decided to distribute signals on the principle of quality
rating, which is calculated based on a variety of parameters. The higher
the quality rating, the higher the position of the signal in the list,
and the greater its credibility among potential subscribers. In this
scheme, potentially dangerous signals appear at the end of the list.

However, not only the rating protects users from the danger of
subscribing to a losing signal. Depending on the degree of risk, the
subscription option can be disabled for some signals. In special cases, a
warning message is shown, like 'A large drawdown may occur on the
account again', 'This is a newly opened account, and the trading results
may be of random nature', 'Low trading activity - only 3 trades
detected in the last month', etc.

We also implemented additional security measures: signal suppliers do
not know who their subscribers are (they known only the number of
signal subscribers), every deal carries a unique digital signature,
service developers do not collect personal information of subscribers
and do not have access to their accounts. We have done everything to
ensure security and help you enjoy the service, being confident that
your account is protected.

### Signals Statistics

Much work has been done to improve the statistics of the signals.
When selecting a suitable signal, a trader primarily checks the
statistics carefully to understand how successfully the signal provider
trades and how reliable the signals are. Therefore, it was important to
provide adequate demonstration of trade statistics.

At the start of the Signals service in the autumn of 2012, only the
two basic charts of growth and balance were available. The first one
showed deposit growth in percentage terms calculated based on the
results of trading operations, the second chart showed the amount of
funds on the account without floating unfixed profit of current open
positions.

A little later, in early 2013, we added a new chart of Distribution
that showed the distribution of the symbols and the Sell/Buy ratio. This
allowed the service users to get a better understanding of ​​the
trading strategy of a selected signals vendor.

![Chart of Distribution](https://c.mql5.com/2/13/distribution_en.png)

In the summer of 2013 we added the "Equity" chart to show the account
equity taking into account the current open positions. A few months
later the statistics again were significantly expanded: a vertical line
appeared in the growth and balance charts to divide trading before
connection to the monitoring and after that, color change was introduced
for every non-trade transactions (depositing or withdrawal), a new
option for tracking the best and worst trade series was added to the
statistics. The aim of these changes in statistics was to provide
potential subscribers with a complete picture of the signals offered by
suppliers.

![Monitoring Line on Charts](https://c.mql5.com/2/13/monitoring_line_en.png)

With the same purpose in mind, a little earlier, in January 2013, we
introduced a unique innovation for the entire industry - [visualization \\
on charts](https://www.metatrader5.com/en/releasenotes/terminal/756 "Visualization of Signal's Trading History on the Terminal's Charts"). This new feature made the service even more convenient for
traders, because now they can see the entire history of signal trading
on charts. A single button opens charts of all currency pairs, which
were traded on the signal account. Despite the unique nature of this
service, traders very quickly got used to it and began to perceive it as
an integral part of copy trading. This is a good example of how we set
the standard for excellence in social trading. The main competitor of
MetaQuotes is MetaQuotes, so the development of the service is similar
to a race with ourselves. Where others would have rested on their
laurels of first small achievements, we are always a step ahead and
constantly strive to improve copy trading mechanisms.

![Visualization of Signal's Trading History on the Terminal's Charts](https://c.mql5.com/2/13/metatrader5_signals_visualize_trading_history.png)

Signals in MetaTrader are like a living organism, they evolve in
accordance with the laws dictated by the developers. We are not in
content with the current situation, and we constantly review the quality
ratings, select signal evaluation criteria and change the calculation
formula. These actions are intended to feature only the best and proven
to succeed signals in the top list.

[![](https://c.mql5.com/2/13/signals-en_small.png)](https://c.mql5.com/2/13/signals-en.png)

### Compatibility of trading conditions of the subscriber and the signal provider

Another important aspect of the Signals service is compatibility of
the trading conditions of the subscriber and the signal provider. This
is a critical element for the normal operation of the service, since
differences in conditions may lead to a worse copy quality or even
complete inability to copy trades. If trading conditions are
incompatible, a signal is either invisible to the subscriber, or is in
the list of prohibited subscription. It is important that the subscriber
is always informed of the differences with the provider's trading
conditions, both when subscribing and during each synchronization of the
terminal with the signal server.

![Compatibility of Trading Condition](https://c.mql5.com/2/13/synchronisation_positive_english__1.png)

Over time, we have enhanced this element further. It started as a
comparison of the subscriber's account settings with those of the signal
provider (deposit currency and leverages), then the comparison of
settings of each trading instrument was introduced. From the comparison
of trading conditions in the subscriber's terminal, we then proceeded to
provide this comparison for subscriptions that were performed mql5.com
site. After some time, we added mapping of instruments - now without
defining the matching of an instrument, the service can find the most
appropriate one.

For example, if the subscriber does not have EURUSD,
which is traded by the provider, signals can be copied for EURUSD.m, if
the subscriber has this one. Mapping has greatly expanded the
possibilities of copy trading, as even on different servers of the same
broker names of instruments may vary.

### We continue to improve the Signals Service

We
continue to enhance the Signals service, improve the mechanisms, add
new functions and fix flaws. The MetaTrader Signals Service of 2012 and
the current MetaTrader Signals Service are like two completely different
services. Currently, we are implementing [A Virtual Hosting Cloud](https://www.mql5.com/en/forum/31359 "Vurtual Cloud Hosting")
service which consists of a network of servers to support specific
versions of the MetaTrader client terminal. Traders will need to
complete only 5 steps in order to rent the virtual copy of their
terminal with minimal network latency to their broker's trade server
directly from the MetaTrader client terminal. This will provide round
the clock operation of the terminal where traders copy trades of signal
providers. Furthermore, we are planning to introduce even better
statistics of signals and provide a new option for traders to form their
own portfolio of signals.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1400](https://www.mql5.com/ru/articles/1400)

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
**[Go to discussion](https://www.mql5.com/en/forum/39185)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
14 Jun 2015 at 17:01

thank all lot of the most great article and help newbie trade


![Ellis Chen](https://c.mql5.com/avatar/2021/5/609ACB71-985D.jpeg)

**[Ellis Chen](https://www.mql5.com/en/users/junestormfx)**
\|
6 Feb 2016 at 01:54

well written. Thank you for sharing.


![Indicator for Constructing a Three Line Break Chart](https://c.mql5.com/2/10/logo.png)[Indicator for Constructing a Three Line Break Chart](https://www.mql5.com/en/articles/902)

This article is dedicated to the Three Line Break chart, suggested by Steve Nison in his book "Beyond Candlesticks". The greatest advantage of this chart is that it allows filtering minor fluctuations of a price in relation to the previous movement. We are going to discuss the principle of the chart construction, the code of the indicator and some examples of trading strategies based on it.

![How we developed the MetaTrader Signals service and Social Trading](https://c.mql5.com/2/11/signals_icon.png)[How we developed the MetaTrader Signals service and Social Trading](https://www.mql5.com/en/articles/1100)

We continue to enhance the Signals service, improve the mechanisms, add new functions and fix flaws. The MetaTrader Signals Service of 2012 and the current MetaTrader Signals Service are like two completely different services. Currently, we are implementing A Virtual Hosting Cloud service which consists of a network of servers to support specific versions of the MetaTrader client terminal.

![Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://c.mql5.com/2/11/fundamental_analysis_statistica_MQL5_MetaTrader5.png)[Regression Analysis of the Influence of Macroeconomic Data on Currency Prices Fluctuation](https://www.mql5.com/en/articles/1087)

This article considers the application of multiple regression analysis to macroeconomic statistics. It also gives an insight into the evaluation of the statistics impact on the currency exchange rate fluctuation based on the example of the currency pair EURUSD. Such evaluation allows automating the fundamental analysis which becomes available to even novice traders.

![MQL5 Cookbook - Multi-Currency Expert Advisor and Working with Pending Orders in MQL5](https://c.mql5.com/2/0/Pending-Orders.png)[MQL5 Cookbook - Multi-Currency Expert Advisor and Working with Pending Orders in MQL5](https://www.mql5.com/en/articles/755)

This time we are going to create a multi-currency Expert Advisor with a trading algorithm based on work with the pending orders Buy Stop and Sell Stop. This article considers the following matters: trading in a specified time range, placing/modifying/deleting pending orders, checking if the last position was closed at Take Profit or Stop Loss and control of the deals history for each symbol.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/1400&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083426951483366275)

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