---
title: The Easy Way to Evaluate a Signal: Trading Activity, Drawdown/Load and MFE/MAE Distribution Charts
url: https://www.mql5.com/en/articles/2704
categories: Trading, Trading Systems
relevance_score: 1
scraped_at: 2026-01-23T21:36:00.070741
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/2704&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071955342943793550)

MetaTrader 5 / Trading


Subscribers often search for an appropriate signal by analyzing the total growth on the signal provider's account, which is not a bad idea. However, it is also important to analyze potential risks of a particular trading strategy. In this article we will show a simple and efficient way to evaluate a Trading Signal based on its performance values:

- [Trading Activity](https://www.mql5.com/en/articles/2704#activity)
- [Drawdown Chart](https://www.mql5.com/en/articles/2704#drawdown)
- [Deposit Load Chart](https://www.mql5.com/en/articles/2704#load)
- [MFE and MAE Distribution](https://www.mql5.com/en/articles/2704#distributions)

### Trading Activity

Trading on a particular day and hour is determined by the rules of a trading strategy. Long-term strategies may open positions once a week or even less than that, and the position lifetime can be up to weeks or months. The lifetime of scalping positions can be from a few minutes to several hours, and such trades can be performed several times a day.

![](https://c.mql5.com/2/24/trade_activity.png)

Trading activity is displayed as the percentage of the total time when there were open positions on the account. If this value approaches 100%, it means the account almost always has open positions, and therefore the subscriber's deposit will be constantly exposed to the risk of a sudden loss. For example, remember the recent United Kingdom European Union membership referendum (Brexit), after which the British pound fell sharply for a few minutes, or the decision of the Swiss Central Bank to abandon the franc's exchange rate peg against EUR, after which the Swiss franc soared as much as 30% in less than 20 minutes. Some investors managed to make a profit on these events, but those who guessed the direction wrong suffered huge losses. Thus, an extremely high trading activity can ruin your deposit in case of strong market movements, and this is a substantial risk.

What trading strategies produce high activity? Such strategies include martingale-based systems and grid strategies with averaging techniques, as well as arbitrage on currency baskets, reversal systems, etc. A high trading activity is not always bad, but you should carefully analyze HOW the provider trades. If a strategy targets profit by closing multiple positions of an instrument at the same time, the signal provider might be trying to average wrong market entries. This approach can be efficient for quite a long time, and the signal may show a steady balance increase over months. But finally your account balance my fail to cover a huge drawdown, and trading will be closed in a natural way by a Stop Out.

On the other hand, a very low trading activity (less than 2-5%) indicates that the signal provider enters the market for a very short period of time and immediately exits it while taking profit or loss. It seems to be good tactics at first glance. But there is another risk for a subscriber, since the provider's trades can be missed, or they can be copied with a large slippage. As a result, the signal provider will have a profit, while the subscriber will have a losing trade. Check out the average position holding time and statistics of slippage between the provider's server and your broker. You should be prepared for possible negative results.

Therefore, try to find a "golden mean" when the provider does not trade too often by opening short-time minor trades, and at the same avoids being all the time in the market. An ideal signal is the one that trades several different financial instruments (applying diversification by symbols), but without additional volume increases or increased position holding time for losing positions. Such strategies may compensate losses on one symbol through winning trades on other instruments. Use the filter in order to see signals with the desired trading activity range.

![](https://c.mql5.com/2/24/filter_activity.png)

### Drawdown Chart

The account balance and equity values of each signal are monitored since the account registration in the Signals service. The difference between these values ​​is positive, if currently open positions show a floating profit. But if the equity is less than the balance value, it means that the trading account is having a drawdown or unrecorded loss.

![](https://c.mql5.com/2/24/drawdown.png)

Review the chart of drawdown over the entire signal monitoring period in order to understand what was the risk on the account before a position was closed with a profit.

For example, in this case we see that the profit over the trading account lifetime was 18% to 76% per month, while the floating drawdown reached more than 16% in May. Compare the account drawdown and profit to decide if you are ready to lose 18% (or multiply it by 3, that is 18 \* 3 = 54%) of the deposit if the same drawdown happens on the provider's account again.

![](https://c.mql5.com/2/24/growth.png)

### Deposit Load Chart

The deposit load value shows the percent of account funds used to open positions. Load calculation formula:

Load = Margin / Equity \* 100%

If the current account value is 10,000 USD, and the margin of open positions is 5,000 USD, the account load is 50% = 5,000 USD / 10,000 USD \* 100%. Larger traded lot values cause larger equity fluctuations on the provider's account in case of market price changes, resulting in a larger load on the trading account. In other words, the higher the load on the account, the higher the risks.

Margin depends on the account leverage, which means that the load on an account with the 1:500 leverage will be 5 times less than that on the account with the 1:100 leverage. But the risk will be the same. So in addition to the load value, you should also pay attention to the provider's account leverage. We recommend you to read the [Forex Trading ABC](https://www.mql5.com/en/articles/1453) article, which explains the dependence of margin value on the account leverage.

Margin (in the symbol base currency) = Lot value (in the symbol base currency) / Leverage\_value

For example, if a provider trades with the 1:500 leverage, and the recorded deposit load reaches 40% or more, a subscriber with the leverage of 1:100 would have a load of over 200% (40%\*(1:100/1:500) = 40% \* 5). What does it means, what are additional risks for signal subscribers in this case? The subscriber's funds may not be enough to cover a drawdown, and all losing positions may be closed by a Stop Out. Meanwhile, the signal provider can be able hold the position and close it with a zero loss or even with a profit.

![](https://c.mql5.com/2/24/risks.png)

Thus, a high load on a provider's trading account can mean an increased risk for a subscriber and can drain their account. Every signal is provided with the Growth tab, on which you can check the maximum load recorded during the account monitoring time.

![](https://c.mql5.com/2/24/max_deposit_load.png)

You can reduce the load on the deposit by additionally limiting the traded deposit percentage in your trading terminal: [Use no more than % of deposit](https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings "https://www.metatrader5.com/en/terminal/help/signals/signal_subscriber#settings") parameter. This option can protect your account from huge losses, especially in cases where the provider and the subscriber have different amounts of deposit and leverage. For example, a subscriber may choose to limit the percentage of the deposit load to 30%. In this case trade volumes are calculated automatically.

![](https://c.mql5.com/2/24/options.png)

### MFE and MAE Distribution

The Risks section features the deposit load chart, and it additionally contains MFE and MAE distribution graphs. These graphs describe the characteristics of every closed position during its lifetime. The detailed explanations of these characteristics are available in the [Mathematics in Trading. How to Estimate Trade Results](https://www.mql5.com/en/articles/1492) article. Here we only show you how to understand the provider's trading style in 1 second by analyzing these clusters.

Green points in the upper quadrant of the MFE graph mark the trades that had a floating profit larger than the one recorded during position closure. It means that for this position Take Profit was not used in order to lock in profits. Thus, a large number of such green dots in the upper right corner of the MFE graph means that trading is based on the "Let profits run" principle. It is a common attribute of trending strategies.

![](https://c.mql5.com/2/24/MAE_MFE.png)

The same applies to the MAE graph – a large percentage of red dots in the lower left corner means that a provider is inclined to "outstay" losses. A graph with a large number of red dots suggests that the provider prefers not use Stop Loss thus breaking the first trading rule "Cut losses, let your profits run".

These distribution graphs help you understand the provider's trading style:

- too many **green** points on the MFE graph – Take Profit orders are not used, the **provider lets the profit run** upon a proper market entry;
- too many **red** points on the MAE graph – Stop Loss orders are not used, the **provider does not limit losses** in case of an unsuccessful entry;

### Quick General Evaluation of a Signal

Based on the four described criteria, you can determine the signal provider's trading style. Low trading activity (<5%) may be problematic for signal copying, and high trading activity exposes the subscriber's account to a constant risk.

Higher drawdown values ​​may be the other side of a stable profitability growth, which can be achieved through longer losing position holding period or by adding volumes in case of wrong market entries.

High deposit load, especially when combined with a high leverage on the provider's account, may also lead to losses on subscribers' accounts with a lower leverage.

MFE and MAE distribution graphs point to the use of Stop Loss and Take Profit orders by the signal provider. We can generally derive the following rule:

**A smooth and even growth chart of a signal is usually accompanied by highly uneven drawdown and load charts.**

Every Signal featured in the showcase is provided with extensive statistical information. Be sure to carefully analyze the provided data when choosing a trading strategy to copy.

**Related articles:**

- [How to Subscribe to Trading Signals](https://www.mql5.com/en/articles/523)
- [General Information on Trading Signals for MetaTrader 4 and MetaTrader 5](https://www.mql5.com/en/articles/618)
- [How to Prepare a Trading Account for Migration to Virtual Hosting](https://www.mql5.com/en/articles/994)
- [Tips for Selecting a Trading Signal to Subscribe. Step-By-Step Guide](https://www.mql5.com/en/articles/1838)


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2704](https://www.mql5.com/ru/articles/2704)

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
**[Go to discussion](https://www.mql5.com/en/forum/96593)**
(14)


![Walleee](https://c.mql5.com/avatar/avatar_na2.png)

**[Walleee](https://www.mql5.com/en/users/walleee)**
\|
30 Oct 2017 at 04:29

Thank you for a very useful article. However, I couldn't understand the drawdown chart section. Can anyone helps me please? 1) "Review the chart of drawdown over the entire signal monitoring period in order to understand what was the risk on the account before a position was closed with a profit." How can I determine the potential risk from the chart? 2) "For example, in this case we see that the profit over the trading account lifetime was 18% to 76% per month, while the floating drawdown reached more than 16% in May. Compare the account drawdown and profit to decide if you are ready to lose 18% (or multiply it by 3, that is 18 \* 3 = 54%) of the deposit if the same drawdown happens on the provider's account again”. Why did the write consider that 18% is a loss, isn't that a profit from May? Also, why should I multiply it by 3? Where did this 3 came from? Lastly, is the drawdown 54% or 18%? I'm sorry if these questions seemed obvious, I'm beginner and I'd really appreciate if someone explains to me. Thank you :)


![drayzen](https://c.mql5.com/avatar/2019/3/5C88A332-A082.png)

**[drayzen](https://www.mql5.com/en/users/drayzen)**
\|
15 Jun 2019 at 08:53

Why do you not factor the Leverage again in the Deposit Load calculation so that the statistic can be directly comparable across all Signals?


![Cheang Jia Kang](https://c.mql5.com/avatar/2018/12/5C171041-546D.jpg)

**[Cheang Jia Kang](https://www.mql5.com/en/users/cheang178)**
\|
30 Sep 2021 at 09:05

Nice article, it cleared my doubt of MAE vs MFE, thanks!


![Kamil Maitah](https://c.mql5.com/avatar/2025/5/681fbcbf-5134.png)

**[Kamil Maitah](https://www.mql5.com/en/users/kamilma)**
\|
25 Jan 2022 at 21:34

Great article, this should be mandatory reading for everyone who wants to start with signals.


![rfh2022](https://c.mql5.com/avatar/avatar_na2.png)

**[rfh2022](https://www.mql5.com/en/users/rfh2022)**
\|
11 Jul 2023 at 23:03

Thanks, but is the Drawdown really only unrealized losses? Drawdown should also be realized losses the reduces the balance relative to  the highest recorded balance on the account


![Working with currency baskets in the Forex market](https://c.mql5.com/2/24/articles_234.png)[Working with currency baskets in the Forex market](https://www.mql5.com/en/articles/2660)

The article describes how currency pairs can be divided into groups (baskets), as well as how to obtain data about their status (for example, overbought and oversold) using certain indicators and how to apply this data in trading.

![Portfolio trading in MetaTrader 4](https://c.mql5.com/2/24/Portfolio_Modeller.png)[Portfolio trading in MetaTrader 4](https://www.mql5.com/en/articles/2646)

The article reveals the portfolio trading principles and their application to Forex market. A few simple mathematical portfolio arrangement models are considered. The article contains examples of practical implementation of the portfolio trading in MetaTrader 4: portfolio indicator and Expert Advisor for semi-automated trading. The elements of trading strategies, as well as their advantages and pitfalls are described.

![Statistical Distributions in MQL5 - taking the best of R and making it faster](https://c.mql5.com/2/25/MQL5_statistics_R_.png)[Statistical Distributions in MQL5 - taking the best of R and making it faster](https://www.mql5.com/en/articles/2742)

The functions for working with the basic statistical distributions implemented in the R language are considered. Those include the Cauchy, Weibull, normal, log-normal, logistic, exponential, uniform, gamma distributions, the central and noncentral beta, chi-squared, Fisher's F-distribution, Student's t-distribution, as well as the discrete binomial and negative binomial distributions, geometric, hypergeometric and Poisson distributions. There are functions for calculating theoretical moments of distributions, which allow to evaluate the degree of conformity of the real distribution to the modeled one.

![LifeHack for trader: "Quiet" optimization or Plotting trade distributions](https://c.mql5.com/2/24/avaf2i.png)[LifeHack for trader: "Quiet" optimization or Plotting trade distributions](https://www.mql5.com/en/articles/2626)

Analysis of the trade history and plotting distribution charts of trading results in HTML depending on position entry time. The charts are displayed in three sections - by hours, by days of the week and by months.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/2704&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071955342943793550)

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