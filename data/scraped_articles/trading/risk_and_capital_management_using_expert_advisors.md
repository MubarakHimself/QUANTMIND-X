---
title: Risk and capital management using Expert Advisors
url: https://www.mql5.com/en/articles/11500
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:30:36.157620
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/11500&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049181253086848650)

MetaTrader 5 / Trading


### Introduction

This paper is not about how much to invest or to risk on a trade. No, not at all. These are subjects well treated and forever concluded by many authors in the last decades. This paper is about what you can not see in a backtest report, what you should expect using automated trading software, how to manage your money if you are using expert advisors, and how to cover a significant loss to remain in the trading activity when you are using automated procedures. In short, how to invest using expert advisors? I am [Cristian Mihail Pauna](https://www.mql5.com/go?link=https://pauna.pro/algorithms "https://pauna.pro/algorithms"), engineer, economist, and Ph.D. in economic informatics. I've been producing trading algorithms and automated trading systems since 1998, a long time before MetaTrader existed. It is my everyday activity, and this is the first article to read if you decide to use automated software to make money.

### Automated trading algorithms

Using expert advisors is a simple task today. The fantastic development of MetaTrader 4 and 5 allows us to spend no more than a couple of minutes to download, test, and run an automatic software to trade our money. Thousands of expert advisors made by hundreds of authors are available online, and to buy one, you need no more than two clicks. A positive backtest report makes you feel that you met the right software. Then, you install it on a demo account and run it for a while. If it is working with virtual money, you buy it and run it on a real account waiting for the day-by-day profit. Usually, it comes, like the backtest presumes, but sometimes it is not!

Why? Why an algorithm that made you money for a long time becomes unprofitable? Can an algorithm that made only trades with a maximum 2% drawdown for more than ten years blow your account? Yes, it is possible! The year 2022, when I write this article, is the best year to prove that algorithms working irreproachable for more than ten years are not good enough. There are so many today, and they are still for sale online for big money. To find the answer to how this is possible, you can read a lot nowadays. Some believers in the conspiracy theory will say that the brokerage companies are acting against the traders. Others suggest it is about the big central banks collaborating and making unpredictable decisions to move the money from small accounts into their huge and unlimited accounts. Others will also sustain that it is about some market makers organized into superior forums deciding who to win or not. Well, these are only stories for children, and none of the above are true in our case. Not at all!

An algorithm is a finite set of rules, a priori defined, made to solve a specific task. In our case, a trading algorithm is made to transform the market input data into trading decisions such as buy, sell or stay away from the market risk. The trading algorithm receives the historical quote price data, apply different computation and transformation functions, and builds the specific trading signals. Every trading algorithm has its own parameter set. These are coefficients set and optimized using the previous market data to obtain the maximum profitability and minimum values for the drawdown into a past time interval. The software authors optimize their algorithms using two, three, five, or more years of historical market data to obtain a better algorithm. In this way, a trading algorithm includes the market behavior through its parameters. In other words, it is made for a specific time interval and particular market behavior to perform well.

What in this world can assure you that tomorrow's market behavior will be the same as today or yesterday? Nothing!

Which mathematical principles can sustain that tomorrow the market will behave like in the last ten years? No one!

What facts from real life can guarantee you that tomorrow's events will be the same as in the last fifty years? None!

We think the market is stable. We want to be. This is a hypothesis. We hope the current market conditions are the same for at least a short time. But this hypothesis is not valid from time to time. There is no mathematical guarantee that the price action will be the same, as there is no motive to trust that humans, nature, or hazards will act the same way every day. The market price movement depends mainly on human behavior. It depends on all global decisions, economic facts, and natural or geopolitical events. It also mainly depends on all the market participants, their beliefs, ideas, fear, or trust. All of these variables are unbounded through the hazard theory. Consequently, the price action has a significant degree of unpredictability and is also unbounded.

An algorithm optimized for the last period will perform well if the market works like in the past, but it will not perform if the market changes significantly. The algorithm will deliver the same results only if the market price action is within the same limits and behavior as in the quotes series used for the procedure optimization. When an unprecedented event is happening, like a pandemic, war, economic crisis, or other significant events that can dramatically change the investment appetite, the algorithms can record substantial losses instead of a profit. The market change is responsible for this fact.

Is this conclusion a big NO for using trading algorithms? Of course NOT! With all of these, we are still using successfully automated trading algorithms, and this article shows you how to do it properly. We have to know what to expect, how much we can trust a trading algorithm, how to recognize the situation when an algorithm is not evolving like it was designed, how to identify an unstable algorithm, and how to avoid significant losses. The most important thing is to understand how to make proper risk and money management to limit the losses and to pass a substantial possible loss to remain in the trading activity for a long time using automated trading algorithms.

### What the backtest reports don't tell

We all are using the backtest reports to evaluate an expert advisor, to optimize it, and to have a suggestion for the risk involved by its functionality. The strategy tester procedures are more and more evolved. They present a vast amount of information, giving you the idea that you know everything about your algorithm. Well, you can find almost everything about a ready-made and optimized algorithm in the backtest results, even an estimation for the profit in the next period. All are made based on the hypothesis that the market behavior will never change. But, as we have seen, this is not true, and from time to time, the market evolution is so much altered that the perfect algorithm is giving you a loss when you don't expect it. The backtest reports don't tell you how the algorithm evolves if the market changes dramatically. But, we still have good news: we can also use the backtest results to evaluate the stability of a trading algorithm when it is out of the optimal form.

In my activity, I have tested thousands of trading algorithms made by me or by many others, and I found out about hundreds of algorithms that were performing exceptionally well for a significant period and changed dramatically without any notice when the market changed its behavior. After a significant period from the market change, a backtest report can reveal the phenomenon with more accuracy. During all this time, I found out that there are three different classes of algorithms when we speak about how the losses appear instead of the profit when the market is drastically changed. These classes are:

1. Lazy algorithms
2. Persistent algorithms
3. Unstable algorithms

The _lazy algorithms_ are those trading procedures that perform well for a while, and after a significant market change, the algorithms start to make very long trades. Some of them still remain profitable, but due to the accumulation of the swap commissions, the profit is meager compared with the initial evolution. Others are opening so long trades that will never be profitable, mainly if brokers with high commissions and swaps are used. To understand this case easily, an algorithm that made for me trades that lasted a maximum of 8 hours for more than five years, after a significant market change, it made trades that lasted at least 8 months. You have to pay enough attention to the most extended time trade and how many positions are open every week or month. Otherwise, a test report made using data from many years back can trick you and give you the false idea that the algorithm is still profitable. Many strategy testers are not presenting the shortest and the longest time trade in an explicit way. In some cases, you have to compute this information as an internal procedure of the optimized algorithm to find out about this fact. The danger in the case of using lazy algorithms is not a very high one. Suppose you are following the open positions in your accounts. In this case, after a while, you will be noticed that the swap commissions are accumulating more than usual, and you will start to ask yourself about the used expert advisor.

The _persistent algorithms_ are those expert advisors producing loss trades, one after another, more negative transactions than the profitable ones, after a significant period in which the algorithm made only or mostly profitable trades. For example, I found an algorithm that made 98% profitable trades during eight years. This means only two losing trades at every 100 open transactions. It was a fantastic procedure! However, after the pandemic crisis in February 2020, this algorithm makes 72% loss trades and only 28% profitable trades, no matter how you optimize it. It is the perfect example of a trading procedure that became an unprofitable algorithm with a negative expectancy due to the market behavior change. Some inexperienced traders using expert advisors will say this case is not dangerous because the algorithms usually use stop-loss points, and significant losses can not appear. This is false! The danger in the case of using persistent algorithms is the fact that you are not noticed the market behavior change. Even using protective stop-loss, without any notice, the algorithm will perform one, two, or more negative trades. The jeopardy, in this case, became even from the trader himself. Looking at a backtest report made for that expert advisor, the trader will see that the probability of a losing trade is very low. Therefore, he will keep the algorithm working, thinking that the following trades will be positive and the drawdown will be recovered. However, that algorithm will make mainly loss trades because the market behavior was changed forever. After a while, the accumulated loss can be so significant that the involved negative expectancy algorithm will never recover it. In this case, if you see that an algorithm made a double or a triple loss than in the normal backtest, you have to ask yourself about that procedure's profitability. For this purpose, you must implement a "Global Stop-Loss" for your entire account, a subject treated in other articles.

The _unstable algorithms_ are the most common trading procedures I found. They have been performing well for a long time, and once the market has changed, the algorithms make a very large or total loss. These are the most dangerous algorithms and must be avoided at any cost, even if they promise significant profitability. Algorithms using an unlimited number of trades, procedures using tens or endless hedging steps, or expert advisors using a wrong risk management procedure are usually unstable algorithms, but not only. A young trader asked me to test an algorithm he had made for a specific market. It was an innovative idea, and the backtest result was optimistic from the beginning. After many tests, I was amazed. The algorithm was profitable for only one parameter set. I have changed the take profit with only one point, and surprise: from time to time, the algorithm was not reaching the take profit point, and very large losses appeared. When I changed the used spread (which usually is variated by the broker with no notice), the loss trades became much higher than the profitable transactions. In this case, the algorithm was perfect for particular market behavior, which is usually impossible in practice. The algorithm appeared after a while on the market, and it is still for sale. Anyone without experience can test that algorithm with the set files provided by the author and can conclude that it is a good one. Anyone can buy it and run it to make losses from the beginning without knowing at least what was wrong. To avoid unstable algorithms, you can use the strategy tester. If you change a little the original parameter set and the algorithm is blowing the test account, that is an unstable algorithm, and keep it away from your interest. You can also find algorithms that appear stable when you set the risk level, for example, 2%. It gives you 2.03% capital exposure and a maximum drawdown of 2,12%. If the same algorithm runs with a 3% capital risk and will provide you a much higher capital exposure, for example, 52%, skip it forever.

### ![Capital evolution of an unstable automated trading algorithm](https://c.mql5.com/2/49/UnstableAlgorithm__3.png)

Figure 1. Capital evolution of an unstable automated trading algorithm.

### A suitable risk management strategy

One suitable risk and capital management strategy is the key to controlling everything. Everyone reading this paper knows that financial trading is a very risky activity. At the same time, made in the right way, it can be a profitable activity; otherwise, it would not exists. The usage of automated trading software is not reducing the involved risk at all. I even consider that the automated expert advisors increase the capital risk, especially when the software author is different than the trader using those procedures or if the procedure optimization is not made frequently. To build a suitable risk and capital management structure, we have first to respect the three fundamental investor rules:

1. Never risk more than you can afford to lose!
2. Never risk all your available money!
3. Never risk others' money!

The general purpose of the trading activity is to make a profit, but losses can happen regardless of the investor's will. The usage of one or more expert advisors does not guarantee a profit. The algorithms are optimized using the last period statistics, but the markets can change the price evolution anytime without notice. Therefore, you must be prepared to admit and accept any money loss. To achieve this stage mind, the trader needs experience and to reduce the invested capital amount until he is comfortable with that eventual loss. This is about the first and the most important rule from above. Once the trader can afford any loss in his accounts, he must follow the second rule to ensure that the trading activity will not negatively impact his life. Finally, everyone must respect the third rule and not imply other unprepared persons in this risky activity. If someone cannot respect one of the above basic financial investment rules, he must stay away from this activity and not invest!

Once we have a capital amount we can afford to lose, we can involve it in financial investment and trade it using expert advisors to make a profit. For this purpose, we need some good and profitable expert advisors and a suitable risk and capital management strategy. Suppose we have 20000 USD and five good expert advisors that can be set to run each one with a maximum drawdown of 2%. By hypothesis, they will run together with a 10% capital exposure for a minimal capital of 1000 USD. The system profitability expectation is to double the invested money in a while. The scenario is realistic and can be done using expert advisors available on the market.

Even though each expert advisor gives us a maximum drawdown of 2% by the backtest reports, the risk involved in the long term is not 2% at all. In time, the market behavior will be so different after a while that the backtest will be obsolete. Usually, we notice this after we take an essential and unprecedented loss. To remain in the trading activity for a long time, we must limit that possible loss and cleverly cover the loss to continue the investment from the last profitability stage. To limit the loss in the case when the expert advisors are evolving differently than we have expected, we have to set a global stop-loss for the whole account. From experience, a suitable global stop-loss value is 2 or 3 times the nominal drawdown. In our example, we can set a global stop-loss at 30% of the deposited capital. To implement it in the account, we can use specialized utility software to stop all trading procedures and to close all positions when a drawdown of 30% is met. Some authors can comment that a 30% limit is too small once some algorithms can reduce the drawdown by their usual run. The global stop-loss limit's maximum and functional value can be set at 50% of the invested capital. This value can also depend on the risk appetite and the algorithm's ability to recover that possible loss. In any case, a global stop-loss procedure set for the trading account will protect the rest of the capital.

The second important stage in building a suitable risk and capital management strategy is to find a way to cover the possible losses. The desired profit will never be made if, after each significant loss, we start from the beginning of the investment plan. The constructive idea is to divide the capital into two main parts: the active capital and the reserve capital. The active capital is the one deposited into the main trading account, where the expert advisors are used to make a profit. The reserve capital is also deposited into a capital account, but it is not risked at all. This passive capital stays and waits to be used only to cover possible losses. In the following table, this strategy is presented for a particular case. The capital is divided into two equal parts from the beginning. When the capital is doubled, half of the profit is withdrawn from the active account into the reserve one.

![Long-time investment plan to cover possible losses](https://c.mql5.com/2/49/InvestmentPlan__1.png)

Figure 2.  Long-time investment plan to cover possible losses

This plan is a stable one in the long-time run. At the beginning of each step, the total reserve capital is equal to the active capital amount. This fact allows the investor to cover from the reserve capital any possible loss at any time during the investment activity. Even if all the active capital is lost, the investor can allocate all the reserve capital for that specific step to recover the loss. Some conservative investors can allocate only half of the reserve capital for a possible loss, keeping half for a major case situation. This strategy is viable, especially when we use global stop-loss protection and not all active capital is at risk. Anyway, this capital strategy allows the investor not to take the whole road from the beginning if all the active capital or important parts are lost during an investment step. Also, after a major loss, if the investor doesn't want to invest anymore, using the above strategy, he remains with the reserve capital, which is a considerable amount after each step, in any case, higher or equal to the loss met.

The above-presented strategy is only an example. More improvements can be made to increase the protective capability for risk and capital management. For example, some traders can withdraw from the profit accumulated in the active account every week or month, thinking that a more extensive period involves a higher risk and can meet a significant loss. Other investors are building different additional strategy steps when it is about to cover a loss. Some will cover only large losses from the reserve account, and others will cover any loss met at the end of each week or month. In any case, a stable strategy must include a rule to rebuild the reserve capital after the trading procedures recover each loss. In this way, the reserve capital will be ready for any other recovery in the future step.

From this point of view, this paper is addressed to any type of investor and trader. The beginners will find a stable way to build their risk and capital management strategy, and the advanced investors will reconsider their capital strategies, especially those who don't have a clear plan to recapitalize their reserve accounts. We have to note that the table above does not include an estimated time for each step once the capital efficiency depends on the financial market behavior. There are good and bad weeks, months, and years, and there are better and worse automated trading procedures, but in any case, the investor must follow his long-time investment plan.

### Conclusion

Using automated software to trade capital markets is not reducing the risk but the opposite. Any algorithm can increase the involved risk without notice because of the market behavior change. Although, nothing in the world can guarantee that the markets will evolve in the future like in the past. A significant part of the invested capital can be protected by using the global stop-loss procedure. A clever capital management strategy can provide the available capital for any loss recovery. Dividing the invested capital into an active and a reserve part offers us a stable investment plan for long-time activity. In any case, never risk more than you can afford to lose, never risk all your available money, and never risk others' money! The trading activity is risky, and investment in the financial markets is a long process involving losses in the way to profit.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot](https://www.mql5.com/en/articles/11933)

**[Go to discussion](https://www.mql5.com/en/forum/433684)**

![DoEasy. Controls (Part 10): WinForms objects — Animating the interface](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__4.png)[DoEasy. Controls (Part 10): WinForms objects — Animating the interface](https://www.mql5.com/en/articles/11173)

It is time to animate the graphical interface by implementing the functionality for object interaction with users and objects. The new functionality will also be necessary to let more complex objects work correctly.

![Neural networks made easy (Part 20): Autoencoders](https://c.mql5.com/2/48/Neural_networks_made_easy_020.png)[Neural networks made easy (Part 20): Autoencoders](https://www.mql5.com/en/articles/11172)

We continue to study unsupervised learning algorithms. Some readers might have questions regarding the relevance of recent publications to the topic of neural networks. In this new article, we get back to studying neural networks.

![Developing a trading Expert Advisor from scratch (Part 22): New order system (V)](https://c.mql5.com/2/47/development__5.png)[Developing a trading Expert Advisor from scratch (Part 22): New order system (V)](https://www.mql5.com/en/articles/10516)

Today we will continue to develop the new order system. It is not that easy to implement a new system as we often encounter problems which greatly complicate the process. When these problems appear, we have to stop and re-analyze the direction in which we are moving.

![Learn how to design a trading system by Accelerator Oscillator](https://c.mql5.com/2/49/why-and-how.png)[Learn how to design a trading system by Accelerator Oscillator](https://www.mql5.com/en/articles/11467)

A new article from our series about how to create simple trading systems by the most popular technical indicators. We will learn about a new one which is the Accelerator Oscillator indicator and we will learn how to design a trading system using it.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/11500&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049181253086848650)

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