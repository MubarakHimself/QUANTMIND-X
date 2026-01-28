---
title: How to choose an Expert Advisor: Twenty strong criteria to reject a trading bot
url: https://www.mql5.com/en/articles/11933
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:28:00.905160
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/11933&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070307750539367310)

MetaTrader 5 / Tester


### Introduction

A professional algorithmic trading portfolio includes at least 10 to 20 capital accounts running from 10 to 50 expert advisors on at least 10 to 20 capital markets. This is not a standard, but it is the practice I can see around. This article tries to answer the question: how can we choose the right expert advisors? Which are the best for our portfolio, and how can we filter the large trading bots list available on the market? This article will present twenty clear and strong criteria to reject an expert advisor.

Each criterion will be presented and well explained to help you make a more sustained decision and build a more profitable expert advisor collection for your profits. Some criteria are very simple and can be debated fast; others need strategy test results for a clear conclusion. Anyway, all requirements are universal, do not depend on the used trading platform, and can be applied by anyone with little experience and no programming or advanced coding skills.

I am [Cristian Mihail Pauna](https://www.mql5.com/go?link=https://pauna.pro/ "https://pauna.pro/"), engineer, economist, and Ph.D. in economic informatics. I've been producing and testing trading algorithms and automated trading systems made by me and by many others since 1998. Intentionally this article was written in a negative way, considering the rejection criteria instead of acceptance ideas, as most of the trading bots on the market do not qualify for all of these criteria. This article presents my own conclusions about how to reject a trading bot and how it can be considered to be included in an investment portfolio. The criteria list is still open, and anyone can complete it with new ideas in the comments section. Enjoy!

### The rejecting criteria list

Profitability is the first criterion most inexperienced traders consider when they pick an expert advisor. Of course, the profit is the reason for using an expert advisor, but is the profit all that counts when you have to decide to buy and use a trading bot? In my opinion, it is not! Of course, the software must be profitable, but many other requirements count more in my decision before measuring how profitable that trading bot is. Here is my list of rejecting criteria:

1\. Reject if the bot depends on the author's action!

2\. Reject if the bot has no optimal parameter set!

3\. Reject if the bot depends on news-related actions!

4\. Reject if the bot depends on your actions in the run!

5\. Reject if the bot has initialization or running errors!

6\. Reject if the bot has intentional running restrictions!

7\. Reject if the bot has tight spread running conditions!

8\. Reject if the bot includes specific scalping conditions!

9\. Reject if the bot works on a high-commission market!

10\. Reject if the bot depends on a particular broker!

11\. Reject if there are more negative than positive trades!

12\. Reject if the profit is obtained in a tight period of time.

13\. Reject if there have been no positive results in the last five years!

14\. Reject if there are no positive results in more than three months!

15\. Reject if you can not set the capital exposure or the risk level!

16\. Reject if the profitability for small and large capital is much different!

17\. Reject if the test results fail for specific parameters you can set!

18\. Reject if the test results fail for any other parameter set than optimal!

19\. Reject if the number of trades is too small or concentrated in time!

20\. Reject if the live results are much different than the test results!

The majority of the rejection criteria presented here can be evaluated with small resources using the free demo version of any expert advisor before buying the paid version by simply observing the test results. Only the last criterion presumes the acquisition of the trading bot and comparing the live results with the test results. The expert advisor's profitability is also important and must fit the long-time [risk and capital management plan](https://www.mql5.com/en/articles/11500).

### 1\. Reject if the bot depends on the author's action!

Your trading tools and investment portfolio must be independent of others' will or actions. You have to be the only one who controls your software and your servers. I initially reject any expert advisor when I find in the product description sentences like: "after purchase, contact me to give you the best parameter set" or "after the purchase, just ask me how you can get the best results using my software." It is like the author wants you to buy his software without testing it before and without knowing anything about the possible results. I am still wondering who buys products like this. Moreover, in this case, if the author has no time or availability to answer you or doesn't want you to have the best results with that software, he can decide for you and your spent money, which is unfair.

### 2\. Reject if the bot has no optimal parameter set!

I reject from the start any trading bot that is not containing the optimal parameter values or if the author is not presenting the best set file. Hundreds of bots on the market are offered together with the invitation to optimize them and find your best parameter values. Some authors even invite the buyers to communicate a better parameter set file if they find one. It is like the author was not able to find the best configuration, but he pretends to have a very good trading bot. In my opinion, a professional expert advisor must have set by default the best parameter set for a specified market, and the buyer must use it to test that product before the purchase.

### 3\. Reject if the bot depends on news-related actions!

I usually reject trading bots if I am invited to stop them before specific or critical news. An expert advisor like this obliges you to be aware of the current economic calendar news and gives you a full-time job. This is the case of a semiautomated trading bot, which is not under my preferences. I am also skeptical when a trading bot depends on the news calendar, reading the events on a specific site. I am not against this idea, but from experience, I saw that important news could come without any notice, with no schedule, and without even having the possibility of finding out about the subject before the price change. These kinds of bots are sensitive to big event news and can generate losses in unprecedented cases. A professional expert advisor must manage all the cases, no matter what news appears and how the price moves in exceptional circumstances.

### 4\. Reject if the bot depends on your actions in the run!

If a trading bot asks for your action from time to time to function well, you can reject it from the beginning. I met expert advisors, some of them with good profitability on the short-term tests, which ask you to do specific tasks after a while. For example, “Restart the bot every seven days to increase the computing speed,” “Restart the bot after every Sunday,” “In case of large volatility, set that parameter to false,” or “Reoptimize that parameter every month.” These are only some of the cases I met. Like the previous rejection criterion, these cases of trading bots are employing you forever, and you have to work hard to keep them updated. Usually, these kinds of bots can not be tested for significant periods because they include procedures that need special functional conditions from time to time that can not be automated. What if I miss doing that action? What if I forget to restart the bot? I will miss the profit, of course, and for this reason, I skip them from the beginning.

### 5\. Reject if the bot has initialization or running errors!

After testing a trading bot in the strategy test module, I carefully read the log file. It can tell you a lot about the program and about the professionalism of the author. If I meet logs like “Dividing by zero at line 298,” or “Array out of range at line 412 character 22,” I will reject that bot from my interest. I will consider the same decision if I find tens or hundreds of warnings in the log file. The decision is not because such errors or warnings make the bot unusable. No, it is still running, but I can not trust the author who sent that program with critical errors or important warnings. It is like he does not care about the functionality at all. A professional trading bot must have no errors or warnings during the typical run.

### 6\. Reject if the bot has intentional running restrictions!

Over time, I met trading bots that can run only if specific conditions are met. This is not about the account number or the investor name, parameters that the authors can set in different cases to protect their copyright. It is about conditions like: “This bot is running only GMT+2 time.” Maybe this condition can be met by my broker today, but what if the broker decides to change the time in the future? Or what about I will choose to change the broker who manages my capital? What will I do with that bot? I will throw it away for sure. To simplify my life, I throw it away from the beginning. There are many other profitable bots with no particular conditions in the market.

### 7\. Reject if the bot has tight spread running conditions!

Bots running great strategies with significant capital efficiency if very low spread values are numerous today. Many brokerage companies are providing variable spread margin accounts with very low spreads. The strategy tests in these cases look amazing. But what happens when the real market conditions are met? A variable spread account presumes that the spread can have any value with no notice during the time. Does that program still remain profitable? Usually no! If I read in the product description: “this bot is designed to work in low spread conditions under 5 pips,” I reject the bot from my interest. The reason is that the bot is usually non-profitable in real market conditions. During the large volatility periods, when the spread is not minimal, the bot will make losing trades, even if it is profitable in the rest of the cases.

### 8\. Reject if the bot includes specific scalping conditions!

There are several meanings for the scalping term nowadays. I am referring here to the case of making a profit from hundreds or thousands of trades per day, in which the profit per trade is lower than the spread. I avoid using this kind of trading bot. The profitability presumes there is a very low latency execution of all the involved trades, which usually is not happening in real market conditions. Moreover, each broker has a limit for the number of daily operations sent to a brokerage server. When you test the bot, this limit is not active. However, it is active, and you will find out about it after you buy the bot and you try to run it in real conditions.

### 9\. Reject if the bot works on a high-commission market!

I reject a bot when I see that the profit trade is less than the spread plus commission. Even if the bot is profitable and the rest of the conditions are met, I reject the bot from an economic point of view. In my opinion, there is no reason to make more money for the broker and loss for me. The balance must be in the other sense. I accept bots if the profit is at least two times more than the spread plus the commission paid for each trade. Of course, this is only an opinion, but there are so many expert advisors in the market with excellent capital efficiency that there is no good reason to use a low-profitability one. Anyone can choose a better one than the presented case.

### 10\. Reject if the bot depends on a particular broker!

I will never buy a bot made especially for a brokerage company. There are many for sale on the market. Even some brokerage companies are selling expert advisors made only for their particular servers to attract new clients. This rejection reason is clear. You must not depend on a specific brokerage company. Professional traders are using more brokers and need the option of changing the brokerage company when the commissions are significantly increased. If you build your activity depending on a specific broker, sooner or later, you will feel the limits. A professional expert advisor must be universal and must be used with any broker under the same trading platform.

### 11\. Reject if there are more negative than positive trades!

There are numerous expert advisors in the market using strategies that obtain profit in a small number of trades, the rest being losing trades. I usually avoid this kind of strategy. If the positive trades percentage is only 10 or 20% of all the trades, at a small market change, there is a very high probability of having less profitable trades than before. Therefore, after years of testing trading strategies, I include only bots with at least 80-90% positive trades percentage in my expert advisors portfolio. Even with this high profitability rate, additional performance criteria must be met.

### 12\. Reject if the profit is obtained in a tight period of time.

I have found expert advisors making a profit only on a specific month of the year. Others are recording earnings on the last week of each month. Also, there are trading bots opening trades only at a specific date and, even worst, during a particular five minutes period every night. In principle, there is nothing wrong with this idea, but what if the brokerage company increases the spread ten times in exactly those 5 minutes of every night? In this case, we have to take goodbye from the profit. This rejection reason must be well understood and applied. For sure, there are some particular strategies that can make a profit. We have only to be sure that the real market conditions are the same as the ones used to test that system. Usually, from my experience, a very tight particularization of the trading period indicates bad results in real-time trading.

### 13\. Reject if there have been no positive results in the last five years!

Testing a trading bot for the last year and obtaining positive results can be a good reason to admit it to your portfolio. But what if the profitability over the last three or five years is negative? Will you reject it or not? I will reject it! From the statistical point of view, a good and representative sample for the time price series must include at least 1000 days. I usually use the last five years. I am considering the fifth year back to have special conditions which adapt the market behavior from the past at the 1000 days in the middle of the sample, and the last year to have particular behavior made by the last time events. If a strategy has not stable evolution in the last five years, it is not good enough to be included in a portfolio, in my opinion.

### 14\. Reject if there are no positive results in more than three months!

Every expert advisor can present losing trades from time to time. I usually reject a strategy if there is a losing period longer than three months in the last five years. This criterion is a subjective one. Anyone can use different numbers. For me, this losing maximum of three months must be recovered within two months after the drawdown. A third criterion must be met at the same time. For acceptance, a trading bot must have the maximal drawdown of at least a third of the yearly profitability. If not, it will be rejected from my interest. These criteria will ensure long-lasting results at the end of the year, especially if you install two or more expert advisors in the same capital account.

### 15\. Reject if you can not set the capital exposure or the risk level!

This is probably the first criterion I am looking for when I test a trading bot. To include an expert advisor in an investment plan, you must set the risk according to your risk and management plan. If that bot offers you only 15% or 25% capital exposure, it will not be suitable for many investors. We don't need to customize the risk with decimals. A scale with 1%, 2%, 3%, or 5% gradient risk is good enough for anyone. Many authors need to include risk customization in their expert advisors. They consider that the risk will result from the tests if you can set the trading volume. But a professional expert advisor will indicate the risk involved, considering the test results in the last five or ten years.

### 16\. Reject if the profitability for small and large capital is much different!

When testing an expert advisor, we can set up the initial capital. If you get, for example, a 25% yearly profitability testing the trading bot for 10000 USD and 580% testing the same bot for 20000 USD, be sure there is a considerable difference between the test results and the real-time results. Usually, by doubling the initial capital, the bot must double the traded volume and obtain something like a double profit. Of course, can be differences, and more trades can be opened if the capital is doubled, but if the profitability is increased in an unbelievable way, be sure there is a problem. I met cases when the profitability was increased when the capital was increased but by increasing the risk involved without even the author knowing about the case.

### 17\. Reject if the test results fail for specific parameters you can set!

Over time I have rejected many expert advisors that were profitable only for one particular parameter set. Another type is represented by those trading bots that fail the test if you set some input parameters in a specific mode. For example, I have rejected a bot that failed the test if you set the risk at 5% and the capital at 1000 USD. Meanwhile, the 5% risk can be set for 10000 USD, and the bot worked fine. The author must validate all the input variables and display warnings if a variable is set out of the operating range. This is not the job of the user. For example, cases like this can generate losses if the risk is set too high for a specific low capital amount.

### 18\. Reject if the test results fail for any other parameter set than optimal!

The stability of any trading strategy can be tested by changing the parameters. I would reject a trading bot if it became not profitable by changing the optimal parameters set received from the author. Over time I have met expert advisors that are profitable for only one particular parameter set. This type is an unstable algorithm that will record losses anytime when the real-time market behaves differently than the market included in the time used for the optimization. Another is made by trading bots that are stable and profitable for parameters that can not be met in real-time trading conditions. This category includes many bots working fine only for small spreads or low-latency data connections. These conditions can not be met in the real-time trading environment.

### 19\. Reject if the number of trades is too small or concentrated in time!

Three or ten trades per year are too small to be interesting for a professional investor. There are such expert advisors available on the market. The small number of transactions over a considerable period can signify randomness. From the statistical point of view, you need much more years to test that trading bot for a general conclusion. The investors can consider their own numbers. I am not accepting a bot if it trades an average of fewer than two trades per week. Another case includes bots trading more positions but in a tight period. The rest of the time, they are waiting for nothing. This case is a waste of time and resources, in my opinion.

### 20\. Reject if the live results are much different than the test results!

For a substantial percentage of the expert advisors on the market, there are significant differences between the strategy test results and the real-time trading results. To consider all the rejection criteria presented above, you can do only the strategy test with significant time back, using the free demo version of any expert advisor. To find out if there are important differences between the test results and the real trading result, you have to buy the software. Believe me or not, 75% of the bots I bought until now were rejected because of the essential differences between the test and the real-time results. In some cases, the differences were so significant that I noticed the authors about the situation. Some of them excluded the bot from the public listings, but many others are still selling those bots. In any case, I have inserted feedback for general information. Before considering including a trading bot in your usual investment portfolio, this is the last test you have to do. For this test, you must spend real money on the software acquisition and risk some real capital. From my observations, if a bot gives you back through profit the spent money for this test in a period between 3 and 12 months without any significant losses, you can use it for good. Of course, everything depends on the market behavior and the used capital for the test, but the idea remains valuable in most cases.

### Conclusion

Including an expert advisor in a long-time investment plan must pass all the above twenty tests presented above. The first criteria in the list can be considered by testing the free version of any expert advisor. The last one presumes to test the trading bot's paid version to check if the real-time trading results align with the test results. A significant percentage of the advisors are not passing this last test. With all the above conditions, the trading bot's profitability is essential in any case. Regarding this subject, the investors have their own numbers. Anyway, the profit level must be considered together with the capital drawdown numbers, the average number of executed trades, recovery, and profit factor. There is no perfect expert advisor on the market. All of them have advantages and disadvantages. This rejection list criteria try to exclude those bots that can generate significant losses or who can consume resources for nothing. All of these are only the author's conclusions. Anyone is invited to complete the criteria list with different and interesting ideas.

\-\-\-

This article was published by [Cristian Mihail Pauna](https://www.mql5.com/go?link=https://pauna.pro/ "https://pauna.pro/") first on [Research Gate](https://www.mql5.com/go?link=https://www.researchgate.net/publication/367328519_How_to_choose_an_Expert_Advisor_Twenty_strong_criteria_to_reject_a_trading_bot "https://www.researchgate.net/publication/367328519_How_to_choose_an_Expert_Advisor_Twenty_strong_criteria_to_reject_a_trading_bot").

DOI: [https://doi.org/10.13140/RG.2.2.20912.43528/1](https://www.mql5.com/go?link=http://dx.doi.org/10.13140/RG.2.2.20912.43528/1 "http://dx.doi.org/10.13140/RG.2.2.20912.43528/1")

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Risk and capital management using Expert Advisors](https://www.mql5.com/en/articles/11500)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/441785)**
(25)


![Alexey Oreshkin](https://c.mql5.com/avatar/2023/1/63b4c27e-be89.jpg)

**[Alexey Oreshkin](https://www.mql5.com/en/users/desead)**
\|
23 Apr 2023 at 22:28

that's bullshit.


![Ricardo Rodrigues Lucca](https://c.mql5.com/avatar/avatar_na2.png)

**[Ricardo Rodrigues Lucca](https://www.mql5.com/en/users/rlucca)**
\|
23 May 2023 at 16:20

I'd like to ask about this " 11\. Reject if there are more negative than positive trades! " . Usually, I agree, this is a rule for a trader doing scalper. Do you have a swing rule? What percentage of positive trades would you expect? 35% to 50%?

![Cristian Mihail Pauna](https://c.mql5.com/avatar/2022/4/625F1963-FB7C.jpg)

**[Cristian Mihail Pauna](https://www.mql5.com/en/users/cpauna)**
\|
23 May 2023 at 16:26

**Ricardo Rodrigues Lucca [#](https://www.mql5.com/en/forum/441785/page2#comment_47058268):**

I'd like to ask about this " 11\. Reject if there are more negative than positive trades! " . Usually, I agree, this is a rule for a trader doing scalper. Do you have a swing rule? What percentage of positive trades would you expect? 35% to 50%?

Hello, I usually accept an EA in my portfolio only if it generates at least 80% positive trades in the last 3-5 years.

![Gilberto Fernandes da Silva](https://c.mql5.com/avatar/avatar_na2.png)

**[Gilberto Fernandes da Silva](https://www.mql5.com/en/users/gilfernandes)**
\|
28 May 2023 at 00:59

I've already used several robots on a real account and what I've noticed is that the results between a real and demo account are very different indeed and have unfortunately caused more losses than gains. I've now been testing some robots with artificial intelligence for 30 days on the demo account and they've shown excellent profitability, but I'm apprehensive and afraid to [put them](https://www.mql5.com/en/articles/1171 "Why Virtual Hosting on the MetaTrader 4 and MetaTrader 5 platforms is better than the usual VPSs") on the real account due to the losses I've already suffered with other robots that proved to be highly profitable on the demo account, but when they went to the real account, everything changed for the worse.


![Harun Öner](https://c.mql5.com/avatar/avatar_na2.png)

**[Harun Öner](https://www.mql5.com/en/users/msert8286)**
\|
28 Nov 2024 at 09:38

**MetaQuotes:**

Check out the new article: [How to choose an Expert Advisor: Twenty strong criteria for rejecting a trading robot](https://www.mql5.com/en/articles/11933).

Author: [Cristian Mihail Pauna](https://www.mql5.com/en/users/cpauna "cpauna")

I want to see the demo operations of the robot live


![Creating an EA that works automatically (Part 01): Concepts and structures](https://c.mql5.com/2/49/Aprendendo-a-construindo.png)[Creating an EA that works automatically (Part 01): Concepts and structures](https://www.mql5.com/en/articles/11216)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode.

![DoEasy. Controls (Part 31): Scrolling the contents of the ScrollBar control](https://c.mql5.com/2/51/MQL5-avatar-doeasy-library-2.png)[DoEasy. Controls (Part 31): Scrolling the contents of the ScrollBar control](https://www.mql5.com/en/articles/11926)

In this article, I will implement the functionality of scrolling the contents of the container using the buttons of the horizontal scrollbar.

![Creating an EA that works automatically (Part 02): Getting started with the code](https://c.mql5.com/2/50/Aprendendo-a-construindo_part_II_avatar.png)[Creating an EA that works automatically (Part 02): Getting started with the code](https://www.mql5.com/en/articles/11223)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we discussed the first steps that anyone needs to understand before proceeding to creating an Expert Advisor that trades automatically. We considered the concepts and the structure.

![Creating a ticker tape panel: Improved version](https://c.mql5.com/2/49/Letreiro_de_Cotar2o_avatar.png)[Creating a ticker tape panel: Improved version](https://www.mql5.com/en/articles/10963)

How do you like the idea of reviving the basic version of our ticker tape panel? The first thing we will do is change the panel to be able to add an image, such as an asset logo or some other image, so that the user could quickly and easily identify the displayed symbol.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=odgzxuppuioavsjugiybxgqtoqgpnnps&ssn=1769185679000395963&ssn_dr=0&ssn_sr=0&fv_date=1769185679&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11933&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20choose%20an%20Expert%20Advisor%3A%20Twenty%20strong%20criteria%20to%20reject%20a%20trading%20bot%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691856796913612&fz_uniq=5070307750539367310&sv=2552)

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