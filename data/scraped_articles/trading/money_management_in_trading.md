---
title: Money management in trading
url: https://www.mql5.com/en/articles/12550
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:07:46.086158
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/12550&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069139751363084480)

MetaTrader 5 / Trading


### Introduction

Money management is a very important aspect for successful trading. Its main purpose is to minimize risks and maximize profits. With the right use of money management, it is possible to achieve improved trading results.

In general, money management is a set of rules that allows traders to calculate the optimal volume of positions, taking into account all the possibilities and limitations. Today, there are quite a few money management strategies to fit every taste. We will try to consider several ways to manage money based on different mathematical growth models.

### Trading strategy and money management

Any method of money management can accelerate the growth of the trading balance, or, in other words, to increase the profitability of the trading strategy. Thus, the trading strategy is the basis, while money management is an addition. Let's see what requirements a trading strategy must meet in order to apply money management to it.

First, it is the mandatory use of stop losses and take profits. With their help, you can control trading risks: stop loss allows you to limit possible losses, and take profit makes it possible to assess the potential profit for each deal.

Second, it is a positive mathematical expectation. It enables traders to evaluate the expected profitability of a trading strategy in the long term, which allows them to make rational decisions and manage their capital more efficiently. However, mathematical expectation is important not only for the trading strategy as a whole, but also for a newly opened position.

Let's introduce the following variables:

- **_p_** – probability of a profitable trade;
- **_SL_** – difference between the open price and stop loss in points;
- **_TP_** – difference between the open price and take profit.

First, we need to estimate the probability of profit. Let **_m_** be the number of closed profitable deals, while **_n_** is a total number of trades. Then the probability of receiving profit is:

> ![](https://c.mql5.com/2/54/a.png)

The mathematical expectation (in points) for the position being opened can be found using the equation:

> ![](https://c.mql5.com/2/54/a__1.png)

For a profitable trading strategy, the mathematical expectation should be positive. In this case, the use of any method of money management can bring additional profit.

However, the mathematical expectation may turn out to be negative or zero. This may happen at the very beginning of trading, when the number of losing trades can have a very large impact on the estimation of the profit probability. In this case, the volume of the opened position should be as small as possible. This means the trader needs to apply money management according to the linear growth model with minimal risk.

### Linear growth

This is one of the most famous and simple growth patterns. Its use in trading is even simpler - a trader needs to choose a fixed position size and use it throughout the entire trading session. This model can be described using the linear function equation:

> ![](https://c.mql5.com/2/54/a__2.png)

For convenience, convert this equation into a discrete form. Let **_deposit\[i\]_** be the value of the trading balance on the _i_ th step. Then the linear balance growth equation will look like this:

> ![](https://c.mql5.com/2/54/a__3.png)

where **_L_** is some constant that determines the rate of linear growth.

Let's rearrange this equation a bit. Let the **_res\[i\]_** variable will mark the results of the _**i**_ th trade. Then we get the following equality:

> ![](https://c.mql5.com/2/54/a__4.png)

Let's assume that we have **_n_** completed trades. Then we can use the least squares method to estimate the **_L_** value:

> ![](https://c.mql5.com/2/54/a__5.png)

In this case, the **_L_** value will be equal to the arithmetic mean of the results of all deals.

> ![](https://c.mql5.com/2/54/a__6.png)

However, the arithmetic mean can also provide an incorrect result. Let's assume that you have just started trading and there have been several losing trades. Then, the **_L_** value will be negative, and it will not be possible to use it in further calculations.

Let's introduce an additional condition – the **_L_** value should aim for the best possible one. Only if this condition is met, the deposit will grow at the maximum rate. Then we will use the following expression to evaluate **_L_**:

> ![](https://c.mql5.com/2/54/a__7.png)

In this case, **_L_** will be equal to the mean square of the results of all trades. This estimate allows us to get the ideal linear growth rate, which will always be greater than the real growth rate of the deposit.

Now it is time to see how we can determine the optimal position size. Let's introduce the following variables:

- **_PV_** – price of one point in the deposit currency;
- **_Lot_** – position volume.

Let's find the **_L_** first:

> ![](https://c.mql5.com/2/54/a__8.png)

The position may turn out to be profitable. Then its result will be **_Lot\*TP\*PV_**. Obviously, in this case, any trader will be interested in increasing the deposit growth rate and maximizing **_L_**. It can be expressed like this:

> ![](https://c.mql5.com/2/54/a__9.png)

But the deal may also turn out to be unprofitable. Then the trader will try to avoid the loss leading to an increase in **_L_**:

> ![](https://c.mql5.com/2/54/a__10.png)

Now we can combine both of these conditions and calculate the optimal position size:

> ![](https://c.mql5.com/2/54/a__11.png)

We can make the model more versatile by adding risk to it. Let's add a new variable:

- **_R_** – parameter that determines the degree of risk.

Then, the expression for finding the optimal lot will look like this:

> ![](https://c.mql5.com/2/54/a__12.png)

Thia means the trader should be prepared to lose a little more than is provided by the strict model. In this case, the optimal position size will be as follows:

> ![](https://c.mql5.com/2/54/a__13.png)

The **_R_** variable should be at least 1. The higher the variable, the higher the risk and the stronger the deviation from the linear growth model. This approach to money management may prove to be more attractive. For example, this is how the balance curve looks like when **_R_** = 3.

> ![](https://c.mql5.com/2/54/1__1.png)

This is how the position volume changed. This chart shows by how many steps the minimum allowable position volume has been increased.

> ![](https://c.mql5.com/2/54/2__1.png)

The increase in risk led to a rapid increase in the volume of positions. This made it possible to increase the initial capital by 7.5 times.

In addition, we can apply an empirical approach to the linear growth model. Let's rewrite the original lot calculation equation as follows:

> ![](https://c.mql5.com/2/54/a__14.png)

That is, we explicitly indicate that the size of the position depends on two related factors - the possible loss and potential profit. There will be minimal losses only if a position with a minimum lot is opened. Let us denote the minimum position size of the **_lot_** variable and add the ability to manage risk. In this case, we get the following equation:

> ![](https://c.mql5.com/2/54/a__15.png)

Note that the larger the **_R_** variable, the less the trading risk. This approach is a bit like the fixed-fractional method proposed by Ryan Jones in his book "The Trading Game. Playing by the Numbers to Make Millions". This method can be called linear growth with a gradual increase in speed. All depends on the **_L_** variable. As long as its value is stable, trading is carried out with a fixed lot. This is how the graphs of balance and position volume changes look like if **_R_** = 1.

> ![](https://c.mql5.com/2/54/3__1.png)

> ![](https://c.mql5.com/2/54/4__1.png)

### Exponential growth

In everyday life, the "exponential growth" expression is most often used to refer to a very rapid increase in some parameter. [Compound interest](https://en.wikipedia.org/wiki/Compound_interest "https://en.wikipedia.org/wiki/Compound_interest") serves as an example of such growth. The exponential growth model in trading can be implemented using [Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion "https://en.wikipedia.org/wiki/Kelly_criterion") and [optimal f by Ralph Vince](https://www.mql5.com/en/articles/4162). The simplest implementation of this growth is trading with a fixed percentage. The trader only needs to find the optimal percentage for trading. Let's see how this can be done.

The discrete equation of exponential growth can be written as follows:

> ![](https://c.mql5.com/2/54/a__16.png)

We can find the value of the growth parameter using the following equation:

> ![](https://c.mql5.com/2/54/a__17.png)

In this case, the change in the trading balance can be described by the following equation:

> ![](https://c.mql5.com/2/54/a__18.png)

It goes without saying that a trader is interested in obtaining the maximum end result. Let's see how we can achieve this.

First, we need to clear the result of each trade from the influence of the lot. To do this, we need to divide the obtained result by the deal volume:

> ![](https://c.mql5.com/2/54/a__19.png)

In other words, **_Res\[i\]_** is a result of the **_i_** th trade in case its volume were equal to 1 lot.

Now we need to find such a position volume that the following condition is fulfilled:

> ![](https://c.mql5.com/2/54/a__20.png)

But that is not all. Exponential growth can bring big gains, but losses can also be huge. To reduce the possible loss, we need to supplement the lot calculation with the possible results of a future deal. Then the volume of the future position can be calculated as follows. Let's find the sum first:

> ![](https://c.mql5.com/2/54/a__21.png)

Then, the optimal volume for the opened position will be equal to:

> ![](https://c.mql5.com/2/54/a__22.png)

Of course, this equation can be modified. For example, a trader may be cautious and assume the worst-case scenario of future events always expecting to lose. Then the equation for calculating the position volume will be as follows:

> ![](https://c.mql5.com/2/54/a__23.png)

Here is an interesting feature. When considering linear growth models, we introduced risk at our discretion. Unlike that, in the exponential growth model, the risk appears as a result of a rigorous mathematical solution. The higher the **_R_** parameter, the lower the risk.

This is how the balance and lot curves look like in case of exponential growth.

> ![](https://c.mql5.com/2/54/5__1.png)

> ![](https://c.mql5.com/2/54/6.png)

### Hyperbolic growth

The main feature of hyperbolic growth is that it can reach an infinite value in a finite number of steps. Other models cannot boast of such a feature. Hyperbolic growth is very slow at first. It is too slow compared to exponential and even linear growth. But it is gaining momentum very quickly and there comes a moment when no one can catch up with it.

> ![](https://c.mql5.com/2/54/9.png)

In general, the hyperbolic growth equation looks like this:

> ![](https://c.mql5.com/2/54/a__24.png)

where **_N_** is a total number of model steps, while _**n**_ is a number of steps already taken. The smaller the difference between them, the faster growth accelerates. If these parameters are equal, we get infinity.

Such an equation is not suitable for use in trading - we know neither the total number of steps, nor how many steps we have already passed. Fortunately, the discrete hyperbolic growth equation is free from these shortcomings:

> ![](https://c.mql5.com/2/54/a__25.png)

Unfortunately, there are no easy ways to calculate the optimal position size. For calculations, we will have to use numerical methods.

First, assign the minimum value to the Lot variable and find the value of the sum:

> ![](https://c.mql5.com/2/54/a__26.png)

Now let's take into account the possible options for the position being opened and get the final value:

> ![](https://c.mql5.com/2/54/a__27.png)

Save the [absolute value](https://www.mql5.com/en/docs/math/mathabs) **_D_**. Then increase the **_Lot_** variable one step and repeat the calculations from the beginning. If the new value **_D_** turned out to be less than the previous one, then it is necessary to increase the **_Lot_** variable again and repeat the calculations. If the **_D_** value is higher than the previous one, the calculations are stopped. The optimal position volume will be equal to the **_Lot_** variable obtained at the previous step.

The use of the hyperbolic growth model is associated with a very high risk. In order to reduce it, we can use the **_R_** variable. The higher it is, the lower the risk. Of course, we can prepare for a loss in advance. Then **_D_** is calculated using the following equation:

> ![](https://c.mql5.com/2/54/a__28.png)

This is how the balance change according to the hyperbolic law looks like.

> ![](https://c.mql5.com/2/54/7.png)

Now it is time for a surprise. Look at the volume change graph. We see that after the 30th deal, the lot becomes fixed. It may seem that the hyperbolic growth has been replaced by the linear one.

> ![](https://c.mql5.com/2/54/8.png)

This is a normal behavior for the hyperbolic growth model. It is too perfect for the real world - the trading balance cannot grow indefinitely. In this case, we can say that the model decided that the hyperbolic growth is at the very beginning, so the change in the volume of positions became small. Whether we can see an ascending branch of hyperbolic growth depends on the trading strategy and luck.

### Conclusion

So, we got acquainted with the basic mathematical models of growth. Now let's see if these models can be applied in practice.

Considering margin requirements is a must here. The model may suggest a position volume that is larger than the allowable one.

In addition, the trading strategy may turn out to be asymmetric in the direction of deals. For example, Buy positions may be more profitable than Sell positions. In this case, the lot calculation should depend on the type of a position being opened.

Now let's look at the results we can expect in the market conditions. To do this, we will write a simple Expert Advisor that opens positions when simple moving averages cross. Positions are closed when the stop loss or take profit is reached. MinLot money management — all positions are opened with a minimum volume. Below are the EA testing parameters.

- Symbol: EURUSD
- Timeframe: H1
- Test period: 2021.01.01 - 2022.12.31
- Initial deposit: 10,000
- Total trades: 509

| Money Management | Risk | Total Net Profit | Gross Profit | Gross Loss | Profit Factor | Expected Payoff |
| --- | --- | --- | --- | --- | --- | --- |
| MinLot | - | **466.70** | **2 119.92** | **-1 653.22** | **1.28** | **0.92** |
| Lin1 | 7 | **39 593.81** | **190 376.71** | **-150 782.90** | **1.26** | **77.79** |
| Lin2 | 1 | **1 719.91** | **7 625.00** | **-5 905.09** | **1.29** | **3.38** |
| Exp | 3 | **12 319.19** | **77 348.22** | **-65 029.03** | **1.19** | **24.20** |
| Hyp | 5 | **24 946.38** | **100 778.15** | **-75 831.77** | **1.33** | **49.01** |

Of course, pay attention to how risk affects return.

Programs that were used when writing the article.

| Name | Type | Features |
| --- | --- | --- |
| Money Management | script | The script simulates money management.<br>- **_ProbabilityWin_** \- profit probability<br>- **_Trades_** \- number of simulated trades<br>- **_StartDeposit_** \- initial deposit<br>- **_SL_** & **_TP_** \- stop loss and take profit of deals<br>- **_Risk_** \- risk value<br>- **_Lin1_** ... **_Hyp_** \- money management type<br>- **_ScreenShot_** \- saves the graph in the Files folder |
| Growth | script | Shows the difference between linear, exponential and hyperbolic growth. |
| EA Money Management | EA | Only for testing different money management methods.<br>- **_MoneyManagement_** \- selecting a money management method<br>- **_Risk_** \- acceptable risk<br>- **_SL_** & **_TP_** \- stop loss and take profit of positions<br>- **_PeriodMA1_** & **_PeriodMA2_** \- moving average periods |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12550](https://www.mql5.com/ru/articles/12550)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12550.zip "Download all attachments in the single ZIP archive")

[Money\_Management.mq5](https://www.mql5.com/en/articles/download/12550/money_management.mq5 "Download Money_Management.mq5")(6.9 KB)

[Growth.mq5](https://www.mql5.com/en/articles/download/12550/growth.mq5 "Download Growth.mq5")(2.11 KB)

[EA\_Money\_Management.mq5](https://www.mql5.com/en/articles/download/12550/ea_money_management.mq5 "Download EA_Money_Management.mq5")(7.6 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Polynomial models in trading](https://www.mql5.com/en/articles/16779)
- [Trend criteria in trading](https://www.mql5.com/en/articles/16678)
- [Cycles and trading](https://www.mql5.com/en/articles/16494)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/448571)**
(8)


![Aleksey Nikolayev](https://c.mql5.com/avatar/2018/8/5B813025-B4F2.jpeg)

**[Aleksey Nikolayev](https://www.mql5.com/en/users/alexeynikolaev2)**
\|
27 Jul 2023 at 14:38

We take a coin, toss it 5 times and get tails 3 times, whence p=3/5=0.6. Then we toss the same coin again 5 times and get tails 2 times, whence p=2/5=0.4.

A paradox? The probability is different for the same coin?

No, just another confusion from mixing the concepts of probability and frequency. Frequency can only be an estimate of probability, but only if certain conditions are fulfilled - those involved in the formulation of the law of large numbers, for example.

![isanchez96](https://c.mql5.com/avatar/avatar_na2.png)

**[isanchez96](https://www.mql5.com/en/users/isanchez96)**
\|
8 Nov 2024 at 23:58

Good, I have some doubts about the formulas as I apply them in excel but I do not get results, could not you make a more detailed example with 5 operations and giving the data of these operations and the calculation of the variables for these 5 operations.

I don't know if I make myself clear, what I mean is if you can apply the formula without unknowns and with the prices of tp,sl, pip price, etc.

Thanks.

Thanks.

![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
9 Nov 2024 at 04:19

**isanchez96 [#](https://www.mql5.com/es/forum/452190#comment_55072279):**

Good, I have some doubts about the formulas since I apply them in excel but I don't get results, couldn't you make a more detailed example with 5 operations and giving the data of these operations and the calculation of the variables for these 5 operations.

I don't know if I make myself clear, what I mean is if you can apply the formula without unknowns and with the prices of tp,sl, pip price, etc.

Thank you.

Thanks.

Let me rewrite the script for you; it now creates a file where it writes the modelling result of each trade.

It is possible that you are not taking into account the impact of the pip value on the deposit currency. The result of each trade should be +/- lot\*(TP / SL)\*pip value.

![isanchez96](https://c.mql5.com/avatar/avatar_na2.png)

**[isanchez96](https://www.mql5.com/en/users/isanchez96)**
\|
9 Nov 2024 at 11:54

**Aleksej Poljakov [#](https://www.mql5.com/es/forum/452190#comment_55072723):**

Let me rewrite the script for you; now create a file in which you write the modelling result of each operation.

It is possible that you are not taking into account the impact of the pip value on the deposit currency. The result of each trade should be +/- lot\*(TP / SL)\*pip value.

Thanks for the script, it has cleared up some doubts, although I still have some questions, since this strategy is used with fixed SL and TP, can't it be used with variable SL and TP?

![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
9 Nov 2024 at 12:29

**isanchez96 [#](https://www.mql5.com/es/forum/452190#comment_55073982):**

Thanks for the script, it has cleared up some doubts, although I still have some questions, since this strategy is used with fixed SL and TP, can't it be used with variable SL and TP?

Of course, you can use floating stop losses and still make a profit. Also, according to my observations, the optimal stop losses and profit taking for buy and sell positions should be different from each other. The only requirement is that the mathematical expectation is strictly positive. I used fixed values to simplify the modelling.

![Category Theory (Part 9): Monoid-Actions](https://c.mql5.com/2/55/category_theory_p9_avatar.png)[Category Theory (Part 9): Monoid-Actions](https://www.mql5.com/en/articles/12739)

This article continues the series on category theory implementation in MQL5. Here we continue monoid-actions as a means of transforming monoids, covered in the previous article, leading to increased applications.

![How to create a custom Donchian Channel indicator using MQL5](https://c.mql5.com/2/55/donchian_channel_indicator_avatar.png)[How to create a custom Donchian Channel indicator using MQL5](https://www.mql5.com/en/articles/12711)

There are many technical tools that can be used to visualize a channel surrounding prices, One of these tools is the Donchian Channel indicator. In this article, we will learn how to create the Donchian Channel indicator and how we can trade it as a custom indicator using EA.

![Rebuy algorithm: Math model for increasing efficiency](https://c.mql5.com/2/54/mathematical_model_to_increase_efficiency_Avatar.png)[Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)

In this article, we will use the rebuy algorithm for a deeper understanding of the efficiency of trading systems and start working on the general principles of improving trading efficiency using mathematics and logic, as well as apply the most non-standard methods of increasing efficiency in terms of using absolutely any trading system.

![Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://c.mql5.com/2/54/perceptron_avatar.png)[Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)

The article provides an example of using a perceptron as a self-sufficient price prediction tool by showcasing general concepts and the simplest ready-made Expert Advisor followed by the results of its optimization.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/12550&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069139751363084480)

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