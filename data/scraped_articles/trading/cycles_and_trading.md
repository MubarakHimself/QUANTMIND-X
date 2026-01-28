---
title: Cycles and trading
url: https://www.mql5.com/en/articles/16494
categories: Trading, Trading Systems, Expert Advisors, Strategy Tester
relevance_score: 0
scraped_at: 2026-01-24T13:27:50.045379
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/16494&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082872359536300196)

MetaTrader 5 / Tester


### Introduction

The main task facing a trader is to predict price movements. Traders build their forecasts based on one model or another. One of the simplest and most visual models is the cyclical price movement model.

The basic idea behind any cyclical pattern is that various factors interact to create cycles in price movement. These cycles may differ from each other in their duration and strength. If you know the parameters of these cycles, then trading operations will become very simple: open a buy position when the cycle has reached its minimum, sell when the cycle has reached its maximum.

Let's see how this model can be used in practice.

### Simple cycle

When describing cycles, we usually apply [sine](https://www.mql5.com/en/docs/math/mathsin) and [cosine](https://www.mql5.com/en/docs/math/mathcos) trigonometric functions. But the cycle can be defined in another way - with the help of [finite differences](https://en.wikipedia.org/wiki/Finite_difference "https://en.wikipedia.org/wiki/Finite_difference").

For example, let's consider an equation in which the 2nd difference is proportional to the value of the time series:

![](https://c.mql5.com/2/156/01__22.png)

This is a [harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator "https://en.wikipedia.org/wiki/Harmonic_oscillator") equation. At first glance, there is nothing special about it. But this equation has one interesting property. If the value of the R ratio lies in the range from 0 to 4, then this equation can only belong to a sinusoid. This ratio depends on the N cycle period. In other words, its value can be chosen such that the equation reacts only to cycles with a certain frequency:

![](https://c.mql5.com/2/156/01__23.png)

Now let's see how this equation can be applied in practice. If we apply it to p\[\] prices, we will need the value of the m level the price fluctuates around.

![](https://c.mql5.com/2/156/01__24.png)

To avoid calculating the value of this average level, I will resort to a little trick. Let's remember what happens mathematically when a cycle reaches a maximum or minimum point. At these moments, the 1st derivative changes sign.

Instead of the derivative, we will use the 1st difference of the original equation. This is the difficult fate of a trader - to look for differences from differences. But this approach greatly simplifies the calculations. We only need to set the value of the R ratio. The final form of the equation is as follows:

![](https://c.mql5.com/2/156/01__25.png)

Further, I will call such equations trading algorithm equations. Based on this equation we can create a trading strategy. First, I will make a small addition - we can use any version of the conventional [moving average](https://www.mql5.com/en/docs/indicators/ima) instead of the current price. The essence of the strategy itself is very simple. Opening and closing positions occurs when the sign of the equation changes:

- from negative to positive - open buy, close sell;
- from positive to negative - open sell, close buy.

The EA balance change can be like this.

![](https://c.mql5.com/2/157/1__4.png)

This trading strategy has prospects. But it needs some improvement.

### Cycle with damping

The strategy we looked at earlier has one serious drawback. It is based on the assumption that the cycle will continue for quite a long time. In real life, the cycle depends on many external factors and can collapse almost immediately after it begins.

To get rid of this drawback, we can use the harmonic oscillator model with damping of oscillations. The equation of such a cycle is also set by finite differences:

![](https://c.mql5.com/2/156/01__26.png)

The S parameter determines how quickly the oscillations die out. The higher its value, the faster the damping occurs. The critical value of this parameter can be found using the following equation:

![](https://c.mql5.com/2/156/01__27.png)

Starting from this value, the oscillator's movement becomes non-oscillatory.

We already know what to do with such an equation. We need to find its 1st difference, which will generate trading signals.

By adding a new parameter, the trading strategy may become more flexible and profitable. For example, this is what the balance change looks like at a critical value of S.

![](https://c.mql5.com/2/157/2__4.png)

By definition, parameter S should be positive. But this requirement is not mandatory. If we set a negative value for this parameter, the original equation will describe oscillations with an ever-increasing amplitude. In this way, EA can react to the beginning of the cycle and possibly increase profitability. The trading results in this case may look like this:

![](https://c.mql5.com/2/157/3__4.png)

As you can see, making the cycle model more complex can improve trading results. Let's try to complicate the model even more.

### Elliott Waves

[Elliott Wave Principle](https://en.wikipedia.org/wiki/Elliott_wave_principle "https://en.wikipedia.org/wiki/Elliott_wave_principle") has been known for a very long time and is widely used in trading. Let's try to apply the cyclical model to Elliott waves.

The first five Elliott waves can be simulated as the sum of two sinusoids with different amplitudes and periods:

![](https://c.mql5.com/2/156/01__28.png)

In this case, the following conditions should be met:

![](https://c.mql5.com/2/156/01__29.png)

As a result, we can get something like this:

![](https://c.mql5.com/2/156/4__4.png)

It does not look exactly like in the books - a little angular, but pretty similar.

We can simulate the sum of two sinusoids using finite differences:

![](https://c.mql5.com/2/156/01__30.png)

where  ![](https://c.mql5.com/2/156/01__31.png)  — the value of the finite difference of order n, on the price reading with index i. We already know what to do with this equation - find the difference from the sum of the differences and get trading signals:

![](https://c.mql5.com/2/156/01__32.png)

Using the sum of two sinusoids makes it possible to adjust the EA to more complex market situations. Using this trading algorithm can lead to both an increase in market entries and an improvement in their profitability.

This is the result that an EA based on the Elliott wave principle can show:

![](https://c.mql5.com/2/157/5__4.png)

As you can see, even a small complication of the cyclical model can bring positive results.

### Generalized harmonic oscillator

We have already looked at three models based on the harmonic oscillator. Each of these models was built on its own initial assumptions. But as a result, we still got some sum of finite differences.

We can build a more general model. The essence of this model is very simple - first we take the difference of a predetermined order and add them to the differences of lower orders, down to the 1st one. In this case, each difference of low order is taken with its own weighting ratio.

For example, I took the difference of the 7th order. Then the equation for the trading algorithm will be as follows:

![](https://c.mql5.com/2/156/01__33.png)

Such a model can include a wide variety of combinations of cycles and their behavior. We have seen earlier that making the model more complex improves trading results. In this case, the balance change may be as follows:

![](https://c.mql5.com/2/157/6__4.png)

Increasing the order of the leading difference makes the model more flexible. The generalized oscillator will only react to certain price combinations. In other words, it looks for certain patterns. At the same time, you need to remember that the higher the order of the oscillator, the less often its patterns will occur.

Another feature of this algorithm is that the ratios can vary within very wide limits. Even assessing these limits can be difficult. On the other hand, the selection of ratios allows you to configure the algorithm in such a way that it will respond to non-cyclical patterns in price behavior.

### Non-linear oscillators

Linear models are simple and intuitive. But price behavior on the market can also be non-linear. For example, a sharp change in price is easier to describe using non-linear models. The main advantage of non-linear models is that they can describe very complex time series behavior. When you hear the words " [chaos](https://en.wikipedia.org/wiki/Chaos_theory "https://en.wikipedia.org/wiki/Chaos_theory") and " [stochasticity](https://en.wikipedia.org/wiki/Stochastic "https://en.wikipedia.org/wiki/Stochastic")", then most probably you are dealing with non-linear models.

One of the simplest models is the signature oscillator. Its equation is in many ways similar to the harmonic oscillator. The [sign function](https://en.wikipedia.org/wiki/Sign_function "https://en.wikipedia.org/wiki/Sign_function") is used as a non-linear function. The trading algorithm equation looks like this:

![](https://c.mql5.com/2/156/01__34.png)

where 'sign' is the sign function.

![](https://c.mql5.com/2/156/01__35.png)

Thanks to this feature, the oscillator can operate in 5 different modes. Switching modes depends on the price movement. The non-linearity of the oscillator gives us hope that trading with its help will be more flexible and stable. The balance change using this algorithm might look like this:

![](https://c.mql5.com/2/157/7__3.png)

Now, let's consider an oscillator with quadratic non-linearity. Its equation is very simple:

![](https://c.mql5.com/2/156/01__36.png)

In order to apply this equation in practice, we need to refine it a little.

First of all, we need an average level around which the price fluctuates. Let me remind you that instead of real prices we can use their smoothed value (SMA). Then the average level will be equal to the average value of the moving averages.

Calculating the average of averages may seem a bit scary at first. In 1910, [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov "https://en.wikipedia.org/wiki/Andrey_Markov") published his work "Calculating Finite Differences". In this work, he showed that taking an average of a large number of SMAs is equivalent to applying an LWMA to all values of the time series. Once again, finite differences have simplified the calculations.

The equation for the quadratic oscillator trading algorithm will be as follows:

![](https://c.mql5.com/2/156/01__37.png)

Quadratic non-linearity allows switching the oscillator operating modes more smoothly, but with a larger amplitude. This can improve trading results.

![](https://c.mql5.com/2/157/8__3.png)

Another example of quadratic non-linearity is [van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator "https://en.wikipedia.org/wiki/Van_der_Pol_oscillator"). This oscillator was one of the first examples of simulating deterministic chaos. Due to its properties, the Van der Pol equation has found application not only in [oscillation theory](https://en.wikipedia.org/wiki/Oscillation_theory "https://en.wikipedia.org/wiki/Oscillation_theory"), but also in other areas - physics, biology, etc.

The van der Pol oscillator itself is a modification of the oscillator with quadratic non-linearity, and the equation for its trading algorithm looks like this:

![](https://c.mql5.com/2/156/01__38.png)

At first glance, this oscillator should be unstable. In fact, its parameters can change over a fairly wide range without affecting trading results.

![](https://c.mql5.com/2/157/9__3.png)

In 1918, [Georg Duffing](https://en.wikipedia.org/wiki/Georg_Duffing "https://en.wikipedia.org/wiki/Georg_Duffing") investigated the equation of an oscillator with cubic non-linearity. This oscillator can also simulate chaotic price behavior. But it has one more feature: depending on external conditions and its parameters, this oscillator can be in two stable states.

The equation of the Duffing Oscillator trading algorithm is the sum of a 3rd degree harmonic oscillator and [cubic binomial](https://en.wikipedia.org/wiki/Binomial_theorem "https://en.wikipedia.org/wiki/Binomial_theorem"):

![](https://c.mql5.com/2/156/01__39.png)

The performance of this oscillator is comparable to other non-linear models.

![](https://c.mql5.com/2/157/10__3.png)

In general, a lot of non-linear models can be created. Any non-linear function can become a chaos generator. The non-linearity of the three previous oscillators was based on some physical principles and ideas. Econophysics is a useful direction, but it is not at all necessary to follow it at all times.

For example, I will add non-linearity to the trading algorithm based on the logarithmic function:

![](https://c.mql5.com/2/156/01__40.png)

You may ask: where did these logarithms come from, and on what grounds did I add them? My answer will be very simple - I wanted it that way.

But if I have time to think about the answer, it will be a little different. What is the difference between consecutive prices? Let's say 100 points. And what about the difference between the logarithms of these prices? Will there be anything there besides zeros? Here is my new answer: on the left we have a regular oscillator, and on the right we have a filter that can influence the opening of positions. The filter will only show itself when the price moves strongly. We check its work.

![](https://c.mql5.com/2/157/11__3.png)

The result is similar to the operation of an oscillator with quadratic non-linearity. The filter is working. But it is important to remember that all ideas should first be tested in the tester before applying them in practice.

### External forces

So far we have been looking at autonomous oscillators. Imagine a pendulum. It swings evenly from side to side. Its cycles repeat over and over again. Why this pendulum swings, why it swings in this particular way - these are questions we cannot answer. This is an example of a free-running oscillator.

Now, push that pendulum. First one way, then the other. The pendulum cycles began to change - its instantaneous frequency and amplitude changed. Under the influence of external forces, the pendulum ceased to be autonomous.

Let's try to apply the same approach to the harmonic oscillator. Its equation will change and look like this:

![](https://c.mql5.com/2/156/01__41.png)

where F\[n\] is an external force. What kind of force is this, how does it act, where is it directed - I know nothing about it. I just assume that it exists. And now the question arises: is such a complication of the model justified? Let's answer it.

![](https://c.mql5.com/2/157/12__1.png)

I obtained this result by assuming that this external force is related to price values as follows:

![](https://c.mql5.com/2/156/01__42.png)

I made this assumption for only one reason - I talked about finite differences throughout the article, and in this case I also used the ratios of the 2nd difference. You are free to make your own assumptions. For example, external force may be related to the difference between closing and opening prices, tick volumes, etc. You can also add an element of non-linearity:

![](https://c.mql5.com/2/156/01__43.png)

After all, this unknown force may depend on several components. Summarize as much as you can as long as possible. Any assumption you make can have a positive impact on your trading results.

### Conclusion

As you can see, using cyclical patterns in trading is quite justified. The main advantage of such models is that they have hundreds of combinations of parameters that give a stable result over a long period of time. The trader needs to select several dozen options that do not correlate with each other.

In this article, I have only touched upon a few of the most popular oscillators. There are still quite a few solutions that can be used to build cyclic models (among other things).

I tested the EA under the following conditions: EURUSD, H1, 2024.01.01 - 2024.10.30

The following programs were used in writing this article:

| Name | Type | Description |
| --- | --- | --- |
| Harmonic Oscillator | EA | - **_iPeriod_** \- moving average period<br>- **_Type_** \- moving average type<br>- **_R_** \- oscillator ratio, permissible value: 1 ... 399 |
| Damped Harmonic Oscillator | EA | - **_S_** \- dumping ratio, permissible value: -1000 ... +1000 |
| scr Elliott wave | script | - **_A1_**, **_A2_** \- amplitudes of sinusoids<br>- **_N1_**, **_N2_** \- periods of sinusoids<br>- **_Width_** \- line width<br>- **_Pause_** \- delay in displaying the result<br>- **_Screenshot_** \- if 'true', then the image is created and saved in the Files folder |
| Elliott wave | EA | - **_R_**, **_S_** \- oscillator ratios, acceptable value: 0 ... 1000 |
| Generalized Harmonic Oscillator | EA | permissible value of ratios: -1000 ... +1000 |
| Sign Oscillator | EA |  |
| Quadratic Oscillator | EA |  |
| Van der Pol Oscillator | EA |  |
| Duffing Oscillator | EA |  |
| Log Oscillator | EA |  |
| Non-Autonomous Oscillator | EA |  |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16494](https://www.mql5.com/ru/articles/16494)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16494.zip "Download all attachments in the single ZIP archive")

[Harmonic\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/harmonic_oscillator.mq5 "Download Harmonic_Oscillator.mq5")(8.14 KB)

[Damped\_Harmonic\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/damped_harmonic_oscillator.mq5 "Download Damped_Harmonic_Oscillator.mq5")(8.36 KB)

[scr\_Elliott\_wave.mq5](https://www.mql5.com/en/articles/download/16494/scr_elliott_wave.mq5 "Download scr_Elliott_wave.mq5")(3.52 KB)

[Elliott\_wave.mq5](https://www.mql5.com/en/articles/download/16494/elliott_wave.mq5 "Download Elliott_wave.mq5")(8.43 KB)

[Generalized\_Harmonic\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/generalized_harmonic_oscillator.mq5 "Download Generalized_Harmonic_Oscillator.mq5")(9.42 KB)

[Sign\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/sign_oscillator.mq5 "Download Sign_Oscillator.mq5")(9.43 KB)

[Quadratic\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/quadratic_oscillator.mq5 "Download Quadratic_Oscillator.mq5")(9.16 KB)

[Van\_der\_Pol\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/van_der_pol_oscillator.mq5 "Download Van_der_Pol_Oscillator.mq5")(9.57 KB)

[Duffing\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/duffing_oscillator.mq5 "Download Duffing_Oscillator.mq5")(9.3 KB)

[Log\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/log_oscillator.mq5 "Download Log_Oscillator.mq5")(8.02 KB)

[Non-Autonomous\_Oscillator.mq5](https://www.mql5.com/en/articles/download/16494/non-autonomous_oscillator.mq5 "Download Non-Autonomous_Oscillator.mq5")(9.37 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How can century-old functions update your trading strategies?](https://www.mql5.com/en/articles/17252)
- [Polynomial models in trading](https://www.mql5.com/en/articles/16779)
- [Trend criteria in trading](https://www.mql5.com/en/articles/16678)
- [Cycles and Forex](https://www.mql5.com/en/articles/15614)
- [Practicing the development of trading strategies](https://www.mql5.com/en/articles/14494)
- [Angle-based operations for traders](https://www.mql5.com/en/articles/14326)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/491106)**
(11)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
16 Jul 2025 at 10:14

Its a great article, but it is also a Cookie Jar, so many alternatives to sample.  Have you considered creating an adaptive engine that evaluates the market conditions and attempts to select the best optimization for the current conditions


![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
16 Jul 2025 at 19:07

**CapeCoddah [#](https://www.mql5.com/ru/forum/477252#comment_57539277):**

This is a great article, but it's also a biscuit jar, so many alternatives to try. Have you considered building an adaptive mechanism that evaluates market conditions and tries to choose the best optimisation for current conditions

One way to adapt is to evaluate multiple cycles at once. Moreover, this is easier than it seems. For example, you can do it this way. The first cycle - take counts in a row. The second cycle - take the price samples one after another. And so on. The combination of these cycles will give a unique picture of the market state at the moment.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
17 Jul 2025 at 12:31

Thanks for the great suggestion.  I will try it and let you know but it will be a while.


![TraderW](https://c.mql5.com/avatar/avatar_na2.png)

**[TraderW](https://www.mql5.com/en/users/jj201)**
\|
19 Jul 2025 at 11:20

### 🚫 Red Flags:

1. **Unusable default parameters**:

   - iPeriod = 870 , R = -940 , S = 450 → absurd values for short-term trading
2. **No trades triggered**:

   - The EA evaluates the signal **only once per new bar**, and signal logic thresholds almost never hit with default parameters.
3. **CalcLWMA() uses static accumulators** in the original — causing totally invalid results over time.

4. **No [backtest](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") or validation in code** — and the **indicator isn’t provided in the article** for real-time visual inspection.

5. **Boasts of equity growth without sharable evidence or MQ5 Signals links**.


![Aleksej Poljakov](https://c.mql5.com/avatar/2017/5/591B60CC-05D2.png)

**[Aleksej Poljakov](https://www.mql5.com/en/users/aleksej1966)**
\|
19 Jul 2025 at 15:22

**TraderW backtests or validation in the code** \- and **no indicator is provided in the article** for real-time visual inspection. **Brags about stock growth without providing any proof or references to MQ5 Signals**.

1\. About the absurdity of the values - you know better

2\. The EA opens positions when a new bar is opened. If you need some other logic, you can implement everything yourself

3\. it should be clear from the article that CalcWMA is used to calculate the average of all SMAs.

![MQL5 Trading Tools (Part 4): Improving the Multi-Timeframe Scanner Dashboard with Dynamic Positioning and Toggle Features](https://c.mql5.com/2/157/18786-mql5-trading-tools-part-4-improving-logo.png)[MQL5 Trading Tools (Part 4): Improving the Multi-Timeframe Scanner Dashboard with Dynamic Positioning and Toggle Features](https://www.mql5.com/en/articles/18786)

In this article, we upgrade the MQL5 Multi-Timeframe Scanner Dashboard with movable and toggle features. We enable dragging the dashboard and a minimize/maximize option for better screen use. We implement and test these enhancements for improved trading flexibility.

![From Basic to Intermediate: Recursion](https://c.mql5.com/2/102/Do_bbsico_ao_intermedilrio_Recursividade__LOGO.png)[From Basic to Intermediate: Recursion](https://www.mql5.com/en/articles/15504)

In this article we will look at a very interesting and quite challenging programming concept, although it should be treated with great caution, as its misuse or misunderstanding can turn relatively simple programs into something unnecessarily complex. But when used correctly and adapted perfectly to equally suitable situations, recursion becomes an excellent ally in solving problems that would otherwise be much more laborious and time-consuming. The materials presented here are intended for educational purposes only. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Implementing Practical Modules from Other Languages in MQL5 (Part 02): Building the REQUESTS Library, Inspired by Python](https://c.mql5.com/2/157/18728-implementing-practical-modules-logo.png)[Implementing Practical Modules from Other Languages in MQL5 (Part 02): Building the REQUESTS Library, Inspired by Python](https://www.mql5.com/en/articles/18728)

In this article, we implement a module similar to requests offered in Python to make it easier to send and receive web requests in MetaTrader 5 using MQL5.

![Developing a Replay System (Part 75): New Chart Trade (II)](https://c.mql5.com/2/102/Desenvolvendo_um_sistema_de_Replay_Parte_75___LOGO.png)[Developing a Replay System (Part 75): New Chart Trade (II)](https://www.mql5.com/en/articles/12442)

In this article, we will talk about the C\_ChartFloatingRAD class. This is what makes Chart Trade work. However, the explanation does not end there. We will complete it in the next article, as the content of this article is quite extensive and requires deep understanding. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/16494&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082872359536300196)

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