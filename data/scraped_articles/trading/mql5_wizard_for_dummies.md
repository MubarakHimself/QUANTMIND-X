---
title: MQL5 Wizard for Dummies
url: https://www.mql5.com/en/articles/287
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:22:36.853446
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ymefmiyspbgeibewpnlezwnvsojkgdxa&ssn=1769181755378598880&ssn_dr=0&ssn_sr=0&fv_date=1769181755&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F287&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20for%20Dummies%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918175599019594&fz_uniq=5069387691235148859&sv=2552)

MetaTrader 5 / Examples


In early 2011 we released the first version of the [MQL5 Wizard](https://www.metatrader5.com/en/automated-trading/mql5wizard "MQL5 Wizard"). This new application provides traders a simple and convenient tool to automatically generate trading robots. Any [MetaTrader 5](https://www.metatrader5.com/ "MetaTrader 5") user can create a custom Expert Advisor without even knowing how to program in MQL5.

In the [new version of the Wizard](https://www.metatrader5.com/en/news/122 "New MQL5 Wizard: A Fully Functional Robot for Every Trader") we have expanded the functionality of the program. Now it enables you to create Expert Advisors based on a combination of several signals. This innovation allows the use of sophisticated analysis in an Expert Advisor to get detailed accurate signals. Nevertheless, this innovation does not complicate the process of generating an EA. It still implies a step-by-step selection of required parameters as a base for EA construction.

Let's consider each step separately and go through all the steps for creating an Expert Advisor. First you will need to define the tool and the timeframe, on which the EA will trade: EUR/USD and M10. We will use the following signals:

- EMA('EURUSD',M10,31) - An exponential [moving average](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma "Moving Average");

- Stochastic('EURUSD',M10,8,3,3) - [Stochastic oscillator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_stochastic "Stochastic Oscillator");

- EMA('EURUSD',H1,24) - An exponential [moving average](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_ma "Moving Average") from another timeframe to confirm the first EMA;

- Stochastic('EURJPY',H4,8,3,3) - [Stochastic oscillator](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_stochastic "Stochastic Oscillator") from another symbol and timeframe to confirm the first one;

- IntradayTimeFilter – A [time filter](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal/signal_time_filter "Intraday Time Filter"), which reveals the efficiency of all other signals during certain hours and days of the week.


Thus, we have outlined the basic parameters for our Expert Advisor. Now we can start working in the MQL5 Wizard. To start the program, open the MetaEditor program and click "Create" in the "File" tab of the main menu. In the appeared window, select "Generate Expert Advisor”:

![Generate Expert Advisor](https://c.mql5.com/2/3/Create_Expert_en.png)

In the next window, specify the symbol and the timeframe, on which the EA will trade. If we leave the default current value, we will receive a universal Expert Advisor able to trade on any symbol and timeframe to which it is connected. But we want to create an EA for intraday trading on EURUSD, as if applied to other instruments it may produce unexpected results. Therefore specify EURUSD and M10.

![General Properties of the Expert Advisor](https://c.mql5.com/2/3/Properties_of_EA_en.png)

Now proceed to the most interesting step - select the signals, based on which our Expert Advisor will trade. At this point the standard library includes [ready-made modules of 20 trading signals](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal "Ready-made Modules of 20 Trading Signals"), which are based on the logic of standard indicators. We select the necessary symbols among them.

![Ready-made Modules of Trading Signals](https://c.mql5.com/2/3/Parameters_of_Signal_Module_en.png)

Then, configure the parameters for each signal we have chosen - from the MA to IntradayTimeFilter.

![Signals of Moving Average](https://c.mql5.com/2/3/Signals_of_MA_en.png)

Each selected signal has its own set of parameters. For example, for the exponential MA we need to specify its period (31), its shift from the current bar (0), the averaging method (Exponential), the price to apply the MA (Close price) and the weight of the signal (1.0).

A detailed description of each trading signal can be found in the [MQL5 Reference](https://www.mql5.com/en/docs/standardlibrary/expertclasses/csignal).

![Selected Signals](https://c.mql5.com/2/3/Selected_Signals_en.png)

After we've set up the parameters of all signals, our Expert Advisor is almost ready. Now, we only need to configure trailing stop and money management modules. However, this step is beyond the scope of this article. Therefore, our Expert Advisor will be constructed without trailing, and will trade the fixed lot.

For further study on how to configure these parameters, please read the article [MQL5 Wizard: Creating Expert Advisors without Programming](https://www.mql5.com/en/articles/171 "MQL5 Wizard: Creating Expert Advisors without Programming").

After all above steps, we must select a money management strategy and return to the MetaEditor where the code of the resulting Expert Advisor is available.

[![Code of the Resulting Expert Advisor](https://c.mql5.com/2/3/trend_expert2__1.png)](https://c.mql5.com/2/3/trend_expert2.png)

To compile the EA, press "Compile" in the control panel. After this, start the MetaTrader 5 Client Terminal and run the Expert Advisor by selecting it in the Navigator -> Expert Advisors.

[![Created Expert Advisor in MetaTrader 5 Strategy Tester](https://c.mql5.com/2/3/MQL5_robot_test_en__1.png)](https://c.mql5.com/2/3/MQL5_robot_test_en.png)

So we've created a full-fledged Expert Advisor with minimum time and effort. Now any MetaTrader 5 user can create an EA just as easy and quick.

The MQL5 Wizard is a powerful tool for creating trading robots. Now anyone can create a fully functional Expert Advisor regardless of programming skills and experience. In a few clicks you select the options you want - and the Expert Advisor created will be trading according to your strategy.

Try MQL5 Wizard today and create EAs with ease according to your specific trading strategy!

[Download MetaTrader 5](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/287](https://www.mql5.com/ru/articles/287)

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
**[Go to discussion](https://www.mql5.com/en/forum/3835)**
(17)


![Victor Kirillin](https://c.mql5.com/avatar/avatar_na2.png)

**[Victor Kirillin](https://www.mql5.com/en/users/unclevic)**
\|
19 Aug 2011 at 11:59

**Medvedev:**

Really want to redo to 3 instruments. So far the class has not been redesigned?

Just trying to initialise 3 objects:

CExpert ExtExpert1;

CExpert ExtExpert2;

CExpert ExtExpert3;

and it still says wrong character.

Last year I managed to convert to multicurrency this example -

[https://www.mql5.com/en/articles/148?source=metaeditor5\_article](https://www.mql5.com/en/articles/148?source=metaeditor5_article "https://www.mql5.com/en/articles/148?source=metaeditor5_article")

but the expert created by wizard does not make a multicurrency ;-((((.

Regards, Andrey.

Try to look [here](https://www.mql5.com/ru/forum/3948/page6#comment_88528).


![Medvedev](https://c.mql5.com/avatar/avatar_na2.png)

**[Medvedev](https://www.mql5.com/en/users/medvedev)**
\|
22 Aug 2011 at 10:30

Great! Thank you so much!

This is just what I needed. I didn't think of how to fix it, but here it is and it works.

All that's left is to choose pairs, make optimisation and launch multicurrency for the championship!

very happy!!!!!! ;-)))

![Владимир](https://c.mql5.com/avatar/2014/2/52F3DA59-AEFA.png)

**[Владимир](https://www.mql5.com/en/users/erm955)**
\|
22 Aug 2011 at 14:05

**Medvedev:**

Great! Thank you so much!

This is just what I needed. I didn't think of how to fix it, but here it is and it works.

All that's left is to choose pairs, make optimisation and launch multicurrency for the championship!

very happy!!!!!! ;-)))

[![](https://c.mql5.com/3/5/cerbre.PNG)](https://c.mql5.com/3/5/kzrwo6.PNG "https://c.mql5.com/3/5/kzrwo6.PNG")

By the way, this is how you can do a run on all symbols one by one.

![kapitan444](https://c.mql5.com/avatar/avatar_na2.png)

**[kapitan444](https://www.mql5.com/en/users/kapitan444)**
\|
27 Sep 2011 at 12:18

**bulat-latypov:**

The article [MQL5 Wizard for "Dummies"](https://www.mql5.com/en/articles/287) has been published:

Author: [Bulat Latypov](https://www.mql5.com/en/users/bulat-latypov "https://www.mql5.com/en/users/bulat-latypov")

**bulat-latypov:**

The article [MQL5 Wizard for "Dummies"](https://www.mql5.com/en/articles/287) has been published:

Author: [Bulat Latypov](https://www.mql5.com/en/users/bulat-latypov "https://www.mql5.com/en/users/bulat-latypov")

I have done everything as described in the article, but the Expert Advisor is not tested and is not installed on the chart. I have absolutely no programming experience, so please help!


![farhadmax](https://c.mql5.com/avatar/avatar_na2.png)

**[farhadmax](https://www.mql5.com/en/users/farhadmax)**
\|
10 Jul 2023 at 16:25

the 4th signal's symbol ( [Stochastics](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "MetaTrader 5 Help: Stochastic Oscillator Indicator")(8,3,3)) in the wizard should be "EURJPY"


![The Fundamentals of Testing in MetaTrader 5](https://c.mql5.com/2/0/tester_basis_Metatrader5__1.png)[The Fundamentals of Testing in MetaTrader 5](https://www.mql5.com/en/articles/239)

What are the differences between the three modes of testing in MetaTrader 5, and what should be particularly looked for? How does the testing of an EA, trading simultaneously on multiple instruments, take place? When and how are the indicator values calculated during testing, and how are the events handled? How to synchronize the bars from different instruments during testing in an "open prices only" mode? This article aims to provide answers to these and many other questions.

![Statistical Estimations](https://c.mql5.com/2/0/MQL5_Statistical_estimation.png)[Statistical Estimations](https://www.mql5.com/en/articles/273)

Estimation of statistical parameters of a sequence is very important, since most of mathematical models and methods are based on different assumptions. For example, normality of distribution law or dispersion value, or other parameters. Thus, when analyzing and forecasting of time series we need a simple and convenient tool that allows quickly and clearly estimating the main statistical parameters. The article shortly describes the simplest statistical parameters of a random sequence and several methods of its visual analysis. It offers the implementation of these methods in MQL5 and the methods of visualization of the result of calculations using the Gnuplot application.

![Advanced Adaptive Indicators Theory and Implementation in MQL5](https://c.mql5.com/2/0/Advanced_adaptive_indicators_MQL5.png)[Advanced Adaptive Indicators Theory and Implementation in MQL5](https://www.mql5.com/en/articles/288)

This article will describe advanced adaptive indicators and their implementation in MQL5: Adaptive Cyber Cycle, Adaptive Center of Gravity and Adaptive RVI. All indicators were originally presented in "Cybernetic Analysis for Stocks and Futures" by John F. Ehlers.

![MQL5 Wizard: New Version](https://c.mql5.com/2/0/New_Master_MQL5.png)[MQL5 Wizard: New Version](https://www.mql5.com/en/articles/275)

The article contains descriptions of the new features available in the updated MQL5 Wizard. The modified architecture of signals allow creating trading robots based on the combination of various market patterns. The example contained in the article explains the procedure of interactive creation of an Expert Advisor.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qhnfcnhfxcnrmpvpezjwbmigencjmsco&ssn=1769181755378598880&ssn_dr=0&ssn_sr=0&fv_date=1769181755&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F287&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20for%20Dummies%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918175598978591&fz_uniq=5069387691235148859&sv=2552)

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