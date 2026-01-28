---
title: Visualize a Strategy in the MetaTrader 5 Tester
url: https://www.mql5.com/en/articles/403
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:01:07.238034
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/403&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071509555403237729)

MetaTrader 5 / Examples


We all know the saying "Better to see once than hear a hundred times". You can read various books about Paris or Venice, but based on the mental images you wouldn't have the same feelings as on the evening walk in these fabulous cities. The advantage of visualization can easily be projected on any aspect of our lives, including work in the market, for example, the analysis of price on charts using indicators, and of course, the visualization of strategy testing.

### Does everyone know the possibilities of the strategy tester?

As practice shows, not everyone. The advantage of the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading platform over its competitors is not only a more convenient user-friendly interface, ample opportunities of the terminal and [MQL5](https://www.metatrader5.com/en/automated-trading/mql5 "https://www.metatrader5.com/en/automated-trading/mql5"), [a multicurrency tester](https://www.metatrader5.com/en/automated-trading/strategy-tester "https://www.metatrader5.com/en/automated-trading/strategy-tester") with the possibility of cloud computing using the [MQL5 Cloud Network](https://cloud.mql5.com/ "https://cloud.mql5.com/") and many other options. The advantage is also that all the tools that a trader needs are available in one place.

The purpose of this article is to show traders one more important feature of all the possibilities of MetaTrader 5 - namely visualization of strategy testing and optimization. Analysis of the behavior of an Expert Advisor on historical data and selection of the best parameters is not only a laborious analysis of figures, trades, balance, equity, etc. Sit down in a chair, make yourself comfortable, put on your "3D" glasses and let's start.

### And I think to myself what a wonderful world chart...

When publishing Expert Advisors in the [Code Base](https://www.mql5.com/en/code) or the [Market](https://www.mql5.com/en/market), authors usually attach a statistical report on their testing, as well as the balance and equity graphs. However, more interesting charts are available in the statistics on the ["Results"](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") tab of the strategy tester:

![Charts of testing results](https://c.mql5.com/2/4/tester_stat__1.png)

Based on these charts you can analyze the months, days or hours, when your EA trades best, when it actively enters the market. You can also evaluate the performance of the EA in terms of whether profitable positions are closed in proper time, whether the EA "sits out" losses based on the MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion) distribution graphs:

![MFE/MAE distribution charts](https://c.mql5.com/2/4/mfe_mae.png)

Further scrolling down the testing results, you will find another graph:

![Distribution of profits and position holding time](https://c.mql5.com/2/4/tester_result_profit.png)

The diagram shows the dependence of the position profit on its lifetime. This diagram will be very helpful for the upcoming [Automated Trading Championship 2012](https://championship.mql5.com/2012/en "https://championship.mql5.com/2012/en"). One of the ATC rules prohibits scalping. Test your Expert Advisor and make sure it complies with the rules.

### A Historical Movie or Visual Testing

One of the revolutionary features of the strategy tester is its [visual testing mode](https://www.metatrader5.com/en/terminal/help/algotrading/visualization "https://www.metatrader5.com/en/terminal/help/algotrading/visualization"). Well, the analysis of deals by dates, plotting of charts and other "interesting" routine procedures constitute quite a complicated process. However, the strategy tester is like giving you a player remote control, you press "Play" and watch a historical movie. You can slow down or speed up the film, stop it and analyze the situation. In real time you see how charts are constructed based on the simulated historical data and how the Expert Advisor responses to the price changes.

YouTube

The visualizer fully supports multicurrency, like the strategy tester. The Expert Advisor in the video uses 4 currency pairs for trading. All simulated prices are available in the Market Watch and on the charts.

### 2D Optimization Results

Before we introduce a three-dimensional image that has become quite popular among television manufacturers, have a look at the 2D visualization in the strategy tester. The rectangles that may seem strange at first glance, allow you to see the mutual influence of two optimization parameters on the optimization criterion (in our case it is the maximum balance value). The darker the shade of green, the higher the balance:

![2D Optimization Results](https://c.mql5.com/2/4/2d__1.png)

We see that the relationship is of a wave-like character, and the maximum result is achieved when using the mean values ​​of the period and shift of the moving average. These are the results of optimization of the _Moving Average_ EA, which is available in the standard distribution pack of the terminal. Even here you can find something interesting.

### Now in 3D

A three-dimensional image provides even better visualization options. Below is the same dependence of the two parameters and the final result. You can switch to this advanced mode using the context menu of the [Optimization Graph](https://www.metatrader5.com/en/terminal/help/algotrading/testing "https://www.metatrader5.com/en/terminal/help/algotrading/testing") tab.

![3D Optimization Results](https://c.mql5.com/2/4/3d.png)

After the three-dimensional optimization appeared in the MetaTrader 5 Strategy Tester, traders started to post their own examples of visualization of mathematical calculations on [the forum](https://www.mql5.com/ru/forum/3445):

![](https://c.mql5.com/2/4/sink_3d__1.png)

Three-dimensional graphics are fully interactive - you can zoom in and out, rotate to a convenient angle, etc. An important feature is that any results of testing and optimization can be exported as images or XML/HTML reports.

### Visualizing Optimization

And finally welcome - [work with optimization results](https://www.mql5.com/en/docs/optimization_frames)! To process results, the trader had to prepare data, download them and process somewhere. Now you can do it on the spot during optimization! To demonstrate this possibility, we need several header files that implement the simplest examples of such processing.

Download the MQH files attached below and save them in folder MQL5\\Include. Take any Expert Advisor and paste this block at the end of its code:

```
//--- Add a code for working with optimization results
#include <FrameGenerator.mqh>
//--- генератором фреймов
CFrameGenerator fg;
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
//--- Insert here your own function for calculating the optimization criterion
   double TesterCritetia=MathAbs(TesterStatistics(STAT_SHARPE_RATIO)*TesterStatistics(STAT_PROFIT));
   TesterCritetia=TesterStatistics(STAT_PROFIT)>0?TesterCritetia:(-TesterCritetia);
//--- Call at each end of testing and pass the optimization criterion as a parameter
   fg.OnTester(TesterCritetia);
//---
   return(TesterCritetia);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
//--- Prepare the chart to show the balance graphs
   fg.OnTesterInit(3); //The parameter sets the number of balance lines on the chart
  }
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
//--- Process the testing results and show the graphics
   fg.OnTesterPass();
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//--- End of optimization
   fg.OnTesterDeinit();
  }
//+------------------------------------------------------------------+
//|  Handling of events on the chart                           |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   //--- Starts frames after the end of optimization when clicking on the header
   fg.OnChartEvent(id,lparam,dparam,sparam,100); // 100 is a pause between frames in ms
  }
//+------------------------------------------------------------------+
```

We use the Expert Advisor _Moving Averages.mq5_ available in the standard delivery pack. Paste the code and save the Expert Advisor as _Moving Averages With Frames.mq5._ Compile and run the optimization.

ma frames - YouTube

[Photo image of MQL5.community](https://www.youtube.com/channel/UC8bwZMk2yh5WDToFos49OjQ?embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F403)

MQL5.community

1.91K subscribers

[ma frames](https://www.youtube.com/watch?v=e3P91OPNnsk)

MQL5.community

Search

Watch later

Share

Copy link

Info

Shopping

Tap to unmute

If playback doesn't begin shortly, try restarting your device.

More videos

## More videos

Share

Include playlist

An error occurred while retrieving sharing information. Please try again later.

[Watch on](https://www.youtube.com/watch?v=e3P91OPNnsk&embeds_referring_euri=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F403)

0:00

0:00 / 0:46

•Live

•

Thus, traders now can run visual optimization and process its results or display the required information during the optimization process.

### Do You Want More?

We keep expanding the possibilities of the MetaTrader 5 trading platform by adding new tools to help traders. Post your comments and share your ideas on how to further improve visualization in the strategy tester. We will try to implement the most interesting and useful of them.

### How to Use the Files

All files with the .mqh extension should be put in the MQL5\\Include folder, that's where the compiler will look for them when compiling the _Moving Average\_With\_Frames.mq5_ Expert Advisor. The EA files can be put right in the MQL5\\Experts folder or in any of its sub-folders.

Start the attached Exert Advisors yourself in the MetaTrader 5 Strategy Tester to make the reading of this article even more interesting. You will like it for sure.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/403](https://www.mql5.com/ru/articles/403)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/403.zip "Download all attachments in the single ZIP archive")

[multimovings.mq5](https://www.mql5.com/en/articles/download/403/multimovings.mq5 "Download multimovings.mq5")(10.95 KB)

[simpletable.mqh](https://www.mql5.com/en/articles/download/403/simpletable.mqh "Download simpletable.mqh")(10.79 KB)

[specialchart.mqh](https://www.mql5.com/en/articles/download/403/specialchart.mqh "Download specialchart.mqh")(7.79 KB)

[moving\_average\_with\_frames.mq5](https://www.mql5.com/en/articles/download/403/moving_average_with_frames.mq5 "Download moving_average_with_frames.mq5")(8.64 KB)

[colorprogressbar.mqh](https://www.mql5.com/en/articles/download/403/colorprogressbar.mqh "Download colorprogressbar.mqh")(4.89 KB)

[framegenerator.mqh](https://www.mql5.com/en/articles/download/403/framegenerator.mqh "Download framegenerator.mqh")(15.08 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/6907)**
(34)


![Marsel](https://c.mql5.com/avatar/2018/8/5B6DB2D4-F590.jpg)

**[Marsel](https://www.mql5.com/en/users/marsel)**
\|
27 Sep 2012 at 09:39

**Integer:**

Application files are not loaded: framegenerator.mqh, colourprogressbar.mqh, error 404.

Fixed. Thanks.

![Maxim Khrolenko](https://c.mql5.com/avatar/2018/1/5A6B2B7F-D1C2.png)

**[Maxim Khrolenko](https://www.mql5.com/en/users/paladin800)**
\|
27 Apr 2013 at 17:55

I would like it to be possible to save it as a file after optimisation, so that it can be loaded into MT5 at any time to view e.g. 2D, 3D charts. 2D, 3D charts. It's not about writing data to Excel and then processing it (that's what people do nowadays), it would be nicer to have it in MT5.


![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
27 Apr 2013 at 18:00

**paladin800:**

I would like it to be possible to save it as a file after optimisation, so that it can be loaded into MT5 at any time to view e.g. 2D, 3D charts. 2D, 3D charts. It is not about the fact that the data can be written to Excel and then processed (that's what people do nowadays), it would be more pleasant in MT5.

There is already such a request in Service Desk and I think even Renat approved it on the forum once (if I'm not confused). So we can only wait.

You can also write a proposal to Service Desk, maybe in this way its priority will increase. ))

![Bogdan Chirukin](https://c.mql5.com/avatar/2019/10/5DB0ACA0-0D60.JPG)

**[Bogdan Chirukin](https://www.mql5.com/en/users/cheater.wot)**
\|
26 Oct 2019 at 08:53

Is it possible to export [test results](https://www.metatrader5.com/en/terminal/help/algotrading/testing "Synopsis: Test results") to a file?

![ddcmql](https://c.mql5.com/avatar/avatar_na2.png)

**[ddcmql](https://www.mql5.com/en/users/ddcmql)**
\|
14 Oct 2020 at 22:14

Is it possible to customize the graphs in the [strategy tester](https://www.mql5.com/en/articles/239 "Article \"The Fundamentals of Testing in MetaTrader 5\"") report to be based on a custom optimization criteria? Eg my EA is optimizing for win rate, not profit. Can I have the graphs showing win rate by month, day of week, hour of day, month? Thanks!


![The Golden Rule of Traders](https://c.mql5.com/2/12/1021_34.png)[The Golden Rule of Traders](https://www.mql5.com/en/articles/1349)

In order to make profits based on high expectations, we must understand three basic principles of good trading: 1) know your risk when entering the market; 2) cut your losses early and allow your profit to run; 3) know the expectation of your system – test and adjust it regularly. This article provides a program code trailing open positions and actualizing the second golden principle, as it allows profit to run for the highest possible level.

![Get 200 usd for your algorithmic trading article!](https://c.mql5.com/2/0/new_article_system.png)[Get 200 usd for your algorithmic trading article!](https://www.mql5.com/en/articles/408)

Write an article and contribute to the development of algorithmic trading. Share your experience in trading and programming, and we will pay you $200. Additionally, publishing an article on the popular MQL5.com website offers an excellent opportunity to promote your personal brand in a professional community. Thousands of traders will read your work. You can discuss your ideas with like-minded people, gain new experience, and monetize your knowledge.

![Limitless Opportunities with MetaTrader 5 and MQL5](https://c.mql5.com/2/0/TW_logoMarket_60x60.png)[Limitless Opportunities with MetaTrader 5 and MQL5](https://www.mql5.com/en/articles/392)

In this article, I would like to give an example of what a trader's program can be like as well as what results can be achieved in 9 months, having started to learn MQL5 from scratch. This example will also show how multi-functional and informative such a program can be for a trader while taking minimum space on the price chart. And we will be able to see just how colorful, bright and intuitively clear to the user trade information panels can get. As well as many other features...

![Kernel Density Estimation of the Unknown Probability Density Function](https://c.mql5.com/2/0/Kernel_Density_Estimation_MQL5.png)[Kernel Density Estimation of the Unknown Probability Density Function](https://www.mql5.com/en/articles/396)

The article deals with the creation of a program allowing to estimate the kernel density of the unknown probability density function. Kernel Density Estimation method has been chosen for executing the task. The article contains source codes of the method software implementation, examples of its use and illustrations.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/403&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071509555403237729)

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