---
title: Practical application of neural networks in trading
url: https://www.mql5.com/en/articles/7031
categories: Trading Systems, Integration
relevance_score: 3
scraped_at: 2026-01-23T19:35:54.540650
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/7031&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070414098224584041)

MetaTrader 5 / Examples


### Introduction

This article considers the application of neural networks when creating trading robots. This is the narrow sense of this problem. More broadly, we will try to answer some questions and to address several problems:

1. Can a profitable system be created using machine learning?
2. What can a neural network give us?
3. The rationale for training neural networks for decision making.
4. Neural Network: is it difficult or simple?
5. How to integrate a neural network into a trading terminal?
6. How to test a neural network? Testing stages.
7. About training samples.

### 1\. Can a profitable system be created using machine learning?

Probably, many beginners who have just started practicing real trading in the forex market without any particular system, took a sheet of paper and wrote a list of possibly suitable indicators, putting plus or minus sign next to them, or arrows, or price movement probability, based on the indicator chart in the terminal. Then the user would sum up observations and makes a certain decision to enter the market in a certain direction (or a decision of whether it is a good time to enter the market or better wait for another opportunity).

So, what happens in the most advanced neural network, i.e. our brain? After the observation of indicators, we have some image of a composite indicator generating the final signal, based on which we make a decision. Or a chain of signals is compiled into an indicator. Think about the following: if we study indicators at a certain point in time and look into the past for a maximum of several periods, how can we study these indicators simultaneously over several previous years and then compile a single composite indicator, which can further be optimized.

This is the answer to our second question: What can a neural network give us? Let's rephrase the question: what do you want to obtain form a neural network as a result of its training? Logically, the first question can also have an affirmative answer. This can be done programmatically. You can see how this is implemented in practice by watching my video: [https://youtu.be/5GwhRnSqT78](https://www.mql5.com/go?link=https://youtu.be/5GwhRnSqT78 "https://youtu.be/5GwhRnSqT78"). You can also watch the video playlist featuring online testing of neural network modules at [https://youtu.be/3wEMQOXJJNk](https://www.mql5.com/go?link=https://youtu.be/3wEMQOXJJNk "https://youtu.be/3wEMQOXJJNk")

### 2\. The rationale for training neural networks for decision making.

Before starting the development of any trading system, answer the following question: On what principles will this system function? We have two fundamental principles: trading flat and trend continuation. We will not consider derivatives from these two systems, such as intraday trading, use of fundamental data and news, trading at market opening time, etc. I came across descriptions of neural network products, in which authors suggested using them to forecast prices, such as stocks, currencies and so on.

![Chart shows the operation of a neural network trained for price forecast](https://c.mql5.com/2/36/aoh9lg_2019_07_09_12_48_42_228.png)

1\. Chart shows the operation of a neural network trained for price forecast

We can see that the neural network values repeat the price chart, but they are one step behind. The result does not depend on whether we predict price data or their derivatives. We can make some conclusion here. For example: "What is 'yesterday' for us is 'today' for the neural network." It is not quite useful, isn't it? However, this variant can be used after certain revision.

But our purpose is "What is 'today' for NN (neural network) is 'tomorrow' for us." Kind of a time machine. However, we understand that the best neural network is our brain. The efficiency of it is 50% (if we talk about the yes/no probability), or even worse. There is also the third option – "What is yesterday for the NN is today for us." Or: "What is today for us is yesterday for NN". Let us consider what the above situations mean in trading:

- **_First_** — we execute a deal and receive an answer from NN, whether the direction was right or not. However, we can know it without the NN.
- **_Second_** — we receive information from the NN, execute a deal and see the next day of the recommendation was correct.
- **_Third_** — we receive information from the NN about when to execute this or that deal, or if we need to execute a deal now - if we do, then in which direction.

The first variant is not suitable at all. The second and the third variants are quite suitable for trading. However, the second variant can be regarded as a glimpse into the future. Roughly speaking, a signal from the NN is received at a certain moment of time, for example at a day close - with a forecast of the next day closing level (and we are currently not interested in the price movement before the deal is closed). At this stage, this idea is hard to implement for a purely automated trading (for profitable trading). The idea of the third variant is that we track the response of the NN during the trading session. We interpret this response and either buy or sell an asset. Here, we need to understand the main thing.

The variant to be implemented depends on how we are going to train the neural network. In this case, the third variant is easier to implement. In the second variant, we use any information with the purpose of receiving the next-day result — its closure (the day is only used as an example, so this can be any period). In the third variant, we use the information received one step earlier, before we make a decision, showing where the price will move. I use the third variant in my systems.

### 3\. Neural network: challenging or easy?

We are trying to create a trading system. So, where do we take a neural network, how should we train it and how can we integrate it into the trading terminal? As for me, I am using ready-made neural networks: NeuroSolutions and Matlab. These platforms allow choosing a suitable network, train it and create an executable file with a desired interface. The resulting neural network program can look like this:

![A neural network module created in Matlab environment](https://c.mql5.com/2/36/xfsqr7_2019_06_27_10_32_05_182.png)

2\. A neural network module created in Matlab environment

[https://c.mql5.com/2/36/m1h6i_8j2zhdu_e5c.png](https://c.mql5.com/2/36/m1h6i_8j2zhdu_e5c.png "https://c.mql5.com/2/36/m1h6i_8j2zhdu_e5c.png") or like this:

![A neural network module created in Neuro Solutions environment](https://c.mql5.com/2/36/4pv1z4_2019_07_08_19_37_33_643.png)

3\. A neural network module created using Neuro Solutions

When studying the possibilities of neural network application in financial markets, I came to the conclusion that neural networks can be used not only as the main signal generator, but also as an option for unloading the software part of the trading Expert Advisor. Imagine that you decide to write an Expert Advisor that uses a dozen indicators. These indicators have different parameters; they need to be analyzed and compared at some time period. Moreover, you use several time windows. Thus, you will receive an overloaded Expert Advisor for real trading, which is extremely hard to test.

What we can do is entrust the indicator calculation task to the neural network, after appropriate training. Further, the neural network will be trained using these indicators. It means that only the relative price data used in indicator formulas will need to be input to the neural network module from the Expert Advisor. The neural network will output "ones" and "zeros", which we can compare and make a decision.

Let us view the result using Stochastic Oscillator as an example. We will use the following price data as inputs. The indicator itself will be used as a training example.

![Price Data](https://c.mql5.com/2/36/4vyvvk_2019_06_04_17_34_52_5111.png)

4\. Price Data

```
FileWrite(handle,

                   iClose(NULL,0,i+4)-iLow(NULL,0,i+4),
                   iHigh(NULL,0,i+4)-iClose(NULL,0,i+4),
                   iHigh(NULL,0,i+4)-iLow(NULL,0,i+4),
                   iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,5,i+4))-iLow(NULL,0,iLowest(NULL,0,MODE_LOW,5,i+4)),

                   iClose(NULL,0,i+3)-iLow(NULL,0,i+3),
                   iHigh(NULL,0,i+3)-iClose(NULL,0,i+3),
                   iHigh(NULL,0,i+3)-iLow(NULL,0,i+3),
                   iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,5,i+3))-iLow(NULL,0,iLowest(NULL,0,MODE_LOW,5,i+3)),

                   iClose(NULL,0,i+2)-iLow(NULL,0,i+2),
                   iHigh(NULL,0,i+2)-iClose(NULL,0,i+2),
                   iHigh(NULL,0,i+2)-iLow(NULL,0,i+2),
                   iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,5,i+2))-iLow(NULL,0,iLowest(NULL,0,MODE_LOW,5,i+2)),

                   iClose(NULL,0,i+1)-iLow(NULL,0,i+1),
                   iHigh(NULL,0,i+1)-iClose(NULL,0,i+1),
                   iHigh(NULL,0,i+1)-iLow(NULL,0,i+1),
                   iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,5,i+1))-iLow(NULL,0,iLowest(NULL,0,MODE_LOW,5,i+1)),

                   iClose(NULL,0,i)-iLow(NULL,0,i),
                   iHigh(NULL,0,i)-iClose(NULL,0,i),
                   iHigh(NULL,0,i)-iLow(NULL,0,i),
                   iHigh(NULL,0,iHighest(NULL,0,MODE_HIGH,5,i))-iLow(NULL,0,iLowest(NULL,0,MODE_LOW,5,i)),

                   iStochastic(NULL,0,5,3,3,MODE_SMA,1,MODE_MAIN,i),
                   TimeToStr(iTime(NULL,60,i)));
```

After training, the neural network will output the following result.

![Neural network response](https://c.mql5.com/2/36/pj35ez_2019_06_04_17_04_19_8541.png)

5\. Neural network response

For a better visual study, let us move this data to the trading terminal as an indicator.

![Stochastic and neural network indicator](https://c.mql5.com/2/36/wkncw9_2019_06_04_17_23_42_2961.png)

6\. Stochastic and neural network indicator

The upper window shows the standard indicator available in the terminal. The lower window shows the indicator created by the neural network. We can see visually that the indicator created by the neural network has all the characteristics of a standard indicator, including levels, intersections, reversals, divergences, etc. Remember, we did not use any complex formulas to train the network.

Thus, we can draw the following block diagram of a trading system.

[![Trading system block diagram](https://c.mql5.com/2/36/bdlhu_qu4jx2f_f1o.png)](https://c.mql5.com/2/36/mkrv9_4h6taag_e1h.png "https://c.mql5.com/2/36/mkrv9_4h6taag_e1h.png")

7\. Trading system block diagram

MT4 blocks represent our Expert Advisor. "Input\_mat" is the price file. "Open1,2,3" is the signal file. Examples of these files are provided in the next section.

The main work will concern the blocks "Net1" and "Net2". For these blocks, we will need to use several scripts and EAs to prepare historical data and to test signals from these blocks. Actually, when the system is ready as a complex, its modification, development and experimenting with it does not take much time. The following video shows an example: [https://youtu.be/k\_OLEKJCxPE](https://www.mql5.com/go?link=https://youtu.be/k_OLEKJCxPE "https://youtu.be/k_OLEKJCxPE"). In general, preparation of files, training of Net1 and Net2, and the first testing stage in which we optimize the system, take 10 minutes.

### 4\. Integrating a neural network into the trading terminal

Integration of a neural network and the trading terminal is not difficult. I solved this question by passing data via files created by the terminal and the neural network program. One may say that this may slow down decision making by the system. However, this method has its advantages. Firstly, the terminal passes a minimum of data, only a few tens of bytes. See below the file line written by the terminal.

![File of normalized prices](https://c.mql5.com/2/36/7waiwv_2019_05_31_15_32_02_547.png)

8\. File of normalized prices

Although this data transmission method enables deal opening only at the next tick after, following the arrival of a signal from the neural network. However, if the system does not trade ultra short-term moments, this is not crucial. In this article, the system works using open prices. Also, systems utilizing this data transmission method, requires testing by checkpoints or using the every tick mode. Tests of neural network-based systems in these two modes are almost identical. When developing traditional trading robots, I came across the situations when testing in the every tick mode showed much worse results.

The main advantage of this data transmission mode is that we can control the data we receive and sent at every stage. I consider this to be one of the foundations for further successful trading using a neural network. Thus, our bulky preparation of the neural network system turns into an advantage in real work. This way we can reduce to a minimum the probability of receiving a program error in the system's logical structure. This is because the system requires a step-by-step triple testing before usage. We will get back to this part later.

The image below shows files “Input\_mat” and “Bar”. These files are generated by the trading terminal. Files Open1,2,3 are generated by the NN program. The only inconvenience is that in the NN program we need to explicitly set the paths to these files based on how we use the EA - for testing or for trading.

![Files generated by the neural network module and the Expert Advisor](https://c.mql5.com/2/36/ig4jph_2019_05_31_15_32_43_499_1.png)

9\. Files generated by the neural network module and the Expert Advisor

“Bar” is an auxiliary file which is used as a counter.

![Bar file](https://c.mql5.com/2/36/o0vjbc_2019_05_31_15_42_45_579_1.png)

The NN response is received into files Open1,2,3. The first line shows the previous response. The second line shows real-time response. This format is a special case. The format may differ depending on trading conditions. The number of response files may also be different. We have three of them, because the NN module uses three networks trained in different time intervals.

![Neural network module response in files Open1,2,3](https://c.mql5.com/2/36/zeo4cv_2019_05_31_15_42_52_695_1.png)

10\. Neural network module response in files Open1,2,3

### 5\. How to test a neural network? Testing stages

I use three testing stages when preparing trading systems based on neural networks. The first stage is rapid testing. This is the main system preparation stage in terms of its general performance. At this stage, we can optimize the system, while the optimization does not take much time. Here we use a script or an Expert Advisor for preparing a file with historical data, with the period following the historical period in which the NN was trained and up to the current time. Then, we receive NN responses at this interval, using a Matlab script, and create an indicator based on the responses. Then we use this indicator to optimize our NN responses for market entries and exits. The below figure shows an example of this indicator. This indicator is an interpretation of 52 derivatives from 12 custom indicators. These may include standard terminal indicators.

![An indicator based on neural network responses](https://c.mql5.com/2/36/EURUSDH1.png)

11\. An indicator based on neural network responses

Next, we can optimize our trading strategy.

![Results of testing the indicator based on neural network responses](https://c.mql5.com/2/36/zwvzib_2019_07_09_17_59_50_192.png)

12\. Results of testing neural network responses

In the second testing stage, we train and write neural networks in the Matlab environment using the Neural Network Toolbox.

![Neural Fitting](https://c.mql5.com/2/36/o7876y_2019_06_10_17_17_52_632.png)

13\. Neural Fitting

![Resulting neural networks](https://c.mql5.com/2/36/uih9h3_2019_06_10_17_20_31_895.png)

14\. Resulting neural networks

Get a response from these neural networks via the command window.

![Receiving a response from the neural network](https://c.mql5.com/2/36/9ed9m_4pczkz8_n2c.png)

15\. Receiving responses from neural networks

Thus, we will receive another indicator which should be identical to the previous one. Accordingly, testing of a strategy based on this indicator should also be identical.

If everything is good, we can move further.

We can test these neural networks using a script of the neural network module which will be used in the system. The signals should be tested at any time interval, using control points. If this test coincides with the identical time interval of the previous indicator test, then we are moving in the right direction. Launch this script in the Matlab environment. At the same time, launch the Expert Advisor in the trading terminal.

![Launching a script in the Matlab environment](https://c.mql5.com/2/36/ciwkba_2019_06_10_17_40_47_712.png)

16\. Launching a script in the Matlab environment

![Launching the Expert Advisor in the terminal](https://c.mql5.com/2/36/2si4bc_2019_07_09_18_25_54_567.png)

17\. Launching the Expert Advisor in the terminal

Here is the result:

![The result of testing the Matlab script and the MT4 Expert Advisor ](https://c.mql5.com/2/36/j8gvsa_2019_07_09_21_09_17_169.png)

18\. The result of testing the Matlab script and the MT4 Expert Advisor

Next, we need to create the user interface, compile the neural network module and test as is described above.

![Testing the compiled neural network module](https://c.mql5.com/2/36/7gq3hw_2019_06_27_14_34_52_661.png)

19\. Testing the compiled neural network module

If the result is similar to the previous one, we can proceed to real trading using according to our neural network system.

### 5\. About training samples

Depending on what training samples will be used to prepare the neural network, we will get different indicators based on NN responses. Therefore, different trading strategies can be created. Furthermore, a combination of different strategies will give us a more stable final result. One of the variants was shown in previous sections. In that example, we made a selection based on trading period extreme points. Let me give you another example.

![Indicator of responses of a neural network trained on a different sample](https://c.mql5.com/2/36/tjfi9g_2019_06_17_14_27_47_282.png)

20\. Indicator of responses of a neural network trained on a different sample

In this case, I trained two neural networks. One - for buying, the other one - for selling. Training is performed on samples when the minimum price has been reached, while the maximum price has not yet been reached. And vice versa. These two indicators shown in the figure reflect the interpretation of twelve custom indicators. Highs of the red line show when the minimum price is reached. Gray highs are the maximum price. Now, it is possible to optimize these indicators, either separately or together. For example, we can test their intersections or difference in their values, as well as level intersections and so on.

It would be much more difficult to optimize twelve indicators.

### Conclusion

There are many articles on the use of neural networks in trading. However, there is very little material regarding how to apply systems based on neural networks in practice. Furthermore, publications are intended for users with specific programming knowledge. It is quite difficult to provide a complete description in one article. I tried to explain the usage specifics without adding excessive theoretical material in my book "Neural network trading system. MetaTrader 4 + MATLAB. Step-by-step development. Second edition" (in Russian).

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/7031](https://www.mql5.com/ru/articles/7031)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Practical application of neural networks in trading (Part 2). Computer vision](https://www.mql5.com/en/articles/8668)
- [Practical application of neural networks in trading. Python (Part I)](https://www.mql5.com/en/articles/8502)
- [Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/352286)**
(40)


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
29 Jul 2021 at 07:50

**MetaQuotes:**

New article [Practical applications of neural networks in trading](https://www.mql5.com/en/articles/7031) has been published in:.

Author: [Andrey Dibrov](https://www.mql5.com/en/users/tomcat66 "Tomcat66.")

![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
29 Jul 2021 at 07:52

What you are saying is like talking to a normal person! Please make it simple, clear and short! I bought from google, so I would like to hear a reply from google!


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
29 Jul 2021 at 07:53

**ゆうじ 保坂:**

What you are saying is like talking to a normal person! Please make it simple, clear and short! I bought from google, so I would like to hear a reply from google!

![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
29 Jul 2021 at 09:51

What you are saying is like talking to a normal person! Please make it simple, clear and short! I bought from google, so I would like to hear a reply from google!


![ゆうじ 保坂](https://c.mql5.com/avatar/2021/7/60F06EBE-82A4.jpg)

**[ゆうじ 保坂](https://www.mql5.com/en/users/popaipapa111-gmail)**
\|
29 Jul 2021 at 09:52

**ゆうじ 保坂:**

What you are saying is like talking to a normal person! Please make it simple, clear and short! I bought from google, so I would like to hear a reply from google!

![Practical application of neural networks in trading. It's time to practice](https://c.mql5.com/2/39/neural_DLL.png)[Practical application of neural networks in trading. It's time to practice](https://www.mql5.com/en/articles/7370)

The article provides a description and instructions for the practical use of neural network modules on the Matlab platform. It also covers the main aspects of creation of a trading system using the neural network module. In order to be able to introduce the complex within one article, I had to modify it so as to combine several neural network module functions in one program.

![Manual charting and trading toolkit (Part I). Preparation: structure description and helper class](https://c.mql5.com/2/39/MQL5-set_of_tools.png)[Manual charting and trading toolkit (Part I). Preparation: structure description and helper class](https://www.mql5.com/en/articles/7468)

This is the first article in a series, in which I am going to describe a toolkit which enables manual application of chart graphics by utilizing keyboard shortcuts. It is very convenient: you press one key and a trendline appears, you press another key — this will create a Fibonacci fan with the necessary parameters. It will also be possible to switch timeframes, to rearrange layers or to delete all objects from the chart.

![Quick Manual Trading Toolkit: Basic Functionality](https://c.mql5.com/2/39/Frame_1.png)[Quick Manual Trading Toolkit: Basic Functionality](https://www.mql5.com/en/articles/7892)

Today, many traders switch to automated trading systems which can require additional setup or can be fully automated and ready to use. However, there is a considerable part of traders who prefer trading manually, in the old fashioned way. In this article, we will create toolkit for quick manual trading, using hotkeys, and for performing typical trading actions in one click.

![Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://c.mql5.com/2/39/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 44): Collection class of indicator buffer objects](https://www.mql5.com/en/articles/7886)

The article deals with creating a collection class of indicator buffer objects. I am going to test the ability to create and work with any number of buffers for indicators (the maximum number of buffers that can be created in MQL indicators is 512).

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iozslnwybwtkyeoqoeoscqvrpfekdikt&ssn=1769186153170053453&ssn_dr=0&ssn_sr=0&fv_date=1769186153&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F7031&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Practical%20application%20of%20neural%20networks%20in%20trading%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918615364594669&fz_uniq=5070414098224584041&sv=2552)

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