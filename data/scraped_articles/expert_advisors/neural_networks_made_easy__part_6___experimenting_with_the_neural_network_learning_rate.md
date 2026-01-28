---
title: Neural networks made easy (Part 6): Experimenting with the neural network learning rate
url: https://www.mql5.com/en/articles/8485
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:28:24.033519
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=eaoflmghvbzsbtbspjjdfycqfoxoikss&ssn=1769192903833240840&ssn_dr=0&ssn_sr=0&fv_date=1769192903&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8485&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20networks%20made%20easy%20(Part%206)%3A%20Experimenting%20with%20the%20neural%20network%20learning%20rate%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919290313983731&fz_uniq=5071857855776108486&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/8485#para1)
- [1\. The problem](https://www.mql5.com/en/articles/8485#para2)
- [2\. Experiment 1](https://www.mql5.com/en/articles/8485#para3)
- [3\. Experiment 2](https://www.mql5.com/en/articles/8485#para4)
- [4\. Experiment 3](https://www.mql5.com/en/articles/8485#para5)
- [Conclusions](https://www.mql5.com/en/articles/8485#para6)
- [References](https://www.mql5.com/en/articles/8485#para7)
- [Programs used in the article](https://www.mql5.com/en/articles/8485#para8)

### Introduction

In earlier articles, we considered the principles of operation and methods of implementing a fully connected perceptron, convolutional and recurrent networks. We used gradient descent to train all networks. According to this method, we determine the network prediction error at each step and adjust the weights in an effort to decrease the error. However, we do not completely eliminate the error at each step, but only adjust weights to reduce the error. Thus, we are trying to find such weights that will closely repeat the training set along its entire length. The learning rate is responsible for the error minimizing speed at each step.

### 1\. The problem

What is the problem with learning rate selection? Let us outline the basic questions related to the selection of the learning rate.

1\. Why cannot we use the rate equal to "1" (or a close value) and immediately compensate the error?

In this case, we would have a neural network overtrained for the last situation. As a result, further decision will be made based only on the latest data while ignoring the history.

2\. What is the problem with a knowingly small rate, which would allow the averaging of values over the entire sample?

The first problem with this approach is the training period of the neural network. If the steps are too small, a large number of such steps will be needed. This requires time and resources.

The second problem with this approach is that the path to the goal is not always smooth. It may have valleys and hills. If we move in too small steps, we can get stuck in one of such values, mistakenly determining it as a global minimum. In this case we will never reach the goal. This can be partially solved by using a momentum in the weight update formula, but still the problem exists.

![](https://c.mql5.com/2/40/1189765377431.png)

3\. What is the problem with a knowingly large rate, which would allow the averaging of values over a certain distance and avoid local minima?

An attempt to solve the local minimum problem by increasing the learning rate leads to another problem: the use of a large learning rate often does not allow minimizing the error, because with the next update of the weights their change will be greater than the required one, and as a result we will jump over the global minimum. If we return to this further again, the situation will be similar. As a result, we will oscillate in around the global minimum.

![](https://c.mql5.com/2/40/3929263491779.png)

These are well known problems, and they are often discussed, but I haven't found any clear recommendations regarding the learning rate selection. Everyone suggests the empirical selection of the rate for each specific task. Some other authors suggest the gradual reduction of rate during the learning process, in order to minimize risk 3 described above.

In this article, I propose to conduct some experiments training one neural network with different learning rates and to see the effect of this parameter on the neural network training as a whole.

### 2\. Experiment 1

For convenience, let us make the _eta_ variable from the _CNeuronBaseOCL_ class a global variable.

```
double   eta=0.01;
#include "NeuroNet.mqh"
```

and

```
class CNeuronBaseOCL    :  public CObject
  {
protected:
   ........
   ........
//---
   //const double      eta;
```

Now, create three copies of the Expert Advisor with different learning rate parameters (0,1; 0,01; 0,001). Also, create the fourth EA, in which the initial learning rate is set to 0.01 and it will be reduced by 10 times every 10 epochs. To do this, add the following code to the training loop in the Train function.

```
         if(discount>0)
            discount--;
         else
           {
            eta*=0.1;
            discount=10;
           }
```

All the four EAs were simultaneously launched in one terminal. In this experiment, I used parameters from earlier EA tests: symbol EURUSD, timeframe H1, data of 20 consecutive candlesticks are fed into the network, and training is performed using the history for the last two years. The training sample was about 12.4 thousand bars.

All EAs were initialized with random weights ranging from -1 to 1, excluding zero values.

Unfortunately, the EA with the learning rate equal to 0.1 showed an error close to 1, and therefore it is not shown in charts. The learning dynamics of other EAs is shown in the charts below.

After 5 epochs, the error of all EAs reached the level of 0.42, where it continued to fluctuate for the rest of the time. The error of the EA with the learning rate equal to 0.001 was slightly lower. The differences appeared in the third decimal place (0.420 against 0.422 of the other two EAs).

The error trajectory of the EA with a variable learning rate follows the error line of the EA with a learning factor of 0.01. This is quite expected in the first ten epochs, but there is no deviation when the rate decreases.

![](https://c.mql5.com/2/40/2290175589834.png)

Let us take a closer look at the difference between the errors of the above EAs. Almost throughout the entire experiment, the difference between the errors of EAs with constant learning rates of 0.01 and 0.001 fluctuated around 0.0018. Furthermore, a decrease in the EA's learning rate every 10 epochs has almost no effect and the deviation from the EA with a rate of 0.01 (equal to the initial learning rate) fluctuates around 0.

![](https://c.mql5.com/2/40/6222168534212.png)

The obtained error values show that the learning rate of 0.1 is not applicable in our case. The use of a learning rate of 0.01 and below produces similar results with an error of about 42%.

The statistical error of the neural network is quite clear. How will this affect the EA performance? Let us check the number of missed fractals. Unfortunately, all EAs showed bad results during the experiment: they all missed nearly 100% of fractals. Furthermore, an EA with the learning rate of 0.01 determines about 2.5% fractals, while with the rate of 0.001 the EA skipped 100% of fractals. After the 52nd epoch, the EA with the learning rate of 0.01 showed a tendency towards a decrease in the number of skipped fractals. No such tendency was shown by the EA with the variable rate.

![](https://c.mql5.com/2/40/3953110846112.png)![](https://c.mql5.com/2/40/1322837470065.png)

The chart of missing fractal percentage deltas also shows a gradual increase in the difference in favor of the EA with a learning rate of 0.01.

We have considered two neural network performance metrics, and so far the EA with a lower learning rate has a smaller error, but it misses fractals. Now, let us check the third value: "hit" of the predicted fractals.

The charts below show a growth of the "hit" percentage in the training of EAs with a learning rate of 0.01 and with a dynamically decreasing rate. The rate of variable growth decreases with a decrease in the learning rate. The EA with the learning rate of 0.001 had the "hit" percentage stuck around 0, which is quite natural because it misses 100% of fractals.

![](https://c.mql5.com/2/40/2955588828793.png)![](https://c.mql5.com/2/40/3328421160139.png)

The above experiment shows that the optimal learning rate or training a neural network within our problem is close to 0.01. A gradual decrease in the learning rate did not give a positive result. Perhaps the effect of rate decrease will be different if we decrease it less often than in 10 epochs. Perhaps, results would be better with 100 or 1000 epochs. However, this needs to be verified experimentally.

### 3\. Experiment 2

In the first experiment, the neural network weight matrices were randomly initialized. And therefore, all the EAs had different initial states. To eliminate the influence of randomness on the experiment results, load the weight matrix obtained from the previous experiment with the EA having a learning rate equal to 0.01 into all three EAs and continue training for another 30 epochs.

The new training confirms the earlier obtained results. We see an average error around 0.42 across all three EAs. The EA with the lowest learning rate (0.001) again had a slightly smaller error (with the same difference of 0.0018). The effect of a gradual decrease in the learning rate is practically equal to 0.

![](https://c.mql5.com/2/40/1897352257097.png)![](https://c.mql5.com/2/40/4129192498637.png)

As for the percentage of missed fractals, the earlier obtained results are confirmed again. The EA with a lower learning factor approached 100% of missed fractals in 10 epochs, i.e. the EA is unable to indicate fractals. The other two EAs show a value of 97.6%. The effect of a gradual decrease in the learning rate is practically equal to 0.

![](https://c.mql5.com/2/40/4794037999782.png)![](https://c.mql5.com/2/40/3612229884440.png)

The "hit" percentage of the EA with the learning rate of 0.001 continues to grow gradually. A gradual decrease in the learning rate does not affect this value.

![](https://c.mql5.com/2/40/770831154124.png)![](https://c.mql5.com/2/40/5239824590907.png)

### 4\. Experiment 3

The third experiment is a slight deviation from the main topic of the article. Its idea came about during the first two experiments. So, I decided to share it with you. While observing the neural network training, I noticed that the probability of the absence of a fractal fluctuates around 60-70% and rarely falls below 50%. The probability of emergence of a fractal, wither buy or sell, is around 20-30%. This is quite natural, as there are much less fractals on the chart than there are candlesticks inside trends. Thus, our neural network is overtrained, and we obtain the above results. Almost 100% of fractals are missed, and only rare ones can be caught.

![Training the EA with the learning rate of 0.01](https://c.mql5.com/2/40/EURUSD_i_PERIOD_H1__20Fractal_OCL239.png)

To solve this problem, I decided to slightly compensate for the unevenness of the sample: for the absence of a fractal in the reference value, I specified 0.5 instead of 1 when training the network.

```
            TempData.Add((double)buy);
            TempData.Add((double)sell);
            TempData.Add((double)((!buy && !sell) ? 0.5 : 0));
```

This step produced a good effect. The Expert Advisor running with a learning rate of 0.01 and a weight matrix obtained from previous experiments shows the error stabilization of about 0.34 after 5 training epochs. The share of missed fractals decreased to 51% and the percentage of hits increased to 9.88%. You can see from the chart that the EA generates signals in group and thus shows some certain zones. Obviously, the idea requires additional development and testing. But the results suggest that this approach is quite promising.

![Learning with 0.5 for no fractal](https://c.mql5.com/2/40/EURUSD_i_PERIOD_H1__20Fractal_OCL217.png)

![](https://c.mql5.com/2/40/1510468199161.png)

![](https://c.mql5.com/2/40/338396624537.png)

![](https://c.mql5.com/2/40/6041214782135.png)

### Conclusions

We have implemented three experiments in this article. The first two experiments have shown the importance of the correct selection of the neural network learning rate. The learning rate affects the overall neural network training result. However, there is currently no clear rule for choosing the learning rate. That is why you will have to select it experimentally in practice.

The third experiment has shown that a non-standard approach to solving a problem can improve the result. But the application of each solution must be confirmed experimentally.

### References

1. [Neural networks made easy](https://www.mql5.com/en/articles/7447 "Neural networks made easy")
2. [Neural networks made easy (Part 2): Network training and testing](https://www.mql5.com/en/articles/8119 "Neural networks made easy (Part 2): Network training and testing")
3. [Neural networks made easy (Part 3): Convolutional networks](https://www.mql5.com/en/articles/8234)
4. [Neural networks made easy (Part 4): Recurrent networks](https://www.mql5.com/en/articles/8385)
5. [Neural networks made easy (Part 5): Multithreaded calculations in OpenCL](https://www.mql5.com/en/articles/8435)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Fractal\_OCL1.mq5 | Expert Advisor | An Expert Advisor with the classification neural network (3 neurons in the output layer) using the OpenCL technology Learning rate = 0.1 |
| 2 | Fractal\_OCL2.mq5 | Expert Advisor | An Expert Advisor with the classification neural network (3 neurons in the output layer) using the OpenCL technology Learning rate = 0.01 |
| 3 | Fractal\_OCL3.mq5 | Expert Advisor | An Expert Advisor with the classification neural network (3 neurons in the output layer) using the OpenCL technology Learning rate = 0.001 |
| 4 | Fractal\_OCL\_step.mq5 | Expert Advisor | An Expert Advisor with the classification neural network (3 neurons in the output layer) using the OpenCL technology Learning rate with a 10x decrease from 0.01 every 10 epochs |
| 5 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 6 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8485](https://www.mql5.com/ru/articles/8485)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8485.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/8485/mql5.zip "Download MQL5.zip")(1582.99 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/359807)**
(13)


![Gexon](https://c.mql5.com/avatar/2023/8/64cd1951-a9dc.png)

**[Gexon](https://www.mql5.com/en/users/gexon)**
\|
9 Nov 2023 at 12:15

"...in the absence of a fractal in the reference value, when [training the network](https://www.mql5.com/en/articles/8119 "Article: Neural networks are easy (Part 2): Training and testing a network "), I specified 0.5 instead of 1."

Why exactly 0.5, where did this figure come from?

![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
9 Nov 2023 at 15:11

**Gexon training the network specified 0.5 instead of 1."**
**Why exactly 0.5, where did this figure come from?**

During training, the model learns the [probability distribution](https://www.mql5.com/en/articles/271 "Article: Statistical Probability Distributions in MQL5 ") of each of the 3 events. Since the probability of fractal absence is much higher than the probability of its appearance, we artificially underestimate it. We specify 0.5, because at this value we come to approximately equal level of maximum probabilities of events. And they can be compared.

I agree that this approach is very controversial and is dictated by observations from the training sample.

![Gexon](https://c.mql5.com/avatar/2023/8/64cd1951-a9dc.png)

**[Gexon](https://www.mql5.com/en/users/gexon)**
\|
17 Nov 2023 at 11:50

> _double rsi=RSI.Main(bar\_t);_
>
> _double cci=CCI.Main(bar\_t);_
>
> _double atr=ATR.Main(bar\_t);_
>
> _if(open==EMPTY\_VALUE \|\|_ _!TempData.Add(rsi) \|\| !TempData.Add(cci) \|\| !TempData.Add(atr) )_
>
> _break;_

It looks like the data is not normalised, is this supposed to be the case or does this also work?

![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
17 Nov 2023 at 12:02

**Gexon [#](https://www.mql5.com/ru/forum/353300#comment_50586034):**

> _double rsi=RSI.Main(bar\_t);_
>
> _double cci=CCI.Main(bar\_t);_
>
> _double atr=ATR.Main(bar\_t);_
>
> _if(open==EMPTY\_VALUE \|\|_ _!TempData.Add(rsi) \|\| !TempData.Add(cci) \|\| !TempData.Add(atr) )_
>
> _break;_

It looks like the data is not normalised, is this supposed to be the case or does this also work?

We will talk about normalising the data a little [later](https://www.mql5.com/en/articles/9207).

![Gexon](https://c.mql5.com/avatar/2023/8/64cd1951-a9dc.png)

**[Gexon](https://www.mql5.com/en/users/gexon)**
\|
18 Nov 2023 at 10:11

**Dmitriy Gizlyk [#](https://www.mql5.com/ru/forum/353300#comment_50586088):**

We'll talk about normalising the data a little [later](https://www.mql5.com/en/articles/9207).

got it, thanks.)


![Gradient boosting in transductive and active machine learning](https://c.mql5.com/2/41/yandex_catboost__2.png)[Gradient boosting in transductive and active machine learning](https://www.mql5.com/en/articles/8743)

In this article, we will consider active machine learning methods utilizing real data, as well discuss their pros and cons. Perhaps you will find these methods useful and will include them in your arsenal of machine learning models. Transduction was introduced by Vladimir Vapnik, who is the co-inventor of the Support-Vector Machine (SVM).

![Optimal approach to the development and analysis of trading systems](https://c.mql5.com/2/40/optimal-approach.png)[Optimal approach to the development and analysis of trading systems](https://www.mql5.com/en/articles/8410)

In this article, I will show the criteria to be used when selecting a system or a signal for investing your funds, as well as describe the optimal approach to the development of trading systems and highlight the importance of this matter in Forex trading.

![Analyzing charts using DeMark Sequential and Murray-Gann levels](https://c.mql5.com/2/41/steps.png)[Analyzing charts using DeMark Sequential and Murray-Gann levels](https://www.mql5.com/en/articles/8589)

Thomas DeMark Sequential is good at showing balance changes in the price movement. This is especially evident if we combine its signals with a level indicator, for example, Murray levels. The article is intended mostly for beginners and those who still cannot find their "Grail". I will also display some features of building levels that I have not seen on other forums. So, the article will probably be useful for advanced traders as well... Suggestions and reasonable criticism are welcome...

![Timeseries in DoEasy library (part 57): Indicator buffer data object](https://c.mql5.com/2/41/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 57): Indicator buffer data object](https://www.mql5.com/en/articles/8705)

In the article, develop an object which will contain all data of one buffer for one indicator. Such objects will be necessary for storing serial data of indicator buffers. With their help, it will be possible to sort and compare buffer data of any indicators, as well as other similar data with each other.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=tllhvsfcgunuvofqmehippjbxniagifu&ssn=1769192903833240840&ssn_dr=0&ssn_sr=0&fv_date=1769192903&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F8485&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20networks%20made%20easy%20(Part%206)%3A%20Experimenting%20with%20the%20neural%20network%20learning%20rate%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919290313935592&fz_uniq=5071857855776108486&sv=2552)

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