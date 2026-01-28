---
title: Experiments with neural networks (Part 1): Revisiting geometry
url: https://www.mql5.com/en/articles/11077
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:29:54.144501
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/11077&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070333889710330880)

MetaTrader 5 / Tester


### Introduction

In this article, I would like to share my experiments with neural networks.  After reading a large amount of information available on MQL5, I came to the conclusion that the theory is sufficient. There are plenty of good articles, libraries and source codes. Unfortunately, all this data does not lead to a logical conclusion – a profitable trading system. Let's try to fix this.

I am not an expert in this field, much less a writer or a journalist, but I will try to express my thoughts in an accessible way to share my experience.

The material is primarily designed for beginners, like myself.

### My understanding. The basics

It is generally believed that neural networks are good at recognizing patterns, while the data passed to the neural network for training is of utmost importance. I will start with this assumption. Let's use geometry. I will transfer geometric shapes to a neural network. To begin with, let's take a regular perceptron, whose specimen I found here ( [МTC Сombo - expert for MetaTrader 4](https://www.mql5.com/en/code/7917)). While performing tests, I decided to abandon the oscillators and use MA. Tests involving oscillators did not yield good results. I believe, everyone knows about the so-called divergence when the price moves up and the oscillator goes down. MA parameters are closer to the price itself.

### Shapes and lines

The basis will consist of two [Moving Average](https://www.mql5.com/en/docs/indicators/ima) indicators with the parameters 1 and 24, the Simple method is applied to Close. In other words, the idea is to pass not only the current indicator locations but also the states, in which they were before the current one. In many examples that I have seen, the price is passed directly to the neural network, which I consider to be fundamentally wrong.

I pass all values in points, which is very important since these values have a certain range beyond which they cannot go. Passing a price to a neural network has no sense since it may oscillate in different ranges for, say, 10 years. Also, keep in mind that we can use different number of indicator parameters when building shapes. Shapes can be complex or simple. A few possible options are provided below. Of course, you can come up with your own ones.

**Shape 1: Simple lines**

The distances in points on the closed candle 1, 4, 7 and 10 between MA 1 and MA 24.

![perceptron 1](https://c.mql5.com/2/47/p1__2.png)

```
double perceptron1()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;

   double a1 = (ind_In1[1]-ind_In2[1])/Point();
   double a2 = (ind_In1[4]-ind_In2[4])/Point();
   double a3 = (ind_In1[7]-ind_In2[7])/Point();
   double a4 = (ind_In1[10]-ind_In2[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

**Shape 2: Simple lines**

Distances in points between the closed candle 1-4, 4-7 and 7-10 of MA 1.

![perceptron 2](https://c.mql5.com/2/47/p2__2.png)

```
double perceptron2()
  {
   double w1 = y1 - 100.0;
   double w2 = y2 - 100.0;
   double w3 = y3 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[4])/Point();
   double a2 = (ind_In1[4]-ind_In1[7])/Point();
   double a3 = (ind_In1[7]-ind_In1[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3);
  }
```

**Shape 3: Simple lines**

Distances in points between the closed candle 1-4, 4-7 and 7-10 of MA 24.

![perceptron 3](https://c.mql5.com/2/47/p3__2.png)

```
double perceptron3()
  {
   double w1 = z1 - 100.0;
   double w2 = z2 - 100.0;
   double w3 = z3 - 100.0;

   double a1 = (ind_In2[1]-ind_In2[4])/Point();
   double a2 = (ind_In2[4]-ind_In2[7])/Point();
   double a3 = (ind_In2[7]-ind_In2[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3);
  }
```

**Shape 4: Butterfly (envelope)**

Distances in points between the closed candle 1-10 of MA 1. And the distance in points between the closed candle 1-10 of MA 24.  The distance in points between the candle 1 of MA 1 and the candle 10 of MA 24. The distance in points between the candle 1 of MA 24 and the candle 10 of MA 1. The result is a butterfly.

![perceptron 4](https://c.mql5.com/2/47/p4__2.png)

```
double perceptron4()
  {
   double w1 = f1 - 100.0;
   double w2 = f2 - 100.0;
   double w3 = f3 - 100.0;
   double w4 = f4 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[10])/Point();
   double a2 = (ind_In2[1]-ind_In2[10])/Point();
   double a3 = (ind_In1[1]-ind_In2[10])/Point();
   double a4 = (ind_In2[1]-ind_In1[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

**Shape 5: Quadrilateral**

The distances in points between the closed candle 1-1, 10-10 between the indicators. And the distance in points between 1-10 of MA 1 and the distance in points between 1-10 of the indicator MA 24. The result is a quadrilateral.

![perceptron 5](https://c.mql5.com/2/47/p5__2.png)

```
double perceptron5()
  {
   double w1 = c1 - 100.0;
   double w2 = c2 - 100.0;
   double w3 = c3 - 100.0;
   double w4 = c4 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[10])/Point();
   double a2 = (ind_In2[1]-ind_In2[10])/Point();
   double a3 = (ind_In1[1]-ind_In2[1])/Point();
   double a4 = (ind_In1[10]-ind_In2[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

**Shape 6: Complex**

Here I will combine all the above shapes into a complex one.

![perceptron 6](https://c.mql5.com/2/47/p6__2.png)

```
double perceptron6()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;

   double w5 = y1 - 100.0;
   double w6 = y2 - 100.0;
   double w7 = y3 - 100.0;

   double w8 = z1 - 100.0;
   double w9 = z2 - 100.0;
   double w10 = z3 - 100.0;

   double w11 = f1 - 100.0;
   double w12 = f2 - 100.0;
   double w13 = f3 - 100.0;
   double w14 = f4 - 100.0;

   double a1 = (ind_In1[1]-ind_In2[1])/Point();
   double a2 = (ind_In1[4]-ind_In2[4])/Point();
   double a3 = (ind_In1[7]-ind_In2[7])/Point();
   double a4 = (ind_In1[10]-ind_In2[10])/Point();

   double a5 = (ind_In1[1]-ind_In1[4])/Point();
   double a6 = (ind_In1[4]-ind_In1[7])/Point();
   double a7 = (ind_In1[7]-ind_In1[10])/Point();

   double a8 = (ind_In2[1]-ind_In2[4])/Point();
   double a9 = (ind_In2[4]-ind_In2[7])/Point();
   double a10 = (ind_In2[7]-ind_In2[10])/Point();

   double a11 = (ind_In1[1]-ind_In1[10])/Point();
   double a12 = (ind_In2[1]-ind_In2[10])/Point();
   double a13 = (ind_In1[1]-ind_In2[10])/Point();
   double a14 = (ind_In2[1]-ind_In1[10])/Point();

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   w5 * a5 + w6 * a6 + w7 * a7   +   w8 * a8 + w9 * a9 + w10 * a10   +   w11 * a11 + w12 * a12 + w13 * a13 + w14 * a14);
  }
```

### Angles

Let's consider another interesting method of passing price data to a perceptron - indicator slope angles. This data also cannot go beyond a certain range, which is quite suitable for us since we want to pass a certain template, just like in the previous case with shapes and lines.

I have come across a lot of methods for defining angles, but many of them depend on the scale of the price chart, which is not suitable for me. So rather than calculating the angle, I am going to calculate the angle tangent using the ratio of the number of points to the number of bars. tg(α) angle tangent is a ratio of the opposite leg a to the adjacent leg b.

![ta](https://c.mql5.com/2/47/ta.png)

It is also possible to use a different number of indicators, handle complex structures and use a different number of candles for analysis. The slope angles are displayed as unfixed lines on the screenshots. Let's consider several examples.

**Perceptront1. 3 slope angles of MA 1**

MA 1 slope angle between candles 1-4, between candles 1-7, between candles 1-10.

![perceptron t1](https://c.mql5.com/2/47/a1.png)

```
double perceptront1()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[4])/4;
   double a2 = (ind_In1[1]-ind_In1[7])/7;
   double a3 = (ind_In1[1]-ind_In1[10])/10;

   return (w1 * a1 + w2 * a2 + w3 * a3);
  }
```

**Perceptront2. 4 slope angles of MA 1 and MA 24**

MA 1 slope between the candles 1-5, between the candles 1-10.  MA 24 slope between the candles 1-5, between the candles 1-10.

![perceptron t2](https://c.mql5.com/2/47/a2.png)

```
double perceptront2()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[5])/5;
   double a2 = (ind_In1[1]-ind_In1[10])/10;
   double a3 = (ind_In2[1]-ind_In2[5])/5;
   double a4 = (ind_In2[1]-ind_In2[10])/10;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

**Perceptront3. 4 slope angles of MA 1 and MA 24 (more or less complex design used as an example)**

The angles are tied with a slope between MA 1 and MA 24.

![perceptron t3](https://c.mql5.com/2/47/a3__2.png)

```
double perceptront3()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[10])/10;
   double a2 = (ind_In2[1]-ind_In1[4])/4;
   double a3 = (ind_In2[1]-ind_In1[7])/7;
   double a4 = (ind_In2[1]-ind_In1[10])/10;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

**Perceptront4. 4 slope angles of MA 1 and MA 24 (more or less complex design used as an example)**

The angles are tied with a slope between MA 1 and MA 24.

![perceptron t4](https://c.mql5.com/2/47/a4.png)

```
double perceptront4()
  {
   double w1 = x1 - 100.0;
   double w2 = x2 - 100.0;
   double w3 = x3 - 100.0;
   double w4 = x4 - 100.0;

   double a1 = (ind_In1[1]-ind_In1[10])/10;
   double a2 = (ind_In2[1]-ind_In1[10])/10;
   double a3 = (ind_In1[1]-ind_In1[10])/10;
   double a4 = (ind_In2[1]-ind_In2[10])/10;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

### Strategy

I decided to use a counter-trend strategy for training by explicitly specifying in the code below. For selling, MA 1 on the first candle is above MA 24. The case is reversed for buying. This is necessary for a clear separation of buying and selling. You, on the other hand, can do the opposite - use a trend.

You can also use other indicators or their values, such as the TEMA indicator. It is impossible to forecast the price movement on 400 points on a five-digit symbol. No one knows where the market will go. Therefore, in order to perform a test, I have set a fixed stop loss of 600 points and a take profit of 60 points for a five-digit symbol. You can download the ready-made EAs below. Let's have a look at the results.

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++

if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (perceptron1()<0) &&(SpreadS1<=MaxSpread)){//v1
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++

if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (perceptron1()>0) && (SpreadS1<=MaxSpread)){//v1
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment);
}
```

### Optimization, testing and resources

As we know, optimization of neural networks requires considerable computing resources. So, while using the strategy tester for optimization, I recommend the "Open prices only" mode with the explicit indication of close prices in the code itself. Otherwise, this is not a feasible task with my rather modest capacities. But even with this optimization mode, I recommend using the [Cloud Network](https://cloud.mql5.com/ "https://cloud.mql5.com/") service. The goal of optimization is finding certain profitable patterns. The occurrences of such patterns (the number of profitable trades) should be much higher than the unprofitable ones. All depends on the StopLoss/TakeProfit ratio.

Looking ahead, I should say that I performed 10 optimizations of each EA continuously. There are a lot of optimization values, which leads to results of about 10,000-15,000 per pass in the Genetic Algorithm mode. Accordingly, the more passes, the higher the chance to find the desired values of the weight ratios. This issue should be solved by means of MQL5. I really do not want to give up the strategy tester.

In contrast to the article referenced above (where the step is equal to 1), the step of 5 optimization values was not chosen by chance. During the experiments, I noticed that this leads to more scattered results of the perceptron weight ratios, which has a better effect on the results.

_**Optimizing EA 1 perceptron 1 figure. Simple lines. (One perceptron with one shape).**_

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), (Maximum profit).
- Initial deposit 10,000.
- TakeProfit = 60, StopLoss = 600.
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4  - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![1 perceptron 1 figure](https://c.mql5.com/2/47/1_1__1.png)

As you can see, the results are far from promising. The best result is 0.87. There is no point in running a forward test.

_**Optimizing EA 1 perceptron 4 figure. (One perceptron with four shapes).**_

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), (Maximum profit).
- Initial deposit 10,000.
- TakeProfit = 60, StopLoss = 600.
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4, y1, y2, y3, z1, z2, z3, f1, f2, f3, f4 - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![1 perceptron 4 figure](https://c.mql5.com/2/47/1_2__1.png)

The result is very similar to the previous one. The best result is 0.94.

_**Optimizing EA 4 perceptron 4 figure. (Four perceptrons with four different shapes).**_

The main code looks as follows:

```
//SELL++++++++++++++++++++++++++++++++++++++++++++++++

if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_SELL, EAComment)==0) && (ind_In1[1]>ind_In2[1]) && (perceptron1()<0) && (perceptron2()<0) && (perceptron3()<0) && (perceptron4()<0) && (SpreadS1<=MaxSpread)){//v1
  OpenSell(symbolS1.Name(), LotsXSell, TakeProfit, StopLoss, EAComment);
}

//BUY++++++++++++++++++++++++++++++++++++++++++++++++

if ((CalculatePositions(symbolS1.Name(), Magic, POSITION_TYPE_BUY, EAComment)==0) && (ind_In1[1]<ind_In2[1]) && (perceptron1()>0) && (perceptron2()>0) && (perceptron3()>0) && (perceptron4()>0) && (SpreadS1<=MaxSpread)){//v1
  OpenBuy(symbolS1.Name(), LotsXBuy, TakeProfit, StopLoss, EAComment);
}
```

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), ( [Maximum of the complex criterion](https://www.metatrader5.com/en/releasenotes/terminal/2157 "https://www.metatrader5.com/en/releasenotes/terminal/2157")).
- Initial deposit 10,000.
- TakeProfit = 200, StopLoss = 200.
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4, y1, y2, y3, z1, z2, z3, f1, f2, f3, f4 - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![4 perceptron 4 figure](https://c.mql5.com/2/47/4_p_4_f_rn.png)

Forward test date from 2021.05.31 to 2022.05.31.Out of all the results, we should choose the one featuring the largest profit factor with the maximum of the complex criterion exceeding 40-50.

![Test 1](https://c.mql5.com/2/47/4_p_4_f_test.png)

![Test 2](https://c.mql5.com/2/47/4_p_4_f_test_2.png)

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), (Maximum profit).
- Initial deposit 10,000.
- TakeProfit = 60, StopLoss = 600
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4, y1, y2, y3, z1, z2, z3, f1, f2, f3, f4 - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![4 perceptron 4 figure](https://c.mql5.com/2/47/1_3__1.png)

The result has been obtained. The best result is 32. Change the date from 2021.05.31 to 2022.05.31 and run a forward test. Out of all the results, we should choose the one featuring the largest profit factor with the minimum number of deals not less than 10-20.

![test 1](https://c.mql5.com/2/47/f_1__1.png)

![test 2](https://c.mql5.com/2/47/f_2__1.png)

_Optimizing EA 4 perceptron 4 tangent. ( _**Four perceptrons**_ with four different angles)._

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), ( [Maximum of the complex criterion](https://www.metatrader5.com/en/releasenotes/terminal/2157 "https://www.metatrader5.com/en/releasenotes/terminal/2157")).
- Initial deposit 10,000.
- TakeProfit = 200, StopLoss = 200.
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4, y1, y2, y3, z1, z2, z3, f1, f2, f3, f4 - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![4 perceptron 4 tangent](https://c.mql5.com/2/47/4_p_4_t.png)

Forward test date from 2021.05.31 to 2022.05.31. Out of all the results, we should choose the one featuring the largest profit factor with the maximum of the complex criterion exceeding 20-40.

![Test 1](https://c.mql5.com/2/47/4_p_4_t_test_2.png)

![Test 2](https://c.mql5.com/2/47/4_p_4_t_test.png)

- Optimization date from 2010.05.31 to 2021.05.31.
- Modes (Open prices only), (Genetic algorithm), (Maximum profit).
- Initial deposit 10,000.
- TakeProfit = 60, StopLoss = 600.
- Period H1.
- Fixed lot 0.01.
- Optimized parameters x1, x2, x3, x4, y1, y2, y3, z1, z2, z3, f1, f2, f3, f4 - perceptron weight ratios. Optimized by values from 0 to 200 in increments of 5.

Optimization and forward test results.

![4 perceptron 4 tangent](https://c.mql5.com/2/47/1_4.png)

The result has been obtained. The best result is 32. I will leave a forward test as a homework. I think, it will be more interesting this way. In addition, it should be noted that the number of trades in relation to the profit factor has increased.

**_Over the course of my experiments, I have encountered several issues that should be solved._**

- First. Due to the complexity of optimizing multiple weight ratio parameters, it is necessary to move them inside the EA code.
- Second. We should have a database for all optimized parameters and then use them in the EA for trading at the same time. I think, it will be possible to use files of .CSV type.

### Conclusion

I really hope that my experiments will lead you to new discoveries and, eventually, success. My objective was to obtain a ready-made profitable strategy. I have partially achieved it by getting good forward test results. However, there is still a lot of work ahead. It is time to move on to more complex systems while benefiting from the experience gained. Besides, we should make more use of what we have. Let's discuss this and much more in the second part of our experiments. Do not miss it! Things are about to get more exciting.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11077](https://www.mql5.com/ru/articles/11077)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11077.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/11077/ea.zip "Download EA.zip")(181.54 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)
- [Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Experiments with neural networks (Part 4): Templates](https://www.mql5.com/en/articles/12202)
- [Testing and optimization of binary options strategies in MetaTrader 5](https://www.mql5.com/en/articles/12103)
- [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/430104)**
(26)


![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
10 Feb 2023 at 05:36

**Aleksey Vyazmikin [#](https://www.mql5.com/ru/forum/426944/page2#comment_44922318):**

Goody-goody :)

How does it feel then? :)

70-80

![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
10 Feb 2023 at 06:05

**Roman Poshtar [#](https://www.mql5.com/ru/forum/426944/page2#comment_44928846):**

70-80

Interesting.

Have you thought about generating "perceptrons"?

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
10 Feb 2023 at 06:34

**Aleksey Vyazmikin [#](https://www.mql5.com/ru/forum/426944/page2#comment_44928980):**

Interesting.

Have you thought about generating "perceptrons"?

More details?

![Dominic Michael Frehner](https://c.mql5.com/avatar/2024/11/672504f5-a016.jpg)

**[Dominic Michael Frehner](https://www.mql5.com/en/users/cryptonist)**
\|
9 Mar 2025 at 10:08

Thanks a lot for your article, it was a very interesting read! How exactly did you manage to solve the optimization amount of so many passes? Because of so many passes I can't do more than 10k.

[![](https://c.mql5.com/3/458/2882872581371__1.png)](https://c.mql5.com/3/458/2882872581371.png "https://c.mql5.com/3/458/2882872581371.png")

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
10 Mar 2025 at 05:29

**Dominic Michael Frehner [#](https://www.mql5.com/en/forum/430104#comment_56114782):**

Thanks a lot for your article, it was a very interesting read! How exactly did you manage to solve the optimization amount of so many passes? Because of so many passes I can't do more than 10k.

You need to repeat several times.

![Data Science and Machine Learning — Neural Network (Part 01): Feed Forward Neural Network demystified](https://c.mql5.com/2/48/forward_neural_network.png)[Data Science and Machine Learning — Neural Network (Part 01): Feed Forward Neural Network demystified](https://www.mql5.com/en/articles/11275)

Many people love them but a few understand the whole operations behind Neural Networks. In this article I will try to explain everything that goes behind closed doors of a feed-forward multi-layer perception in plain English.

![DoEasy. Controls (Part 8): Base WinForms objects by categories, GroupBox and CheckBox controls](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__2.png)[DoEasy. Controls (Part 8): Base WinForms objects by categories, GroupBox and CheckBox controls](https://www.mql5.com/en/articles/11075)

The article considers creation of 'GroupBox' and 'CheckBox' WinForms objects, as well as the development of base objects for WinForms object categories. All created objects are still static, i.e. they are unable to interact with the mouse.

![Learn how to design a trading system by Bear's Power](https://c.mql5.com/2/48/why-and-how__3.png)[Learn how to design a trading system by Bear's Power](https://www.mql5.com/en/articles/11297)

Welcome to a new article in our series about learning how to design a trading system by the most popular technical indicator here is a new article about learning how to design a trading system by Bear's Power technical indicator.

![Learn how to design a trading system by Force Index](https://c.mql5.com/2/48/why-and-how__2.png)[Learn how to design a trading system by Force Index](https://www.mql5.com/en/articles/11269)

Welcome to a new article in our series about how to design a trading system by the most popular technical indicators. In this article, we will learn about a new technical indicator and how to create a trading system using the Force Index indicator.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=bzkkfspnwdgjbsfxhlzrdvciqadeseas&ssn=1769185793339631579&ssn_dr=0&ssn_sr=0&fv_date=1769185793&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11077&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Experiments%20with%20neural%20networks%20(Part%201)%3A%20Revisiting%20geometry%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918579324226498&fz_uniq=5070333889710330880&sv=2552)

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