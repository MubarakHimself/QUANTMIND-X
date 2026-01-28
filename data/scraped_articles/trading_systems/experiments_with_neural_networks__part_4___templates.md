---
title: Experiments with neural networks (Part 4): Templates
url: https://www.mql5.com/en/articles/12202
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:26:51.585724
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=nsuuibuskgmgkefxdnuhpoeouxwauryo&ssn=1769185609999985809&ssn_dr=0&ssn_sr=0&fv_date=1769185609&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12202&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Experiments%20with%20neural%20networks%20(Part%204)%3A%20Templates%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691856092521111&fz_uniq=5070290669454431048&sv=2552)

MetaTrader 5 / Trading systems


I would like to share my experience with you more often, but as you understand, such an activity requires time and computer resources, which, unfortunately, are very meager compared to the tasks set.

In the previous articles ( [Part 1](https://www.mql5.com/en/articles/11077), [Part 2](https://www.mql5.com/en/articles/11186), [Part 3](https://www.mql5.com/en/articles/11949)), we experimented with shapes and angles whose values were passed to the perceptron and the neural network built on the basis of the DeepNeuralNetwork.mqh library. We also conducted experiments on optimization methods in the strategy tester. I was not entirely satisfied with the results of the work of neural networks based on the DeepNeuralNetwork.mqh library, as well as with the slow work of the smart optimization algorithm described in [Part 3](https://www.mql5.com/en/articles/11949). The results on a simple perceptron turned out to be better than a neural network. Perhaps, we are passing inappropriate data to the neural network for such tasks, and its scatter range cannot yield a stable result. In addition, the feedback about previous articles contained criticism of the TakeProfit to StopLoss ratio. All that was taken into account in the following experiments.

In my searches, I came across an interesting pattern tracking algorithm on our favorite MQL5 forum. Its essence was to bring the price to a certain pattern to determine the entry into a position, while not concerning its use in neural networks.

I called this "technology" templates. I do not know if it is correct or not, but it seemed the most appropriate word to me.

An important task in the current experiments was to track the influence of the amount of transmitted data and the depth of history we take this data from. In addition, we needed to reveal patterns, whether short or long templates are better, as well as whether we should use fewer or more parameters for passing.

Now I am writing this introduction, and frankly, I do not know what result I will come to in the end. As always, I will use only MetaTrader 5 tools without any third-party software. This article will most likely be similar to a step-by-step instruction. I will try to explain everything as clearly and simply as possible.

### 1\. Currency pair. Optimization and forward test range. Settings

Here I will provide all the parameters for optimization and forward testing, so as not to repeat myself in the text:

- Forex;
- EURUSD;
- Timeframe: H1;
- Templates: Fan, Parallelogram, Triangle;
- Stop Loss and Take Profit for the corresponding modifications of 600 and 60, 200 and 230 for balance, taking into account the spread, add 30 points for five decimal places to TakeProfit, 200 and 430 Take Profit is 2 times more than Stop Loss, 30 points for five decimal places to TakeProfit for balance;
- "Open prices only" and "Complex Criterion max" optimization and testing modes. It is very important to use the "Maximum complex criterion" mode, it showed more stable and profitable results compared to "Maximum profitability";
- Optimization range 3 years. 2019.02.18 - 2022.02.18. 3 years is not a reliable criterion. You can experiment with this parameter on your own;
- Forward test range is 1 year. 2022.02.18 - 2023.02.18. Check everything based on the algorithm described in my article ( [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)). This means simultaneous trading of several best optimization results;
- We will now perform the optimization 20 times. Let's increase it by 2 times compared to the previous tests and look at the results.
- In all forward tests, 40 optimization results were used simultaneously. The value is increased 2 times in comparison with the previous tests;
- Optimization of EAs with the "Fast (genetic based algorithm)" perceptron;
- EA optimization on the DeepNeuralNetwork.mqh "Fast (genetic based algorithm)" library. Due to the slow optimization of the algorithm considered in ( [Part 2](https://www.mql5.com/en/articles/11186)), it was decided to optimize using MetaTrader 5 directly;
- Initial deposit 10,000 units;
- Leverage 1:500.

I almost forgot to tell you how I optimize 20, 40 or more times in the "Fast (genetic algorithm)" mode. To do this, I use a small autoclicker program I wrote in Delphi. I cannot post it here, but I will send it to anyone who needs it in a private message. It works as follows:

1. Enter the required number of optimizations.
2. Hover the mouse cursor over the Start button in the strategy optimizer.
3. Wait.

Optimization ends after the specified cycles and the program closes. The autoclicker responds to the change in the color of the Start button. The program is displayed in the screenshot below.

![Autoclicker](https://c.mql5.com/2/52/Pr__1.png)

### 2\. Templates

A template is a kind of construction similar to a "floating pattern". Its values are constantly changing depending on the situation on the market, but each of the values is in a certain range, which is what we need for our experiments. Since we already know that the data that we transmit to the neural network should be in a certain range, the value in the template is rounded up to an integer for simplicity and better understanding by the perceptron and the neural network. Thus, we get more situations for triggering conditions and less load on the perceptron and the neural network. Below you see the first of the templates that came to my mind. I called it a fan. I think, the similarity is obvious. We will not use indicators in this article, instead we work with candlesticks.

Below are examples using history zooming so we can analyze a shorter or deeper history.

Using an equal number of candles in templates is not a prerequisite, which gives an additional field for reflection on the relevance of previous price values. In our case, these are the closing prices of the candles.

It is important to understand that in the examples using the DeepNeuralNetwork.mqh library for 24 candles, we use different libraries that I described in the previous articles. They have different input settings. Namely, 4 and 8 parameters for the input of the neural network. You do not have to worry about it. I have already added EAs and necessary libraries in the attachment.

2.1 Fan template of four values stretched over 24 candles. It is equal to one day on H1.

![Fan 4 24](https://c.mql5.com/2/51/S_4_24__1.png)

Let's describe what we will transfer to the perceptron and the neural network for better understanding:

1. Rounded distance in points from point 1 to point 2;
2. Rounded distance in points from point 1 to point 3;
3. Rounded distance in points from point 1 to point 4;
4. Rounded distance in points from point 1 to point 5;

This is how it will look in the code for perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   a1 = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

This is how it will look in the code for EAs based on the DeepNeuralNetwork.mqh library:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.2 Fan template of eight values stretched over 24 candles. It is equal to one day on H1.

![Template 8 24](https://c.mql5.com/2/51/S_8_24__1.png)

Let's see what we pass to the perceptron and the neural network for better understanding:

1. Rounded distance in points from point 1 to point 2;
2. Rounded distance in points from point 1 to point 3;
3. Rounded distance in points from point 1 to point 4;
4. Rounded distance in points from point 1 to point 5;
5. Rounded distance in points from point 1 to point 6;
6. Rounded distance in points from point 1 to point 7;
7. Rounded distance in points from point 1 to point 8;
8. Rounded distance in points from point 1 to point 9;

This is how it will look in the code for perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   a1 = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,4))/Point()));
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,10))/Point()));
   a4 = (int)(a4/100)*100;

   int b1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   b1 = (int)(b1/100)*100;
   int b2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,16))/Point()));
   b2 = (int)(b2/100)*100;
   int b3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   b3 = (int)(b3/100)*100;
   int b4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,22))/Point()));
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

This is how it will look in the code for EAs based on the DeepNeuralNetwork.mqh library:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,4))/Point()));
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,10))/Point()));
   xInputs[3] = (int)(a4/100)*100;

   int g1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   xInputs[4] = (int)(g1/100)*100;
   int g2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,16))/Point()));
   xInputs[5] = (int)(g2/100)*100;
   int g3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   xInputs[6] = (int)(g3/100)*100;
   int g4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,22))/Point()));
   xInputs[7] = (int)(g4/100)*100;

   return(1);

  }
```

2.3 Fan template of four values stretched over 48 candles. Equal to two days on H1.

![Fan 48](https://c.mql5.com/2/51/S_4_48.png)

Let's provide a description for better understanding:

1. Rounded distance in points from point 1 to point 2;
2. Rounded distance in points from point 1 to point 3;
3. Rounded distance in points from point 1 to point 4;
4. Rounded distance in points from point 1 to point 5;

This is how it will look in the code for perceptron EAs. We see the previous example stretched in time for 2 days:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   a1 = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point()));
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point()));
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

This is how it will look in the code in DeepNeuralNetwork.mqh. We see the previous example stretched in time for 2 days:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point()));
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point()));
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.4 Fan template of eight values stretched over 48 candles. Equal to two days on H1.

![Fan 8 48](https://c.mql5.com/2/51/S_8_48__1.png)

The reference points have shifted by an equal number of candles:

1. Rounded distance in points from point 1 to point 2;
2. Rounded distance in points from point 1 to point 3;
3. Rounded distance in points from point 1 to point 4;
4. Rounded distance in points from point 1 to point 5;
5. Rounded distance in points from point 1 to point 6;
6. Rounded distance in points from point 1 to point 7;
7. Rounded distance in points from point 1 to point 8;
8. Rounded distance in points from point 1 to point 9;

The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   a1 = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   a4 = (int)(a4/100)*100;

   int b1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point()));
   b1 = (int)(b1/100)*100;
   int b2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,31))/Point()));
   b2 = (int)(b2/100)*100;
   int b3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point()));
   b3 = (int)(b3/100)*100;
   int b4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,43))/Point()));
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

The code in the EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point()));
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point()));
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point()));
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point()));
   xInputs[3] = (int)(a4/100)*100;

   int g1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point()));
   xInputs[4] = (int)(g1/100)*100;
   int g2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,31))/Point()));
   xInputs[5] = (int)(g2/100)*100;
   int g3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point()));
   xInputs[6] = (int)(g3/100)*100;
   int g4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,43))/Point()));
   xInputs[7] = (int)(g4/100)*100;

   return(1);

  }
```

2.5 Parallelogram template of four values stretched over 24 candles. A more complex construction for passing parameters. Equal to one day on H1.

![Parallelogram](https://c.mql5.com/2/51/S_4_24__3.png)

Below is a description of what values we pass:

1. In my case, add 800 points for five decimal places from point 10 to point 2;
2. Subtract 800 points from point 10 to point 1;
3. Add 800 points from point 9 to point 3;
4. Subtract 800 points from point 9 to point 4;
5. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 3);
6. Rounded value from point 3 to point 7;
7. Rounded value from point 2 to point 6;

8. Rounded value from point 1 to point 5;


The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,7)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

The code in the EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,7)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.6 Parallelogram template of four values stretched over 48 candles. Greater coverage over time. Equal to two days on H1.

![Parallelogram](https://c.mql5.com/2/51/S_4_48__1.png)

Below is a description of what values we pass:

1. In my case, add 1200 points for five decimal places from point 10 to point 2; I have increased the value in points, since the price can vary widely within two days. This way our construction remains intact;
2. Subtract 1200 points from point 10 to point 1;
3. Add 1200 points from point 9 to point 3;
4. Subtract 1200 points from point 9 to point 4;
5. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 3);
6. Rounded value from point 3 to point 7;
7. Rounded value from point 2 to point 6;

8. Rounded value from point 1 to point 5;

The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,13)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

The code in the EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,13)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.7 Parallelogram template of eight values stretched over 24 candles. It is equal to one day on H1.

![Parallelogram](https://c.mql5.com/2/52/S_8_24__1.png)

Below is a description of what values we pass:

01. In my case, add 800 points for five decimal places from point 9;
02. Subtract 800 points from point 9, get point 4;
03. Add 800 points from point 12 to point 2;
04. Subtract 800 points from point 12 to point 1;
05. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 3);
06. Rounded value from point 7 to point 3;
07. Rounded value from point 13 to point 4;

08. Rounded value from point 6 to point 3;
09. Rounded value from point 2 to point 6;

10. Rounded value from point 2 to point 10;

11. Rounded value from point 12 to point 11 (point 1 is found as the difference between points 4 and 1);

12. Rounded value from point 1 to point 5;


The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,5)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,9)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,13)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())))/Point());
   a4 = (int)(a4/100)*100;

   int b1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   b1 = (int)(b1/100)*100;
   int b2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,17))/Point());
   b2 = (int)(b2/100)*100;
   int b3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,21))/Point());
   b3 = (int)(b3/100)*100;
   int b4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))/2))/Point()) ;
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

This is how it looks in the code for EAs based on the DeepNeuralNetwork.mqh library:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,5)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,9)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,13)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())))/Point());
   xInputs[3] = (int)(a4/100)*100;

   int g1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[4] = (int)(g1/100)*100;
   int g2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)+(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,17))/Point());
   xInputs[5] = (int)(g2/100)*100;
   int g3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,21))/Point());
   xInputs[6] = (int)(g3/100)*100;
   int b4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,24)-(800*Point()))/2))/Point()) ;
   xInputs[7] = (int)(g4/100)*100;

   return(1);

  }
```

2.8 Parallelogram template of eight values stretched over 48 candles for greater history coverage. Equal to two days on H1.

![Parallelogram](https://c.mql5.com/2/52/S_8_48.png)

Below is a description of what values we pass:

01. In my case, add 1200 points for five decimal places from point 9;
02. Subtract 1200 points from point 9, get point 4;
03. Add 1200 points from point 12 to point 2;
04. Subtract 1200 points from point 12 to point 1;
05. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 3);
06. Rounded value from point 7 to point 3;
07. Rounded value from point 13 to point 4;

08. Rounded value from point 6 to point 3;
09. Rounded value from point 2 to point 6;

10. Rounded value from point 2 to point 10;

11. Rounded value from point 12 to point 11 (point 1 is found as the difference between points 4 and 1);

12. Rounded value from point 1 to point 5;

The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,9)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,17)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,25)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())))/Point());
   a4 = (int)(a4/100)*100;

   int b1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   b1 = (int)(b1/100)*100;
   int b2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,33))/Point());
   b2 = (int)(b2/100)*100;
   int b3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,41))/Point());
   b3 = (int)(b3/100)*100;
   int b4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))/2))/Point()) ;
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

The code for EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,9)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())))/Point());
   xInputs[0] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,17)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())))/Point());
   xInputs[0] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,25)-(iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())))/Point());
   xInputs[0] = (int)(a4/100)*100;

   int g1 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   xInputs[0] = (int)(g1/100)*100;
   int g2 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)+(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,33))/Point());
   xInputs[0] = (int)(g2/100)*100;
   int g3 = (int)(((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))-iClose(symbolS1.Name(),PERIOD_CURRENT,41))/Point());
   xInputs[0] = (int)(g3/100)*100;
   int g4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point()))+(iClose(symbolS1.Name(),PERIOD_CURRENT,48)-(1200*Point()))/2))/Point()) ;
   xInputs[0] = (int)(g4/100)*100;

   return(1);

  }
```

2.9 Triangle template of four values stretched over 24 candles. It is equal to one day on H1.

![Triangle](https://c.mql5.com/2/52/t_4_24.png)

Below is a description of what values we pass:

1. In my case, add 800 points for five decimal places from point 4 to point 2;
2. Subtract 800 points from point 4 to point 3;
3. Rounded value from point 8 to point 4 (point 8 is found as the difference between points 2 and 1);
4. Rounded value from point 3 to point 5;
5. Rounded value from point 1 to point 6;
6. Rounded value from point 1 to point 7;

The code in perceptron EAs:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

The code for EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.10 Triangle template of four values, history coverage of 48 candles. Two days on H1.

![Triangle](https://c.mql5.com/2/52/t_4_48.png)

Below are the values passed to the perceptron and the neural network:

1. In my case, add 1200 points for five decimal places from point 4 to point 2;
2. Subtract 1200 points from point 4 to point 3;
3. Rounded value from point 8 to point 4 (point 8 is found as the difference between points 2 and 1);
4. Rounded value from point 3 to point 5;
5. Rounded value from point 1 to point 6;
6. Rounded value from point 1 to point 7;

Template code in perceptron EA:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   a4 = (int)(a4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4);
  }
```

Template code for EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   xInputs[3] = (int)(a4/100)*100;

   return(1);

  }
```

2.11 Triangle template of eight values stretched over 24 candles. History coverage for analysis is one day on H1.

![Triangle](https://c.mql5.com/2/52/t_8_24.png)

Passed values are described below:

01. In my case, add 800 points for five decimal places from point 4 to point 2;
02. Subtract 800 points from point 4 to point 3;
03. Rounded value from point 8 to point 4 (point 8 is found as the difference between points 2 and 1);
04. Rounded value from point 3 to point 5;
05. Rounded value from point 1 to point 6;
06. Rounded value from point 1 to point 7;
07. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 1);

08. Rounded value from point 8 to point 10 (point 8 is found as the difference between points 2 and 1);

09. Rounded value from point 8 to point 11 (point 8 is found as the difference between points 2 and 1);

10. Rounded value from point 8 to point 12 (point 8 is found as the difference between points 2 and 1);


Template code in perceptron EA:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   a4 = (int)(a4/100)*100;

   int b1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,4))/Point());
   b1 = (int)(b1/100)*100;
   int b2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,10))/Point());
   b2 = (int)(b2/100)*100;
   int b3 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,16))/Point());
   b3 = (int)(b3/100)*100;
   int b4 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,22))/Point());
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

Template code for EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,7))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,24)-iClose(symbolS1.Name(),PERIOD_CURRENT,19))/Point());
   xInputs[3] = (int)(a4/100)*100;

   int g1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,4))/Point());
   xInputs[4] = (int)(g1/100)*100;
   int g2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,10))/Point());
   xInputs[5] = (int)(g2/100)*100;
   int g3 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,16))/Point());
   xInputs[6] = (int)(g3/100)*100;
   int g4 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(800*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,24))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,22))/Point());
   xInputs[7] = (int)(g4/100)*100;

   return(1);

  }
```

2.12 Triangle template of eight values stretched over 48 candles. History coverage for analysis is two days on H1.

![Triangle](https://c.mql5.com/2/52/t_8_48.png)

Passed values are described below:

01. In my case, add 1200 points for five decimal places from point 4 to point 2, the added value is increased by 48 candles;
02. Subtract 1200 points from point 4 to point 3;
03. Rounded value from point 8 to point 4 (point 8 is found as the difference between points 2 and 1);
04. Rounded value from point 3 to point 5;
05. Rounded value from point 1 to point 6;
06. Rounded value from point 1 to point 7;
07. Rounded value from point 8 to point 9 (point 8 is found as the difference between points 2 and 1);

08. Rounded value from point 8 to point 10 (point 8 is found as the difference between points 2 and 1);

09. Rounded value from point 8 to point 11 (point 8 is found as the difference between points 2 and 1);

10. Rounded value from point 8 to point 12 (point 8 is found as the difference between points 2 and 1);

Template code in perceptron EA:

```
//+------------------------------------------------------------------+
//|  The PERCEPRRON - a perceiving and recognizing function          |
//+------------------------------------------------------------------+
double perceptron1()
  {
   double w1 = x1 - 10.0;
   double w2 = x2 - 10.0;
   double w3 = x3 - 10.0;
   double w4 = x4 - 10.0;

   double v1 = y1 - 10.0;
   double v2 = y2 - 10.0;
   double v3 = y3 - 10.0;
   double v4 = y4 - 10.0;

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   a1 = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   a2 = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   a3 = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   a4 = (int)(a4/100)*100;

   int b1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,6))/Point());
   b1 = (int)(b1/100)*100;
   int b2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,18))/Point());
   b2 = (int)(b2/100)*100;
   int b3 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,31))/Point());
   b3 = (int)(b3/100)*100;
   int b4 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,43))/Point());
   b4 = (int)(b4/100)*100;

   return (w1 * a1 + w2 * a2 + w3 * a3 + w4 * a4   +   v1 * b1 + v2 * b2 + v3 * b3 + v4 * b4);
  }
```

Triangle template code for EAs based on DeepNeuralNetwork.mqh:

```
//+------------------------------------------------------------------+
//|percentage of each part of the candle respecting total size       |
//+------------------------------------------------------------------+
int CandlePatterns(double &xInputs[])
  {

   int a1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,1))/Point());
   xInputs[0] = (int)(a1/100)*100;
   int a2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,13))/Point());
   xInputs[1] = (int)(a2/100)*100;
   int a3 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,25))/Point());
   xInputs[2] = (int)(a3/100)*100;
   int a4 = (int)((iClose(symbolS1.Name(),PERIOD_CURRENT,48)-iClose(symbolS1.Name(),PERIOD_CURRENT,37))/Point());
   xInputs[3] = (int)(a4/100)*100;

   int g1 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)+(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,6))/Point());
   xInputs[4] = (int)(g1/100)*100;
   int g2 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,18))/Point());
   xInputs[5] = (int)(g2/100)*100;
   int g3 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,31))/Point());
   xInputs[6] = (int)(g3/100)*100;
   int g4 = (int)((((iClose(symbolS1.Name(),PERIOD_CURRENT,1)-(1200*Point())+iClose(symbolS1.Name(),PERIOD_CURRENT,48))/2)-iClose(symbolS1.Name(),PERIOD_CURRENT,43))/Point());
   xInputs[7] = (int)(g4/100)*100;

   return(1);

  }
```

### 3\. Expert Advisors

So, let's get down to the most interesting part, namely optimization and testing of our templates. As you might remember, the optimization and testing of Expert Advisors on the DeepNeuralNetwork.mqh library was carried out using standard MQL5 tools without using the optimization technology described in ( [Article 2](https://www.mql5.com/en/articles/11186)). It has also been noticed that 20 passes is a bit small for these EAs. I recommend that you do the optimization yourself with a greater number of iterations. I am sure this will improve the result. A large number of parameters to be optimized requires more time to identify the best results. In this article, I only want to show non-standard methods for transmitting data in a neural network.

The Expert Advisors based on the DeepNeuralNetwork.mqh library with four parameters contain the 4-4-3 neural network scheme in the template, while in case of eight parameters, the scheme is 8-4-3.

I tried to name each of the EAs in accordance with its strategy and the template used for analysis. So I think it is hard to get lost. Anyway, you can always contact me on the forum or via private messages.

The first 40 best results obtained during optimization in the “Complex Criterion max” mode were used in each EA for forward tests. I will post the results in the format of optimization results and below are the results of forward testing.

In this article, I will test EAs based on the fan template and draw conclusions about the presented “technology”. You can test the rest of the templates yourselves if interested. I posted the technical part in the form of ready-made codes for all templates above, so I think you will have no problems replacing the template codes in the EAs posted at the end of the article and carrying out optimization and testing.

If you have troubles in understanding the test, read [Part 3](https://www.mql5.com/en/articles/11949) of the series. Everything is described in detail there.

3.1 Perceptron-based EAs

**Perceptron fan 4 SL TP 24 - trade**\- four parameters on 24 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/51/Opt.png)

![Forward](https://c.mql5.com/2/51/Trade.png)

The result of forward testing is not encouraging. We can see a smooth draining of the deposit throughout the entire history. The profit factor of test results is at a high level. Probably, this has to do with the ratio of Stop Loss to Take Profit.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/51/Opt__1.png)

![Forward](https://c.mql5.com/2/51/Trade__1.png)

The profit factor barely exceeds the value of 1.8. Large deposit drawdowns can be seen throughout the annual history. Minimum deposit growth. We are simply treading in place.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/51/Opt__2.png)

![Forward](https://c.mql5.com/2/51/Trade__2.png)

The fluctuation of the deposit is seen on the entire range of the forward test. The profits match the losses and the results are unstable. The profit factor during the optimization is about 2.

**Perceptron fan 4 SL TP 48 - trade**\- four parameters on 48 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/51/Opt__3.png)

![Forward](https://c.mql5.com/2/51/Trade__3.png)

A stable growth of the deposit for the first six months is followed by a certain decline. I believe, this happens due to the lack of additional optimization for such a long period of time. More stable results are observed by increasing the time in the passed parameters. The profit factor is much higher than the result on 24 candles. No signs of binding to the Stop Loss to Take Profit ratio.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/51/Opt__4.png)

![Forward](https://c.mql5.com/2/51/Trade__4.png)

A stable growth of the deposit for the first five months, then a decline. Perhaps, the market has changed in contrast to the selected conditions in the perceptron. The uneven graph of the first five months still suggests the instability of the system.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/51/Opt__5.png)

![Forward](https://c.mql5.com/2/51/Trade__5.png)

The most stable result in tests on the perceptron with the fan template. Steady growth for the first six months. Then, a slight decline due to the lack of re-optimization. TakeProfit more than twice StopLoss gives a good result according to the basic trading rules. The profit is greater than the stop. When optimized, the profit factor is at the level of 1.6, which I think is natural given the ratio of Stop Loss to Take Profit.

**Perceptron fan 8 SL TP 24 - trade**\- eight parameters on 24 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt__12.png)

![Forward](https://c.mql5.com/2/52/Trade__12.png)

The graph is quite uneven but the result is fundamentally different from the EA with four parameters with the same Take Profit and Stop Loss. There is also a slight decline after the first half of the year. The average profit factor during optimization is about 6, which is quite a lot.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__13.png)

![Forward](https://c.mql5.com/2/52/Trade__13.png)

In this forward test, we experience a complete failure. I think this is due to the ratio of TakeProfit and StopLoss one to one. Optimization showed a profit factor around 1.7, but this did not save the situation.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__14.png)

![Forward](https://c.mql5.com/2/52/Trade__14.png)

This option is also a complete failure, although there is some resistance at first. Optimization showed a profit factor around 1.8, but that was of no help again.

**Perceptron fan 8 SL TP 48 - trade**\- eight parameters on 48 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt__15.png)

![Forward](https://c.mql5.com/2/52/Trade__15.png)

Uneven graph. Taking into account the ratio of TakeProfit to StopLoss, we get a loss. Optimization showed a profit factor about 3.5-4, but forward tests showed a loss.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__16.png)

![Forward](https://c.mql5.com/2/52/Trade__16.png)

We also get fluctuations in one place in this version. Strangely enough, the decline in balance occurs immediately at the beginning of the test. Optimization showed a profit factor about 2.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__17.png)

![Forward](https://c.mql5.com/2/52/Trade__17.png)

A very good start at the beginning for approximately the first 3 months. Then the lack of new optimization apparently showed itself. The ratio of Take Profit to Stop Loss two to one did not save the situation for such a long period. When optimizing, the average profit factor was 1.4.

3.2 EAs based on the DeepNeuralNetwork.mqh library.

**4-4-3 fan 4 SL TP 24 - trade**\- four parameters on 24 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt.png)

![Forward](https://c.mql5.com/2/52/Trade.png)

The profit factor 20 which is a lot. The forward test shows positive results, but we have large Stop Losses. But still, a larger number of small positive Take Profits saves the situation.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__1.png)

![Forward](https://c.mql5.com/2/52/Trade__1.png)

The balance chart is treading water in one place. Optimization showed a profit factor of about 1.7.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__2.png)

![Forward](https://c.mql5.com/2/52/Trade__2.png)

Slow but sure decline. The ratio of TakeProfit to StopLoss does not save the situation. Optimization showed a profit factor of about 2.

**4-4-3 fan 4 SL TP 48 - trade**\- four parameters on 48 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt__3.png)

![Forward](https://c.mql5.com/2/52/Trade__3.png)

Passing the template at 48 candles did not show a positive result in contrast to the same template at 24 candles. Apparently, such a template that is so long in time does not work well with such a Take Profit to Stop Loss ratio. Optimization showed a profit factor of about 14, which is quite a lot.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__4.png)

![Forward](https://c.mql5.com/2/52/Trade__4.png)

When optimizing, we get a profit factor of 2.5. As we can see, TakeProfit 230 StopLoss 200 are not helpful. We get a slow decline in the balance.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__5.png)

![Forward](https://c.mql5.com/2/52/Trade__5.png)

In this case, we have no progress. Throughout the year, the balance falls and grows. When optimizing, we get a profit factor of 2.7.

**8-4-3 fan 8 SL TP 24 - trade**\- eight parameters on 24 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt__6.png)

![Forward](https://c.mql5.com/2/52/Trade__6.png)

Very interesting results, only one StopLoss in the entire history of forward testing. But still, the results may be random. When optimizing, the profit factor goes off scale at around 29.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__7.png)

![Forward](https://c.mql5.com/2/52/Trade__7.png)

As we can see from the screenshot, there is no progress again. When optimizing, the profit factor is at the level of 2.7, which should have been enough with the current TakeProfit to StopLoss ratio, but it was not. Some upsurge is seen for the first six months.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__8.png)

![Forward](https://c.mql5.com/2/52/Trade__8.png)

For the first two or three months, there was a slight increase, then, apparently, the situation on the market changed, and the EA started losing the deposit. Apparently, constant optimization is still needed. When optimizing, the profit factor is at the level of 3.9.

**8-4-3 fan 8 SL TP 48 - trade**\- eight parameters on 48 candles, fan template:

Take profit 60 stop loss 600:

![Optimization](https://c.mql5.com/2/52/Opt__9.png)

![Forward](https://c.mql5.com/2/52/Trade__9.png)

The behavior of this EA is similar to the previous one on a 24-candle pattern. More losing trades. When optimizing, the profit factor was at the level of 26. In case of 24 candles, it was at the level of 29.

Take Profit 230 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__10.png)

![Forward](https://c.mql5.com/2/52/Trade__10.png)

When optimizing, the profit factor is at the level of 3. The balance level stays the same. Losses alternate with profits.

Take Profit 430 Stop Loss 200:

![Optimization](https://c.mql5.com/2/52/Opt__11.png)

![Forward](https://c.mql5.com/2/52/Trade__11.png)

Take Profit being twice more than Stop Loss does not lead to a positive result. Most likely, the neural network is unable to predict such a large Take Profit. When optimizing, the profit factor is at the level of 3.

### Conclusion

We can draw both positive and negative conclusions from the work done. I will provide them in the form of a small list so as not to lose my train of thought.

- In terms of scaling, it turned out to be a very flexible system. It is possible to apply an unlimited number of templates and the parameters that we pass to them. Come up with new patterns and observe the results, which has a better effect on forward testing.
- It may be necessary to try systems with several perceptrons and different templates in them to determine the entry signal.
- Computer power is clearly not enough. Systems with a huge number of cores are needed. Ideally, a two CPU assembly with a total of 16 or more cores. As you might know, the strategy tester uses only physical cores, not threads. The ability to use [MQL5 Cloud Network](https://cloud.mql5.com/en "https://cloud.mql5.com/en") can significantly increase the productivity of our search.
- The number of passed inputs significantly increases the load on a perceptron or a neural network. Rounding the values of the inputs increased the number of positive results for about twice.
- Before choosing a system for further development, it is necessary to check a few more options for passing data to the perceptron and the neural network, namely indicators that move in a certain range, as well as such an interesting phenomenon as divergence. I think, I will do this in the near future.

The list of attached files:

01. DeepNeuralNetwork - original library;
02. DeepNeuralNetwork2 - modified library for the 4-4-3 structure neural network;
03. DeepNeuralNetwork3 - modified library for the 8-4-3 structure neural network;
04. perceptron fan 4 SL TP 24 - opt - perceptron-based EA for fan template optimization with four parameters on 24 candles;
05. perceptron fan 4 SL TP 48 - opt  - perceptron-based EA for fan template optimization with four parameters on 48 candles;
06. perceptron fan 8 SL TP 24 - opt - perceptron-based EA for fan template optimization with eight parameters on 24 candles;
07. perceptron fan 8 SL TP 48 - opt - perceptron-based EA for fan template optimization with eight parameters on 48 candles;
08. perceptron fan 4 SL TP 24 - trade (600 60), (200 230), (200 430) - optimized perceptron-based EAs, fan template with four parameters on 24 candles;
09. perceptron fan 4 SL TP 48 - trade (600 60), (200 230), (200 430) - optimized perceptron-based EAs, fan template with four parameters on 48 candles;

10. perceptron fan 8 SL TP 24 - trade (600 60), (200 230), (200 430) - optimized perceptron-based EAs, fan template for eight parameters on 24 candles;

11. perceptron fan 8 SL TP 48 - trade (600 60), (200 230), (200 430) - optimized perceptron-based EAs, fan template for eight parameters on 48 candles;

12. 4-4-3 fan 4 SL TP 24 - opt -  library-based EA for fan template optimization with four parameters on 24 candles;
13. 4-4-3 fan 4 SL TP 48 - opt -  library-based EA for fan template optimization with four parameters on 48 candles;

14. 8-4-3 fan 4 SL TP 24 - opt -  library-based EA for fan template optimization with eight parameters on 24 candles;

15. 8-4-3 fan 4 SL TP 48 - opt -  library-based EA for fan template optimization with eight parameters on 48 candles;

16. 4-4-3 fan 4 SL TP 24 - trade (600 60), (200 230), (200 430) - optimized library-based EAs, fan template with four parameters on 24 candles;
17. 4-4-3 fan 4 SL TP 48 - trade (600 60), (200 230), (200 430) - optimized library-based EAs, fan template with four parameters on 48 candles;

18. 8-4-3 fan 4 SL TP 24 - trade (600 60), (200 230), (200 430) - optimized library-based EAs, fan template with for eight parameters on 24 candles;

19. 8-4-3 fan 4 SL TP 48 - trade (600 60), (200 230), (200 430) - optimized library-based EAs, fan template with for eight parameters on 48 candles;

Thank you for your attention!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12202](https://www.mql5.com/ru/articles/12202)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12202.zip "Download all attachments in the single ZIP archive")

[EA.zip](https://www.mql5.com/en/articles/download/12202/ea.zip "Download EA.zip")(4191.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)
- [Experiments with neural networks (Part 6): Perceptron as a self-sufficient tool for price forecast](https://www.mql5.com/en/articles/12515)
- [Experiments with neural networks (Part 5): Normalizing inputs for passing to a neural network](https://www.mql5.com/en/articles/12459)
- [Testing and optimization of binary options strategies in MetaTrader 5](https://www.mql5.com/en/articles/12103)
- [Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)
- [Experiments with neural networks (Part 2): Smart neural network optimization](https://www.mql5.com/en/articles/11186)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/446348)**
(16)


![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
8 Apr 2023 at 18:02

**Sergei Poliukhov day of the week, hours**
**, for example from 15-17 volatile hours and figures breakout triangles.**

**Maybe the first Friday of the month (nonfarm). Also can add imbalance (middle of impulse candle, orderblocks, only those with imbalances.**

**For example, on Friday after strong news, the last day of the month is usually nasty. Or the last day of the month, too. I also noticed behaviour that the last minute which is a multiple of 15, 30 minutes is imbalanced.**

Everything can be done. If you want to. Thanks for the feedback. [Trading sessions](https://www.mql5.com/en/docs/marketinformation/symbolinfosessionquote "MQL5 Documentation: Getting Market Information") are under development.

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
8 Apr 2023 at 18:03

**Sergei Poliukhov [#](https://www.mql5.com/ru/forum/442691/page2#comment_46144416):**

I can also give mvli power for testing for free during working hours if you are a developer.

The team is being recruited. Write in PM.

![Guilherme Mendonca](https://c.mql5.com/avatar/2018/9/5B98163A-29AC.jpg)

**[Guilherme Mendonca](https://www.mql5.com/en/users/billy-gui)**
\|
28 Apr 2023 at 03:15

Great job on your article series!

Your adventures in the world of [neural networks](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") seem to parallel mine.

Thanks for taking the time to write and share with all of us.

![Roman Poshtar](https://c.mql5.com/avatar/2019/12/5DE4DBC6-D492.jpg)

**[Roman Poshtar](https://www.mql5.com/en/users/romanuch)**
\|
28 Apr 2023 at 05:19

**Guilherme Mendonca [#](https://www.mql5.com/en/forum/446348#comment_46539780):**

Great job on your article series!

Your adventures in the world of [neural networks](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") seem to parallel mine.

Thanks for taking the time to write and share with all of us.

Thanks for the feedback. Very pleased to hear.

![Top-T](https://c.mql5.com/avatar/avatar_na2.png)

**[Top-T](https://www.mql5.com/en/users/top-t)**
\|
27 Jun 2023 at 20:12

Hey Roman, Thank you very much for these articles you shared.

I would like to know if you have provided an explination for what these values are refering to?

string [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis")\[\]\[37\]=

{

    {"1.71225","-1","-1","-1","0.6","-0.5","0.5","-1","-1","-1","-1","-0.3","-0.5","-0.3","0.6","0","-0.4","0.1","0.1","-0.5","0.9","-0.3","0.5","-0.4","0","0.1","0.1","-0.2","-0.8","-0.5","0.5","0.3","0.8","-1","-1","-1"},

.....

in the 4-4-3 fan 4 SL TP 24 - trade (200 230) script.

and woukd I be able to get these values from other pairs?

Thank you in advanced.

Taher

![Creating an EA that works automatically (Part 09): Automation (I)](https://c.mql5.com/2/50/aprendendo_construindo_009_avatar.png)[Creating an EA that works automatically (Part 09): Automation (I)](https://www.mql5.com/en/articles/11281)

Although the creation of an automated EA is not a very difficult task, however, many mistakes can be made without the necessary knowledge. In this article, we will look at how to build the first level of automation, which consists in creating a trigger to activate breakeven and a trailing stop level.

![How to create a custom indicator (Heiken Ashi) using MQL5](https://c.mql5.com/2/54/heikin_ashi_avatar.png)[How to create a custom indicator (Heiken Ashi) using MQL5](https://www.mql5.com/en/articles/12510)

In this article, we will learn how to create a custom indicator using MQL5 based on our preferences, to be used in MetaTrader 5 to help us read charts or to be used in automated Expert Advisors.

![Creating an EA that works automatically (Part 10): Automation (II)](https://c.mql5.com/2/50/aprendendo_construindo_010_avatar.png)[Creating an EA that works automatically (Part 10): Automation (II)](https://www.mql5.com/en/articles/11286)

Automation means nothing if you cannot control its schedule. No worker can be efficient working 24 hours a day. However, many believe that an automated system should operate 24 hours a day. But it is always good to have means to set a working time range for the EA. In this article, we will consider how to properly set such a time range.

![Neural networks made easy (Part 36): Relational Reinforcement Learning](https://c.mql5.com/2/52/Neural_Networks_Made_036_avatar.png)[Neural networks made easy (Part 36): Relational Reinforcement Learning](https://www.mql5.com/en/articles/11876)

In the reinforcement learning models we discussed in previous article, we used various variants of convolutional networks that are able to identify various objects in the original data. The main advantage of convolutional networks is the ability to identify objects regardless of their location. At the same time, convolutional networks do not always perform well when there are various deformations of objects and noise. These are the issues which the relational model can solve.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/12202&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070290669454431048)

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