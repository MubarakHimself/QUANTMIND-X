---
title: Learn how to design a trading system by Stochastic
url: https://www.mql5.com/en/articles/10692
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:14:17.269326
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/10692&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069266474373153350)

MetaTrader 5 / Trading


### Introduction

One of the most beautiful things in technical analysis trading is that we have a lot of tools and we can use one tool individually or combine more than one tools and this can be useful and profitable for us. In addition to that, we have an amazing tool which can be very helpful in enhancing our trading results and performance. This tool is programming allowing us to create programs which can be very useful and helpful: we give the computer instructions to do what we need automatically and accurately according to our conditions.

This article is another contribution to trying to learn and teach what can be helpful in this context of technical trading and programming of profitable trading strategies. We will discuss through this article a new tool that can be useful in our trading and it is the "Stochastic Oscillator" indicator which is one of the most popular in trading using technical analysis. If you read my other articles here, you should know that I like learning and teaching roots of things to understand all the details and the mechanism of things working as much as I can. So, I will discuss with you this topic using the same method. The following topics will be covered in this article:

1. [Stochastic definition](https://www.mql5.com/en/articles/10692#definition)
2. [Stochastic strategy](https://www.mql5.com/en/articles/10692#strategy)
3. [Stochastic blueprint](https://www.mql5.com/en/articles/10692#blueprint)
4. [Stochastic trading system](https://www.mql5.com/en/articles/10692#system)
5. [Conclusion](https://www.mql5.com/en/articles/10692#conclusion)

Through the "Stochastic definition" topic, we will learn about the Stochastic Oscillator indicator in detail to know what it is, what it measures. Then, in "Stochastic strategy", we will learn some simple strategies that we can use. After that we will design a blueprint for these simple strategies to help us code them in MQL5 so that further we can use them in MetaTrader 5 — this will be considered in the "Stochastic blueprint" topic. Then we will proceed to learning how to write a code to design a trading system for these simple strategies — this will be done in the topic of "Stochastic trading system".

We will use the MetaTrader 5 trading terminal and MetaQuotes Language, which is built-in with MetaTrader 5. The terminal can be downloaded at the following link: [https://www.metatrader5.com/en/download](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download")

Disclaimer: All information is provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

Now, we will start our new journey about a new interesting topic, let us do it.

### Stochastic definition

In this part, we will learn in more detail about one of the most popular technical indicators, which is the Stochastic Oscillator indicator. It was developed by George Lane and this what is mentioned in "New Trading Systems and Methods" book by Perry J. Kaufman. But there are other opinions about that, one of them is that it is not absolutely clear who created the Stochastic indicator — this is mentioned in book "Technical Analysis, The Complete Resource for Financial Market Technicians" by Charles D. Kirkpatrick and Julie R. Dahlquist and the other opinion is that the indicator was popularized by George Lane as per what is mentioned in "Technical Analysis of The Financial Markets" book by John J. Murphy.

The idea behind the Stochastic indicator is as follows: there was an observation that during the uptrend, closing prices tend to close closer to the upper part of the price range for a specific period and vice versa, during the downtrend, closing prices tend to close closer to the lower part of the price range for a specific period. It measures the relationship between the current closing price and both the highest and lowest prices of a specific period.

The Stochastic Oscillator indicator consists of two lines and these two lines oscillate between zero level and 100 level:

- %K line is the fastest line.
- %D line is the slowest line, it is smoothed.

There is more than one version for the Stochastic Oscillator indicator. The most popular versions are the following two:

- The fast Stochastic.
- The slow Stochastic.

We will see the difference in the methods in which the Stochastic Oscillator indicator can be calculated.

The following are the steps to calculate the Stochastic Oscillator indicator:

1. Determine the period that is needed to calculate the indicator.
2. Determine the highest high value of the determined period.
3. Determine the lowest low value of the determined period.
4. Get %K of fast stochastic = 100\*((current close-lowest low)/(highest high-lowest low))
5. Get %D of fast stochastic = 3 moving average of %K of fast stochastic
6. Get %K of slow stochastic = %D of fast stochastic
7. Get %D of slow stochastic = 3 moving average of (%K of slow stochastic which is the same of %D of fast stochastic)

Now, let us see an example of how to apply these calculations to get the Stochastic Oscillator indicator. If we have the following data:

| Days | Close | High | Low |
| --- | --- | --- | --- |
| 1 | 100 | 110 | 90 |
| 2 | 130 | 140 | 120 |
| 3 | 140 | 160 | 120 |
| 4 | 130 | 150 | 110 |
| 5 | 120 | 140 | 100 |
| 6 | 140 | 150 | 130 |
| 7 | 160 | 170 | 150 |
| 8 | 170 | 180 | 160 |
| 9 | 155 | 170 | 150 |
| 10 | 140 | 170 | 130 |
| 11 | 160 | 180 | 155 |
| 12 | 180 | 190 | 175 |
| 13 | 190 | 220 | 190 |
| 14 | 200 | 230 | 200 |
| 15 | 210 | 215 | 205 |
| 16 | 200 | 200 | 190 |
| 17 | 190 | 195 | 180 |
| 18 | 185 | 105 | 180 |
| 19 | 195 | 210 | 185 |
| 20 | 200 | 220 | 190 |

To calculate the Stochastic Oscillator indicator we will follow the previously mentioned steps, so we do the following:

- Determine the period which will be = 14, so we need first to have 14 trading periods to start calculating the indicator.
- Determine the highest high of the last 14 periods, it will be (230) and the following picture will show us that for the first value and the rest of the data:

![14 Highest high](https://c.mql5.com/2/46/14_Highest_high.png)

- Determine the lowest low of the last 14 periods, it will be (90) and the following picture will show us that for the first value and the rest of the data:

![14 lowest low](https://c.mql5.com/2/46/14_lowest_low.png)

- Get the %K of fast stochastic = 100\*((current close - lowest low)/(highest high- lowest low)),
- Get the %D of fast stochastic = 3 moving average of %K of fast stochastic,
- The following picture will show us the results of the calculation for the available data:

![Fast - K-D](https://c.mql5.com/2/46/Fast_-_KbD.png)

- Get the %K of slow stochastic = %D of fast stochastic,
- Get the %D of slow stochastic = 3 moving average of %K of slow stochastic,
- The following picture will show us the results of the calculation for the available data:

![Slow - K-D](https://c.mql5.com/2/46/Slow_-_KdD.png)

Through the previous steps, we calculated the Stochastic Oscillator indicator manually. Fortunately, we do not need to calculate it manually every time as we have a ready-to-use built-in indicator in the MetaTrader5 trading terminal. We can simply choose it from the list of available indicators in the platform by clicking on the Insert tab from MetaTrader 5 as it is shown in the following picture:

![Stoch insert](https://c.mql5.com/2/46/Stoch_insert.png)

After choosing the "Stochastic Oscillator", the following window will be opened for the parameters of the indicator:

![ Stoch insert window](https://c.mql5.com/2/46/Stoch_insert_window.png)

Parameters of the indicator:

1. Determines the period
2. Determines the moving average period for %D fast
3. Determines the moving average period for %D slow
4. Determines the price filed to measure the relation to it
5. Determine the moving average type which will be used in the calculation
6. Determine %K's line color, line style, and line thickness
7. Determine %D's line color, line style, and line thickness

After determining parameters according to your preferences, the indicator will be attached to the chart and will appear like in the following picture:

![Stoch on the chart](https://c.mql5.com/2/46/Stoch_on_the_chart.png)

After inserting the Stochastic Oscillator, we can find according to the previous picture that we have the indicator plotted in the lower window with 2 lines (%K, %D) oscillating between zero and 100 and we can find it the same as the below:

- Stoch(14, 3,3): represents the name of the indicator and chosen parameters: 14 periods, 3 for the moving average of %D of the fast stochastic, 3 for the moving average of %D of the fast stochastic.
- 36.85: represents the %k value
- 38.28: represents the %D value

According to the calculation of the Stochastic Oscillator indicator, it will move to the direction of the market in a specific form most of the time as we knew that closing prices tend to close near to the upper or lower price range of a specific period according to market movement.

During uptrend:

An uptrend is a market direction in which prices move up because of the control of buyers or bulls. Usually prices create higher lows and higher highs. In this case, we can call this market a bullish market.

We can find during this market direction that most of the time, closing prices tend to close near the upper part of range price during that period of movement "Uptrend" and the stochastic indicator will oscillate between 50 level and 100 level. An example of this situation is shown in the following picture:

![Stochastic during uptrend](https://c.mql5.com/2/46/Stochastic_during_uptrend.png)

During downtrend:

A downtrend is a market direction in which prices move down because of the control of sellers or bears. The prices create lower highs and lower lows. In this case, we can call this market a bearish market.

During this market direction most of the time closing prices tend to close near the lower range of price movements during that "Downtrend" movement period. So, we may find that the stochastic indicator moves between 50 level and 0 levels during downward movement period. An example of this situation is shown in the following picture:

![Stochastic during downtrend](https://c.mql5.com/2/46/Stochastic_during_downtrend.png)

During sideways:

A sideways is a price movement that is a movement without a clear direction either up or down. It can be considered as a balance between buyers and sellers or bulls and bears. It is any movement except uptrend or downtrend.

During this sideways movement there is no clear movement up or down, the prices tend to close near to mid-range of prices for that period of movement "Sideways" and so, we may find that most of the time, the stochastic moves around 20 and 80 levels and the following picture is an example for that:

![Stochastic during sideways](https://c.mql5.com/2/46/Stochastic_during_sideways.png)

Now that we have analyzed how the stochastic indicator may act in different market movements. Now we need to learn how we can use it in our favor, and this is what we will learn through the following topic.

### Stochastic strategy

In this part, we will talk about how we can use this indicator through simple strategies. We can get signals from the stochastic indicator according to market trend and these strategies are uptrend strategy, downtrend strategy, and sideways strategy.

- **Strategy one: Uptrend strategy**

According to this strategy, we need to check if the %K line and %D line are below the 50 level, then, the buy signal will be generated when the %K line crosses above the %D line. We can take profit according to another effective tool like price action by searching for a lower low for example.

%K, %D < 50 --> %K > %D = buy signal

- **Strategy two: downtrend strategy**

According to this strategy, we need to check if the %K line and %D line are above the 50 level, then, the sell signal will be generated when the %K line crosses below the %D line. We can take profit according to another effective tool like price action by searching for a higher high for example.

%K, %D > 50 --> %K < %D = sell signal

- **Strategy three: sideways strategy**

  - The buy signal:

According to this strategy, we need to check if the %K line and %D line are below the 20 level, then, the buy signal will be generated when the %K line crosses above the %D line. When the %K line and %D line are above 80, then the take profit signal will be generated when the %K line crosses below the %D line.

%K, %D < 20 --> %K > %D = buy signal

%K, %D > 80 --> %K < %D = take profit

- **The sell signal**

According to this strategy, we need to check if the %K line and %D line are above the 80 level, then, the sell signal will be generated when the %K line crosses below the %D line. When the %K line and %D are below the 20 level, then, the take profit signal will be generated when the %K line crosses above the %D line.

%K, %D > 80 --> %K < %D = sell signal

%K, %D < 20 --> %K > %D = take profit

I'd like to mention here that there are many stochastic strategies that can be used, from simple to complicated strategies and this indicator can be used individually or combined with another tool in a way that can give better results but here we mention only simple strategies to understand how this indicator can be used.

### Stochastic blueprint

In this part, we will design a blueprint for previously mentioned strategies which should help us easily create a trading system. This blueprint will be a step-by-step guideline to identify what we need the program to do exactly.

- **Strategy one: uptrend strategy**

%K, %D < 50 --> %K > %D = buy signal

We first need the program to check (%K, %D) and decide if it is below or above the 50 level, then if it is below the 50 level, it will wait to do nothing till the %K line crosses above %D line to give the buy signal. If (%K, %D) is above or equal to the 50 level, the program will do nothing.

The following picture is a blueprint for this uptrend strategy:

![SSS- uptrend strategy blueprint](https://c.mql5.com/2/46/SSS-_uptrend_strategy_blueprint.png)

- **Strategy two: downtrend strategy**

%K, %D > 50 --> %K < %D = sell signal

We first need the program to check (%K, %D) and decide if it is above or below the 50 level, then if it is above the 50 level, it will wait to do nothing till the %K line crosses below %D line to give the sell signal. If (%K, %D) is below or equal to the 50 level, the program will do nothing.

The following picture is a blueprint of this strategy:

![SSS- downtrend strategy blueprint](https://c.mql5.com/2/46/SSS-_downtrend_strategy_blueprint.png)

- **Strategy three: sideways strategy**

The buy signal:

%K, %D < 20 --> %K > %D = buy signal

%K, %D > 80 --> %K < %D = take profit

We first need the program to check (%K, %D) and decide if it is below or above the 20 level, then if it is below the 20 level, it will wait to do nothing till the %K line crosses above %D line to give the buy signal. If (%K, %D) is above or equal to the 20 level, the program will do nothing. Then, the program will check (%K, %D) and decide if it is above or below the 80 level, then if it is above the 80 level, it will wait to do nothing till the %K line crosses below the %D line to give the take profit signal. If (%K, %D) is below or equal to the 80 level, the program will do nothing.

The following picture is a blueprint for this strategy:

![SSS- sideways strategy - buy blueprint](https://c.mql5.com/2/46/SSS-_sideways_strategy_-_buy_blueprint.png)

**The sell signal:**

%K, %D > 80 --> %K < %D = sell signal

%K, %D < 20 --> %K > %D = take profit

We first need the program to check (%K, %D) and decide if it is above or below the 80 level, then if it is above the 80 level, it will wait to do nothing till the %K line crosses below %D line to give the sell signal. If (%K, %D) is below or equal to the 80 level, the program will do nothing. Then, the program will check (%K, %D) and decide if it is below or above the 20 level, then if it is below the 20 level, it will wait to do nothing till the %K line crosses above the %D line to give the take profit signal. If (%K, %D) is above or equal to the 20 level, the program will do nothing.

The following picture is a blueprint for this sell signal of sideways strategy:

![SSS- sideways strategy - sell blueprint](https://c.mql5.com/2/46/SSS-_sideways_strategy_-_sell_blueprint.png)

So far, we have considered simple strategies that can be used by the stochastic indicator and we have designed blueprints for these strategies to help us to create a trading system. In the next part, we will learn how to design this trading system.

### Stochastic trading system

Now, we will start an interesting part in this article to learn how to code a trading system that works by these strategies. Programming is a great tool that helps us to trade effectively as it helps us to execute our plan and strategy in a disciplined way without involving emotions in our trading decision in addition to the accuracy in execution but the most important thing is to test any strategy before executing it. Let us write our trading system according to the earlier discussed strategies...

First, we need to write a code for a program that provides values of stochastic lines on the chart, and through writing this code we will learn line by line what we are writing to be able after this step to understand and create the trading system.

First steps, we need to create arrays for the %K line and %D line and we will use "double" which is presenting values with a fractional part:

```
double Karray[];
double Darray[];
```

Then, we need to sort these created arrays from the current data and we will use "ArraySetAsSerious" function:

```
ArraySetAsSeries(Karray, true);
ArraySetAsSeries(Darray, true);
```

Then, we need to define the "Stochastic" indicator and we will use "iStochastic" function after creating a variable with "StochDef" to be equal to the "Stochastic" definition:

```
int StochDef = iStochastic(_Symbol,_Period,14,3,3,MODE_SMA,STO_LOWHIGH);
```

Then, we need to fill array with prices data and we will use "CopyBuffer" function:

```
CopyBuffer(StochDef,0,0,3,Karray);
CopyBuffer(StochDef,1,0,3,Darray);
```

Then, we will calculate the value of the %K line and %D line of current data and we will use "float" to decrease the size of fractions and to be approximated:

```
float KValue = Karray[0];
float DValue = Darray[0];
```

The last code line will be to let the program present the %K line value and the %D line value on the chart in two lines and we will use the "Comment" function:

```
Comment("%K value is ", KValue,"\n""%D Value is ",DValue);
```

And the following is the same as the previous code but in one block to be able to copy it in one step if you need that:

```
//+------------------------------------------------------------------+
//|                       Simple Stochastic System - Lines Value.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
//creating arrays for %K line and %D line
   double Karray[];
   double Darray[];

//sorting arrays from the current data
   ArraySetAsSeries(Karray, true);
   ArraySetAsSeries(Darray, true);

//defining the stochastic indicator
   int StochDef = iStochastic(_Symbol,_Period,14,3,3,MODE_SMA,STO_LOWHIGH);

//filling arrays with price data
   CopyBuffer(StochDef,0,0,3,Karray);
   CopyBuffer(StochDef,1,0,3,Darray);

//calculating value of %K and %D line of cuurent data
   float KValue = Karray[0];
   float DValue = Darray[0];

//commenting calcualted values on the chart
   Comment("%K value is ",KValue,"\n"
   "%D Value is ",DValue);

  }
//+------------------------------------------------------------------+
```

So far, we wrote code of a program that can present current values of stochastic indicator lines on the chart and now we need to execute this program on our trading terminal which is Meta Trader5 as we can find it now existing in the navigator window after compiling:

![Nav 1](https://c.mql5.com/2/46/Nav_1.png)

After dragging and dropping the file of Simple Stochastic System - Lines Value, the following window will open:

![SSS-lines value window](https://c.mql5.com/2/46/SSS-lines_value_window.png)

After pressing OK, the program or the (Expert Advisor) will be attached to the chart the same as the below picture:

![SSS-lines value attached](https://c.mql5.com/2/46/SSS-lines_value_attached.png)

Then, we can see that the program generates values of stochastic lines the same as in the below picture:

![SSS- lines value](https://c.mql5.com/2/46/SSS-_lines_value.png)

Now, we will start to write the code to create a trading system for our strategies, and what we need to do here is to create a trading system that is able to present the suitable signal as a comment on the chart according to the used strategy. Let us start to do that...

- **Strategy one: uptrend strategy**

%K, %D < 50 --> %K > %D = buy signal

The following code is for creating a trading system to execute what we need according to this uptrend strategy:

```
//+------------------------------------------------------------------+
//|                  Simple Stochastic System - Uptrend Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   string signal="";

   double Karray[];
   double Darray[];

   ArraySetAsSeries(Karray, true);
   ArraySetAsSeries(Darray, true);

   int StochDef = iStochastic(_Symbol,_Period,14,3,3,MODE_SMA,STO_LOWHIGH);

   CopyBuffer(StochDef,0,0,3,Karray);
   CopyBuffer(StochDef,1,0,3,Darray);

   double KValue0 = Karray[0];
   double DValue0 = Darray[0];

   double KValue1 = Karray[1];
   double DValue1 = Darray[1];

   if (KValue0<50&&DValue0<50)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "BUY";
      }

   Comment("SSS - Uptrend Strategy - Signal is ",signal);

  }
//+------------------------------------------------------------------+
```

As you can see that the differences here is:

Created a variable for the "signal" with an empty assignment as we will calculate it after that:

```
string signal="";
```

Created two variables for %K values (0,1) and two variables for %D values (0,1) for current data and data before the current:

```
   double KValue0 = Karray[0];
   double DValue0 = Darray[0];

   double KValue1 = Karray[1];
   double DValue1 = Darray[1];
```

To evaluate the crossover between %K and %D as we need the crossover to up, in other words, we need %K to break %D to up to move above it after this crossover and this is what we can find in the line code of "if":

```
   if (KValue0<50&&DValue0<50)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "BUY";
      }

```

After evaluation of the crossover, if it is true, we need a signal for "BUY" and this will be the assignment for "signal". Then we need a comment with this signal on the chart and it will be according to the following line of code:

```
Comment("SSS - Uptrend Strategy - Signal is ",signal);
```

So for, we created the code of the program and after compiling we can find it in the navigator window:

![Nav 2](https://c.mql5.com/2/46/Nav_2.png)

After dragging and dropping this file on the chart, the following window will appear:

![ SSS-Uptrend window](https://c.mql5.com/2/46/SSS-Uptrend_window.png)

After pressing OK, it will be attached:

![SSS-Uptrend attached](https://c.mql5.com/2/46/SSS-Uptrend_attached.png)

The signal will appear the same as the following example while testing:

![SSS - uptrend - buy](https://c.mql5.com/2/46/SSS_-_uptrend_-_buy.png)

- **Strategy two: downtrend strategy**

%K, %D > 50 --> %K < %D = sell signal

The following is for creating a program to execute this downtrend strategy:

```
//+------------------------------------------------------------------+
//|                Simple Stochastic System - Downtrend Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   string signal="";

   double Karray[];
   double Darray[];

   ArraySetAsSeries(Karray, true);
   ArraySetAsSeries(Darray, true);

   int StochDef = iStochastic(_Symbol,_Period,5,3,3,MODE_SMA,STO_LOWHIGH);

   CopyBuffer(StochDef,0,0,3,Karray);
   CopyBuffer(StochDef,1,0,3,Darray);

   double KValue0 = Karray[0];
   double DValue0 = Darray[0];

   double KValue1 = Karray[1];
   double DValue1 = Darray[1];

   if (KValue0>50&&DValue0>50)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "SELL";
      }

   Comment("SSS - Downtrend Strategy - Signal is ",signal);

  }
//+------------------------------------------------------------------+
```

The differences are:

- Conditions of the crossover

```
   if (KValue0>50&&DValue0>50)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "SELL";
      }

```

- The comment according to this strategy:

```
   Comment("SSS - Downtrend Strategy - Signal is ",signal);
```

After compiling, we can find the file of this program in the navigator window:

![Nav 3](https://c.mql5.com/2/46/Nav_3.png)

Drag and drop this file on the chart and the following window will open:

![SSS-Downtrend window](https://c.mql5.com/2/46/SSS-Downtrend_window.png)

After pressing OK, the file will be attached to the chart:

![SSS-Downtrend attached](https://c.mql5.com/2/46/SSS-Downtrend_attached.png)

The signal will appear as a comment on the chart according to the conditions of this strategy and the following picture is an example of that:

![SSS - downtrend - sel](https://c.mql5.com/2/46/SSS_-_downtrend_-_sell.png)

- **Strategy three: sideways strategy**

  - **Buy signal:**

%K, %D < 20 --> %K > %D = buy signal

%K, %D > 80 --> %K < %D = take profit signal

The following code is for creating a program that is able to execute this sideways strategy:

```
//+------------------------------------------------------------------+
//|           Simple Stochastic System - Sideways - Buy Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   string signal="";

   double Karray[];
   double Darray[];

   ArraySetAsSeries(Karray, true);
   ArraySetAsSeries(Darray, true);

   int StochDef = iStochastic(_Symbol,_Period,5,3,3,MODE_SMA,STO_LOWHIGH);

   CopyBuffer(StochDef,0,0,3,Karray);
   CopyBuffer(StochDef,1,0,3,Darray);

   double KValue0 = Karray[0];
   double DValue0 = Darray[0];

   double KValue1 = Karray[1];
   double DValue1 = Darray[1];

   if (KValue0<20&&DValue0<20)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "BUY";
      }

   Comment("SSS - Sideways - Buy Strategy - Signal is ",signal);

    if (KValue0>80&&DValue0>80)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "Take Profit";
      }

   Comment("SSS - Sideways - Buy Strategy - Signal is ",signal);

  }
//+------------------------------------------------------------------+
```

The differences are:

- Conditions of crossover according to the buy signal of sideways strategy:

```
   if (KValue0<20&&DValue0<20)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "BUY";
      }

```

- The comment according to the buy signal of sideways strategy:

```
   Comment("SSS - Sideways - Buy Strategy - Signal is ",signal);
```

- Conditions of take profit signal of sideways strategy:

```
    if (KValue0>80&&DValue0>80)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "Take Profit";
      }

```

- The comment according to take profit signal of sideways strategy:

```
   Comment("SSS - Sideways - Buy Strategy - Signal is ",signal);
```

After compiling, the file of this trading system will appear in the Navigator window:

![Nav 4](https://c.mql5.com/2/46/Nav_4.png)

Drag and drop the file on the chart, the following window will be opened:

![SSS-Sideways - buy - window](https://c.mql5.com/2/46/SSS-Sideways_-_buy_-_window.png)

After pressing OK, the program will be attached to the chart:

![SSS-Sideways - buy - attached](https://c.mql5.com/2/46/SSS-Sideways_-_buy_-_attached.png)

The signal will be shown as a comment on the chart according to this strategy. The following picture shows an example of such a signal:

![SSS - Sideways - buy - buy](https://c.mql5.com/2/46/SSS_-_Sideways_-_buy_-_buy.png)

The "take profit" signal example:

![SSS - Sideways - buy - TP](https://c.mql5.com/2/46/SSS_-_Sideways_-_buy_-_TP.png)

- **Sideways strategy: sell signal**

%K, %D > 80 --> %K < %D = sell signal

%K, %D < 20 --> %K > %D = take profit signal

The following is for how to write the code of a program that is execute this strategy:

```
//+------------------------------------------------------------------+
//|          Simple Stochastic System - Sideways - Sell Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   string signal="";

   double Karray[];
   double Darray[];

   ArraySetAsSeries(Karray, true);
   ArraySetAsSeries(Darray, true);

   int StochDef = iStochastic(_Symbol,_Period,5,3,3,MODE_SMA,STO_LOWHIGH);

   CopyBuffer(StochDef,0,0,3,Karray);
   CopyBuffer(StochDef,1,0,3,Darray);

   double KValue0 = Karray[0];
   double DValue0 = Darray[0];

   double KValue1 = Karray[1];
   double DValue1 = Darray[1];

    if (KValue0>80&&DValue0>80)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "SELL";
      }

   Comment("SSS - Sideways - Sell Strategy - Signal is ",signal);

   if (KValue0<20&&DValue0<20)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "TAKE PROFIT";
      }
   Comment("SSS - Sideways - Sell Strategy - Signal is ",signal);
  }
//+------------------------------------------------------------------+
```

The differences are:

- Conditions of crossover according to the sell signal of sideways strategy:

```
    if (KValue0>80&&DValue0>80)
      if ((KValue0<DValue0) && (KValue1>DValue1))
      {
         signal = "SELL";
      }
```

- The comment according to the sell signal of sideways strategy:

```
   Comment("SSS - Sideways - Sell Strategy - Signal is ",signal);
```

- Conditions of take profit signal of sideways strategy:

```
   if (KValue0<20&&DValue0<20)
      if ((KValue0>DValue0) && (KValue1<DValue1))
      {
         signal = "TAKE PROFIT";
      }
```

- The comment according to the take profit signal of sideways strategy:

```
   Comment("SSS - Sideways - Sell Strategy - Signal is ",signal);
```

After compiling, the file of this trading system will appear in the Navigator window:

![Nav 5](https://c.mql5.com/2/46/Nav_5.png)

Drag and drop it on the chart and the following window will open:

![SSS-Sideways - sell - window](https://c.mql5.com/2/46/SSS-Sideways_-_sell_-_window.png)

After pressing OK, the program will be attached to the chart to be ready to generate signals:

![SSS-Sideways - sell - attached](https://c.mql5.com/2/46/SSS-Sideways_-_sell_-_attached.png)

The following picture is an example of a generated sell signal according to this strategy:

![SSS - Sideways - sell - sell](https://c.mql5.com/2/46/SSS_-_Sideways_-_sell_-_sell.png)

The following picture is an example of a generated take profit signal according to this strategy:

![SSS - Sideways - sell - TP](https://c.mql5.com/2/46/SSS_-_Sideways_-_sell_-_TP.png)

### Conclusion

At the end of this article, we can say that we have learnt the basics of how to create a simple trading system using the "Stochastic Oscillator" technical indicator which is one of the most popular indicators in the world of trading. I hope you have understood what the "Stochastic Oscillator" indicator is, what it measures, and how to calculate it — we have considered it in detail. I hope you have found out something new about how to use the Stochastic indicator by learning some simple strategies that can be used. Also the article presented a blueprint for these simple strategies based on which it is possible to create a trading system easily and smoothly. Also, we have considered how to design a trading system based on these mentioned simple strategies.

I hope that you found this article useful for you and your trading, and that it will assist in finding new ideas by giving you insights into one of the most popular and useful indicators — the "Stochastic Oscillator" in the world of trading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10692.zip "Download all attachments in the single ZIP archive")

[Simple\_Stochastic\_System\_-\_Lines\_Value.mq5](https://www.mql5.com/en/articles/download/10692/simple_stochastic_system_-_lines_value.mq5 "Download Simple_Stochastic_System_-_Lines_Value.mq5")(1.31 KB)

[Simple\_Stochastic\_System\_-\_Uptrend\_Strategy.mq5](https://www.mql5.com/en/articles/download/10692/simple_stochastic_system_-_uptrend_strategy.mq5 "Download Simple_Stochastic_System_-_Uptrend_Strategy.mq5")(1.29 KB)

[Simple\_Stochastic\_System\_-\_Downtrend\_Strategy.mq5](https://www.mql5.com/en/articles/download/10692/simple_stochastic_system_-_downtrend_strategy.mq5 "Download Simple_Stochastic_System_-_Downtrend_Strategy.mq5")(1.29 KB)

[Simple\_Stochastic\_System\_-\_Sideways\_-\_Buy\_Strategy.mq5](https://www.mql5.com/en/articles/download/10692/simple_stochastic_system_-_sideways_-_buy_strategy.mq5 "Download Simple_Stochastic_System_-_Sideways_-_Buy_Strategy.mq5")(1.5 KB)

[Simple\_Stochastic\_System\_-\_Sideways\_-\_Sell\_Strategy.mq5](https://www.mql5.com/en/articles/download/10692/simple_stochastic_system_-_sideways_-_sell_strategy.mq5 "Download Simple_Stochastic_System_-_Sideways_-_Sell_Strategy.mq5")(1.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)
- [How to build and optimize a volume-based trading system (Chaikin Money Flow - CMF)](https://www.mql5.com/en/articles/16469)
- [MQL5 Integration: Python](https://www.mql5.com/en/articles/14135)
- [How to build and optimize a volatility-based trading system (Chaikin Volatility - CHV)](https://www.mql5.com/en/articles/14775)
- [Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)
- [Building and testing Keltner Channel trading systems](https://www.mql5.com/en/articles/14169)
- [Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/399644)**
(18)


![58051101 Jaurès ADOMOU](https://c.mql5.com/avatar/avatar_na2.png)

**[58051101 Jaurès ADOMOU](https://www.mql5.com/en/users/58051101)**
\|
26 Nov 2022 at 16:10

**Emmanuel Tiousse [#](https://www.mql5.com/fr/forum/431669#comment_42362959):**

I'd like to know if I don't have a visa card, how can I make the withdrawal after I've won?

Write to me on the watsapp number +229 57442981 so that I can show you how it's done.


![JackHuynh91](https://c.mql5.com/avatar/avatar_na2.png)

**[JackHuynh91](https://www.mql5.com/en/users/jackhuynh91)**
\|
10 Aug 2023 at 04:13

thank u for sharing strategy. It's really good. Can i have a question ? In your article. You use stoch 14 3 3 for Uptrend Strategy. But Downtrend Strategy and sideway buy [sell signal](https://www.mql5.com/en/articles/591 "Article: How to Become a Signal Provider for MetaTrader 4 and MetaTrader 5 ") is stoch 5 3 3. Is that correct?


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
10 Aug 2023 at 07:58

**JackHuynh91 [#](https://www.mql5.com/en/forum/399644#comment_48661609):**

thank u for sharing strategy. It's really good. Can i have a question ? In your article. You use stoch 14 3 3 for Uptrend Strategy. But Downtrend Strategy and sideway buy [sell signal](https://www.mql5.com/en/articles/591 "Article: How to Become a Signal Provider for MetaTrader 4 and MetaTrader 5 ") is stoch 5 3 3. Is that correct?

Thanks for your kind comment. it is correct.

![eudesdionatas](https://c.mql5.com/avatar/2023/10/6526c5b3-e2fe.png)

**[eudesdionatas](https://www.mql5.com/en/users/eudesdionatas)**
\|
23 Nov 2023 at 02:01

Congratulations on the article, [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud) . Very good!


![Quentier Marc](https://c.mql5.com/avatar/avatar_na2.png)

**[Quentier Marc](https://www.mql5.com/en/users/marc_q)**
\|
6 Dec 2024 at 06:05

Many thanks for all your articles which are very helpful to learn developing EA. Really appreciated!


![DirectX Tutorial (Part I): Drawing the first triangle](https://c.mql5.com/2/45/ramka.png)[DirectX Tutorial (Part I): Drawing the first triangle](https://www.mql5.com/en/articles/10425)

It is an introductory article on DirectX, which describes specifics of operation with the API. It should help to understand the order in which its components are initialized. The article contains an example of how to write an MQL5 script which renders a triangle using DirectX.

![Using the CCanvas class in MQL applications](https://c.mql5.com/2/45/canvas-logo-3.png)[Using the CCanvas class in MQL applications](https://www.mql5.com/en/articles/10361)

The article considers the use of the CCanvas class in MQL applications. The theory is accompanied by detailed explanations and examples for thorough understanding of CCanvas basics.

![Graphics in DoEasy library (Part 97): Independent handling of form object movement](https://c.mql5.com/2/45/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 97): Independent handling of form object movement](https://www.mql5.com/en/articles/10482)

In this article, I will consider the implementation of the independent dragging of any form objects using a mouse. Besides, I will complement the library by error messages and new deal properties previously implemented into the terminal and MQL5.

![Learn how to design a trading system by MACD](https://c.mql5.com/2/46/why-and-how__1.png)[Learn how to design a trading system by MACD](https://www.mql5.com/en/articles/10674)

In this article, we will learn a new tool from our series: we will learn how to design a trading system based on one of the most popular technical indicators Moving Average Convergence Divergence (MACD).

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/10692&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069266474373153350)

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