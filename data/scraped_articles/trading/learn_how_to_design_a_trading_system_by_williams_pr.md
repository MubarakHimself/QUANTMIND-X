---
title: Learn how to design a trading system by Williams PR
url: https://www.mql5.com/en/articles/11142
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:12:25.241211
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11142&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069233815441834454)

MetaTrader 5 / Trading


### Introduction

In this new article from our series about how to design a trading system based on the most popular technical indicators, we will learn how to design different simple trading systems based on a new technical indicator which is the Williams Percent Range (WPR). In addition to understanding the Williams Percent Range technical indicator in detail to learn how we can use it effectively based on the main idea behind it. I believe that If we can understand the root of things, we will not only be able to use it effectively but also we can find new insights or ideas around the main concept or any related tool which can be a reason to get exceptional and better results from trading. So, I have an approach to try learning and teaching the root of things to deepen our understanding more and more.

We will learn that through many topics in this article the same as the following:

1. [Williams' %R definition](https://www.mql5.com/en/articles/11142#definition)
2. [Williams' %R strategy](https://www.mql5.com/en/articles/11142#strategy)
3. [Williams' %R strategy blueprint](https://www.mql5.com/en/articles/11142#blueprint)
4. [Williams' %R trading system](https://www.mql5.com/en/articles/11142#system)
5. [Conclusion](https://www.mql5.com/en/articles/11142#conclusion)

Through the first topic which is Williams' %R definition, we will learn what is Williams' %R, what it measures, and how we can calculate it manually. Then we will learn how can we use this indicator properly through simple strategies based on the main concept behind this indicator, we will learn that through the topic of Williams' %R strategy. We will design a step-by-step blueprint for each mentioned strategy to help us to create the main task in this article which is creating a trading system by this indicator. This blueprint will be done through the topic of Williams' %R strategy blueprint. Then, we will come to the most interesting topic in this article is to create a trading system based on mentioned strategies through the topic of Williams' %R trading system.

After these mentioned topics, I think that we will learn the fundamentals of this technical indicator in addition to learning how we create a simple trading system based on it to be able to use this trading system in the MetaTrader 5 to generate signals automatically. I encourage you to apply everything by yourself as this will be very helpful to understand deeply. I confirm also to test every signal strategy before using it on your real account to make sure that it will be useful for you.

We will use the MetaTrader 5 trading terminal and the built-in MetaEditor to write our MQL5 codes to create our trading system in a form of expert advisors. If you do not know how to get the MetaTrader 5 or how to write your MQL5 codes, you can read [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) topic from a previous article to know more about that.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

Now, let us start learning a new tool that may be a transition point in our trading journey to get better results.

### Williams' %R definition

Same as I mentioned in the introduction we will learn about the Williams' %R indicator in detail. The Williams' %R (WPR) is a momentum indicator that is developed by Larry Williams. We can say that WPR is the opposite of the Stochastic Oscillator as the difference between them is how they are scaled, we can say also that the WPR can be used the same as the Stochastic also. If you want to learn more about the Stochastic indicator, you can read my previous article about [Learn how to design a trading system by Stochastic](https://www.mql5.com/en/articles/10692). The WPR measures the relationship between the closing price and the high-low range. The WPR indicator also oscillates between 0 and -100. Because the WPR is a momentum indicator, it may be used to measure the strength of the trend also. If you need to know more about the trend and types of trends, you can read the [trend definition](https://www.mql5.com/en/articles/10715#trend) topic from a previous article as it may be helpful to understand that. The indicator can be meaningful by watching important levels of -80, -20, and the mid-point -50.

It is important to try to combine another technical tool with this indicator for example price action tools, chart patterns, volume, and moving average like what we will do through this article on the strategies topic as this will be very useful as you will be able to clarify more perspective in the instrument that you study and this will be useful to take the suitable decisions.

Now, we need to know how we can calculate the WPR indicator. Simply, we can calculate it by the following steps:

1. Calculate the highest high = the highest value of highs in the calculated period.
2. Calculate the lowest low = the lowest value of low in the calculated period.
3. WPR = (highest high - close)/(highest high - lowest low) \* -100

For more understanding we need to see an example with real values to apply that and calculate the WPR, So, if we have the following data for an instrument.

| Period | High | Low | Close |
| --- | --- | --- | --- |
| 1 | 1.078 | 1.0678 | 1.0733 |
| 2 | 1.0788 | 1.0726 | 1.0777 |
| 3 | 1.0766 | 1.0697 | 1.0727 |
| 4 | 1.0733 | 1.0662 | 1.0724 |
| 5 | 1.074 | 1.0642 | 1.068 |
| 6 | 1.0749 | 1.0661 | 1.0734 |
| 7 | 1.0698 | 1.0558 | 1.0689 |
| 8 | 1.0599 | 1.0532 | 1.056 |
| 9 | 1.0608 | 1.046 | 1.0586 |
| 10 | 1.0565 | 1.046 | 1.0466 |
| 11 | 1.0556 | 1.0429 | 1.0547 |
| 12 | 1.0444 | 1.0388 | 1.0431 |
| 13 | 1.0421 | 1.035 | 1.0411 |
| 14 | 1.053 | 1.0353 | 1.0379 |
| 15 | 1.0577 | 1.0502 | 1.0511 |
| 16 | 1.0586 | 1.0525 | 1.0527 |
| 17 | 1.0594 | 1.0495 | 1.0555 |
| 18 | 1.0601 | 1.0482 | 1.0551 |
| 19 | 1.0642 | 1.0493 | 1.054 |
| 20 | 1.0632 | 1.0505 | 1.0621 |
| 21 | 1.0579 | 1.0491 | 1.052 |
| 22 | 1.0567 | 1.0491 | 1.051 |
| 23 | 1.0602 | 1.0381 | 1.0547 |
| 24 | 1.0509 | 1.0359 | 1.0443 |
| 25 | 1.0486 | 1.0396 | 1.0414 |
| 26 | 1.052 | 1.04 | 1.0408 |
| 27 | 1.0644 | 1.0505 | 1.0515 |
| 28 | 1.0775 | 1.0611 | 1.0614 |
| 29 | 1.0749 | 1.0671 | 1.0714 |
| 30 | 1.0715 | 1.0652 | 1.0699 |

Now, we will go through calculation steps to get the WPR the same as the following. Note that, we will use the default settings of the period is 14.

1- Getting highest high.

![HH values](https://c.mql5.com/2/47/Screen_Shot_2022-06-29_at_11.56.49_PM.png)

2- Getting lowest low.

![LL values](https://c.mql5.com/2/47/Screen_Shot_2022-06-29_at_11.56.57_PM.png)

3- Getting WPR.

![](https://c.mql5.com/2/47/Screen_Shot_2022-06-29_at_11.57.04_PM.png)

Now, we calculated the WPR values. It will be a line that oscillates between 0 and -100 to measure the momentum the same as we mentioned. Nowadays, we do not need to calculate it manually as we can find it ready in the MetaTrader 5 trading terminal and all we need to do is to choose it among the available indicators the same as the following steps.

![W_R insert](https://c.mql5.com/2/47/W_R_insert.png)

After that, we will find its window to determine its parameters the same as the following:

![ W_R parameters](https://c.mql5.com/2/47/W_R_parameters.png)

1- To set the desired period.

2- To determine the color of the WPR line.

3- To determine the style of the WPR line.

4- To determine the thickness of the WPR line.

We can also control levels of the indicator through the Levels tab the same as the following:

![W_R lvls](https://c.mql5.com/2/47/W_R_lvls.png)

We can see in the previous picture we have two levels -80 and -20 which are important levels to record high and low readings. Also, we can add other levels as per what we may find that will be useful like adding a -50 level for example by pressing Add and then determining the level to appear on the indicator window on the chart.

After determining all desired parameters and pressing "OK", the indicator will be attached to the chart the same as the following.

![ W_R attached](https://c.mql5.com/2/47/W_R_attached.png)

As we can see in the previous picture, the black line in the lower window in the chart is oscillating between 0 and -100 to measure the momentum. Also, the mid-point of -50 is an important level and it has meaningful insights into the price movement as when the WPR crosses above the -50 level, this means that the price trades in the upper half part of the high-low range during the calculated period and vice versa when the WPR crosses below -50 level, this means that the price trades in the lower half part of the high-low range in the calculated period. Otherwise, when we see the indicator below the area of -80 this means a low reading which indicates that the price trades near its low but when we see the indicator above the area of -20 level this means a high reading which indicates that the price trades near its high. these levels of -80 and -20 are very important areas because they indicate that prices reach overbought and oversold areas the same as we will see later.

### Williams' %R strategy

In this part, we will learn how we can use the Williams' %R through simple strategies based on the main concept of the indicator. We will learn three different strategies which can be used based on the idea behind the Williams %R indicator. The first strategy, we will call it Williams %R - OB and OS, the second one, we can call it Williams %R - crossover, and the third one we will call it Williams %R - MA. in the following lines we will identify how these strategies can be used.

- Strategy one: Williams %R - OB and OS:

According to this strategy, we need to be notified when the instrument is in the overbought area or the oversold area based on a specific condition. When Williams %R value is lower than the -80 level this will be an oversold signal. When the Williams %R value is greater than the -20 level this will be an overbought signal.

This strategy will be helpful to notify us when prices reach overbought or oversold areas and this will help us to expect the upcoming movement by reading the status of momentum.

WPR value < -80 --> oversold

WPR value > - 20 --> overbought

- Strategy two: Williams %R - crossover:

According to this strategy, we need to get a signal when there is a buy or sell signal based on the crossover between the current Williams %R, the previous Williams %R (WPR), and the -50 level. When the previous WPR is lower than - 50 and the current WPR is greater than -50, this will be a buy signal. When the previous WPR is greater than -50 and the current WPR is lower than -50, this will be a sell signal.

This strategy will be useful as it will generate buy and sell signals based on the crossover with a very important level which is -50, and it will be more effective if we combine another technical tool to confirm these signals.

Prev. WPR < -50 and current WPR > -50 --> buy signal

Prev. WPR > -50 and current WPR < -50 --> sell signal

- Strategy three: Williams %R - MA:

According to this strategy, we need to get a notification when we have a buy or sell signal also but it will be based on another condition as we need to get a buy signal when the ask is greater than the moving average value and the Williams %R value is greater than -50 level. We need to get a sell signal when the bid is lower than the moving average value and the Williams %R value is lower than the moving average value.

This strategy allows us to get more defined buy and sell signals based on its conditions because here we combined another technical tool which is the moving average to confirm the current signal and filter false breakouts.

Ask > MA value and WPR > -50 --> buy signal

Bid < MA value and WPR < -50 --> sell signal

### Williams' %R strategy blueprint

In this topic, we will design a step-by-step blueprint to create a trading system for every mentioned strategy. I consider this step a very important step in any trading system creation process as it will help us to understand what we need to do through organized steps.

- Strategy one: Williams %R - OB and OS:

Based on this strategy, we will design a step-by-step blueprint to create a trading system as we need the program or expert to check three values every tick and make a comparison to determine the position of one of these values compared to others and these three values are the current WPR , -80, and -20 level. The expert will decide after the previous comparison if the current WPR value is lower than the -80 level we need the expert to generate a comment on the chart with "Over Sold". In the other scenario, when the current WPR value is greater than the -20 level, we need the expert to generate a different comment on the chart with "Over Bought".

The following is the blueprint for this strategy:

![ Williams_R - OB _ OS Blueprint](https://c.mql5.com/2/48/Williams_R_-_OB___OS_Blueprint.png)

- Strategy two: Williams %R - crossover:

Based on this strategy, we need the program to generate buy or sell signals based on checking three values the previous WPR, current WPR, and -50. This signal will be generated based on a comparison after every tick checking by the expert for these three values to determine the position of the previous and current WPR value from the -50 level. We the expert according to what we mentioned to generate a buy signal, the Williams %R value, and the Williams %R previous value as a comment on the chart if the previous WPR is lower than the -50 and the current WPR value is greater than the -50 level. The other sell signal, the Williams %R value, and the previous Williams %R value as a comment on the chart if the previous WPR is greater than -50 and the current WPR is lower than -50.

The following is the blueprint for this strategy as a diagram.

![ Williams_R - Crossover blueprint](https://c.mql5.com/2/48/Williams_R_-_Crossover_blueprint.png)

- Strategy three: Williams %R and MA:

We need to create a trading system according to this strategy to generate buy or sell signals based on other conditions or other measurements. As we need to create a simple expert advisor that can be able to generate these signals based on continuously checking for the Ask, Bid, WPR value, moving average, and -50 level. When the expert finds that the Ask price is greater than the moving average and the WPR value is greater than the -50 level, we need the expert to generate a buy signal, Williams %R value, and the exponential moving average value as a comment on the chart. The other case to generate a sell signal, Williams %R value, and the exponential moving average value as a comment also on the chart when the Bid price is lower than the moving average and the WPR is lower than the -50 level.

The following is for the diagram blueprint to create this kind of trading system.

![ Williams_R _ MA blueprint](https://c.mql5.com/2/48/Williams_R___MA_blueprint.png)

### Williams' %R trading system

In this part, we will learn how to create a trading system for each mentioned strategy which is the most interesting part of this article. First, we will create a simple trading system that is generating a comment on the chart by the Williams' %R value to use it as a base for all strategies.

The following is for how to write the code of this trading system:

- Create an array by using the "double" function which represents values with fractions.

```
double WPArray[];
```

- Sorting this created array from current data by using the "ArraySetAsSeries" function which returns true or false as it returns a boolean value. Parameters of this function are an array\[\] and flag.

```
ArraySetAsSeries(WPArray,true);
```

- Defining the Williams' %R properties by using the "iWPR" after creating an integer variable for WPDef of Williams' %R definition. the "iWPR" function returns the handle of the Larry Williams' Percent Range indicator. Parameters are symbol name, period, and averaging period.

  - We will use (\_Symbol) to be applied on the currently used symbol and (\_Period) to be applied on the currently used timeframe.

```
int WPDef = iWPR(_Symbol,_Period,14);
```

- Copying price data to the created array by using the "CopyBuffer" function which returns the copied data count or -1 if there is an error. Parameters are indicator handle, indicator buffer time, start position, amount to copy, and target array to copy.

```
CopyBuffer(WPDef,0,0,3,WPArray);
```

- Getting the Williams' %R value of the current data by using the "NormalizeDouble" function after creating a double variable of WPValue for the current Williams' %R value. Parameters of the "NormalizeDouble" function are the normalized number and the number of digits after the decimal point.

```
double WPVal = NormalizeDouble(WPArray[0],2);
```

- Using the "Comment" function to generate the Williams' %R value as a comment on the chart.

```
Comment("Williams' %R Value is",WPVal);
```

To see the full code lines consequently of the previous trading system, we can see that through the following full code:

```
//+------------------------------------------------------------------+
//|                                            Simple Williams%R.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double WPArray[];

   ArraySetAsSeries(WPArray,true);

   int WPDef = iWPR(_Symbol,_Period,14);

   CopyBuffer(WPDef,0,0,3,WPArray);

   double WPVal = NormalizeDouble(WPArray[0],2);

   Comment("Williams' %R Value is",WPVal);

  }
//+------------------------------------------------------------------+
```

After compiling this code we can find the expert in the "Expert Advisors" folder file in the Navigator window the same as the following picture.

![WPR nav1](https://c.mql5.com/2/47/WPR_nav1.png)

By dragging and dropping on the desired chart to execute it, the window of this trading system will appear as follows.

![Simple W_R window](https://c.mql5.com/2/47/Simple_W_R_window.png)

After pressing "OK", it will be attached to the chart as follows.

![ Simple W_R attached](https://c.mql5.com/2/47/Simple_W_R_attached.png)

The following is an example of generated signals of this trading system from testing.

![ Simple W_R signal](https://c.mql5.com/2/47/Simple_W_R_signal.png)

As we can see in the previous example, the expert generated a comment signal on the upper left corner of the chart with the current WPR value. To make sure that generated signal is the same as the built-in indicator in MetaTrader 5, we can see that in the following picture.

![Simple W_R - same signal](https://c.mql5.com/2/47/Simple_W_R_-_same_signal.png)

As we can see in the previous picture that the expert is attached and the indicator is inserted and both of them generated the same signal or value. In the upper right corner when can the Simple Williams %R expert is attached and in the upper left corner, we can find its generated signal as a comment which is equal to -78.15 which is the same as the value of the built-in indicator which is inserted to the chart in the lower window and appear the same value above the WPR line on the left side.

- Strategy one: Williams%R - OB & OS:

Based on this strategy, we need to create a trading system that can be able to generate signals for us to inform us automatically when the price reached overbought or oversold areas. The following is the full code to create a trading system for this strategy:

```
//+------------------------------------------------------------------+
//|                                         Williams%R - OB & OS.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double WPArray[];

   ArraySetAsSeries(WPArray,true);

   int WPDef = iWPR(_Symbol,_Period,14);

   CopyBuffer(WPDef,0,0,3,WPArray);

   double WPVal = NormalizeDouble(WPArray[0],2);

   if(WPVal<-80)
   {
      Comment("Over Sold");
   }

   if(WPVal>-20)
   {
      Comment("Over Bought");
   }

  }
//+------------------------------------------------------------------+
```

Differences in this code are the same as the following.

Conditions of oversold:

```
   if(WPVal<-80)
   {
      Comment("Over Sold");
   }
```

Conditions of overbought:

```
   if(WPVal>-20)
   {
      Comment("Over Bought");
   }
```

After compiling this code, when we want to execute it in the MetaTrader 5 to generate automatic signals we will do the following step:

Find the expert file in the navigator window.

![WPR nav3](https://c.mql5.com/2/47/WPR_nav3.png)

Drag and drop it on the desired chart that we need to extract signals from. The following is for its window.

![Williams_R - OB _ OS window](https://c.mql5.com/2/47/Williams_R_-_OB___OS_window.png)

Tick next to "Allow Algo Trading" --> Press "OK". Find it attached to the chart.

![Williams_R - OB _ OS attached](https://c.mql5.com/2/47/Williams_R_-_OB___OS_attached.png)

The following is an example of generating signals based on this strategy from testing.

Overbought:

![Williams_R - OB _ OS - OB signal](https://c.mql5.com/2/47/Williams_R_-_OB___OS_-_OB_signal.png)

As we can see in the previous example, the expert generated an over-bought signal as the WPR value is above the -20 level which means that prices are trading near to their high during the 14 periods.

Oversold:

![Williams_R - OB _ OS - OS signal](https://c.mql5.com/2/47/Williams_R_-_OB___OS_-_OS_signal.png)

According to the previous example, we found that the WPR is below the -80 level which means that prices are trading near to their lows, So, the signal here is oversold.

- Strategy two: Williams%R - Crossover:

According to this strategy, we need to create a trading system to generate both of buying and selling signals based on the crossover. The following is for the full code to create a trading system based on this strategy:

```
//+------------------------------------------------------------------+
//|                                       Williams%R - Crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double WPArray[];

   ArraySetAsSeries(WPArray,true);

   int WPDef = iWPR(_Symbol,_Period,14);

   CopyBuffer(WPDef,0,0,3,WPArray);

   double WPVal = NormalizeDouble(WPArray[0],2);
   double WPPrevVal = NormalizeDouble(WPArray[1],2);

   if(WPPrevVal<-50 && WPVal>-50)
     {
      Comment("Buy signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "Williams % R Previous Value is",WPPrevVal);
     }

   if(WPPrevVal>-50 && WPVal<-50)
     {
      Comment("Sell signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "Williams % R Previous Value is",WPPrevVal);
     }

  }
//+------------------------------------------------------------------+
```

Differences in this code are the same as the following.

Getting the previous Williams' %R value by using the "NormalizeDouble" function after creating a double variable of WPPrevValue for the previous Williams' %R value. Parameters of the "NormalizeDouble" function are the normalized number and number of digits after the decimal point.

```
double WPPrevVal = NormalizeDouble(WPArray[1],2);
```

Conditions of the Williams%R - crossover strategy.

In case of the buy signal:

```
   if(WPPrevVal<-50 && WPVal>-50)
     {
      Comment("Buy signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "Williams % R Previous Value is",WPPrevVal);
     }
```

In case of the sell signal:

```
   if(WPPrevVal>-50 && WPVal<-50)
     {
      Comment("Sell signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "Williams % R Previous Value is",WPPrevVal);
     }
```

After compiling this code, we can find it in the navigator window the same as the following.

![WPR nav2](https://c.mql5.com/2/47/WPR_nav2.png)

By double-clicking on the expert of this strategy as a second option to execute it on the desired chart, the window of the expert will appear.

![Williams_R - Crossover window](https://c.mql5.com/2/47/Williams_R_-_Crossover_window.png)

After pressing "OK", it will be attached to the chart.

![Williams_R - Crossover attached](https://c.mql5.com/2/47/Williams_R_-_Crossover_attached.png)

The following is for an example of generated signals of this strategy from testing.

Buy signal:

![Williams_R - Crossover signal - Buy signal](https://c.mql5.com/2/47/Williams_R_-_Crossover_signal_-_Buy_signal.png)

The same as we can see on the chart, the expert generated a buy signal, the Williams %R value, and the Williams %R value in the upper left corner of the chart as a comment because the conditions of this strategy are met because the previous WPR = -55.93 which is lower than the -50 level and the current WPR = -41.09 which is greater than -50 level.

Sell signal:

![Williams_R - Crossover signal - Sell signal](https://c.mql5.com/2/47/Williams_R_-_Crossover_signal_-_Sell_signal.png)

As we can see in the previous example of a sell signal based on this strategy, we can find in the upper left corner of the chart the sell signal, the current WPR, and the previous WPR values as a comment because of the conditions of this strategy are met.

The previous WPR value = -29.95 which is greater than the -50 level and the current WPR = -58.8 which is lower than the -50 level.

- Strategy three: Williams%R & MA:

The following is for the full code to create a trading system for this strategy:

```
//+------------------------------------------------------------------+
//|                                              Williams%R & MA.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double WPArray[];
   double MAArray[];

   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   ArraySetAsSeries(WPArray,true);
   ArraySetAsSeries(MAArray,true);

   int WPDef = iWPR(_Symbol,_Period,14);
   int MADef = iMA(_Symbol,_Period,100,0,MODE_EMA,PRICE_CLOSE);

   CopyBuffer(WPDef,0,0,3,WPArray);
   CopyBuffer(MADef,0,0,3,MAArray);

   double WPVal = NormalizeDouble(WPArray[0],2);
   double MAVal = NormalizeDouble(MAArray[0],2);

   if(Ask>MAVal && WPVal>-50)
     {
      Comment("Buy signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }

   if(Bid<MAVal && WPVal<-50)
     {
      Comment("Sell signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }

  }
//+------------------------------------------------------------------+
```

Differences in this code are the same as the following.

Creating arrays for WP and MA.

```
double WPArray[];
double MAArray[];
```

Getting values of ask and bid after creating double variables for them.

```
double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);
```

Sorting this created arrays from current data.

```
ArraySetAsSeries(WPArray,true);
ArraySetAsSeries(MAArray,true);
```

Defining Williams' PR by using the iWPR function and Moving Average by using the iMA function. Parameters of iWPR mentioned before. Parameters of iMA are symbol name, period, averaging period, horizontal shift, smoothing type, and type of price.

```
int WPDef = iWPR(_Symbol,_Period,14);
int MADef = iMA(_Symbol,_Period,100,0,MODE_EMA,PRICE_CLOSE);
```

Copying price data to the created arrays by using the "CopyBuffer" function.

```
CopyBuffer(WPDef,0,0,3,WPArray);
CopyBuffer(MADef,0,0,3,MAArray);
```

Getting values of the current Williams RP and Moving average.

```
double WPVal = NormalizeDouble(WPArray[0],2);
double MAVal = NormalizeDouble(MAArray[0],2);
```

Conditions of strategy.

In case of buy signal:

```
   if(Ask>MAVal && WPVal>-50)
     {
      Comment("Buy signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }
```

In case of sell signal

```
   if(Bid<MAVal && WPVal<-50)
     {
      Comment("Sell signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }
```

After compiling this code, we will find the expert in the navigator window of the MetaTrader 5 the same as the following.

![WPR nav4](https://c.mql5.com/2/47/WPR_nav4.png)

By dragging and dropping it on the desired chart, we will find its window the same as the following.

![ Williams_R _ MA window](https://c.mql5.com/2/47/Williams_R___MA_window.png)

After ticking next to "Allow Algo Trading" and then pressing "OK", the expert will be attached to the chart the same as the following.

![ Williams_R _ MA attached](https://c.mql5.com/2/47/Williams_R___MA_attached.png)

The following is an example of generated signals based on this strategy.

In case of buy signal:

![ Williams_R _ MA - Buy](https://c.mql5.com/2/47/Williams_R___MA_-_Buy.png)

As we can find on the chart the generated signal in the upper left corner,

- Buy signal
- WPR value
- EMA value

Conditions are met. the Ask is greater than MA and WPR is greater than -50

In case of sell signal:

![ Williams_R _ MA - Sell](https://c.mql5.com/2/47/Williams_R___MA_-_Sell.png)

As we can find on the chart the generated signal in the upper left corner,

- Sell signal
- WPR value
- EMA value

Conditions are met. The bid is lower than MA and WPR is lower than the -50 level.

### Conclusion

Williams' PR is an important technical tool that can be considered as one of the most popular indicators that measure the momentum the same as we mentioned. It can be combined also with another technical indicator to magnify benefits because this is one of the most important benefits of technical analysis is that we can use more than one tool which can clear more than one perspective to understand the instrument well to take a suitable decision.

We learned through this article more topics about the Williams' PR technical indicator, as we learned what it is, what it measures, and how we can calculate it by applying that through an example to deepen our understanding, then we learned how we can insert the indicator, and how we can read it to understand what it wants to inform us.

We learned also, how we can use it through simple strategies based on the main concept behind it as we learned three simple strategies and they were:

- Williams %R - OB and OS: to generate signals of overbought and oversold when prices reach these zones.
- Williams %R - crossover: to generate signals of buy and sell signals based on specific and meaningful conditions.
- Williams %R - MA: to generate buy and sell signals based on WPR and exponential moving average reading.

After that, we designed a step-by-step blueprint for each mentioned strategy to help us create a trading system for each one of them. Then we learned and created a trading system for each strategy by the MQL5 to use them in the MetaTrader 5 trading terminal to generate signals automatically and accurately.

All that we learned in this article is a simple example of how we can do using programming as the Algorithmic trading is an incredible tool that can help us to trade smoothly, easily, and effectively. It can be a very useful tool to let money works for you while sleeping literally. So, I encourage everyone to learn or think about how he can benefit from this tool in a way that increases the probability of maximizing profits.

I need to confirm here again that you must test any strategy before using it on your real account because there is nothing that is suitable for everyone, what is suitable for me may not be for you. especially since the main purpose of this article is educational so we may find that strategies need optimization. I hope that you tried to apply what read by yourself as this will be an important factor to understand what you learn. I hope also that you find this article useful for you to enhance your trading to get better results and if you want to read similar articles about how to design a trading system by the most popular technical indicators, you can read my previous articles in this series and I hope that you find them useful for you.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11142.zip "Download all attachments in the single ZIP archive")

[Simple\_WilliamswR.mq5](https://www.mql5.com/en/articles/download/11142/simple_williamswr.mq5 "Download Simple_WilliamswR.mq5")(0.9 KB)

[WilliamskR\_-\_OB\_q\_OS.mq5](https://www.mql5.com/en/articles/download/11142/williamskr_-_ob_q_os.mq5 "Download WilliamskR_-_OB_q_OS.mq5")(0.97 KB)

[WilliamsbR\_-\_Crossover.mq5](https://www.mql5.com/en/articles/download/11142/williamsbr_-_crossover.mq5 "Download WilliamsbR_-_Crossover.mq5")(1.27 KB)

[WilliamsbR\_2\_MA.mq5](https://www.mql5.com/en/articles/download/11142/williamsbr_2_ma.mq5 "Download WilliamsbR_2_MA.mq5")(1.53 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/428153)**
(18)


![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
22 Aug 2022 at 15:39

**Pavel Malyshko [#](https://www.mql5.com/ru/forum/431239/page2#comment_41561601):**

Well, you write about stress tests... about some rubbish... and so on... I, for example, can only say thank you to mql5... and in the community there should be something to unite... users should help each other... why praise this most simple thing.

So you don't even have a minimum of research skills? Or is there no one else to talk to on free topics?

Stop floundering. And if you need help, you can contact me. )

![Pavel Malyshko](https://c.mql5.com/avatar/2022/9/631563ec-074c.jpg)

**[Pavel Malyshko](https://www.mql5.com/en/users/malishko89)**
\|
23 Aug 2022 at 19:15

**Anatoli Kazharski [#](https://www.mql5.com/ru/forum/431239/page2#comment_41561623):**

So you don't even have a minimum of research skills? Or is there no one else to talk to on free topics?

Stop floundering. And if you need help, you can contact me. )

I, for example, do not like all these off-topic comments.

I'm leaving this thread. don't bother replying to me.

![Anatoli Kazharski](https://c.mql5.com/avatar/2022/1/61D72F6B-7C12.jpg)

**[Anatoli Kazharski](https://www.mql5.com/en/users/tol64)**
\|
23 Aug 2022 at 19:27

**Pavel Malyshko [#](https://www.mql5.com/ru/forum/431239/page2#comment_41593911):**

you've got a free topic and you've made a mess of it by seeing the author's mistake.

...

Where did you see it? Show me. )

For those who do not understand, I will explain. The point was that the author wrote a lot of almost identical articles, the only difference being that they offer different indicators. The whole code base is full of this useless "good".

And you imagined something about an error.

**Pavel Malyshko [#](https://www.mql5.com/ru/forum/431239/page2#comment_41593911):**

...

I'm leaving this topic. Don't bother replying to me.

You shouldn't have started.

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
24 Aug 2022 at 09:19

Thank you all for your comments. The blueprint error has been edited and sent for republishing. I need to clarify one more thing about my articles in this first series on mql5, because the main purpose of this series is to share the most technical indicators and create simple strategies that can be used based on the basic concept of each indicator.You can find similar content, but the main topic for each article is different, because in each article I share a new indicator with different technical strategies and create a simple trading system for those strategies and they cannot be in one article. So, this series is for beginners in addition to we all don't use the same tools or indicators in our trading but when we provide the most popular technical indicators based on what I mentioned earlier, nowadays everyone can find what they need. be suitable for their trading in addition to understanding how to code as a beginner.

Thanks.

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
5 Jan 2023 at 18:06

Happy New Year to all!

Many thanks to the author for his work. There is one nuance, which, in my opinion, still needs to be corrected when normalising the [moving average](https://www.mql5.com/en/code/42 "Moving Average of Oscillator is the difference between oscillator and oscillator smoothing") price. Instead of rounding a floating point number to 2:

```
double MAVal = NormalizeDouble(MAArray[0],2);
```

rounding to \_Digits:

```
double MAVal = NormalizeDouble(MAArray[0],_Digits);
```

because the condition highlighted in colour is often not met:

```
   if(Ask>MAVal && WPVal>-50)
     {
      Comment("Buy signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }

   if(Bid<MAVal && WPVal<-50)
     {
      Comment("Sell signal","\n",
              "Williams % R Value is",WPVal,"\n",
              "EMA Value is",MAVal);
     }
```

Regards, Vladimir.

![Indicators with on-chart interactive controls](https://c.mql5.com/2/46/interactive-control.png)[Indicators with on-chart interactive controls](https://www.mql5.com/en/articles/10770)

The article offers a new perspective on indicator interfaces. I am going to focus on convenience. Having tried dozens of different trading strategies over the years, as well as having tested hundreds of different indicators, I have come to some conclusions I want to share with you in this article.

![Developing a trading Expert Advisor from scratch (Part 13): Time and Trade (II)](https://c.mql5.com/2/46/development__4.png)[Developing a trading Expert Advisor from scratch (Part 13): Time and Trade (II)](https://www.mql5.com/en/articles/10412)

Today we will construct the second part of the Times & Trade system for market analysis. In the previous article "Times & Trade (I)" we discussed an alternative chart organization system, which would allow having an indicator for the quickest possible interpretation of deals executed in the market.

![Neural networks made easy (Part 14): Data clustering](https://c.mql5.com/2/48/Neural_networks_made_easy_014.png)[Neural networks made easy (Part 14): Data clustering](https://www.mql5.com/en/articles/10785)

It has been more than a year since I published my last article. This is quite a lot time to revise ideas and to develop new approaches. In the new article, I would like to divert from the previously used supervised learning method. This time we will dip into unsupervised learning algorithms. In particular, we will consider one of the clustering algorithms—k-means.

![Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://c.mql5.com/2/46/development__3.png)[Developing a trading Expert Advisor from scratch (Part 12): Times and Trade (I)](https://www.mql5.com/en/articles/10410)

Today we will create Times & Trade with fast interpretation to read the order flow. It is the first part in which we will build the system. In the next article, we will complete the system with the missing information. To implement this new functionality, we will need to add several new things to the code of our Expert Advisor.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/11142&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069233815441834454)

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