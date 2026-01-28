---
title: Learn how to design a trading system by Accelerator Oscillator
url: https://www.mql5.com/en/articles/11467
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:47:06.589106
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/11467&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6410228046700074137)

MetaTrader 5 / Trading


### Introduction

Here is a new article from our series about how to design a trading system based on the most popular technical indicators. We will learn in this article a new technical tool that can be used in our favor in trading. We will learn about the Accelerator Oscillator indicator (AC) in detail as we learn what it is, what it measures, how it can be calculated, how to read it and use it also through simple trading strategies then how we can create a trading system based on them.

The following topics will be our path to learn more about this indicator:

1. [Accelerator Oscillator definition](https://www.mql5.com/en/articles/11467#definition)
2. [Accelerator Oscillator strategy](https://www.mql5.com/en/articles/11467#strategy)
3. [Accelerator Oscillator strategy blueprint](https://www.mql5.com/en/articles/11467#blueprint)
4. [Accelerator Oscillator trading system](https://www.mql5.com/en/articles/11467#system)
5. [Conclusion](https://www.mql5.com/en/articles/11467#conclusion)

We will use the MQL5 (MetaQuotes Language 5) to write our codes by it which is built-in in the MetaTrader 5 trading terminal which will be used to execute our trading system. If you do not know how to download the MetaTrader 5 and how to use MQL5, you can read the topic of [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) from a previous article to learn more about that. All mentioned strategies here are designed for educational purposes only and you must test them before using them in your real account to make sure that they are profitable and suitable for you and I advise you to try to apply what you learn if you want to improve your trading and coding skills as this step is an important step as it will be helpful to get the full benefit from this article.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Accelerator Oscillator definition

The Accelerator Oscillator (AC) is a momentum indicator which is developed by Bill Williams. It measures if there is a change in momentum. If the momentum of an uptrend is decreased, this may mean that there is less interest to buy the instrument and, in this case, we may see a different movement in the opposite direction of selling and vice versa, if the momentum of a downtrend is decreased, this may mean that there is a less interest to sell the instrument and we may see a buying power. It is a leading indicator also as it may move before price.

So, the main objective of this indicator is to measure the acceleration and deceleration of the market power in both directions up or down to get insights into how long the current price movement will continue and be ready for any change.

Now, we will learn how we can calculate this indicator manually through the following steps:

AC = AO - SMA (AO, 5)

Where:

Ac = Accelerator Oscillator.

AO = Awesome Oscillator, by the way you can read my previous article about the [Awesome Oscillator](https://www.mql5.com/en/articles/11468) to know more about it.

SMA = Simple Moving Average.

5 = the period of SMA.

To calculate the AO following the following steps:

MEDIAN PRICE = (HIGH + LOW) / 2

AO = SMA (MEDIAN PRICE, 5) - SMA (MEDIAN PRICE, 34)

Fortunately, we do not need to calculate this indicator manually as we have it in the MetaTrader 5 trading platform, and all that we need to do is choose it from the available indicators. While opening the trading terminal we will press Insert --> Indicators --> Bill Williams --> Accelerator Oscillator

![ AC insert](https://c.mql5.com/2/49/AC_insert.png)

Then we will find the window of the indicator parameters to set our preferences the same as the following:

![ AC param](https://c.mql5.com/2/49/AC_param.png)

1 - the color up values.

2 - the thickness of the histogram.

3 - the color of down values.

After determining the previous parameters as per our preferences and pressing "OK" we will find the indicator is inserted to the chart the same as the following:

![ AC attached](https://c.mql5.com/2/49/AC_attached.png)

As we can see in the previous chart, we have the indicator inserted in the lower part of the chart and we have the AC values are oscillating around zero based on the acceleration of the momentum.

The zero-level means that there is a balance between the two market parties bulls and bears. If the AC is greater than the zero level, it means that we may find a continuation of moving upward and vice versa, if the AC is lower than the zero level, it means that we may find a continuation of moving downward.

### Accelerator Oscillator strategy

In this topic, we will learn simple trading strategies to be used based on the basic concept of the AC indicator, but you must test them before using them on your real account as the main objective here is educational only. So, you must make sure that it is suitable and profitable for your trading.

**Strategy one: AC Zero Crossover**

Based on this strategy, we need to use the AC to know when there is a bullish or bearish signal by comparing two values to determine their positions the current AC value and the zero level of the AC indicator. If the current AC value is greater than the zero level, it will be a signal of bullishness. In the other scenario, if the current AC value is lower than the zero level, it will be a signal of bearishness.

Simply,

AC > Zero level --> Bullish

AC < Zero level --> Bearish

**Strategy two: AC Strength**

Based on this strategy, we need to get signals with the strength of the AC movement based on comparing the current AC value with the maximum AC and the minimum AC values of the last ten AC values to determine the position of every value to get the suitable signal. If the current AC value is greater than the maximum AC value, it will be a signal of strength. In the other scenario, if the current AC value is lower than the minimum AC value, it will be a signal of weakness.

Simply,

AC > Max AC value --> AC is strong

AC < Min AC value --> AC is weak

**Strategy three: AC & MA Strategy**

Based on this strategy, we need to get signals of buying or selling depending on checking the five values and they are the closing price, the 50- period exponential moving average, the current AC, the maximum AC value, and the minimum AC value of the last ten AC values to determine the position of them to get the suitable signal. If the current AC is greater than the maximum AC value and the closing price is greater than the 50- period EMA, it will be a buy signal. In the other scenario, if the current AC is lower than the minimum AC value and the closing price is lower than the 50- period EMA, it will be a sell signal.

Simply,

AC > Max AC value and Close > 50- EMA --> Buy

AC < Min AC value and Close < 50- EMA --> Sell

### Accelerator Oscillator strategy blueprint

In this topic, we will design a step-by-step blueprint for every mentioned strategy to help us to create a trading system smoothly.

**Strategy one: AC Zero Crossover**

According to this strategy we need to create a trading system that can be used to generate automatically signals of bullish or bearish as a comment on the chart based on continuously checking for the current AC value and the zero level of the AC indicator to determine the position of them to return the suitable signal. If the current AC value is greater than the zero level, we need the trading system to return comment on the chart with the following values:

- Bullish
- AC Value

In the other case, if the AC value is lower than the zero level, we need the trading system to return comment on the chart with the following values:

- Bearish
- AC Value

The following is the blueprint of this trading system:

![ AC Zero Crossover blueprint](https://c.mql5.com/2/49/AC_Zero_Crossover_blueprint.png)

**Strategy two: AC Strength**

According to this strategy, we need to create a trading system that can be used to generate a signal with the strength of AC movement if it is strong or weak based on continuously checking for three values and they are the current AC value, the maximum value, the minimum value of the last ten values of the AC indicator. If the current AC value is greater than the maximum value, we need the trading system to return comment on the chart with the following values:

- AC is strong
- AC value
- AC maximum value
- AC minimum value

In the other case, if the current AC value is lower than the minimum value we need the trading system to return a comment on the chart with the following values:

- AC is weak
- AC value
- AC maximum value
- AC minimum value

The following is the blueprint of this trading system:

![ AC strength blueprint](https://c.mql5.com/2/49/AC_strength_blueprint.png)

**Strategy three: AC & MA Strategy**

According to this strategy we need to create a trading system that can be used to generate signals of buy and sell as a comment on the chart based on continuously checking the following five values, the current AC, the maximum AC, the minimum AC, the closing price, and the moving average value to determine the position of them to generate the suitable signal. If the current AC is greater than the maximum AC value and the closing price is greater than the moving average, we need the trading system to generate a comment on the chart as a signal with the following values:

- Buy
- Closing price
- AC value
- AC Max
- AC Min
- MA value

In the other case, if the current AC is lower than the minimum AC value and the closing price is lower than the moving average, we need the trading system to return comment on the chart with the following values:

- Sell
- Closing price
- AC value
- AC Max
- AC Min
- MA value

The following is the blueprint of this trading system:

![ AC _ MA Strategy blueprint](https://c.mql5.com/2/49/AC___MA_Strategy_blueprint.png)

### Accelerator Oscillator trading system

In this interesting topic, we will create our trading system based on the mentioned strategies, but we will create a base for them by creating a simple trading system that will be able to return a comment on the chart with the current AC value and the following is for steps to do that:

Creating an array for the acArray by using the "double" function:

```
double acArray[];
```

Using the "ArraySetAsSeries" function to the acArray to return a true or false as a Boolean value. Its parameters are:

- array\[\]: we will use acArray.
- flag: We will use "true" as an array indexing direction.

```
ArraySetAsSeries(acArray,true);
```

Defining the AC indicator by using the "iAC" function to return the handle of the Accelerator Oscillator indicator. Its parameters are:

- symbol: we will use the (\_Symbol) to be applied to the current symbol.
- period: we will use the (\_period) to be applied to the current period.

```
int acDef = iAC(_Symbol,_Period);
```

Getting data from the buffer of the AC indicator by using the "CopyBuffer" function. Its parameters are:

- indicator\_handle: we will use the (acDef) of the predefined AC indicator handle.
- buffer\_num: to determine the indicator buffer number, we will use (0).
- start\_pos: to determine the starting position, we will use (0).
- count: to determine the amount to copy, we will use (3).
- buffer\[\]: to determine the target array to copy, we will use (acArray).

```
CopyBuffer(acDef,0,0,3,acArray);
```

Define the AC value after creating a double variable for the acVal and Normalizing it by using the "NormalizeDouble". Parameters of the "NormalizeDouble" are:

- value: Normalized number. We will use (acArray\[0\]).
- digits: to determine the number of digits after the decimal. We will use (7).

```
double acVal = NormalizeDouble(acArray[0],7);
```

Return the comment of the current AC value on the chart by using the "Comment" function.

```
Comment("AC Value is ",acVal);
```

The following is the full code of this trading system:

```
//+------------------------------------------------------------------+
//|                                                    Simple AC.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
  void OnTick()
  {
   double acArray[];
   ArraySetAsSeries(acArray,true);
   int acDef = iAC(_Symbol,_Period);
   CopyBuffer(acDef,0,0,3,acArray);
   double acVal = NormalizeDouble(acArray[0],7);
   Comment("AC Value is ",acVal);
  }
//+------------------------------------------------------------------+
```

After compiling this code, we will find it in the navigator in the Expert Advisors folder the same as the following:

![ Simple AC nav](https://c.mql5.com/2/49/Simple_AC_nav.png)

To execute this file, we will drag and drop it on the desired chart then we will find the window of it the same as the following:

![ Simple AC win](https://c.mql5.com/2/49/Simple_AC_win.png)

![ Simple AC attached](https://c.mql5.com/2/49/Simple_AC_attached.png)

As we can see in the previous picture in the top right corner, we have the expert is attached to the chart. Now, we're ready to receive our signal and it will be the same as the following:

![ Simple AC signal](https://c.mql5.com/2/49/Simple_AC_signal.png)

As we can see in the top left corner, we have a comment on the current AC value.

**Strategy one: AC Zero Crossover**

The following is for the full code to create the trading system of this strategy:

```
//+------------------------------------------------------------------+
//|                                            AC Zero Crossover.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
  void OnTick()
  {
   double acArray[];
   ArraySetAsSeries(acArray,true);
   int acDef = iAC(_Symbol,_Period);
   CopyBuffer(acDef,0,0,3,acArray);
   double acVal = NormalizeDouble(acArray[0],7);
   if(acVal > 0)
     {
      Comment("Bullish","\n"
              "AC Value is ",acVal);
     }
   if(acVal < 0)
     {
      Comment("Bearish","\n"
              "AC Value is ",acVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code:

Conditions of the strategy.

In case of bullish signal:

```
   if(acVal > 0)
     {
      Comment("Bullish","\n"
              "AC Value is ",acVal);
     }
```

In case of bearish signal

```
   if(acVal < 0)
     {
      Comment("Bearish","\n"
              "AC Value is ",acVal);
     }
```

After compiling this code, making sure that there are no errors, and executing it by dragging and dropping on the chart the same as what we learned before we will find it is attached to the chart the same as the following:

![ AC Zero Crossover attached](https://c.mql5.com/2/49/AC_Zero_Crossover_attached.png)

In the top right corner, we can see the expert of this strategy is attached to the chart. We're ready now to receive our signals.

In case of a bullish signal:

![ AC Zero Crossover - bullish signal](https://c.mql5.com/2/49/AC_Zero_Crossover_-_bullish_signal.png)

We can see in the top left corner that we have the comment as a signal of this strategy with the following values:

- Bullish
- AC Value

In case of a bearish signal:

![ AC Zero Crossover - bearish signal](https://c.mql5.com/2/49/AC_Zero_Crossover_-_bearish_signal.png)

We can see in the top left corner that we have the following values:

- Bearish
- AC Value

**Strategy two: AC Strength**

The following is for the full code to create the trading system of this strategy:

```
//+------------------------------------------------------------------+
//|                                                  AC Strength.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double acArray[];
   ArraySetAsSeries(acArray,true);
   int acDef = iAC(_Symbol,_Period);
   CopyBuffer(acDef,0,0,11,acArray);
   double acCurrVal = NormalizeDouble(acArray[0],7);
   int acMaxArray = ArrayMaximum(acArray,1,WHOLE_ARRAY);
   int acMinArray = ArrayMinimum(acArray,1,WHOLE_ARRAY);
   double acMaxVal = NormalizeDouble(acArray[acMaxArray],7);
   double acMinVal = NormalizeDouble(acArray[acMinArray],7);
   if(acCurrVal>acMaxVal)
     {
      Comment("AC is strong ","\n",
              "AC Value is ",acCurrVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal);
     }
   if(acCurrVal<acMinVal)
     {
      Comment("AC is weak ","\n",
              "AC Value is ",acCurrVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code.

Defining the current value of AC:

```
double acCurrVal = NormalizeDouble(acArray[0],7);
```

Defining the maximum value in the whole array of AC by using the "ArrayMaximum" function to return the maximum value. Its parameters:

- array\[\]: we will use acArray.
- start: the starting point to check from. We will use (1).

- count: total elements from array to check. We will use (WHOLE\_ARRAY) to check the whole of this array.


```
int acMaxArray = ArrayMaximum(acArray,1,WHOLE_ARRAY);
```

Defining the minimum value in the whole array of AC by using the "ArrayMinimum" function to return the minimum value. Its parameters:

- array\[\]: we will use (acArray).
- start: the starting point to check from. We will use (1).
- count: total elements from array to check. We will use (WHOLE\_ARRAY) to check the whole of this array.

```
int acMinArray = ArrayMinimum(acArray,1,WHOLE_ARRAY);
```

Normalizing values of maximum and minimum by using the "NormalizeDouble" function.

```
   double acMaxVal = NormalizeDouble(acArray[acMaxArray],7);
   double acMinVal = NormalizeDouble(acArray[acMinArray],7);
```

Conditions of the strategy:

In case of a strong signal,

```
   if(acCurrVal>acMaxVal)
     {
      Comment("AC is strong ","\n",
              "AC Value is ",acCurrVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal);
     }
```

In case of a weak signal,

```
   if(acCurrVal<acMinVal)
     {
      Comment("AC is weak ","\n",
              "AC Value is ",acCurrVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal);
     }
```

After compiling and executing this code on the desired chart, we will find it attached the same as the following:

![AC strength attached](https://c.mql5.com/2/49/AC_strength_attached.png)

As we can in the top right of the chart that the expert is attached, we are ready to receive signals of this strategy the same as the following examples from testing.

In case of a strong signal:

![AC strength - strong signal](https://c.mql5.com/2/49/AC_strength_-_strong_signal.png)

As we can see on the chart in the top left chart as we received our signal with the following values:

- AC is strong
- AC value
- AC maximum value
- AC minimum value

In case of a weakness:

![AC strength - weak signal](https://c.mql5.com/2/49/AC_strength_-_weak_signal.png)

As we can that we got the signal with the following values:

- AC is weak
- AC value
- AC maximum value
- AC minimum value

**Strategy three: AC & MA Strategy**

The following is for the full code to create the trading system of this strategy:

```
//+------------------------------------------------------------------+
//|                                             AC & MA Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlRates pArray[];
   double acArray[];
   double maArray[];
   int Data=CopyRates(_Symbol,_Period,0,1,pArray);
   ArraySetAsSeries(acArray,true);
   ArraySetAsSeries(maArray,true);
   int acDef = iAC(_Symbol,_Period);
   int maDef = iMA(_Symbol,_Period,50,0,MODE_EMA,PRICE_CLOSE);
   CopyBuffer(acDef,0,0,3,acArray);
   CopyBuffer(maDef,0,0,3,maArray);
   int acMaxArray = ArrayMaximum(acArray,1,WHOLE_ARRAY);
   int acMinArray = ArrayMinimum(acArray,1,WHOLE_ARRAY);
   double closingPrice = pArray[0].close;
   double acVal = NormalizeDouble(acArray[0],7);
   double acMaxVal = NormalizeDouble(acArray[acMaxArray],7);
   double acMinVal = NormalizeDouble(acArray[acMinArray],7);
   double maVal = NormalizeDouble(maArray[0],7);
   if(acVal > acMaxVal && closingPrice > maVal)
     {
      Comment("Buy","\n"
              "Closing Price is ",closingPrice,"\n",
              "Ac Value is ",acVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal,"\n",
              "MA Value is ",maVal);
     }
   if(acVal < acMinVal && closingPrice < maVal)
     {
      Comment("Sell","\n"
              "Closing Price is ",closingPrice,"\n",
              "Ac Value is ",acVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal,"\n",
              "MA Value is ",maVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code:

Creating arrays of pArray, acArray, and maArray. We will use the double function for the acArray and maArray but we will use the MqlRates function for pArray to store information about the price, volume, and spread.

```
   MqlRates pArray[];
   double acArray[];
   double maArray[];
```

Setting the AS\_SERIES flag to arrays of (acArray) and (maArray) like what we mentioned before and defining Data by using the "CopyRates" function to get historical data of MqlRates structure and its parameters are:

- symbol\_name: we will use \_Symbol to be applied to the current symbol.
- timeframe: we will use \_Period to be applied to the current period.
- start\_pos: to determine the starting position, we will use (0).
- count: to determine the data count to copy, we will use (1).
- rates\_array\[\]: to determine the target array to copy, we will use the pArray.

```
   int Data=CopyRates(_Symbol,_Period,0,1,pArray);
   ArraySetAsSeries(acArray,true);
   ArraySetAsSeries(maArray,true);
```

We will define the AC, MA:

AC by using the "iCA" function the same as we mentioned. But MA, we will use the "iMA" function, its parameters:

- symbol: we will use (\_Symbol)
- period: we will use (\_period)
- ma\_period: to determine the period of the moving average, we will use 50
- ma\_shift: to determine the horizontal shift, we will use (0)
- ma\_method: to determine the type of the moving average, we will use exponential MA
- applied\_price: to determine the type of useable price, we will use the closing price

```
   int acDef = iAC(_Symbol,_Period);
   int maDef = iMA(_Symbol,_Period,50,0,MODE_EMA,PRICE_CLOSE);
```

We will get data from the buffer of the AC and MA indicators by using the "CopyBuffer" function.

```
   CopyBuffer(acDef,0,0,3,acArray);
   CopyBuffer(maDef,0,0,3,maArray);
```

Getting maximum and minimum values of the acArray.

```
   int acMaxArray = ArrayMaximum(acArray,1,WHOLE_ARRAY);
   int acMinArray = ArrayMinimum(acArray,1,WHOLE_ARRAY);
```

Defining values of AC, AC maximum, AC minimum, and Exponential Moving Average.

```
   double acVal = NormalizeDouble(acArray[0],7);
   double acMaxVal = NormalizeDouble(acArray[acMaxArray],7);
   double acMinVal = NormalizeDouble(acArray[acMinArray],7);
   double maVal = NormalizeDouble(maArray[0],7);
```

Conditions of the strategy.

In case of a buy signal:

```
   if(acVal > acMaxVal && closingPrice > maVal)
     {
      Comment("Buy","\n"
              "Closing Price is ",closingPrice,"\n",
              "Ac Value is ",acVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal,"\n",
              "MA Value is ",maVal);
     }
```

In case of a sell signal:

```
   if(acVal < acMinVal && closingPrice < maVal)
     {
      Comment("Sell","\n"
              "Closing Price is ",closingPrice,"\n",
              "Ac Value is ",acVal,"\n",
              "AC Max is ",acMaxVal,"\n",
              "AC Min is ",acMinVal,"\n",
              "MA Value is ",maVal);
     }
```

After compiling this code and executing it to be attached to receive our signal, it will be attached the same as the following:

![AC _ MA Strategy attached](https://c.mql5.com/2/49/AC___MA_Strategy_attached.png)

As we can see in the top right corner that the expert is attached to the chart. Now, we can receive our signals.

In case of a buy signal:

![AC & MA Strategy - buy signal](https://c.mql5.com/2/49/AC_4_MA_Strategy_-_buy_signal.png)

As we can see in the top left corner that we have the following values:

- Buy
- Closing price
- AC value
- AC Max
- AC Min
- MA value

In case of a sell signal:

![AC & MA Strategy - sell signal](https://c.mql5.com/2/49/AC_t_MA_Strategy_-_sell_signal.png)

We have the comment with the following values:

- Sell
- Closing price
- AC value
- AC Max
- AC Min
- MA value

### Conclusion

After all that we learned in this article, it is supposed that you are understanding the Accelerator Oscillator indicator well as we covered it through this article and we learned what it is, what it measures, how we can calculate it, how we can read and use it through simple trading strategies which were the same as the following:

- AC Zero Crossover: to get bullish and bearish signals based on the crossover between the AC value and the zero level of the AC indicator.
- AC Strength: to get a signal of the strength of the AC movement based on comparing between the current AC value with the maximum and the minimum of the last ten AC values.
- AC & MA Strategy: to get buying and selling signals based on the positioning of the closing price, the 50- period exponential moving average, the AC current value, AC maximum value, and the AC minimum value.

Then, we designed a step-by-step blueprint for all mentioned strategies to help us to create our trading system easily, effectively, and smoothly. Then, we came to the most interesting part of the article as we wrote our codes to create a trading system based on these mentioned strategies to be executed in the MetaTrader 5 to generate automatic signals without any manually reading or monitoring of conditions to be applied.

I hope that tried to apply what you learned the same as I told you at the start of this article and got complete benefits from this article by getting more insights about the topic of this article or any related topic. I need to confirm here another time that you must test any mentioned strategy before using it on your real account to make sure that it will be profitable or suitable for your trading style as there is nothing suitable for all. In addition to that, the main objective here is education only.

I hope that you found this article useful for your trading to get better results and if you want to read more similar articles about how to design a trading system based on the most popular technical indicators like RSI, MACD, MA, Stochastic, Bollinger Bands ...etc. You can read my previous articles in this series to learn more about that.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11467.zip "Download all attachments in the single ZIP archive")

[Simple\_AC.mq5](https://www.mql5.com/en/articles/download/11467/simple_ac.mq5 "Download Simple_AC.mq5")(0.86 KB)

[AC\_Zero\_Crossover.mq5](https://www.mql5.com/en/articles/download/11467/ac_zero_crossover.mq5 "Download AC_Zero_Crossover.mq5")(1.02 KB)

[AC\_Strength.mq5](https://www.mql5.com/en/articles/download/11467/ac_strength.mq5 "Download AC_Strength.mq5")(1.46 KB)

[AC\_e\_MA\_Strategy.mq5](https://www.mql5.com/en/articles/download/11467/ac_e_ma_strategy.mq5 "Download AC_e_MA_Strategy.mq5")(1.98 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/433611)**
(1)


![vorster Charles](https://c.mql5.com/avatar/avatar_na2.png)

**[vorster Charles](https://www.mql5.com/en/users/vorster)**
\|
20 Oct 2022 at 12:21

How to install the robot


![Neural networks made easy (Part 20): Autoencoders](https://c.mql5.com/2/48/Neural_networks_made_easy_020.png)[Neural networks made easy (Part 20): Autoencoders](https://www.mql5.com/en/articles/11172)

We continue to study unsupervised learning algorithms. Some readers might have questions regarding the relevance of recent publications to the topic of neural networks. In this new article, we get back to studying neural networks.

![MQL5 Wizard techniques you should know (Part 03): Shannon's Entropy](https://c.mql5.com/2/49/Regression_Analysis.png)[MQL5 Wizard techniques you should know (Part 03): Shannon's Entropy](https://www.mql5.com/en/articles/11487)

Todays trader is a philomath who is almost always looking up new ideas, trying them out, choosing to modify them or discard them; an exploratory process that should cost a fair amount of diligence. These series of articles will proposition that the MQL5 wizard should be a mainstay for traders.

![Risk and capital management using Expert Advisors](https://c.mql5.com/2/49/Risk-and-capital-management-using-Expert-Advisors.png)[Risk and capital management using Expert Advisors](https://www.mql5.com/en/articles/11500)

This article is about what you can not see in a backtest report, what you should expect using automated trading software, how to manage your money if you are using expert advisors, and how to cover a significant loss to remain in the trading activity when you are using automated procedures.

![Matrix and Vector operations in MQL5](https://c.mql5.com/2/48/matrix_and_vectors_2.png)[Matrix and Vector operations in MQL5](https://www.mql5.com/en/articles/10922)

Matrices and vectors have been introduced in MQL5 for efficient operations with mathematical solutions. The new types offer built-in methods for creating concise and understandable code that is close to mathematical notation. Arrays provide extensive capabilities, but there are many cases in which matrices are much more efficient.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/11467&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6410228046700074137)

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