---
title: Learn how to design a trading system by Alligator
url: https://www.mql5.com/en/articles/11549
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:10:36.083518
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/11549&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069197540148052329)

MetaTrader 5 / Trading


### Introduction

Here is a new article from our series about how to design a trading system based on the most popular technical indicators. In this article, we will learn in detail about the Alligator indicator by learning what it is, it measures, how we can calculate it, and how we can read and use it. Then we will create a trading system based on some simple strategies based on the main objective of this indicator. We will cover this indicator through the following topics:

1. [Alligator definition](https://www.mql5.com/en/articles/11549#definition)
2. [Alligator strategy](https://www.mql5.com/en/articles/11549#strategy)
3. [Alligator strategy blueprint](https://www.mql5.com/en/articles/11549#blueprint)
4. [Alligator trading system](https://www.mql5.com/en/articles/11549#system)
5. [Conclusion](https://www.mql5.com/en/articles/11549#conclusion)

I advise you to try to apply what you learn by yourself especially the codes in this article to develop your programming skills as this is an important step to achieve this objective of development. We will use in this article MQL5 (MetaQuotes Language 5) IDE which is built-in in the MetaTrader 5 trading terminal, if you do not know how to download the MetaTrader 5 and how to use MQL5 you can read the topic of [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) to learn more about that.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Alligator definition

The Alligator technical indicator is a trend-following, developed by Bill Williams. It is based on that trend is not the most time of the age of financial market instruments and it is only 15% to 30% of the time while it consumes the most of time ranging or sideways it takes about 85% to 70% of the time. So, we have a trend sometimes either uptrend or downtrend but most of the time we have ranging periods. If you want to learn more about trends, their types, and how we can identify them, you can read the topic of [trend definition](https://www.mql5.com/en/articles/10715#trend) from my previous article it will be useful in this context. It uses some of Fibonacci numbers in the calculation the same as we will see as it uses three smoothed moving averages with a set of numbers five, eight, thirteen periods. As we said that the Alligator indicator consists of three smoothed moving averages, and they are the same as the following:

- The Alligator Jaw.
- The Alligator Teeth.
- The Alligator Lips.

We can calculate this indicator manually through the following steps:

The Alligator Jaw = SMMA (Median price, 13, 8)

The Alligator Teeth = SMA (Median price, 8, 5)

The Alligator Lips = SMMA (Median price, 5, 3)

Median Price = (High + Low)/2

Where:

SMMA = smoothed Moving Average

Median price = the median price of data

High = the highest price

Low = the lowest price

After the previous steps, we will get the Alligator indicator, but we do not need to calculate it manually as we have it as a built-in indicator in the MetaTrader 5 terminal and all we need to do is choose it from the available technical indicators the same as the following:

While Opening the MetaTrader 5 terminal, we will press Insert --> Indicators --> Bill Williams --> Alligator.

![ Alligator insert](https://c.mql5.com/2/49/Alligator_insert.png)

After that we will find the parameters of it the same as the following:

![ Alligator param](https://c.mql5.com/2/49/Alligator_param.png)

1 - the period of the Alligator Jaws.

2 - the horizontal shift of the Alligator Jaws.

3 - the period of the Alligator Teeth.

4 - the horizontal shift of the Alligator Teeth.

5 - the period of the Alligator Lips.

6 - the horizontal shift of the Alligator Lips.

7 - the method of smoothing.

8 - the usable price type.

Through the following window we can determine the style of the indicator:

![ Alligator param1](https://c.mql5.com/2/49/Alligator_param1.png)

1 - the color of the Jaws line.

2 - the type of the jaws line.

3 - the thickness of the jaws line.

4 - the color of the teeth line.

5 - the type of the teeth line.

6 - the thickness of the teeth line.

7 - the color of the lips line.

8 - the type of the lips line.

9 - the thickness of the lips line.

After determining all parameters and pressing "OK", we can find that the indicator is attached to the chart the same as the following:

![ Alligator attached](https://c.mql5.com/2/49/Alligator_attached.png)

As we can see in the previous chart we have the indicator inserted into the chart and we have three lines:

1 - Alligator Lips

2 - Alligator Teeth

3 - Alligator Jaws

We can read the indicator simply by observing the position of these lines from prices. If the price is above lips, teeth, then jaws, we can say the trend is up. If the price is below lips, teeth, then jaws, we can say that the trend is down. If the price is moving through the three lines, we can say that we have a sideways.

### Alligator strategy

There are many ways to use the Alligator indicator in trading, but I'll mention simple strategies that can be used based on the main idea of this topic for education and you must test them before using them in your real account to make sure that they will be profitable. You may find that you need to optimize mentioned strategies, and this is normal you can do that to reach the best formula that can be useful for your trading style.

**Strategy one: Alligator Trend Identifier**

Based on this strategy, we need to get signals of bullish and bearish signals based on specific conditions. If the lips value is greater than two values the teeth value and jaws value and at the same time the teeth value and the jaws value, it will be a bullish signal. In other case, if the lips value is lower than the teeth value and jaws value and at the same time the teeth value is lower than the jaws value, it will be a bearish signal.

lips > teeth value, lips > jaws value, and teeth > jaws value --> bullish

lips < teeth value, lips < jaws value, and teeth < jaws value --> bearish

**Strategy two: Alligator Signals System**

Based on this strategy, we need to get signals of buy and sell depending on the crossover between lips, teeth, and jaw values. If the lips value is greater than the teeth value and the jaw value, this will be a buy signal. In the other scenario, if the lips value is lower than the teeth and jaws value, it will be a sell signal.

lips > teeth value, lips > jaws value --> buy

lips < teeth value, lips < jaws value --> sell

**Strategy three: Alligator Signals System Enhancement**

Based on this strategy, we need to get signals of buy and sell based on the crossover between lips, teeth, and jaws after that the crossover between the closing price and the teeth value. If the lips line is lower than the teeth and the jaws then the closing price is greater than the teeth value, it will be a buy signal. In the other case, if the lips line is greater than the teeth and the jaws then the closing price is lower than the teeth value, it will be a sell signal.

lips < teeth value, lips < jaws value and closing price > teeth --> buy

lips > teeth value, lips > jaws value and closing price < teeth --> sell

### Alligator strategy blueprint

Now, we need to create a trading system based on mentioned strategies in the previous topic, and to do that we will design a step-by-step blueprint to help us create this trading system.

**Strategy one: Alligator Trend Identifier**

According to this strategy, we need to create a trading system that can be used to generate signals of bullish or bearish based on continuously checking for the following values:

- Lips value
- Teeth value
- Jaws value

We need the trading system to determine the positions of these values to generate suitable signals based on that. If the lips line is greater than teeth, lips greater than jaws, and teeth line is greater than the jaws, in this case, we need the trading system to return the following values as a comment on the chart:

- Bullish
- Jaws value = n
- teeth value = n
- lips value = n

In the other case, if the lips line is lower than the teeth line, the lips line is lower than the jaws line and at the same time, the teeth line is lower than the jaws line, in this case, we need the trading system to return the following values as a comment on the chart:

- Bearish
- Jaws value = n
- teeth value = n
- lips value = n

The following is a step-by-step blueprint of this trading system:

![Alligator Trend Identifier blueprint](https://c.mql5.com/2/49/Alligator_Trend_Identifier_blueprint.png)

**Strategy two: Alligator Signals System**

According to this strategy, we need to create a trading system that can be used to generate signals of buy or sell based on continuously checking the following values to determine the positions of each one of them:

- Lips value
- Teeth value
- Jaws value

If the lips line is greater than the teeth value and at the same time, the lips line is greater than the jaws value, in this case, we need the trading system to return a comment on the chart with the following values:

- Buy
- Jaws value = n
- teeth value = n
- lips value = n

In the other case, if the lips line is lower than the teeth value and at the same time, the lips line is lower than the jaws line, we need the trading system to return a comment on the chart as a signal with the following values:

- Sell
- Jaws value = n
- teeth value = n
- lips value = n

The following is a step-by-step blueprint of this trading system:

![Alligator signals system blueprint](https://c.mql5.com/2/49/Alligator_signals_system_blueprint.png)

**Strategy three: Alligator signals System Enhancement**

According to this strategy, we need to create a trading system to generate buy and sell signals based on checking the following values to determine the position of each one of them:

- Lips value
- Teeth value
- Jaws value
- Closing price

If the lips value is lower than the teeth one, the lips line is lower than the jaws line and then, the closing price became above the teeth value, we need the trading system to return a comment on the chart as a signal with the following values:

- Buy
- Lips value = n
- Teeth value = n
- Closing price = n
- Jaws value = n

If the lips line is greater than the teeth line, the lips line is higher than the jaws line and then, the closing price became below the teeth value, we need the trading system to return a comment on the chart as a signal with the following values:

- Sell
- Lips value = n
- Teeth value = n
- Closing price = n
- Jaws value = n

The following is a step-by-step blueprint of this trading system:

![Alligator Signals System Enhancement blueprint](https://c.mql5.com/2/49/Alligator_Signals_System_Enhancement_blueprint.png)

### Alligator trading system

Now, we will create a trading system for each mentioned strategy to be executed in the MetaTrader 5 terminal to generate desired signals automatically. First, we will create a base trading system to generate a signal of the Alligator components values to use in our trading system. The following are for steps to create this type of system:

Creating arrays of each one of the Alligator components (Lips, Teeth, Jaws).

```
   double jawsArray[];
   double teethArray[];
   double lipsArray[];
```

Sorting data in these arrays by using the "ArraySetAsSeries" function. Its parameters:

- array\[\]
- flag

```
   ArraySetAsSeries(jawsArray,true);
   ArraySetAsSeries(teethArray,true);
   ArraySetAsSeries(lipsArray,true);
```

Defining the Alligator by using the "iAlligator" function. Its parameters:

- symbol: we will use (\_Symbol) to be applied to the current symbol.
- period: we will use (\_Period) to be applied to the current time frame.
- jaw\_period: to determine the period of the calculation of jaws, we will use (13).
- jaw\_shift: to determine the horizontal shift of jaws, we will use (8).
- teeth\_period: to determine the period of the calculation of teeth, we will use (8).
- teeth\_shift: to determine the horizontal shift of teeth, we will use (5).
- lips\_period: to determine the period of the calculation of lips, we will use (5).
- lips\_shift: to determine the horizontal shift of lips, we will use (3).
- ma\_method: to determine the type of moving average, we will use(MODE\_SMA).
- applied\_price: to determine the type of price, we will use (PRICE\_MEDIAN).

```
int alligatorDef=iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN);
```

Defining data and storing results by using the "CopyBuffer" function. Its parameters:

- indicator\_handle: to determine the indicator handle, we will use (alligatorDef).
- buffer\_num: to determine the indicator buffer number, we will use (0 for jaws), (1 for teeth), and (2 for lips).
- start\_pos: to determine the start position, we will determine (0).
- count: to determine the amount to copy, we will use (3).
- buffer\[\]: to determine the target array to copy, we will use (jawsArray, teethArray, lipsArray).

```
   CopyBuffer(alligatorDef,0,0,3,jawsArray);
   CopyBuffer(alligatorDef,1,0,3,teethArray);
   CopyBuffer(alligatorDef,2,0,3,lipsArray);
```

Getting values of three components.

```
   double jawsValue=NormalizeDouble(jawsArray[0],5);
   double teethValue=NormalizeDouble(teethArray[0],5);
   double lipsValue=NormalizeDouble(lipsArray[0],5);
```

Commenting with three components of the indicator.

```
   Comment("jawsValue = ",jawsValue,"\n",
   "teethValue = ",teethValue,"\n",
   "lipsValue = ",lipsValue);
```

The following is the full code to create this trading system:

```
//+------------------------------------------------------------------+
//|                                      Simple Alligator System.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//--------------------------------------------------------------------
void OnTick()
  {
   //creating price array
   double jawsArray[];
   double teethArray[];
   double lipsArray[];
   //Sorting data
   ArraySetAsSeries(jawsArray,true);
   ArraySetAsSeries(teethArray,true);
   ArraySetAsSeries(lipsArray,true);
   //define Alligator
   int alligatorDef=iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN);
   //define data and store result
   CopyBuffer(alligatorDef,0,0,3,jawsArray);
   CopyBuffer(alligatorDef,1,0,3,teethArray);
   CopyBuffer(alligatorDef,2,0,3,lipsArray);
   //get value of current data
   double jawsValue=NormalizeDouble(jawsArray[0],5);
   double teethValue=NormalizeDouble(teethArray[0],5);
   double lipsValue=NormalizeDouble(lipsArray[0],5);
   //comment on the chart
   Comment("jawsValue = ",jawsValue,"\n",
   "teethValue = ",teethValue,"\n",
   "lipsValue = ",lipsValue);
  }
//+------------------------------------------------------------------+
```

After compiling this code, we will find it in the navigator among Expert Advisors folder the same as the following:

![Simple Alligator System nav](https://c.mql5.com/2/49/Simple_Alligator_System_nav.png)

By dragging and dropping it on the desired chart, we will find the window of this EA the same as the following:

![Simple Alligator System win](https://c.mql5.com/2/49/Simple_Alligator_System_win.png)

After pressing "OK" after ticking next to "Allow Algo Trading", we will the EA attached to the chart the same as the following:

![Simple Alligator System attached](https://c.mql5.com/2/49/Simple_Alligator_System_attached.png)

As we can on the previous chart in the top right corner that the EA is attached. Now, we're ready to receive signals. The following is an example from testing to check for generated signals.

![ Simple Alligator System signal](https://c.mql5.com/2/49/Simple_Alligator_System_signal.png)

As we can see in the previous chart, we have a comment on the top left corner with the following values:

Jaws value = n

teeth value = n

lips value = n

**Strategy one: Alligator Trend Identifier**

We can create a trading system based on this strategy the same as the following full code:

```
//+------------------------------------------------------------------+
//|                                   Alligator Trend Identifier.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
//creating three arrays of Alligator components
   double jawsArray[];
   double teethArray[];
   double lipsArray[];
//Sorting data
   ArraySetAsSeries(jawsArray,true);
   ArraySetAsSeries(teethArray,true);
   ArraySetAsSeries(lipsArray,true);
//define Alligator
   int alligatorDef=iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN);
//define data and store result
   CopyBuffer(alligatorDef,0,0,13,jawsArray);
   CopyBuffer(alligatorDef,1,0,13,teethArray);
   CopyBuffer(alligatorDef,2,0,13,lipsArray);
//get value of current data
   double jawsValue=NormalizeDouble(jawsArray[0],5);
   double teethValue=NormalizeDouble(teethArray[0],5);
   double lipsValue=NormalizeDouble(lipsArray[0],5);
//conditions of strategy
   if(lipsValue>teethValue && lipsValue>jawsValue && teethValue>jawsValue)
     {
      Comment("Bullish","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
   if(lipsValue<teethValue && lipsValue<jawsValue && teethValue<jawsValue)
     {
      Comment("Bearish","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code.

Conditions of the strategy, in case of bullish:

```
   if(lipsValue>teethValue && lipsValue>jawsValue && teethValue>jawsValue)
     {
      Comment("Bullish","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
```

In case of bearish:

```
   if(lipsValue<teethValue && lipsValue<jawsValue && teethValue<jawsValue)
     {
      Comment("Bearish","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
```

After compiling this code, and executing the created expert the same as we learned, we will find this EA is attached to the chart the same as the following:

![Alligator Trend Identifier attached](https://c.mql5.com/2/49/Alligator_Trend_Identifier_attached.png)

As we can see in the previous chart that we have the EA is attached to the chart in the top right corner and we're ready to get our bullish or bearish signals based on this strategy. The following are examples of these signals:

In case of bullish:

![Alligator Trend Identifier - bullish signal](https://c.mql5.com/2/49/Alligator_Trend_Identifier_-_bullish_signal.png)

As we can see in the previous chart, we  have a comment as a signal on the top right corner with the following values:

- Bullish
- Jaws value = n
- teeth value = n
- lips value = n

We have three lines moving below prices. So, we got a bullish signal.

In case of bearish:

![ Alligator Trend Identifier - bearish signal](https://c.mql5.com/2/49/Alligator_Trend_Identifier_-_bearish_signal.png)

As we can see in the previous chart, we  have a comment as a signal of bearish on the top right corner with the following values:

- Bearish
- Jaws value = n
- teeth value = n
- lips value = n

We have three lines moving above prices, So, we got a bearish signal.

**Strategy two: Alligator Signals System**

The following code is for creating a trading system based on this strategy:

```
//+------------------------------------------------------------------+
//|                                    Alligator Signals System .mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
//creating three arrays of Alligator components
   double jawsArray[];
   double teethArray[];
   double lipsArray[];
//Sorting data
   ArraySetAsSeries(jawsArray,true);
   ArraySetAsSeries(teethArray,true);
   ArraySetAsSeries(lipsArray,true);
//define Alligator
   int alligatorDef=iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN);
//define data and store result
   CopyBuffer(alligatorDef,0,0,13,jawsArray);
   CopyBuffer(alligatorDef,1,0,13,teethArray);
   CopyBuffer(alligatorDef,2,0,13,lipsArray);
//get value of current data
   double jawsValue=NormalizeDouble(jawsArray[0],5);
   double teethValue=NormalizeDouble(teethArray[0],5);
   double lipsValue=NormalizeDouble(lipsArray[0],5);
//conditions of strategy
   if(lipsValue>teethValue && lipsValue>jawsValue)
     {
      Comment("Buy","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
   if(lipsValue<teethValue && lipsValue<jawsValue)
     {
      Comment("Sell","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code.

Conditions of this strategy.

In case of buy signal:

```
   if(lipsValue>teethValue && lipsValue>jawsValue)
     {
      Comment("Buy","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
```

In case of sell signal:

```
   if(lipsValue<teethValue && lipsValue<jawsValue)
     {
      Comment("Sell","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "lipsValue = ",lipsValue);
     }
```

After compiling this code and executing it in the trading terminal, we can find it is attached to the chart the same as the following:

![Alligator Signals System attached](https://c.mql5.com/2/49/Alligator_Signals_System_attached.png)

We see on the top right corner in the previous chart that the EA is attached and we're ready to receive our buy or sell signals based on this strategy the following are for examples from testing.

In case of buy signal:

![Alligator Signals System - buy signal](https://c.mql5.com/2/49/Alligator_Signals_System_-_buy_signal.png)

As we can see that we have our desired signal with the following values:

- Buy
- Jaws value = n
- teeth value = n
- lips value = n

We have three lines moving below prices, So, we got a buy signal.

In case of sell signal:

![ Alligator Signals System - sell signal](https://c.mql5.com/2/49/Alligator_Signals_System_-_sell_signal.png)

We have our desired signal with the following values:

- Sell
- Jaws value = n
- teeth value = n
- lips value = n

We have three lines that are moving above prices, So, we got a sell signal.

**Strategy three: Alligator Signals System Enhancement**

The following is for the full code to create a trading system based on this strategy:

```
//+------------------------------------------------------------------+
//|                         Alligator Signals System Enhancement.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
//creating three arrays of Alligator components
   double jawsArray[];
   double teethArray[];
   double lipsArray[];
   MqlRates pArray[];
//Sorting data
   ArraySetAsSeries(jawsArray,true);
   ArraySetAsSeries(teethArray,true);
   ArraySetAsSeries(lipsArray,true);
   int Data=CopyRates(_Symbol,_Period,0,1,pArray);
//define Alligator
   int alligatorDef=iAlligator(_Symbol,_Period,13,8,8,5,5,3,MODE_SMMA,PRICE_MEDIAN);
//define data and store result
   CopyBuffer(alligatorDef,0,0,13,jawsArray);
   CopyBuffer(alligatorDef,1,0,13,teethArray);
   CopyBuffer(alligatorDef,2,0,13,lipsArray);
//get value of current data
   double jawsValue=NormalizeDouble(jawsArray[0],5);
   double teethValue=NormalizeDouble(teethArray[0],5);
   double lipsValue=NormalizeDouble(lipsArray[0],5);
//conditions of strategy
   if(lipsValue<teethValue && lipsValue<jawsValue && pArray[0].close>teethValue)
     {
      Comment("Buy","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "closingPrice = ",pArray[0].close,"\n",
              "lipsValue = ",lipsValue);
     }
   if(lipsValue>teethValue && lipsValue>jawsValue && pArray[0].close<teethValue)
     {
      Comment("Sell","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "closingPrice = ",pArray[0].close,"\n",
              "lipsValue = ",lipsValue);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code.

Creating one more array for prices by using the "MqlRates" function to store information about prices, volumes, and spread.

```
MqlRates pArray[];
```

Getting the historical data of MqlRates by using the "CopyRates". Its parameters:

- symbol\_name: to determine the symbol name, we will use (\_Symbol).
- timeframe: to determine the period, we will use (\_period).
- start\_pos: to determine the start position, we will use (0).
- count: to determine the data count to copy, we will use (1).
- rates\_array\[\]: to determine the target array to copy, we will use (pArray).

```
int Data=CopyRates(_Symbol,_Period,0,1,pArray);
```

Conditions of the strategy.

In case of buy signal:

```
   if(lipsValue<teethValue && lipsValue<jawsValue && pArray[0].close>teethValue)
     {
      Comment("Buy","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "closingPrice = ",pArray[0].close,"\n",
              "lipsValue = ",lipsValue);
     }
```

In case of sell signal:

```
   if(lipsValue>teethValue && lipsValue>jawsValue && pArray[0].close<teethValue)
     {
      Comment("Sell","\n",
              "jawsValue = ",jawsValue,"\n",
              "teethValue = ",teethValue,"\n",
              "closingPrice = ",pArray[0].close,"\n",
              "lipsValue = ",lipsValue);
     }
```

After compiling this code and executing the EA we will find it attached to the chart the same as the following to get our buy and sell signals.

![ Alligator Signals System Enhancement attached](https://c.mql5.com/2/49/Alligator_Signals_System_Enhancement_attached.png)

We have the EA attached in the top right corner and we can receive our desired signals the following are examples from testing.

In case of buy signal:

![ Alligator Signals System Enhancement - buy signal](https://c.mql5.com/2/49/Alligator_Signals_System_Enhancement_-_buy_signal.png)

We have a comment in the top left corner with the following values:

- Buy
- Jaws value = n
- teeth value = n
- Closing price = n
- lips value = n

As three lines are moving above prices then we can find that the closing price is closed above the Alligator teeth line.

In case of sell signal:

![ Alligator Signals System Enhancement - sell signal](https://c.mql5.com/2/49/Alligator_Signals_System_Enhancement_-_sell_signal.png)

We have a comment as a signal in the top left corner:

- Sell
- Jaws value = n
- teeth value = n
- Closing price = n
- lips value = n

As three lines are moving below prices then we can find that the closing price is closed below the Alligator teeth line.

### Conclusion

We learned in this article in detail about the Alligator technical indicator which can be used to confirm the trend in addition to generating buy and sell signals. We learned how we can calculate it manually, and how we can use based on the simple mentioned strategies:

- Alligator Trend Identifier: To generate bullish or bearish signals based on the position of the three lines of Alligator (Lips, Teeth, and Jaws).
- Alligator Signals System: To generate buy or sell signals by classical method crossover based on the crossover between the three lines of the indicator.
- Alligator Signals System Enhancement: To generate buy or sell signals by another method to get these signals earlier based on the position of three lines of the indicators and the crossover between the closing price and the teeth.

We designed a step-by-step blueprint for each mentioned strategy to help us to organize our ideas to create a trading system smoothly, easily, and effectively. We also created a trading system for each mentioned strategy to be used in the MetaTrader 5 trading terminal and generate signals automatically as per what we designed and coded. I need to confirm here again that you must test any mentioned strategy before using it on your real account to make sure that they are useful and profitable for you as there is nothing is suitable for all of us in addition to that the main objective here is to share knowledge or the main purpose of this article is education only.

I hope that you tried to apply what you learned by yourself the same I advised you to improve your coding skills and get complete benefit from reading this article and I hope that you found it useful for your trading and got useful insights into the topic of this article or even any related topic. If you want to read more similar articles, you can read my previous articles in this series as we shared the most popular technical indicators like stochastic, RSI, Bollinger Bands, Moving Averages, Envelopes, MACD, ADX, etc., and how we can create a trading system based on them.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11549.zip "Download all attachments in the single ZIP archive")

[Simple\_Alligator\_System.mq5](https://www.mql5.com/en/articles/download/11549/simple_alligator_system.mq5 "Download Simple_Alligator_System.mq5")(1.49 KB)

[Alligator\_Trend\_Identifier.mq5](https://www.mql5.com/en/articles/download/11549/alligator_trend_identifier.mq5 "Download Alligator_Trend_Identifier.mq5")(1.88 KB)

[Alligator\_Signals\_System.mq5](https://www.mql5.com/en/articles/download/11549/alligator_signals_system.mq5 "Download Alligator_Signals_System.mq5")(1.83 KB)

[Alligator\_Signals\_System\_Enhancement.mq5](https://www.mql5.com/en/articles/download/11549/alligator_signals_system_enhancement.mq5 "Download Alligator_Signals_System_Enhancement.mq5")(2.06 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/434432)**
(5)


![panicle](https://c.mql5.com/avatar/avatar_na2.png)

**[panicle](https://www.mql5.com/en/users/panicle)**
\|
12 Oct 2022 at 20:59

**MetaQuotes:**

New article [Learn how to design a trading system by Alligator](https://www.mql5.com/en/articles/11549) has been published:

Author: [Mohamed Abdelmaaboud](https://www.mql5.com/en/users/M.Aboud "M.Aboud")

Hi thanks for this trading tutorial, where do you actually begin when designing this system, I am fairly new in trading


![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
15 Nov 2022 at 09:33

Hi,

I found that this link of your article is not valid: [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/11549/118781/edit%20en=). One get a 404 error.

Beside that this is wrong:

We have three lines moving below prices, So, we got a bearish signal.

We have three lines moving **below** prices, **So**, we got a bearish signal.

It should be (imho):

We have three lines moving **above** prices, **so**, we got a bearish signal

Please can you correct it?

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
15 Nov 2022 at 14:30

**Carl Schreiber [#](https://www.mql5.com/en/forum/434432#comment_43248275):**

Hi,

I found that this link of your article is not valid: [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/11549/118781/edit%20en=). One get a 404 error.

Beside that this is wrong:

We have three lines moving below prices, So, we got a bearish signal.

It should be (imho):

Please can you correct it?

Thanks for your note and I will correct them.


![Sergiy Kolesnyk](https://c.mql5.com/avatar/2020/7/5F23F478-6A8E.png)

**[Sergiy Kolesnyk](https://www.mql5.com/en/users/sergiykolesnyk)**
\|
23 Jan 2023 at 03:37

**Mohamed Abdelmaaboud [#](https://www.mql5.com/en/forum/434432#comment_43253647):**

Thanks for your note and I will correct them.

it still is 404


![xielp](https://c.mql5.com/avatar/avatar_na2.png)

**[xielp](https://www.mql5.com/en/users/xielp)**
\|
28 Feb 2023 at 15:33

[Backtesting](https://www.mql5.com/en/articles/2612 "Article: Testing trading strategies on real ticks ") doesn't look good.


![DoEasy. Controls (Part 14): New algorithm for naming graphical elements. Continuing work on the TabControl WinForms object](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__2.png)[DoEasy. Controls (Part 14): New algorithm for naming graphical elements. Continuing work on the TabControl WinForms object](https://www.mql5.com/en/articles/11288)

In this article, I will create a new algorithm for naming all graphical elements meant for building custom graphics, as well as continue developing the TabControl WinForms object.

![DoEasy. Controls (Part 13): Optimizing interaction of WinForms objects with the mouse, starting the development of the TabControl WinForms object](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__1.png)[DoEasy. Controls (Part 13): Optimizing interaction of WinForms objects with the mouse, starting the development of the TabControl WinForms object](https://www.mql5.com/en/articles/11260)

In this article, I will fix and optimize handling the appearance of WinForms objects after moving the mouse cursor away from the object, as well as start the development of the TabControl WinForms object.

![Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://c.mql5.com/2/48/Neural_networks_made_easy_023.png)[Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://www.mql5.com/en/articles/11273)

In this series of articles, we have already mentioned Transfer Learning more than once. However, this was only mentioning. in this article, I suggest filling this gap and taking a closer look at Transfer Learning.

![Neural networks made easy (Part 22): Unsupervised learning of recurrent models](https://c.mql5.com/2/48/Neural_networks_made_easy_022.png)[Neural networks made easy (Part 22): Unsupervised learning of recurrent models](https://www.mql5.com/en/articles/11245)

We continue to study unsupervised learning algorithms. This time I suggest that we discuss the features of autoencoders when applied to recurrent model training.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/11549&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069197540148052329)

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