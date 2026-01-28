---
title: Learn how to design a trading system by Relative Vigor Index
url: https://www.mql5.com/en/articles/11425
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:10:45.711370
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/11425&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069202303266783604)

MetaTrader 5 / Trading


### Introduction

Welcome to my new article in our series to learn new technical tools in detail and design a simple trading system with this tool. It is the Relative Vigor Index (RVI) indicator, we will cover it through the following topics:

1. [Relative Vigor Index definition](https://www.mql5.com/en/articles/11425#definition)
2. [Relative Vigor Index strategy](https://www.mql5.com/en/articles/11425#strategy)
3. [Relative Vigor Index strategy blueprint](https://www.mql5.com/en/articles/11425#blueprint)
4. [Relative Vigor Index trading system](https://www.mql5.com/en/articles/11425#system)
5. [Conclusion](https://www.mql5.com/en/articles/11425#conclusion)

We will use MQL5 (MetaQuotes Language 5) IDE to write our codes and the MetaTrader 5 trading terminal to execute our trading system. I advise you to try to apply what you will learn by yourself if you want to improve your coding skills as a beginner because practice is very important to do. If you do not know how to download and use MetaTrader 5 and MQL5, you can read the topic of [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) from a previous article for me.

All mentioned strategies are for education purposes only, so, you may find that they need optimization and this is normal as the main objective of this article and these simple strategies is to learn the main concept behind it. So, you must test them very well before using them in your real account to make sure that they will be useful for your trading.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Relative Vigor Index definition

In this topic, we will learn what is the Relative Vigor Index (RVI) in detail. It is a momentum and Oscillator technical indicator. By comparing the closing price to its opening level, this indicator measures the strength of the trend. During the uptrend, the closing price closes higher than the open price but during the downtrend, the closing price closes lower than the open price. In addition to that, the results were smoothed by using the simple moving average. This RVI indicator is performing better during trending markets than ranging periods according to its nature. We can find that the calculation of this RVI is similar to the Stochastic Oscillator but the RVI is comparing the closing price to its opening one and the Stochastic Oscillator is comparing the closing to its price range during a specific period. If we want to calculate the RVI manually, we can do that through the following steps:

RVI = (CLOSE - OPEN) / (HIGH - LOW)

Where:

CLOSE = Closing price

OPEN = Opening price

HIGH = Highest price

LOW = Lowest price

But usually, the RVI that we use appears as two lines with different components in calculation and the following is for how we can calculate them.

1- 1st line Calculation:

- Nominator calculation:

MovAverage = (CLOSE-OPEN) + 2 \* (CLOSE-1 - OPEN-1) + 2 \* (CLOSE-2 - OPEN-2) + (CLOSE-3 - OPEN-3)

Where:

CLOSE = Current closing price

CLOSE 1, CLOSE 2, CLOSE 3 = Closing prices of 1, 2, and 3 periods ago

OPEN = Current opening price

OPEN 1, OPEN 2, OPEN 3 = Opening prices of 1, 2, and 3 periods ago

- Denominator calculation:

RangeAverage = (HIGH-LOW) + 2 x (HIGH-1 - LOW-1) + 2 x (HIGH-2 - LOW-2) + (HIGH-3 - LOW-3)

Where:

HIGH = Last Maximal Price

HIGH, HIGH 2, HIGH 3 = Last Maximal Prices of 1, 2, and 3 periods ago

LOW = Last Minimal Price

LOW 1, LOW 2, LOW 3 = Last Minimal Prices of 1, 2, and 3 periods ago

- Calculate the sum of these averages the same as the following:

![RVI average](https://c.mql5.com/2/48/Screen_Shot_2022-09-11_at_2.02.58_PM.png)

2- 2nd line calculation:

RVIsignal = (RVIaverage + 2 \* RVIaverage-1 + 2 \* RVIaverage-2 + RVIaverage-3)/6

Now, we understood the main concept behind this indicator after learning how to calculate it manually but we do not need to do that because it is calculated in the MetaTrader 5 trading terminal and all that we need to do is to choose it from the available indicators. The following is for how to insert it into the chart:

While opening the MetaTrader 5 terminal --> Insert --> Indicators --> Oscillators --> Relative Vigor Index

![RVI insert](https://c.mql5.com/2/48/RVI_insert.png)

After pressing on the Relative Vigor Index, we will see the window of its parameters the same as the following:

![RVI param](https://c.mql5.com/2/48/RVI_param.png)

1- To determine the period.

2- To determine the color of the RVI line.

3- To determine the style of the RVI line.

4- To determine the thickness of the RVI line.

5- To determine the color of the signal line.

6- To determine the style of the signal line.

7- To determine the thickness of the signal line.

After determining all of the parameters, we will find that the indicator is attached the same as the following:

![RVI attached](https://c.mql5.com/2/48/RVI_attached.png)

As we can see in the previous chart that we have the RVI indicator is attached to the chart and it has two lines oscillating around the zero level.

### Relative Vigor Index strategy

After learning the main concept behind the RVI, we will learn how to use it based on its main concept through simple strategies. What I need to confirm here again is to make sure to test any strategy before using it on your real account to make sure that it is suitable for your trading and profitable as the main concept here is to share simple strategies for educational purposes only and it is normal that you find that they may need optimizations.

- Strategy one: RVI Crossover - Uptrend:

Based on this strategy, we need to get buy and close signals during the uptrend by a specific condition. When the RVI current value and RVI signal current value are greater than the zero level at the same time that RVI current value is greater than the current value of the RVI signal, this will be a buy signal. Vice Versa, when the RVI current value and RVI signal current value is below zero level at the same time that the RVI current value is below the current value of the RVI signal, this will be a close signal.

Simply,

RVI value > 0 and RVI signal value > 0 and RVI value > RVI signal value --> buy

RVI value < 0 and RVI signal value < 0 and RVI value < RVI signal value --> close

- Strategy two: RVI Crossover - Downtrend:

Based on this strategy, we need to get the opposite signals of the previous RVI Crossover - Uptrend strategy as we need to get short and cover signals. When the RVI current value and RVI signal current value are lower than the zero level at the same time that RVI current value is lower than the current value of the RVI signal, this will be a short signal. Vice Versa, when the RVI current value and RVI signal current value is above the zero level at the same time that the RVI current value is above the current value of the RVI signal, this will be a cover signal.

Simply,

RVI value < 0 and RVI signal value < 0 and RVI value < RVI signal value --> short

RVI value > 0 and RVI signal value > 0 and RVI value > RVI signal value --> cover

- Strategy three: RVI and MA Crossover:

Based on this strategy, we need to get buy and sell signals based on specific conditions as we need to get a buy signal when the closing price is greater than the 100 -period moving average at the same time that the current RVI value is greater than the current RVI signal value. In the other scenario, we need to get a sell signal when the closing price is lower than the 100 -period moving average at the same time that the current RVI value is lower than the current RVI signal value.

Simply,

Closing price >100- Moving average and RVI value > RVI signal value --> buy

Closing price <100- Moving average and RVI value < RVI signal value --> sell

### Relative Vigor Index strategy blueprint

After learning how we can use the RVI through our mentioned strategy, we will create a step-by-step blueprint for each strategy to help us to create a trading system smoothly.

- Strategy one: RVI Crossover - Uptrend:

According to this strategy, we need to create a trading system that gives buy and close signals during the uptrend by continuously checking the following values:

- The current RVI value
- The current RVI signal value
- The zero level of the RVI indicator

If the current RVI value is greater than the zero level, the current RVI signal value is greater than the zero level, and the current RVI value is greater than the current RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Buy Signal
- Relative Vigor Index Value
- RVI Signal Value

In the other scenario, if the current RVI value is lower than the zero level, the current RVI signal value is lower than the zero level, and the current RVI value is lower than the current RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Close Signal
- Relative Vigor Index Value
- RVI Signal Value

The following is the step-by-step blueprint for this strategy:

![RVI crossover - U blueprint](https://c.mql5.com/2/48/RVI_crossover_-_U_blueprint.png)

- Strategy two: RVI Crossover - Downtrend:

According to this strategy, we need to create a trading system that can give us signals of short and cover during the downtrend by continuously checking the same values of the previous strategy:

- The current RVI value
- The current RVI signal value
- The zero level of the RVI indicator

But conditions will be slightly different as per the desired signals. If the current RVI value is lower than the zero level, the current RVI signal value is lower than the zero level at the same time, the RVI value is lower than the RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Short Signal
- Relative Vigor Index Value
- RVI Signal Value

On the other hand, if the current RVI value is higher than the zero level, the current RVI signalvalue is higher than the zero level at the same time, the RVI value is higher than the RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Cover
- Relative Vigor Index Value
- RVI Signal Value

The following is the blueprint of this trading system:

![RVI crossover - D blueprint](https://c.mql5.com/2/48/RVI_crossover_-_D_blueprint.png)

- Strategy three: RVI & MA Crossover

According to this strategy, we need to create a trading system that gives buy and sell signals by continuously checking the following values:

- The closing price
- The 100- period moving average (MA)
- The current RVI value
- The current RVI signal value

If the closing price is higher than the 100- period MA and the current RVI value is higher than the current RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Buy Signal
- Closing price
- MA Value
- Relative Vigor Index Value
- RVI Signal Value

On the other hand,  If the closing price is lower than the 100- period MA and the current RVI value is lower than the current RVI signal value, we need the trading system to return the following values as a comment on the chart:

- Sell Signal
- Closing price
- MA Value
- Relative Vigor Index Value
- RVI Signal Value

The following is the blueprint of this strategy:

![ RVI, MA crossover blueprint](https://c.mql5.com/2/48/RVIo_MA_crossover_blueprint.png)

### Relative Vigor Index trading system

After learning how to use the RVI through simple strategies and creating a step-by-step blueprint for each strategy to help us to create our trading system. Now, we will create a trading system for each mentioned strategy. First, we will create a simple trading system that returns the current RVI value as a comment on the chart and the following is for the code to create this trading system step by step.

Creating two arrays for the RVI and RVI signal by using the double function:

```
   double rviArray[];
   double rviSignalArray[];
```

Setting the AS\_SERIES flag to arrays to return a true or false which is a boolean value by using the "ArraySetAsSeries" function:

```
   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);
```

Defining the RVI by using the "iRVI" function that returns the handle of the Relative Vigor Index indicator. Its parameters are:

- symbol: to determine the symbol name, we will use (\_Symbol) to be applied to the current symbol.
- period: to set the useable period, we will use (\_period) to be applied to the current timeframe.
-  ma\_period: to set the averaging period, we will use(10).

```
int rviDef = iRVI(_Symbol,_Period,10);
```

Getting data from the buffer of the RVI indicator by using the "CopyBuffer" function. Its parameters are:

- indicator\_handle: we will use the (rviDef, rviDef) of the predefined indicator handle.
- buffer\_num: to determine the indicator buffer number, we will use (0, 1).
- start\_pos: to determine the starting position, we will use (0, 0).
- count: to determine the amount to copy, we will use (3, 3).
- buffer\[\]: to determine the target array to copy, we will use (rviArray, rviSignalArray).

```
   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);
```

Defining the values of RVI and RVI signal by using the "NormalizeDouble" function to return a value of double type. Its parameters are:

- value: to determine the normalized number. We will use (rviArray\[0\], rviSignalArray\[0\]).
- digits: to determine the number of digits after the decimal. We will use (3).

```
   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);
```

Using the "comment" function to return the value of the current RVI indicator as a comment on the chart.

```
   Comment("Relative Vigor Index Value is ",rviVal,"\n",
           "RVI Signal Value is ",rviSignalVal);
```

The full code will be the same as the following:

```
//+------------------------------------------------------------------+
//|                                                   Simple RVI.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double rviArray[];
   double rviSignalArray[];

   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);

   int rviDef = iRVI(_Symbol,_Period,10);

   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);

   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);

   Comment("Relative Vigor Index Value is ",rviVal,"\n",
           "RVI Signal Value is ",rviSignalVal);

  }
//+------------------------------------------------------------------+
```

After compiling this code, we will find the expert in the navigator window the same as the following:

![ RVI nav](https://c.mql5.com/2/48/RVI_nav.png)

By double-clicking the expert to execute it in the trading terminal, we will find the window of this expert the same as the following:

![Simple RVI win](https://c.mql5.com/2/48/Simple_RVI_win.png)

By pressing "OK" after ticking next to the "Allow Algo Trading", we will find the expert is attached t the chart the same as the following:

![ Simple RVI attached](https://c.mql5.com/2/48/Simple_RVI_attached.png)

Now, we're ready to get generated signals based on this trading system. So, we will get a signal the same as the following:

![Simple RVI signal](https://c.mql5.com/2/48/Simple_RVI_signal.png)

As we can see on the top left corner of the chart we have a comment with the current RVI value.

- Strategy one: RVI Crossover - Uptrend:

The following is the full code to create a trading system based on this strategy:

```
//+------------------------------------------------------------------+
//|                                      RVI Crossover - Uptrend.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double rviArray[];
   double rviSignalArray[];

   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);

   int rviDef = iRVI(_Symbol,_Period,10);

   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);

   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);

   if(rviVal>0 && rviSignalVal>0 && rviVal>rviSignalVal)
     {
      Comment("Buy Signal","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
   if(rviVal<0 && rviSignalVal<0 && rviVal<rviSignalVal)
     {
      Comment("Close","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }

  }
//+------------------------------------------------------------------+
```

Differences in this code.

Conditions of the strategy:

In case of buy signal

```
   if(rviVal>0 && rviSignalVal>0 && rviVal>rviSignalVal)
     {
      Comment("Buy Signal","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

In case of close

```
   if(rviVal<0 && rviSignalVal<0 && rviVal<rviSignalVal)
     {
      Comment("Close","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

After compiling this code and executing it to the trading terminal in the same way as what we mentioned in the simple RVI trading system, we will find the expert is attached to the chart the same as the following:

![RVI crossover - U attached](https://c.mql5.com/2/48/RVI_crossover_-_U_attached.png)

Then, we can get desired signals based on this trading system the same as the following:

In case of buy signal:

![ RVI crossover - U - buy signal](https://c.mql5.com/2/48/RVI_crossover_-_U_-_buy_signal.png)

As we can see on the chart in the top left corner that we received a signal with the following values:

- Buy Signal
- Relative Vigor Index value
- RVI signal value

In case of close signal:

![ RVI crossover - U - close signal](https://c.mql5.com/2/48/RVI_crossover_-_U_-_close_signal.png)

As we can see on the chart in the top left corner that we received a signal with the following values:

- Close
- Relative Vigor Index value
- RVI signal value

It is clear that this signal is the same as what we need based on our mentioned trading strategy and its conditions during the uptrend.

- Strategy two: RVI Crossover - Downtrend:

The following is the full code to create a trading system for this trading strategy:

```
//+------------------------------------------------------------------+
//|                                    RVI Crossover - Downtrend.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
void OnTick()
  {
   double rviArray[];
   double rviSignalArray[];

   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);

   int rviDef = iRVI(_Symbol,_Period,10);

   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);

   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);

   if(rviVal<0 && rviSignalVal<0 && rviVal<rviSignalVal)
     {
      Comment("Short Signal","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
   if(rviVal>0 && rviSignalVal>0 && rviVal>rviSignalVal)
     {
      Comment("Cover","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code.

Conditions of the strategy:

In case of short signal

```
   if(rviVal<0 && rviSignalVal<0 && rviVal<rviSignalVal)
     {
      Comment("Short Signal","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

In case of cover

```
   if(rviVal>0 && rviSignalVal>0 && rviVal>rviSignalVal)
     {
      Comment("Cover","\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

After compiling and executing this code the same as what we learned before, we will find that the expert is attached to the chart the same as the following:

![ RVI crossover - D attached](https://c.mql5.com/2/48/RVI_crossover_-_D_attached.png)

Now, we're ready to receive signals based on this strategy and trading system. The following are examples of signals from testing:

In case of short signal:

![ RVI crossover - D - short signa](https://c.mql5.com/2/48/RVI_crossover_-_D_-_short_signal.png)

As we can see on the chart in the top left corner that we received a signal with the following values:

- Short signal
- Relative Vigor Index value
- RVI signal value

In case of cover:

![RVI crossover - D - cover signa](https://c.mql5.com/2/48/RVI_crossover_-_D_-_cover_signal.png)

As we can see that we have a signal of the following values:

- Cover
- Relative Vigor Index value
- RVI signal value

Now, we have created a trading system based on the strategy of RVI Crossover - Downtrend to generate signals during the downtrend.

- Strategy three: RVI & MA Crossover:

The following is the full code to create a trading system for this trading strategy:

```
//+------------------------------------------------------------------+
//|                                           RVI & MA Crossover.mq5 |
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
   double maArray[];
   double rviArray[];
   double rviSignalArray[];

   int Data=CopyRates(_Symbol,_Period,0,1,pArray);
   ArraySetAsSeries(maArray,true);
   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);

   int rviDef = iRVI(_Symbol,_Period,10);
   int maDef = iMA(_Symbol,_Period,100,0,MODE_EMA,PRICE_CLOSE);

   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);
   CopyBuffer(maDef,0,0,3,maArray);

   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);
   double maVal = NormalizeDouble(maArray[0],3);

   if(pArray[0].close>maVal && rviVal>rviSignalVal)
     {
      Comment("Buy Signal","\n",
              "Closing price is ",pArray[0].close,"\n",
              "MA Value is ",maVal,"\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
   if(pArray[0].close<maVal && rviVal<rviSignalVal)
     {
      Comment("Sell Signal","\n",
              "Closing price is ",pArray[0].close,"\n",
              "MA Value is ",maVal,"\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }

  }
//+------------------------------------------------------------------+
```

Differences in this code.

Creating four arrays (pArray, maArray, rviArray, rviSignalArray). We will use the double function the same as what we mentioned except for the pArray, we will use the MqlRates function which stores information about the price, volume, and spread.

```
   MqlRates pArray[];
   double maArray[];
   double rviArray[];
   double rviSignalArray[];
```

Setting the AS\_SERIES flag to arrays to return a true or false which is a boolean value by using the "ArraySetAsSeries" function. Defining Data by using the "CopyRates" function to get historical data of MqlRates structure and its parameters are:

- symbol\_name: to determine the symbol name, we will use \_Symbol to be applied to the current symbol.
- timeframe: to determine the period and we will use \_Period to be applied to the current period.
- start\_pos: to determine the starting position, we will use (0).
- count: to determine the data count to copy, we will use (1).
- rates\_array\[\]: to determine the target array to copy, we will use the pArray.

```
   int Data=CopyRates(_Symbol,_Period,0,1,pArray);
   ArraySetAsSeries(maArray,true);
   ArraySetAsSeries(rviArray,true);
   ArraySetAsSeries(rviSignalArray,true);
```

Defining RVI by using the iRVI function the same as we mentioned and Moving average by using the iMA function to return the handle of the moving average indicator and its parameters are:

- symbol: to determine the symbol name, we will use (\_Symbol)
- period: to determine the period of the timeframe, we will use (\_period)
- ma\_period: to determine the period of the moving average, we will use 100
- ma\_shift: to determine the horizontal shift, we will use (0)
- ma\_method: to determine the type of the moving average, we will use exponential MA
- applied\_price: to determine the type of price, we will use closing price

```
   int rviDef = iRVI(_Symbol,_Period,10);
   int maDef = iMA(_Symbol,_Period,100,0,MODE_EMA,PRICE_CLOSE);
```

Getting data from the buffer of the RVI and MA indicators by using the "CopyBuffer" function.

```
   CopyBuffer(rviDef,0,0,3,rviArray);
   CopyBuffer(rviDef,1,0,3,rviSignalArray);
   CopyBuffer(maDef,0,0,3,maArray);
```

Defining values of RVI, RVI signal, and MA value by using the NormalizeDouble function.

```
   double rviVal = NormalizeDouble(rviArray[0],3);
   double rviSignalVal = NormalizeDouble(rviSignalArray[0],3);
   double maVal = NormalizeDouble(maArray[0],3);
```

Conditions of the strategy.

In case of buy signal

```
   if(pArray[0].close>maVal && rviVal>rviSignalVal)
     {
      Comment("Buy Signal","\n",
              "Closing price is ",pArray[0].close,"\n",
              "MA Value is ",maVal,"\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

In case of sell signal

```
   if(pArray[0].close<maVal && rviVal<rviSignalVal)
     {
      Comment("Sell Signal","\n",
              "Closing price is ",pArray[0].close,"\n",
              "MA Value is ",maVal,"\n",
              "Relative Vigor Index Value is ",rviVal,"\n",
              "RVI Signal Value is ",rviSignalVal);
     }
```

After compiling this code and executing the expert in the terminal, we will find that the expert is attached to the chart the same as the following:

![ RVI, MA crossover attached](https://c.mql5.com/2/48/RVIz_MA_crossover_attached.png)

Now, we can get generated signals based on this trading system the same as the following examples from testing.

In case of buy signal:

![ RVI, MA crossover - buy signal](https://c.mql5.com/2/48/RVIc_MA_crossover_-_buy_signal.png)

As we can see in the previous chart in the top left corner that we have a signal with the following values:

- Buy Signal
- Closing price
- MA Value
- Relative Vigor Index Value
- RVI Signal Value

In case of sell signal:

![RVI, MA crossover - sell signal](https://c.mql5.com/2/48/RVIm_MA_crossover_-_sell_signal.png)

We have a signal with the following values:

- Sell Signal
- Closing price
- MA Value
- Relative Vigor Index Value
- RVI Signal Value

Now, we created a trading system for each mentioned strategy to generate an automated signal based on its conditions.

### Conclusion

After what we learned through previous topics in this article, it is supposed that we understood this indicator deeply as we learned about it in detail. We learned what it is, what it measures, and how we can calculate it manually to understand the main concept behind the RVI indicator. We learned how to use it through three simple strategies and they were:

- RVI Crossover - Uptrend: to generate buy and close signals during the uptrend.
- RVI Crossover - Downtrend: to generate short and cover signals during the downtrend.
- RVI & MA Crossover: to generate buy and sell signals based on a specific crossover technique.

We designed a step-by-step blueprint for each mentioned strategy to help us to create trading for them smoothly and easily. We created a trading system for each of mentioned strategies to generate automatic signals after executing them in MetaTrader 5.

I hope that you tried to apply what you learned by yourself to get the maximum benefit from this article and develop your programming skills. I hope also that you found this article useful for you to improve your trading results and get useful insights related to the topic of the article or any related topic. If you want to read more similar articles to learn how to design a trading system based on the most popular technical indicators, you can read my other article in this series about that.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11425.zip "Download all attachments in the single ZIP archive")

[Simple\_RVI.mq5](https://www.mql5.com/en/articles/download/11425/simple_rvi.mq5 "Download Simple_RVI.mq5")(1.12 KB)

[RVI\_Crossover\_-\_Uptrend.mq5](https://www.mql5.com/en/articles/download/11425/rvi_crossover_-_uptrend.mq5 "Download RVI_Crossover_-_Uptrend.mq5")(1.45 KB)

[RVI\_Crossover\_-\_Downtrend.mq5](https://www.mql5.com/en/articles/download/11425/rvi_crossover_-_downtrend.mq5 "Download RVI_Crossover_-_Downtrend.mq5")(1.45 KB)

[RVI\_f\_MA\_Crossover.mq5](https://www.mql5.com/en/articles/download/11425/rvi_f_ma_crossover.mq5 "Download RVI_f_MA_Crossover.mq5")(1.92 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/432690)**
(1)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
23 Nov 2022 at 07:13

Everything is very beautiful, beautiful, magnificent. Except for one thing - a guarantee of deposit drain


![The price movement model and its main provisions (Part 2): Probabilistic price field evolution equation and the occurrence of the observed random walk](https://c.mql5.com/2/47/StatLab-icon_12Litl.png)[The price movement model and its main provisions (Part 2): Probabilistic price field evolution equation and the occurrence of the observed random walk](https://www.mql5.com/en/articles/11158)

The article considers the probabilistic price field evolution equation and the upcoming price spike criterion. It also reveals the essence of price values on charts and the mechanism for the occurrence of a random walk of these values.

![DoEasy. Controls (Part 9): Re-arranging WinForms object methods, RadioButton and Button controls](https://c.mql5.com/2/47/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 9): Re-arranging WinForms object methods, RadioButton and Button controls](https://www.mql5.com/en/articles/11121)

In this article, I will fix the names of WinForms object class methods and create Button and RadioButton WinForms objects.

![Learn how to design a trading system by Awesome Oscillator](https://c.mql5.com/2/48/why-and-how__9.png)[Learn how to design a trading system by Awesome Oscillator](https://www.mql5.com/en/articles/11468)

In this new article in our series, we will learn about a new technical tool that may be useful in our trading. It is the Awesome Oscillator (AO) indicator. We will learn how to design a trading system by this indicator.

![CCI indicator. Upgrade and new features](https://c.mql5.com/2/47/new_oscillator.png)[CCI indicator. Upgrade and new features](https://www.mql5.com/en/articles/11126)

In this article, I will consider the possibility of upgrading the CCI indicator. Besides, I will present a modification of the indicator.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/11425&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069202303266783604)

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