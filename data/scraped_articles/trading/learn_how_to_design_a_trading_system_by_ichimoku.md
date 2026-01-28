---
title: Learn how to design a trading system by Ichimoku
url: https://www.mql5.com/en/articles/11081
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:12:45.041757
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/11081&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6413686624949830123)

MetaTrader 5 / Trading


### Introduction

Welcome to a new article in our series which can learn through it how to design a trading system by the most common technical indicators. In this article, we will learn a new technical tool in detail that can be used in our favor to learn how to create a simple trading system based on simple strategies based on the main concept or idea of this indicator.

We will cover this indicator through the following topics:

1. [Ichimoku definition](https://www.mql5.com/en/articles/11081#definition)
2. [Ichimoku strategy](https://www.mql5.com/en/articles/11081#strategy)
3. [Ichimoku strategy blueprint](https://www.mql5.com/en/articles/11081#blueprint)
4. [Ichimoku trading system](https://www.mql5.com/en/articles/11081#system)
5. [Conclusion](https://www.mql5.com/en/articles/11081#conclusion)

Throughout the topic of Ichimoku definition, we will learn what is Ichimoku, what is the construction of the indicator, how we can calculate it, and what it measures. So, we will learn the indicator in detail to learn the main concept behind it to be able to use it effectively. Ichimoku strategy will be the topic that we will learn through it, simple strategies based on the basic concept behind the indicator. Then, we will design a step-by-step strategy blueprint for each mentioned strategy to help us to create a trading system for them and we will learn that through the topic of Ichimoku strategy blueprint. In addition to that, we will learn the most interesting part in this article and the core of it is how we can create a trading system by MQL5 to execute it in the MetaTrader 5 trading platform for each mentioned strategy as we will learn that through the topic of Ichimoku trading system.

We will use the MetaQuotes Language (MQL5) which is a built-in editor in the MetaTrader 5. If you want to learn how to download the MetaTrader 5 to use the MetaEditor, you can read the [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) topic from a previous article.

I advise you to test everything and practice what you read as this will be very helpful to deepen your understanding or open your eyes to new ideas.

Disclaimer: All content of this article is made for the purpose of education only not for anything else. So, you will be responsible for any action you take based on the content of this article, as the content of this article does not guarantee any kind of results.

Now, let us start our learning journey through this new article and its topics.

### Ichimoku definition

In this part, we will learn the Ichimoku indicator in detail by identifying what is it, what it measures, its construction, and how we can calculate it. So, we can understand all that we need deeply about the indicator for effective use after that.

The Ichimoku indicator was developed by Goichi Hosoda. It is one of the Japanese school tools that can be used to get more insights into the financial instrument as we can identify the trend direction, support and resistance, momentum, and trading signals. If you want to know more about how to identify the trend, you can read the topic of [trend definition](https://www.mql5.com/en/articles/10715#trend) from a previous article.

It has five plots:

- Tenkan-Sen (Conversion Line)
- Kijun-Sen (Base Line)
- Senkou Span A (Leading Span A)
- Senkou Span B (Leading Span B)
- Chikou Span (Lagging Span)

Senkou Span A (Leading Span A) and Senkou Span B (Leading Span B) construct the cloud (Kumo).

Now, we need to learn how we can calculate the Ichimoku indicator manually to know the concept behind this indicator or use the indicator effectively. So, we will learn that by calculating every element from the Ichimoku.

Tenkan-Sen (Conversion Line)= (High of 9 periods + Low of 9 periods)/2

Kijun-Sen (Base Line)= (High of 26 periods + Low of 26 periods)/2

Senkou Span A (Leading Span A) = (Tenkan-Sen + Kijun-Sen)/2 but it will be plotted 26 periods in the future

Senkou Span B (Leading Span B)= (High of 52 periods + Low of 52 periods)/2 but it will be plotted 26 periods in the future also

Chikou Span (Lagging Span)= Close of today but it will be plotted 26 periods in the past

Nowadays, we do not need to calculate it manually but we learn it now to identify the concept behind the indicator. We have trading platforms that include this indicator and all we need is to select the Ichimoku among available indicators to be plotted on the chart without any calculation. Here, we need to insert the Ichimoku indicator to the chart in the MetaTrader5, we will press the Insert tab --> Indicators --> Ichimoku Kinko Hyo

By the way, Ichimoku called Ichimoku Kinko Hyo also.

![ Ichimoku insert](https://c.mql5.com/2/47/Ichimoku_insert.png)

After that, we will find the window of the indicator appear the same as the following:

![ Ichimoku insert parameter.](https://c.mql5.com/2/47/Ichimoku_insert_parameter.png)

The previous picture is to set the desired parameters of the indicator:

1- To set the period of Tenkan-sen

2- To set the period of Kijun-sen

3- To set the period of Senkou Span B

We can also set the desired style and appearance of the indicator from the colors tab:

![Ichimoku insert colors](https://c.mql5.com/2/47/Ichimoku_insert_colors.png)

1- To select the color of Tenkan-sen

1A- To select the style of Tenkan-sen line

1B- To select the thickness of Tenkan-sen line

2- To select the color of the Kijun-sen

2A- To select the style of the Kijun line

2B- To select the thickness of the Kijun line

3- To select the color of the Chikou Span

3A- To select the style of the Chikou Span

3B- To select the thickness of the Chikou Span

4- To select the color of the cloud in case of upward movement (Up Kumo)

4A- To select the style of the cloud in case of upward movement (Up Kumo)

4B- To select the thickness of the cloud in case of upward movement (Up Kumo)

5- To select the color of the cloud in case of downward (Down Kumo)

5A- To select the style of the cloud in case of downward (Down Kumo)

5B- To select the thickness of the cloud in case of downward (Down Kumo)

After setting our preferences, we can find the indicator on the chart the same as the following.

![Ichimoku attached](https://c.mql5.com/2/47/Ichimoku_attached.png)

As we can see from the indicator on the chart in the previous picture, the blue and red lines are for Tenkan-Sen and Kijun-Sen, and the green line is for Chikou Span which is the close price plotted 26 hours in the past according to the timeframe, in addition to the cloud (Senkou Span A and Senkou Span B) that is plotted 26 hours in the future and it is red which means that it is moving to down.

### Ichimoku strategy

In this topic, we will learn how to use the Ichimoku indicator based on simple strategies based on the basic concept of the indicator. We will see a strategy that can be used to identify the trend through the Ichimoku trend identifier. We will learn a strategy that can be used to inform us about the strength of the trend through the strategy of the Ichimoku strength, then we will learn a strategy that can be used to alert us in case of a bullish or bearish signal based on two different methods of crossover the same as what we will see through the strategy of Ichimoku price-Ki signal and the strategy of Ichimoku ten-ki signal.

I need to confirm here before starting to mention strategies that you must test any strategy before using it even if it was tested as there is nothing suitable for everyone. Every one of us has a personality in trading or personal trading style based on his characteristics. So, what might be suitable for me, does not mean it will be suitable for you.

In addition to that, even if you find the strategy is suitable for you as an idea or concept, you may find that you need optimization to be useful for you. So, it is very important that you do not use any strategy on your real account before testing and finding it will be useful for you.

- Strategy one: Ichimoku trend identifier:

According to this strategy, we need a trigger that can be used to inform us about the trend type, if it is an uptrend or downtrend. We will check three values to do that and these values are the closing price, Senkou Span A, and Senkou Span B. If the closing price is greater than the Senkou Span B and at the same time the closing price is greater than Senkou Span A, this will be the trigger to know that the trend is up. Vice versa, if the closing price is lower than the Senkou Span B and at the same time the closing price is lower than the Senkou Span A, this will be the trigger to the downtrend.

Closing price > Senkou Span B and closing price > Senkou Span A --> Uptrend

Closing price < Senkou Span B and closing price < Senkou Span A --> Downtrend

- Strategy two: Ichimoku trend strength:

Based on this strategy, we need a trigger that can inform us the current trend is strong. We will check three values to do that and these values are the current Senkou Span A, the previous Senkou Span A, and the Senkou Span B. If the current Senkou Span A is greater than the previous Senkous Span A and at the same time the current Senkou Span A is greater than the Senkou Span B, this is a trigger that the trend is up and strong. Vice versa, if the current Senkou Span A is lower than the previous Senkou Span A and at the same time, the current Senkou Span A is lower than the Senkou Span B, this will be a signal that the trend is down and strong.

Current Senkou Span A > previous Senkou Span A and current Senkou Span A > Senkou Span B --> the uptrend is strong

Current Senkou Span A < previous Senkou Span A and current Senkou Span A < Senkou Span B --> the downtrend is strong

- Strategy three: Ichimoku price-Ki signal:

According to this strategy, during the uptrend, we need a trigger that can alert us about the bullish signal, and during the downtrend, we need a trigger than can alert us about the bearish signal. We will check based on this strategy two values, closing price, and Kijun-Sen. If the closing price is greater than the Kijun-sen value, this will be a trigger to a bullish signal. Vice versa, if the closing price is lower than the Kijun-sen value, this will be a bearish signal.

During uptrend, closing price > Kijun -sen --> bullish signal

Duuring downtrend, closing price < Kijun -sen --> bearish signal

- Strategy four: Ichimoku ten-ki signal:

According to this strategy, during the uptrend, we need another trigger or method to alert us when there is a bullish signal or during the downtrend, we need a signal of bearishness. We will check based on this strategy two values, Tenkan-sen and Kijum-sen. If the Tenkan-sen value is greater than the Kijun-sen, this will be a signal of bullishness. Vice versa, if the Tenkan-sen is lower than the Kijun-sen, this will be a signal of bearishness.

Tenkan-sen > Kijun-sen --> bullish signal

Tenkan-sen < Kijun-sen --> bearish signal

### Ichimoku strategy blueprint

In this part, we will create a blueprint for each strategy, I consider this step the most important step in our mission to create a trading system as it will help us to design a step-by-step blueprint that will help us to understand what we want to do exactly.

- Strategy one: Ichimoku trend identifier:

Based on this strategy, we need to create a trading system that is able to check the values of closing prices, Senkou Span A, and Senkou span B continuously. We need the trading system to make a comparison between these values to decide which one is bigger or smaller to decide if there is an uptrend or downtrend and appear as a comment on the chart with values of the closing price and Ichimoku lines. If the closing price is greater than span B and the closing price is greater than span A, then the trend is up. If the closing price is lower than span B and the closing price is lower than span A, then the trend is down.

![Ichimoku trend identifier blueprint](https://c.mql5.com/2/47/Ichimoku_trend_identifier_blueprint.png)

- Strategy two: Ichimoku trend strength:

Based on this strategy, we need to create a trading system to alert us to the strength of the current trend. So, we need the trading system to check the values of current Senkou span A, previous senkou span A, and senkou span B continuously. We need the trading system to make a comparison between these values to decide which one is bigger or smaller to return the strength of the current trend based on that. If the current span A is greater than the previous span A and the current span A is greater than span B, then the trend is up and strong and appears as a comment on the chart with values of Ichimoku lines.

![Ichimoku trend strength blueprint](https://c.mql5.com/2/47/Ichimoku_trend_strength_blueprint.png)

- Strategy three: Ichimoku price - Ki signal:

According to this strategy, we need to create a trading system to appear a comment on the chart with the bullish or bearish signal and values of closing prices and the Kijun sen line. So, we need the trading system to check the values of the closing price and the Kijun sen line continuously to make a comparison to decide which one is bigger or smaller. If the closing price is greater than Kijun sen, appear bullish signal, closing price, and Kijun sen values as comments on the chart. If the closing price is lower than the Kijun sen, appear bearish signal, closing price, and Kijun sen values as comments on the chart.

![ Ichimoku Price-Ki signal blueprint](https://c.mql5.com/2/47/Ichimoku_Price-Ki_signal_blueprint.png)

- Strategy four: Ichimoku ten-Ki signal:

According to this strategy, we need to create a trading system to appear a comment on the chart with a bullish or bearish signal based on a comparison between Tenkan sen and Kijun values to decide which one is bigger or smaller. So, we need the trading system to check these two values continuously to return a bullish signal, the Tenkan Sen value, and the Kijun Sen value if the Tenkan Sen value is greater than the Kijun Sen value to return a bearish signal, the Tenkan Sen value, Kijun Sen value if the Tenkan Sen value is lower than Kijun Sen value.

![Ichimoku ten-Ki signal blueprint](https://c.mql5.com/2/47/Ichimoku_ten-Ki_signal_blueprint.png)

### Ichimoku trading system

In this part, we will learn how to design a trading system by Ichimoku indicator based on mentioned strategies. So, we will learn how to design a trading system based on Ichimoku trend identifier strategy, trend strength strategy, price, and Kijun-sen strategy, in addition to Tenken-sen and Kijun-sen strategy.

Now, we will design the simple Ichimoku system that shows us all Ichimoku values as a comment on the chart to use as a base for the mentioned strategies.

We will place the #include command to the file name Indicators/Trend.mqh then use CiIchimoku class to use the data of the Ichimoku indicator.

```
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
```

Through the void OnInit(), we will call the indicator and then create double variables values of Tenkan-sen, Kijun-sen, Senkou span A, Senkou span B, and Chikou span.

```
   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);
```

Comment these values on the chart and each value in a separate line.

```
   Comment("Tenkan Sen Value is: ",TenkanVal,"\n",
           "Kijun Sen Value is: ",KijunVal,"\n",
           "Senkou Span A Value is: ", SpanAVal,"\n",
           "Senkou Span B Value is: ",SpanBVal,"\n",
           "Chikou Span Value is: ",ChikouVal);
```

So, the full code will be the same as the following:

```
//+------------------------------------------------------------------+
//|                                       Simple Ichimoku system.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
//+------------------------------------------------------------------+
void OnInit()
  {
   Ichimoku = new CiIchimoku();
   Ichimoku.Create(_Symbol,PERIOD_CURRENT,9,26,52);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);

   Comment("Tenkan Sen Value is: ",TenkanVal,"\n",
           "Kijun Sen Value is: ",KijunVal,"\n",
           "Senkou Span A Value is: ", SpanAVal,"\n",
           "Senkou Span B Value is: ",SpanBVal,"\n",
           "Chikou Span Value is: ",ChikouVal);
  }
//+------------------------------------------------------------------+
```

After compiling this code, we will find it in the navigator window.

![Nav - Ichi](https://c.mql5.com/2/47/Nav_-_Ichi.png)

To execute it, we will double-click it or drag and drop it on the chart and the window of the Ichimoku system will appear.

![Simple Ichimoku system window](https://c.mql5.com/2/47/Simple_Ichimoku_system_window.png)

After pressing Ok, it will be attached to the chart.

![Simple Ichimoku system attached](https://c.mql5.com/2/47/Simple_Ichimoku_system_attached.png)

The following is an example of generated signals of this system.

![Simple Ichimoku system signal](https://c.mql5.com/2/47/Simple_Ichimoku_system_signal.png)

As we can see from the previous picture, we find that there is a comment on the chart with the following values:

- Tenkan Sen Value
- Kijun Sen Value
- Senkou Span A Value
- Senkou Span B Value
- Chikou Span Value

These previous values represent the Ichimoku elements.

- Strategy one: Ichimoku trend identifier:

Now, we need to design a trading system that gives me a comment with the current trend definition based on this strategy and the following is the full code to write this kind of strategy.

```
//+------------------------------------------------------------------+
//|                                    Ichimoku trend identifier.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
//+------------------------------------------------------------------+
void OnInit()
  {
   Ichimoku = new CiIchimoku();
   Ichimoku.Create(_Symbol,PERIOD_CURRENT,9,26,52);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlRates PArray[];
   int Data=CopyRates(_Symbol,_Period,0,1,PArray);

   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);

   if(PArray[0].close>SpanBVal&&PArray[0].close>SpanAVal)
     {
      Comment("The trend is up","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }

   if(PArray[0].close<SpanBVal&&PArray[0].close<SpanAVal)
     {
      Comment("The trend is down","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code will be the same as the following:

Creating a price array by using the MqlRates function that stores price information.

```
MqlRates PArray[];
```

Filling the price array by using the CopyRates function after creating an integer variable for Data. CopyRates function gets historical data of MqlRates.

```
int Data=CopyRates(_Symbol,_Period,0,1,PArray);
```

Conditions of the Ichimoku trend identifier strategy based on the type of the trend,

In case of the uptrend:

```
   if(PArray[0].close>SpanBVal&&PArray[0].close>SpanAVal)
     {
      Comment("The trend is up","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
```

In case of the downtrend:

```
   if(PArray[0].close<SpanBVal&&PArray[0].close<SpanAVal)
     {
      Comment("The trend is down","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
```

After compiling this code, we can find it in the navigator window in the Expert Advisors folder and the following is a picture that shows it.

![Nav - Ichi 2](https://c.mql5.com/2/47/Nav_-_Ichi_2.png)

By double-clicking or dragging and dropping it on the chart the Ichimoku trend identifier window will appear the same as the following:

![Ichimoku trend identifier window.](https://c.mql5.com/2/47/Ichimoku_trend_identifier_window.png)

After pressing "Ok", it will be attached to the chart.

![Ichimoku trend identifier attached](https://c.mql5.com/2/47/Ichimoku_trend_identifier_attached.png)

The following is an example of generating signals for uptrend and downtrend based on the strategy conditions.

Uptrend signal:

![Ichimoku trend identifier up signal](https://c.mql5.com/2/47/Ichimoku_trend_identifier_up_signal.png)

As we can see in the previous picture, there are comments on the charts with the following:

- The trend is up
- Close price Value
- Tenkan Sen Value
- Kijun Sen Value
- Senkou Span A Value
- Senkou Span B Value
- Chikou Span Value

This strategy provides the uptrend type with different values of price and Ichimoku elements.

Downtrend signal:

![Ichimoku trend identifier down signal](https://c.mql5.com/2/47/Ichimoku_trend_identifier_down_signal.png)

As we can see in the previous picture, there are comments on the charts with the following:

- The trend is down
- Close price Value
- Tenkan Sen Value
- Kijun Sen Value
- Senkou Span A Value
- Senkou Span B Value
- Chikou Span Value

This strategy provides the uptrend type with different values of price and Ichimoku elements.

- Strategy two: Ichimoku trend strength:

According to this strategy, we need to create a trading system that appears as a comment on the chart with the trend strength if it is strong as an uptrend or downtrend. The following code is for writing this trading system on MQL5 that can do that:

```
//+------------------------------------------------------------------+
//|                                      Ichimoku trend strength.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
//+------------------------------------------------------------------+
void OnInit()
  {
   Ichimoku = new CiIchimoku();
   Ichimoku.Create(_Symbol,PERIOD_CURRENT,9,26,52);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanAPrevVal= Ichimoku.SenkouSpanA(-25);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);

   if(SpanAVal>SpanAPrevVal&&SpanAVal>SpanBVal)
     {
      Comment("The trend is up and strong","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Prev. Value is: ", SpanAPrevVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }

   if(SpanAVal<SpanAPrevVal&&SpanAVal<SpanBVal)
     {
      Comment("The trend is down and strong","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Prev. Value is: ", SpanAPrevVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code are conditions of the strategy,

In case of a strong uptrend:

```
   if(SpanAVal>SpanAPrevVal&&SpanAVal>SpanBVal)
     {
      Comment("The trend is up and strong","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Prev. Value is: ", SpanAPrevVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
```

In case of a strong downtrend:

```
   if(SpanAVal<SpanAPrevVal&&SpanAVal<SpanBVal)
     {
      Comment("The trend is down and strong","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n",
              "Senkou Span A Prev. Value is: ", SpanAPrevVal,"\n",
              "Senkou Span A Value is: ", SpanAVal,"\n",
              "Senkou Span B Value is: ",SpanBVal,"\n",
              "Chikou Span Value is: ",ChikouVal);
     }
```

After compiling this code, we will find it in the navigator window.

![Nav - Ichi 3](https://c.mql5.com/2/47/Nav_-_Ichi_3.png)

By double-clicking to execute it, we will find the window of the expert the same as the following one:

![ Ichimoku trend strength window](https://c.mql5.com/2/47/Ichimoku_trend_strength_window.png)

After pressing "OK", the expert will be attached to the chart.

![Ichimoku trend strength attached](https://c.mql5.com/2/47/Ichimoku_trend_strength_attached.png)

The following is an example of generated signals from testing, in case of a strong uptrend:

![Ichimoku trend strength up and strong signal](https://c.mql5.com/2/47/Ichimoku_trend_strength_up_and_strong_signal.png)

As we can see in the previous picture, we can find comments on the chart with the following values:

- The trend is up and strong
- Tenkan Sen Value
- Kijun Sen Value
- Senkou Span A previous Value
- Senkou Span A Value
- Senkou Span B Value
- Chikou Span Value

Here we can find that the comment informs us that the uptrend is strong. The following is an example of generated signals from testing, in case of a strong downtrend:

![ Ichimoku trend strength down and strong signal](https://c.mql5.com/2/47/Ichimoku_trend_strength_down_and_strong_signal.png)

As we can see in the previous picture, we can find comments on the chart with the following values:

- The trend is down and strong
- Tenkan Sen Value
- Kijun Sen Value
- Senkou Span A previous Value
- Senkou Span A Value
- Senkou Span B Value
- Chikou Span Value

On the opposite, we can find that the comment informs us that the downtrend is strong.

- Strategy three: Ichimoku Price-Ki signal:

According to this strategy, we need to create a trading system to generate a comment on the chart with bullish or bearish signals based on the crossover between price and Kijun-sen, the following is the full code to do that.

```
//+------------------------------------------------------------------+
//|                                     Ichimoku Price-Ki signal.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
//+------------------------------------------------------------------+
void OnInit()
  {
   Ichimoku = new CiIchimoku();
   Ichimoku.Create(_Symbol,PERIOD_CURRENT,9,26,52);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   MqlRates PArray[];

   int Data=CopyRates(_Symbol,_Period,0,1,PArray);

   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);

   if(PArray[0].close>KijunVal)
     {
      Comment("Bullish signal","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }

   if(PArray[0].close<KijunVal)
     {
      Comment("Bearish signal","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code:

Creating and filling the price array,

```
   MqlRates PArray[];

   int Data=CopyRates(_Symbol,_Period,0,1,PArray);
```

Condition of the Ichimoku Price-Ki strategy, in case of a bullish signal,

```
   if(PArray[0].close>KijunVal)
     {
      Comment("Bullish signal","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
```

Condition of the Ichimoku Price-Ki strategy, in case of a bearish signal,

```
   if(PArray[0].close<KijunVal)
     {
      Comment("Bearish signal","\n",
              "Close Value is: ",PArray[0].close,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
```

After compiling this code, we can find it in the navigator window.

![Nav - Ichi 5](https://c.mql5.com/2/47/Nav_-_Ichi_5.png)

By dragging and dropping it on the chart to execute the expert, the window of the expert will appear.

![ Ichimoku Price-Ki signal window](https://c.mql5.com/2/47/Ichimoku_Price-Ki_signal_window.png)

After pressing "OK", it will be attached to the chart.

![Ichimoku Price-Ki signal attached](https://c.mql5.com/2/47/Ichimoku_Price-Ki_signal_attached.png)

The following is an example of the generated bullish signal from testing.

![Ichimoku Price-Ki signal bullish](https://c.mql5.com/2/47/Ichimoku_Price-Ki_signal_bullish.png)

As we can see in the previous example, there is a comment on the chart with:

- Bullish signal
- Closing price value
- Kijun-sen value

This bullish signal is generated based on the crossover between price and Kijun-sen. The following is an example of the generated bearish signal from testing.

![ Ichimoku Price-Ki signal bearish](https://c.mql5.com/2/47/Ichimoku_Price-Ki_signal_bearish.png)

As we can see in the previous example, there is a comment on the chart with:

- Bearish signal
- Closing price value
- Kijun-sen value

This bearish signal is generated based on the crossover between price and Kijun-sen.

- Strategy four: Ichimoku ten-ki strategy:

According to this strategy, we need to create a trading system to generate a comment on the chart with bullish or bearish signals based on the crossover between Tenkan-sen and Kijun-sen, the following is the full code to do that.

```
//+------------------------------------------------------------------+
//|                                       Ichimoku ten-Ki signal.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Indicators/Trend.mqh>
CiIchimoku*Ichimoku;
//+------------------------------------------------------------------+
void OnInit()
  {
   Ichimoku = new CiIchimoku();
   Ichimoku.Create(_Symbol,PERIOD_CURRENT,9,26,52);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   Ichimoku.Refresh(-1);
   double TenkanVal= Ichimoku.TenkanSen(0);
   double KijunVal= Ichimoku.KijunSen(0);
   double SpanAVal= Ichimoku.SenkouSpanA(-26);
   double SpanBVal= Ichimoku.SenkouSpanB(-26);
   double ChikouVal= Ichimoku.ChinkouSpan(26);

   if(TenkanVal>KijunVal)
     {
      Comment("Bullish signal","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }

   if(TenkanVal<KijunVal)
     {
      Comment("Bearish signal","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code are conditions of the Ichimoku ten-Ki strategy,

In case of a bullish signal:

```
   if(TenkanVal>KijunVal)
     {
      Comment("Bullish signal","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
```

In case of a bearish signal:

```
   if(TenkanVal<KijunVal)
     {
      Comment("Bearish signal","\n",
              "Tenkan Sen Value is: ",TenkanVal,"\n",
              "Kijun Sen Value is: ",KijunVal,"\n");
     }
```

After compiling this code, we will find the expert in the navigator.

![Nav - Ichi 4](https://c.mql5.com/2/47/Nav_-_Ichi_4.png)

By double-clicking, its window will appear.

![Ichimoku ten-Ki signal window](https://c.mql5.com/2/47/Ichimoku_ten-Ki_signal_window.png)

After pressing "OK", the expert will be attached to the chart.

![Ichimoku ten-Ki signal attached](https://c.mql5.com/2/47/Ichimoku_ten-Ki_signal_attached.png)

The following is an example of generated bullish signal from testing based on this strategy:

![Ichimoku ten-Ki signal bullish](https://c.mql5.com/2/47/Ichimoku_ten-Ki_signal_bullish.png)

As we can see in the previous example, there is a comment on the chart with:

- Bullish signal
- Tenkan-sen value
- Kijun-sen value

This bullish signal is generated based on the crossover between tenkan-sen and Kijun-sen. The following is an example of generated bearish signal from testing based on this strategy:

![Ichimoku ten-Ki signal bearish](https://c.mql5.com/2/47/Ichimoku_ten-Ki_signal_bearish.png)

As we can see in the previous example, there is a comment on the chart with:

- Bearish signal
- Tenkan-sen value
- Kijun-sen value

This bearish signal is generated based on the crossover between tenkan-sen and Kijun-sen.

### Conclusion

The Ichimoku indicator can be used as a complete system to let you identify more than one perspective on the chart as we learned in this article as you can identify the trend, the strength of the trend, and get trading bullishness or bearishness signals. So, it is very useful in trading and it can enhance our results by taking better decisions by identifying many perspectives by one indicator.

I think that we learned many topics about this indicator to cover it as much as we can but for sure, you may find yourself needing to read more about this indicator and this is normal as we provided basics about it and I encourage you to do that.

We covered many topics in this article about Ichimoku, we learned what is it, what it measures, its construction, and how we can calculate it through the topic of Ichimoku definition. We learn how we can use it through simple strategies based on the basic concept behind it as we learned the following strategies:

- Ichimoku trend identifier: to inform us of the trend type, if it is an uptrend or a downtrend.
- Ichimoku trend strength: to inform us when the trend is strong.
- Ichimoku price-ki signal: to inform us when there is a bullish or bearish signal.
- Ichimoku ten-ki signal: to inform us when there is a bullish or bearish signal based on another method.

We created also a step-by-step blueprint to help us create our trading system based on mentioned strategies. We created a trading system for every mentioned strategy by MQL5 to be used in the MetaTrader 5. I hope that you applied what you learned in this article by yourself as the practice is a very important factor in any educational process. I confirm again to test any strategy before using it on your real account as there is nothing suitable for everyone.

Algorithmic trading is an incredible tool that can help us to trade very well as it helps us to avoid human emotions that can be harmful to our trading because you have to know that emotions play a big role in our trading and it will be a reason for losses. So, when we find a tool that can execute our trades based on our winning strategy automatically, this will be a treasure literally. In addition to that, the time that can be available for us to do other useful things at the same time that our trading system is working for us.

At the end of this article, I hope that you find it useful for you to enhance your trading. If you want to read more similar articles, you can read my other articles in this series about how to design a trading system by the most popular technical indicator.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11081.zip "Download all attachments in the single ZIP archive")

[Simple\_Ichimoku\_system.mq5](https://www.mql5.com/en/articles/download/11081/simple_ichimoku_system.mq5 "Download Simple_Ichimoku_system.mq5")(1.52 KB)

[Ichimoku\_trend\_identifier.mq5](https://www.mql5.com/en/articles/download/11081/ichimoku_trend_identifier.mq5 "Download Ichimoku_trend_identifier.mq5")(2.22 KB)

[Ichimoku\_trend\_strength.mq5](https://www.mql5.com/en/articles/download/11081/ichimoku_trend_strength.mq5 "Download Ichimoku_trend_strength.mq5")(2.22 KB)

[Ichimoku\_Price-Ki\_signal.mq5](https://www.mql5.com/en/articles/download/11081/ichimoku_price-ki_signal.mq5 "Download Ichimoku_Price-Ki_signal.mq5")(1.73 KB)

[Ichimoku\_ten-Ki\_signal.mq5](https://www.mql5.com/en/articles/download/11081/ichimoku_ten-ki_signal.mq5 "Download Ichimoku_ten-Ki_signal.mq5")(1.64 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/427516)**
(7)


![MathLearner](https://c.mql5.com/avatar/2019/8/5D56E639-62D7.png)

**[MathLearner](https://www.mql5.com/en/users/hermansuriato)**
\|
13 Jul 2022 at 14:44

Very good and useful lesson.

Thank you

![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
4 Feb 2023 at 10:24

I think it's better to combine come described strategies in one in order to make trading signals stronger.

![R4tna C](https://c.mql5.com/avatar/avatar_na2.png)

**[R4tna C](https://www.mql5.com/en/users/r4tna)**
\|
14 May 2024 at 18:51

Excellent article - only one minor issue is that I got memory leaks with the sample code

[![](https://c.mql5.com/3/435/1816495918868__1.png)](https://c.mql5.com/3/435/1816495918868.png "https://c.mql5.com/3/435/1816495918868.png")

I found by deleting the object in DeInit() fixed this

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   delete Ichimoku;
  }
```

![Ngoc Kien](https://c.mql5.com/avatar/2023/11/6565A4E7-FC00.png)

**[Ngoc Kien](https://www.mql5.com/en/users/nguyenngockien0888288528)**
\|
14 Sep 2024 at 16:47

hi all, help me to bugging mql5:


![Ngoc Kien](https://c.mql5.com/avatar/2023/11/6565A4E7-FC00.png)

**[Ngoc Kien](https://www.mql5.com/en/users/nguyenngockien0888288528)**
\|
15 Sep 2024 at 00:31

my mind :

draw spanB limegreen when close > past clouds and close > past tenkansen, close > past kijunsen and close > Present clouds, spanA > spanB :

#include <Indicators/Trend.mqh>

CiIchimoku\*Ichimoku;

//\-\-\- indicator settings

[#property indicator\_chart\_window](https://www.mql5.com/en/docs/basis/preprosessor/compilation "MQL5 Documentation: Program Properties (#property)")

#property indicator\_buffers 9

#property indicator\_plots   1

#property indicator\_type1   DRAW\_LINE

#property indicator\_color1  LimeGreen

#property indicator\_width1  2

double sp1tl\[\];

double sp2tl\[\];

double trendtang\[\];

double tenqk\[\];

double kijqk\[\];

double sp1ht\[\];

double sp2ht\[\];

double sp1qk\[\];

double sp2qk\[\];

void OnInit()

{

Ichimoku = new CiIchimoku();

Ichimoku.Create(\_Symbol,PERIOD\_CURRENT,9,26,52);

SetIndexBuffer(0,trendtang,INDICATOR\_DATA);

SetIndexBuffer(1,sp1tl,INDICATOR\_DATA);

SetIndexBuffer(2,sp2tl,INDICATOR\_DATA);

SetIndexBuffer(3,tenqk,INDICATOR\_DATA);

SetIndexBuffer(4,kijqk,INDICATOR\_DATA);

SetIndexBuffer(5,sp1ht,INDICATOR\_DATA);

SetIndexBuffer(6,sp2ht,INDICATOR\_DATA);

SetIndexBuffer(7,sp1qk,INDICATOR\_DATA);

SetIndexBuffer(8,sp2qk,INDICATOR\_DATA);

IndicatorSetInteger(INDICATOR\_DIGITS,\_Digits+1);

//\-\-\- sets first bar from what index will be drawn

PlotIndexSetInteger(0,PLOT\_DRAW\_BEGIN,51);

//\-\-\- lines shifts when drawing

PlotIndexSetInteger(0,PLOT\_SHIFT,25);

}

int OnCalculate(const int rates\_total,

                const int prev\_calculated,

                const datetime &time\[\],

                const double &open\[\],

                const double &high\[\],

                const double &low\[\],

                const double &close\[\],

                const long &tick\_volume\[\],

                const long &volume\[\],

                const int &spread\[\])

{

int start;

//---

if(prev\_calculated==0)

      start=0;

else

      start=prev\_calculated-1;

//\-\-\- main loop

for(int i=start; i<rates\_total && !IsStopped(); i++)

     {

     MqlRates PArray\[\];

int Data=CopyRates(\_Symbol,\_Period,0,1,PArray);

Ichimoku.Refresh(-1);

double spanAtl= Ichimoku.SenkouSpanA(0);

double spanBtl= Ichimoku.SenkouSpanB(0);

double spanAht= Ichimoku.SenkouSpanA(-25);

double spanBht= Ichimoku.SenkouSpanB(-25);

double spanAqk= Ichimoku.SenkouSpanA(-51);

double spanBqk= Ichimoku.SenkouSpanB(-51);

double tenkanqk= Ichimoku.TenkanSen(-25);

double kijunqk= Ichimoku.KijunSen(-25);

      sp1tl\[i\]=spanAtl;

      sp2tl\[i\]=spanBtl;

      tenqk\[i\]=tenkanqk;

      kijqk\[i\]=kijunqk;

      sp1ht\[i\]=spanAht;

      sp2ht\[i\]=spanBht;

      sp1qk\[i\]=spanAqk;

      sp2qk\[i\]=spanBqk;

if(

sp1tl\[i\]>=sp2tl\[i\]

&& close\[i\]>tenqk\[i\]

&& close\[i\]>kijqk\[i\]

&& close\[i\]>sp1ht\[i\]

&& close\[i\]>sp2ht\[i\]

&& close\[i\]>sp1qk\[i\]

&& close\[i\]>sp2qk\[i\]

)

{

trendtang\[i\]=sp2tl\[i\];

}

else

{

trendtang\[i\]=EMPTY\_VALUE;

}

     }

//---

return(rates\_total);

}

![Developing a trading Expert Advisor from scratch (Part 9): A conceptual leap (II)](https://c.mql5.com/2/46/development__1.png)[Developing a trading Expert Advisor from scratch (Part 9): A conceptual leap (II)](https://www.mql5.com/en/articles/10363)

In this article, we will place Chart Trade in a floating window. In the previous part, we created a basic system which enables the use of templates within a floating window.

![DoEasy. Controls (Part 4): Panel control, Padding and Dock parameters](https://c.mql5.com/2/46/MQL5-avatar-doeasy-library-2__3.png)[DoEasy. Controls (Part 4): Panel control, Padding and Dock parameters](https://www.mql5.com/en/articles/10756)

In this article, I will implement handling Padding (internal indents/margin on all sides of an element) and Dock parameters (the way an object is located inside its container).

![Data Science and Machine Learning (Part 05): Decision Trees](https://c.mql5.com/2/48/tree_decision__1.png)[Data Science and Machine Learning (Part 05): Decision Trees](https://www.mql5.com/en/articles/11061)

Decision trees imitate the way humans think to classify data. Let's see how to build trees and use them to classify and predict some data. The main goal of the decision trees algorithm is to separate the data with impurity and into pure or close to nodes.

![How to master Machine Learning](https://c.mql5.com/2/47/machine-learning.png)[How to master Machine Learning](https://www.mql5.com/en/articles/10431)

Check out this selection of useful materials which can assist traders in improving their algorithmic trading knowledge. The era of simple algorithms is passing, and it is becoming harder to succeed without the use of Machine Learning techniques and Neural Networks.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/11081&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6413686624949830123)

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