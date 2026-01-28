---
title: How to deal with lines using MQL5
url: https://www.mql5.com/en/articles/11538
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:10:14.624246
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/11538&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069189761962279252)

MetaTrader 5 / Trading


### Introduction

We all as traders draw many lines while trading to help us to observe some important levels to take a suitable trading decision based on them. So, these lines are very important for us as traders and we may wonder if we have a method that can be used to draw these lines or take a suitable decision based on them automatically because I think that it will help us a lot. The answer is yes, we have a method to do that by MQL5 (MetaQuotes Language 5).

This article is an example to learn how we can deal with these lines by the MQL5 because there are many methods to do something like this and these methods depend on what you need to do. There are many types of lines but we will mention in this article three of them only because of their popularity and they are trend lines, support, and resistance lines.

We will cover them through the following topics:

1. [Trend lines and MQL5](https://www.mql5.com/en/articles/11538#trend)
2. [Support and MQL5](https://www.mql5.com/en/articles/11538#support)
3. [Resistance and MQL5](https://www.mql5.com/en/articles/11538#resistance)
4. [Conclusion](https://www.mql5.com/en/articles/11538#conclusion)

Generally speaking, there are many methods that can be used to deal with lines by MQL5 we will share here one simple method and you may find that you need to develop it to match your strategy we will share here a method to detect price levels then drawing an updated type of line.

We will learn what are lines in detail by identifying their types, how we can draw them manually, and how we can use them in trading. We learn how we can use MQL5 to deal with trend lines, support, and resistance levels to help us to ease our trading process and enhance our trading results by automation things. We will use MQL5 (MetaQuotes Language 5) to code our programs which is built-in with the MetaTrader 5 trading terminal and if you do not know how to use them, you can read the topic of [Writing MQL5 code in MetaEditor](https://www.mql5.com/en/articles/10748#editor) from a previous article to learn more about that.

My advice here is to try to write and apply what you read by yourself as this step is very important to improve your coding skills and it will deepen your understanding in addition to getting more insights about the topic of this article and/or any related topic which can be useful to develop and enhance your trading results.

Disclaimer: All information provided 'as is' only for educational purposes and is not prepared for trading purposes or advice. The information does not guarantee any kind of result. If you choose to use these materials on any of your trading accounts, you will do that at your own risk and you will be the only responsible.

### Trendlines and MQL5

Lines are technical tools that can be used for trading to get more effective results. These lines can be drawn around important prices or levels that can be used as triggers to take effective decisions and there are many types of these lines. We will mention three of these important lines in this article and they are trend lines, support, and resistance levels.

Now, we will start identifying and dealing with Trend lines. Trend lines are lines that can be used to visualize price levels that can be used as rejection levels to up or down. It can be drawn above or below prices based on the direction of the market. So, we can say that the Trend line may help us to detect the direction of the market trend. If there is a price bounce to the upside from three points as a minimum on the same levels of the drawn line, it will be an upward trend line. Vice versa, if there is a bounce to the downside from three points as a minimum on the same levels of the drawn line, it will be a downtrend line. It is important to mention here that a trend line is just a tool and you can recognize and draw it correctly or not.

We can use the Trend line in trading by placing our orders based on the trend line type. If there is an upward trend line, we may expect the price moves down to test this trend line from above then rebounding to up and then we can place our buying order around this trend line. Vice versa, if there is a downward trend line, we may that the price moves up to test this trend line from below then rebounding to down and then we can place our shorting or selling order around this downward trend line.

The following is for the upward trend line:

![ Up trendline](https://c.mql5.com/2/50/Up_trendline__1.png)

We can see that it is clear in the previous figure that we have an up movement and if we try to connect between the last three lows we can find that they are on the same line heading up.

The following is an example of the upward trend line from the market:

![Up trendline](https://c.mql5.com/2/50/Up_trendline.png)

As we can see in the previous chart we have an upward trend line that touched many times and rebounded after finding the buying power to head up which confirms the existence of an uptrend.

The following is for the downtrend line:

![Down trendline](https://c.mql5.com/2/50/Down_trendline__1.png)

We can see that it is clear in the previous figure that we have a down movement and if we try to connect between the last three highs we can find that they are on the same line move to down.

The following is an example of a downtrend line from the market:

![Down trendline](https://c.mql5.com/2/50/Down_trendline.png)

As we can see in the previous chart we have a downward trend line that touched many times and rebounded after finding the selling power to move down which confirms the existence of a downtrend.

**1\. Upward Trendline System**

We need to create a program that can be used to create an upward trend line by the MetaTrader 5 automatically that can be seen below prices for potential up movement. We need the program to check price lows and if there is any upward trend line every tick. Then deleting the previous upward trend line, creating an updated blue one below the lows of the price.

To do that, we can follow the below steps as one of the methods that can be used.

Creating first candle detection by creating an integer variable to be equal to the returned long type of value of the corresponding property of the current chart by using the "ChartGetInteger" function and its parameters are:

- chart\_id: to determine the chart and we will use (0) to represent the current chart.
- prop\_id: to determine the chart property ID and we can use here one of the (ENUM\_CHART\_PROPERTY\_INTEGER) values. We will use (CHART\_FIRST\_VISIBLE\_BAR).
- sub\_window=0: to determine the number of chart sub-window. We will use (0) to represent the main chart window.

```
int candles = (int)ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);
```

Creating an array of candles of low prices by creating a double variable type for pLow.

```
double pLow[];
```

Sorting the created array from the current data by using the "ArraySetAsSeries" function to return a boolean value which will be true on success or false in the other case.

```
ArraySetAsSeries(pLow,true);
```

Filling the created pLow array with data by using the " CopyLow" function to get into the lowest price of the current symbol of the current time. Its parameters are:

- symbol\_name: to determine the symbol, we will use (\_Symbol) to be applied for the current symbol.
- timeframe: to determine the timeframe, we will use (\_Period) to be applied for the current time frame.
- start\_pos: to determine the start position of element to copy, we will use (0).

- count: to determine the data count to copy, we will use the pre created variable ( candles).

- low\_array\[\]: to determine the target array to copy, we will use (pLow) array.


```
int copy_low = CopyLow(_Symbol,_Period,0,candles,pLow);
```

Setting condition if copy\_low is greater than 0

```
if(copy_low>0)
```

In the body of if condition, creating and calculating the low of the candle variable by creating an integer variable of " candleLow" to be equal to the minimum value of the array and we will use the "ArrayMinimum" function to return the lowest element in the array. Its parameters:

- array\[\]: to determine the array, we will use the ( pLow) array.

- start=0: to determine the index to start checking with, we will use the (0) value.

- count=WHOLE\_ARRAY: to determine the number of checked elements, we will use the (candles).


```
int candleLow = ArrayMinimum(pLow,0,candles);
```

Creating an array of price "pArray" by using the "MqlRates" function which stores information about prices, volume, and spread.

```
MqlRates pArray[];
```

Sorting the price array of "pArray" by using the " ArraySetAsSeries" function.

```
ArraySetAsSeries(pArray,true);
```

Copying price data to the price array after creating an integer variable of "Data" and then using the " CopyRates" to get the historical data of the MqlRates structure. Its parameters are:

- symbol\_name: to determine the symbol, we will use (\_Symbol) to be applied to the current chart.
- timeframe: to determine the time frame, we will use (\_Period) to be applied to the current time frame.
- start\_pos: to determine the start position, we will use (0) value.
- count: to determine the data count to copy, we will use the ( candles) created variable.

- rates\_array\[\]: to determine the target array to copy, we will use the ( pArray).


```
int Data = CopyRates(_Symbol,_Period,0,candles,pArray);
```

Deleting any previously created trend line to be updated by using the " ObjectDelete" function to remove any specified object from any specified chart. Its parameters are:

- chart\_id: to determine the chart identifier, we will use (\_Symbol) to be applied for the current symbol.
- name: to specify the name of the object that we want to remove, we will specify the ("UpwardTrendline") as a name.

```
ObjectDelete(_Symbol,"UpwardTrendline");
```

Creating a new trend line by using the " ObjectCreate" to create a trend line object. Its parameters are:

- chart\_id: to determine the symbol, we will use (\_Symbol) to be applied for the current symbol.
- name: to specify the object name, we will specify the ("UpwardTrendline") as a name.
- type: to determine the object type, we will use the (OBJ\_TREND) to create a trend line.
- nwin: to determine the window index, we will use (0) value to use the main chart window.
- time1: to determine the time of the first anchor point, we will use the (pArray\[candleLow\].time).
- price1: to determine the price of the first anchor point, we will use the (pArray\[candleLow\].low).
- timeN: to determine the time of the N of anchor point, we will use the (pArray\[0\].time).
- priceN: to determine the price of the N of anchor point, we will use the (pArray\[0\].low).

```
ObjectCreate(_Symbol,"UpwardTrendline",OBJ_TREND,0,pArray[candleLow].time,pArray[candleLow].low,pArray[0].time,pArray[0].low);
```

Determining trend line color by using the " ObjectSetInteger" function to set the value as a color of the object. Its parameters are:

- chart\_id: the chart identifier, we will use (0) value which means the current chart.
- name: the object name, we will use the ("UpwardTrendline") which is the predefined name of the object.
- prop\_id: the ID of the object property, we will use one of the ENUM\_OBJECT\_PROPERTY\_INTEGER which is the (OBJPROP\_COLOR).
- prop\_value: the value of the property, we will use the (Blue) as a color of the created trend line.

```
ObjectSetInteger(0,"UpwardTrendline",OBJPROP_COLOR,Blue);
```

Determining trend line style by using the ObjectSetInteger function also but we will use another one of the  ENUM\_OBJECT\_PROPERTY\_INTEGER which is the (OBJPROP\_STYLE) and the prop\_value will be (STYLE\_SOLID) as a style of the created trend line.

```
ObjectSetInteger(0,"UpwardTrendline",OBJPROP_STYLE,STYLE_SOLID);
```

Determining trend line width by using the "ObjectSetInteger" function also but we will use another one of the ENUM\_OBJECT\_PROPERTY\_INTEGER which is the (OBJPROP\_WIDTH) and the prop\_value will be (3) as a width of the created trend line.

```
ObjectSetInteger(0,"UpwardTrendline",OBJPROP_WIDTH,3);
```

Determining trend line ray by using the "ObjectSetInteger" function also but we will use another one of ENUM\_OBJECT\_PROPERTY\_INTEGER which is the (OBJPROP\_RAY\_RIGHT) and the value of the prop will be true.

```
ObjectSetInteger(0,"UpwardTrendline",OBJPROP_RAY_RIGHT,true);
```

And the following is for the full code to create this system to create an upward trend line automatically:

```
//+------------------------------------------------------------------+
//|                                       UpwardTrendline System.mq5 |
//+------------------------------------------------------------------+
void OnTick()
  {
   int candles = (int)ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);
   double pLow[];
   ArraySetAsSeries(pLow,true);
   int copy_low = CopyLow(_Symbol,_Period,0,candles,pLow);
   if(copy_low>0)
     {
      int candleLow = ArrayMinimum(pLow,0,candles);
      MqlRates pArray[];
      ArraySetAsSeries(pArray,true);
      int Data = CopyRates(_Symbol,_Period,0,candles,pArray);
      ObjectDelete(0,"UpwardTrendline");
      ObjectCreate(0,"UpwardTrendline",OBJ_TREND,0,pArray[candleLow].time,pArray[candleLow].low,
                   pArray[0].time,pArray[0].low);
      ObjectSetInteger(0,"UpwardTrendline",OBJPROP_COLOR,Blue);
      ObjectSetInteger(0,"UpwardTrendline",OBJPROP_STYLE,STYLE_SOLID);
      ObjectSetInteger(0,"UpwardTrendline",OBJPROP_WIDTH,3);
      ObjectSetInteger(0,"UpwardTrendline",OBJPROP_RAY_RIGHT,true);
     }
  }
//+------------------------------------------------------------------+
```

After writing this code we will compile it, it must resulting no errors.  After that we will find the Expert Advisor (EA) in the navigator window the same as the following:

![ Lines nav](https://c.mql5.com/2/50/Lines_nav.png)

By dragging and dropping the EA on the chart, we will find the window of the expert the same as the following:

![ Lines Win](https://c.mql5.com/2/50/Lines_Win.png)

After ticking next to the "Allow Algo Trading" and pressing "OK", we will the EA is attached to the chart and we will be ready to see the desired action based on this EA the same as the following example:

![ Upward trendline example](https://c.mql5.com/2/50/Upward_trendline_example.png)

As we can see in the previous chart that we have the EA is attached to the chart the same as it is obvious in the top right corner and we have the blue upward trend line is drawn below prices.

**2\. Downward Trendline System**

We need to create a program that can be used to create a downward trend line by the MetaTrader 5 automatically that can be seen above prices for potential down movement. We need the program to check price highs and if there is any downward trend line every tick. Then delete the previous downward trend line and create an updated blue one above the highs of the price.

To do that, we can follow the below full code as one of the methods that can be used.

```
//+------------------------------------------------------------------+
//|                                     DownwardTrendline System.mq5 |
//+------------------------------------------------------------------+
void OnTick()
  {
   int candles = (int)ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);
   double pHigh[];
   ArraySetAsSeries(pHigh,true);
   int copy_high = CopyHigh(_Symbol,_Period,0,candles,pHigh);
   if(copy_high>0)
     {
      int candleHigh = ArrayMaximum(pHigh,0,candles);
      MqlRates pArray[];
      ArraySetAsSeries(pArray,true);
      int Data = CopyRates(_Symbol,_Period,0,candles,pArray);
      ObjectDelete(0,"DnwardTrendline");
      ObjectCreate(0,"DnwardTrendline",OBJ_TREND,0,pArray[candleHigh].time,pArray[candleHigh].high,
                   pArray[0].time,pArray[0].high);
      ObjectSetInteger(0,"DnwardTrendline",OBJPROP_COLOR,Blue);
      ObjectSetInteger(0,"DnwardTrendline",OBJPROP_STYLE,STYLE_SOLID);
      ObjectSetInteger(0,"DnwardTrendline",OBJPROP_WIDTH,3);
      ObjectSetInteger(0,"DnwardTrendline",OBJPROP_RAY_RIGHT,true);
     }
  }
//+------------------------------------------------------------------+
```

Differences in this code:

Creating an array of candles of high prices.

```
double pHigh[];
```

Sorting the created array from current data.

```
ArraySetAsSeries(pHigh,true);
```

Filling the array with data.

```
int copy_high = CopyHigh(_Symbol,_Period,0,candles,pHigh);
```

Creating and calculating the high of the (candle) variable

```
int candleHigh = ArrayMaximum(pHigh,0,candles);
```

Deleting any previous downward trend line

```
ObjectDelete(_Symbol,"DnwardTrendline");
```

Creating a new downward trend line

```
ObjectCreate(_Symbol,"DnwardTrendline",OBJ_TREND,0,pArray[candleHigh].time,pArray[candleHigh].high,pArray[0].time,pArray[0].high);
```

Determining the downward trend line color to be blue by using the "OBJPROP\_COLOR"

```
ObjectSetInteger(0,"DnwardTrendline",OBJPROP_COLOR,Blue);
```

Determining the downward trend line style to be solid by using "OBJPROP\_STYLE" and STYLE\_SOLID

```
ObjectSetInteger(0,"DnwardTrendline",OBJPROP_STYLE,STYLE_SOLID);
```

Determining the downward trend line width to be 3 value by using the "OBJPROP\_WIDTH"

```
ObjectSetInteger(0,"DnwardTrendline",OBJPROP_WIDTH,3);
```

Determining the downward trend line ray by using the "OBJPROP\_RAY\_RIGHT"

```
ObjectSetInteger(0,"DnwardTrendline",OBJPROP_RAY_RIGHT,true);
```

After compiling without any error we will find the expert also in the navigator window the same as we mentioned before. After attaching the EA we will be ready to get the required action based on this EA and we can find the downward trend line is drawn the same as the following:

![Downward trendline example](https://c.mql5.com/2/50/Downward_trendline_example.png)

As we can see on the chart in the top right corner we have the EA is attached and we have the downtrend line drawn above prices as there is a buying power around this level.

### Support levels and MQL5

The support level is a price level or zone that can be found below current prices and we can find a rebound to up around it because there is a buying power around these levels. So, they are very important as we can trigger them as buying points. There are many forms of support levels and one of them is the horizontal support level. The following is what this form looks like:

![ Support lvl](https://c.mql5.com/2/50/Support_lvl.png)

The following is an example of the support level from the market:

![Support lvl 1](https://c.mql5.com/2/50/Support_lvl_1.png)

As we can see in the previous chart we have a green support line below prices that can be used as a rejection level to the upside.

We need the program to check price lows and if there is any support line every tick. Then delete the previous support line and create an updated green one below the lows of the price. We can draw and deal with this support line by MQL5 the same as the full below code:

```
//+------------------------------------------------------------------+
//|                                          Support Line System.mq5 |
//+------------------------------------------------------------------+
void OnTick()
  {
   int candles=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);
   double pLow[];
   ArraySetAsSeries(pLow,true);
   CopyLow(_Symbol,_Period,0,candles,pLow);
   int candleLow = ArrayMinimum(pLow,0,candles);
   MqlRates pArray[];
   ArraySetAsSeries(pArray,true);
   int Data = CopyRates(_Symbol,_Period,0,candles,pArray);
   ObjectDelete(_Symbol,"supportLine");
   ObjectCreate(_Symbol,"supportLine",OBJ_HLINE,0,pArray[candleLow].time,pArray[candleLow].low,pArray[0].time,pArray[0].low);
   ObjectSetInteger(0,"supportLine",OBJPROP_COLOR,Green);
   ObjectSetInteger(0,"supportLine",OBJPROP_STYLE,STYLE_SOLID);
   ObjectSetInteger(0,"supportLine",OBJPROP_WIDTH,3);
   ObjectSetInteger(0,"supportLine",OBJPROP_RAY,true);
  }
//+------------------------------------------------------------------+
```

Difference in this code:

Deleting any previous support lines with the name "supportLine"

```
ObjectDelete(_Symbol,"supportLine");
```

Creating a new support line by using "OBJ\_HLINE" as a type of the required object

```
ObjectCreate(_Symbol,"supportLine",OBJ_HLINE,0,pArray[candleLow].time,pArray[candleLow].low, pArray[0].time,pArray[0].low);
```

Determining the support line color by using the "OBJPROP\_COLOR"

```
ObjectSetInteger(0,"supportLine",OBJPROP_COLOR,Green);
```

Determining the support line style by using OBJPROP\_STYLE and STYLE\_SOLID

```
ObjectSetInteger(0,"supportLine",OBJPROP_STYLE,STYLE_SOLID);
```

Determining the support line width by using the "OBJPROP\_WIDTH"

```
ObjectSetInteger(0,"supportLine",OBJPROP_WIDTH,3);
```

Determining the support line ray by using the "OBJPROP\_RAY"

```
ObjectSetInteger(0,"supportLine",OBJPROP_RAY,true);
```

After compiling this code without any error, we can attach it to the chart the same as we mentioned before. After that, we will find it attached to the chart and we can find its action as we can find the support line is drawn the same as the below example:

![Support line](https://c.mql5.com/2/50/Support_line.png)

As we can see in the previous chart that we have the Support System EA is attached the same as what we find in the top right corner and we have a support line below prices.

### Resistance levels and MQL5

The resistance level is a price level or zone that can be found above current prices and we can find a rebound to down around it because there is a selling power around these levels. So, they are very important as we can trigger them as selling points. There are many forms of these resistance levels and one of them is the horizontal resistance level the following is what this form looks like:

![ Resistance lvl](https://c.mql5.com/2/50/Resistance_lvl.png)

The following is an example of the resistance from the market:

![ Resistance lvl 1](https://c.mql5.com/2/50/Resistance_lvl_1.png)

As we can see in the previous chart we have a red resistance line above prices that can be used as a rejection level to the downside as there is a buying power around this level.

We need the program to check price highs and if there is any resistance line every tick. Then delete the previous resistance line and create an updated red one above the highs of the price. We can draw and deal with this resistance line by MQL5 the same as the full below code:

```
//+------------------------------------------------------------------+
//|                                       Resistance Line System.mq5 |
//+------------------------------------------------------------------+
void OnTick()
  {
   int candles=ChartGetInteger(0,CHART_FIRST_VISIBLE_BAR,0);
   double pHigh[];
   ArraySetAsSeries(pHigh,true);
   CopyHigh(_Symbol,_Period,0,candles,pHigh);
   int candleHigh = ArrayMaximum(pHigh,0,candles);
   MqlRates pArray[];
   ArraySetAsSeries(pArray,true);
   int Data = CopyRates(_Symbol,_Period,0,candles,pArray);
   ObjectDelete(_Symbol,"resistanceLine");
   ObjectCreate(_Symbol,"resistanceLine",OBJ_HLINE,0,pArray[candleHigh].time,pArray[candleHigh].high,pArray[0].time,pArray[0].high);
   ObjectSetInteger(0,"resistanceLine",OBJPROP_COLOR,Red);
   ObjectSetInteger(0,"resistanceLine",OBJPROP_STYLE,STYLE_SOLID);
   ObjectSetInteger(0,"resistanceLine",OBJPROP_WIDTH,3);
   ObjectSetInteger(0,"DnwardTrendline",OBJPROP_RAY_RIGHT,true);
  }
//+------------------------------------------------------------------+
```

Difference in this code:

Deleting any previous resistance line with the name "resistanceLine"

```
ObjectDelete(_Symbol,"resistanceLine");
```

Creating a new resistance line by using "OBJ\_HLINE" as a type of the required object

```
ObjectCreate(_Symbol,"resistanceLine",OBJ_HLINE,0,pArray[candleHigh].time,pArray[candleHigh].high,pArray[0].time,pArray[0].high);
```

Determining the resistance line color by using OBJPROP\_COLOR

```
ObjectSetInteger(0,"resistanceLine",OBJPROP_COLOR,Red);
```

Determining the resistance line style by using OBJPROP\_STYLE and STYLE\_SOLID

```
ObjectSetInteger(0,"resistanceLine",OBJPROP_STYLE,STYLE_SOLID);
```

Determining the resistance line width by using OBJPROP\_WIDTH

```
ObjectSetInteger(0,"resistanceLine",OBJPROP_WIDTH,3);
```

Determining the resistance line ray by using OBJPROP\_RAY\_RIGHT

```
ObjectSetInteger(0,"DnwardTrendline",OBJPROP_RAY_RIGHT,true);
```

After compiling this code without any error, we can attach it to the chart the same as we mentioned before. After that, we will find the Resistance System  EA is attached to the chart and we can find its action the same as the following:

![Resistance line](https://c.mql5.com/2/50/Resistance_line.png)

As we can see in the previous chart that we have the Resistance System EA is attached the same as what we can find in the top right corner and we have a resistance line below the prices.

### Conclusion

Now, it is supposed that you understand three of the important lines that can be used in our trading in detail. The trend lines (upward, and downward trend lines), Support, and resistance lines as we learned what they are and how we can use them. We learned also how we can deal with them by the MQL5 to draw them automatically by creating our own system for every one of them. You can also develop this code by adding sending orders to execute trades based on them and you must test them before using them on your real account to make sure that they are profitable as the main objective of this article is educational only.

I hope that you find this article useful for you and you learned how to use these mentioned tools in your favor by using them as a part of another advanced system or individually. So, I hope that you got useful insights into the topic of this article or any related topic to get better trading results and enhance them with useful and profitable tools in addition to hoping that you tried to test and write mentioned codes by yourself to get benefit and improve your coding skills.

If you find this article useful and you like to read more articles about the MQL5 and how to design a trading system using the most popular technical indicators you can read my previous article to learn more about that as you can find articles like how to design a trading system based on the most popular technical indicators like The Moving Average, RSI, MACD, Stochastic, Parabolic SAR, etc.

If you want also to learn more also about some of the basics of MQL5 and why and how to design your algorithmic trading system, you can also read my previous article about that.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11538.zip "Download all attachments in the single ZIP archive")

[Support\_System.mq5](https://www.mql5.com/en/articles/download/11538/support_system.mq5 "Download Support_System.mq5")(1.04 KB)

[Resistance\_System.mq5](https://www.mql5.com/en/articles/download/11538/resistance_system.mq5 "Download Resistance_System.mq5")(1.07 KB)

[UpwardTrendline\_System.mq5](https://www.mql5.com/en/articles/download/11538/upwardtrendline_system.mq5 "Download UpwardTrendline_System.mq5")(1.14 KB)

[DownwardTrendline\_System.mq5](https://www.mql5.com/en/articles/download/11538/downwardtrendline_system.mq5 "Download DownwardTrendline_System.mq5")(1.15 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/436010)**
(7)


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
20 Mar 2023 at 15:43

**MrBrooklin [#](https://www.mql5.com/ru/forum/439977#comment_45332169):**

Interesting article for beginners, but for EAs to work correctly, a little tweaking of codes is required.

Regards, Vladimir.

Thanks for your kind comment, I will review it.


![Andrei Luiz Pereira](https://c.mql5.com/avatar/2022/10/633C1402-C350.jpg)

**[Andrei Luiz Pereira](https://www.mql5.com/en/users/andreipereira96)**
\|
4 Apr 2023 at 16:09

The way you drew the line is very nice. One observation about [trend lines](https://www.mql5.com/en/docs/constants/objectconstants/enum_object "MQL5 documentation: object types"): It doesn't do much good if the second anchor point is always the 0 index bar, because with each new bar the trend line will be updated and so the trend line will be of no use. The ideal would be to define the second anchor point based on a specific criterion. For example, the second bar with the highest price, with a minimum distance of 30 bars from the first, or take the bar with the highest price the next day, and so on.

Congratulations on the article.


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
6 Apr 2023 at 12:46

**Andrei Pereira trend lines: It doesn't do much good if the second anchor point is always the 0 index bar, because with each new bar the trend line will be updated and so the trend line will be of no use. The ideal would be to define the second anchor point based on a specific criterion. For example, the second bar with the highest price, at least 30 bars away from the first, or take the bar with the highest price the next day, and so on.**

**Congratulations on the article.**

Thanks for your comment, it's a good observation.

![MrBrooklin](https://c.mql5.com/avatar/2022/11/6383f326-c19f.png)

**[MrBrooklin](https://www.mql5.com/en/users/mrbrooklin)**
\|
13 Jan 2024 at 09:01

**Mohamed Abdelmaaboud [#](https://www.mql5.com/ru/forum/439977#comment_45738451):**

Thanks for the kind comment, I will revise it.

It's been almost 10 months and no changes to his codes have been made by the author of the article. ))

Regards, Vladimir.

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
13 Jan 2024 at 14:50

**MrBrooklin [#](https://www.mql5.com/ru/forum/439977#comment_51689203):**

Almost 10 months have passed, and the author of the article hasn't made any changes in his codes. ))

Regards, Vladimir.

Thank you for your reminder and contribution. It will be sent for editing.

![Developing a trading Expert Advisor from scratch (Part 29): The talking platform](https://c.mql5.com/2/48/development__5.png)[Developing a trading Expert Advisor from scratch (Part 29): The talking platform](https://www.mql5.com/en/articles/10664)

In this article, we will learn how to make the MetaTrader 5 platform talk. What if we make the EA more fun? Financial market trading is often too boring and monotonous, but we can make this job less tiring. Please note that this project can be dangerous for those who experience problems such as addiction. However, in a general case, it just makes things less boring.

![Neural networks made easy (Part 27): Deep Q-Learning (DQN)](https://c.mql5.com/2/48/Neural_networks_made_easy.png)[Neural networks made easy (Part 27): Deep Q-Learning (DQN)](https://www.mql5.com/en/articles/11369)

We continue to study reinforcement learning. In this article, we will get acquainted with the Deep Q-Learning method. The use of this method has enabled the DeepMind team to create a model that can outperform a human when playing Atari computer games. I think it will be useful to evaluate the possibilities of the technology for solving trading problems.

![DoEasy. Controls (Part 20): SplitContainer WinForms object](https://c.mql5.com/2/49/MQL5-avatar-doeasy-library-2__1.png)[DoEasy. Controls (Part 20): SplitContainer WinForms object](https://www.mql5.com/en/articles/11524)

In the current article, I will start developing the SplitContainer control from the MS Visual Studio toolkit. This control consists of two panels separated by a vertical or horizontal movable separator.

![Neural networks made easy (Part 26): Reinforcement Learning](https://c.mql5.com/2/48/Networks_easy_26.png)[Neural networks made easy (Part 26): Reinforcement Learning](https://www.mql5.com/en/articles/11344)

We continue to study machine learning methods. With this article, we begin another big topic, Reinforcement Learning. This approach allows the models to set up certain strategies for solving the problems. We can expect that this property of reinforcement learning will open up new horizons for building trading strategies.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/11538&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069189761962279252)

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