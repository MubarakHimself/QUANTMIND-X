---
title: Learn how to design a trading system by Envelopes
url: https://www.mql5.com/en/articles/10478
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:14:47.572044
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/10478&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069274334163305061)

MetaTrader 5 / Trading


### Introduction

In my previous article, I wrote about one of the most popular indicators — Bollinger Bands — which allows us to trade bands. This article is available at the following link: [Learn how to design a trading system by Bollinger Bands](https://www.mql5.com/en/articles/3039/)

As I mentioned in this article about the Bollinger Band, the concept of band trading has a variety of existing methods that we can use. Every method has a different approach as per its calculation and its strategy but the common thing is the concept that we trade the bands.

Now, I will share another method withing the concept of band trading — the Envelopes indicator. The indicator is represented by two bands which surround the price and its moving average. In this article, we will consider some more details, including the following topics:

- [Envelopes definition](https://www.mql5.com/en/articles/10478#definition)
- [Envelopes strategy](https://www.mql5.com/en/articles/10478#strategy)
- [Envelopes trading strategy system blueprint](https://www.mql5.com/en/articles/10478#blueprint)
- [Envelopes trading system](https://www.mql5.com/en/articles/10478#system)
- [Conclusion](https://www.mql5.com/en/articles/10478#conclusion)

Through these topics, I will try to explain the indicator in a way to allow a deep understand of the indicator. The information should assist in developing a better trading plan or strategy which can generate better results. So, through the topic of Envelopes definition, I will share with you more details about the concept of the Envelopes indicator. We will consider its calculation for a better understanding of the concept. Then, we will try to develop strategies which will enhance our trading results.

In the Envelopes Strategy part, I will share with you some simple basic strategies which can be used in our favor as examples of how can we use it and this can open our eyes on more strategies to get much better results. Then, in the 'Envelopes trading strategy system blueprint' part, which I consider a very interesting topic, I will share with you a blueprint of how to create a trading system using strategies of the Envelopes indicator. Later, in the 'Envelopes trading system' part, I will share with you the ways to design a trading system based in the Envelopes indicator strategies in the MQL5 language, which can further be used in MetaTrader 5.

I hope that this article will be useful for you and that you will find some insights which will help you develop your own trading system or will help you improve your trading results.

**Important!** All codes in this article are written in MQL5. These codes can be executed in MetaTrader 5. So, you will need to have MetaTrader 5 to run the applications. If you do not have the MetaTrader 5 terminal yet, you can download it from the following link: [https://www.metatrader5.com/en/download](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download")

After downloading it the following picture will show the window of MetaTrader 5:

![MT5 platform](https://c.mql5.com/2/45/1-_MT5_platform.png)

To open the MetaQuotes Language Editor (MetaEditor) to write codes the following picture shows you how to do that:

![Metaeditor opening 1](https://c.mql5.com/2/45/2-_Metaeditor_opening_1.png)

Another method to open it:

![Metaeditor opening 2](https://c.mql5.com/2/45/3-_Metaeditor_opening_2.png)

Or you can press F4 while the MetaTrader 5 window is open.

Then the following window will be opened:

![Metaeditor window](https://c.mql5.com/2/45/4-_Metaeditor_window.png)

Here is how to open a new file:

![Metaeditor window](https://c.mql5.com/2/45/5-_Metaeditor_window.png)

![Metaeditor - New file](https://c.mql5.com/2/45/6-_Metaeditor_-_New_file.png)

1. Open a new file to write an Expert Advisor
2. Open a new file to write a custom indicator
3. Open a new file to write a script

For more info, you can read my previous article: [Learn Why and How to Design Your Algorithmic Trading System](https://www.mql5.com/en/articles/10293)

**Disclaimer**: The content in this article is made for educational purpose only not for anything else. So, you will be responsible for any action you take based on the content of this article, as the content of this article does not guarantee any kind of results.

Now, let’s start our journey to studying the Envelopes indicator to see how it may be beneficial to our trading.

### Envelopes definition

The Envelopes indicator is one of the methods of trading bands and like what I mentioned before, there are many methods of trading bands like Bollinger Bands, Channels, Envelopes and others.

Simply put, we can say that Envelopes are two bands that surround the moving average by a certain fixed percentage and normally they surround prices. These Envelopes can be used to filter trend movements and to trade during sideways.

Sometimes during trends, we face some movements in the opposite direction, such as false breakouts or whipsaws. Such movements can generate a fake signal, after which the situation returns to its opposite scenario. For example, if prices cross the moving average upwards, this serves as a signal of an upward movement. Then they turn and cross the moving average downwards, i.e. the opposite of the previous upward crossover. By the way, you can read more about the Moving Averages, about how to use them and to create a trading system based on them, you can read my previous article: [Learn how to design different Moving Average systems](https://www.mql5.com/en/articles/3040)

![MA crosses false break](https://c.mql5.com/2/45/1-_MA_crosses_false_break.png)

In the above figure, after the price crosses the Moving Average from above, it turns and moves up again.

Sometimes also, we have sideways and there is a balance between buyers and sellers or there is no an obvious trend and we need to benefit from this movement.

In these cases, the Envelopes indicator can be useful for us as it can be a filter for movements during trends. If the price crosses the Moving Average upwards, we wait for the price to cross above the upper band of the Envelopes indicator.

![MA crosses without envelopes](https://c.mql5.com/2/45/2-_MA_crosses_without_envelopes.png)

As you can see in the above figure for the same movement that we mentioned before, the price crossed the Moving Average upwards but did not cross the upper band of Envelopes indicator. So, we cannot take action according to this crossover only then we found that prices moved to down again.

And here I need to mention something interesting about Technical Analysis as it has many methods and strategies that can be combined in a way which generates better results.

Like what I mention before, Envelopes are two bands surrounding prices, so the wider bands provide a lower probability to be touched and the narrow bands provide a higher probability to be touched.

Now how can the Envelopes indicator be calculated? Envelopes are set at a certain percentage from the moving average. This percentage depends on the price action, time frame and the user strategy.

To calculate the Envelopes:

1. Calculate the Moving Average
2. The % of Envelopes = Moving Average \* %
3. Upper band of Envelopes = Moving Average + The determined % of Moving Average
4. Lower band of Envelopes = Moving Average - The determined % of Moving Average

Let us see an example of how to do that.

> If we have the following data for prices of 14 days:

![Calculation example](https://c.mql5.com/2/45/3-_calculation_example.png)

- Get the Moving Average (here we calculate a simple moving average, but you can use any - some of them are described in detail in my previous article: [Learn how to design different Moving Average systems](https://www.mql5.com/en/articles/3040)

![Env Calculation](https://c.mql5.com/2/45/4-_Env_Calculation.png)

- Calculate the % of Envelopes as per your preferences

![Env Calculation 2](https://c.mql5.com/2/45/5-_Env_Calculation_2.png)

- Calculate the Upper Band of Envelopes

![Env Calculation 3](https://c.mql5.com/2/45/6-_Env_Calculation_3.png)

- Calculate the Lower Band of Envelopes

![Env Calculation 4](https://c.mql5.com/2/45/7-_Env_Calculation_4.png)

This calculation will be performed at every tick, then values of upper band will be plotted above prices and values of lower band will be plotted below prices.

But today we will not do that manually as we have it calculated and built-in in the platform and all what we need to do to choose it and it will be plotted on our chart and the following pictures are about how to do that:

![Indicator insert](https://c.mql5.com/2/45/8-_Indicator_insert.png)

![Indicator insert 1](https://c.mql5.com/2/45/9-_Indicator_insert_1.png)

![Indicator parameter](https://c.mql5.com/2/45/10-_Indicator_parameter.png)

The figure above shows the parameters of Envelopes:

1. To determine the period of moving average.
2. % Of Envelopes.
3. Kind of price (close, Median price, Typical price, etc.)
4. Value to be shifted for the Envelopes on the chart.
5. Type of Moving Average.
6. Color of upper band of Envelopes.
7. Style of line of upper band of Envelopes.
8. Color of lower band of Envelopes.
9. Style of line of lower band of Envelopes.

And after determining all parameter, the indicator will appear on the chart, just as is shown in the following picture:

![Indicator on chart](https://c.mql5.com/2/45/11-_Indicator_on_chart.png)

### Envelopes strategy

In this part, I will share with you some of Envelopes strategies. These strategies are used according to market condition or market direction.

As we all know, in the market there us an uptrend, a downtrend, and sideways directions. According to the specific market direction, we will use a specific strategy which will allow us to benefit from this market movement.

It is possible to can combine the Envelopes with other tools and create a strategy based on this combination. The results of such a combined  use can be better than using one tool only. So, you can combine the Envelopes indicator with another tool like price action pattern for example: take a signal from Envelopes indicator and take another signal or confirmation signal from a price pattern. Optionally, it is possible to use another technical analysis indicator. I think this is an amazing approach in technical analysis trading to combine more than one concept in a way to get better results and improve your decisions.

According to that, we will divide our strategies to strategies which can be used during trending movement and others which can be used during sideways movements.

**During trending markets**

- Strategy 1: Uptrend

During uptrend markets, prices move in the uptrend making higher lows and higher highs. Now, we need to know how prices and the Envelopes indicator will act during the uptrend.

As this indicator is a percentage of moving average which is an average for a period of prices, during the uptrend prices will move above upper band of Envelopes indicator. So, our strategy here during uptrend will be as follows:

We will get a buy signal when prices cross the upper band of Envelopes upward, then take profit when prices cross the lower band of Envelopes downward.

- Prices > Upper band = buy
- Prices < Lower band = take profit

![Strategy 1 - Uptrend](https://c.mql5.com/2/45/7-_Strategy_1_-_Uptrend.png)

- Strategy 2: Downtrend

During downtrend markets, prices move in the downtrend making lower highs and lower lows. Now, we need to know how prices and the Envelopes indicator will act during downtrend.

During downtrend prices will move below lower band of Envelopes indicator. So, our strategy here during downtrend will be as follows:

We will get a short signal when prices cross the lower band of Envelopes downward, then take profit when prices cross upper band of Envelopes indicator upward.

- Prices < Lower band = short
- Prices > Upper band = take profit

![Strategy 2 - downtrend](https://c.mql5.com/2/45/8-_Strategy_2_-_downtrend.png)

- Strategy 3: Sideways:

During sideways markets, prices move in some of sideways formations making same highs and same lows. Or there is a balance between buyers and sellers without controlling one party on the other and this the simple form ranging between two levels. It shows the balance. This does not mean that there are not other forms for sideways but it is good to know that sideways movement is any movement or form except uptrend and downtrend. Now, we need to know how prices and the Envelopes indicator will act during sideways.

During sideways, prices will move between lower band and upper band of the Envelopes indicator. So, our strategy here during sideways will be as follows:

We will get a buy signal when prices touch the lower band of Envelopes, then take profit when prices touch the upper band of Envelopes. We will get a short signal when prices touch the upper band of Envelopes, then take profit when prices touch the lower band of Envelopes.

- Prices = Lower band = buy
- Prices = Upper band = take profit

- Prices = Upper band = short
- Prices = Lower band = take profit

![Strategy 3 - Sideways](https://c.mql5.com/2/45/9-_Strategy_3_-_Sideways.png)

Please note that these strategies are only provided here to help you in understanding how the indicator can be used. This requires improvement and further development for proper use. Be sure to always thoroughly test every strategy you read or learn. Never use them without testing and checking its results especially if it was made for educational purposes.

### Envelopes trading strategy system blueprint

In this interesting part, we will design our mentioned strategies as a blueprint to determine what we want to do exactly and what we want the program to do. So, let us do that.

**During trending markets**

- Strategy 1: Uptrend

- Prices > Upper band = buy
- Prices < Lower band = take profit

So, the blueprint for this strategy will be like the following:

![Uptrend strategy blueprint](https://c.mql5.com/2/45/12-_Uptrend_strategy_blueprint.png)

- Strategy 2: Downtrend:

- Prices < Lower band = short
- Prices > Upper band = take profit

So, the blueprint for this strategy will be like the following:

![Downtrend strategy blueprint](https://c.mql5.com/2/45/13-_Downtrend_strategy_blueprint.png)

**During Sideways**,

- Strategy 3: Sideways:

- Prices = Lower band = buy
- Prices = Upper band = take profit

- Prices = Upper band = short
- Prices = Lower band = take profit

So, the blueprint for this strategy will be like the following:

![Sideways strategy blueprint](https://c.mql5.com/2/45/14-_Sideways_strategy_blueprint.png)

### Envelopes trading system

In this part, I will share with you how you can code these strategies using MQL5, in the form of EA (Expert Advisors) programs. This can help someone enhance their trading and identify signals automatically and easily. Now, I will share with you how to code a basic Envelopes EA and give instructions to our program to comment Envelopes indicator upper and lower values on the chart.

Let us design a program which will allow us to see values of Envelopes upper and lower bands as comments on the chart:

```
//+------------------------------------------------------------------+
//|                                          Simple Envelopes EA.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array for price info
   MqlRates PriceInfo[];

   //Sorting it from current data to old data
   ArraySetAsSeries(PriceInfo, true);

   //Copy prices data to array
   int Data = CopyRates(Symbol(), Period (), 0, Bars(Symbol(), Period()), PriceInfo);

   //Creating two arrays for prices of upper and lower band
   double UpperBandArray[];
   double LowerBandArray[];

   //Identify Envelopes indicator
   int EnvelopesIdentify = iEnvelopes(_Symbol,_Period,14,0,MODE_SMA,PRICE_CLOSE,0.100);

   //Copying prices data to arrays
   CopyBuffer(EnvelopesIdentify,0,0,1,UpperBandArray);
   CopyBuffer(EnvelopesIdentify,1,0,1,LowerBandArray);

   //Calculation for the current data
   double UpperBandValue=NormalizeDouble(UpperBandArray[0],6);
   double LowerBandValue=NormalizeDouble(LowerBandArray[0],6);

   //Comments on the chart with values of Band of Envelopes
   Comment("Envelopes Upper Band =",UpperBandValue,"\n" "Envelopes Lower Band =",LowerBandValue,"\n");

  }
//+------------------------------------------------------------------+
```

The code in MetaTrader 5 can be executed as is shown in the following pictures:

![Basic Envelopes EA Insert](https://c.mql5.com/2/45/11-_Basic_Envelopes_EA_Insert.png)

Drag and drop the file on the chart or double click on the file then the code will be executed and the following window will be opened.

![1](https://c.mql5.com/2/45/1.png)

After that the program will appear on the chart:

![11](https://c.mql5.com/2/45/11__2.png)

After the execution of the code in the MetaTrader 5 trading terminal we will see the following chart and signals on it:

![Basic Envelopes EA](https://c.mql5.com/2/45/10-_Basic_Envelopes_EA.png)

We have two bands of the Envelopes indicator — Lower band and Upper band — and we can see their values on the chart.

Now, we will code our mentioned three strategies, strategy one is about buy and take profit during uptrend, strategy two is about short and take profit during downtrend, then strategy three is about sideways and buy on the lower band and take profit on the upper band then short on the upper band and take profit on the lower band.

**During trending markets**

- Strategy 1: Uptrend:

- Prices > Upper band = buy
- Prices < Lower band = take profit

The following code is to design the program to execute this strategy:

```
//+------------------------------------------------------------------+
//|                                   Envelopes uptrend Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array for price info
   MqlRates PriceInfo[];

   //Sorting it from current data to old data
   ArraySetAsSeries(PriceInfo, true);

   //Copy prices data to array
   int Data = CopyRates(Symbol(), Period (), 0, Bars(Symbol(), Period()), PriceInfo);

   //Creating two arrays for prices of upper and lower band
   double UpperBandArray[];
   double LowerBandArray[];

   //Identify Envelopes indicator
   int EnvelopesIdentify = iEnvelopes(_Symbol,_Period,14,0,MODE_SMA,PRICE_CLOSE,0.100);

   //Copying prices data to arrays
   CopyBuffer(EnvelopesIdentify,0,0,1,UpperBandArray);
   CopyBuffer(EnvelopesIdentify,1,0,1,LowerBandArray);

   //Calculation for the current data
   double UpperBandValue=NormalizeDouble(UpperBandArray[0],6);
   double LowerBandValue=NormalizeDouble(LowerBandArray[0],6);

   //Comment Buy signal on the chart or no signal
   if (PriceInfo[0].close > UpperBandValue)
   Comment("Envelopes Buy Signal");

   if (PriceInfo[0].close < LowerBandValue)
   Comment("Envelopes Take Profit or Stop Loss");

  }
//+------------------------------------------------------------------+
```

Code from MetaTrader can be executed as is shown in the following pictures:

![Envelopes Uptrend Strategy Insert](https://c.mql5.com/2/45/13-_Envelopes_Uptrend_Strategy_Insert.png)

Drag and drop the file on the chart or double click on the file then the code will be executed and the following window will be opened.

![2](https://c.mql5.com/2/45/2.png)

After that the program is attached to the chart:

![22](https://c.mql5.com/2/45/22.png)

After the execution of the code in the MetaTrader 5 trading terminal we will see the following chart and signals on it:

![Envelopes Uptrend Strategy - Buy signal](https://c.mql5.com/2/45/11-_Envelopes_Uptrend_Strategy_-_Buy_signal.png)

As we can see on the previous chart, the program gives us a buy signal when prices cross above the upper band during uptrend.

![Envelopes Uptrend Strategy - Take Profit](https://c.mql5.com/2/45/12-_Envelopes_Uptrend_Strategy_-_Take_Profit.png)

As we can see on the previous chart, the program gives us a take profit signal when prices cross below the lower band.

- Strategy 2: Downtrend

- Prices < Lower band = short
- Prices > Upper band = take profit

The following code is to design the program to execute this strategy:

```
//+------------------------------------------------------------------+
//|                                 Envelopes Downtrend Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array for price info
   MqlRates PriceInfo[];

   //Sorting it from current data to old data
   ArraySetAsSeries(PriceInfo, true);

   //Copy prices data to array
   int Data = CopyRates(Symbol(), Period (), 0, Bars(Symbol(), Period()), PriceInfo);

   //Creating two arrays for prices of upper and lower band
   double UpperBandArray[];
   double LowerBandArray[];

   //Identify Envelopes indicator
   int EnvelopesIdentify = iEnvelopes(_Symbol,_Period,14,0,MODE_SMA,PRICE_CLOSE,0.100);

   //Copying prices data to arrays
   CopyBuffer(EnvelopesIdentify,0,0,1,UpperBandArray);
   CopyBuffer(EnvelopesIdentify,1,0,1,LowerBandArray);

   //Calculation for the current data
   double UpperBandValue=NormalizeDouble(UpperBandArray[0],6);
   double LowerBandValue=NormalizeDouble(LowerBandArray[0],6);

   //Comment Buy signal on the chart or no signal
   if (PriceInfo[0].close < LowerBandValue)
   Comment("Envelopes Short Signal");

   if (PriceInfo[0].close > UpperBandValue)
   Comment("Envelopes Take Profit or Stop Loss");

  }
//+------------------------------------------------------------------+
```

We can execute the code from MetaTrader 5 like what we can see at the following pictures:

![Envelopes Downtrend Strategy Insert](https://c.mql5.com/2/45/16-_Envelopes_Downtrend_Strategy_Insert.png)

We can drag and drop the file on the chart or double click on the file then the code will be executed and the following window will be opened.

![3](https://c.mql5.com/2/45/3.png)

After that we see the program attached to the chart:

![33](https://c.mql5.com/2/45/33.png)

![Envelopes Downtrend Strategy - Short signal](https://c.mql5.com/2/45/13-_Envelopes_Downtrend_Strategy_-_Short_signal.png)

As we can see on the previous chart, the program gives us a short signal when prices cross below the lower band during downtrend.

![Envelopes Downtrend Strategy - Take Profit](https://c.mql5.com/2/45/14-_Envelopes_Downtrend_Strategy_-_Take_Profit.png)

As we can see on the previous chart, the program gives us a take profit signal when prices cross above the upper band during downtrend.

- Strategy 3: Sideways

- Prices = Lower band = buy
- Prices = Upper band = take profit

The following code is to design the program to execute this strategy:

```
//+------------------------------------------------------------------+
//|                              Envelopes Sideways Buy Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array for price info
   MqlRates PriceInfo[];

   //Sorting it from current data to old data
   ArraySetAsSeries(PriceInfo, true);

   //Copy prices data to array
   int Data = CopyRates(Symbol(), Period (), 0, Bars(Symbol(), Period()), PriceInfo);

   //Creating two arrays for prices of upper and lower band
   double UpperBandArray[];
   double LowerBandArray[];

   //Identify Envelopes indicator
   int EnvelopesIdentify = iEnvelopes(_Symbol,_Period,14,0,MODE_SMA,PRICE_CLOSE,0.150);

   //Copying prices data to arrays
   CopyBuffer(EnvelopesIdentify,0,0,1,UpperBandArray);
   CopyBuffer(EnvelopesIdentify,1,0,1,LowerBandArray);

   //Calculation for the current data
   double UpperBandValue=NormalizeDouble(UpperBandArray[0],6);
   double LowerBandValue=NormalizeDouble(LowerBandArray[0],6);

   //Comment Buy signal on the chart or no signal
   if (PriceInfo[0].close <= LowerBandValue)
   Comment("Envelopes Buy Signal");

   if (PriceInfo[0].close >= UpperBandValue)
   Comment("Envelopes Take Profit");

   if (PriceInfo[0].close > LowerBandValue
   &&  PriceInfo[0].close < UpperBandValue)
   Comment("No Signal");
  }
//+------------------------------------------------------------------+
```

The code in MetaTrader 5 can be executed as is shown in the following pictures:

![Envelopes Sideways Buy Strategy Insert](https://c.mql5.com/2/45/20-_Envelopes_Sideways_Buy_Strategy_Insert.png)

Drag and drop the file on the chart or double click on the file then the code will be executed and the following window will be opened.

![4](https://c.mql5.com/2/45/4.png)

After that the program will appear on the chart:

![44](https://c.mql5.com/2/45/44.png)

![Envelopes Sideways Buy Strategy - No signal](https://c.mql5.com/2/45/15-_Envelopes_Sideways_Buy_Strategy_-_No_signal.png)

After the execution of the code in the MetaTrader 5 trading terminal we will see the following chart and signals on it:

![Envelopes Sideways Buy Strategy - Buy signal](https://c.mql5.com/2/45/16-_Envelopes_Sideways_Buy_Strategy_-_Buy_signal.png)

As we can see on the previous chart, the program gives us a buy Signal as prices touch the lower band during sideways.

![Envelopes Sideways Buy Strategy - Take Profit](https://c.mql5.com/2/45/17-_Envelopes_Sideways_Buy_Strategy_-_Take_Profit.png)

As we can see on the previous chart, the program gives us a take profit Signal as prices touch the upper band during sideways

- Prices = Upper band = short
- Prices = Lower band = take profit

The following code is to design the program to execute this strategy:

```
//+------------------------------------------------------------------+
//|                            Envelopes Sideways Short Strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //Creating an array for price info
   MqlRates PriceInfo[];

   //Sorting it from current data to old data
   ArraySetAsSeries(PriceInfo, true);

   //Copy prices data to array
   int Data = CopyRates(Symbol(), Period (), 0, Bars(Symbol(), Period()), PriceInfo);

   //Creating two arrays for prices of upper and lower band
   double UpperBandArray[];
   double LowerBandArray[];

   //Identify Envelopes indicator
   int EnvelopesIdentify = iEnvelopes(_Symbol,_Period,14,0,MODE_SMA,PRICE_CLOSE,0.150);

   //Copying prices data to arrays
   CopyBuffer(EnvelopesIdentify,0,0,1,UpperBandArray);
   CopyBuffer(EnvelopesIdentify,1,0,1,LowerBandArray);

   //Calculation for the current data
   double UpperBandValue=NormalizeDouble(UpperBandArray[0],6);
   double LowerBandValue=NormalizeDouble(LowerBandArray[0],6);

   //Comment Buy signal on the chart or no signal
   if (PriceInfo[0].close >= UpperBandValue)
   Comment("Envelopes Short Signal");

   if (PriceInfo[0].close <= LowerBandValue)
   Comment("Envelopes Take Profit");

   if (PriceInfo[0].close > LowerBandValue
   &&  PriceInfo[0].close < UpperBandValue)
   Comment("No Signal");

  }
//+------------------------------------------------------------------+
```

The code in MetaTrader 5 can be executed as is shown in the following pictures:

![Envelopes Sideways Sell Strategy Insert](https://c.mql5.com/2/45/24-_Envelopes_Sideways_Sell_Strategy_Insert.png)

Drag and drop the file on the chart or double click on the file then the code will be executed and the following window will be opened.

![5](https://c.mql5.com/2/45/5.png)

After that the program will appear on the chart:

![55](https://c.mql5.com/2/45/55.png)

![Envelopes Sideways Sell Strategy - No signal](https://c.mql5.com/2/45/18-_Envelopes_Sideways_Sell_Strategy_-_No_signal.png)

As we can see on the previous chart, the program gives us No Signal as prices move between the upper band and the lower band during sideways.

![Envelopes Sideways Sell Strategy - Sell signal](https://c.mql5.com/2/45/19-_Envelopes_Sideways_Sell_Strategy_-_Sell_signal.png)

As we can see on the previous chart, the program gives us a short Signal as prices touch the upper band during sideways.

![Envelopes Sideways Sell Strategy - Take Profit](https://c.mql5.com/2/45/20-_Envelopes_Sideways_Sell_Strategy_-_Take_Profit.png)

As we can see on the previous chart, the program gives us a take profit Signal as prices touch the lower band during sideways.

### Conclusion

In this article, we considered the concept of trading bands through the Envelopes indicator. We also considered the benefits of trading bands and the ways in which the Envelopes indicator can be used during trends and during sideways movements.

We have also seen how the Envelopes are calculated indicator, while understanding the insides of what we are using is always good. Envelopes is one of the indicators which can give us new insights and ideas on how we can improve our trading results and our decisions.

Furthermore, we have considered a simple strategy, allowing to use the indicator in different market conditions: trending market uptrend and downtrend or sideways. We have also seen here how to design a trading system blueprint to help us to understand and to be able to code these strategies easily.

Also, the article features the code in the MQL5 language, related to the implementation of the Envelopes-based strategies. And I want again to mention that the main objective of this article is to learn how coding can be helpful and useful for us in trading and how it can help us to trade easily and effectively.

I would recommend you to write your own codes and to execute them in order to enhance your understanding of the topic. I hope that you found this article useful for you and your trading.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/10478.zip "Download all attachments in the single ZIP archive")

[Simple\_Envelopes\_EA.mq5](https://www.mql5.com/en/articles/download/10478/simple_envelopes_ea.mq5 "Download Simple_Envelopes_EA.mq5")(1.79 KB)

[Envelopes\_Uptrend\_Strategy.mq5](https://www.mql5.com/en/articles/download/10478/envelopes_uptrend_strategy.mq5 "Download Envelopes_Uptrend_Strategy.mq5")(1.86 KB)

[Envelopes\_Downtrend\_Strategy.mq5](https://www.mql5.com/en/articles/download/10478/envelopes_downtrend_strategy.mq5 "Download Envelopes_Downtrend_Strategy.mq5")(1.86 KB)

[Envelopes\_Sideways\_Buy\_Strategy.mq5](https://www.mql5.com/en/articles/download/10478/envelopes_sideways_buy_strategy.mq5 "Download Envelopes_Sideways_Buy_Strategy.mq5")(1.96 KB)

[Envelopes\_Sideways\_Short\_Strategy.mq5](https://www.mql5.com/en/articles/download/10478/envelopes_sideways_short_strategy.mq5 "Download Envelopes_Sideways_Short_Strategy.mq5")(1.97 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/390056)**
(13)


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
11 Jun 2022 at 14:53

**Alexander Miller [#](https://www.mql5.com/en/forum/390056#comment_28652415):**

Mohamed this is truly great material for learning! Thank you very much for your work and contribution.

Thanks for your comment. I hope that it will help to enhance your trading.

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
11 Jun 2022 at 14:56

**jesseh1970 [#](https://www.mql5.com/en/forum/390056#comment_40104901):**

Please help me for the "comment" on the chart. For the UpTrend strategy, the "comment" of " [take profit](https://www.mql5.com/en/articles/7113 "Article: Scratching Profits to the Last Pip ") or stop loss" is set when the price is lower than LowBandValue.

However, I observe it always indicated even the price is higher than LowBandValue.

Thanks.

Try to revise conditions of strategy of your code.

![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
29 Jan 2023 at 09:03

Uptrend and Downtrend strategies should be better combined into one: just close the current position on the opposite signal. In the same way, you can combine Sideways Long and Sideways Short.

![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
29 Jan 2023 at 19:45

**Ivan Titov [#](https://www.mql5.com/en/forum/390056/page2#comment_44654571):**

Uptrend and Downtrend strategies should be better combined into one: just close the current position on the opposite signal. In the same way, you can combine Sideways Long and Sideways Short.

Yes, you can combine them.

![Junaid Asghar Bhatti](https://c.mql5.com/avatar/2024/5/66573251-bc4a.jpg)

**[Junaid Asghar Bhatti](https://www.mql5.com/en/users/juni1122)**
\|
8 May 2024 at 14:32

hi, thank you very much, you have done a really good job, i just wanted to know how can i add the option of lot size and [take profit levels](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_double "MQL5 documentation:") and stoploss in the code.


![Learn how to design a trading system by RSI](https://c.mql5.com/2/45/why-and-how__4.png)[Learn how to design a trading system by RSI](https://www.mql5.com/en/articles/10528)

In this article, I will share with you one of the most popular and commonly used indicators in the world of trading which is RSI. You will learn how to design a trading system using this indicator.

![Graphics in DoEasy library (Part 93): Preparing functionality for creating composite graphical objects](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__5.png)[Graphics in DoEasy library (Part 93): Preparing functionality for creating composite graphical objects](https://www.mql5.com/en/articles/10331)

In this article, I will start developing the functionality for creating composite graphical objects. The library will support creating composite graphical objects allowing those objects have any hierarchy of connections. I will prepare all the necessary classes for subsequent implementation of such objects.

![The correct way to choose an Expert Advisor from the Market](https://c.mql5.com/2/44/mql5_avatar_adviser_choose.png)[The correct way to choose an Expert Advisor from the Market](https://www.mql5.com/en/articles/10212)

In this article, we will consider some of the essential points you should pay attention to when purchasing an Expert Advisor. We will also look for ways to increase profit, to spend money wisely, and to earn from this spending. Also, after reading the article, you will see that it is possible to earn even using simple and free products.

![Visual evaluation of optimization results](https://c.mql5.com/2/44/visual-estimation.png)[Visual evaluation of optimization results](https://www.mql5.com/en/articles/9922)

In this article, we will consider how to build graphs of all optimization passes and to select the optimal custom criterion. We will also see how to create a desired solution with little MQL5 knowledge, using the articles published on the website and forum comments.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/10478&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069274334163305061)

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