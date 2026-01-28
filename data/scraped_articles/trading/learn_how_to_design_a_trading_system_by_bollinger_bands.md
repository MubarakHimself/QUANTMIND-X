---
title: Learn how to design a trading system by Bollinger Bands
url: https://www.mql5.com/en/articles/3039
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T20:49:04.534276
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/3039&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051706938670044431)

MetaTrader 5 / Trading


### Introduction

In the world of trading, there are many tools and methods that we can use in our favor to achieve out trading goals. These tools can be used according to market condition or according to market status.

One of such methods is trading bands. The main concept of trading bands is to trade between two bands buying and selling to benefit from the determined bands. One of these tools is the Bollinger Bands indicator. This indicator is one of the most popular or one of the most commonly used one in the world of technical analysis and trading.

In this article, I will share with you information about this Bollinger Bands indicator to enhance our understanding about what it is, how we calculate it, and how we can use it in our favor. After that you will be able to use it in your favorite way according to your strategy. Next, I will share with you how we can design a trading system by using this Bollinger Bands indicator in an accurate and easy way. Thus, we will cover the following topics:

1. [Bollinger Bands Definition](https://www.mql5.com/en/articles/3039#definition)
2. [Bollinger Bands Strategies](https://www.mql5.com/en/articles/3039#strategies)
3. [Bollinger Bands Strategies System Designing](https://www.mql5.com/en/articles/3039#design)
4. [Conclusion](https://www.mql5.com/en/articles/3039#conclusion)
5. [References](https://www.mql5.com/en/articles/3039#references)

As we will learn about Bollinger Band, it measures the dispersion of data around its mean. This indicator was created by John Bollinger. It is constructed by two bands surrounding with 20 days moving average to measure the dispersion of data (prices) around its mean (20 days moving average).

It may seem that Bollinger Bands indicator is the same like Envelopes indicator as it also has two bands surrounding the prices. But this is not true because the difference between Bollinger Bands and Envelopes is that the Bollinger Bands indicator is not placed at a fixed percentage beyond the moving average but its calculation allows that Bollinger Bands indicator expands or contracts according to the standard deviation of the moving average. We will learn this and other details in the Bollinger Bands Definition section or topic.

We will also learn how we can use Bollinger Bands through some strategies of Bollinger Bands in a way to be expressive and beneficial for our trading and this is for sure will be in the Bollinger Bands Strategies section.

Then we will move to the most interesting part of this article: how to use these strategies in an algorithmic trading system in an accurate and reliable manner and this is what we will learn at Bollinger Bands Strategies Blueprint and Bollinger Bands Strategies System Designing sections.

Note that:

- All codes through this article will be written by MQL5 and execution will on MetaTrader 5.
- I advice you to write and execute codes by yourself if you want to practice and enhance your learning.
- So, you will need MetaTrader 5 terminal to execute codes and MetaEditor of MQL5 to write codes and the following pictures are for them.

After downloading and installing the MetaTrader 5 on your device and you can download it from the following link: [https://www.metatrader5.com/en/download](https://www.metatrader5.com/en/download "https://www.metatrader5.com/en/download"). After that you will find the MetaTrader 5 window will be the same like the following picture:

![MT5 platform](https://c.mql5.com/2/45/3-_MT5_platform.png)

You can open MetaEditor by press F4 when open the MetaTrader 5 terminal or from Tools tap we can choose MetaQuotes Language Editor. The following pictures show how to open MetaEditor to write your codes from MetaTrader 5:

![Metaeditor opening 1](https://c.mql5.com/2/45/4-_Metaeditor_opening_1.png)

![Metaeditor opening 2](https://c.mql5.com/2/45/5-_Metaeditor_opening_2.png)

The following picture shows the open MetaEditor window:

![Metaeditor window](https://c.mql5.com/2/45/6-_Metaeditor_window.png)

The following picture shows how to create a new file to write your code:

![Metaeditor - New file](https://c.mql5.com/2/45/8-_Metaeditor_-_New_file.png)

1. Open a new file to write an Expert Advisor
2. Open a new file to write a custom indicator
3. Open a new file to write a script

For more info, you can read my previous article through the following link: [https://www.mql5.com/en/articles/10293](https://www.mql5.com/en/articles/10293)

As I like to mention always in different events, programming or coding is an amazing tool which allows us to make our life easy and smooth with accurate actions which are done automatically. So, it is an important objective to put an investment at this kind of field to learn how to use it in a proper way to generate desired goals in different areas of life.

When it comes to trading, I need to imagine how life can be easy and enjoyable when you give your instructions to the computer to do what you want to do and at the time you want then the computer do what you want exactly without any objections and you can do anything else in your life. It is an amazing lifestyle, so, it will be an important objective to give this coding or programming your interest in your suitable way even if you will code by yourself or let someone else to do it for you.

### Disclaimer

All content of this article is made for the purpose of education only not for anything else. So, you will be responsible for any action you take based on the content of this article, as the content of this article does not guarantee any kind of results.

So, let's go through our article sections to learn more about this interesting topic and the indicator.

### Bollinger Bands Definition

The history of the idea of trading bands is long and interesting and is utilized by many methods. Trading bands is mainly based on constructing bands above and below some measure of tendency.

For example, we can use moving averages shifted up and down by a percentage of itself or Envelopes. Also, there are channels which are two parallel lines above and below prices to detect turning points. But all these methods are fixed and do not react according to price movements. In other words, they do not expand or contract with the price movements.

Bollinger Bands can do that because of its specific calculation. We will see the details of the indicator calculation in this part.

The following picture shows an example of the Envelopes indicator and how it appears on prices:

![EURUSDH1 - Envelopes](https://c.mql5.com/2/45/1-_EURUSDH1_-_Envelopes.png)

As you can see in the previous picture, we have two bands that surround the prices: the lower band surrounds the prices from the below and upper the band surrounds them from the above. So, the main idea here that we need to know is that there are a lot of methods which are based on trading bands from the history and the most common were percentage bands.

There are also many developments on the trading bands idea. One of these developments is the Bollinger Bands indicator which is used to trade bands. It differs from the previously mentioned methods in that it can be expanded and contracted as its calculation allows it to do that.

The Bollinger Bands indicator is created by John Bollinger in the early 1980s, he is one of the popular experts in the field of financial market and trading. He is a CFA ( Chartered Financial Analyst) and a CMT ( Chartered Market Technician).

The Bollinger Bands indicator is a popular technical indicators, it measures the volatility and can be expanded and contracted according to the market condition. It can be used in all financial markets like stocks, forex...etc. Now, let’s talk about the construction of Bollinger Bands:

Like what we mention before the concept of trading bands is to start with some measure of central tendency. Then we construct bands above and below this measure. For the Bollinger Bands indicator, the measure of central tendency is the simple moving average and the interval is determined by a measure of volatility of a moving standard deviation.

The Bollinger bands indicator:

- It measures dispersion of data around its mean (Moving Average).
- It a volatility indicator.
- The difference between Bollinger bands and Envelopes or any percentage trading bands method is that they have specific percentages of moving average above and below moving average but Bollinger Bands is a standard deviation of Moving average.
- The bands automatically widen when volatility increases and contract when volatility decreases. Their dynamic nature allows them to be used on different securities with the standard settings.

- Bollinger Bands are a trading tool used to determine entry and exit points for a trade.
- The bands are often used to determine overbought and oversold conditions.
- It can be used to identify M-Tops and W-Bottoms or to determine the strength of the trend.


The below formula shows how Bollinger Bands are calculated:

![BB calculation](https://c.mql5.com/2/45/BB_calculation.png)

Let us now see an example of calculation of Bollinger Bands:

Suppose we have closing prices for 20 trading days as in the following table:

| Day number | Closing Price $ |
| --- | --- |
| 1 | 20 |
| 2 | 30 |
| 3 | 35 |
| 4 | 30 |
| 5 | 40 |
| 6 | 45 |
| 7 | 50 |
| 8 | 55 |
| 9 | 40 |
| 10 | 45 |
| 11 | 50 |
| 12 | 35 |
| 13 | 40 |
| 14 | 50 |
| 15 | 60 |
| 16 | 65 |
| 17 | 70 |
| 18 | 60 |
| 19 | 70 |
| 20 | 75 |

![BB calculation](https://c.mql5.com/2/45/BB_calculation__1.png)

So, Bollinger Bands calculation will be the same like the following:

![ BB calculation example](https://c.mql5.com/2/45/BB_calculation_example__3.png)

![BB calculation example 1](https://c.mql5.com/2/45/BB_calculation_example_1.png)

**Adjusting the Settings:**

Now If some asked about if we can adjust settings of Bollinger Bands in any other way to be adopted with our strategy or trading plan, the answer is yes we can do that especially if these adjustments tested and gave us a good results. And also we have to know that Bollinger recommends to make a small adjustments for the standard deviation multiplier. Also what we have to know that changing the period of the moving average affects the period used to calculate the standard deviation. Bollinger also suggests to increase the multiplier of standard deviation to 2.1 if we will use a 50 period simple moving average and decrease it to 1.9 if we will use a 10 period simple moving average.

Now, in the current world, we do not need to calculate it manually but we have the indicator ready and built-in in MetaTrader 5. The following images show how to add it or attach the indicator to your chart. Select the Insert menu in MetaTrader 5:

![1- BB indicator insert](https://c.mql5.com/2/45/1-_BB_indicator_insert.png)

Then choose Indicators, then Trend, then Bollinger Bands:

![2- BB indicator insert](https://c.mql5.com/2/45/2-_BB_indicator_insert.png)

After you select the Bollinger Bands indicator, the indicator window parameters will appear. It shows the indicator’s default parameters. It has the following adjustable parameters:

> 1\.    Period to select the period of Moving average.
>
> 2\.    Deviations to select deviation parameter of upper and lower bands.
>
> 3\.    Apply to: to select the kind of prices that parameters which will be applied (Close, Open, High, Low, Median Price, Typical Price, and Weighted Close).
>
> 4\.    Style: to select the style of indicator (Color, shape and width of lines of indicator.
>
> 5\.    Shift: to set a value to apply a shift for indicator.
>
> And like what I mention before we can adjust these settings according to the trader inputs and his strategy.

![3- BB parameters](https://c.mql5.com/2/45/3-_BB_parameters__1.png)

Now, we have the Bollinger Bands indicator drawing on the price chart, and it will be shown the same like the following picture:

![EURUSDH1 - BB](https://c.mql5.com/2/45/2-_EURUSDH1_-_BB.png)

As we can find on the previous picture, prices surrounded by Two Bands (Bollinger Bands) lower band is below the moving average (mean) and prices and upper band is above the moving average (mean) and prices.

### Bollinger Bands Strategies

At this we will talk about Bollinger Bands strategies and we will know how to use it in a beneficial way.

We can use Bollinger Bands with different market condition or different market movements, we can use during uptrend, downtrend, and sideways. And with every market condition, there are many strategies that we can use but here we will mention some of them.

- During Uptrend

First, we need to understand how prices will move during uptrend with Moving Average, as we can find that prices move most of the time during the uptrend above its mean (Moving Average).

So, we will find that during the uptrend prices move between Moving Average and Upper Band most of the time. It means that prices make higher lows and higher highs as there is control from buyers.

![Uptrend](https://c.mql5.com/2/45/3-_Uptrend.png)

![Uptrend](https://c.mql5.com/2/45/Uptrend.png)

Here Bollinger bands can help us to identify this uptrend as we can find that prices move between Moving Average and upper Band. The following picture is an example for that: as you can clearly see, the prices in the green zone move during an uptrend and move between Moving average and upper band:

![EURUSDWeekly - BB with uptrend](https://c.mql5.com/2/45/4-_EURUSDWeekly_-_BB_with_uptrend.png)

The strategy that we will mention here is that we need to:

- Buy when prices crossover above Moving Average and the target will be the Upper Band.

  - Prices > MA = Buy, Upper Band = Target

![EURUSDWeekly - BB up strategy](https://c.mql5.com/2/45/5-_EURUSDWeekly_-_BB_up_strategy.png)

- During Downtrend

First, we need to understand how prices will move during downtrend with Moving Average, as we can find that prices move most of the time during the downtrend below its mean (Moving Average).

So, we will find that during the downtrend prices move between Moving Average and Lower Band most of the time. It means that prices make lower highs and lower lows as Sellers control the market and prices move down.

![Downtrend](https://c.mql5.com/2/45/6-_Downtrend.png)

![Downtrend](https://c.mql5.com/2/45/Downtrend.png)

Here Bollinger Bands can help us to identify this downtrend as we can find that prices move between Moving Average and Lower Band. The following picture is an example for that and we can find clearly that prices in the red zone move in a downtrend and move between moving average and lower band:

![EURUSDDaily - BB with downtrend](https://c.mql5.com/2/45/7-_EURUSDDaily_-_BB_with_downtrend.png)

The strategy that we will mention here is that we need to:

- Short when prices crossover below Moving Average and the target will be the Lower Band.

  - Prices < MA = Short, Lower Band = Target

![EURUSDDaily - BB down strategy](https://c.mql5.com/2/45/8-_EURUSDDaily_-_BB_down_strategy.png)

- During Sideways

First, we need to understand how prices move during sideways as we can find that sideways are any movement except uptrend and downtrend.

Sideways show a balance between buyers and sellers as there is no full control on the market from one party of them. The following pictures are examples for some of sideways movements:

![Sideways 1](https://c.mql5.com/2/45/Sideways_1.png)

![Sideways 2](https://c.mql5.com/2/45/Sideways_2.png)

![Sideways 3](https://c.mql5.com/2/45/Sideways_3.png)

![Sideways 4](https://c.mql5.com/2/45/Sideways_4.png)

The following is a real example for sideways as we can find clearly that prices in the green zone move sideways without clear direction to up or down:

![ Sideways](https://c.mql5.com/2/45/Sideways.png)

The previous figures are examples for sideways movement and there are more than what I mentioned but the concept of sideways movement is it is any movement except uptrend and downtrend.

Now, we need to know how sideways movement will act with Moving Average as prices do not respect Moving Average as it will crossover to above and below. And the following figure is an example for that:

![ MA with sideways](https://c.mql5.com/2/45/MA_with_sideways.png)

So, how can prices move with Bollinger Bands indicator? Prices move between lower band and upper band and the following picture is an example for that:

![GBPUSD1H - BB with sideways](https://c.mql5.com/2/45/9-_GBPUSD1H_-_BB_with_sideways.png)

According to what we know that prices move between lower band and upper band, the strategy here during sideways is to buy on the Lower Band and the target will the Upper Band, or short on the Upper Band and the target will be the Lower Band.

- Prices on Lower Band = Buy, Upper Band = Target
- Prices on Upper Band = Sell, Lower Band = Target

![GBPUSD1H - BB with sideways strategy](https://c.mql5.com/2/45/10-_GBPUSD1H_-_BB_with_sideways_strategy.png)

In the previous picture:

- We can buy on the lower band and take profit on the upper band.
- We can sell on the upper band and take profit on the lower band.

### Bollinger Bands Strategies System Designing

In this interesting part, we will learn how to code an algorithmic trading system based on the Bollinger Bands indicator. We will also learn how to create three trading strategies which we mentioned before (Uptrend strategy, Downtrend strategy, and sideways strategy).

You can design your trading system whatever you want, from a simple trading system to an advanced trading system. For example, you can create and design a simple trading system which will generate a simple signal based on a simple concept then you will act according to this signal or you can create and design an advanced trading system which will not only give you a simple signal but gives an advanced signal based on a combination of related concepts and execute it automatically. There are a lot of different approaches. Within this article, we will deal with a simple trading systems - we will create it for educational purposes only to learn the concept. Based on this knowledge, you will be able to develop similar systems or to design your own trading system in your own terms. I hope that you find this article useful for you and it helps you to achieve your trading or investment goals.

First, we will learn how to design a pure Bollinger Bands code which will allow to appear three comments on the chart with Moving Average Value, Upper Band Value, and Lower Band Value.

The following is the code to do that:

```
//+------------------------------------------------------------------+
//|                                       Simple Bollinger Bands.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //create an array for several prices
   double MiddleBandArray[];
   double UpperBandArray[];
   double LowerBandArray[];

   //sort the price array from the cuurent candle downwards
   ArraySetAsSeries(MiddleBandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);

   //define Bollinger Bands
   int BollingerBands = iBands(_Symbol,_Period,20,0,2,PRICE_CLOSE);

   //copy price info into the array
   CopyBuffer(BollingerBands,0,0,3,MiddleBandArray);
   CopyBuffer(BollingerBands,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBands,2,0,3,LowerBandArray);

   //calcualte EA for the cuurent candle
   double MiddleBandValue=MiddleBandArray[0];
   double UpperBandValue=UpperBandArray[0];
   double LowerBandValue=LowerBandArray[0];

   //comments
   Comment("MiddleBandValue: ",MiddleBandValue,"\n",
   "UpperBandValue: ",UpperBandValue,"\n","LowerBandValue: ",LowerBandValue,"\n");

  }
//+------------------------------------------------------------------+
```

And now after writing the code and compiling it, we have to make sure that the code has no errors or warning the same like the following picture:

![1- SBB - no errors](https://c.mql5.com/2/45/1-_SBB_-_no_errors.png)

Then find the file in the navigator window and drag and drop it on the chart or double click on it to execute and attach it to the chart as in the following pictures:

![ 2- SBB - file](https://c.mql5.com/2/45/2-_SBB_-_file.png)

![3- SBB execution window](https://c.mql5.com/2/45/3-_SBB_execuion_window.png)

The following image shows how a chart with our program running on it:

![4- SBB attached to chart](https://c.mql5.com/2/45/4-_SBB_attached_to_chart.png)

Now, we will learn how to design our simple Bollinger Bands strategies.

**Strategy 1:**

- Uptrend Bollinger Band Strategy Blueprint:
- Prices > MA = Buy, Upper Band = Take Target, Or = No Signal

![Uptrend BB strategy blueprint](https://c.mql5.com/2/45/11-_Uptrend_BB_strategy_blueprint.png)

The following is the code to do that:

```
//+------------------------------------------------------------------+
//|                                          Uptrend BB strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //define Ask, Bid
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   //create an array for several prices
   double MiddleBandArray[];
   double UpperBandArray[];
   double LowerBandArray[];

   //sort the price array from the cuurent candle downwards
   ArraySetAsSeries(MiddleBandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);

   //define Bollinger Bands
   int BollingerBands = iBands(_Symbol,_Period,20,0,2,PRICE_CLOSE);


   //copy price info into the array
   CopyBuffer(BollingerBands,0,0,3,MiddleBandArray);
   CopyBuffer(BollingerBands,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBands,2,0,3,LowerBandArray);

   //calcualte EA for the cuurent candle
   double MiddleBandValue=MiddleBandArray[0];
   double UpperBandValue=UpperBandArray[0];
   double LowerBandValue=LowerBandArray[0];

   //giving buy signal when price > MA
   if (
      (Ask>=MiddleBandArray[0])
   && (Ask<UpperBandArray[0])
      )
         {
         Comment("BUY");
         }

   //check if we have a take profit signal
   if (
      (Bid>=UpperBandArray[0])
      )
         {
         Comment("TAKE PROFIT");
         }

      //check if we have no signal
   if (
      (Ask<MiddleBandArray[0])
      )
         {
         Comment("NO SIGNAL");
         }
  }
//+------------------------------------------------------------------+
```

The following screenshots show the signals of this strategy:

![Uptrend BB - buy](https://c.mql5.com/2/45/4-_Uptrend_BB_-_buy.png)

![Uptrend BB - take profit](https://c.mql5.com/2/45/5-_Uptrend_BB_-_take_profit.png)

![Uptrend BB - No signal](https://c.mql5.com/2/45/6-_Uptrend_BB_-_No_signal.png)

Next we need to perform the following steps to:

- Make sure that your code has no errors or warnings
- Find the file
- Execute the code by drag and drop or double click on the file to be attached on the chart

![5- UBB - no errors](https://c.mql5.com/2/45/5-_UBB_-_no_errors.png)

![ 6- UBB file](https://c.mql5.com/2/45/6-_UBB_file.png)

![7- UBB execution window](https://c.mql5.com/2/45/7-_UBB_execuion_window.png)

The following image shows the chart after the execution of the code. Here the program is running on the chart:

![8- UBB attached to chart](https://c.mql5.com/2/45/8-_UBB_attached_to_chart.png)

**Strategy 2:**

- Downtrend Bollinger Band Strategy Blueprint:
- Prices < MA = Short, Lower Band = Target

![Downtrend BB strategy blueprint](https://c.mql5.com/2/45/12-_Downtrend_BB_strategy_blueprint.png)

The following is the code to do that:

```
//+------------------------------------------------------------------+
//|                                        Downtrend BB strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //define Ask, Bid
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   //create an array for several prices
   double MiddleBandArray[];
   double UpperBandArray[];
   double LowerBandArray[];

   //sort the price array from the cuurent candle downwards
   ArraySetAsSeries(MiddleBandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);

   //define Bollinger Bands
   int BollingerBands = iBands(_Symbol,_Period,20,0,2,PRICE_CLOSE);


   //copy price info into the array
   CopyBuffer(BollingerBands,0,0,3,MiddleBandArray);
   CopyBuffer(BollingerBands,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBands,2,0,3,LowerBandArray);

   //calcualte EA for the cuurent candle
   double MiddleBandValue=MiddleBandArray[0];
   double UpperBandValue=UpperBandArray[0];
   double LowerBandValue=LowerBandArray[0];

   //giving sell signal when price < MA
   if (
      (Bid<=MiddleBandArray[0])
   && (Bid>LowerBandArray[0])
      )
         {
         Comment("SELL");
         }

   //check if we have a take profit signal
   if (
      (Ask<=LowerBandArray[0])
      )
         {
         Comment("TAKE PROFIT");
         }

      //check if we have no signal
   if (
      (Bid>MiddleBandArray[0])
      )
         {
         Comment("NO SIGNAL");
         }
  }
//+------------------------------------------------------------------+
```

The below screenshots display the signals which were generated by this strategy:

![Downtrend BB - sell](https://c.mql5.com/2/45/7-_Downtrend_BB_-_sell.png)

![Downtrend BB - take profit](https://c.mql5.com/2/45/8-_Downtrend_BB_-_take_profit.png)

![Downtrend BB - no signal](https://c.mql5.com/2/45/9-_Downtrend_BB_-_no_signal.png)

Next we again perform the steps (shown in the screenshots) to:

- Make sure that your code has no errors or warnings
- Find the file
- Execute the code by drag and drop or double click on the file to be attached on the chart

![ 9- DBB - no errors](https://c.mql5.com/2/45/9-_DBB_-_no_errors.png)

![ 10- DBB file](https://c.mql5.com/2/45/10-_DBB_file.png)

![ 11- DBB execution window](https://c.mql5.com/2/45/11-_DBB_execuion_window.png)

And the following picture after the execution of the code and how it appears on the chart or how the program attached to the chart:

![ 12- DBB attached to chart](https://c.mql5.com/2/45/12-_DBB_attached_to_chart.png)

**Strategy 3:**

- Sideways Bollinger Band Strategy Blueprint:
- Prices on Lower Band = Buy, Upper Band = Target
- Prices on Upper Band = Sell, Lower Band = Target

![13- Sideways BB strategy blueprint](https://c.mql5.com/2/46/13-_Sideways_BB_strategy_blueprint__1.png)

The following is the code to do that:

> A-   Buy signal which is generated when touching lower band,

```
//+------------------------------------------------------------------+
//|                                     Buy sideways BB strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //define Ask, Bid
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   //create an array for several prices
   double MiddleBandArray[];
   double UpperBandArray[];
   double LowerBandArray[];

   //sort the price array from the cuurent candle downwards
   ArraySetAsSeries(MiddleBandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);

   //define Bollinger Bands
   int BollingerBands = iBands(_Symbol,_Period,20,0,2,PRICE_CLOSE);


   //copy price info into the array
   CopyBuffer(BollingerBands,0,0,3,MiddleBandArray);
   CopyBuffer(BollingerBands,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBands,2,0,3,LowerBandArray);

   //calcualte EA for the cuurent candle
   double MiddleBandValue=MiddleBandArray[0];
   double UpperBandValue=UpperBandArray[0];
   double LowerBandValue=LowerBandArray[0];

   //giving buy signal when price > MA
   if (
      (Ask<=LowerBandArray[0])
      )
         {
         Comment("BUY");
         }

   //check if we have a take profit signal
   if (
      (Bid>=UpperBandArray[0])
      )
         {
         Comment("TAKE PROFIT");
         }

  }
//+------------------------------------------------------------------+
```

The signals generated by this strategy:

![Buy signal - sideways BB](https://c.mql5.com/2/45/10-_Buy_signal_-_sideways_BB.png)

![TP of Buy signal - sideways BB](https://c.mql5.com/2/45/11-_TP_of_Buy_signal_-_sideways_BB.png)

Steps (as are shown in the screenshots below) to:

- Make sure that your code has no errors or warning
- Find the file
- Execute the code by drag and drop or double click on the file to be attached on the chart

![13- BSBB - no errors](https://c.mql5.com/2/45/13-_BSBB_-_no_errors.png)

![ 14- BSBB file](https://c.mql5.com/2/45/14-_BSBB_file.png)

![15- BSBB execution window](https://c.mql5.com/2/45/15-_BSBB_execuion_window.png)

And the following picture after the execution of the code and how it appears on the chart or how the program attached to the chart:

![ 16- BSBB attached to chart](https://c.mql5.com/2/45/16-_BSBB_attached_to_chart.png)

> B-   Sell signal which is generated when touching upper band,

The following is the code to do that:

```
//+------------------------------------------------------------------+
//|                                    Sell sideways BB strategy.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   //define Ask, Bid
   double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits);
   double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits);

   //create an array for several prices
   double MiddleBandArray[];
   double UpperBandArray[];
   double LowerBandArray[];

   //sort the price array from the cuurent candle downwards
   ArraySetAsSeries(MiddleBandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);

   //define Bollinger Bands
   int BollingerBands = iBands(_Symbol,_Period,20,0,2,PRICE_CLOSE);


   //copy price info into the array
   CopyBuffer(BollingerBands,0,0,3,MiddleBandArray);
   CopyBuffer(BollingerBands,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBands,2,0,3,LowerBandArray);

   //calcualte EA for the cuurent candle
   double MiddleBandValue=MiddleBandArray[0];
   double UpperBandValue=UpperBandArray[0];
   double LowerBandValue=LowerBandArray[0];

   //giving sell signal when price < MA
   if (
      (Bid>=UpperBandArray[0])
      )
         {
         Comment("SELL");
         }

   //check if we have a take profit signal
   if (
      (Ask<=LowerBandArray[0])
      )
         {
         Comment("TAKE PROFIT");
         }
  }
//+------------------------------------------------------------------+
```

The following are screen shots for signal which is generated by this strategy:

![Sell signal - sideways BB](https://c.mql5.com/2/45/12-_Sell_signal_-_sideways_BB.png)

![TP of Sell signal - sideways BB](https://c.mql5.com/2/45/13-_TP_of_Sell_signal_-_sideways_BB.png)

Steps (as are shown in the screenshots below) to:

- Make sure that your code has no errors or warning
- Find the file
- Execute the code by drag and drop or double click on the file to be attached on the chart

![ 17- SSBB - no errors](https://c.mql5.com/2/45/17-_SSBB_-_no_errors.png)

![ 18- SSBB file](https://c.mql5.com/2/45/18-_SSBB_file.png)

![ 19- SSBB execution window](https://c.mql5.com/2/45/19-_SSBB_execuion_window.png)

And the following picture after the execution of the code and how it appears on the chart or how the program attached to the chart:

![20- SSBB attached to chart](https://c.mql5.com/2/45/20-_SSBB_attached_to_chart.png)

### Conclusion

The concept of Trading Bands is very good as it gives us a clear expectation for the upcoming movement to take an advantage from this movement and benefit from it and there are many methods to trade bands like Bollinger Bands and like what we knew that it is a good tool or methods as it can expand and contract with price movements and we can use it with all market conditions or trends.

And like what I mention before, at this article I tried to share with the concept of Bollinger Bands indicators, how we can calculate it to enhance the understanding of it, and mentioned simple trading strategies by Bollinger Bands indicator then learned how to design simple trading systems for these simple strategies.

Now, you can develop or design your own strategy which will be suitable for your trading style. I hope you find this article useful for you and your trading.

### References

- Bollinger on Bollinger Bands by John Bollinger
- https://www.bollingerbands.com/about-us
- https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends
- https://school.stockcharts.com/doku.php?id=technical\_indicators:bollinger\_bands

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3039.zip "Download all attachments in the single ZIP archive")

[Simple\_Bollinger\_Bands.mq5](https://www.mql5.com/en/articles/download/3039/simple_bollinger_bands.mq5 "Download Simple_Bollinger_Bands.mq5")(1.61 KB)

[Uptrend\_BB\_strategy.mq5](https://www.mql5.com/en/articles/download/3039/uptrend_bb_strategy.mq5 "Download Uptrend_BB_strategy.mq5")(2.12 KB)

[Downtrend\_BB\_strategy.mq5](https://www.mql5.com/en/articles/download/3039/downtrend_bb_strategy.mq5 "Download Downtrend_BB_strategy.mq5")(2.48 KB)

[Buy\_signal\_-\_sideways\_BB\_strategy.mq5](https://www.mql5.com/en/articles/download/3039/buy_signal_-_sideways_bb_strategy.mq5 "Download Buy_signal_-_sideways_BB_strategy.mq5")(1.95 KB)

[Sell\_signal\_-\_sideways\_BB\_strategy.mq5](https://www.mql5.com/en/articles/download/3039/sell_signal_-_sideways_bb_strategy.mq5 "Download Sell_signal_-_sideways_BB_strategy.mq5")(1.95 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/389146)**
(21)


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
19 May 2022 at 10:36

**Rusmin Nuryadin [#](https://www.mql5.com/en/forum/389146/page2#comment_39675246):**

Dear Sir,

Could you please help to also provide the code for auto trade EA for this trading system? So we can have an EA for auto trade/robot. Thank you.

Best Regards,

Rusmin

Thanks for your comment and I will try.

![John Winsome Munar](https://c.mql5.com/avatar/2022/6/62A73E8B-F64C.jpg)

**[John Winsome Munar](https://www.mql5.com/en/users/trozovka)**
\|
21 May 2022 at 20:01

Thanks for this useful info.


![Mohamed Abdelmaaboud](https://c.mql5.com/avatar/2018/5/5AE8D3AC-DEC5.jpg)

**[Mohamed Abdelmaaboud](https://www.mql5.com/en/users/m.aboud)**
\|
21 May 2022 at 23:45

**John Winsome Munar [#](https://www.mql5.com/en/forum/389146/page2#comment_39721021):**

Thanks for this useful info.

Thanks for your comment.

![Ivan Titov](https://c.mql5.com/avatar/2024/9/66d71f0c-3796.png)

**[Ivan Titov](https://www.mql5.com/en/users/goldrat)**
\|
2 Feb 2023 at 06:47

I think there is an issue in your Trend system: if the current price is near the Upper/Lower band for a long time, it will generate too many signals. To reduce their number, you can add some **offset** from the Upper/Lower band to the Middle line:

```
   if (
      (Ask>=MiddleBandArray[0])
   && (Ask<UpperBandArray[0] - 0.002)
      )
         {
         Comment("BUY");
         }
...
   if (
      (Bid<=MiddleBandArray[0])
   && (Bid>LowerBandArray[0] + 0.002)
      )
         {
         Comment("SELL");
         }
```

or you can make it wait for a **rollback** beyond the Middle line and its re-crossing.

![Petr Michalica](https://c.mql5.com/avatar/2022/1/61D0F730-8743.jpg)

**[Petr Michalica](https://www.mql5.com/en/users/petr.michalica)**
\|
16 Jul 2023 at 11:23

You can use only one order per  expert .

Some like:

if(m\_Position.SelectByMagic( Symbol() , [order\_magic](https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_property_integer "MQL5 documentation:")))

![An Analysis of Why Expert Advisors Fail](https://c.mql5.com/2/45/Why-Expert-Advisors-Fail.png)[An Analysis of Why Expert Advisors Fail](https://www.mql5.com/en/articles/3299)

This article presents an analysis of currency data to better understand why expert advisors can have good performance in some regions of time and poor performance in other regions of time.

![Graphics in DoEasy library (Part 91): Standard graphical object events. Object name change history](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__3.png)[Graphics in DoEasy library (Part 91): Standard graphical object events. Object name change history](https://www.mql5.com/en/articles/10184)

In this article, I will refine the basic functionality for providing control over graphical object events from a library-based program. I will start from implementing the functionality for storing the graphical object change history using the "Object name" property as an example.

![Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://www.mql5.com/en/articles/10237)

In the article, I will create the class of the standard graphical object memory allowing the object to save its states when its properties are modified. In turn, this allows retracting to the previous graphical object states.

![Improved candlestick pattern recognition illustrated by the example of Doji](https://c.mql5.com/2/44/doji.png)[Improved candlestick pattern recognition illustrated by the example of Doji](https://www.mql5.com/en/articles/9801)

How to find more candlestick patterns than usual? Behind the simplicity of candlestick patterns, there is also a serious drawback, which can be eliminated by using the significantly increased capabilities of modern trading automation tools.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/3039&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051706938670044431)

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