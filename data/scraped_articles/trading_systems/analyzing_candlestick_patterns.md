---
title: Analyzing Candlestick Patterns
url: https://www.mql5.com/en/articles/101
categories: Trading Systems
relevance_score: 2
scraped_at: 2026-01-23T21:31:36.465009
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fgjbxmksjmpplamjjmkibbluyzdfhozv&ssn=1769193093511570916&ssn_dr=0&ssn_sr=0&fv_date=1769193093&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F101&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Analyzing%20Candlestick%20Patterns%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919309336120559&fz_uniq=5071898451806990468&sv=2552)

MetaTrader 5 / Trading systems


![](https://c.mql5.com/2/1/candles__2.png)

"Through Inquiring of the Old We Learn the New"

### Introduction

Plotting of candlestick charts and analysis of candlestick patterns is an amazing line of technical analysis. The advantage of candlesticks is that they represent data in a way that it is possible to see the momentum within the data.

Candlesticks give a vivid mental picture of trading. After reading and a little practice, candlesticks will be part of your analytical arsenal. Japanese candlestick charts can help you penetrate "inside" of financial markets, which is very difficult to do with other graphical methods. They are equally suitable for all markets.

### 1\. Types of Candlesticks

One of the first analysts who started to predict the movement of prices in the future, based on past prices, was the legendary Japanese [Munehisa Homma](https://www.mql5.com/go?link=http://www.forexwk.com/therms/109 "http://www.forexwk.com/therms/109"). Trading principles applied by Homma in trading on the rice market, initiated the technique of Japanese candlesticks, which is now widely used in Japan and abroad.

![Structure of a candlestick](https://c.mql5.com/2/1/candles__4.png)

Figure 1. Structure of a candlestick

Consider the structure of a candlestick (Fig. 1). The rectangle representing the difference between the open and close prices, is called the body of the candlestick. The height of the body represents the range between the open and close prices of the trading period. When the close price is higher than the open price, the candlestick body is white (Fig. 1 a). If the body is black (Fig. 1 b), this means the close price was below the open price.

Candlesticks can have shadows - the upper and lower, the length of shadows depends on the distance between the open/close prices and the minimum/maximum prices.

Candlestick are plotted on the chart one by one, forming various patterns. According to the theory, some patterns can indicate with a certain probability that the trend is changing, or confirm the trend, or show that the market is indecisive.

Long bodies of candlesticks, as a rule, tell about pressure from buyers or sellers (depending on the color of the candlestick). Short bodies mean that the struggle between the bulls and bears was weak.

| Candlesticks | Description |
| --- | --- |
| ![](https://c.mql5.com/2/1/p1__1.jpg) | **"Long Candlesticks".** Link to long candlesticks is widespread in the literature on Japanese candlesticks. <br>The term "long" refers to a candlestick body length, the difference between the open price and the close price. <br>It is better to consider the most recent price movements to determine what is long, and what is not.<br>Five or ten previous days - that's quite an adequate period to arrive at a correct conclusion. |
| ![](https://c.mql5.com/2/1/p2__1.jpg) | **"Short Candlesticks".** The determination of short candlesticks can be based on the same methodology as in the case of long candlesticks, with similar results. <br>In addition, there are a lot of candlesticks, which do not fall into any of these two categories. |
| ![](https://c.mql5.com/2/1/p3__1.jpg) | **"Marubozu"**. In Japanese "Marubozu" means almost bold. <br>In any case, the meaning of the term reflects the fact that the body of the candlestick either does not have up and/or down shadows at all, or they are very small.<br>Black marubozu - a long black body without a shadow on one of the sides. It often becomes a part of the bearish continuation pattern or bull reversal pattern, especially it appears in a downtrend. A long black candlestick indicates a major victory of bears, so it often appears the first day of many reversal patterns of the bullish character. <br>White marubozu - a long white body without a shadow on one of the sides. This is a very strong candle. In contrast to the black marubozu it often turns out to be part of the bullish pattern of continuation or a bearish reversal pattern. |
| ![](https://c.mql5.com/2/1/p4__1.jpg) | **"Doji"**. If the body of the candlestick is so small that the open and close prices are the same, it is called Doji.<br>The requirement that the open and close prices should be exactly equal, imposes ti strict restrictions on the data, and Doji would appear very rarely.<br>If the price difference between the open and close prices does not exceed a few ticks (minimum price change), this is more than enough. |
| ![](https://c.mql5.com/2/1/p5__1.jpg) | **"Spinning Tops"** are short candlesticks with an upper and/or lower shadow longer than the body. <br>Sometimes they are referred to as "white" and "black" Doji. Koma indicates indecision of bulls and bears.<br>The color of the Koma body, as well as the length of its shadow, is not important. The small body relative to the shadows is what makes the spinning top. |
| ![](https://c.mql5.com/2/1/p6__1.jpg) | **"Hanging man" and "Hammer"**. These are candlesticks with long lower shadows and short bodies. The bodies are at the top of the range of prices.<br>The surprising property of these candlesticks is that they can be bullish and bearish, depending on the phase of the trends, in which they appear.<br>The appearance of these candles in a downward trend is a signal that its dominance in the market is coming to an end, in this case the candlestick is called "the hammer". <br>If the candlestick appears during an uptrend, it indicates its possible end, and the candlestick has an ominous name - "hanging man". |
| ![](https://c.mql5.com/2/1/p7__1.jpg) | **"A start"** appears each time when the small body appears whenever a small body opens up or down from the previous long body, body color is unimportant.<br>Ideally, the gap must catch also shadows, but it is not entirely necessary. The star indicates some uncertainty prevailing in the market. <br>Stars are included in many candlestick patterns, mostly reversal. |

Table 1. Types of Candlesticks

Separate candlesticks are extremely important for the analysis of combinations of candlesticks. When an analyst uses them separately, and then in combination with other candlesticks, the psychological state of the market is revealed.

### 2\. Identification of the basic types of candlesticks

**2.1. Necessary structures**

Candlestick patterns can be a separate candlestick or consist of a few of them. For the candlestick patterns, there are certain rules of recognition.

_Example:_ **_Evening star (bearish pattern)_**. The trend upwards. The first and third candlesticks are "long". Shadows of the stars are short, the color does not matter. The classical pattern: separation of the star from the Close of the first candlestick, for forex and within the day: Close of the first candlestick and Open of the star are equal. The third candlestick is closed inside the body of the first one.

So first let's learn to recognize the types of candlesticks. For this purpose, we write a function RecognizeCandle, which will recognize the type of candlesticks and return the necessary information.

```
//+------------------------------------------------------------------+
//|   Function of candlestick type recognition                       |
//+------------------------------------------------------------------+
bool RecognizeCandle(string symbol,ENUM_TIMEFRAMES period, datetime time,int aver_period,CANDLE_STRUCTURE &res)
```

where:

- symbol - the name of the symbol

- period – chart period,

- time – open time of the candlestick,

- aver\_period - period of averaging

- res - a structure, in which the result is returned.

Let's define what results we need, based on the rules of recognition of candlestick patterns:

- open, close, high and low;
- open time of the candlestick;
- trend direction;
- bullish or bearish candlestick;
- size of the candlestick body – an absolute value;
- type of candlestick (from Table 1).

Let's create a structure:

```
//+------------------------------------------------------------------+
//| Structure CANDLE_STRUCTURE                                       |
//+------------------------------------------------------------------+
struct CANDLE_STRUCTURE
  {
   double            open,high,low,close; // OHLC
   datetime          time;     //Time
   TYPE_TREND       trend;    //Trend
   bool              bull;     //Bull candlestick
   double            bodysize; //Body size
   TYPE_CANDLESTICK  type;    //Type of candlestick
  };
```

where trend and type are variables of enumeration type:

```
//+------------------------------------------------------------------+
//|   ENUM TYPE CANDLESTICK                                          |
//+------------------------------------------------------------------+
enum TYPE_CANDLESTICK
  {
   CAND_NONE,          //Unrecognized
   CAND_MARIBOZU,       //Marubozu
   CAND_MARIBOZU_LONG, //Marubozu long
   CAND_DOJI,           //Doji
   CAND_SPIN_TOP,       //Spinning Tops
   CAND_HAMMER,         //Hammer
   CAND_INVERT_HAMMER, //Reverse hammer
   CAND_LONG,           //Long
   CAND_SHORT,          //Short
   CAND_STAR            //Star
  };
//+------------------------------------------------------------------+
//|   TYPE_TREND                                                     |
//+------------------------------------------------------------------+
enum TYPE_TREND
  {
   UPPER,   //Upward
   DOWN,    //Downward
   LATERAL  //Lateral
  };
```

Let us consider the RecognizeCandle function.

**2.2. Recognition of the candlestick type**

```
//+------------------------------------------------------------------+
//| Function of recognition of candlestick type                      |
//+------------------------------------------------------------------+
bool RecognizeCandle(string symbol,ENUM_TIMEFRAMES period, datetime time,int aver_period,CANDLE_STRUCTURE &res)
  {
   MqlRates rt[];
//--- Get data of previous candlesticks
   if(CopyRates(symbol,period,time,aver_period+1,rt)<aver_period)
     {
      return(false);
     }
```

First, using the [CopyRates](https://www.mql5.com/en/docs/series/copyrates) function obtain data from previous aver\_period +1 candlesticks. Note the order, in which data is stored in the array we obtain.

If data were received without errors, start filling out our return structure CANDLE\_STRUCTURE with data.

```
   res.open=rt[aver_period].open;
   res.high=rt[aver_period].high;
   res.low=rt[aver_period].low;
   res.close=rt[aver_period].close;
   res.time=rt[aver_period].time;
```

**Defining trend.** What is a trend? If this question had a fairly complete answer, the secrets of the market would have been disclosed. In this article we will use the method for determining the trend using a moving average.

MA=(C1+C2+…+Cn)/N,

where C – close prices, N – number of bars.

L. Morris in his book ["Candlestick Charticng Explained. Timeless techniques for Trading Stocks and Futures"](https://www.mql5.com/go?link=http://www.alpina.ru/book/199/ "http://www.alpina.ru/book/199/") uses to a moving average with a period of ten to identify a short-term trend; if the close price is above average - the trend is up, if lower - it is downward.

That's how it looks like:

```
//--- Define the trend direction
   double aver=0;
   for(int i=0;i<aver_period;i++)
   {
      aver+=rt[i].close;
   }
   aver=aver/aver_period;

   if(aver<res.close) res.trend=UPPER;
   if(aver>res.close) res.trend=DOWN;
   if(aver==res.close) res.trend=LATERAL;
```

Next we define if our candle is bullish or bearish, we calculate the absolute value of the candlestick body, the size of shadows, the average body size of the candlestick during aver\_period and other necessary intermediate data.

```
//--- Define of it bullish or bearish
   res.bull=res.open<res.close;
//--- Get the absolute value of the candlestick body size
   res.bodysize=MathAbs(res.open-res.close);
//--- Get the size of shadows
   double shade_low=res.close-res.low;
   double shade_high=res.high-res.open;
   if(res.bull)
     {
      shade_low=res.open-res.low;
      shade_high=res.high-res.close;
     }
   double HL=res.high-res.low;
//--- Calculate the average body size of previous candlesticks
   double sum=0;
   for(int i=1; i<=aver_period; i++)
      sum=sum+MathAbs(rt[i].open-rt[i].close);
   sum=sum/aver_period;
```

Now let's deal with the identification of the types of candlesticks.

**2.3. Rules of identification of candlestick types**

**"Long"** candlesticks. To define "long" candlesticks, check the value of the current candlestick relative to the average value of aver\_period previous candles.

**(Body) > (average body of the last five days) \*1.3**

```
//--- long
   if(res.bodysize>sum*1.3) res.type=CAND_LONG;
```

**"Short"** candlesticks. To define "short" candlesticks, use the same principle as for the "long" ones, but with a changed condition.

**(Body) > (average body of the last X days) \*0.5**

```
//--- short
   if(res.bodysize<sum*0.5) res.type=CAND_SHORT;
```

**Doji**. Doji occurs when open and close prices are equal. This is a very strict rule. In the case of most types of data, we can tolerate some deviations in finding patterns. The formula allows finding the percentage difference between the two prices within acceptable limits.

**(Dodji body) < (range from the highest to the lowest prices) \* 0.03**

```
//--- doji
   if(res.bodysize<HL*0.03) res.type=CAND_DOJI;
```

**"Marubozu"**. This is a candlestick with no high or low, or they are very small

**(lower shadow) < (body) \* 0.03 or (upper shadow) < (body) \* 0.03**

```
//--- maribozu
   if((shade_low<res.bodysize*0.01 || shade_high<res.bodysize*0.01) && res.bodysize>0)
     {
      if(res.type==CAND_LONG)
         res.type=CAND_MARIBOZU_LONG;
      else
         res.type=CAND_MARIBOZU;
     }
```

When writing an indicator for this article, it was necessary to separate the "long" "Maribozu", for which I had to add the condition for checking "long" candlesticks.

**"Hammer"** and **"Hanging Man"**.  The body is located in the upper part of the daily range and the lower shadow is much longer than the body. It is also necessary to consider the length of the upper shadow, if there is any. The ratio between the body and the lower shadow is defined as the ratio of the body length to the length of the lower shadow:

**(lower shadow)>(body)\*2 and  (upper shadow)< (body)\*0.1**

```
//--- hammer
   if(shade_low>res.bodysize*2 && shade_high<res.bodysize*0.1) res.type=CAND_HAMMER;
```

**"Shooting Star"** and **"Inverted Hammer"** are similar to the "Hammer", but with the opposite condition:

**(lower shadow)<(body)\*0.1 and (upper shadow)> (body)\*2**

```
//--- invert hammer
   if(shade_low<res.bodysize*0.1 && shade_high>res.bodysize*2) res.type=CAND_INVERT_HAMMER;
```

**"Spinning Tops"**. These are "short" candlesticks" with shadows longer than the body:

**(lower shadow) > (body) and (upper shadow) > (body)**

```
//--- spinning top
   if(res.type==CAND_SHORT && shade_low>res.bodysize && shade_high>res.bodysize) res.type=CAND_SPIN_TOP;
```

The source text of the function and structure descriptions are in the attached file CandlestickType.mqh.

A;so to the article, the Candlestick Type Color.mq5 indicator is attached, which paints candlesticks on the chart depending on their type.

![](https://c.mql5.com/2/1/type__1.gif)

Figure 2. Example of Candlestick Type Color.mq5

So we have created a function that returns all the necessary data.

Now we cane proceed creating an indicator that will recognize candlestick patterns.

### 3\. Candlestick Patterns and Algorithms for Their Identification

A candlestick pattern can be either a single candlestick, or consist of several candlesticks, seldom more than five or six. In Japanese literature, they sometimes refer to the patterns consisting of a larger number of candlesticks. The order in which we will consider the patterns, does not reflect their importance or predictive capabilities.

Candlestick patterns are divided into two types - **_reversal patterns_** and _**continuation patterns**_.

We will consider simple patterns (one candlestick) first, and then complex (several candlesticks). The figure containing the pattern will start with two small vertical lines. These lines simply indicate the direction of the previous trend of the market, and should not be used as a direct indication of the relations between patterns.

The patterns will be presented in a table, in the first line - bull pattern, in the second - the opposite bearish pattern, if there is such.

**3.1. REVERSAL PATTERNS OF CANDLESTICKS**

**3.1.1. Patterns consisting of a single candlestick**

Obtain data for a single candlestick:

```
//--- calculation of candlestick patterns
   for(int i=limit;i<rates_total-1;i++)
     {
      CANDLE_STRUCTURE cand1;
      if(!RecognizeCandle(_Symbol,_Period,time[i],InpPeriodSMA,cand1))
         continue;
/* Checking patterns with one candlestick */
```

and recognize patterns.

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Inverted**<br>**hammer**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Inverted_Hammer.gif) | ![](https://c.mql5.com/2/1/Inverted_Hammer__1.gif) | Downward trend.<br>The upper shadow is not less than 2 and no more than 3 times larger than the body.<br>There is no lower shadow, or it is very short (no more than 10% of the candlestick range).<br>The color of the body in the long game is not important, with the short - white hammer is much stronger than the black one.<br>Confirmation is suggested. |
| **Hanging Man**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Handing_Man.gif) | ![](https://c.mql5.com/2/1/Handing_Man__1.gif) | Uptrend.<br>The lower shadow is not less than 2 and no more than 3 times larger than the body.<br>There is no upper shadow, or it is very short (no more than 10% of the candlestick range).<br>The color of the body in the long game is not important, with the short - the black hanging man is much stronger than the white one.<br>Confirmation is suggested. |

```
      //------
      // Inverted Hammer the bull model
      if(cand1.trend==DOWN && // check the trend direction
         cand1.type==CAND_INVERT_HAMMER) // check the "inverted hammer"
        {
         comment=_language?"Inverted hammer";
         DrawSignal(prefix+"Inverted Hammer the bull model"+string(objcount++),cand1,InpColorBull,comment);
        }
      // Handing Man the bear model
      if(cand1.trend==UPPER && // check the trend direction
         cand1.type==CAND_HAMMER) // check "hammer"
        {
         comment=_language?"Hanging Man";
         DrawSignal(prefix+"Hanging Man the bear model"+string(objcount++),cand1,InpColorBear,comment);
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Hammer**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Hammer.gif) | ![](https://c.mql5.com/2/1/Hammer__1.gif) | Downward trend.<br>The lower shadow is not less than 2 and no more than 3 times larger than the body.<br>There is no upper shadow, or it is very short (no more than 10% of the candlestick range).<br>The color of the body in the long game is not important, with the short - white hammer is much stronger than the black one.<br>Confirmation is suggested. |
| **Shooting Star**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Shooting_Star.gif) | ![](https://c.mql5.com/2/1/Shooting_Star_FX.gif) | Uptrend.<br>The upper shadow is not less than 3 times is larger than the body.<br>There is no lower shadow, or it is very short (no more than 10% of the candlestick range).<br>The price gap between the star and the previous candlestick.<br> Forex: the Close price of the previous candlestick and Open of the Star are equal.<br>The body color does not matter.<br>Confirmation is suggested. |

```
      //------
      // Hammer the bull model
      if(cand1.trend==DOWN && //check the trend direction
         cand1.type==CAND_HAMMER) // check the hammer
        {
         comment=_language?"Hammer";
         DrawSignal(prefix+"Hammer the bull model"+string(objcount++),cand1,InpColorBull,comment);
        }
      //------
      // Shooting Star the bear model
      if(cand1.trend==UPPER && cand2.trend==UPPER && //check the trend direction
         cand2.type==CAND_INVERT_HAMMER) // check the inverted hammer
        {
         comment=_language?"Shooting Star";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open) // close 1 less equal open 1
              {
               DrawSignal(prefix+"Shooting Star the bear model"+string(objcount++),cand2,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.close<cand2.close) // 2 candlestick detached from 1
              {
               DrawSignal(prefix+"Shooting Star the bear model"+string(objcount++),cand2,InpColorBear,comment);
              }
           }
        }
```

I would like to draw your attention to the fact that in the case of "Shooting Star" we actually need two candlesticks, because under the terms of recognizing the body of the previous day is taken into account.

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Belt Hold**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Belt_Hold_Bull.gif) | The pattern<br>is not<br>implemented | Downward trend.<br>Opening of a candlestick with a large gap in the direction of the trend.<br> White candlestick — «marubozu» «long».<br> The body of the white candlestick is much larger than the body of the previous candlestick.<br>Confirmation is suggested. |
| **Belt Hold**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Belt_Hold_Bear.gif) | The pattern<br>is not<br>implemented | Uptrend.<br>Opening of a candlestick with a large gap in the direction of the trend.<br> Black candlestick — «marubozu» «long».<br> The body of the black candlestick is much larger than the body of the previous candlestick.<br>Confirmation is suggested. |

```
      //------
      // Belt Hold the bull model
      if(cand2.trend==DOWN && cand2.bull && !cand1.bull &&// check the trend direction and direction of the candlestick
         cand2.type==CAND_MARIBOZU_LONG && // check the "long" marubozu
         cand1.bodysize<cand2.bodysize && cand2.close<cand1.close) // the body of the first candlestick is smaller than the body of the second one, close of the second one is below the close of the first
        {
         comment=_language?"Belt Hold";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Belt Hold the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
           }
        }
      // Belt Hold the bear model
      if(cand2.trend==UPPER && !cand2.bull && cand1.bull && // check the trend direction and direction of the candlestick
         cand2.type==CAND_MARIBOZU_LONG && // check the "long" marubozu
         cand1.bodysize<cand2.bodysize && cand2.close>cand1.close) // the body of the first candlestick is smaller than the body of the second one, close of the second one is above the close of the first
        {
         comment=_language?"Belt Hold";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Belt Hold the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
           }
        }
```

Like in the case with the "Shooting Star", two candlesticks are used, because the body of the previous day is taken into account for pattern recognition.

**3.1.2. Patterns consisting of two candlesticks**

Add another candle:

```
/* Checking patterns with two candlesticks */
      CANDLE_STRUCTURE cand2;
      cand2=cand1;
      if(!RecognizeCandle(_Symbol,_Period,time[i-1],InpPeriodSMA,cand1))
         continue;
```

and recognize patterns:

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Engulfing**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Engulfing_Bull.gif) | ![](https://c.mql5.com/2/1/Engulfing_Bull_FX.gif) | Downward trend.<br>The body of the second candlestick completely covers the body of the first one.<br>Forex: Close of the black candlestick and Open of the white one match.<br>Shadows do not matter.<br>Confirmation is suggested. |
| **Engulfing**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Engulfing_Bear.gif) | ![](https://c.mql5.com/2/1/Engulfing_Bear_FX.gif) | Uptrend.<br>The body of the second candlestick completely covers the body of the first one.<br> Forex: Close of the white candlestick and Open of the black one match.<br>Shadows do not matter.<br>Confirmation is suggested. |

```
      //------
      // Engulfing the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.trend==DOWN && cand2.bull && // check the trend direction and direction of the candlestick
         cand1.bodysize<cand2.bodysize) // the body of the second candlestick is larger than that of the first
        {
         comment=_language?"Engulfing";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open && cand1.open<cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Engulfing the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.open<cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Engulfing the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
        }
      // Engulfing the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.trend==UPPER && !cand2.bull && // check the trend direction and direction of the candlestick
         cand1.bodysize<cand2.bodysize) // the body of the second candlestick is larger than that of the first
        {
         comment=_language?"Engulfing";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open && cand1.open>cand2.close) //the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Engulfing the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.open>cand2.close) // close 1 less equal to open 2 or open 1 larger equal to close 2
              {
               DrawSignal(prefix+"Engulfing the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Harami Cross (Bull)** | **Buy** | ![](https://c.mql5.com/2/1/Harami_Cross_Bull.gif) | ![](https://c.mql5.com/2/1/Harami_Cross_Bull_FX.gif) | Downward trend.<br>The first candlestick of the pattern is long black.<br>Doji is within the range of the first candlestick, including the shades.<br>Forex: doji is on the level of Close of the first candlestick. If Doji shadows are short, the pattern should be considered a Doji Star for forex.<br>Confirmation is suggested. |
| **Harami Cross**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Harami_Cross_Bear.gif) | ![](https://c.mql5.com/2/1/Harami_Cross_Bear_FX.gif) | Uptrend.<br>The first candlestick of the pattern is long white.<br>Doji is within the range of the first candlestick, including the shades.<br> Forex: doji is on the level of Close of the first candlestick. If Doji shadows are short, the pattern should be considered a Doji Star for forex.<br>Confirmation is suggested. |

```
      //------
      // Harami Cross the bull model
      if(cand1.trend==DOWN && !cand1.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && cand2.type==CAND_DOJI) // checking the "long" first candlestick and doji
        {
         comment=_language?""Harami Cross";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open && cand1.close<=cand2.close && cand1.open>cand2.close) // doji inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami Cross the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.close<cand2.close && cand1.open>cand2.close) // doji inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami Cross the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
        }
      // Harami Cross the bear model
      if(cand1.trend==UPPER && cand1.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && cand2.type==CAND_DOJI) //  checking the "long" candlestick and doji
        {
         comment=_language?"Harami Cross";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open && cand1.close>=cand2.close && cand1.close>=cand2.close) //  doji inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami Cross the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.close>cand2.close && cand1.open<cand2.close) //  doji inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami Cross the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Harami**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Harami_Bull.gif) | ![](https://c.mql5.com/2/1/Harami_Bull_FX.gif) | Downward trend.<br>The body of the first "long" candlestick" completely engulfs the body of the second one.<br>Shadows do not matter.<br>Forex: Close of the black candlestick and Open of white match.<br>Confirmation is suggested. |
| **Harami**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Harami_Bear.gif) | ![](https://c.mql5.com/2/1/Harami_Bear_FX.gif) | Uptrend.<br> The body of the first "long" candlestick" completely engulfs the body of the second one.<br>Shadows do not matter.<br>Forex: Close of the white candlestick and Open of black match.<br>Confirmation is suggested. |

```
      //------
      // Harami the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.bull &&// check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) &&  // checking the "long"first candlestick
          cand2.type!=CAND_DOJI && cand1.bodysize>cand2.bodysize) // the second candlestick is not doji and the body of the first is larger than the body of the second
        {
         comment=_language?"Harami";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open && cand1.close<=cand2.close && cand1.open>cand2.close) // body of the second candlestick is inside the body of the first
              {
               DrawSignal(prefix+"Harami the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.close<cand2.close && cand1.open>cand2.close) // body of the second candlestick is inside the body of the first
              {
               DrawSignal(prefix+"Harami the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
              }
           }
        }
      // Harami the bear model
      if(cand1.trend==UPPER && cand1.bull && !cand2.bull &&// check the trend direction and direction of candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) &&  // checking the "long" first candlestick
          cand2.type!=CAND_DOJI && cand1.bodysize>cand2.bodysize) // the second candlestick is not doji and body of the first candlestick is larger than that of the second
        {
         comment=_language?"Harami";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open && cand1.close>=cand2.close && cand1.close>=cand2.close) // doji is inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.close>cand2.close && cand1.open<cand2.close) //doji is inside the body of the first candlestick
              {
               DrawSignal(prefix+"Harami the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Doji Star**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Star_doji_Bull.gif) | ![](https://c.mql5.com/2/1/Star_doji_Bull_FX.gif) | Downward trend.<br>The first candlestick of the pattern is long black.<br>The second session - doji with a break in the trend direction.<br>Forex: no break.<br>Shadows of doji are short.<br>Confirmation is suggested. |
| **Doji Star**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Star_doji_Bear.gif) | ![](https://c.mql5.com/2/1/Star_doji_Bear_FX.gif) | Uptrend.<br>The first candlestick of the pattern is long white.<br> The second session - doji with a break in the trend direction.<br>Forex: no break.<br>Shadows of doji are short.<br>Confirmation is suggested. |

```
      //------
      // Doji Star the bull model
      if(cand1.trend==DOWN && !cand1.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && cand2.type==CAND_DOJI) // checking 1 "long" candlestick and 2 doji
        {
         comment=_language?"Doji Star";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open) // open of doji is below or equal to close of the first candlestick
              {
               DrawSignal(prefix+"Doji Star the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);

              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.close>cand2.close) // the body of doji is alienated from the body of the first candlestick
              {
               DrawSignal(prefix+"Doji Star the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);

              }
           }
        }
      // Doji Star the bear model
      if(cand1.trend==UPPER && cand1.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && cand2.type==CAND_DOJI) // checking 1 "long" candlestick and 2 doji
        {
         comment=_language?"Doji Star";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open) // // open of doji is above or equal to close of the first candlestick
              {
               DrawSignal(prefix+"Doji Star the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);

              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.close<cand2.close) // // the body of doji is alienated from the body of the first candlestick
              {
               DrawSignal(prefix+"Doji Star the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);

              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Piercing Pattern**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Gleam_in_clouds.gif) | ![](https://c.mql5.com/2/1/Gleam_in_clouds_FX.gif) | Downward trend.<br> Both candlesticks are long.<br> Open of the white candlestick is below Low of the black.<br> Forex: Close of the black candlestick and Open of white are equal.<br>The white candlestick is closed inside the black one and covers more than 50% of the body. (For stock markets: unlike Dark-cloud cover, this requirement has no exceptions.)<br>Confirmation is not required for the classical model, is required for Forex. |
| **Dark Cloud Cover (Bear)** | **Sell** | ![](https://c.mql5.com/2/1/Veil_from_dark_clouds.gif) | ![](https://c.mql5.com/2/1/Veil_from_dark_clouds_FX.gif) | Uptrend.<br> Both candlesticks are long.<br> Open of the black candlestick is above High of the white candlestick.<br> Forex: Close of the white candlestick and Open of black are equal.<br>The black candle closes inside white and covers more than 50% of the body. <br>Confirmation is not required for the classical model, is required for Forex. |

```
            //------
            // Piercing line the bull model
            if(cand1.trend==DOWN  &&  !cand1.bull  &&  cand2.trend==DOWN  &&  cand2.bull && // check the trend direction and direction of the candlestick
               (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
               cand2.close>(cand1.close+cand1.open)/2)// close of the second is above the middle of the first
              {
               comment=_language?"Piercing Line";
               if(_forex)// if forex
                 {
                  if(cand1.close>=cand2.open && cand2.close<=cand1.open)
                    {
                     DrawSignal(prefix+"Gleam in clouds"+string(objcount++),cand1,cand2,InpColorBull,comment);
                    }
                 }
               else
                 {
                  if(cand2.open<cand1.low && cand2.close<=cand1.open) // open of the second candlestick is below LOW of the first,
                    {
                     DrawSignal(prefix+"Piercing Line"+string(objcount++),cand1,cand2,InpColorBull,comment);
                    }
                 }
              }
            // Dark Cloud Cover the bear model
            if(cand1.trend==UPPER  &&  cand1.bull  &&  cand2.trend==UPPER && !cand2.bull && // check the trend direction and direction of the candlestick
               (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
               cand2.close<(cand1.close+cand1.open)/2)// close 2 is below the middle of the body of 1
              {
               comment=_language?"Dark Cloud Cover";
               if(_forex)// if forex
                 {
                  if(cand1.close<=cand2.open && cand2.close>=cand1.open)
                    {
                     DrawSignal(prefix+"Dark Cloud Cover"+string(objcount++),cand1,cand2,InpColorBear,comment);

                    }
                 }
               else
                 {
                  if(cand1.high<cand2.open && cand2.close>=cand1.open)
                    {
                     DrawSignal(prefix+"Dark Cloud Cover"+string(objcount++),cand1,cand2,InpColorBear,comment);

                    }
                 }
              }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Meeting Lines**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Meeting_Lines_Bull.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first candlestick of the pattern is long black.<br> Open of the white candlestick is with a large gap and is below the Low of the black candlestick.<br> Close prices of both candlesticks are the same.<br> The body of the white candlestick is larger than the body of the black candlestick.<br>Confirmation is suggested. |
| **Meeting Lines**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Meeting_Lines_Bear.gif) | The pattern<br>is not<br>implemented | Uptrend.<br>The first candlestick of the pattern is long white.<br> Open of the black candlestick is with a large gap and is above the High of the white candlestick.<br> Close prices of both candlesticks are the same.<br> The body of the black candlestick is larger than the body of the white candlestick.<br>Confirmation is suggested. |

```
      // Meeting Lines the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.trend==DOWN && cand2.bull && // check the trend direction and the candlestick direction
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand1.close==cand2.close && cand1.bodysize<cand2.bodysize && cand1.low>cand2.open) // close prices are equal, size of the first one is less than of the second one, open of the second is below Low of the first
        {
         comment=_language?"Meeting Lines";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Meeting Lines the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
           }
        }
      // Meeting Lines the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.trend==UPPER && !cand2.bull && // check the trend direction and the candlestick direction
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand1.close==cand2.close && cand1.bodysize<cand2.bodysize && cand1.high<cand2.open) // // close prices are equal, size of first is less than of second, open of the second is above High of the first
        {
         comment=_language?"Meeting Lines";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Meeting Lines the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Matching Low**<br>**(bullish)** | ****Buy**** | ![](https://c.mql5.com/2/1/Matching_Low_Bull.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first candlestick of the pattern must not necessarily be long.<br> Open of the second candlestick is inside the body of the first one.<br>Close prices of both prices are the same.<br>there are no lower shadows or they are very short.<br>Confirmation is suggested. |

```
      //------
      // Matching Low the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.trend==DOWN && !cand2.bull && // check the trend direction and the candlestick direction
         cand1.close==cand2.close && cand1.bodysize>cand2.bodysize) // close prices are equal, size of the first one is larger than of the second
        {
         comment=_language?"Matching Low";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Matching Low the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Homing Pigeon<br1(bullish)** Bullish Homing Pigeon | ****Buy**** | ![](https://c.mql5.com/2/1/Homing_Pigeon_Bull.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first candlestick of the pattern must not necessarily be long.<br> Open of the second candlestick is inside the body of the first one.<br>Close prices of both prices are the same.<br>there are no lower shadows or they are very short.<br>Confirmation is suggested. |

```
      //------
      // Homing Pigeon the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.trend==DOWN && !cand2.bull && // check the trend direction and the candlestick direction
        (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) &&  // checking the "long" candlestick
         cand1.close<cand2.close && cand1.open>cand2.open) // body of the second is inside that of the first candlestick
        {
         comment=_language?"Homing Pigeon";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Homing Pigeon the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
           }
        }
```

**3.1.3. Patterns consisting of three candlesticks**

```
/* Checking patterns with three candlesticks */
      CANDLE_STRUCTURE cand3;
      cand3=cand2;
      cand2=cand1;
      if(!RecognizeCandle(_Symbol,_Period,time[i-2],InpPeriodSMA,cand1))
         continue;
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Abandoned Baby**<br>**(bullish)** | ### **Buy** | ![](https://c.mql5.com/2/1/Abandoned_Baby_Bull.gif) | The pattern<br>is not<br>implemented | This is a rare but important reversal pattern. <br>Downward trend.<br>The first candlestick of the pattern is long black.<br>The second candlestick is doji with a gap, and the gap is not only between the candlestick bodies, but also between shadows.<br>The third candlestick is "long" white candlestick with the same gap between the shadows and Close inside the body of the first candlestick. |
| **Abandoned Baby**<br>**(bearish)** | ### Sell | ![](https://c.mql5.com/2/1/Abandoned_Baby_Bear.gif) | The pattern<br>is not<br>implemented | This is a rare but important reversal pattern. <br>Uptrend.<br>The first candlestick of the pattern is long white.<br>The second candlestick is doji with a gap, and the gap is not only between the candlestick bodies, but also between shadows.<br>The third candlestick is "long" black candlestick with the same gap between the shadows and Close inside the body of the first candlestick. |

```
      //------
      // The Abandoned Baby, the bullish model
      if(cand1.trend==DOWN && !cand1.bull && cand3.trend==DOWN && cand3.bull && // check direction of trend and direction of candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // check of "long" candlestick
         cand2.type==CAND_DOJI && // check if the second candlestick is Doji
         cand3.close<cand1.open && cand3.close>cand1.close) // the third one is closed inside of body of the first one
        {
         comment=_language?"Abandoned Baby (Bull)":"Abandoned Baby";
         if(!_forex)// if it's not forex
           {
            if(cand1.low>cand2.high && cand3.low>cand2.high) // gap between candlesticks
              {
               DrawSignal(prefix+"Abandoned Baby the bull model"+string(objcount++),cand1,cand1,cand3,InpColorBull,comment);
              }
           }
        }
      // The Abandoned Baby, the bearish model
      if(cand1.trend==UPPER && cand1.bull && cand3.trend==UPPER && !cand3.bull && // check direction of trend and direction of candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // check of "long" candlestick
         cand2.type==CAND_DOJI && // check if the second candlestick is Doji
         cand3.close>cand1.open && cand3.close<cand1.close) // // the third one is closed inside of body of the second one
        {
         comment=_language?"Abandoned Baby (Bear)":"Abandoned Baby";
         if(!_forex)// if it's not forex
           {
            if(cand1.high<cand2.low && cand3.high<cand2.low) // gap between candlesticks
              {
               DrawSignal(prefix+"Abandoned Baby the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Morning Star**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Morning_star.gif) | ![](https://c.mql5.com/2/1/Morning_star_FX.gif) | Downward trend.<br> The first and the third sessions are "long" candlesticks.<br>Shadows of the stars are short, the color does not matter.<br> Separation of the star from the Close of the first candlestick.<br>Forex: Close of the first candlestick and Open of the star are equal.<br> The third candlestick is closed inside the body of the first one. |
| **Evening Star**<br>**(bearish** **)** | ### Sell | ![](https://c.mql5.com/2/1/Evening_star.gif) | ![](https://c.mql5.com/2/1/Evening_star_FX.gif) | Uptrend.<br> The first and the third sessions are "long" candlesticks.<br>Shadows of the stars are short, the color does not matter.<br> Separation of the star from the Close of the first candlestick.<br>Forex: Close of the first candlestick and Open of the star are equal.<br> The third candlestick is closed inside the body of the first one. |

```
      // Morning star the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand3.trend==DOWN && cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand2.type==CAND_SHORT && // checking "short"
         cand3.close>cand1.close && cand3.close<cand1.open) // the third candlestick is closed inside the body of the first one
        {
         comment=_language?"Morning star";
         if(_forex)// if forex
           {
            if(cand2.open<=cand1.close) // open of the second candlestick is below or equal to close of the first one
              {
               DrawSignal(prefix+"Morning star the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
         else // another market
           {
            if(cand2.open<cand1.close && cand2.close<cand1.close) // separation of the second candlestick from the first one
              {
               DrawSignal(prefix+"Morning star the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Evening star the bear model
      if(cand1.trend==UPPER && cand1.bull && cand3.trend==UPPER && !cand3.bull && //check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand2.type==CAND_SHORT && // checking "short"
         cand3.close<cand1.close && cand3.close>cand1.open) // the third candlestick is closed inside the body of the first one
        {
         comment=_language?"Evening star";
         if(_forex)// if forex
           {
            if(cand2.open>=cand1.close) // open of the second is above or equal to close of the first one
              {
               DrawSignal(prefix+"Evening star the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
         else // another market
           {
            if(cand2.open>cand1.close && cand2.close>cand1.close) //  separation of the second candlestick from the first one
              {
               DrawSignal(prefix+"Evening star the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Morning Doji Star**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Star_doji_the_morning.gif) | ![](https://c.mql5.com/2/1/Star_doji_the_morning_FX.gif) | Downward trend.<br> The first candlestick of the pattern is long black.<br> The second session - doji with a break in the trend direction.<br>Forex: no break.<br>Shadows of doji are short.<br> The third candlestick is closed inside the body of the first one. |
| **Evening Doji Star** **(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Star_doji_the_evening.gif) | ![](https://c.mql5.com/2/1/Star_doji_the_evening_FX.gif) | Uptrend.<br>The first candlestick of the pattern is long white.<br> The second session - doji with a break in the trend direction.<br>Forex: no break.<br>Shadows of doji are short.<br>Confirmation is suggested. |

```
      //------
      // Morning Doji Star the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand3.trend==DOWN && cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand2.type==CAND_DOJI && // checking "doji"
         cand3.close>cand1.close && cand3.close<cand1.open) // third cand. is closed inside body of first
        {
         comment=_language?"Morning Doji Star";
         if(_forex)// if forex
           {
            if(cand2.open<=cand1.close) // open of doji is below or equal to close of the first
              {
               DrawSignal(prefix+"Morning Doji Star the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
         else // another market
           {
            if(cand2.open<cand1.close) // separation of doji from the first
              {
               DrawSignal(prefix+"Morning Doji Star the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Evening Doji Star the bear model
      if(cand1.trend==UPPER && cand1.bull && cand3.trend==UPPER && !cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand2.type==CAND_DOJI && // checking "doji"
         cand3.close<cand1.close && cand3.close>cand1.open) // third cand. is closed inside body of first
        {
         comment=_language?"Evening Doji Star";
         if(_forex)// if forex
           {
            if(cand2.open>=cand1.close) // open of doji is above or equal to close of the first
              {
               DrawSignal(prefix+"Evening Doji Star the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
         else // another market
           {
            if(cand2.open>cand1.close) // separation of doji from the first
               // check close 2 and open 3
              {
               DrawSignal(prefix+"Evening Doji Star the bear model"+string(objcount++),cand1,cand3,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Bearish Upside Gap Two Crows** | **Sell** | ![](https://c.mql5.com/2/1/Upside_Gap_Two_Crows.gif) | The pattern<br>is not<br>implemented | Uptrend.<br>The first candlestick is "long" white.<br>A gap between white and black candlesticks.<br>The third candlestick opens higher than the second and engulfs it.<br>Confirmation is suggested.<br> The meaning of the pattern: if prices could not go up during the 4th session, we should expect prices to fall. |
| **Two Crows**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Two_Crows.gif) | The pattern<br>is not<br>implemented | Uptrend.<br>The first candlestick of the pattern is long white.<br>The gap between white and the first black candlestick.<br>The third candlestick is black and necessarily "long"; opens inside or above the second and closes within or below the white candlestick, covering the gap. <br>Confirmation is suggested.<br>If the second crow (the third candle) engulfs a white candle, the confirmation is not required. |

```
      //------
      // Upside Gap Two Crows the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.trend==UPPER && !cand2.bull && cand3.trend==UPPER && !cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand1.close<cand2.close && cand1.close<cand3.close && // separation of the second and third from the first one
         cand2.open<cand3.open && cand2.close>cand3.close) // the third candlestick engulfs second
        {
         comment=_language?"Upside Gap Two Crows";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Upside Gap Two Crows the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
        }
      //------
      // Two Crows the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.trend==UPPER && !cand2.bull && cand3.trend==UPPER && !cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG|| cand1.type==CAND_MARIBOZU_LONG) &&(cand3.type==CAND_LONG|| cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand1.close<cand2.close && // separation of the second from the first one
         cand3.open>cand2.close && // third one opens higher than close of the second
         cand3.close<cand1.close) // third one closes than close of the first
        {
         comment=_language?"Two Crows";
         if(!_forex)// if not forex
           {
            DrawSignal(prefix+"Two Crows the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three Star in the South**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Three_Star_in_the_South.gif) | ![](https://c.mql5.com/2/1/Three_Star_in_the_South_FX.gif) | Downward trend.<br>The first candlestick is a long black day with a long lower shadow.<br>The second candlestick is shorter than the first, its Low is above the Low of the first candlestick.<br>The third candlestick is a small black marubozu or a star, an internal day in relation to the second session.<br>Confirmation is suggested. |
| **Deliberation**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Deliberation.gif) | ![](https://c.mql5.com/2/1/Deliberation_FX.gif) | Uptrend.<br>Three white candlesticks with higher close prices. The first two candlesticks are long days. <br>Open price of each candlestick is inside the body of the preceding one. <br> Forex: open/close prices of white candlesticks are the same.<br> The third candlestick opens at about the close level of the second candlestick.<br> the third candlestick is a star or a spinning top.<br>Confirmation is suggested. |

```
      //------
      // Three Star in the South the bull model

      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && !cand3.bull && //check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand3.type==CAND_MARIBOZU || cand3.type==CAND_SHORT) && // checking the "long" candlestick and marubozu
         cand1.bodysize>cand2.bodysize && cand1.low<cand2.low && cand3.low>cand2.low && cand3.high<cand2.high)
        {
         comment=_language?"Three Star in the South";
         if(_forex)// if forex
           {
            DrawSignal(prefix+"Three Star in the South the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
         else // another market
           {
            if(cand1.close<cand2.open && cand2.close<cand3.open) // open is inside the previous candlestick
              {
               DrawSignal(prefix+"Three Star in the South the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Deliberation the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.bull && cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         (cand3.type==CAND_SPIN_TOP || cand3.type==CAND_SHORT)) // the third candlestick is a star or spinning top
        {
         comment=_language?"Deliberation";
         if(_forex)// if forex
           {
            DrawSignal(prefix+"Deliberation the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
         else // another market
           {
            if(cand1.close>cand2.open && cand2.close<=cand3.open) // open is inside the previous candlestick
               // check close 2 and open 3
              {
               DrawSignal(prefix+"Deliberation the bear model"+string(objcount++),cand1,cand3,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three White Soldiers**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Three_White_Soldiers.gif) | ![](https://c.mql5.com/2/1/Three_White_Soldiers_FX.gif) | Downward trend.<br> Three long white candlesticks appear one after another, close prices of each of them is higher than that of the previous.<br> Open price of each soldier is inside the body of the previous candlestick.<br> Forex: Close/Open of solders are the same.<br> Upper shadows of the soldiers are short.<br> Confirmation is not required. |
| **Three Black Crows**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Three_Black_Crows.gif) | The pattern<br>is not<br>implemented | Uptrend.<br> Three long black candlesticks appear one after another, close prices of each of them is lower than that of the previous.<br> Open price of each crow is inside the body of the previous candlestick.<br> Forex: corresponds to pattern Identical three crows.<br> The lower shadows of the crows are short.<br> Confirmation is not required. |

```
      //------
      // Three White Soldiers the bull model
      if(cand1.trend==DOWN && cand1.bull && cand2.bull && cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick and marubozu
         (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG)) // checking the "long" candlestick and marubozu
        {
         comment=_language?"Three White Soldiers";
         if(_forex)// if forex
           {
            DrawSignal(prefix+"Three White Soldiers the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
         else // another market
           {
            if(cand1.close>cand2.open && cand2.close>cand3.open) // open inside the previous candlestick
              {
               DrawSignal(prefix+"Three White Soldiers the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Three Black Crows the bear model
      if(cand1.trend==UPPER && !cand1.bull && !cand2.bull && !cand3.bull && //check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick and marubozu
         (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick and marubozu
         cand1.close<cand2.open && cand2.close<cand3.open)
        {
         comment=_language?"Three Black Crows";
         if(!_forex) // non-forex
           {
               DrawSignal(prefix+"Three Black Crows the bear model"+string(objcount++),cand1,cand3,cand3,InpColorBear,comment);
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three Outside Up**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Three_Outside_Up.gif) | ![](https://c.mql5.com/2/1/Three_Outside_Up_FX.gif) | Downward trend.<br> First the pattern of Engulfing (bull) is formed: the body of the second candlestick completely covers the body of the first one.<br> Forex: Close of a black candlestick and Open of the white one are equal.<br>Shadows do not matter.<br> Then, on the third day there is a higher close.<br>Confirmation is not required: the pattern itself is a confirmation to the bull Engulfing |
| **Three Outside Down**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Three_Outside_Down.gif) | ![](https://c.mql5.com/2/1/Three_Outside_Down_FX.gif) | Uptrend.<br> First the pattern of Engulfing (bear) is formed: the body of the second candlestick completely covers the body of the first one.<br> Forex: Close of the white candlestick and Open of the first black one match.<br>Shadows do not matter.<br> Then, on the third day there is a lower close.<br>Confirmation is not required: the pattern itself is a confirmation to the bear Engulfing |

```
      //------
      // Three Outside Up the bull model
      if(cand1.trend==DOWN && !cand1.bull && cand2.trend==DOWN && cand2.bull && cand3.bull && // check the trend direction and direction of the candlestick
         cand2.bodysize>cand1.bodysize && // the body of the second candlestick is larger than that of the first
         cand3.close>cand2.close) // the third day is closed higher than the second
        {
         comment=_language?"Three Outside Up";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open && cand1.open<cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Three Outside Up the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.open<cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Three Outside Up the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Three Outside Down the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.trend==UPPER && !cand2.bull && !cand3.bull && // check the trend direction and direction of the candlestick
         cand2.bodysize>cand1.bodysize && // the body of the second candlestick is larger than that of the first
         cand3.close<cand2.close) // the third day is closed lower than the second
        {
         comment=_language?"Three Outside Down";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open && cand1.open>cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Three Outside Down the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.open>cand2.close) // the body of the first candlestick is inside the body of the second
              {
               DrawSignal(prefix+"Three Outside Down the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three Inside Up**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Three_Inside_Up.gif) | ![](https://c.mql5.com/2/1/Three_Inside_Up_FX.gif) | Downward trend.<br> In the first two sessions the Harami (bull) pattern is formed: a small white body is engulfed by a large black one.<br> Close in the third session is higher than High of the first two candlesticks.<br>Confirmation is not required: the pattern itself is a confirmation to the bull Harami. |
| **Three Inside Down**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Three_Inside_Down.gif) | ![](https://c.mql5.com/2/1/Three_Inside_Down_FX.gif) | Uptrend.<br> In the first two sessions the Harami (bear) pattern is formed: a small black body is engulfed by a large white one.<br> Close in the third session is lower than Low of the first two candlesticks.<br>Confirmation is not required: the pattern itself is a confirmation to the bear Harami. |

```
      //------
      // Three Inside Up the bull model
      if(cand1.trend==DOWN  &&  !cand1.bull  &&  cand2.bull && cand3.bull && //check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) &&  //checking the "long" candlestick
         cand1.bodysize>cand2.bodysize && // the body of the first candlestick is larger than that of the second one
         cand3.close>cand2.close) // the third day closes higher than the second
        {
         comment=_language?"Three Inside Up";
         if(_forex)// if forex
           {
            if(cand1.close<=cand2.open && cand1.close<=cand2.close && cand1.open>cand2.close) // the body of the second candlestick is inside the body of the first one
              {
               DrawSignal(prefix+"Three Inside Up the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
         else
           {
            if(cand1.close<cand2.open && cand1.close<cand2.close && cand1.open>cand2.close) // the body of the second candlestick is inside the body of the first one
              {
               DrawSignal(prefix+"Three Inside Up the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Three Inside Down the bear model
      if(cand1.trend==UPPER && cand1.bull && !cand2.bull && !cand3.bull &&//check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick
         cand1.bodysize>cand2.bodysize && // the body of the first candlestick is larger than that of the second one
         cand3.close<cand2.close) // the third day closes lower than the second
        {
         comment=_language?"Three Inside Down";
         if(_forex)// if forex
           {
            if(cand1.close>=cand2.open && cand1.close>=cand2.close && cand1.close>=cand2.close) // inside the body of the first candlestick
              {
               DrawSignal(prefix+"Three Inside Down the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
         else
           {
            if(cand1.close>cand2.open && cand1.close>cand2.close && cand1.open<cand2.close) // inside the body of the first candlestick
              {
               DrawSignal(prefix+"Three Inside Down the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three Stars**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Tri_Star_Bull.gif) | ![](https://c.mql5.com/2/1/Tri_Star_Bull_FX.gif) | Downward trend.<br>The gap between the first doji and previous candlesticks is not required.<br> All the three candlesticks are doji.<br> The middle doji has a gap up or down.<br> Forex: all the three doji are on one level.<br>Confirmation is suggested. |
| **Three Stars**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Tri_Star_Bear.gif) | ![](https://c.mql5.com/2/1/Tri_Star_Bear_FX.gif) | Uptrend.<br>The gap between the first doji and previous candlesticks is not required.<br> All the three candlesticks are doji.<br> The middle doji has a gap up or down.<br> Forex: all the three doji are on one level.<br>Confirmation is suggested. |

```
      //------
      // Three Stars the bull model
      if(cand1.trend==DOWN  && // check the trend direction
         cand1.type==CAND_DOJI && cand2.type==CAND_DOJI && cand3.type==CAND_DOJI) // check doji
        {
         comment=_language?"Bullish Three Stars";
         if(_forex)// if forex
           {
               DrawSignal(prefix+"Three Stars the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
         else
           {
            if(cand2.open!=cand1.close && cand2.close!=cand3.open) // the second candlestick is on a different level
              {
               DrawSignal(prefix+"Three Stars the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
              }
           }
        }
      // Three Stars the bear model
      if(cand1.trend==UPPER && // check the trend direction
         cand1.type==CAND_DOJI && cand2.type==CAND_DOJI && cand3.type==CAND_DOJI) // check doji
        {
         comment=_language?"Bearish Three Stars";
         if(_forex)// if forex
           {
               DrawSignal(prefix+"Three Stars the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
         else
           {
            if(cand2.open!=cand1.close && cand2.close!=cand3.open) //  the second candlestick is on a different level
              {
               DrawSignal(prefix+"Three Stars the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Identical Three Crows**<br>**(bearish)** | ****Sell**** | ![](https://c.mql5.com/2/1/Identical_Three_Crows.gif) | ![](https://c.mql5.com/2/1/Identical_Three_Crows__1.gif) | Uptrend.<br> Three long black candlesticks appear one after another, close prices of each of them is lower than that of the previous.<br> The open price of each crow is approximately equal to Close of the preceding candlestick.<br> The lower shadows of the crows are short.<br> Confirmation is not required. |

```
      // Identical Three Crows the bear model
      if(cand1.trend==UPPER && !cand1.bull && !cand2.bull && !cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // checking the "long" candlestick or marubozu
         (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG)) // checking the "long" candlestick or marubozu
        {
         comment=_language?"Identical Three Crows";
         if(_forex)// if forex
           {
            DrawSignal(prefix+"Identical Three Crows the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
         else // a different market
           {
            if(cand1.close>=cand2.open && cand2.close>=cand3.open) // open is less or equal to close of the preceding candlestick
              {
               DrawSignal(prefix+"Identical Three Crows the bear model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
              }
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Unique Three River Bottom**<br>**(bullish)** | ****Buy**** | ![](https://c.mql5.com/2/1/Unique_Three_River_Bottom.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first candlestick of the model is a long black with short shadows.<br> On the second day Harami appears, but with a black body.<br> The lower shadow of the second day gives a new Low.<br> The third day is a short white day lower than the middle day.<br>Confirmation is not necessary but desirable. |

```
      // Unique Three River Bottom the bull model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && cand3.bull && // check the trend direction and the candlestick direction
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && cand3.type==CAND_SHORT && // check the "long" candlestick or "marubozu" third short day
         cand2.open<cand1.open && cand2.close>cand1.close && cand2.low<cand1.low && // body of the second candlestick is inside the first, Low is lower than the first
         cand3.close<cand2.close) // the third candlestick is below the second
        {
         comment=_language?"Unique Three River Bottom";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Unique Three River Bottom the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
        }
```

**3.1.4. Patterns consisting of four candlesticks**

```
/* Checking patterns with four candlesticks */
      CANDLE_STRUCTURE cand4;
      cand4=cand3;
      cand3=cand2;
      cand2=cand1;
      if(!RecognizeCandle(_Symbol,_Period,time[i-3],InpPeriodSMA,cand1))
         continue;
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Concealing Baby Swallow (bull)** | ****Buy**** | ![](https://c.mql5.com/2/1/Concealing_Baby_Swallow.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first two sessions are two black marubozu.<br> The third session opens with a break down, but trade is conducted inside the body of the second candlestick, which leads to the formation of a long upper shadow.<br> The fourth black candlestick completely engulfs the third one, including the shadow.<br> Confirmation is not required. |

```
      //------
      // Concealing Baby Swallow the bull model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && !cand3.bull && !cand4.bull && // check the trend direction and the candlestick direction
         cand1.type==CAND_MARIBOZU_LONG && cand2.type==CAND_MARIBOZU_LONG && cand3.type==CAND_SHORT && // checking "marubozu"
         cand3.open<cand2.close && cand3.high>cand2.close && // third candlestick with a gap down, High is inside the 2nd candlestick
         cand4.open>cand3.high && cand4.close<cand3.low) // the fourth candlestick completely engulfs the third
        {
         comment=_language?"Concealing Baby Swallow";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Concealing Baby Swallow the bull model"+string(objcount++),cand1,cand2,cand4,InpColorBull,comment);
           }
        }
```

**3.1.5. Patterns consisting of two candlesticks**

```
/* Checking patterns with five candlesticks */
      CANDLE_STRUCTURE cand5;
      cand5=cand4;
      cand4=cand3;
      cand3=cand2;
      cand2=cand1;
      if(!RecognizeCandle(_Symbol,_Period,time[i-4],InpPeriodSMA,cand1))
         continue;
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Breakaway**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Breakaway_Bull.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first two sessions — a "long" black candlestick and a "short" black candlestick (star) with a gap.<br> The third session is "short", can be of any color.<br> The fourth candlestick is "Short" black.<br> The fifth one is "long" white with Close inside the gap.<br>Confirmation is suggested. |
| **Breakaway**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Breakaway_Bear.gif) | The pattern<br>is not<br>implemented | Uptrend.<br> The first two sessions — a "long" white candlestick and a "short" white candlestick (star) with a gap.<br> The third session is a "short" candlestick, can be of any color.<br>The fourth day is "short" white.<br> The fifth one is "long" black with Close inside the gap.<br>Confirmation is suggested. |

```
      //------
      // Breakaway the bull model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && !cand4.bull && cand5.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG|| cand1.type==CAND_MARIBOZU_LONG) &&  //checking the "long" first candlestick
         cand2.type==CAND_SHORT && cand2.open<cand1.close && // the second is "short", separated from the first
         cand3.type==CAND_SHORT && cand4.type==CAND_SHORT && // third and fourth are "short"
         (cand5.type==CAND_LONG || cand5.type==CAND_MARIBOZU_LONG) && cand5.close<cand1.close && cand5.close>cand2.open) // fifth is "long" white with close inside the gap
        {
         comment=_language?"Bullish Breakaway";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Breakaway the bull model"+string(objcount++),cand1,cand2,cand5,InpColorBull,comment);
           }
        }
      // Breakaway the bear model
      if(cand1.trend==UPPER && cand1.bull && cand2.bull && cand4.bull && !cand5.bull && //  check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG|| cand1.type==CAND_MARIBOZU_LONG) &&  // checking the "long" first candlestick
         cand2.type==CAND_SHORT && cand2.open<cand1.close && // the second is "short", separated from the first
         cand3.type==CAND_SHORT && cand4.type==CAND_SHORT && // third and fourth are "short"
         (cand5.type==CAND_LONG || cand5.type==CAND_MARIBOZU_LONG) && cand5.close>cand1.close && cand5.close<cand2.open) // fifth is "long" black with close inside the gap
        {
         comment=_language?"Bearish Breakaway";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Breakaway the bear model"+string(objcount++),cand1,cand2,cand5,InpColorBear,comment);
           }
        }
```

**3.2. CONTINUATION PATTERNS**

Continuation patterns are the time when the market is resting.

Whatever the model, you need to make a decision about your current position, even if it is to change nothing.

**3.2.1. Patterns consisting of a single candlestick**

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Kicking**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Kicking_Bull.gif) | The pattern<br>is not<br>implemented | The black "marubozu" is followed by the white "marubozu".<br> There is a gap between the bodies.<br> Confirmation is not required. |
| **Kicking**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Kicking_Bear.gif) | The pattern<br>is not<br>implemented | The white "marubozu" is followed by the black "marubozu".<br> There is a gap between the bodies.<br> Confirmation is not required. |

```
      //------
      // Kicking the bull model
      if(!cand1.bull && cand2.bull && // check the trend direction and direction of the candlestick
         cand1.type==CAND_MARIBOZU_LONG && cand2.type==CAND_MARIBOZU_LONG && // two marubozu
         cand1.open<cand2.open) // a gap between them
        {
         comment=_language?"Bullish Kicking";
         if(!_forex)// if non-forex
           {
            DrawSignal(prefix+"Kicking the bull model"+string(objcount++),cand1,cand2,InpColorBull,comment);
           }
        }
      // Kicking the bear model
      if(cand1.bull && !cand2.bull && // check the trend direction and direction of the candlestick
         cand1.type==CAND_MARIBOZU_LONG && cand2.type==CAND_MARIBOZU_LONG && // two marubozu
         cand1.open>cand2.open) // a gap between them
        {
         comment=_language?"Bearish Kicking";
         if(!_forex)// if non-forex
           {
            DrawSignal(prefix+"Kicking the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
           }
        }
```

**3.2.2. Patterns consisting of two candlesticks**

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Bearish On Neck Line** | **Sell** | ![](https://c.mql5.com/2/1/On_Neck_Line.gif) | The pattern<br>is not<br>implemented | Downward trend.<br>The first candlestick of the pattern is long black.<br> The white candlestick closes below Low of the black one and closes at about Low of the black candlestick.<br> The white candlestick - not necessarily a long day.<br>Confirmation is suggested. |
| **In Neck Line**<br>**(bearish)** | ****Sell**** | ![](https://c.mql5.com/2/1/In_Neck_Line.gif) | The pattern<br>is not<br>implemented | Downward trend.<br>The first candlestick of the pattern is long black.<br> The white candlestick opens below Low of the black one and closes a little higher than Close of the black candlestick.<br> The white candlestick - not necessarily a long day.<br> The body of the white candlestick is smaller than the body of the black candlestick.<br> The upper shadow of the white candlestick is very small.<br>Confirmation is suggested. |
| **Thrusting Line**<br>**(bearish)** | ### **Sell** | ![](https://c.mql5.com/2/1/Thrusting_Line.gif) | The pattern<br>is not<br>implemented | Downward trend.<br>The first candlestick of the pattern is long black.<br> The white candlestick opens below Low of the black one and closes a higher than Close of the black candlestick, but the close price is still below the middle of the black candlestick.<br> The white candlestick - not necessarily a long day.<br>Confirmation is suggested. |

These three models have much in common, so their implementation is somewhat different, please pay attention to it.

```
      //------ Check On Neck patterns
      if(cand1.trend==DOWN && !cand1.bull && cand2.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG)) // the first candlestick is "long"
        {
         // On Neck Line the bear model
         if(cand2.open<cand1.low && cand2.close==cand1.low) // the second candlestick opens below 1st and closes at Low of the first
           {
            comment=_language?"On Neck Line Bear";
            if(!_forex)// if not forex
              {
               DrawSignal(prefix+"On Neck Line the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
              }
           }
         else
           {
            // In Neck Line the bear model
            if(cand1.trend==DOWN && !cand1.bull && cand2.bull && // check the trend direction and direction of the candlestick
               (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && // the first candlestick is "long"
               cand1.bodysize>cand2.bodysize && // body of 2nd candlestick is smaller than that of the 1st one
               cand2.open<cand1.low && cand2.close>=cand1.close && cand2.close<(cand1.close+cand1.bodysize*0.01)) // the second candlestick opens lower than first and closes a little higher than Close of the first
              {
               comment=_language?"In Neck Line Bear";
               if(!_forex)// если не форекс
                 {
                  DrawSignal(prefix+"In Neck Line the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
                 }
              }
            else
              {
               // Thrusting Line the bear model
               if(cand1.trend==DOWN && !cand1.bull && cand2.bull && // check the trend direction and direction of the candlestick
                  (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && // the first candlestick is "long"
                  cand2.open<cand1.low && cand2.close>cand1.close && cand2.close<(cand1.open+cand1.close)/2) //  the second candlestick opens lower than first and closes a higher than Close of the first but lower than middle
                 {
                  comment=_language?"Thrusting Line Bea";
                  if(!_forex)// if non-forex
                    {
                     DrawSignal(prefix+"Thrusting Line the bear model"+string(objcount++),cand1,cand2,InpColorBear,comment);
                    }
                 }
              }
           }
        }
```

**3.2.3. Patterns consisting of three candlesticks**

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Upside Gap Three Methods**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Upside_Gap_Three_Methods.gif) | The pattern<br>is not<br>implemented | Uptrend.<br> The first two candlesticks are two "long" white candlesticks with a gap.<br> The third candlestick opens inside the body of the second and fills in the gap.<br>Confirmation is suggested. |
| **Upside Gap Three Methods**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Downside_Gap_Three_Methods.gif) | The pattern<br>is not<br>implemented | Downward trend.<br> The first two candlesticks are two "long" black candlesticks with a gap.<br> The third candlestick opens inside the body of the second and fills in the gap.<br>Confirmation is suggested. |

```
      //------
      // Upside Gap Three Methods the bull model
      if(cand1.trend==UPPER && cand1.bull && cand2.bull && !cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // the first two candlesticks are "long"
         cand2.open>cand1.close && // a gap between the first and the second
         cand3.open>cand2.open && cand3.open<cand2.close && cand3.close<cand1.close) // the third candlestick opens inside the body of the second and fills in the gap
        {
         comment=_language?"Upside Gap Three Methods";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Upside Gap Three Methods the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
        }
      //------
      // Downside Gap Three Methods the bull model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && cand3.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // the first two candlesticks are "long"
         cand2.open<cand1.close && // a gap between the first and the second
         cand3.open<cand2.open && cand3.open>cand2.close && cand3.close>cand1.close) //the third candlestick opens inside the body of the second and fills in the gap
        {
         comment=_language?"Downside Gap Three Methods";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Downside Gap Three Methods the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
        }
```

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Upside Tasuki Gap**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Upside_Tasuki_Gap.gif)< | The pattern<br>is not<br>implemented | Uptrend.<br>The gap between two neighboring white candles. White candlesticks are not necessarily "long".<br>The third session opens within the body of the second candlestick.<br>The third session closes inside the gap, but the gap is partially unfilled.<br>Confirmation is suggested. |
| **Downside Tasuki Gap**<br>**(bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Downside_Tasuki_Gap.gif) | The pattern<br>is not<br>implemented | Downward trend.<br>The gap between two neighboring black candles. Black candlesticks are not necessarily "long".<br>The third session opens within the body of the second candlestick.<br>The third session closes inside the gap, but the gap is partially unfilled.<br>Confirmation is suggested. |

```
      //------
      // Upside Tasuki Gap the bull model
      if(cand1.trend==UPPER && cand1.bull && cand2.bull && !cand3.bull && // check the trend direction and direction of the candlestick
         cand1.type!=CAND_DOJI && cand2.type!=CAND_DOJI && // the first two candlesticks are not doji
         cand2.open>cand1.close && // a gap between the first and the second
         cand3.open>cand2.open && cand3.open<cand2.close && cand3.close<cand2.open && cand3.close>cand1.close) // 3rd candlestick opens inside 2nd and closes inside the gap
        {
         comment=_language?"Upside Tasuki Gap";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Upside Tasuki Gap the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBull,comment);
           }
        }
      //------
      // Downside Tasuki Gap the bull model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && cand3.bull && // check the trend direction and direction of the candlestick
         cand1.type!=CAND_DOJI && cand2.type!=CAND_DOJI && // the first two candlesticks are not doji
         cand2.open<cand1.close && // a gap between the first and the second
         cand3.open<cand2.open && cand3.open>cand2.close && cand3.close>cand2.open && cand3.close<cand1.close) // 3rd candlestick opens inside 2nd and closes inside the gap
        {
         comment=_language?"Downside Tasuki Gap";
         if(!_forex)// non-forex
           {
            DrawSignal(prefix+"Downside Tasuki Gap the bull model"+string(objcount++),cand1,cand2,cand3,InpColorBear,comment);
           }
        }
```

**3.2.4. Patterns consisting of four candlesticks**

| Name of the Pattern | Order | Classical pattern | Forex | Pattern recognition |
| --- | --- | --- | --- | --- |
| **Three-Line Strike**<br>**(bullish)** | **Buy** | ![](https://c.mql5.com/2/1/Three-line_strike_Bull.gif) | ![](https://c.mql5.com/2/1/Three-line_strike_Bull_FX.gif) | The bullish trend is continuing with three candlesticks, similar to the Three white soldiers pattern: long white candlesticks with short shadows.<br> Forex: Open/close prices of white candlesticks are the same.<br> The fourth candlestick opens with a gap up and closes below open of the first white candlestick.<br> Forex: close of 3rd and open of 4tha candlesticks are equal.<br>Confirmation is suggested. |
| **Three-Line Strike (bearish)** | **Sell** | ![](https://c.mql5.com/2/1/Three-line_strike_Bear.gif) | ![](https://c.mql5.com/2/1/Three-line_strike_Bear_FX.gif) | The bearish trend is continuing with three candlesticks, similar to the Three black crows pattern: long black candlesticks with short shadows.<br> Forex: Open/close prices of black candlesticks are the same (similar to Identical three crows).<br> The fourth candlestick opens with a gap down and closes above open of the first black candlestick.<br> Forex: close of 3rd and open of 4tha candlesticks are equal.<br>Confirmation is suggested. |

```
      //------
      // Three-line strike the bull model
      if(cand1.trend==UPPER && cand1.bull && cand2.bull && cand3.bull && !cand4.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && // check the "long: candlestick or "maruozu"
         (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && //  check the "long: candlestick or "maruozu"
         cand2.close>cand1.close && cand3.close>cand2.close && cand4.close<cand1.open) // close of 2nd is higher than 1st, that of 3rd is higher than 2nd, the fourth candlestick closes lower than the 1st
        {
         comment=_language?"Three-line strike";
         if(_forex)// if forex
           {
            if(cand4.open>=cand3.close) // 4th opens higher than the 3rd
              {
               DrawSignal(prefix+"Three-line strike the bull model"+string(objcount++),cand1,cand3,cand4,InpColorBull,comment);
              }
           }
         else // a different market
           {
            if(cand4.open>cand3.close) // 4th opens higher than th 3rd
              {
               DrawSignal(prefix+"Three-line strike the bull model"+string(objcount++),cand1,cand3,cand4,InpColorBull,comment);
              }
           }
        }
      //------
      // Three-line strike the bear model
      if(cand1.trend==DOWN && !cand1.bull && !cand2.bull && !cand3.bull && cand4.bull && // check the trend direction and direction of the candlestick
         (cand1.type==CAND_LONG || cand1.type==CAND_MARIBOZU_LONG) && (cand2.type==CAND_LONG || cand2.type==CAND_MARIBOZU_LONG) && //  check the "long: candlestick or "maruozu"
         (cand3.type==CAND_LONG || cand3.type==CAND_MARIBOZU_LONG) && //  check the "long: candlestick or "maruozu"
         cand2.close<cand1.close && cand3.close<cand2.close && cand4.close>cand1.open) // close of 2nd is lower than 1st, close of 3rd is lower than 2nd, the fourth candlestick closes higher than 1st
        {
         comment=_language?"Three-line strike";
         if(_forex)// if forex
           {
            if(cand4.open<=cand3.close) // the fourth opens lower than or equal to the third
              {
               DrawSignal(prefix+"Three-line strike the bear model"+string(objcount++),cand1,cand3,cand4,InpColorBear,comment);
              }
           }
         else // a different market
           {
            if(cand4.open<cand3.close) // the fourth opens lower than 3rd
              {
               DrawSignal(prefix+"Three-line strike the bear model"+string(objcount++),cand1,cand3,cand4,InpColorBear,comment);
              }
           }
        }
```

### **4\. The implementation of the indicator**

Let's select required input parameters.

**![Input parameters of the Candlestick Patterns indicator](https://c.mql5.com/2/1/Input__1.PNG)**

Figure 3. Input parameters of the Candlestick Patterns indicator

The first parameter is the "Averaging period" - we need it for determining trend direction, defining "long" and "short" candlesticks.

The next parameter is "Enable signal" - responsible for enabling/disabling the signal. The signal notifies that anew pattern has appeared.

![Signals generated by the Candlestick Patterns indicator ](https://c.mql5.com/2/1/Alert__1.PNG)

Figure 4. Signals generated by the Candlestick Patterns indicator

Parameter "Number of bars for calculation" is designed to facilitate the work of the indicator. If its value is less or equal to 0 candlestick patterns are searched through the entire available history, otherwise a set number of bars is used for search, which considerably facilitates the work of the indicator.

I think everything is clear with colors...

"Enable comments" means enable/disable names of candlestick patterns.

"Font size" sets the font size for comments.

The implementation of the indicator is very simple.

Expect the appearance of a new bar:

```
//--- Wait for a new bar
   if(rates_total==prev_calculated)
     {
      return(rates_total);
     }
```

Then we calculate the initial value of the counter to start the cycle.

```
   int limit;
   if(prev_calculated==0)
     {
      if(InpCountBars<=0 || InpCountBars>=rates_total)
         limit=InpPeriodSMA*2;
      else
         limit=rates_total-InpCountBars;
     }
   else
      limit=prev_calculated-1;
```

In the loop we check the combinations of candlesticks and display them as it shown in Fig. 5:

- Combination with one candlestick: an arrow with the pattern name above or below it.
- Combination with two candlesticks: a thin rectangle with the pattern name above or below the first candlestick.
- Combination with three or more candlesticks: a thick rectangle with the pattern name above or below the last candlestick.

![Example of how the Candlestick Patterns indicator works](https://c.mql5.com/2/1/EURUSDDaily__1.png)

Figure 5. Example of how the Candlestick Patterns indicator works

Please note that due to the possibility of [function overloading](https://www.mql5.com/en/docs/basis/function/functionoverload), the output of signs of different patterns is done via the functions with the same name DrawSignal(), but with different numbers of parameters.

```
void DrawSignal(string objname,CANDLE_STRUCTURE &cand,color Col,string comment)
```

```
void DrawSignal(string objname,CANDLE_STRUCTURE &cand1,CANDLE_STRUCTURE &cand2,color Col,string comment)
```

```
void DrawSignal(string objname,CANDLE_STRUCTURE &cand1,CANDLE_STRUCTURE &cand2,CANDLE_STRUCTURE &cand3,color Col,string comment)
```

The full text of the indicator is in the file Candlestick\_Patterns.mq5 attached to this article.

### **Conclusion**

In this article we have reviewed most of the candlestick patterns, methods of detecting them and provided examples of how to implement them in the MQL5 programming language. The attachment to the article contains two indicators and an include file. To use them, place indicators in the \\Indicators folder and include file in \\Include folder, then compile them.

I hope that the analysis of candlestick patterns will help you improve the results of your work.

Looking ahead, I'd like to add that filtered out candlesticks give better results than most of technical indicators, but we will consider this subject in the following article, in which I am going to create a trading system and an Expert Advisor trading by the candlestick patterns.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/101](https://www.mql5.com/ru/articles/101)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/101.zip "Download all attachments in the single ZIP archive")

[candlesticktype.mqh](https://www.mql5.com/en/articles/download/101/candlesticktype.mqh "Download candlesticktype.mqh")(4.32 KB)

[candlestick\_type\_color.mq5](https://www.mql5.com/en/articles/download/101/candlestick_type_color.mq5 "Download candlestick_type_color.mq5")(5.64 KB)

[candlestick\_patterns.mq5](https://www.mql5.com/en/articles/download/101/candlestick_patterns.mq5 "Download candlestick_patterns.mq5")(63.5 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [An Example of a Trading System Based on a Heiken-Ashi Indicator](https://www.mql5.com/en/articles/91)
- [The Price Histogram (Market Profile) and its implementation in MQL5](https://www.mql5.com/en/articles/17)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2187)**
(54)


![Allan Jheyson Ramos Goncalves](https://c.mql5.com/avatar/avatar_na2.png)

**[Allan Jheyson Ramos Goncalves](https://www.mql5.com/en/users/allnef)**
\|
23 Jun 2021 at 08:57

Congratulations!

Thank you!!!

![期货周报](https://c.mql5.com/avatar/2021/6/60CC79D8-A6D9.png)

**[期货周报](https://www.mql5.com/en/users/13451938885)**
\|
3 Jul 2021 at 06:58

It's full of grammatical errors and won't pass at all!


![Edson Carvalho Do Nascimento](https://c.mql5.com/avatar/2013/11/52878CC0-2F4C.jpeg)

**[Edson Carvalho Do Nascimento](https://www.mql5.com/en/users/edsonnascimento)**
\|
28 Apr 2023 at 11:47

Thank you very much, you have greatly enriched my knowledge of [candlestick patterns](https://www.mql5.com/en/articles/4236 "Article: Using OpenCL to Test Candlestick Patterns ").


![Mikhail Ostashov](https://c.mql5.com/avatar/2024/8/66C758C1-EDE1.jpg)

**[Mikhail Ostashov](https://www.mql5.com/en/users/mihail_anato)**
\|
19 Dec 2024 at 18:43

It's a cannon! But just a moment. Is it possible to make the names in Russian?

![Petr Iarlykov](https://c.mql5.com/avatar/2021/5/60B356E5-1DD4.jpg)

**[Petr Iarlykov](https://www.mql5.com/en/users/dem0n71)**
\|
19 Apr 2025 at 06:32

The patterns seem fine, but I would like to be able to choose which ones to display specifically. I have some questions about colouring. Why do you colour bearish candles as longing candles? .... The situation is similar with maribose candles. There are long candles, but there are no short candles, and the colouring is wrong by direction.

Can this be corrected? I would like an implementation for mt4.

![Evaluation of Trade Systems - the Effectiveness of Entering, Exiting and Trades in General](https://c.mql5.com/2/0/Measure_Trade_Efficiency_MQL5.png)[Evaluation of Trade Systems - the Effectiveness of Entering, Exiting and Trades in General](https://www.mql5.com/en/articles/137)

There are a lot of measures that allow determining the effectiveness and profitability of a trade system. However, traders are always ready to put any system to a new crash test. The article tells how the statistics based on measures of effectiveness can be used for the MetaTrader 5 platform. It includes the class for transformation of the interpretation of statistics by deals to the one that doesn't contradict the description given in the "Statistika dlya traderov" ("Statistics for Traders") book by S.V. Bulashev. It also includes an example of custom function for optimization.

![Controlling the Slope of Balance Curve During Work of an Expert Advisor](https://c.mql5.com/2/0/Balance_Angle_Control_MQL5.png)[Controlling the Slope of Balance Curve During Work of an Expert Advisor](https://www.mql5.com/en/articles/145)

Finding rules for a trade system and programming them in an Expert Advisor is a half of the job. Somehow, you need to correct the operation of the Expert Advisor as it accumulates the results of trading. This article describes one of approaches, which allows improving performance of an Expert Advisor through creation of a feedback that measures slope of the balance curve.

![Interview with Berron Parker (ATC 2010)](https://c.mql5.com/2/0/Berron_ava.png)[Interview with Berron Parker (ATC 2010)](https://www.mql5.com/en/articles/530)

During the first week of the Championship Berron's Expert Advisor has been on the top position. He now tells us about his experience of EA development and difficulties of moving to MQL5. Berron says his EA is set up to work in a trend market, but can be weak in other market conditions. However, he is hopeful that his robot will show good results in this competition.

![Protect Yourselves, Developers!](https://c.mql5.com/2/17/846_12.gif)[Protect Yourselves, Developers!](https://www.mql5.com/en/articles/1572)

Protection of intellectual property is still a big problem. This article describes the basic principles of MQL4-programs protection. Using these principles you can ensure that results of your developments are not stolen by a thief, or at least to complicate his "work" so much that he will just refuse to do it.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/101&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071898451806990468)

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