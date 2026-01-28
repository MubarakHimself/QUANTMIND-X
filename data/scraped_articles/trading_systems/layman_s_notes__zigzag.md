---
title: Layman's Notes: ZigZag…
url: https://www.mql5.com/en/articles/1537
categories: Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:57:00.523215
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1537&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083233789624194939)

MetaTrader 4 / Trading systems


### Introduction

Surely, a fey thought to trade closely to extremums has visited every apprentice trader when he/she saw "enigmatic" polyline for the first time. It's so simple, indeed. Here is the maximum. And there is the minimum. A beautiful picture on the history. And what is in practice? A ray is drawn. It should seem, that is it, the peak! It is time to sell. And now we go down. But hell no! The price is treacherously moving upwards. Haw! It's a trifle, not an indicator. And you throw it out!

Again and again, we come back to the ZigZag indicator after having read some intelligent books about Elliot's wave theory, Fibo levels, Gartley's patterns, etc. An endless circle. An infinite subject to discuss.

The ZigZag indicator is a polyline that consequently joins the peaks and troughs on a price chart. In such a manner, we display the path the price has "followed" with the course of time. This brings up a question now...What are these peaks and troughs?

The article contains the following:

\- the description of ZigZag;

\- the description of a method to find price models "Peak" and "Trough";

\- the definition of potential reverse points;

\- the revelation of the indicator's additional features.

### What Is ZigZag?

I think, only few ones are surprised with the modern representation of the price chart as a bar. A time length is cut from the ticks history. The enter (Open) and exit (Close) prices, the maximal (High) and minimal (Low) values are fixed. But we can say nothing about the intrabar movement, about its direction at the exit... There is not enough information to make a trade decision. Additional tools are necessary for market analysis.

It is another pair of shoes, when the matter is the ZigZag polyline... We know the level and the approximate time of the last movement start, its direction. On the basis of statistical data processing (peaks and troughs), the possibility of forecasting the probable level and probable reverse time appears... There is enough information to make a "proper" trade decision.

It is important to understand that the current quotes are used for ZigZag drawing. It doesn't use any "quackery" and nor it adds anything to what already exists. **The ZigZag is one of the methods of representing a price chart in a more compact form.**

What quotes are usually used for ZigZag drawing? High and Low. That is the answer. But we can draw a polyline using only the open or only the close price of the bar (in my opinion, these are most preferable variants, because we know the event time for sure).

### How Can We Identify Peaks and Troughs?

Let's remember the definition of a fractal, before we answer this question. Let me quote some phrases from [_New Trading Dimensions_ by Bill Williams:](https://www.mql5.com/go?link=https://www.amazon.com/New-Trading-Dimensions-Commodities-Marketplace/dp/0471295418 "http://www.amazon.com/New-Trading-Dimensions-Commodities-Marketplace/dp/0471295418")

_"The fractal pattern is a simple one. The market makes a move in one direction or the other. After a period of time, all the willing buyer, have bought (on an up move) and the market falls back because of a lack of buyers. Then some new incoming information (Chaos) begins to affect the traders. There is an influx of new buying, and the market, finding that place of equal disagreement on value and agreement on price, moves up. If the momentum and the buyers' strength are strong enough to exceed the immediately preceding up fractal, we would place an order to buy one tick over the high of the fractal."_

In practice, it is applicable to ZigZag. We search for the new models of Peaks and Troughs during its composing. How can we find them? And here comes a quote again:

_“The technical definition of a fractal is: a series of a minimum of five consecutive bars where the highest high is preceded by two lower highs and is followed by two lower highs. (The opposite configuration would be applicable to a sell fractal.)”._

_![](https://c.mql5.com/2/16/index_image488_1.png)_

**The technical definition of the Peak price model is: a series of N number of consecutive bars, the ExtDepth of bars and CheckBar bars with lower maximums after it are located before the highest maximum in it. (The opposite configuration corresponds the Trough price model). Obtained price model is compared with the last known model for the observance of the ZigZag's main rule: the consecutive alternation of models.**

**ExtDepth** and **CheckBar** are the names of the parameters used in the **ZigZagTF** indicator.

The above mentioned algorithm is entirely implemented. Two buffers are used to write the extremums.

### Reverse Points

The only parameter **ExtDepth** is used in the standard ZigZag implementation. **CheckBar** =0, i.e. the extremum is to be searched on the current bar. This is the reason for the permanent refreshing and redrawing of the last ray. _“This ability to correct_
_its values by the following price changes makes Zigzag a perfect tool_
_for analyzing price changes that have already happened. Therefore, you_
_should not try to create a trade system basing on the Zigzag. It is_
_more suitable for analyzing historical data than for making prognoses."_ (The citation of standard ZigZag developers). It is hard to disagree.

Let's refer to the fractal again. The models of fractal alternate on the price chart, i.e. the fractal of Buying is changed with the Selling fractal and vice versa. But there are some regions on the chart, where fractals do not alternate and the same fractal model occurs again. Usually, it's repeated 2-3 times.

![](https://c.mql5.com/2/16/test1_1.gif)

ExtDepth = 2, CheckBar = 2.

Let's try to imitate the working of [Fractals](https://www.mql5.com/ru/code/8193 "https://www.mql5.com/ru/code/8193") indicator (a modification of the standard indicator without redrawing of the last fractal) using our **ZigZagTF** indicator. The **ExtDepth** and **CheckBar** parameters are equal to 2. For a new ray of ZigZag to arise the confirmation of the extremum by two bars is necessary. As soon as the ray appeared on the third bar, we will obtain the **potential reverse point**. The renewing of the ray is possible but not so often as in the standard ZigZag indicator.

Now the ZigZag indicator became more usable for the creation of a trade strategy.

### The Indicator Features

The implementation of possibility of using different price properties for ZigZag composition was already mentioned above.

**Price** Parameters:

0 – High and Low;

1 – bar's open price (Open);

2 – bar's minimum (Low);

3 – bar's maximum (High);

4 – bar's close price (Close).

![](https://c.mql5.com/2/16/test2_1.gif)

ZigZag is drawn by the close prices. ExtDepth = 2, CheckBar = 2.

The situation, when the more accurate information is necessary for drawing the ZigZag, may happen. For example, when both extremums are found on a bar, it is so called external bar. How should we correctly draw the polyline, in this case? The possibility of using smaller timeframes is implemented.

The **TimeFrame** parameter: 0 – the current timeframe.

The **Zzcolor** and **Zzwidth** parameters change the color and the width of the ZigZag line, respectively.

It is useful to know the information about the level of extremum possible appearance, especially when a new ray has not been drawn yet, when there is no confirmation signal ( **CheckBar** parameter), and the level of extremum appearance is broken through.

The **Levels** parameter – enables disables the drawing of levels.

The **Lcolor** and **Hcolor** parameters change the color of the lower and the higher level, respectively.

![](https://c.mql5.com/2/16/test_1.gif)

The levels of new extremum appearance

There is a possibility of writing all points of the ZigZag's polyline after it's drawn into a file. The file is at the: client\_terminal\_root\_directory\\experts\\files folder. The writing of found points: bar number, type of the point, price.

147;Down;209.11

141;Up;210.77

133;Down;208.57

131;Up;209.63

128;Down;208.67

121;Up;209.97

117;Down;209.57

112;Up;210.6

109;Down;209.64

106;Up;210.39

103;Down;209.4

The **Record** parameter – enables writing of the reverse points into the file.

And the last function that is frequently used in trading is the connection of Fibo levels. The **Fibo1** and **Fibo2** parameters allow to enable/disable the Fibo levels on the last and the next to the last ray, respectively. We choose the necessary color by the **FiboColor1** and **FiboColor2** parameters.

![](https://c.mql5.com/2/16/test3_1.gif)

### Conclusion

In this article, I have compared the Fractal and the ZigZag indicators. It appeared that the Fractal model was a particular case. We can also remember the Gann's indicators: the small trend, the intermediate trend, and the main trend. They are the particular cases of the ZigZag, as well.

Maybe it is the first time when the ZigZag is defined as the consecutive alternation of the Peak-and-Trough price models, the technical method of finding them on the price chart.

Now the work is over. Worthful community obtained a convenient tool for trading and market analysis. **Luck and profits to everyone!**

Translated from Russian by MetaQuotes Software Corp.

Original article: [http://articles.mql4.com/ru/articles/1537](https://www.mql5.com/ru/articles/1537)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1537](https://www.mql5.com/ru/articles/1537)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1537.zip "Download all attachments in the single ZIP archive")

[ZigZagTF.mq4](https://www.mql5.com/en/articles/download/1537/ZigZagTF.mq4 "Download ZigZagTF.mq4")(19.41 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39480)**
(10)


![blueeagle](https://c.mql5.com/avatar/2014/1/52E8F957-52E2.jpg)

**[blueeagle](https://www.mql5.com/en/users/blueeagle)**
\|
17 Nov 2009 at 10:09

**grandemario:**

Very Interesting, I'm trying to build an ea based on zigzag indicator and I show you the result of backtesting from may 2008 till now. [![](https://c.mql5.com/3/54/strategytesterlzigmario2_small__1.gif)](https://c.mql5.com/3/54/strategytesterlzigmario2__1.gif)

No bad isn't it?

Can you share your EA?

![Ankit Jain](https://c.mql5.com/avatar/avatar_na2.png)

**[Ankit Jain](https://www.mql5.com/en/users/ankit29030)**
\|
1 Nov 2012 at 09:01

does it repaints like the normal zig zag??

![Ankit Jain](https://c.mql5.com/avatar/avatar_na2.png)

**[Ankit Jain](https://www.mql5.com/en/users/ankit29030)**
\|
1 Nov 2012 at 09:13

unable to use it..not adding or showing on graph..


![Daniel Castro](https://c.mql5.com/avatar/2017/4/58F79B51-FCB1.jpg)

**[Daniel Castro](https://www.mql5.com/en/users/danielfcastro)**
\|
31 May 2018 at 21:12

I am trying to read the results of the 4 buffers and then understand what it is, however I am only getting zero in most part of the positions of the output buffers.  What am I doing wrong?

How can I get the most recent High and Low?

```
//+------------------------------------------------------------------+
//|                                                      TesteTF.mq4 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

//indicador
const string indicator_name = "ZigZagTF";
const int ExtDepth = 12;
const int checkbar = 2;
const int timeframe = 0;
const int price = 0;
const color ZZcolor = clrYellow;
const int ZZwidth = 2;
const bool Levels = true;
const bool Record = true;
const bool Lcolor = clrAqua;
const bool Hcolor = clrAqua;
const bool Fibo1 = true;
const bool Fibo2 = true;
const color FiboColor1 = clrAqua;
const color FiboColor2= clrAqua;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(){
//---
   if(IsNewCandle()){
   for(int i=0;i<50;i++){
      printf("LowBuffer["+i+"]: "+GetZigZagLowBuffer(i));
      printf("HighBuffer["+i+"]: "+GetZigZagHighBuffer(i));
      printf("LowLevel["+i+"]: "+GetZigZagLowLevel(i));
      printf("HighLevel["+i+"]: "+GetZigZagHighLevel(i));
   }}
}
bool IsNewCandle(){
   //Somente é inicializado uma vez e somente percebido pela função que inicializa
    static int BarsOnChart = 0;
    if(BarsOnChart == Bars) return false;
    BarsOnChart = Bars;
    return true;
}
//+------------------------------------------------------------------+
double GetZigZagLowBuffer(int i){
   double temp = iCustom (Symbol(),PERIOD_CURRENT,indicator_name,
                              //NEXT LINES WILL BE INDCATOR INPUTS
                              ExtDepth,checkbar,timeframe,price,ZZcolor,ZZwidth,Levels,Record,Lcolor,Hcolor,Fibo1,Fibo2,FiboColor1,FiboColor2,
                              0,//Index Buffer
                              i//Shift used is 1 because signals is at Close of bar
                              );
   return temp;
}

double GetZigZagHighBuffer(int i){
   double temp = iCustom (Symbol(),PERIOD_CURRENT,indicator_name,
                              //NEXT LINES WILL BE INDCATOR INPUTS
                              ExtDepth,checkbar,timeframe,price,ZZcolor,ZZwidth,Levels,Record,Lcolor,Hcolor,Fibo1,Fibo2,FiboColor1,FiboColor2,
                              1,//Index Buffer
                              i//Shift used is 1 because signals is at Close of bar
                              );
   return temp;
}

double GetZigZagLowLevel(int i){
   double temp = iCustom (Symbol(),PERIOD_CURRENT,indicator_name,
                              //NEXT LINES WILL BE INDCATOR INPUTS
                              ExtDepth,checkbar,timeframe,price,ZZcolor,ZZwidth,Levels,Record,Lcolor,Hcolor,Fibo1,Fibo2,FiboColor1,FiboColor2,
                              2,//Index Buffer
                              i//Shift used is 1 because signals is at Close of bar
                              );
   return temp;
}

double GetZigZagHighLevel(int i){
   double temp = iCustom (Symbol(),PERIOD_CURRENT,indicator_name,
                              //NEXT LINES WILL BE INDCATOR INPUTS
                              ExtDepth,checkbar,timeframe,price,ZZcolor,ZZwidth,Levels,Record,Lcolor,Hcolor,Fibo1,Fibo2,FiboColor1,FiboColor2,
                              3,//Index Buffer
                              i//Shift used is 1 because signals is at Close of bar
                              );
   return temp;
}
```

![Tee Yan Sheng](https://c.mql5.com/avatar/avatar_na2.png)

**[Tee Yan Sheng](https://www.mql5.com/en/users/teezai)**
\|
30 May 2021 at 09:39

Remove the following line:

```
 if(MarketInfo(Symbol(),MODE_TRADEALLOWED) != 1){return(1);}
```

should allow the indicator to be attach to offline charts.

![Trend-Hunting](https://c.mql5.com/2/15/574_6.gif)[Trend-Hunting](https://www.mql5.com/en/articles/1515)

The article describes an algorithm of volume increase of a profit trade. Its implementation using MQL4 means is presented in the article.

![The Statistic Analysis of Market Movements and Their Prognoses](https://c.mql5.com/2/16/634_10.jpg)[The Statistic Analysis of Market Movements and Their Prognoses](https://www.mql5.com/en/articles/1536)

The present article contemplates the wide opportunities of the statistic approach to marketing. Unfortunately, beginner traders deliberately fail to apply the really mighty science of statistics. Meanwhile, it is the only thing they use subconsciously while analyzing the market. Besides, statistics can give answers to many questions.

![All about Automated Trading Championship: Registration](https://c.mql5.com/2/16/698_8.gif)[All about Automated Trading Championship: Registration](https://www.mql5.com/en/articles/1548)

This article comprises useful materials that will help you learn more about the procedure of registration for participation in the Automated Trading Championship.

![Market Diagnostics by Pulse](https://c.mql5.com/2/15/589_8.gif)[Market Diagnostics by Pulse](https://www.mql5.com/en/articles/1522)

In the article, an attempt is made to visualize the intensity of specific markets and of their time segments, to detect their regularities and behavior patterns.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1537&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083233789624194939)

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