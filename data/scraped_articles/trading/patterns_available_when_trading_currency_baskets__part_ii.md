---
title: Patterns available when trading currency baskets. Part II
url: https://www.mql5.com/en/articles/2960
categories: Trading, Trading Systems
relevance_score: 1
scraped_at: 2026-01-23T21:35:40.761213
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=dwgikwnfasawberkcapdtocjdayqwwar&ssn=1769193339291863767&ssn_dr=0&ssn_sr=0&fv_date=1769193339&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2960&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Patterns%20available%20when%20trading%20currency%20baskets.%20Part%20II%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919333979045412&fz_uniq=5071951584847409531&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article](https://www.mql5.com/en/articles/2816) about the patterns emerging when trading currency baskets, we focused our attention on the combined indicators based on oscillators. The combined Williams’ Percent Range indicator was used as an example. As a result, we obtained a number of patterns, analyzed their pros and cons and made conclusions about the applicability of each of them in real trading.

However, this is not enough. The combined indicators based on oscillators cannot cover the needs of all traders who want to apply technical analysis to evaluate the currency basket. The trader's toolkit can be reinforced with combined trend-following indicators having their own patterns. Only after studying them, we are able to consider our set of technical tools to be complete.

Let's develop a test indicator to solve the issue. We have already performed this task, therefore we can use the code from the previous article with minimal changes. But first, we should consider some specific features of combined trend-following indicators. We will use the terminology that is already familiar to readers of the previous articles.

### Features of the combined trend-following indicators

Combined trend-following indicators cannot be based on any parent trend-following indicator due to certain **limitations**.

**Limitation #1**. Combined indicators should be placed in a separate window. There is no point in displaying this indicator in the price chart window. Since the averaging principle is used for constructing combined indicators, it is not clear what they display in that case. Besides, the applied measurement units are different from the ones used on the chart. Thus, neither moving averages, nor Bollinger bands, nor other chart indicators cannot be used as a parent indicator.

**Limitation #2**. The combined indicator shows the status of only one currency; therefore, you need two combined indicators to display a current pair status. Since each of them is located in a separate window, we need two additional windows. The reason for such separation is a difference in scale. The combined indicators based on oscillators always change within pre-defined limits. However, this is not the case with trend-following ones. Neither maximum, nor minimum value are known in advance. This means that the previously described approach involving the moving average applied to the readings of the two combined indicators is pointless. Such collaborative calculations are not suitable when combining trend-following indicators.

Limitations of the parent indicator list prevent from using combined trend-following indicators to the fullest extent. For example, according to the preliminary examination, only ADX and StdDev are suitable for us out of the entire list of trend-following indicators in the MetaTrader 5 menu.

But that is not a reason to abandon the task. We will use the provided tools and start with the combined indicator that we know already — basket currency index.

### Basket currency index with the moving average

Let's develop the testIndexMA.mq5 test indicator similar to the one described [here](https://www.mql5.com/en/articles/2660) and add the moving average to it:

//+------------------------------------------------------------------+

//\|                                                 testDistance.mq5 \|

//\|                                   2016 MetaQuotes Software Corp. \|

//\|                                              http://www.mql5.com \|

//+------------------------------------------------------------------+

#property copyright"Copyright 2016, MetaQuotes Software Corp."

#property link"http://www.mql5.com"

#property version"1.00"

#property indicator\_separate\_window

#property indicator\_buffers2

#property indicator\_plots2

inputcolor   clr= clrGreen;

inputcolor   clrMA = clrMagenta;

inputint maperiod  = 10; //Period MA

double ind\[\],ma\[\];

//+------------------------------------------------------------------+

//\| Custom indicator initialization function                         \|

//+------------------------------------------------------------------+

//int h,h1;

intOnInit()

{

//\-\-\- indicator buffers mapping

ArraySetAsSeries(ind,true);

SetIndexBuffer(0,ind);

IndicatorSetString(INDICATOR\_SHORTNAME,"testdistance");

IndicatorSetInteger(INDICATOR\_DIGITS,2);

PlotIndexSetInteger(0,PLOT\_DRAW\_TYPE,DRAW\_LINE);

PlotIndexSetInteger(0,PLOT\_LINE\_STYLE,STYLE\_SOLID);

PlotIndexSetInteger(0,PLOT\_LINE\_WIDTH,2);

PlotIndexSetInteger(0,PLOT\_LINE\_COLOR,clr);

PlotIndexSetString(0,PLOT\_LABEL,"\_tstdistance\_");

ArraySetAsSeries(ma,true);

SetIndexBuffer(1,ma);

PlotIndexSetInteger(1, PLOT\_DRAW\_TYPE, DRAW\_LINE           );

PlotIndexSetInteger(1, PLOT\_LINE\_STYLE, STYLE\_SOLID            );

PlotIndexSetInteger(1, PLOT\_LINE\_WIDTH, 1            );

PlotIndexSetInteger(1, PLOT\_LINE\_COLOR, clrMA            );

PlotIndexSetString (1, PLOT\_LABEL, "\_tstdistance\_MA" );

//---

return(INIT\_SUCCEEDED);

}

string pair\[\]={"EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"};

bool bDirect\[\]={false,false,false,false,true,true,true};

int iCount=7;

double GetValue(int shift)

{

double res=1.0,t;

double dBuf\[1\];

for(int i=0; i<iCount; i++)

      {

       t=CopyClose(pair\[i\],PERIOD\_CURRENT,shift,1,dBuf);

if(!bDirect\[i\]) dBuf\[0\]=1/dBuf\[0\];

       res\*=dBuf\[0\];

      }//end for (int i = 0; i < iCount; i++)

return (NormalizeDouble(MathPow (res, 1/(double)iCount), \_Digits) );

}

//+------------------------------------------------------------------+

//\| Custom indicator iteration function                              \|

//+------------------------------------------------------------------+

intOnCalculate(constint rates\_total,

constint prev\_calculated,

constdatetime &time\[\],

constdouble &open\[\],

constdouble &high\[\],

constdouble &low\[\],

constdouble &close\[\],

constlong &tick\_volume\[\],

constlong &volume\[\],

constint &spread\[\])

{

if(prev\_calculated==0 \|\| rates\_total>prev\_calculated+1)

      {

int rt=rates\_total;

for(int i=1; i<rt; i++)

         {

          ind\[i\]= GetValue(i);

         }

          rt -= maperiod;

for (int i = 1; i< rt; i++)

            {

             ma\[i\] = GetMA(ind, i, maperiod, \_Digits);

            }

      }

else

      {

          ind\[0\]= GetValue(0);

           ma\[0\] = GetMA(ind, 0, maperiod, \_Digits);

      }

//\-\-\- return value of prev\_calculated for next call

return(rates\_total);

}

voidOnDeinit(constint reason)

{

string text;

switch(reason)

      {

caseREASON\_PROGRAM:

          text="Indicator terminated its operation by calling the ExpertRemove() function";break;

caseREASON\_INITFAILED:

          text="This value means that OnInit() handler "+\_\_FILE\_\_+" has returned a nonzero value";break;

caseREASON\_CLOSE:

          text="Terminal has been closed"; break;

caseREASON\_ACCOUNT:

          text="Account was changed";break;

caseREASON\_CHARTCHANGE:

          text="Symbol or timeframe was changed";break;

caseREASON\_CHARTCLOSE:

          text="Chart was closed";break;

caseREASON\_PARAMETERS:

          text="Input-parameter was changed";break;

caseREASON\_RECOMPILE:

          text="Program "+\_\_FILE\_\_+" was recompiled";break;

caseREASON\_REMOVE:

          text="Program "+\_\_FILE\_\_+" was removed from chart";break;

caseREASON\_TEMPLATE:

          text="New template was applied to chart";break;

default:text="Another reason";

      }

PrintFormat("%s",text);

}

//+------------------------------------------------------------------+

double GetMA(constdouble& arr\[\], int index , int period, int digit) {

double m = 0;

for (int j = 0; j < period; j++)  m += arr\[index + j\];

    m /= period;

return (NormalizeDouble(m,digit));

}

Using this set of input data, the indicator plots the USD index with the fast moving average. Modify the lines 49 and 50 the following way:

string pair\[\]={"EURUSD", "EURJPY", "EURCHF", "EURGBP", "EURNZD", "EURCAD", "EURAUD"};

bool bDirect\[\]={true,true,true,true,true,true,true};

Repeat the compilation with testIndexMA2.mq5. As a result, we obtain a similar indicator showing the EUR index. Place it to EURUSD H1:


![](https://c.mql5.com/2/26/EURUSDH1-1.png)

We are not interested in the absolute indicator values yet. Let's count the MA indicator crossing points with potential market entry points. As stated in the previous articles, these points should be fixed at **the candle closure** which is exactly what we do. Mark detected entry points with vertical lines: blue ones standing for buys and red ones — for sells. The positive result is evident. However, the profit is rather small and unstable, thus it makes sense to increase profitability. First, do not forget about the second currency of the pair and add the USD index indicator in a separate subwindow:

![](https://c.mql5.com/2/26/EURUSDH1-2.png)

Mark МА and USD index graph crossings in vertical lines. Let's analyze the result.

- The index graph crossing MA **hints at a possible trend reversal** which is more probable if a similar (although reverse in nature) crossing is detected at the second pair of the currency. For example, in case of EURUSD, if the USD index graph crosses МА upwards, the EUR index should cross MA downwards. The situation signals the strengthening of one currency with simultaneous weakening of the other.

- If crossing points on the both currencies index graphs are in the same direction, **do not enter the market**. In this case, the probability of a flat is high.
- Crossing points **should be clearly visible**. We considered that in the previous article.

Thus the first practical conclusion: **Consider the indices of both currencies when entering the market**. It is recommended to enter the market when one of the currencies becomes weaker, while the second one gets stronger. One of the first signals of that is an index graph crossing MA. However, this signal is not sufficient: first, wait for the second currency moving in the opposite direction.

The delay issue remains open: What is the maximum possible distance between an index graph and MA crossing points for both currencies of a pair? Obviously, the minimum (and perfect) distance is zero. It is difficult to provide a clear answer using the maximum delay. Although, it is clear that a certain distance should be applied. It is dangerous to enter the market if weakening of one currency and strengthening of another is greatly separated in time. In this case, we face a divergence and a trend weakening.

Thus, we considered market entries based on a combined trend-following indicator. In order to evaluate a potential entry point more accurately, let's move on to the indicator absolute values already mentioned above.

### Quick analysis using ZigZag

For our further work, let's use one of the indicators based on ZigZag from [this article](https://www.mql5.com/en/articles/2774#z7) by my respected colleague [Dmitry Fedoseev](https://www.mql5.com/en/users/integer "Integer"). Let's place iUniZigZagPriceSW.mq5 directly on USD index graph:

![](https://c.mql5.com/2/26/EURUSDH1-zz1.png)

Here, ZigZag is shown as a thick blue line. Our objective is to analyze and organize ZigZag segment length. This way we may be able to get the "swing amplitude" of the USD index.

Let's modify the indicator code a bit:

//+------------------------------------------------------------------+

//\|                                                 iUniZigZagSW.mq5 \|

//\|                        Copyright 2016, MetaQuotes Software Corp. \|

//\|                                             https://www.mql5.com \|

//+------------------------------------------------------------------+

#property copyright"Copyright 2016, MetaQuotes Software Corp."

#property link"https://www.mql5.com"

#property version"1.00"

#property indicator\_separate\_window

#property indicator\_buffers6

#property indicator\_plots3

//\-\-\- plot High

#property indicator\_label1  "High"

#property indicator\_type1   DRAW\_LINE

#property indicator\_color1  clrGreen

#property indicator\_style1  STYLE\_SOLID

#property indicator\_width1  1

//\-\-\- plot Low

#property indicator\_label2  "Low"

#property indicator\_type2   DRAW\_LINE

#property indicator\_color2  clrGreen

#property indicator\_style2  STYLE\_SOLID

#property indicator\_width2  1

//\-\-\- plot ZigZag

#property indicator\_label3  "ZigZag"

#property indicator\_type3   DRAW\_SECTION

#property indicator\_color3  clrRed

#property indicator\_style3  STYLE\_SOLID

#property indicator\_width3  1

//\-\-\- plot Direction

#property indicator\_label4  "Direction"

#property indicator\_type4   DRAW\_LINE

#property indicator\_style4  STYLE\_SOLID

#property indicator\_width4  1

//\-\-\- plot LastHighBar

#property indicator\_label5  "LastHighBar"

#property indicator\_type5   DRAW\_LINE

#property indicator\_style5  STYLE\_SOLID

#property indicator\_width5  1

//\-\-\- plot LastLowBar

#property indicator\_label6  "LastLowBar"

#property indicator\_type6   DRAW\_LINE

#property indicator\_style6  STYLE\_SOLID

#property indicator\_width6  1

#include <ZigZag\\CSorceData.mqh>

#include <ZigZag\\CZZDirection.mqh>

#include <ZigZag\\CZZDraw.mqh>

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

enum EDirection

{

    Dir\_NBars=0,

    Dir\_CCI=1

};

//\-\-\- input parameters

input EDirection  DirSelect=Dir\_NBars;

inputint                  CCIPeriod   =  14;

inputENUM\_APPLIED\_PRICE   CCIPrice    =  PRICE\_TYPICAL;

inputint                  ZZPeriod=14;

inputstring               name="index-usd-zz.txt";

CZZDirection\*dir;

CZZDraw\*zz;

//\-\-\- indicator buffers

double         HighBuffer\[\];

double         LowBuffer\[\];

double         ZigZagBuffer\[\];

double         DirectionBuffer\[\];

double         LastHighBarBuffer\[\];

double         LastLowBarBuffer\[\];

//+------------------------------------------------------------------+

//\| Custom indicator initialization function                         \|

//+------------------------------------------------------------------+

int h;

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

intOnInit()

{

switch(DirSelect)

      {

case Dir\_NBars:

          dir=new CNBars(ZZPeriod);

break;

case Dir\_CCI:

          dir=new CCCIDir(CCIPeriod,CCIPrice);

break;

      }

if(!dir.CheckHandle())

      {

Alert("Indicator 2 download error");

return(INIT\_FAILED);

      }

    zz=new CSimpleDraw();

//\-\-\- indicator buffers mapping

SetIndexBuffer(0,HighBuffer,INDICATOR\_DATA);

SetIndexBuffer(1,LowBuffer,INDICATOR\_DATA);

SetIndexBuffer(2,ZigZagBuffer,INDICATOR\_DATA);

SetIndexBuffer(3,DirectionBuffer,INDICATOR\_CALCULATIONS);

SetIndexBuffer(4,LastHighBarBuffer,INDICATOR\_CALCULATIONS);

SetIndexBuffer(5,LastLowBarBuffer,INDICATOR\_CALCULATIONS);

    h=FileOpen(name,FILE\_CSV\|FILE\_WRITE\|FILE\_ANSI,',');

//---

return(INIT\_SUCCEEDED);

}

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

voidOnDeinit(constint reason)

{

if(CheckPointer(dir)==POINTER\_DYNAMIC)

      {

delete(dir);

      }

if(CheckPointer(zz)==POINTER\_DYNAMIC)

      {

delete(zz);

      }

}

//+------------------------------------------------------------------+

//\| Custom indicator iteration function                              \|

//+------------------------------------------------------------------+

int ind=0;

//+------------------------------------------------------------------+

//\|                                                                  \|

//+------------------------------------------------------------------+

intOnCalculate(constint rates\_total,

constint prev\_calculated,

constint begin,

constdouble &price\[\]

                 )

{

int start;

if(prev\_calculated==0)

      {

       start=0;

      }

else

      {

       start=prev\_calculated-1;

      }

for(int i=start;i<rates\_total;i++)

      {

       HighBuffer\[i\]=price\[i\];

       LowBuffer\[i\]=price\[i\];

      }

int rv;

    rv=dir.Calculate(rates\_total,

                     prev\_calculated,

                     HighBuffer,

                     LowBuffer,

                     DirectionBuffer);

if(rv==0)return(0);

    zz.Calculate(rates\_total,

                 prev\_calculated,

                 HighBuffer,

                 LowBuffer,

                 DirectionBuffer,

                 LastHighBarBuffer,

                 LastLowBarBuffer,

                 ZigZagBuffer);

if(ind<= 10) ind++;

if(ind == 10)

      {

double mx=0,mn=1000000;

double lg;

for(int i=0;i<rates\_total;i++)

         {

if(ZigZagBuffer\[i\]==0 \|\| ZigZagBuffer\[i\]==EMPTY\_VALUE) continue;

if(ZigZagBuffer\[i\] > mx) mx = ZigZagBuffer\[i\];

if(ZigZagBuffer\[i\] < mn) mn = ZigZagBuffer\[i\];

         }

       lg=mx-mn;

PrintFormat("Min index: %.05f Max index: %.05f Length: %.05f",mn,mx,lg);

       lg/=100;

double levels\[100\];

int    count\[100\];

ArrayInitialize(count,0);

for(int i=1; i<101; i++) levels\[i-1\]=NormalizeDouble(lg\*i,\_Digits);

       mn=0;

for(int i=0;i<rates\_total;i++)

         {

if(ZigZagBuffer\[i\]==0 \|\| ZigZagBuffer\[i\]==EMPTY\_VALUE) continue;

if(mn==0) mn=ZigZagBuffer\[i\];

else

            {

             lg=MathAbs(mn-ZigZagBuffer\[i\]);

for(int j=0; j<100; j++)

               {

if(lg<levels\[j\])

                  {

                   count\[j\]++;

break;

                  }

               }

             mn=ZigZagBuffer\[i\];

            }

         }

for(int i=0; i<100; i++)

         {

PrintFormat("%d level: %.05f count: %d",i,levels\[i\],count\[i\]);

FileWrite(h,i,levels\[i\],count\[i\]);

         }

FileClose(h);

      }

return(rates\_total);

}

//+------------------------------------------------------------------+

The indicator starts working and defines the maximum possible ZigZag segment size on the tenth tick. Using this size as 100%, we are able to calculate one percent and organize the values of the remaining ZigZag segments. As a result, we have an array containing the number of ZigZag segments from 1% to 100% of the maximum one. The results are shown in the file and on the Libre Office Calc diagram (download them from the ZZdata.zip archive). Let's show the beginning of the file here together with the corresponding diagram section:

![](https://c.mql5.com/2/26/diag1.jpg)

| Number | Segment length | Number of segments |
| --- | --- | --- |
| 0 | 0.01193 | 2975 |
| 1 | 0.02387 | 850 |
| 2 | 0.0358 | 197 |
| 3 | 0.04773 | 54 |
| 4 | 0.05967 | 17 |

Other diagram areas are of little interest for us since they are mostly filled with zeros. We can continue and fine tune this analysis if we decrease the step, but let's stick to the current results for now. We are already able to make the main practical conclusion:

- We should be cautious when entering the market following a trend if the segment size of the ZigZag applied to the combined indicator of the basket currency index exceeds a certain value. Here, the segment size is a length of the ZigZag segment projection to the price (Y) axis.


We should define the size of this "critical" segment. To do this, apply statistical methods to the above data. You may perform such an analysis on your own using your own risk preferences. In my opinion, the "critical" segment should not be equal to 0.03.

In this example, we analyzed the entire available history. However, if we want to capture the most recent market movements, we should use smaller periods (year, quarter or month).

A similar analysis can be performed for all currency baskets and most timeframes. It would be interesting to see how diagrams change for different "currency — timeframe" sets. Developers will immediately notice that the algorithm can be easily implemented in the code. Of course, you should not rely solely on one signal. Look for confirmation. The technique shown above can be applied to many common trend indicators. However, it is meaningless when dealing with oscillators.

Thus, we have made another step towards improving the quality of the market entries by using the combined indicators. Since it is **necessary to look for a signal confirmation**, this is what we are going to do now.

### Joint use of different combined indicators

As you may know, we already have a combined indicator based on WPR. We examined it in details in our [previous article](https://www.mql5.com/en/articles/2816), including its code and available patterns. This time, let's try applying it in conjunction with the combined index indicator. The resulting structure is expected to be quite efficient, since many trading systems are built the same way: trend-following indicator + oscillator.

Let's use the indicators testDistance.mq5, testWPR.mq5 and testWPRjpy.mq5 from the previous article and place them to the chart together with the combined EUR index testDistance.mq5 indicator. In our previous article, we have studied EURJPY, therefore, the testWPRjpy indicator should be re-written for working with USD. We will preserve the indicator's name in order not to alter the testDistance.mq5 indicator. All indicators from this section can be found in the wpr.zip archive.

The combined WPR indicator is to plot the difference between the combined WPRs of the currencies included into our currency pair (all that has been described in details in the original article). Our objective is to detect the previously described patterns on a single indicator when using another one as a filter:


![](https://c.mql5.com/2/26/EURUSDH1-4__1.png)

Potential market entry points (not all) are marked in the image. The entry points, at which the combined WPR (top window) and the combined EUR index (bottom window) show unidirectional patterns, are considered more reliable. Here, these are entries 1 and 6. These are the crossing points of the indicator and the moving average.

The entry 2 is of particular interest, since the combined WPR forms an almost academically accurate pattern crossing the oversold line. Other entries do not provide us with sufficient confirmation. Although, the actual market entry in such points would not cause losses, I would not take such a risk in real trading.

How reasonable is it to apply the combined indicator plotting the difference between the combined WPRs included in the currency basket? Wouldn't it be more correct to use the combined WPR for EUR in tandem with the combined EUR index? Let's try to do this by replacing testDistance.mq5 with testWPR.mq5:

![](https://c.mql5.com/2/26/EURUSDH1-5__1.png)

Here we can see the combined WPR indicator for EUR. Is it justified? In this case, it is. The indicator has corrected our entry by one candle in points 2 and 6 (arrows are used to specify the direction). The entry 1 is not confirmed enough. The entries 5 and 3 are not recommended. The point 4 has been corrected by the indicator into the point 7.

It would seem that the obtained results vote in favor of the combined WPR indicator of the currency basket rather than the indicator of the difference between the combined WPRs for two currencies of the pair. However, this is true only for this particular case. I would recommend applying both kinds of the combined oscillator in everyday usage till significant statistics is accumulated.

You may ask "What is so special about that? The combined usage of several indicators is not new to us. Why talk about it in the article? Besides, that has nothing to do with the patterns".

However, the objective of this article section is to answer the three questions:

- What is the best kind of oscillators to be used in this case?

- Are there any explicit limitations on using one of the forms?
- How correctly are patterns identified and confirmed when applying various kinds of oscillators?


We have answered these questions to the best of our ability. Besides, the way to test the conclusions has been made clear.

### Conclusion

In this article, we have discussed the simplest patterns occurring when trading currency baskets. But does that mean an end to the entire topic? Not in the least. There are still a lot of interesting opportunities.

Let me once again repeat the obvious thing: **The indicators attached to the articles are not intended for real trading!** They are unstable and used for illustrative purposes only.

### Programs used in the article:

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | testIndexMA.mq5 | Indicator | The test combined USD indicator with the moving average. |
| 2 | testIndexMA2.mq5 | Indicator | The test combined EUR indicator with the moving average. |
| 3 | testIndexZig-Zag1.mq5 | Indicator | The test ZigZag indicator capable of measuring and logging the lengths of individual segments. |
| 4 | testWPR.mq5 | Indicator | The test combined WPR indicator for EUR. |
| 5 | testWPRjpy.mq5 | Indicator | The test combined WPR indicator for USD. |
| 6 | testDistance.mq5 | Indicator | The test combined indicator plotting the difference between the two others. Here, these are testWPR.mq5 and testWPRjpy.mq5 (EUR and USD). |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2960](https://www.mql5.com/ru/articles/2960)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2960.zip "Download all attachments in the single ZIP archive")

[testIndexMA.mq5](https://www.mql5.com/en/articles/download/2960/testindexma.mq5 "Download testIndexMA.mq5")(4.82 KB)

[testIndexMA2.mq5](https://www.mql5.com/en/articles/download/2960/testindexma2.mq5 "Download testIndexMA2.mq5")(4.96 KB)

[testIndexZig-Zag1.mq5](https://www.mql5.com/en/articles/download/2960/testindexzig-zag1.mq5 "Download testIndexZig-Zag1.mq5")(13.88 KB)

[ZZdata.zip](https://www.mql5.com/en/articles/download/2960/zzdata.zip "Download ZZdata.zip")(37.28 KB)

[wpr.zip](https://www.mql5.com/en/articles/download/2960/wpr.zip "Download wpr.zip")(45.32 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**[Go to discussion](https://www.mql5.com/en/forum/190383)**

![Universal Trend with the Graphical Interface](https://c.mql5.com/2/26/MQL5_Universalni_trend.png)[Universal Trend with the Graphical Interface](https://www.mql5.com/en/articles/3018)

In this article a universal trend indicator is created based on a number of standard indicators. An additionally created graphical interface allows selecting the type of indicator and adjusting its parameter. The indicator is displayed in a separate window with rows of colored icons.

![Graphical Interfaces X: Sorting, rebuilding the table and controls in the cells (build 11)](https://c.mql5.com/2/26/MQL5-avatar-X-tableSort-001.png)[Graphical Interfaces X: Sorting, rebuilding the table and controls in the cells (build 11)](https://www.mql5.com/en/articles/3104)

We continue to add new features to the rendered table: data sorting, managing the number of columns and rows, setting the table cell types to place controls into them.

![Graphical Interfaces X: Word wrapping algorithm in the Multiline Text box (build 12)](https://c.mql5.com/2/27/MQL5-avatar-RedSquare-001.png)[Graphical Interfaces X: Word wrapping algorithm in the Multiline Text box (build 12)](https://www.mql5.com/en/articles/3173)

We continue to develop the Multiline Text box control. This time our task is to implement an automatic word wrapping in case a text box width overflow occurs, or a reverse word wrapping of the text to the previous line if the opportunity arises.

![A Universal Channel with the Graphical Interface](https://c.mql5.com/2/26/MQL5-avatar-Universalni-oscilyator-001.png)[A Universal Channel with the Graphical Interface](https://www.mql5.com/en/articles/2888)

All channel indicators are displayed as three lines, including central, top and bottom lines. The drawing principle of the central line is similar to a moving average, while the moving average indicator is mostly used for drawing channels. The top and bottom lines are located at equal distances from the central line. This distance can be determined in points, as percent of price (the Envelopes indicator), using a standard deviation value (Bollinger Bands), or an ATR value (Keltner channel).

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fawrviayvsmzxsvwshciuimuadxlcrxa&ssn=1769193339291863767&ssn_dr=0&ssn_sr=0&fv_date=1769193339&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2960&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Patterns%20available%20when%20trading%20currency%20baskets.%20Part%20II%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919333979037340&fz_uniq=5071951584847409531&sv=2552)

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