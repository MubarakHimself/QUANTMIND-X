---
title: Drawing Indicator's Emissions in MQL5
url: https://www.mql5.com/en/articles/26
categories: Trading
relevance_score: 6
scraped_at: 2026-01-23T11:30:39.862405
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/26&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062518908952421294)

MetaTrader 5 / Examples


### Introduction

Certainly, many traders and developers of trading strategies are interested in these questions:

- [How the strong market movements  are emerging?](https://www.mql5.com/go?link=http://translate.google.com/translate?js=y&prev=_t&hl=en&ie=UTF-8&layout=1&eotf=1&u=http://forum.mql4.com/ru/27371&sl=ru&tl=en)

- How to determine the correct direction of upcoming changes?
- How to open profitable position for trade?
- How to close position with maximal profit?


Finding the answers for these questions led me to creation of a new approach to the market research: construction and analysis of indicator emissions. To make it clearer, take a look at the following figures:

![](https://c.mql5.com/2/0/f1.jpg)

Fig. 1 Emission of DCMV indicator.

![](https://c.mql5.com/2/0/f2.jpg)

Fig. 2. Emission of indicator, based on iMA envelopes.

It shows the emission from different indicators, but the principle of their construction is the same. More and more points with different color and shape appear after the each tick. They form numerous clusters in the forms of nebulae, clouds, tracks, lines, arcs, etc. These shapes can help to detect the invisible springs and forces that affect the movement of market prices. The research and analysis of these emissions are something like chiromancy.

### Emission and its Properties

The emission is a set of points, located at the intersection points of specific lines of the indicator.

The properties of emissions haven't been still clear yet, they are still waiting for the researchers. Here is a list of known properties:

- the points of same type tend to cluster;
- the emission has a direction - from the present to the future or to the past;
- the clusters are important - the dense clusters can attract or, conversely, can repel the price.

### Calculation of the Indicator's Emission

Let's consider the fundamentals of emission calculation using an
example. Let's take two indicators - [iBands](https://www.mql5.com/en/docs/indicators/ibands) and [iMA](https://www.mql5.com/en/docs/indicators/ima) \- and find the intersection of their lines. We will use them to draw the points of emission. For this we will need [graphic objects](https://www.mql5.com/en/docs/objects). The algorithm is implemented in Expert Advisors, but it can be done in the indicators.

The initial indicators are presented in Fig. 3.:

![](https://c.mql5.com/2/0/f3.gif)

Fig. 3. The iBands (green) and iMA (red) indicators.

We need an Expert Advisor to create emission points. It's better to use the MQL5 Wizard to create an Expert Advisor template.

![](https://c.mql5.com/2/0/f4.png)

Fig. 4. Creating an Expert Advisor template using MQL5 Wizard.

```
//+------------------------------------------------------------------+
//|                                      Emission of Bands && MA.mq5 |
//|                                                 Copyright DC2008 |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "DC2008"
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---

//---
   return(0);
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
void OnTick()
  {
//---

  }
//+------------------------------------------------------------------+
```

First, we need some auxiliary plottings. We need to continue indicator lines using rays (Fig. 5.). It will allow to control the correctness of calculation and visualization of emission points. Later, we will remove these lines from the chart.

![](https://c.mql5.com/2/0/f5.gif)

Fig. 5. Auxiliary
plottings. Continuation of the indicator's lines using rays.

Thus, let's add the graphic objects (horizontal and trend lines) to the code of our Expert Advisor.

```
input bool     H_line=true;   // flag to enable drawing of the horizontal lines
input bool     I_line=true;   // flag to enable drawing of the indicator's lines
//---
string         name;
//---- indicator buffers
double      MA[];    // array for iMA indicator
double      BBH[];   // array for iBands indicator  - UPPER_BAND
double      BBL[];   // array for iBands indicator - LOWER_BAND
double      BBM[];   // array for iBands indicator - BASE_LINE
datetime    T[];     // array for time coordinates
//---- handles for indicators
int         MAHandle;   // iMA indicator handle
int         BBHandle;   // iBands indicator handle
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   MAHandle=iMA(Symbol(),0,21,0,MODE_EMA,PRICE_CLOSE);
   BBHandle=iBands(Symbol(),0,144,0,2,PRICE_CLOSE);
//---
   if(H_line)     // Horizontal lines of iBands indicator
      {
         //--- iBands - UPPER_BAND
         name="Hi";
         ObjectCreate(0,name,OBJ_HLINE,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,Red);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_DOT);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
         //--- iBands - LOWER_BAND
         name="Lo";
         ObjectCreate(0,name,OBJ_HLINE,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,Blue);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_DOT);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
         //--- iBands - BASE_LINE
         name="MIDI";
         ObjectCreate(0,name,OBJ_HLINE,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,DarkOrange);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_DOT);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
      }
//---
   if(I_line)     // Indicator lines
      {
         //--- iMA
         name="MA";
         ObjectCreate(0,name,OBJ_TREND,0,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,Red);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_SOLID);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,2);
         ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_LEFT,1);
         //--- iBands - UPPER_BAND
         name="BH";
         ObjectCreate(0,name,OBJ_TREND,0,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,MediumSeaGreen);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_SOLID);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_LEFT,1);
         //--- iBands - LOWER_BAND
         name="BL";
         ObjectCreate(0,name,OBJ_TREND,0,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,MediumSeaGreen);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_SOLID);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_LEFT,1);
         //--- iBands - BASE_LINE
         name="BM";
         ObjectCreate(0,name,OBJ_TREND,0,0,0,0);
         ObjectSetInteger(0,name,OBJPROP_COLOR,MediumSeaGreen);
         ObjectSetInteger(0,name,OBJPROP_STYLE,STYLE_SOLID);
         ObjectSetInteger(0,name,OBJPROP_WIDTH,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_RIGHT,1);
         ObjectSetInteger(0,name,OBJPROP_RAY_LEFT,1);
      }
   return(0);
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
void OnTick()
  {
   //--- filling the arrays with current values
   CopyBuffer(MAHandle,0,0,2,MA);
   ArraySetAsSeries(MA,true);
   CopyBuffer(BBHandle,0,0,2,BBM);
   ArraySetAsSeries(BBM,true);
   CopyBuffer(BBHandle,1,0,2,BBH);
   ArraySetAsSeries(BBH,true);
   CopyBuffer(BBHandle,2,0,2,BBL);
   ArraySetAsSeries(BBL,true);
   CopyTime(Symbol(),0,0,10,T);
   ArraySetAsSeries(T,true);

   //--- Horizontal lines of iBands indicator (correction)
   if(H_line)
      {
      name="Hi";
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBH[0]);
      name="Lo";
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBL[0]);
      name="MIDI";
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBM[0]);
      }
   //--- Indicator's lines (correction)
   if(I_line)
      {
      name="MA";  //--- iMA
      ObjectSetInteger(0,name,OBJPROP_TIME,T[1]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,MA[1]);
      ObjectSetInteger(0,name,OBJPROP_TIME,1,T[0]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,1,MA[0]);
      name="BH";  //--- iBands - UPPER_BAND
      ObjectSetInteger(0,name,OBJPROP_TIME,T[1]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBH[1]);
      ObjectSetInteger(0,name,OBJPROP_TIME,1,T[0]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,1,BBH[0]);
      name="BL";  //--- iBands - LOWER_BAND
      ObjectSetInteger(0,name,OBJPROP_TIME,T[1]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBL[1]);
      ObjectSetInteger(0,name,OBJPROP_TIME,1,T[0]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,1,BBL[0]);
      name="BM";  //--- iBands - BASE_LINE
      ObjectSetInteger(0,name,OBJPROP_TIME,T[1]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,BBM[1]);
      ObjectSetInteger(0,name,OBJPROP_TIME,1,T[0]);
      ObjectSetDouble(0,name,OBJPROP_PRICE,1,BBM[0]);
      }
  }
//+------------------------------------------------------------------+
```

Since the emission continues to the past and to the future, the trend line [properties](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property) should be following:

- OBJPROP\_RAY\_LEFT = 1,  (Ray goes left);
- OBJPROP\_RAY\_RIGHT = 1, (Ray goes right).


As a result, the chart with additional lines will look as presented in Fig. 6.

The preparatory phase is
completed, now let's proceed to the emission. We are going to create the first series of points at the intersection of the following lines:

- between the "MA" (iMA) line and "BH" line (iBands = UPPER\_BAND);
- between the "MA" (iMA) line and "BL" line (iBands = LOWER\_BAND);
- between the "MA" (iMA) line and "BM" line (iBands = BASE\_BAND).

![](https://c.mql5.com/2/0/f6.png)

Fig. 6. Auxiliary plottings. Continuation of the indicator's lines using straight lines.

Now it's time to calculate the coordinates of intersection and to draw the points of the emission. Let's create the function:

```
void Draw_Point(
                string   P_name,     // Object name (OBJ_ARROW)
                double   P_y1,       // Y-coordinate of the 1st line at the [1] bar
                double   P_y0,       // Y-coordinate of the 1st line at the [0] bar
                double   P_yy1,      // Y-coordinate of the 2nd line at the [1] bar
                double   P_yy0,      // Y-coordinate of the 2nd line at the [0] bar
                char     P_code1,    // Char at the right side of the [0] bar
                char     P_code2,    // Char at the left side of the [0] bar
                color    P_color1,   // Color of point at the right side of the [0] bar
                color    P_color2    // color of point at the left side of the [0] bar
                )
  {
   double   P,X;
   datetime P_time;
   if(MathAbs((P_yy0-P_yy1)-(P_y0-P_y1))>0)
     {
      P=P_y1+(P_y0-P_y1)*(P_y1-P_yy1)/((P_yy0-P_yy1)-(P_y0-P_y1));
      X=(P_y1-P_yy1)/((P_yy0-P_yy1)-(P_y0-P_y1));
      if(X>draw_period)
        {
         P_time=T[0]+(int)(X*PeriodSeconds());
         ObjectCreate(0,P_name,OBJ_ARROW,0,0,0);
         ObjectSetDouble(0,P_name,OBJPROP_PRICE,P);
         ObjectSetInteger(0,P_name,OBJPROP_TIME,P_time);
         ObjectSetInteger(0,P_name,OBJPROP_WIDTH,0);
         ObjectSetInteger(0,P_name,OBJPROP_ARROWCODE,P_code1);
         ObjectSetInteger(0,P_name,OBJPROP_COLOR,P_color1);
         if(X<0)
           {
            ObjectSetInteger(0,P_name,OBJPROP_ARROWCODE,P_code2);
            ObjectSetInteger(0,P_name,OBJPROP_COLOR,P_color2);
           }
        }
     }
  }
```

And adding the following lines of code to the function [OnTick](https://www.mql5.com/en/docs/basis/function/events#ontick):

```
//+------------------------------------------------------------------+
   int GTC=GetTickCount();
//+------------------------------------------------------------------+
   name="H"+(string)GTC;
   Draw_Point(name,BBH[1],BBH[0],MA[1],MA[0],170,178,Red,Red);
   name="L"+(string)GTC;
   Draw_Point(name,BBL[1],BBL[0],MA[1],MA[0],170,178,Blue,Blue);
   name="M"+(string)GTC;
   Draw_Point(name,BBM[1],BBM[0],MA[1],MA[0],170,178,Green,Green);
//---
   ChartRedraw(0);
```

Now let's run the Expert Advisor and look at the result  (Fig. 7.).

It's good, but there are some other intersection cases, that we haven't considered. For example, the [iBands](https://www.mql5.com/en/docs/indicators/ibands) indicator has three lines that intersect each other and can complement the overall picture.

![](https://c.mql5.com/2/0/f7.png)

Fig. 7. The emission of the iMA and iBands indicators (3 intersections).

Now, let's try to add another one series of point to the calculated emission, the intersection between the following lines:

- between the line "BH" (iBands = UPPER\_BAND) and line "BL" (iBands =
LOWER\_BAND);
- between the line  "BH" (iBands = UPPER\_BAND) and line "BM" (iBands =
BASE\_BAND);
- between the line  "BL" (iBands = LOWER\_BAND) and line "BM" (iBands =
BASE\_BAND).

Due to these intersections, we would get 3 points, but all they will have the same coordinates. Therefore, it's sufficient to use an only one intersection, between the line "BH" and line "BL".

Let's add these lines of code to our Expert Advisor, and take a look at the result (Fig. 8.).

```
   name="B"+(string)GTC;
   Draw_Point(name,BBH[1],BBH[0],BBL[1],BBL[0],170,178,Magenta,Magenta);
```

![](https://c.mql5.com/2/0/f8.png)

Fig. 8. Emission of the iMA and iBands indicators (4 intersections).

So, we have got the emission, but there is a feeling that we have missed something important. Nevertheless, what we have missed?

Why we have used just such input parameters? What we will get if we change them? And anyway, what is their role in the emissions?

All right, the emission we've got corresponds to a single frequency, resulted from the input parameters of the indicator. To calculate the full multi-frequency spectrum, it's necessary to perform the same calculations for other frequencies. As an example, here is my version of the possible emission spectrum:

```
//---- handles for indicators
int         MAHandle[5];   // handles array of iMA indicators
int         BBHandle[7];   // handles array of iBands indicator
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   MAHandle[0]=iMA(NULL,0,21,0,MODE_EMA,PRICE_CLOSE);
   MAHandle[1]=iMA(NULL,0,34,0,MODE_EMA,PRICE_CLOSE);
   MAHandle[2]=iMA(NULL,0,55,0,MODE_EMA,PRICE_CLOSE);
   MAHandle[3]=iMA(NULL,0,89,0,MODE_EMA,PRICE_CLOSE);
   MAHandle[4]=iMA(NULL,0,144,0,MODE_EMA,PRICE_CLOSE);
//---
   BBHandle[0]=iBands(NULL,0,55,0,2,PRICE_CLOSE);
   BBHandle[1]=iBands(NULL,0,89,0,2,PRICE_CLOSE);
   BBHandle[2]=iBands(NULL,0,144,0,2,PRICE_CLOSE);
   BBHandle[3]=iBands(NULL,0,233,0,2,PRICE_CLOSE);
   BBHandle[4]=iBands(NULL,0,377,0,2,PRICE_CLOSE);
   BBHandle[5]=iBands(NULL,0,610,0,2,PRICE_CLOSE);
   BBHandle[6]=iBands(NULL,0,987,0,2,PRICE_CLOSE);
//---
   return(0);
  }
```

To consider all possible combinations, let's add the following code to the Expert Advisor:

```
//+------------------------------------------------------------------+
   CopyTime(NULL,0,0,10,T);
   ArraySetAsSeries(T,true);
   int GTC=GetTickCount();
//+------------------------------------------------------------------+
   int iMax=ArraySize(BBHandle)-1;
   int jMax=ArraySize(MAHandle)-1;
   for(int i=0; i<iMax; i++)
     {
      for(int j=0; j<jMax; j++)
        {
         //--- filling the arrays with current values
         CopyBuffer(MAHandle[j],0,0,2,MA);
         ArraySetAsSeries(MA,true);
         CopyBuffer(BBHandle[i],0,0,2,BBM);
         ArraySetAsSeries(BBM,true);
         CopyBuffer(BBHandle[i],1,0,2,BBH);
         ArraySetAsSeries(BBH,true);
         CopyBuffer(BBHandle[i],2,0,2,BBL);
         ArraySetAsSeries(BBL,true);

         name="H"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBH[1],BBH[0],MA[1],MA[0],250,158,Aqua,Aqua);
         name="L"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBL[1],BBL[0],MA[1],MA[0],250,158,Blue,Blue);
         name="M"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBM[1],BBM[0],MA[1],MA[0],250,158,Green,Green);
         name="B"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBH[1],BBH[0],BBL[1],BBL[0],250,158,Magenta,Magenta);
        }
     }
//---
   ChartRedraw(0);
```

The more frequencies is involved in the emission spectrum, the better picture will be on chart, but you should not abuse it - it's a simplest way to exhaust the computer resources, and to get the chaos on the chart. The number of frequencies can be determined experimentally. For the better perception of graphics, we must pay special attention to the drawing style.

![](https://c.mql5.com/2/0/f9.png)

Fig. 9. Multi-frequency emission spectrum.

### Of Emission's Drawing Styles

MQL5 language provides a wide range of [Web Colors](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) and [Windings characters](https://www.mql5.com/en/docs/constants/objectconstants/wingdings) for drawing emissions. I would like to share my thoughts about it:


1. Each person has his own perception of graphic images, so you'll need some time to customize the emissions.
2. The "chaos" in Fig.9. doesn't allow to recognize any regularities or patterns in images. It's an example of bad drawing.
3. Try to use the neighbor colors in the rainbow spectrum.
4. The character codes for the past (from the left side of \[0\] bar) and for the future (from the right side of \[0\] bar) should differ.
5. The successful combination of colors and shapes of points is able to
turn the emission into the masterpieces, which will not only help in the
    trade, but will also pleasure your eyes.

As an example, here is my version of drawing style for the emission (see Figures 10-17):

```
         name="H"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBH[1],BBH[0],MA[1],MA[0],250,158,Aqua,Aqua);
         name="L"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBL[1],BBL[0],MA[1],MA[0],250,158,Blue,Blue);
         name="M"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBM[1],BBM[0],MA[1],MA[0],250,158,Magenta,Magenta);
         name="B"+(string)GTC+(string)i+(string)j;
         Draw_Point(name,BBH[1],BBH[0],BBL[1],BBL[0],250,158,DarkOrchid,DarkOrchid);
```

### The iMA and iBands Emissions Gallery

The images with this emissions are presented in this chapter.

![](https://c.mql5.com/2/0/f10.gif)

Fig. 10.

![](https://c.mql5.com/2/0/f11.gif)

Fig. 11

![](https://c.mql5.com/2/0/f12.gif)

Fig. 12.

![](https://c.mql5.com/2/0/f13.gif)

Fig. 13.

![](https://c.mql5.com/2/0/f14.gif)

Fig. 14.

![](https://c.mql5.com/2/0/f15.gif)

Fig. 15.

![](https://c.mql5.com/2/0/f16.gif)

Fig. 16.

![](https://c.mql5.com/2/0/f17.gif)

Fig. 17.

### Emission Analysis

The analysis of emissions is a separate task. The most useful thing is to look at its dynamics in a real time, it's the best way to understand many effects and patterns.

Pay attention to the price corrections - it seems that the emission "knows" the target price. In addition, you can see the support, resistance and equilibrium price levels.

### Conclusion

1. The emissions of the indicators might be interesting to traders and trade systems developers, who are looking for new approaches in market research and analysis.
2. As an introductory article, it doesn't contain the ready solutions.
However, the presented technology for emission calculation can be applied in other indicators or their combinations.
3. Preparing this article, I have collected more questions than answers. Here are some of them: how to optimize the algorithm of emission drawing;  what is the role of the emission spectrum characteristics in the structure of emission;  how to use the emissions in automated trading?

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/26](https://www.mql5.com/ru/articles/26)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/26.zip "Download all attachments in the single ZIP archive")

[emission\_bands\_and\_ma\_en.mq5](https://www.mql5.com/en/articles/download/26/emission_bands_and_ma_en.mq5 "Download emission_bands_and_ma_en.mq5")(8.62 KB)

[emission\_bands\_and\_ma\_spectrum\_en.mq5](https://www.mql5.com/en/articles/download/26/emission_bands_and_ma_spectrum_en.mq5 "Download emission_bands_and_ma_spectrum_en.mq5")(5.63 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing multi-module Expert Advisors](https://www.mql5.com/en/articles/3133)
- [3D Modeling in MQL5](https://www.mql5.com/en/articles/2828)
- [Statistical distributions in the form of histograms without indicator buffers and arrays](https://www.mql5.com/en/articles/2714)
- [The ZigZag Indicator: Fresh Approach and New Solutions](https://www.mql5.com/en/articles/646)
- [Calculation of Integral Characteristics of Indicator Emissions](https://www.mql5.com/en/articles/610)
- [Testing Performance of Moving Averages Calculation in MQL5](https://www.mql5.com/en/articles/106)
- [Migrating from MQL4 to MQL5](https://www.mql5.com/en/articles/81)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/735)**
(11)


![nerobot](https://c.mql5.com/avatar/avatar_na2.png)

**[nerobot](https://www.mql5.com/en/users/nerobot)**
\|
5 Aug 2010 at 19:26

Really interesting, but when I compile and use them, nothing happens.

Not sure if I'm missing something?

Steven

![nerobot](https://c.mql5.com/avatar/avatar_na2.png)

**[nerobot](https://www.mql5.com/en/users/nerobot)**
\|
5 Aug 2010 at 19:48

Never mind, it's working now.


![Mario](https://c.mql5.com/avatar/2010/9/4C90BB1F-3416.jpg)

**[Mario](https://www.mql5.com/en/users/maryan.dirtyn)**
\|
13 Sep 2010 at 21:26

it's showing something else. [![](https://c.mql5.com/3/2/hh__1.png)](https://c.mql5.com/3/2/hh.png "https://c.mql5.com/3/2/hh.png")

![Sergey Pavlov](https://c.mql5.com/avatar/2010/2/4B7AECD8-6F67.jpg)

**[Sergey Pavlov](https://www.mql5.com/en/users/dc2008)**
\|
13 Sep 2010 at 22:58

**maryan.dirtyn:**

It's showing something else.

Try looking at M1.


![Aleksey Rodionov](https://c.mql5.com/avatar/2019/1/5C44359A-8B22.jpg)

**[Aleksey Rodionov](https://www.mql5.com/en/users/zeleniy)**
\|
17 Oct 2011 at 14:46

It doesn't look like that at all and in addition it loads the machine a lot (I looked at real EURUSD M1 quotes).

[![](https://c.mql5.com/3/5/EURUSDM1__3.png)](https://c.mql5.com/3/5/EURUSDM1__2.png "https://c.mql5.com/3/5/EURUSDM1__2.png")

![Introduction to MQL5: How to write simple Expert Advisor and Custom Indicator](https://c.mql5.com/2/0/a03__1.png)[Introduction to MQL5: How to write simple Expert Advisor and Custom Indicator](https://www.mql5.com/en/articles/35)

MetaQuotes Programming Language 5 (MQL5), included in MetaTrader 5 Client Terminal, has many new possibilities and higher performance, compared to MQL4. This article will help you to get acquainted with this new programming language. The simple examples of how to write an Expert Advisor and Custom Indicator are presented in this article. We will also consider some details of MQL5 language, that are necessary to understand these examples.

![The Drawing Styles in MQL5](https://c.mql5.com/2/0/180x180LoongIndicator.png)[The Drawing Styles in MQL5](https://www.mql5.com/en/articles/45)

There are 6 drawing styles in MQL4 and 18 drawing styles in MQL5. Therefore, it may be worth writing an article to introduce MQL5's drawing styles. In this article, we will consider the details of drawing styles in MQL5. In addition, we will create an indicator to demonstrate how to use these drawing styles, and refine the plotting.

![MQL5: Analysis and Processing of Commodity Futures Trading Commission (CFTC) Reports in MetaTrader 5](https://c.mql5.com/2/0/trader_mql5__1.png)[MQL5: Analysis and Processing of Commodity Futures Trading Commission (CFTC) Reports in MetaTrader 5](https://www.mql5.com/en/articles/34)

In this article, we will develop a tool for CFTC report analysis. We will solve the following problem: to develop an indicator, that allows using the CFTC report data directly from the data files provided by Commission without an intermediate processing and conversion. Further, it can be used for the different purposes: to plot the data as an indicator, to proceed with the data in the other indicators, in the scripts for the automated analysis, in the Expert Advisors for the use in the trading strategies.

![Practical Implementation of Digital Filters in MQL5 for Beginners](https://c.mql5.com/2/0/Filter.png)[Practical Implementation of Digital Filters in MQL5 for Beginners](https://www.mql5.com/en/articles/32)

The idea of digital signal filtering has been widely discussed on forum topics about building trading systems. And it would be imprudent not to create a standard code of digital filters in MQL5. In this article the author describes the transformation of simple SMA indicator's code from his article "Custom Indicators in MQL5 for Newbies" into code of more complicated and universal digital filter. This article is a logical sequel to the previous article. It also tells how to replace text in code and how to correct programming errors.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=xebqqbylunvvhvzhsdknqafvwkdmlyex&ssn=1769157038159954596&ssn_dr=0&ssn_sr=0&fv_date=1769157038&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F26&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Drawing%20Indicator%27s%20Emissions%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915703866488827&fz_uniq=5062518908952421294&sv=2552)

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