---
title: Plotting trend lines based on fractals using MQL4 and MQL5
url: https://www.mql5.com/en/articles/1201
categories: Trading, Trading Systems
relevance_score: 4
scraped_at: 2026-01-23T17:39:42.404569
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/1201&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068497971579910700)

MetaTrader 5 / Trading


### Table of Contents

- [Introduction](https://www.mql5.com/en/articles/1201#intro)
- [1\. Input parameters, DeInit() function and initial declaration of variables](https://www.mql5.com/en/articles/1201#c1)
- [2\. Searching for nearest fractals](https://www.mql5.com/en/articles/1201#c2)
- [3\. Determining price and time values of fractals](https://www.mql5.com/en/articles/1201#c3)
- [4\. Creating objects and editing their properties. Redrawing the lines](https://www.mql5.com/en/articles/1201#c4)
- [5\. Checking the bars history loading](https://www.mql5.com/en/articles/1201#c5)
- [6\. Signals of trend lines breakthroughs, push notifications](https://www.mql5.com/en/articles/1201#c6)
- [7\. Practical use of trend lines in trading](https://www.mql5.com/en/articles/1201#c7)
- [Conclusion](https://www.mql5.com/en/articles/1201#summary)

### Introduction

Recently, I have been thinking about using trend lines. There was a question about choosing a method of determining the points for plotting lines, and also about plotting accuracy. I decided to use fractals as a basis.

I often analyze markets at my main job where I can spend some time on trading. Also, you can't just draw the lines on a larger timeframe - the line should be plotted by extreme points with accuracy up to 15 minutes. The reason for this is that the fractal time on a larger timeframe doesn't always match the time of the same extreme point on M15. In short, this is where automation comes to help. It happened that I began writing the code using MQL5 and then moved to MQL4, because I needed this program for MetaTrader 4.

In this article I presented my solution of the problem using MQL4 and MQL5. The article provides the comparative view, but it would be inappropriate to compare the efficiency of MQL4 and MQL5 here. Also, I understand that there are probably other solutions, more effective than mine. The article can be useful to beginners who write scripts using either MQL4 or MQL5, especially to those who plan to use fractals and trend lines.

### 1\. Input parameters, DeInit() function and initial declaration of variables

I used the following variables as input parameters:

```
input color Resistance_Color=Red;       // setting the resistance line color
input ENUM_LINE_STYLE Resistance_Style; // setting the resistance line style
input int Resistance_Width=1;           // setting the resistance line width
input color Support_Color=Red;          // setting the support line color
input ENUM_LINE_STYLE Support_Style;    // setting the support line style
input int Support_Width=1;              // setting the support line width
```

These parameters are the same for MQL4 and MQL5.

In MQL5 we need to create the indicator in advance:

```
//--- iFractals indicator handle
int Fractal;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- getting the iFractals indicator handle
   Fractal=iFractals(Symbol(),PERIOD_D1);
//---
   return(INIT_SUCCEEDED);
  }
```

Because the program will be drawing graphical objects, it makes sense to remove them when removing the Expert Advisor from the chart:

```
void OnDeinit(const int reason)
  {
   ObjectDelete(0,"TL_Resistance");
   ObjectDelete(0,"TL_Support");
  }
```

Plotting two lines (support and resistance) requires four points. To determine the line passing point we need to know the time and price .

The coordinates are determined in this order: first, we find the extreme bar, knowing the extreme bar we can determine the price and time of the extreme point.

Declaring variables in the [OnTick()](https://www.mql5.com/en/docs/basis/function/events#ontick) function:

| MQL4 |
| --- |
| ```<br>//--- declaration of variables<br>int n,UpperFractal_1,UpperFractal_2,LowerFractal_1,LowerFractal_2;<br>``` |

| MQL5 |
| --- |
| ```<br>//--- declaration of variables<br>int n,UpperFractal_1,UpperFractal_2,LowerFractal_1,LowerFractal_2;<br>//--- declaring the arrays for writing values of the iFractal indicator buffer<br>double FractalDown[],FractalUp[];<br>double UpFractal_1,UpFractal_2,LowFractal_1,LowFractal_2;<br>``` |

First off, I declared only those variables which store indexes of bars with formed fractals.

In MQL4:

1. n - the variable is needed for finding the nearest known fractal using the [for loop operator](https://book.mql4.com/en/operators/for);
2. UpperFractal\_1, UpperFractal\_2,  LowerFractal\_1, LowerFractal\_2 - these variables will store the index of a bar at the first and the second nearest extreme point with the highest/lowest price (in terms of determining fractals);


In MQL5 we introduce additional variables:

1. FractalDown\[\],FractalUp\[\]; - declaring arrays of double values for storing the values of the [iFractals](https://www.mql5.com/en/docs/indicators/ifractals) indicator buffer;
2. Next, the double type variables: UpFractal\_1,UpFractal\_2,LowFractal\_1,LowFractal\_2. They will store the price values of extreme points.


### 2\. Searching for nearest fractals

To find the index of a bar with a formed fractal we use the [for loop operator](https://book.mql4.com/en/operators/for).

Let's determine the indexes of the first two bars which correspond to the first and second upper fractals:

| MQL4 |
| --- |
| ```<br>//--- finding the bar index of the first nearest upper fractal<br>   for(n=0; n<(Bars-1);n++)<br>     {<br>      if(iFractals(NULL,1440,MODE_UPPER,n)!=NULL)<br>         break;<br>      UpperFractal_1=n+1;<br>     }<br>//--- finding the bar index of the second nearest upper fractal<br>   for(n=UpperFractal_1+1; n<(Bars-1);n++)<br>     {<br>      if(iFractals(NULL,1440,MODE_UPPER,n)!=NULL)<br>         break;<br>      UpperFractal_2=n+1;<br>     }<br>``` |

| MQL5 |
| --- |
| ```<br>//--- first, we need to write the Fractal indicator buffer values into the arrays<br>//--- filling arrays with buffer values<br>   CopyBuffer(Fractal,0,TimeCurrent(),Bars(Symbol(),PERIOD_D1),FractalUp);<br>   CopyBuffer(Fractal,1,TimeCurrent(),Bars(Symbol(),PERIOD_D1),FractalDown);<br>//--- indexing like in timeseries<br>   ArraySetAsSeries(FractalUp,true);<br>   ArraySetAsSeries(FractalDown,true);<br>//--- next, we use the for loop operator to find the first upper fractal<br>   for(n=0; n<Bars(Symbol(),PERIOD_D1); n++)<br>     {<br>      //--- if the value is not empty, break the loop<br>      if(FractalUp[n]!=EMPTY_VALUE)<br>         break;<br>     }<br>//--- writing the price value of the first fractal into the variable<br>   UpFractal_1=FractalUp[n];<br>//--- writing the index of the first fractal into the variable<br>   UpperFractal_1=n;<br>//--- finding the second upper fractal <br>   for(n=UpperFractal_1+1; n<Bars(Symbol(),PERIOD_D1); n++)<br>     {<br>      if(FractalUp[n]!=EMPTY_VALUE) //if the value is not empty, break the loop<br>         break;<br>     }<br>//--- writing the price value of the second fractal into the variable<br>   UpFractal_2=FractalUp[n];<br>//--- writing the index of the second fractal into the variable<br>   UpperFractal_2=n;<br>``` |

Here I clearly demonstrated one of the key differences between MQL5 and MQL4 - using the [functions for accessing timeseries](https://www.mql5.com/en/docs/series).

In MQL4 I immediately started finding the index of the bar with a formed fractal, but in MQL5 I specified the FractalUp\[\] and FractalDown\[\] arrays for storing the price values of upper and lower fractals by accessing the [iFractals](https://www.mql5.com/en/docs/indicators/ifractals) indicator with the [CopyBuffer()](https://www.mql5.com/en/docs/series/copybuffer) function. Next, I set the indexing of these arrays as in timeseries using the [ArraySetAsSeries()](https://www.mql5.com/en/docs/array/arraysetasseries) function.

In MQL4 I got only indexes of the bars with known fractals, but in MQL5 I used the [CopyBuffer()](https://www.mql5.com/en/docs/series/copybuffer) function to get the bar indexes and the price values of fractals.

Similarly, we find first two lower fractals:

| MQL4 |
| --- |
| ```<br>//--- finding the bar index of the first nearest lower fractal<br>   for(n=0; n<(Bars-1);n++)<br>     {<br>      if(iFractals(NULL,1440,MODE_LOWER,n)!=NULL)<br>         break;<br>      LowerFractal_1=n+1;<br>     }<br>//--- finding the bar index of the second nearest lower fractal<br>   for(n=LowerFractal_1+1; n<(Bars-1);n++)<br>     {<br>      if(iFractals(NULL,1440,MODE_LOWER,n)!=NULL)<br>         break;<br>      LowerFractal_2=n+1;<br>     }<br>``` |

| MQL5 |
| --- |
| ```<br>//--- finding the values of the lower fractals<br>//--- finding the first lower fractal<br>   for(n=0; n<Bars(Symbol(),PERIOD_D1); n++)<br>     {<br>      //--- if the value is not empty, break the loop<br>      if(FractalDown[n]!=EMPTY_VALUE)<br>         break;<br>     }<br>//--- writing the price value of the first fractal into the variable<br>   LowFractal_1=FractalDown[n];<br>//--- writing the index of the first fractal into the variable<br>   LowerFractal_1=n;<br>//--- finding the second lower fractal <br>   for(n=LowerFractal_1+1; n<Bars(Symbol(),PERIOD_D1); n++)<br>     {<br>      if(FractalDown[n]!=EMPTY_VALUE)<br>         break;<br>     }<br>//--- writing the price value of the second fractal into the variable<br>   LowFractal_2=FractalDown[n];<br>//--- writing the index of the second fractal into the variable<br>   LowerFractal_2=n;<br>``` |

As you see, the code is very similar in MQL4 and MQL5. There is a slight difference in syntax.

### 3\. Determining price and time values of fractals

To draw the line we need to determine the time and price of a fractal. Of course, in MQL4 we could simply use the High\[\] and Low\[\] [predefined timeseries](https://docs.mql4.com/en/predefined), and the [iTime()](https://docs.mql4.com/en/series/itime) function, however we also need to get more precise time coordinates to ensure the correct plotting of the trend line.

Fig. 1-2 show the difference between the time values of extreme points on H4 and M15 timeframes.

![Fig.1. The extreme point time value on H4](https://c.mql5.com/2/12/GBPNZD_mH4.png)

Fig.1. The extreme point time value on H4

![Fig.2. The extreme point time value on M15](https://c.mql5.com/2/12/GBPNZD_mM15.png)

Fig.2. The extreme point time value on M15

I came to a conclusion that the extreme point accuracy of 15 minutes is quite sufficient for my purposes.

In general, the principle of extreme point clarification is almost the same for both MQL4 and MQL5, but there are certain differences in details:

| MQL4 | MQL5 |
| --- | --- |
| 1. Determine the extreme point time value on a larger timeframe;<br>2. Using the found time value, determine the index of the extreme bar on a smaller timeframe with the [iBarShift()](https://docs.mql4.com/en/series/ibarshift) function;<br>3. Because 24 hours can be represented as an array of 96 15-minute bars, we search for an extreme point (the highest/lowest value) among these 96 elements using the [iHigh()](https://docs.mql4.com/en/series/ihigh), [iLow()](https://docs.mql4.com/en/series/ilow) , [iTime()](https://docs.mql4.com/en/series/itime), [ArrayMaximum()](https://docs.mql4.com/array/arraymaximum) and [ArrayMinimum()](https://docs.mql4.com/array/arrayminimum) functions. | 1. Determine the extreme point time value on a larger timeframe;<br>2. Using the found time value, determine the generation time of the next day bar. We need this value for use in the [CopyHigh()](https://www.mql5.com/en/docs/series/copyhigh), [CopyLow()](https://www.mql5.com/en/docs/series/copylow) and [CopyTime()](https://www.mql5.com/en/docs/series/copytime) functions;<br>3. Declare and fill the arrays for storing the price and time values for the 15-minute timeframe;<br>4. Using the [ArrayMaximum()](https://www.mql5.com/en/docs/array/arraymaximum) and [ArrayMinimum()](https://www.mql5.com/en/docs/array/arrayminimum) functions, find the lowest and highest price values, and the time values of the clarified extreme points. |

The code for each step is shown below:

| MQL4 |
| --- |
| ```<br>// Step 1. Determining the extreme point time value on a larger timeframe:<br>//--- determining the time of fractals<br>   datetime UpFractalTime_1=iTime(NULL, 1440,UpperFractal_1);<br>   datetime UpFractalTime_2=iTime(NULL, 1440,UpperFractal_2);<br>   datetime LowFractalTime_1=iTime(NULL, 1440,LowerFractal_1);<br>   datetime LowFractalTime_2=iTime(NULL, 1440,LowerFractal_2);<br>``` |
| ```<br>// Step 2.  Determining the index of the extreme bar on a smaller timeframe:   <br>//--- finding the fractal index on M15<br>   int UpperFractal_1_m15=iBarShift(NULL, 15, UpFractalTime_1,true);<br>   int UpperFractal_2_m15=iBarShift(NULL, 15, UpFractalTime_2,true);<br>   int LowerFractal_1_m15=iBarShift(NULL, 15, LowFractalTime_1,true);<br>   int LowerFractal_2_m15=iBarShift(NULL, 15, LowFractalTime_2,true);<br>``` |
| ```<br>// Step 3. Using the arrays to find the clarified extreme points on М15:<br>//--- using the arrays to find the clarified extreme points<br>//--- introducing the i variable to use in the for loop operator<br>   int i;<br>//--- 1. First, find the lower extreme points<br>//--- 3.1 Finding the first lower extreme point<br>//--- declaring the array for storing the index values of the bars<br>   int Lower_1_m15[96];<br>//--- declaring the array for storing the price values<br>   double LowerPrice_1_m15[96];<br>//--- starting the for loop:<br>   for(i=0;i<=95;i++)<br>     {<br>      //--- filling the array with the bar index values<br>      Lower_1_m15[i]=LowerFractal_1_m15-i;<br>      //--- filling the array with the price values<br>      LowerPrice_1_m15[i]=iLow(NULL,15,LowerFractal_1_m15-i);<br>     }<br>//--- determining the minimum price value in the array<br>   int LowestPrice_1_m15=ArrayMinimum(LowerPrice_1_m15,WHOLE_ARRAY,0);<br>//--- determining the bar with the lowest price in the array<br>   int LowestBar_1_m15=Lower_1_m15[LowestPrice_1_m15];<br>//--- determining the time of the lowest price bar<br>   datetime LowestBarTime_1_m15=iTime(NULL,15,Lower_1_m15[LowestPrice_1_m15]);<br>//--- 3.2 Finding the second lower extreme point<br>   int Lower_2_m15[96];<br>   double LowerPrice_2_m15[96];<br>   for(i=0;i<=95;i++)<br>     {<br>      //--- filling the array with the bar index values<br>      Lower_2_m15[i]=LowerFractal_2_m15-i;<br>      //--- filling the array with the price values<br>      LowerPrice_2_m15[i]=iLow(NULL,15,LowerFractal_2_m15-i);<br>     }<br>//--- determining the minimum price value in the array<br>   int LowestPrice_2_m15=ArrayMinimum(LowerPrice_2_m15,WHOLE_ARRAY,0);<br>//--- determining the bar with the lowest price in the array<br>   int LowestBar_2_m15=Lower_2_m15[LowestPrice_2_m15];<br>//--- determining the time of the lowest price bar<br>   datetime LowestBarTime_2_m15=iTime(NULL,15,Lower_2_m15[LowestPrice_2_m15]);<br>//--- 3.3 Finding the first upper extreme point<br>   int Upper_1_m15[96];<br>   double UpperPrice_1_m15[96];<br>   for(i=0;i<=95;i++)<br>     {<br>      //--- filling the array with the bar index values<br>      Upper_1_m15[i]=UpperFractal_1_m15-i;<br>      //--- filling the array with the price values<br>      UpperPrice_1_m15[i]=iHigh(NULL,15,UpperFractal_1_m15-i);<br>     }<br>//--- determining the maximum price value in the array<br>   int HighestPrice_1_m15=ArrayMaximum(UpperPrice_1_m15,WHOLE_ARRAY,0);<br>//--- determining the bar with the highest price in the array<br>   int HighestBar_1_m15=Upper_1_m15[HighestPrice_1_m15];<br>//--- determining the time of the highest price bar<br>   datetime HighestBarTime_1_m15=iTime(NULL,15,Upper_1_m15[HighestPrice_1_m15]);<br>//--- 3.4 Finding the second upper extreme point<br>   int Upper_2_m15[96];<br>   double UpperPrice_2_m15[96];<br>   for(i=0;i<=95;i++)<br>     {<br>      //--- filling the array with the bar index values<br>      Upper_2_m15[i]=UpperFractal_2_m15-i;<br>      //--- filling the array with the price values<br>      UpperPrice_2_m15[i]=iHigh(NULL,15,UpperFractal_2_m15-i);<br>     }<br>``` |

| MQL5 |
| --- |
| ```<br>// Step 1. Determining the extreme point time value on a larger timeframe:<br>//--- declaring the arrays for storing the time values of the corresponding bar index on a larger timeframe<br>   datetime UpFractalTime_1[],LowFractalTime_1[],UpFractalTime_2[],LowFractalTime_2[];<br>//--- determining the time of fractals on a larger timeframe<br>   CopyTime(Symbol(),PERIOD_D1,UpperFractal_1,1,UpFractalTime_1);<br>   CopyTime(Symbol(),PERIOD_D1,LowerFractal_1,1,LowFractalTime_1);<br>   CopyTime(Symbol(),PERIOD_D1,UpperFractal_2,1,UpFractalTime_2);<br>   CopyTime(Symbol(),PERIOD_D1,LowerFractal_2,1,LowFractalTime_2);<br>``` |
| ```<br>// Step 2. Determining the generation time of the next day bar:<br>//--- determining the generation time of the next day bar (the stop time for CopyHigh(), CopyLow() and CopyTime())<br>   datetime UpFractalTime_1_15=UpFractalTime_1[0]+86400;<br>   datetime UpFractalTime_2_15=UpFractalTime_2[0]+86400;<br>   datetime LowFractalTime_1_15=LowFractalTime_1[0]+86400;<br>   datetime LowFractalTime_2_15=LowFractalTime_2[0]+86400;<br>``` |
| ```<br>// Step 3. Declaring and filling the arrays for storing the price and time values for the 15-minute timeframe:   <br>//--- declaring the arrays for storing the maximum and minimum price values<br>   double High_1_15[],Low_1_15[],High_2_15[],Low_2_15[];<br>//--- filling the arrays with the CopyHigh() and CopyLow() functions<br>   CopyHigh(Symbol(),PERIOD_M15,UpFractalTime_1[0],UpFractalTime_1_15,High_1_15);<br>   CopyHigh(Symbol(),PERIOD_M15,UpFractalTime_2[0],UpFractalTime_2_15,High_2_15);<br>   CopyLow(Symbol(),PERIOD_M15,LowFractalTime_1[0],LowFractalTime_1_15,Low_1_15);<br>   CopyLow(Symbol(),PERIOD_M15,LowFractalTime_2[0],LowFractalTime_2_15,Low_2_15);<br>//--- declaring the arrays for storing the time values corresponding to the extreme bar indexes  <br>   datetime High_1_15_time[],High_2_15_time[],Low_1_15_time[],Low_2_15_time[];<br>//--- filling the arrays<br>   CopyTime(Symbol(),PERIOD_M15,UpFractalTime_1[0],UpFractalTime_1_15,High_1_15_time);<br>   CopyTime(Symbol(),PERIOD_M15,UpFractalTime_2[0],UpFractalTime_2_15,High_2_15_time);<br>   CopyTime(Symbol(),PERIOD_M15,LowFractalTime_1[0],LowFractalTime_1_15,Low_1_15_time);<br>   CopyTime(Symbol(),PERIOD_M15,LowFractalTime_2[0],LowFractalTime_2_15,Low_2_15_time);<br>``` |
| ```<br>// Step 4. Finding the lowest and highest price values, and the time values of the clarified extreme points:<br>//--- determining the highest and lowest price and time values with the ArrayMaximum() and ArrayMinimum() functions<br>   int Max_M15_1=ArrayMaximum(High_1_15,0,96);<br>   int Max_M15_2=ArrayMaximum(High_2_15,0,96);<br>   int Min_M15_1=ArrayMinimum(Low_1_15,0,96);<br>   int Min_M15_2=ArrayMinimum(Low_2_15,0,96);<br>``` |

Eventually, we have determined the following trend line coordinates:

1\. For the support line:

| MQL4 | MQL5 |
| --- | --- |
| 1. First time coordinate -  LowestBarTime\_2\_m15;<br>2. First price coordinate  - LowerPrice\_2\_m15\[LowestPrice\_2\_m15\];<br>3. Second time coordinate  - LowestBarTime\_1\_m15;<br>4. Second price coordinate  - LowerPrice\_1\_m15\[LowestPrice\_1\_m15\]. | 1. First time coordinate - Low\_2\_15\_time\[Min\_M15\_2\];<br>2. First price coordinate - Low\_2\_15\[Min\_M15\_2\];<br>3. Second time coordinate - Low\_1\_15\_time\[Min\_M15\_1\];<br>4. Second price coordinate - Low\_1\_15\[Min\_M15\_1\]. |

2\. For the resistance line:

| MQL4 | MQL5 |
| --- | --- |
| 1. First time coordinate -  HighestBarTime\_2\_m15;<br>2. First price coordinate  - UpperPrice\_2\_m15\[HighestPrice\_2\_m15\];<br>3. Second time coordinate  - HighestBarTime\_1\_m15;<br>4. Second price coordinate  - UpperPrice\_1\_m15\[HighestPrice\_1\_m15\]. | 1. First time coordinate - High\_2\_15\_time\[Max\_M15\_2\];<br>2. First price coordinate - High\_2\_15\[Max\_M15\_2\];<br>3. Second time coordinate - High\_1\_15\_time\[Max\_M15\_1\];<br>4. Second price coordinate - High\_1\_15\[Max\_M15\_1\]. |

### 4\. Creating objects and editing their properties. Redrawing the lines

Now, when we know the coordinates of the line, we only need to create the graphical objects:

| MQL4 |
| --- |
| ```<br>//--- creating the support line<br>   ObjectCreate(0,"TL_Support",OBJ_TREND,0,LowestBarTime_2_m15,LowerPrice_2_m15[LowestPrice_2_m15],<br>                LowestBarTime_1_m15,LowerPrice_1_m15[LowestPrice_1_m15]);<br>   ObjectSet("TL_Support",OBJPROP_COLOR,Support_Color);<br>   ObjectSet("TL_Support",OBJPROP_STYLE,Support_Style);<br>   ObjectSet("TL_Support",OBJPROP_WIDTH,Support_Width);<br>//--- creating the resistance line<br>   ObjectCreate(0,"TL_Resistance",OBJ_TREND,0,HighestBarTime_2_m15,UpperPrice_2_m15[HighestPrice_2_m15],<br>                HighestBarTime_1_m15,UpperPrice_1_m15[HighestPrice_1_m15]);<br>   ObjectSet("TL_Resistance",OBJPROP_COLOR,Resistance_Color);<br>   ObjectSet("TL_Resistance",OBJPROP_STYLE,Resistance_Style);<br>   ObjectSet("TL_Resistance",OBJPROP_WIDTH,Resistance_Width);<br>``` |

| MQL5 |
| --- |
| ```<br>//--- creating the support line<br>   ObjectCreate(0,"TL_Support",OBJ_TREND,0,Low_2_15_time[Min_M15_2],Low_2_15[Min_M15_2],Low_1_15_time[Min_M15_1],Low_1_15[Min_M15_1]);<br>   ObjectSetInteger(0,"TL_Support",OBJPROP_RAY_RIGHT,true);<br>   ObjectSetInteger(0,"TL_Support",OBJPROP_COLOR,Support_Color);<br>   ObjectSetInteger(0,"TL_Support",OBJPROP_STYLE,Support_Style);<br>   ObjectSetInteger(0,"TL_Support",OBJPROP_WIDTH,Support_Width);<br>//--- creating the resistance line<br>   ObjectCreate(0,"TL_Resistance",OBJ_TREND,0,High_2_15_time[Max_M15_2],High_2_15[Max_M15_2],High_1_15_time[Max_M15_1],High_1_15[Max_M15_1]);<br>   ObjectSetInteger(0,"TL_Resistance",OBJPROP_RAY_RIGHT,true);<br>   ObjectSetInteger(0,"TL_Resistance",OBJPROP_COLOR,Resistance_Color);<br>   ObjectSetInteger(0,"TL_Resistance",OBJPROP_STYLE,Resistance_Style);<br>   ObjectSetInteger(0,"TL_Resistance",OBJPROP_WIDTH,Resistance_Width);<br>``` |

So I created the necessary lines and specified their parameters based on the input parameters.

Now we need to implement redrawing of the trend lines.

When the market situation changes, for example, when a new extreme point appears, we can just remove the existing line:

| MQL4 |
| --- |
| ```<br>//--- redrawing the support line<br>//--- writing the values of the support line time coordinates into the variables<br>   datetime TL_TimeLow2=ObjectGet("TL_Support",OBJPROP_TIME2);<br>   datetime TL_TimeLow1=ObjectGet("TL_Support",OBJPROP_TIME1);<br>//--- if the line coordinates don't match the current coordinates<br>   if(TL_TimeLow2!=LowestBarTime_1_m15 && TL_TimeLow1!=LowestBarTime_2_m15)<br>     {<br>      //--- remove the line<br>      ObjectDelete(0,"TL_Support");<br>     }<br>//--- redrawing the resistance line<br>//--- writing the values of the resistance line time coordinates into the variables<br>   datetime TL_TimeUp2=ObjectGet("TL_Resistance",OBJPROP_TIME2);<br>   datetime TL_TimeUp1=ObjectGet("TL_Resistance",OBJPROP_TIME1);<br>//--- if the line coordinates don't match the current coordinates<br>   if(TL_TimeUp2!=HighestBarTime_1_m15 && TL_TimeUp1!=HighestBarTime_2_m15)<br>     {<br>      //--- remove the line<br>      ObjectDelete(0,"TL_Resistance");<br>     }<br>``` |

| MQL5 |
| --- |
| ```<br>//--- redrawing the support line<br>//--- writing the values of the support line time coordinates into the variables<br>   datetime TL_TimeLow2=(datetime)ObjectGetInteger(0,"TL_Support",OBJPROP_TIME,0);<br>   datetime TL_TimeLow1=(datetime)ObjectGetInteger(0,"TL_Support",OBJPROP_TIME,1);<br>//--- if the line coordinates don't match the current coordinates<br>   if(TL_TimeLow2!=Low_2_15_time[Min_M15_2] && TL_TimeLow1!=Low_1_15_time[Min_M15_1])<br>     {<br>      //--- remove the line<br>      ObjectDelete(0,"TL_Support");<br>     }<br>//--- redrawing the resistance line<br>//--- writing the values of the resistance line time coordinates into the variables<br>   datetime TL_TimeUp2=(datetime)ObjectGetInteger(0,"TL_Resistance",OBJPROP_TIME,0);<br>   datetime TL_TimeUp1=(datetime)ObjectGetInteger(0,"TL_Resistance",OBJPROP_TIME,1);<br>//--- if the line coordinates don't match the current coordinates<br>   if(TL_TimeUp2!=High_2_15_time[Max_M15_2] && TL_TimeUp1!=High_1_15_time[Max_M15_1])<br>     {<br>      //--- remove the line<br>      ObjectDelete(0,"TL_Resistance");<br>     }<br>``` |

### 5\. Checking the bars history loading

During the testing, I realized that the lines didn't always draw correctly.

At first, I thought there was a bug in the code or my solution didn't work at all, but then I realized that the problem was caused by insufficient loading of the bar history on a smaller timeframe, M15 in my case. To warn the user about these issues, I decided to make the program additionally check if a bar exists on M15.

For this purpose, in MQL4 I used the [iBarShift()](https://docs.mql4.com/en/series/ibarshift) function capabilities which I originally used in the section ["Determining price and time values of fractals"](https://www.mql5.com/en/articles/1201#c3).

If a bar is not found, the [iBarShift()](https://docs.mql4.com/en/series/ibarshift) function returns -1. Therefore, we can output this warning:

| MQL4 |
| --- |
| ```<br>//--- checking the bars history loading<br>//--- if at least one bar is not found on M15<br>   if(UpperFractal_1_m15==-1 || UpperFractal_2_m15==-1<br>      || LowerFractal_1_m15==-1 || LowerFractal_2_m15==-1)<br>     {<br>      Alert("The loaded history is insufficient for the correct work!");<br>     }<br>``` |

In MQL5 I used the [Bars()](https://www.mql5.com/ru/docs/series/bars) function which returns an empty value, if the timeseries data haven't been generated in the terminal:

|  |
| --- |
| ```<br>//--- checking the bars history loading<br>//--- 1. determining the number of bars on a specified timeframe<br>   int High_M15_1=Bars(Symbol(),PERIOD_M15,UpFractalTime_1[0],UpFractalTime_1_15);<br>   int High_M15_2=Bars(Symbol(),PERIOD_M15,UpFractalTime_2[0],UpFractalTime_2_15);<br>   int Low_M15_1=Bars(Symbol(),PERIOD_M15,LowFractalTime_1[0],LowFractalTime_1_15);<br>   int Low_M15_2=Bars(Symbol(),PERIOD_M15,LowFractalTime_2[0],LowFractalTime_2_15);<br>//--- 2. check if the loaded history is insufficient for the correct line drawing<br>//--- if at least one bar is not found<br>   if(High_M15_1==0 || High_M15_2==0 || Low_M15_1==0 || Low_M15_2==0)<br>     {<br>      Alert("The loaded history is insufficient for the correct work!");<br>     }<br>``` |

### 6\. Signals of trend lines breakthroughs, push notifications

To complete the picture, I decided to implement a signal of the trend line breakthrough. The trend line is plotted through the extreme points of the day timeframe, but to identify the breakthrough earlier, the bar must be closed lower or higher than the trend line on H4.

In general, we can break the process into three steps:

1. Determine the bar closing price and the trend line price;
2. Determine the conditions in which the price breaks through the trend line;
3. Send the push notification about the breakthrough.

| MQL4 |
| --- |
| ```<br>// 1. Getting the price parameters of the trend line <br>//--- determining the closing price of a bar with index 1<br>   double Price_Close_H4=iClose(NULL,240,1);<br>//--- determining the time of a bar with index 1<br>   datetime Time_Close_H4=iTime(NULL,240,1);<br>//--- determining the bar index on H4<br>   int Bar_Close_H4=iBarShift(NULL,240,Time_Close_H4);<br>//--- determining the price of the line on H4<br>   double Price_Resistance_H4=ObjectGetValueByShift("TL_Resistance",Bar_Close_H4);<br>//--- determining the price of the line on H4   <br>   double Price_Support_H4=ObjectGetValueByShift("TL_Support",Bar_Close_H4);<br>``` |
| ```<br>// 2. Conditions for trend line breakthroughs<br>//--- for breaking through the support line<br>   bool breakdown=(Price_Close_H4<Price_Support_H4);<br>//--- for braking through the resistance line<br>   bool breakup=(Price_Close_H4>Price_Resistance_H4);<br>``` |
| ```<br>// 3. Delivering the push notifications<br>   if(breakdown==true)<br>     {<br>      //--- send no more than one notification per 4 hours<br>      int SleepMinutes=240;<br>      static int LastTime=0;<br>      if(TimeCurrent()>LastTime+SleepMinutes*60)<br>        {<br>         LastTime=TimeCurrent();<br>         SendNotification(Symbol()+"The price has broken through the support line");<br>        }<br>     }<br>   if(breakup==true)<br>     {<br>      //--- send no more than one notification per 4 hours<br>      SleepMinutes=240;<br>      LastTime=0;<br>      if(TimeCurrent()>LastTime+SleepMinutes*60)<br>        {<br>         LastTime=TimeCurrent();<br>         SendNotification(Symbol()+"The price has broken through the resistance line");<br>        }<br>     }<br>``` |

| MQL5 |
| --- |
| ```<br>// 1. Getting the price parameters of the trend line<br>   double Close[];<br>   CopyClose(Symbol(),PERIOD_H4,TimeCurrent(),10,Close);<br>//--- setting the array indexing order<br>   ArraySetAsSeries(Close,true);<br>//---<br>   datetime Close_time[];<br>   CopyTime(Symbol(),PERIOD_H4,TimeCurrent(),10,Close_time);<br>//--- setting the array indexing order<br>   ArraySetAsSeries(Close_time,true);<br>//---<br>   double Price_Support_H4=ObjectGetValueByTime(0,"TL_Support",Close_time[1]);<br>   double Price_Resistance_H4=ObjectGetValueByTime(0,"TL_Resistance",Close_time[1]);<br>``` |
| ```<br>// 2. Conditions for trend line breakthroughs<br>   bool breakdown=(Close[1]<Price_Support_H4);<br>   bool breakup=(Close[1]>Price_Resistance_H4);<br>``` |
| ```<br>// 3. Delivering the push notifications<br>   if(breakdown==true)<br>     {<br>      //--- send no more than one notification per 4 hours<br>      int SleepMinutes=240;<br>      static int LastTime=0;<br>      if(TimeCurrent()>LastTime+SleepMinutes*60)<br>        {<br>         LastTime=(int)TimeCurrent();<br>         SendNotification(Symbol()+"The price has broken through the support line");<br>        }<br>     }<br>   if(breakup==true)<br>     {<br>      //--- send no more than one notification per 4 hours<br>      int SleepMinutes=240;<br>      static int LastTime=0;<br>      if(TimeCurrent()>LastTime+SleepMinutes*60)<br>        {<br>         LastTime=(int)TimeCurrent();<br>         SendNotification(Symbol()+"The price has broken through the resistance line");<br>        }<br>     }<br>``` |

To identify a breakthrough, I used the [ObjectGetValueByShift()](https://docs.mql4.com/en/objects/objectgetvaluebyshift) function in MQL4 and the [ObjectGetValueByTime()](https://docs.mql4.com/en/objects/objectgetvaluebyshift) function in MQL5.

Perhaps, I could just set 1 instead of Bar\_Close\_H4 as a parameter for [ObjectGetValueByShift()](https://docs.mql4.com/en/objects/objectgetvaluebyshift), but I decided to determine the index on H4 first. I used the solution for limiting the number of sent messages published on [this forum thread](https://www.mql5.com/ru/forum/109093), and I would like to thank the author very much.

### 7\. Practical use of trend lines in trading

The most simple way: identify a breakthrough, wait for a pullback and enter the market after it.

Ideally, you should get something like this:

![Fig. 3. Trend line breakthrough](https://c.mql5.com/2/18/Fig3_TrendLine_breakdown.png)

Fig. 3. Trend line breakthrough

You can then use your imagination and try to identify the formations, i.e. the patterns of technical analysis, for example, a triangle:

![Fig.4. Triangle pattern](https://c.mql5.com/2/18/Fig4_Triangle_pattern.png)

Fig.4. Triangle pattern

The lines haven't been clarified by a smaller timeframe on the images above.

### Conclusion

This concludes the article, I hope you will find it useful. The article was intended for beginners in programming, for amateurs like me.

I learned a lot when writing this article: first, I started to make more meaningful code comments; second, at the beginning I had a more cumbersome and complex solution for extreme points clarification, but then I came up with a more simple solution which I demonstrated here.

Thank you for reading, any feedback is appreciated.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1201](https://www.mql5.com/ru/articles/1201)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1201.zip "Download all attachments in the single ZIP archive")

[trendlines.mq4](https://www.mql5.com/en/articles/download/1201/trendlines.mq4 "Download trendlines.mq4")(19.87 KB)

[trendlines.mq5](https://www.mql5.com/en/articles/download/1201/trendlines.mq5 "Download trendlines.mq5")(20.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Movement continuation model - searching on the chart and execution statistics](https://www.mql5.com/en/articles/4222)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/58407)**
(29)


![Almat Kaldybay](https://c.mql5.com/avatar/2018/3/5AB77329-242E.jpg)

**[Almat Kaldybay](https://www.mql5.com/en/users/almat)**
\|
12 May 2020 at 09:48

**fxalert:**

Hi - I cannot pretend to understand all of the coding, but I am working my way through it as it does exactly what I have been trying to code myself (badly as very much a novice).

I have recreated the EA and successfully complied in MQL4 and on first initialisation it will draw the support and resistance lines and will send the notifications but it doesn't redraw NEW support and resistance lines when NEW extreme point fractals appear- is it supposed too? Is there something I am missing something.

Also I couldn't get the fractals to show on the chart but testing the EA in [strategy tester](https://www.mql5.com/en/articles/239 "Article \"The Fundamentals of Testing in MetaTrader 5\"") the fractals would correctly appear after stopping the test?? Any ideas what I a missing again?

Genius work by the way, fractals and trend lines against the extreme points is not easy to explain let alone code it :)

Best Regards

Andy

Hi, about first question: if it doesn't redraw new trendlines maybe you don't used code, which delete created objects. Look 4-th part of article. About second question: you can create new template with fractals for tester and called it "tester.tpl". Thereafter fractals will always show on the chart when you use tester

![fxalert](https://c.mql5.com/avatar/avatar_na2.png)

**[fxalert](https://www.mql5.com/en/users/fxalert)**
\|
12 May 2020 at 10:57

**Almat Kaldybay:**

Hi, about first question: if it doesn't redraw new trendlines maybe you don't used code, which delete created objects. Look 4-th part of article. About second question: you can create new template with fractals for tester and called it "tester.tpl". Thereafter fractals will always show on the chart when you use tester

Thanks for the reply, I left the EA attached to two charts overnight (1x 4hr chart and 1x 15m=M chart) and it has redrawn them so my bad for little patience.

Thanks for the tip regarding tester.pl.

Many thanks

![Konstantin Seredkin](https://c.mql5.com/avatar/2017/3/58BD719C-0537.png)

**[Konstantin Seredkin](https://www.mql5.com/en/users/tramloyr)**
\|
24 May 2020 at 08:38

In order to correctly work drawing lines in the tester and they do not stick to the first point in time, you need to put the control of lines from the bottom above the creation of trend lines

[![](https://c.mql5.com/3/320/2020-05-24_16h35_39__1.png)](https://c.mql5.com/3/320/2020-05-24_16h35_39.png "https://c.mql5.com/3/320/2020-05-24_16h35_39.png")

![Zeke Yaeger](https://c.mql5.com/avatar/2022/6/629E37C1-8BFC.jpg)

**[Zeke Yaeger](https://www.mql5.com/en/users/ozymandias_vr12)**
\|
27 Aug 2020 at 03:41

Hi,

Thank you for your article, I will explore the idea and try to adapt it for my EA's.

Thanks again and happy trading.


![Sergei Naumov](https://c.mql5.com/avatar/avatar_na2.png)

**[Sergei Naumov](https://www.mql5.com/en/users/snov)**
\|
21 May 2022 at 07:24

**Сергей Дыбленко [#](https://www.mql5.com/ru/forum/57988/page2#comment_13923212):**

Neither as an Expert Advisor nor as an indicator in MQL4 it does not show any signs at all!!!!!!!!!!!!!!!!!!!. Is it such a fun to upload anything and a lot just to raise your rating?

![](https://c.mql5.com/3/387/2213267885451.png)

Let's add OnTick function after initialisation to make it work on weekends and in general as soon as you stick it on the chart, not when new prices arrive.

![Tips for Purchasing a Product on the Market. Step-By-Step Guide](https://c.mql5.com/2/18/metatrader-market.png)[Tips for Purchasing a Product on the Market. Step-By-Step Guide](https://www.mql5.com/en/articles/1776)

This step-by-step guide provides tips and tricks for better understanding and searching for a required product. The article makes an attempt to puzzle out different methods of searching for an appropriate product, sorting out unwanted products, determining product efficiency and essentiality for you.

![Trading Ideas Based on Prices Direction and Movement Speed](https://c.mql5.com/2/18/zbm4cy.png)[Trading Ideas Based on Prices Direction and Movement Speed](https://www.mql5.com/en/articles/1747)

The article provides a review of an idea based on the analysis of prices' movement direction and their speed. We have performed its formalization in the MQL4 language presented as an expert advisor to explore viability of the strategy being under consideration. We also determine the best parameters via check, examination and optimization of an example given in the article.

![MQL5 Cookbook: Implementing an Associative Array or a Dictionary for Quick Data Access](https://c.mql5.com/2/18/MQL5_Associative_Arrays__1.png)[MQL5 Cookbook: Implementing an Associative Array or a Dictionary for Quick Data Access](https://www.mql5.com/en/articles/1334)

This article describes a special algorithm allowing to gain access to elements by their unique keys. Any base data type can be used as a key. For example it may be represented as a string or an integer variable. Such data container is commonly referred to as a dictionary or an associative array. It provides easier and more efficient way of problem solving.

![Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal API, Part 2](https://c.mql5.com/2/17/HedgeTerminalaArticle200x200_2p2.png)[Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal API, Part 2](https://www.mql5.com/en/articles/1316)

This article describes a new approach to hedging of positions and draws the line in the debates between users of MetaTrader 4 and MetaTrader 5 about this matter. It is a continuation of the first part: "Bi-Directional Trading and Hedging of Positions in MetaTrader 5 Using the HedgeTerminal Panel, Part 1". In the second part, we discuss integration of custom Expert Advisors with HedgeTerminalAPI, which is a special visualization library designed for bi-directional trading in a comfortable software environment providing tools for convenient position management.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ynfeilvlzjdeezljhoibofdptvgiujpa&ssn=1769179181662595726&ssn_dr=0&ssn_sr=0&fv_date=1769179181&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1201&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Plotting%20trend%20lines%20based%20on%20fractals%20using%20MQL4%20and%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691791811441707&fz_uniq=5068497971579910700&sv=2552)

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