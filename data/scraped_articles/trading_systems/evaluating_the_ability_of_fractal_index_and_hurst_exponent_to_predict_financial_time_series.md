---
title: Evaluating the ability of Fractal index and Hurst exponent to predict financial time series
url: https://www.mql5.com/en/articles/6834
categories: Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:52:56.191928
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/6834&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083182198477035142)

MetaTrader 5 / Trading systems


### Introduction

The modern financial market is an example of a "natural" complexly balanced system. On the one hand, the market is pretty chaotic, because it
is influenced by a large number of participants. On the other hand, the market is characterized by definite stable processes, which are
determined by the market participants' actions. One of econophysics tasks concerns the description of social interaction processes,
which form the price dynamics observed on the exchange. Therefore, it is highly desirable to define and present specific properties of
financial time series, which will distinguish such data from other natural processes. In modern theories, price series are defined as
different-scale fractals (from several minutes to dozens of years).

They show a much more complicated behavior, than many model and natural processes \[3\]. One of the tools for finding out the details of such
behavior is the numerical analysis of the series, the purpose of which is to study the dynamics of the series. The typical algorithms for the
reliable evaluation of fractal dimension require large datasets (about 10,000-100,000 samples), which characterize a series over a long
time interval, during which the behavior can change, and sometimes it can change repeatedly. For real trading tasks, we need methods to
determine the local fractal characteristics of a series. In this article, we will discuss and demonstrate a method for determining the
fractal dimension of the series of price sequences, using the numerical method described in \[1, 2\].

### The concept of fractal dimension and statistical properties of time series

The fractal dimension estimates how the data set takes up space. There are many methods for estimating the fractal dimension. Their common
feature is that volume or area are calculated in the space, in which this set is located. Let us use the example of the time series for the
financial instrument, which consists of Close prices {Close(t)}. If the {Close(t)} series levels are independent, there are no clear
trends on the symbol chart, while the behavior will be similar to the "white noise". The value of the fractal dimension

**_D_** will be close to the value of the topological dimension of the plane, in other words **_D->2_**.
If the {Close(t)} series levels are not independent, the D value will be significantly less than 2, which indicates that the time series has a
"memory", i.e. upward and downward trends will be observed at some time intervals, alternating with the undefined periods (Fig. 1).

![](https://c.mql5.com/2/36/ip31__1.png)

Fig.1. An example of a random series and a series with a trend, and the corresponding
fractal dimension

### Fractal dimension evaluation methods and their features

There are different methods for calculating the fractal dimension of a time series. Let us consider the evaluation method utilizing the Hurst
exponent.

The Hurst exponent H is determined based on the following equation

![Hurst coefficient definition](https://c.mql5.com/2/36/b4d2a.png)
                              (1a)

where the angle brackets indicate time averaging. The relationship of the Hurst exponent with the fractal
dimension is obtained by the method of normalized range or by the R/S analysis based on the following equations

> > **_DH = 2-H_**
> >
> > **H = log(R/ S) / log(N / 2)**(1b)

where **R**— max {Close(t)} - min {Close(t)},  i = 1..N is the range of the Close(t) series deviations,  **S** — is the
standard deviation of Close(t) values. The method was described in more detail in the article

[Calculating the Hurst exponent](https://www.mql5.com/en/articles/2930) by Dmitry Piskarev.

If the Hurst exponent for the time series is in the range between 0.5 and 1, such a series is considered to be persistent, or trend-resistant,
which means that the {Close(t)} series is not random, contains a trend and the behavior of the series can be predicted with a good enough
accuracy. The closer the

**H** value to 1, the greater is the correlation between the {Close(t)} series values.

The disadvantage of this method is that a large amount of data (thousands of data series values) is needed in order to obtain a reliable
estimate of the Hurst exponent, otherwise the estimates obtained may be incorrect. In addition, the series values must have a normal
distribution law, which is not always the case. Since the reliable calculation of both DH and H requires a large representative sample of a
large data amount, the series behavior can repeatedly change during the relevant long trading period. In order to link the local dynamics of
the analyzed process with the fractal dimension of the observed series, we need to locally determine the

**_D_** dimension.

### Fractal dimension estimation based on minimum covered area

A more efficient method when forecasting econometric series is the one based on the calculation of the minimum coverage dimension
![](https://c.mql5.com/2/36/weh2c.png)\[1, 2\]. In 1919,
Hausdorff suggested the following formula for determining a fractal:

![Hausdorff Fractal](https://c.mql5.com/2/36/lg41b.png).

where![](https://c.mql5.com/2/36/glk1z.png)
is the lowest number of balls of radius
![](https://c.mql5.com/2/36/wq72d_small.png),
which cover this set. Note that if the original set is in Euclidean space, any other simple shapes (such as cells) can be used for the
set approximation with the geometric factor ![](https://c.mql5.com/2/36/wq72d_small__1.png)
instead of covering the set using balls.

For example, the _f(t)_ function is set in the \[a, b\] interval. Let us evenly split the interval wm = \[a=t0<t1<t2...tm=b\], while the
scope of split is defined as
![TimeScale](https://c.mql5.com/2/36/abi3a.png)
If we cover these sets using, for example, cells sized
![delta](https://c.mql5.com/2/36/j6e2d_small.png),
then if the factor
![delta](https://c.mql5.com/2/36/9cv2d_small.png) is
decreased, the number of cells

**_N_** will increase according to the power law:

![Nlaw](https://c.mql5.com/2/36/ede1a.png)

where **_D_** is the fractal dimension.

When determining the **_D_** dimension using the cells method, the surface in which the time series graph is located is
divided into cells of size
![delta](https://c.mql5.com/2/36/j6e2d_small__1.png),
and then a calculation is performed to count the number of cells

**_N(_**
**_![delta](https://c.mql5.com/2/36/j6e2d_small__1.png))_**,
to which at least one point of this graph belongs. Then
![delta](https://c.mql5.com/2/36/j6e2d_small__1.png) changes
and the

**_N(_**
**_![delta](https://c.mql5.com/2/36/j6e2d_small__1.png))_**
function graph is plotted in the double logarithmic state. Further, the resulting set of points is approximated using the least square (LS) method. **_D_**
is determined based on the line slope.

The minimum coverage area of the function graph at this scale, in the **_\[a, b\]_** interval will be equal to the
sum of areas of the rectangles with the base
![delta](https://c.mql5.com/2/36/j6e2d_small__5.png) and
the hight equal to the variation
![MaxMinValue](https://c.mql5.com/2/36/csc2f.png) — the
difference between the maximum and minimum of the

_**f(t)**_ function at each **_\[ti-1, ti\]_** interval. The minimum coverage area
![MinSquare](https://c.mql5.com/2/36/c5j2e.png) can
be calculated using the following formula:

![SquareMinOverload](https://c.mql5.com/2/36/4gw2g.png)
                  (2)

where
![AmplSum](https://c.mql5.com/2/36/e1m2h.png)is
the sum of amplitude variations of function

_**f(t)**_ in the _**\[a, b\]**_ interval. The estimate
![AmplSum](https://c.mql5.com/2/36/e1m2h.png) depends
on the selected magnitude. The smaller
![deltamin](https://c.mql5.com/2/36/xo72d.png),
the more accurate the calculation of
![AmplSum](https://c.mql5.com/2/36/e1m2h.png).
In this case, the value
![AmplSum](https://c.mql5.com/2/36/e1m2h.png)changes
according to a power lay when
![deltamin](https://c.mql5.com/2/36/xo72d__1.png)changes:

![V_Ampl_law](https://c.mql5.com/2/36/o9a4b.png)

(3)

where
![muValue](https://c.mql5.com/2/36/2zs4c.png).
The value
![DimMinCover](https://c.mql5.com/2/36/pbz4e.png) is
called "Dimension of the minimal cover", while index
![mu](https://c.mql5.com/2/36/22y4d.png)is
referred to as the fractal index.

The dependence of the minimal cover area from different
![deltamin](https://c.mql5.com/2/36/xo72d__2.png)values
for the time series consisting of 32 observations is shown in Fig. 2.

![Calculate cover](https://c.mql5.com/2/36/s6z4_ruqen.png)

Fig. 2. Calculating cover area with various values
![deltamin](https://c.mql5.com/2/36/xo72d__3.png)

Reference \[2\] states that the fractal dimension which is calculated using the cell covering and covering with
rectangle, based on the function variation, coincide. An important property of the algorithm which uses function variations, is its much
faster convergence, which allows determining of the time series fractal dimension value locally, using a small set of values.

Applying a logarithm to (3), we obtain the following:

![MuEquation](https://c.mql5.com/2/36/0gv5a.png)

                    (4)

To determine
![DimMinCover](https://c.mql5.com/2/36/pbz4e.png),
a dependence (3) chart is plotted in double logarithmic coordinates using the least squares (LS) method, and then the tangent of the
straight line angle is determined. Based on expression (4), calculate
![mu](https://c.mql5.com/2/36/22y4d.png),
the fractal index, which is the local characteristic of the time series. As is shown in reference \[1\], the
![](https://c.mql5.com/2/36/g864e.png)determining accuracy is
much higher that the accuracy of determining of other fractal characteristics, such as the cellular dimension
![](https://c.mql5.com/2/36/c9h4_c.png) or the dimension
calculated based on the Hurst exponent. In addition, the method has no limitations on the distribution of series
![](https://c.mql5.com/2/36/niw2c.png). Reference \[1\]
also shows that a reliable estimate can be obtained if the time series
![](https://c.mql5.com/2/36/niw2c__1.png) includes no less than
32 observations. Normally, financial sets have a much longer history. This approach enables the use of the fractal index as a function of
time
![mu(t)](https://c.mql5.com/2/36/zgq4e.png),
in which each value is determined based on the previous 32 values of the time series
![](https://c.mql5.com/2/36/niw2c__1.png).

Fig. 3 shows the example of calculation of fractal index
![mu](https://c.mql5.com/2/36/22y4d.png) based
on the angle of the approximating straight line. According to the figure, the coefficient of determination of the regression equation

**_R_**
**_2_**, which approximates the dependence, is equal to 0.96 — this indicates that the fractal index of 0.4544 is calculated
quite accurately for a fragment of a series of 32 points.

![Dependence in double log coordinates](https://c.mql5.com/2/36/inu5m.png)

Fig. 3. Approximation of dependence
![lnV(delta)](https://c.mql5.com/2/36/y2f5c.png) in
double logarithmic coordinates and determination of the fractal index

The fractal dimension can be evaluated using either the cell dimension method or the Hurst index. As an example,
let us consider Lukoil stock quotes (MICEX) before the crisis, which happened at the beginning of the century. This time can be interpreted
as a stable trend with a gradual increase (persistent series). Fig. 4 shows the results of the fractal dimension evaluation in 1999.

![FractalDimensionLHOL](https://c.mql5.com/2/36/4be6.png)

Fig. 4. a) LS approximation of the fractal measure using the cell covering (D=1.1894),
b) Log-log plot of the numerical estimate of the Hurst parameter (D=1.6)

The fractal dimension of the series **_D_** = 1.18 points to its persistent trendy nature. A value close to one indicates
the nearing end of the trend, which happened in 2000-2001. Hurst exponent value

**_H_** =0.40. Pay attention to the relatively low coefficient of determination **_R_**
**_2_** **_=_** 0.56 with the confidence interval of 0.95. According to formulas (1), the fractal dimension calculated
by the Hurst exponent is equal to

**_D_** = 1.6, which indicates the random behavior of a series and an increased level of stochasticity. However, this does not
concern Lukoil stocks in the period of 1999.

Another interesting and illustrative example of the fractal index and Hurst exponent estimation accuracy as of local indicators of time series
is provided in reference \[2\]. This parameter assessment is more appropriate for the trading tasks related to market analysis of the
operational qualitative and quantitative behavior of time series. The source price series of Alcoa Inc., including 8145 points, was
divided into 8113 overlapping intervals of 32 days each, shifted relative to each other by one day. The following was used as the calculation
accuracy parameters: the width of the confidence interval 95% for

**_H_** and
![mu](https://c.mql5.com/2/36/22y4d.png),
evaluation of accuracy of real points hitting the theoretical line

**_K = 1- R_**
**_2_**,where **_R_**
**_2_** is the coefficient of determination (of the exactly fall into the line, then **_R_**
**_2_** =1 and **_K_** =0).

The following values were calculated at each of the 8113 intervals:

- **_H_** — Hurst exponent;
- ![mu](https://c.mql5.com/2/36/22y4d.png)—
fractal index;
- ![](https://c.mql5.com/2/36/delta1.png)— width 95 %
of the confidence interval for

**_H_**;
- ![](https://c.mql5.com/2/36/deltaMu.png)— width 95
% of the confidence interval for
![](https://c.mql5.com/2/36/g2l4d.png);
- ![](https://c.mql5.com/2/36/K_H.png)\- the accuracy
of correspondence of experimental and the obtained straight line for

**_H_**;
- ![](https://c.mql5.com/2/36/K_MU.png)\- the
accuracy of correspondence of experimental and the obtained straight line for
![mu](https://c.mql5.com/2/36/22y4d.png).

Typical fragments of graphs of functions
![](https://c.mql5.com/2/36/deltaMuwtn.png),![](https://c.mql5.com/2/36/deltaHktq.png)and

![](https://c.mql5.com/2/36/K_mueto.png),

![](https://c.mql5.com/2/36/K_Hftu.png),
built for the intervals, the right value of which coincides with the time _t_, are shown in Fig. 5a and
5b. It can be seen from these figures, that in most cases index ![mu](https://c.mql5.com/2/36/22y4d.png) is
determined much more accurately, than **_H_**.

![Fig.5a delta_H(t), delta_mu(t)](https://c.mql5.com/2/36/ftb7a.png)

Fig. 5a. Typical fragment of the time
series of width of confidence intervals created based on the series of Close prices for AlcoaInc.

![Fig. 5b. K_h(t), K_mu(t)](https://c.mql5.com/2/36/wmv7b.png)

Fig. 5b.The
corresponding series fragment for the values showing the accuracy of coincidence of experimental points and the theoretical line,built
for the same series

Based on these images, it is possible to conclude that in the overwhelming majority of cases the fractal
index ![mu](https://c.mql5.com/2/36/22y4d.png)is
determined much more accurately than _**H**_.

The main advantage of the index
![mu](https://c.mql5.com/2/36/22y4d.png)in
relation to other fractal indicators (including in particular the Hurst exponent) is that the corresponding
![AmplSum](https://c.mql5.com/2/36/e1m2h.png)value
quickly enters the asymptotic mode. This enables the use of
![mu](https://c.mql5.com/2/36/22y4d.png) as
the local characteristic by determining the dynamics of the initial process, since the order of the scale for its accurate determining
matches that of the main scale of determining process states. Such states include the relative calm periods (flat) and the long term upward
or downward movement periods (trends). An efficient solution for linking value
![mu](https://c.mql5.com/2/36/22y4d.png) with
the series behavior, is to add the function
![](https://c.mql5.com/2/36/8344e.png) as a value
![mu](https://c.mql5.com/2/36/22y4d.png) which
is determined in the minimum interval preceding

**_t_**, in which
![mu](https://c.mql5.com/2/36/22y4d.png) can
still be calculated with acceptable accuracy.

### The correlation of the time series nature and the fractal index

Anyone willing to use an indicator based on the fractal index should know some of its specific features \[2\].

The behavior of the series defines the ![mu](https://c.mql5.com/2/36/22y4d.png)value:

1. **![](https://c.mql5.com/2/36/uzi4e.png)** =
    0.5 indicates random price walk (Wiener process). Investors behave independently and there is no obvious trend in the
    behavior of the price. In this case, we can say that the price has a "normal" stability, because the price is weakly dependent on
    external influences, there is no "feedback" and thus there are no arbitrage opportunities.
2. **![](https://c.mql5.com/2/36/uzi4e.png)** <
    0.5 suggests that the price has a higher stability against external influence, which can be connected with the investors'
    confidence in the relevant company stability and of absence of any new information in the market. In this case, stock prices
    fluctuate within a quite narrow price range. There are still enough sellers when prices grow, as well as there are enough
    buyers when prices fall, and their actions get the prices back to the initial range. "Correlation" in this case is negative and
    it mitigates stock price changes while preserving stable price behavior.
3. **![](https://c.mql5.com/2/36/uzi4e.png)** \>
    0.5 corresponds to reduced price stability. This may indicate the emergence of new information and the reaction to this
    information. It can be assumed that all market participants estimate the incoming information approximately equally, and
    thus a tendency appears in the price movement corresponding to the information received. Under some conditions, this
    situation leads to sharp changes in a stock price.

The fractal index and Hurst exponent are related as
![mu](https://c.mql5.com/2/36/22y4d.png)**_=_**
**_1-H_**, which enables the inheritance of classification variants from chaotic time series:

1. When **_![mu](https://c.mql5.com/2/36/22y4d.png) =_**
**_0.5, H = 0.5_** a time series is the Wiener process ("brown" noise). The main property of the process is the absence of memory: series
    evolution is not connected with previous values.
2. When **_0.5 <_**
**_![mu](https://c.mql5.com/2/36/22y4d.png) <= 1,_**
**_0 <= H < 0.5_**, the process is considered as the "pink" noise. It is characterized by the "negative" memory: if positive
    increment was registered in the past, it will be probably followed by a negative increment, and vice versa.
3. If **_0 <=_**
**_![mu](https://c.mql5.com/2/36/22y4d.png) <_**
**_0.5 , 0.5 < H <=1_**, the time series is a "black" noise with the positive memory: if positive trend occurred in
    the past, it is likely to remain in the future and vice versa.

### Indicator for evaluating Fractal index and Hurst exponent

Successful trading on a scale of days, weeks and months is associated with an understanding of the chaotic state of financial time series. Based
on the stable evaluation of fractal indexes in short data fragments, we can develop an indicator for stocks (the evolution of which is
determined by the will of a large number of people), which will help the trader to identify and forecast financial time series.

The indicator evaluates the fractal index, the confidence interval for it, the values of the coefficient of determination and the Hurst
exponent. The below charts show the aforementioned

![Fractal Index](https://c.mql5.com/2/36/MUut3_2.png),

![](https://c.mql5.com/2/36/deltaMuwtn.png),
and
![](https://c.mql5.com/2/36/K_mueto.png)function graphs.

In the indicator, you can set the length of the time series segment, for which calculation will be performed and the parameter
evaluation window will be provided. Upon the indicator launch, the series is calculated along the Close prices while the window is
shifted by one count. Since the length of the evaluation window (interval) is equal to the power of two, we can obtain a set of values and
evaluate the fractal index by performing the linear approximation of the set.

```
double CFractalIndexLine::CalculateFractalIndex(const double &series[],const int N0,const int N1,
                                                const double hourSampling,int CountFragmentScale=0)
  {
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// series[] - time series
// N0, N1 - the left and right boundary points of the series[] array fragment, based on which the fractal index will be estimated
// hourSampling - discretization between points in HOURS
// CountFragmentScale - the number of requested scales to form a set of points, for which the fractal index is calculated
//
// RESULT
// the fractal index (Mu), the Hurst index (Hurst), the Confidence interval 95% (ConfInterval[2],
// coefficient of determination (R2det) - the closer to 1, the more accurately the calculation points fall on the approximating line
// determining stability for coefficient KR2 = 1-R2det. The closer to zero, the more accurate the calculated value of Mu
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// 1. Load the internal fragment with values from the time series
   LoadFragment(series,N0,N1,hourSampling);

// 2. Determine the number of cycles to determine the points of the approximating line
   int   nn2 = (int)floor(Nfrgm/2);  // Partition limits - no less than two points
   int npow2 = (int)ipow2(nn2);      // The number of the powers of two in the Possible partitioning limit;

   if(CountFragmentScale==0) CountFragmentScale=npow2; // default

   int Count=fmin(CountFragmentScale,npow2);           // limiting the number of variants of series fragment division
   int NumPartDivide;

   for(int i=0; i<=Count; i++)
     {
      NumPartDivide = (int)pow (2,i);      // Number of pieces in the series fragment division
      CalcAmplVariation(NumPartDivide, i); // Calculating a point for the approximating line model
      i=i;
     }
// 4. Evaluation of the Fractal Index and on the limits of the Index confidence intervals
   Mu=fCalculateConfidenceIntervalMU(LogDeltaScales,LogAmplVariations,Count,ConfInterval,R2det);
   Hurst=1-Mu;		// Hurst exponent
   KR2=1-R2det;

   return Mu;
  }
//----------------------------------------------------------------------------------------------------------------------------------

double CFractalIndexLine::CalcAmplVariation(const int NumPartDivide,int idxAmplVar=-1)
  {
// If idxAmplVar=-1, then index in the array is determined automatically (based on the contribution of the power of two in NumPartDivide)
// ALREADY PREVIOUSLY DONE: copying the fragment, setting the time of discretization of the series IN DAYS
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// 1. DETERMINE THE BORDERS OF INTERVALS CORRESPONDING TO THE SPECIFIC NUMBERS
   int nCheckPoint=0,nIntervalPoints=0; // the number of points to check in one division interval
   double dayDeltaScales=BoundaryArray(NumPartDivide,fragment,0,Nfrgm-1,hSampling,Boundaries,nIntervalPoints);

// 2. GO THROUGH INTERVALS TO DETERMINE LIMIT VALUES OF FUNCTIONS AND OF AMPLITUDE VARIATION
   int countInterval=Boundaries.CountNonEmty();
   int  maxFuncIdx=0,minFuncIdx=0;
   double A,V=0.;

   nCheckPoint=(int)(Boundaries.y[0]-Boundaries.x[0])+1;
   for(int i=0; i<countInterval; i++)
     {
      maxFuncIdx = ArrayMaximum(fragment,(int)Boundaries.x[i],nCheckPoint); // INDEX WITH MAX. VALUE
      minFuncIdx = ArrayMinimum(fragment,(int)Boundaries.x[i],nCheckPoint);
      A = fragment[maxFuncIdx] - fragment[minFuncIdx];
      V = V+A;
      i=i;
     }

// 3. ACCUMULATION OF RESULTS IN STORAGE
   if(idxAmplVar==-1) idxAmplVar=ipow2(NumPartDivide); // index in the storage array

   LogDeltaScales   [idxAmplVar] = log(dayDeltaScales); // log-scale of the current division
   LogAmplVariations[idxAmplVar] = log(V);              // log-Amplitude Variation in the current division scale

   return V;
  }
//--------------------------------------------------------------------------------------------------------------------------------------
```

CFragmentIndexLine.mqh file fragments execute loops for the calculation of covering area, as it is shown in Fig.2. The sequence of actions in the program is
explained through detailed comments.

### Demonstration of indicator operation on real data

We call the indicator, requesting the evaluation of 600 days with the evaluation window of 64 points. The result contains 536 values of the
fractal index and it is shown in Fig.6.



![FigGAZP](https://c.mql5.com/2/36/GAZP.png)

Fig.6 Close prices of Gazprom and the fractal index evaluation results

The figure shows the correlation of the index values and the behavior of prices. The blue color of the index graph corresponds to the trend
state of the system, indicates the trend stability and the ability to predict future behavior. Violet color indicates anti-persistence of
the "pink noise" type, which corresponds to "negative" memory and flat. Yellow corresponds to the "Brownian motion", i.e. the movement is
random and cannot be predicted.

### Conclusions

Local fractal analysis can be interesting in trading for the following purposes:

1. Determining of disorder, i.e. of the moment when statistical characteristics of a time series change;
2. Prediction of a time series.

It should be taken into account that the scale for determining index **_![mu](https://c.mql5.com/2/36/22y4d.png)_** with
a suitable accuracy is two orders of magnitude less than a similar scale for calculating the Hurst exponent H. This difference allows
using index

**_![mu](https://c.mql5.com/2/36/22y4d.png)_** as
the local fractal index. That is why it can be considered that index

**_![mu](https://c.mql5.com/2/36/22y4d.png)_** describes
the stability of a time series. Case

**_![mu](https://c.mql5.com/2/36/22y4d.png)_****<0.5** can be interpreted as a trend, and case **_![mu](https://c.mql5.com/2/36/22y4d.png)_ >0.5** can be treated as a flat. **_![mu](https://c.mql5.com/2/36/22y4d.png)_ ~**
**0.5** is considered to be the Brownian motion. Thus, using function **![](https://c.mql5.com/2/36/uzi4e.png)** we
can classify initial price series and provide basis for forecasts.

### List of References

1. Dubovikov M.M., Starchenko N.V. Econophysics and fractal analysis of financial time series
2. Dubovikov M.M., Starchenko N.V. Econophysics and analysis of financial time series // Collected. "ECONOPHYSICS. Modern physics in search
    of economic theory"
3. Peters, J. Chaos and Order in the Capital Markets A New View of Cycles, Prices and Market Volatility

4. [Krivonosova \\
    E.K., Pervadchuk V.P., Krivonosova E.A.Comparison of the fractal characteristics of time series of economic indicators](https://www.mql5.com/go?link=https://www.science-education.ru/pdf/2014/6/701.pdf "https://www.science-education.ru/pdf/2014/6/701.pdf")

5. Starchenko N.V. Local fractal analysis in physical applications.


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6834](https://www.mql5.com/ru/articles/6834)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6834.zip "Download all attachments in the single ZIP archive")

[FRACTAL\_upd.zip](https://www.mql5.com/en/articles/download/6834/fractal_upd.zip "Download FRACTAL_upd.zip")(18.5 KB)

[fFractalSegmentSeriesAnalysis.mqh](https://www.mql5.com/en/articles/download/6834/ffractalsegmentseriesanalysis.mqh "Download fFractalSegmentSeriesAnalysis.mqh")(0.75 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Using MATLAB 2018 computational capabilities in MetaTrader 5](https://www.mql5.com/en/articles/5572)
- [Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://www.mql5.com/en/articles/3172)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/317674)**
(41)


![SME_FX](https://c.mql5.com/avatar/2021/3/603E364A-E28E.png)

**[SME\_FX](https://www.mql5.com/en/users/sme_fx)**
\|
2 Mar 2021 at 13:02

GREAT article, thank you!!


![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
2 Jun 2023 at 11:10

Good afternoon.

Started the indicator, timeframe 1 hour, [symbol EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis"), quotes metaquotes, default settings.

After a few seconds of work, it gives an error array out of range in 'CFractalSeriesSet.mqh' (108,17).

I have this page 108

```
MuIndexes[ii] = IndexCalculater.Mu;
```

I have done the sprinting.

```
16:01:59.441    Fractal Index (EURUSD,H1)       ii = 539
16:01:59.441    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
16:01:59.441    Fractal Index (EURUSD,H1)       ii = 538
16:01:59.441    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
16:01:59.441    Fractal Index (EURUSD,H1)       ii = 537
16:01:59.441    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
16:02:10.682    Fractal Index (EURUSD,H1)       ii = 1610
16:02:10.682    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
16:02:10.685    Fractal Index (EURUSD,H1)       array out of range in 'CFractalSeriesSet.mqh' (108,17)
```

I can't understand why the array out of range occurs. The array size is 1074, the index is 1610, where is the overrun?

And it is strange that the indexes are descending, and does not reach zero becomes 1610, before it was all logical.

```
15:49:38.445    Fractal Index (EURUSD,H1)       ii = 1
15:49:38.445    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
15:49:38.445    Fractal Index (EURUSD,H1)       ii = 0
15:49:38.445    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
15:49:47.403    Fractal Index (EURUSD,H1)       ii = 1073
15:49:47.403    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
15:49:47.403    Fractal Index (EURUSD,H1)       ii = 1072
15:49:47.403    Fractal Index (EURUSD,H1)       MuIndexes.Size() = 1074
```

Can you tell me what could be the reason for this?

How to fix it?

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
2 Jun 2023 at 11:35

**Aleksandr Slavskii [#](https://www.mql5.com/ru/forum/314566/page4#comment_47259289):**

I can't understand why the array is overrun. The array size is 1074, index is 1610, where is the overrun?

You answered yourself

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
2 Jun 2023 at 11:56

**Rashid Umarov [#](https://www.mql5.com/ru/forum/314566/page4#comment_47259435):**

You answered it yourself.

Oops. I see it now.)

For some reason the figure 1610, the brain perceived as 1061 and I am so confused, where is the exit beyond the limits.

![Aleksandr Slavskii](https://c.mql5.com/avatar/2017/4/58E88E5E-2732.jpg)

**[Aleksandr Slavskii](https://www.mql5.com/en/users/s22aa)**
\|
2 Jun 2023 at 13:09

I'm not sure if this is correct (I'm sure it's not), but I added a line to the code.

In the CFractalSeriesSet.mqh file before the line

```
ii = (CircleCount - 1 - i) + IndexCount; // FRESH SCORES ARE STORED AT THE END OF THE ARRAYS
```

before the line

```
IndexCount = IndexCount >= (int)MuIndexes.Size() ? 0 : IndexCount;
```

Now it does not go outside the [array](https://www.mql5.com/en/articles/2555 "Article: What checks a trading robot must pass before publishing in the Marketplace"). But I don't know how it will affect the calculations of the indicator.

![Price velocity measurement methods](https://c.mql5.com/2/36/Article_Logo__1.png)[Price velocity measurement methods](https://www.mql5.com/en/articles/6947)

There are multiple different approaches to market research and analysis. The main ones are technical and fundamental. In technical analysis, traders collect, process and analyze numerical data and parameters related to the market, including prices, volumes, etc. In fundamental analysis, traders analyze events and news affecting the markets directly or indirectly. The article deals with price velocity measurement methods and studies trading strategies based on that methods.

![Library for easy and quick development of MetaTrader programs (part VII): StopLimit order activation events, preparing the functionality for order and position modification events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__2.png)[Library for easy and quick development of MetaTrader programs (part VII): StopLimit order activation events, preparing the functionality for order and position modification events](https://www.mql5.com/en/articles/6482)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the sixth part, we trained the library to work with positions on netting accounts. Here we will implement tracking StopLimit orders activation and prepare the functionality to track order and position modification events.

![Grokking market "memory" through differentiation and entropy analysis](https://c.mql5.com/2/36/snip_20190614154924__2.png)[Grokking market "memory" through differentiation and entropy analysis](https://www.mql5.com/en/articles/6351)

The scope of use of fractional differentiation is wide enough. For example, a differentiated series is usually input into machine learning algorithms. The problem is that it is necessary to display new data in accordance with the available history, which the machine learning model can recognize. In this article we will consider an original approach to time series differentiation. The article additionally contains an example of a self optimizing trading system based on a received differentiated series.

![Library for easy and quick development of MetaTrader programs (part VI): Netting account events](https://c.mql5.com/2/36/MQL5-avatar-doeasy__1.png)[Library for easy and quick development of MetaTrader programs (part VI): Netting account events](https://www.mql5.com/en/articles/6383)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the fifth part of the article series, we created trading event classes and the event collection, from which the events are sent to the base object of the Engine library and the control program chart. In this part, we will let the library to work on netting accounts.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jxmpzmaiixwqbocfcgtcnugrqsehvhak&ssn=1769251975792184502&ssn_dr=0&ssn_sr=0&fv_date=1769251975&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F6834&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Evaluating%20the%20ability%20of%20Fractal%20index%20and%20Hurst%20exponent%20to%20predict%20financial%20time%20series%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925197503555785&fz_uniq=5083182198477035142&sv=2552)

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