---
title: William Blau's Indicators and Trading Systems in MQL5. Part 1: Indicators
url: https://www.mql5.com/en/articles/190
categories: Trading Systems, Indicators
relevance_score: 9
scraped_at: 2026-01-22T17:39:30.489050
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/190&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049283808315943070)

MetaTrader 5 / Trading systems


_Technical trading can only be exploited if good tools are available._

_The tools of a good trader are experience, judgement, and a mathematical hierarchy provided by good trading computer program._ _William Blau_

### Introduction

The first part of the article "Indicators and Trade Systems in MQL5 by William Blau. Part 1: Indicators " is a description of indicators and oscillators, described by William Blau in the book ["Momentum, Direction, and Divergence"](https://www.mql5.com/go?link=https://www.amazon.com/Momentum-Direction-Divergence-Indicators-Technical/dp/0471027294 "http://www.amazon.com/Momentum-Direction-Divergence-Indicators-Technical/dp/0471027294").

The indicators and oscillators, described in this article, are presented as a source codes in MQL5 language and attached in the archive file "Blau\_Indicators\_MQL5\_en.zip".

**The key idea of analysis by William Blau**

The technical analysis by William Blau consists of four phases:

1. Using the price series data (q bars) the indicator is calculated and plotted at chart. _The indicator does not reflect the general trend of the price movement, and does not allow to determine the trend reversal points._
2. The indicator is smoothed several times using the EMA method: the first time (with period r), the second time (with period s), and the third time (with period u); a smoothed indicator is plotted. _A smoothed indicator fairly accurately and reproduces the price fluctuations with a minimum lag. It allows to determine the trend of the price movement and the reversal points and eliminates the price noise._
3. The smoothed indicator is normalized, a normalized smoothed indicator is plotted. _The normalization allows the indicator value to be interpreted as the overbought or oversold states of the market._
4. A normalized smoothed indicator is smoothed once by the EMA method (period ul); an oscillator is constructed - the indicator histogram and the signal line, the levels of overbought and oversold of the market are added. _Oscillator allows us to distinguish the overbought/oversold states of the market, the reveral points and the end of a trend._

### **Indicators**

The article describes the following groups of indicators:

1. Indicators, based on the Momentum:


   - [Momentum](https://www.mql5.com/en/articles/190#Blau_Mtm)(Blau\_Mtm.mq5)
   - [The True Strength Index](https://www.mql5.com/en/articles/190#Blau_TSI) (Blau\_TSI.mq5)
   - [Ergodic Oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic) (Blau\_Ergodic.mq5)
2. Indicators, based on Stochastic:


   - [Stochastic](https://www.mql5.com/en/articles/190#Blau_TStoch) (Blau\_TStoch.mq5)
   - [Stochastics Index](https://www.mql5.com/en/articles/190#Blau_TStochI) (Blau\_TStochI.mq5)
   - [Stochastic Oscillator](https://www.mql5.com/en/articles/190#Blau_TS_Stochastic) (Blau\_TS\_Stochastic.mq5)
3. Indicators, based on the Stochastic Momentum:


   - [Stochastic Momentum](https://www.mql5.com/en/articles/190#Blau_SM) (Blau\_SM.mq5)
   - [Stochastic Momentum Indicator](https://www.mql5.com/en/articles/190#Blau_SMI) (Blau\_SMI.mq5)
   - [Stochastic Momentum Oscillator](https://www.mql5.com/en/articles/190#Blau_SM_Stochastic) (Blau\_SM\_Stochastic.mq5)
4. Indicators, based on a Mean Deviation from the market trends:


   - [Mean Deviation Index Indicator](https://www.mql5.com/en/articles/190#Blau_MDI) (Blau\_MDI.mq5)
   - [Ergodic MDI-oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic_MDI) (Blau\_Ergodic\_MDI.mq5)
5. Indicators based on the Moving Average Convergence/Divergence:


   - [MACD indicator](https://www.mql5.com/en/articles/190#Blau_MACD) (Blau\_MACD.mq5)
   - [Ergodic MACD-oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic_MACD) (Blau\_Ergodic\_MACD.mq5)
6. Indicators, based on the Candlestick Momentum:


   - [Candlestick Momentum](https://www.mql5.com/en/articles/190#Blau_CMtm) (Blau\_CMtm.mq5)
   - [The Candlestick Momentum Index](https://www.mql5.com/en/articles/190#Blau_CMI) (Blau\_CMI.mq5)
   - [Candlestick Index Indicator](https://www.mql5.com/en/articles/190#Blau_CSI) (Blau\_CSI.mq5)
   - [Ergodic CMI-Oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic_CMI) (Blau\_Ergodic\_CMI.mq5)
   - [Ergodic CSI-Oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic_CSI) (Blau\_Ergodic\_CSI.mq5)
7. Indicators, based on a Composite High-Low Momentum:


   - [Indicator of the Virtual Close](https://www.mql5.com/en/articles/190#Blau_HLM) (Blau\_HLM.mq5)
   - [Directional Trend Index Indicator](https://www.mql5.com/en/articles/190#Blau_DTI) (Blau\_DTI.mq5)
   - [Ergodic DTI Oscillator](https://www.mql5.com/en/articles/190#Blau_Ergodic_DTI) (Blau\_Ergodic\_DTI.mq5)

For each group of the indicators the following are presented:

- Smoothed indicator index;
- The index of the normalized smoothed indicator;
- The oscillator, based on the index of the normalized smoothed index.

The [True Strength Index](https://www.mql5.com/en/articles/190#Blau_Mtm) section contains:

- A detailed analysis of William Blau's approach in the aspect of the technical analysis of the price chart;
- A detailed description of the algorithm and code of each indicator of the Momentum-based indicators groups.

As a smoothing method William Blau uses the [exponentially smoothed](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") [Moving Average](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") [s](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma") (EMA). The Exponential Moving Average is calculated by adding to the previous value of the Moving Average, a certain percentage of the current price.

When using the EMA, the latest prices have a greater weight.

The function of calculating of EMA:

```
EMA(k,n) = EMA(k-1,n) + 2/(n+1) * (price(k) - EMA(k-1,n))
         = price(k) * 2/(n+1) + EMA(k-1,n) * (1 - 2/(n+1))
```

where:

- EMA(k,n) - exponentially smoothed moving average of period n for the moment of period k;
- price(k) - the price at the moment of period k.

The description of the four types of moving averages and the methods of their use in technical analysis (see also [iMA](https://www.mql5.com/en/docs/indicators/ima)) can be found in the "MetaTrader 5 Help" ( ["Analytics/Technical Indicators/Trend Indicators/Moving Average"](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma")).

**The Library of Functions**

The library of functions for calculating the moving averages is located in the "MovingAverages.mqh". We are concerned with the ExponentialMAOnBuffer(), which fills the output array buffer\[\] with EMA values of the input array price\[\]. However, the implementation of the ExponentialMAOnBuffer() presented in the "MovingAverages.mqh" has the disadvantage that it does not work with the period n = 1.

See in the source code:

```
if(period<=1 || rates_total-begin<period) return(0);
```

However, William Blau in his book uses the smoothing period n = 1 as the absence of smoothing.

Therefore, the code of the ExponentialMAOnBuffer() function has undergone a few changes:

```
if(period<1 || rates_total-begin<period) return(0);
```

and we obtain the ExponentialMAOnBufferWB(). The code of this function is located in the file "WilliamBlau.mqh".

The file "WilliamBlau.mqh" also has the following the functions:

- The PriceName() function returns the price type as a string:

```
string PriceName(
                 const int applied_price // price type
                )
```

- The CalculatePriceBuffer() function calculates the price array of this price type:

```
int CalculatePriceBuffer(
                         const int applied_price,   // price type
                         const int rates_total,     // rates total
                         const int prev_calculated, // bars, processed at the last call
                         const double &Open[],      // Open[]
                         const double &High[],      // High[]
                         const double &Low[],       // Low[]
                         const double &Close[],     // Close[]
                         double &Price[]           // calculated prices array
                        )
```

**The applied price type and the timeframe of the price chart**

William Blau considers the closing prices of the Daily timeframe. The indicators, developed in this article, allow you to choose the price type (see [price constants](https://www.mql5.com/en/docs/constants/indicatorconstants/prices)) the timeframe of the price chart depends on the timeframe of the indicator (see [chart timeframes](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)).

### 1\. The True Strength Index

The considered indicators (see attachment):

1. **Blau\_Mtm.mq5**\- Indicator of the rate (q-period Momentum; smoothed q-period Momentum);
2. **Blau\_TSI.mq5** \- True strengths Index (Normalized smoothed q-period Momentum);
3. **Blau\_Ergodic.mq5** \- Ergodic Oscillator (based on the True Strength Index).

**1.1. Momentum**

The description of the built-in technical indicator [Momentum](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/momentum "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/momentum"), and its use is in technical analysis can be found in the "MetaTrader 5 Help" section ["Analytics/Technical Indicators/Oscillators/Momentum"](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/momentum "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/momentum") (see also [iMomentum](https://www.mql5.com/en/docs/indicators/imomentum)). In contrast to the standard Momentum ( [iMomentum](https://www.mql5.com/en/docs/indicators/imomentum)) the Momentum by William Blau calculates the Momentum as the _absolute_ price change.

An example of the MQL5-implementation of the True Strength Indicator (TSI) by William Blau is presented in the article ["MQL5: Create Your Own Indicator"](https://www.mql5.com/en/articles/10).

**1.1.1. Technical analysis using Momentum indicator**

The object of the technical analysis is the price chart of the financial instrument. Each element of the chart is a price [bar](https://www.mql5.com/en/docs/series/bars). The price bar has the following characteristics: [opening time](https://www.mql5.com/en/docs/series/copytime), [opening price](https://www.mql5.com/en/docs/series/copyopen), [maximum price](https://www.mql5.com/en/docs/series/copyhigh), [minimum price](https://www.mql5.com/en/docs/series/copylow), [closing price](https://www.mql5.com/en/docs/series/copyclose), [trading volumes](https://www.mql5.com/en/docs/series/copyrealvolume), and [other](https://www.mql5.com/en/docs/series). The price bar is formed and reflects the behavior of prices during a specific discrete time period (chart [timeframe](https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes)).

_**The task of the technical analysis of the price chart** is to determine the current trend of the price movement, reveal the price peaks and bottoms and predict the direction of the price change in the coming period of time._ The complexity of this, is that the price, while moving within the limits of its basic tendency, makes multi-directional fluctuations creating a so-called price-noise.

**_What William Blau has proposed._** _**The first difference: the Momentum.**_ William Blau calculated the Momentum as a relative to the price change \[closing\] for every \[day\] period of time; and created the Momentum indicator. _From a mathematical point of view_  the Momentum function is the first derivative of the price.

![Fig. 1.1. Momentum Indicator (q-period Momentum)](https://c.mql5.com/2/2/mtm01.PNG)

Fig. 1.1. Momentum Indicator (q-period Momentum)

The Momentum displays one-day period price fluctuations shows the speed (magnitude) and the direction of the price changes over this period, but it does not reflect the general trend of the price movement, and does not determine the trend reversal points.

_**The second difference is the smoothing.**_ The moving average of the Momentum (the cumulative sum of daily price fluctuations) almost exactly reproduces both, the major and local variations of the curve prices. Fig. 1.2 (a) in the subwindows I, II present the smoothed Momentum (moving averages with periods 20 and 300, respectively).

The _higher_ is the period of the Moving Average, the _more accurately_ the smoothed Momentum approximates (reproduces) the fluctuations of the price curve. _From a mathematical point of view_ the function of smoothing the Momentum is the integral function of the momentum, or the restored function of the price.

![Fig. 1.2 (a). Momentum Indicator (smoothed q-period Momentum)](https://c.mql5.com/2/2/mtm02a.PNG)

Fig. 1.2 (a). Momentum Indicator (smoothed q-period Momentum)

![Fig. 1.2 (b). Momentum Indicator (smoothed q-period Momentum)](https://c.mql5.com/2/2/mtm02b.PNG)

Fig. 1.2 (b). Momentum Indicator (smoothed q-period Momentum)

In Fig. 1.2 (a), in the main window, the EMA-smoothed (with periods of 5, 20, 100) indicators are presented. A slight increase in the period of the moving average leads to a lag and the moving average practically becomes unable to reproduce the fluctuations of the price curve.

_**The third difference is the resmoothing.**_ The first smoothing of the Momentum defines the main trend of the price movement, as well as the reversal points, but does not eliminate the noise. To eliminate the price noise a re-smoothing is needed with a _small_ period of the moving average.

Fig. 1.2 (b), in the sub-window I presents the smoothed Momentum indicator (moving average with period 20), the subwindows II and III present  the double-and triple-smoothed Momentum (periods of moving average of 5, 3). A repeated smoothing eliminates the price noise, but adds a slight shift of the curve (a lag).

_**The fourth difference: the difference in a signal of changing trends.**_ The smoothing of Momentum with a small averaging period may lead to a divergence of the smoothed Momentum with the trend of the price curve.

On Fig. 1.2 (a), the discrepancy is observed in the subwindow I, and on Fig. 1.2 (b) - in the subwindows I, II, III (the direction of the price changes diverges from the direction of the change in the smoothed Momentum). Such differences often indicates a trend change. _From a mathematical point of view_  the divergence  is a function of the smoothing period.

The reliability of the interpretation of these differences as a signal of changing trends can be improved if we consider the divergence only for the overbought or oversold areas (see п. 1.2.1).

**1.1.2. Definition of the Momentum**

**The Momentum** is _a relative_ price change.

_The sign of the Momentum shows_ the direction of the price change: a positive Momentum - the price increased over the period, a negative - the price has declined over the period. _The magnitude of the Momentum_ \- is the relative speed of the price change (first derivative of the price).

![Fig. 1.3. Definition of the Momentum](https://c.mql5.com/2/2/mtm03.PNG)

Fig. 1.3. Definition of the Momentum

**Formula of the Momentum**:

```
mtm(price) = price - price[1]
```

where:

- price - price \[closing\] of the current period;
- price \[1\] - price of \[closing\] of the previous period.

William Blau examines the momentum as the difference of the price of \[closing\] of the current period and the price of \[closing\] of the previous period. _William Blau, in his calculation of a **single** period momentum, uses the prices of **two** periods (the current and the previous periods)._

We introduce into the formula for calculating the momentum a period indicator, _**q - is the number of time periods involved in the calculation**_ (By William Blau q = 2).

**Formula of q-period Momentum**:

```
mtm(price,q) = price - price[q-1]
```

where:

- q - number of bars, used in the calculation of the momentum;
- price - price \[closing\] of the current period;
- price \[q-1\] - price of \[closing\] (q-1) periods ago.

In the resulting formula, our **two period** Momentum corresponds to **one period** relative Momentum of William Blau.

**Formula of a smoothed q-period Momentum**:

```
Mtm(price,q,r,s,u) = EMA(EMA(EMA( mtm(price,q) ,r),s),u)
```

where:

- price - price of \[closing\] - the price base of the price chart;
- q - number of bars, used in calculation of the Momentum;
- mtm(price,q)=price-price\[q-1\] - q-period Momentum;
- EMA (mtm (price, q), r) - the first smoothing - the EMA(r), applied to the q-period Momentum;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - the EMA(u), applied to the result of the 2nd smoothing.

**1.1.3. Mtm(price,q,r,s,u) - rate indicator (momentum). Specification**

- **File name**: **Blau\_Mtm.mq5**
- **The name**: Momentum (q-period Momentum; smoothed q-period Momentum) by William Blau.
- **Input parameters**:

  - q - the period for which the Momentum is calculated (default q = 2);
  - r -period of the 1-st EMA, applied to the Momentum (default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s or u are equal to 1, the EMA smoothing is not used. For example, if you set Mtm (price, 2,20,5,1), we obtain a double-smoothed momentum, but if you set Mtm (price, 2,1,1,1), we obtain a nonsmoothed momentum;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

### 1.2. The True Strengths Index

**1.2.1. Technical analysis using the True Strength Index**

**_Continued_**: See the beginning in Section 1.1.1.

_**The fifth: normalization.**_ Bringing through the normalization of values of the smoothed Momentum to a single scale (mapping to the interval \[-1, +1\]), allows us to determine the overbought or oversold states of the market. Repeated multiplication of the values ​​of the normalized smoothed momentum a factor of 100 converts the numerical series in the percentage range (mapping to the interval \[-100, 100\]).

![Fig. 1.4. Normalized Smoothed Momentum](https://c.mql5.com/2/2/mtm04.PNG)

Fig. 1.4. Normalized Smoothed Momentum

A discrepancy as a signal of changing trends can be considered reliable if the normalized smoothed momentum is in the state of an overbought or oversold.

**1.2.2. The definition of the True Strength Index**

**The True Strength Index** (True Strength Index, TSI) - is an indicator of the normalized Momentum (normalized q-period Momentum). Bringing the values of the ​​smoothed Momentum to a single scale (mapping to  the interval \[-1, +1\]) is provided with the normalization of each value of the smoothed Momentum (the cumulative sum of the smoothed q-period price fluctuations) by the value of the smoothed Momentum, taken in absolute value.

Multiplication by a coefficient of 100 changes the interval of the display to \[-100, +100\] (percent). Normalization allows the interpretation of the TSI value as a level of overbought (positive) or oversold (negative) market.

**The formula of the True Strength Index**:

```
                     100 * EMA(EMA(EMA( mtm(price,q) ,r),s),u)         100 * Mtm(price,q,r,s,u)
TSI(price,q,r,s,u) = –––––––––––––––––––––––––------–––––––––– = ––––––––––––––––------–––––––––––––––
                       EMA(EMA(EMA( |mtm(price,q)| ,r),s),u)     EMA(EMA(EMA( |mtm(price,q)| ,r),s),u)
```

```
if EMA(EMA(EMA(|mtm(price,q)|,r),s),u)=0, then TSI(price,q,r,s,u)=0
```

where:

- price - price of \[closing\] - the price base of the price chart;
- q - period of the Momentum;
- mtm(price,q)=price-price\[q-1\] - q-period momentum;
- \| Mtm (price, q) \| - the absolute value of the q-period Momentum;
- Mtm (price, q, r, s, u) - three times smoothed q-period Momentum;
- EMA (..., r) - the first smoothing - the EMA of period r, applied to:

1) q-period Momentum;

2) absolute value of the q-period Momentum;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - EMA(u), applied to the result of the 2nd smoothing.

**1.2.3. TSI(price,q,r,s,u) - the True Strength Index. Specification**

- **File name**: **Blau\_TSI.mq5**
- **The name**: The True Strength Index (normalized smoothed q-period relative Momentum) by William Blau.
- **Input parameters**:

  - q - the period for which the momentum is calculated (default q = 2);
  - r -period of the 1-st EMA, applied to the Momentum (default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphical plotting - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) two-levels (default is -25 and +25) - add/remove a level; change the value, the level description, change the rendering style of the levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

### 1.3. Ergodic oscillator

**1.3.1. Technical analysis using the Ergodic Oscillator**

**_Continued_**: See the beginning in Secs. 1.1.1, 1.2.1.

_**Sixth: the areas of an overbought and oversold market.**_ Unit interval \[-1, +1\] or a percentage interval \[-100.100\], within which changes occur in the values ​​of the normalized smoothed momentum, allows you to define the areas overbought or oversold market.

_The class of indexes of technical analysis, which characterize the state of overbought or oversold market, is called the oscillator._ For each oscillator, levels are determined, at the approach to which, the signals of an overbought or oversold market are received. Oscillators are ineffective on trending markets, as the market can be in an overbought/oversold conditions for an arbitrarily long period.

_**Seventh: The Signal Line.**_ To obtain a signal about the end of a trend and a reversal trend of a price movement, a signal line is used. The signal to buy is received when the main line crosses the signal line from the bottom up. The signal to sell is received when the main line crosses the signal line from the top down. In the case where there is a main line - this is an ergodic (true strength index), then a re-smoothing of the ergodic forms a signal line. The re-smoothing procedure is equal to the last process of ergodic smoothing.

_**Eighth: the trend of the price movement.**_ The trend of the price movement is upwards (upward trend), when the main line (ergodic) passes above the signal line. The trend of the price movement is downwards (downward trend), when the main line (ergodic) passes under the signal line.

![Fig. 1.5. Ergodic Oscillator](https://c.mql5.com/2/2/mtm05.PNG)

Fig. 1.5. Ergodic Oscillator

**1.3.2. Definition of the Ergodic Oscillator**

```
Ergodic(price,q,r,s,u) = TSI(price,q,r,s,u)
```

```
SignalLine(price,q,r,s,u,ul) = EMA( Ergodic(price,q,r,s,u) ,ul)
```

where:

- Ergodic() - ergodic - True Strength Index TSI(price,q,r,s,u);
- The SignalLine() -a signal line - the EMA(ul), applied to the ergodic;
- ul - an EMA period of a signal line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic. For example, if you are using a double smoothing Ergodic (price, q, r, s, u) = Ergodic (price, 2,20,5,1), then by William Blau ul = s = 5.

**1.3.3. Ergodic (price, q,r,s,u,ul) - ergodic oscillator. Specification**

- **File name**: **Blau\_Ergodic.mq5**
- **Name**: Ergodic Oscillator (based on a true strength index) by William Blau.
- **Input parameters**:

  - graphic plot #0 - Ergodic (a true strength index):

    - q - the period for which the momentum is calculated (default q = 2);
    - r -period of the 1-st EMA, applied to the Momentum (default r = 20);
    - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
    - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - the signal line:

    - ul - period EMA signal line, is applied to the ergodic (by default ul = 3);

  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphic plots - the color, thickness, line style (the "Colors" tab);
  - two levels (by default -25 and +25) - add/remove a level, change the value, level description, change the rendering style of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) bounds of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - ul>0. If ul = 1, then the Signal Line and Ergodic lines are the same;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

**1.4. The Code (detailed description)**

**1.4.1. "Blau\_Mtm.mq5" - indicator Mtm(price,q,r,s,u) - momentum**

The code of the indicator Mtm (price,q,r,s,u):

```
//+------------------------------------------------------------------+
//|                                                     Blau_Mtm.mq5 |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp." // copyright
#property link      "https://www.mql5.com"                       // url
#property description "q-period Momentum (William Blau)"        // description
#include <WilliamBlau.mqh>              // include file (terminal_data_folder\MQL5\Include)
//--- indicator settings
#property indicator_separate_window     // indicator in a separate window
#property indicator_buffers 5           // number of buffers used
#property indicator_plots   1           // number of plots
//--- main graphic plot #0
#property indicator_label1  "Mtm"       // graphic plot label #0
#property indicator_type1   DRAW_LINE   // draw as a line
#property indicator_color1  Blue        // color
#property indicator_style1  STYLE_SOLID // line style - solid line
#property indicator_width1  1           // line width
//--- input parameters
input int    q=2;  // q - period of Momentum
input int    r=20; // r - 1st EMA, applied to momentum
input int    s=5;  // s - 2nd EMA, applied to the 1st EMA
input int    u=3;  // u - 3rd EMA, applied to the 2nd EMA
input ENUM_APPLIED_PRICE AppliedPrice=PRICE_CLOSE; // AppliedPrice - price type
//--- dynamic arrays
double MainBuffer[];     // u-period 3rd EMA (for graphic plot #0)
double PriceBuffer[];    // price array
double MtmBuffer[];      // q-period Momentum
double EMA_MtmBuffer[];  // r-period 1st EMA
double DEMA_MtmBuffer[]; // s-period 2nd EMA
//--- global variables
int    begin1, begin2, begin3, begin4; // data starting indexes
int    rates_total_min; // total rates min
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers
   // plot buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);             // u-period 3rd EMA
   // buffers for intermediate calculations
   SetIndexBuffer(1,PriceBuffer,INDICATOR_CALCULATIONS);    // price buffer
   SetIndexBuffer(2,MtmBuffer,INDICATOR_CALCULATIONS);      // q-period Momentum
   SetIndexBuffer(3,EMA_MtmBuffer,INDICATOR_CALCULATIONS);  // r-period 1st EMA
   SetIndexBuffer(4,DEMA_MtmBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA
/*
//--- graphic plot #0 (Main)
   PlotIndexSetString(0,PLOT_LABEL,"Mtm");             // label
   PlotIndexSetInteger(0,PLOT_DRAW_TYPE,DRAW_LINE);    // drawing type as a line
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,Blue);        // line color
   PlotIndexSetInteger(0,PLOT_LINE_STYLE,STYLE_SOLID); // line style
   PlotIndexSetInteger(0,PLOT_LINE_WIDTH,1);           // line width
*/
//--- precision
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
//---
   begin1=q-1;        //                             - MtmBuffer[]
   begin2=begin1+r-1; // or =(q-1)+(r-1)             - EMA_MtmBuffer[]
   begin3=begin2+s-1; // or =(q-1)+(r-1)+(s-1)       - DEMA_MtmBuffer[]
   begin4=begin3+u-1; // or =(q-1)+(r-1)+(s-1)+(u-1) - MainBuffer[]
   //
   rates_total_min=begin4+1; // minimal size
//--- starting index for plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_BEGIN,begin4);
//--- short indicator name
   string shortname=PriceName(AppliedPrice)+","+string(q)+","+string(r)+","+string(s)+","+string(u);
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_Mtm("+shortname+")");
//--- OnInit done
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(
                const int rates_total,     // rates total
                const int prev_calculated, // bars, calculated at previous call
                const datetime &Time[],    // Time
                const double &Open[],      // Open
                const double &High[],      // High
                const double &Low[],       // Low
                const double &Close[],     // Close
                const long &TickVolume[],  // Tick Volume
                const long &Volume[],      // Real Volume
                const int &Spread[]        // Spread
               )
  {
   int i,pos;
//--- check rates
   if(rates_total<rates_total_min) return(0);
//--- calc PriceBuffer[]
   CalculatePriceBuffer(
                        AppliedPrice,        // applied price
                        rates_total,         // rates total
                        prev_calculated,     // bars, calculated at previous call
                        Open,High,Low,Close, // Open[], High[], Low[], Close[] arrays
                        PriceBuffer          // price buffer
                       );
//--- calculation of q-period Momentum
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              // calc all values starting from begin1
      for(i=0;i<pos;i++)       // pos values
         MtmBuffer[i]=0.0;     // zero values
     }
   else pos=prev_calculated-1; // overwise recalc only last value
   // calculate MtmBuffer[]
   for(i=pos;i<rates_total;i++)
      MtmBuffer[i]=PriceBuffer[i]-PriceBuffer[i-(q-1)];
//--- EMA smoothing
   // r-period 1st EMA
   ExponentialMAOnBufferWB(
                           rates_total,     // rates total
                           prev_calculated, // bars, calculated at previous call
                           begin1,          // starting index
                           r,               // smoothing period
                           MtmBuffer,       // input array
                           EMA_MtmBuffer    // output array
                          );
   // s-period 2nd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_MtmBuffer,DEMA_MtmBuffer);
   // u-period 3rd EMA (for plot #0)
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_MtmBuffer,MainBuffer);
//--- OnCalculate done. Return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Let's consider the code in detail.

#### 1.4.1.1. Indicator settings Mtm (price,q,r,s,u)

##### Literature

What to read about the settings of the indicator in the [MQL5 Reference](https://www.mql5.com/en/docs):

1. Section " [Custom Indicators](https://www.mql5.com/en/docs/customind)".
2. [The relationship between the properties of the indicator and the corresponding functions](https://www.mql5.com/en/docs/customind/propertiesandfunctions) (See " [Custom Indicators](https://www.mql5.com/en/docs/customind)").
3. [Programs properties (# property)](https://www.mql5.com/en/docs/basis/preprosessor/compilation) (See "Language Basics/ Preprocessor").
4. [Rendering styles (graphic plot properties)](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) (See section "Standard constants, enumerations, and structures / indicator constants").
5. [Properties of custom indicators](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties) (See section "Standard constants, enumerations, and structures / indicator constants").

##### Copyright. Description of the indicator

```
#property copyright "Copyright 2011, MetaQuotes Software Corp." // copyright
#property link      "https://www.mql5.com"                       // url
#property description "q-period Momentum (William Blau)"        // description
```

Settings only through the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive. The сopyright (parameters _copyright_ and _link_), version (the parameter _version_) and a description of the mql5-program (parameter _description_) are displayed in the "Properties" of the indicator window (the "Properties" tab, box "Additional").

##### Include file

```
#include <WilliamBlau.mqh>              // include file (terminal_data_folder\MQL5\Include)
```

[Preprocessor](https://www.mql5.com/en/docs/basis/preprosessor) replaces the _#Include <WilliamBlau.mqh>_ line with the contents of the "WilliamBlau.mqh" file. _Angle brackets_ indicate that the file "WilliamBlau.mqh" will be taken from the terminal data folder. For more information see [Including files](https://www.mql5.com/en/docs/basis/preprosessor/include).

On the contents of the file "WilliamBlau.mqh" see the introduction.

##### Indicator settings (in general)

[The custom Indicator](https://www.mql5.com/en/docs/customind) \- is few graphic plots. Graphic plot of the indicator can be displayed either in the main window of the price chart or in a separate window. Each graphic plot has a certain drawing method, color, style, and thickness.

The data for the rendering of the graphic plot is taken from the indicator buffers (each graphic plot corresponds from one to five indicators buffers). We use an indicator array as an indicator buffer.

To set up the indicator, it is necessary to (see Fig. 1.6):

01. Specify the window for displaying the indicators.
02. Specify the number of graphic plots.
03. Specify the number of indicator buffers.
04. Declaration of the indicator arrays.
05. Set up a link: indicator array -> indicator buffer -> graphic plot.
06. Describe the properties of each graphic plot.
07. Specify the display precision of the indicator values.
08. Specify for each graphical construction, the number of initial bars without the rendering of the graphic plot.
09. Set up the horizontal levels, and describe the properties of each horizontal level ( **_not present_**.)
10. Set the scale restrictions for the separate indicator window ( **_not present_**.)
11. Specify the short name of the indicator.

![Fig. 1.6. Momentum Indicator Mtm (price,q,r,s,u)](https://c.mql5.com/2/2/mtm06.PNG)

Fig. 1.6. Momentum Indicator Mtm (price,q,r,s,u)

Indicator settings are performed:

- a) either through the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive,

- b) or using the [Special Functions](https://www.mql5.com/en/docs/customind).


For more information see [Connection between Indicator Properties and Corresponding Functions](https://www.mql5.com/en/docs/customind/propertiesandfunctions).

The difference in the methods of setting up the indicator is that the settings through the [_#property_](https://www.mql5.com/en/docs/basis/preprosessor/compilation)  directive are available before the indicator is attached to the price chart, while the settings through _special functions_ are available **_after_**  the indicator is attached to the price chart. The configuration of the settings is performed from the "Properties" window of the indicator.

##### The settings: a window for displaying the indicator (1)

```
#property indicator_separate_window     // indicator in a separate window
```

The configuration is mandatory and is only possible through the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive. There are two options of indicator display:

1. In the main window of the price chart - _indicator\_chart\_window_;
2. In a separate window - _indicator\_separate\_window_.

##### Settings: The number of buffers (3) and graphic plots (2)

```
#property indicator_buffers 5           // number of buffers used
#property indicator_plots   1           // number of plots
```

The configuration is mandatory and is possible only through the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive. The number of indicator buffers (parameter _indicator\_buffers_) and the number of graphic plots (parameter _indicator\_plots_) is not limited.

##### Settings: Indicator Arrays (4)

```
//--- dynamic arrays
double MainBuffer[];     // u-period 3rd EMA (for graphic plot #0)
double PriceBuffer[];    // price array
double MtmBuffer[];      // q-period Momentum
double EMA_MtmBuffer[];  // r-period 1st EMA
double DEMA_MtmBuffer[]; // s-period 2nd EMA
```

Indicator arrays are declared at [global level](https://www.mql5.com/en/docs/basis/variables/global) as one-dimensional [dynamic arrays](https://www.mql5.com/en/docs/basis/types/dynamic_array) of type [double](https://www.mql5.com/en/docs/basis/types/double).

##### Settings: Setting up the link (5) between the indicator arrays, indicator buffers, and graphic plots.

```
// graphic plot #0
SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);             // u-period 3rd EMA
// buffers for intermediate calculations
SetIndexBuffer(1,PriceBuffer,INDICATOR_CALCULATIONS);    // price buffer
SetIndexBuffer(2,MtmBuffer,INDICATOR_CALCULATIONS);      // q-period Momentum
SetIndexBuffer(3,EMA_MtmBuffer,INDICATOR_CALCULATIONS);  // r-period 1st EMA
SetIndexBuffer(4,DEMA_MtmBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA
```

The code is written in the function [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) of the event handler [Init()](https://www.mql5.com/en/docs/runtime/event_fire#init).

The link of the indicator buffer with the corresponding one-dimensional array is set up with the function [SetIndexBuffer()](https://www.mql5.com/en/docs/customind/setindexbuffer):

```
bool SetIndexBuffer(
   int                 index,    // index of the indicator buffer (starts from 0)
   double              buffer[], // dynamic array
   ENUM_INDEXBUFFER_TYPE data_type // type of data, stored in the indicator array
   );
```

The indicator buffer is a one-dimensional [dynamic array](https://www.mql5.com/en/docs/basis/types/dynamic_array) of [double](https://www.mql5.com/en/docs/basis/types/double) type, the size of which is controlled by the client terminal, so that it always corresponded to the number of bars on which the indicator is calculated. The indexation of indicator buffers starts from 0.

An indicator buffer can store three [types of data](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_indexbuffer_type_enum): _INDICATOR\_DATA_, _INDICATOR\_COLOR\_INDEX_, _INDICATOR\_CALCULATIONS_. Each graphic plot, depending on the method of its display, can be corresponded to by one to five indicator buffers: one to four indicator buffer values (data type _INDICATOR\_DATA_), and one color buffer (data type _INDICATOR\_COLOR\_INDEX_.)

Indicator buffers with the _INDICATOR\_CALCULATIONS_ data of type are designed for intermediate calculations. After binding, the indicator array will have [indexation](https://www.mql5.com/en/docs/series/bufferdirection) just like in conventional arrays (see below in Section 1.4.1.2).

##### Settings: Properties of graphic plots (6)

For the configuration of each set of graphic plots, the following things are specified:

1. Label;

2. Drawing Type (see all 18 types in the [ENUM\_DRAW\_TYPE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_draw_type) enumeration);
3. Line Color;

4. Line Style (see the possible styles enumerated in [ENUM\_LINE\_STYLE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_line_style));

5. Line Width.

There are two possible ways to configure:

1) Through the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive (implemented this way):

```
//--- graphic plot #0 (Main)
#property indicator_label1  "Mtm"       // label of graphic plot #0
#property indicator_type1   DRAW_LINE   // Drawing type: DRAW_LINE - line
#property indicator_color1  Blue        // Line color - Blue
#property indicator_style1  STYLE_SOLID // Line style: STYLE_SOLID - solid line
#property indicator_width1  1           // Line width
```

2) Using a [group of functions](https://www.mql5.com/en/docs/customind) of the settings of properties of the graphic plot [PlotIndexSetDouble()](https://www.mql5.com/en/docs/customind/plotindexsetdouble), [PlotIndexSetInteger()](https://www.mql5.com/en/docs/customind/plotindexsetinteger), [PlotIndexSetString()](https://www.mql5.com/en/docs/customind/plotindexsetstring):

```
//--- graphic plot #0 (Main)
   PlotIndexSetString(0,PLOT_LABEL,"Mtm");            // label
   PlotIndexSetInteger(0,PLOT_DRAW_TYPE,DRAW_LINE);    // drawing type as a line
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,Blue);       // line color
   PlotIndexSetInteger(0,PLOT_LINE_STYLE,STYLE_SOLID); // line style
   PlotIndexSetInteger(0,PLOT_LINE_WIDTH,1);          // line width
```

The code is written in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler. Specification of the [PlotIndexSet \*()](https://www.mql5.com/en/docs/customind) function:

```
bool PlotIndexSetDouble|Integer|String(
   int                             plot_index, // index of the graphic plot
   int                             prop_id,    // identifier of the property of the graphic plot
   double|int,char,bool,color|string  prop_value  // new value of the property
   );
```

To refine the display of the selected type of graphic plot, we use the property IDs of graphic plot, listed in the [ENUM\_PLOT\_PROPERTY](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles) enumeration.

The indexing of graphic plots starts from 0. Regarding the preferableness of configuring through a [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) directive see above in the "Indicator Preferences" section. Some properties of graphic plots (the color, style, line width) are available for change from the "Properties" window (the "Colors" tab) of the indicator.

##### Settings: The precision of the display of the indicator values ​​(7)

```
//--- precision
   IndicatorSetInteger(INDICATOR_DIGITS,_Digits);
```

The code is written in the  [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler. The specification of the function of indicator settings configuration [IndicatorSet \* ()](https://www.mql5.com/en/docs/customind):

```
bool IndicatorSetDouble|Integer|String(
   int                    prop_id,   // ID of indicator property
   double|int,color|string  prop_value // new value of a property
   );
```

Identifiers of indicator properties are listed in the [ENUM\_CUSTOMIND\_PROPERTY](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties) enumeration.

The precision of the display of the indicator values is given only by the [IndicatorSetInteger()](https://www.mql5.com/en/docs/customind/indicatorsetinteger) function, the ID of the indicator properties _INDICATOR\_DIGITS_, [ENUM\_CUSTOMIND\_PROPERTY\_INTEGER](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_customind_property_integer) enumeration.

In an example where the values ​​of the indicator buffers, which are intended to render, under display: next to the short name of the indicator, in a pop-up message, when the mouse pointer is placed over the indicator line - will be rounded up to [\_Digits](https://www.mql5.com/en/docs/predefined/_digits) \- number of digits after the decimal point in the price of the instrument, to which the indicator is attached.

##### Settings: Number of initial bars without rendering (8)

The data for rendering the q-period Momentum of William Blau is formed in four steps:

Step 1. On the basis of the data from the PriceBuffer\[\] prices array, the Momentum (the period q) is calculated. The values ​​of the q-period Momentum are placed into the MtmBuffer\[\] array. Since the indexation of the prices array starts from 0, the significant data in the prices array also start at index 0, then the significant data in the MtmBuffer\[\] array start with the index (q-1).

Step 2. Significant data in the MtmBuffer\[\] array is smoothed (smoothing period r). The values ​​of the smoothed q-period Momentum are placed in the EMA\_MtmBuffer\[\] array. Since the indexation of the MtmBuffer\[\] array starts from 0, the significant data in the MtmBuffer\[\] array starts with the index (q-1), then the significant data in the EMA\_MtmBuffer\[\] array start with the index (q-1) + (r-1).

The 3rd and 4th steps. Similar considerations are given for determining from which bar starts the meaningful data in the DEMA\_MtmBuffer\[\] array (smoothing period s) and in the MainBuffer\[\] array (smoothing period u). See Fig. 1.7.

![Fig. 1.7. The meaningful data of the Mtm (price,q,r,s,u) indicator](https://c.mql5.com/2/2/mtm07.PNG)

Fig. 1.7. The meaningful data of the Mtm (price,q,r,s,u) indicator

On a [global level](https://www.mql5.com/en/docs/basis/variables/global) the variables are declared:

```
//--- global variables
int    begin1, begin2, begin3, begin4; // data starting indexes
```

The values ​​of the variables - is the index of the bar, from which begins the meaningful data, in the corresponding to the variable indicator array. Variable values ​​are calculated in the function [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) event handler [Init](https://www.mql5.com/en/docs/runtime/event_fire#init), and will be used in the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function of the [Calculate](https://www.mql5.com/en/docs/runtime/event_fire#calculate) event handler.

```
//---
   begin1=q-1;        //                             - MtmBuffer[]
   begin2=begin1+r-1; // or =(q-1)+(r-1)             - EMA_MtmBuffer[]
   begin3=begin2+s-1; // or =(q-1)+(r-1)+(s-1)       - DEMA_MtmBuffer[]
   begin4=begin3+u-1; // or =(q-1)+(r-1)+(s-1)+(u-1) - MainBuffer[]
   //
   rates_total_min=begin4+1; // minimal size
//--- starting index for plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_BEGIN,begin4);
```

The number of initial bars without the showing at the graphic plot is specified using the [PlotIndexSetInteger()](https://www.mql5.com/en/docs/customind/plotindexsetinteger) function, the identifier of the indicator property _PLOT\_DRAW\_BEGIN_ enumerations [ENUM\_PLOT\_PROPERTY\_INTEGER](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_plot_property_integer).

##### Configuration: The short name of the indicator (11)

```
//--- short indicator name
   string shortname=PriceName(AppliedPrice)+","+string(q)+","+string(r)+","+string(s)+","+string(u);
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_Mtm("+shortname+")");
```

The code is written in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler. The short name of the indicator is specified only by using the [IndicatorSetString()](https://www.mql5.com/en/docs/customind/indicatorsetstring) function, identifier of the indicator properties _INDICATOR\_SHORTNAME_  ( [ENUM\_CUSTOMIND\_PROPERTY\_STRING](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_customind_property_string) enumeration). The **PriceName ()** function returns the name of the [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices), depending on the value of _AppliedPrice_ input parameter. The code of the _PriceName ()_  function is located in the file "WilliamBlau.mqh" (see Introduction).

##### Input parameters

```
//--- input parameters
input int    q=2;  // q - period of Momentum
input int    r=20; // r - 1st EMA, applied to momentum
input int    s=5;  // s - 2nd EMA, applied to the 1st EMA
input int    u=3;  // u - 3rd EMA, applied to the 2nd EMA
input ENUM_APPLIED_PRICE AppliedPrice=PRICE_CLOSE; // AppliedPrice - price type
```

For more information see [input variables](https://www.mql5.com/en/docs/basis/variables/inputvariables). Input parameters are available for change from the "Properties" window (the "Inputs" tab) of the indicator.

#### 1.4.1.2. The calculation of the indicator Mtm (price,q,r,s,u)

##### Calculation: The algorithm

The algorithm for calculating the indicator Mtm(price,q,r,s,u):

1. Check whether there is enough data to calculate the indicator.
2. The calculation of the prices array according to the specified [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \- formation of the PriceBuffer\[\] array
3. The determination of the index bar, from with which to begin/continue the calculation of the q-period Momentum.
4. The calculation of the q-period momentum - the filling of the MtmBuffer\[\] array.
5. The first smoothing by the EMA method (period r) - the filling of the EMA\_MtmBuffer\[\] array.
6. The second smoothing by the EMA method (period s) - the filling of the DEMA\_MtmBuffer\[\] array.
7. The third smoothing by the EMA method (period u) - the filling of the MainBuffer\[\]  array - the calculation of values ​​for the rendering of the graphic plot #0.

##### Calculation: The function OnCalculate()

The calculation of the indicator values ​​is performed in the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function of the [Calculate](https://www.mql5.com/en/docs/runtime/event_fire#calculate) event handler. We use the [second form of OnCalculate() function call](https://www.mql5.com/en/docs/basis/function/events#oncalculate2).

```
int OnCalculate(
                const int rates_total,     // rates total
                const int prev_calculated, // bars, calculated at the previous call
                const datetime &Time[],    // Time
                const double &Open[],      // Open
                const double &High[],      // High
                const double &Low[],       // Low
                const double &Close[],     // Close
                const long &TickVolume[],  // Tick Volume
                const long &Volume[],      // Real Volume
                const int &Spread[]        // Spread
               )
  {
//---
//--- OnCalculate done. Return value of prev_calculated for next call
   return(rates_total);
  }
```

The _rates\_total_  argument  is the number of bars of the price chart, which are rendered and are available to the indicator for processing. The _prev\_calculated_ \- is the number of bars of the price chart that have been processed by the indicator at the time of the _start_ of the current [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function call.

The [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function returns the number of bars of the price chart that have been processed by the indicator at the time of the _end_  of the current call. This function returns the _rates\_total_ parameter and must be constructed in such a way, that on the very first call, _all_ of the unprocessed bars of the price chart, would be processed.

That is, if on the first call of the OnCalculate() function, the parameter _prev\_calculated_ is equal to 0, then the on the second call, the parameter _prev\_calculated_ is either equal to _rates\_total_ or _rates\_total +1_, and starting from the second call, the OnCalculate() function handles (counts) _only_ the last bar. For further clarification with an example, see [here](https://www.mql5.com/en/docs/customind).

Indicator buffers and Time\[\], Open\[\], High\[\], Low\[\], Close\[\], TickVolume\[\], Volume\[\], and Spread\[\]  arrays have a default [direction of indexing](https://www.mql5.com/en/docs/series/bufferdirection) from left to right, from the beginning to the end of the array, from the oldest to the latest data. The index of the first element is equal to 0. The size of the indicator buffer is controlled by the client terminal, so that it always corresponded to the number of bars on which the indicator is calculated.

##### Calculation: Check whether there is enough data to calculate the indicator (1)

```
//--- check rates
   if(rates_total<rates_total_min) return(0);
```

The [global](https://www.mql5.com/en/docs/basis/variables/global) variable _rates\_total\_min_  is the minimum size of the input timeseries of the indicator, calculated in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler.

```
   rates_total_min=begin4+1; // minimum size of the input timeseries of the indicator
```

##### Calculation: The prices arrays PriceBuffer\[\] (2)

```
//--- calculation of the prices array PriceBuffer[]
   CalculatePriceBuffer(
                        AppliedPrice,        // price type
                        rates_total,         // size of the input timeseries
                        prev_calculated,     // bars, processed on the previous call
                        Open,High,Low,Close, // Open[], High[], Low[], Close[]
                        PriceBuffer          // calculate the prices array
                       );
```

To fill the PriceBuffer\[\] prices array, the **CalculatePriceBuffer()** function is used. The code of the _CalculatePriceBuffer()_  function is located in the file "WilliamBlau.mqh" (see introduction). [Price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) is specified in the input parameter _AppliedPrice_.

##### Calculation: The definition of the bar index, from with which to begin/continue the calculation of the q-period Momentum (3)

The _pos_ [local variable](https://www.mql5.com/en/docs/basis/variables/local) is the index of the bar, from which the indicator will be calculated on the current call of the [OnCalculate()](https://www.mql5.com/en/docs/basis/function/events#oncalculate) function. Let's combine the calculation of the _pos_  variable with the stage of preparing the MtmBuffer\[\] array to the calculation (the stage of zeroing the insignificant elements of the MtmBuffer\[\] array).

##### Calculation: q-period Momentum (4)

```
//--- calculation of q-period Momentum
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              // calc all values starting from begin1
      for(i=0;i<pos;i++)       // pos values
         MtmBuffer[i]=0.0;     // zero values
     }
   else pos=prev_calculated-1; // overwise recalc only last value
   // calculate MtmBuffer[]
   for(i=pos;i<rates_total;i++)
      MtmBuffer[i]=PriceBuffer[i]-PriceBuffer[i-(q-1)];
```

The q-period Momentum is calculated as a difference between the current period PriceBuffer\[i\], and the price(q-1) of the previous periods PriceBuffer\[i-(q-1)\].

##### Calculation: smoothing by the EMA method (5-7)

```
//--- EMA smoothing
   // r-period 1st EMA
   ExponentialMAOnBufferWB(
                           rates_total,     // rates total
                           prev_calculated, // bars, calculated at previous call
                           begin1,          // starting index
                           r,               // smoothing period
                           MtmBuffer,       // input array
                           EMA_MtmBuffer    // output array
                          );
   // s-period 2nd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_MtmBuffer,DEMA_MtmBuffer);
   // u-period 3rd EMA (for plot #0)
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_MtmBuffer,MainBuffer);
```

The **ExponentialMAOnBuffer()** function is decribed in the introduction. On the example of the calculation of the r-period moving 1st EMA: the _ExponentialMAOnBuffer()_ function fills the EMA\_MtmBuffer\[\] output array with the values of ​​EMA (r) of the MtmBuffer\[\] input array; with insignificant data up to the index (begin1-1) inclusive, are filled with zero values.

**1.4.2. "Blau\_TSI.mq5" - indicator TSI(price,q,r,s,u) - the true strength index**

The code of the indicator TSI (price,q,r,s,u) (is built on the bases of changes and additions to the code "Blau\_Mtm.mq5"):

```
//+------------------------------------------------------------------+
//|                                                     Blau_TSI.mq5 |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp." // copyright
#property link      "https://www.mql5.com"                       // URL
#property description "True Strength Index (William Blau)"      // description
#include <WilliamBlau.mqh>               // include file (terminal_data_folder\MQL5\Include)
//--- indicator settings
#property indicator_separate_window      // indicator in a separate window
#property indicator_buffers 10           // number of buffers used
#property indicator_plots   1            // graphic plots
//--- horizontal levels
#property indicator_level1 -25           // level #0 (vertical)
#property indicator_level2 25            // level #1 (vertical)
#property indicator_levelcolor Silver    // level color
#property indicator_levelstyle STYLE_DOT // level style
#property indicator_levelwidth 1         // level width
//--- indicator min/max
#property indicator_minimum -100         // minimum
#property indicator_maximum 100          // maximum
//--- graphic plot #0 (Main)
#property indicator_label1  "TSI"        // label for graphic plot #0
#property indicator_type1   DRAW_LINE    // draw as a line
#property indicator_color1  Blue         // line color
#property indicator_style1  STYLE_SOLID  // line style
#property indicator_width1  1            // line width
//--- input parameters
input int    q=2;  // q - period of Momentum
input int    r=20; // r - 1st EMA, applied to Momentum
input int    s=5;  // s - 2nd EMA, applied to the 1st smoothing
input int    u=3;  // u - 3rd EMA, applied to the 2nd smoothing
input ENUM_APPLIED_PRICE AppliedPrice=PRICE_CLOSE; // AppliedPrice - price type
//--- dynamic arrays
double MainBuffer[];        // TSI (graphic plot #0)
double PriceBuffer[];       // price array
double MtmBuffer[];         // q-period Momentum
double EMA_MtmBuffer[];     // r-period 1st EMA
double DEMA_MtmBuffer[];    // s-period 2nd EMA
double TEMA_MtmBuffer[];    // u-period 3rd EMA
double AbsMtmBuffer[];      // q-period Momentum (absolute value)
double EMA_AbsMtmBuffer[];  // r-period 1st EMA (absolute value)
double DEMA_AbsMtmBuffer[]; // s-period 2nd EMA (absolute value)
double TEMA_AbsMtmBuffer[]; // u-period 3rd EMA (absolute value)
//--- global variables
int    begin1, begin2, begin3, begin4; // starting index
int    rates_total_min; // rates total min
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                // TSI
   // intermediate buffers; (not used for plot)
   SetIndexBuffer(1,PriceBuffer,INDICATOR_CALCULATIONS);       // price array
   SetIndexBuffer(2,MtmBuffer,INDICATOR_CALCULATIONS);         // q-period Momentum
   SetIndexBuffer(3,EMA_MtmBuffer,INDICATOR_CALCULATIONS);     // r-period 1st EMA
   SetIndexBuffer(4,DEMA_MtmBuffer,INDICATOR_CALCULATIONS);    // s-period 2nd EMA
   SetIndexBuffer(5,TEMA_MtmBuffer,INDICATOR_CALCULATIONS);    // u-period 3rd EMA
   SetIndexBuffer(6,AbsMtmBuffer,INDICATOR_CALCULATIONS);      // q-period моментум (absolute value)
   SetIndexBuffer(7,EMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);  // r-period 1st EMA (absolute value)
   SetIndexBuffer(8,DEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (absolute value)
   SetIndexBuffer(9,TEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // u-period 3rd EMA (absolute value)
/*
//--- graphic plot #0 (Main)
   PlotIndexSetString(0,PLOT_LABEL,"TSI");             // label of graphic plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_TYPE,DRAW_LINE);    // draw as a line
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,Blue);        // line color
   PlotIndexSetInteger(0,PLOT_LINE_STYLE,STYLE_SOLID); // line style
   PlotIndexSetInteger(0,PLOT_LINE_WIDTH,1);           // line width
*/
//--- precision
   IndicatorSetInteger(INDICATOR_DIGITS,2);
/*
//--- horizontal levels
   IndicatorSetInteger(INDICATOR_LEVELS,2);                // number of levels
   IndicatorSetDouble(INDICATOR_LEVELVALUE,0,-25);         // level #0
   IndicatorSetDouble(INDICATOR_LEVELVALUE,1,25);          // level #1
   IndicatorSetInteger(INDICATOR_LEVELCOLOR,Silver);       // level color
   IndicatorSetInteger(INDICATOR_LEVELSTYLE,STYLE_DOT);    // level style
   IndicatorSetInteger(INDICATOR_LEVELWIDTH,1);            // level width
   IndicatorSetString(INDICATOR_LEVELTEXT,0,"Oversold");   // level 0 description "Oversold"
   IndicatorSetString(INDICATOR_LEVELTEXT,1,"Overbought"); // level 1 description "Overbought"
//--- indicator scale
   IndicatorSetDouble(INDICATOR_MINIMUM,-100); // minimum
   IndicatorSetDouble(INDICATOR_MAXIMUM,100);  // maximum
*/
//---
   begin1=q-1;        //                             - MtmBuffer[], AbsMtmBuffer[]
   begin2=begin1+r-1; // or =(q-1)+(r-1)             - EMA_...[]
   begin3=begin2+s-1; // or =(q-1)+(r-1)+(s-1)       - DEMA_...[]
   begin4=begin3+u-1; // or =(q-1)+(r-1)+(s-1)+(u-1) - TEMA_...[], MainBuffer[]
   //
   rates_total_min=begin4+1; // rates total min
//--- starting index for plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_BEGIN,begin4);
//--- short indicator name
   string shortname=PriceName(AppliedPrice)+","+string(q)+","+string(r)+","+string(s)+","+string(u);
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_TSI("+shortname+")");
//--- OnInit done
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(
                const int rates_total,     // rates total
                const int prev_calculated, // bars, calculated at previous call
                const datetime &Time[],    // Time
                const double &Open[],      // Open
                const double &High[],      // High
                const double &Low[],       // Low
                const double &Close[],     // Close
                const long &TickVolume[],  // Tick Volume
                const long &Volume[],      // Real Volume
                const int &Spread[]        // Spread
               )
  {
   int i,pos;
   double value1,value2;
//--- check rates
   if(rates_total<rates_total_min) return(0);
//--- calc PriceBuffer[]
   CalculatePriceBuffer(
                        AppliedPrice,        // price type
                        rates_total,         // rates total
                        prev_calculated,     // bars, calculated at previous tick
                        Open,High,Low,Close, // Open[], High[], Low[], Close[]
                        PriceBuffer          // price buffer
                       );
//--- calculation of  mtm and |mtm|
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              // calc all values starting from begin1
      for(i=0;i<pos;i++)       // pos
        {
         MtmBuffer[i]=0.0;     // zero values
         AbsMtmBuffer[i]=0.0;  //
        }
     }
   else pos=prev_calculated-1; // overwise calc only last bar
   // calculate MtmBuffer[] and AbsMtmBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      MtmBuffer[i]=PriceBuffer[i]-PriceBuffer[i-(q-1)];
      AbsMtmBuffer[i]=MathAbs(MtmBuffer[i]);
     }
//--- EMA smoothing
   // r-period 1st EMA
   ExponentialMAOnBufferWB(
                           rates_total,     // rates total
                           prev_calculated, // bars, calculated at previous call
                           begin1,          // starting index
                           r,               // smoothing period
                           MtmBuffer,       // input array
                           EMA_MtmBuffer    // output array
                          );
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,r,AbsMtmBuffer,EMA_AbsMtmBuffer);
   // s-period 2nd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_MtmBuffer,DEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_AbsMtmBuffer,DEMA_AbsMtmBuffer);
   // u-period 3rd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_MtmBuffer,TEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_AbsMtmBuffer,TEMA_AbsMtmBuffer);
//--- TSI calculation (graphic plot #0)
   if(prev_calculated==0)      // at first call
     {
      pos=begin4;              // calc all values starting from begin4
      for(i=0;i<pos;i++)       //
         MainBuffer[i]=0.0;    // zero values
     }
   else pos=prev_calculated-1; // overwise calc only last bar
   // calculation of MainBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      value1=100*TEMA_MtmBuffer[i];
      value2=TEMA_AbsMtmBuffer[i];
      MainBuffer[i]=(value2>0)?value1/value2:0;
     }
//--- OnCalculate done. Return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Let us consider in detail only the modifications and additions to the code "Blau\_Mtm.mq5".

#### 1.4.2.1. The configurations of the indicator TSI (price,q,r s,u) (alterations and additions to the code "Blau\_Mtm.mq5")

##### Indicator settings (in general)

The configurations of the indicator _TSI(price,q,r,s,u)_ differ from the configurations of the indicator _Mtm(price,q,r,s,u)_ (see Fig. 1.8):

01. Specify the window for displaying the indicators ( _no chang_ e)
02. Specify the number of graphical structures ( _no change_)
03. Specify the number of indicator buffers ( _the number of buffers has increased_.)
04. Declaration of the indicator arrays ( _added to the arrays_.)
05. Assign the arrays/buffer/plots: the indicator array -> indicator buffer -> graphic plot ( _restructuring_.)
06. Describe the properties of each graphic plot ( _label has been changed_)
07. Specify the accuracy of the display of the indicator values ​​( _changed accuracy_)
08. Specify, for each graphic plot, the number of initial bars without showing on the graphic plot ( _no change_)
09. Set the horizontal levels and describe the properties of each horizontal level ( _**new**_)
10. Set limits for the scale of the separate indicator window ( _**new**_)
11. Specify the short indicator name ( _name changed_.)

![Fig. 1.8. True Strength Index TSI (price,q,r,s,u) indicator](https://c.mql5.com/2/2/mtm08.PNG)

Fig. 1.8. True Strength Index TSI (price,q,r,s,u) indicator

##### Configurations (changes)

In the code "Blau\_Mtm.mq5", the following minor modifications are made.

1\. The short description of the mql5-program is changed:

```
#property description "True Strength Index (William Blau)"      // description
```

2\. (in configuration 6) The number of graphic plots has not increased, the drawing method (DRAW\_LINE - line), the line color (Blue), the line style (STYLE\_SOLID - solid line), and the line width (1) remained unchanged, but the label for the graphic plot #0 has changed:

```
#property indicator_label1  "TSI"        // label for graphic plot #0
```

3\. (in configuration 7) The accuracy of the display of the indicator values is changed:

```
   IndicatorSetInteger(INDICATOR_DIGITS,2);
```

4\. (in configuration 11) the short name of the indicator is changed:

```
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_TSI("+shortname+")");
```

##### Configurations: horizontal levels (9)

To configure the horizontal levels, the following must be specified for each level:

1. The value on the vertical axis;

2. The description of the level (optional). Horizontal layers have a single style of rendering:


1. Color for the display of the line;
2. Line style (see the possible styles enumerated in [ENUM\_LINE\_STYLE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_line_style));

3. The thickness of the line.

There are two possible ways to configure:

1) Using the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive (Implemented this way).

```
//--- horizontal levels
#property indicator_level1 -25           // level #0 (vertical)
#property indicator_level2 25            // level #1 (vertical)
#property indicator_levelcolor Silver    // level color
#property indicator_levelstyle STYLE_DOT // level style
#property indicator_levelwidth 1         // level width
```

2) Using the group of the [IndicatorSet \*()](https://www.mql5.com/en/docs/customind) functions:

```
//--- horizontal levels
   IndicatorSetInteger(INDICATOR_LEVELS,2);                // number of levels
   IndicatorSetDouble(INDICATOR_LEVELVALUE,0,-25);         // level #0
   IndicatorSetDouble(INDICATOR_LEVELVALUE,1,25);          // level #1
   IndicatorSetInteger(INDICATOR_LEVELCOLOR,Silver);       // level color
   IndicatorSetInteger(INDICATOR_LEVELSTYLE,STYLE_DOT);    // level style
   IndicatorSetInteger(INDICATOR_LEVELWIDTH,1);            // level width
   IndicatorSetString(INDICATOR_LEVELTEXT,0,"Oversold");   // level 0 description "Oversold"
   IndicatorSetString(INDICATOR_LEVELTEXT,1,"Overbought"); // level 1 description "Overbought"
```

The code is written in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler. Indexation of the horizontal levels starts from 0. To refine the display of the horizontal level, the identifiers of the properties of the _INDICATOR\_LEVEL \*_ index are used, which are listed in the [ENUM\_CUSTOMIND\_PROPERTY](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_customind_property_integer) enumeration.

The description of each level is set only using the [IndicatorSetString()](https://www.mql5.com/en/docs/customind/indicatorsetstring) function, the identifier of the indicator property _INDICATOR\_LEVELTEXT_ ( [ENUM\_CUSTOMIND\_PROPERTY\_STRING](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_customind_property_string) enumeration). The description of the level is placed directly above the level, on the left.

You can add/remove horizontal levels, change the values, the description of each level, and the style of level rendering from the "Properties" window (the "Levels" tab) of the indicator.

##### Configurations: Limits of the scale of the separate indicator window (10)

There are two possible ways to configure:

1) Using the [#property](https://www.mql5.com/en/docs/basis/preprosessor/compilation) preprocessor directive (Implemented this way).

```
//--- indicator min/max
#property indicator_minimum -100         // minimum
#property indicator_maximum 100          // maximum
```

2) Using the [IndicatorSetDouble()](https://www.mql5.com/en/docs/customind/indicatorsetdouble) function, the identifiers of the properties of the indicators _INDICATOR\_MINIMUM_ and _INDICATOR\_MAXIMUM_  ( [ENUM\_CUSTOMIND\_PROPERTY\_DOUBLE](https://www.mql5.com/en/docs/constants/indicatorconstants/customindicatorproperties#enum_customind_property_double) enumeration).

```
//--- indicator scale
   IndicatorSetDouble(INDICATOR_MINIMUM,-100); // minimum
   IndicatorSetDouble(INDICATOR_MAXIMUM,100);  // maximum
```

The code is written in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function of the [Init](https://www.mql5.com/en/docs/runtime/event_fire#init) event handler. The lower and upper bounds of the scale of a separate indicator window are available for change from the "Properties" window (the "Scale" tab) of the indicator.

##### Configurations (changes): The indicator buffers (3-5)

The changes in the configuration "indicator array -> indicator buffer -> graphic plot":

1\. (in configuration 3) The number of buffers increased:

```
#property indicator_buffers 10           // the number of buffers for the calculation of the indicator
```

2\. (in configuration 4) Added indicator arrays that are needed to calculate the absolute value of the q-period Momentum:

```
double AbsMtmBuffer[];      // q-period Momentum (absolute value)
double EMA_AbsMtmBuffer[];  // r-period 1st EMA (absolute value)
double DEMA_AbsMtmBuffer[]; // s-period 2nd EMA (absolute value)
double TEMA_AbsMtmBuffer[]; // u-period 3rd EMA (absolute value)
```

the purpose of the MainBuffer\[\] array is changed:

```
double MainBuffer[];        // TSI (graphic plot #0)
double TEMA_MtmBuffer[];    // u-period 3rd EMA
```

3\. (in configuration 5) The connection of "indicator array -> indicator buffer -> graphic plot" is changed:

```
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                // TSI
   // intermediate buffers; (not used for plot)
   SetIndexBuffer(1,PriceBuffer,INDICATOR_CALCULATIONS);       // price array
   SetIndexBuffer(2,MtmBuffer,INDICATOR_CALCULATIONS);         // q-period Momentum
   SetIndexBuffer(3,EMA_MtmBuffer,INDICATOR_CALCULATIONS);     // r-period 1st EMA
   SetIndexBuffer(4,DEMA_MtmBuffer,INDICATOR_CALCULATIONS);    // s-period 2nd EMA
   SetIndexBuffer(5,TEMA_MtmBuffer,INDICATOR_CALCULATIONS);    // u-period 3rd EMA
   SetIndexBuffer(6,AbsMtmBuffer,INDICATOR_CALCULATIONS);      // q-period моментум (absolute value)
   SetIndexBuffer(7,EMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);  // r-period 1st EMA (absolute value)
   SetIndexBuffer(8,DEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (absolute value)
   SetIndexBuffer(9,TEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // u-period 3rd EMA (absolute value)
```

#### 1.4.2.2. The calculation of the indicator TSI (price,q,r,s,u) (alterations and additions to the code "Blau\_Mtm.mq5")

##### Calculation: The algorithm

The algorithm for calculating the TSI (price,q,r,s,u) indicator:

1. Check whether there is enough data to calculate the indicator.
2. The calculation of the prices array according to the specified [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \- formation of the PriceBuffer\[\] array.
3. The determination of the index bar, from with which to begin/continue the calculation of the q-period Momentum.
4. The calculation of the q-period Momentum, and its absolute value - the filling of MtmBuffer\[\] and AbsMtmBuffer\[\] arrays.
5. The first smoothing by the EMA method (period r) - the filling of  EMA\_MtmBuffer\[\] and EMA\_AbsMtmBuffer\[\] arrays.
6. The second smoothing by the EMA method (period s) - the filling of DEMA\_MtmBuffer\[\] and DEMA\_AbsMtmBuffer\[\] arrays.
7. The third method smoothing by the EMA method (period u) - the filling of TEMA\_MtmBuffer\[\] and TEMA\_AbsMtmBuffer\[\] arrays.
8. The determination of the index bar, from with which to begin/continue the calculation of the true strength index.
9. The calculation of the the true strength index - the filling of the MainBuffer\[\] array - the calculation of values ​​for graphic plot #0.

The essence of the changes in the algorithm (briefly):

- a) (see paragraph 4-7) parallel to the calculation of the q-period momentum (group of arrays \* MtmtBuffer\[\]) the calculation of the absolute value of the q-period Momentum (\*AbsMtmBuffer\[\] group of arrays) is performed;
- b) (see Section 8-9) calculation of TSI is added.

##### Calculation: the q-period Momentum its absolute value (3-7)

```
//--- calculation of  mtm and |mtm|
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              // calc all values starting from begin1
      for(i=0;i<pos;i++)       // pos
        {
         MtmBuffer[i]=0.0;     // zero values
         AbsMtmBuffer[i]=0.0;  //
        }
     }
   else pos=prev_calculated-1; // overwise calc only last bar
   // calculate MtmBuffer[] and AbsMtmBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      MtmBuffer[i]=PriceBuffer[i]-PriceBuffer[i-(q-1)];
      AbsMtmBuffer[i]=MathAbs(MtmBuffer[i]);
     }
//--- EMA smoothing
   // r-period 1st EMA
   ExponentialMAOnBufferWB(
                           rates_total,     // rates total
                           prev_calculated, // bars, calculated at previous call
                           begin1,          // starting index
                           r,               // smoothing period
                           MtmBuffer,       // input array
                           EMA_MtmBuffer    // output array
                          );
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,r,AbsMtmBuffer,EMA_AbsMtmBuffer);
   // s-period 2nd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_MtmBuffer,DEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_AbsMtmBuffer,DEMA_AbsMtmBuffer);
   // u-period 3rd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_MtmBuffer,TEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_AbsMtmBuffer,TEMA_AbsMtmBuffer);
```

**Calculation: The True Strength Index (8-9)**

```
//--- TSI calculation (graphic plot #0)
   if(prev_calculated==0)      // at first call
     {
      pos=begin4;              // calc all values starting from begin4
      for(i=0;i<pos;i++)       //
         MainBuffer[i]=0.0;    // zero values
     }
   else pos=prev_calculated-1; // overwise calc only last bar
   // calculation of MainBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      value1=100*TEMA_MtmBuffer[i];
      value2=TEMA_AbsMtmBuffer[i];
      MainBuffer[i]=(value2>0)?value1/value2:0;
     }
```

**1.4.3. "Blau\_Ergodic.mq5" - Ergodic(price,q,r,s,u,ul) - Ergodic Oscillator**

The code of the Ergodic (price,q,r,s,u,ul) indicator is based on changes of the code of "Blau\_TSI.mq5":

```
//+------------------------------------------------------------------+
//|                                                 Blau_Ergodic.mq5 |
//|                        Copyright 2011, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2011, MetaQuotes Software Corp." // copyright
#property link      "https://www.mql5.com"                       // URL
#property description "Ergodic Oscillator (William Blau)"       // description
#include <WilliamBlau.mqh>                 // include file (terminal_data_folder\MQL5\Include)
//--- indicator settings
#property indicator_separate_window        // indicator in a separate window
#property indicator_buffers 11             // number of buffers
#property indicator_plots   2              // indicator plots
//--- horizontal levels
#property indicator_level1 -25             // level #0
#property indicator_level2 25              // level #1
#property indicator_levelcolor Silver      // level color
#property indicator_levelstyle STYLE_DOT   // level style
#property indicator_levelwidth 1           // level width
//--- min/max
#property indicator_minimum -100           // minimum
#property indicator_maximum 100            // maximum
//--- graphic plot #0 (Main)
#property indicator_label1  "Ergodic"      // graphic plot #0
#property indicator_type1   DRAW_HISTOGRAM // draw as a histogram
#property indicator_color1  Silver         // histogram color
#property indicator_style1  STYLE_SOLID    // line style
#property indicator_width1  2              // line width
//--- graphic plot #1 (Signal Line)
#property indicator_label2  "Signal"       // graphic plot #1
#property indicator_type2   DRAW_LINE      // draw as a line
#property indicator_color2  Red            // line color
#property indicator_style2  STYLE_SOLID    // line style
#property indicator_width2  1              // line width
//--- input parameters
input int    q=2;  // q - period of Momentum
input int    r=20; // r - 1st EMA, applied to Momentum
input int    s=5;  // s - 2nd EMA, applied to the 1st smoothing
input int    u=3;  // u - 3rd EMA, applied to the 2nd smoothing
input int    ul=3; // ul- period of a Signal Line
input ENUM_APPLIED_PRICE AppliedPrice=PRICE_CLOSE; // AppliedPrice - price type
//--- dynamic arrays
double MainBuffer[];        // Ergodic (graphic plot #0)
double SignalBuffer[];      // Signal line: ul-period EMA of Ergodic (graphic plot #1)
double PriceBuffer[];       // price array
double MtmBuffer[];         // q-period Momentum
double EMA_MtmBuffer[];     // r-period of the 1st EMA
double DEMA_MtmBuffer[];    // s-period of the 2nd EMA
double TEMA_MtmBuffer[];    // u-period of the 3rd EMA
double AbsMtmBuffer[];      // q-period Momentum (absolute value)
double EMA_AbsMtmBuffer[];  // r-period of the 1st EMA (absolute value)
double DEMA_AbsMtmBuffer[]; // s-period of the 2nd EMA (absolute value)
double TEMA_AbsMtmBuffer[]; // u-period of the 3rd EMA (absolute value)
//--- global variables
int    begin1, begin2, begin3, begin4, begin5; // starting indexes
int    rates_total_min; // rates total min
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                 // Ergodic
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);               // signal line: ul-period EMA of Ergodic
   // buffers for intermediate calculations
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);        // price array
   SetIndexBuffer(3,MtmBuffer,INDICATOR_CALCULATIONS);          // q-period моментум
   SetIndexBuffer(4,EMA_MtmBuffer,INDICATOR_CALCULATIONS);      // r-period of the 1st EMA
   SetIndexBuffer(5,DEMA_MtmBuffer,INDICATOR_CALCULATIONS);     // s-period of the 2nd EMA
   SetIndexBuffer(6,TEMA_MtmBuffer,INDICATOR_CALCULATIONS);     // u-period of the 3rd EMA
   SetIndexBuffer(7,AbsMtmBuffer,INDICATOR_CALCULATIONS);       // q-period Momentum (absolute value)
   SetIndexBuffer(8,EMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);   // r-period of the 1st EMA (absolute value)
   SetIndexBuffer(9,DEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);  // s-period of the 2nd EMA (absolute value)
   SetIndexBuffer(10,TEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // u-period of the 3rd EMA (absolute value)
/*
//--- graphic plot #0 (Main)
   PlotIndexSetString(0,PLOT_LABEL,"Ergodic");           // label of graphic plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_TYPE,DRAW_HISTOGRAM); // draw as a histogram
   PlotIndexSetInteger(0,PLOT_LINE_COLOR,Silver);        // line color
   PlotIndexSetInteger(0,PLOT_LINE_STYLE,STYLE_SOLID);   // line style
   PlotIndexSetInteger(0,PLOT_LINE_WIDTH,2);             // line width
//--- graphic plot #1 (Signal Line)
   PlotIndexSetString(1,PLOT_LABEL,"Signal");            // label of graphic plot #1
   PlotIndexSetInteger(1,PLOT_DRAW_TYPE,DRAW_LINE);      // draw as a line
   PlotIndexSetInteger(1,PLOT_LINE_COLOR,Red);           // line color
   PlotIndexSetInteger(1,PLOT_LINE_STYLE,STYLE_SOLID);   // line style
   PlotIndexSetInteger(1,PLOT_LINE_WIDTH,1);             // line width
*/
//--- precision
   IndicatorSetInteger(INDICATOR_DIGITS,2);
/*
//--- horizontal levels
   IndicatorSetInteger(INDICATOR_LEVELS,2);                // number of indicator levels
   IndicatorSetDouble(INDICATOR_LEVELVALUE,0,-25);         // level #0
   IndicatorSetDouble(INDICATOR_LEVELVALUE,1,25);          // level #1
   IndicatorSetInteger(INDICATOR_LEVELCOLOR,Silver);       // level color
   IndicatorSetInteger(INDICATOR_LEVELSTYLE,STYLE_DOT);    // level style
   IndicatorSetInteger(INDICATOR_LEVELWIDTH,1);            // level width
   IndicatorSetString(INDICATOR_LEVELTEXT,0,"Oversold");   // level #0 "Oversold"
   IndicatorSetString(INDICATOR_LEVELTEXT,1,"Overbought"); // level #1 "Overbought"
//--- min/max values
   IndicatorSetDouble(INDICATOR_MINIMUM,-100); // min
   IndicatorSetDouble(INDICATOR_MAXIMUM,100);  // max
*/
//---
   begin1=q-1;         //                                    - MtmBuffer[], AbsMtmBuffer[]
   begin2=begin1+r-1;  // or =(q-1)+(r-1)                    - EMA_...[]
   begin3=begin2+s-1;  // or =(q-1)+(r-1)+(s-1)              - DEMA_...[]
   begin4=begin3+u-1;  // or =(q-1)+(r-1)+(s-1)+(u-1)        - TEMA_...[], MainBuffer[]
   begin5=begin4+ul-1; // or =(q-1)+(r-1)+(s-1)+(u-1)+(ul-1) - SignalBuffer[]
   //
   rates_total_min=begin5+1; // rates total min
//--- starting bar index for plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_BEGIN,begin4);
//--- starting bar index for plot #1
   PlotIndexSetInteger(1,PLOT_DRAW_BEGIN,begin5);
//--- short indicator name
   string shortname=PriceName(AppliedPrice)+","+string(q)+","+string(r)+","+string(s)+","+string(u)+","+string(ul);
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_Ergodic("+shortname+")");
//--- OnInit done
   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(
                const int rates_total,     // rates total
                const int prev_calculated, // bars, calculated at previous call
                const datetime &Time[],    // Time
                const double &Open[],      // Open
                const double &High[],      // High
                const double &Low[],       // Low
                const double &Close[],     // Close
                const long &TickVolume[],  // Tick Volume
                const long &Volume[],      // Real Volume
                const int &Spread[]        // Spread
               )
  {
   int i,pos;
   double value1,value2;
//--- rates total
   if(rates_total<rates_total_min) return(0);
//--- calculation of PriceBuffer[]
   CalculatePriceBuffer(
                        AppliedPrice,        // price type
                        rates_total,         // rates total
                        prev_calculated,     // bars, calculated at the previous call
                        Open,High,Low,Close, // Open[], High[], Low[], Close[]
                        PriceBuffer          // price array
                       );
//--- calculation of mtm and |mtm|
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              // starting from begin1
      for(i=0;i<pos;i++)       // pos
        {
         MtmBuffer[i]=0.0;     // zero values
         AbsMtmBuffer[i]=0.0;  //
        }
     }
   else pos=prev_calculated-1; // overwise calc only last bar
   // calculate MtmBuffer[] and AbsMtmBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      MtmBuffer[i]=PriceBuffer[i]-PriceBuffer[i-(q-1)];
      AbsMtmBuffer[i]=MathAbs(MtmBuffer[i]);
     }
//--- EMA smoothing
   // r-period of the 1st EMA
   ExponentialMAOnBufferWB(
                           rates_total,     // rates total
                           prev_calculated, // bars, calculated at previous call
                           begin1,          // starting index
                           r,               // smoothing period
                           MtmBuffer,       // input array
                           EMA_MtmBuffer    // output array
                          );
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,r,AbsMtmBuffer,EMA_AbsMtmBuffer);
   // s-period of 2nd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_MtmBuffer,DEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin2,s,EMA_AbsMtmBuffer,DEMA_AbsMtmBuffer);
   // u-period 3rd EMA
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_MtmBuffer,TEMA_MtmBuffer);
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin3,u,DEMA_AbsMtmBuffer,TEMA_AbsMtmBuffer);
//--- calculation of Ergodic (graphic plot #0)
   if(prev_calculated==0)      // at first call
     {
      pos=begin4;              // starting from begin4
      for(i=0;i<pos;i++)       // pos
         MainBuffer[i]=0.0;    // zero values
     }
   else pos=prev_calculated-1; // overwise calculate only last bar
   // calculation of MainBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      value1=100*TEMA_MtmBuffer[i];
      value2=TEMA_AbsMtmBuffer[i];
      MainBuffer[i]=(value2>0)?value1/value2:0;
     }
//--- calculation of Signal Line (graphic plot #1)
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin4,ul,MainBuffer,SignalBuffer);
//--- OnCalculate done. Return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
```

Let us consider in detail only the modifications and additions to the code "Blau\_TSI.mq5".

#### 1.4.3.1. Configurations of the indicator Ergodic (price,q,r,s,u,ul) (alterations and additions to the code "Blau\_TSI.mq5")

##### Indicator settings (in general)

The configurations of the indicator _Ergodic (price,q,r,s,u,ul)_ differ from the configurations of the indicator _TSI (price,q,r,s,u)_ (See Fig. 1.9):

01. Specify the window for displaying the indicators ( _no change_)
02. Specify the number of graphic plots ( _**a graphic plot is added**_)
03. Specify the number of indicator buffers ( _the number of buffers has increased_)
04. Declaration of the indicator arrays ( _added to the array_)
05. Set up a relation: the indicator array -> indicator buffer -> graphic plot ( _restructuring_.)
06. Describe the properties of each graphic plot ( _properties altered,a graphic plot is added_.)
07. Specify the display precision of the indicator values ​​( _no change_.)
08. Specify for each graphical structure the number of initial bars without the showing at the graphic plot ( _added a graphic plot_.)
09. Set the horizontal levels, and describe the properties of each horizontal level ( _no change_.)
10. Set the limit of the separate scale of the indicator window ( _no change_.)
11. Specify the short indicator name ( _name changed_.)

![Fig. 1.9. Ergodic (price,q,r,s,u,ul) indicator](https://c.mql5.com/2/2/mtm09.PNG)

Fig. 1.9. Ergodic (price,q,r,s,u,ul) indicator

##### Configurations (changes)

The code "Blau\_TSI.mq5" has been changed in the following ways.

1\. The short description of the mql5-program is changed:

```
#property description "Ergodic Oscillator (William Blau)"       // description
```

2\. An input parameter has been added:

```
input int    ul=3; // ul- period of a Signal Line
```

3\. (in configuration 11) change is made to the short name of the indicator:

```
//--- short indicator name
   string shortname=PriceName(AppliedPrice)+","+string(q)+","+string(r)+","+string(s)+","+string(u)+","+string(ul);
   IndicatorSetString(INDICATOR_SHORTNAME,"Blau_Ergodic("+shortname+")");
```

##### Configurations (changes): Graphic plots (2, 6)

1\. (in configuration 2) Added one more graphic plot (Signal Line):

```
#property indicator_plots   2              // indicator plots
```

2\. (in configuration 6) a) Changed the properties of the first graphic plot #0 "Ergodic".

Previously, as a way to display the line, we used the (identifier _DRAW\_LINE_), now we use a histogram from the zero line ( _DRAW\_HISTOGRAM_ of the [ENUM\_DRAW\_TYPE](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_draw_type) enumeration)

Changed the color for displaying the lines and the lines width:

```
//--- graphic plot #0 (Main)
#property indicator_label1  "Ergodic"      // graphic plot #0
#property indicator_type1   DRAW_HISTOGRAM // draw as a histogram
#property indicator_color1  Silver         // histogram color
#property indicator_style1  STYLE_SOLID    // line style
#property indicator_width1  2              // line width
```

b) Added a graphic plot #1 "Signal" (Signal Line):

```
//--- graphic plot #1 (Signal Line)
#property indicator_label2  "Signal"       // graphic plot #1
#property indicator_type2   DRAW_LINE      // draw as a line
#property indicator_color2  Red            // line color
#property indicator_style2  STYLE_SOLID    // line style
#property indicator_width2  1              // line width
```

##### Configurations (changes): The indicator buffers (3-5)

The changes in the configuration "indicator array -> indicator buffer -> graphical structure":

1\. (in configuration 3) The number of buffers increased:

```
#property indicator_buffers 11             // number of buffers
```

2\. (in configuration 4) Added an indicator array, which is required to calculate and render the signal line values:

```
double SignalBuffer[];      // Signal line: ul-period EMA of Ergodic (graphic plot #1)
```

3\. (in configuration 5) The relation "indicator array -> indicator buffer -> graphical structure" is changed:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                 // Ergodic
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);               // signal line: ul-period EMA of Ergodic
   // buffers for intermediate calculations
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);        // price array
   SetIndexBuffer(3,MtmBuffer,INDICATOR_CALCULATIONS);          // q-period моментум
   SetIndexBuffer(4,EMA_MtmBuffer,INDICATOR_CALCULATIONS);      // r-period of the 1st EMA
   SetIndexBuffer(5,DEMA_MtmBuffer,INDICATOR_CALCULATIONS);     // s-period of the 2nd EMA
   SetIndexBuffer(6,TEMA_MtmBuffer,INDICATOR_CALCULATIONS);     // u-period of the 3rd EMA
   SetIndexBuffer(7,AbsMtmBuffer,INDICATOR_CALCULATIONS);       // q-period Momentum (absolute value)
   SetIndexBuffer(8,EMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);   // r-period of the 1st EMA (absolute value)
   SetIndexBuffer(9,DEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS);  // s-period of the 2nd EMA (absolute value)
   SetIndexBuffer(10,TEMA_AbsMtmBuffer,INDICATOR_CALCULATIONS); // u-period of the 3rd EMA (absolute value)
```

##### Settings: Number of initial bars without rendering (8)

- The number of initial bars without the rendering of the graphic plot #0 "Ergodic" has not changed. The method of calculation is set forth in Section 1.4.1.1.
- The methods of calculating the number of initial bars without the rendering of the graphic plot #1 "Signal" is the same. The SignalBuffer\[\] array is the result of the smoothing of the significant data of the array MainBuffer\[\] (the smoothing period ul).


Since the indexation of the MainBuffer\[\] array starts from 0 and the significant data in the MainBuffer\[\] array start with the index (q-1)+(r-1)+(s-1)+(u-1), the significant data in the SignalBuffer\[\] array start with the index (q-1)+(r-1)+(s-1)+(u-1)+(ul-1).

The [global](https://www.mql5.com/en/docs/basis/variables/global) variable _begin5_  is declared:

```
int    begin1, begin2, begin3, begin4, begin5; // starting indexes
```

Calculation (complete, additionally see section 1.4.1.1):

```
//---
   begin1=q-1;         //                                    - MtmBuffer[], AbsMtmBuffer[]
   begin2=begin1+r-1;  // or =(q-1)+(r-1)                    - EMA_...[]
   begin3=begin2+s-1;  // or =(q-1)+(r-1)+(s-1)              - DEMA_...[]
   begin4=begin3+u-1;  // or =(q-1)+(r-1)+(s-1)+(u-1)        - TEMA_...[], MainBuffer[]
   begin5=begin4+ul-1; // or =(q-1)+(r-1)+(s-1)+(u-1)+(ul-1) - SignalBuffer[]
   //
   rates_total_min=begin5+1; // rates total min
//--- starting bar index for plot #0
   PlotIndexSetInteger(0,PLOT_DRAW_BEGIN,begin4);
//--- starting bar index for plot #1
   PlotIndexSetInteger(1,PLOT_DRAW_BEGIN,begin5);
```

#### 1.4.3.2. The calculation of the Ergodic (price,q,r,s,u,ul) indicator (alterations and additions to the code "Blau\_TSI.mq5")

##### Calculation: The algorithm

The algorithm for calculating the indicator Ergodic (price,q,r,s,ul):

01. Check whether there is enough data to calculate the indicator.
02. The calculation of the prices array according to the specified [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \- filling of the PriceBuffer\[\] array.
03. The determination of the index bar, from with which to begin/continue the calculation of the q-period Momentum.
04. The calculation of the q-period momentum, and its absolute value - the filling of MtmBuffer\[\] and AbsMtmBuffer\[\] arrays.
05. The first smoothing by the EMA method (period r) - the filling of EMA\_MtmBuffer\[\] and EMA\_AbsMtmBuffer\[\] arrays.
06. The second smoothing by the EMA method (period s) - the filling of DEMA\_MtmBuffer\[\] and DEMA\_AbsMtmBuffer\[\] arrays.
07. The third method smoothing by the EMA method (period u) - the filling of TEMA\_MtmBuffer\[\] and TEMA\_AbsMtmBuffer\[\] arrays.
08. The determination of the index bar, from with which to begin/continue the calculation of the True Strength Index.
09. The calculation of the Ergodic (True Strength Index) - the filling of the MainBuffer\[\] array - the calculation of values ​​for rendering the graphic plot #0.
10. The calculation of the signal line - the smoothing of the Ergodic by the EMA method (period ul) - the filling of the SignalBuffer\[\] array - the calculation of values for the rendering of the graphic plot #1.

The essence of the changes in the algorithm (briefly) a) (see Section 1) the requirement for the minimum size of the indicator input timeseries has changed; b) (see paragraph 10) the calculation of the Signal Line has changed.

##### Calculation (change): Check whether there is enough data to calculate the indicator (1)

There are no changes In the algorithm:

```
//--- rates total
   if(rates_total<rates_total_min) return(0);
```

The values of the [global](https://www.mql5.com/en/docs/basis/variables/global) variable _rates\_total\_min_ has cahnged (the minimum size of the input timeseries of the indicator; calculated in the [OnInit()](https://www.mql5.com/en/docs/basis/function/events#oninit) function at the [Initialization](https://www.mql5.com/en/docs/runtime/event_fire#init) event):

```
   rates_total_min=begin5+1; // the minimum size of the input timeseries of the indicator
```

##### Calculation: signal line (10)

```
//--- calculation of Signal Line (graphic plot #1)
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin4,ul,MainBuffer,SignalBuffer);
```

### 2\. Stochastic Momentum

The considered indicators (see the attachment) are divided into two groups.

I. Indicators, based on the Stochastic:

1. **Blau\_TStoch.mq5** \- Stochastic (q-period Stochastic; smoothed q-period Stochastic);
2. **Blau\_TStochI.mq5** \- Stochastic Index (normalized smoothed q-period Stochastic);
3. **Blau\_TS\_Stochastic.mq5** \- Stochastic TS-oscillator (based on the index of the Stochastic).

II. Indicators, based on the Stochastic Momentum:

1. **Blau\_SM.mq5** \- Stochastic Momentum (q-period Stochastic Momentum; smoothed q-period Stochastic Momentum);
2. **Blau\_SMI.mq5** \- Stochastic Momentum Index (normalized smoothed q-period Momentum);
3. **Blau\_SM\_Stochastic.mq5** \- Stochastic SM-Oscillator (based on the Stochastic Momentum Index).

**2.1. Indicators based on the Stochastic**

The "User's Guide to the MetaTrader client terminal", in the section " [Analysis/Technical Indicators/Oscillators/Stochastic Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so")" provides a description of the built-in client terminal MetaTrader 5 of the technical indicators of the Stochastic Oscillator and the ways of its use in technical analysis (see also [iStochastic](https://www.mql5.com/en/docs/indicators/istochastic).)

**2.1.1. George Lane's Stochastic Oscillator**

**Stochastic**, **stochastic oscilliator** ( [Stochastic, Stochastic Oscillator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/so")) \- is an indicator, which shows the price, in relation to the price fluctuation for the previous q periods. The author and popularizer of the indicator is George Lane.

Distinguish:

- _Fast Stochastic_, sometimes called %K;
- _Slow Stochastic_ (Signal Line), sometimes called %D.

**The formula of Stochastic by** **George Lane**:

```
           price - LL(q)
%K = 100 * -------------
           HH(q) - LL(q)
```

```
%D = SMA(%k, ul)
```

where:

- % K - Fast Stochastic;
- % D - Slow Stochastic (Signal Line);
- price - price \[closing\] of the current period;
- q - the number of time periods of the prices chart used in calculation of the Stochastic;
- HH (q) - the maximum value for the previous q periods of the highest prices for the period q;
- LL (q) - the minimum value for the previous q periods of the lowest price for the period q;
- SMA (% K, ul) - the simple moving average of order ul, applied to the fast stochastic (% K).

**According to the interpretation of George Lane**, the basic idea is that during the trend of a price increase (upward trend), the price tends to stop, close to the previous maximums. With the trend of price decrease (downward trend), the price tends to stop, close to the previous minimums.

**2.1.2. William Blau's Stochastic Oscillator**

![Fig. 2.1. William Blau's indicators, based on the Stochastic](https://c.mql5.com/2/2/stoch01.PNG)

Fig. 2.1. William Blau's indicators, based on the Stochastic

#### 2.1.2.1. Stochastic

**Stochastic** \- is the distance from the price \[closing\] of the current period to the lowest point of the range of price fluctuations, for the previous q periods. _The value of the q-period stochastic shows_ by how much the price is shifted, relative to the lowest point of the q-period range of price fluctuations. The values ​​of the q-period Stochastic are positive or equal to zero.

![Fig. 2.2. Definition of the Stochastic](https://c.mql5.com/2/2/stoch02__1.PNG)

Fig. 2.2. Definition of the Stochastic

**The formula of the q-period Stochastic**:

```
stoch(price,q) = price - LL(q)
```

where:

- price - price \[closing\] of the current period;
- q - the number of time periods of the prices graph, involved in the calculation of the stochastic;
- LL (q) - the minimum value, for the previous q periods, of the lowest price for the period q.

**The formula of the smoothed q-period Stochastic**:

```
TStoch(price,q,r,s,u) = EMA(EMA(EMA( stoch(price,q) ,r),s),u)
```

where:

- price - price of \[closing\] - the price base of the price chart;
- q - the number bars, used in the calculation of the Stochastic;
- stoch(price,q)=price-LL(q) - q-period Stochastic;
- EMA (stoch (price,q),r) - first smoothing - EMA of period r, applied to the q-period stochastic;
- EMA (EMA(..., r),s) - the second smoothing - EMA of period s, applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - EMA of period u, applied to the result of the 2nd smoothing.

#### **TStoch(price,q,r,s,u) - Stochastic. Specification**

- **File name**: **Blau\_TStoch.mq5**
- **Name**: Stochastic Indicator (q-period Stochastic; smoothed q-period Stochastic), according to William Blau.
- **Input parameters**:

  - q - period, for which the stochastic is calculated (by default q = 5);
  - r -period of the 1st EMA, applied to the Stochastic (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**2.1.2.2. The Stochastic Index**

The Stochastic Index indicator is the normalized smoothed q-period Stochastic.

The values ​​of the smoothed q-period Stochastic are mapped to a percentage format (the interval \[0, 100\]). Each value of the smoothed q-period Stochastic is normalized by the value of the q-period price range. _The normalization allows_ to interpret the value of the smoothed normalized q-period Stochastic as the degree of the overbought/oversold states of the market.

**The formula of the Stochastic Index**:

```
                         100 * EMA(EMA(EMA( price-LL(q) ,r),s),u)       100 * TStoch(price,q,r,s,u)
TStochI(price,q,r,s,u) = ---------------------------------------- = ----------------------------------
                            EMA(EMA(EMA( HH(q)-LL(q) ,r),s),u)      EMA(EMA(EMA( HH(q)-LL(q) ,r),s),u)
```

```
if EMA(EMA(EMA(HH(q)-LL(q),r),s),u)=0, then TStochI(price,q,r,s,u)=0
```

where:

- price - price of \[closing\] - the price base of the price chart;
- q - the number bars, used in the calculation of the Stochastic;
- LL (q) - the minimum value of the lowest price for the period q;
- HH (q) - the maximum value of the highest price for the period q;
- stoch(q)=price-LL(q) - q-period Stochastic;
- TStoch(price,q,r,s,u) - three times smoothed q-period Stochastic;
- HH(q)-LL(q) - q-period Price Range;
- EMA (..., r) - the first smoothing - the EMA(r), applied to:


1. to the q-period Stochastic;
2. to the q-period Price Range;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - EMA(u), applied to the result of the 2nd smoothing.

**TStochI(price,q,r,s,u) - Stochastic Index. Specification**

- **File name**: **Blau\_TStochI.mq5**
- **Name**: Stochastic Index (normalized smoothed q-period Stochastic), according to William Blau.
- **Input parameters**:

  - q - period, for which the stochastic is calculated (by default q = 5);
  - r -period of the 1st EMA, applied to the Stochastic (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphic plot - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) Two-levels (by default 40 and 60) - add/remove a level; change the value and description of the level, change the style of the rendering of the levels (the "Levels" tab);
  - change the lower (by default 0), and the upper (by default 100) limits of the scale of the separate indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

#### 2.1.2.3. Stochastic Oscillator

The definition of the Stochastic Oscillator:

```
TS_Stochastic(price,q,r,s,u) = TStochI(price,q,r,s,u)
```

```
SignalLine(price,q,r,s,u,ul) = EMA( TS_Stochastic(price,q,r,s,u) ,ul)
```

where:

- TS\_Stochastic() - Fast Stochastic, %k - Stochastic Index TStochI(price,q,r,s,u);
- SignalLine() - Slow Stochastic (Signal Line),% d - EMA of period ul, applied to the Fast Stochastic (% k);
- ul - period EMA signal line - according to William Blau, the ul value must be equal to the period of the last significant (> 1) EMA fast stochastic.

**TS\_Stochastic(price,q,r,s,u,ul) - Stochastic Oscillator. Specification**

- **File name**: **Blau\_TS\_Stochastic.mq5**
- **Name**: Stochastic Oscillator (based on the Stochastic Index), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - Fast Stochastic (stochastic index),% k:

    - q - period, for which the Stochastic is calculated (by default q = 5);
    - r -period of the 1st EMA, applied to Stochastic (by default r = 20);
    - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
    - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - Slow Stochastic (Signal Line),% d:

    - ul - period EMA Signal Line, applied to the Fast Stochastic (by default ul = 3);

  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphical plotting - the color, thickness, line style (the "Colors" tab);
  - two levels (by default 40 and 60) - add/remove a level; change the value and description of the level; change the style of the rendering of the levels (the "Levels" tab);
  - change the lower (by default 0), and the upper (by default 100) limits of the scale of the separate indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - ul>0. If ul = 1, then the Slow Stochastic (Signal line) and the Fast Stochastic lines are the same;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

#### 2.1.2.4. Continuity

William Blau's Stochastic Oscillator includes the Stochastic Oscillator by George Lane. In order for the TS\_Stochastic (William Blau) to correspond to the standard Stochastic Oscillator (George Lane), implemented in MetaTrader 5, the following must be specified:

```
TS_Stochastic( price=Close, q=KPeriod, r=1, s=1, u=1, ul=DPeriod )
```

```
Stochastic( KPeriod=q, DPeriod=ul, Slowing=1, price="Low/High", method="Exponential" )
```

![Fig. 2.3. William Blau Stochastic Oscillator contains George Lane's Stochastic Oscillator](https://c.mql5.com/2/2/stoch03__1.PNG)

Fig. 2.3. William Blau Stochastic Oscillator contains George Lane's Stochastic Oscillator

#### 2.1.2.5. The code of the Stochastic Oscillator

On the example of the indicator TS\_Stochastic (price,q,r,s,u,ul):

1) The relation between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);              // fast Stochastic
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);            // slow Stochastic: ul-period EMA of the fast Stochastic
   // buffers, used for intermediate calculations
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);      // price array
   SetIndexBuffer(3,LLBuffer,INDICATOR_CALCULATIONS);         // min value (q bars)
   SetIndexBuffer(4,HHBuffer,INDICATOR_CALCULATIONS);         // max value (q bars)
   SetIndexBuffer(5,StochBuffer,INDICATOR_CALCULATIONS);      // q-period Stochastic
   SetIndexBuffer(6,EMA_StochBuffer,INDICATOR_CALCULATIONS);  // r-period of the 1st EMA
   SetIndexBuffer(7,DEMA_StochBuffer,INDICATOR_CALCULATIONS); // s-period of the 2nd EMA
   SetIndexBuffer(8,TEMA_StochBuffer,INDICATOR_CALCULATIONS); // u-period of the 3rd EMA
   SetIndexBuffer(9,HHLLBuffer,INDICATOR_CALCULATIONS);       // q-period price range
   SetIndexBuffer(10,EMA_HHLLBuffer,INDICATOR_CALCULATIONS);  // r-period of the 1st EMA (price range)
   SetIndexBuffer(11,DEMA_HHLLBuffer,INDICATOR_CALCULATIONS); // s-period of the 2nd EMA (price range)
   SetIndexBuffer(12,TEMA_HHLLBuffer,INDICATOR_CALCULATIONS); // u-period of the 3rd EMA (price range)
```

2) The calculation algorithm for the q-period Stochastic and the q-period Price Range:

```
   // calculation of StochBuffer[], HHLLBuffer[], LLBuffer[], HHBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      // LLBuffer[] - search for the minimal price (q bars)
      // HHBuffer[] - search for the maximal price (q bars)
      min=1000000.0;
      max=-1000000.0;
      for(k=i-(q-1);k<=i;k++)
        {
         if(min>Low[k])  min=Low[k];
         if(max<High[k]) max=High[k];
        }
      LLBuffer[i]=min;
      HHBuffer[i]=max;
      // StochBuffer[] - q-period Stochastic
      StochBuffer[i]=PriceBuffer[i]-LLBuffer[i];
      // HHLLBuffer[] - q-period price range
      HHLLBuffer[i]=HHBuffer[i]-LLBuffer[i];
     }
```

**2.2. Indicators, based on the Stochastic Momentum**

![Fig. 2.4. William Blau's indicators, based on the Stochastic Momentum](https://c.mql5.com/2/2/stoch04__1.PNG)

Fig. 2.4. William Blau's indicators, based on the Stochastic Momentum

**2.2.1. Stochastic Momentum**

The **Stochastic Momentum** (Stochastic Momentum, SM) - is the distance from the price of the current period to the middle of the price range over the previous q periods. _The value of the q-period Stochastic Momentum shows_ the position of price in the price range.

_The sign of the q-period stochastic momentum shows_ the price position, relative to the middle of the q-period price range: a positive Stochastic Momentum - the price is above the midpoint, a negative - the price is below the midpoint.

![Fig. 2.5. The definition of the Stochastic Momentum](https://c.mql5.com/2/2/stoch05__1.PNG)

Fig. 2.5. The definition of the Stochastic Momentum

**The formula of the q-period Stochastic Momentum**:

```
sm(price,q) = price - 1/2 * [LL(q) + HH(q)]
```

where:

- price - price \[closing\] of the current period;
- q - the number of bars, used in calculation of the Stochastic Momentum;
- LL (q) - the minimum value of the lowest price for the period q;
- HH (q) - the maximum value of the highest prices for the period q;
- 1/2\* \[LL(q)+HH (q)\] - the middle of the q-period price range.

**The formula of the smoothed q-period Stochastic Momentum**:

```
SM(price,q,r,s,u) = EMA(EMA(EMA( sm(price,q) ,r),s),u)
```

where:

- price - price of \[closing\] - the price base of the price chart;
- q - the number of bars, used in the calculation of the Stochastic momentum;
- sm(price,q)=price-1/2\*\[LL(q)+HH(q)\] - the q-period Stochastic Momentum;
- EMA (sm(price,q),r) - the first smoothing - the EMA(r), applied to the q-period Stochastic Momentum;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA(EMA(EMA(sm(q),r),s),u) - the third smoothing - the EMA(u), applied to the result of the 2nd smoothing.

#### 2.2.1.2. SM(price,q,r,s,u) - Stochastic Momentum. Specification

- **File name**: **Blau\_SM.mq5**
- **Name**: Stochastic Momentum Indicator (q-period stochastic momentum, smoothed q-period stochastic momentum), according to William Blau.
- **Input parameters**:

  - q - the period by which the stochastic momentum is calculated (by default q = 5);
  - r - period of the 1-st EMA, applied to the Stochastic Momentum (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**2.2.2. The Stochastic Momentum Index**

The **Stochastic Momentum Index** (SMI) - is an indicator of a normalized stochastic rate (normalized smoothed q-period stochastic momentum). The values ​​of the q-period smoothed Stochastic Momentum is given in the percentage format (interval of display \[-100, 100\]).

Each value of the smoothed q-period Stochastic Mmomentum is normalized by the value of half of the q-period range of price fluctuations. _Normalization allows for the_ interpretation of the value of SMI as a degree of an overbought level (positive value) or oversold level (negative) of the market.

**The formula of the Stochastic Momentum Index**:

```
                     100 * EMA(EMA(EMA( price-1/2*[LL(q)+HH(q)] ,r),s),u)           100 * SM(price,q,r,s,u)
SMI(price,q,r,s,u) = ---------------------------------------------------- = ----------------------------------------
                           EMA(EMA(EMA( 1/2*[HH(q)-LL(q)] ,r),s),u)         EMA(EMA(EMA( 1/2*[HH(q)-LL(q)] ,r),s),u)
```

```
if EMA(EMA(EMA(1/2*[HH(q)-LL(q)],r),s),u)=0, then SMI(price,q,r,s,u)=0
```

where:

- price - price of \[closing\] - the price base of the price chart;
- LL (q) - the minimum value of the lowest price for the period q;
- HH (q) - the maximum value of the highest prices for the period q;
- sm(price,q)=price-1/2\*\[LL(q)+HH(q)\] - the q-period Stochastic Momentum;
- SM(price,q,r,s,u) - three times smoothed q-period Stochastic Momentum;
- HH(q)-LL(q) - q-period price range;
- 1/2\* \[LL (q)+HH(q)\] - the middle of the q-period price range;
- 1/2\*\[HH(q)-LL(q)\] - half of the q-period of the price range;
- EMA (..., r) - the first smoothing - EMA(r), applied to:

1) the q-period Stochastic Momentum

2) half of the q-period Price Range;
- EMA (EMA(..., r),s) - the second smoothing - EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - EMA(u), applied to the result of the 2nd smoothing.

#### 2.2.2.2. SMI(price,q,r,s,u) - Stochastic Momentum Index. Specification

- **File name**: **Blau\_SMI.mq5**
- **Name**: Stochastic Momentum Index (normalized smoothed q-period Stochastic Momentum) according to William Blau.
- **Input parameters**:

  - q - the period by which the Stochastic Momentum is calculated (by default q = 5);
  - r - period of 1-st EMA, applied to Stochastic Momentum (by default r = 20);
  - s - period of the 2nd EMA, applied to the results of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the results of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphical plotting - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) Two-levels (by default -40 and +40) - add/remove a level; change the value and description of the level, change the style of the rendering of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**2.2.3. The Stochastic Oscillator**

The definition of the Stochastic Oscillator:

```
SM_Stochastic(price,q,r,s,u) = SMI(price,q,r,s,u)
```

```
SignalLine(price,q,r,s,u,ul) = EMA( SM_Stochastic(price,q,r,s,u) ,ul)
```

where:

- SM\_Stochastic() - Stochastic Momentum Index SMI(price,q,r,s,u);
- SignalLine() - Signal Line - EMA of period, ul, applied to the Stochastic Momentum Index;

- ul - period EMA signal line - according to William Blau, the ul value must be equal to the period of the last significant (>1) EMA index of the stochastic rate.

#### **2.2.3.1.** SM\_Stochastic(price,q,r,s,u,ul) - Stochastic Oscillator. Specification

- **File name**: **Blau\_SM\_Stochastic.mq5**
- **The name**: Stochastic Oscillator (based on the Stochastic Momentum), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - the Stochastic Momentum Index:

    - q - the period by which the stochastic momentum is calculated (by default q = 5);
    - r - period of the 1st EMA, applied to the Stochastic Momentum (by default r = 20);
    - s - period of the 2nd EMA, applied to result of the 1st smoothing (by default s = 5);
    - u - period of the 3rd EMA, applied to result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - the signal line:

    - ul - period EMA signal line, with regards to the index of the stochastic rate (by default ul = 3);

  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphical plotting - the color, thickness, line style (the "Colors" tab);
  - two levels (by default -40 and +40) - add/remove a level; change the value and description of the level, change the rendering style of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - ul>0. If ul = 1, then the signal line coincides with the index of the stochastic rate;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

**2.2.4. The code of the Stochastic Oscillator**

The SM\_Stochastic (price, q, r, s, u, ul):

1) The relation between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                   // Stochastic Momentum Index
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);                 // Signal Line: ul-period EMA of Stochastic Momentum Index
   // buffers for intermediate calculations (not used for plotting)
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);          // price array
   SetIndexBuffer(3,LLBuffer,INDICATOR_CALCULATIONS);             // minimal price value (q bars)
   SetIndexBuffer(4,HHBuffer,INDICATOR_CALCULATIONS);             // maximal price value (q bars)
   SetIndexBuffer(5,SMBuffer,INDICATOR_CALCULATIONS);             // q-period Stochastic Momentum
   SetIndexBuffer(6,EMA_SMBuffer,INDICATOR_CALCULATIONS);         // r-period of the 1st EMA
   SetIndexBuffer(7,DEMA_SMBuffer,INDICATOR_CALCULATIONS);        // s-period of the 2nd EMA
   SetIndexBuffer(8,TEMA_SMBuffer,INDICATOR_CALCULATIONS);        // u-period of the 3rd EMA
   SetIndexBuffer(9,HalfHHLLBuffer,INDICATOR_CALCULATIONS);       // half of price range (q bars)
   SetIndexBuffer(10,EMA_HalfHHLLBuffer,INDICATOR_CALCULATIONS);  // r-period of the 1st EMA (half of price range)
   SetIndexBuffer(11,DEMA_HalfHHLLBuffer,INDICATOR_CALCULATIONS); // s-period of the 2nd EMA (half of price range)
   SetIndexBuffer(12,TEMA_HalfHHLLBuffer,INDICATOR_CALCULATIONS); // u-period of the 3rd EMA (half of price range)
```

2) The algorithm of calculation of the q-period Stochastic Momentum and half of the q-period price range:

```
//--- calculation of q-period Stochastic Momentum and half of price range (q bars)
   if(prev_calculated==0)       // at first call
     {
      pos=begin1;               // starting from 0
      for(i=0;i<pos;i++)        // pos values
        {
         SMBuffer[i]=0.0;       // zero values
         HalfHHLLBuffer[i]=0.0; //
         LLBuffer[i]=0.0;       //
         HHBuffer[i]=0.0;       //
        }
     }
   else pos=prev_calculated-1;  // overwise calculate only last value
   // calculation of SMBuffer[], HalfHHLLBuffer[], LLBuffer[], HHBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      // calculation of LLBuffer[] - search for the minimal price (q bars)
      // calculation of HHBuffer[] - search for the maximal price (q bars)
      min=1000000.0;
      max=-1000000.0;
      for(k=i-(q-1);k<=i;k++)
        {
         if(min>Low[k])  min=Low[k];
         if(max<High[k]) max=High[k];
        }
      LLBuffer[i]=min;
      HHBuffer[i]=max;
      // calculation of SMBuffer[] - q-period Stochastic Momentum
      SMBuffer[i]=PriceBuffer[i]-0.5*(LLBuffer[i]+HHBuffer[i]);
      // calculation of HalfHHLLBuffer[] - half of price range (q bars)
      HalfHHLLBuffer[i]=0.5*(HHBuffer[i]-LLBuffer[i]);
     }
```

### 3\. The indicator of deviation from the trend

The considered indicators (see the attachment) are divided into two groups.

I. Indicators, based on a deviation from the market trend.

1. **Blau\_MDI.mq5**\- An indicator of an Average Deviation from the trend (mean deviation, moving average deviation);
2. **Blau\_Ergodic\_MDI.mq5**\- Ergodic MDI oscillator (based on the mean deviation).

II. Indicators, based on the Moving Averages Convergence/Divergence.

1. **Blau\_MACD.mq5**\- Moving Averages Convergence/Divergence (MACD; smoothed MACD);
2. **Blau\_Ergodic\_MACD.mq5**\- Ergodic MACD-Oscillator (based on the MACD indicator).

**3.1. Indicators, based on the deviation from the market trends**

![Fig. 3.1. William Blau's indicators are based on a deviation from the market trends](https://c.mql5.com/2/2/md01.PNG)

Fig. 3.1. William Blau's indicators are based on a deviation from the market trends

**3.1.1. The Mean Deviation Indicator**

**The mean deviation from the trend** is the distance between the price and the EMA (exponentially smoothed moving average) of period r, applied to the price.

**The trend of market development**: the EMA(r), applied to the price is used to determine the upward trend (exponential increase), or downtrend (exponential decrease) of prices.

The moving average smooths out the price curve, but a slight increase of the moving average period leads to a lag, which is clearly visible at the points of price reversal (see additionally 1.1.1, Fig. 1.2). _The value of the average deviation from the trend shows_ the distance to the EMA(r), applied to the price.

_The sign of the average deviation from the trend shows_ the position of the price, relative to the EMA(r) applied to the price: a positive deviation from the trend - the price is higher than the exponent, negative - the price is lower than the exponent.

**The formula for the mean deviation from the trend**:

```
md(price,r) = price - EMA(price,r)
```

where:

- price - price of the current period;
- EMA (price,r) - the market trend - EMA of the r period, applied to the price.

See in the "User's Guide to the client terminal MetaTrader", in the section ["Anatyics/Technical Indicators/Trend Indicators"](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators"):

1. [Double Exponential Moving Average, DEMA;](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/dema "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/dema")

2. [Triple Exponential Moving Average, TEMA.](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/tema "https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/tema")

A similar index is used by Alexander Elder in his Bears Power and Bulls Power indicators. See in the "User's Guide to the MetaTrader client terminal" in the section ["Analysis/Technical Indicators/Oscillators"](https://www.metatrader5.com/en/terminal/help/indicators/oscillators "https://www.metatrader5.com/en/terminal/help/indicators/oscillators"):

1. [Bears Power;](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bears")
2. [Bulls Power.](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/bulls")

The **indicator of the mean deviation from the trend** (Mean Deviation Index, MDI) - is a smoothed average deviation from the market trend.

**The formula of the indicator of the mean deviation from the trend**:

```
MDI(price,r,s,u) = EMA(EMA( md(price,r) ,s),u) = EMA(EMA( price-EMA(price,r) ,s),u)
```

where:

- price - price of \[closing\] - the price base of the price chart;
- EMA (price, r) - the market trend - the first smoothing of the EMA(r), applied to the price;
- md (price,r) = price-EMA (price,r) - the mean deviation from the trend - the deviation of the price from the EMA(r), applied to the price;
- EMA (md (price, r), s) - the second smoothing - the EMA(s), applied to the mean deviation from the trend;
- EMA (EMA (md(price,r),s),u) - the third smoothing - the EMA(u), applied to the result of the second smoothing.

#### 3.1.1.3. MDI(price,r,s,u) - Mean Deviation Index. Specification

- **File name**: **Blau\_MDI.mq5**
- **Name**: The indicator of the mean deviation from the market (mean deviation; a smoothed mean deviation), according to William Blau.
- **Input parameters**:

  - r - period of the 1st EMA, applied to the price (by default r=20);
  - s - period of the 2nd EMA, applied to mean deviation (by default, s = 5);
  - u - period of the 3rd EMA, applied to result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - r>1;
  - s>0, u>0. If s or u are equal to 1, the EMA smoothing is not used;
  - the minimum size of the prices array = (r+s+u-3+1).

**3.1.2. Ergodic MDI-oscillator**

Definition of the Ergodic MDI-oscillator:

```
Ergodic_MDI(price,r,s,u) = MDI(price,r,s,u)
```

```
SignalLine(price,r,s,u,ul) = EMA( Ergodic_MDI(price,r,s,u) ,ul)
```

where:

- Ergodic\_MDI() - Ergodic - Mean Deviation Index MDI(price,r,s,u);
- The SignalLine() -a Signal line - EMA of period ul, applied to the Ergodic;
- ul - an EMA period of a Signal line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic.

#### 3.1.2.2. Ergodic\_MDI(price,r,s,u,ul) - Ergodic MDI-oscillator. Specification

- **File name**: **Blau\_Ergodic\_MDI.mq5**
- **Name**: The Ergodic MDI-oscillator (based on the Mean Deviation Index), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - Ergodic (the indicator of the mean deviation from the trend):

    - r - period of the 1st EMA, applied to the price (by default r=20);
    - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default, s = 5);
    - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - Signal Line:

    - ul - period EMA signal line, applied to the Ergodic (by default ul = 3);

  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the style of the rendering of each graphical structure - the color, width, line style (the "Colors" tab).

- **Limitations**:
  - r>1;
  - s>0, u>0. If s or u are equal to 1, the EMA smoothing is not used;
  - ul>0. If ul = 1, then the Signal line and the Ergodic lines are the same;
  - the minimum size of the prices array = (r+s+u+ul-4+1).

**3.1.3. The code of the Ergodic oscillator**

As example, let's consider the Ergodic\_MDI (price,r,s,u,ul) indicator:

1) The relation between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);              // ergodic: u-period 3rd EMA
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);            // signal line: ul-period EMA of Ergodic
   // buffers for intermediate calculations; not used for plotting
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);     // price array
   SetIndexBuffer(3,EMA_PriceBuffer,INDICATOR_CALCULATIONS); // r-period 1st EMA (price)
   SetIndexBuffer(4,MDBuffer,INDICATOR_CALCULATIONS);        // среднее отклонение
   SetIndexBuffer(5,DEMA_MDBuffer,INDICATOR_CALCULATIONS);   // s-period 2nd EMA
```

2) The algorithm for calculating the mean deviation:

```
//--- calculation of the mean deviation
   if(prev_calculated==0)      // at first call
     {
      pos=begin2;              // starting from 0
      for(i=0;i<pos;i++)       // pos data
         MDBuffer[i]=0.0;      // zero values
     }
   else pos=prev_calculated-1; // overwise calculate only last bar
   // r-period 1st EMA: calculation of EMA_PriceBuffer[]
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,r,PriceBuffer,EMA_PriceBuffer);
   // calculation of MDBuffer[]
   for(i=pos;i<rates_total;i++)
      MDBuffer[i]=PriceBuffer[i]-EMA_PriceBuffer[i];
```

**3.2. Indicators, based on the Moving Average Convergence/Divergence**

![Fig. 3.2. Indicators by William Blau are based on the Moving Averages Convergence/Divergence](https://c.mql5.com/2/2/md02.PNG)

Fig. 3.2. Indicators by William Blau are based on the Moving Averages Convergence/Divergence

**3.2.1. The indicator of Moving Averages Convergence/Divergence**

The **Moving Average Convergence/Divergence**( [Moving Average Convergence/Divergence, MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd")) \- is the difference between two exponentially smoothed moving averages: the fast EMA(s) the slow EMA(r), applied to the price.

_The sign MACD shows the_ position of the Fast EMA(s), relative to the slow EMA(r): a positive MACD - EMA(s) is above the EMA(r), a negative MACD - EMA(s) is below EMA(r). _Change of the MACD by the absolute value_: an increase\|MACD\| indicates the discrepancy between the moving averages, a decrease\|MACD\| indicates a convergence of the moving averages.

**The formula of the Moving Average Convergence/Divergence:**

```
macd(price,r,s) = EMA(price,s) - EMA(price,r)
```

```
s < r
```

where:

- price - price \[closing\] of the current period;
- EMA(price,r) - Slow EMA(r), applied to the price;
- EMA(price,s) - Fast EMA(s), applied to the price.

The **MACD indicator** show the relationship between the fast and the slow exponential averages (smoothed convergence/divergence of the moving averages).

The **formula of the MACD indicator**:

```
MACD(price,r,s,u) = EMA( macd(price,r,s) ,u) = EMA( EMA(price,s)-EMA(price,r) ,u)
```

```
s < r
```

where:

- price - price of \[closing\] - the price of the price chart;
- EMA(price,r) - the first smoothing - the slow exponential of the EMA(r), applied to the price;
- EMA(price,s) - the second smoothing - the fast EMA(s), s, applied to the price;
- macd(r,s)=EMA(price,s)-EMA (price,r) - the MACD;
- EMA(macd (r,s),u) - the third smoothing - the EMA(u), applied to the MACD: a fast EMA (price,s) and a slow EMA (price,r).

#### 3.2.1.1. MACD(price,r,s,u) - the Moving Average Convergence/Divergence indicator. Specification

- **File name**: **Blau\_MACD.mq5**
- **Name**: The MACD indicator (MACD;smoothed MACD), according to William Blau.
- **Input parameters**:

  - r - period of the 1st  EMA (slow), applied to the price (by default r = 20);
  - s - period of the 2nd EMA (fast), applied to the price (by default s = 5)
  - u - period of the 3rd EMA, applied to the moving averages convergence/divergence (by default u = 3);
  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - r>1, s>1;
  - s <r (limit by the requirements of the theory, is not checked on the program level);
  - u>0. If u = 1, smoothing is not performed;
  - the minimum size of the prices array = (\[max(r,s)\]+u-2+1).

**3.2.2. Ergodic MACD-oscillator**

The definition of the Ergodic MACD-oscillator:

```
Ergodic_MACD(price,r,s,u) = MACD(price,r,s,u)
```

```
SignalLine(price,r,s,u,ul) = EMA( Ergodic_MACD(price,r,s,u) ,ul)
```

where:

- Ergodic\_MACD () - Ergodic - is an indicator of moving averages convergence/divergence MACD(price,r,s,u);
- The SignalLine() -a Signal Line - an EMA(ul), applied to the ergodic;
- ul - an EMA period of a signal line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic.

The "User's Guide to the MetaTrader client terminal", in the ["Analytics/Technical Indicators/Oscillators/MACD"](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") section, describes the technical indicator [Convergence/Divergence of the moving averages (MACD)](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd"), built-in in the MetaTrader 5 client terminal, and how to use it in technical analysis (see also [iMACD](https://www.mql5.com/en/docs/indicators/imacd).)

In contrast to the standard MACD, William Blau uses the _exponentially smoothed_ moving average (in the standard MACD the _simple_ moving average is used).

#### 3.2.2.1. Ergodic\_MACD(price,r,s,u,ul) - Ergodic MACD-oscillator. Specification

- **File name**: **Blau\_Ergodic\_MACD.mq5**
- **Name**: Ergodic MACD-oscillator (based on the moving averages convergence/divergence indicator), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - Ergodic (the moving averages convergence/divergence):

    - r - period of the 1st EMA (slow), applied to the price (by default r = 20);
    - s - period of the 2nd EMA (fast) applied to the price (by default s = 5)
    - u - period of the 3rd EMA, applied to the moving averages convergence/divergence (by default u = 3);

  - graphic plot #1 - the Signal Line:

    - ul - period EMA signal line, is applied to the ergodic (by default ul = 3);

  - AppliedPrice - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) (default AppliedPrice=PRICE\_CLOSE).

- **Additionally**:

  - displayed in a separate window;
  - change the style of the rendering of each graphical structure - the color, width, line style (the "Colors" tab).

- **Limitations**:
  - r>1, s>1;
  - s <r (limit by the requirements of the theory, is not checked on the program level);
  - u>0. If u = 1, smoothing is not performed;
  - ul>0. If ul = 1, then the signal line coincides with the ergodic;
  - the minimum size of the prices array =(\[max(r,s)\]+u+ul-3+1).

**3.2.3. The code of the Ergodic MACD-Oscillator**

As example, let's consider the Ergodic\_MACD (price,r,s,u,ul) indicator:

1) The link between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);               // Ergodic: u-period 3rd EMA
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);             // Signal Line: ul-period EMA, applied to Ergodic
   // buffers for intermediate calculations
   SetIndexBuffer(2,PriceBuffer,INDICATOR_CALCULATIONS);      // price array
   SetIndexBuffer(3,EMA1_PriceBuffer,INDICATOR_CALCULATIONS); // r-period 1st EMA (slow), applied to price
   SetIndexBuffer(4,EMA2_PriceBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (fast), applied to price
   SetIndexBuffer(5,MACDBuffer,INDICATOR_CALCULATIONS);       // moving averages convergence/divergence
```

2) The algorithm of moving averages convergence/divergence:

```
//--- calculation of moving average convergence/divergence
   if(prev_calculated==0)      // at first call
     {
      pos=begin2;              //
      for(i=0;i<pos;i++)       // pos
         MACDBuffer[i]=0.0;    // zero values
     }
   else pos=prev_calculated-1; // overwise calculate only last value
   // r-period 1st EMA: calculation of EMA1_PriceBuffer[]
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,r,PriceBuffer,EMA1_PriceBuffer);
   // s-period 2nd EMA: calculation of EMA2_PriceBuffer[]
   ExponentialMAOnBufferWB(rates_total,prev_calculated,begin1,s,PriceBuffer,EMA2_PriceBuffer);
   // calculation of MACDBuffer[]
   for(i=pos;i<rates_total;i++)
      MACDBuffer[i]=EMA2_PriceBuffer[i]-EMA1_PriceBuffer[i];
```

**3.3. Addition**

In calculating the Ergodic MDI-oscillator and the MACD-Oscillator, according to William Blau, the normalization is not used (for reference see pp. 1.2.1, 1.3.1). Therefore, the **Ergodic MDI-Oscillator and the MACD-Oscillator cannot be used to interpret the degree of the overbought or the oversold market**.

For example, the recommendations for using the MACD indicator signals from the "User's Guide to the MetaTrader client terminal" of the ["Analytics/Technical Indicators/Oscillators/MACD"](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") section:

The MACD is also useful as an overbought/oversold indicator. When the shorter moving average pulls away dramatically from the longer moving average (i.e., the MACD rises), it is likely that the security price is overextending and will soon return to more realistic levels.

in this case, from the aspect of technical analysis.

### 4\. Candlestick Momentum

The considered indicators (see the attachment) are divided into two groups.

1. **Blau\_CMtm.mq5**\- is the Candlestick Momentum indicator (momentum of the q-period candlestick; smoothed q-period Candlestick Momentum);
2. The Indexes (normalized smoothed q-period Candlestick Momentum):

   - **Blau\_CMI.mq5**\- the Candlestick Momentum Index (normalization by the absolute value of the q-period Candlestick Momentum);
   - **Blau\_CSI.mq5**\- the Candlestick Index (the normalized by the length q-period Candlestick);

4. The ergodic oscillator of the candlestick
   - **Blau\_Ergodic\_CMI.mq5**\- the Ergodic CMI-Oscillator (based on the Candlestick Momentum Index);
   - **Blau\_Ergodic\_CSI.mq5**\- the Ergodic CSI-Oscillator (based on the Candlestick Index).

![Fig. 4.1. Indicators by William Blau, based on the Candlestick Momentum (normalized by the absolute value of the q-period Candlestick Momentum)](https://c.mql5.com/2/2/cmtm01__1.PNG)

Fig. 4.1. Indicators by William Blau, based on the Candlestick Momentum (normalized by the absolute value of the q-period Candlestick Momentum)

![Fig. 4.2. Indicators by William Blau, based on the Candlestick Momentum (normalized by the length of the q-period Candlestick)](https://c.mql5.com/2/2/cmtm02__2.PNG)

Fig. 4.2. Indicators by William Blau, based on the Candlestick Momentum (normalized by the length of the q-period Candlestick)

**4.1. The Candlestick Momentum**

**4.1.1. The definition of the Candlestick Momentum**

The Momentum (see p. 1.1) - is the difference between the current price (usually, today's closing price) and the previous price (usually yesterday's closing price). The momentum can reflect the price change at any time period of the price graph.

The **Candlestick Momentum** (according to William Blau) - is the difference between the closing price and the opening price, within the same period (within one candlestick). The sign of the Candlestick Momentum shows the direction of the price change: a positive Candlestick Momentum - the price has increased over the period, a negative - the price has decreased over the period.

**The formula of the Candlestick Momentum**:

```
cmtm = close - open
```

where:

- close - the closing price of \[the current\] period of the (candlestick);
- open - the opening price of \[the current\] period of the (candlestick).

From the standpoint of universality, _let's extend the definition of the candlestick momentum_:

1. The Candlestick Momentum can reflect the price change for any time period of the price chart;

2. The price base (the closing price, opening price) can be arbitrary.

![Fig. 4.3. The definition of the q-period Candlestick](https://c.mql5.com/2/2/cmtm03.PNG)

Fig. 4.3. The definition of the q-period Candlestick

**The formula of the q-period Candlestick Momentum**:

```
cmtm(price1,price2,q) = price1 - price2[q-1]
```

where:

- q - is the number of bars of the price chart, used in calculation of the Candlestick Momentum;
- price1 - price \[closing\] at the end of period q;
- price2\[q-1\] - price\[opening\] at the beginning of period q.

The **formula of the smoothed q-period Candlestick Momentum**:

```
CMtm(price1,price2,q,r,s,u) = EMA(EMA(EMA( cmtm(price1,price2,q) ,r),s),u)
```

where:

- q - the number of bars of the price chart, used in calculation the q-period of Candlestick Momentum;
- price1 - price \[closing\] at the end of period q;
- price2 - price\[opening\] at the beginning of period q;
- cmtm(price1,price2,q)=price1-price2\[q-1\] - q-period Candlestick Momentum;
- EMA (cmtm (price1, price2, q), r) - the first smoothing - EMA(r), applied to the q-period Candlestick Momentum;
- EMA (EMA(..., r),s) - the second smoothing - EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - EMA(u), applied to the result of the 2nd smoothing.

**4.1.2. CMtm(price1,price2,q,r,s,u) - Candlestick Momentum indicator. Specification**

- **File name**: **Blau\_CMtm.mq5**
- **Name**: The Candlestick Momentum indicator (smoothed q-period Candlestick Momentum), according to William Blau.
- **Input parameters**:

  - q - the period of Candlestick Momentum (by default q = 1);
  - r - period of the 1st EMA, applied to the q-period Candlestick Momentum (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice1 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[closing\] (by default AppliedPrice=PRICE\_CLOSE);
  - AppliedPrice2 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[opening\] (by default AppliedPrice=PRICE\_OPEN).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**4.2. Normalized Candlestick Momentum**

**4.2.1. Candlestick Momentum Index**

The **Candlestick Momentum Index** (CMI) - is the normalized q-period Candlestick Momentum.

The values of the smoothed momentum of the q-period Candlestick are given as a percentage (mapping interval \[-100, 100\]). Each value of the smoothed momentum of the q-period Candlestick is normalized by the value of the smoothed q-period Candlestick Momentum, taken in the absolute value. _Normalization allows_ the CMI value to be interpreted as a degree of an overbought (positive value) or oversold (negative value) market level.

**The formula for the Candlestick Momentum Index**:

```
                             100 * EMA(EMA(EMA( cmtm(price1,pric2,q) ,r),s),u)          100 * CMtm(price1,pric2,q,r,s,u)
CMI(price1,price2,q,r,s,u) = –––––––––––-------------––––––––-–––––––––––––––– = –––––––––––––––-------------–––-–––––––––––––
                               EMA(EMA(EMA( |cmtm(price1,pric2,q)| ,r),s),u)     EMA(EMA(EMA( |cmtm(price1,pric2,q)| ,r),s),u)
```

```
if EMA(EMA(EMA(|cmtm(price1,pric2,q)|,r),s),u)=0, then CMI(price1,price2,q,r,s,u)=0
```

where:

- q - the number of time periods of the price graph, involved in calculating the momentum of the q-period of the candlestick;
- price1 - price \[closing\] at the end of period q;
- price2 - price\[opening\] at the beginning of period q;
- cmtm(price1,pric2,q)=price1-pric2\[q-1\], - q-period Candlestick Momentum;
- \|cmtm(price1,pric2,q)\| - absolute value of the q-period Candlestick Momentum;
- CMtm (price,q,r,s,u) - three times smoothed q-period Candlestick Momentum;
- EMA (..., r) - first smoothing - the EMA(r), applied to:

1) the q-period Candlestick Momentum

2) the absolute value of the q-period Candlestick Momentum;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - the EMA(u), applied to the result of the 2nds smoothing.

#### 4.2.1.1. CMI(price1,price2,q,r,s,u) - Candlestick Momentum Index. Specification

- **File name**: **Blau\_CMI.mq5**
- **Name**: q-period Candlestick Momentum Index (normalized smoothed q-period Candlestick Momentum; normalization by the absolute value of the q-period Candlestick Momentum), according to William Blau.
- **Input parameters**:

  - q - the period of the Candlestick Momentum (by default q = 1);
  - r - period of the 1st EMA, applied to q-period Candlestick Momentum (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice1 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[closing\] (by default AppliedPrice=PRICE\_CLOSE);
  - AppliedPrice2 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[opening\] (by default AppliedPrice=PRICE\_OPEN).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphical plotting - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) two-levels (default is -25 and +25) - add/remove a level; change the value, the level description, change the rendering style of the levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**4.2.2. The Candlestick Index**

The **Candlestick index** (CSI) - is an indicator of the normalized q-period Candlestick  Momentum (normalized smoothed q-period Candlestick Momentum). The values of the smoothed q-period Candlestick  Momentum are given as a percentage of the scale (mapping interval \[-100, 100\]).

Each value of the smoothed q-period Candlestick Momentum is normalized by the value of the q-period price range (or by the length of the q-period candlestick). _Normalization allows_ to interpret the value of CSI as a degree of an overbought (positive value) or oversold (negative value) market level.

**The formula of the Candlestick Index**:

```
                             100 * EMA(EMA(EMA( cmtm(price1,pric2,q) ,r),s),u)    100 * CMtm(price1,pric2,q,r,s,u)
CSI(price1,price2,q,r,s,u) = –––––––––––––––––––-–––-------------––––––––––––– = ––––––––––––––––--––-–––––––––––––
                                    EMA(EMA(EMA( HH(q)-LL(q) ,r),s),u)           EMA(EMA(EMA( HH(q)-LL(q) ,r),s),u)
```

```
if EMA(EMA(EMA(HH(q)-LL(q),r),s),u)=0, then CSI(price1,price2,q,r,s,u)=0
```

where:

- q - the number of bars of the price chart, used in calculation of the q-period Candlestick Momentum;
- price1 - price \[closing\] at the end of period q;
- price2 - price\[opening\] at the beginning of period q;
- cmtm(price1,pric2,q)=price1-price2\[q-1\] - q-period Candlestick Momentum;
- LL (q) - the minimum value of the lowest price for the period q;
- HH(q) - the maximum value of the highest price for period q
- HH(q)-LL(q) - q-period price range (the length of the q-period candlestick);
- CMtm(price1,pric2,q,r,s,u) - three times smoothed q-period Candlestick Momentum;
- EMA (..., r) - the first smoothing - the EMA(r), applied to:

1) the q-period Candlestick Momentum,

2) the q-period Price Range (or the length of the q -period candlestick);
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - the EMA(u), applied to the result of the 2nd smoothing.

#### 4.2.2.1. CSI(price1,price2,q,r,s,u) - Candlestick Index. Specification

- **File name**: **Blau\_CSI.mq5**
- **Name**: q-period Candlestick Index (normalized smoothed q-period Candlestick Momentum; normalization by the length of the q-period candlestick), according to William Blau.
- **Input parameters**:

  - q - the period for which the q-period Candlestick Momentum is calculated (by default q = 1);
  - r - period of the 1st EMA, applied to the q-period candlestick Momentum (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);
  - AppliedPrice1 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[closing\] (by default AppliedPrice=PRICE\_CLOSE);
  - AppliedPrice2 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[opening\] (by default AppliedPrice=PRICE\_OPEN).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphical plotting - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) two-levels (default is -25 and +25) - add/remove a level; change the value, the level description, change the rendering style of the levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**4.3. The Ergodic Oscillators of the candlestick**

**4.3.1. The Ergodic CMI-oscillator**

The definition of the Ergodic CMI-oscillator:

```
Ergodic_CMI(price1,pric2,q,r,s,u) = CMI(price1,pric2,q,r,s,u)
```

```
SignalLine(price1,pric2,q,r,s,u,ul) = EMA( Ergodic_CMI(price1,pric2,q,r,s,u) ,ul)
```

where:

- Ergodic\_CMI() - Ergodic - Candlestick Momentum Index CMI(price1,price2,q,r,s,u);
- The SignalLine() -a Signal Line - EMA(ul), applied to the Ergodic;
- ul - an EMA period of a signal line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic.

#### Ergodic\_CMI(price1,pric2,q,r,s,u,ul) - ergodic CMI-oscillator. Specification

- **File name**: **Blau\_Ergodic\_CMI.mq5**
- **Name**: Ergodic CMI-Oscillator (based on the Candlestick Momentum Index), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - Ergodic (Candlestick Momentum Index):

    - q - the period of Candlestick Momentum (by default q = 1);
    - r - period of the 1st EMA, applied to q-period Candlestick Momentum (by default r = 20);
    - s - period of the 2nd EMA, applied to result of the 1st smoothing (by default s = 5);
    - u - period of the 3rd EMA, applied to result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - the Signal Line:

    - ul - period of Signal Line, applied to the Ergodic (by default ul = 3);

  - AppliedPrice1 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[closing\] (by default AppliedPrice=PRICE\_CLOSE);
  - AppliedPrice2 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[opening\] (by default AppliedPrice=PRICE\_OPEN).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphical plotting - the color, thickness, line style (the "Colors" tab);
  - two levels (by default -25 and +25) - add/remove a level, change the value, level description, change the rendering style of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - ul>0. If ul = 1, then the signal line coincides with the ergodic;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

#### The code of the Ergodic CMI-oscillator

As example, let's consider the Ergodic\_CMI (price1,price2,r,s,u,ul) indicator:

1) The relation between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                  // Ergodic
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);                // Signal Line: EMA(ul), applied to Ergodic
   // buffers for intermediate calculations
   SetIndexBuffer(2,Price1Buffer,INDICATOR_CALCULATIONS);        // price array [close]
   SetIndexBuffer(3,Price2Buffer,INDICATOR_CALCULATIONS);        // price array [open]
   SetIndexBuffer(4,CMtmBuffer,INDICATOR_CALCULATIONS);          // q-period Candlestick Momentum
   SetIndexBuffer(5,EMA_CMtmBuffer,INDICATOR_CALCULATIONS);      // r-period 1st EMA
   SetIndexBuffer(6,DEMA_CMtmBuffer,INDICATOR_CALCULATIONS);     // s-period 2nd EMA
   SetIndexBuffer(7,TEMA_CMtmBuffer,INDICATOR_CALCULATIONS);     // u-period 3rd EMA
   SetIndexBuffer(8,AbsCMtmBuffer,INDICATOR_CALCULATIONS);       // q-period Candlestick Momentum (absolute value)
   SetIndexBuffer(9,EMA_AbsCMtmBuffer,INDICATOR_CALCULATIONS);   // r-period 1st EMA (absolute value)
   SetIndexBuffer(10,DEMA_AbsCMtmBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (absolute value)
   SetIndexBuffer(11,TEMA_AbsCMtmBuffer,INDICATOR_CALCULATIONS); // u-period 3rd EMA (absolute value)
```

2) Algorithm of calculating cmtm and \|cmtm\|:

```
//--- calculation of Price1Buffer[] and Price2Buffer[]
   CalculatePriceBuffer(
                        AppliedPrice1,       // applied price [close]
                        rates_total,         // rates total
                        prev_calculated,     // number of bars, calculated at previous call
                        Open,High,Low,Close, // Open[], High[], Low[], Close[]
                        Price1Buffer         // target array
                       );
   CalculatePriceBuffer(AppliedPrice2,rates_total,prev_calculated,Open,High,Low,Close,Price2Buffer);
//--- calculation of cmtm and |cmtm|
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              //
      for(i=0;i<pos;i++)       //
        {
         CMtmBuffer[i]=0.0;    // zero values
         AbsCMtmBuffer[i]=0.0; //
        }
     }
   else pos=prev_calculated-1; // overwise calculate only last value
   // calculation of CMtmBuffer[] and AbsCMtmBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      CMtmBuffer[i]=Price1Buffer[i]-Price2Buffer[i-(q-1)];
      AbsCMtmBuffer[i]=MathAbs(CMtmBuffer[i]);
     }
```

**4.3.2. The Ergodic CSI-oscillator**

The Ergodic CSI-oscillator is defined as follows:

```
Ergodic_CSI(price1,pric2,q,r,s,u) = CSI(price1,pric2,q,r,s,u)
```

```
SignalLine(price1,pric2,q,r,s,u,ul) = EMA( Ergodic_CSI(price1,pric2,q,r,s,u) ,ul)
```

where:

- Ergodic\_CSI() - Ergodic - Candlestick index CSI(price1,price2,q,r,s,u);
- The SignalLine() -a Signal Line - the EMA(u)l, applied to the Ergodic;
- ul - an EMA period of a Signal Line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic.

#### 4.3.2.1. Ergodic\_CSI(price1,pric2,q,r,s,u,ul) - ergodic CSI-oscillator. Specification

- **File name**: **Blau\_Ergodic\_CSI.mq5**
- **Name**: Ergodic CSI-Oscillator (based on the Candlestick Index), according to William Blau.
- **Input parameters**:

  - graphic plot #0 - Ergodic (Candlestick Index):

    - q - the period for which the q-period Candlestick Momentum is calculated (by default q = 1);
    - r - period of the 1st EMA, applied to the q-period Candlestick Momentum (by default r = 20);
    - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
    - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3);

  - graphic plot #1 - the Signal Line:

    - ul - period EMA signal line, is applied to the Ergodic (by default ul = 3);

  - AppliedPrice1 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[closing\] (by default AppliedPrice=PRICE\_CLOSE);
  - AppliedPrice2 - [price type](https://www.mql5.com/en/docs/constants/indicatorconstants/prices) \[opening\] (by default AppliedPrice=PRICE\_OPEN).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphical plotting - the color, thickness, line style (the "Colors" tab);
  - two levels (by default -25 and +25) - add/remove a level, change the value, level description, change the rendering style of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) boundaries of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - ul>0. If ul = 1, then the signal line coincides with the ergodic;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

#### 4.3.2.2. The code of the Ergodic CSI-oscillator

On the example of the indicator Ergodic\_CSI (price1, price2,r,s,u,ul):

1) The relation between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);               // Ergodic
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);             // Signal Line: EMA(ul), applied to Ergodic
   // buffers, used for intermediate calculations
   SetIndexBuffer(2,Price1Buffer,INDICATOR_CALCULATIONS);     // price array [close]
   SetIndexBuffer(3,Price2Buffer,INDICATOR_CALCULATIONS);     // price arrya [open]
   SetIndexBuffer(4,LLBuffer,INDICATOR_CALCULATIONS);         // lowest prices (q bars)
   SetIndexBuffer(5,HHBuffer,INDICATOR_CALCULATIONS);         // highest prices (q bars)
   SetIndexBuffer(6,CMtmBuffer,INDICATOR_CALCULATIONS);       // q-period Candlestick Momentum
   SetIndexBuffer(7,EMA_CMtmBuffer,INDICATOR_CALCULATIONS);   // r-period 1st EMA
   SetIndexBuffer(8,DEMA_CMtmBuffer,INDICATOR_CALCULATIONS);  // s-period 2nd EMA
   SetIndexBuffer(9,TEMA_CMtmBuffer,INDICATOR_CALCULATIONS);  // u-period 3rd EMA
   SetIndexBuffer(10,HHLLBuffer,INDICATOR_CALCULATIONS);      // price range (q bars)
   SetIndexBuffer(11,EMA_HHLLBuffer,INDICATOR_CALCULATIONS);  // r-period 1st EMA (price range)
   SetIndexBuffer(12,DEMA_HHLLBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (price range)
   SetIndexBuffer(13,TEMA_HHLLBuffer,INDICATOR_CALCULATIONS); // u-period 3rd EMA (price range)
```

2) The algorithm of calculation for the cmtm and the q-period price range:

```
//--- calculation of Price1Buffer[] and Price2Buffer[]
   CalculatePriceBuffer(
                        AppliedPrice1,       // price type [close]
                        rates_total,         // rates total
                        prev_calculated,     // number of bars, calculated at previous call
                        Open,High,Low,Close, // Open[], High[], Low[], Close[]
                        Price1Buffer         // target array
                       );
   CalculatePriceBuffer(AppliedPrice2,rates_total,prev_calculated,Open,High,Low,Close,Price2Buffer);
//--- calculation of cmtm and price range (q bars)
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              //
      for(i=0;i<pos;i++)       //
        {
         CMtmBuffer[i]=0.0;    // zero values
         HHLLBuffer[i]=0.0;    //
         LLBuffer[i]=0.0;      //
         HHBuffer[i]=0.0;      //
        }
     }
   else pos=prev_calculated-1; // overwise calculate only last value
   // calculation of CMtmBuffer[], HHLLBuffer[], LLBuffer[], HHBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      // CMtmBuffer[] - q-period Candlestick Momentum
      CMtmBuffer[i]=Price1Buffer[i]-Price2Buffer[i-(q-1)];
      // LLBuffer[] - search for the lowest price (q bars)
      // HHBuffer[] - search for the highest price (q bars)
      min=1000000.0;
      max=-1000000.0;
      for(k=i-(q-1);k<=i;k++)
        {
         if(min>Low[k])  min=Low[k];
         if(max<High[k]) max=High[k];
        }
      LLBuffer[i]=min;
      HHBuffer[i]=max;
      // HHLLBuffer[] - Price Range (q bars)
      HHLLBuffer[i]=HHBuffer[i]-LLBuffer[i];
     }
```

### 5\. Directional Trend

The considered indicators (see attachment):

1. **Blau\_HLM.mq5**\- is an indicator of the Virtual Close (q-period Composite High-Low Momentum; the smoothed q-period Composite High-Low Momentum);
2. **Blau\_DTI.mq5**\- the Directional Trend Index (normalized smoothed q-period Composite High-Low Momentum);
3. **Blau\_Ergodic\_DTI.mq5**\- the Ergodic DTI-oscillator (based on the Directional Trend Index).

![Fig. 5.1. Directional Trend Index Indicators](https://c.mql5.com/2/2/hlm01__2.PNG)

Fig. 5.1. Directional Trend Index Indicators

**5.1. The Composite High-Low Momentum**

**5.1.1. Defining the momentum of the up-trend and down-trend**

**One of the definitions of the trend.** If the values ​​of the maximum prices increase, then there is an _upward trend_. If the values ​​of the minimum prices are decreasing, then there is a _downward trend_.

A group of Momentum indicators, discussed in Section 1, can be used tp calculate the momentum for the maximums of the prices:

```
Mtm( price=High, q, r, s, u )
TSI( price=High, q, r, s, u )
Ergodic( price=High, q, r, s, u )
```

and for the minimum prices:

```
Mtm( price=Low, q, r, s, u )
TSI( price=Low, q, r, s, u )
Ergodic( price=Low, q, r, s, u )
```

The **up-trend Momentum** or the **High Momentum Up**  (HMU) is the _positive_ difference between the maximum price of the current period, and the maximum price at the beginning of the q-period price range. _The value of the q-period Momentum of the up-trend shows a_ relative velocity of the _growth_ of the maximum price for the current period, compared to the maximum price at the beginning of the q-period range of price fluctuations.

**The formula of the q-period momentum of the up-trend**:

```
HMU(q) = High - High[q-1], if High - High[q-1] > 0
```

```
HMU(q) = 0, if High - High[q-1] <= 0
```

where:

- q - is the number of time periods of the price graph, involved in the calculation of the up-trend momentum;
- High - the maximum price for the current period;
- High\[q–1\] - maximum price (q-1) periods ago.

The **down-trend momentum** or the **Low Momentum Down** (LMD) - this is a _positive_ difference between the minimum price of the current period, and the lowest price for the beginning of the q-period range of price fluctuations. _The value of the q-period momentum of the down-trend shows the_ relative velocity of the _decrease_ of the minimum price of the current period, compared with the lowest price for the beginning of the q-period price range.

**The formula of the q-period down-trend Momentum**:

```
LMD(q) = -(Low - Low[q-1]), if Low - Low[q-1] < 0
```

```
LMD(q) = 0, if Low - Low[q-1] >= 0
```

where:

- q - is the number of time periods of the price chart, used in the calculation of the down-trend momentum;
- Low - the minimum price for the current period;
- Low\[q-1\] - the minimum price (q-1) periods ago.

**A Composite High-Low Momentum**(High-Low Momentum, HLM) - is the difference between the q-period Momentum of the up-trend and the q-period Momentum of the down-trend. The sign of the composite High-Low Momentum indicates the trend of price changes: a positive HLM - a trend of price increase (upward trend), and a negative - the trend of price decrease (downward trend).

**Formula**:

```
HLM(q) = HMU(q) - LMD(q)
```

where:

- q - the number of time periods of the price graph, involved in the calculation of the momentums of the up-trend and down-trend;
- HMU(q) - the momentum of the up-trend for the period q;
- LMD(q) - the momentum of the down-trend for the period q.

The **formula of the smoothed q-period Composite High-Low Momentum**(Virtual Close):

```
HLM(q,r,s,u) = EMA(EMA(EMA( HLM(q) ,r),s),u) = EMA(EMA(EMA( HMU(q)-HMD(q) ,r),s),u)
```

where:

- q - the number of time periods of the price graph, involved in the calculation of the momentums of the up-trend and down-trend;
- HMU(q) - the momentum of the up-trend for the period q;
- LMD(q) - the momentum of the down-trend for the period q;
- HLM(q) = HMU(q)-LMD(q) - the q-period Composite High-Low Momentum;
- EMA (HLM (q), r) - the first smoothing - the EMA(r), applied to the q-period Composite High-Low Momentum;
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing

- EMA (EMA (EMA (..., r), s), u) - the third smoothing - the EMA(u), applied to the result of the 2ndsmoothing.

The curve of the graph of the accumulated sum of complex momentums for the maximums and minimums is called a _virtual close_.

**5.1.2. HLM(q,r,s,u) - Virtual Close Indicator. Specification**

- **File name**: **Blau\_HLM.mq5**
- **Name**: Indicator of the virtual Close (q-period Composite High-Low Momentum; a smoothed q-period Composite High-Low Momentum), according to William Blau.
- **Input parameters**:

  - q - the period for which the HLM (by default q = 2) is calculated;
  - r - period of the 1st EMA, applied to the HLM (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3).

- **Additionally**:

  - displayed in a separate window;
  - changes of the rendering of the graphical plotting - the color, thickness, line style (the "Colors" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**5.2. Directional Trend Index**

**5.2.1. The definition of the Directional Trend Index**

The **Directional Trend Index**(Directional Trend Index, DTI) - is an indicator of a normalized q-period Composite High-Low Momentum (normalized smoothed HLM). The values of the smoothed HLM are given as a percentage of the scale (interval of display \[-100, 100\]).

Each value of the smoothed HLM is normalized by the value of a smoothed HLM, taken as an absolute value. _Normalization allows_ the DTI value to be interpreted as a degree of an overbought (positive value) or oversold (negative value) market level.

**The formula of the Directional Trend Index**:

```
               100 * EMA(EMA(EMA( HLM(q) ,r),s),u)          100 * HLM(q,r,s,u)
DTI(q,r,s,u) = –––––––––––––––––––––––––---––––––– = ––––––––––––––--–––––––––––––––
                 EMA(EMA(EMA( |HLM(q)| ,r),s),u)     EMA(EMA(EMA( |HLM(q)| ,r),s),u)
```

```
if EMA(EMA(EMA(|HLM(q)|,r),s),u)=0, then DTI(price,q,r,s,u)=0
```

where:

- q - the number of time periods of the price graph, involved in the calculation of the momentums of the up-trend and down-trend;
- HLM(q) = HMU(q)-LMD(q) - a complex q-period momentum for the maximums and minimums;
- \|HLM(q)\| - absolute value HLM(q);
- HLM(q,r,s,u) - three times smoothed HLM(q);
- EMA(..., r) - the first smoothing - the EMA(r), applied to:

1) to the HLM (q)

2) to the absolute value of the HLM (q);
- EMA (EMA(..., r),s) - the second smoothing - the EMA(s), applied to the result of the 1st smoothing;
- EMA (EMA (EMA (..., r), s), u) - the third smoothing - the EMA(u), applied to the result of the 2nd smoothing.

**5.2.2. DTI(q,r,s,u) - Directional Trend Index. Specification**

- **File name**: **Blau\_DTI.mq5**
- **Name:** Directional Trend Index (normalized smoothed q-period Composite High-Low Momentum), according to William Blau.
- **Input parameters**:

  - q - the period for which the HLM (by default q = 2) is calculated;
  - r - period of the 1st EMA, applied to the HLM (by default r = 20);
  - s - period of the 2nd EMA, applied to the result of the 1st smoothing (by default s = 5);
  - u - period of the 3rd EMA, applied to the result of the 2nd smoothing (by default, u = 3).

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of the graphical plotting - the color, thickness, line style (the "Colors" tab);
  - ( **_optional_**) two-levels (default is -25 and +25) - add/remove a level; change the value, the level description, change the rendering style of the levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, then in the corresponding EMA period, smoothing will not be performed;
  - the minimum size of the prices array = (q-1 + r + s + u-3 +1).

**5.3. The Ergodic DTI-oscillator**

**5.3.1. The definition of the Ergodic DTI-oscillator**

```
Ergodic_DTI(q,r,s,u) = DTI(q,r,s,u)
```

```
SignalLine(q,r,s,u,ul) = EMA( Ergodic_DTI(q,r,s,u) ,ul)
```

where:

- Ergodic\_DTI() - Ergodic - Directional Trend Index DTI(q,r,s,u);
- The SignalLine() - a Signal Line - an exponentially moving average of period ul, applied to the Ergodic;
- ul - an EMA period of a Signal Line - according to William Blau, the ul value must be equal to the period of the last significant (>1) of the EMA ergodic.

**5.3.2. Ergodic\_DTI(q,r,s,u,ul) - Ergodic DTI-oscillator. Specification**

- **File name**: **Blau\_Ergodic\_DTI.mq5**
- **Name**: Ergodic DTI-Oscillator (based on the Directional Trend Index) by William Blau.
- **Input parameters**:

  - graphic plot #0 - ergodic (index of the directional trend):

    - q - the period for which the HLM (by default q = 2) is calculated;
    - r - period of the 1st EMA, with regards to the HLM (by default r = 20);
    - s - period of the 2nd EMA, with respect to the results of the first smoothing (by default s = 5);
    - u - period of the 3rd EMA, with respect to the result of the second smoothing (by default, u = 3);

  - graphical construction # 1 - the signal line:
    - ul - period EMA signal line, is applied to the ergodic (by default ul = 3);

- **Additionally**:

  - displayed in a separate window;
  - change the rendering style of each graphical plotting - the color, thickness, line style (the "Colors" tab);
  - two levels (by default -25 and +25) - add/remove a level, change the value, level description, change the rendering style of levels (the "Levels" tab);
  - change the lower (by default -100) and the upper (by default 100) limits of the scale of the single indicator window (the "Scale" tab).

- **Limitations**:
  - q>0;
  - r>0, s>0, u>0. If r, s, or u are equal to 1, the EMA smoothing is not used;
  - ul>0. If ul = 1, then the signal line coincides with the ergodic;
  - the minimum size of the prices array = (q-1 + r + s + u + ul-4 +1).

**5.4. The code of the** **Ergodic DTI-oscillator**

The Ergodic\_DTI (q,r,s,u,ul) indicator:

1) The link between the indicator arrays, indicator buffers, and graphic plots:

```
//--- indicator buffers
   // graphic plot #0
   SetIndexBuffer(0,MainBuffer,INDICATOR_DATA);                 // Ergodic Line
   // graphic plot #1
   SetIndexBuffer(1,SignalBuffer,INDICATOR_DATA);               // Signal Line: EMA(ul), applied to Ergodic
   // buffers, used for intermediate calculations
   SetIndexBuffer(2,HMUBuffer,INDICATOR_CALCULATIONS);          // q-period Up Trend Momentum
   SetIndexBuffer(3,LMDBuffer,INDICATOR_CALCULATIONS);          // q-period Down Trend Momentum
   SetIndexBuffer(4,HLMBuffer,INDICATOR_CALCULATIONS);          // Composite q-period High/Low Momentum
   SetIndexBuffer(5,EMA_HLMBuffer,INDICATOR_CALCULATIONS);      // r-period 1st EMA
   SetIndexBuffer(6,DEMA_HLMBuffer,INDICATOR_CALCULATIONS);     // s-period 2nd EMA
   SetIndexBuffer(7,TEMA_HLMBuffer,INDICATOR_CALCULATIONS);     // u-period 3rd EMA
   SetIndexBuffer(8,AbsHLMBuffer,INDICATOR_CALCULATIONS);       // Composite q-period High/Low Momentum (absolute values)
   SetIndexBuffer(9,EMA_AbsHLMBuffer,INDICATOR_CALCULATIONS);   // r-period 1st EMA (absolute values)
   SetIndexBuffer(10,DEMA_AbsHLMBuffer,INDICATOR_CALCULATIONS); // s-period 2nd EMA (absolute values)
   SetIndexBuffer(11,TEMA_AbsHLMBuffer,INDICATOR_CALCULATIONS); // u-period 3rd EMA (absolute values)
```

2) Algorithm of calculation of HLM and \|HML\|:

```
//--- calculation of HLM and |HLM|
   if(prev_calculated==0)      // at first call
     {
      pos=begin1;              //
      for(i=0;i<pos;i++)       //
        {
         HLMBuffer[i]=0.0;     // zero values
         AbsHLMBuffer[i]=0.0;  //
         HMUBuffer[i]=0.0;     //
         LMDBuffer[i]=0.0;     //
        }
     }
   else pos=prev_calculated-1; // overwise calculate only last value
   // calculation of HLMBuffer[], AbsHLMBuffer[], HMUBuffer[], LMDBuffer[]
   for(i=pos;i<rates_total;i++)
     {
      HMUBuffer[i]=High[i]-High[i-(q-1)];    HMUBuffer[i]=(HMUBuffer[i]>0)?HMUBuffer[i]:0;
      LMDBuffer[i]=-1*(Low[i]-Low[i-(q-1)]); LMDBuffer[i]=(LMDBuffer[i]>0)?LMDBuffer[i]:0;
      HLMBuffer[i]=HMUBuffer[i]-LMDBuffer[i];
      AbsHLMBuffer[i]=MathAbs(HLMBuffer[i]);
     }
```

### **Conclusion**

The first part of the article "William Blau's Indicators and Trading Systems on MQL5. Part 1: Indicators" provides a description of the developed indicators and oscillators in MQL5, from the book ["Momentum, Direction, and Divergence"](https://www.mql5.com/go?link=https://www.amazon.com/Momentum-Direction-Divergence-Indicators-Technical/dp/0471027294 "http://www.amazon.com/Momentum-Direction-Divergence-Indicators-Technical/dp/0471027294") by William Blau.

The use of these indicators and oscillators when making trading decisions will be described in the second part of the article "William Blau's Indicators and Trading Systems in MQL5. Part 2: Trading Systems".

The contents of the attachment archive of this article ("Blau\_Indicators\_MQL5\_en.zip"):

| File | Description |
| --- | --- |
| The included file. Location: "terminal\_data\_folder\\MQL5\\Include" |
| WilliamBlau.mqh |  |
| Indicators. Location: "terminal\_data\_folder\\MQL5\\Indicators" |
|  | **Indicators, based on the Momentum** |
| Blau\_Mtm.mq5 | The Momentum Indicator (q-period momentum, smoothed q-period momentum) |
| Blau\_TSI.mq5 | The True Strength Index (Normalized smoothed q-period momentum) |
| Blau\_Ergodic.mq5 | Ergodic Oscillator (based on the true strength index) |
|  | **Indicators, based on the Stochastic** |
| Blau\_TStoch.mq5 | Stochastic (q-period stochastic, smoothed q-period Stochastic) |
| Blau\_TStochI.mq5 | Stochastic index (normalized smoothed q-period Stochastic) |
| Blau\_TS\_Stochastic.mq5 | Stochastic TS-oscillator (based on the Stochastic Index) |
|  | **Indicators, based on the Stochastic Momentum** |
| Blau\_SM.mq5 | Stochastic Momentum (q-period Stochastic Momentum, smoothed q-period Stochastic Momentum) |
| Blau\_SMI.mq5 | Stochastic Momentum Index (Smoothed normalized q-stochastic momentum RSI) |
| Blau\_SM\_Stochastic.mq5 | Stochastic SM-Oscillator (based on the Stochastic Momentum Index) |
|  | **Indicators, based on a deviation from the market trend** |
| Blau\_MDI.mq5 | Mean Deviation Indicator (Mean Deviation, smoothed mean deviation) |
| Blau\_Ergodic\_MDI.mq5 | Ergodic MDI-oscillator (based on the Mean Deviation indicator) |
|  | **Indicators, based on the Moving Average Convergence/Divergence** |
| Blau\_MACD.mq5 | Indicator of the convergence/divergence of the moving averages (MACD; smoothed MACD) |
| Blau\_Ergodic\_MACD.mq5 | Ergodic MACD-oscillator (based on the MACD indicator) |
|  | **Indicators, based on the Candlestick Momentum** |
| Blau\_CMtm.mq5 | Candlestick Momentum Indicator (q-period Candlestick Momentum, smoothed q-period Candlestick Momentum) |
| Blau\_CMI.mq5 | Candlestick Momentum Index (normalized smoothed q-period Candlestick Momentum; normalized by the absolute value of the q-period Candlestick Momentum) |
| Blau\_CSI.mq5 | The Candlestick Index (normalized smoothed q-period Candlestick Momentum; normalization by the length of the q-period candlestick) |
| Blau\_Ergodic\_CMI.mq5 | Ergodic CMI-oscillator (based on the Candlestick Momentum Index) |
| Blau\_Ergodic\_CSI.mq5 | Ergodic CSI-oscillator (based on the Candlestick Index) |
|  | **Indicators, based on the Composite Momentum** |
| Blau\_HLM.mq5 | Indicator of Virtual Close (q-period Composite High-Low Momentum; the smoothed q-period Composite High-Low Momentum) |
| Blau\_DTI.mq5 | Index of a directional trend (normalized smoothed q-period Composite High-Low Momentum) |
| Blau\_Ergodic\_DTI.mq5 | Ergodic DTI-oscillator (based on the Directional Trend Index) |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/190](https://www.mql5.com/ru/articles/190)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/190.zip "Download all attachments in the single ZIP archive")

[blau\_indicators\_mql5\_en.zip](https://www.mql5.com/en/articles/download/190/blau_indicators_mql5_en.zip "Download blau_indicators_mql5_en.zip")(52.11 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/4291)**
(7)


![---](https://c.mql5.com/avatar/avatar_na2.png)

**[\-\-\-](https://www.mql5.com/en/users/sergeev)**
\|
22 Apr 2011 at 23:42

**fozi:**

here is the book the author refers to. True, they want some "gold" and I found it that way ))

\*.djvu format

shit... why it doesn't appear after attachment.

zip


![ak20 ak20](https://c.mql5.com/avatar/avatar_na2.png)

**[ak20 ak20](https://www.mql5.com/en/users/ak20)**
\|
16 Aug 2011 at 18:25

Very insightful article. I'm looking forward to part 2. Thank you!


![pierre wang](https://c.mql5.com/avatar/avatar_na2.png)

**[pierre wang](https://www.mql5.com/en/users/upgnaw)**
\|
10 Dec 2011 at 09:55

It's just what I am lookiing for, thank you very much


![Marik](https://c.mql5.com/avatar/avatar_na2.png)

**[Marik](https://www.mql5.com/en/users/marik)**
\|
31 Jul 2015 at 10:41

Kudos to the author for the hard work!


![akarapone tonsomboon](https://c.mql5.com/avatar/2022/7/62DE6E87-1A99.png)

**[akarapone tonsomboon](https://www.mql5.com/en/users/sekiro_okami)**
\|
3 Aug 2022 at 07:24

appreciate your effort. <3


![3 Methods of Indicators Acceleration by the Example of the Linear Regression](https://c.mql5.com/2/0/Indirocket.png)[3 Methods of Indicators Acceleration by the Example of the Linear Regression](https://www.mql5.com/en/articles/270)

The article deals with the methods of indicators computational algorithms optimization. Everyone will find a method that suits his/her needs best. Three methods are described here.One of them is quite simple, the next one requires solid knowledge of Math and the last one requires some wit. Indicators or MetaTrader5 terminal design features are used to realize most of the described methods. The methods are quite universal and can be used not only for acceleration of the linear regression calculation, but also for many other indicators.

![Enhancing the Quality of the Code with the Help of Unit Test](https://c.mql5.com/2/17/936_17.png)[Enhancing the Quality of the Code with the Help of Unit Test](https://www.mql5.com/en/articles/1579)

Even simple programs may often have errors that seem to be unbelievable. "How could I create that?" is our first thought when such an error is revealed. "How can I avoid that?" is the second question which comes to our mind less frequently. It is impossible to create absolutely faultless code, especially in big projects, but it is possible to use technologies for their timely detection. The article describes how the MQL4 code quality can be enhanced with the help of the popular Unit Testing method.

![Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://c.mql5.com/2/0/Fisher_Transform_MQL5__1.png)[Applying The Fisher Transform and Inverse Fisher Transform to Markets Analysis in MetaTrader 5](https://www.mql5.com/en/articles/303)

We now know that probability density function (PDF) of a market cycle does not remind a Gaussian but rather a PDF of a sine wave and most of the indicators assume that the market cycle PDF is Gaussian we need a way to "correct" that. The solution is to use Fisher Transform. The Fisher transform changes PDF of any waveform to approximately Gaussian. This article describes the mathematics behind the Fisher Transform and the Inverse Fisher Transform and their application to trading. A proprietary trading signal module based on the Inverse Fisher Transform is presented and evaluated.

![The Player of Trading Based on Deal History](https://c.mql5.com/2/0/MQL5_Trade_Player.png)[The Player of Trading Based on Deal History](https://www.mql5.com/en/articles/242)

The player of trading. Only four words, no explanation is needed. Thoughts about a small box with buttons come to your mind. Press one button - it plays, move the lever - the playback speed changes. In reality, it is pretty similar. In this article, I want to show my development that plays trade history almost like it is in real time. The article covers some nuances of OOP, working with indicators and managing charts.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=xumxfdzgxdhogelrpqncvuplosqyamjp&ssn=1769092766736785216&ssn_dr=0&ssn_sr=0&fv_date=1769092766&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F190&back_ref=https%3A%2F%2Fwww.google.com%2F&title=William%20Blau%27s%20Indicators%20and%20Trading%20Systems%20in%20MQL5.%20Part%201%3A%20Indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909276683725785&fz_uniq=5049283808315943070&sv=2552)

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