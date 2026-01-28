---
title: Simple Methods of Forecasting Directions of the Japanese Candlesticks
url: https://www.mql5.com/en/articles/1374
categories: Trading
relevance_score: 1
scraped_at: 2026-01-23T21:36:30.291828
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1374&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071960728832782762)

MetaTrader 4 / Trading


### Introduction

Japanese candlesticks represent one of the ways of displaying information on prices over a certain period of time showing the correspondence between open, close, high and low prices. Candlesticks are primarily used in technical analysis and interpreted in accordance with well-known models.

After writing several indicators for forecasting the direction of the market movement, I came up with an idea of analyzing the possibility of doing a basic forecast of directions of the Japanese candlesticks based on the available data on the directions of the preceding candlesticks.

The article deals with a few simple forecasting methods and introduces several EAs for testing the ideas and assessing their effectiveness.

Attention! The Expert Advisors attached to the article are intended for use in the Strategy Tester only.

The first method is based on the assumption that the direction of the next candlestick coincides with the direction of the preceding one. The second method assumes that the direction of the next candlestick will be opposite to that of the preceding one. The benefit of these methods lies in their simplicity. We will review them in the "Naive Methods" section.

The third method is a combination of the first two. At first, we choose either the first or the second method and the chosen method is then used unless and until the forecast appears to be different from the reality (in case an error is detected). In this case the method that is currently in use is replaced with the other one. This method is more complex than the first two as it takes into consideration errors that occurred at the previous stages. We will therefore review this method in the "Adaptive Method" section.

### Basic Data for Forecasting

All the methods under consideration rest upon the information on the direction of the preceding candlestick. Therefore, we should first think of a way of getting the information on the direction of any given candlestick.

The term "direction of the candlestick" will be used to refer to the value characterizing the sign of the difference between opening and closing prices of a given candlestick. In view of this, the direction of any candlestick can take one of three values - "direction up", "direction down" and "no direction".

Information on opening and closing prices of the candlestick can be accessed using the **Open** and **Close** arrays, respectively. These arrays use shifts as indices. A shift is the number of a candlestick counted from the right hand side of the chart, starting with zero. That means that the shift of the current candlestick is zero, while the shift of the preceding candlestick is one.

```
// getting the direction of the candlestick
double GetCandleDirection(int shift) {
   double diff=Close[shift]-Open[shift];
   if (diff>0.0) {
      return (DIRECTION_UP);
   }
   else if (diff<0.0) {
      return (DIRECTION_DOWN);
   }
   else /*if (diff==0.0)*/ {
      return (DIRECTION_NONE);
   }
}
```

The function receives the **shift** of the candlestick and calculates the difference of its opening and closing prices. Depending on the sign of the resulting figure, the function returns the value of one of the three constants, **DIRECTION\_UP**, **DIRECTION\_DOWN** or **DIRECTION\_NONE**. These constants correspond to "direction up", "direction down" and "no direction", respectively.

It should be noted that all the forecasting methods under consideration provide for one special case: if the direction of the preceding candlestick cannot be determined (the **GetCandleDirection(1)** call has returned the value of **DIRECTION\_NONE**), the direction of the next candlestick will also be undetermined as a result of the forecast. In such cases, the Expert Advisors written to test the methods do not enter the market.

Note that values of the constants used for indicating different directions should be defined so as not to cause difficulties when calculating the opposite direction. The sign change can serve as one of the ways of calculating the opposite direction. These constants can then be defined as follows:

```
// determining the possible candlestick directions
#define DIRECTION_NONE 0
#define DIRECTION_UP 1
#define DIRECTION_DOWN -1
```

Let's arrange a separate output of the candlestick directions in the chart, using the CandleColor.mq4 indicator attached to the article. The indicator displays the directions of the candlesticks (the blue and red histograms), as well as a simple moving average based on the data on the directions which allows us to get a general idea of the candlesticks prevailing over a certain period of time.

![](https://c.mql5.com/2/13/02_candle_color_detailed.gif)

Candlestick directions: Behavior over time.

As shown by the indicator, the directions of candlesticks very often change, alternating or remaining the same for a few candlesticks in a row. Clearly, forecasting the direction of even a single candlestick is a challenging task.

Let's now proceed to the consideration and detailed analysis of the proposed methods.

### Naive Methods

We will start with the first two methods:

1. The direction of the next candlestick will coincide with the direction of the preceding one.

2. The direction of the next candlestick will be opposite to that of the preceding one.


To test the effectiveness of both methods, an Expert Advisor (the CandlePredTest\_naive.mq4 file attached to the article) was developed and tested.

The Expert Advisor uses opening prices. On each new bar, it closes all positions that were opened earlier (takes profit or stops loss) and opens a position in accordance with the hypothesized direction of the next candlestick (just now opened). The forecast of the next candlestick direction is performed by the **ModelPredict** function:

```
// forecasting based on the current model
int ModelPredict() {
   if (modelState==STATE_NORMAL) {
      return (GetCandleDirection(1));
   }
   else /*if (modelState==STATE_REVERSE)*/ {
      return (-GetCandleDirection(1));
   }
}
```

The function implements two forecasting methods. The choice of the method depends on the value of the **modelState** variable.

If the **modelState** variable takes on the value of **STATE\_NORMAL**, the forecasting will follow the first method whereby the function calls **GetCandleDirection(1)** for the direction of the preceding candlestick and returns the obtained value as the direction of the next candlestick.

The second forecasting method is used if the **modelState** variable value is **STATE\_REVERSE**. In this case, the calculation of the opposite direction is done through the sign change, i.e. the **ModelPredict** function returns the value of **-GetCandleDirection(1)**.

The **ModelPredict** function does not use the express comparison with the **STATE\_REVERSE** constant corresponding to the second method. In the code of the Expert Advisor, it is assumed that the data entered by the user is always correct and the **modelState** variable always takes on one of the two values: **STATE\_NORMAL** or **STATE\_REVERSE**. Express comparison can be enabled by removing the comment brackets " **/\***" and " **\*/**" from the body of the function, thus adding another conditional operator **if** to the code.

Although the **modelState** variable is missing from the list of parameters of the Expert Advisor, its value is copied from the **initialModelState** external variable upon launching the Expert Advisor:

```
// model state
extern int initialModelState=STATE_NORMAL;
```

This additional complication of the code is provided due to the fact that the number of states may further increase, while not all of those states will be allowed at the start of operation. You can use any values as indications of states. For example, in the Expert Advisor under consideration, they are defined as follows:

```
// defining the model states
#define STATE_NORMAL 0
#define STATE_REVERSE 1
```

The remaining code of the Expert Advisor is provided with very detailed comments and should not be of any difficulty to the reader.

The conditions for testing the Expert Advisor are as shown below:

- Position volume - 0.1 lot.
- Initial deposit - $10,000.
- Time frames - all time frames available.
- Operation modes - **STATE\_NORMAL**, **STATE\_REVERSE**.
- Testing period - all available history.
- Instruments:




01. USDCHF
02. GBPUSD
03. EURUSD
04. USDJPY
05. AUDUSD
06. USDCAD
07. EURGBP
08. EURCHF
09. EURJPY
10. GBPJPY
11. GBPCHF
12. EURAUD

The criteria according to which the results were compared are net profit and percentage of profitable trades.

The following two tables demonstrate the results of testing the Expert Advisor in the **STATE\_NORMAL** mode (first method).

**_Net profit:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | -9902 | -9900 | -9903 | -9907 | -9901 | -9900 | -9927 |
| GBPUSD | -9868 | -9864 | -9869 | -9887 | -9873 | -9868 | -9924 |
| EURUSD | -9921 | -9957 | -9949 | -9941 | -9934 | -9910 | -9879 |
| USDJPY | -9900 | -9905 | -9900 | -9905 | -9935 | -9926 | -9904 |
| AUDUSD | -9952 | -9957 | -9966 | -9962 | -9961 | -9956 | -10000 |
| USDCAD | -9901 | -9903 | -9901 | -9900 | -9902 | -9904 | -8272 |
| EURGBP | -9862 | -9861 | -9864 | -9869 | -9875 | -9871 | -5747 |
| EURCHF | -9862 | -9865 | -9865 | -9874 | -9869 | -9862 | -5750 |
| EURJPY | -9866 | -9877 | -9964 | -9877 | -9869 | -9867 | -10000 |
| GBPJPY | -9848 | -9841 | -9840 | -9845 | -9848 | -9870 | -9849 |
| GBPCHF | -9891 | -9885 | -9850 | -9844 | -9857 | -9856 | -9891 |
| EURAUD | -9865 | -9863 | -9868 | -9874 | -9861 | -9891 | -10000 |

**_Percentage of profitable trades:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | 19.55 | 27.82 | 31.50 | 38.18 | 39.86 | 43.29 | 43.95 |
| GBPUSD | 21.18 | 27.81 | 30.29 | 33.81 | 35.20 | 41.39 | 46.71 |
| EURUSD | 26.06 | 33.20 | 34.59 | 37.55 | 41.64 | 43.73 | 44.37 |
| USDJPY | 19.28 | 31.68 | 34.13 | 35.48 | 39.04 | 42.99 | 46.85 |
| AUDUSD | 19.71 | 21.30 | 26.25 | 27.20 | 33.15 | 39.96 | 44.69 |
| USDCAD | 21.88 | 25.13 | 27.59 | 31.79 | 33.07 | 39.48 | 46.40 |
| EURGBP | 21.78 | 28.90 | 31.49 | 34.55 | 35.24 | 42.11 | 47.04 |
| EURCHF | 28.70 | 27.86 | 27.85 | 31.13 | 32.90 | 39.08 | 47.65 |
| EURJPY | 23.69 | 30.62 | 35.81 | 38.89 | 39.06 | 44.04 | 48.16 |
| GBPJPY | 10.11 | 19.47 | 33.48 | 37.39 | 37.75 | 43.09 | 47.42 |
| GBPCHF | 23.17 | 23.95 | 30.84 | 33.50 | 36.03 | 42.43 | 47.90 |
| EURAUD | 1.65 | 7.35 | 16.55 | 22.24 | 27.65 | 36.11 | 44.64 |

The testing of the first method showed poor results. Regardless of the period, we see the loss of the deposit for all the instruments; that said, the number of profitable trades increases on bigger time frames but it never exceeds 50%.

The following two tables show the results of testing the Expert Advisor in the **STATE\_REVERSE** mode (second method).

**_Net profit:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | -9900 | -9912 | -9902 | -9917 | -9915 | -9901 | +83 |
| GBPUSD | -9864 | -9866 | -9869 | -9874 | -9870 | -9911 | -5565 |
| EURUSD | -9920 | -9917 | -9923 | -9947 | -9922 | -9886 | +4249 |
| USDJPY | -9906 | -9901 | -9906 | -9900 | -9900 | -9928 | +2003 |
| AUDUSD | -9955 | -9956 | -9957 | -9956 | -9969 | -9947 | +1203 |
| USDCAD | -9900 | -9900 | -9900 | -9900 | -9901 | -9906 | -3129 |
| EURGBP | -9868 | -9862 | -9862 | -9863 | -9867 | -9902 | -9867 |
| EURCHF | -9863 | -9861 | -9863 | -9862 | -9869 | -9869 | -6556 |
| EURJPY | -9861 | -9864 | -9866 | -9868 | -9897 | -9886 | -10000 |
| GBPJPY | -9850 | -9887 | -9905 | -9857 | -9969 | -9848 | -9964 |
| GBPCHF | -9846 | -9841 | -9842 | -9842 | -9886 | -9914 | -9850 |
| EURAUD | -9870 | -9866 | -9874 | -9910 | -9872 | -9872 | -2411 |

**_Percentage of profitable trades:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | 18.38 | 27.41 | 33.76 | 37.46 | 42.28 | 46.71 | 52.15 |
| GBPUSD | 18.59 | 27.71 | 34.07 | 39.01 | 43.79 | 47.33 | 49.39 |
| EURUSD | 23.37 | 33.88 | 41.05 | 43.80 | 43.56 | 48.16 | 52.56 |
| USDJPY | 16.88 | 30.39 | 38.13 | 40.29 | 43.94 | 46.62 | 49.34 |
| AUDUSD | 18.84 | 22.52 | 25.39 | 29.83 | 36.36 | 43.51 | 49.84 |
| USDCAD | 22.44 | 24.04 | 27.07 | 31.06 | 35.69 | 43.51 | 49.84 |
| EURGBP | 20.52 | 28.89 | 35.18 | 38.82 | 42.41 | 45.82 | 45.55 |
| EURCHF | 27.43 | 31.82 | 32.80 | 35.05 | 36.74 | 43.89 | 46.04 |
| EURJPY | 18.79 | 28.15 | 37.09 | 39.80 | 43.05 | 44.15 | 46.92 |
| GBPJPY | 10.02 | 19.76 | 31.92 | 38.06 | 41.84 | 44.32 | 47.53 |
| GBPCHF | 19.87 | 27.24 | 33.72 | 35.70 | 39.74 | 47.59 | 46.11 |
| EURAUD | 1.43 | 7.66 | 17.24 | 26.95 | 34.46 | 41.29 | 47.07 |

We have got mixed results when testing the second method. The results on all the time frames, except for D1, have been about the same as the ones obtained for the first method. Trading on D1, the Expert Advisor gained profit on four instruments out of the twelve. The percentage of profitable trades was close to 50%, reaching 52% in two cases.

In general, the Expert Advisor's testing results leave much to be desired. Let's try to use a combination of the naive methods to cancel out their disadvantages.

### Adaptive Method

The essence of the third method is that we do the forecast using one of the two naive forecasting methods which is replaced with the other one in case of an error. In this way, this method adapts to the behavior of the forecast series.

The effectiveness of this method was verified by testing the corresponding Expert Advisor (the CandlePredTest\_adaptive.mq4 file attached to the article).

This Expert Advisor represents an improved version of the CandlePredTest\_naive Expert Advisor. Its key difference is in that each previous forecast is analyzed for errors on each new bar.

```
// working function of the Expert Advisor
int start() {
   // positions are opened once on each bar
   if (!IsNewBar()) {
      return (0);
   }
   // close all open positions
   TradeClosePositions(DIRECTION_UP);
   TradeClosePositions(DIRECTION_DOWN);
   // update the model state
   if (prediction!=DIRECTION_NONE) {
      ModelUpdate(prediction==GetCandleDirection(1));
   }
   // predict the color of the next candlestick
   prediction=ModelPredict();
   if (prediction==DIRECTION_NONE) {
      return (0);
   }
   // open a position in the right direction
   TradeOpenPosition(prediction,"");
   return (0);
}
```

Errors are analyzed only if the Expert Advisor opened positions on the previous bar, i.e. if the forecast did not return the **DIRECTION\_NONE** constant. The errors are all cases of discrepancy between the forecast and the actual direction of the closed candlestick whose shift becomes equal to one upon opening of the new bar.

One naive forecasting method is replaced by the other one using the **ModelUpdate** function that receives the forecast correction flag ( **prediction==GetCandleDirection(1)**) as a parameter.

```
// updating the model state
void ModelUpdate(bool correct) {
   // if the forecast was erroneous, change the model state
   if (!correct) {
      if (modelState==STATE_NORMAL) {
         modelState=STATE_REVERSE;
      }
      else /*if (modelState==STATE_REVERSE)*/ {
         modelState=STATE_NORMAL;
      }
   }
}
```

The function analyzes the forecast correction flag ( **correct**) and in case of an error replaces the selected naive method of forecasting by changing the value of the **modelState** variable. The initial value of the **modelState** variable is copied from the **initialModelState** variable upon initialization. It is not material as the Expert Advisor can change the forecasting method during the course of its operation.

Forecasting is the responsibility of the **ModelPredict** function similar to that used in the CandlePredTest\_naive Expert Advisor. As a result of changes in the **modelState** variable value, the forecast is performed based on the first and second methods, alternating each other.

The Expert Advisor was tested under the same conditions, with the only exception being the **initialModelState** variable value which was fixed ( **STATE\_NORMAL**).

**_Net profit:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | -9903 | -9905 | -9902 | -9915 | -9902 | -9907 | -3738 |
| GBPUSD | -9866 | -9875 | -9876 | -9864 | -9867 | -9933 | -6404 |
| EURUSD | -9924 | -9924 | -9931 | -9920 | -9920 | -9903 | -3779 |
| USDJPY | -9904 | -9900 | -9900 | -9904 | -9920 | -9902 | -6493 |
| AUDUSD | -9955 | -9954 | -9955 | -9995 | -9955 | -9965 | -8649 |
| USDCAD | -9903 | -9902 | -9901 | -9911 | -9900 | -9904 | -1525 |
| EURGBP | -9861 | -9867 | -9865 | -9862 | -9881 | -9877 | -9913 |
| EURCHF | -9861 | -9862 | -9872 | -9881 | -9870 | -9875 | -9258 |
| EURJPY | -9861 | -9865 | -9866 | -9868 | -9876 | -9905 | -9880 |
| GBPJPY | -9842 | -9844 | -9867 | -9840 | -9919 | -10000 | -9933 |
| GBPCHF | -9841 | -9843 | -9917 | -9840 | -9893 | -9848 | -9895 |
| EURAUD | -9866 | -9869 | -9863 | -9872 | -9867 | -9904 | -7656 |

**_Percentage of profitable trades:_**

|  | M1 | M5 | M15 | M30 | H1 | H4 | D1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDCHF | 19.22 | 26.82 | 32.29 | 35.71 | 40.29 | 44.15 | 47.93 |
| GBPUSD | 20.12 | 28.16 | 32.01 | 34.87 | 40.05 | 41.97 | 49.39 |
| EURUSD | 25.24 | 32.65 | 38.32 | 39.48 | 41.50 | 45.81 | 48.88 |
| USDJPY | 19.16 | 29.48 | 35.02 | 37.03 | 40.19 | 44.45 | 47.34 |
| AUDUSD | 19.63 | 21.67 | 25.76 | 28.16 | 32.11 | 41.11 | 46.76 |
| USDCAD | 22.84 | 24.46 | 27.39 | 29.79 | 34.44 | 42.26 | 48.46 |
| EURGBP | 21.44 | 29.45 | 33.05 | 36.55 | 38.19 | 41.86 | 46.23 |
| EURCHF | 27.51 | 29.98 | 30.92 | 34.44 | 34.09 | 40.87 | 44.27 |
| EURJPY | 21.24 | 28.56 | 35.58 | 38.78 | 40.50 | 44.23 | 47.12 |
| GBPJPY | 11.30 | 19.90 | 31.91 | 37.15 | 36.76 | 41.11 | 46.24 |
| GBPCHF | 22.00 | 26.81 | 32.13 | 34.38 | 36.99 | 43.11 | 47.16 |
| EURAUD | 1.50 | 6.56 | 15.14 | 22.25 | 31.63 | 37.96 | 47.70 |

The testing did not show significantly better results as compared to the first and second methods. The profit was negative for all instruments on all time frames, while the percentage of profitable trades did not exceed 50%.

### Conclusion

The testing results provided in the article have demonstrated poor effectiveness of the reviewed methods.

- The first method turned out to be extremely ineffective. Probably, the assumption that the direction of the next candlestick coincides with the direction of the preceding one needs further verification in forecasting.
- The second method showed the best results out of all other methods under consideration. However it requires a more detailed analysis. For example, data on the number of trades, profit on which was not taken might prove interesting.
- The third method demonstrated average results compared to the first and second method and needs further improvement. The main idea behind it was the improvement of the characteristics as compared to the characteristics used in the naive methods. However it did not bring about any improvements. Apparently, this method very often interchanges the forecasting methods which leads to additional errors.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1374](https://www.mql5.com/ru/articles/1374)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1374.zip "Download all attachments in the single ZIP archive")

[CandlesPrediction.zip](https://www.mql5.com/en/articles/download/1374/CandlesPrediction.zip "Download CandlesPrediction.zip")(3.42 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [LibMatrix: Library of Matrix Algebra (Part One)](https://www.mql5.com/en/articles/1365)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39136)**
(5)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
19 Sep 2013 at 08:07

So, what is your intention doing **Simple Methods of Forecasting Directions of the [Japanese Candlesticks](https://www.mql5.com/en/docs/constants/chartconstants/chart_view#enum_chart_mode "MQL5 documentation: Chart Representation")** if the result is bad?

![Osama Shaban](https://c.mql5.com/avatar/2013/7/51ED7E66-4468.JPG)

**[Osama Shaban](https://www.mql5.com/en/users/oshaban)**
\|
17 Jul 2015 at 00:58

I'm a researcher. As I see, as we got a result, regardless of it is good or bad one, it is a result we can consider it in an other researches or avoid using the same concept to save other researchers time and effort. Also, it could direct to some new ideas or directions for the same topic.

So, I personally consider it a value added info for adding a new info to our knowledge.

Thanks to the author, Evgeniy Logunov :)

![jaffer wilson](https://c.mql5.com/avatar/2018/5/5AFC2828-5318.jpeg)

**[jaffer wilson](https://www.mql5.com/en/users/jafferwilson)**
\|
21 May 2018 at 09:12

Is there any code for MT5? I am not using MT4.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
14 Jan 2019 at 10:56

This is an old article. I have read with interest.

I chanced upon the "miracle of candlesticks" recently when I was talking to someone about what kind of [market structure](https://www.mql5.com/en/articles/8184 "Article: What are trends and what is the structure of the markets - trend or flat? ") will trading be more successful.

Then it falls on me that that market structure is actually a particular candlestick on a higher time frame.

And so, I have also begun coding an independent test module for it. Of course, this article becomes one of my references.

Win ratio is almost guaranteed for a specific candlestick of a certain category, but only for certain time frames.

My own opinion is that I think you cannot trade every candlestick that comes, like what the author did in his case.

Then it will be like throwing multiple darts and hoping at least some of them land accurately and wins exceed losses.

That explains the negative report in this article.

Well I think if more people contribute their ideas, even this seemingly useless article can also become an interesting topic in itself.

Just need more people to see the rationale behind it.

![SpiceTrader](https://c.mql5.com/avatar/avatar_na2.png)

**[SpiceTrader](https://www.mql5.com/en/users/spicetrader)**
\|
19 Sep 2020 at 04:20

Hello,  I read the article with some interest.  I suspect that if you incorporate an extreme [Bollinger Band](https://www.mql5.com/en/code/14 "The Bollinger Bands Indicator (BB) is similar to Envelopes") location into the mix, your accuracy will greatly improve.  By this I mean, that if the price location is outside an extreme Bollinger Band then the likelihood that the correct model is STATE\_REVERSE is increased.  Granted, this will not always be the situation, yet much more likely for the smaller time frame charts.  Similarly, if the current price location is near the median of the Bollinger Band then it might be more likely that the correct model STATE\_NORMAL is accurate.  This is just another widget to throw into your EA to see if the accuracy of the adaptive model can be improved.  Thanks again for your article.


![Trading Signal Generator Based on a Custom Indicator](https://c.mql5.com/2/0/icustom_ava.png)[Trading Signal Generator Based on a Custom Indicator](https://www.mql5.com/en/articles/691)

How to create a trading signal generator based on a custom indicator? How to create a custom indicator? How to get access to custom indicator data? Why do we need the IS\_PATTERN\_USAGE(0) structure and model 0?

![Building an Automatic News Trader](https://c.mql5.com/2/0/cover.png)[Building an Automatic News Trader](https://www.mql5.com/en/articles/719)

This is the continuation of Another MQL5 OOP class article which showed you how to build a simple OO EA from scratch and gave you some tips on object-oriented programming. Today I am showing you the technical basics needed to develop an EA able to trade the news. My goal is to keep on giving you ideas about OOP and also cover a new topic in this series of articles, working with the file system.

![Expert Advisor for Trading in the Channel](https://c.mql5.com/2/17/834_22.gif)[Expert Advisor for Trading in the Channel](https://www.mql5.com/en/articles/1375)

The Expert Advisor plots the channel lines. The upper and lower channel lines act as support and resistance levels. The Expert Advisor marks datum points, provides sound notification every time the price reaches or crosses the channel lines and draws the relevant marks. Upon fractal formation, the corresponding arrows appear on the last bars. Line breakouts may suggest the possibility of a growing trend. The Expert Advisor is extensively commented throughout.

![How Reliable is Night Trading?](https://c.mql5.com/2/17/841_4.gif)[How Reliable is Night Trading?](https://www.mql5.com/en/articles/1373)

The article covers the peculiarities of night flat trading on cross currency pairs. It explains where you can expect profits and why great losses are not unlikely. The article also features an example of the Expert Advisor developed for night trading and talks about the practical application of this strategy.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/1374&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071960728832782762)

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