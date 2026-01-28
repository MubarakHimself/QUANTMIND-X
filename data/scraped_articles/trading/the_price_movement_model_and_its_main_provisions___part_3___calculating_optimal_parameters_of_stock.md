---
title: The price movement model and its main provisions. (Part 3): Calculating optimal parameters of stock exchange speculations
url: https://www.mql5.com/en/articles/12891
categories: Trading, Trading Systems, Indicators
relevance_score: 0
scraped_at: 2026-01-24T13:30:10.569277
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/12891&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082907874620870960)

MetaTrader 5 / Trading


### Introduction

In the previous articles ( [Part 1](https://www.mql5.com/en/articles/10955) and [Part 2](https://www.mql5.com/en/articles/11158)), I presented the fundamental principles and latent mechanisms for generating price dynamics, which was purely theoretical in nature and even went beyond the scope of what was observed (being, however, the basis of it). In this and subsequent articles, I will try to lay the foundations of a new engineering discipline (where many calculations will be of an evaluative nature), which would allow users to draw practically useful conclusions from the observed price dynamics and directly apply them in trading. In this article, I will talk about engineering approaches and algorithms that, in general, are able to provide sustainable profits, as well as probabilistic calculations of those optimal take profit and stop loss values that would allow achieving the maximum average profit.

### 1\. Model.

In the previous article ( [Part 2](https://www.mql5.com/en/articles/11158)), I  obtained the (II.3) equation for the price probability flow (for brevity, from now on, the (N) equation of the Part R article is numbered as (R.N), where R is a Roman numerical). Such a probability flow in reduced or observed form is expressed in "up" and "down" price movement probabilities, or, to be more precise, it creates such probabilities. Let's formulate the approach to the practical assessment of such probabilities.

In the discrete time representation  (based on the bar concept), when the ![](https://c.mql5.com/2/56/xi.png)price history segment (Open, Close, High or Low)  is presented as the ![](https://c.mql5.com/2/56/rxi6.png)series (the numeration order here is so that subsequent bars have higher numbers than that of the previous ones), the price moves in discrete steps. On large scales or in case of the quite large ![](https://c.mql5.com/2/56/Mgrite1.png), this allows us to discuss the probabilities of such price leaps assessed for the upward price movement probability, as ![](https://c.mql5.com/2/56/p.png),  where ![](https://c.mql5.com/2/56/Mj.png)  \- number of ![](https://c.mql5.com/2/56/d4p.png)set members, or for the downward one ![](https://c.mql5.com/2/56/q.png), where ![](https://c.mql5.com/2/56/M-.png)  \- number of  ![](https://c.mql5.com/2/56/1c-.png)members.  At the same time, it is possible to calculate the longevity of the average leap

![](https://c.mql5.com/2/56/1_1.png)                                                                                        (1.1)

of the ![](https://c.mql5.com/2/56/xi__1.png)price. In practice, we can find out that a chaotically moving price for the  ![](https://c.mql5.com/2/56/M.png)period deviates from its current average (defined by such probabilities) due to a random walk. The deviation is of the order of

![](https://c.mql5.com/2/56/1_2.png),                                                                                                      (1.2)

(this is confirmed by the [Casual Channel](https://www.mql5.com/en/market/product/71806) indicator whose channel lines are ![](https://c.mql5.com/2/56/Casu_Chan.png)  or deviations (1.2) from the moving average of the ![](https://c.mql5.com/2/56/M__1.png)period).

![](https://c.mql5.com/2/56/1._EURUSDM15.png)

Fig. 1. [Casual Channel](https://www.mql5.com/en/market/product/71806) indicator

Obviously, the characteristic time of the random price deviation by ![](https://c.mql5.com/2/56/delt_x.png)  of the order of ![](https://c.mql5.com/2/56/T.png) , where ![](https://c.mql5.com/2/56/Tij.png)  is a temporary bar length of the appropriate timeframe.  We would see the same deviation (1.2) from the price from the average, if the price randomly moved by similar leaps exactly equal to ![](https://c.mql5.com/2/56/delt.png).

Therefore, in the model simplification provided here, we will assume that the price moves in similar ![](https://c.mql5.com/2/56/delt__1.png)leaps, with the probabilities of their directions of ![](https://c.mql5.com/2/56/p_.png)  and  ![](https://c.mql5.com/2/56/q_.png).

The ![](https://c.mql5.com/2/56/delt__2.png)leap, calculated using the methods described above, as well as the proabilities were already relevant  for the previous ![](https://c.mql5.com/2/56/ixi8.png)interval as a whole, i.e. as averages for the value range, rather than for the current price movements formed under the influence of different  ![](https://c.mql5.com/2/56/Delt_pred.png)leaps and, most importantly, ![](https://c.mql5.com/2/56/p_pred.png)and  ![](https://c.mql5.com/2/56/q_pred.png)probabilities that is yet to be forecast.

As I mentioned in [Part 2](https://www.mql5.com/en/articles/11158), **the methods of conventional statistics and its mathematical apparatus are not suitable in case of the price dynamics formed out of superpositions of the probability waves** causing considerable errors. Thus, the analysis based on applying observed data and the appropriate probabilistic and statistic calculations have approximate nature.

### 2\. Practical determination of previously operating probabilities and normalized price velocity. The principle of using these parameters to calculate the future price distribution.

Average speed of price movement over the averaging interval ![](https://c.mql5.com/2/56/M__2.png) is equal to

![](https://c.mql5.com/2/56/2_1.png)                                                                   (2.1)

and shows the ![](https://c.mql5.com/2/56/2_1_1.png)moving average change rate (where ![](https://c.mql5.com/2/56/i.png)  is a bar index) with the appropriate averaging period, and not some kind of speed fluctuations. Therefore, the average speed is a practically calculable quantity

![](https://c.mql5.com/2/56/2_2.png) .         (2.2)

Equating (2.1) to (2.2), we obtain an empirically determined parameter

![](https://c.mql5.com/2/56/2_3.png) ,                                                                                           (2.3)

Let's call this a _normalized price speed_, since ![](https://c.mql5.com/2/56/2_3_1.png) , while ![](https://c.mql5.com/2/56/2_3_2.png). ![](https://c.mql5.com/2/56/2_3_3.png) is easy to demonstrate. In fact, for example, from (1.1) follows the inequality

![](https://c.mql5.com/2/56/2_4.png) ,                                                                           (2.4)

entailing  ![](https://c.mql5.com/2/56/2_3_3__1.png) together with (2.3). We can also assume ![](https://c.mql5.com/2/56/2_4_1.png), and since the probability is ![](https://c.mql5.com/2/56/2_4_2.png), then ![](https://c.mql5.com/2/56/2_3_1__1.png). Use (2.3)  and  ![](https://c.mql5.com/2/56/2_4_3.png)to find the probabilities

![](https://c.mql5.com/2/56/2_5_1.png)  and ![](https://c.mql5.com/2/56/2_5_2.png) ,                                                                        (2.5)

as well as another expression for the normalized speed

![](https://c.mql5.com/2/56/2_6.png) ,                                                                                    (2.6)

where the parameter

![](https://c.mql5.com/2/56/2_7.png) .                                                                   (2.7)

In further calculations, we will also need the parameter

![](https://c.mql5.com/2/56/2_8.png) ,                                                            (2.8)

through which the normalized speed itself is expressed as follows

![](https://c.mql5.com/2/56/2_9.png)  .                                                                                            (2.9)

The (2.5) probabilities of leaps calculated on the ![](https://c.mql5.com/2/56/2_9_1.png) price interval are the average ones for the interval and participate in forming the ![](https://c.mql5.com/2/56/2_9_2.png)end price. Therefore, if we know (at the current moment of ![](https://c.mql5.com/2/56/i__1.png)beforehand when the ![](https://c.mql5.com/2/56/xi__2.png)price is known) these average probabilities, we can predict the price ![](https://c.mql5.com/2/56/2_9_3.png) in the future  ![](https://c.mql5.com/2/56/2_9_4.png), or, to be more precise, the probability distribution ![](https://c.mql5.com/2/56/2_9_5.png) of the price. When comparing the ![](https://c.mql5.com/2/56/xi__3.png)price charts and the ![](https://c.mql5.com/2/56/bi.png) normalized speed (from which ![](https://c.mql5.com/2/56/pi.png) and ![](https://c.mql5.com/2/56/qi.png) are calculated), we can trace their strong similarity (by the coincidence of their vertices locations), showing that it was this speed (more precisely, the probabilities corresponding to it) that formed the current price of ![](https://c.mql5.com/2/56/xi__4.png).

![](https://c.mql5.com/2/56/2._EURUSDM30.png)

Fig. 2. The figure shows the normalized speed graph.

Indeed, from (2.3) it follows ![](https://c.mql5.com/2/56/2_9_6.png). This means that the ![](https://c.mql5.com/2/56/2_9_2__1.png) price is formed from the ![](https://c.mql5.com/2/56/xi__5.png) price by the array of future velocities ![](https://c.mql5.com/2/56/2_9_7.png) or their averages by the interval

![](https://c.mql5.com/2/56/2_10.png),                         (2.10)

where the average normalized speed over the future interval, displayed on its graph at the point   ![](https://c.mql5.com/2/56/2_9_4__1.png)

![](https://c.mql5.com/2/56/2_11.png),                                                     (2.11)

while the averages of the probability interval ![](https://c.mql5.com/2/56/2_11_1.png) and ![](https://c.mql5.com/2/56/2_11_2.png) are found substituting ![](https://c.mql5.com/2/56/2_11_3.png) to the equations (2.5) (obviously, if in (2.10) the ![](https://c.mql5.com/2/56/2_11_4.png)member is around ![](https://c.mql5.com/2/56/xi__6.png) , then we have the graphs resembling ![](https://c.mql5.com/2/56/2_9_2__2.png) and ![](https://c.mql5.com/2/56/2_11_3__1.png)). Then, having predicted sufficiently smooth functions ![](https://c.mql5.com/2/56/2_11_5.png) on ![](https://c.mql5.com/2/56/M__3.png) bars forward, we calculate the necessary probabilities ![](https://c.mql5.com/2/56/2_11_1__1.png) and ![](https://c.mql5.com/2/56/2_11_2__1.png), which allows us to calculate (at the moment of ![](https://c.mql5.com/2/56/i__2.png)) the  ![](https://c.mql5.com/2/56/2_9_5__1.png) probability distribution of the ![](https://c.mql5.com/2/56/2_9_2__3.png) future price and its parameters necessary for trading (position opening direction and stop order position).

The essence of the normalized speed forecast used here is as follows. The ![](https://c.mql5.com/2/56/beta.png) temporary function of the normalized speed fluctuates within the range of ![](https://c.mql5.com/2/56/1__2.png) near its mathematical expectation equal to zero (or the ![](https://c.mql5.com/2/56/beta0.png) small value displaying the velocity of the global trend, if it covers the entire area under consideration). In this case, for example, the simplest statistical forecast based on the conditional mathematical expectation taking the form of ![](https://c.mql5.com/2/56/2__2.png) will bring the predictive function closer to zero or ![](https://c.mql5.com/2/56/beta0__1.png) according to ![](https://c.mql5.com/2/56/3__1.png). In other words, as the autocorrelation function of the corresponding process ![](https://c.mql5.com/2/56/beta__1.png) decreases sharply reducing the number of positions opened by a condition like ![](https://c.mql5.com/2/56/4__1.png) and making the game even less profitable, than a game with a trivial forecast based on the last value ![](https://c.mql5.com/2/56/5__2.png). On the other hand, price dynamics are well modeled and predicted by oscillatory processes. The idea behind this was revealed in previous articles. At this stage of theory development, the ![](https://c.mql5.com/2/56/6__3.png)forecast of the ![](https://c.mql5.com/2/56/beta__2.png) function (having an oscillatory nature) on ![](https://c.mql5.com/2/56/7__2.png) bars forward was made based on Fourier extrapolation calculated on the basis of empirical historical data ![](https://c.mql5.com/2/56/beta__3.png), since the use of the wavelet extrapolation proposed in previous articles in this case has not yet provided noticeable advantages.

### 3\. Trend quality. Assessing the extent of current and future trends, adequate work horizon.

If the trend lasts longer than the ![](https://c.mql5.com/2/56/3_0.png) averaging time, then the natural increase in price during the averaging time (according to (2.1) and (2.3)) is of the order of

![](https://c.mql5.com/2/56/3_1.png) .                                    (3.1)

Increment uncertainty

![](https://c.mql5.com/2/56/3_2.png) ,                                                                                               (3.2)

then the full range of price movement (see Fig. 1), when it moves from one border of the [Casual Channel](https://www.mql5.com/en/market/product/71806)indicator channel to another and at the same time drifts on average at the ![](https://c.mql5.com/2/56/beta__4.png) normalized speed, is estimated as

![](https://c.mql5.com/2/56/3_3.png) ,                                                               (3.3)

where

![](https://c.mql5.com/2/56/3_4.png) .                                                                                 (3.4)

If we follow the trend, then it is desirable that the value of the shift (3.1) remains significantly greater than the uncertainty (3.2) of this shift

![](https://c.mql5.com/2/56/3_5.png) ,                                                                (3.5)

from it and from (2.6) and (2.9), we obtain a lower estimate for the required averaging time

![](https://c.mql5.com/2/56/3_6.png) ,                                                   (3.6)

when fulfilled, in accordance with (3.4), the averaging time (in bars) is calculated as

![](https://c.mql5.com/2/56/3_7.png) .                                                                                  (3.7)

If the ![](https://c.mql5.com/2/56/delt_x__1.png) uncertainty price increment cannot be neglected, then the averaging time is considered as the positive root of the quadratic equation (3.4)

![](https://c.mql5.com/2/56/3_7_1.png) .                                                                                        (3.7.1)

Define the _trend quality_

![](https://c.mql5.com/2/56/3_8.png) ,                                                                                        (3.8)

as the ratio of its natural increment to its uncertainty or noise. It is quite clear that a stable profitable trend-following strategy requires the high ![](https://c.mql5.com/2/56/3_8_1.png) quality. But the [Quality Trend](https://www.mql5.com/en/market/product/79505) indicator calculating the quality (Fig. 3), reaches the value of several units for a currency at best. Moreover, even having identified a high-quality trend, it is not possible to determine when it will end due to the unpredictability of the emergence of strong external events that can disrupt the market’s own dynamics and end the trend or even reverse it. Therefore, a profitable strategy can only be based on taking profits on relatively small fluctuations in the direction of the trend .

![](https://c.mql5.com/2/56/3._EURUSDH4.png)

Fig. 3. The [Quality Trend](https://www.mql5.com/en/market/product/79505) indicator where the price increment is not taken modulo, i.e. the sign of the indicator indicates the direction of the trend.

Note that [Quality Trend](https://www.mql5.com/en/market/product/79505) indicator readings being proportional to the normalized speed, as was shown by the (2.10) ratio for it, turn out to be similar to the price history in the positions of their peaks and, accordingly, the [Quality Trend](https://www.mql5.com/en/market/product/79505) readings turn out to not lag behind. Moreover, the indicator readings may surpass (which they often do) price movements, since even before the trend changes, the corresponding speed of a price movement (growth for an uptrend or fall for a downtrend) decreases. However, such predictive behavior of this indicator occurs only in the absence of strong influences on the market that disrupt its own movement. After such influences, during their relaxation time, the [Quality Trend](https://www.mql5.com/en/market/product/79505) readings become "ordinary" lagging ones, with the lag determined by its averaging period.

Let's analyze the behavior of the ![](https://c.mql5.com/2/56/QtMr.png) function and evaluate the possible extent of trends, provided there are no strong third-party influences on the market. With the decreasing ![](https://c.mql5.com/2/56/M__4.png)average period, the calculated values ![](https://c.mql5.com/2/56/beta__5.png) of the normalized velocity may increase (after all, its “instant” values in a constantly changing market change with a greater amplitude than average ones and the greater the averaging, the smaller such variations), i.e. in the quality factor ratio, the members, for which ![](https://c.mql5.com/2/56/1__3.png) and vice versa ![](https://c.mql5.com/2/56/2__3.png)are multiplied, which makes it possible for the ![](https://c.mql5.com/2/56/QsMd.png)function to have maximums. However, on a very small interval, when the true probabilities of ![](https://c.mql5.com/2/56/p_r.png) and ![](https://c.mql5.com/2/56/q_r.png) leaps are constant, due to small changes in the market situation on it, their statistically calculated values ![](https://c.mql5.com/2/56/p___1.png) and ![](https://c.mql5.com/2/56/q___1.png) over this short averaging period, will most likely differ greatly from the true probabilities, because with the uncertainty period decrease, the ![](https://c.mql5.com/2/56/dp.png) and ![](https://c.mql5.com/2/56/dq.png) of the probabilities calculated on it increase. Therefore, for the ![](https://c.mql5.com/2/56/M__5.png) averaging period providing the calculation of more or less reliable probabilities, the ![](https://c.mql5.com/2/56/3__2.png) type ratios should be satisfied, which will determine its minimum value. Otherwise, when ![](https://c.mql5.com/2/56/4__2.png) (although this is a broader case than the case of large fluctuations in the instantaneous normalized velocity at small averaging intervals, since such a ratio can also occur at large intervals with a rapid change in the true probability ![](https://c.mql5.com/2/56/p_r__1.png)), statistically calculated probability values cannot be used. Note that in cases of strong fluctuations of the normalized velocity, the ![](https://c.mql5.com/2/56/Q9Mc.png) function based on it also fluctuates strongly near its maximum, so the maximum used to analyze the market situation should be chosen such that it forms smoothly, which, as follows from the above, is achieved with sufficiently large averaging periods. If the ![](https://c.mql5.com/2/56/5__3.png) condition is met, which is assumed to be fulfilled in the further theory, the estimated probabilities ![](https://c.mql5.com/2/56/p___2.png) and ![](https://c.mql5.com/2/56/q___2.png) can be identified with acting ![](https://c.mql5.com/2/56/p_r__2.png) and ![](https://c.mql5.com/2/56/q_r__1.png) probabilities, which then we will also write as ![](https://c.mql5.com/2/56/p___3.png) and ![](https://c.mql5.com/2/56/q___3.png).

Precisely in those areas where the ![](https://c.mql5.com/2/56/p___4.png) and ![](https://c.mql5.com/2/56/q___4.png) probabilities are constant, a stable trend is formed, while the drop in the prevailing probability ![](https://c.mql5.com/2/56/p___5.png) (i.e. ![](https://c.mql5.com/2/56/7__3.png) ) will reduce the growth rate of the increment ![](https://c.mql5.com/2/56/6__4.png) and maybe even (when the inverse relationship ![](https://c.mql5.com/2/56/8__1.png) is achieved) reverse the trend, which also leads to a drop in the calculated quality factor. On the contrary, high quality factor and its growth indicate not only a strong predominance of the prevailing probability ![](https://c.mql5.com/2/56/p___6.png) over ![](https://c.mql5.com/2/56/q___5.png), but also about its constancy and even increase. . Therefore, the higher the trend quality (3.8), the greater the probability that it is present there, i.e. over the entire ![](https://c.mql5.com/2/56/M__6.png) interval, while the low quality value ![](https://c.mql5.com/2/56/9__2.png) indicates a flat. It is also clear that if we increase the averaging period ![](https://c.mql5.com/2/56/M__7.png)covering not only a trend (with the length of ![](https://c.mql5.com/2/56/10.png) ) but also a flat and, moreover, a section of price history with an oppositely directed trend, then the quality factor will drop sharply; therefore the length ![](https://c.mql5.com/2/56/Mx.png) of the trend is identified by the ![](https://c.mql5.com/2/56/11.png) peak quality factor.

If we increase the ![](https://c.mql5.com/2/56/M__8.png)period so that we cover with it a unidirectional trend of a larger scale ![](https://c.mql5.com/2/56/12.png), than the scale of a smaller trend section with the length of ![](https://c.mql5.com/2/56/Mx__1.png), the trend quality, on the contrary, increases, since due to the similarity of charts in different timeframes (provided that ![](https://c.mql5.com/2/56/3__3.png)), the scale of normalized velocities in (3.8) remains almost unchanged with an increase in ![](https://c.mql5.com/2/56/M__9.png), while ![](https://c.mql5.com/2/56/sqrtM.png) will increase. In addition (this already requires the ![](https://c.mql5.com/2/56/delt_x__2.png) value calculation correction, appearing in the quality equation (3.8)), noise contamination of trend is sharply amplified by large and small chaotic price jumps that go beyond the statistical distribution formed by "standard" leaps (corresponding to the model under consideration) with the probabilities of ![](https://c.mql5.com/2/56/p_r__3.png) and ![](https://c.mql5.com/2/56/q_r__2.png). Such non-standard jumps are the same for all scales and create “additional” noise to the trend, so the weight of this additional noise decreases with increasing scales the trend is identified at. All of the above entails the fact that the ![](https://c.mql5.com/2/56/Q6Mv.png) function can be used to define a number of quality peaks in case of a global unidirectional trend, which will go up with an increase in ![](https://c.mql5.com/2/56/M__10.png) or an increase in the scale of identified trend areas.

![](https://c.mql5.com/2/56/4._USDCHFH1.png)

Fig. 4. Current ![](https://c.mql5.com/2/56/QpMx.png) function. The X axis here displays the averaging period for [Quality Trend](https://www.mql5.com/en/market/product/79505) from 10 to 160, rather than time.

Finally, the game is not based on an already formed history, but in real time, the knowledge of a number of forecasts ![](https://c.mql5.com/2/56/13.png), based on the ![](https://c.mql5.com/2/56/14.png) set of normalized velocity forecast values, is necessary. Therefore, to assess the possible length ![](https://c.mql5.com/2/56/15.png) of a newly emerging trend, we need to go through the entire spectrum of averaging periods and identify a number of maximums of the forecast quality ![](https://c.mql5.com/2/56/16.png), when its forecasting goes forward by ![](https://c.mql5.com/2/56/M__11.png) bars from the current bar, i.e. calculated

![](https://c.mql5.com/2/56/3_9.png) ,                                                                 (3.9)

where ![](https://c.mql5.com/2/56/3_9_1.png)  is an identification function of the ![](https://c.mql5.com/2/56/i__3.png)th maximum. In this case, we also need to set the maximum peak

![](https://c.mql5.com/2/56/3_10.png) ,                                                                                    (3.10)

and its corresponding point ![](https://c.mql5.com/2/56/3_10_1.png)on the averaging scale.

It is obvious that at smaller averaging intervals ![](https://c.mql5.com/2/56/3_10_2.png) going to the maximum peak ![](https://c.mql5.com/2/56/Qmax.png) of the forecast quality provided that at these intervals the quality is also significant and grows monotonically or in a sequence of increasing (also based on a growing backlog) peaks, there will be a corresponding unidirectional trend (with rollbacks after each peak of quality). After ![](https://c.mql5.com/2/56/3_10_3.png) of the maximum peak ![](https://c.mql5.com/2/56/Qmax__1.png) of quality, when it begins to fall on the scale of the corresponding averaging ![](https://c.mql5.com/2/56/M__12.png), there is a slowdown in the trend, which may soon lead to a reversal. The latter is most likely when the peak value ![](https://c.mql5.com/2/56/Qmax__2.png)is very significant in the sense that the quality factor for the exchange instrument under consideration rarely reaches values larger than ![](https://c.mql5.com/2/56/Qmax__3.png) . In any case, the increasing trend will continue until the mark ![](https://c.mql5.com/2/56/3_10_1__1.png), before reaching which we need to close the position opened according to this trend.

Let us now try to estimate the lengths ![](https://c.mql5.com/2/56/My.png) of trend segments that are promising for trading, which do not necessarily have to be equal to the predicted length of the ![](https://c.mql5.com/2/56/15__1.png) trend. _Firstly,_, due to the low reliability of the work of predictive mathematics itself (which applies to all its types, even various frequency and other extrapolators, including neural networks and ARIMA, etc.), the profit should be taken on relatively small segments ![](https://c.mql5.com/2/56/My__1.png) of the identified future trend ![](https://c.mql5.com/2/56/15__2.png), the trend is more likely to form on. Therefore, as follows from the previous paragraph, the ![](https://c.mql5.com/2/56/3_10_4.png)inequality should necessarily be fulfilled. _Second_, the presented model uses estimates of future probability values ![](https://c.mql5.com/2/56/p_pred__1.png) and ![](https://c.mql5.com/2/56/q_pred__1.png), as well as the average ![](https://c.mql5.com/2/56/3_10_5.png)leaps are assumed to be constant, since there are forecasts that work when the market develops by inertia and according to its own laws. However, as shown in the first article ( [Part 1](https://www.mql5.com/en/articles/10955)), **the interval of the predictable market development begins from the last strong external event and continues until the occurrence of the next such event**. Therefore, there is the _adequate work horizon_![](https://c.mql5.com/2/56/3_10_6.png) of the entire mathematical apparatus being developed here, where the quantity ![](https://c.mql5.com/2/56/3_10_6__1.png) is equal to the number of bars from the current bar to the future bar of the onset of a strong external event. If we try to use such a mathematical apparatus (which is extremely important) beyond a given horizon, this will cause errors in its operation and inevitable losses. To determine the possible horizon for the adequate operation of such a mathematical apparatus, it is necessary to build on fundamental analysis or expert research assessing the strength of influence of all current and future political and economic events on the state of the market. Therefore, the length of the forecast trend section promising for trading is estimated from above by the ratio

![](https://c.mql5.com/2/56/3_11.png) ,                                                                        (3.11)

and from below it should be estimated on the basis of the previously established ratio of the smallness of the probability uncertainty (fluctuations) compared to the probability itself

![](https://c.mql5.com/2/56/3_12.png) ,                                                                                    (3.12)

which is also determined from the predicted quality factor graph ![](https://c.mql5.com/2/56/3_12_1.png)and corresponds to those areas where this graph changes quite smoothly. The expected natural price change in this section of the trend is

![](https://c.mql5.com/2/56/3_13.png) ,                                                                                    (3.13)

which corresponds to the order of profit obtained with a purely trend-following strategy.

_Third_, selection of the ![](https://c.mql5.com/2/56/My__2.png) length of the trend segment should also be based on the calculations presented below, which essentially allows us to set the interval ![](https://c.mql5.com/2/56/3_13_1.png) values ![](https://c.mql5.com/2/56/M__13.png), on which it is possible to obtain an average statistical profit under given market conditions, i.e. ![](https://c.mql5.com/2/56/3_13_2.png). In addition to all this, traders chooses the timeframe on their own, and the true (and not the model, calculated by formula (3.8)) quality decreases with a decrease in the timeframe due to price noise on all timeframes by its non-model (identical on all timeframes) leaps. Therefore, traders are offered a choice of options with a high trend quality, but a long wait for profit, which is the case for large timeframes; or quickly making a profit on lower quality trends (and, accordingly, greater risks), which is typical for small timeframes.

### 4\. Probabilistic calculation of the take profit and stop loss values that yield maximum profit at constant operating probabilities and the expression of the latter.

_Setting a task_.

The price moves in jumps in the vertical dimension from the zero mark. The probability of an upward price jump ![](https://c.mql5.com/2/56/p___7.png), the probability of a downward price jump ![](https://c.mql5.com/2/56/q___6.png), respectively, ![](https://c.mql5.com/2/56/1__4.png). Of course, there are predicted averages here ![](https://c.mql5.com/2/56/2__4.png)and ![](https://c.mql5.com/2/56/3__4.png), which is not important now. At the top, there is the take profit at the distance of "a", while at the bottom, there is the stop loss at the distance of "в" from the zero mark (when viewed in the coordinate axes ![](https://c.mql5.com/2/56/4__3.png)). Find the parameters of the stock exchange game that ensure maximum profit.

_Solution_.

              The price can go to the point with coordinate "n" or from below from the point "n-1" or from above from the point "n+1". Therefore, the probability of finding the price at point "n" is equal to

![](https://c.mql5.com/2/56/4_1.png) .                                                                                     (4.1)

From (4.1), we get the finite differences equation

![](https://c.mql5.com/2/56/4_2.png)                                                                  (4.2)

_Equiprobable jumps_.

              Let us first consider the case of equiprobable jumps ![](https://c.mql5.com/2/56/4_2_1.png). Here we get the following from (4.2)

![](https://c.mql5.com/2/56/4_3.png) ,                                           (4.3)

where ![](https://c.mql5.com/2/56/4_3_1.png) is a constant, from which we find

![](https://c.mql5.com/2/56/4_4.png) .                                                                                                 (4.4)

              The probability of the price being at zero at the starting moment of its movement is ![](https://c.mql5.com/2/56/4_4_1.png), therefore,

![](https://c.mql5.com/2/56/4_5.png) .                                                                                                      (4.5)

              Let's assume that the stop loss "в" together with the take profit "a" constitute the characteristic (estimated here (3.4) in average price jumps ![](https://c.mql5.com/2/56/delt__3.png)) price range ![](https://c.mql5.com/2/56/4_5_1.png)of the movement over the ![](https://c.mql5.com/2/56/M__14.png)period of its averaging (and movement) the constancy of probabilities ![](https://c.mql5.com/2/56/p___8.png)and ![](https://c.mql5.com/2/56/q___7.png)is based on. The probability that the price is already at the stop loss point where ![](https://c.mql5.com/2/56/4_5_2.png)reaches a take profit reaches zero is ![](https://c.mql5.com/2/56/4_5_3.png). Substituting (4.5), we obtain

![](https://c.mql5.com/2/56/4_6.png)                                                                                                         (4.6)

together with (4.5), this provides us with the probability of achieving take profit equal to

![](https://c.mql5.com/2/56/4_7.png)                                                                                                         (4.7)

while the probability of a stop loss being triggered

![](https://c.mql5.com/2/56/4_8.png) .                                                                                         (4.8)

Therefore, with equally probable price jumps in different directions, the average profit in the number of jumps

![](https://c.mql5.com/2/56/4_9.png)                                                           (4.9)

is always zero (the spread, of course, makes it negative) regardless of the position of the take profit and stop loss, which can be anything.

_There is a tendency to move towards take profit_.

Let ![](https://c.mql5.com/2/56/4_9_1.png) (or, to be more accurate, ![](https://c.mql5.com/2/56/4_9_2.png)) then, multiplying all equations (4.2), we find

![](https://c.mql5.com/2/56/4_10.png) ,                                                      (4.10)

reducing identical factors in (4.10), using notation (2.7) ![](https://c.mql5.com/2/56/4_10_1.png) and considering that ![](https://c.mql5.com/2/56/4_4_1__1.png), we obtain

![](https://c.mql5.com/2/56/4_11.png) .                                                                                 (4.11)

Let's display ![](https://c.mql5.com/2/56/4_11_1.png)as the sum of the differences of adjacent terms of the probability series ![](https://c.mql5.com/2/56/4_11_2.png)using further relations (4.11) and the equation for summing the geometric progression

![](https://c.mql5.com/2/56/4_12.png) ,             (4.12)

![](https://c.mql5.com/2/56/4_12_1.png) , therefore,

![](https://c.mql5.com/2/56/4_13.png)                                                                                      (4.13)

![](https://c.mql5.com/2/56/4_4_1__2.png) , hence,

![](https://c.mql5.com/2/56/4_14.png) ,                                                                                           (4.14)

dividing (4.13) by (4.14), we obtain the probability of achieving take profit "а"

![](https://c.mql5.com/2/56/4_15.png) .                                                                                                      (4.15)

The probability of the stop loss being triggered is, accordingly, equal to ![](https://c.mql5.com/2/56/4_15_1.png). Then the average profit **per one position** in price jumps ![](https://c.mql5.com/2/56/delt__4.png) is equal to

![](https://c.mql5.com/2/56/4_16.png) ,         (4.16)

which in the representation (4.16) is a function of the stop loss value "в", which in this representation is simply a number of ![](https://c.mql5.com/2/56/4_16_1.png)jumps, but in fact there is a value of ![](https://c.mql5.com/2/56/4_16_2.png). The profit is ![](https://c.mql5.com/2/56/4_16_3.png). It is clear that with an increase in the probability of price movement towards an open position, the average profit (4.16) increases. When ![](https://c.mql5.com/2/56/4_16_4.png) takes the value of ![](https://c.mql5.com/2/56/4_16_5.png), i.e. ![](https://c.mql5.com/2/56/4_16_6.png) is a growing function from ![](https://c.mql5.com/2/56/4_16_7.png).

Let's find the maximum of the average statistical profit (4.16) subject to the given values of _**N**_ and ![](https://c.mql5.com/2/56/4_16_7__1.png). To achieve this, let's equate its derivative to zero

![](https://c.mql5.com/2/56/4_17.png),                                                             (4.17)

from where we find the value of the desired stop loss in ![](https://c.mql5.com/2/56/delt__5.png) price leaps

![](https://c.mql5.com/2/56/4_18.png) .                                                                                        (4.18)

Since ![](https://c.mql5.com/2/56/4_18_1.png), the ![](https://c.mql5.com/2/56/4_18_2.png) logarithm is positive and ![](https://c.mql5.com/2/56/4_18_3.png). Accordingly, the logarithm ![](https://c.mql5.com/2/56/4_18_4.png) value should be positive. This condition is satisfied if

![](https://c.mql5.com/2/56/4_19.png) ,                                                                                               (4.19)

or

![](https://c.mql5.com/2/56/4_20.png) ,                                                      (4.20)

where ![](https://c.mql5.com/2/56/4_20_1.png). The (4.20) inequality is strictly satisfied for any ![](https://c.mql5.com/2/56/4_20_2.png) , since the ![](https://c.mql5.com/2/56/4_20_3.png)exponent passes above the straight line ![](https://c.mql5.com/2/56/4_20_4.png) touching it only at ![](https://c.mql5.com/2/56/4_20_5.png).

The second derivative of the function (4.16)

![](https://c.mql5.com/2/56/4_21.png)                                                              (4.21)

is always negative under these conditions, i.e. the curvature of the ![](https://c.mql5.com/2/56/4_16_6__1.png) function is directed downward and we have the maximum at (4.18). The function (4.16) at N=100 and ![](https://c.mql5.com/2/56/4_21_1.png) is displayed in Fig. 5.

![](https://c.mql5.com/2/56/5__1.png)

Fig.5. Dependence of the profit function on stop loss.

Keep in mind that in order for the average profit ![](https://c.mql5.com/2/56/4_16_6__2.png)to be positive, the ![](https://c.mql5.com/2/56/4_16_7__2.png) ratio should significantly exceed one. Indded, if ![](https://c.mql5.com/2/56/4_21_2.png), where  ![](https://c.mql5.com/2/56/4_21_3.png)and we can neglect the second term of the expansion, leaving only the first term ![](https://c.mql5.com/2/56/4_21_4.png), then the average profit per trade

![](https://c.mql5.com/2/56/4_22.png)    (4.22)

will be equal to zero (as in the case of equal probabilities of opposite jumps). If the second term of the expansion cannot be neglected, then taking into account that the number of jumps ![](https://c.mql5.com/2/56/4_22_1.png) is big enough or ![](https://c.mql5.com/2/56/4_22_2.png), we have

![](https://c.mql5.com/2/56/4_23.png) ,                               (4.23)

which will give a positive value for the average profit (4.16)

![](https://c.mql5.com/2/56/4_24.png) ,                          (4.24)

since ![](https://c.mql5.com/2/56/4_24_1.png),  ![](https://c.mql5.com/2/56/4_24_2.png) (and, therefore, ![](https://c.mql5.com/2/56/4_24_3.png)).

              The approximate average profit (4.24) relative to the argument ![](https://c.mql5.com/2/56/4_16_1__1.png)is an inverted quadratic parabola whose maximum is reached at ![](https://c.mql5.com/2/56/4_24_4.png) (which is the equality of stop loss and take profit), when ![](https://c.mql5.com/2/56/4_24_5.png).

Here is a very important point. In the theory presented above, the average profit was calculated only on the basis of average price values, which, in fact, fluctuates greatly and can even greatly exceed the corresponding average shifts in the range of its fluctuations. However, stop orders (take profit and stop loss) are closed not at the average price values, but precisely at the edges of the band of its fluctuations. Therefore, in order for the presented mathematical apparatus (based on average values) to work, the stop loss should greatly exceed ![](https://c.mql5.com/2/56/4_24_6.png) the price uncertainty ![](https://c.mql5.com/2/56/delt_x__3.png) (so that its fluctuation triggering differs little from the model triggering in terms of the average and these fluctuations can be neglected), i.e., according to (1.2),

![](https://c.mql5.com/2/56/4_25.png) .                                                                                                         (4.25)

In this case, using the (3.7.1) expressions for the averaging period, get from (4.25) the function the following inequality should be satisfied for

![](https://c.mql5.com/2/56/4_26.png) ,                                                                 (4.26)

which is a criterion for the smallness of price fluctuations, where ![](https://c.mql5.com/2/56/beta__6.png) is found from the (2.6) ratio. By substituting into (4.26) the (4.18) stop loss, we get the function graph (Fig. 6), which makes clear that such a function is not much greater than zero, but, on the contrary, is fundamentally negative, i.e. the ratio (4.25) is never satisfied with the optimal stop loss (4.18).

MATLAB code

```
>> [N,a]=meshgrid([3:200],[1.01:0.01:3]);
>> b=log(N.*log(a)./(1-a.^(-N)))./log(a);
>> beta=(a-1)./(a+1);
>> s=(N.*beta+1).^(1/2)./beta;
>> y=b-s;
>> plot3(N,a,y)
>> grid on
```

![](https://c.mql5.com/2/56/6__2.png)

Fig.6. "y" function graph when changing Alpha from 1 to 3 and changing N from 3 to 200.

Thus, the use of the stop order values calculated above will lead to average statistical losses, since price fluctuations turn out to be fundamentally greater than the optimal one in the model of its average stop loss movement

![](https://c.mql5.com/2/56/4_27.png) .                                                                                                              (4.27)

This means we need to change the size of the stop loss itself, rather than look for the averaging period which makes the optimal stop loss (4.18) relatively small (4.25) (since this task has no solution). This will, of course, change the profit as well.

The optimal take profit for the average price movement model coincides with the point of the forecast moving average price, which is located ![](https://c.mql5.com/2/56/M__15.png)bars ahead of the current bar. But if we take into account strong price deviations from the average, at which stop orders are closed, then (as can be seen in the [Casual Channel](https://www.mql5.com/en/market/product/71806) indicator graph in Fig. 1), to ensure a profitable game, such an optimal take profit should be reduced by an amount greater than the average deviation

![](https://c.mql5.com/2/56/4_28.png) ,                                                                                                  (4.28)

where the ![](https://c.mql5.com/2/56/4_28_1.png) ratio should be slightly greater than one for weak trends (having almost no profit) and approximately ![](https://c.mql5.com/2/56/4_28_2.png) for strong trends, being here exactly the parameter whose exact value should be sought through optimization, and the stop loss should be increased by the same amount, i.e.

![](https://c.mql5.com/2/56/4_29.png) .                                                                                                  (4.29)

Then, a stop loss, as a value separated from the average value of the possible price deviation (for ![](https://c.mql5.com/2/56/M__16.png) future bars) against an open position by ![](https://c.mql5.com/2/56/4_29_1.png), will be triggered much less often with a probability lower than ![](https://c.mql5.com/2/56/qb.png), while take profit will be triggered more often with a greater probability exceeding ![](https://c.mql5.com/2/56/pa.png). Accordingly, for the maximum profit, we obtain the estimate

![](https://c.mql5.com/2/56/4_30.png) , (4.30)

where ![](https://c.mql5.com/2/56/4_16_6__3.png) is a value from (4.16), or considering (3.7.1)

![](https://c.mql5.com/2/56/4_31.png)                                                     (4.31)

whose function (in case of the optimal **_b_** from (4.18)) can be constructed, so that we are able to find the _**N**_ value maximizing it, as well as the averaging period.

MATLAB code for **_k=3_**

```
>> [N,a]=meshgrid([3:200],[1.01:0.01:3]);
>> b=log(N.*log(a)./(1-a.^(-N)))./log(a);
>> beta=(a-1)./(a+1);
>> s=(N.*beta+1).^(1/2)./beta;
>> s0=N.*(1-a.^(-b))./(1-a.^(-N))-b;
>> Profit=s0-3*s;
>> plot3(N , a, Profit)
>> grid on
```

![](https://c.mql5.com/2/56/7__1.png)

Fig.7. Profit function graph (in model price jumps) as Alpha changes from 1 to 3 and N from 3 to 200.

The graph shows that the profit with a positive mathematical expectation is generally possible and increases with increasing Alpha and _**N**_.

To find the most promising averaging period ![](https://c.mql5.com/2/56/My__3.png), we need to construct a predictive quality factor function ![](https://c.mql5.com/2/56/4_31_1.png). This is why the [CalculateScientificTradePeriod](https://www.mql5.com/en/market/product/99909) script has been developed. We need to define the most promising period ![](https://c.mql5.com/2/56/My__4.png)by the ![](https://c.mql5.com/2/56/Qmax__4.png) maximum location where ![](https://c.mql5.com/2/56/4_31_2.png)when this maximum is reached smoothly (the (3.12) ratio is fulfilled) and is not located further than _the adequate work horizon_. If the ![](https://c.mql5.com/2/56/My__5.png)value found in this way provides a positive (as well as exceeding at least a couple of spreads) profit value (4.31) and a sufficiently high probability of winning (4.15), which sets the interval ![](https://c.mql5.com/2/56/3_13_1__1.png), then the trading decision should be based on it here. In order to calculate the optimal (maximizing the average profit) take profit and stop loss, as well as to determine the further price trend, I have developed the [ScientificTrade](https://www.mql5.com/en/market/product/98467) indicator, whose algorithms are based on the entire theory presented above.

Note that the [CalculateScientificTradePeriod](https://www.mql5.com/en/market/product/99909) script algorithm is very resource-intensive, so we use the script, not the indicator, which would run this algorithm on every tick and would freeze the computer. The [FindScientificTradePeriod](https://www.mql5.com/en/market/product/99926) indicator is used to display data calculated by the script.

![](https://c.mql5.com/2/56/8.png)

Fig. 8. [ScientificTrade](https://www.mql5.com/en/market/product/98467) and [FindScientificTradePeriod](https://www.mql5.com/en/market/product/99926) indicators.

![](https://c.mql5.com/2/56/9.gif)

Fig. 9. [ScientificTrade](https://www.mql5.com/en/market/product/98467) indicator results.

### 5\. Irremovable error of applied calculations within the framework of the mathematical apparatus itself. The approach to identifying moments of naturally occurring price rebounds and reversals.

As was previously said, the trends predicted by the [ScientificTrade](https://www.mql5.com/en/market/product/98467) indicator and calculated by the stop loss and take profit location indicator are based on the ![](https://c.mql5.com/2/56/p_pred__2.png) and ![](https://c.mql5.com/2/56/q_pred__2.png)forecast values, which may turn out to be erroneous due to the unreliability of the forecasting apparatus itself (in the Fourier extrapolator indicator). Therefore, such forecasts may turn out to be false within the _adequate work horizon_ of all the mathematical apparatus presented above.

To exclude at least some cases of false mathematical forecasts, trends calculated by the ScientificTrade indicator on the interval determined by the [CalculateScientificTradePeriod](https://www.mql5.com/en/market/product/99909) script, should coincide with the forecast trends provided by authoritative fundamental analysis experts for a given interval. It is clear that if both ScientificTrade and the experts provide the same false forecast (which we cannot know about), then losses are also inevitable. According to my subjective observations, experts are more likely to make mistakes than the ScientificTrade indicator in combination with the CalculateScientificTradePeriod script, which, in my opinion, is due to the fact that inner laws of market development have a stronger impact than most external events causing changes in trends experts are not able to determine. Moreover, such market reversals caused by internal reasons often occur before the onset of strong external events, as well as more often than those events. The appropriate mechanisms will be discussed below.

To express the essence of the above problem, it should first be noted that the price, does not always move in equidistant jumps even when the market develops according to its own laws (when the price is not pushed by strong external events). This concept is a simplified model, which allows us to understand at least something and make calculations in the market chaos. In reality, the price occasionally makes short-term (beyond the scope of classical statistics) strong movements contrary to average static trends, according to which it slowly drifts (with large fluctuations) in a certain direction. Moreover, if such strong movements are not caused by external influences on the market, but originate from its inner processes, then they are usually directed against statistical trends. Therefore, such strong movements knock down stop losses and create maximum losses for most small traders.

In fact (if we exclude the purposeful knocking down of stop losses by providers of trading services and quotes), Le Chatelier’s principle works here in conjunction with the dialectical law of the transition of quantity into quality. Upon reaching a specific (also dependent on the market) growth (or fall) level of a certain market instrument, a sharp jump in quality occurs, which, in accordance with Le Chatelier’s principle (the action of which extends to any complex system in equilibrium, including the economy, which most of the time evolves passing through close quasi-equilibrium states), tends to resist the growth of the above-mentioned quantity sharply lowering it (in a jump). Since the market as a system, with its monotonous development, gradually passes through close quasi-equilibrium states, Le Chatelier’s principle does not react to its small changes, but acts abruptly when large quantitative changes are already accumulating in a given system. From the standpoint of the market wave model ( [Part 1](https://www.mql5.com/en/articles/10955)), such jumps can be explained by the advancing proximity (or equality) of the phases of partial probability waves of the corresponding market instrument.

Theoretically, the approach of a natural price jump can be identified using the ratio (II.17). However, in practice, it is much easier to detect an approaching jump using the predicted quality factor graph. In particular, if the predicted quality factor at some future moment ![](https://c.mql5.com/2/56/Mb.png)(distant from the current bar by ![](https://c.mql5.com/2/56/Mb__1.png)bars) exceeds or approaches a certain critical value for a given market instrument on the corresponding timeframe ![](https://c.mql5.com/2/56/Qcr.png), i.e. ![](https://c.mql5.com/2/56/1__6.png)(if we consider the current and not the forecast situation, then simply ![](https://c.mql5.com/2/56/2__6.png)), then at this moment a change in the global trend is possible.

In general, natural price jumps (of any nature, both global and small) at the level of probability amplitudes of its distribution are described by antisymmetric wavelets ( [Part 1](https://www.mql5.com/en/articles/10955), ratio (I.17)), when, after realizing the proximity and even equality of the phases of all partial price waves, from which its total probability amplitude is composed, the phase of the latter, due to the antisymmetry of the corresponding wavelets, is inverted, which leads to a sharp change in the actual ![](https://c.mql5.com/2/56/p_r__4.png)and ![](https://c.mql5.com/2/56/q_r__3.png)probabilities, which will then be fundamentally different from the forecast probabilities ![](https://c.mql5.com/2/56/p_pred__3.png)and ![](https://c.mql5.com/2/56/q_pred__3.png). It is clear that such critical situations, when jumps in quality occur due to the market’s own laws, should be excluded from trading. Such antisymmetry of partial price waves ensures their similarity to fermions, which determines the constant desire to change price levels and their significant width (which is interpreted as a result of price fluctuations). Therefore, it is more correct to describe the evolution of market instruments not by ordinary statistics, but by Fermi-Dirac statistics.

Due to the described effect of price wave inversion ( [Part 1](https://www.mql5.com/en/articles/10955)) at the most intense phase of its movement (the maximum modulus of the amplitude of its probability), it is also funny to note that, it would seem, the most optimal trading parameters based on the highest ![](https://c.mql5.com/2/56/Qcr__1.png)quality factor should also ensure maximum profit. However, these trading parameters identified by most traders (intuitively or mathematically) on the basis of habitual ideas (characteristic of everything observed in the physical macrocosm) about the monotony of processes and their inertia, in fact, cause maximum losses.

As a result, the everyday "law" turns out to be true here too: money begets money, the bulk of which is stored in banks, so the market pulls money out of small traders. After all, the market, for the reasons mentioned above, suddenly (which it does regularly) begins to develop contrary to the trends predicted by the majority of traders playing against them. This process has no human "evil" intent. The market does not have constant inertia (used by small traders to get profit for some period of time) characteristic of physical macro processes and, at certain moments (unpredictable for experts and traders not armed with the appropriate theory), it easily reverses the phases of all individual partial price waves emitted by different market participants under the influence of its inner laws operating at its emergent level (surpassing the influence of even strong external events).

Overall, the market is ruled by chaos. That's why, if any market order is identified to the maximum extent, then it must be violated, which can be considered a fairly well-observed law of the market. It is impossible to obtain a stable profit without knowing it.

### Conclusion

The article presents my engineering approach to creating a profitable trading strategy. This approach shows that the market leaves traders an extremely narrow set of conditions for opening and closing positions, which could provide them with a game with a positive expected profit. This set is not identifiable by classical methods. But a game with a positive profit expectation is still possible, which to some extent is confirmed by my personal use of the [ScientificTrade](https://www.mql5.com/en/market/product/98467) indicator based on this engineering approach and the use of Fourier extrapolation for prediction (although the statistics is far from sufficient so far). Of course, this indicator still needs to be improved. At the moment, its main drawback is the use of an insufficiently accurate mathematical forecasting apparatus.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12891](https://www.mql5.com/ru/articles/12891)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [The price movement model and its main provisions (Part 2): Probabilistic price field evolution equation and the occurrence of the observed random walk](https://www.mql5.com/en/articles/11158)
- [The price movement model and its main provisions (Part 1): The simplest model version and its applications](https://www.mql5.com/en/articles/10955)

**[Go to discussion](https://www.mql5.com/en/forum/457451)**

![Brute force approach to patterns search (Part V): Fresh angle](https://c.mql5.com/2/57/Avatar_The_Bruteforce_Approach_Part_5.png)[Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)

In this article, I will show a completely different approach to algorithmic trading I ended up with after quite a long time. Of course, all this has to do with my brute force program, which has undergone a number of changes that allow it to solve several problems simultaneously. Nevertheless, the article has turned out to be more general and as simple as possible, which is why it is also suitable for those who know nothing about brute force.

![Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://c.mql5.com/2/60/Data_Science_and_Machine_LearningdPart_15g__Logo.png)[Data Science and Machine Learning (Part 15): SVM, A Must-Have Tool in Every Trader's Toolbox](https://www.mql5.com/en/articles/13395)

Discover the indispensable role of Support Vector Machines (SVM) in shaping the future of trading. This comprehensive guide explores how SVM can elevate your trading strategies, enhance decision-making, and unlock new opportunities in the financial markets. Dive into the world of SVM with real-world applications, step-by-step tutorials, and expert insights. Equip yourself with the essential tool that can help you navigate the complexities of modern trading. Elevate your trading game with SVM—a must-have for every trader's toolbox.

![The case for using Hospital-Performance Data with Perceptrons, this Q4, in weighing SPDR XLV's next Performance](https://c.mql5.com/2/60/Insurance_Claims_Data_with_Perceptrons__Logo.png)[The case for using Hospital-Performance Data with Perceptrons, this Q4, in weighing SPDR XLV's next Performance](https://www.mql5.com/en/articles/13715)

XLV is SPDR healthcare ETF and in an age where it is common to be bombarded by a wide array of traditional news items plus social media feeds, it can be pressing to select a data set for use with a model. We try to tackle this problem for this ETF by sizing up some of its critical data sets in MQL5.

![Developing a Replay System — Market simulation (Part 12): Birth of the SIMULATOR (II)](https://c.mql5.com/2/54/replay-p12-avatar.png)[Developing a Replay System — Market simulation (Part 12): Birth of the SIMULATOR (II)](https://www.mql5.com/en/articles/10987)

Developing a simulator can be much more interesting than it seems. Today we'll take a few more steps in this direction because things are getting more interesting.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/12891&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082907874620870960)

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