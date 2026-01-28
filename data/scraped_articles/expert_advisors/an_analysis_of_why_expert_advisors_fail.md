---
title: An Analysis of Why Expert Advisors Fail
url: https://www.mql5.com/en/articles/3299
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:27:33.769510
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/3299&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071847341696167828)

MetaTrader 5 / Expert Advisors


### Introduction

Expert advisors (EAs) may perform well over a period of months and even a few years, but there are always periods of time where performance is poor. EAs rarely show a steady profit over an extended period of time, such as ten years. If an EA has past periods of poor performance, then how can it be expected to always perform well in the future?

In this analysis, two questions arise: What changes occur in the time series data that strongly affect the performance of the EA? Is there a technical indicator that can predict when an EA will have poor performance and when it will have good performance? Such an indicator would be most valuable!

### Moving Average Crossover Triggers

To study these issues, 15 minute currency data over a period of sixteen years is used along with a test-bed strategy, the Moving Average Crossover. For the Moving Average Crossover (MAC) trigger, a buy signal is generated when the fast Simple Moving Average (SMA) crosses up over the slow SMA and a sell signal is generated when the fast SMA crosses down over the slow SMA. The MAC strategy should provide a good test-bed for study. Key input parameters for the MAC strategy are the fast and slow SMA periods. The fast SMA period is typically 1 to 4 in these studies and the slow SMA period is 10 to 150 and both are assumed to be fixed during the test period.

Figure 1 shows the profit return behavior of the SMA Crossover trigger for the EURUSD currency pair from January 2006 through December 2021. For the optimized SMA fast and slow periods of 2 and 80, the EA has a positive return for some intervals and negative return for other intervals. The period of 1/2008 – 6/2013 has a steady profit growth. Figure 2 shows the profit return behavior of the SMA Crossover trigger for the GBPUSD currency pair for the same interval for the optimized SMA periods of 1 and 80. The GBPUSD chart shows only brief periods of positive profit performance.

![Fig_1_EURUSD_SMACross](https://c.mql5.com/2/45/Fig_1_Tester_EURUSDM15_SMACross.png)

Figure 1.  EURUSD SMA Cross Trigger Profit (0.1 Lot)

![Fig_2_GBPUSD_SMACross](https://c.mql5.com/2/45/Fig_2_Tester_GBPUSDM15_SMACross.png)

Figure 2. GBPUSD SMA Cross Trigger Profit (0.1 Lot)

Given the time behaviors shown in Figures 1 and 2, we can examine the time behavior of common technical indicators to determine if there is any relationship between technical indicator time behavior and moving average crossover profit return time behavior.

### Behavior of Traditional Technical Indicators

Several standard technical indicators can be examined to see if their behavior is different in time regions of positive and negative return. Figure 3 shows the monthly Average True Range (ATR) indicator over a sixteen year period for EURUSD. The ATR indicator provides a measure of the currency pair volatility. It exhibits definite time dependent behavior but nothing corresponding to the profit time behavior in Figure 1. Interestingly, Figure 3 shows the extreme volatility around the economic crisis of late 2008 and again in early 2020. It also suggests that using absolute thresholds of price change for triggering logic is unwise and that the smoothed ATR value should be considered. Another common technical indicator is the spread of the upper and lower Bollinger Bands. This indicator should be influenced by both volatility and trend strength. Surprisingly, its long term monthly behavior, as shown in Figure 4, is nearly identical to the ATR indicator. Like the ATR indicator, it has no behavior corresponding to positive and negative return time regions in the EURUSD data of Figure 1. Similar results are seen for the GBPUSD data. Long-term volatility does not account for the time behavior seen in figures 1 and 2.

![Fig_3_ATRvsMth](https://c.mql5.com/2/45/Fig_3_EURUSD_ATRvsMth_.png)

Figure 3. EURUSD ATR, Monthly Average

![FIg_4_BBSpreadVsMth](https://c.mql5.com/2/45/Fig_4_EURUSD_BBSprdVsTime_.png)

Figure 4. EURUSD Bollinger Band Spread, Monthly Average

### Autocorrelation Function as an Indicator

Unlike the ATR and Bollinger Bands indicators, the AutoCorrelation Function (ACF) does not have a dependence on volatility. The ACF is a helpful tool to find patterns in time series data. The ACF measures the correlation of elements in a time series with elements in this time series delayed by a lag time. When a trend is present in the data, the ACF, for small lags, will have positive values.In this analysis, the Yi element in the time series is defined as the (Close Price - Open Price) of bar i.

For a time series of N elements Yi , i =1...N, the ACF is defined as:

![ACF_EquationV2](https://c.mql5.com/2/45/ACF_EquationV2.png)

The autocorrelation function for a series of price data is much different than the autocorrelation of a series of bar to bar price change data. For instance, for a series of numbers {1,2,3..14} the ACF is 0.786. For a series of differences between consecutive elements of that array, the ACF will be close to zero (assuming a little noise in the values to avoid divide by zero).

Computation of the ACF is implemented in the following code snippet:

```
 double GetAtCorrVal(double &ClsOpn[],int CorrPer, int LagPer,int joff ) {
   double corr;
   double AIn[],BIn[];
   double XMean,XNum,XDen;
   int jj;
   ArrayResize(AIn,CorrPer);
   ArrayResize(BIn,CorrPer);
   XMean = 0.;
   XNum = 0.;
   XDen = 0.;
   corr = 0.;
   if(CorrPer<2)
    {
     Print("No AutoCorr Processing Allowed ");
     return(corr);
    }
   // mean
   for(jj=0;jj<CorrPer;jj++)
    {
     XMean +=ClsOpn[jj+joff];
    }
   XMean = XMean/CorrPer;
  // variances
   for(jj=0;jj<CorrPer;jj++)
    {
     if(jj<(CorrPer-LagPer))
       XNum  += (ClsOpn[jj+joff]-XMean)*(ClsOpn[jj+LagPer+joff]-XMean);
     XDen += (ClsOpn[jj+joff]-XMean)*(ClsOpn[jj+joff]-XMean);
    }
    if(XDen==0.)
     {
      corr = 0.;
     }
    else
    corr = XNum/XDen;
   return(corr);
  }
//----------------------------------------------------------------
```

Figure 5 shows the monthly average of the ACF over the sixteen year period for EURUSD data with a smoothing period of six months. Figure 1 shows a profitable performance between 2008 and the end of 2012, while Figure 5 also shows a slightly higher ACF value in this time region, suggesting possible relationship between profit performance of the MAC strategy and the value of the ACF.

![Fig_5_EURUSD_ACFvsMth](https://c.mql5.com/2/45/Fig_5_EURUSD_ACFvsMth_.png)

**Figure 5. EURUSD AutoCorrelation Function Monthly Average**

For the GBUUSD data shown in Figure 6, a weak linkage between profit performance (shown in Figure 2) and the ACF value can be seen.

![Fig_6_GBPUSD_ACFvsMth](https://c.mql5.com/2/45/Fig_6_GBPUSD_ACFvsMth_.png)

Figure 6. GBPUSD AutoCorrelation Function Monthly Average

The AutoCorrelation Function can be used as an indicator (AutoCorr.mq5), but as seen from Figure 7, it is not a good trend identification indicator.

![Fig_7_EURUSD_ACFIndicator](https://c.mql5.com/2/45/FIg_7_EURUSDH1__2.png)

Figure 7. ACF Indicator, Lag=1

### \#\#\# ACF Time Behavior and the SMA Crossover Strategy

More information can be found by looking at the time behavior of the ACF just prior to the SMA crossover point (Trigger ACF) and also, just after the crossover point, while the trade is still open (Trade ACF). The two types of ACFs, Trade ACF and Trigger ACF have distinct properties. Figures 8 and 9 show the Trade ACF for the EURUSD and GBPUSD data. Using the Strategy Tester, the ACF value can be plotted separately for profit trades and for loss trades as a function of time. Both charts show a time varying separation of the ACF between profit and loss trades. The profitable region 1/2008-6/2013 in Figure 1 (EURUSD) roughly matches a region in Figure 8 where the separation of the ACF is also large. Starting at 1/2018, there is little separation between ACF values for profit and loss trades. In Figure 1, this corresponds to a flat region in profit returns.

![Fig_8_EURUSD_TradeACF](https://c.mql5.com/2/45/Fig_8_TradeACF_EURUSDM15-A___2.png)

**Figure 8.  Trade AutoCorrelation for EURUSD for SMA Cross Triggers**

![Fig_9_GBPUSD_TradeACF](https://c.mql5.com/2/45/Fig_9_TradeACF_GBPUSDM14-A__2.png)

Figure 9. Trade AutoCorrelation for GBPUSD for SMA Cross Triggers

Figures 10 and 11 show the Trigger ACF for the EURUSD and GBPUSD data. The purpose of these two charts is to investigate whether the Trigger ACF value can be used as a filter in an expert advisor to improve the profitability of the SMA Crossover strategy. Compared to the Trade ACF data, the trigger ACF data shows less separation between positive and negative profit returns. This weakens its value as a trigger filter. However, the ACF value is largely flat in time indicating that a fixed threshold for the ACF could be used as a trigger filter. The profit/loss curves are largely bouncing across each other which means there are regions in time that trades would not be profitable.

![Fig_10_EURUSD_TriggerCF](https://c.mql5.com/2/45/Fig_10_TriggerACF_EURUSDM15-A___2.png)

**Figure 10.  Trigger AutoCorrelation for EURUSD for SMA Cross Triggers**

![Fig_11_GBPUSD_TriggerACF](https://c.mql5.com/2/45/Fig_11_TriggerACF_GBPUSDM15-A__2.png)

Figure 11. Trigger AutoCorrelation for GBPUSD for SMA Cross Triggers

### ACF Filter for the SMA Crossover Trigger

Despite the small ACF differences between trades with positive and negtive returns (Figures 10 and 11), if an autocorrelation threshold requirement is added to the SMA Crossover trigger, the return performance of the expert advisor is much improved in regions of poor profit performance. Figures 12 and 13 show the profit return of the EA with an additional ACF filter for EURUSD and GBPUSD data. The full sixteen year duration is used. Comparing figures 1 and 2 with figures 12 and 13, the return performance for both the EURUSD and GBPUSD data is greatly improved when a autocorrelation threshold requirement is applied.

![Fig_12_EURUSD_SMACrossCorr](https://c.mql5.com/2/45/Fig_12_EURUSD_SMACrossCorr.png)

Figure 12. EURUSD Crossover Trigger With ACF Filter

![Fig_13_GBPUSD_SMACrossCorr](https://c.mql5.com/2/45/FIg_13_GBPUSD_SMACross-Corr.png)

Figure 13. GBPUSD Crossover Trigger With ACF Filter

Figure 14 shows a table of performance results from the Strategy Tester. A significant improvement is seen for both EURUSD and GBPUSD data. In both cases, the total profit increases by approximately 50% when the ACF filter is enabled and the profit per trade (PO) is also greatly improved.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Trigger Type** | **\# Trades** | **Profit $**<br>**(0.1 Lot)** | **Profit/Loss (PF)** | **Prof/Trade (PO)** |
| EURUSD SMA Cross | 3160 | 5830\. | 1.10 | 1.85 |
| EURUSD SMA Cross + ACF | 1548 | 8511\. | 1.33 | 5.50 |
| GBPUSD SMA Cross | 3672 | 5352\. | 1.07 | 1.46 |
| GBPUSD SMA Cross + ACF | 3096 | 8317\. | 1.13 | 2.69 |

Figure 14. Test Performance Table for EURUSD and GBPUSD Data

### Dependence of ACF on SMA Slow Period for SMA Crossover Strategy

It is also of interest to examine the dependence of the average autocorrelation function on choice of SMA slow period for winning and losing trades. This is shown for EURUSD M15 data in Figure 15. No correlation threshold is used as part of the trade opening strategy. Without regard to the ACF value, the optimal slow period for the SMA Crossover strategy was determined to be 80. Figure 15 shows that the largest separation of average ACF value, between winning and losing trades, occurs for SMA slow periods between 65 and 80. Further analysis may lead to using the ACF information to determine the optimal slow period based on measured autocorrelation value at the time of trade opening.

### ![Fig_15_AvgACVvsSlowPeriod](https://c.mql5.com/2/45/Fig_15_AvgCorrVsSMAPer_46_1.png)

Figure 15. Average ACF versus SMA Slow Period for EURUSD

### Conclusion

The autocorrelation function is a valuable indicator for improving expert advisor performance. It improves the performance of the test-bed trigger by filtering out regions where performance is poor.

In general, correlation coefficients, including the autocorrelation function, provide a fruitful area of research for improving trading efficiency. They are immune to volatility effects and are sensitive to price patterns including trend formation and reversal.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3299.zip "Download all attachments in the single ZIP archive")

[AutoCorr.mq5](https://www.mql5.com/en/articles/download/3299/autocorr.mq5 "Download AutoCorr.mq5")(4.12 KB)

[SMACrossAutoCorr\_EURUDM15.mq5](https://www.mql5.com/en/articles/download/3299/smacrossautocorr_eurudm15.mq5 "Download SMACrossAutoCorr_EURUDM15.mq5")(66.68 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/389383)**
(11)


![Vladimir Gulakov](https://c.mql5.com/avatar/2025/3/67c4a5ad-1b40.png)

**[Vladimir Gulakov](https://www.mql5.com/en/users/fobuysel)**
\|
9 Apr 2022 at 16:29

Traders and programmers follow a parallel course. An attempt to cross from either side is the reason for failure.

If there were a forum here only for professionals of both sides, we could discuss it in detail. But here it is far from that. So I will decipher briefly. Passion for one thing harms another and vice versa. This is a natural process...

![Roman Shiredchenko](https://c.mql5.com/avatar/2012/3/4F76634B-9044.jpg)

**[Roman Shiredchenko](https://www.mql5.com/en/users/r0man)**
\|
10 Apr 2022 at 07:32

Yes.... By the way, the topic is interesting, I'll take it under consideration.

![Cenk](https://c.mql5.com/avatar/2022/6/62A472E7-9C25.png)

**[Cenk](https://www.mql5.com/en/users/konuskancilek1)**
\|
6 May 2022 at 05:40

Thank you for the article.

I have a question;

Is the "ACF" filter approach with the "AutoCorr" indicator only valid for EURUSD and [GBPUSD pairs](https://www.mql5.com/en/quotes/currencies/gbpusd "GBPUSD chart: techical analysis")?

For example, can we say that we may not show the same performance in pairs such as other major parities, USDJPY, USDCHF, AUDUSD etc.?

Best.

![Richard Poster](https://c.mql5.com/avatar/avatar_na2.png)

**[Richard Poster](https://www.mql5.com/en/users/raposter)**
\|
8 May 2022 at 23:22

Cenk

My experience with [autocorrelation](https://www.mql5.com/en/articles/5451 "Article: An Econometric Approach to Finding Market Patterns: Autocorrelation, Heatmaps, and Scatterplots ") functions is that they work well with most currency pairs.

![John Winsome Munar](https://c.mql5.com/avatar/2022/6/62A73E8B-F64C.jpg)

**[John Winsome Munar](https://www.mql5.com/en/users/trozovka)**
\|
21 May 2022 at 14:43

Good article


![Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__4.png)[Graphics in DoEasy library (Part 92): Standard graphical object memory class. Object property change history](https://www.mql5.com/en/articles/10237)

In the article, I will create the class of the standard graphical object memory allowing the object to save its states when its properties are modified. In turn, this allows retracting to the previous graphical object states.

![Learn how to design a trading system by Bollinger Bands](https://c.mql5.com/2/45/why-and-how__2.png)[Learn how to design a trading system by Bollinger Bands](https://www.mql5.com/en/articles/3039)

In this article, we will learn about Bollinger Bands which is one of the most popular indicators in the trading world. We will consider technical analysis and see how to design an algorithmic trading system based on the Bollinger Bands indicator.

![Developing a trading Expert Advisor from scratch](https://c.mql5.com/2/44/Robozinho.png)[Developing a trading Expert Advisor from scratch](https://www.mql5.com/en/articles/10085)

In this article, we will discuss how to develop a trading robot with minimum programming. Of course, MetaTrader 5 provides a high level of control over trading positions. However, using only the manual ability to place orders can be quite difficult and risky for less experienced users.

![Graphics in DoEasy library (Part 91): Standard graphical object events. Object name change history](https://c.mql5.com/2/44/MQL5-avatar-doeasy-library3-2__3.png)[Graphics in DoEasy library (Part 91): Standard graphical object events. Object name change history](https://www.mql5.com/en/articles/10184)

In this article, I will refine the basic functionality for providing control over graphical object events from a library-based program. I will start from implementing the functionality for storing the graphical object change history using the "Object name" property as an example.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/3299&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071847341696167828)

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