---
title: Angles in Trading. Further Study Required
url: https://www.mql5.com/en/articles/3237
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-24T13:37:04.887376
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/3237&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082992700224967375)

MetaTrader 4 / Examples


### Introduction. Angles in Geometry and in Trading

In geometry, an angle is the figure formed by two rays (the sides of the angle), sharing a common endpoint (the vertex of the angle).

![](https://c.mql5.com/2/29/1__1.png)

The concept of the angle is slightly modified in trading. An angle is the ratio of the number of bars to the number of points. Bars here mean the time offered by standard (M1, M5, etc.) and non-standard timeframes. Points represent the unit of price measurement with an accuracy of 4 or 5 decimal places.

![](https://c.mql5.com/2/29/2__1.png)

The following angle calculation and construction options are provided in the MetaTrader 4 terminal.

1) A simple angle (Trend Line By Angle).

![](https://c.mql5.com/2/29/3__1.png)

2) Gann Fan Angles.

![Gann Fan Angle.](https://c.mql5.com/2/27/y2c3_10lyu_yi0p1.png)

3) Gann Grid.

![Gann Grid.](https://c.mql5.com/2/27/8blp7_l2x8y.png)

The above examples available in the terminal are based on popular trading methods using angles described by Gann.

There are some advanced trading developments that are based on the angle analysis. One of the positive innovative approaches to technical analysis using angles is the indicator [SinFractal](https://www.mql5.com/en/market/product/14320#full_description) by A.Praslov. The below figure describes the essence of trend analysis using angles.

![SinFractal](https://c.mql5.com/2/27/1ed0d5863c.png)

Based on SinFractal calculation, the author managed to create a series of angle indicators for the simultaneous analysis of 28 currency pairs on multiple timeframes. This method enables successful trading using a basket strategy.

A question may arise here: Why are ready angle solutions offered by Gann not enough? For me personally, the answer to this question lies in the fact that Gann continued to study angles, trying to improve the angular trend analysis technique.

In my opinion, a clear theory of how to use angles in trading has not yet been developed. In addition to traditional angles (including those used in SinFractal), I suggest adding new types to improve the quality of the analysis: **vertical and turning** angles.

### Light Beam Physics and Trend Movement: Optical Analogies in Technical Analysis

The author of Forks and the median line Andrews suggested that financial markets are subject to the laws of physics. Andrews used to draw a normal cycle and to correlate cycle trend movements with its oscillation phases.

![](https://c.mql5.com/2/29/4__1.png)

Andrews' idea that the laws of physics can help in understanding what is happening with the trend suggests that the trend looks like a beam of light that falls from the sellers' environment to the customers' environment. This phenomenon can be correlated with the optical laws of reflection (the angles of an incidence ray and a reflection ray). In optics we rely on the angles formed by the falling angle and the OY, while all the above angles in trading are formed by a trend line and the axis OX axis.

![](https://c.mql5.com/2/29/5.png)

Thus, following the laws of optics, we get a new type of angle on our charts, i.e. a **vertical** angle (which is an angle between the trend and the OY axis).

And, finally a **turning angle** which is well known for all traders. Consider this angle from the point of view of optics.

The laws of optics allow considering trend similarly to a ray of light. Accordingly, we will try to extrapolate the physics of the light beam propagation to the behavior of a trend. A light ray is reflected from the surface, and trend direction is changed to the opposite, i.e. reverses. If we use the laws of light dispersion (decomposition) of light, we can observe a number of interesting analogies.

- In optics, dispersion is the decomposition of white light into a spectrum consisting of monochromatic colors. In technical analysis, dispersion is the decomposition of the whole trend into a series of separate trends formed on smaller timeframes.

- In optics, the media refractive index depends on its frequency (color), and in technical analysis this means that the turning angle for each of the minor trends within the major one will be different, depending on the timeframe.

These trend properties borrowed from optics, **allow analyzing the trend as a wave**. This analysis can be related to the Elliott wave theory, which is also very popular in technical analysis.

### Measuring Angles on Price Charts

To measure angles by three points, we use the **Angles\_3\_points angle indicator**, which calculates the angle based on the point-to-minute ratio.

The indicator calculates the angles using inverse trigonometric functions: the tangent of the angle is calculated first, using the ratio of the opposite leg to the adjacent one, whose lengths are known. Then arctangent is calculated based on it, i.e. the angle size in degrees. This value and the difference between extremes in pips and values are displayed on the price chart.

**Important note:** **the value of the actual calculation angle can visually differ from the angle shown on the chart.**

The actual calculated angle is a certain **precise angle value**. The original algorithm of precise angle calculation is provided in the indicator source code.

The visual angle shown on the chart is converted by the terminal for the convenience of perception **of the entire price chart** with a specific time and price scale.

Example of measuring the turning angle on the M1 timeframe.

![Turning angle on М1.](https://c.mql5.com/2/27/kjno_9xst2o74v_6b_s1.png)

The **M1** timeframe provides a very detailed market picture which inevitably contains a huge amount of strong price noise. Filtering this noise on M1 is a non-trivial task, and we don't need it now. Let us remember the value of the lower turning angles on this timeframe: **48 degrees**.

Let's increase the timeframe. The turning angle has changed on **М5** and is now equal to **104 degrees**. That's how it looks like:

![Turning angle on М5.](https://c.mql5.com/2/27/hxew_xkeeuyv5x_ix_z5.png)

The angle has become larger on **М15** and is now equal to **112 degrees**.

![Turning angle on М15.](https://c.mql5.com/2/27/ogge_wdhc9e7wc_6s_p15.png)

For Н1:

![Turning angle on H1.](https://c.mql5.com/2/27/eiym_6na4lbell_fu_61..png)

For Н4:

![Turning angle on H4.](https://c.mql5.com/2/27/h8xl_9ehdswu5j_wv_74.png)

D1:

![Turning angle on D1.](https://c.mql5.com/2/27/oz5h_urn1sfaj5_99_D1..png)

Examples of different turning angles confirming the assumption that the current trend consists of a sum of trends on separate timeframes are similar to Newton's law of the separation of white light into component waves.

The following **conclusions can be made** based on the above.

1\. Individual trend reversal angles exist on each timeframe for each currency pair.

2\. Upper and lower turning angles have individual sizes.

4\. Vertical levels (formed by the trend and the OY axis) can be used for the analysis similar to traditional horizontal angles (formed by the trend and the OY axis).

5\. Since each timeframe contributes to the general trend, there are two methods of trend analysis using angles:

a) analyzing each trend on each timeframe separately

b) analyzing the overall (mean average) trend

However, using pure laws of optics for trend interpretation is not enough. The trend, like any physical process, needs to be considered in detail.

For example, the **effect of the angle of incidence** is still useful for the technical analysis purposes.

![](https://c.mql5.com/2/29/6.png)

Let's try to interpret this figure from a book on optics in terms of trend classification. In figure 2 we see the full reflection angle, when the reflected ray is directed along the OX axis.

**Conclusion**: There are 4 trend types on any timeframe.

1\. Uptrend

2\. Downtrend

3\. Upper flat trend (an uptrend has ended, but the trend has not reversed yet)

4\. Lower flat trend (a downtrend has ended, but the trend has not reversed yet)

The concept of flat can be expanded if analyzed in terms of optics. In addition to the standard trading definition of flat, we can add that it can occur in the upper part of an uptrend (upper flat) or in the lower part of a downtrend (lower flat). When analyzing flat, it is necessary to check the timeframe it occurred on.

Example of flat on D1

![Flat on D1.](https://c.mql5.com/2/27/EURUSD1.png)

Here the flat on D1 is limited by a channel of 1374 points (for 4-digit quotes). One cannot expect a stable trend on this timeframe until the price exits this channel.

From the point of view of the angular analysis, it seems that the trend moves horizontally while gaining and increasing its turning angle, and then trend reverses. But sometimes the trend is influenced by the trend of a higher timeframe. In this case, the reversal is canceled, and the trend continues into the "major" direction as per the general movement of the higher timeframe.

Based on the above idea, we can formulate the basic **angular analysis problems**. These are:

-  Determining the most effective angles for analyzing the trend within one timeframe

-  Identifying angle signals that can be useful for trading
-  Determining trend reversal angles within one timeframe


### What Angles Are More Effective for Technical Analysis?

As already described above, we can use the following types of angles for trend analysis:

-  Horizontal (between the trend and the OX axis);
-  Vertical (between the trend and the OY axis);
-  Turning angles (between two different trend directions).


Let's consider the features of horizontal and vertical angles when used for different timeframes.

The below image shows how angles appear on **М1**:

![](https://c.mql5.com/2/29/7.png)

The most effective angle for M1 is horizontal, while the rest angles are difficult to read or analyze.

Consider the **M5** timeframe.

![](https://c.mql5.com/2/29/8.png)

**М15.**

![](https://c.mql5.com/2/29/9.png)

**Н1.**

![Effective angles for the analysis on H1.](https://c.mql5.com/2/27/kmf356qll4z_6n7_iqitnvu_3nig_71.png)

**Н4.**

![](https://c.mql5.com/2/29/10.png)

**D1.**

![Effective angles for the analysis on D1.](https://c.mql5.com/2/27/sbjp92dokke_5rh_gjbseq4_j2ex_D1.png)

Conclusions about the use of different angles for different timeframes:

-   on lower timeframes (М1, М5), it is visually more convenient to use horizontal angles (which are clearly visible on the chart);
-   on medium timeframes (М15, Н1, Н4), turning, horizontal and vertical angles look good;
-   on higher timeframes (above H4), vertical angles and turning angles increase substantially, while horizontal angles are minimized and become difficult to read;
-   the angle indicator is best suited for timeframes М1 to Н4; horizontal or vertical angles are more convenient on separate timeframes;
-   turning angles are often increased before the new trend direction, but still they are more like angles anticipating trend continuation rather than reversal;

-   real turning angles that would unambiguously mean upcoming trend reversal have not been found yet;

-   also, a useful practical application of a combination of turning, horizontal and vertical angles on one timeframe and the explicit relationship between them have not been found yet;
-   for an effective trend analysis, new angles types should be studied.


### New Angles for Technical Analysis

Let us do the following, in order to quickly receive the final result and the required types of angles:

1) We take Andrew's median line, but it will be built based on a bisector of an angle instead of the median;

2) Now we draw a turning, horizontal and vertical angles for the Andrews' median line.

If the horizontal Andrews' angle (the red line color) is greater than the vertical Andrews' angle, it is a sign of a falling trend.

If the vertical Andrews' angle (the blue line color) is greater than the horizontal Andrews' angle, it is a sign of a rising trend.

![](https://c.mql5.com/2/29/11.png)

They look pretty clear. Let's try to find effective signals for determining the trend in these angles.

The result of detecting signals using new Andrews' angles is as follows:

**Sell (sell close)** signals.

![The sell and sell close signals.](https://c.mql5.com/2/27/kfixxr8_sell_x_sell_close.png)

**Buy (buy close)** signals.

![The Buy and Buy close signals.](https://c.mql5.com/2/27/jdrz4il_buy_g_buy_close.png)

An example for **М1** (5 decimal places).

![An example of signals on М1.](https://c.mql5.com/2/27/2nt9xv_7ncljb26_yn_j1_vlo_5_n3e7hb_ddau7_amp4z6c.png)

**Sell** signals: the upper red peak after an upper read trough.

**Sell close (Buy)** signals: the upper blue trough after an upper peak.

**М5.**

![An example of signals on М5.](https://c.mql5.com/2/27/lawq4e_ipmwrk5p_2l_a5_fnk_5_lzlxit_voghz_kn2w6a0.png)

**М15.**

![An example of signals on М15.](https://c.mql5.com/2/27/p2kmsb_7cqqfw6k_54_j15_mee_5_b4maxo_3wlon_vln5azj.png)

The angle indicator performs well on historic data, while current values are redrawn.

To smooth the redraw effect, Elder method can be used, during which we simultaneously analyze several different timeframes of the same currency pair (three Elder screens).

For our example, we will use two timeframes instead of three (i.e. two angle indicators with a proportional number of bars) and obtain the following results. You can see in the above figure that indicator values on a lower timeframe (thin lines) can be corrected based on the values of a higher timeframe.

**Н1 with two indicators**

![](https://c.mql5.com/2/29/12.png)

Let's consider, what angles allow receiving signals for different timeframes from М1 to Н1.

Description of trading signals:

Preliminary signals:

1\. "Attention, get ready for buy open"P: the upper blue peak;

2\. "Attention, get ready for sell open": the upper red trough;

3\. "Attention, get ready for buy close": the upper red peak;

4\. "Attention, get ready for sell close": the upper blue trough;

Executive signals:

1\. "Buy open": the upper blue trough;

2\. "Sell open": the upper red peak;

3\. "Buy close": the upper red peak;

4\. "Sell close": the upper blue trough;

5\. "Add buy": the second consecutive upper blue trough;

6\. "Add sell": the second consecutive upper red peak.

If we continue to use the indicator on higher timeframes, e.g. **H4**, we obtain the following results:

![An example of signals on H4.](https://c.mql5.com/2/27/ok62ou_xoql0iue_8j_24..png)

As you can see, the accuracy of the angle indicator is lost on H4. The use of angle indicators does not provide a serious practical advantage on H4. Other analysis tools and approaches should be used here.

### Conclusions

We have conducted a small study on the use of angles for technical analysis. We have shown variants of use of various angles.

We still need to study the features of summation of angles of different timeframes. Good results were achieved by A. Prasov. He provides an advanced useful strategy of how to use angles in trading.

The study of new angles continues, and there are more unexplored and unknown points than already known. These indicators are obtained as a result of multiple attempts to find an approach to the measurement of angles.

In the article, I used angle indicators developed to order (I do not program myself, alas):

**The Angles\_3\_points indicator** is designed for manual setting of three points for measuring angles on history (set three points by clicking on the "Points" menu) or the current angle (two points are set from "Points", and the third one is set by clicking "Bid"). A point is set at the required location by a double click, similar to the way it is done in the terminal. Once the point location is fixed, the next one can be set. Points on the chart can be moved. To do this, double-click on the desired point and drag it to a new location, then double-click on it again. The result of measuring the angles is shown in the price chart.

We tried to eliminate the deformation connected with timeframe switching (that is why the values of indicators are more reliable than the standard MT tool "Trendline by angle").

**The ChartAngles indicator** automatically measures angles and displays them in the additional window below the chart. The indicator draws two types of angles: usual (horizontal, vertical, turning) and Andrews' angles (horizontal, vertical and turning). Required levels should be enabled in the settings, while the rest should be disabled. The color of various angle lines can be configured in the indicator settings. The indicator uses the ZigZag indicator. The indicator uses the principle of screen deformation elimination described in Angles\_3\_points. The number of bars is sent in parameters, based on which the indicator plots angles.

**The DegreeAngles\_Zero indicator** automatically measures angles and draws them in the additional window below the chart for two different angles using the different number of bars. Its distinctive feature is that instead of using the Zigzag indicator, it uses the regression line. This makes the indicator values more accurate. Set the number of bars for the two indicators, the color and width of the lines, and also the type of angles. Then the indicator will draw the angles.

I repeat that practically useful signals of these indicators were obtained for Andrews' lines using a bisector. Vertical and horizontal angles based on the bisector, as well as their intersection, help to determine the trend. The turning Andrews' angle using the bisector also seems to be potentially useful for the practical trading.

An Expert Adviser using these indicators is currently being developed.

For a proper operation of the **Angles\_3\_points indicator**, it is necessary to install the attached font.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/3237](https://www.mql5.com/ru/articles/3237)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/3237.zip "Download all attachments in the single ZIP archive")

[ChartAngles\_v3.ex4](https://www.mql5.com/en/articles/download/3237/chartangles_v3.ex4 "Download ChartAngles_v3.ex4")(58.18 KB)

[6ijt0\_GILSANUB.zip](https://www.mql5.com/en/articles/download/3237/6ijt0_gilsanub.zip "Download 6ijt0_GILSANUB.zip")(46.57 KB)

[Angles\_3\_points.mq4](https://www.mql5.com/en/articles/download/3237/angles_3_points.mq4 "Download Angles_3_points.mq4")(100.23 KB)

[DegreeAnglesZero.mq4](https://www.mql5.com/en/articles/download/3237/degreeangleszero.mq4 "Download DegreeAnglesZero.mq4")(47.65 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/213865)**
(4)


![Greco608 No Fear](https://c.mql5.com/avatar/2016/12/585926CC-FEBA.png)

**[Greco608 No Fear](https://www.mql5.com/en/users/greco608)**
\|
21 Aug 2017 at 00:58

nice fictional narrative. thumbs up.


![Domenico De Sandro](https://c.mql5.com/avatar/2017/12/5A350857-07C3.jpg)

**[Domenico De Sandro](https://www.mql5.com/en/users/mimmods)**
\|
12 Dec 2017 at 16:10

Hello I like your idea, but I cannot set the ChartAngle indicator becouse I receive this window setting:

![Carsten Oliver Nielsen](https://c.mql5.com/avatar/avatar_na2.png)

**[Carsten Oliver Nielsen](https://www.mql5.com/en/users/carsten103)**
\|
7 Apr 2018 at 17:33

Where do I install the font file ? Libraries ?


![mj milly](https://c.mql5.com/avatar/avatar_na2.png)

**[mj milly](https://www.mql5.com/en/users/mjmilly2-gmail)**
\|
6 Jul 2022 at 08:32

**Carsten Oliver Nielsen [#](https://www.mql5.com/en/forum/213865#comment_7031730):**

Where do I install the font file ? Libraries ?

Where do install it


![Graphical Interfaces XI: Rendered controls (build 14.2)](https://c.mql5.com/2/28/av.png)[Graphical Interfaces XI: Rendered controls (build 14.2)](https://www.mql5.com/en/articles/3366)

In the new version of the library, all controls will be drawn on separate graphical objects of the OBJ\_BITMAP\_LABEL type. We will also continue to describe the optimization of code: changes in the core classes of the library will be discussed.

![Patterns available when trading currency baskets. Part III](https://c.mql5.com/2/28/articles_234.png)[Patterns available when trading currency baskets. Part III](https://www.mql5.com/en/articles/3266)

This is the final article devoted to the patterns that occur when trading currency pair baskets. It considers combined trend-following indicators and application of standard graphical constructions.

![Testing patterns that arise when trading currency pair baskets. Part I](https://c.mql5.com/2/28/articles_234__1.png)[Testing patterns that arise when trading currency pair baskets. Part I](https://www.mql5.com/en/articles/3339)

We begin testing the patterns and trying the methods described in the articles about trading currency pair baskets. Let's see how oversold/overbought level breakthrough patterns are applied in practice.

![Graphical Interfaces XI: Refactoring the Library code (build 14.1)](https://c.mql5.com/2/28/MQL5-avatar-XI-build14.png)[Graphical Interfaces XI: Refactoring the Library code (build 14.1)](https://www.mql5.com/en/articles/3365)

As the library grows, its code must be optimized again in order to reduce its size. The version of the library described in this article has become even more object-oriented. This made the code easier to learn. A detailed description of the latest changes will allow the readers to develop the library independently based on their own needs.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/3237&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082992700224967375)

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