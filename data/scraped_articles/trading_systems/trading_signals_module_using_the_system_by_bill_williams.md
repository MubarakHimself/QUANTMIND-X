---
title: Trading signals module using the system by Bill Williams
url: https://www.mql5.com/en/articles/2049
categories: Trading Systems, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:41:36.251627
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=xnemkkfbyqcgnetxqnajyncjpujtraht&ssn=1769193695510457593&ssn_dr=0&ssn_sr=0&fv_date=1769193695&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F2049&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20signals%20module%20using%20the%20system%20by%20Bill%20Williams%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919369508249513&fz_uniq=5072023650103669470&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

The trading system by Bill Williams described in his book called " [New trading dimensions](https://www.mql5.com/go?link=https://www.amazon.com/New-Trading-Dimensions-Profit-Commodities/dp/0471295418 "http://www.amazon.com/New-Trading-Dimensions-Profit-Commodities/dp/0471295418")" is certainly something that any trader is familiar with. This is one of the systems that contains clear and understandable rules for a majority of beginners. But the simplicity of rules is only apparent — the trading system comprises more than a dozen of trading patterns.

Many have attempted to create an Expert Advisor themselves based on this system, but pattern formalization, correct search and interpretation frequently prove difficult. In order to automate trading as well as identify and mark the system patterns, I have developed a module of trading signals for creating robots in [MQL5 Wizard](https://www.mql5.com/en/articles/171).

I aimed to create maximum convenience for those potential users of the MetaTrader 5 terminal, who may wish to study the trading system independently. The difference of the suggested trading module from other [60 published modules for MQL5 Wizard](https://www.mql5.com/en/search#!module=mql5_module_codebase&keyword=MQL5%20Wizard) is that it contains configuration options with a visual interface.

So, these are the main features of the trading module:

1. Adjusting settings of the trading system with a graphic panel.
2. Ability to disable identification and marking of selected patterns.
3. Ability to disable trading with selected patterns.
4. Ability to optimize parameters of the trading system.

Structure (the source code is contained in the _billwilliamsts.zip_ archive attached to this article):

1. MQL5 _CBillWilliamsTS_ class. Contains all logic of identifying system's trading patterns and logic of trading with found patterns. Marking found patterns on the trading instrument chart can be executed using the class (optional). The class is contained in the _BillWilliamsTS.mqh_ file.
2. MQL5 _CBillWilliamsDialog_ graphic panel class. It is intended to display the settings panel for interactive management of the _CBillWilliamsTS_ class object. The class is contained in the _BillWilliamsPanel.mqh_ file.
3. MQL5 _SignalBillWilliams class._ The trading signal module used in MQL5 Wizard for automated creation of an Expert Advisor.
4. MQL5 _BillWilliamsEA_ Expert Advisor.A trading expert developed on the basis of trading classes and the graphic panel. It is intended for automated trading with patterns of the trading system by Bill Williams and is contained in the _BillWilliamsEA.mq5_ file.

The materials are provided in the following order:

1. Brief description of the trading strategy by Bill Williams, trading patterns used, and marking performed by the developed Expert Advisor.
2. Description of the graphic panel.
3. Results of testing on various trading instruments.

### 1\. Brief overview of the trading system by Bill Williams

### 1.1. General information

In his book "New trading dimensions", Bill Williams claims that it is necessary to know the market structure to achieve profitable trading results on financial markets. From the author's point of view, the market has five dimensions, that, if studied cumulatively, can help you get the true picture and take up to 80% of the trend movement from the market:

1. The fractal (phase space)
2. Momentum (phase energy)
3. Acceleration/deceleration (phase force)
4. Zone (phase energy/force combination)
5. Balance line


In addition to five dimensions, Bill Williams introduces well-known market conditions — trend and flat. To identify them, the system's author suggests using the Alligator indicator, that he developed, and working only at the trending market areas.

Elements of the trading system, trading patterns based on them, and peculiarities of marking found signals, using the developed module of trading signals, are considered further in the article.

### 1.2. Alligator

The Alligator indicator is a combination of three moving averages (Figure 1):

- Jaws, slow line (blue), normally is a 13-period moving average;
- Teeth, average line (red), normally is an 8-period moving average;
- Lips, fast line (green), normally is a 5-period moving average;

![Fig. 1. Alligator](https://c.mql5.com/2/22/alligator.png)

Fig. 1. Alligator

According to the system, trading operations must be executed only when the Alligator lines are arranged towards the trend in a descending order of their period value: price, lips, teeth, jaws. The figure shows the beginning and the end of the downtrend.

It's obvious that the Alligator is a severely delayed indicator, the same with all other indicators on the basis of moving averages. However, the intersection of moving averages doesn't act as a signal for a market entry, but only filters executed trades.

### 1.3. Fractals - signals of the first market dimension

Fractal is a formation consisting of 5 candles. Sell fractal is a fractal where the Low price of the average candle is minimum. Buy fractal, on the other hand, is a fractal where the High price of the average candle is maximum. Fractals are also called the first market dimension (dimension 1):

![Figure 2. Fractals](https://c.mql5.com/2/22/fractals.png)

Figure 2. Fractals

A fractal is considered valid if it is formed above the Alligator's average line (teeth) for an uptrend, and below the average line for a downtrend.

The developed Expert Advisor marks valid fractals in the following way (FrB — FractalBuy — valid buy fractals, FrS — FractalSell — valid sell fractals):

![Figure 3. Operating buy fractal](https://c.mql5.com/2/20/figure3_-_fractalbuy.jpg.png)

Figure 3. Valid buy fractal

![Figure 4. Operating sell fractal](https://c.mql5.com/2/20/figure4_-_fractalsell.jpg.png)

Figure 4. Valid sell fractal

A Buy Stop pending order is positioned 1 pip higher than the bar's maximum, where the valid buy fractal is formed. A Sell Stop pending order is positioned 1 pip lower than the bar's minimum, where the valid sell fractal is formed. Additional positions (by fractals and other indicators) are opened only after overcoming the first fractal when the Alligator changes the trend.

### 1.4. Awesome Oscillator — AO — signals of the second market dimension

#### 1.4.1. General information

The awesome Oscillator (AO) determines the momentum of the market. It is the difference between the 34-period SMA and the 5-period SMA, that are calculated with central values of bars. On the chart, the indicator is presented as a histogram:

![Figure 5. Awesome oscillator](https://c.mql5.com/2/20/ao.png)

Figure 5. Awesome Oscillator

There are 6 patterns based on the oscillator in the trading strategy. Their descriptions and marking via the Expert Advisor are presented hereinbelow.

#### 1.4.2. "Saucer" buy pattern

![Figure 6. "Saucer" buy pattern](https://c.mql5.com/2/20/figure6_-_ao_dish_buy.jpg.png)

Figure 6. "Saucer" buy pattern

The pattern consists of three columns. The first column must be higher than the middle column and can be of any color. The middle column must be red. The third column (signal) must be green. The signal is displayed by the Expert Advisor on the AO indicator and is referred to as DiB (Dish Buy).

#### 1.4.3. "Saucer" sell pattern

![Figure 7. "Saucer" sell pattern](https://c.mql5.com/2/20/figure6_-_ao_dish_sell.jpg.png)

Figure 7. "Saucer" sell pattern

The pattern consists of three columns. The first column must be lower than the middle column and can be of any color. The middle column must be green. The third column (signal) must be red. The signal is displayed on the AO indicator by the Expert Advisor and is referred to as DiS (Dish Sell).

#### 1.4.4. "Zero line crossing" buy pattern

![Figure 8. "Crossing the zero line" buy pattern](https://c.mql5.com/2/20/figure8_-_ao_cross_buy.jpg.png)

Figure 8. "Zero line crossing" buy pattern

The signal appears when the histogram crosses the zero line above. The column that crosses the zero line is the signal column. The signal is displayed on the AO indicator by the Expert Advisor and is referred to as CrB (Cross Buy).

#### 1.4.5. "Zero line crossing" sell pattern

![Figure 9. "Crossing the zero line" sell pattern](https://c.mql5.com/2/20/figure9_-_ao_cross_sell.jpg.png)

Figure 9. "Zero line crossing" sell pattern

The signal appears when the histogram crosses the zero line below. The column that crosses the zero line is the signal column. The signal is displayed on the AO indicator by the Expert Advisor and is referred to as CrS (Cross Sell).

#### 1.4.6. "Twin peaks" buy pattern

![Figure 10. "Two peaks" buy pattern](https://c.mql5.com/2/20/figure10_-_ao_2peak_buy.jpg.png)

Figure 10. "Twin peaks" buy pattern

The buy signal is formed when the histogram is below the zero line, and the last bottom of the indicator is above the previous one. Herewith, between these extremums the histogram didn't rally above zero. The signal is displayed on the AO indicator by the Expert Advisor and is referred to as 2pB (2 peak Buy).

#### 1.4.7. "Twin peaks" sell pattern

![Figure 11. "Two peaks" sell pattern](https://c.mql5.com/2/20/figure11_-_ao_2peak_sell.jpg.png)

Figure 11. "Twin peaks" sell pattern

The sell signal is formed when the histogram is below the zero line, and the last peak of the indicator is below the previous one. Herewith, between these extremums the histogram didn't rally below zero. The signal is displayed on the AO indicator by the Expert Advisor and is referred to as 2pS (2 peak Sell).

#### 1.4.7. Setting orders

When the buy signal column appears, a Buy Stop pending order is set 1 pip higher than the signal bar maximum. When the sell signal column appears, a Sell Stop pending order is set 1 pip lower than the signal bar minimum.

### 1.5. Acceleration/Deceleration Oscillator — AC — signals of the third market dimension

The Acceleration/Deceleration (АС) histogram is the difference between the Awesome Oscillator histogram and 5-period moving average on the Awesome Oscillator:

![Figure 12. AC Oscillator](https://c.mql5.com/2/20/ac.png)

Figure 12. AC Oscillator

The buy signal is formed if two consecutive columns with higher values than the last smallest column (the histogram is above the zero line) appear; if the histogram is below the zero line, then three consecutive green columns (figure 13, B signal — Buy) must be formed.

The sell signal is formed if two consecutive columns with lower values than the last highest column (the histogram is below the zero line) appear; if the histogram is above the zero line, then three consecutive red columns (figure 13, S signal — Sell) must be formed.

![Figure 13. Patterns of the AC Oscillator](https://c.mql5.com/2/20/figure13_-_ac_patterns.jpg.png)

Figure 13. Patterns of the AC Oscillator

The signal is displayed on the AC indicator by the Expert Advisor and is referred to as S (Sell) or B (Buy). When the buy signal column appears, a Buy Stop pending order is set 1 pip higher than the signal bar maximum. When the sell signal column appears, a Sell Stop pending order is set 1 pip lower than the signal bar minimum.

### 1.6. Zone trading — signals of the fourth market dimension

Bill Williams introduces the term of trading zones: green and red. If the current columns АС and АО are green, the price is positioned in the green zone. If the current columns АС and АО are red, the price is in the red zone.

To open new buy positions in the green zone (sell positions in the red zone) at least two green (red) bars in a row are required, and the closing price of the second bar must be higher (lower) than the closing price of the previous bar. However, after five green or red bars in a row, positions are no longer opened.

In case the fifth green (red) bar appears, it is necessary to place a Stop Loss order 1 pip lower than the minimum (higher than the maximum) price of the fifth bar. If the pending order is not executed at the following bar, then it must be changed to the level that is 1 pip lower than the minimum (higher than the maximum) price of the sixth bar, and so on.

Zone trading signals are displayed on the AC indicator as ZS (Zone Sell) and ZB (Zone Buy) (normally they match the signals from the AC indicator):

![Figure 14. Zone trading signals](https://c.mql5.com/2/20/figure14_-_zone_trade.jpg.png)

Figure 14. Zone trading signals

### 1.7. Trading from the balance line — signals of the fifth market dimension

The "buying above the balance line" pattern is formed by two bars when the price is higher than the Alligator indicator. If the opening price of the zero bar (it is also the maximum price of this bar at this moment) is lower than the first previous maximum bar price (can be found few bars behind), then the found maximum price will be the price for opening a buy position for the green zone. If the price is lower than the Alligator line, then one more maximum above the entry price in the green zone is required.

Selling below the balance line is inverse.

The logic of trading from the balance line is described in more details in the article ["Expert Advisor based on the book by Bill Williams"](https://www.mql5.com/en/articles/139).

The Expert Advisor marks patterns with a horizontal line in the place of setting a pending order:

![Figure 15. Places of setting pending orders](https://c.mql5.com/2/20/figure15_-_dimension_5.jpg.png)

Figure 15. Places for setting pending orders

### 1.8. Closing positions

Bill Williams has proposed several ways of closing positions:

- If the bar crosses Alligator's teeth (red line) with a closing price, when a trend exists on the market, then positions must be closed;
- Stop Loss is set after five bars in a row appear in the green (red) zone under the extremum of the last bar;
- If a signal in the opposite direction appears, then all opened positions must be closed.

### 2\. Graphic panel

### 2.1. General information

The interface of the graphic panel is shown below:

![Figure 16. Graphic panel to manage the Expert Advisor](https://c.mql5.com/2/20/figure16_-_graph_panel.jpg.png)

Figure 16. Graphic panel to manage the Expert Advisor

The graphic panel consists of four logical blocks:

- Analyzer settings;
- Alligator settings;
- Settings for displaying and trading by signals of five dimensions;
- Settings for trading.

After changing settings press "Accept" button to save them.

### 2.2. Analyzer settings

Following elements refer to the analyzer settings:

- Show Signals — option for displaying found patterns based on the trading strategy by Bill Williams;
- RGB — color settings to display found patterns;
- Bar count — calculation of the given amount of history bars to mark the chart (if a zero value is given, then the whole chart is marked).

### 2.3. Alligator settings

Alligator settings are the standard settings of this indicator. There is an additional option to disable the display of this indicator ('Show' parameter).

### 2.4. Settings for displaying and trading with signals of five dimensions

They offer a way of displaying separate signals ('Show' parameter), and also trading with separate signals ('Trade' parameter):

- Fractals (Dim1.Fractals line);
- AO (Dim2.AO line);
- AC (Dim3.AC line);
- Zone trading (Dim4.Zones line);
- Trading from the balance line (Dim5.Balance line).

Additionally, there is an option to display all signals irrespective of the current trend ('Show out of trend signals' parameter).

### 2.5. Trading settings

Trading settings have only one parameter – the Lot size.

### 2.6. Main window

The interface of the working chart is shown below (settings panel is minimized):

![Figure 17. Main window](https://c.mql5.com/2/20/sample.png)

Figure 17. Main window

### 3\. Creating Expert Advisor in the MQL5 Wizard

### 3.1. Preparation

Before creating an Expert Advisor you must download the attached archive _billwilliamsts.zip_ and copy its files to the relevant folders of the trading terminal's data catalog.

### 3.2. Creating Expert Advisor

The following steps must be performed for an automated Expert Advisor generation:

Select "New" in the MQL editor, and when a new window appears select "Expert Advisor (generate)":

![Figure 18. MQL Wizard - step 1](https://c.mql5.com/2/22/mqlmaster-1.png)

Figure 18. Creating Expert Advisor — step 1

Enter a name of the Expert Advisor that you wish to create:

![Creating Expert Advisor - step 2](https://c.mql5.com/2/22/mqlmaster-2.png)

Figure 19. Creating Expert Advisor — step 2

The next step requires adding the signal generator used:

![Creating Expert Advisor - step 3](https://c.mql5.com/2/22/mqlmaster-3.png)

Figure 20. Creating Expert Advisor — step 3

Select "Signal of BillWilliams trading system" as a signal generator:

![Creating Expert Advisor - step 4](https://c.mql5.com/2/22/mqlmaster-4.png)

Figure 21. Creating Expert Advisor — step 4

The next step is confirmed without changes:

![Creating Expert Advisor - step 5](https://c.mql5.com/2/22/mqlmaster-5.png)

Figure 22. Creating Expert Advisor — step 5

Selection of a trading signal module is confirmed further:

![Creating Expert Advisor - step 6](https://c.mql5.com/2/22/mqlmaster-6.png)

Figure 23. Creating Expert Advisor — step 6

Trailing Stop parameters are set, if necessary:

![Creating Expert Advisor - step 7](https://c.mql5.com/2/22/mqlmaster-7.png)

Figure 24. Creating Expert Advisor — step 7

Then, money management parameters are set:

![Creating Expert Advisor - step 8](https://c.mql5.com/2/22/mqlmaster-8.png)

Figure 25. Creating Expert Advisor — step 8

The file of the created Expert Advisor must be edited, so that it could react to changes of parameters in the trading panel:

It is necessary to find this section of the code in the file:

```
//--- Creating filter CBillWilliamsSignal
CBillWilliamsSignal *filter0=new CBillWilliamsSignal;
```

```

```

And change it to:

```
filter0=new CBillWilliamsSignal;
```

Declare the _filter0_ global variable:

```
//+------------------------------------------------------------------+
//| Global expert object                                             |
//+------------------------------------------------------------------+
CExpert ExtExpert;
CBillWilliamsSignal *filter0;
```

```

```

And add the chart's event handler:

```
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
  filter0.ChartEvent(id,lparam,dparam,sparam);
}
```

```

```

The created Expert Advisor is now ready to be used.

### 3.3. Restrictions

The Expert Advisor that was created in Wizard has restrictions imposed by the Standard library API:

1. There is no ability to scale the position, if it is already open (improvement to the standard library is required);
2. The Expert Advisor trades only market orders.

To eliminate these disadvantages an additional Expert Advisor _BillWilliamsEA.mq5_, also placed in the attached file, was developed based on the class of trading signals.

### 4\. Test results

### 4.1. EURUSD D1, 2015

![Testing chart EURUSD D1, 2015](https://c.mql5.com/2/22/eurd1-2015__1.png)

Figure 26. Testing chart EURUSD D1, 2015

Detailed results are in the [EUR-D1-2015.zip](https://www.mql5.com/en/articles/download/2049/42894/eur-d1-2015.zip) file.

### 4.2. EURUSD D1, 2010 - 2015

![Testing chart EURUSD D1, 2010-2015 ](https://c.mql5.com/2/22/eurd1-2010-2015.png)

Figure 27. Testing chart EURUSD D1, 2010-2015

Detailed results are in the [EUR-D1-2010-2015.zip](https://www.mql5.com/en/articles/download/2049/42894/eur-d1-2010-2015.zip) file.

### 4.3. EURJPY D1, 2010 - 2015

![Testing chart EURJPY D1, 2010-2015](https://c.mql5.com/2/22/eurjpy__2.png)

Figure 28. Testing chart EURJPY D1, 2010-2015

Detailed results are in the [EURJPY-D1-2010-2015.zip](https://www.mql5.com/en/articles/download/2049/42894/eurjpy-d1-2010-2015.zip) file.

### Conclusion

According to the testing results, we may conclude that the Expert Advisor performs well in trending sections, however, is below break-even on the flat market (in fact, this is a typical situation that Bill Williams has mentioned himself).

In order to achieve acceptable results, techniques of setting stop-loss orders must be combined, because the system by Bill Williams gets considerably delayed and enters a trend only when it develops sufficiently.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/2049](https://www.mql5.com/ru/articles/2049)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/2049.zip "Download all attachments in the single ZIP archive")

[billwilliamsts.zip](https://www.mql5.com/en/articles/download/2049/billwilliamsts.zip "Download billwilliamsts.zip")(12.97 KB)

[eur-d1-2010-2015.zip](https://www.mql5.com/en/articles/download/2049/eur-d1-2010-2015.zip "Download eur-d1-2010-2015.zip")(84.57 KB)

[eur-d1-2015.zip](https://www.mql5.com/en/articles/download/2049/eur-d1-2015.zip "Download eur-d1-2015.zip")(50.88 KB)

[eurjpy-d1-2010-2015.zip](https://www.mql5.com/en/articles/download/2049/eurjpy-d1-2010-2015.zip "Download eurjpy-d1-2010-2015.zip")(88.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/74306)**
(23)


![apirakkamjan](https://c.mql5.com/avatar/avatar_na2.png)

**[apirakkamjan](https://www.mql5.com/en/users/apirakkamjan)**
\|
3 Sep 2019 at 08:41

This is one of the best to learn form.

Thanks Master,  [Nikolay Churbanov](https://www.mql5.com/en/users/mikalaha "mikalaha")

If you get an error when compiling.

![](https://c.mql5.com/3/289/image__91.png)

go to:

SignalBillWilliams.mqh

comments out line# 96 then

at line # 134

Change

```
return INIT_FAILED;
```

To

```
return false;
```

![jprinsloo](https://c.mql5.com/avatar/avatar_na2.png)

**[jprinsloo](https://www.mql5.com/en/users/jprinsloo)**
\|
16 Oct 2019 at 23:59

Thanks Nicolay, this article is very helpful.

I'm using Metatrader 4 [trading platform](https://www.mql5.com/en/trading "Web terminal for the MetaTrader trading platform") which doesn't seem to allow MetaEditor 5

Is there a way that I can get the Trading Signals Module to run on MetaTrader4?

![roger barry](https://c.mql5.com/avatar/avatar_na2.png)

**[roger barry](https://www.mql5.com/en/users/rbarry)**
\|
6 Dec 2019 at 09:46

I have loaded your EA and made the correction as detailed in your instructions.

I have been checking the alerts and point of exit and entry but are having difficulty in what they are indicating

that I should do .. Like in AC ZS S and on the main chart FrS when they are way under the lips of the alligator and the bars are

pulling away supporting the down trend.  [![](https://c.mql5.com/3/300/Image__2.JPG)](https://c.mql5.com/3/300/Image__1.JPG "https://c.mql5.com/3/300/Image__1.JPG")

Could be I do not understand how it works?

Appreciate any help. Regards, Roger

RE

l

![SULAIMAN LALANI](https://c.mql5.com/avatar/2019/12/5DEAD27A-A88D.jpg)

**[SULAIMAN LALANI](https://www.mql5.com/en/users/sslalani)**
\|
18 Sep 2020 at 04:55

I read this article today (17 Sep 2020), some 4 years after it was added.  Very interesting. Not sure if anyone has used the EA with real money and what are the results it generated.


![Daniel Steven Betancourt Correa](https://c.mql5.com/avatar/2025/5/6834b2f6-bc8d.png)

**[Daniel Steven Betancourt Correa](https://www.mql5.com/en/users/btradersoficial)**
\|
22 Oct 2025 at 14:53

For the date 2025, I installed the proposed BillWilliams advisor and honestly I find that it is not good, because it opens many martingale trades and there are so many trades that when I have positive trades it does not overcome the losses. Unfortunately this EA in my personal experience does not work.


![Graphical Interfaces I: Testing Library in Programs of Different Types and in the MetaTrader 4 Terminal (Chapter 5)](https://c.mql5.com/2/21/Graphic-interface__4.png)[Graphical Interfaces I: Testing Library in Programs of Different Types and in the MetaTrader 4 Terminal (Chapter 5)](https://www.mql5.com/en/articles/2129)

In the previous chapter of the first part of the series about graphical interfaces, the form class was enriched by methods which allowed managing the form by pressing its controls. In this article, we will test our work in different types of MQL program such as indicators and scripts. As the library was designed to be cross-platform so it could be used in all MetaTrader platforms, we will also test it in MetaTrader 4.

![Graphical Interfaces I: Functions for the Form Buttons and Deleting Interface Elements (Chapter 4)](https://c.mql5.com/2/21/Graphic-interface__3.png)[Graphical Interfaces I: Functions for the Form Buttons and Deleting Interface Elements (Chapter 4)](https://www.mql5.com/en/articles/2128)

In this article, we are going to continue developing the CWindow class by adding methods, which will allow managing the form by clicking on its controls. We will enable the program to be closed by a form button as well as implement a minimizing and maximizing feature for the form.

![Adding a control panel to an indicator or an Expert Advisor in no time](https://c.mql5.com/2/22/avatar.png)[Adding a control panel to an indicator or an Expert Advisor in no time](https://www.mql5.com/en/articles/2171)

Have you ever felt the need to add a graphical panel to your indicator or Expert Advisor for greater speed and convenience? In this article, you will find out how to implement the dialog panel with the input parameters into your MQL4/MQL5 program step by step.

![Studying the CCanvas Class. Anti-aliasing and Shadows](https://c.mql5.com/2/21/CCanvas_class_Standard_library_MetaTrader5.png)[Studying the CCanvas Class. Anti-aliasing and Shadows](https://www.mql5.com/en/articles/1612)

An anti-aliasing algorithm of the CCanvas class is the base for all constructions where anti-aliasing is being used. The article contains information about how this algorithm operates, and provides relevant examples of visualization. It also covers drawing shades of graphic objects and has a detailed algorithm developed for drawing shades on canvas. The numerical analysis library ALGLIB is used for calculations.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/2049&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5072023650103669470)

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