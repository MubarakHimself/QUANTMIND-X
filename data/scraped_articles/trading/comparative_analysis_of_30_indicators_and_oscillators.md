---
title: Comparative Analysis of 30 Indicators and Oscillators
url: https://www.mql5.com/en/articles/1518
categories: Trading
relevance_score: 6
scraped_at: 2026-01-23T11:30:50.695336
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1518&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062520940471952312)

MetaTrader 4 / Trading


### Introduction

The abundance of indicators and oscillators developed nowadays inevitably leads to the problem of choosing the most efficient of them. Very often a beginning trader first facing this plenty of available analysis and forecasting tools starts testing them on history data and demo accounts. After that a set of conclusions is made about the efficiency and uselessness of this or that indicator in certain situations.

However such an estimation is not always objective because of the large number of parameters (symbol, chart period, volatility, etc.) and therefore the large amount of the analyzed information. More reliable conclusions about certain indicators and oscillators can be made after conducting their comparative analysis.

Actually the comparative analysis of indicators and oscillators is impossible without their simultaneous consideration bounding them to a price chart. However attaching even ten different indicators to a chart considerably overloads it. The obtained figure becomes tangled and unsuitable for analysis. An Expert Advisor described in this article can solve this problem and make the conduction of the comparative analysis more convenient.

### Expert Advisor

Pay attention, this EA is not intended for the execution of live trading and therefore does not contain the money management block. Block of trade execution is implemented in a very simple way. The main task of the Expert Advisor is the provision of information about the presence or absence of signals from different indicators in connection with a price chart.

The program analyzes the following indicators and oscillators:

01. Acceleration/Deceleration — АС
02. Accumulation/Distribution - A/D
03. Alligator & Fractals
04. Gator Oscillator
05. Average Directional Movement Index - ADX
06. Average True Range - ATR
07. Awesome Oscillator
08. Bears Power
09. Bollinger Bands
10. Bulls Power
11. Commodity Channel Index
12. DeMarker
13. Envelopes
14. Force Index
15. Ichimoku Kinko Hyo (1)
16. Ichimoku Kinko Hyo (2)
17. Ichimoku Kinko Hyo (3)
18. Money Flow Index – MFI
19. Moving Average
20. MACD (1)
21. MACD (2)
22. Moving Average of Oscillator (MACD Histotgram) (1)
23. Moving Average of Oscillator (MACD Histotgram) (2)
24. Parabolic SAR
25. RSI
26. RVI
27. Standard Deviation
28. Stochastic Oscillator (1)
29. Stochastic Oscillator (2)
30. Williams Percent Range

For the implementation of this task a digital matrix consisting of "-1", "0" and "1" is drawn on a chart. Each matrix line belongs to a certain indicator or oscillator. The matrix columns are formed at each moment of time (according to the selected chart period). The appearance of "-1" in a certain matrix line denotes the presence of a signal to sell produced by a certain indicator (oscillator); appearance of "1" - presence of a buy signal, "0" denotes the absence of any signal. Fig. 1 illustrates the program operation results.

Fig. 2 illustrates the analysis of RVI operation based on the matrix. In the line 26 (it contains data about this indicator) "1" is recorded when the main line moves above the signal one (the indicator recommends to buy), "-1" - when the signal line is above the main one (Sell signal). Due to the indicator characteristics this line does not contain "0".

![](https://c.mql5.com/2/16/2.jpg)

Due to characteristics of indicators and oscillators, we can detect two types of indexes in the matrix: constantly significant ones, the corresponding flags (values in the matrix) of which are never equal to zero, and pointwise significant, the flags of which can accept zero and make signals only in some certain moments of time (for example, Parabolic SAR). It is shown in Fig. 3.

![](https://c.mql5.com/2/16/3_2.jpg)

It is recommended to analyze the obtained information and then form a package of indicators the following way. First of all select the chart part with a trend. Then the period of possible trades is specified. The beginning of such a trend can be a flat period before the selected trend or trend origination; the end of the period is the last time moment when a trade is still profitable. Thus anticipatory and late indicator signals are not taken into account. In fig. 4 such a period is indicated by green lines. After that information by indicators (oscillators) is analyzed inside this period: all indicators showing trend correctly are taken, other ones are shifted away. After that the package can be extended or restricted by the analogous analysis on other time intervals, as a result the final package will be formed.

The program provides the testing of formed packages. For this purpose in a corresponding line of the strategy processing Block enumerate conditions (indicator showings) based on which the final decision about selling (or buying) is made.

Thus the combined analysis of the price behavior and signals produced by each certain indicator (oscillator) provides the possibility of selecting the most efficient of them for further formation of a package of indicators.

### Algorithm

At the first stage values of "flags" (-1, 0, 1) are defined for each indicator (oscillator). Assuming that one indicator (for example, MACD) produces signals different ways (convergence/divergence, crossing of a zero line, etc.), the program code contains the description of its defining principle. For example, analysis of the oscillator "Williams Percent Range" is implemented so:

```
//30. Williams Percent Range
//Buy: crossing -80 upwards
//Sell: crossing -20 downwards
if (iWPR(NULL,piwpr,piwprbar,1)<-80&&iWPR(NULL,piwpr,piwprbar,0)>=-80)
{f30=1;}
if (iWPR(NULL,piwpr,piwprbar,1)>-20&&iWPR(NULL,piwpr,piwprbar,0)<=-20)
{f30=-1;}
```

After that unique objects of the 'Text' type are formed from the obtained digital values of flags (to avoid duplication of object names current time value is used) and the displayed:

```
timeident=TimeCurrent(); //Time to form the unique object name
for (i=0;i<=29;i++) //Loop for displaying values

//Forming unique object names
{ident=DoubleToStr(30-i,0)+"   "+DoubleToStr(timeident,0);

//Creating objects, indicating their location
ObjectCreate(ident,OBJ_TEXT,0,timeident,WindowPriceMin()+Point*5*(i+1));

info=DoubleToStr(f[30-i],0); //Forming a text line to be displayed
ObjectSetText(ident,info,6,"Arial", Black);} //Describing the display format
```

To check the formed packages the program includes "The block of processing a strategy and placing the Main Flag". This block contains conditions providing which the EA must buy (if the Main Flag is equal to 1) and sell (if the Main Flag is equal to -1). If the described conditions are not fulfilled, the Main Flag stays equal to zero and trades are not executed. The EA also contains a block of position closing (it is commented).

```
if(f8==1&&f21==1) //Set of conditions, providing which Buy is executed
flag=1;
if(f8==-1&&f21==-1) //Set of conditions, providing which Sell is executed
flag=-1;
```

Parameters of each indicator (oscillator) are described in the form of variables, which allows their automatic optimization (initial parameters are those generally accepted).

### Conclusion

As said above, the offered Expert Advisor is not a ready automated trading system. Its main task is the conduct of a comparative analysis of indicators and oscillators for further formation of a package. The EA demonstrates signals produced by indicators not overloading a trader's desktop making the analysis very convenient.

The flexibility of the EA is in the following: a new set of indicators can be included into the matrix and formed packages can be tested on history data.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1518](https://www.mql5.com/ru/articles/1518)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1518.zip "Download all attachments in the single ZIP archive")

[Matrix.mq4](https://www.mql5.com/en/articles/download/1518/Matrix.mq4 "Download Matrix.mq4")(32.39 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39451)**
(14)


![Michael](https://c.mql5.com/avatar/avatar_na2.png)

**[Michael](https://www.mql5.com/en/users/nondisclosure)**
\|
21 May 2008 at 20:23

I have to add Bears into the "what's the point" section.  It's always getting set to 0.

```
if (iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,2)>0&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,1)>0&&iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,0)>0&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,2)>iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,1)&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,1)>iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,0))
{f10=-1;}
f10=0; //Ïîêà íå èñïîëüçóåì
```

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
21 May 2008 at 23:43

**nondisclosure:**

I have to add Bears into the "what's the point" section.  It's always getting set to 0.

```
if (iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,2)>0&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,1)>0&&iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,0)>0&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,2)>iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,1)&&iBullsPower
    (NULL,pibull,pibullu,PRICE_CLOSE,1)>iBullsPower(NULL,pibull,pibullu,PRICE_CLOSE,0))
{f10=-1;}
f10=0; //Ïîêà íå èñïîëüçóåì
```

Would it not be possible (since the resulting [matrix](https://www.mql5.com/en/docs/matrix/matrix_types " MQL5 Documentation: Matrix and vector types") is known after calculation) to make another pass through the data and mark via color codes or some other graphic scheme the data much as has been marked up above in the description above?  It would greatly help those with weak eyesight (like me).


![Mehmet Bastem](https://c.mql5.com/avatar/avatar_na2.png)

**[Mehmet Bastem](https://www.mql5.com/en/users/mehmet)**
\|
22 May 2008 at 00:43

**Mehmet:**

Please translate English. Your article is Rusian language.

Thank You.  but  expert advisor BUY or SELL not work. Order Send Error=131

Please Help Me ?

[OrderSend](https://docs.mql4.com/trading/ordersend "OrderSend")(Symbol(),OP\_BUY,Lot,Ask,slipp,Ask-distance\*Point,Ask+distance\*Point,"Buy")

![MisterH](https://c.mql5.com/avatar/2011/9/4E7B281E-AF2F.jpg)

**[MisterH](https://www.mql5.com/en/users/misterh)**
\|
23 May 2008 at 22:04

Definitely one of the better ideas on this site. Keep up the good work.

![Jose](https://c.mql5.com/avatar/avatar_na2.png)

**[Jose](https://www.mql5.com/en/users/jcalderon)**
\|
28 Dec 2008 at 02:09

I have download and save under expert folder.

The system takes long time working and nothing is show on the main window. I have check with diferent colour (black, yellow) and size fonts on the [ObjectSetText](https://docs.mql4.com/objects/objectsettext "ObjectSetText") function.

I have check on weekend, so there is no live market. Is this the reason?

I have change also distance=12 due ODL needs minimum of 10.

Also checked with 'strategy test' and nothing appear!!!

Can some one give some help?

Many thanks

![Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://c.mql5.com/2/15/601_23.gif)[Fallacies, Part 1: Money Management is Secondary and Not Very Important](https://www.mql5.com/en/articles/1526)

The first demonstration of testing results of a strategy based on 0.1 lot is becoming a standard de facto in the Forum. Having received "not so bad" from professionals, a beginner sees that "0.1" testing brings rather modest results and decides to introduce an aggressive money management thinking that positive mathematic expectation automatically provides positive results. Let's see what results can be achieved. Together with that we will try to construct several artificial balance graphs that are very instructive.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization](https://c.mql5.com/2/15/575_71.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization](https://www.mql5.com/en/articles/1516)

This article dwells on implementation algorithm of simplest trading systems. The article will be useful for beginning traders and EA writers.

![Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part II)](https://c.mql5.com/2/15/576_89.gif)[Expert Advisors Based on Popular Trading Systems and Alchemy of Trading Robot Optimization (Part II)](https://www.mql5.com/en/articles/1517)

In this article the author continues to analyze implementation algorithms of simplest trading systems and describes some relevant details of using optimization results. The article will be useful for beginning traders and EA writers.

![Metalanguage of Graphical Lines-Requests. Trading and Qualified Trading Learning](https://c.mql5.com/2/15/597_26.gif)[Metalanguage of Graphical Lines-Requests. Trading and Qualified Trading Learning](https://www.mql5.com/en/articles/1524)

The article describes a simple, accessible language of graphical trading requests compatible with traditional technical analysis. The attached Gterminal is a half-automated Expert Advisor using in trading results of graphical analysis. Better used for self-education and training of beginning traders.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=rqhmaswfujvqdahbgbvgtmiwbqkhbwyj&ssn=1769157048355705920&ssn_dr=0&ssn_sr=0&fv_date=1769157048&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1518&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Comparative%20Analysis%20of%2030%20Indicators%20and%20Oscillators%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176915704885125981&fz_uniq=5062520940471952312&sv=2552)

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