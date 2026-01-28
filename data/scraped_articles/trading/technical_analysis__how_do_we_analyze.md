---
title: Technical Analysis: How Do We Analyze?
url: https://www.mql5.com/en/articles/174
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:40:24.946823
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/174&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083031449419912087)

MetaTrader 5 / Trading


### Introduction

Looking through various publications, somehow related to the use of technical analysis, we come across information that sometimes leaves you indifferent, and sometimes a desire to comment on the information read appears. It is this desire that led to the writing of this article, which first of all is pointed at a try to analyze our actions and results once again, using a particular method of analysis.

### Redrawing

If you look at comments on the indicators published at [https://www.mql5.com/en/code](https://www.mql5.com/en/code), you will notice that the vast majority of users have extremely negative attitude towards those of them whose previously calculated values are changing and being redrawn during the formation of the next bar.

Once it becomes clear that the indicator is redrawn, it is no more interesting to anyone. Such an attitude to the redrawing of indicators is often quite reasonable, but in some cases, redrawing may be not as terrible as it seems at first glance. To demonstrate this, we will analyze the simplest SMA indicator.

In the Figure 1 blue color shows the impulse characteristic of the low rate filter, which corresponds to the SMA (15) indicator. For the case shown in the figure, the SMA (15) is a sum of the last 15 counts of the input sequence, where each of the input counts is multiplied by 1/15 corresponding to the impulse characteristic presented. Now, having SMA (15) value calculated on the interval of 15 counts, we have to decide to what point of time we should assign this value.

Accepting SMA (15) as an average of the previous 15 input counts, this value should appear as shown on the upper chart, thus it should correspond to a zero bar. In case of accepting SMA (15) as a low rate filter with the impulse characteristic of a finite length, thus the calculated value, taking into account the delay in the filter must match the bar number seven, as shown on the bottom chart.

So, by the simple shift, we transform the chart of a moving average into the chart of a low rate filter with zero latency.

Note that in case of using zero latency charts, some traditional analysis methods slightly change their meaning. For example, the intersection of two MA plots with different periods and the intersection of the same plots with compensated delay will occur at different times. In the second case we will get intersection moments, which will be determined only by MA periods, but not by their latency.

Returning to the figure 1 it is easy to see in the bottom plot the SMA (15) curve doesn't reach the most recent counts of the input signal by the value equal to the half of the averaging period. An area of the seven counts is forming, where the SMA (15) value is not defined. We can assume that having compensated the delay, we have lost some information because of an area of ambiguity has appeared, but that is fundamentally wrong.

The same ambiguity is on the upper chart (figure 1), but due to the shift it is hidden in the right side, where there are no input counts. The MA chart loses its time binding to the input sequence and the delay size depends on the MA smoothing period because of the shift.

![Figure 1. Impulse response of the SMA (15)](https://c.mql5.com/2/2/Fig_0001__1.png)

Figure 1. Impulse response of the SMA (15)

If all the delays occurred are always compensated when using MA with different periods, this will result in charts with a certain time binding to the input sequence and to each other. But besides the irrefutable advantages, this approach assumes areas of ambiguity. The reason of their occurrence is well known features of time finite length sequences processing, but not our reasoning mistakes.

We are facing the problems occurring on the edges of such sequences using interpolation algorithms, various filtrations, smoothing, etc. And nobody has a thought to hide the result part by its shifting.

I must admit that MA charts with a certain not-drawn part are a more correct representation of a filtering, but they look very unusual. From the formal point of view, we can't calculate the value of filter SMA (15), Shift=-7 for output counts with the index lower than 7. So is there any other way to smooth down counting on the edge if the input sequence?

Let's try to filter these counts using the same SMA algorithm, but decreasing its period of smoothing with each bar approaching to the zero. Also, we should not forget about the delay compensation of the filter used.

![Figure 2. Modified SMA](https://c.mql5.com/2/2/Image2__1__2.png)

Figure 2\. Modified SMA

Figure 2 shows how output counts with indexes from 0 up to 6 will be formed in this case. The counts which will take part in the calculating of average value are conventionally marked with colored dots in the bottom of the figure, and vertical lines show to which output count this average will be assigned. On the zero bar no processing is made, the value of the input sequence is assigned to the output one. For the output sequence with indexes seven or greater calculations are made with the usual SMA (15) Shift =- 7.

It is obvious that when using such an approach the output chart on the index interval from 0 up to 6 will be redrawn with each new bar occurrence, and the intensity of redrawing will increase with a decrease of the index. At the same time a delay for any count of the output sequence is compensated.

In the example analyzed, we have got a redrawing indicator which is an analog of the standard SMA (15), but with a zero delay and extra information on the edge of input sequence which is absent in the standard SMA (15). Accepting zero delay and extra information as an advantage, nevertheless we've got a redrawing indicator, but it's more informative than the standard SMA indicator.

It should be emphasized that the redrawing in this example does not lead to any catastrophic consequences. On the resulting plot there is the same information as to the standard SMA, with its counts shifted to the left.

In the example considered the odd SMA period was chosen, which completely compensated the delay in time, which is for SMA:

t = (N-1)/2,

where N is a smoothing period.

Due to the fact that for even values of N the delay can't be fully compensated using this approach and the offered method of counts smoothing on the sequence edge is not the only possible, the variant of indicator construction is considered only as an example here, but not as a complete indicator.

### Multi-timeframe

On the MQL4 and [MQL5](https://www.mql5.com/en/code) websites you can see the so called multi-timeframe indicators. Let's try to figure out what multi-timeframe gives us by an example of " [iUniMA MTF](https://www.mql5.com/en/code/180)" indicator.

Assume we are in the lowest M1 timeframe window, and are going to show smoothed Open or Close of the M30 timeframe value in the same window, applying SMA (3) for smoothing. It is known M30 timeframe sequence forms from M1 timeframe sequence by sampling every thirtieth value and discarding the remaining 29 values. The doubts appear whether it's reasonable to use the M30 timeframe sequence.

If we have an access to a certain amount of information on M1 timeframe, then what's the point of contacting with M30 timeframe, which contains only one-thirties part of that information? In the considered case we intensionally eliminate the most of the information available and process what remains from SMA (3) and display the result in the M1 timeframe source window.

It's obvious that the actions described look quite strange. Isn't it easier just to apply SMA (90) to the complete sequence of M1 timeframe? The frequency of SMA (90) filter slice on M1 timeframe is equal to the frequency of SMA (3) filter slice on M30 timeframe.

In figure 3 an example of using multi-timeframe indicator " [iUniMA MTF](https://www.mql5.com/en/code/180)" on the chart of EURUSD M1 currency pair is shown. The blue curve is the result of applying SMA (3) to the M30 timeframe sequence. In the same figure the curve of red color is the result obtained with the regular "Moving Average" indicator. Hence the result of applying the standard SMA (90) indicator is more natural.

And no special techniques are required.

![Figure 3. Multi-timeframe indicator usage](https://c.mql5.com/2/2/Fig_0003__1.png)

Figure 3. Multi-timeframe indicator usage

Another variant of multi-timeframe indicators usage is possible, when an information from the lowest timeframe according to the current one is shown on a terminal. This variant can be useful if you need to compress the scale of quote displaying even more than it is allowed by the terminal on the lowest timeframe. But in this case also, no additional information about the quotes can be obtained.

It's easier to turn to the lowest timeframe and to handle all the data processing with regular indicators, but not with multi-timeframe ones.

When developing custom indicators or Expert Advisors special situations can occur, when an organization of access to various timeframe sequences is reasonable and is the only possible solution, but even in this case we should remember that the higher timeframe sequences are formed from the lower ones and don't carry any additional unique information.

### Candlestick charts

In publications of technical analysis we can often meet excited relation to everything connected with candlestick charts. For example, in the article " [Analysing Candlestick Patterns](https://www.mql5.com/en/articles/101)" is said: "The advantage of candlesticks is that they represent data in a way that it is possible to see the momentum within the data". ... Japanese candlestick charts can help you penetrate "inside" of financial markets, which is very difficult to do with other graphical methods.

And that is not the only source of such statements. Let's try to figure out whether candlestick charts allow us to get into the financial markets.

"Low", "High", "Open" and "Close" values sequences are used for rates representation in form of candlestick charts. Let's remember what kind of values these are. "Low" and "High" values are equal to the minimal and the maximal rates values on the period of chosen timeframe. The "Open" value is equal to the first known value of the rates in the analyzed period. The "Close" value is equal to the last known value of the rates in the analyzed period. What could this mean?

This primarily means that somewhere there are market rates from which values "Low", "High", "Open" and "Close" sequences are formed. «Low», «High», «Open» and «Close» values in this method of their formation are not strictly bound to the time. Besides, there is no way to restore initial rates by these sequences. The most interesting thing is that the same combination of «Low», «High», «Open» and «Close» values on any bar of any timeframe can be formed by an infinite number of variants of the original rates sequence. These conclusions are trivial and based on well-known facts.

Thus the original information is irreversibly distorted if using market rates in form of candlestick charts. Using strict math methods of analysis for rates behavior assessment by any of «Low», «High», «Open» or «Close» sequences the results are connected not to market rates, but to their distorted representation. Nevertheless, we should admit candlestick charts analysis has many advocates.

How could that be explained? Perhaps the secret is that initially the target of rates representation in form of candlestick charts was fast visual intuitive market analysis, but not applying math analysis methods to candlestick charts.

Thus, to understand how rates representation in form of candlestick charts can be used with technical analysis, let's turn to the pattern recognition theory, which is closer to usual human decision methods, than the formal math analysis methods are.

In figure 4, according to the pattern recognition theory a simplified scheme of decision making is drawn. A decision in this case can be a definition of trend beginning or ending moment and detection of optimal moments for opening a position of time moments, etc.

![Figure 4. Decision making scheme](https://c.mql5.com/2/2/Image4__1__2.png)

Figure 4. Decision making scheme

As it's shown on the figure 4 initial data (rates) is preliminary treated and significant features are formed from them in block 2. In our case these values are «Low», «High», «Open» and «Close». We can't impact on processes in blocks 1 and 2. On the terminal side only that features which are already dedicated for us are available. These features come to block 3, where decisions are made on their base.

Decision making algorithm can be implemented in software or manually by strict adherence to the specifications. We can develop and in some way implement decision making algorithms, but we can't choose significant features from analyzed rates sequence, because this sequence is not available for us.

From the point of increasing the probability of making the right decision the most crucial thing is the choice of significant features and their essential amount, but we don't have this important possibility. In this case, to impact on the reliability of this or that market situation recognition is quiet difficult, since even the most advanced decision making algorithm isn't able to compensate the disadvantages connected with nonoptimal choice of features.

What is a decision making algorithm according to this scheme? In our case, that is a set of rules published in the candlestick charts analysis research. For example, the definition of candlestick charts types, the disclosure of their various combinations meaning, etc.

Referring to the theory of pattern recognition, we come to the conclusion that candlestick charts analysis fits the scheme of this theory, but we don't have any reason to assert that the choice of «Low», «High», «Open» and «Close» values as significant features is the best. Also, a nonoptimal choice of features can dramatically reduce the probability of making correct decisions in a process of rates analyzing.

Going back to the beginning, we can confidently say candlestick chart analysis would hardly "penetrate "inside" of financial markets" or "see the momentum within the data". Moreover, its efficiency compared with other methods of technical analysis can cause serious doubts.

### Conclusion

Technical analysis is a fairly conservative area. Basic postulates formation of technical analysis took part in the 18-19 centuries, and this basis reached our days almost unchanged. At the same time over the past decade, the global market structure dramatically changed during its development. Development of online trading contributed the nature of market behavior.

In this situation, even usage of the most popular theories and methods of classical technical analysis doesn't always provide us with sufficient trade efficiency.

Nevertheless, the availability of computers and interest in trading on markets being shown by people of various professions, can stimulate the development of technical analysis methods. It is obvious that today market analysis needs more accurate and sensitive analytical tools development.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/174](https://www.mql5.com/ru/articles/174)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to the Empirical Mode Decomposition Method](https://www.mql5.com/en/articles/439)
- [Kernel Density Estimation of the Unknown Probability Density Function](https://www.mql5.com/en/articles/396)
- [The Box-Cox Transformation](https://www.mql5.com/en/articles/363)
- [Time Series Forecasting Using Exponential Smoothing (continued)](https://www.mql5.com/en/articles/346)
- [Time Series Forecasting Using Exponential Smoothing](https://www.mql5.com/en/articles/318)
- [Analysis of the Main Characteristics of Time Series](https://www.mql5.com/en/articles/292)
- [Statistical Estimations](https://www.mql5.com/en/articles/273)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/2548)**
(15)


![Алёша](https://c.mql5.com/avatar/2017/4/59038297-75D6.jpg)

**[Алёша](https://www.mql5.com/en/users/alex_bondar)**
\|
20 Jan 2013 at 09:21

Very true article, thank you !

![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
13 Feb 2014 at 15:51

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/en/forum)

[Press review](https://www.mql5.com/en/forum/12423/page90#comment_764589)

[newdigital](https://www.mql5.com/en/users/newdigital "https://www.mql5.com/en/users/newdigital"), 2014.02.13 15:46

3 Types of Forex Analysis(based on [dailyfx article](https://www.mql5.com/go?link=https://www.dailyfx.com/forex/education/trading_tips/chart_of_the_day/2014/02/13/3_methods_of_forex_analysis.html "http://www.dailyfx.com/forex/education/trading_tips/chart_of_the_day/2014/02/13/3_methods_of_forex_analysis.html"))

**Fundamental**

Forex fundamental centers mostly around the currency’s interest rate. Other fundamental factors are included such as Gross Domestic Product, inflation, manufacturing, economic growth activity. However, whether those other fundamental releases are good or bad is of less importance than how those releases affect that country’s interest rate.

As you review the fundamental releases, keep in mind how it might affect the future movement of the interest rates. When investors are in a risk seeking mode, money follows yield and higher rates could mean more investment. When investors are in a risk adverse mentality, then money leaves yield for safe haven currencies.

**Technical**

Forex technical analysis involves looking at patterns in price history to determine the higher probability time and place to enter and exit a trade. As a result, forex technical analysis is one of the most widely used types of analysis.

Since FX is one of the largest and most liquid markets, the movements on a chart from the price action generally gives clues about hidden levels of supply and demand. Other patterned behavior such as which currencies are trending the strongest can be obtained by reviewing the price chart.

Other technical studies can be conducted through the use of indicators. Many traders prefer using indicators because the signals are easy to read and it makes forex trading simple.

[http://www.dailyfx.com/forex/education/trading_tips/daily_trading_lesson/2012/01/20/Technical_Versus_Fundamental_Analysis_in_Forex.html?CMP=SFS-70160000000Nc3HAAS](https://www.mql5.com/go?link=https://www.dailyfx.com/forex/education/trading_tips/daily_trading_lesson/2012/01/20/Technical_Versus_Fundamental_Analysis_in_Forex.html?CMP=SFS-70160000000Nc3HAAS "http://www.dailyfx.com/forex/education/trading_tips/daily_trading_lesson/2012/01/20/Technical_Versus_Fundamental_Analysis_in_Forex.html?CMP=SFS-70160000000Nc3HAAS")**Sentiment**

Forex sentiment is another widely popular form of analysis. When you see sentiment overwhelmingly positioned to one direction that means the vast majority of traders are already committed to that position.

Since we know there is a large pool of traders who have already BOUGHT, then these buyers become a future supply of sellers. We know that because eventually, they are going to want to close out the trade. That makes the EUR to USD vulnerable to a sharp pull back if these buyers turn around and sell to close out there trades.

![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
15 Feb 2014 at 07:00

[Forum on trading, automated trading systems and testing trading strategies](https://www.mql5.com/en/forum)

[Press review](https://www.mql5.com/en/forum/12423/page91#comment_766927)

[newdigital](https://www.mql5.com/en/users/newdigital "https://www.mql5.com/en/users/newdigital"), 2014.02.15 06:58

Trader Styles and Flavors(based on dailyfx article)

**Technical vs. Fundamental**

Technical analysis is the art of studying past price behavior and attempting to anticipate price moves in the future. These are traders that focus solely on price charts and often times incorporate indicators and tools to assist them. They look at price action, support and resistance levels, and chart patterns to create trading strategies that hopefully will turn a profit.

Fundamental analysis looks at the underlying economic conditions of each currency. Traders will turn to the Economic Calendar and Central Bank Announcements. They attempt to predict where price might be headed based on interest rates, [jobless claims](https://www.metatrader5.com/en/terminal/help/fundamental/economic_indicators_usa/usa_jobless_claims "Jobless Claims (Initial claims)"), treasury yields and more. This can be done by looking at patterns in past economic news releases or by understanding a country’s economic situation.

**Short-Term vs. Medium-Term vs. Long-Term**

Deciding what time frame we should use is mostly decided by how much time you have to devote to the market on a day-to-day basis. The more time you have each day to trade, the smaller the time frame you could trade, but the choice is ultimately yours.

Short-Term trading generally means placing trades with the intention of closing out the position within the same day, also referred to as

“Day Trading” or “Scalping” if trades are opened and closed very rapidly. Due to the speed at which trades are opened and closed, short-term traders use small time-frame charts (Hourly, 30min, 15min, 5min, 1min).

Medium-Term trades or “Swing Trades” typically are left open for a few hours up to a few days. Common time frames used for this type of trading are Daily, 4-hour and hourly charts.

Long-Term trading involves keeping trades open for days, weeks, months and possibly years. Weekly and Daily charts are popular choices for long term traders. If you are a part-time trader, it might be suitable to begin by trading long term trades that require less of your time.

**Discretionary vs. Automated**

Discretionary trading means a trader is opening and closing trades by using their own discretion. They can use any of the trading styles listed above to create a strategy and then implement that strategy by placing each individual trade.

The first challenge is creating a winning strategy to follow, but the second (and possibly more difficult) challenge is diligently following the strategy through thick and thin. The psychology of trading can wreak havoc on an otherwise profitable strategy if you break your own rules during crunch time.

Automated trading or algorithmic trading requires the same time and dedication to create a trading strategy as a discretionary trader, but then the trader automates the actual trading process. In other words, computer software opens and closes the trades on its own without needing the trader’s assistance. This has three main benefits. First, it saves the trader quite a bit of time since they no longer have to monitor the market as closely to input trades. Second, it takes the emotions out of trading by letting a computer open and close trades on your behalf. This means you are following your strategy to the letter and are not able to deviate. And third, automated strategies can trade 24 hours a day, 5 days a week giving your account the ability to take advantage of any opportunity that comes its way no matter the time of day.

![Sergey Golubev](https://c.mql5.com/avatar/2012/12/50D09349-3D90.jpg)

**[Sergey Golubev](https://www.mql5.com/en/users/newdigital)**
\|
13 Jun 2014 at 19:28

[Learning to Read Forex Charts](https://www.mql5.com/go?link=http://www.fxwire.com/forex_news/global-forex-news/learning-read-forex-charts "http://www.fxwire.com/forex_news/global-forex-news/learning-read-forex-charts")

Technical analysis is considered one of the easiest ways to analyze the foreign exchange market. It involves the analysis of charts and graphs to ascertain future currency price movements, and differs massively from fundamental analysis in that it does not require the analysis of forex news, reports or other economic releases to establish future price movements.

**Becoming a Technical Analyst**

The first step in becoming a successful technical analyst is to learn how to read forex charts. Outlined below are some simple steps that every trader should take when first starting out with technical analysis:

When analyzing a currency pair you will need to look out for a prevailing trend. Start off with charts that provide long-term data (for example days, weeks and months) and go back over the course of a number of years. Such charts contain an exhaustive amount of data, thus providing a much clearer picture of exactly what the currency pair is doing than if using short-term charts (5 minutes, 15 minutes, 30 minutes or one hour). This extra data also makes the technical indicators much more steadfast and reliable.

**How to Identify a Trend**

To identify a trend simply look at the graph presented before you and decide whether it is rising more than it is falling, or vice versa. Trends can be shallow or sharp, weeks short or years long. Practice identifying trends and locating the moment where the trends change direction.

Even if you are a short-term scalper or day trader who wishes to place a trade for no longer than an hour, it is still important to identify trends. Identifying a trend is one of the best steps a trader can take in executing more accurate, profitable trades.

Upon identifying a trend in a long-term chart you will then be able to compare that trend with the one that you have found in the short-term charts. Within the path set by the prevailing trend you will discover that there are a variety of short-term and intermediate-term trends. Overall the pattern on the graph will follow a particular path as set by the longest-term trend.

**Identifying Support and Resistance Levels**

After this point you will then need to locate the support and resistance levels. In technical analysis these are regarded as the ‘floor’ and ‘ceiling’ points on a graph and are key locations on a chart where the price continually refuses to break through. The price will reach a peak or a valley, after which point it will not go any further, but will instead alter its direction. The more frequently this occurs, the stronger the support and resistance levels are.

Draw a straight line as you pass through most of the support points. Draw another line as you pass through most of the resistance points. This provides you with a lucid picture of the price channel, or the path that the currency pair’s trend is following. This is a highly powerful yet incredibly simply tool for determining a currency’s future pathway.

**What is a Range- Bound?**

In the event that ‘range bound’ occurs, this simply means that the support and resistance levels are so strong that the graph’s movements appear to ‘bounce’ in a sideways pattern. Nevertheless this generally occurs 80% of the time and many traders prefer to trade within the channels.

**Breaking out of a Price Channel**

In the event that a currency pair becomes released from a price channel, in some instances it falls back into the channel, and in others it gains momentum and continues to move. The latter movement is better known as a ‘momentum market’, and is an alternative way to trade the range: by setting an entry order for the price to break out, either below or above the channel.

![Simalb](https://c.mql5.com/avatar/2017/11/59FF5B41-9224.png)

**[Simalb](https://www.mql5.com/en/users/simalb)**
\|
21 Nov 2017 at 10:43

With what I read in your articles, I feel less ignorant. I hope it's true . Anyway thanks

![Designing and implementing new GUI widgets based on CChartObject class](https://c.mql5.com/2/0/Design_Widgets_MQL5.png)[Designing and implementing new GUI widgets based on CChartObject class](https://www.mql5.com/en/articles/196)

After I wrote a previous article on semi-automatic Expert Advisor with GUI interface it turned out that it would be desirable to enhance interface with some new functionalities for more complex indicators and Expert Advisors. After getting acquainted with MQL5 standard library classes I implemented new widgets. This article describes a process of designing and implementing new MQL5 GUI widgets that can be used in indicators and Expert Advisors. The widgets presented in the article are CChartObjectSpinner, CChartObjectProgressBar and CChartObjectEditTable.

![The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor](https://c.mql5.com/2/0/Trade_Classes_Stdlib_MQL5.png)[The Use of the MQL5 Standard Trade Class libraries in writing an Expert Advisor](https://www.mql5.com/en/articles/138)

This article explains how to use the major functionalities of the MQL5 Standard Library Trade Classes in writing Expert Advisors which implements position closing and modifying, pending order placing and deletion and verifying of Margin before placing a trade. We have also demonstrated how Trade classes can be used to obtain order and deal details.

![Dimitar Manov: "I fear only extraordinary situations in the Championship" (ATC 2010)](https://c.mql5.com/2/0/manov_avatar.png)[Dimitar Manov: "I fear only extraordinary situations in the Championship" (ATC 2010)](https://www.mql5.com/en/articles/536)

In the recent review by Boris Odintsov the Expert Advisor of the Bulgarian Participant Dimitar Manov appeared among the most stable and reliable EAs. We decided to interview this developer and try to find the secret of his success. In this interview Dimitar has told us what situation would be unfavorable for his robot, why he's not using indicators and whether he is expecting to win the competition.

![Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://c.mql5.com/2/0/anufrenko_ava.png)[Alexander Anufrenko: "A danger foreseen is half avoided" (ATC 2010)](https://www.mql5.com/en/articles/535)

The risky development of Alexander Anufrenko (Anufrenko321) had been featured among the top three of the Championship for three weeks. Having suffered a catastrophic Stop Loss last week, his Expert Advisor lost about $60,000, but now once again he is approaching the leaders. In this interview the author of this interesting EA is describing the operating principles and characteristics of his application.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pamlbtxrghuixmhuxqopiyrpnrfqpqvx&ssn=1769251223951769474&ssn_dr=0&ssn_sr=0&fv_date=1769251223&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F174&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Technical%20Analysis%3A%20How%20Do%20We%20Analyze%3F%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17692512238686257&fz_uniq=5083031449419912087&sv=2552)

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