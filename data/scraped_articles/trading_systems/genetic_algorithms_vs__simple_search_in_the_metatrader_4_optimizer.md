---
title: Genetic Algorithms vs. Simple Search in the MetaTrader 4 Optimizer
url: https://www.mql5.com/en/articles/1409
categories: Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:40:02.262036
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=euxpdrlpsivtxzcdankxraadngncicne&ssn=1769092801822968626&ssn_dr=0&ssn_sr=0&fv_date=1769092801&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1409&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Genetic%20Algorithms%20vs.%20Simple%20Search%20in%20the%20MetaTrader%204%20Optimizer%20-%20MQL4%20Articles&scr_res=1920x1080&ac=17690928012682102&fz_uniq=5049289872809765056&sv=2552)

MetaTrader 4 / Tester


### 1\. What Are Genetic Algorithms?

The MetaTrader 4 platform now offers genetic algorithms of optimization the Expert
Advisors' inputs. They reduce optimization time significantly without any significant
invalidation of testing. Their operation principle is described in article named
[Genetic Algorithms: Mathematics](https://www.mql5.com/en/articles/1408) in details.

**This article is devoted to EAs' inputs optimization using genetic algorithms compared**
**to the results obtained using direct, complete search of parameter values.**

### 2\. The Expert Advisor

**For my experiments, I slightly completed the EA named CrossMACD that you may have**
**known from the article named** [Orders Management - It's Simple](https://www.mql5.com/ru/articles/1404):

- Added StopLoss and TakeProfit to the placed positions.
- Added Trailing Stop.
- Used parameter OpenLuft to filter signals: Now signal will come if the zero line
is crossed at a certain amount of points (with the accuracy to one decimal place).

- Added parameter CloseLuft for the similar filtering of close signals.
- Put in expernal variables the periods of the slow and the fast moving averages used
for MACD calculations.

Now it is a practically completed Expert Advisor. It will be convenient to optimize
it and use in trading. You can download EA [**CrossMACD\_DeLuxe.mq4**](https://www.mql5.com/en/articles/download/1409/CrossMACD_DeLuxe.mq4 "Скачать") to your PC and test it independently.

### 3\. Optimization

Now we can start to optimize the EA. **Three tests** will be conducted with different amounts of optimizing searches. This will help
to compare profits obtained using genetic algorithms in various situations.

After each test, I will manually remove the **tester cache** for the subsequent tests not to use combinations already found. This is necessary
only for the experiment to be more precise - normally, automated chaching of results
just enhances the repeated optimization.

To **compare the results**, optimization using genetic algorithms will be made twice: first time - in order to find the maximal profit (Profit),
second time – to find the highest profit factor (Profit Factor). After that, the best three results for both
optimization methods will be given in the summary report table sorted by the given columns.

Optimization is purely experimental. This article is not aimed at finding inputs
that would really make greatest profits.

### Test 1

- chart symbol – **EURUSD**;
- chart timeframe – **Н1**;
- testing period – **2 years**;
- modelling – " **Open prices only**";
- inputs searched in:

|     |     |     |     |
| --- | --- | --- | --- |
| **Variable Name** | **Starting Value** | **Step** | **Final Value** |
| **StopLoss** | 0 | 10 | 100 |
| **TakeProfit** | 0 | 10 | 150 |
| **TrailingStop** | 0 | 10 | 100 |
| **OpenLuft** | 0 | 5 | 50 |
| **CloseLuft** | 0 | 5 | 50 |
| **Number of searches** | **234256** |

It must be noted that, when using genetic algorithms, the **expected time** of optimization is approximately the same as that of optimization using direct
inputs search. The difference is that a genetic algorithm continuously screens
out certainly unsuccessful combinations and, in this way, reduces the amount of
necessary tests several times (perhaps several tens, hundreds, thousands of times).
This is why you should not be geared to the expected optimization time when using
genetic algorithms. The real optimization time will always be shorter:

![](https://c.mql5.com/2/14/time0_1_.gif)

**Direct search**

![](https://c.mql5.com/2/14/time1_1_.gif)

**Genetic algorithm**

As you see, optimization using genetic algorithms took less than four minutes instead
of the expected five and a half hours.

Optimization graph with genetic algorithms also differs from that with direct search.
Since bad combinations have already been screened out, the subsequent tests are
conducted with combinations of inputs that are more profitable by default. This
is why the balance graph goes up:

![](https://c.mql5.com/2/14/testergraph_2_.gif)

Let us consider the results of both optimization methods in all details.

**Results Table:**

|     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | **Direct search** | **Genetic algorithm** |
| **Total optimization time** | **4 h 13 min 28 sec** | **3 min 50 sec** |
|  | **SL** | **TP** | **TS** | **Open Luft** | **Close Luft** | **Profit** | **SL** | **TP** | **TS** | **Open Luft** | **Close Luft** | **Profit** |
| **1** | 70 | 140 | 0 | 20 | 30 | 1248.08 | 70 | 140 | 0 | 20 | 30 | 1248.08 |
| **2** | 70 | 140 | 0 | 20 | 35 | 1220.06 | 70 | 140 | 0 | 20 | 35 | 1220.06 |
| **3** | 70 | 150 | 0 | 20 | 30 | 1176.54 | 70 | 150 | 0 | 20 | 30 | 1176.54 |
|  | **SL** | **TP** | **TS** | **Open Luft** | **Close Luft** | **Profit Factor** | **SL** | **TP** | **TS** | **Open Luft** | **Close Luft** | **Profit Factor** |
| **1** | 100 | 50 | 40 | 50 | 5 | 4.72 | 0 | 50 | 40 | 50 | 5 | 4.72 |
| **2** | 90 | 50 | 40 | 50 | 5 | 4.72 | 90 | 50 | 40 | 50 | 5 | 4.72 |
| **3** | 80 | 50 | 40 | 50 | 5 | 4.72 | 80 | 50 | 40 | 50 | 0 | 4.72 |

As you can see from the table, optimization using genetic algorithms is **some tens of times faster!** The results are practically the same. There are several results with maximal profit
of 4.72, this is why different combinations of inputs are reported, but it is not
very important.

Now let's try to decrease the amount of searches, but increase the testing time.
We will use the "All ticks" model for this.

### Test 2

- chart symbol – **EURUSD**;
- chart timeframe – **Н1**;
- testing period – **2 years**;
- modelling – " **All ticks**";
- inputs searched in:


|     |     |     |     |
| --- | --- | --- | --- |
| **Variable Name** | **Start Value** | **Step** | **End Value** |
| **StopLoss** | 0 | 10 | 100 |
| **TakeProfit** | 0 | 10 | 150 |
| **TrailingStop** | 0 | 10 | 100 |
| **OpenLuft** | 0 | 10 | 50 |
| **Number of searches** | **11 616** |

**Results table:**

|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | **Direct search** | **Genetic algorithm** |
| **Total optimization time** | **32 h 32 min 37 sec** | **1 h 18 min 51 sec** |
|  | **SL** | **TP** | **TS** | **Open Luft** | **Profit** | **SL** | **TP** | **TS** | **Open Luft** | **Profit** |
| **1** | 50 | 0 | 0 | 20 | 1137.89 | 50 | 0 | 0 | 20 | 1137.89 |
| **2** | 70 | 0 | 0 | 20 | 1097.87 | 70 | 0 | 0 | 20 | 1097.87 |
| **3** | 60 | 0 | 0 | 20 | 1019.95 | 60 | 0 | 0 | 20 | 1019.95 |
|  | **SL** | **TP** | **TS** | **Open Luft** | **Profit Factor** | **SL** | **TP** | **TS** | **Open Luft** | **Profit Factor** |
| **1** | 50 | 90 | 60 | 50 | 4.65 | 50 | 90 | 60 | 50 | 4.65 |
| **2** | 50 | 140 | 60 | 50 | 4.59 | 50 | 140 | 60 | 50 | 4.59 |
| **3** | 100 | 90 | 60 | 50 | 4.46 | 70 | 90 | 60 | 50 | 4.46 |

For such an amount of searches, the optimization rate differs **25 times** which is not bad either. The results conincide by practically 100%, the only difference
is in the StopLoss value on the third pass. The profit factor remains maximal.

Now let's try to increase the amount of searches and descrease the testing time.
Let us use the "Control points" model for this.

### Test 3

- chart symbol – **EURUSD**;
- chart timeframe – **Н1**;
- testing period – **2 years**;
- modelling – " **Control points**";
- inputs searched in:


|     |     |     |     |
| --- | --- | --- | --- |
| **Variable Name** | **Start Value** | **Step** | **Final Value** |
| **StopLoss** | 0 | 10 | 100 |
| **OpenLuft** | 0 | 5 | 50 |
| **CloseLuft** | 0 | 5 | 50 |
| **Number of searches** | **1 331** |

**Results table:**

|     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | **Direct search** | **Genetic algorithm** |
| **Total optimization time** | **33 min 25 sec** | **31 min 55 sec** |
|  | **SL** | **Open Luft** | **Close Luft** | **Profit** | **SL** | **Open Luft** | **Close Luft** | **Profit** |
| **1** | 0 | 0 | 45 | 1078.03 | 0 | 0 | 45 | 1078.03 |
| **2** | 70 | 20 | 15 | 1063.94 | 70 | 20 | 15 | 1063.94 |
| **3** | 70 | 20 | 25 | 1020.19 | 70 | 20 | 25 | 1020.19 |
|  | **SL** | **Open Luft** | **Close Luft** | **Profit Factor** | **SL** | **Open Luft** | **Close Luft** | **Profit Factor** |
| **1** | 80 | 50 | 15 | 2.73 | 80 | 50 | 15 | 2.73 |
| **2** | 70 | 50 | 15 | 2.73 | 70 | 50 | 15 | 2.73 |
| **3** | 90 | 50 | 15 | 2.65 | 90 | 50 | 15 | 2.65 |

The situation has changed. The optimization periods coincide (an insignificant error
is admissible), and the results are identical. This can be explained through that
optimization consisted of only 1331 searches and this amount of passes is just
not enough for using genetic algorithms. they have no time to "pick up speed"
\- the optimization is faster due to screening out certainly losing inputs combinations,
but having such amount of combinations as above, genetic algorithms cannot define
what "parents" (inputs combinations) generate bad "off-spring".
So, there is no sense to use them.

### 4\. Conclusions

Genetic algorithms are a nice addition to the МТ 4 strategies optimizer. Optimization is dramatically enhanced
if the amount of searches is large, the results coincide with those obtained by regular optimization.

Now there is no sense to use the full search in inputs. Genetic algorithms will
find the best result faster and no less effectively.

### 5\. Afterword

After having written the article, I satisfied my curiosity and launched optimization
of [**CrossMACD\_DeLuxe**](https://www.mql5.com/en/articles/download/1409/CrossMACD_DeLuxe.mq4 "Скачать") on all inputs. The amount of combinations made over one hundred million (103 306 896). The optimization using genetic algorithms took only 17 hours, while optimization using search in all inputs would take approximately 35 years (301 223 hours).

Conclusions are up to you.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1409](https://www.mql5.com/ru/articles/1409)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1409.zip "Download all attachments in the single ZIP archive")

[CrossMACD\_DeLuxe.mq4](https://www.mql5.com/en/articles/download/1409/CrossMACD_DeLuxe.mq4 "Download CrossMACD_DeLuxe.mq4")(6.75 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [How to Order an Expert Advisor and Obtain the Desired Result](https://www.mql5.com/en/articles/235)
- [Equivolume Charting Revisited](https://www.mql5.com/en/articles/1504)
- [Testing Visualization: Account State Charts](https://www.mql5.com/en/articles/1487)
- [An Expert Advisor Made to Order. Manual for a Trader](https://www.mql5.com/en/articles/1460)
- [Testing Visualization: Trade History](https://www.mql5.com/en/articles/1452)
- [Sound Alerts in Indicators](https://www.mql5.com/en/articles/1448)
- [Filtering by History](https://www.mql5.com/en/articles/1441)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39299)**
(2)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
8 Sep 2007 at 22:25

I have been looking for ways to optimize optimization, and this article intrigues
me. How exactly would I go about enabling "genetic optimization?"

Thanks

Nick

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
27 Sep 2007 at 16:29

Thanks for the great explanation and thorough testing Andrey! Very helpful.

Nick, there is a checkbox in the [Strategy Tester](https://www.mql5.com/en/articles/239 "Article \"The Fundamentals of Testing in MetaTrader 5\"") when you click "Expert properties"
\-\- it is on the first tab "Testing"


![A Method of Drawing the Support/Resistance Levels](https://c.mql5.com/2/14/233_1.png)[A Method of Drawing the Support/Resistance Levels](https://www.mql5.com/en/articles/1439)

This article describes the process of creating a simple script for detecting the support/resistance levels. It is written for beginners, so you can find the detailed explanation of every stage of the process. However, though the script is very simple, the article will be also useful for advanced traders and the users of the MetaTrader 4 platform. It contains the examples of the data export into the tabular format, the import of the table to Microsoft Excel and plotting the charts for the further detailed analysis.

![Pivot Points Helping to Define Market Trends](https://c.mql5.com/2/14/333_1.png)[Pivot Points Helping to Define Market Trends](https://www.mql5.com/en/articles/1466)

Pivot point is a line in the price chart that shows the further trend of a currency pair. If the price is above this line, it tends to grow. If the price is below this line, accordingly, it tends to fall.

![Displaying of Support/Resistance Levels](https://c.mql5.com/2/14/237_1.png)[Displaying of Support/Resistance Levels](https://www.mql5.com/en/articles/1440)

The article deals with detecting and indicating Support/Resistance Levels in the MetaTrader 4 program. The convenient and universal indicator is based on a simple algorithm. The article also tackles such a useful topic as creation of a simple indicator that can display results from different timeframes in one workspace.

![Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://c.mql5.com/2/13/129_2.gif)[Expert System 'Commentator'. Practical Use of Embedded Indicators in an MQL4 Program](https://www.mql5.com/en/articles/1406)

The article describes the use of technical indicators in programming on MQL4.

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/1409&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049289872809765056)

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