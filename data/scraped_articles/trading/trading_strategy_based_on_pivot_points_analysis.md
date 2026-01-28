---
title: Trading Strategy Based on Pivot Points Analysis
url: https://www.mql5.com/en/articles/1465
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:24:06.380420
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/1465&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069407237631313044)

MetaTrader 4 / Trading


### Introduction

Pivot Points (PP) analysis is one of the simplest and most effective strategies
for high intraday volatility markets. It was used as early as in the precomputer
times, when traders working at stocks could not use any ADP equipment, except for
counting frames and arithmometers. Analysis of this kind can often be found in
a number of articles on technical analysis in the sections devoted to excursions
into history. The main advantage of this technique is its computational efficiency
that allows traders to make calculations mentally or on a sheet of paper.

Since four arithmetic operations are used in calculations, every trader using this
technique always wanted either outrun the competitors or, at least, "outcalculate"
them. Correspondingly, there are many formulas to calculate pivot points and support/resistance
levels (see examples in the table below).

|     |     |
| --- | --- |
| Range | Possible Formulas to Calculate PP |
| RANGE: High - Low<br>RANGE %: (High - Low) / (Previous Close) | PP1=(H+L+C)/3<br>PP2=(H+L+O)/3<br>PP3=(H+L+C+O)/4<br>PP4=(H+L+C+C)/4<br>PP5=(H+L+O+O)/4<br>PP6=(H+L)/2<br>PP7=(H+C)/2<br>PP8=(L+C)/2 |
| Change |
| Change: Close - Previous Close<br>Change %: (Close - Previous Close) / (Previous Close) |
| Trend % |
| Calculation: ABS (CLOSE - OPEN) / RANGE |

|     |     |
| --- | --- |
| Classic Formula | Woodie Pivot Points |
| R4 = R3 + RANGE (same as: PP + RANGE \* 3)<br>R3 = R2 + RANGE (same as: PP + RANGE \* 2)<br>R2 = PP + RANGE<br>R1 = (2 \* PP) - LOW<br>PP = (HIGH + LOW + CLOSE) / 3<br>S1 = (2 \* PP) - HIGH<br>S2 = PP - RANGE<br>S3 = S2 - RANGE (same as: PP - RANGE \* 2)<br>S4 = S3 - RANGE (same as: PP - RANGE \* 3) | R4 = R3 + RANGE<br>R3 = H + 2 \* (PP - L) (same as: R1 + RANGE)<br>R2 = PP + RANGE<br>R1 = (2 \* PP) - LOW<br>PP = (HIGH + LOW + CLOSE) / 3<br>S1 = (2 \* PP) - HIGH<br>S2 = PP - RANGE<br>S3 = L - 2 \* (H - PP) (same as: S1 - RANGE)<br>S4 = S3 - RANGE |
| Camarilla Pivot Points | Tom DeMark "Pivot Points" |
| R4 = C + RANGE \* 1.1/2<br>R3 = C + RANGE \* 1.1/4<br>R2 = C + RANGE \* 1.1/6<br>R1 = C + RANGE \* 1.1/12<br>PP = (HIGH + LOW + CLOSE) / 3<br>S1 = C - RANGE \* 1.1/12<br>S2 = C - RANGE \* 1.1/6<br>S3 = C - RANGE \* 1.1/4<br>S4 = C - RANGE \* 1.1/2 | R1 = X / 2 - L<br>PP = X / 4 (this is not an official DeMark number but merely a reference point based<br>on the calculation of X)<br>S1 = X / 2 - H |
| Condition if Open after Close |
| if C < O then X = (H + (L \* 2) + C) |
| if C > O then X = ((H \* 2) + L + C) |
| if C = 0 then X = (H + L + (C \* 2)) |

### Problems and Disappointments

In the world of probability Forex also relates to, finding of a Pivot Point with
unique computing result is something like an oasis in a desert. This unambiguity
and simplicity of arithmetics attract novice traders.

However, this notorious unambiguity is the result of arithmetic operations and has
no relation to Forex. The duality of this situation irritates, in case the results
of calculations performed by traders using data from different DataCenters are
different. The differences between the results and the forecasts of analyst Rudolph
Axel, the acknowledged leader in Pivot techniques application, are even more irritating.
Let us try to separate the husk from the grain.

### Not All Roses

To generate a pivot point and support/resistance levels for a future period of time,
Pivot Points Analysis uses the minimal amount of inputs: High, Low and Close of
the preceding tick period. Initially, such period was a trading session.

Far back in the past, when the main rules of Pivot and support/resistance levels
were developed, a "trading session" and a "trading day" were
perhaps the same. At present, the trading day time in forex consists of three main
trading sessions, so attempts to use the rules of the Pivot Points Analysis without
taking these change into consideration are not quite correct. Time is the parameter
that is present in trading, but not shown in calculating formulas. _In our topic under consideration, it determines High, Low and Close of the period_
_used in calculations. This is the first "thorn" in the idea._

### Another "Thorn"

It's the terminal's internal time. Instead of being the same (GMT) in all the terminal,
it is different in different Data Centers. This results in an interesting effect:
the time within which a candlestick is formed is the same for only timeframes smaller
than H1, then there are divergences observed. So the analysis, or its reliability
and unambiguity on charts of different Data Centers, is open to question.

_To exclude the situation when the internal terminal time influences calculations,_
_it is necessary to use one-hour candlesticks corrected for the difference between_
_the terminal time and GMT._

This thorn can be eliminated using indicator DailyPivot\_Shift ( [https://www.mql5.com/ru/code/8864](https://www.mql5.com/ru/code/8864)). Indicator DailyPivot\_Shift differs from the normal indicator DailyPivot through
that the basic levels can be calculated with a shift in relation to the beginning
of the day. Thus, the levels can be calculated on the basis of local time, not
the server time, for example, GMT. As well, the indicator does not consider information
about weekend's quotes when it builds charts on Mondays.

### The Third "Thorn"

We want to use one-hour candlesticks, but we can only get them in the manual mode,
separately for each currency pair. The author does not mean an advanced programmer,
but, for example, a physician or an economist.

_This means that the time needed for analyzing will be wasted on nonproductive manual_
_operations._

### On Computational Accuracy

The table below gives absolute values of Pivot levels at different values of Close,
and the absolute value of deviations in points.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Calculation of Pivot levels at different values of Close |
|  | -30 | -10 | 0 | 10 | 30 |
|  | GBPUSD | GBPUSD | GBPUSD | GBPUSD | GBPUSD |
| R3 | 1,8566 | 1,8580 | 1,8586 | 1,8593 | 1,8606 |
| R2 | 1,8524 | 1,8530 | 1,8534 | 1,8537 | 1,8544 |
| R1 | 1,8450 | 1,8464 | 1,8470 | 1,8477 | 1,8490 |
| Pivot | 1,8408 | 1,8414 | 1,8418 | 1,8421 | 1,8428 |
| S1 | 1,8334 | 1,8348 | 1,8354 | 1,8361 | 1,8374 |
| S2 | 1,8292 | 1,8298 | 1,8302 | 1,8305 | 1,8312 |
| S3 | 1,8218 | 1,8232 | 1,8238 | 1,8245 | 1,8258 |
| H | 1,8481 | 1,8481 | 1,8481 | 1,8481 | 1,8481 |
| L | 1,8365 | 1,8365 | 1,8365 | 1,8365 | 1,8365 |
| C | 1,8377 | 1,8397 | 1,8407 | 1,8417 | 1,8437 |
|  | -30 | -10 | 0 | 10 | 30 |

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| deviations of the mean value, in points |
| \* | GBPUSD | GBPUSD | GBPUSD | GBPUSD | GBPUSD |
| R3 | -20 | -6,7 | 1,8586 | 6,7 | 20 |
| R2 | -10 | 3,3 | 1,8534 | 3,3 | 10 |
| R1 | -20 | 6,7 | 1,8470 | 6,7 | 20 |
| Pivot | -10 | 3,3 | 1,8418 | 3,3 | 10 |
| S1 | -20 | 6,7 | 1,8354 | 6,7 | 20 |
| S2 | -10 | 3,3 | 1,8302 | 3,3 | 10 |
| S3 | -20 | 6,7 | 1,8238 | 6,7 | 20 |
| H | 1,8481 | 1,8481 | 1,8481 | 1,8481 | 1,8481 |
| L | 1,8365 | 1,8365 | 1,8365 | 1,8365 | 1,8365 |
| C | 1,8377 | 1,8397 | 1,8407 | 1,8417 | 1,8437 |

Deviation of the period close price (or the summarized deviation of H+L+C) by 30
points results in the error of 10 points.

### Quick Calculation

The classical formula is as follows: PP = (HIGH + LOW + CLOSE) / 3

A variation looks like this: PP = (H + L) / 2

Suppose H = 1.9100, L = 1.9000, Range = 100. Then, by definition, "Close" must be within the range of
1.9000 – 1.9100.

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| High | Low | Close | (H+L+C)/3 | (H+L)/2 | /3 -'/2 |
| 1.9100 | 1.9000 | 1.9000 | 1.9033 | 1.9050 | -17 |
| 1.9100 | 1.9000 | 1.9010 | 1.9037 | 1.9050 | -13 |
| 1.9100 | 1.9000 | 1.9020 | 1.9040 | 1.9050 | -10 |
| 1.9100 | 1.9000 | 1.9030 | 1.9043 | 1.9050 | -7 |
| 1.9100 | 1.9000 | 1.9040 | 1.9047 | 1.9050 | -3 |
| 1.9100 | 1.9000 | 1.9050 | 1.9050 | 1.9050 | 0 |
| 1.9100 | 1.9000 | 1.9060 | 1.9053 | 1.9050 | 3 |
| 1.9100 | 1.9000 | 1.9070 | 1.9057 | 1.9050 | 7 |
| 1.9100 | 1.9000 | 1.9080 | 1.9060 | 1.9050 | 10 |
| 1.9100 | 1.9000 | 1.9090 | 1.9063 | 1.9050 | 13 |
| 1.9100 | 1.9000 | 1.9100 | 1.9067 | 1.9050 | 17 |

We can see that the deviation of the close price from (H+L)/2 up to 30 points results
in an error within 10 points. It means that if the movement has not started and
the price does not break High and Low levels, stays somewhere in the middle of
the range, we get a PP using Andrews' Pitchfork directly in the chart, whereas
deviations are within 10 points from Axel's data. Moreover, I didn't do it myself
due to absence Axel's archives. Something may be done through searching in Axel's
Forecast and (H+L+C)/3, (H+L)/2 (on the preceding session).

### Support/Resistance Levels

The formulas are given above. It is necessary to avoid the erroneous assumption
about comprehension unambiguity of calculation results, such as R3 = 1. 9356, nothing
more or less than that, and accept the following calculation order. The support/resistance
level calculation result is accurate up to the nearest support/resistance level
of the real history in the chart. This is what Rudolph Axel demonstrates us, actually.

Example: "The intraday EURJPY: The pair is traded near the minor resistance
of 158. 38 (maximum of the 9th of February). If this level is broken through, the
pair will aim at 158.76 (maximum of the 14th of February). The minor support is
located near 157.78 (minimum of Wednesday) and at 157.28."

### Conclusion

I haven't found any theory that would prove or disprove the Pivot Point or Support/Resistant
Levels calculation results. We will use the generalized, accepted by practically
all scientists rules: "Practice is the sole criterion of truth". Experienced
traders in all around the world consider these reference values to be rather close
to the truth with rather high statistical advantage over coin tossing.

One may dispute the usability, correctness and accuracy of these calculations as
long and well-reasoned as one like. However, simple formulas and monotonicity of
their usage allow a trader of any level to gain experiences and find legs. "Begin
at the beginning."

P.S. The article was prepared by Moderator of forum [www.forum.profiforex.ru](https://www.mql5.com/go?link=http://www.forum.profiforex.ru/), Vladimir aka dedd.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1465](https://www.mql5.com/ru/articles/1465)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Pivot Points Helping to Define Market Trends](https://www.mql5.com/en/articles/1466)

**[Go to discussion](https://www.mql5.com/en/forum/39384)**

![Object Approach in MQL](https://c.mql5.com/2/15/499_6.gif)[Object Approach in MQL](https://www.mql5.com/en/articles/1499)

This article will be interesting first of all for programmers both beginners and professionals working in MQL environment. Also it would be useful if this article were read by MQL environment developers and ideologists, because questions that are analyzed here may become projects for future implementation of MetaTrader and MQL.

![Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://c.mql5.com/2/14/367_27.png)[Automated Choice of Brokerage Company for an Efficient Operation of Expert Advisors](https://www.mql5.com/en/articles/1476)

It is not a secret that for an efficient operation of Expert Advisors we need to find a suitable brokerage company. This article describes a system approach to this search. You will get acquainted with the process of creating a program with dll for working with different terminals.

![Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://c.mql5.com/2/14/315_5.png)[Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://www.mql5.com/en/articles/1458)

The article describes the way of connecting to the remote PC with installed MT4 Client Terminal via a PDA.

![MT4TerminalSync - System for the Synchronization of MetaTrader 4 Terminals](https://c.mql5.com/2/14/418_30.png)[MT4TerminalSync - System for the Synchronization of MetaTrader 4 Terminals](https://www.mql5.com/en/articles/1488)

This article is devoted to the topic "Widening possibilities of MQL4 programs by using functions of operating systems and other means of program development". The article describes an example of a program system that implements the task of the synchronization of several terminal copies based on a single source template.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lnpwcwayfrrgywmcmihuygkitebeuprn&ssn=1769181844781942855&ssn_dr=0&ssn_sr=0&fv_date=1769181844&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1465&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Trading%20Strategy%20Based%20on%20Pivot%20Points%20Analysis%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176918184494920497&fz_uniq=5069407237631313044&sv=2552)

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