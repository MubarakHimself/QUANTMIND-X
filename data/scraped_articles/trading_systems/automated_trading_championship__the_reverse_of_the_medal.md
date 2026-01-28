---
title: Automated Trading Championship: The Reverse of the Medal
url: https://www.mql5.com/en/articles/1541
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:02:34.697276
---

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1541&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071529449691752889)

MetaTrader 4 / Trading systems


Automated Trading Championship based on online trading platform MetaTrader 4 is being conducted for the third time and accepted by many people as a matter-of-course yearly event being waited for with impatience. Thanks in large part to the incidence of the Championship where hundreds of Expert Advisors (trading robots in MQL4) compete with each other, and to its permanent online coverage, automated trading rose to a much higher level. However, larger scale of the contest places more serious demands on the Participants. This is precisely the topic we're going to discuss in this article.

![Automated Trading Championship 2007](https://c.mql5.com/2/16/atc07_chart_1.png)

Final diagram of the Participants of Automated Trading Championship 2007.

The purpose of the [Automated Trading Championship](https://championship.mql5.com/), as we alleged repeatedly before, is to popularize automated trading and MQL4 as the most suitable programming language for coding trading robots. Previous Championships aroused large interests of traders worldwide, and specialized journals gave coverage to it.

It seems unthinkable that nobody before MetaQuotes Software Corp. thought of conducting such a Championship welcomed by all traders. In fact, organizing such an event needs more than just a desire to do it. The necessary amount of money is not a sufficient condition for conducting such a competition, though. Technicality comes to the fore here.

Many people have already got used to that they can launch in one PC a number of copies of MetaTrader 4 Client Terminal installed in different directories. Since not traders themselves participate in the Championship, but their programs written in [MQL4](https://docs.mql4.com/), the first question now arises of how to start some hundreds of EAs, each provided with its individual copy of the client terminal.

Despite the smallest size of the distributive (3.5 Mb), there are natural limitations determined by the power of the operating system itself. A lite version of the terminal was written especially for the Championship. Everything unnecessary for an EA trading independently within a number of months was removed in this version. This allows us to launch 40 copies of MetaTrader 4 Client Terminal on one server. Do you know any other trading platform that would allow doing this?

However, simultaneous work of dozens of EAs on one server sets certain requirements to the participating Expert Advisors themselves. In the past, we had some examples of incorrectly coded EAs creating giant log files of some gigabytes within one day and taking all free disk space of the server.

In some cases, these were terminal logs containing order opening error messages. In other cases, they were technical messages of the EAs themselves writing some reports at each arriving tick. The work of these EAs would result in inability of other Participants' EAs to work properly, this is why the incorrectly coded Expert Advisors were unconditionally excluded from the competition.

Except the disk space shared, another important resource is CPU. In recent years, modern processor speeds have increased manifold, but no supercomputer, even the most modern one, can work efficiently with a badly coded program algorithm.

A correctly optimized algorithm works by an order faster than that written with errors. In 2005, when the new online trading [platform MetaTrader 4](https://www.metaquotes.net/en/metatrader5) had been officially released, we gave the results of tests measuring the speed and the accuracy of calculations in the new programming language, MQL4, as compared to other languages.

MQL4 was only behind special programming languages, but got to windward of the languages of other trading platforms. However, even with this high speed of calculations, it would be unfair to consume most CPU resources at the expense of other Participants' Expert Advisors on the same server.

This is why all EAs that consume CPU resources uneconomically were and will be excluded from the contest. At the stage of Registration, we are going to detect such incorrectly written Expert Advisors providing certain test time limitations in the eight-month interval in the "Every tick" mode. If the EA's testing time exceeds 5 minutes, this EA will not be allowed to participate in the contest.

When preparing the Championship, we immediately emphasized the online coverage of the contest. A task was set to show the state of each competing account in the real-time mode and to display open positions, current balance and equity, as well as many other statistical parameters. For this purpose, we created a special website that was developed and is being supported by a team of programmers, developers of web applications.

Every site visitor could both view the information of his or her interest and leave a message in the thread of each Participant. We were doing our best to both provide "dry" information and publish interviews with Participants and some survey analytical materials at each stage of the contest.

![Hardware for the Automated Trading Championship 2006](https://c.mql5.com/2/16/server_automated_trading.jpg)

Especially for the purposes of the very first Championship, [Automated Trading Championship 2006](https://championship.mql5.com/), [a server priced at US$ 35 000](https://championship.mql5.com/)
was purchased, but that was not the greatest investment. The main expenses were working hours spent on creating new technologies for adapting the trade server data to the web functions of the Championship's website. We expected a heavy load on the contest website, but it still turned out to exceed the planned load. Organizing the first Championship produced much useful experience for creating online technologies of the kind.

The site of the Championship was visited by about 6 500 unique visitors per day, each visitor having viewed 30 pages at the average. Besides displaying the current state of the account, we also showed the current history of closed trades and the terminal logs of each Participant. All this is a considerable amount of information!

Not everything always clicks into place. There were [errors on our side](https://championship.mql5.com/), [equipment failures](https://championship.mql5.com/), [non-market spikes during quoting](https://championship.mql5.com/).
It is impossible to hedge against all risks, but we did and are going to do our best to conduct the Championship 2008 at the highest possible technological level.

It will be recalled that over 600 prize contenders were registered at Automated Trading Championship 2006, 258 of which were admitted to participation. As to [ATC 2007](https://championship.mql5.com/), there were over 2000 registered contenders, 603 of which became Participants after detailed checks. We cannot predict the amount of registrations for Automated Trading Championship 2008, but we will do our best to protect Olympic principles of equal opportunities and of fair play.

Only one thing is required from you as from the participant of the forthcoming contest - creation of a correctly coded Expert Advisor and non-infringement of the Rules. And may the battle be to the strong!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1541](https://www.mql5.com/ru/articles/1541)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39473)**
(5)


![Hongbin Gu](https://c.mql5.com/avatar/avatar_na2.png)

**[Hongbin Gu](https://www.mql5.com/en/users/valleyfir)**
\|
3 Jul 2008 at 02:56

Referring to what is said in the article: "If the EA's testing time exceeds 5 minutes, this EA will not be allowed to participate int the contest." I wonder the testing time of not exceeding 5 minutes, is the testing implemented on your server or an ordinary PC just like what we use our MT4 terminal client software on? Thanks.

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
3 Jul 2008 at 12:43

It's no matter. Preliminary back testing will run only for one EA at moment.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
22 Jul 2008 at 07:12

**valleyfir:**

Referring to what is said in the article: "If the EA's testing time exceeds 5 minutes, this EA will not be allowed to participate int the contest." I wonder the testing time of not exceeding 5 minutes, is the testing implemented on your server or an ordinary PC just like what we use our MT4 terminal client software on? Thanks.

This is probably just a [check](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") to make sure you had some idea of what you were doing. Poorly written code is written by inexperienced developers who probably aren't going to finish in the top ten anyway. I'm sure the 5 minute rule is far above any realistic use of the processor. They probably just don't want some infinite loop crap running on during the competition.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
11 Apr 2011 at 17:24

If you want to increase your forex trading, and profitability in the market, then this article will definitely help you. This article is very constructive for discouraged forex traders. This article is very helpful if you want to earn profit in less than half an hour with almost no effort. This will be very helpful for expert as well as new forex traders. I assure you that you will not find such a useful product any where else.

\-\-\---------------

\[url=http://www.best4xrobots.com/\]Free [forex robot](https://www.mql5.com/en/market "A Market of Applications for the MetaTrader 5 and MetaTrader 4")\[/url\]

\[url=http://www.best4xrobots.com/\]Forex expert advisor\[/url\]

![Alamdar](https://c.mql5.com/avatar/2020/8/5F427C19-8AD7.jpg)

**[Alamdar](https://www.mql5.com/en/users/mohsenalamdar)**
\|
19 Dec 2022 at 01:17

why it is stopped ???


![Market Diagnostics by Pulse](https://c.mql5.com/2/15/589_8.gif)[Market Diagnostics by Pulse](https://www.mql5.com/en/articles/1522)

In the article, an attempt is made to visualize the intensity of specific markets and of their time segments, to detect their regularities and behavior patterns.

![Two-Stage Modification of Opened Positions](https://c.mql5.com/2/16/612_6.gif)[Two-Stage Modification of Opened Positions](https://www.mql5.com/en/articles/1529)

The two-stage approach allows you to avoid the unnecessary closing and re-opening of positions in situations close to the trend and in cases of possible occurrence of divirgence.

![The Statistic Analysis of Market Movements and Their Prognoses](https://c.mql5.com/2/16/634_10.jpg)[The Statistic Analysis of Market Movements and Their Prognoses](https://www.mql5.com/en/articles/1536)

The present article contemplates the wide opportunities of the statistic approach to marketing. Unfortunately, beginner traders deliberately fail to apply the really mighty science of statistics. Meanwhile, it is the only thing they use subconsciously while analyzing the market. Besides, statistics can give answers to many questions.

![A Pattern Trailing Stop and Exit the Market](https://c.mql5.com/2/15/605_15.gif)[A Pattern Trailing Stop and Exit the Market](https://www.mql5.com/en/articles/1527)

Developers of order modification/closing algorithms suffer from an imperishable woe - how to compare results obtained by different methods? The mechanism of checking is well known - it is Strategy Tester. But how to make an EA to work equally for opening/closing orders? The article describes a tool that provides strong repetition of order openings that allows us to maintain a mathematically correct platform to compare the results of different algorithms for trailing stops and for exiting the market.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/1541&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071529449691752889)

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