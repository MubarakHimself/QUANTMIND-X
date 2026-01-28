---
title: Dealing with Time (Part 1): The Basics
url: https://www.mql5.com/en/articles/9926
categories: Trading, Indicators, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:32:51.838669
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=kgeoefwzuimimzxttuswqylywjamsppo&ssn=1769250769387973898&ssn_dr=0&ssn_sr=0&fv_date=1769250769&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9926&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Dealing%20with%20Time%20(Part%201)%3A%20The%20Basics%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925076992517236&fz_uniq=5082941465560093137&sv=2552)

MetaTrader 5 / Tester


### Which Time

Accurate timing may be a crucial element in trading. At the current hour, is the stock exchange in London or New York already open or not yet open, when does the trading time for Forex trading start and end? For a trader who trades manually and live, this is not a big problem. Through various internet tools, the specifications of the financial instruments, one's own time, one can quickly see when is the right time to trade for one's strategy. For some traders who only look at the development of the prices, buy and sell whenever the price development provides him with a signal, the time is rather unimportant. It is different for the traders who do 'night scalping', who trade after the (stock) market close in New York and before (stock) markets in the EU open, or those who trade specifically, for example, the 'London breakout', or otherwise during the time with the highest turnover, or all those who trade stocks or futures. These traders cannot follow the broker's times, but must use the actual local times in the USA, Japan, London, EU or Moscow at which the respective exchange opens or closes or the special times at which a special future is traded.

### Summer Time, Winter Time, and Broker Offset

As said, traders who sit in front of the screen to buy and sell can easily handle the different times. Be it via the internet or certain functions, values or clocks that the PC keeps available and that can be retrieved at any time with MQL functions like _TimeGMT()_, _TimeGMTOffset()_ and others. The situation is different if one wants to write and test a trading program with historical data that trades time-related. Actually, analogous to live trading, the question should be just as easy to answer.

Starting from UTC (Universal Coordinated Time) or Greenich Mean Time (GMT, [https://greenwichmeantime.com/what-is-gmt/](https://www.mql5.com/go?link=https://greenwichmeantime.com/what-is-gmt/ "https://greenwichmeantime.com/what-is-gmt/")), one would simply have to add the geographic or instrument-related time shift and one would know what time it is where. But it is not that simple. There is the changeover of winter and summer time (DST or Daylight Saving Time), which, depending on the time, adds or subtracts 1 hour. However, not uniform, but each country or region such as Europe has its own definition, sometimes constant for years (EU and USA) or changed every now and then as in Russia, where it was abolished in 2014. In the EU, there was a discussion in 2018, including a survey on the question of abolishing the annual time change ( [https://ec.europa.eu/germany/news/20180831-konsultation-sommerzeit\_de](https://germany.representation.ec.europa.eu/index_de "https://ec.europa.eu/germany/news/20180831-konsultation-sommerzeit_de")), which resulted in a majority in favor of abolition, so that the Commission submitted a legislative proposal for the end of the time change ( [https://ec.europa.eu/germany/news/20180914-kommission-gesetzesvorschlag-ende-zeitumstellung\_de](https://germany.representation.ec.europa.eu/index_de "https://ec.europa.eu/germany/news/20180914-kommission-gesetzesvorschlag-ende-zeitumstellung_de")), which then came to nothing again.

As if that's not chaotic enough, now brokers are adding to the mix how they define their respective server time. In 2015, a German broker wrote to me:

1. Until March8, 2015 our MT4 server time is set to London Time ( = GMT ).
2. March 8 2015, until March 29, it is set to CET ( GMT + 1 )
3. As of March 29 the server is set to London Time ( = GMT + 1 )


That is, the German broker uses GMT + DST-USA, or (mainly) London time + DST of the USA, you have to come up with that first, although the reasons are understandable, because forex trading starts at 17:00 New York time, and that is subject to the American time change. As a result, for someone based in Frankfurt, forex trading sometimes starts at 21:00, usually at 22:00 and for one/two weeks in the fall at 23:00.

For us as traders and clients, this seems more like being in a kindergarten. Everyone does what and how he wants. The trader or the developer is therefore confronted with many different times, all of which could be important, and all of which would have to be determined from the 'on-board resources' of MQL - but unfortunately not if an indicator, script, or an EA is running in the strategy tester.

We will change that. Without having to ask the broker, we will determine its respective time shifts, so that GMT can be determined quite simply at any moment for any time stamp in the strategy tester and subsequently, of course, any further required local time like New York time. In addition, because it just came up in this context, also the remaining time that forex market is still open, for all traders who want to close their open positions before the weekend, if necessary. However, we will come to these functions only in the second article, because now we will develop some macro substitutions, which will simplify for us for the mentioned functions the calculations and representations for control.

### Macrosubstitutions

Everything coming is in the (attached) include file _DealingWithTimePart1.mqh_. At the beginning of this file there are the various macro substitutions.

First of all, there are some time shifts of different regions, so that you don't have to derive or search for them again and again:

```
//--- defines
#define TokyoShift   -32400                           // always 9h
#define NYShift      18000                            // winter 17h=22h GMT: NYTime + NYshift = GMT
#define LondonShift  0                                // winter London offset
#define SidneyShift  -39600                           // winter Sidney offset
#define FfmShift     -3600                            // winter Frankfurt offset
#define MoskwaShift  -10800                           // winter Moscow offset
#define FxOPEN       61200                            // = NY 17:00 = 17*3600
#define FxCLOSE      61200                            // = NY 17:00 = 17*3600
#define WeekInSec    604800                           // 60sec*60min*24h*7d = 604800 => 1 Week
```

For a better understanding, we agree that GMT is a non-variable time anchor, while the different local times, New York, London, or Moscow, each differ from it by a different (variable) time offset. Likewise, we will also refer to the shifts due to summer or winter time as variable. The reason is simple. The signs of the variable time differences are defined in such a way that applies:

                   variable time + variable time shift = GMT

#### Some examples:

                               Moscow (16h) + Moscow offset (-3h)                                = GMT (13h)

                    New York time (16h) + New York offset (+5h)                             = GMT (21h)

                    New York time (16h) + (New York offset (+5h) + DST\_US(-1h))  = GMT (20h)

                    Frankfurt time (16h) + (Frankfurt offset (-1h) + DST\_US(-1h))  = GMT (14h)

Mind the parenthesis! They become important when the variable times are to be calculated from GMT:

                   New York time (16h) =  GMT (20h) - (New York offset (+5h) + DST\_US(-1h)) => 20 -(+5 + -1) = 20 - 5 +1 = 16h

                    Frankfurt time (16h) =  GMT (14h) - (Frankfurt offset (-1h) + DST\_US(-1h)) => 14 -( -1 + -1) = 14 +1 +1 = 16h

Believe me, it is eternal source of sign errors and misunderstandings.

In this form also MQL5 and the PC handle the time differences like _TimeGMTOffset(): TimeLocal() + TimeGMTOffset() = TimeGMT())_ or _TimeDaylightSavings(): TimeLocal() + TimeDaylightSavings()_ = winter or standard local time of the PC.

As last of this part there are _FxOpen_ and _FxClose_ and the time duration of a whole week in seconds _WeekInSec_, because these designations are easier to understand than if in the code suddenly e.g. 604800 is added to a variable.

Then comes two simple simplifications of the variable output. TOSTR(A) it prints the name of the variable and its value - quite nice. Then the array with the abbreviations of the weekdays. It is used several times in the code developed for this project and helps to save space in the line but also for the orientation of calculated time stamps:

```
#define  TOSTR(A) #A+":"+(string)(A)+"  "             // Print (TOSTR(hGMT)); => hGMT:22 (s.b.)
string _WkDy[] =                                      // week days
  {
   "Su.",
   "Mo.",
   "Tu.",
   "We.",
   "Th.",
   "Fr.",
   "Sa.",
   "Su."
  };
```

Now follow alternatives for time calculation that avoid the detour of assigning a time to the MQ structure _MqlDateTime_ to read a scalar value. They are calculated directly:

```
#define  DoWi(t) ((int)(((t-259200)%604800)/86400))      // (int)Day of Week Su=0, Mo=1,...
#define  DoWs(t) (_WkDy[DoWi(t)])                        // Day of Week as: Su., Mo., Tu., ....
```

_DoWi(t)_ (Day of Week as integer) determines the day of the week as an integer (Sunday:0, Monday:1, ...) while _DoWs(t)_ Day of Week as string) returns the abbreviated day of the week as text using _DoWi(t)_ and the array _\_WkDy\[\]_. Since the second part of the article deals with the last and first hours of the weekends, these macro substitutions will be used more often.

```
#define  SoD(t) ((int)((t)%86400))                       // Seconds of Day
#define  SoW(t) ((int)((t-259200)%604800))               // Seconds of Week
```

_SoD(t)_ (Seconds of the day) returns the number of seconds since 00:00 of the passed time. So _SoD(TimeCurrent())_ calculates in this simple way the duration in seconds that a day candle already exists. Similarly, _SoW(t)_ calculates the number of seconds since the last Sunday 00:00. This way it is easy to calculate the percentage of elapsed time (in seconds) and the remaining time of this day candle. If you divide this value by 864 (=0.01\*24\*60\*60) you get the percentage that the day has already passed:

```
(double)SoD(D’20210817 15:34‘) / 864. = 64.86%
86400 - SoD(D’20210817 15:34‘)      =
Print("D'20210817 15:34': ",
      DoubleToString((double)SoD(D'20210817 15:34')/864.0, 2),"%  ",
      86400 - SoD(D’20210817 15:34‘),“ sec left“
);
```

These calculations work accordingly (whoever it needs):

```
#define  MoH(t) (int(((t)%3600)/60))                     // Minute of Hour
#define  MoD(t) ((int)(((t)%86400)/60))                  // Minute of Day 00:00=(int)0 .. 23:59=1439
```

The function _ToD(t)_ returns the seconds of the day as datatype " _datetime_". This prevents the warning messages of the compiler in one or the other situation:

```
#define ToD(t) ((t)%86400)    // Time of Day in Sec (datetime) 86400=24*60*60
```

The function _HoW(t)_ returns the number of hours that have passed since Sunday 00:00:

```
#define HoW(t) (DoW(t)*24+HoD(t))  // Hour of Week 0..7*24 = 0..168 0..5*24 = 0..120
```

This can be used to calculate the hour of the day in a simple way, the result is an integer, not a date:

```
#define HoD(t) ((int)(((t)%86400)/3600))  //Hour of Day: 2018.02.03 17:55 => 17
```

The same only 'commercial' rounded:

```
#define rndHoD(t) ((int)((((t)%86400)+1800)/3600))%24 // rounded Hour of Day 17:55 => 18
```

and again the same as type _datetime_:

```
#define rndHoT(t) ((t+1800)-((t+1800)%3600))  // rounded Hour of Time: 2018.02.03 17:55:56 => (datetime) 2018.02.03 18:00:00
```

The time of the start of the day as _datetime_, which otherwise would have to be determined extra with the call of the day time frame D1:

```
#define BoD(t) ((t)-((t)%86400))     // Begin of day 17.5 12:54 => 17.5. 00:00:00
```

The time of the beginning of the week as _datetime_, which would otherwise have to be determined by calling the weekly time frame:

```
#define BoW(t) ((t)-((t-172800)%604800 - 86400))  // Begin of Week, Su, 00:00
```

The following snipptes use the MQL structures and functions, because the conversion was too complicated for me, for example because of the leap years, but they follow the same format principle from above, the time functions with names of mostly three letters or acronyms, DoY, (=Day of Year), MoY (Month of Year), and YoY (Year of Year):

```
MqlDateTime tΤ; // hidden auxiliary variable: the Τ is a Greek characker, so virtually no danger
int DoY(const datetime t) {TimeToStruct(t,tΤ);return(tΤ.day_of_year); }
int MoY(const datetime t) {TimeToStruct(t,tΤ);return(tΤ.mon); }
int YoY(const datetime t) {TimeToStruct(t,tΤ);return(tΤ.year); }
```

Here is still the calculation of the week of the year according to US definition:

```
int WoY(datetime t) //Su=newWeek Week of Year = nWeek(t)-nWeeks(1.1.) CalOneWeek:604800, Su.22:00-Su.22:00 = 7*24*60*60 = 604800
  {
   return(int((t-259200) / 604800) - int((t-172800 - DoY(t)*86400) / 604800) + 1); // calculation acc. to USA
  }
```

(More about the weekly calculations here: [https://en.wikipedia.org/wiki/Week#Numbering](https://en.wikipedia.org/wiki/Week#Numbering "https://en.wikipedia.org/wiki/Week#Numbering"))

### The seasonal and local time differences

Having declared and explained the simplifications required by the functions, and a bit more, we now prepare the wide field of different local and seasonal time shifts and how the brokers and MQL5 handle them. Let's start with our programming language MQL5.

#### Time in MQL5

The first and immediate time is the one with which the broker provides its quotes, which is reflected in the opening time of the bars, and which can be queried with _TimeCurrent()_ for the last quote received. In many cases - and this may be surprising - this time is relatively unimportant and only serves to sort the bars, prices but also trading activities in time. So it has its relevance only in the environment of the terminal and that in live trading as well as in the strategy tester.

Then there is _TimeTradeServer()_: "Returns current calculation time of the trading server. Unlike the TimeCurrent() function, the time calculation is performed in the client terminal and depends on time settings in the user computer." BUT: "IDuring testing in the strategy tester, TimeTradeServer() is simulated according to historical data and always equal to TimeCurrent()." Thus, this function is not much help for a program that is to be optimized for live operation in the strategy tester.

The same is true for _TimeGMT()_ and _TimeLocal():_ "During testing in the strategy tester, TimeGMT() is always equal to TimeTradeServer() simulated server time.", and this is equal to TimeCurrent().

This also makes _TimeGMTOffset()_ 'useless' in the strategy tester, because _TimeGMTOffset() = TimeGMT() - TimeLocal()_, and that is then always and at all times zero. :(

The last thing left is _TimeDaylightSavings()_: "Returns correction for daylight saving time in seconds if the transition to daylight saving time occurred. Depends on settings in the user computer. ... If the transition to winter time (standard time) occurred, returns 0."

Here is the printout from a live demo account (MQ, as you can see from the date, is currently daylight saving time in the EU and US at the time of the query):

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    Broker: MetaQuotes Software Corp.

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeCurrent:             10:25

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeTradeServer:      10:25

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeLocal:                 09:25

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeGMT:                  07:25

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeDaylightSavings:  -3600 h: -1

2021.08.26 09:25:45.321    MasteringTime (EURUSD,M1)    TimeGMTOffset:         -7200 h: -2

and here the printout of the same account in the strategy tester (debugging):

2021.08.26 10:15:43.407    2021.06.18 23:54:59   Broker: MetaQuotes Software Corp.

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeCurrent:             23:54

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeTradeServer:      23:54

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeLocal:                 23:54

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeGMT:                  23:54

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeDaylightSavings:  0 h: 0

2021.08.26 10:15:43.407    2021.06.18 23:54:59   TimeGMTOffset:         0 h: 0

All these functions do not help in the strategy tester, but only in the live situation. So we have only TimeCurrent() to calculate all other times by ourselves and that with the mess. Well, easily anyone can ;)

Theoretically you could think, you can get everything from the broker, how he defines the time shifts for the timestamps of your quotes. For example, here is the answer from Alpari:

Kindly note that our trading sessions start on Monday at 00:05:00 Server Time

(GMT+2/GMT +3 DST) and end on Friday at 23:55:00 Server Time. During Summer

hours trading starts at 21:05 GMT on a Sunday, and during winter hours it is

22:05 GMT on a Sunday.

It turns out that this is only partially true. Firstly, the transition time when EU and USA are not equally in summer or winter time is not mentioned and secondly, the prices in this transition time, e.g. on 26.3.2021, do not end at 23:55, but at 22:55. before and afterwards again at 23:55.

But before we go on, a small overview of the different time zones and their acronyms. Most helpful to this whole topic seems to me to be this site: [https://24timezones.com/time-zones](https://www.mql5.com/go?link=https://24timezones.com/time-zones "https://24timezones.com/time-zones"). The table there lists 200 different time zones. Here are the ones that are relevant for us:

| **Acronym** | **Time Zone** | **Site** | **UTC Diff.** | **GMT Diff.** |
| --- | --- | --- | --- | --- |
| CEST | Central European Summer Time | Frankfurt, summer time | UTC+2 | GMT+2 |
| CET | Central European Time | Frankfurt, normal time | UTC+1 | GMT+1 |
| EDT | Eastern Daylight Time | New York, normal time | UTC-4 | GMT-4 |
| EEST | Eastern European Summer Time | Zypern, summer time | UTC+3 | GMT+3 |
| EET | Eastern European Time | Zypern, normal time | UTC+2 | GMT+2 |
| EST | Eastern Standard Time | New York, normal time | UTC-5 | GMT-5 |
| ET | Eastern Time | New York | UTC-5/UTC-4 | GMT-5/GMT-4 |
| GMT | Greenwich Mean Time | London, Normaltime | UTC+0 | GMT+0 |
| UTC | Coordinated Universal Time | London, Normaltime | UTC | GMT |

Note, the normal time difference from New York is -5 hours in summer -4, so one hour 'less', while that from Frankfurt is +1 or from MQ +2, but in summer +2 respectively +3 so one hour 'more', if you look purely at the numbers. Do not be confused by that, remember the [examples](https://www.mql5.com/en/articles/9926#example1)!

Now let's look at what are the timestamps of the first and last quotes of a demo account of MQ around a weekend: On the M1 chart of "EURUSD" we first activate the period separation, then press Enter on the chart to enter in each case the date of Monday in the format dd.mm.yyyy (not MQ date format). Then move the chart a little to the right with the mouse. Now the bar with the vertical line is the first of the new week and the bar to the left of it is the last of the past week. Here are the dates of the last bar of Friday and the first bar after the weekend:

The weekends around the time changeover in autumn 2020:

| **Summer-EU** | **Summer-EU** | **Summer-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Winter-US** | **Winter-US** | **Winter-US** | **Winter-US** | **Summer-US** |
| **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** |
| 2020.10.16 | 2020.10.19 | 2020.10.23 | 2020.10.26 | 2020.10.30 | 2020.11.02 | 2021.03.05 | 2021.03.08 | 2021.03.12 | 2021.03.15 |
| 23:54 | 00:05 | 23:54 | 00:05 | 22:54 | 00:02 | 23:54 | 00:03 | 23:54 | 00:00 |
| 1,1716 | 1,17195 | 1,18612 | 1,18551 | 1,16463 | 1,16468 | 1,19119 | 1,19166 | 1,19516 | 1,19473 |
| 1,17168 | 1,17208 | 1,18615 | 1,18554 | 1,16477 | 1,16472 | 1,19124 | 1,19171 | 1,19521 | 1,19493 |
| 1,1716 | 1,1718 | 1,18596 | 1,18529 | 1,16462 | 1,16468 | 1,19115 | 1,19166 | 1,19514 | 1,19473 |
| 1,1716 | 1,17188 | 1,18598 | 1,18534 | 1,16462 | 1,16472 | 1,1912 | 1,19171 | 1,19519 | 1,19491 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 29 | 22 | 35 | 48 | 30 | 3 | 33 | 2 | 23 | 25 |
| 4 | 11 | 2 | 6 | 2 | 61 | 1 | 38 | 1 | 29 |

And here the weekends around the time changeover in spring 2021:

| **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Winter-EU** | **Summer-EU** | **Summer-EU** | **Summer-EU** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Winter-US** | **Winter-US** | **Winter-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** | **Summer-US** |
| **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** | **Fr.** | **Mo.** |
| 2021.03.05 | 2021.03.08 | 2021.03.12 | 2021.03.15 | 2021.03.19 | 2021.03.22 | 2021.03.26 | 2021.03.29 | 2021.04.02 | 2021.04.05 |
| 23:54 | 00:03 | 23:54 | 00:00 | 22:54 | 00:00 | 22:54 | 00:07 | 23:54 | 00:03 |
| 1,19119 | 1,19166 | 1,19516 | 1,19473 | 1,19039 | 1,18828 | 1,17924 | 1,17886 | 1,17607 | 1,17543 |
| 1,19124 | 1,19171 | 1,19521 | 1,19493 | 1,19055 | 1,18835 | 1,17936 | 1,17886 | 1,17608 | 1,17543 |
| 1,19115 | 1,19166 | 1,19514 | 1,19473 | 1,19038 | 1,18794 | 1,17922 | 1,17884 | 1,17607 | 1,17511 |
| 1,1912 | 1,19171 | 1,19519 | 1,19491 | 1,19044 | 1,18795 | 1,17933 | 1,17886 | 1,17608 | 1,17511 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 33 | 2 | 23 | 25 | 41 | 43 | 17 | 3 | 2 | 3 |
| 1 | 38 | 1 | 29 | 1 | 20 | 5 | 68 | 28 | 79 |

Interesting and not quite understandable to me is the fact that sometimes the last quotes before the weekend arrive around 23:00 (22:54), but usually at 24:00 (23:54), but the first quotes always arrive at 00:00. This sometimes creates, on a purely hourly view, a 'price hole' of 1 hour (Close at 23:00, Open at 00:00) while the Forex market always closes at 17:00 (NY time) on Friday and always opens at 17:00 (NY time) on Sunday. Let's look specifically at the weekends that switch over and calculate the respective time in the other time zones that are relevant to us:

| **Date** |  | **Date Last Fr.** |  |  |  | **Date** |  |  |  |  |  | **last m1 Bar** | **next m1 Bar** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Switch US** |  |  | **NY-Time** | **NY-GMT** | **GMT** | **Switch EU** |  | **CET-GMT** | **CET** | **EET-GMT** | **EET** | **of MQ** | **of MQ** |
|  | **Summer-US** | Fr, 16. Oct 20 | 17:00 | GMT + 4 | 21 |  | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | 24 | 23:54 |  |
|  | **Summer-US** | So, 18. Oct 20 | 17:00 | GMT + 4 | 21 |  | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | 24 |  | 00:05 |
|  | **Summer-US** | Fr, 23. Oct 20 | 17:00 | GMT + 4 | 21 | 25.10.20 | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | **24** | 23:54 |  |
|  | **Summer-US** | So, 25. Oct 20 | 17:00 | GMT + 4 | 21 |  | **Winter-EU** | **GMT + 1** | **22** | **GMT + 2** | **23** |  | **00:05** |
| 01.11.20 | **Summer-US** | Fr, 30. Oct 20 | 17:00 | GMT + 4 | **21** |  | **Winter-EU** | **GMT + 1** | **22** | **GMT + 2** | **23** | **22:54** |  |
|  | **Winter-US** | So, 1. Nov 20 | 17:00 | GMT + 5 | **22** |  | **Winter-EU** | GMT + 1 | 23 | GMT + 2 | 24 |  | 00:02 |
|  | **Winter-US** | Fr, 6. Nov 20 | 17:00 | GMT + 5 | 22 |  | **Winter-EU** | GMT + 1 | 23 | GMT + 2 | 24 | 23:54 |  |
|  | **Winter-US** | So, 8. Nov 20 | 17:00 | GMT + 5 | 22 |  | **Winter-EU** | GMT + 1 | 23 | GMT + 2 | 24 |  | 00:03 |
| 14.03.21 | **Winter-US** | Fr, 12. Mrz 21 | 17:00 | GMT + 5 | **22** |  | **Winter-EU** | GMT + 1 | 23 | GMT + 2 | **24** | 23:54 |  |
|  | **Summer-US** | So, 14. Mrz 21 | 17:00 | GMT + 4 | **21** |  | **Winter-EU** | **GMT + 1** | **22** | **GMT + 2** | **23** |  | **00:00** |
|  | **Summer-US** | Fr, 26. Mrz 21 | 17:00 | GMT + 4 | 21 | 28.03.21 | **Winter-EU** | **GMT + 1** | **22** | **GMT + 2** | **23** | **22:54** |  |
|  | **Summer-US** | So, 28. Mrz 21 | 17:00 | GMT + 4 | 21 |  | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | 24 |  | 00:07 |
|  | **Summer-US** | Fr, 2. Apr 21 | 17:00 | GMT + 4 | 21 |  | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | 24 | 23:54 |  |
|  | **Summer-US** | So, 4. Apr 21 | 17:00 | GMT + 4 | 21 |  | **Summer-EU** | GMT + 2 | 23 | GMT + 3 | 24 |  | 00:03 |

In the week(s) when the clocks in the EU and the US do not both show local daylight saving time or standard time respectively, the forex market in New York opens Sunday at 23:00 EET and closes Friday at 23:00 EET. However, various brokers, as well as MQ's demo accounts, always provide the first quotes on Sunday at 24:00 (or Monday 00:00). Now one could think that not with the first changeover, but the time stamp of the quotes always follows the second time changeover. Then, however, on Friday the last courses before the closing of the Forex market would have to have a time stamp of shortly before 24:00, because only then there are the full 24\*5=120 hours, but so one hour is missing. This raises the question, when is the hour missing: on the Friday before the weekend or on the Sunday after it?

Since the closing prices of the week are the more important ones compared to the first trading hour on Sunday, it can be assumed that if on Friday the last prices are from shortly before 23:00, but the next ones are only at 24:00 or 00:00, the first hour of the Forex market is missing, the one from Sunday 23:00-24:00, but not the last one, i.e. the one from Friday 23:00-24:00. However, if the first prices would have a timestamp of Sunday shortly after 23:00, it would be quite easy not only to recognize the time change, but also to know when the Forex market closes again (5d\*24h=120h after the opening on Sunday), if you follow a cautious strategy that closes all positions before the weekend. Well as said simple can anyone.

First, we consider what assumptions we can make. In the strategy tester we only have _TimeCurrent()_. With this we will determine GMT or UTC, so that we can then easily calculate all times of other time zones. Subject to possible holidays or other reasons that affect the times for opening and closing the forex market in the USA, we assume:

1. The Forex market closes at 17:00 New York time on Friday.
2. The Forex market opens on Sunday at 17:00 New York time.
3. It is normally open for (5\*24=) 120 h and closed for (2\*24=) 48 h.
4. If there are missing hours between Fri. 17:00 and Sun. 17:00, then there will be missing quotes on Sunday until the first quote and not on Friday after the last quote received.
5. Incoming quotes (no matter what time stamp) are always up to date (and not 1 hour old for example).
6. Broker does not change its policy of time change, as it was last, it was before for its historical quotes.

### Outlook

We have now defined the functions, or more precisely macro substitutions, that we will use later and clarified the conditions to develop in the next article "Dealing with Time Part 2: The Functions" the functions that will calculate GMT for us from any time given.

Attached is DealingwithTimePart1.mqh, which contains the code parts discussed here. It will be extended with the mentioned functions in the second article.

Please write suggestions, comments and hints in the comment section of the article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/9926.zip "Download all attachments in the single ZIP archive")

[DealingWithTimePart1.mqh](https://www.mql5.com/en/articles/download/9926/dealingwithtimepart1.mqh "Download DealingWithTimePart1.mqh")(7.18 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Dealing with Time (Part 2): The Functions](https://www.mql5.com/en/articles/9929)
- [Cluster analysis (Part I): Mastering the slope of indicator lines](https://www.mql5.com/en/articles/9527)
- [Enhancing the StrategyTester to Optimize Indicators Solely on the Example of Flat and Trend Markets](https://www.mql5.com/en/articles/2118)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/378827)**
(11)


![Rolom27](https://c.mql5.com/avatar/avatar_na2.png)

**[Rolom27](https://www.mql5.com/en/users/rolom27)**
\|
14 Jun 2022 at 22:13

**Miguel Angel Vico Alba [#](https://www.mql5.com/es/forum/380476#comment_40165530):**

The time zone in the terminal is that of the server location, i.e. if the broker is located in London, but the server is in, for example, Luxembourg, the time zone is different. It has always been and will always be like that.

Thank you very much, I am checking if the trader is real, today I checked and the prices of the cfds reported in the operations, taking into account the time difference mentioned before, if they coincide with the chart values of the London market that I consult today.

This is the report of meta trader 5 of a trade today that I passed an acquaintance who recently opened an account with that broker, what do you think seems real?

PS I have known of fake brokers that create websites and make it look like they are real brokers and in a few months disappear.

![Miguel Angel Vico Alba](https://c.mql5.com/avatar/2025/10/68e99f33-714e.jpg)

**[Miguel Angel Vico Alba](https://www.mql5.com/en/users/mike_explosion)**
\|
15 Jun 2022 at 01:04

**Rolom27 [#](https://www.mql5.com/es/forum/380476#comment_40193612):**

Thank you very much, I am verifying if the trader is real, today I checked and the prices of the cfds reported in the operations, taking into account the time difference mentioned before, if they coincide with the values of London market charts that I consult today.

This is the report of meta trader 5 of a trade today that I got from an acquaintance who recently opened an account with this broker, what do you think seems real?

PS I have known of fake brokers that create websites and make it look like they are real brokers and in a few months disappear.

Send me by private message more information about that broker and I can tell you if it is regulated, etc.. Such discussions are forbidden here on the forum.

![Christian Linden](https://c.mql5.com/avatar/2023/5/64613c2e-006a.jpg)

**[Christian Linden](https://www.mql5.com/en/users/lindomatic)**
\|
23 Sep 2023 at 23:33

Brilliant article and help! I wasted many months looking at [test results](https://www.mql5.com/en/docs/common/TesterStatistics "MQL5 Documentation: TesterStatistics function") that just were invalid.

There should be any hint if the tester is used: "Take care with time dependend strategies in the tester.. " something like that. How could one know?

I think in the first table with the time zones and acronyms you mean "New York, summer time" in row 3 (EDT, UTC-4), not New York, normal time.

![Carl Schreiber](https://c.mql5.com/avatar/2018/2/5A745EEE-EB76.PNG)

**[Carl Schreiber](https://www.mql5.com/en/users/gooly)**
\|
24 Sep 2023 at 06:53

Thank you very much, but I won't change anything here as the calculation of some of the integrated functions has been changed (I assume in the context of the extended use of matrices and vectors) I posted a new version of all functions and macros and one more here: [https://www.mql5.com/en/code/45287](https://www.mql5.com/en/code/45287 "https://www.mql5.com/en/code/45287").

And there are two programs to test und check the way it works and can be used.

![Christian Linden](https://c.mql5.com/avatar/2023/5/64613c2e-006a.jpg)

**[Christian Linden](https://www.mql5.com/en/users/lindomatic)**
\|
24 Sep 2023 at 10:35

**Carl Schreiber [#](https://www.mql5.com/en/forum/378827#comment_49514680):**

Thank you very much, but I won't change anything here as the calculation of some of the integrated functions has been changed (I assume in the context of the extended use of matrices and vectors) I posted a new version of all functions and macros and one more here: [https://www.mql5.com/en/code/45287](https://www.mql5.com/en/code/45287 "https://www.mql5.com/en/code/45287").

And there are two programs to test und check the way it works and can be used.

Alright, will check your new versions, thanks again.

![Combinatorics and probability theory for trading (Part III): The first mathematical model](https://c.mql5.com/2/43/gix1_2.png)[Combinatorics and probability theory for trading (Part III): The first mathematical model](https://www.mql5.com/en/articles/9570)

A logical continuation of the earlier discussed topic would be the development of multifunctional mathematical models for trading tasks. In this article, I will describe the entire process related to the development of the first mathematical model describing fractals, from scratch. This model should become an important building block and be multifunctional and universal. It will build up our theoretical basis for further development of this idea.

![Graphics in DoEasy library (Part 82): Library objects refactoring and collection of graphical objects](https://c.mql5.com/2/43/MQL5-avatar-doeasy-library3-2__1.png)[Graphics in DoEasy library (Part 82): Library objects refactoring and collection of graphical objects](https://www.mql5.com/en/articles/9850)

In this article, I will improve all library objects by assigning a unique type to each object and continue the development of the library graphical objects collection class.

![Dealing with Time (Part 2): The Functions](https://c.mql5.com/2/43/mql5-dealing-with-time__1.png)[Dealing with Time (Part 2): The Functions](https://www.mql5.com/en/articles/9929)

Determing the broker offset and GMT automatically. Instead of asking the support of your broker, from whom you will probably receive an insufficient answer (who would be willing to explain a missing hour), we simply look ourselves how they time their prices in the weeks of the time changes — but not cumbersome by hand, we let a program do it — why do we have a PC after all.

![Combinatorics and probability theory for trading (Part II): Universal fractal](https://c.mql5.com/2/42/Centropolis2.png)[Combinatorics and probability theory for trading (Part II): Universal fractal](https://www.mql5.com/en/articles/9511)

In this article, we will continue to study fractals and will pay special attention to summarizing all the material. To do this, I will try to bring all earlier developments into a compact form which would be convenient and understandable for practical application in trading.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/9926&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082941465560093137)

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