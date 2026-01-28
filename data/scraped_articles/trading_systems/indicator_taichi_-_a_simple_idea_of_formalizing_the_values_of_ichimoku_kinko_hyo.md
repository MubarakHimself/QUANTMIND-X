---
title: Indicator Taichi - a Simple Idea of Formalizing the Values of Ichimoku Kinko Hyo
url: https://www.mql5.com/en/articles/1501
categories: Trading Systems
relevance_score: 3
scraped_at: 2026-01-23T21:02:53.800463
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/1501&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071533624399964620)

MetaTrader 4 / Trading systems


### Introduction

What forced the creation of the indicator Taichi and a trading system based on this
indicator?

Let me start with a small prehistory. First I invested much time into reading special
literature and analyzing a large number of indicators and their combinations. I
suppose, this is a common way to begin for if not everyone, for at least 90% of
those starting to work in the financial market independently. Simultaneously I
had several attempts of manual trading using generally accepted and widely known
trading systems. The conclusion was simple - without a serious psychological training
no indicator or trading system can be profitable in practical application, a deposit
is invariably lost.

While my previous experience allowed to learn MQL4 rather quickly, the idea of using
the possibilities of automated trading for eliminating the psychological factor
seemed very tempting. This was the beginning of the long process of defining principles,
formalizing and writing my own trading system. At that moment I unexpectedly made
a conclusion that may seem simple - all indicators (or at least the majority of
known indicators) in a graphical way draw one and the same thing. It is price.
Price at the current moment, price an hour ago, price in its historical presentation,
different price aspects and characteristics.

We always know how a price behaved in the past, which enables "analysts"
to explain authentically why an event on a chart happened or did not happen. It
is a common opinion that a price is the last thing to change in the market. Probably
it is true, but for the automated trading this does not matter. Automated trading
is based on technical analysis and data from indicators. And this is actually an
attempt to formalize a price behavior in the past and to project this behavior
onto a probable price development in the future. And that is where the most interesting
things start - the possibility of seeing exact entry and exit points is a dream
of any trader. Defining flat zone and its end in an automated mode is also a very
important task. Perhaps the indicator Taichi will help you to detect the most important
events on a chart not in the past, but at the moment of their appearance.

### Wanted: Exact Entrance! Is it Trend?

From the large amount of existing indicators the most interesting for me seemed
[Ichimoku Kinko Hyo](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh). The reason for this is quite simple - its workability that is proved by years.
Still there is nothing ideal in this world, and the indicator itself was developed
not for Forex. Of course, Hosodo contributed greatly to trading. Moreover, all
the indicator principles are actively used in the suggested variant of the indicator
Taichi. Actually the indicator Taichi is an evolutionary mutation of [Ichimoku Kinko Hyo](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh).

I suppose, I am not the only one who faced problems when interpreting Ichimoku values.
Operation methods and setup parameters are described in the library of technical
analysis [Ichimoku Kinko Hyo](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh), so I will not dwell on them in this article. The main idea of the indicator Taichi
is formalizing signals of [Ichimoku Kinko Hyo](https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ikh) and detecting a prospective flat/trend.

What do we need for a successful trading? The first thing is detecting an entrance
point. Let us try to detect it using the indicator Taichi.

Principle and lines of the indicator:

Taichi - weighted average Tenkan+Kijun+SpanA+SpanB. The principle according to
which these lines were combined are hard to explain technologically, it should
be understood intuitively. Still it may be defined as an average value of market
moods. Of course, this idea is not original, but it seems to be working.

TaichiFor - weighted average SpanA+SpanB with a shift Kijun. It is not hard to
understand its meaning - it is an average of a cloud.

Signal - moving average with a period Kijun.

SSignal - moving average with a period Senkou.

Flat - paints over flat zones.

- The line Taichi consolidates average values Tenkan+Kijun+SpanA+SpanB/4. Consolidated
direction of the line Taichi and the position of a price relative to it allows
to define the current state of events quite precisely. The line Taichi is also
a fast line of a trend; it compensates by its values average speculating price
movements.

- The line TaichiFor consolidates average values SpanA+SpanB/2 with a shift Kijun.
Consolidated direction of the line TaichiFor and the position of Taichi relative
to it defines a possible long-term trend. The line TaichiFor is a slow line of
a trend; it compensates by its values strong speculating price movements. Moreover,
inheriting the cloud properties it forecasts the probable development of price
movement.

- The line Signal is a moving average upon Taichi with a period Kijun. The line Signal
is introduced for smoothing the values of Taichi; it allows to filter out false
signals.
- The line SSignal is a moving average upon Taichi with a period Senkou. The line
SSignal is introduced for smoothing values of Taichi; it also allows to filter
out false signals, but at stronger price movements.
- The zone Flat is painted over providing that the average difference Taichi-Signal,
Taichi-SSignal and Signal-SSignal in points does not exceed the sensibility level,
set up by an external variable FlatSE (preliminary research showed that the optimal
level is 6-10 points, depending on the strategy aggression).


Below is a chart with a possible entering point into a short position and a possible
position closing point.

![](https://c.mql5.com/2/15/taichi_1.gif)

Viewing the above chart, we can conclude that for entering a short position we need
the following combination of indicator values with no painted flat zone:

1. The current price is less that Taichi (preliminary signal)
2. Taichi is less than TaichiFor (confirming signal)
3. Signal is less than SSignal (signal for entering)

But it is not so easy. It is not as definite as we would like it to be. No one can
guarantee that a situation is favorable for getting good profit. I suppose many
of you have noticed that favorable indicator combinations occur when a price achieves
pivot points. It is quite a hard task to define pivot points, besides it is hard
to implement in an automatic mode. That is why for trying to filter out false signals,
I recommend using an additional indicator.

You can use any indicator you wish (or a combination of several indicators - indicators
show some aspects of price states). In our variant we use DeMarker attaching two
MA lines to it.

![](https://c.mql5.com/2/15/demarkertaichi_1.gif)

What we expect from DeMarker is the moment, when DeMarker crosses fast and slow
MA lines somwhere about 0.7 for a short position. This is the first preliminary
signal. Then we need the intersection of MA lines. This is the main signal after
which we start working with the values of the indicator Taichi. If MA lines did
not intersect, the signal is considered to be false and the price movement is expected
to continue.

Offered parameters for H1: DeMarker 64, fast MA 42, slow MA 86. The parameters were
selected in an experimental way, so in you personal case the values may differ.

### Wanted: Exact Exit! Is it Flat or Pivot?

Well, we have entered. But you understand that it is not all we need. Position closing
is probably the most difficult moment in terms of psychological tension. Let us
try to close a position with minimal loses of a possible profit, besides we should
not face the situation, when after waiting too long the position stops being profitable.

Closing a position at a first negative signal may result in a strong disappointment
if the price then will continue its movement in its previous direction. After that
you may try to reopen the position (quite an often case), which is likely to result
in losses.

A position closing technique, offered for a certain variant, is quite simple and
has several aims. The first one is defending the profit acquired from a position.
But when should this be done? Price movement is almost unpredictable and has quite
a large deviation at a movement in one direction. The values of the indicator Taichi
are too late for this purpose, that is why let us try to use the filtering combination
of the indicator DeMarker with two MA lines, as described above.

After opening a position (in our case it is a short one) check the value of the
fast MA line on DeMarker. When the fast MA line upon DeMarker makes an attempt
to turn upwards, activate a trailing stop and set stop level at 25-30 points away
from the current price. On hour charts this level is quite enough for eliminating
a quick stop activation and prevents from losing already acquired profit.

There is a high probability that we reached a true pivot point, or a point of turning
into flat, so it is reasonable to close part of a position together with setting
stop level. If a lot size is decimal and a brokerage company allows a partial position
closing, we can close 20% of the position. As a result we have the following: we
have acquired profit, protected the remaining part of the position by a stop and
allowed the position to get more profit in case if the signal was false (fast MA
line and slow MA line did not intersect). What is more important, we do not have
to bother and be nervous.

Experience shows that the above described situation most often leads to a pivot
or passing into flat. So when the pivot combination occurs (fast MA line and slow
MA line intersect), close the remaining part of the position.

![](https://c.mql5.com/2/15/close_1.gif)

### Flat - Time to Rest?

And what should we do, if the combination was formed, but values of the indicator
Taichi are still directed into the same direction. The advice is simple - wait.

Do not be in a hurry trying to win a little profit. Today is not the last day of
trading and Forex will not cease to exist tomorrow. The experience showed that
after the pivot combination on DeMarkeris formed, Taichi is likely to enter the
state of flat zone indication. In such a period, despite the values of a filtering
indicator, it is not recommended to enter the market.

Although, there are exceptions. The flat zone may last very long. In such periods
you may try to trade using a system of an equidistant channel. But this is not
the topic of this article, so we will not dwell on it.

![](https://c.mql5.com/2/15/flat_1.gif)

### Conclusion

The main idea I wanted to share with you is the following: despite the fact that
the market exists already for a long time and one could think everything is already
invented, the market itself is a very interesting object to be investigated. We
should not take on trust everything that was written earlier - check it, try to
use all the developed earlier as the basis of your own ideas. I cannot guarantee
that you will find something new. But I am sure it will be interesting for you
and will help to develop your own understanding of the Forex market.

This article contains an example of an attempt to find my own way in trading based
on development and ideas that were expressed long ago. Perhaps, this will be useful
for you.

All the written in this article is only my personal opinion and my own view on processes
and events.

Attached is the indicator Taichi and the indicator DeMarker with MA attached to
it. Prefix Cronex in the names of the files is used only for the identification
of indicators upon their belonging to a trading strategy.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1501](https://www.mql5.com/ru/articles/1501)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1501.zip "Download all attachments in the single ZIP archive")

[Cronex\_DeMarker.mq4](https://www.mql5.com/en/articles/download/1501/Cronex_DeMarker.mq4 "Download Cronex_DeMarker.mq4")(2.54 KB)

[Cronex\_Taichi.mq4](https://www.mql5.com/en/articles/download/1501/Cronex_Taichi.mq4 "Download Cronex_Taichi.mq4")(4.23 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39388)**
(17)


![bobby](https://c.mql5.com/avatar/avatar_na2.png)

**[bobby](https://www.mql5.com/en/users/funyoo)**
\|
3 May 2009 at 17:16

Thanks for sharing this article.

I have made an EA based on these two indicators [here](https://www.mql5.com/go?link=http://www.tradingsystemforex.com/expert-advisors-backtesting/1319-taichi-demarker-ea.html "http://www.tradingsystemforex.com/expert-advisors-backtesting/1319-taichi-demarker-ea.html").

![bigpipn](https://c.mql5.com/avatar/avatar_na2.png)

**[bigpipn](https://www.mql5.com/en/users/bigpipn)**
\|
13 Dec 2009 at 19:17

I call this Cronex Taichi line The money line. :) I have a select few SMAs that I follow and when I see the Taichi rolling over or lifting off with my SMA squeezing it, it is guaranteed to be a big move. It's the most impressive squiggly you can add to any chart. :) It is also amazing how the yens cascade then you get a small bounce and the market comes up and tags taichi to the tick then continues its journey. I never can get enough of it.


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
31 Mar 2010 at 13:14

Hi !

Can you make CRONEX TAICHI indicator a MTF indicator?

best Regards,

Dilip


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
29 Oct 2011 at 20:14

Hi, this is an excellent indicator set.

Please can you advise on your setting parameters for your Cronex [DeMarker](https://www.mql5.com/en/code/26 "The Demarker Indicator is based on the comparison of the period maximum with the previous period maximum") ...

Thankyou

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
30 Oct 2011 at 10:00

**Desander:**

As a new Forex- and commodities trader, i discovered your two indicators.

**What are they beautyfull** !

**I don't use anything-else anymore** !

This is **realy** seeing the trend, **and** the trendbreaking before the quote changes.

Thank you a lot, this will make me rich in the future !!

Thank you, thank you...

Gilbert Despeghel,

Belgium.

Hi, what settings do you use for the two indicators and time frame?


![Displaying a News Calendar](https://c.mql5.com/2/15/520_12.gif)[Displaying a News Calendar](https://www.mql5.com/en/articles/1502)

This article contains the description of writing a simple and convenient indicator displaying in a working area the main economic events from external Internet resources.

![MQL4 Language for Newbies. Custom Indicators (Part 1)](https://c.mql5.com/2/15/516_15.gif)[MQL4 Language for Newbies. Custom Indicators (Part 1)](https://www.mql5.com/en/articles/1500)

This is the fourth article from the series "MQL4 Languages for Newbies". Today we will learn to write custom indicators. We will get acquainted with the classification of indicator features, will see how these features influence the indicator, will learn about new functions and optimization, and, finally, we will write our own indicators. Moreover, at the end of the article you will find advice on the programming style. If this is the first article "for newbies" that you are reading, perhaps it would be better for you to read the previous ones. Besides, make sure that you have understood properly the previous material, because the given article does not explain the basics.

![MQL4 Language for Newbies. Custom Indicators (Part 2)](https://c.mql5.com/2/15/536_27.gif)[MQL4 Language for Newbies. Custom Indicators (Part 2)](https://www.mql5.com/en/articles/1503)

This is the fifth article from the series "MQL4 Languages for Newbies". Today we will learn to use graphical objects - a very powerful development tool that allows to widen substantially possibilities of using indicators. Besides, they can be used in scripts and Expert Advisors. We will learn to create objects, change their parameters, check errors. Of course, I cannot describe in details all objects - there are a lot of them. But you will get all necessary knowledge to be able to study them yourself. This article also contains a step-by-step guide-example of creating a complex signal indicator. At that, many parameters will be adjustable which will make it possible to change easily the appearance of the indicator.

![Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://c.mql5.com/2/14/315_5.png)[Terminal Service Client. How to Make Pocket PC a Big Brother's Friend](https://www.mql5.com/en/articles/1458)

The article describes the way of connecting to the remote PC with installed MT4 Client Terminal via a PDA.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=udmharipalvndqyuluhhgioddpnqbran&ssn=1769191372832350833&ssn_dr=0&ssn_sr=0&fv_date=1769191372&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1501&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Indicator%20Taichi%20-%20a%20Simple%20Idea%20of%20Formalizing%20the%20Values%20of%20Ichimoku%20Kinko%20Hyo%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176919137289972514&fz_uniq=5071533624399964620&sv=2552)

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