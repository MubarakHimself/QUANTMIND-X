---
title: Creation of an Automated Trading System
url: https://www.mql5.com/en/articles/1426
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:53:33.249624
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1426&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062793859873810699)

MetaTrader 4 / Trading systems


### Introduction

You must admit that it sounds alluringly - you become a fortunate possessor of a
program that can develop for you a profitable automated trading system (ATS) within
a few minutes. All you need is to enter desirable inputs and press Enter. And -
here you are, take your ATS tested and having positive expected payoff. Where thousands
of people spend thousands of hours on developing that very unique ATS, which will
"wine and dine", these statements sound, to put it mildly, very hollow.

On the one hand, this really looks a little larger than life... However, to my mind,
this problem can be solved. We all know that the nature of market fluctuations
changes all the time whereas the tools frequently used by traders have usually
low adaptivity to varied conditions. This objectively results from that, prior
to adaptation, one should have a clear idea of what to adapt to. In most cases,
we ourselves cannot define clearly what surrounds us, in order to elaborate an
adequate behavior pattern.

If it is referred to creation of algorithms under uncertainty, how can one blind
(a man) lead another blind (a computer) where to go? On the other hand, everything
repeats in this world earlier or later, so the situation that took place in the
past can take place again in future. It is this fact that underlies the most ATS's.
I'm not referring to adaptation to varied conditions on the market. I'd better
stress patience and ability to wait until the moment, which brought statistically
acceptable results in the past, comes back.

### What Are Automated Trading Systems for?

One is tempted to answer: ATS's are for traders who don't know what operations to
perform on the market. I mean, if a trader knows what to do, there is no need to
use an ATS. But this is out of this article's scope. At the same time, if a human
always knows what to do, it is not a human.

This is a joke, of course. However, this is why we sometimes feel blind, like in
total darkness. So nature cared of the man and created cycles for such situations
of uncertainty. We can consider alternation of day and night as an obvious cycle.
It is common knowledge that the night follows the day and vice versa. We also can
easily see that the day is over - it starts to become dark. It is approximately
the same situation in the market. If we see a classical flat, it means that it
will be followed by a trend for sure. And vice versa, if we see a trend in a chart,
it will surely be followed by a flat or by an opposite, but weaker trend.

In other words, to elaborate an adequate market behavior pattern for the future,
we have to define clearly and classify the current state of the market. As practice
proves, it is the point that represents the challenge. Every trader has his or
her own idea about what state the market is having at the moment. This is determined
by that everybody watches charts through his or her own alembic of time. Yet at
the same time, if we divide the entire stock into all traders with their time prospects,
it will turn out that the most money falls within traders who work (invest in open
positions) for periods of half a year or more.

If it were different, we would never see trends several years long. The specificity
of financial markets itself defines this scale. This is because, if we drastically
reduce investing time in open positions and invest in larger volumes and in shorter
periods, the market will either just die or become absolutely illiquid, so working
on such a market will be a kind of charity. So large operators have to prolong
periods of position forming by several weeks and even months. Respectively, a large
position cannot be closed within one hour, as well.

Whereas Forex operates on volumes of 3-4 trillion US dollars a day, the prevailing
amount of these volumes is made using leverage. This means that open positions
are swapped on closing and the money really operated on is much smaller. If somebody
decides to change currency by direct conversion, this operation cannot be covered
by a reverse swap, so one will have to find somebody to cover the risk which results
in multiplicative effect.

Forming of such open positions with the volume of "just" several milliards
of dollars may seriously "move" the market. Such volumes are quite normal
for large pension and hedge funds that don't speculate in differences, they just
look for more profitable interest tools in certain currencies and, having found
them, invest in them for a period of a year or more. The resulting currency difference
can be considered as an extra bonus.

Summarizing all above, we can define that rush, practically non-predictable fluctuations
during flat periods start exactly as a result of large direct conversion operations.
The trend itself results from those operations because market makers start to cover
the risks obtained and open their positions in the same direction. The longer is
the flat (the larger is the position and the period it is being formed), the stronger
will be the following trend.

### Market Patterns

Much as large operators hide their actions concerning conversion operations, the
latter ones will sooner or later become visible. A different matter is when a common
trader understands what happens in the market. It is to the benefit of large operators
\- to mask their actions as long as possible, until they are completed. It is to
the benefit of small (and not very small) traders - to recognize such actions as
soon as possible in order to take profitable positions beforehand. Anyway, forming
a large position is a complicated and long-term process, i.e. everything must be
computed up to details and estimated beforehand.

It is the fact that the regulations of forming a large position are still regulations
and comply with certain rules that allows recognition of these actions, at least,
at the stage of performing them, not post factum. This is what underlies all sorts
of market patterns. Specific actions are reflected as a specific pattern on price
charts. To start making detailed analysis of market patterns, it would be reasonable
to try and get to know what they actually represent and what we want from them.
As an initial example, we can take market patterns described in details in classical
books on technical analysis.

For example, chart patterns like "Double Top/Bottom", "Head and Shoulders",
"Triangle", etc. are all market patterns since they after having been
formed on the chart, in the most cases, are followed be a specific and quite expected
price movement. Thus, the market pattern can be defined as follows:

Market pattern is a certain drawing on a price chart followed by the price movement
in the expected direction and for the expected distance.

I would add - in the most statistical cases.

If we just consider price charts, various market patterns can be recognized visually,
in spite of existing errors. However, where it is referred to ATS's, we will have
to use mathematical methods since computers love accuracy and cannot stand uncertainty.

### Cleanness of Market Patterns and Their Predictive Capability

Any market pattern can be described in some mathematical terms. The more parameters
are used in the market pattern description, the clearer it becomes and the easier
it is to recognize it in future. On the other hand, prior to start describing market
models mathematically, it would be reasonable to select those giving, on the most
cases the expected results. However, there are some difficulties here.

So how should we define and select these patterns? Thinking of this, I found some
associations to other topics. Let's consider, for example, a horse breeder who
breeds pedigreed horses. He or she knows definitely from the previous experiences
what requirements must be met to breed a stakes winning horse. For example, this
horse's parents and other ancestors must also be pedigreed racers. Housing conditions,
feeding and training must also meet special requirements. Well, if the breeder
meets all the requirements, he or she will get young horses, the most of which
will answer his or her expectations.

In the market terms, these conditions are our market pattern that must answer our
expectations in future. The only difference is that a trader cannot create these
conditions, but he or she is able to wait until these conditions occur. Adverting
to the above statements, we have to define what patterns should be selected to
be described mathematically and, most important, how to recognize them early on.
It would be reasonable to act by contradiction. I mean, we define our expected
results first, and then we will analyze conditions that provided such results in
the past.

For example, we would like to get 100 pips per trade. Therefore, we need such price
movements that would exceed these 100 pips + spread + contingency stock (let it
be 30-35%). It is highly desirable, in this case, that these price movements go
into one pulse and don't contain any deep corrections. It would also be reasonable
to describe market patterns using mathematical tools normally utilized by traders.
The set of indicators and their inputs should be defined beforehand. Further actions
can be described as follows:

1. select and mark on the chart the areas meeting our requirements, namely, pulse movements
traveling by 130-140 pips;
2. define points/areas where they started;
3. describe in mathematical terms the point/area where the pulse started;
4. try to find common values of parameters in the descriptions obtained.

If the common parameters are found, it is quite possible to use them as a basis
for a market pattern. If every situation looks like unique, it may be reasonable
to change indicators to describe the pattern or to search for better parameters
for the existing ones.

The next stage is back testing of the market pattern obtained. At this stage, it
can turn out that the price does not always goes at 130-140 pips in the same conditions,
it sometimes doesn't reach this level and sometimes considerably exceeds it. So
you will have to consider the amount of cases where your pattern provides the expected
result and of those cases that can be considered to be statistical error. If that
statistical error exceeds a certain level, we will have to input additional parameters
to describe the pattern in order to refine our cases according to pour purposes.
So this is how we can describe a market pattern for our specific purposes (100
pips changes per trade).

Having such a description, we can create an ATS that would fix forming of the given
market pattern and give a signal to start a new trade. In my opinion, the most
difficult thing here is that all patterns are built on historical data, so nobody
can predict their behavior in future. Pedigreed horses sometimes sire bad colts.

Our last recourse will be to continue testing. It means, having obtained a satisfactory
result on historical data, you start trading and continue to watch statistical
parameters by accrued method. As soon as the results stop satisfying you, you will
have to recompute and re-describe everything.

### Automation of ATS Creation

Let us think at the end whether it is possible to automate selection and description
of patterns to underlie an ATS. Technically speaking, I don't see any considerable
difficulties here. As I wrote before, the description of a market pattern starts
with definition of objectives. Then we select in the historical database situations
that meet the preset criteria.

I think умут a programmer of little experience is able to select pulses of preset length in historical data
and register them. But the next stage related to description of the selected situations may require a fresh approach
and much experience. At describing a point/area of a pulse movement, it will be necessary to search in a huge
amount of indicators having manifold combinations of parameters. The more indicators and their parameters are
involved in the description, the exacter the pattern will be described and the easier it will be to filter market
situations that don't completely meet the description.

Later on, from thew entire multitude of parameters describing the point/area where
the pulse movement starts, we will only keep those repeated in the most selected
cases and use them in the pattern description.

### Conclusion

In other words, this process can be automated. The matter is that I, personally,
do not know how much time this may consume. This is because I'm not a real programmer
myself, though I tried to be. I would like to suggest everybody interested in this
problem and wishing to participate in its solving to create a trading systems laboratory
and combine efforts and experiences in order to obtain a positive result.

As an initial objective, we can try to aim at creation an algorithm of selection
market situations with preset parameters. After having solved this task, we can
start developing the algorithm of market patterns description.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1426](https://www.mql5.com/ru/articles/1426)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Social Trading. Can a profitable signal be made even better?](https://www.mql5.com/en/articles/4191)
- [How to conduct a qualitative analysis of trading signals and select the best of them](https://www.mql5.com/en/articles/3166)
- [Synthetic Bars - A New Dimension to Displaying Graphical Information on Prices](https://www.mql5.com/en/articles/1353)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39352)**
(1)


![molanis](https://c.mql5.com/avatar/avatar_na2.png)

**[molanis](https://www.mql5.com/en/users/molanisfx)**
\|
11 Mar 2010 at 21:51

For a quick start use this tool [Strategy Builder](https://www.mql5.com/go?link=http://www.molanis.com/ "http://www.molanis.com/") to create EA. It's a visual environment to create EA and indicators. Really fast and easy. no coding is required.

There are also some examples and even a forum.


![MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://c.mql5.com/2/15/466_27.gif)[MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://www.mql5.com/en/articles/1496)

This is the third article from the series "MQL4 Language for Newbies". Now we will learn to use built-in functions and functions for working with technical indicators. The latter ones will be essential in the future development of your own Expert Advisors and indicators. Besides we will see on a simple example, how we can trace trading signals for entering the market, for you to understand, how to use indicators correctly. And at the end of the article you will learn something new and interesting about the language itself.

![How to Cut an EA Code for an Easier Life and Fewer Errors](https://c.mql5.com/2/14/441_19.png)[How to Cut an EA Code for an Easier Life and Fewer Errors](https://www.mql5.com/en/articles/1491)

A simple concept described in the article allows those developing automated trading systems in MQL4 to simplify existing trading systems, as well as reduce time needed for development of new systems due to shorter codes.

![Construction of Fractal Lines](https://c.mql5.com/2/14/210_2.png)[Construction of Fractal Lines](https://www.mql5.com/en/articles/1429)

The article describes construction of fractal lines of various types using trend lines and fractals.

![Practical Use of the Virtual Private Server (VPS) for Autotrading](https://c.mql5.com/2/14/373_44.png)[Practical Use of the Virtual Private Server (VPS) for Autotrading](https://www.mql5.com/en/articles/1478)

Autotrading using VPS. This article is intended exceptionally for autotraders and autotrading supporters.

[![](https://www.mql5.com/ff/si/0nfwvn6yhmgzf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F117%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dorder.expert%26utm_content%3Dorder.freelance%26utm_campaign%3D0622.MQL5.com.Internal&a=tunpwtbhegzufrqocbwiszessdutnobs&s=d9e7484e15300021b4066b1df77a94a1352f9e7c326d5113006bb4f6476bafeb&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=gxwrvmtitxthcxxntwzodxzfbqhwaurd&ssn=1769158412498671398&ssn_dr=0&ssn_sr=0&fv_date=1769158412&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1426&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Creation%20of%20an%20Automated%20Trading%20System%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176915841211479473&fz_uniq=5062793859873810699&sv=2552)

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