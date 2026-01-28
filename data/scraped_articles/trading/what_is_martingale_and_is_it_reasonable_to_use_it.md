---
title: What is Martingale and Is It Reasonable to Use It?
url: https://www.mql5.com/en/articles/1481
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:42:21.815727
---

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/1481&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083055977978139660)

MetaTrader 4 / Trading


### What Is Martingale?

If you write "martingale" in a search engine box, it will return a large
number of pages with the description of this system. It is interesting that among
others you will meet web-sites of online casinos, which assure that this system
works, all you need is entering your credit card number to start scooping up money.
What is strange - are the casinos ready to give their money such easily? If the
Martingale really works so good, then why have not all the casinos turned bankrupt
yet?

So, what is Martingale? Here is the definition from Wikipedia [http://ru.wikipedia.org/wiki/Martingale_system](https://en.wikipedia.org/wiki/Martingale_(betting_system) "http://ru.wikipedia.org/wiki/Martingale_system"):

The Martingale is a betting system in gambling. The meaning is the following:

- A game starts with a certain minimal bet;
- After each each loss the bet should be increased so, that the win would recover
all previous losses plus a small profit;
- In case of win a gambler returns to the minimal bet.

(Translated from Russian Wikipedia by MetaQuotes Software Corp.)

More information is here: [https://en.wikipedia.org/wiki/Martingale\_system](https://en.wikipedia.org/wiki/Martingale_system "https://en.wikipedia.org/wiki/Martingale_system")

### Where Is Martingale Used?

The simplest gamble for analyzing the Martingale is chuck-farthing. The chances
to win and to lose are equal - the gambler wins if a coin comes up heads and loses
if the coin comes up tails. The Martingale system for this game works in such a
way:

- Start the game with a small bet;
- After each loss double the bet;
- In case of win return to the minimal bet.

The Martingale can also be used in playing the roulette, betting on red or black.
The chances are less than 50/50, because there is also Zero, still very close to
it.

As applied to trading, the following variant of the game can be used. Analogous
to tossing a coin we open a position in any direction (short or long) with stop-loss
and take-profit equally distant from the trade price. As we open the position in
a random direction, the probability of profit and loss is analogous - 50/50. So
in this article I will describe only the classical problem of tossing a coin with
doubling the bet at a loss.

### Mathematical Part

Let us conduct a mathematical calculation of the dependence of the loss probability
on the possible profit at the game with a coin using the Martingale system. Let
us introduce the following symbols:

- _Set_ – a set of tosses, ending by a winning one. I.e. all tosses except the last one are losing. At the first toss
the bet is minimal, at each next toss in the set the bet is doubled;
- _Q_ – initial deposit;
- _q_– price of the starting bet;
- _k_– maximal number of tosses (losing) in the set, leading to bankruptcy (suppose after k toss the deposit is equal
to zero).

As we double the bet after each losing toss, we can derive the following equation:

![](https://c.mql5.com/2/15/formula1.jpg)

Each set with the amount of tosses less than k-1 returns the profit q. As the probability of winning at a toss
= ½, the average set length is 2\*. Let us label by P(N) – the probability that we will not turn bankrupt within
N tosses. As N tosses constitute approximately N/2 sets (the average set length is 2), and the probability to
win in the set is (1/2)^k-1 , then

![](https://c.mql5.com/2/15/formula2.jpg)

We get the function of the win dependence on N. But the total number of tosses (N)
is not informative enough, so let us try to bind N with an expected profit. Suppose,
in the result we want to double our capital. As in set each we win q=Q/(2^k-1),
the total profit is calculated according to the rule of the compound interest (more
information about compound interest is [here](https://www.mql5.com/go?link=https://www.google.com/search?hl=en&newwindow=1&q=compound+interest "http://www.google.com/search?hl=en&amp;newwindow=1&amp;q=compound+interest")):

![](https://c.mql5.com/2/15/formula3.jpg)

After simple transformations we get the following formula for N:

![](https://c.mql5.com/2/15/formula4.jpg)

After calculating the probability of the profit P(N) using the equities (1)-(2)
we get the following results:

If we consider N a noninteger (do not round off the results of the equity (2) to
a whole number), then P(N) does not depend on k and is equal to 1/2 (you can easily
verify it, inserting (2) into (1) and using the simplest properties of logarithms).
I.e. using the Martingale does not provide any advantages; we could as well bet
all our capital Q and the winning probability would be the same (1/2).

### Conclusions of the Mathematical Part

Frankly speaking, at the beginning of preparing calculations for this article I
expected that the Martingale would increase the probability of loss. It appeared
to be wrong and the risk of loss is not increased. Still this article very vividly
describes the meaninglessness of using the Martingale.

### Expert Advisor

After getting the above formulas, the first thing I did was writing a small program,
emulating the process of playing chuck-farthing and composing the statistics of
the losing probability (P) dependence on the coefficient k. After the check I found
that the program results (it can be called "an experiment") coincide
with mathematical calculations.

Of course, the ideal variant would be writing an Expert Advisor, trading by the
same rules as in chuck-farthing and making sure that theoretical and experimental
data are identical. But it is impossible because the starting bet is calculated
using the formula:

![](https://c.mql5.com/2/15/formula6_1.jpg)

And in the Forex we can "bet" only a sum multiple of 1/10 of a lot. That
is why it is impossible to write an Expert Advisor, vividly proving the above formulas.
Nevertheless, for completeness of analysis, we still can write an Expert Advisor,
using the Martingale. But here the starting bet will be fixed - 0.1 of a lot. Analogous,
the bet will be doubled at a loss and return to the starting one at profit. As
described in the beginning of the article, a trade will be opened in the following
way: a trade is opened in a random direction with the probability 50%, stoploss
and takeprofit are fixed and equally distant.

![](https://c.mql5.com/2/15/screen7.jpg)

The above screenshot displays the results of testing this Expert Advisor. You see,
though the general direction of the curve is upwards, from time to time it suffers
large dips. As a result of the last dip the Expert Advisor stops trading, because
the balance is not enough for the next bet with a doubled lot. And at the moment
of stop the balance is positive - here is the difference from the theoretical calculation
in "the mathematical part".

P.S. The files attached contain the screenshot of all necessary mathematical calculations
and the Expert Advisor.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1481](https://www.mql5.com/ru/articles/1481)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1481.zip "Download all attachments in the single ZIP archive")

[martin.mq4](https://www.mql5.com/en/articles/download/1481/martin.mq4 "Download martin.mq4")(2.21 KB)

[solution.jpg](https://www.mql5.com/en/articles/download/1481/solution.jpg "Download solution.jpg")(202.27 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Displaying a News Calendar](https://www.mql5.com/en/articles/1502)
- [Displaying of Support/Resistance Levels](https://www.mql5.com/en/articles/1440)
- [A Method of Drawing the Support/Resistance Levels](https://www.mql5.com/en/articles/1439)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39356)**
(8)


![alexander supertrade](https://c.mql5.com/avatar/avatar_na2.png)

**[alexander supertrade](https://www.mql5.com/en/users/supertrade)**
\|
31 Mar 2011 at 05:06

How to implement profitable martingale:

1 - start with low base lots (0.1 or less)

2 - high win rate.. at least 70%

3 - low consecutive losses (minimize z-score)

4 - semi-martingale.. rather then strictly doubling the number of lots.. allow this factor to be varied based on market/performance.. in my experience, doubling is a very aggressive approach

5 - equity stoploss.. force martingale to reset to minimum lots if a given equity drawdown is observed.. you will take a large hit, but still be in the green & in the game

..I have such a system trading on a live account.. due to the limitations above it is not hugely profitable but has been running for over a year with a constant return on capital

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
7 Feb 2012 at 10:13

this is a Maringale System and you can use very good. but you sell its helps and then i will send 3 next days EA for you. password PDF for openning is "business".


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
7 Feb 2012 at 10:17

## 2

![Jon Grah](https://c.mql5.com/avatar/2014/3/43393_1322257541.png)

**[Jon Grah](https://www.mql5.com/en/users/4evermaat1)**
\|
13 May 2012 at 08:30

its funny the articles that are written on martingale. It is true that without an edge, even 0.50% above 50/50, you would lose over time (over a continuous series of repititions). This was also proven in an article detailing [the gambler's fallacy](https://www.mql5.com/go?link=https://wizardofodds.com/gambling/betting-systems/ "https://www.mql5.com/go?link=https://wizardofodds.com/gambling/betting-systems/") (see follow up also [here)](https://www.mql5.com/go?link=https://wizardofodds.com/ask-the-wizard/betting-systems/gamblers-fallacy/ "https://www.mql5.com/go?link=https://wizardofodds.com/ask-the-wizard/betting-systems/gamblers-fallacy/") . The good news is that the financial markets can only go so far in any particular direction before a pullback. You can use this fact alone to your advantage (edge). I think the largest single direction trend a market ever made was EURUSD 2000 pips without a pullback of at least 38.2%, There was another move of 3600 pips, but there were significant retracements in-between. Even better news is that martingale is not required to become profitable in the markets. There is a [spreadsheet where](https://www.mql5.com/go?link=http://www.traderslaboratory.com/forums/forex-trading-laboratory/11152-dollar-cost-averaging-spreadsheet-alternative-method.html "https://www.mql5.com/go?link=http://www.traderslaboratory.com/forums/forex-trading-laboratory/11152-dollar-cost-averaging-spreadsheet-alternative-method.html") you can work out the numbers yourself and see.

![Proximus](https://c.mql5.com/avatar/2014/3/83537_1375629221.jpg)

**[Proximus](https://www.mql5.com/en/users/proximus)**
\|
24 Aug 2013 at 03:00

It works if the net profit factor is above 1 and the win rate is higher than 50%, martingale is a double or nothing either doubles your money or doubles your losses, so if you have a 60% win rate with 1:1 RR ratio you can use it safely, if not then dont.

Whats funny about forex that you dont start from 50% win rate from the start because the market is changing not a fix probability set like a roulette or blackjack game.So if you start it like a betting system you will have like 40% win rate with 1:1 RR if you take trades random, maybe on the 9999999999999999999999th trade you hit 49.9% but thats still not enough.So it is better to filter out crappy trades first and then increase your win rate to be martingale compatible! And this is the advantage of investing vs gambling, you can filter out bad trades, on the roulette or blackjack you cant filter out bad hands or spins unless you cheat, but surely not the statistical way!!

This is how my 60% win rate, real martingale system looks like, and how it should suppose to look like, on LEVEL 7 settings (2^7)

Here are my martingale type systems:

**1) CLASSICAL MARTINGALE** AFTER 567 TRADES (60% WR, 1:1 RR)

[![](https://c.mql5.com/3/54/martin_small.gif)](https://c.mql5.com/3/54/martin.gif)

As you can see after 500 trades it barely hit LEVEL 7 and even if we would lost that we would lose only half of the profit and continue from there to grow it back!

Of course you need a big account for this like one that can support like 10 lot size trades to be only 1% account risk, but statistically its very improbable to blow your account since its only 1% risk versus huge potential gains...The martingale presented in this article is BS with like 40-45% win rate which is sadly not enough, not even 50% is, must be 51 or higher...

**2) PROGRESSIVE DYNAMIC GROWTH MARTINGALE** (60% WR, 1:1 RR)

[![](https://c.mql5.com/3/54/growth_small.gif)](https://c.mql5.com/3/54/growth.gif)

**3) PROGRESSIVE STATIC GROWTH MARTINGALE** (60% WR, 1:1 RR)

[![](https://c.mql5.com/3/54/testergraph_small.gif)](https://c.mql5.com/3/54/testergraph.gif)

**4) ANTI MARTINGALE or INVERSE MARTINGALE** (60% WR, 1:1 RR)

[![](https://c.mql5.com/3/54/testergraphfff_small.gif)](https://c.mql5.com/3/54/testergraphfff.gif)

enjoy and good programming ;)

![Mathematics in Trading: How to Estimate Trade Results](https://c.mql5.com/2/14/442_223.gif)[Mathematics in Trading: How to Estimate Trade Results](https://www.mql5.com/en/articles/1492)

We all are aware of that "No profit obtained in the past will guarantee any success in future". However, it is still very actual to be able to estimate trading systems. This article deals with some simple and convenient methods that will help to estimate trade results.

![Construction of Fractal Lines](https://c.mql5.com/2/14/210_2.png)[Construction of Fractal Lines](https://www.mql5.com/en/articles/1429)

The article describes construction of fractal lines of various types using trend lines and fractals.

![Breakpoints in Tester: It's Possible!](https://c.mql5.com/2/14/203_15.jpg)[Breakpoints in Tester: It's Possible!](https://www.mql5.com/en/articles/1427)

The article deals with breakpoint emulation when passed through Tester, debug information being displayed.

![MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://c.mql5.com/2/15/466_27.gif)[MQL4 Language for Newbies. Technical Indicators and Built-In Functions](https://www.mql5.com/en/articles/1496)

This is the third article from the series "MQL4 Language for Newbies". Now we will learn to use built-in functions and functions for working with technical indicators. The latter ones will be essential in the future development of your own Expert Advisors and indicators. Besides we will see on a simple example, how we can trace trading signals for entering the market, for you to understand, how to use indicators correctly. And at the end of the article you will learn something new and interesting about the language itself.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/1481&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083055977978139660)

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