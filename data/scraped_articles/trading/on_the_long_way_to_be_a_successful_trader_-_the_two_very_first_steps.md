---
title: On the Long Way to Be a Successful Trader - The Two Very First Steps
url: https://www.mql5.com/en/articles/1571
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:40:54.365117
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=enjkkiroclyqmganmrjffhgkvcevcfbw&ssn=1769251252556396706&ssn_dr=0&ssn_sr=0&fv_date=1769251252&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1571&back_ref=https%3A%2F%2Fwww.google.com%2F&title=On%20the%20Long%20Way%20to%20Be%20a%20Successful%20Trader%20-%20The%20Two%20Very%20First%20Steps%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925125257737581&fz_uniq=5083036968452887475&sv=2552)

MetaTrader 4 / Trading


**Introduction**

It's all
about Money Management. You
probably have heard this dozens of times, some of you may even have implemented
some kind of Lot management in their EA or
trading strategies and believe that it will do, alone as it is.

Money
Management is a complex question and it does not get covered with some
linear/quadratic/fancy relation between volume traded and current balance or free
margin.

In order to
perform a correct Money Management one MUST understand what the concept behind
the two words is. Eventually you'll figure out yourself that the number of Lots
you are going to trade in your next trade should not be chosen by a function, the only variable
of which is how much money you currently have on your account.

Looking at
many EAs in the codebase I often recognize pieces of code coming from the article " _Fallacies, Part 1: Money Management is Secondary and Not Very Important_"
and I understand that many think that the article gives the final solution to Money
Management problem.

Well, as
far as I've loved that piece of paper when I was first approaching the Money
Management issue, it is my current opinion that it just raises the question and
shows a first rudimental approach increasing the amount of money traded as the
balance goes up, but it doesn’t touch at all the core of the problem of Money
Management.

The main
point of this article is to show a practical way to implement an effective MM.
This can be achieved only by using a certain kind of strategies that we need to
identify and describe first. In the following we’ll cover the basic concepts of how to build such strategy and we’ll point out the
common mistakes which always end up in draining a Trader’s account.

Writing
this paper I assume you've read the article " _Fallacies, Part 1: Money_
_Management is Secondary and Not Very Important_". If you haven’t, I
strongly recommend you to go there now ([http://articles.mql4.com/en/articles/1526](https://www.mql5.com/en/articles/1526))before continuing with this one.

**The Meaning**
**of Money Management**

A concept
in just two words: MANAGEMENT and MONEY.

"To manage?
comes from the Italian "maneggiare" which literally means to handle. Managing
something requires skills and knowledge and the intimate meaning of the word
management can probably be expressed as "handling with knowledge".

When
talking about Money Management we obviously talk about money. What we want is
to have a direct control (handling) over the way we invest our capital.

Now, when
we place an order we have control on its volume (Lots), no figure in "money" terms is directly involved. If we want to handle our money with knowledge, we
must be able to translate that volume to our deposit currency in order to
estimate the potential loss associated with the order.

**Successful**
**Traders**

We’ve been
told that statistically speaking Successful Traders never risk more than 2% of
their capital on a single trade.

It is a
simple rule indeed… why not to give it a try?

What you
have to do is just to quantify how much is 2% of your capital and avoid to risk
more than that sum next time you open a position… sounds stupidly easy… so why
didn’t you do it so far? :)

Again, you
open positions in Lots and you need a way to convert _“2% of your Capital"_ into
Lots. That’s what
we are going to learn in the following.

**If You Want**
**to Win You Must Be Able to Lose**

When
trading, your main goal is to increase your Capital. Profitable trades increase
your balance while losing ones decreases it.

At first
glance one can say: winning=good, loosing=bad.

Following
this reasoning one is driven to think that avoiding losses is a good starting
point on the long way to become a Successful Trader.

As a matter
of fact there are plenty of strategies and EAs which "solve" the problem of
losses by simply not dealing with it. Those strategies make no use of StopLoss
and let the loss grow until the Market comes eventually back in their favour
and the position becomes profitable.

EAs
developed on those strategies usually show very good profits and no losses at
all (win ratio is usually more than 95%). This behaviour is anyway stable for a
relative short period and when the loss comes it comes unexpected and huge…
sometimes one single series of losses is enough to drain the account empty. It's the typical mistake of the newbie that attempts to code his first Grail. Been there.

Systems
like this work fine until the Market has a strong trend in one direction but
becomes absolutely dangerous when the trend disappears or changes direction.

**_Ignoring_**
**_losses is definitely not the way to go if we want to see our Capital grow on_**
**_long/medium term period._**

We want our
strategies to be able to survive the toughest conditions we can find out there
so we do want to avoid using a system that works only for short periods
introducing the risk to lose most of our capital when things go bad.

Looking
better at the dynamic of capital growth we can write down the following simple
equation:

Profit = %Win \* AVGwin – %Loss \* AVGloss _(1)_

where:

> %Win
> = nr. profitable trades / nr. of total trades(percentage of profitable trades)
>
> AVGwin
> = total profit / nr. profitable trades(average profit)
>
> AVGloss
> = total loss / nr. losing trades(average loss)

Equation
(1) is valid in any circumstances, whichever trading system one uses.

Considering
that %Loss is equal to \[1-%Win\] we can write the equation (1) in this form:

Profit = %Win \* AVGwin – (1-%Win) \* AVGloss _(2)_

As you can
see the equation (2) is function of three variables, namely:

- %Win
- AVGwin
- AVGloss

At this
point the simple concept “winning=good, loosing=bad” doesn’t seem so obvious anymore… in fact
it sounds kind of wrong.

We can
pretty much look at the equation (2) as translation into formal math terms of
the say _“it’s not about being right or wrong rather how much you make when_
_you are right and how much you lose when you are wrong”_.

Both
equation and statement complicates the game but they introduce interesting
degrees of freedom which allow us to get into loosing trades as much as \[ (1-win%) \* AVGloss \] remains smaller than \[ %Win\* AVGwin \].

The equation
(2) is also well described in the article "Be In-Phase" by Mikhail Korolyuk
that you can find here ( [https://championship.mql5.com](https://championship.mql5.com/)). I warmly recommend you to study it cause it explains some key concept that
will help to improve your understanding of the dynamic of strategies/trades.

Now we know
that the success of your strategy is determined by a relation which is function
of three variables… to be successful we have to "drive" them in our favour and
maximize the profit.

But what do
we know about each one of these variables?

> **%Win**
> is a characteristic of the strategy used. Usually is not affected by the size
> of the positions opened. Is worth to point out that %Win is not known before
> running a few tests (on demo account or Strategy Tester). It can be increased by
> altering the strategy, by choosing better entry points and it may change as
> consequence of changes to TakeProfit and StopLoss.
>
> Practically
> speaking %Win is a variable on which we don’t have much control as it really
> depends on the intimate structure of our strategy.

> **AVGwin**
> is also tied to the strategy used. It can be changed directly by changing
> order’s Volume or TakeProfit levels or, more generally, the criteria for which
> the strategy closes profitable orders.
>
> Although you can play with it, you will always come to face the fact that the more you
> increase TakeProfit levels (or force your exit points up) the smaller your %Win
> will be. As we are trying to maximize the product \[ %Win\*\
> AVGwin \] we must be very careful in changing TakeProfit levels or we
> could end up in worsening the situation.
>
> Furthermore
> we have to notice that at the moment a position is opened we have no clue on
> how much the price will move in our favour; therefore we cannot really rely on
> strong AVGwin to maximize the profit given by equation (2).

> **AVGloss**
> depends of course on the strategy used but it can be "adjusted" by changing
> StopLoss levels and Volume of our orders (Lots).
>
> Acting on
> StopLoss or Lots has different effect on how our strategy performs. Let’s see
> it…

- Reducing StopLoss levels
results in smaller losses per single trade. A collateral effect could be
that more orders are open as positions are closed more often for StopLoss
and this leaves room for other positions to be opened. The effect on %Win
is pretty much unpredictable;
- Increasing StopLoss levels
results in bigger losses per single trade. A collateral effect could be
that the StopLoss levels are so high that they never trigger and when they
do the loss is too big for our account to stand it. More optimistically
the number of orders can be considerably reduced as positions are kept
longer. The effect on %Win is again unpredictable;
- Changing the volume of orders
results in smaller losses per single trade. Usually it does not affect the
%Win of a strategy.

Finally it’s
clear that out of the three variables the only one that we can control directly
BEFORE placing an order is AVGloss. Here is where Money Management kicks in.

Let’s look
again at equation (2):

Profit = %Win \* AVGwin – (1-%Win) \* AVGloss _(2)_

…it’s
pretty obvious that minimizing AVGloss we can increase our profit. **Managing**
**our losses is then the key to increase our profit.**

**Again:** we
cannot plan our gains as those depend on how much the Market moves in our
favour but we can always decide how much we are willing to loose before giving
up the position when things go wrong.

AVGloss =
TotalLoss / nr. losing trades

If we look
at the relation above we clearly see that in order to minimize AVGLoss we have
to minimize TotalLoss, which is the sum of every single loss cumulated by our
trading system. To keep down TotalLoss is sufficient to reduce to minimum the
loss on every single losing order.

**Cut the Loss and Let the Profit Grow**

Experience
shows that if you want to be successful in trading you must quickly get rid of
loosing positions and treasure profitable ones.

We just
learnt that we can play with Lots or SL levels to "adjust" AVGloss
variable. We’ve seen also there is not much we can do with %Win and AVGwin.

We need now
to find a criterion to choose SL levels and Lot
size in order to optimize the result of equation (2).

This is the
equation we are going to use to implement our Money Management strategy:

StopLoss \*
PipValue = Capital at risk _(3)_

where StopLoss is given in pips and PipValue is the
value of a single pip in the deposit currency.

PipValue is obviously tied to the volume of the order
and this relation will be the key for handling our money with knowledge.

Let’s talk about StopLoss first.

**There Is No Safe Trading without StopLoss**

Before we
come to describe how to implement Money Management in a practical way, it is worth
to mention something that kills a large number of Traders.

Many
beginners and experienced Traders trade without using any hard SL. Who
codes an EA tends to think that it will take care of everything, following the
market closely and babysitting open positions ready to close them when things
go wrong. Nothing could be more far from the truth.

In fact when you don’t place a hard SL you don’t really know when the
position will be closed in case the Market turns against you… you actually
cannot be sure that it will be closed at all!!!

How can you
pretend then to handle your money with knowledge?

**_A_**
**_strategy which makes no use of hard StopLoss CANNOT implement a sound Money_**
**_Management!_**

Such
strategies handle closure of position by watching the Market for particular
events, events that may never occur driving the account to bankruptcy. Using
such strategies/systems/EAs is not possible in fact to quantify the losses at
the moment of position opening. Losses on those strategies can be quantified
only on a statistical base.

Of course
there are exceptions: extremely complex EA characterized by a strong
ripetibility… but they will not be taken into consideration in the scope of
this article.

Many will
have to smash their head on it but at some point it will become pretty clear
that a strategy which makes use of hard SL is much more reliable than one that
doesn’t.

So, next time
you attempt to set up your strategy or code your EA keep in mind that you MUST
use hard SL and you have to address the issue of how to choose SL in an
effective way.

This will
be your first big step to the success.

**How to Choose the Proper StopLoss level**

This matter
alone would require pages of explanations, examples and talking. We will not
cover it in detail but just give some hint and underline what is really
important when talking about SL.

SL
level shall always be chosen looking at the current condition of the Market, so
called Price Action. Who uses fixed SL (say very common 20 or 50 pips)
has probably a very limited understanding of the Market and trading in general.
In fact SL shall be wide enough to avoid being wiped out in case the
Market retraces back against you… but not too wide otherwise one would have a
very limited return compared to the risk he is taking with the trade.

As a
general rule:

**_you_**
**_want to have as many support/resistance levels as possible between the current_**
**_Price and your SL_**

This will
ensure that if the Market turns against you it has a chance to slow down and go
back in the right direction before hitting your SL.

Of course
you can practically include a minimum number of S/R levels between current
Price and SL level to avoid having too wide SL.

_Let’s see a_
_practical example._

In the following H1 chart you can see the price rising
after a double top in a clear and strong movement (last two bars).

Say we want
to use round figures and buy at 1.2700.

If we use a
dumb strategy we set our SL 50 pips below the entry (see the red line) at
1.2650.

Note how
the SL falls just in the middle of the 1st bar… it is highly
probable that it will be wiped out in case a retracement occurs. But our strategy just uses 50 pips SL and does not take into account price action... too bad.

![](https://c.mql5.com/2/17/1.gif)

SL example H1 chart: Open price and 50 pips SL

Let’s see
the same moment in M15 chart.

It appears
very clear that a resistance level has been broken around 1.2575 and is also
evident that there is no support next to the chosen SL level while the Market is
stable way below.

![](https://c.mql5.com/2/17/2_1.gif)

SL example: setup on M15 chart

In fact with
a little help of Fractals a Successful Trader can identify four major supports
of interest:

- first is at 1.2566
- second at 1.2536
- third at 1.2529
- forth at 1.2525

Without
entering into too much detail the Successful Trader would just state that the
forth support level is the most reliable one, therefore he’ll set his SL right
there (continuous blue line). Of course SL could be set lower but this would
penalize the ROI of the trade. Remember that you want to keep SL as smaller as
possible!

So… after a
careful analysis of the price action the Successful Trader ended up with a SL
of 175 pips. Surely not one of the smallest… but he wants to take this trade
and be reasonably sure that he doesn’t end up loosing… and according to him
there is not much of a choice here.

Remember that
we instead placed our dumb 50 pips SL on the red line at 1.2650.

Let’s see
now how the two trades moved on…

![](https://c.mql5.com/2/17/3.gif)

SL example: trade on M15 chart

…there you
go! M15 chart.

Just after
we opened the position a retracement started and in a few hours wiped out the
dumb SL leaving a hole into our account :(

We’ve just lost.
Right after the order has been closed by SL the price starts climb again and in
less then one hour we see that the position would have been profitable if only
the SL was set a bit lower.

I bet you
have seen this many many times.

What
happened to the Successful Trader?

He is
probably laughing at us while he is gaining profit from a smooth trade.

As simple
as that: place your SL correctly and you can go on the beach with no worries. The Successful Trader knows that.

Again, **an**
**EA/strategy which uses fixed SL has very limited probability of success**
**as it does not take into full account what’s happening in the Market**.

If you are
using such strategy you should change your approach immediately and every time
you open a trade without studying where to place your SL you should slap
yourself hard in the face! :)

**REMEMBER:** the first step when attempting any trading is to analyze the Market and understand where to place our hard SL. Period.

**Volume Is**
**the Key**

Should be
clear by now that SL levels are pretty much tied to the Market and in
most of the cases we are forced to "accept" SL levels where they come and
not where we would like them to be.

Let’s see
again the equation (3) introduced before:

SL \*
PipValue = Capital at risk _(3)_

If we want
to trade like a Successful Trader risking not more than 2% of our Capital on a
single trade, means that SL (in pips) of any order by the value of
one single pip (in deposit currency) shall be smaller than 2% of our Capital.

SL \*
PipValue <= Capital \* 0.02 _(4)_

**Since SL is imposed by the Market the only degree of freedom we**
**have left to adjust our Risk is to choose a proper PipValue.**

Reversing
the equation (4) we can extract PipValue:

PipValue
<=(Capital \* 0.02) / StopLoss _(5)_

Given a
fixed Risk and a choosen StopLoss we can quantify PipValue.

_Let’s make_
_an example._

Suppose we
have a Capital of 10 000 EUR and we want to take a maximum risk of 2%. That means
we are willing to risk 200 EUR on our next trade. If in the current Market we
can set SL at 80 pips we are in the following situation:

PipValue = 200
/ 80 = 2.5 EUR

In other
words we know we may end up the trade losing 80 pips and to limit the loss to
200 EUR we have to open a position with a Volume for which each pip is worth
2.5 EUR.

Ok, in our
example we know that PipValue must be 2.5 EUR…

…but which
is the relation between PipValue and Lot size?
How do we get each pip to be worth 2.5 EUR?

Let’s
define _NominalPipValue_ as the value of a single pip in deposit currency when
Volume=1.00.

NominalPipValue
can be calculated knowing the current exchange rate of the chosen pair.

Let’s say
you are trading EURUSD on a standard account with leverage 1:100. If your
deposit currency is USD then NominalPipValue will always be 10$, no matter
which is the exchange rate.

If your
deposit currency is EUR then you must consider the exchange rate, more
precisely the inverse of the exchange rate.

NominalPipValue
= (10 / exchange rate)

If the
current exchange rate EURUSD is for example 1.3333 then you’ll have:

NominalPipValue
= (10$ / 1.3333) = 7.519 EUR

As you can
see it’s pretty easy. Things get more complicated when you trade a pair like
USDCHF and your deposit currency is in EUR or GBPJPY and your deposit currency
is either EUR or USD. In fact in these cases you have to take into account a
double exchange rate.

Fortunately
we don’t need to do that as MetaTrader has a specific function that easily tells
us NominalPipValue:

NominalPipValue
= MarketInfo(Symbol(),MODE\_TICKVALUE) _(6)_

This will
work whichever pair your are trading and whichever your deposit currency is… even
too easy now :)

**ATTENTION:** NominalPipValue
changes with exchange rates so its value at order’s opening is different from
the one at order’s closure. Actual profit/losses are calculated using exchange
rate at order’s closure.

Unfortunately
we do not know this value at the moment in which we open the order… anyway for
variations of exchange rate in the order of typical SL the difference in NominalPipValue
at opening and closure of the position is small enough to be ignored for the
purpose of our calculation (less than 1% on for SL <= 100pips).

At this
point we know that when buying/selling 1.00 Lots PipValue equals
NominalPipValue. More in general we can say:

1.00 /
NominalPipValue = X / PipValue _(7)_

where “X”
is the nr. of Lots to trade.

We extract
X from the equation (7) turning it into:

X =
PipValue / NominalPipValue _(8)_

**Good, we**
**reached our target. Now we know how to convert Risk Capital in Lots!!!**

_A final_
_example will clarify the procedure._

Let’s say keep going with our trade: risk 200 EUR when SL is 80 pips and the current exchange
rate of EURUSD is 1.3333.

Remember
that what we want to get is “X”, the nr. of Lots to trade!

According
the equations described earlier we have:

X =
PipValue / NominalPipValue

which according equations (5) and (6) corresponds in general to:

X = ( Risk
Capital / StopLoss ) / MarketInfo(Symbol(),MODE\_TICKVALUE)

or, for a
standard EUR account with leverage 1:100

X = ( Risk
Capital / StopLoss ) / (10 / exchange rate)

Using the
numbers of the example:

X = ( 200 /
80 ) / ( 10 / 1.3333 ) = 2.5 / 7.5 = 0.33

We’ll open
then a position with Volume 0.33!

This will ensure that each pip is worth exactly 2.5 EUR. DONE :)

**Conclusions**

We've just seen a practical method for implementation of a sound MM system.

This article shows that the size of the order is not the only variable to be controlled/considered while implementing MM but also the size of SL plays a very important role.

We went
through some theory and identified a bunch of equations useful to calculate the
proper size of a position as function of the available Capital (FreeMargin) and SL. Moreover we stressed the importance of trading using hard SL and pointed
reader’s attention on the fact that SL should NEVER be set in a fixed size
irrespective of price action (Market’s conditions).

In other
words a Trader should never gamble. A Trader should never gain from his trades
less than expected and never lose more than planned.

Right, a Trader MUST have a plan.

As recalled
in the article the plan of many Successful Traders is to risk less than 2% of
available Capital (FreeMargin) on each trade… personally I don’t quite agree on
the 2% rule… but that will be subject of another article :)

Anyway, one
thing that Successful Traders do for sure is to split the preparation of a
trade into two simple operations:

1)identify
the proper SL

2)calculate
the size of the position by spreading the potential loss equally on each pip of
the SL

Once this
is done the trade may win or lose… If it’s a winning one, then cheers… If it’s
a losing one nobody gets hurt cause the possible loss was taken into account
and handled with knowledge.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39547)**
(16)


![Avery Horton](https://c.mql5.com/avatar/2014/4/535FD084-E727.png)

**[Avery Horton](https://www.mql5.com/en/users/therumpledone)**
\|
9 Sep 2009 at 02:25

"Look,
for example, at this elegant little experiment. A rat was put in a
T-shaped maze with a few morsels of food placed on either the far right
or left side of the enclosure. The placement of the food is randomly
determined, but the dice is rigged: over the long run, the food was
placed on the left side sixty per cent of the time. How did the rat
respond? It quickly realized that the left side was more rewarding. As a result, it always went to the left, which resulted in a sixty percent success rate. The rat didn't strive for perfection.
It didn't search for a Unified Theory of the T-shaped maze, or try to
decipher the disorder. Instead, it accepted the inherent uncertainty of
the reward and learned to settle for the best possible alternative.

The experiment was then repeated with Yale undergraduates. Unlike
the rat, their swollen brains stubbornly searched for the elusive
pattern that determined the placement of the reward. They made
predictions and then tried to learn from their prediction errors. The
problem was that there was nothing to predict: the randomness was real.
Because the students refused to settle for a 60 percent success rate,
they ended up with a 52 percent success rate. Although
most of the students were convinced they were making progress towards
identifying the underlying algorithm, they were actually being
outsmarted by a rat."
\- P64 _**"HOW WE DECIDE"**_

What I realized was all conventional trading wisdom is "wrong".

I have distilled trading into the following:

**ALL YOU NEED TO KNOW ABOUT TRADING**

- Price either goes up or down.
- No one knows what will happen next.
- Keep losses small and let winners run.
- POSITION SIZE = RISK / STOP LOSS
- The reason you entered has no bearing on the outcome of your trade.
- You can control the size of your loss (skill) but you can't control the size of your win (luck).
- You need to know when to pick up your chips and cash them in.

Expectancy = (Probability of Win \* Average Win) - (Probability of Loss \* Average Loss)

You can not control the probabilities of wining or losing.

You can not control your average win size.

The only part of the equation of the equation that you can control is your average loss size.

**THE ILLUSION OF CONTROL**

_"Individuals appear hard-wired to overattribute success to skill, and to underestimate the role of chance, when both are in fact present."_

\[Langer, E. J., The Illusion of Control, Journal of Personality and\
\
Social Psychology 32 (2), 311-328 (1975)\]

**FINANCIAL MANAGEMENT**

_"After a full cycle of rise and fall after which stocks were valued just where they were at the start, all his clients lost money (Don Guyon, 1909)._

_Many academic works suggest that most managers underperform "buy-and-hold" strategy; persistence of winners is very rare, etc._

_Most funds consistently fail to overperform random strategies (dart throwing)."_

**OVER-OPTIMIZATION**

**Rats beat humans in simple games**

**People makes STORIES!**

_"Normal people have an "interpreter" in their left brain that takes all the random, contradictory details of whatever they are doing or remembering at the moment, and smoothes everything in one coherent story. If there are details that do not fit, they are edited out or revised!"_

(T. Grandin and C. Johnson, Animals in translation (Scribner,

New York, 2005)

[PDF here](https://www.mql5.com/go?link=http://www.er.ethz.ch/presentations/Illusion_of_control_Zurich_CCSS-conf19Aug08.pdf "PDF here")

**The rat will beat you if you do not understand this.**

![David Lynch](https://c.mql5.com/avatar/avatar_na2.png)

**[David Lynch](https://www.mql5.com/en/users/pipwatcher)**
\|
2 May 2010 at 17:01

**Zypkin:**

**SEBAZ:**

Good jobs pals! This is really great.

A lot of people think that trading forex is just trying to make money, without considering really what will happen if you fail to secure what you have made. I would like you to comment on the good strategies to adopt while scalping using the 5, 15 and 30 minutes chart window.

Keep moving.

TONY.

Tnx for your comment Tony.

Well, I'm really not a big fan of scalping cause I think it is "physiologically" dangerous on the Forex Market.

What I mean is that in live trading your plan of gaining i.e. 5 pips can be easily spoiled by a requote of the server.

I'll make you an example:

think you have in your hands the M5 chart of today (I mean actually all the bars that still have not come are already known to you) and you can open your orders knowing exactly every movement of the market.

Now, you want to get the most out of it and scalp on M5 to get profit from ALL the movements... after all you know them already, why you shoulndt get advantage of it? :)

When doing scalping in M5 it's fair to assume that your target are 5 to max 10 pips, right?

So now place your orders, chart in your hands, and let's see what happens.

Well... most likely many of your requests will be requoted, both in opening and closure of the trades and this will result in a disaster.

Remember you have 2 pips spread... add 2 pips requote on opening and 1 pip on closure and your 5 pips profit are gone already!!!

Do you get my point? :)

I think scalping works fine only on the strategy tester where no requote is applied and where transaction are made instantly. In the real world things are different.

One can overcome this with pending orders (which I consider a superior way of trading) and as a matter of fact if you set your openings and TP with a pending order you can be sure you'll get it right. Question is... without your magic chart of the future, do you have a system for scalping that gives you 5 pips of margin for placing a pending order? :)

You see that the question is still open.

Anyway, I strongly recommend not to trade on any timeframe lower than M15 cause this is the minimum for proper selection of SL. I hope I can explain this better in another article. Also movements on M15 are a bit wider and this will allow you coping with requotes most of the time.

Hope this short answer gave you some good idea :)

Cheers, _Andrea_

I have RARELY been re-quoted at my current broker, but always was [requoted](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes "MQL5 Documentation: Requote") at my other. I think if this is your only reason for not scalping...then you should get a new broker...in my opinion!

David


![Luis Leal](https://c.mql5.com/avatar/2013/12/52A1FC3B-E443.jpg)

**[Luis Leal](https://www.mql5.com/en/users/firstdimension)**
\|
16 Apr 2011 at 19:29

The solution is not in the SL and TP.

Supose that to make a minimal FX trading team with success you need:

Fx is on activity 24 hours a day > We need to stay in it 24 hours a day.

An FX team needs several kind of intelectual capital: management staff, strategical staff, director staff and operational staff. As we will need 24 hours available, we will need a minimum set of four members in each time frame (4 members x 3 time frame) = 12, plus one of each for the backup... we will need 12 + 4=16 members minimum...

Have you this amount of time or effort to invest in FX? The FX as all the investments have your own level of information, the war is on that level, more information, more success.

The first level of MM is at the [project](https://www.mql5.com/en/articles/7863 "Article: Projects allow you to create profitable trading robots! But it is not exactly") startup and in your own conception.

Minimum amount as equity... $5M :) All the rest is food for shark

It's only an appart of a newbie, sorry for the poor English

Luís


![blackmore](https://c.mql5.com/avatar/avatar_na2.png)

**[blackmore](https://www.mql5.com/en/users/blackmore)**
\|
15 Nov 2011 at 12:51

this is a good article however is the author still around..as I may have some coding questions (which I hope he can answer?)

basically, I want to put as much of this article has to offer into my stoploss, risk [money management](https://www.mql5.com/en/articles/4162 "Article: Money Management by Vince. Implementation as a MQL5 Wizard Module") into my EAs, or even as library which can be use on ends.

I am very impressed by the well-thought of equations, nominal pip values, etc. unfortunately, this article does not come with any codes. so if anyone can lend a hand, really appreciated, and if the author is seeing this, wherever you are, Im here.

![decipheringme](https://c.mql5.com/avatar/avatar_na2.png)

**[decipheringme](https://www.mql5.com/en/users/decipheringme)**
\|
23 Jan 2017 at 15:10

Great article, [@Andrea Bronzini](https://www.mql5.com/en/users/zypkin1). Thank you very much for writing this up.

It's very interesting to single out SL as the only thing a trader can practically use to turn certain odds in his or her favor. And it provides a practical handle on how to set stop losses. Invaluable and it makes this article a must-read for any trader, I think. I love the idea of using support and resistance levels for setting SL values. But why not using those for TP levels as well?

Am I correct to understand that this article argues that using support and resistance levels for SL will positively benefit the bottom line as it actually impacts the balance of loss vs gains. But TP levels do not impact this same balance?

My second question is then, how to best determine TP levels (outside of blindly optimizing)?

Cheers!

![Superposition and Interference of Financial Securities](https://c.mql5.com/2/17/800_14.gif)[Superposition and Interference of Financial Securities](https://www.mql5.com/en/articles/1570)

The more factors influence the behavior of a currency pair, the more difficult it is to evaluate its behavior and make up future forecasts. Therefore, if we managed to extract components of a currency pair, values of a national currency that change with the time, we could considerably delimit the freedom of national currency movement as compared to the currency pair with this currency, as well as the number of factors influencing its behavior. As a result we would increase the accuracy of its behavior estimation and future forecasting. How can we do that?

![Using Neural Networks In MetaTrader](https://c.mql5.com/2/16/777_40.gif)[Using Neural Networks In MetaTrader](https://www.mql5.com/en/articles/1565)

This article shows you how to easily use Neural Networks in your MQL4 code taking advantage of best freely available artificial neural network library (FANN) employing multiple neural networks in your code.

![Interaction between MеtaTrader 4 and MATLAB Engine (Virtual MATLAB Machine)](https://c.mql5.com/2/16/782_20.gif)[Interaction between MеtaTrader 4 and MATLAB Engine (Virtual MATLAB Machine)](https://www.mql5.com/en/articles/1567)

The article contains considerations regarding creation of a DLL library - wrapper that will enable the interaction of MetaTrader 4 and the MATLAB mathematical desktop package. It describes "pitfalls" and ways to overcome them. The article is intended for prepared C/C++ programmers that use the Borland C++ Builder 6 compiler.

![Alert and Comment for External Indicators](https://c.mql5.com/2/17/789_15.gif)[Alert and Comment for External Indicators](https://www.mql5.com/en/articles/1568)

In practical work a trader can face the following situation: it is necessary to get "alert" or a text message on a display (in a chart window) indicating about an appeared signal of an indicator. The article contains an example of displaying information about graphical objects created by an external indicator.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/1571&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083036968452887475)

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