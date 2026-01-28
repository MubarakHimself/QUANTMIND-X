---
title: Money Management Revisited
url: https://www.mql5.com/en/articles/1367
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:20:57.386747
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/1367&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069364202059006934)

MetaTrader 4 / Trading


Epigraph:

![](https://c.mql5.com/2/13/4mmfrm00.png)

### Introduction

Trading activity can be divided into two relatively independent parts. The first part called trading system (ТS) analyzes the current situation, makes decision to enter the market, defines position type (buy\\sell) and moment of market exit. Amount of funds used in each deal is defined by the second part called money management system. In this article, we make an attempt to analyze some MM strategies depending on changes in their parameters. Simulation method has been selected for analysis. However, results of analytical decisions are also considered in some cases. Tools of analysis are МТ4 trading terminal and Excel. The libraries providing pseudorandom number generation (PRNG) \[1\], statistical functions \[2\] and module for transmitting data from МТ4 to Excel \[3\] are additionally used.

It is assumed that any trading activity (TA) has some degree of uncertainty. In other words, TA parameters are known with some degree of accuracy and they can never be defined precisely. The strategy of "random bets" can serve as a good and, perhaps, the simplest example of such TA. Using this strategy, a trader randomly (for example, by tossing a coin) stakes on whether one currency rate will rise or fall for a certain number of points relative to another one. Most probably, generation of currency rates is not related to the results of tossing a coin. Therefore, we have a TA, in which deal results are not related to each other (Bernoulli ones). Besides, we are unable to define the result of the next coin toss, as well as predict if it matches the currency direction in the future. We only know that the match will approximately occur in _50_ cases out of _100_ ones in case of sufficiently large number of attempts. Many traders believe that their trade is different from that. Perhaps, they are right but first we will look at this particular case. Then, we will analyze behavior and performance of the system, in which more than _50_ cases out of _100_ ones can be predicted.

Structurally, the article is arranged, so that the most interesting MM parameters are analyzed first using "theoretical" examples. Next, we will try to model the behavior of MM having data similar to the real Forex trading conditions. It should be noted that no particular TS will be analyzed. It is assumed that whatever TS is used, it is merely provides us with data on wins and losses with a specified probability, as well as preset win and loss values. Matters relating to the definition of independence (Bernoulli) of the actual deals' results and evaluation of TS stationarity in time are not considered here.

As already mentioned, the simulation will be used. The essence of simulation is in the fact that the result of the next bet (win or loss) is defined based on the generation of pseudorandom numbers with preset parameters. The bet size is defined by the selected ММ strategy. In a case of a loss, the placed bet is subtracted from the trader's current funds. In case of success, the funds are increased. A specified number of deals is simulated and the total results are calculated afterwards. Then the process is repeated multiple times (from a few hundreds to several hundred thousands), after which the results are averaged in the most appropriate way.

### Some Basic Terms and Abbreviations

The concept of Terminal Wealth Relative ( _**TWR**_) should be mentioned first. It represents the total profit from the series of transactions as a multiplier to the initial capital. In other words, if we divide the final funds by the initial ones, we will obtain **_TWR_**. For example, if the profit comprises _12%_, _**TWR** =1.12_. if the loss comprises _18%_, _**TWR** =0.82_. It is possible to compare the results of various TAs regardless of the absolute value of the initial funds. The term **_TWR_** is used by analogy with \[4\], \[5\] and \[6\].

The next important thing is the concept of "win". Any result is considered to be a win if its value exceeds the initial one. In other words, win is a case when _**TWR** >1_. Accordingly, the loss is a result that does not satisfy the mentioned condition, i.e. _**TWR** <=1_. Thus, the case when the end funds are equal to the initial ones, _**TWR** =1_, are also considered to be a loss. But this case will be considered separately if necessary. In addition, there is the concept of "loss" indicating the loss of funds, whether as a result of a single transaction or a series of deals, followed by inability to continue trading activity. For example, it may be the loss of all funds ( _**TWR** <=0_) or having the funds that are less than a certain minimum (security deposit).

Now, let's consider conventional signs used in the article. Winning probability is denoted by _**p**_ symbol. Its normal dimension is unit fractions. The same applies to loss probability, _**q** =1- **p**_. Total number of deals - **_N_**, number of profitable deals - **_V_**, while number of loss-making ones - **_L_**. The size of winning deals in absolute terms are defined as **_a_**, while the size of loss-making ones - as **_b_**, profit/loss ratio is **_k= a/b_**. In case we discuss the sizes of profitable\\loss-making deals relative to the funds size, the symbols are **_a%_** _and_ **_b%_** respectively. Ratio of **_a%_** _to_ **_b%_** is **_k_**. Bet size relative to the funds is denoted as **_f_**.

Probability of events received during the calculations is denoted as **_Prob_**. Other signs will also be implemented if necessary.

Besides, there is no distinction between "bet" and "deal" concepts in the article. They refer to the same thing - a single trading operation (SТО). A series of such SТОs is called a game or trading depending on the context. Although many traders do not like it, the word _game_ properly characterizes uncertainties occurring during trading activity.

### Historical Background

The properties of a TS based on tossing a coin have been researched for quite a long period of time. In its most simple form it was a gamble called "heads or tails". The game is played by two players who initially have some funds. A coin is tossed and a player who called the face-up side receives some part of another player's funds. Otherwise, the player has to give a part of his or her own funds. It is the classic "gambler's ruin" problem for mathematicians. It is studied fairly well and results depending on the initial parameters are well-known.

Fundamental results of solving the problem "of the ruin of the gambler playing against a very wealthy opponent" are more important for traders. A very wealthy opponent is represented by a dealing center (DC) here. If you are sure that you do not act against the DC, you may consider all other Forex participants to be a very wealthy opponent.

What is the problem in this kind of game? We will not discuss the cases of the infinite game. No one is able to play infinitely and possible outcomes are deplorable anyway. Let's consider that the game is finite and consists of **_N_** deals. Suppose that probability of your guessing right is _**p** >0.50_, while the values of both wins and losses are equal to each other, _**a** = **b**_. Let's also suppose that you want your win for **_N_** deals to be the largest of possible ones.

The best strategy to achieve the maximum profit in this game is to place the maximum possible bet, i.e. all the funds your have. However, it has been mathematically proven that in this case the probability of ruin can be calculated by the equation _1-( **pN**)_ and that probability depends on **_N_**. \[7\]\[8\] The greater the value of **_N_**, i.e. the number of bets, the higher the probability of ruin. Consequently, ruin is least probable in case of _**N** =1_ in case the game is mandatory. If the game is not mandatory, the most winning strategy is not to play, as the probability of ruin is zero.

> Note: there is one common misconception. Many believe that if game odds are equal then their possible gain is approximately zero. This would be the case if they played against an opponent having a similar volume of funds.

In real trading, that means that even if you are able to win in _2_ cases out of _3_ ones while putting all your funds at stake each time, the following thing will happen. The probability of ruin during the first deal is equal to _1/3_. That is a good result but it will be equal to _~0.98_ already by the tenth deal.

Thus, the requirement to maximize profits leads to an absurd situation when all the funds should be put at stake, however, only once. That is certainly not a desirable outcome for you, as you expect that the more bets you make, the larger your expected end profit should be in your game.

It is possible to increase the duration of such a game and reduce the probability of ruin by eliminating the maximum profit requirement. In other words, only part of the funds should be used when making bets. If this part is very small, the game may last for quite a long time. This has also been proven mathematically. \[7\]\[8\] However, in this case, the final gain will also be small. Thus, high bets increase potential profit while also increasing the risk of loss. Low bets reduce the risk decreasing potential profit. Thus, there is a question what part of the funds is the most reasonable for being used in the game (from a certain point of view).

These are usual line of reasoning and assumptions. This issue has been thoroughly studied. The entire field called "money management" has appeared in order to solve it. There are several MM methods providing various opportunities to meet the requirements of the ruin probability and profit value. We will not examine all the methods due to the article limitations, therefore, we will focus on two of them: the method based on defining fixed-sized bets and the method of putting a fixed fraction of the funds at stake.

### A Bit of Theory

If we omit all mathematical wisdom, the key moment in MM is a question that can be formulated as follows: what is the probability of an event (for example, funds growth or the ruin) after a certain period of time (for example, number of deals). In fact, it is a question concerning perspectives, therefore, the time can be fixed and only two parameters can be considered - event probability and the event itself.

If we consider "heads or tails" in its most simple form, then we can calculate some things quite easily knowing the number of _**N**_ bets. For example, mathematical expectation ( _**МО**_) of revenue per one bet (1) or expected revenue for a series of bets (2). Note that _**МО**_ is defined relative to the initial funds here. It means that we have positive expectation in case _**МО** >0_ and if _**МО** <0_, it is expected that each bet is loss-making on the average.

![](https://c.mql5.com/2/13/4mmfrm01.png)

For the case mentioned in \[9\]: _**p** =0.45, **q** =0.55, **a%** =0.08, **b%** =0.05, **N** =20_, the results are _**MO** =0.0085, **TWR** =1.170_. This case is interesting in that with the probability of win _**p** <0.5_, _**МО**_ remains positive at one bet and consequently a profit of _~17%_ to the initial funds is expected.

> Warning: another MM method is examined in \[9\]. Therefore, the results will be different despite similar input data.

However, such an expectation is akin to the average across the board. It tells nothing about the probability of occurrence of some of the results depending on the number of winning bets complicating evaluation of risks. Therefore, let's introduce two more equations: for calculating profit for a certain amount of winning bets (3) and calculating probability of occurrence of a certain amount of winning bets in a series (4):

![](https://c.mql5.com/2/13/4mmfrm02.png)

Now, we only have to calculate the values for all _**V** = 0,1, **...,N**, **L** = **N**- **V**_ and create the graph of dependency of **_Prob(V)_** from **_TWR(V)_**. For the previously described case, the graph looks as follows. Note that the graph displays **_Prob_** and **_TWR_** axes without **_(V)_** index. This has been done solely to make the graph look less complicated.

![](https://c.mql5.com/2/13/4mmpic01.png)

Fig. 1

Results of our calculations using equations (3) and (4) are shown as green dots. The graph can be interpreted as follows. For example, the probability of occurring the deals series results, in which _**TWR** =1.04_, is _0.162_. Most often, with probability of _0.177_, _**TWR** =1.170_ and so on. The blue dots represent the same data in the form of cumulative probability. Thus, the probability of loss (i.e., some of the game rounds have _**TWR** <=1.00_) for our input data comprises _0.252_. Extreme values are not displayed on the graph. The case of ruin ( _**TWR** =0.00_) has _**Prob** =6.4E-06_. Maximum gain _**TWR** =2.60_, \- _**Prob** =1.2E-07_. These are very small probabilities. However, their existence creates another important issue.

Let's try to demonstrate this with the following example. Calculations have been performed for the following conditions: _**p** =0.45, **q** =0.55, **a%** =0.05, **b%** =0.05, **N** =50_. Results are displayed in the graph.

![](https://c.mql5.com/2/13/4mmpic02.png)

Fig. 2

As we can see, _**TWR**_ takes on the values from _-1.50_ up to _3.50_. _**TWR** =-1.50_ is possible only in case the game has been performed with the amount of funds less than _0.0_. Thus, the equations we used do not consider the fact that the funds were depleted at one of the intermediary deals and the game could not be continued. Therefore, the new task has emerged, the so-called "boundary absorption issue". This issue considers that there is some boundary (or a limit) for the existing funds. When this boundary is reached, the game is stopped. In its simplest form, it is assumed that the boundary _=0_. However, we are more interested in the case when it can take on arbitrary values. Some aspects of the analytical solution of this issue are revealed in \[7\].

### Range of Opportunities

Let's try to solve this problem numerically, first, by using iterative calculations, and second, using simulations modeling based on stochastic methods (Monte Carlo method). First of all, let's examine the figure and try to sort out the problem.

![](https://c.mql5.com/2/13/4mmpic03.png)

Fig. 3

The figure schematically displays possible change trajectories of the funds during the game. Three cases are displayed, though there are much more of them, of course. It is assumed that players having the funds that exceed, for example, _0.3_ (collateral requirements) are allowed to participate in the game. Suppose that the game took the red course and it became impossible to continue playing. Boundary absorption of the trajectory has occurred meaning complete ruin. The player is dropped out of the game, unlike the games that took green or blue courses. Thus, we need to define the ruin probability at the previous steps in order to define it at some definite one.

The most simple, intuitive and very old technique that can be used to carry out such calculations is Pascal's triangle. In fact, this is a recursive procedure in which the next value is calculated using the previous ones. The slightly modified triangle is displayed below.

![](https://c.mql5.com/2/13/4mmpic04.png)

Fig. 4

Green dots indicate possible locations on the graph **_TWR_** trajectory can pass through. In this case, TWR can be calculated using equation (3). The dots have **_Prob_** values (numerator) and **_TWR_** values (denominator). **_z%_** symbol designates the boundary value at which absorption occurs (black line). The red horizontal line is drawn through _**z%** + **b%**_ value.

What does location of dots relative to the red line represent? If the dot is located higher, the next round is possible. When the red line crosses the dot, this is the last chance for one more step. In case of success, the game continues, otherwise, absorption occurs. The dots between the red and black lines are achievable, but the next step out of them is impossible, as the funds are insufficient for the next bet. In other words, this is not a complete ruin but the game still cannot be continued.

> Note: of course, this is not possible in case of integer bets, as when using coins, but if the bet is equal to, say, _0.15_ of the funds, the overall picture is exactly like the one described above.

![](https://c.mql5.com/2/13/4mmpic05.png)

Fig. 5

Here is another figure containing calculation results in case of different boundary conditions. If we compare it with the previous one, we should be able to notice the difference.

![](https://c.mql5.com/2/13/4mmpic06.png)

Fig. 6

Now, having such data, we are able to know the probability of any event. For example, what is **_Prob_** of _**TWR** =1.4_ in case of _**N** =6_ for Fig. 6? The answer is _0.234_. Or what is **_Prob_** of _**TWR** >1 in case of **N** =6_? Corresponding **_Prob_** values should be summed. The answer is _0.344_. Or what is **_Prob_** of _**TWR** =1.1 in case of **N** =6_? The answer is _0_. And so on.

The last example of this series having "disfigured" input values: _**p** =2/3, **q** =1/3, **a%** =0.1, **b%** =0.2, **z%** =0.2_ is shown to demonstrate how **_Prob_** and **_TWR_** change in this case. As we can see, the win probability exceeds the loss one twice. However, the win size is half the size of a loss.

![](https://c.mql5.com/2/13/4mmpic07.png)

Fig. 7

If _**TWR** >1_, _**Prob** =0.351_. This **_Prob_** value is pretty close to the case displayed on Fig. 6. But it is clear that **_TWR_** values that can be achieved in a comparable number of bets are much less.

Another important thing that we have not discussed yet concerns **_f_** parameter indicating the share of funds involved in the bet. In fact, this is the share of the funds that are used in the deal, i.e. **_b%_**. In classic "heads or tails", it is possible to lose only a countable number of "coins". Thus, such an expression as _1/ **f**_ displays the number of consecutive losses that leads to ruin. _1/ **f**_ parameter may be non-integral in our case. At the same time, the number of bets cannot be non-integral considering non-divisibility of bets. It means that some part of the funds may still remain when the game cannot be continued (and this part will be less than **_b%_**). In other words, this part is absolutely risk-free, as it cannot be lost. In this regard, the actual amount of the funds participating in the game is less by this value. Thus, the actual **_z%_** parameter is larger by this value (see Fig. 5). In this case, if _**z%** >0_, then **_f_** tends to exceed **_b%_**. Considering all that, the actual **_f_** can be calculated the following way:

![](https://c.mql5.com/2/13/4mmfrm03.png)

where **_int_** symbol stands for truncation. For the example displayed in Fig. 5, _**b%** =1/5_, but the actual _**f** =1/3_.

### Duration of Opportunities

Let's consider our results to be a correspondence between **_Prob_** and **_TWR_**. Several different sets of input data have been selected as an example. All the series have the same length - _**N** =15_.

![](https://c.mql5.com/2/13/4mmpic08.png)

Fig. 8

Below are some explanations concerning what and how is displayed on the graph. Each curve starts with an "empty" dot (if we look from left to right). These are conditional points (in some cases), which may not exist actually. Nevertheless, they are applied indicating the probability of the absorption occurring on some of the intermediate steps. In other words, this is the probability of the trajectory not reaching the last step. The next dot on the graph represents actual data on the probability of the absorption occurring at the last step considering all previous ruins (with some exceptions that are not important here).

The next image demonstrates how the probability of ruin changes depending on **_N_** length of the series. The same input data as for Fig. 8 has been used for the example. As we will see further, the results are significantly different from each other.

![](https://c.mql5.com/2/13/4mmpic09.png)

Fig. 9

The most important thing is that if the series length increases, so does the ruin probability. This key issue cannot be eliminated in case of a boundary absorption game. However, not everything is as bad as it seems at first glance. The probability of ruin can turn out to be very small depending on TS parameters (see the red line). If TS parameters are not very successful, the ruin occurs rapidly (see the brown line).

Another important issue is the probability of total win. We have already considered the case of _**TWR** <( **z%** + **b%**)_, now it is time to define the _**TWR** >1_ probability behavior depending on **_N_**.

![](https://c.mql5.com/2/13/4mmpic10.png)

Fig. 10

In the case shown in green and black colors, we deal with neutral strategies, i.e. the ones having _**MO** =0_ and we have all reasons to expect that the win probability should be about _0.5_. However, this is not the case. And that has to do with boundary absorption.

That is how the probability of possible _**TWR**_ occurring at a certain step looks. In this case, _**N** =50_, which is similar to the Fig. 10 value.

![](https://c.mql5.com/2/13/4mmpic11.png)

Fig. 11

These are ordinary distribution curves. Their difference from the conventional distribution curve is that they are asymmetric relative to their maximum value, as well as their minimum and maximum **_TWR_**. Besides, some curves are noticeably skewed and resemble lognormal curves. Another interesting moment is that if we compare Figures 10 and 11, we will see the following. While most distributions in Figure 11 have the most probable **_TWR_** value of more than _1_, the probability of **_TWR_** value being more than _1_ is less than _0.5_. Moreover, for cases with _**MO** =0_ (black and green lines), the most probable **_TWR_** is also not equal to _1_, as it could be expected. There is no paradox in that.

Now it is time to finish our brief review of some aspects of "boundary absorption task". The next thing we should do is to perform stochastic simulation and compare results with the previously obtained ones to evaluate the correctness of reasoning and simulation accuracy.

### Simulation

In general, the following simulation algorithm is used. The amount of funds is checked before each "coin" toss. If the funds are insufficient to continue the game, it is stopped. If the game can be continued, PRNG is used to define if the bet was a winning or losing one. The funds are increased or decreased according to the results. The algorithm remains the same to the end of the game. The large amount of game rounds are performed, results are averaged. It is all very simple. The only problem of stochastic methods lies in their accuracy. When using this method, the accurate solution of the task is impossible to be found and that has been proven mathematically (see de Moivre paradox). Therefore, we should define to what extent the results are consistent with other ones before using this model in further calculations.

Let's compare two solution cases for the following parameters: _**p** =0.5, **q** =0.5, **a%** =0.2, **b%** =0.1, **z%** =0.0_. Below are two figures (12 and 13) displaying the correspondence of the results.

![](https://c.mql5.com/2/13/4mmpic12.png)

Fig. 12

![](https://c.mql5.com/2/13/4mmpic13.png)

Fig. 13

Previously obtained values are shown in green. Black dots show simulation results. The red color represents the error, the ratio of the expected value to the simulation one. Figure 12 displays the values, while Figure 13 is an accumulated sum. The match is quite good. The error rate at the midpoint of the values is less than _0.5%_. Of course, the error rate is higher at the edges of the range. However, there is nothing to worry about. We just should consider that in the future. Besides, we are more interested in the accumulated sum. The error rate is much less there (this is a feature of the cumulative curves when errors from different values cancel each other).

Another feature of the simulation results is that it is almost impossible to receive the probability of extremely rare values. In the above example, maximum _**TWR** =11_, while _**Prob** = 8.88E-16_. These values could not be obtained in simulation.

Another example that demonstrates the above mentioned fact in case of _**N** =250_: _**p** =0.5, **q** =0.5, **a%** =0.1, **b%** =0.1, **z%** =0.0_. No further comments are needed here.

![](https://c.mql5.com/2/13/4mmpic14.png)

Fig. 14

![](https://c.mql5.com/2/13/4mmpic15.png)

Fig. 15

Now we can use this model as a basis for solving the more complex problem than the one we considered above.

### Trading Operations

Before we continue our discussion, I would like to focus on concepts and equations. At first glance, real trading is quite different from "heads or tails" game described above - considering this fact requires additional efforts during simulation. Therefore, it is necessary to clarify at once what we will have to deal with in the future. Of course, most readers have their own ideas about that. Below are some useful concepts and notations.

> Note: for simplicity, only transactions with "direct quotation" pairs, such as EURUSD and GBPUSD, will be examined. "Indirect quotation" pairs and cross rates are calculated in a different way. For indirect quotation pairs, the point's price changes according to the current quote. For cross rates, the current quote of the base (first) currency to USD is additionally considered. Besides, ASK and BID concepts are not used here.

The funds used in trading will be called _**Deposit**_. We have the right to buy and sell contracts having a certain **_LotSize_** using **_Leverage_**. The contract may be fractional, so we need the concept of **_Lots_** as a parameter indicating the applied contract size. The actual size of the used lot in base currency will be called **_LotPrice_** accordingly. We should pay **_Margin_** for the right to buy or sell. If we express **_Margin_** as a share of **_Deposit_**, we will receive **_Margin%_** parameter. We will also use **_StopOut_** indicating the minimum part of **_Margin_**. Reaching it leads to closing of the current deal and the trading is stopped forcedly. Thus, there are two different situations when further trading is impossible (at the desired rate of **_LotPrice_**), i.e. the case of ruin. Another parameter **_Sigma_** has additionally been included. This is the ratio of the funds used in trading operations considering credit to equity. Ratio of the used funds to the actual capital, i.e. the analogue of a leverage, though it is applied to the entire **_Deposit_** instead of **_LotPrice_**.

![](https://c.mql5.com/2/13/4mmfrm04.png)

One of the basic concepts that characterize trading process is _**Quote**_ \- current exchange rate. Minimum rate change is **_Tick_**. The size of the symbol's rate minimum change is **_TickPrice_** considered as part of **_Deposit_** \- **_Tick%_**.

![](https://c.mql5.com/2/13/4mmfrm05.png)

Besides, we have a few more parameters connected with rate change: the so-called **_TP_** (TakeProfit) and **_SL_** (StopLoss) - rate change, at which profit or loss is fixed. If we express these parameters in currency, they will be called **_TPprice_** and **_SLprice_**. The ones expressed as the part of **_Deposit_** will be called **_TP%_** and **_SL%_**. Also, there are various dealer commissions including **_Spread_**, **_Swap_** etc. We will not examine the entire diversity of these parameters focusing only on **_Spread_**, which is traditionally displayed in points of rate change. If necessary, **_Swap_** data can be considered quite easily if it is also displayed in points. If **_Swap_** is presented as an actual interest rate on borrowed funds used with the help of a leverage, the situation turns out to be somewhat more complicated.

![](https://c.mql5.com/2/13/4mmfrm06.png)

Let's compare obtained parameters with the ones used in "boundary absorption" problem. **_SL%_** clearly corresponds to **_b%_**, while _**TP%** = **a%**_. The initial funds have been presented as _**TWR** =1_. The current case is similar, as our calculations are based on parameters displayed in fractions of a unit. _**z%** + **b%**_ boundary value can be conventionally considered as **_Margin%_**. There is some difference between these concepts but it is not so critical in case we do not consider the existence of **_StopOut_**. It appears that the tasks are similar at the first approximation. We will check this claim later.

If we closely examine the equations (9, 11, 13, 15), we will see _**Sigma**_ parameter in all of them. As already mentioned, this is an equivalent of a leverage used in all critical equations. _**Leverage**_ can directly affect only _**Margin%**_.

![](https://c.mql5.com/2/13/4mmpic16.png)

Fig. 16

Correctness of the calculations can be checked by the following example. Suppose that if _**Leverage** =100_, _**LotSize** =100000_, _**Lots** =0.1_, _**Deposit** =10000_, the margin size is _100_. If we look at the appropriate (brown) graph, we will see that _**Margin%** =0.01_. In case _**Deposit** =10000_, we will receive _100_.

Two features can be observed in the graphs: first, the larger **_Sigma_** value, the higher **_Margin%_**, in other words, the less free funds that can be used in trading. Second, decreasing **_Leverage_** value also increases **_Margin%_**. Let's examine Figure 9 where similar cases are addressed (blue and green lines). These cases differ only in **_z%_** value. It can be seen that increase of the boundary value also leads to an increase of the ruin probability under otherwise equal conditions.

Since there is no direct relationship between **_SL%_** and **_Leverage_**, let's examine how this parameter depends on **_Sigma_** and ( **_SL+Spread_**).

![](https://c.mql5.com/2/13/4mmpic17.png)

Fig. 17

Increase of Sigma parameter leads to increase of **_SL%_**. Thus, the ruin becomes much more probable again. This dependence is a linear one. It turns out that in order to decrease **_SL%_**, we need either decrease ( **_SL+Spread_**) or **_Lots_**, since all other values included in **_Sigma_** most probably cannot be changed.

Then we are done with equations. Now, let's briefly describe how the simulation was conducted. The process is almost indistinguishable from the previous one. The possibility of trading has been checked, buy operation has been performed at the current market conditions, the rate has been determined with the help of PRNG, sell operation has been performed and the current funds level has been calculated. All this has been repeated multiple times in the loop. Trading results have eventually been defined. No specific analytical forms have been used, the entire simulation has been performed "as is". **_Tick_** and other related concepts, as well as **_Swap_** have not been used, as they have not been necessary.

In order to assess simulation results, we have used an example with the following basic parameters: _**Deposit** =1000, **Leverage** =100, **LotSize** =100000, **Lots** =0.1, **TP** =0.0040, **SL** =0.0040, **Spread** =0.0002, **p** =0.5, **N** =250_. The value of the series length is approximately equal to the number of working days per year. Thus, if we assume that one trade is performed per day (intraday deal, therefore, **_Swap_** is not used), results are annualized. After applying equations (9, 13, 15), we will receive the following results: _**TP%** =0.038= **a%**, **SL%** =0.042= **b%**, **Margin%** =0.1_ (thus, **_z%_** =0.058). Let's perform stochastic simulation and calculations similar to the ones displayed in Figure 4.

![](https://c.mql5.com/2/13/4mmpic18.png)

Fig. 18

Calculations based on the simulation and on the equations are compared here. As we can see, results of various calculations have a good match. This is another reason to claim that trading process is not much different from the classic "heads or tails" game when dealing with "boundary absorption" problem.

Below are a few brief comments on the graph. The left-most and the smallest dot of the green-red line displays the ruin probability. In our case, it is **_TWR_** <= **_Margin%_** \- _0.375_. The probability of loss is when **_TWR_** <= _1_, \- _0.795_, thus, the win probability is _0.205_.

### ММ \- Fixed Size (FixSize)

The main idea of this MM method is that a fixed part of the initial funds is used each time despite any circumstances. In fact, conventional features of lot size management provided to traders by dealers directly correspond to this method. Besides, this method is implemented in "heads or tails" game. Since we have already covered the basic features of this method before, let's proceed to the issue of individual parameters affecting the system. We will use our model to tackle this issue.

As we consider numerical solution rather than analytical one, the only way to evaluate the effect of changes in the input parameters is to compare calculation results. We will fix the data, change some of the data in a certain range and observe the changes.

Let's return to the case displayed in Figure 16. Perform the calculations with the following parameters: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** = { 0.0030, 0.0040, 0.0050, 0.0060}, **Spread** =0.0002, **p** =0.5, **N** =250, **Leverage** = { 3, 4, 5, 6, 7, 10, 25, 50, 75, 100}_.

![](https://c.mql5.com/2/13/4mmpic19.png)

Fig. 19

Calculation results are shown on a logarithmic scale for greater clarity. Actually, decrease of _**Leverage**_ leads to increase of _**Margin%**_ (black line) enhancing the probability of ruin. The correlation between the ruin probability and the leverage is a non-linear one and its greatest impact can be felt in the area of small _**Leverage**_ values. The negative effect caused by decreasing _**Leverage**_ can be reduced by decreasing _**TP**_ and _**SL**_ levels. But even if _**Leverage**_ value is small enough (in our case, it is 4), the ruin probability is still high enough. In addition, there is another feature that is displayed in the figure below.

![](https://c.mql5.com/2/13/4mmpic20.png)

Fig. 20

The figure shows how the loss probability changes depending on altering _**Leverage**_ value. As we can see, the loss becomes more probable when _**TP**_ and _**SL**_ decrease.

If we increase the funds up to, say, _10 000_, the ruin becomes impossible in case of _**N** =250_. If we compare the figures 20 and 21, we will see that the results coincide in _**Leverage** >10_ area.

![](https://c.mql5.com/2/13/4mmpic21.png)

Fig. 21

Let's consider an example, in which the win probability in one deal is higher than the probability of loss, in other words, _**p** >0.50_. Perform the calculations with the following parameters: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** ={ 0.0030, 0.0040, 0.0050, 0.0060}, **Spread** =0.0002, **p** =0.52, **N** =250, **Leverage** ={ 3, 4, 5, 6, 7, 10, 25, 50, 75, 100}_.

![](https://c.mql5.com/2/13/4mmpic22.png)

Fig. 22

![](https://c.mql5.com/2/13/4mmpic23.png)

Fig. 23

If we compare calculation results displayed in Figure 19, we will see that increasing the gain per one deal decreases the ruin and loss probabilities. That is a natural and expected result.

The following example shows how win probability affects the loss one in one deal considering _**TP**_, _**SL**_ level. The parameters are as follows: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** ={ 0.0020, 0.0030, 0.0040, 0.0050, 0.0060}, **Spread** =0.0002, **p** ={ 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57}, **N** =250, **Leverage** =100_.

> Note: Example has been selected so that the ruin probability value is inconsiderable and can be disregarded. The highest ruin probability for the worst case comprises about _0.04_.

![](https://c.mql5.com/2/13/4mmpic24.png)

Fig. 24

The first important thing is that increase of _**p**_ decreases the probability of loss. Decrease of _**TP**_ and _**SL**_ levels has the opposite effect. This happens due to fixed _**Spread**_ value. Another conclusion that can be drawn from these results is that trading with low levels in each deal is quite a risky thing.

Example below demonstrates the extent to which _**Spread**_ value affects the loss probability. Main parameters of the simulation: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** ={ 0.0020, 0.0030, 0.0040, 0.0050, 0.0060}, **Spread** ={ 0.00000, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045}, **p** =0.55, **N** =250, **Leverage** =100_.

![](https://c.mql5.com/2/13/4mmpic25.png)

Fig. 25

The next figure demonstrates the correspondence between the loss probability and _**MO**_ of the deal calculated by using equation (1) for the data displayed in Fig. 24. Instead of _**a%**_ and _**b%**_ values, the appropriate _**TP%**_ and _**SL%**_ ones have been used.

![](https://c.mql5.com/2/13/4mmpic26.png)

Fig. 26

As we can see, increase in _**MO**_ of the deal reduces the loss probability, though it cannot be reduced to zero. It means that positive _**MO**_ alone does not guarantee protection against loss, as well as ruin. Greater _**TP**_ and _**SL**_ levels certainly reduce the loss probability. But at the same time (see Figures 19 and 22), that increases the ruin probability.

In other words, we have again returned to our previous statement that I expressed when describing the classical "heads or tails" game. Increase in bets leads to an increase of possible gain (and reduces the loss probability), while making the ruin more probable. Unfortunately, no mathematically sound method of selecting a deal volume when applying this MM method exists. We can use only our personal preferences on what level of ruin or loss is acceptable in each case.

Let's consider the issue of determining the probability of reaching a certain _**TWR**_ level during the game. Main parameters of the simulation: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** = 0.0030, **Spread** =0.0002, **p** ={ 0.48, 0.49, 0.50, 0.51, 0.52}, **N** =250, **Leverage** =100_.

![](https://c.mql5.com/2/13/4mmpic27.png)

Fig. 27

Graph data can be interpreted the following way. The probability of _**TWR**_ value decreasing down to _~0.90_ in case of _**p** =0.48_ is _~0.93_. If _**p** =0.52_, the value is _~0.68_.

![](https://c.mql5.com/2/13/4mmpic28.png)

Fig. 28

This figure displays the probability of _**TWR**_ level increase up to a certain value during the game. For example, the probability of _**TWR**_ value reaching _~1.10_ at _**p** =0.48_ is _~0.09_. In case of _**p** =0.52_, the value is _~0.37_. It is quite logical that increase of the win probability in one deal leads to increase in the probability of reaching a certain level.

Now, a few words about _**Drawdown**_. Script \[10\] with minor changes has been used to evaluate this parameter. Besides, unlike the original script, calculation is performed considering _**Spread**_.

Main parameters of the simulation: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** =0.0030, **Spread** =0.0002, **p** ={ 0.48, 0.49, 0.50, 0.51, 0.52}, **N** =250, **Leverage** =100_.

![](https://c.mql5.com/2/13/4mmpic29.png)

Fig. 29

In case of the initial parameters similar to Fig. 27, the probability of _**Maximal Drawdown %** ~0.20_ at _**p=0.48**_ is _~0.85_. In case of _**p** =0.52_, the value is _~0.47_.

The next and final example: _**Deposit** =2000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** ={ 0.0020, 0.0030, 0.0040, 0.0050, 0.0060}, **Spread** =0.0002, **p** =0.50, **N** =250, **Leverage** = 100_.

![](https://c.mql5.com/2/13/4mmpic30.png)

Fig. 30

One small note concerning Figure 30. _**Maximal Drawdown %** level is 0.96_ approximately matching _**Margin**_ level. That is why we can see a sharp break in lines in the graphs. As we can see the probability of reaching greater _**Maximal Drawdown %**_ values has increased, as compared to the case shown in Figure 29 (red line).

Before finishing our discussion on some features of MM having fixed size, let's examine two more distributions of such model characteristics as _**Profit Factor**_ and _**Expected Payoff**_.

![](https://c.mql5.com/2/13/4mmpic31.png)

Fig. 31

![](https://c.mql5.com/2/13/4mmpic32.png)

Fig. 32

Main parameters of the simulation for Figures 31 and 32: _**Deposit** =3000, **LotSize** =100000, **Lots** =0.1, **TP** = **SL** = 0.0030, **Spread** =0.0002, **p** ={ 0.48, 0.49, 0.50, 0.51, 0.52}, **N** =250, **Leverage** =100_.

### ММ \- Fixed Fraction (FixFrac)

This method comprises ММ having a bet as a fixed fraction of the current funds. Share size is initially determined, for example, _10%_ for participating in deals. The amount of funds used in deals is calculated based on the current amount of the overall funds afterwards, regardless of the trading results. Thus, each successful deal increases the volume of the next one and vice versa. Sometimes, this system is called anti-martingale (though, strictly speaking, it is incorrect). This system assumes reinvestment of possible profit as opposed to MM having the fixed size.

Advantages and drawbacks of this MM method are well-known already. This method provides rapid growth of funds in case of successful deals. This growth has the form of a geometric progression as opposed to an arithmetic one in case of the fixed size method. In theory, this method prevents complete loss of funds in case of the infinite divisibility of bets (though this is not the case for real trading, of course).

One of the method's drawbacks is the so-called asymmetric leverage effect. It means that the next bet after the loss-making one has lesser volume. Thus, a player should have more successful deals than loss-making ones in order to recover the losses. Besides, hidden series in deal consequences may affect this method's results (non-Bernoulli).

During its practical use, this MM method may pose an issue of selecting the funds share that can provide the best funds growth rate combined with acceptable ruin probability. Such an asymptotically optimal game strategy has been offered by Kelly.

But first, let's examine some theory and equations. If we know the number of profitable and loss-making bets, as well as win and loss values per one bet, we can calculate _**TWR(V)**_ and _**Prob(V)**_ the same way as in case of the fixed size method. The equation (17) is similar to (4).

![](https://c.mql5.com/2/13/4mmfrm07.png)

Example of calculation using equations (16) and (17) is displayed below. The probability of reaching different _**TWR**_ values is shown as an accumulated value.

![](https://c.mql5.com/2/13/4mmpic33.png)

Fig. 33

The next figure shows the two cases with similar conditions but different MM. The fixed size case has been taken from Fig. 8. Fixed size MM (FixSize) is shown in red, while fixed fraction one (FixFrac) in black.

![](https://c.mql5.com/2/13/4mmpic34.png)

Fig. 34

The equation describing the speed of the funds growth in case of equal profit and loss values looks as follows.

![](https://c.mql5.com/2/13/4mmfrm08.png)

By using simple transformations and reasoning \[8\] and minimizing expected average geometrical growth, we can produce the following equation showing the optimal _**f\***_ bet size.

![](https://c.mql5.com/2/13/4mmfrm09.png)

This is the so-called Kelly criterion. Its idea is quite simple. If you have a ТS with the win probability exceeding _0.5_, while profit and loss values are equal, then you need to use the funds share calculated using equation (19) when placing a bet. For example, _**p** =0.55_, in this case, _**f\*** =0.55-0.45=0.10_. It means that you need to use one tenth of the funds when placing a bet in order to make the funds grow efficiently.

> Warning: please note that Kelly's optimal bet value equation is different from the one offered by Vince in \[4\]. There is no any mistake from Vince's side here and I will explain that later on.

> Note: If we calculate optimal funds share value for _**p** =0.50,_ then we will naturally get _0.00_. In other words, it is recommended not to play at all.

In case profits and losses are **_k_** times different, we should use another equations instead of (18) and (19). \[8\]\[5\].

![](https://c.mql5.com/2/13/4mmfrm10.png)

If we create _**g(f)**_ change graph from _**f**_ using equation (20), we will see something like the following figure. This figure is marked by _**f\***_ point. This is the highest point of the graph where the funds growth rate reaches its maximum. Besides, there is _**fc**_ point representing the intersection point of the graph and a zero line. It is the point where the funds growth rate is zero.

![](https://c.mql5.com/2/13/4mmpic35.png)

Fig. 35

> Note: _**k** =2_ has been selected solely as an irony addressed to the appropriate graphs displayed in Vince's books. Nevertheless, such _**k**_ value provides greater "clarity" and "beauty" to the methods he advocates.

In \[8\], it is stated that if the game is performed along the specified conditions, then _**f\***_ provides maximum funds growth rate and zero ruin probability. If the share less than _**f\***_ is used, ruin probability is also zero, though the funds growth rate is lower. If the used funds share exceeds _**fc**_, the ruin is imminent (in this case, any fund stock is meant as a ruin, no matter how low it is). If the funds occupy the range from _**f\***_ to _**fc**_, growth rate is also slower than the maximum one, though there is no ruin probability.

The results are impressive enough. However, these theoretical calculations do not take some real world features into account. Therefore, Vince recommends to calculate the optimal **_fс_** with consideration to maximum losses. This leads to the fact that his _**f\***_ value becomes lower than that calculated strictly mathematically with all ensuing consequences.

Let's try to consider how it may look on the graphs. Let's take the following case: _**Deposit** =1000, **LotSize** =100000, **f** ={ 0.01, 0.02, 0.03, 0.04, 0.05}, **TP** = **SL** =0.01, **Spread** =0.0000, **p** =0.51, **N** =250, **Leverage** = 100_. With these input data, initial _**Lots** =0.01_, while _**TP%** = **SL%** =0.01_. Besides, _**f\*** =0.02_ and _**fc** =0.04_, as shown in the following figure.

![](https://c.mql5.com/2/13/4mmpic36.png)

Fig. 36

In case of _**f**_ changes, we will have the following correlations between _**Prob**_ and _**TWR**_. Let's perform a few calculations to demonstrate this.

![](https://c.mql5.com/2/13/4mmpic37.png)

Fig. 37

Let's consider this in more detail. The line corresponding to _**f** =0.01_ demonstrates previously discussed case of _**f** < **f\***_ when the funds growth rate is lower than the maximum possible one, while the ruin probability is zero. In this case, the probability of loss _**TWR** <=1_ is equal to _~0.40_.

The next case, _f=0.02_ is the one when the share of the used funds is the optimal one. The loss probability is _~0.45_. In other words, more than a half of game rounds are expected to be profitable.

Calculation version _**f\*** < **f** < **fc**_, i.e. _**f** =0.03_. The loss probability is _~0.45_. The ruin is impossible under the given game conditions. However, the values _**TWR**_ can descend to are more significant than in the previous cases. Greater profits are also possible.

Now the loss probability is _~0.50_, i.e. _**f** = **fc** =0.04_. It is stated that in this case, _**TWR**_ will almost certainly fluctuate between _0_ up to _+infinity_.

And the last case, _**f** > **fc**_. The loss probability is _~0.55_. Very considerable gains are possible in that case, but all funds will be lost in the long (infinite) run and _**TWR**_ will be reduced down to the level that can be classified as ruin.

A little more equations. _**g(f)**_ funds growth rate can be evaluated using equations (18) and (20). Since _**g(f)**_ growth rate and the amount of deals _**N**_ are known, it is possible to calculate _**TWR**_ using the equation (22). Besides, it would be more correct to evaluate expected average profit from the deal for MM using fixed funds share as _**Gmean**_ geometric mean using the equation (23) rather than arithmetic one.

![](https://c.mql5.com/2/13/4mmfrm11.png)

> Recommendation: It would be good to include the calculation of the deals' geometric mean to the MT tester's standard report. It would allow users to evaluate trading systems using fixed fraction MMs more correctly.

Below are two examples of calculation using the equations (23) and (24) and data from Figure 36.

![](https://c.mql5.com/2/13/4mmpic38.png)

Fig. 38

![](https://c.mql5.com/2/13/4mmpic39.png)

Fig. 39

Thus, based on Figure 39, the overall funds will tend to _**TWR** =1.05_ under the given conditions and with the optimal share in bets. As we can see in Figure 37, the probability of loss is _~0.45_.

It would be interesting to know the behavior of different MMs having the same starting conditions. If we use data in Figure 36, we will see the following figure.

![](https://c.mql5.com/2/13/4mmpic40.png)

Fig. 40

In the case of _**f** =0.01_, charts are very similar, while MMs are different. In other words, the results can be similar under certain conditions. All other cases show that fixed size MM has lesser loss probability ( _**TWR** <=1_) than fixed fraction one under the same starting conditions.

> Warning: The above observation applies only to the selected input data. In no case it can be regarded as a general rule.

Let's show another interesting circumstance - a correlation of the values, to which _**TWR**_ tends, with the funds share used in a deal. Calculation for these values can be performed for fixed size MM as multiplication of _**MO**_ by _**N**_, while the equation (22) is used for the fixed fraction MM.

![](https://c.mql5.com/2/13/4mmpic41.png)

Fig. 41

As we can see, the expected payoff when using fixed size MM is higher (for the used input data). However, this is not always the case.

It is possible to examine how the correlations, to which _**TWR**_ tends, change considering various win probabilities per one deal. The graphs for series length _**N** =250_ are shown below.

![](https://c.mql5.com/2/13/4mmpic42.png)

Fig. 42

As we can see, under certain conditions, in terms of the expected _**TWR**_, there may be situations when fixed size ММ is more preferable than fixed fraction one. However, if we consider this issue together with the ruin probability, the whole thing seems to be not so obvious.

Besides, the value of the expected _**TWR**_ is significantly affected by _**N**_ series length. In general, the longer the series, the greater the benefit of fixed fraction MM.

Now, let's finish our examination of some theoretical aspects and move on to simulation. In fact, nothing has changed in our initial model, besides the fact that it is necessary to add minimum and maximum lot concepts, _**MinLot**_ and _**MaxLot**_. Besides, we will need _**LotStep**_ value characterizing minimum possible step of the lot change. The general calculation algorithm remains the same. _**MinDeposit**_ concept is initially introduced as minimum possible deposit value. Thus, trading can be continued if the deposit value exceeds _**Margin**_ and _**MinDeposit**_.

Below is an example of simulating various MMs with the following input data. _**Deposit** =1000, **MinDeposit** =300, **LotSize** =100000, **MinLot** =0.01, **LotStep** =0.01, **MaxLot** =100, **f** ={ 0.01, 0.02, 0.03, 0.04, 0.05}, **TP** = **SL** =0.01, **Spread** =0.0000, **p** =0.54, **N** =250, **Leverage** =100_. Please note that _**Spread**_ is not used in this calculation.

![](https://c.mql5.com/2/13/4mmpic43.png)

Fig. 43

Introduction of _**MinDeposit**_ to calculations leads to the fact that fixed fraction ММ which had theoretically zero ruin probability (at _**f** < **fc**_) now has some non-zero one instead. Besides, discretization and lot size limitations also lead to ruin probability under some certain conditions (not displayed here). The impact of these negative factors can be reduced by taking the difference of _**Deposit**_ and _**MinDeposit**_ values as the initial funds. In fact, that is what Vince suggests - only a part of the funds should be used to calculate the optimal funds share. It seems to be a wise decision but it naturally leads to _**TWR**_ decrease.

Now, let me make a few more clarifications concerning Figure 43. Let's find " _**FixFrac**; **p** =0.54; **f** =0.05_" on the graph where it corresponds to _**Prob** =0.50_. This is a median, the value, above and below which _50%_ of all values are located. In this case, it corresponds to _**TWR** ~2.00_. In other words, half of all games has at least doubled the initial funds. The loss probability comprises _~0.20_, while ruin probability is _~0.03_. Comparing with " _**FixSize**; **p** =0.54; **f** =0.05_" graph, we can note that the ruin probability may increase up to _~0.08_, while the loss probability decreases down to _~0.14_, but the median approximately corresponds to _**TWR** ~2.00_. If we are lucky to get to the number of cases where _**TWR** >2.00_, the result will most probably be lower than when using fixed fraction MM.

Comparing the graphs on Figures 42 and 43, we should note that Figure 42 values, to which _**TWR**_ tends, are nothing else but the median in Figure 43.

Now, let's consider the case displayed in Figure 43 with _**Spread** =0.0002_. All other input data has remained the same.

![](https://c.mql5.com/2/13/4mmpic44_1.png)

Fig. 44

As we can see, _**Spread**_ comprising from _0.4%_ to _2%_ of the profit\\loss level per deal leads to the considerable change of the entire picture. This can be best seen in the following figure where two calculation cases are compared (from Figures 43 and 44).

![](https://c.mql5.com/2/13/4mmpic45.png)

Fig. 45

Thus, _**Spread**_ consideration decreases _**TWR**_ median (for example, blue lines) from _2.00_ down to _1.50_. Loss probability has increased from _0.20_ up to _0.30_. The difference does not seem to be so large, only _0.10_, but if treated differently, one in five players will lose (though not down to ruin) in the fist case, while it will be one in three players in the second one.

Let's try to examine how the loss probability changes affected by the win probability per one deal considering various _**f**_, simulating various ММs with the following input data: _**Deposit** =1000, **MinDeposit** =300, **LotSize** =100000, **MinLot** =0.01, **LotStep** =0.01, **MaxLot** =100, **f** ={ 0.01, 0.02, 0.03, 0.04, 0.05}, **TP** = **SL** =0.0100, **Spread** =0.0002, **p** ={ 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57}, **N** =250, **Leverage** =100_.

![](https://c.mql5.com/2/13/4mmpic46.png)

Fig. 46

This example differs from the one displayed in Figure 24 by the fact that _**TP**_ and _**SL**_ levels are fixed here but lot sizes are changeable. Figure 47 shows the case with the data similar to Figure 46 one at _**TP** = **SL** =0.0050_.

![](https://c.mql5.com/2/13/4mmpic47.png)

Fig. 47

Similar to the case displayed in Figure 24, decreasing _**TP**_ and _**SL**_ levels led to the loss probability increase. Besides, the scatter of values between various cases has decreased. The graphs have become "denser". In other words, affect of _**f**_ value has decreased. This is particularly evident in the figure below.

![](https://c.mql5.com/2/13/4mmpic48.png)

Fig. 48

_**TP** = **SL** =0.0020_ levels were used in this case. As we can see, in order to compensate _**Spread**_ influence and reduce the loss probability down to less than _0.50_, we will need a TS able to provide _**p** =>0.56_. But generally, if a TS can be profitable only in _50_ cases out of _100_ ones, results remain the same, the loss probability after _**250**_ deals is _~0.95_ regardless of MM type and _f_ types.

Let's perform the calculation in order to show how _**TWR**_ will look in case of _**TP** = **SL** =0.0020_ and _**p** =0.56_. The result is shown below. This is actually a case with the loss probability of about _0.40_, as in Figure 48, while expected _**TWR**_ value is _1.01...1.04_. Different ММs display similar values.

![](https://c.mql5.com/2/13/4mmpic49.png)

Fig. 49

As I have already mentioned, this is typical for cases with small levels. If _**Spread**_ had been a floating value and had been charged as a percentage value of the bet size, the entire picture would have been different. That is exactly what happens on other markets other than "Forex for a wide range of customers".

Let's go back to the calculations with the following input data: _**Deposit** =1000, **MinDeposit** =300, **LotSize** =100000, **MinLot** =0.01, **LotStep** =0.01, **MaxLot** =100, **f** ={ 0.01, 0.02, 0.03, 0.04, 0.05}, **TP** = **SL** =0.0100, **Spread** =0.0002, **p** =0.51, **N** =250, **Leverage** =100_. Let's consider how _**TWR**_ decrease probability looks up to a certain level.

![](https://c.mql5.com/2/13/4mmpic50.png)

Fig. 50

The graphs' data should be interpreted the same way as in Figure 27. The probability of _**TWR**_ value decreasing down to _~0.70_ at _**f** =0.05_ is _~0.76_ in case fixed fraction MM is used. If _**f** =0.02_, the value is _~0.34_.

![](https://c.mql5.com/2/13/4mmpic51.png)

Fig. 51

And that is the probability of the deposit increase during the game for a certain value. This graph should be interpreted similar way as Figure 28.

![](https://c.mql5.com/2/13/4mmpic52.png)

Fig. 52

The last graph displays _**Maximal Drawdown %**_ probability of a certain value depending on the used funds share. Generally, we can assume that fixed fraction MM leads to larger _**Drawdown**_ than when using a fixed size one. This can be explained by asymmetric leverage effect. Besides, if _**fc** > **f** > **f\***_ is used, lesser _**Drawdown**_ can be expected, while the profit may be similar to _**f<f\***_ case.

### Conclusion

We have examined two money management methods, reviewed their development backgrounds, as well as briefly learned about some theoretical aspects of the issue and a few simple equations. We have performed stochastic simulation research and evaluated obtained results comparing different methods. Perhaps, the most important conclusion that can be drawn from this study is that it is impossible to select the ultimate method out of the two examined ones. ММ's efficiency depends on a large amount of factors. Various combinations of these factors provide different results. Therefore, in each definite case, we should select the most appropriate MM depending on TS properties, dealing center conditions, as well as trader's possibilities and preferences. Our article has highlighted some possible solutions and crucial features. I hope that at least some traders have found this text useful. Good luck to all!

My future plans include examination of martingale (I think, that part will not take much time and space, as this method's properties are pretty obvious), as well as reviewing R. Jones' method (the author claims that it has been developed as a combination of advantages of the two examined MM methods).

### References and web links (in order of appearance in the text)

01. Agner Fog - Pseudo random number generators - [http://www.agner.org/random](https://www.mql5.com/go?link=http://www.agner.org/random/ "http://www.agner.org/random")
02. strator - Probability Library (part of Cephes) - [https://www.mql5.com/ru/code/10101](https://www.mql5.com/ru/code/10101 "https://www.mql5.com/ru/code/10101") (in Russian)
03. Suvorov V. - MS Excel: Data Exchange and Management - [https://www.mql5.com/en/code/8175](https://www.mql5.com/en/code/8175 "https://www.mql5.com/en/code/8175")
04. Vince R. - The Mathematics of Money Management.
05. Bershadsky А.V. - Research and Development of Risk Management Scenario Methods. - Dissertation, 2002 (in Russian).
06. Smirnov А.V., Guryanova Т.V. - On Ralph Vince's "optimal f" (in Russian).
07. W. Feller - An Introduction to Probability Theory and Its Applications.
08. E. Thorp - The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.
09. S. Bulashov - Statistics for Traders. (online version, p. 199, in Russian)
10. Starikoff S. - How to Evaluate the Expert Testing Results - [https://www.mql5.com/en/articles/1403](https://www.mql5.com/en/articles/1403 "https://www.mql5.com/en/articles/1403") [http://articles.mql4.com/en/articles/1403](https://www.mql5.com/en/articles/1403 "http://articles.mql4.com/en/articles/1403")

### Examples

Publicly available libraries used when writing this article are attached below. The scripts and the application are not attached. This has been done deliberately to stimulate development of trading server emulators among the members of MQL community. It was a trading server emulator that was used in simulation.

### Disclaimers аkа Excuses

The author of this article bears no responsibility for anything. Complaints and suggestions are accepted in writing, in discussion section or via email. Sensible suggestions will be considered, while silly ones will be ignored. All copyrights are specified if known. Otherwise, the aurthor is unknown or the copyright has been lost.

The author admits that the text and calculations may contain inaccuracies or errors. The text is large and the subject is a complicated one, so errors are inevitable. Therefore, the author wanted to assign a $1 award for each detected fault similar to what D. Knuth did. But life has made some adjustments. The cold winds of the global financial crisis are still raging over the country. The most vulnerable and weakest among us - children left without parents - are usually the ones who suffer most at times like these. In order to help them, the author decided not to pay for his own mistakes but to spend the entire fee for charity. I have selected several orphan asylums in remote Russian provinces and sent my modest funds there.

It was an indescribable feeling when I received a response letter from the head of one of the orphanages. It was a simple letter written by hand on a sheet of paper from a squared copy-book. Of course, it contained words of gratitude but most importantly there was a list of purchased items. That wise woman spent the funds neither for a new computer to copy and paste letters of gratitude, nor for new curtains in her office, nor for food provided by the state, though it is much different from what people usually eat in Moscow. All the funds were invested into the education of those kids.

Copy-books, markers, pens, learning games, exercise-books, paints. All expendable materials that are so critical for the development of kids living in a small provincial town or village in an old wooden building possibly built by our ancestors who had just returned from the Great Patriotic War. So, if you can, if this is not your last money, please help them, do not be greedy or lazy. This will not go unnoticed. In fact, I'm not sure. Perhaps, that letter was written by a night watchman who was drinking away this unexpected gift. I'm not interested in that and I will never know the truth. I only made a bet, albeit a small one, that the probability of the fact that I live among normal people is much higher. Unlike heads or tails, this game is fair to all of us.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1367](https://www.mql5.com/ru/articles/1367)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1367.zip "Download all attachments in the single ZIP archive")

[experts.zip](https://www.mql5.com/en/articles/download/1367/experts.zip "Download experts.zip")(94.95 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**[Go to discussion](https://www.mql5.com/en/forum/39128)**

![MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://c.mql5.com/2/0/avatar11.png)[MQL5 Cookbook: Writing the History of Deals to a File and Creating Balance Charts for Each Symbol in Excel](https://www.mql5.com/en/articles/651)

When communicating in various forums, I often used examples of my test results displayed as screenshots of Microsoft Excel charts. I have many times been asked to explain how such charts can be created. Finally, I now have some time to explain it all in this article.

![Another MQL5 OOP Class](https://c.mql5.com/2/0/hand.png)[Another MQL5 OOP Class](https://www.mql5.com/en/articles/703)

This article shows you how to build an Object-Oriented Expert Advisor from scratch, from conceiving a theoretical trading idea to programming a MQL5 EA that makes that idea real in the empirical world. Learning by doing is IMHO a solid approach to succeed, so I am showing a practical example in order for you to see how you can order your ideas to finally code your Forex robots. My goal is also to invite you to adhere the OO principles.

![How Reliable is Night Trading?](https://c.mql5.com/2/17/841_4.gif)[How Reliable is Night Trading?](https://www.mql5.com/en/articles/1373)

The article covers the peculiarities of night flat trading on cross currency pairs. It explains where you can expect profits and why great losses are not unlikely. The article also features an example of the Expert Advisor developed for night trading and talks about the practical application of this strategy.

![Alert and Comment for External Indicators (Part Two)](https://c.mql5.com/2/17/825_12.gif)[Alert and Comment for External Indicators (Part Two)](https://www.mql5.com/en/articles/1372)

Since I published the article "Alert and Comment for External Indicators", I have been receiving requests and questions regarding the possibility of developing an external informer operating based on indicator lines. Having analyzed the questions, I have decided to continue with the subject. Getting data stored in indicator buffers turned out to be another area of interest to users.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1367&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069364202059006934)

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