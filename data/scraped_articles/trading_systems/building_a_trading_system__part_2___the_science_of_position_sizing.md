---
title: Building a Trading System (Part 2): The Science of Position Sizing
url: https://www.mql5.com/en/articles/18991
categories: Trading Systems
relevance_score: 9
scraped_at: 2026-01-22T17:33:42.632430
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/18991&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049217137538606916)

MetaTrader 5 / Trading systems


### Introduction

Position sizing is one of the most critical yet frequently misunderstood components of successful trading. It serves as the cornerstone of risk management and plays a decisive role in the long-term sustainability of any trading system—especially one built on a positive expectancy.

In [Part 1](https://www.mql5.com/en/articles/18587) of this article, we discussed how to construct a trading system with positive expectancy by focusing on win-rate and reward-to-risk ratio (RRR). We also demonstrated how risking 1% of the account balance per trade performed under various conditions using simulations.

In this second installment, we go a step further by examining how _varying the percentage of risk per trade_ can significantly impact the system’s performance, drawdown profile, and emotional burden on the trader. We'll explore different risk models using Monte Carlo simulation to illustrate the effects of position sizing under realistic market scenarios.

Professional traders often advocate for risking only _1% to 2%_ of the account balance per trade. Some take an even more conservative stance, recommending less than 1%, particularly during periods of market volatility or uncertainty. These guidelines are not arbitrary—they are designed to preserve capital and protect traders from the emotional and financial damage that can result from losing streaks.

However, an important question remains: Are these recommendations universal rules, or can traders tailor position sizing to fit their specific account size, objectives, and risk appetite?

In this article, we’ll explore that question in depth—combining theory with quantitative evidence. Using Monte Carlo simulations, we’ll analyze the outcomes of different risk levels, investigate the probabilities of account blowouts, and evaluate whether more aggressive risk-taking ever makes sense.

By the end of this piece, you’ll be better equipped to determine whether to follow the 1%-2% rule—or break from it intelligently.

### The Reality of Risk Across Account Sizes

To put things into perspective, consider a trader managing a $1 million account. Risking 1% per trade would mean placing $10,000 on a single position. Compare that to a trader with a $1,000 account—1% equates to just $10. The difference is striking.

It is more tempting for a small-account trader to risk a larger percentage—or even the entire account—on what appears to be a high-probability (A-plus) setup. The logic often goes: “If this works, I could double or triple my account quickly.” Unfortunately, while the upside may seem appealing, the downside can be devastating. A failed trade could wipe out the account entirely, leading not just to financial loss but significant emotional distress.

**Is 1%-2% a Universal Rule?**

The 1%-2% rule serves as a guideline rather than a hard-and-fast rule. Its purpose is to promote consistency and capital preservation. However, it may not suit every trader’s strategy, psychology, or financial situation.

Aggressive traders with small accounts might feel restricted by this rule, especially if their goal is rapid growth. But without a clear understanding of the probabilities behind losing streaks and the associated drawdowns, taking higher risks often leads to ruin rather than reward.

### Monte Carlo Simulation: Quantifying the Risk

In this article, we will develop a Monte Carlo simulation to dive deeper into the topic of position sizing and help traders achieve their desired results. This simulation will allow us to model hundreds or thousands of possible trading outcomes based on a system’s win-rate, risk-reward ratio, and position size.

Our goal is to understand, in probabilistic terms, whether a trader is better off sticking with the 1%-2% risk rule or designing a personalized risk strategy—without falling into emotional or psychological burnout or experiencing total account blowout.

But before we launch into the simulation itself, we will first explore the concept of consecutive losses under different win-rate and how these streaks interact with position sizing to produce various levels of drawdown. Monte Carlo simulation will be used to bring these scenarios to life and provide data-driven insight into how even a mathematically sound strategy can fail under poor risk management.

### Win-rate and its Associated Consecutive Losses

To demonstrate the impact of position sizing on different trading profiles, we generated a set of _synthetic trading systems using realistic combinations of win rates and reward-to-risk ratios (RRR) that mirror actual trading conditions._ Each system was constructed to ensure a positive expectancy, meaning the RRR for each exceeds the minimum threshold required for profitability at its given win-rate.

For a detailed breakdown of how minimum RRR thresholds are calculated for specific win rates, refer to [Part 1](https://www.mql5.com/en/articles/18587) of this series.

Below is a summary of the trading systems used in our simulation:

Table 1:

| System | Win-rate % | RRR |
| --- | --- | --- |
| 1 | 30% | 2.6 |
| 2 | 50% | 1.7 |
| 3 | 65% | 0.9 |
| 4 | 76% | 0.6 |
| 5 | 83% | 0.3 |

For visual inspection of positive expectancy, we provide graphs for each trading system, illustrating how their respective win rates and reward-to-risk ratios generate a statistically profitable edge over time. These visuals help confirm the mathematical soundness of each setup before we apply position sizing variations in the Monte Carlo simulations.

![syst1](https://c.mql5.com/2/160/system_1__32.png)

Figure 1: System 1(30%, 2.6)

![syst2](https://c.mql5.com/2/160/system_2__50.png)

Figure 2: System 2(50%, 1.5)

![sys3](https://c.mql5.com/2/160/system_3__65.png)

Figure 3: System 3(65%, 0.9)

![sys4](https://c.mql5.com/2/160/system_4__76.png)

Figure 4: System 4(76%, 0.6)

![sys5](https://c.mql5.com/2/160/system_5__83.png)

Figure 5: System 5(83%, 0.3)

As observed in Figures 1 through 5, each graph displays a rising or upward-sloping mean equity curve, which visually confirms the presence of positive expectancy in all the simulated trading systems. The Python code used to generate these graphs can be found in Part 1 of this series.

**Embracing the Inevitable: Losing Streaks in Trading Systems**

Although all the systems demonstrate positive expectancy, they are still susceptible to _consecutive losing streaks._ This is often the most emotionally challenging phase for traders. During such periods, many begin to doubt the system's profitability and may abandon it prematurely.

However, when a trader _understands and anticipates_ the potential for losing streaks, based on the system’s win-rate, they are better equipped to remain disciplined, avoid destructive emotional decisions and manage risk effectively. Over time, a trader who embraces this reality _learns to psychologically harmonize_ with the system’s natural fluctuations, recognizing that _no system wins all the time_. Just as losing streaks are inevitable, so too are winning streaks.

To quantify this aspect, we use Monte Carlo simulation to estimate the range of possible consecutive losing streaks for each win-rate. This helps traders set realistic expectations and build emotional resilience.

Figure 6 presents a box plot showing the distribution of consecutive losing streaks across the different win-rate systems after performing 100 simulation with 500 trades each.

![boxplot](https://c.mql5.com/2/160/boxplot_info.png)

Figure 6: Win-rate vs. Consecutive losses

Table 2 shows the minimum, median and maximum _consecutive losing streak of each win-rate_ as depicted in Figure 6.

| System's win-rate | minimum | median | maximum |
| --- | --- | --- | --- |
| 30% | 10 | 15 | 28 |
| 50% | 5 | 8 | 15 |
| 65% | 4 | 5 | 11 |
| 76% | 3 | 4 | 8 |
| 83% | 2 | 3 | 6 |

**Understanding Losing Streaks: A Trader's Guide to Emotional Resilience**

Understanding the likelihood and extent of consecutive losing streaks is essential for building emotional discipline and effective risk management in trading. Even profitable systems with positive expectancy can encounter extended periods of drawdown. In this section, we analyze the behavior of different win-rate systems: 30%, 65%, and 83% based on results from Monte Carlo simulations, with a focus on how traders can psychologically and strategically prepare for the worst-case scenarios.

- 30% Win-Rate System

Statistical results from the simulation indicate that a 30% win-rate system may experience a minimum of 10, median of 15, and maximum of 28 consecutive losing trades. Being mentally prepared for such prolonged drawdowns is vital. A trader who understands that a 28-trade losing streak is within the range of possibilities can remain emotionally grounded and avoid abandoning the system prematurely.

Accepting this reality allows the trader to apply minimal risk per trade, such as 1% of account balance, reducing the potential drawdown to a _maximum of 28% in the worst-case scenario._ While this is a significant loss, it may be tolerable for traders committed to the system’s long-term profitability. As observed in Figure 6, the _maximum value of 28 is an outlier_, indicating it's statistically less likely but still possible.

Traders must honestly assess whether they have the emotional resilience and capital tolerance to withstand such deep losing streaks. If not, a 30% win-rate system—despite its profitability over time may not be suitable.

- 65% Win-Rate System

For a system with a 65% win-rate, the simulation reveals a minimum of 4, median of 5, and maximum of 11 consecutive losing trades. This is a more forgiving profile compared to lower win-rate systems, yet still presents potential emotional challenges. A trader who risks 1% per trade could face up to an _11% drawdown in a worst-case scenario_. While this is more manageable than the 30% system, it still requires psychological readiness and proper risk controls to stay consistent during downturns.

This system may appeal to swing or intraday traders who prefer balanced setups a solid win-rate with moderate RRR. By understanding and accepting the probability of 11 consecutive losses, the trader is less likely to react emotionally or abandon the system. With disciplined position sizing, the 65% win-rate system offers a smoother equity curve while still achieving long-term profitability. From the figure 6, the 11 maximum consecutive loss is less likely to occur since it falls outside the range.

- 83% Win-Rate System

An 83% win-rate system is often associated with high-frequency or scalping strategies that target small, consistent gains. According to the simulation, such a system may face a minimum of 2, median of 3, and maximum of 6 consecutive losing trades.

Because the likelihood of extended losing streaks is low, traders may be tempted to increase their risk per trade, for example, to 5%. While this may seem acceptable given the system’s high win rate, it's important to recognize that even a _6 trade losing streak at 5% risk can result in a 30% drawdown_, a significant hit to the trading account.

To manage this, traders must set realistic expectations and avoid overconfidence. Accepting that occasional losing streaks will happen, even in high-probability systems, can help prevent emotional breakdowns and impulsive decisions. Building resilience around these numbers is essential for long-term success. The 6 consecutive losing streak is less likely to occur because it fall outside the range. The python code to generate possible consecutive losses is attached in this article.

The code usage:

To evaluate alternative outcomes, modify the parameter values in the following section. This allows for flexible testing under varying conditions while maintaining the underlying analytical framework.

```
import numpy as np
import matplotlib.pyplot as plt

# Systems data: win rates and RRR
win_rates = [0.30, 0.50, 0.65, 0.76, 0.83]
rrrs = [2.6, 1.7, 0.9, 0.6, 0.3]
num_systems = len(win_rates)
num_simulations = 100
num_trades = 500

# Initialize results storage
all_max_consecutive_losses = []
```

Initialize variables and values for computation.

### Position Sizing and its Associated Drawdowns

Position sizing plays a crucial role in trading performance, especially during losing streaks. The fundamental formula for position sizing is:

![position](https://c.mql5.com/2/160/form1.png)

The risk amount can be categorized as either fixed (static) or dynamic (variable). In this section, we explore how each _approach influences drawdowns_ during consecutive losing streaks.

Let define the:

```
fraction of Risk% as f,
previous balance as balance j-1,
current balance as balance j
initial balance as balance i
win-rate as P
```

The risk amounts for each case are defined as follows:

```
Dynamic Risk Amount = Risk% x balance j-1 = f x balance j-1

Fixed Risk Amount = f x balance i
```

**Case 1: Dynamic Risk (Risk % of Current Balance)**

In this model, the risk amount is recalculated for every trade based on the most recent account balance. This allows for compounding during profitable streaks and accelerated decline during drawdowns.

At each trade j:

- If the trade is a _win_:

![df2](https://c.mql5.com/2/160/dform2.png)

- If the trade is a _loss_:

![df2b](https://c.mql5.com/2/160/dform2b.png)

Combining both outcomes probabilistically, we get:

![eqn1](https://c.mql5.com/2/160/eqn1.png)

**Case 2: Fixed Risk (Risk % of Initial Balance)**

In this model, the risk amount remains constant throughout all trades, regardless of account performance. This leads to steady growth during favorable conditions and uniform drawdowns during losing streaks.

- If the trade is a _win_:

![fm3](https://c.mql5.com/2/160/Fform3.png)

- If the trade is a _loss_:

![fm3b](https://c.mql5.com/2/160/Fform3b.png)

Combined, this gives:

![eqn2](https://c.mql5.com/2/160/eqn2.png)

### Simulation Approach

Equations (1) and (2) will be implemented in Monte Carlo simulations to generate a wide range of potential outcomes for _account balance and drawdown under each position sizing strategy._ By comparing the results, traders can better understand how dynamic versus fixed risk exposure affects _equity growth, volatility,_ and _survivability_ of the trading system.

The accompanying charts illustrate single simulation outcomes for each trading system under review. Each system was evaluated using _three risk parameters: 1%, 2%, and 5%_ of either the _current account balance_ or the _initial account balance_. At first glance, the equity curves appear highly promising for this individual test run—similar to the excitement we feel when back-testing over a specific period yields strong results, tempting us to transition to live trading.

To assess the robustness and long-term potential of these systems, we extended our analysis by conducting _500 Monte Carlo simulations_, each consisting of _1,000 trades._ This broader approach allows us to explore a wide range of possible equity trajectories and better understand the variability and risks involved.

The simulation exercise was repeated across multiple system configurations, including:

- Win-rate: 30%, RRR: 2.6
- Win-rate: 65%, RRR: 0.9
- Win-rate: 83%, RRR: 0.3

By evaluating the selected trading systems under varying risk levels of 1%, 2%, and 5% of account equity, we can determine whether it’s feasible to exceed the risk thresholds typically advised by professional traders. These specific risk levels serve as a benchmark to assess the resilience and profitability of different strategies under both conservative and aggressive risk exposures.

The chosen configurations—varying in win-rate and RRR—represent a spectrum of trading strategies, from high-risk/high-reward setups to more conservative, high-probability approaches. This allows for a thorough assessment of performance across different risk profiles.

Additionally, researchers and traders are encouraged to explore other win-rate scenarios, such as 50%, 76%, or any other configuration of interest, to gain more profound insights into potential performance outcomes and system stability under varying conditions.

### Simulation Results Overview

**System 1: Win-Rate = 30%, RRR = 2.6**

Figure 7 through 9 illustrates the equity curve for trades executed at a 1%,2% and 5% risk level. The accompanying table presents aggregated results from 500 Monte Carlo simulations, each comprising 1,000 trades at their respective risk parameter. This comprehensive simulation approach provides statistically significant insights into the system's performance characteristics under controlled risk conditions.

- 1% risk level

![sys30_1pc](https://c.mql5.com/2/160/Sys_30_1per_curves.png)

Figure 7

Table 2: Monte Carlo Simulation results for a system with win-rate=30%, RRR=2.6, 1% risk

| Metrics | Dynamic (1% of Current Balance) | Fixed (1% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $2,143.81 | $1,778.62 |
| Median Final Balance | $1,876.43 | $1,764.00 |
| Min Final Balance | $326.01 | $0.00 |
| Max Final Balance | $7,831.03 | $3,204.00 |
| Mean Drawdown | $649.40 | $398.98 |
| Median Drawdown | $584.29 | $363.00 |
| Max Drawdown | $1,980.21 | $1,278.00 |
| Mean Max Drawdown (%) | 33.72% | 28.47% |
| Median Max Drawdown (%) | 32.02% | 25.72% |
| Max Drawdown (%) | 74.91% | 114.11% |

Key Insight:

For 1% risk, the mean and median final balances indicate a positive outcome, consistent with the system's positive expectancy. The return on investment is approximately 2x the initial equity for both risk models, with dynamic risk achieving a slightly higher return overall.

In some simulations, dynamic risk produced a maximum balance of approximately 8x the initial equity, compared to 3x for fixed risk. This demonstrates dynamic risk's ability to compound gains more aggressively during winning streaks. However, the cost of higher return is higher volatility. Mean and median drawdowns under dynamic risk were larger than those in fixed risk, and while the maximum drawdown under **dynamic risk reached about 75%**, the fixed risk model experienced complete _account blowout in some cases_—exceeding 114% in drawdown, reflecting negative account balances.

The data reveals a clear _survival advantage for dynamic risk sizing (1% of current balance) over fixed risk (1% of initial balance)_ in adverse conditions.

- 2% risk level

![sys30_2pc](https://c.mql5.com/2/160/Sys_30_2per_curves.png)

Figure 8

Table 3: Monte Carlo Simulation results for a system with win-rate=30%, RRR=2.6, 2% risk

| Metrics | Dynamic (2% of Current Balance) | Fixed (2% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $4,418.38 | $2,557.23 |
| Median Final Balance | $2,705.11 | $2,528.00 |
| Min Final Balance | $83.85 | $-1,000.00 |
| Max Final Balance | $46,107.53 | $5,408.00 |
| Mean Drawdown | $2,387.05 | $797.95 |
| Median Drawdown | $1,613.94 | $726.00 |
| Max Drawdown | $22,972.93 | $2,556.00 |
| Mean Max Drawdown (%) | 57.14% | 47.67% |
| Median Max Drawdown (%) | 55.95% | 41.85% |
| Max Drawdown (%) | 94.87% | 212.80% |

Key Insight:

For 2% risk, the system again reflects positive expectancy, but with _increased volatility and divergence between dynamic and fixed outcomes._ The mean final balance under dynamic risk was $4,418.38, significantly higher than the $2,557.23 for fixed risk. However, the median final balance under dynamic risk was higher than fixed, indicating that the equity curve is not stable compared to the fixed risk.

In some of the simulation, dynamic risk resulted in a maximum account growth of over _46x the initial balance_, while fixed risk capped at _around 5.4x._ This showcases the _explosive upside potential of dynamic risk when conditions are favorable._

On the downside, _drawdowns were significantly more severe._ The mean mean/median drawdown hovers around 57%/56% respectively. Some of the simulation results of dynamic risk saw maximum drawdowns nearing 95%, compared to _213% in fixed risk—indicating not only a full loss of capital but debt-level exposure in extreme scenarios._ This level of volatility would be emotionally and financially unsustainable for most traders without strict control mechanisms.

- 5% risk level

![sys30_5pc](https://c.mql5.com/2/160/Sys_30_5per_curves.png)

Figure 9

Table 4: Monte Carlo Simulation results for a system with win-rate=30%, RRR=2.6, 5% risk

| Metrics | Dynamic (5% of Current Balance) | Fixed (5% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $25,741.66 | $4,893.08 |
| Median Final Balance | $1,797.64 | $4,820.00 |
| Min Final Balance | $0.37 | $-4,000.00 |
| Max Final Balance | $1,857,357.21 | $12,020.00 |
| Mean Drawdown | $41,836.96 | $1,994.88 |
| Median Drawdown | $5,488.64 | $1,815.00 |
| Max Drawdown | $4,836,685.44 | $6,390.00 |
| Mean Max Drawdown (%) | 90.30% | 88.15% |
| Median Max Drawdown (%) | 92.07% | 72.69% |
| Max Drawdown (%) | 99.99% | 532.00% |

Key Insight:

At 5% risk, the simulation paints a picture of extreme risk-reward asymmetry. The mean final balance under dynamic risk was an _impressive $25,741.66, compared to $4,893.08 for fixed risk._ However, the _median final balance under dynamic risk was only $1,797.64, signaling that most simulations lost money or barely survived_, despite the high average being skewed by a few outliers. The median for the fixed risk was approximately the same as the mean final balance which suggests the returns on investment _is more predictable and consistent,_ though not necessarily high-return.

In some cases dynamic risk achieved a staggering maximum final balance of over _1,857x the starting capital, reflecting rare but powerful compounding outcomes._ Fixed risk peaked at about 12x. However, this upside came with enormous drawdowns—median drawdowns exceeded 92% in dynamic risk, and maximum drawdowns approached total wipeout at 99.99%.

In fixed risk, for some cases, _maximum drawdown soared to 532%_, again suggesting complete account destruction in worst-case scenarios. While the reward potential is massive, so is the risk— _highlighting why 5% position sizing is often considered reckless for such systems, even if the expectancy is mathematically positive._

**Recommended Risk Levels for a 30% Win-Rate, RRR = 2.6 System**

In a nutshell, for a trading system with a 30% win-rate and reward-to-risk ratio of 2.6, the simulation results suggest that _risking 1% to 2% of the current account balance_ is the most balanced and sustainable approach. This range delivers _good return potential_ on favorable trading days, while keeping _mean and median drawdowns below 60%_, maintaining capital survivability.

On the other hand, _risking 1% to 5% of the initial balance_ provides _more stable and predictable outcomes_—particularly useful for traders seeking consistency. However, this approach is still vulnerable to _total account loss during extended drawdowns_, especially at higher risk levels.

_Risking 5% on either model (fixed or dynamic)_ should generally be avoided, unless the trader is fully aware of the risks and is willing to treat _the account as a high-stakes gamble._ While the upside can be explosive, the probability of _account blowout_ is extremely high. In such cases, the trader must be _psychologically and financially_ prepared to accept any outcome, including _full loss of capital._

**System 2: Win-Rate = 65%, RRR = 0.9**

Figure 10 through 12 illustrates the equity curve for trades executed at a 1%,2% and 5% risk level. The table below summarizes the aggregated outcomes of 500 Monte Carlo simulations, with each simulation running 1,000 trades under specified risk parameters. This extensive simulation framework ensures statistically robust analysis, offering valuable insights into the system's performance under controlled risk conditions.

- 1% risk level

![sys65_1p](https://c.mql5.com/2/160/Sys_65_1per_curves.png)

Figure 10

Table 5: Monte Carlo Simulation results for a system with win-rate=65%, RRR=0.9, 1% risk

| Metrics | Dynamic (1% of Current Balance) | Fixed (1% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $10,408.50 | $3,348.14 |
| Median Final Balance | $9,941.86 | $3,340.50 |
| Min Final Balance | $3,879.52 | $2,400.00 |
| Max Final Balance | $20,668.22 | $4,072.00 |
| Mean Drawdown | $571.54 | $92.96 |
| Median Drawdown | $534.55 | $88.00 |
| Max Drawdown | $1,567.38 | $208.00 |
| Mean Max Drawdown (%) | 8.98% | 5.65% |
| Median Max Drawdown (%) | 8.61% | 5.37% |
| Max Drawdown (%) | 18.99% | 16.07% |

Key Insight:

At 1% risk, the simulation depicts _exponential growth_ for the dynamic risk model, while the fixed risk model exhibits _linear and stable growth._ The mean final balance for dynamic risk reached _10x the initial capital_, while fixed risk returned approximately _3x_. The median final balances for both models are very close to their respective means, indicating _symmetrical performance and low skewness_ in the distribution.

_None of the simulated runs experienced a complete blowout_, suggesting the system is robust and sustainable at this risk level. The maximum drawdown for dynamic risk was about 19%, while fixed risk topped at approximately 16%. The mean and median drawdown percentages were lower in the fixed risk model, suggesting a smoother equity curve with less volatility.

Some simulations under dynamic risk reached a maximum final balance of _20x the initial capital_, while fixed risk peaked around _4x_, reinforcing the compounding advantage of dynamic position sizing.

- 2% risk level

![sys65_2p](https://c.mql5.com/2/160/Sys_65_2per_curves.png)

Figure 11

Table 6: Monte Carlo Simulation results for a system with win-rate=65%, RRR=0.9, 2% risk

| Metrics | Dynamic (2% of Current Balance) | Fixed (2% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $107,088.02 | $5,696.28 |
| Median Final Balance | $90,573.24 | $5,681.00 |
| Min Final Balance | $13,775.17 | $3,800.00 |
| Max Final Balance | $391,748.08 | $7,144.00 |
| Mean Drawdown | $9,787.36 | $185.92 |
| Median Drawdown | $8,491.57 | $176.00 |
| Max Drawdown | $39,558.61 | $416.00 |
| Mean Max Drawdown (%) | 17.33% | 9.48% |
| Median Max Drawdown (%) | 16.71% | 8.79% |
| Max Drawdown (%) | 34.71% | 31.14% |

Key Insight:

At 2% risk, dynamic risk shows a _significant jump in growth potential with a mean final balance of over $107,000—that’s 107x the initial capital._ The median balance, at _approximately $90,573_, also _confirms strong, consistent growth._ Meanwhile, fixed risk produced a _mean final balance of $5,696, nearly 6x the initial capital,_ and the median _closely matched, indicating consistent and predictable results._

The maximum final balance under _dynamic risk exploded to over $391,000_, while fixed risk maxed at _just $7,144._ This highlights the huge upside potential of dynamic risk during extended winning streaks.

On the downside, _maximum drawdown under dynamic risk rose to 34.71%,_ while fixed risk stayed _relatively controlled at 31.14%._ Mean and median drawdown percentages also followed this pattern— _dynamic risk was higher, reflecting greater exposure to volatility, while fixed risk remained more stable but less rewarding._

- 5% risk level

![sys65_5p](https://c.mql5.com/2/160/Sys_65_5per_curves.png)

Figure 12

Table 7: Monte Carlo Simulation results for a system with win-rate=65%, RRR=0.9, 5% risk

| Metrics | Dynamic (5% of Current Balance) | Fixed (5% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $106,016,246.65 | $12,740.69 |
| Median Final Balance | $40,611,334.02 | $12,702.50 |
| Min Final Balance | $362,423.53 | $8,000.00 |
| Max Final Balance | $1,591,375,537.03 | $16,360.00 |
| Mean Drawdown | $18,402,293.20 | $464.79 |
| Median Drawdown | $7,730,838.44 | $440.00 |
| Max Drawdown | $268,258,572.26 | $1,040.00 |
| Mean Max Drawdown (%) | 38.94% | 18.20% |
| Median Max Drawdown (%) | 38.28% | 16.16% |
| Max Drawdown (%) | 66.90% | 71.24% |

Key Insight:

At 5% risk, dynamic risk turned into a _high-stakes compounding machine, with an astronomical mean final balance exceeding $106 million,_ and a _median of $40 million._ The maximum balance in a single simulation hit _$1.59 billion, demonstrating how compounding can produce exceptional outcomes given a strong system._

However, _this explosive growth comes with extreme volatility._ The mean drawdown was over _$18 million_, and the maximum drawdown exceeded _$268 million, with drawdown percentages ranging from 38% to 67%_—levels that would be _emotionally and financially intolerable_ for most traders without strict discipline and risk buffers.

Fixed risk, in contrast, was much _more controlled._ The mean and median final balances were around _$12,700_, with max balance peaking at _$16,360_—roughly _16x the initial capital_. While modest in comparison, this approach offers _greater consistency with maximum drawdowns capped at 71%_, and more realistic levels of volatility for long-term sustainability.

Recommended Risk Levels for System 3 (65% Win Rate, RRR = 0.9)

Based on simulation data, _1% to 2% risk of current balance offers the best balance between growth and capital protection._ Dynamic risk allows for greater upside due to compounding, while fixed risk offers a smoother, more predictable return profile.

At 5% risk, dynamic sizing can _generate extraordinary returns, but with drawdowns so large that they are unsuitable_ for most traders unless they are fully prepared to accept extreme volatility and account fluctuations. Fixed risk at 5% performs reasonably well, but with reduced upside.

In practice, _1% to 2% dynamic risk is recommended for this system, especially for traders who want to preserve capital and grow steadily_ without being exposed to unbearable emotional and financial swings.

For some traders, _risking 5% of the initial balance_ may be a preferred strategy for this system _due to its low mean and median drawdown percentages_, along with a mean final return of approximately 12x the initial equity, which is both impressive and statistically more predictable. While the maximum _drawdown of 71%_ remains a possibility, it is considered a _low-probability_ event within the simulations—making this approach _acceptable for traders who can tolerate moderate risk in exchange for stable growth potential._

**System 3: Win-Rate = 83%, RRR = 0.3**

This system represents a _high-probability, low-reward model_, often used in _scalping strategies_ or systems with frequent small wins.

Figure 13 through 15 depicts the equity curve for trades executed at a 1%,2% and 5% risk level. The table below summarizes the aggregated outcomes of 500 Monte Carlo simulations, with each simulation running 1,000 trades under specified risk parameters. Through an extensive simulation framework, the analysis achieves statistical robustness, providing essential insights into system performance under carefully managed risk conditions.

-  1% risk level

![sys83_1p](https://c.mql5.com/2/161/Sys_83_1per_curves__3.png)

Figure 13

Table 8: Monte Carlo Simulation results for a system with win-rate=83%, RRR=0.3, 1% risk

| Metrics | Dynamic (1% of Current Balance) | Fixed (1% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $2,202.86 | $1,790.73 |
| Median Final Balance | $2,176.49 | $1,790.00 |
| Min Final Balance | $1,452.51 | $1,387.00 |
| Max Final Balance | $3,391.51 | $2,232.00 |
| Mean Drawdown | $118.36 | $72.50 |
| Median Drawdown | $112.90 | $68.00 |
| Max Drawdown | $256.29 | $143.00 |
| Mean Max Drawdown (%) | 7.07% | 5.50% |
| Median Max Drawdown (%) | 6.71% | 5.10% |
| Max Drawdown (%) | 13.55% | 11.60% |

Key Insight

At 1% risk, the system showed _modest but consistent growth_ across both models. The mean final balance under dynamic risk was _$2,202.86—roughly 2.2x the starting equity_—while fixed risk resulted in _$1,790.73, about 1.8x_. The median final balances were very close to their respective means, indicating tight distribution and low volatility in outcomes.

Importantly, _no simulations experienced account blowouts_, and the system proved _statistically stable_. The maximum drawdown for dynamic risk was just _13.55%_, while fixed risk stayed even lower at _11.60%_. The mean and median drawdown percentages were also smaller in the fixed model, reinforcing its smoother equity curve.

In some simulations, dynamic risk reached a max balance of _3.4x the initial equity_, while fixed risk maxed out at _2.2x,_ showcasing the limited but _stable upside_ of this high-win-rate system.

-  2% risk level

![](https://c.mql5.com/2/161/Sys_83_2per_curves__1.png)

Figure 14

Table 9: Monte Carlo Simulation results for a system with win-rate=83%, RRR=0.3, 2% risk

| Metrics | Dynamic (2% of Current Balance) | Fixed (2% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $4,842.45 | $2,581.46 |
| Median Final Balance | $4,621.22 | $2,580.00 |
| Min Final Balance | $2,052.24 | $1,774.00 |
| Max Final Balance | $11,256.45 | $3,464.00 |
| Mean Drawdown | $443.14 | $144.99 |
| Median Drawdown | $407.10 | $136.00 |
| Max Drawdown | $1,214.00 | $286.00 |
| Mean Max Drawdown (%) | 13.77% | 9.42% |
| Median Max Drawdown (%) | 13.13% | 8.63% |
| Max Drawdown (%) | 25.67% | 21.47% |

Key Insight:

At 2% risk, the _performance divergence between dynamic and fixed models becomes clearer._ Dynamic risk produced a mean final balance of _$4,842.45 (almost 5x starting capital),_ compared to _$2,581.46 under fixed risk (2.6x)_. The median final balances closely matched the means, confirming _predictability and consistent performance._

The maximum final balance under dynamic risk hit _$11,256.45_ **,** indicating that a few strong streaks can produce substantial growth, while fixed risk peaked at _$3,464_. However, maximum drawdowns also increased, with dynamic risk hitting _25.67%_, and fixed risk maintaining a lower _21.47%_.

This scenario reflects a moderate risk-reward profile where compounding under dynamic risk enhances returns, while fixed risk offers lower volatility with limited upside.

-  5% risk level

![sys83_5p](https://c.mql5.com/2/161/Sys_83_5per_curves__1.png)

Figure 15

Table 10: Monte Carlo Simulation results for a system with win-rate=83%, RRR=0.3, 5% risk

| Metrics | Dynamic (1% of Current Balance) | Fixed (1% of Initial Balance) |
| --- | --- | --- |
| Mean Final Balance | $50,642.20 | $4,953.64 |
| Median Final Balance | $38,003.88 | $4,950.00 |
| Min Final Balance | $4,884.41 | $2,935.00 |
| Max Final Balance | $360,637.45 | $7,160.00 |
| Mean Drawdown | $9,127.86 | $362.48 |
| Median Drawdown | $7,201.27 | $340.00 |
| Max Drawdown | $62,063.85 | $715.00 |
| Mean Max Drawdown (%) | 31.82% | 18.15% |
| Median Max Drawdown (%) | 30.97% | 16.25% |
| Max Drawdown (%) | 54.29% | 49.28% |

Key Insight:

At 5% risk, the system's potential _explodes under dynamic risk_, yielding a mean final balance of over _$50,000 (50x initial capital)_ and a median of _$38,003_, indicating a strong bias toward profitable outcomes. The maximum final balance was an _extraordinary $360,637,_ confirming the system’s ability to compound small wins into massive returns.

However, this comes at the cost of _high volatility—with mean drawdowns exceeding $9,000_ and maximum drawdowns hitting over _$62,000_. The mean drawdown percentage was nearly _32%_, and the _worst-case scenario reached 54.29%_, which could still be tolerable for well-capitalized traders but may challenge _psychological resilience._

For fixed risk, the performance was more contained. The mean final balance was _$4,953.64 (about 5x capital)_, with a _maximum of $7,160._ Drawdowns were significantly lower than dynamic, with median max drawdown around _16%_, and maximum drawdown capped at _49.28%._

Recommended Risk Levels for System 5 (83% Win Rate, RRR = 0.3)

For a system with such a _high win rate and low RRR_, the simulations affirm it is statistically stable and resilient, especially under conservative risk.

- _1% to 2% risk_ of current balance is optimal for most traders—offering a balance of consistency, growth, and manageable drawdowns.
- Dynamic risk enhances performance through compounding but introduces slightly higher drawdowns.
- Fixed risk produces a more predictable equity curve, suited for traders prioritizing low volatility.

At 5% risk, dynamic risk delivers exceptional upside but demands _emotional discipline_ to endure drawdowns in the _30–50% range_. Fixed risk at 5% yields predictable _5x returns_ with maximum drawdowns under 50%, making it attractive to traders _seeking steady results without extreme swings._

For some traders, risking _5% of the initial balance_ may be a preferred approach for this system, given its low mean and median drawdown percentages and a predictable return potential of approximately _5x the initial capital._ Although the worst-case drawdown reached 49.28% **,** it was a statistically rare outcome in the simulations and could be tolerable for traders comfortable with moderate risk exposure.

For others, risking _5% of the current balance (dynamic model)_ may be more appealing due to its _exceptional return potential_, with a mean final balance exceeding _50x and a median of 38x the initial equity._ Despite the aggressive risk sizing, the system exhibited moderate drawdown behavior, with mean and median drawdowns _around 32%, and a maximum drawdown of 54.29%._ While this level of volatility is significant, it remains within acceptable limits for experienced traders who are prepared—both emotionally and financially—for a high-compounding, high-variability strategy.

For this system, _risking beyond the traditional 2% rule_ is a viable path to achieving trading objectives, _but it requires a robust and reliable trading system to carry you through periods of drawdown and reach your target._

Code Usage

To evaluate alternative outcomes, modify the parameter values in the following section. This allows for flexible testing under varying conditions while maintaining the underlying analytical framework.

```
# Parameters
win_rate = 0.83
rrr = 0.3
num_trades = 1000
initial_balance = 1000
risk_percent = 0.05  # %

np.random.seed(42)

#The use of a predefined seed (e.g., 42) enables result reproducibility.
#Users may change this value/deactivate the seeding logic by commenting out the relevant code block.
```

### Conclusion

In this article, we applied _Monte Carlo simulation_ to objectively quantify the impact of _consecutive losing streaks_ and explore how they interact with different win-rate systems and position sizing strategies. The results indicate that _trading systems with lower win rates_ tend to experience _longer losing streaks_, while _higher win-rate systems_ experience shorter and less frequent streaks.

However, despite the psychological challenges associated with low win-rate systems, those with _positive expectancy_ (i.e., a sufficiently high reward-to-risk ratio) can deliver _significantly higher returns_ when exposed to the same risk levels. This reinforces the point that win rate alone doesn't define profitability—expectancy and risk management are just as important.

Our analysis also revealed that _drawdown severity is directly influenced by both the length of losing streaks and the level of risk per trade._ Naturally, the _higher the risk_, the greater the potential drawdown—but also the greater the potential reward, especially for systems with a mathematical edge.

While the traditional _1%-2% risk rule_ remains a sound foundation, our results suggest that _traders with a robust and proven system can justify going beyond the 2% threshold._ For example, the _83% win-rate, RRR = 0.3 system_ demonstrated stable drawdowns even at _5% risk_, particularly under the fixed risk model. In fact, if such a system had a slightly better RRR (e.g., 0.5), risking more than 2% could be even more viable—though this should be tested thoroughly using simulation tools.

Overall, our simulations consistently showed that _dynamic risk (based on current balance)_ outperformed fixed risk in terms of long-term returns. However, _fixed risk models_ offered _more stable and predictable outcomes_ in certain cases—such as the _5% risk setting in the 65% win-rate, RRR = 0.9 system._

This raises a broader question: _Can we set specific trading targets and reliably achieve them using a well-defined trading system?_

We believe this is at the heart of successful trading— _turning strategy into targeted results_. In the next article, we will explore this concept further and demonstrate how traders can set and pursue realistic profit goals using simulation-backed strategies.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/18991.zip "Download all attachments in the single ZIP archive")

[consecutiveloss.py](https://www.mql5.com/en/articles/download/18991/consecutiveloss.py "Download consecutiveloss.py")(2.92 KB)

[PositionSizing\_simulation.py](https://www.mql5.com/en/articles/download/18991/positionsizing_simulation.py "Download PositionSizing_simulation.py")(4.49 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing a Trading Strategy: Using a Volume-Bound Approach](https://www.mql5.com/en/articles/20469)
- [Developing a Trading Strategy: The Flower Volatility Index Trend-Following Approach](https://www.mql5.com/en/articles/20309)
- [Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)
- [Developing a Trading Strategy: The Triple Sine Mean Reversion Method](https://www.mql5.com/en/articles/20220)
- [Developing a Trading Strategy: The Butterfly Oscillator Method](https://www.mql5.com/en/articles/20113)
- [Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://www.mql5.com/en/articles/19693)
- [Building a Trading System (Part 4): How Random Exits Influence Trading Expectancy](https://www.mql5.com/en/articles/19211)

**[Go to discussion](https://www.mql5.com/en/forum/492914)**

![Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://c.mql5.com/2/162/18971-python-metatrader-5-strategy-logo__1.png)[Python-MetaTrader 5 Strategy Tester (Part 01): Trade Simulator](https://www.mql5.com/en/articles/18971)

The MetaTrader 5 module offered in Python provides a convenient way of opening trades in the MetaTrader 5 app using Python, but it has a huge problem, it doesn't have the strategy tester capability present in the MetaTrader 5 app, In this article series, we will build a framework for back testing your trading strategies in Python environments.

![Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://c.mql5.com/2/162/19052-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 2): Expert Advisor, Backtests, and Optimization](https://www.mql5.com/en/articles/19052)

This article presents a sample Expert Advisor implementation for trading a basket of four Nasdaq stocks. The stocks were initially filtered based on Pearson correlation tests. The filtered group was then tested for cointegration with Johansen tests. Finally, the cointegrated spread was tested for stationarity with the ADF and KPSS tests. Here we will see some notes about this process and the results of the backtests after a small optimization.

![MQL5 Trading Tools (Part 8): Enhanced Informational Dashboard with Draggable and Minimizable Features](https://c.mql5.com/2/162/19059-mql5-trading-tools-part-8-enhanced-logo__2.png)[MQL5 Trading Tools (Part 8): Enhanced Informational Dashboard with Draggable and Minimizable Features](https://www.mql5.com/en/articles/19059)

In this article, we develop an enhanced informational dashboard that upgrades the previous part by adding draggable and minimizable features for improved user interaction, while maintaining real-time monitoring of multi-symbol positions and account metrics.

![From Novice to Expert: Animated News Headline Using MQL5 (VIII) — Quick Trade Buttons for News Trading](https://c.mql5.com/2/160/18975-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (VIII) — Quick Trade Buttons for News Trading](https://www.mql5.com/en/articles/18975)

While algorithmic trading systems manage automated operations, many news traders and scalpers prefer active control during high-impact news events and fast-paced market conditions, requiring rapid order execution and management. This underscores the need for intuitive front-end tools that integrate real-time news feeds, economic calendar data, indicator insights, AI-driven analytics, and responsive trading controls.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/18991&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049217137538606916)

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