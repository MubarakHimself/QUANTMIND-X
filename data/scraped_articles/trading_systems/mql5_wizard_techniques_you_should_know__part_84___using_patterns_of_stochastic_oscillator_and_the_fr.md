---
title: MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion
url: https://www.mql5.com/en/articles/19890
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:48:46.719265
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/19890&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049397457445563084)

MetaTrader 5 / Trading systems


### Introduction

From the [last article](https://www.mql5.com/en/articles/19857), we examined the first 5 signal patterns of the Indicator pairing Stochastic-Oscillator and Fractal Adapting Moving Average. From our small test window, all appeared to have profitable forward walks, with training done over a year and the validation performed over the subsequent year.

We performed these tests while being mindful of the patterns’ suitable market types, whilst also using ‘appropriate’ assets for each market type. The market archetypes we considered were trending/mean-reverting, auto-correlated/decoupled, and highly-volatile/low-volatility markets. Within these types we tried to attribute particular asset types that are better exploited by these patterns, and based on the forward results, our selection could have been appropriate.

We therefore maintain the same asset-type/market-archetype pairing of the last article as we consider the remaining 5 signal patterns of this indicator pairing.

### Pattern-5: Low Volatility Flat line Breakout

Our sixth pattern, pattern-5, is designed to work in calm or low volatility environments. The requirements for this therefore are identifying price compression zones that have both FrAMA and price flattening prior to a pop in a given direction. In essence, we seek to capture the moment volatility starts to breakout, following an extended contraction. Such regimes are no strangers to the asset we tested them on in the last article, and that asset was USD JPY. Alternations between sharp bursts and long consolidations are very common with this forex pair. We implement this as follows in MQL5:

**Buy signal (Pattern 5):** Flat FrAMA + Stoch cross up under 30

![p5buy](https://c.mql5.com/2/174/p5_buy.png)

```
//+------------------------------------------------------------------+
//| Check for Pattern 5.                                             |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::IsPattern_5(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return((FlatFrama(X()) && CrossUp(K(X()), D(X()), K(X() + 1), D(X() + 1)) && K(X()) < 30)  ? true : false);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return((FlatFrama(X()) && CrossDown(K(X()), D(X()), K(X() + 1), D(X() + 1)) && K(X()) > 70) ? true : false);
   }
   return(false);
}
```

The internal logic of this pattern depends on the ‘FlatFrama()’ method, whose code was shared in the last article. This function basically measures the degree to which FrAMA has been flat when using a tolerance that is tuned by the hyperparameter ‘m\_pips’. For completeness, we share this same function below:

```
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::FlatFrama(int ind)
{  const double tol = m_pips * m_symbol.Point();
   for(int i = ind; i < m_past+ind; i++)
      if(MathAbs(FrAMASlope(i)) > tol) return false;
   return true;
}
```

Our boolean logic above simply checks for the ‘flatness’ of the FrAMA slope across a look-back window, another optimized hyperparameter. This check returns true, if indeed, there has been low volatility and a lack of direction over the prior look back period. When we pair this with the Stochastic-Oscillator, and use the stochastic cross-over when close to the threshold zones of 30 or 70, the pattern is essentially trying to anticipate a volatility breakout. To sum up therefore, the intended market behavior is to spot a quiet regime, wait for a stochastic crossover that could spark a momentum push, then enter preemptively, anticipating volatility to spike.

The use of USD JPY for this was important because compression into narrow ranges when between macro-events, is what USD JPY is known for. It tends to offer ideal theoretical ground for volatility-breakout models. So FrAMA’s adaptiveness would confirm USD JPY’s contraction phases while the stochastic re-activation spots early directional bias. When training on this pair, subsequently, several of these instances were present, multiple times, especially in January 2024’s BOJ’s policy window. This meant we had perfect low volatility setups. However, what ended up happening was, as we transitioned into the expected high volatility regimes, especially post 2024 when forward testing, the environment for such setups disappeared with the volatility regimes being mostly erratic. This exposed the pattern’s dependency on specific market regimes and in part explains why we were not able to forward walk profitably with this pattern as indicated in the report below;

![r5](https://c.mql5.com/2/174/r5.png)

In the forward pass, the FrAMA check for flatness was often ‘true’ across multiple bars, even in volatility spikes, and this led to false breakouts as well as late entries. The model was seeking quiet markets but found erratic volatility. Our binary conditions based only on flatness are unable to adapt, fast enough, to new volatility regimes. This could point to the need to have a dynamic and not static, look back period.

### Pattern-6: Trend Resumption, ‘W’ and ‘M’ Formations

This pattern, is designed to capture resumptions in trends after a correction. By being able to identify recurring momentum formations whose visual representation is akin to a ‘W’ for bullish formations and ‘M’ for the bearish, within the stochastic oscillator, we are identifying the spots where momentum dips, stabilizes, and then resumes by reasserting itself in the direction of the main trend. This pattern also merges FrAMA’s slope as a directional check, such that only setups with the prevailing adaptive trend are deemed valid. On paper this makes it amount to a classic trend-continuation-algorithm, that could be ideal for the likes of Gold, where multi-leg directional-waves, are common place. We implement this as follows in MQL5.

**Sell signal (Pattern 6):** Stoch “M” peak above 80 + FrAMA slope up

![p6sell](https://c.mql5.com/2/174/p6_sell.png)

```
//+------------------------------------------------------------------+
//| Check for Pattern 6.                                             |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::IsPattern_6(ENUM_POSITION_TYPE T)
{  bool W = (K(X() + 2) < 20 && K(X() + 1) > K(X() + 2) && K(X()) < 20 && K(X()) > K(X() + 1));
   bool M = (K(X() + 2) > 80 && K(X() + 1) < K(X() + 2) && K(X()) > 80 && K(X()) < K(X() + 1));
   if(T == POSITION_TYPE_BUY)
   {  return((FrAMASlope(X()) < 0 && W)  ? true : false);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return((FrAMASlope(X()) > 0 && M) ? true : false);
   }
   return(false);
}
```

From our source above, to go long, we are seeking a W formation, with the stochastic K forming 2 troughs below 20 with the second of these troughs being higher than the first. In order to go short, we are after a K formation of 2 peaks when above the 80. Again, as with the bullish, the second peak needs to be below the first. In with instance, the FrAMA slope serves as a regime filter. It checks whether these formations are happening in a correctional phase or not. The structure is aiming to spot when momentum pause is exhausted, and the primary trend is about to resume, a setup designed for mid-trend re-entries.

The use of Gold as our asset to test out this signal pattern was because of Gold’s ‘trend-mechanics’. It is often a solid candidate for making re-entries. It also often shows long impulsive legs interspersed with small retracements, a ‘fertile-ground’ for our sought ‘W/M’ stochastic formations. In fact, Gold’s strong institutional and retail participation often implies that momentum oscillators such as the stochastic oscillator show a lot of harmonic repetition in mid-trend pauses. The slope of the FrAMA was then intended to filter the stochastic readings by ensuring alignment with the underlying trend context. When training/ optimizing this signal pattern was promising, successfully identifying these wave based continuations, however we later realize that this success is dependent on the consistency of volatility cycles. These change significantly in the forward test, and it is reflected in the results as shown below:

![r6](https://c.mql5.com/2/174/r6.png)

So, pattern-6 does well in structured trends as was the case from the 2023, runs but not as well in variable volatility phases as shown in 2025. Gold’s ‘rhythm’ altered in 2024 to 2025 with pullbacks being a bit deeper and less symmetric, undermining the ‘W/M’ pattern’s geometry. The unprofitable forward walk demonstrates tendencies for static geometric recognition, despite using binary encoded signals. As suggested for pattern-5, improvements for pattern-6 could consider adaptive windowing that could include volatility adjusted look back bar counts, or using a dynamic FrAMA slope threshold to infuse some robustness.

In a sense, this signal emphasizes the important trade-off that exists between precision and adaptability. While the ‘W/M’ logic was able to spot clean pullbacks in the training trend price action, the changing volatility symmetry in forward markets left it unable to cope. So, for sustained deployment, besides a dynamic look back for robustness, additional context awareness from reinforcement learning, with MQL5 coded and not ONNX imported models, could be explored.

### Pattern-7: Mean-Reverting FrAMA Touch-Reversal

While pattern-6 struggles with volatility symmetry, our next pattern, is designed to ‘thrive’ on it. Pattern-7’s design embodies a mean-reversion rebound logic that gets activated when price touches the adaptive mean, FrAMA while the stochastic oscillator exits at an extreme. This signal-pattern aims at those moments when price briefly reconnects with equilibrium before resuming its oscillatory cycle. This is somewhat an archetype of ‘buy-the-dip’ or ‘sell-the-rally’.

We select Gold as the asset to test out this pattern because it alternates a lot between directional impulses and calm consolidation periods. Gold’s rhythm is rich in touch and recoil sequences. Particularly if one studies its chart going back to say 1980, its consolidation periods are very significant. From April 1980 to July 2006, it was essentially in a consolidation phase. Similarly, recently, from July 2012 to July 2020 we got another consolidation band. Presently, on a large time frame, we are on the move in fact we are at all-time highs. With this backdrop, therefore, the FrAMA would provide an equilibrium that is adaptive and the Stochastic’s K slope would help sharpen short term entries. We implement this in MQL5 as follows:

**Buy signal (Pattern 7):** Price touches FrAMA + Stoch rises past 30

![p7buy](https://c.mql5.com/2/174/p7_buy.png)

```
//+------------------------------------------------------------------+
//| Check for Pattern 7.                                             |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::IsPattern_7(ENUM_POSITION_TYPE T)
{  bool touch = (Low(X()) <= FrAMA(X()) && High(X()) >= FrAMA(X()));
   if(T == POSITION_TYPE_BUY)
   {  return((touch && K(X() - 1) < 20 && K(X()) > 30)  ? true : false);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return((touch && K(X() - 1) > 80 && K(X()) < 70) ? true : false);
   }
   return(false);
}
```

Our listing above checks whether price candle range crosses the FrAMA line. This ensures the market actually interacted with the adaptive mean. For the buy signal, the stochastic K needs to shift upward from below the 20-level to above 30. This represents an exit from the oversold zone. For the sell signal, the K buffer needs to turn downward from above 80 to close at or below 70. Again, this signals the beginning of a mean-reversion from an overbought condition. Our code essentially determines if price met the mean, and that momentum reversed. This could be a reliable counter swing condition, particularly in choppy environments.

We choose Gold for this given its extended consolidation periods, and the importance of identifying when they flip to trends. FrAMA adapts smoothly to Gold’s volatility cycles, by having its mean level recalibrated in real time. Even on smaller timeframes like the 4-hour we are testing on, Gold drifts a lot to its adaptive mean after impulsive extensions, and most of the time whenever it touches the FrAMA, these ‘touch-events’ tend to coincide with the exhaustion of speculation position, a swing in momentum. The stochastic oscillator therefore serves to confirm the new momentum away from the mean, acting as a filter and lessening the risk of fading into a trend continuation. Following our training over 2023 to 2024, the forward walk report showed some profit, as indicated below:

![r7](https://c.mql5.com/2/174/r7.png)

The chosen inputs, when training pattern-7, yielded a moderate trade frequency of about one trade every 4 to 5 bars, which is almost one trade per day. This points to efficient filtering, that we were able to achieve when the optimizer converged at 73, using the complex criterion that typically targets the 100 value. This, on paper, should suggest consistent contribution to profitability. Within the test itself, performance appears to have peaked between October 2023 and March 2024, which were periods marked by oscillatory post-trend consolidations. This validates the pattern's reversion-like design.

When forward testing, signal pattern-7 stayed robust in spite of changing volatility. The FrAMA - touch condition automatically was able to scale with amplitude. A few textbook narratives played out in early 2025 when Gold corrected within a range followed by price piercing the FrAMA with a stochastic rebound. This produced a lot of clean reversals. Importantly, the previous K value gave us a confirmation that was lagging, and yet we did not sacrifice profits from late entries.

So to sum this up, pattern-7 makes the case that mean-reversion can exist with adaptive averages, when encoded as discrete boolean logic. Boolean in the sense that we always need multiple indicator signals to all be registered before a pattern is declared present. Along with a signal we saw in the last article, I think it was pattern-1, the two could be paired to form a resilient regime-switching backbone.

### Pattern-8: High Volatility Midline Cross Momentum

While the pattern we saw at the end of the last article (pattern-4) captured edges of volatility bursts, our ninth signal pattern focuses on the internal momentum transition that follows volatility. This pattern aims to spot times when price, under conditions of high volatility, crosses back through FrAMA’s midline with renewed strength. This often means that volatility is not only increasing, but it is reasserting direction.

Put differently, this pattern is about volatility continuation, trading the second impulse of volatility after a breakout of the first. By requiring both a FrAMA midline cross and a mid-range stochastic confirmation, this pattern sees to it that noise-driven spikes are filtered out. This happens while genuine directional extensions are captured. We implement it in MQL5 as follows:

```
//+------------------------------------------------------------------+
//| Check for Pattern 8.                                             |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::IsPattern_8(ENUM_POSITION_TYPE T)
{  // Midline (50) cross via synthetic “moving” reference
   bool crossUpMid   = (K(X() + 1) <= 50.0 && K(X()) > 50.0);
   bool crossDownMid = (K(X() + 1) >= 50.0 && K(X()) < 50.0);
   if(T == POSITION_TYPE_BUY)
   {  return((FramaTurningUp()   && crossUpMid)  ? true : false);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return((FramaTurningDown() && crossDownMid) ? true : false);
   }
   return(false);
}
```

This setup, as indicated above, will trigger a boolean signal when the close price crosses FrAMA in the same direction as the stochastic oscillator’s midline bias. The buy signal is when the stochastic K is above 55 as a confirmation of continuing long momentum. The sell signal is when the stochastic K crosses the 45 level to close below it, confirming that bearish pressure is sustained. Our ninth pattern aims to avoid overreactions to short-term overshoots by not focusing on the typical extreme zones and instead dwelling on the mid-range oscillator levels. It tracks mature volatility-driven continuations.

Our choice of USD JPY for this pattern was, as with previous similar signals, its volatility-sensitivity. The dynamics of USD JPY as already argued are usually marked by alternations between low volatility spells, and sharp directional follow-throughs. Pattern-8 is meant to capitalize on these phase 2 accelerations when price crosses back above or below the adapting-mean, after an initial volatility burst.

FrAMA’s ability to adapt ensures the midline reflects the prevailing structural mean even in situations when market regime’s shift. The stochastic’s 50ish level threshold confirms whether this move has legs or it is a reflection of noise. This pattern was able to forward walk profitably, as we can see in the report below:

![r8](https://c.mql5.com/2/174/r8.png)

When optimizing/training, the best input settings showed solid volatility-aligned profitability in the trained period. The best performance was centered around two intervals; in late 2023 and early 2024. This was from exploiting large directional moves, following extended compression. We were able to use the K thresholds in the 45/55 range, confirming that mid-band cross momentum can carry some persistence and perform better than extreme stochastic oscillator zones in volatile phases.

The forward walk was profitable, especially in early 2025 when the BOJ shocked the world by hiking their interest rates from 0.25 to 0.5%. With the USD JPY elevated at that time, the directional follow-through was intermittent. This signal pattern avoided the false flags, triggering only after valid cross-momentum conditions were in play. Notably as well, the drawdown remained moderate at about 10.4% proving that the pattern’s ability to sustain performance in changing volatility scenarios was baked in.

### Pattern-9: Decoupled Overextension Reversal

Our final signal-pattern is built as a contrarian decoupling detector. It is meant to capture exhaustion when price momentum stretches too far off its adaptive mean, which for our purposes is the FrAMA, and subsequently indicates divergent oscillator behavior. This pattern seeks to profit from short-term dislocations between the structural trend and stochastic readings, a common occurrence in indices such as the SPX 500 where composite correlated sectors can overshoot prior to reverting. We implement this in MQL5 as follows:

**Sell signal (Pattern 9):** Upward FrAMA + Stoch > 90 and falling

![p9sell](https://c.mql5.com/2/174/p9_sell.png)

```
//+------------------------------------------------------------------+
//| Check for Pattern 9.                                             |
//+------------------------------------------------------------------+
bool CSignalFrAMA_Stochastic::IsPattern_9(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return((FrAMASlope(X()) < 0 && K(X()) < 10 && K(X()) >= K(X() + 1))  ? true : false);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return((FrAMASlope(X()) > 0 && K(X()) > 90 && K(X()) <= K(X() + 1)) ? true : false);
   }
   return(false);
}
```

The buy signal is when price falls significantly beneath the FrAMA and the stochastic gets to deep oversold territory. The sell signal is also when price is trading way above the adaptive mean and the oscillator is in the extreme overbought zone. We use a 3x point distance factor to help quantify these over extensions. This factor is meant to ensure the signal only gets activated or triggered  when we have pronounced deviations. The oscillator component also needs to confirm that internal momentum is saturated in order to imply a high-probability reversal.

We use the SPX 500 Index given its composite nature, it also displays synchronized extensions, from its underlying constituent sectors, before having mean corrections. These crowd-incited overextensions do give an excellent testing environment for signals that are contrarian, like with this pattern. The thesis for this is that in the past broad indices have tended to show sector rotation induced decoupling, where, for example, when technology and energy diverge in price action even though the index is printing new highs. Despite all these good intentions, this pattern, like 5 and 6 did not favorably forward walk. This could be that the SPX 500’s structural resilience that involves ETF rebalancing can limit how long such decoupling last. Because of this, what may initially seem like an overextension can easily peter-out and become a momentum continuation. This can lead to premature reversal entries. The forward walk report is as follows:

![r9](https://c.mql5.com/2/174/r9.png)

When ‘training’ this pattern indicated solid in-sample performance particularly in from THE mid 2023 to the start of 2024 consolidation period. This was when minor pullbacks followed short-lived overshoots. Also, we can say, albeit posthumously, that because our optimization window contained several balanced volatility phases, the pattern appeared well-calibrated. Its boolean threshold check appeared to ‘cleanly’ isolate overextensions with rapid rebounds. In spite of this, most of the profits came from only three clusters, which revealed temporal concentration as opposed to robustness.

In the forward walk, the ‘chickens went to the roast’. The 2025 market was rotational, with price routinely exceeding the 3x point deviation while remaining in sustained trends. The stochastic oscillator’s K buffer also mostly remained pinned near or at the extremes, thus invalidating the assumption that overbought/oversold means exhausted. Instead of marking reversals, the binary rule tended to fade momentum in macro driven continuations. In summary, we ‘now know’ that pattern-9 confused volatility persistence for exhaustion, a classic failure when using overextension logic in markets that are dynamic.

### Comparative Insights

Once again, our exploration of ten boolean encoded signal patterns that this time bring together the Stochastic-Oscillator and the Fractal Adaptive Moving Average has shed some insights into how classic indicators can be transformed into context aware signals that are potentially suited for automated trades. We share code for use in an MQL5 Wizard, to assemble an Expert Advisor because this gives us a rapid way to prototype new ideas.

Since we encoded each pattern as a boolean condition, requiring true only if all pattern conditions are met, else false, this in a sense behaved like a vectorized filter for market noise. This is able to dynamically map sieved chart conditions into a compute-efficient binary state. This ‘binary-state’, or pattern outputs make them easily adoptable to machine learning models which we will explore in the next article where we consider inference learning. They are part of a ‘pipeline’ that is normalizing data for a forecasting model. The results from forward testing our 10 signal patterns are tabulated as follows.

| pattern | market type | asset tested | optimization profit factor | forward profit factor | status | comment |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | trending | XAU USD | 1.66 | 1.48 | Walked | Strong directional continuity |
| 1 | mean-reverting | XAU USD | 1.52 | 1.39 | Walked | Stable oscillation recovery |
| 2 | auto-correlated | SPX 500 | 1.58 | 1.33 | Walked | Captures synchronized rotation |
| 3 | decoupled | SPX 500 | 1.49 | 1.28 | Walked | Reliable context filter |
| 4 | high volatility | USD JPY | 1.72 | 1.43 | Walked | Strong impulse capture |
| 5 | low volatility | USD JPY | 1.34 | 0.87 | Failed | Overfitted to calm regimes |
| 6 | trending | XAU USD | 1.63 | 0.92 | Failed | Static pattern misalignment |
| 7 | mean-reverting | XAU USD | 1.59 | 1.44 | Walked | Excellent equilibrium response |
| 8 | high volatility | USD JPY | 1.74 | 1.52 | Walked | Second-phase volatility continuation |
| 9 | decoupled | SPX 500 | 1.47 | 0.84 | Failed | Misidentified structural exhaustion |

### Conclusion

Throughout our examination of the 10 FrAMA-Stochastic boolean signal patterns, we were able to see how these basic indicator combinations, when encoded to a noise filtering boolean output, can give us a diverse behavioral array of across the market archetypes. Every pattern worked as an experimental proxy for determining if structured indicator logic could stay robust when forward walked past its optimization data window. The evidence, which we’ve summarized in our comparative table above, indicates that while seven of the signal patterns successfully, to a degree, transferred profitability; the others suffered from regime dependency, or volatility asymmetry.

The profitable group of 0,1,2,3,4,7, and 8 showed more robustness and adaptability. However, even though the test results presented here may seem ‘insightful’ they have been harnessed from a limited test window and very specific market conditions/situations. As always, readers need to exercise independent diligence prior to considering or relying on any of the outcomes we have put up here. Market dynamics evolve a lot, and what does well in one phase can do a U turn in another. The need to always validate any findings through independent testing, with extended data and more forward simulations prior to any live deployments, cannot be overstated.

For our follow-up, as per usual, we’ll move on from rule encoding to machine learning. Specifically, we’ll explore inference learning and surmise if indeed it can play a role in turning the fortunes around of our laggard patterns for this article.

### Disclaimer

This is not financial advice. The goal with these articles is not to purport any understanding of the markets, but invite discussion of ideas. The aim is not to point to a grail, but rather to probe the trough of opportunities available and hopefully to exhibit how they could be tapped into. The finishing work and important diligence is always on the part of the reader, should he choose to incorporate some of the material presented here.

| name | description |
| --- | --- |
| WZ\_83.mq5 | Wizard Assembled Expert Advisor whose header lists name and location of referenced files |
| SignalWZ\_83.mqh | Custom Signal Class file used by MQL5 Wizard in assembling Expert Advisor |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19890.zip "Download all attachments in the single ZIP archive")

[WZ\_83.mq5](https://www.mql5.com/en/articles/download/19890/WZ_83.mq5 "Download WZ_83.mq5")(8.5 KB)

[SignalWZ\_83.mqh](https://www.mql5.com/en/articles/download/19890/SignalWZ_83.mqh "Download SignalWZ_83.mqh")(50.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/497652)**
(1)


![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
16 Oct 2025 at 19:40

Hello Stephen,

A very interesting article, especially Pattern 0.

I downloaded your code from the prior article, compiled and ran it on four Major USD currencies, EUR, GBP, CAD, JPY for the H4 timeframe and for the period 1/1/25 - 10/1/25 and ALL produced a loss.

First of all, do I have to retrain it for the specific currency, and secondly, how do I specify using only one or several of the patterns?  Also, can this ea be used to incorporate several of your excellent articles?

Thanks,

CapeCoddah

![Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://c.mql5.com/2/175/19693-building-a-trading-system-final-logo.png)[Building a Trading System (Part 5): Managing Gains Through Structured Trade Exits](https://www.mql5.com/en/articles/19693)

For many traders, it's a familiar pain point: watching a trade come within a whisker of your profit target, only to reverse and hit your stop-loss. Or worse, seeing a trailing stop close you out at breakeven before the market surges toward your original target. This article focuses on using multiple entries at different Reward-to-Risk Ratios to systematically secure gains and reduce overall risk exposure.

![Biological neuron for forecasting financial time series](https://c.mql5.com/2/117/Biological_neuron_for_forecasting_financial_time_series___LOGO.png)[Biological neuron for forecasting financial time series](https://www.mql5.com/en/articles/16979)

We will build a biologically correct system of neurons for time series forecasting. The introduction of a plasma-like environment into the neural network architecture creates a kind of "collective intelligence," where each neuron influences the system's operation not only through direct connections, but also through long-range electromagnetic interactions. Let's see how the neural brain modeling system will perform in the market.

![Dialectic Search (DA)](https://c.mql5.com/2/115/Dialectic_Search____LOGO.png)[Dialectic Search (DA)](https://www.mql5.com/en/articles/16999)

The article introduces the dialectical algorithm (DA), a new global optimization method inspired by the philosophical concept of dialectics. The algorithm exploits a unique division of the population into speculative and practical thinkers. Testing shows impressive performance of up to 98% on low-dimensional problems and overall efficiency of 57.95%. The article explains these metrics and presents a detailed description of the algorithm and the results of experiments on different types of functions.

![Self Optimizing Expert Advisors in MQL5 (Part 15): Linear System Identification](https://c.mql5.com/2/175/19891-self-optimizing-expert-advisors-logo__1.png)[Self Optimizing Expert Advisors in MQL5 (Part 15): Linear System Identification](https://www.mql5.com/en/articles/19891)

Trading strategies may be challenging to improve because we often don’t fully understand what the strategy is doing wrong. In this discussion, we introduce linear system identification, a branch of control theory. Linear feedback systems can learn from data to identify a system’s errors and guide its behavior toward intended outcomes. While these methods may not provide fully interpretable explanations, they are far more valuable than having no control system at all. Let’s explore linear system identification and observe how it may help us as algorithmic traders to maintain control over our trading applications.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=qkhzotsamktlmhhjufneyyvfuwylhopq&ssn=1769093325743100437&ssn_dr=0&ssn_sr=0&fv_date=1769093325&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19890&back_ref=https%3A%2F%2Fwww.google.com%2F&title=MQL5%20Wizard%20Techniques%20you%20should%20know%20(Part%2084)%3A%20Using%20Patterns%20of%20Stochastic%20Oscillator%20and%20the%20FrAMA%20-%20Conclusion%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909332580476367&fz_uniq=5049397457445563084&sv=2552)

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