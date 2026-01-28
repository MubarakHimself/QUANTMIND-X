---
title: Markets Positioning Codex in MQL5 (Part 2):  Bitwise Learning, with Multi-Patterns for Nvidia
url: https://www.mql5.com/en/articles/20045
categories: Trading Systems, Indicators, Expert Advisors, Strategy Tester
relevance_score: 4
scraped_at: 2026-01-23T17:42:36.092324
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/20045&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068554398860245722)

MetaTrader 5 / Tester


### Introduction

From our last pilot [article](https://www.mql5.com/en/articles/20020) on Market-Positioning, we used Nvidia Corp’s stock (NVDA) as our asset in determining if one-position-type strategies could have traction when deployed live following extensive testing. Our test window was constrained due to stock split activity on Nvidia, and so we had to train over a narrow, 2-year band, before doing a 1-year forward walk test, with both periods being in the gap between NVDA’s two recent stock splits. These splits, though jarring when overlooked in Strategy Tester, are often handled by most brokers, including those offering MetaTrader, by closing traders’ positions on the high price and re-opening them with increased volume at the new price without any attrition on the position’s profits. We continue our study of NVDA by looking at patterns 5 to 9 for the indicator pairing RSI and DeMarker.

### Pattern-5: Slope Confluence with Range Expansion

The sixth signal that we have spots market moves that have a ‘high-conviction’ by merging slope-metrics of the RSI and DeMarker with the high-low price expansion. This usually signals an increase in trader activity as well as volatility. The buy condition is marked when the RSI slope across 3 bars is positive, and the DeMarker also indicates a similar state across its 3 bars. The price range of the current price bar, where current is taken to represent the most recent completed price bar that would be indexed 1; would also be greater than that of a previous bar at an optimized distance in the past. Once the current close price is within the top 25 percent of its range, this would signal buying pressure. The chart representation for this would be as follows:

_RSI slope(3) > 0, DeMarker slope(3) > 0, AND current bar range (high-low) > 1.2 \* average(range,10) (i.e., range expansion) — price must close in top 25% of bar._

[![p5](https://c.mql5.com/2/182/p5__1.png)](https://c.mql5.com/2/182/p5.png "https://c.mql5.com/2/182/p5.png")

As a side note, given our long only approach, the sell-requirements, are also achieved when the RSI and DeMarker slopes are negative over a three bar period and the current high-low range has increased more than a past bar-range. This past benchmark is tunable/optimizable, and we are taking its period to match that of our used averages in order to reduce the number of optimized parameters. Readers of course can adjust this, but the rationale here is to be wary of curve fitting. The current close price would also be in the bottom 25 percent of the current price bar range in order to confirm the presence of selling pressure. We implement this in MQL5 as follows:

```
//+------------------------------------------------------------------+
//| Check for Pattern 5.                                             |
//6) Slope Confluence with Volume/Range Expansion
//+------------------------------------------------------------------+
bool CSignalRSI_DeMarker_SL::IsPattern_5(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return(Hi(X() + m_past + 1) - Lo(X() + m_past + 1) < Hi(X() + 1) - Lo(X() + 1)  &&
             RSI(X()) - RSI(X() + 3) > 0.0 &&
             DeMarker(X()) - DeMarker(X() + 3) > 0.0);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return(Hi(X() + m_past + 1) - Lo(X() + m_past + 1) < Hi(X() + 1) - Lo(X() + 1)  &&
             RSI(X()) - RSI(X() + 3) < 0.0 &&
             DeMarker(X()) - DeMarker(X() + 3) < 0.0);
   }
   return(false);
}
```

Pattern-5 is built to be good at filtering for bars showing simultaneous momentum acceleration, as measured by the slopes, while strong moves in price from range expansion are also being logged. In requiring both the oscillators to concur on direction, we validate a volume sensitive range expansion. We thus capture conviction moves and not just spikes that could be transient. The forward test results for this pattern, that run from 2023.06.01 to 2024.06.01, on the 15-minute time frame give us the following report:

[![r5](https://c.mql5.com/2/181/r5__1.png)](https://c.mql5.com/2/181/r5.png "https://c.mql5.com/2/181/r5.png")

With our report above, Pattern-5, that is based on slope confluence with a volume expansion, from our above report, shows a tight but modest performance. Starting with a deposit of 10k, we are able to book a net profit of 2,251 USD. No losers are recorded, 42 positions were opened, all in the long direction as per our testing protocol where we intend to test for multiple signal-patterns concurrently. With an unbounded profit-factor, the expected payoff was 53.6 and the largest winner being 173.76. We were testing with fixed margin, not fixed lots, which means these numbers do not necessarily carry the same significance, but they are noteworthy markers nonetheless. Equity drawdown sat at a sub-one-percent value, with balance drawdown being none existent. The recovery factor was 2.5 on a Sharpe ratio of 6.32. These together with a perfect LR correlation of 1.00 imply the entry logic was lockstep with NVDA’s prevalent trend.

From the concept view point - RSI and DeMarker slope alignment together with range expansion continues to latch onto the bursts of the bullish direction following a compression, however in this forward walk it appears to be doing so selectively. Since we have no booked losses, the prevalent risk with pattern-5 probably lies in the intra-trade equity dips as well as the occurrence of future fake expansions. These were not encountered in our test window given that we were in a major bullish period between two major stock splits, therefore adding explicit volatility checks, firmer stop logic, or conservative position sizing attuned to volatility are all extra measures that could be adopted in enhancing this.

### Pattern-6: Leading Price and Lagging Indicators

Our seventh signal, pattern-6,places trades based on pullback entries by spotting cases where price tends to lead with a clear impulsive break and yet the RSI and DeMarker are lagging without indicating exhaustion. Chart representation of this can appear as follows:

_Price makes an impulsive leg up (higher high) then pulls back to the 21-EMA; RSI is still > 45 and DeMarker > 0.45 (no severe indicator fall) — enter when RSI slope turns positive._

[![p6](https://c.mql5.com/2/182/p6__1.png)](https://c.mql5.com/2/182/p6.png "https://c.mql5.com/2/182/p6.png")

The buy condition is triggered when price makes an impulsive move higher, marking a higher high, then pulls back towards a crucial support level, which for this article is a period optimized exponential moving average. This EMA signifies major dynamic support and if the RSI stays above 45 while the DeMarker is also north of 0.45 then this can be interpreted as no serious loss in momentum. Long entry should be triggered when the RSI turns positive, meaning momentum has resumed. We implement this in MQL5 as follows:

```
//+------------------------------------------------------------------+
//| Check for Pattern 6.                                             |
//7) Price Leading, Indicator Lagging (confirm pullback entries)
//+------------------------------------------------------------------+
bool CSignalRSI_DeMarker_SL::IsPattern_6(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return(High(X() + 2) < High(X() + 1) && High(X() + 1) > High(X()) && Close(X()) <= Cl(X()) &&
             RSI(X()) > 45.0 &&
             DeMarker(X()) > 0.45);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return(Low(X() + 2) > Low(X() + 1) && Low(X() + 1) < Low(X()) && Close(X()) >= Cl(X()) &&
             RSI(X()) < 55.0 &&
             DeMarker(X()) < 0.55);
   }
   return(false);
}
```

The sell condition, on the flip side, is when price makes an impulsive move lower by registering a new low, only to retrace upwards to a notional resistance level. This resistance is dynamic, and as with the buy condition, is an EMA of an optimizable period. The RSI should be below 55 while the DeMarker is also not above 0.55, a sign of weak but flickering bearish momentum. Entry can be made when the RSI slope turns negative, affirming that the bearish follow-through is ensuing. Forward testing of this signal, after training from 2021.08.01 to 2023.06.01 gave us the following report:

[![r6](https://c.mql5.com/2/181/r6__1.png)](https://c.mql5.com/2/181/r6.png "https://c.mql5.com/2/181/r6.png")

This signal-pattern that relies on leading price and lagging indicators nets us a profit of 2,532 off the starting deposit of 10k. A total of 26 positions are opened in this process, with all closing in the green. The expected payoff was 97 and the Sharpe was a respectable 5.32 and a perfect LR correlation of 1.0. More remarkable though was the maximum equity drawdown that was close to zero when testing with broker available ticks. We were able to monetize pull-back confirmation entries into orderly continuation runs.

As a concept, this pattern tends to lean on the notion that price momentum leads and indicators are meant to confirm that these moves are still in play and not exhausted. The largest individual winner was 195, also with no losses, a statement that continues our overriding theme of this is not invincibility but a gentle period in which we happened to forward walk the pattern. The 26 ‘winning streak’, if that even matters in this context, is impressive on paper but speaks nothing to performance in bearish or whipsaw environments. An additional stopping system seems non-negotiable, given that our testing is not even engaging trailing stops. These modifications can easily be made in the wizard-assembly dialog boxes. The need to think about dynamic exit rules as well as volatility aware stops, as presented by prebuilt custom modules in the MQL5 wizard, is certainly something to consider if one plans on surviving any rougher than a bullish NVDA regime.

### Pattern-7: Indicator Mismatch

The eighth signal, pattern-7, takes advantage of a special mismatch between the DeMarker oscillator and the RSI oscillator to grab early entry signals. We look out for M/W formations in the DeMarker while the relative position of the RSI to the neutral 50 level is also under scrutiny. We have our typical chart impression for this pattern as follows:

_DeMarker forms a W (two rising lows) while RSI is still below 50 but turning up (RSI slope(2)>0). Enter when price closes above the last minor swing high._

[![p7](https://c.mql5.com/2/182/p7__1.png)](https://c.mql5.com/2/182/p7.png "https://c.mql5.com/2/182/p7.png")

The buy condition is triggered when the DeMarker makes a W formation with the two bottom U troughs rising, or the latter being at or above the prior; and the RSI is south of the 50 level but with a positive slope across two price bars. Entry can be made when price closes above the last minor swing high, confirming a bullish thesis. We implement this as follows in MQL5:

```
//+------------------------------------------------------------------+
//| Check for Pattern 7.                                             |
//8) Indicator Mismatch (one leading, one confirming)
//+------------------------------------------------------------------+
bool CSignalRSI_DeMarker_SL::IsPattern_7(ENUM_POSITION_TYPE T)
{  vector _demarker;
   _demarker.Init(fmax(5, m_past));
   for(int i = 0; i < fmax(5, m_past); i++)
   {  _demarker[i] = DeMarker(i);
   }
   if(T == POSITION_TYPE_BUY)
   {  return(IsW(_demarker) &&
             RSI(X()) - RSI(X() + 2) > 0.0 &&
             Close(X()) >= High(X() + 1));
   }
   else if(T == POSITION_TYPE_SELL)
   {  return(IsM(_demarker) &&
             RSI(X()) - RSI(X() + 2) < 0.0 &&
             Close(X()) <= Low(X() + 1));
   }
   return(false);
}
```

The sell condition is registered when the DeMarker this time makes an M formation, also marked by two declining lowercase Ns, with the latter being at or below the prior, and RSI being above the 50 but with a negative slope. Entry can be made once price breaks below the last minor swing-low. Our source above spots the local extrema of the DeMarker in a look abc window that is always at least 5, and also verifies the oscillator slopes before looking for corresponding price action validation. A forward test of pattern-7 presented us with the following report:

[![r7](https://c.mql5.com/2/181/r7__1.png)](https://c.mql5.com/2/181/r7.png "https://c.mql5.com/2/181/r7.png")

The eighth signal pattern returns a clean profit profile by raising the 10k initial deposit by 2,570 after 52 trades were placed. Once again, and perhaps unsurprisingly, we are able to achieve a 100 percent win rate by using this indicator mismatch idea. The Sharpe ratio came in at 8.17, above the other two patterns we have looked at above, and the LR correlation is almost perfect at 0.99. So our entries are able to properly track Nvidia’s price action to avoid unnecessary adverse excursions. The worst equity drawdown was 9 percent, which, though large, could be manageable and points to a more stable equity line progression whenever a breakout continuation market regime is in play. Pattern-7 is depending on RSI and DeMarker affirmations and is able to make this work by having synchronized timing to capture consistently the market continuations while avoiding the choppy mean reversion noise.

As a concept, this pattern is trying to use DeMarker as the forbearing signal to strength or weaknesses in the prevalent market trend, while the RSI plays the role of a slower momentum filter. This partnership with our testing is playing out as a controlled hand off as opposed to a tug of war. Our largest winning trade was over 600 with the average profit being almost. Without any losses incurred, our outsize winners did not patch up any equity curve dents, or we had no recovery-factor to write about. We clocked 52 wins, in a somewhat volatile but mostly bullish setting that implies as with the others that we need some re-runs outside this window for us to draw better conclusions. While this lead-lag interplay is certainly promising, further work on the pattern by not just testing outside-of its implied comfort zone seems appropriate. When this is done, the afore mentioned volatility filters and stop loss management mechanisms proposed for the previous patterns, could be considered as supplements.

### Pattern-8: Sequential Slope Ladder

Our penultimate pattern, engages a multi-period slope check as a strategy in order to increase predictive reliability by stacking momentum filters in the short, medium and longer horizons in measuring the RSI slope. This is paired with the current DeMarker slope as a pertinent confirmation of a given trend. A chart representation is as follows for the buy signal:

_RSI slope(1)>0, slope(3)>0, slope(5)>0 (monotone positive slopes across short→long) AND DeMarker slope(1)>0._

[![p8](https://c.mql5.com/2/182/p8__1.png)](https://c.mql5.com/2/182/p8.png "https://c.mql5.com/2/182/p8.png")

With this logic, the buy condition is logged when the RSI across one bar, three bars and five bars is positive while the DeMarker over one is also in the plus. This is meant to portray a monotonically rising slope ladder over consecutive, expanding time horizons. It is supposed to point to upward momentum. We implement this in MQL5 as follows:

```
//+------------------------------------------------------------------+
//| Check for Pattern 8.                                             |
//9) Sequential Slope Ladder (multi-period confirmation)
//+------------------------------------------------------------------+
bool CSignalRSI_DeMarker_SL::IsPattern_8(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return(RSI(X()) - RSI(X() + 1) > 0.0 &&
             RSI(X()) - RSI(X() + 3) > 0.0 &&
             RSI(X()) - RSI(X() + 5) > 0.0 &&
             DeMarker(X()) - DeMarker(X() + 1) > 0.0);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return(RSI(X()) - RSI(X() + 1) < 0.0 &&
             RSI(X()) - RSI(X() + 3) < 0.0 &&
             RSI(X()) - RSI(X() + 5) < 0.0 &&
             DeMarker(X()) - DeMarker(X() + 1) < 0.0);
   }
   return(false);
}
```

The flip side sell conditions see the RSI also registering a negative slope across the one, three and five bar periods with the DeMarker oscillator also being currently in negative slope territory. This signal is the first one that we consider where price action is not considered at all, which by itself ought to be dangerous. A forward test run, from 2023.06.01 to 2024.06.01 following training from 2021.08.01 to 2023.06.01 gave us the following report:

[![r8](https://c.mql5.com/2/181/r8__1.png)](https://c.mql5.com/2/181/r8.png "https://c.mql5.com/2/181/r8.png")

Pattern-8 was a sequential, slope ladder strategy meant to get momentum from stacked RSI and DeMarker slopes put up another clean performance, while executing long only positions, amidst NVDA’s bullish run. Netting, 2094 on the 10k deposit pattern-8 yields a gain of 54.6 per trade. We opened 53 positions where once more all did close in the green, meaning we have an unbound or NaN profit factor, with the indicated zero simply being a placeholder. Indicated equity drawdown came in at 8.8 percent with an absolute amount that's just over USD 900. Based on a starting capital of 10k, these figures sit comfortably within our risk zone while still accommodating room for intra trade noise without forced exits.

Further analysis of pattern-8’s trades shows that purely momentum based entries are not only accurate but probably relentless, given the 100 pct win rate from 53 entries. Nonetheless, this performance risks lulling complacency. As already mentioned, with our other patterns we do not have a proper sense of how bad the losers will be, or whether a few poorly timed reversals could wipe out a chunk of accumulated gains. The mean profit of 49-55 also suggests that pattern-8’s edge is based on repetition, as opposed to home-run or recovery trades.

With this, the same remedies we recommended with patterns 5 to 7 above do still apply. Explicit stops are necessary, and so are volatility filters. Price action confirmation can also see to it that when NVDA’s market regime stops rewarding this stop-ladder logic, the inevitable drawdowns do not nuke our equity curve, and we remain to trade another day.

### Pattern-9: False Break Detection

The last signal pattern for going long on NVDA that uses the RSI and DeMarker oscillators, seeks to spot false breakouts as well as traps by merging price action failure with ‘non-confirmations’ of the RSI and DeMarker. This on paper is meant to signal asymmetric setups for entry that provide a favorable risk-reward dynamic. On a chart, this may appear as follows:

_Price briefly breaks below a support (close below) but within 3 bars reclaims support (close back above) AND RSI did not confirm (RSI did not make new low) while DeMarker showed rising slope — interpret as bear trap, enter long on reclaim._

[![p9](https://c.mql5.com/2/182/p9__1.png)](https://c.mql5.com/2/182/p9.png "https://c.mql5.com/2/182/p9.png")

Going long requires price to break below a key level of support by closing below it. Following this, if within 3 price bars, price is able to reclaim this drop by subsequently closing above this support while the RSI did not back the price drop as a sign of weakening momentum and the DeMarker also indicated a rising slope; this would reinforce evidence of buying pressure. Buys can be entered when price closes and re-opens above this support level. We code this in MQL5 as follows:

```
//+------------------------------------------------------------------+
//| Check for Pattern 9.                                             |
//10) False Break / Trap Detection (price action deviation)
//+------------------------------------------------------------------+
bool CSignalRSI_DeMarker_SL::IsPattern_9(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY)
   {  return(Close(X() + 2) < Cl(X() + 2) && Close(X() + 1) < Cl(X() + 1) && Close(X()) >= Cl(X()) &&
             RSI(X()) - RSI(X() + 1) > 0.0 &&
             DeMarker(X()) - DeMarker(X() + 1) > 0.0);
   }
   else if(T == POSITION_TYPE_SELL)
   {  return(Close(X() + 2) > Cl(X() + 2) && Close(X() + 1) > Cl(X() + 1) && Close(X()) <= Cl(X()) &&
             RSI(X()) - RSI(X() + 1) < 0.0 &&
             DeMarker(X()) - DeMarker(X() + 1) < 0.0);
   }
   return(false);
}
```

The bear trap mirrors what we have above, with price breaking above a resistance level, even closing above it before making a retreat within 3 price bars or less to close below it. The RSI would similarly fail to confirm the breakout by not making a new high, while the DeMarker would also be sloping downward. Shorts are made when price closes and opens below the resistance. Forward testing for pattern-9 gives us the following results from Strategy Tester:

[![r9](https://c.mql5.com/2/181/r9__1.png)](https://c.mql5.com/2/181/r9.png "https://c.mql5.com/2/181/r9.png")

Our last signal pattern that is using the false trap signal in its refined form exhibited a more controlled performance by improving the starting 10k balance by just 1300 over 37 trades. With all positions intentionally long, as argued in the introduction, we once again obtain all as winners. Given no losses, the printed profit factor of 0 above is nominal for an unbound value. The expected payoff was thus modest, given the small profits we obtained relative to the other signal patterns, however the Sharpe ratio of 14.19 indicates that this pattern’s edge is both efficient and resilient. Following this, the equity drawdown was almost minuscule by NVDA’s volatility standards, at 2.6 percent or USD 277 in absolute terms, which is a far cry from the holes of almost 40 to 50 percent we’ve seen printed in the reports of other signal patterns above and in the last article. One could thus say that this sits comfortably with what any traders would deem ‘survivable’, even at the used 1:100 leverage.

We have ended up with a signal pattern that is not only right on direction for the most of the time, but is also more cautious in how it participates in a trend. While 37 lossless wins are suspiciously perfect, the moderate expected profit of USD 36, and the largest win being a modest USD 85 ensure our equity curve is not hinting at lottery expectations. It is a grind, not glory. The question on leverage to use, though, is relevant since a few major jurisdictions would force this amount to be reduced, which would in turn reduce returns while providing some consolation to the risk profile. Even in these bullish regimes that we are testing Nvidia, it is not a gentle ride, as tempered excursions both adverse and favorable were endured. Live deployment would clearly one to watch their position sizes and not assume that our testing years are the only plausible outlook, in addition to adopting some of the measures suggested in the prior patterns above.

### Multiple-Pattern with up to 10 selections

We are implementing only long conditions, as an exploration if this could fare better in forward walks than our previous attempts when doing multi-pattern testing where we took on both long and short positions. Our conditions, of course, are hugely skewed given the bullish phase of NVDA that we are studying, that also had to be restricted further given the stock splits the company made. Our MQL5 code for the long and short conditions is as follows;

```
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalRSI_DeMarker::LongCondition(void)
{  PatternsUsed(GetDualMap(m_dual_map));
   int result  = 0, results = 0;
//--- if the model 0 is used
   if(((m_patterns_usage & 0x01) != 0) && IsPattern_0(POSITION_TYPE_BUY))
   {  result += m_pattern_0;
      results++;
   }
//--- if the model 1 is used
   if(((m_patterns_usage & 0x02) != 0) && IsPattern_1(POSITION_TYPE_BUY))
   {  result += m_pattern_1;
      results++;
   }
//--- if the model 2 is used
   if(((m_patterns_usage & 0x04) != 0) && IsPattern_2(POSITION_TYPE_BUY))
   {  result += m_pattern_2;
      results++;
   }
//--- if the model 3 is used
   if(((m_patterns_usage & 0x08) != 0) && IsPattern_3(POSITION_TYPE_BUY))
   {  result += m_pattern_3;
      results++;
   }
//--- if the model 4 is used
   if(((m_patterns_usage & 0x10) != 0) && IsPattern_4(POSITION_TYPE_BUY))
   {  result += m_pattern_4;
      results++;
   }
//--- if the model 5 is used
   if(((m_patterns_usage & 0x20) != 0) && IsPattern_5(POSITION_TYPE_BUY))
   {  result += m_pattern_5;
      results++;
   }
//--- if the model 6 is used
   if(((m_patterns_usage & 0x40) != 0) && IsPattern_6(POSITION_TYPE_BUY))
   {  result += m_pattern_6;
      results++;
   }
//--- if the model 7 is used
   if(((m_patterns_usage & 0x80) != 0) && IsPattern_7(POSITION_TYPE_BUY))
   {  result += m_pattern_7;
      results++;
   }
//--- if the model 8 is used
   if(((m_patterns_usage & 0x100) != 0) && IsPattern_8(POSITION_TYPE_BUY))
   {  result += m_pattern_8;
      results++;
   }
//--- if the model 9 is used
   if(((m_patterns_usage & 0x200) != 0) && IsPattern_9(POSITION_TYPE_BUY))
   {  result += m_pattern_9;
      results++;
   }
//--- return the result
//if(result > 0)printf(__FUNCSIG__+" result is: %i",result);
   if(results > 0 && result > 0)
   {  return(int(round(result / results)));
   }
   return(0);
}
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalRSI_DeMarker::ShortCondition(void)
{  return(0);
}
```

Our code, as has been the case in previous articles, is in essence a weighted voting system that establishes buy-side conviction of a trading system that is assembled via the [MQL5 Wizard](https://www.mql5.com/en/articles/171). Its operation concept is that each of the ten signal patterns; that, in our case, is derived from the combination of the indicators RSI and DeMarker as covered in the preceding sections; can independently appraise for bullish conditions in price. The ten patterns that we typically represent as IsPattern-0() through IsPattern9() usually evaluate different indicator features that could include divergences, synchronized momentum shifts, or structural M/W formations, as we saw with the RSI-DeMarker patterns above. The function LongCondition() therefore brings together all these pattern checks into one common signal that outputs a weighted mean score of the collective vote of all patterns chosen, towards a bullish outcome.

The choice of pattern is controlled by the integer variable m\_patterns\_usage that acts as a bit-mask. Every ‘bit’ in it (when converted to base-2 or 0s and 1s) gets to correspond to one of the ten exclusive signal-patterns. For instance, assigning this variable the integer value 1, activates only pattern 0, while patterns 2,4,8,16,…,512 activate patterns indexed 1 to 9 respectively. This is when implementing exclusive use. When combining patterns, though, this value would be assigned different integer values, typically in the range 0 to 1023. We stop at 1023 because we have 10 patterns, and 1024 is 2 to the power 10.

The PatternsUsed() function assigns the value of this m\_patterns\_usage variable from the input value, while the bit-mask interpretation happens in the LongCondition() function. We optimize/train for a suitable combination of patterns, 0 to 0 for the RSI and DeMarker over 2021.08.01 to 2023.06.01 as with the single patterns and then perform a forward test on the period 2023.06.01 to 2024.06.01. Our results from this are given below:

[![r-all](https://c.mql5.com/2/181/r-all__1.png)](https://c.mql5.com/2/181/r-all.png "https://c.mql5.com/2/181/r-all.png")

Our test report above was got from running the Expert Advisor with the input parameter for patterns used assigned 605. This corresponds to the use of the signal patterns 0,2,3,4,6 and 9. This forward walk seems robust. From the initial 10k deposit, we are able to net USD 2,309. Strictly speaking this is a bit underwhelming, given the 10 signal patterns that were at disposal, especially considering we were able to achieve higher returns with just one signal pattern as we already covered in this article and the last. 40 trades were placed, again all in the black, with a recovery factor of 2.95, and Sharpe ratio of 6.99 our returns are healthy relative to volatility.

Risk also appeared subdued, with the maximum equity drawdown being 7.61 percent, amounting to USD 703. Our modelling quality was also decent at 99 percent.

In general, this forward walk amounts to a proof of concept that has held together. Starting off the article, we focused on opening only long positions, with the final intent of testing multiple patterns that all open positions in a specific direction. Previously in other articles I have written about the folly of using more than one pattern in an Expert Advisor as signals tend to cancel each other out when forward walking such that the good results obtained when training/ optimizing are unable to replicate since the training results were amounting to a curve fit of longs cancelling shorts and vice versa.

40 trades, all wins, no balance drawdown to talk about, We seem to have made the case for opening one-sided positions only if one wants to trade with multiple signal patterns. However, the over arching goal of using multiple patterns was to have outsized yields, certainly above the single pattern setups, and hypothetically a sum-is-better-than-the-parts' situation. We have clearly fallen short of the former, which is the very basic goal. As things stand, absent of testing in wider windows and implementing stop management and the suggested volatility filters one can still argue that trading with just the one signal pattern that does well, north of 30 percent as we have already seen with some signal patterns, is a more prudent strategy.

### Conclusion

From the testing of our 10 signal patterns on Nvidia Corp, starting with the last article and now concluding with this one, a summarizing common theme could be something that most traders suspect, but never reveal outright per se, markets reward precision but punish overconfidence. Even though plenty of our signal pattern models did post decent wins, a chronic problem was the poor profit-to-loss asymmetry. For instance, pattern-5’s range expansion as well as slope confluence showed enviable momentum capture, but this came with a lot of drawdowns, something that can make many traders second guess their trades. Pattern-6 also, where we used leading price and lagging indicators, seemed to be elegant and efficient with some measured drawdown and yet still we needed to implement a form of stricter stop placement in order to survive extended volatility runs.

In contrast to these initial signal patterns, pattern-8 where we relied on a ‘sequential-slope-ladder’ revealed the dangers of separating signal logic from price structure. Its momentum only methodology collapsed in the forward test given its directional biases, highlighting that even with good win ratios, unbounded losses can wipe out hard-earned gains.

The true highlight though, it seems, is the multi-pattern configuration. This was our last testing, just above, and in it, we allowed the selection of up to 10 patterns to vote through a weighted mask. The winning bit-mask map from training/optimization was 883 or the patterns 0,1,4,6,9 and these yielded a profit factor of 3.0, a Sharpe ratio greater than 20 all with the largest equity drawdown restricted to less than 19 percent. These benchmarks were met while multiplying the starting deposit by 2.5x, again, a very aggressive performance. We used the high leverage of 1 to 100, so using 1 to 50 could in theory still give us a 75 percent return on the year. The broader takeaway here could be that pattern diversity compensates for pattern specific weaknesses. Another way of putting could be that the ensemble, of multi-pattern selection, behaved like a machine-learning model where the blending of confirmation, reversals, and range expansions signals was fomented into a single coherent decision layer.

With his said, there are still no free lunches in trading. While these forward results do look promising, our test window was constrained by NVDA’s stock split dates and these bordered an exceptional bullish period for the stock. Yes, we sustained some losses despite going long only with stops, but the bullish backdrop for this testing was present. How will these fare in a bear market? There are always lurking dangers in taking curve alignment for model robustness. These patterns, all including the multi-pattern, should take on some revalidation in different volatility regimes. Also, this out-of-regime testing should account for some execution latency, changes in spreads, and sensitivity to leverage seeing as we stuck with 1 to 100, yet some traders can not use anything above 1 to 50.

| name | description |
| --- | --- |
| EMC-1.mq5 | Wizard Assembled Expert Advisor with references in header |
| SignalEMC-1.mq5 | Signal Class file of Expert Advisor used with [MQL5 Wizard](https://www.mql5.com/en/articles/171) |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20045.zip "Download all attachments in the single ZIP archive")

[EMC-1.mq5](https://www.mql5.com/en/articles/download/20045/EMC-1.mq5 "Download EMC-1.mq5")(8.42 KB)

[SignalEMC-1.mqh](https://www.mql5.com/en/articles/download/20045/SignalEMC-1.mqh "Download SignalEMC-1.mqh")(50.56 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**[Go to discussion](https://www.mql5.com/en/forum/500428)**

![Building AI-Powered Trading Systems in MQL5 (Part 6): Introducing Chat Deletion and Search Functionality](https://c.mql5.com/2/181/20254-building-ai-powered-trading-logo.png)[Building AI-Powered Trading Systems in MQL5 (Part 6): Introducing Chat Deletion and Search Functionality](https://www.mql5.com/en/articles/20254)

In Part 6 of our MQL5 AI trading system series, we advance the ChatGPT-integrated Expert Advisor by introducing chat deletion functionality through interactive delete buttons in the sidebar, small/large history popups, and a new search popup, allowing traders to manage and organize persistent conversations efficiently while maintaining encrypted storage and AI-driven signals from chart data.

![Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://c.mql5.com/2/181/20065-developing-trading-strategy-logo.png)[Developing Trading Strategy: Pseudo Pearson Correlation Approach](https://www.mql5.com/en/articles/20065)

Generating new indicators from existing ones offers a powerful way to enhance trading analysis. By defining a mathematical function that integrates the outputs of existing indicators, traders can create hybrid indicators that consolidate multiple signals into a single, efficient tool. This article introduces a new indicator built from three oscillators using a modified version of the Pearson correlation function, which we call the Pseudo Pearson Correlation (PPC). The PPC indicator aims to quantify the dynamic relationship between oscillators and apply it within a practical trading strategy.

![Risk Management (Part 2): Implementing Lot Calculation in a Graphical Interface](https://c.mql5.com/2/115/Gesti8n_de_Riesgo_Parte_1_LOGO.png)[Risk Management (Part 2): Implementing Lot Calculation in a Graphical Interface](https://www.mql5.com/en/articles/16985)

In this article, we will look at how to improve and more effectively apply the concepts presented in the previous article using the powerful MQL5 graphical control libraries. We'll go step by step through the process of creating a fully functional GUI. I'll be explaining the ideas behind it, as well as the purpose and operation of each method used. Additionally, at the end of the article, we will test the panel we created to ensure it functions correctly and meets its stated goals.

![From Novice to Expert: Predictive Price Pathways](https://c.mql5.com/2/182/20160-from-novice-to-expert-predictive-logo.png)[From Novice to Expert: Predictive Price Pathways](https://www.mql5.com/en/articles/20160)

Fibonacci levels provide a practical framework that markets often respect, highlighting price zones where reactions are more likely. In this article, we build an expert advisor that applies Fibonacci retracement logic to anticipate likely future moves and trade retracements with pending orders. Explore the full workflow—from swing detection to level plotting, risk controls, and execution.

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/20045&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068554398860245722)

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