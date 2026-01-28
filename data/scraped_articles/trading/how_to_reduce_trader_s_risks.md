---
title: How to reduce trader's risks
url: https://www.mql5.com/en/articles/4233
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:36:07.654593
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=vwidyijylvuuvbhpooxqidrbnnwymczy&ssn=1769250965076592695&ssn_dr=0&ssn_sr=0&fv_date=1769250965&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F4233&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20reduce%20trader%27s%20risks%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925096577134222&fz_uniq=5082982014346334868&sv=2552)

MetaTrader 5 / Trading


### Contents

- [What are financial markets in terms of process dynamics?](https://www.mql5.com/en/articles/4233#n1)
- [What is the probability of making a profit in financial markets?](https://www.mql5.com/en/articles/4233#n2)
- [Market risks associated with price movement dynamics](https://www.mql5.com/en/articles/4233#n3)

  - [Classifying risks associated with market dynamics](https://www.mql5.com/en/articles/4233#n4)
  - [Risks associated with high volatility at the time of a market entry](https://www.mql5.com/en/articles/4233#n5)
  - [Risks associated with the presence of resistance levels at the time of a market entry](https://www.mql5.com/en/articles/4233#n6)
  - [Risks of ending up in an overbought/oversold zone at the time of a market entry](https://www.mql5.com/en/articles/4233#n7)
  - [Risks associated with the absence of a clear trend at the time of a market entry](https://www.mql5.com/en/articles/4233#n8)
  - [Risks associated with the incorrect selection of the indicator calculation period](https://www.mql5.com/en/articles/4233#n9)
  - [Risks associated with the use of pending orders at the time of a market entry](https://www.mql5.com/en/articles/4233#n10)
  - [Risks associated with uncertainty of the price movement amplitude after a market entry](https://www.mql5.com/en/articles/4233#n11)
  - [Risks associated with a price collapse after a market entry](https://www.mql5.com/en/articles/4233#n12)
  - [Risks associated with using only one price movement scale during analysis](https://www.mql5.com/en/articles/4233#n13)
  - [Risks associated with the use of only technical or only fundamental analysis](https://www.mql5.com/en/articles/4233#n14)

- [Risks not associated with market dynamics](https://www.mql5.com/en/articles/4233#n15)

  - [Classifying risks not associated with market dynamics](https://www.mql5.com/en/articles/4233#n16)
  - [Risks associated with the trading system structure](https://www.mql5.com/en/articles/4233#n17)
  - [Risks associated with exceeding a loss limit by deposit (managing investor's risk limit by deposit)](https://www.mql5.com/en/articles/4233#n18)
  - [Risks associated with negative changes of trading conditions](https://www.mql5.com/en/articles/4233#n19)
  - [Risks determined by the quality of connection to a broker's trade server](https://www.mql5.com/en/articles/4233#n20)
  - [Risks associated with the permission for automated trading on the broker's and client's sides](https://www.mql5.com/en/articles/4233#n21)
  - [Risks associated with changes in the financial markets legislation](https://www.mql5.com/en/articles/4233#n22)

- [A simple EA featuring modules for reducing some of the mentioned risks](https://www.mql5.com/en/articles/4233#n23)
- [Conclusion](https://www.mql5.com/en/articles/4233#n24)

### Introduction

First of all, this article will be of help for novice traders and analysts developing their own trading strategies. However, experienced market participants may also find something useful, for example classification of risk types, applying a candle analysis to define overbought/oversold areas, relationship between fundamental and technical analysis, selecting moving average calculation periods, as well as reducing risks associated with a possible price collapse.

The article deals with the following issues:

- the essence of the trading process on financial and stock markets in terms of dynamics;
- profit probability in trading;
- ways to reduce trader's risks when developing a trading system.

I do not promise a fully comprehensive analysis and complete classification of risks when trading in financial markets. Instead, we are going to focus on the main market risks associated with financial instruments' price movement dynamics. Besides, we will shed some light on the risks that are not directly associated with market dynamics but are still important for trading efficiency. When writing the article, I used my experience gained during the analysis and development of trading systems.

The EA's MQL5 version can be found **[here](https://www.mql5.com/en/code/19726)**.

### What are financial markets in terms of process dynamics?

When developing trading strategies (both for manual and automated trading), we should understand what process we are dealing with. Price movement on financial markets is a non-stationary process. It is influenced by multiple factors, and it is often impossible to determine the defining one.

The non-stationary character of the process is determined in the behavior of the market participants. Their reactions are multi-directional. As a result, the change in amplitude and frequency in the market cannot be determined by the deterministic behavior laws. Generally, such process can be considered random.

Although, it still has areas where you can forecast the movement correctly. Let's list the factors that make these areas appear.

- Wave-like price movement. We are able to define the start of the wave: for example, by the candle crossing the group of moving averages or by signals from conventional indicators. Let's not talk about the accuracy of such a definition for now. Here we only note that this is possible.

- Narrow consolidation zones (the price always leaves them).

- Some rules for responding to certain basic, fundamental and technical factors.

### What is the probability of making a profit in financial markets?

This is the main question, since any trader is in fact an investor. The probability of making a profit when trading in financial markets is directly related to the correctness of the trend continuation forecast, since the main movement range is located precisely in trend areas.

There are many methods for calculating trend continuation or reversal. They all have a common drawback since any trading system is a market status model. Evaluating the accuracy of such a model is always problematic. First, the process itself is rather complicated in terms of dynamics due to its non-stationary (random) character. Second, all accuracy evaluation methods "inside" a random process are also complex and their efficiency is ambiguous.

Therefore, in order to assess the possibility of making a profit, we will go the other way and use actual (rather than calculated) data. To do this, we need to analyze statistics on Forex and stock market trading efficiency provided by investment and brokerage companies.

According to the data provided in the article " [Effectiveness of private Forex trading in Russia and the US](https://www.mql5.com/go?link=http://journals.tsu.ru/puf/&journal_page=archive&id=1323&article_id=24308 "/go?link=http://journals.tsu.ru/puf/&amp;journal_page=archive&amp;id=1323&amp;article_id=24308")" (in Russian), the share of profitable private traders in Russia comprised 28%, while in the US it was 33%, as of 2015. Of course, these data cannot be considered exhaustive, since a limited number of companies were included in the study, and the study period was only six months. Anyway, these figures provide us with some insight. What is the possible conclusion?

Conventional methods of analysis used by most market participants do not allow us to effectively predict the process of price movement in financial markets. Additional measures, including money management methods, are also inefficient. The reason lies in the initial inefficiency of the basic forecast of the market dynamics performed using technical and fundamental analysis. As a result, no more than a third of financial market participants receive profit.

As for the stock market, the statistics is not uniform. I will cite relatively "moderate" data. According to the research of the French agency for financial markets, eight out of ten individual traders eventually lose their money. This means, the efficiency of trading on stock market is 20%. This is even worse than in the Forex market, although the value is comparable.

As we can see, a comprehensive accounting of risks is of utmost importance for any investor.

Let's highlight two main groups of risks:

- risks associated with the price movement dynamics;
- risks not associated with market dynamics.

We will have a closer look at these risks and the ways to reduce them.

We will start with the first group of risks. Their list is determined based on the analysis of financial instrument charts. We need to analyze the elements of a chart and their parameters, including the characteristics of both individual candles and their groups, as well as moving averages. We will not use other conventional indicators, since their mathematical models do not correspond to the price movement concept making them inefficient.

### Market risks associated with price movement dynamics

Technically, the main trader's risk is a trend reversal one. A financial instrument price may change direction and start moving opposite to the open position. If a position has not been opened yet, the risk lies in forecasting such a reversal.

Every trader has their own definition of a trend. Strange as it may seem, there is no "official" definition for this concept in technical analysis meaning that evaluation of a trend reversal probability is subjective as well. The critical amplitude (during a trend reversal) defined by every trader depends on the acceptable risk (risk limit). In turn, this value is defined by the amount of funds on each trader's account. In other words, everything depends on the maximum possible drawdown. What is good for a bank is bad for an ordinary trader.

Let's have a look at risk types and search for solutions to reduce them.

#### Classifying risks associated with market dynamics

Traditional analysis highlights three types of trend by their duration: short, medium and long term ones.

Local and global trends are often mentioned as well. Traditionally, local trends are the ones formed in the short or medium term. Global trends are formed in the long term. There are no clearly defined values that determine the time and amplitude boundaries of these concepts. Everything is relative. Therefore, let's simplify the analysis: the reverse movement amplitude is to be used as a criterion of the risk associated with the trend reversal. We will evaluate it with respect to important technical levels or depending on the size of the investor's deposit, taking the given risk limit into account.

To classify the risks associated with the possibility of a trend reversal, it is also necessary to take into account the factors of the market prices dynamics. Let's examine them in brief.

- Risks associated with high volatility at the time of a market entry.
- Risks associated with the presence of resistance levels at the time of a market entry.
- Risks of ending up in an overbought/oversold zone at the time of a market entry.
- Risks associated with the absence of a clear trend at the time of a market entry.
- Risks associated with the incorrect selection of the indicator calculation period.
- Risks associated with the use of pending orders at the time of a market entry.
- Risks associated with uncertainty of the price movement amplitude after a market entry.
- Risks associated with a price collapse after a market entry.
- Risks associated with using only one timeframe.
- Risks associated with the use of only one type of analysis (technical or fundamental).


#### Risks associated with high volatility at the time of a market entry

Many traders do not take this kind of risk into account at all. This is a big mistake, because the movement amplitude (including against the main trend) rises during a high volatility. This is the risk factor.

Evaluation of volatility using traditional indicators is too subjective. First, the indicators' settings themselves are subjective, and second, their mathematical models are not always designed for such an assessment. Therefore, it is more reliable to use the consolidation area as a model defining the necessary degree of volatility when entering the market.

To reduce this risk, we can set the following condition in the entry algorithm: threshold amplitude limit in the nearest quote history. Such a price range can be set:

- amplitude between two opposite fractals
- or amplitude of a candle group.

Limiting candle amplitudes is the simplest way. Let's have a look at the appropriate code fragment:

```
 //---MARKET ENTRY ALGORITHM - BUY-----------------------------------------------------------------------------------------

  if(
   //----REDUCE RISKS RELATED TO HIGH VOLATILITY AT THE MOMENT OF A MARKET ENTRY ----

     //Simulate the absence of high volatility in recent history:
     ( High[1] - Low[1]) <= 200*Point &&                       //limit amplitude of the lower timeframe candles (tfМ1)
     ( High[2] - Low[2]) <= 200*Point &&
     ( High[3] - Low[3]) <= 200*Point &&
     (H_prev_m15 - L_prev_m15) <= 300*Point &&                 //limit amplitude of the higher timeframe candles (tfМ15)
     (H_2p_m15 - L_2p_m15) <= 300*Point &&
     (H_3p_m15 - L_3p_m15) <= 300*Point &&
     (H_prev_m15 - L_3p_m15) <= 300*Point &&                   //limit the channel amplitude of the higher timeframe candles (tfМ15)
     (High[1] - Low[1]) >= (1.1*(High[2] - Low[2])) &&         //limit activity on the previous bar relative to the 2 nd bar in quotes history
     (High[1] - Low[1]) < (3.0*(High[2] - Low[2])) &&          //same
```

Fig. 1. Module in the entry algorithm: Reducing risks associated with high volatility at the time of a market entry

Fig. 1 shows the module in the entry algorithm (Buy).

_Notation_:

- High{1\], High\[2\], High\[3\] — High levels on three previous bars at М1,
- H\_prev\_m15, H\_2p\_m15, H\_3p\_m15 — High levels on three previous bars at М15,
- Low\[1\], Low\[2\], Low\[3\] — Low levels on three previous bars at М1,
- L\_prev\_m15, L\_2p\_m15, L\_3p\_m15 — Low levels on three previous bars at М15.

The module allows you to reduce the risks arising if the entry point is in the considerable volatility area. This is achieved through modeling a horizontal price channel, as well as modeling activity (moderate volatility) at the end of the channel.

Price channels on М1 and М15 timeframes are simulated by limiting the amplitude of candles on the previous three bars in quote history. The amplitude limitations are set both for separate candles and the entire candle group. The channel amplitude is defined as a difference between extreme values of its first and last candles.

Moderate volatility at the end of the channel is simulated by limiting the ratio of the amplitudes of two adjacent candles of M1 timeframe (in this case, the applied ratio is 1.1 to 3.0). Of course, you can use other amplitude ratios.

The question is why we should set the amplitude of the entire channel. Isn't it enough to set the amplitude of each candle? No, it isn't. If we do not define the amplitude of the entire channel, a wave-like (rather than a horizontal) channel may form, and its amplitude may be twice the size of a single candle's one.

This example is for Buy. For Sell, the code remains the same since the channel and separate candles' amplitude is defined as the difference between High and Low, while the direction is set in other entry algorithm modules rather than the current one.

#### Risks associated with the presence of resistance levels at the time of a market entry

Market participants react to resistance levels present on the chart differently. For some, this is the profit goal, for others - the loss limiters, and for the third - the initial goal for the level breakdown. This is why, fractal levels are often surrounded by areas where the price moves differently within a small amplitude. Position opening risks in such zones are increased, while, according to various trade systems test results, a forecast within them is inefficient. Therefore, it seems reasonable when a market entry point is outside resistance levels in the direction of a trend.

How can we take into account this trading risk?

If position is already opened in the direction of a resistance level and the price is approaching it, then it is better to close this position in advance due to a high probability of a strong roll-back interfering with the result.

If you are going to open a position according to your algorithm after the price passes the resistance level, then wait for a reliable breakthrough in the selected direction.

In fact, this is an issue of a false and true breakdowns. One of its possible solutions is as follows. Wait till a new resistance level is formed after a fractal level. Then wait till the price breaks through it. Make sure to use one of the lower timeframes. However, this is not enough as well. The current dynamics should demonstrate clear signs of activity in the direction of breakthrough and beyond the new fractal level.

The drawback of this approach is the uncertainty of a timeframe. It is chosen subjectively, because it is impossible to predict the amplitude of the level breakthrough in advance. Therefore, we simplify the problem by specifying the condition that the level should already be passed at the time of the entry.

When developing a trading system, we can use several options in the entry algorithm to reduce the risk related to the presence of resistance levels.

- **Option 1**. After searching for the nearest resistance fractal (in the code, it is implemented using a loop). The advantage of this method: we will find the real level of resistance on this timeframe. However, there are two drawbacks. First, developing loops may be difficult for novice programmers. Second, the fractal may turn out to be too deep in history meaning it is irrelevant as a resistance level.
- **Option 2**. Use High (for Buy) and Low (for Sell) of the previous candle. There are two advantages of that method. First, it can be easily programmed. Second, you can set several timeframes simultaneously, which is equivalent to searching for older fractals. Disadvantage: some fractals may not be detected, since candle extremes are fractals only in the presence of so-called shadows.

For novice programmers, as well as when developing manual strategies we recommend applying the option 2: it is simple and efficient even though due to some decrease in accuracy because of the loss of some fractals. Here is the code fragment:

```
 //----REDUCE RISKS RELATED TO THE PRESENCE OF RESISTANCE LEVELS AT THE TIME OF MARKET ENTRY-----

     //Simulate the situation when the current price passes through the local resistance levels:
       Bid > High[1] &&           //on М1 (lower timeframe)
       Bid > H_prev_m15 &&        //on М15 (higher timeframe)
```

Fig. 2. Module in the entry algorithm: Reducing risks associated with the presence of resistance levels when entering the market

The Fig. 2 features the module in the entry algorithm for reducing risks associated with the presence of resistance levels for an upward movement, Buy entry. This is achieved by setting a given condition that the current price has already crossed the resistance level as a High of the previous candle (separately on two timeframes).

We should additionally set the movement activity. However, in order to avoid duplication of variables, this has been done in other modules (where both the direction and activity are set).

To enter Sell, use the previous candle's Low levels (M1 and M15 timeframes) as resistance levels considering a downtrend.

#### Risks of ending up in an overbought/oversold zone at the time of a market entry

These risks include the probability of entering at the very end of an active wave-like movement when the remaining amplitude in the entry direction is small and the reversal probability sharply increases. Such areas are nothing other than oversold/overbought areas. Using traditional indicators (RSI etc.) for their definition is often inefficient, in many cases their signals are false. The reason is the same: conventional indicators feature no adequate mathematical algorithms for determining these areas. For more accurate search for overbought/oversold areas, you need to identify the signs of a trend slowing down (including on M1, since it immediately shows the reversals dynamics on all other timeframes).

We will detect the signs of slowing down by combining a fractal and candle analysis according to the following criteria:

- decreasing a distance (amplitude size) between the neighboring fractal resistance levels. In order to compare the amplitudes of two neighboring areas, we need three fractals;
- increasing the correction inside a candle, decreasing the candle "body" (intra-candle analysis);
- changing the direction of the candle's pivot shift relative to the previous candle's pivot (three candles are needed for this).

If this method is used, the condition for the absence of the listed factors should be added to the entry algorithm. For example, this can be done using _false_ and _true_, where _true_ means that the specified slowing factors are present on the market.

However, there is a simpler option. It is not directly related to the search for an overbought/oversold area. An indirect sign is used here: if you provide entry to the market at the beginning of a wave-like movement, the probability of falling into an overbought/oversold area is sharply reduced. In fact, we are _simulating the initial stage of a local trend_.

- First, we need to identify the intersection of one or more moving averages (MA) inside the same candle - this is the possible beginning of a wave-like motion. To confirm the beginning of the wave, we need additional conditions (see below).
- Then simulate the initial phase of the wave-like movement. It will be the beginning of a new trend. To achieve this, we specify: direction of the candle after the crossing; its activity; direction of fast МАs taking part in crossing at the previous bar; the current price location relative to these МАs.

Note: When simulating the initial stage of a local trend, it is recommended to set the direction of fast MAs only. There is no need to set the direction of the slow МАs.

The reason for such a different approach to moving averages is the following: the older MAs, due to their larger delay, do not have time to reverse in the direction of a new trend. Therefore, if we set their direction, the simulated entry point to the market may be far from the beginning of the trend and fall within a dangerous overbought/oversold area.

The second option (simulating the initial stage of a local trend) is simpler and therefore recommended for novice developers. Let's have a look at the code fragment:

```
//---REDUCE RISKS RELATED TO ENTERING THE MARKET IN AN OVERBOUGHT AREA-----

    //Simulate binding to the beginning of the way to reduce the porbability of entering in the overbought area:
     ((MA8_prev > Low[1] && MA8_prev < High[1]) || (MA8_2p > Low[2] && MA8_2p < High[2]) || //start of the wave - no farther than three bars in data history (М1)
     (MA8_3p > Low[3] && MA8_3p < High[3])) &&                                              //same
      MA5_prev_m15 > L_prev_m15 && MA5_prev_m15 < H_prev_m15 &&                             //start of the wave - at the previous bar of the higher timeframe (М15)

```

Fig. 3. Module in the entry algorithm: Reducing risks of falling within an overbought area at the time of a market entry

_Notation_:

- МА8\_prev, МА8\_2p, МА8\_3p — МА with a period of 8 calculated on the previous, 2nd and 3rd bars (respectively) in the quote history (М1),
- МА5\_prev\_m15, МА5\_2p\_m15, МА5\_3p\_m15 — МА with a period of 5 calculated on the previous, 2nd and 3rd bars (respectively) in the quote history (М15),
- candle extreme values are specified earlier (see Fig. 2).

The risk to fall within an overbought area is reduced by binding an entry point to the beginning of an estimated wave. The wave beginning sign is an MA being crossed by the candle: on М1, this is МА with the period of 8, on М15 — МА with the period of 5. The value of the periods of moving averages is chosen from the Fibo series. We will consider this parameter in more details in the "Risks associated with the incorrect selection of the indicator calculation period" section.

This module does not specify parameters that characterize the activity and direction of candles and MAs, as well as the position of the current price relative to MAs. This is done in order not to duplicate the variables. We will set these parameters in the module from the "Risks associated with the absence of a clear trend at the time of a market entry".

Note that the intersection of the candle and the MA on the M1 timeframe is not limited to one bar in history. It is set by the logical OR - either on the previous, or on the 2nd, or on the 3rd bar in the history (on M1). On М15, there is a single crossing option for that case — on the previous bar in the quote history. This set of potential options makes it possible to take into account the multi-variant real market situations associated with the development of a local trend relative to such an intersection.

The above example has been written for Buy entry (avoid the overbought zone). For Sell entry (avoiding the oversold area), the module is the same since the algorithm of МА crossing with a candle does not depend on the movement direction.

So, we considered two ways to enter the market without falling into the overbought/oversold areas, as well as to find these areas. You can experiment yourselves with these methods when creating your trading system or developing trading indicators.

#### Risks associated with the absence of a clear trend at the time of a market entry

The absence of a clearly noticeable trend is yet another factor of dynamics uncertainty and the related risk. We are talking about situations where the market is dominated by either a sideways trend of a small amplitude, or even a flat.

It is difficult to determine the prevailing direction of the price, because it is constantly changing in a narrow range. Therefore, the risk of an error when forecasting a market entry direction is increasing.

The issue is even more complicated since the conventional analytical methods do not provide us with a clear definition of a flat (just like in the case of a trend). Therefore, there is no definition of the boundary between a flat and the beginning of a trend. Existing methods are very subjective: squared deviation method (for example, in the StdDev indicator), as well as more advanced adaptive functions (for example, FRAMA). The issue is even more complicated when dealing with graphical methods of a trend definition. According to different interpretations, various market segments (including the ones with a rather considerable amplitude) are considered to be flat. This leads to loss of profits.

According to my personal experience (confirmed by the TS results), the most efficient way to define a flat border is to set an absolute amplitude value inside a sideways trend.

But keep in mind that if the amplitude is greater than the selected threshold value, this does not mean that we are witnessing a new trend! So do not rush to open a position immediately. You need to confirm these data with the current dynamics.

Applying absolute threshold values seems more promising than estimating the relative amplitude values that are very difficult to define under conditions of a random non-stationary process. Of course, this is a serious simplification, but in practice it gives good results. Without it, you will face a serious issue and complex theoretical calculations, since the boundary between a flat and a trend are the subject of the _fuzzy logic_.

Let's have a look at the code fragment (for Buy entry):

```
 //---REDUCE RISKS RELATED TO THE ABSENCE OF A CLEARLY DEFINED TREND AT THE TIME OF A MARKET ENTRY-------

      //Simulate the candles direction on the lower timeframe:
      Close[2] > Open[2] &&      //upward candle direction on the 2 nd bar in history (М1)
      Close[1] > Open[1] &&      //previous candle's upward direction (М1)

      //Simulate direction of moving averages on the higher timeframe:
      MA5_cur > MA5_2p &&  MA60_cur > MA60_2p &&     //upward МАs: use moving averages with the periods of 5 and 60 (М1)

      //Simulate the hierarchy of moving averages on the lower timeframe:
      MA5_cur > MA8_cur && MA8_cur > MA13_cur &&     //form the "hierarchy" of three МАs on М1 (Fibo periods:5,8,13), this is the indirect sign of the upward movement

      //Simulate the location of the current price relative to the lower timeframes' moving averages:
      Bid > MA5_cur && Bid > MA8_cur && Bid > MA13_cur && Bid > MA60_cur && //current price exceeds МА (5,8,13,60) on М1, this is an indirect sign of the upward movement

      //Simulate the candle direction on the higher timeframe:
      C_prev_m15 > O_prev_m15 &&       //previous candle's upward direction (М15)

      //Simulate the MA direction on the higher timeframe:
      MA4_cur_m15 > MA4_2p_m15 &&     //upward МА with the period of 4 (М15)

      //Simulate the hierarchy of moving averages on the higher timeframe:
      MA4_prev_m15 > MA8_prev_m15 &&  //form the "hierarchy" of two МАs on М15 (periods 4 and 8), this is the indirect sign of the upward movement

      //Simulate the location of the current price relative to the higher timeframes' moving averages:
      Bid > MA4_cur_m15 &&            //current price exceeds МА4 (М15), this is an indirect sign of the upward movement
      Bid > MA24_cur_h1 &&            //current price exceeds МА24 (МН1), this is an indirect sign of the upward movement

      //Simulate a micro-trend inside the current candle of the lower timeframe, as well as the entry point:
      Bid > Open[0] &&               //presence of the upward movement inside the current candle (М1)

     //Simulate sufficient activity of the previous process at the higher timeframe:
     (C_prev_m15 - O_prev_m15) > (0.5*(H_prev_m15 - L_prev_m15)) &&  //share of the candle "body" exceeds 50% of the candle amplitude value (previous М15 candle)
     (H_prev_m15 - C_prev_m15) < (0.25*(H_prev_m15 - L_prev_m15)) && //correction depth limitation is less than 25% of the candle amplitude (previous М15 candle)
      H_prev_m15 > H_2p_m15 &&                                       //upward trend by local resistance levels (two М15 candles)
      O_prev_m15 < H_prev_m15 && O_prev_m15 > L_prev_m15 &&          //presence of a wick (previous М15 candle) relative to the current candle's Open price

     //Simulate sufficient activity of the previous process on the lower timeframe:
     (Close[1] - Open[1]) > (0.5*(High[1] - Low[1])) &&              //share of the candle "body" exceeds 50% of the candle amplitude value (previous М1 candle)
     (High[1] - Low[1]) > 70*Point &&                                //previous candle has an amplitude exceeding the threshold one (excluding an evident flat)
     (High[2] - Close[2]) < (0.25*(High[2] - Low[2])) &&             //correction depth limitation is less than 20% of the candle amplitude (the second candle in the М1 data history)
      High[1] > High[2] &&                                           //upward trend by local resistance levels (two М1 candles)
      Open[1] < High[1] && Open[1] > Low[1] )                        //presence of the wick (previous tfМ1 candle) relative to the current candle's Open price

```

Fig. 4. Module in the entry algorithm: Reducing risks associated with the absence of a clearly defined trend at the time of a market entry

_Notation_:

- Open\[1\], Close\[1\] — Open and Close prices on the previous bar (М1) accordingly;
- МА5\_cur, MA8\_cur, MA13\_cur, MA60\_cur — МА values on the current bar with the periods of 5,8, 13 and 60 accordingly (М1);
- MA4\_cur\_m15, MA4\_prev\_m15, MA4\_2p\_m15 — МА with a period of 4 calculated on the previous and 2 nd (respectively) in the quote history (М15);
- MA8\_prev\_m15 — МА value with the period of 8 on the previous bar in the quote history (М15).

We reduce this risk by simulating a clearly defined trend in two timeframes (M1 and M15) simultaneously:

- previous candles in the direction of position opening (two candles at М1 and one candle at М15);
- direction of MAs (two МАs at М1, one MA at М15);
- hierarchy of moving averages (three МАs at М1 and two MAs at М15);
- position of the current price relative to MAs — three МАs on М1.

This "set of measures" significantly increases the probability that the position entry point will be inside a clearly defined trend and on two timeframes simultaneously. Accordingly, the probability of falling within a flat zone, random fluctuations and other areas of an unclear trend decreases. The risk associated with this unfavorable factor decreases as well.

This example is for Buy entry. For Sell, the opposite algorithm is used — the downward movement of candles and МАs is set instead of an upward one. The entry point is below the specified МАs.

#### Risks associated with the incorrect selection of the indicator calculation period

Each trader sets the indicator period, including the Moving Average, based on personal experience. Someone prefers MA with the period of 200, someone sets the period of 50, while someone follows the Fibo series. We choose specific indicators settings (and first of all, the period that we are most interested in) intuitively.

The reason for this forced intuition is that the conventional methods of analysis provide no mechanisms for identifying frequency-modulated oscillations at a particular moment in time. This leads to uncertainty when setting the indicator period. Of course, there are methods for constructing adaptive functions (Kaufman, FRAMA etc.), but their algorithms also do not take into account the constantly changing frequency of market fluctuations.

Let's consider the partial solution: we use conventional methods of analysis adding some logic in the definition of the MA periods. We will apply some constant factors related to time. In favor of this principle is the fact that the boundaries of the candles of large timeframes are fractal levels for intra-candle movements, but only if these candles have wicks (shadows). If there is no wick, then the movement can continue to the next candle without forming a fractal.

We compare the standard timeframes and numbers of the Fibo series (close in magnitude). As a result, we obtain the approximate correspondence between timeframes and MA calculation periods:

- 1 minute — nearest Fibo value 1 (minute, this is a pivot);
- 5 minutes — nearest Fibo value 5 (minutes);
- 15 minutes – nearest Fibo value 13 (minutes);
- 1 hour (60 minutes) — nearest Fibo value 55 (minutes);
- 4 hours (240 minutes) — nearest Fibo values 3 (hours), 5 (hours), 233 (minutes);
- 1 day (24 hours) — nearest Fibo value 21 (hours);
- 5 days (trading week, 120 hours for Forex market) — nearest Fibo values 89 (hours) and 144 (hours).

In addition to this, we can recommend an option that minimizes the MA delay:

- use only the first numbers of the series at each timeframe: 1 (pivot), 3, 5, 8, 13;

- use them in a complex way: convert into the younger timeframe using a ratio equal to the ratio of timeframes. Using МА with minimum periods allows reducing the delay of these functions.

As a result, we obtain a set of MAs for different timeframes (example of using four timeframes):

- for М1: МАs with the periods of 5, 8, 13, 55 (or 60 =1 hour), 233 (or 240 = 4 hours);
- for М15: МА with the periods of 5, 8, 13, 55 (or 60=4 hours?);
- for Н1: МА with the periods of 5, 8, 13, 21 (or 24 = 1 day), 89, 144 (or 120 = 5 days);
- for D1: МА with the periods of 5, 8, 13, 21 (or 24 = 1 trading month).

Of course, this is only a sample set of MAs. You are free to add your own values. Anyway, this principle of choosing MA periods has the right to exist, since many factors, including the beginning and end of trading sessions, statistics periodicity, paying dividends, etc., have a clearly defined periodicity. Thus, the logical choice of the indicator period (in this case MA) allows us to take into account the frequency of the market events, and therefore, reduce the risks to some extent.

We will use these recommendations on choosing the MAs in the development of other described modules.

#### Risks associated with the use of pending orders for entering the market

Here we mean strategies relying on pending orders (rather than market ones) to set a target level. More often, pending orders are used in strategies associated with the exit of prices from the consolidation zones.

With all the variety of such strategies, let's talk about using a pair of pending orders placed on both sides of the consolidation zone in the hope that one order works correctly, and the second is removed immediately after the first one is triggered. Note that the use of pending orders to determine an entry point is associated with a risk. The level the pending order is set to is always defined before a desired price is reached. The pending order level is defined, for example, in % of the consolidation area amplitude or intuitively, and this has nothing to do with the real price dynamics.

In fact, this is the **main drawback of pending orders compared to market ones – inability to avoid opening a position in the presence of negative factors at the moment of entry.** Right after reaching the target price, the market dynamics may turn out to be unfavorable for this pending order type but the market entry will still take place causing a loss.

Therefore, if you still want to trade from specific fixed levels, then it is more reasonable to assign them inside the entry algorithm virtually instead of applying standard pending orders. Enter the market using fixed levels only if the necessary dynamics has been detected when they are crossed. This, of course, is more complicated than the conventional pending orders, since we require an additional algorithm for controlling dynamics at the time the price crosses a virtual level (programming skills for developing such an algorithm are required as well). But this will allow you to get rid of trading "in the dark", which is inevitable when applying conventional pending orders.

This method also has its drawbacks. When the price reaches the required level, we enter using the market (not pending) order, which will be reflected in the platform of the broker later than the already placed pending orders. If there are a lot of orders, the execution delay probability is high due to orders priority. In case of a sharp price spike at the time of the entry, slippage is possible as well.

Thus, we should choose between the risk of entering "in the dark" by using pending orders and market orders' execution delay risk. At the same time, the advantage of using a virtual level is that the system can automatically cancel the entry, if there are no additional favorable conditions for entering the market when the defined level is reached.

My personal opinion: since the use of pending orders increases the risk, it is necessary to reduce it by abandoning such orders. This is justified, among other things, by the fact that only the market determines the amplitude of price movements. Our only objective is to fix such movements at a certain stage of development and then to accompany them using analytical functions.

#### Risks associated with uncertainty of the price movement amplitude after a market entry

A typical mistake of many traders is that they are excessively addicted to defining specific target levels. Considering a random, non-stationary process of the price movement, **the final trend amplitude is a probability value**. Like with pending orders described above, placing fixed target profit levels (i.e. for defining a market exit point) is intuitive. Therefore, this approach often leads to losses.

The market exit algorithm should feature either a function of controlling the signs of slowing down (adaptive function) or the function for controlling a fixed amplitude value relative to certain levels (entry level, current high or low inside an open position). We are also going to add the risk limitation control to the second option — both per position and the entire deposit. We will use the last, simpler version. Let's have a look at the code fragment:

```
 //REDUCE RISKS RELATED TO THE UNCERTAINTY OF THE PRICE MOVEMENT AMPLITUDE AT THE TIME OF MARKET ENTRY--

              //Track the fixed profit value (per position):
              (Bid > OrderOpenPrice() && (Bid - OrderOpenPrice()) >= 100*Point)                       //exit conditions in the profit area (shadow take profit)
                     ||

             //Manage the maximum available price deviation
             //from the current maximum after entering the market:
             (shift_buy >= 1 &&                                                                          //shift no less than 1 bar from the entry point
             Time_cur > OrderOpenTime() && Max_pos > 0 && OrderOpenTime() > 0 && OrderOpenPrice() > 0 && //there is the current maximum after the entry
             Max_pos > OrderOpenPrice() &&                                                               //current maximum is in the profit area
             Bid < Max_pos &&                                                                            //there is the price reverse movement
             (Max_pos - Bid) >= 200*Point)                                                               //reverse deviation from the current maximum for exiting the market
                    ||

             //Track the pre-defined risk limit (per position):
              (Bid < OrderOpenPrice() && (OrderOpenPrice() - Bid) >= 200*Point)                          //entry conditions in the loss area (shadow stop loss)
                    ||

             //Track the pre-defined risk limit (entire deposit):
              (AccountBalance() <=  NormalizeDouble( (Depo_first*((100 - Percent_risk_depo)/100)), 0)) )  //if the risk limit for the entire deposit has been exceeded during the current trading
```

Fig. 5. Module in the exit algorithm: Reducing risks associated with uncertainty of the price movement amplitude after a market entry

_Notation_:

- OrderOpenPrice() — market entry price;
- Shift\_buy — shift (in bars on М1) relative to the entry point (necessary for defining a maximum inside an open position);
- Max\_pos  — maximum level inside an open position;
- AccountBalance() — current balance in monetary terms;
- Depo\_first — initial deposit in monetary terms;
- Percent\_risk\_depo — maximum allowable losses % for the entire deposit.

This type of risk is reduced by using the following functions:

- Managing the fixed profit per position is, in fact, a "shadow" take profit.
- Managing the maximum allowable price deviation from the current maximum (for Buy) or minimum (for Sell) after a market entry.
- Managing a pre-defined risk limitation per position is a "shadow" stop loss.
- managing a pre-defined risk limitation for the entire deposit.

In all these cases, we control a certain amplitude value relative to levels capable of changing their value, including the entry level, the maximum inside the open position and the initial deposit. Therefore, unlike the tightly set target levels, in this case, these levels are not fixed.

We have considered the sample module in the Buy position close algorithm. For Sell module, changes occur only in the trading direction. Instead of the current maximum within the position (Max\_pos), the current minimum (Min\_pos) is used.

In the next section, we consider risks that also take into account the price speed, which is especially important when the market price collapses.

#### Risks associated with a price collapse after a market entry

One of the dangerous forms of a local trend reversal are the crashes and spikes with a large amplitude in a short time. The jump of one currency in a pair always means a collapse of another. It is especially dangerous if there is an open position and the direction of the collapse is unfavorable to it. In this case, a small deposit may be completely lost.

**The essence of the problem**

- Rates of price collapses leave no time for market participants to adequately respond to them.
- The modern analytics offers no mechanism for identifying dynamic structures inherent for rapid price collapses.
- If there is an open position, the market participants are completely defenseless during price collapses. It is quite difficult to detect a collapse at an early stage and even more so to respond to it, since the main movement peak has already passed or the market is blocked due to panic in the chain of brokers and banks.

As a result, the market participants suffer huge losses. For example, on May 6, 2010, the Dow Jones index fell by 1000 points in 6 minutes, while, according to expert estimates, the market lost about a trillion dollars.

The more recent example (Fig. 6) is Brexit, which provoked the collapse of GBPUSD on June 24, 2016 by 560 points at once. 473 points from them were lost in one minute:

![GBPUSD 24/06/2014](https://c.mql5.com/2/30/pic_6.png)

Fig. 6. GBPUSD price collapse on June 24, 2016

There are three global reasons for price collapses.

1. **The nature of the market itself**. The collapse is not something alien, but a natural manifestation of market dynamics. Among other things, it may cause sharp changes in the price movement rate (for example, due to the growing number of market participants, including trade robots). For example, according to media reports, thousands of ultra-fast fluctuations lasting less than a second are registered in the US stock market.

2. **Analytics development level.** The methods of identifying such market manifestations are imperfect. The adequate response requires the ability to analyze the situation in fractions of a minute, while second timeframes are a rare thing for market platforms.

3. **Faults of a legislation** regulating activities in financial markets. For example, there are no legal mechanisms to combat the artificial price manipulation performed by market makers.


Here is an example of how an MACD-based EA does not have time to react to a sharp collapse of the USDCHF price:

![USDCHF- MACD](https://c.mql5.com/2/30/pic_7.png)

Fig. 7. The MACD-based EA has no time to react to USDCHF price collapse on October 2, 2015

The Fig. 7 shows that the EA opens two positions — before (arrow 1) and after (arrow 2) the price collapse. It did not respond to the collapse itself because it simply "did not notice" it.

Thus, if you have an open position, there is a risk of losing a significant part of the deposit during a price collapse or rally. This means the position close algorithm should have the special protection module.

Let's have a look at the code fragment:

```
 if(
            //REDUCE RISLS RELATED TO PRICE COLLAPSES AT THE TIME OF MARKET ENTRY----------------------

              (Bid < Open[0] && (Open[0] - Bid) >= 100*Point && (Time_cur - Time[0]) <= 20)           //exit conditions (in any zones) during a price collapse (reference point - М1 current candle open price)
                     ||
              (Bid < O_cur_m15 && (O_cur_m15 - Bid) >= 200*Point && (Time_cur - Time_cur_m15) <= 120) //exit conditions (in any area) during the price collapse (reference point - current М15 candle open price)
                     ||
              ((Time_cur - OrderOpenTime()) > 60 && Close[1] < Open[1] &&
              (Open[1] - Close[1]) >= 200*Point)                                                      //exit conditions in any area during a price collapse (reference point - previous М1 candle amplitude)
                     ||
```

Fig. 8. Module in the exit algorithm: Reducing risks associated with price collapses after a market entry

The price collapse risk after opening a position are reduced the following way.

- The limits of the price reverse deviation (collapse) are set relative to the current candle's Open level separately on different timeframes (in our case, these are М1 and М15). The maximum acceptable duration of such a collapse is defined as well.
- The amplitude of the maximum permissible collapse in the form of the completed previous candle on M1 is set (since this is a minute candle, then the time of the collapse is indirectly specified as well).

Thus, the collapse is tracked both at the beginning of the current candle (using the collapse at the previous one) and in the process of its development. If any of the specified conditions appear (by OR logic), the position is closed — the risk is minimized.

The module for closing a Buy position is shown in this example. To close Sell, consider the price movement mirrorwise.

#### Risks associated with using only one timeframe

Price changes are reflected differently in the charts of different timeframes. The apparent trend on a lower timeframe may be only a minor correction, if we consider it at a higher one.

Therefore, we should analyze the price dynamics at several timeframes — one is not enough.

How many timeframes should be used for analysis? The answer to this question is up to you. Personally, I use four timeframes simultaneously (on four screens) because:

- М1: this timeframe is good at showing trend reversals (including global ones) and fast market collapses;
- М15: candles on this period of the chart often reflect the completed dynamic structures of fast market crashes, formed on lower timeframes (М1 and М5);
- Н1: candles on this timeframe are natural time limiters (including in trading session schedule, release of macroeconomic indices, news, etc.);
- D1: candles on this timeframe are natural limiters (including trading days within a week).

I believe, these timeframes form an efficient set since it allows you to accurately analyze the market dynamics. Of course, we need some experience of "volumetric" perception of price movements, but this is indispensable when working in financial markets.

Since the use of one timeframe inevitably increases the risks of trading, the code of the entry algorithm should include the analysis of candles, MAs and other indicators in relation to several timeframes.

#### Risks associated with the use of only one type of analysis (technical or fundamental)

The price movement is affected by many factors. The market is a behavioral system based on the economic interests of the participants. The technical analysis (trading robots are based on) uses only one kind of data — price levels of a particular financial instrument. In turn, the fundamental analysis also considers only one kind of data (for example, a specific macroeconomic factor). Obviously, such an analysis is incomplete, one-sided, and therefore inaccurate. It would be incorrect to analyze only prices without taking into account the factors of their movement or focus only on fundamental indicators without taking into account the real dynamics of the price.

It is more reasonable to combine both types of analysis, as they logically complement each other. Fundamental factors in general (but not absolutely) predetermine the price movement direction, while technical factors confirm it. But the primacy of fundamental factors relative to technical ones is not absolute.

Sometimes, the market reacts more to technical factors rather than fundamental ones, for example when reaching historical highs. A striking example of this is the periodic collapse of bitcoin (sometimes by 20% per day) immediately after reaching a historic high. However, the example with bitcoin is not entirely indicative, since the fundamental analysis of cryptocurrency is problematic because of the market specifics.

However, in case of Forex and especially stock market, the impact of fundamental analysis is obvious. For example, the collapse of USDCHF (in October 2015) was provoked by the Swiss bank "untying" the CHF rate from the EUR one.

Therefore, I believe, there is no point in trying to define the priority of one of the analysis types and use them in conjunction instead. However, there is a question: How can we use the fundamental analysis in automated trading considering that trading system algorithms apply only technical market parameters?

We suggest two options for embedding the fundamental analysis into the TS.

- **Option 1.** Implement the fundamental analysis parameters in the TS algorithm converting them into technical ones (for example, enter the data of the economic calendar after defining their action logic).
- **Option 2**. Manually limit the market entry direction inside an Expert Advisor (only in the direction corresponding to fundamental parameters).

As we can see, using the fundamental analysis in the algorithm is inconvenient, but this is the right way. The combined application of the both methods in manual trading is simpler. This is what novice analysts and traders should strive for.

### Risks not associated with market dynamics

External factors — operational, financial and legal ones — are also part of the trading process. They play an equally important role in the occurrence of risks. Therefore, we should take them into account when developing a trading system if possible.

Let's define their list.

#### Classifying risks not associated with market dynamics

Here are the most common of possible risks not related to the price dynamics.

- Risks associated with the trading system structure.
- Risks associated with exceeding a loss limit by deposit (managing investor's risk limit by deposit).
- Risks associated with negative changes of trading conditions.
- Risks determined by the quality of connection to a broker's trade server.
- Risks associated with the permission for automated trading on the broker's or client's side
- Risks associated with changes in the financial markets legislation.

Let's have a closer look at these risks and the ways to reduce them.

#### Risks associated with the trading system structure

The main rule: **the structure of a trading system should take into account the structure of risks**. It should include the protection against the maximum possible number of risks considered above. This will improve the trading efficiency.

However, while we increase the number of blocks to minimize risks, we inevitably increase the number of filters, since any method of avoiding risks is, in fact, a filter. As a result, we face with the "over-filtering" of a trading robot's entry algorithm. At the same time, most traders believe that the robot should actively trade, opening positions at least daily. In this sense, a trader acts as a robot's customer, and such a requirement seems quite logical from their side. However, I believe, this is incorrect due to the very nature of the process.

Any EA algorithm simulates very complex dynamic processes inevitably simplifying them. The errors are related to incomplete matching of a certain EA's model to the dynamics of the market.

Each developer knows that there is a relationship between the number of entries in a trading system and the amount of filtering in the entry algorithm. The greater the amount of filtering, the smaller the number of entries, and vice versa. Generally, as filtration increases, the financial result improves. But, of course, it all depends on the content of the filters. You need to find a compromise between the number of entries and the financial result, or more precisely the maximum risk (deposit drawdown) for each trading system. This is called conventional optimization.

**But what is filtration?**

If it means the allocation of a useful signal from the entire spectrum, then we need clear criteria of this usefulness. The conventional analysis is unable to offer them. Activation of conventional indicators can hardly be called "correct" filtration of such a complex process as price movement. For example, if we apply Elliott wave structures or technical analysis patterns to find a useful signal, we will realize that they are not suitable for analysis since each trader determines them differently on the same chart.

Therefore, the trading system structure should reduce the risks by sorting out the critical values ​​of the parameters these risks are determined by. But there are a lot of risks, and the degree of filtration grows along with their quantity. Therefore, when evaluating the efficiency of an individual EA, the main thing is not the number of entries, but a stable (albeit low) profit with an acceptable risk (drawdown). In this case, the number of entries (when testing on a single financial instrument) may be insignificant. The profit can be increased by trading simultaneously on several currency pairs.

Hence follows the conclusion:

The number of positions to be opened is a secondary factor in assessing the EA's quality. The main thing is its relative stability when making a profit.

The efficiency of a trading system should be evaluated on several financial instruments. This improves the quality of the simulation and the eventual result:

- a TS is tested on a variety of manifestations of market dynamics;
- in case of a positive result on several instruments, the total profit on the deposit grows satisfying the investor's demand for profit value.

Hence, the optimization of a TS should be reduced to the selection of such settings that show profit when tested on several financial instruments.

#### Risks associated with exceeding a loss limit by deposit

Even before trading, it is necessary to set a limit on the risk of a trade deposit. This may be a fixed and absolute (in monetary terms) or relative (in % of an initial deposit sum) amount. Upon its achievement, trading stops.

If you entrust a more experienced trader with managing your account, set the risk limit in the agreement.

If you trade on your own, you need to add the code to both the entry algorithm (disable an entry if the risk limit is exceeded) as well as to the exit one (we discussed this earlier in the "Risks associated with uncertainty of the price movement amplitude after a market entry" section). The double control of the deposit is necessary because at first, the limit of the risk on the deposit is achieved when there is an open position (this is fixed in the exit algorithm of the market). After that, we need to disable subsequent entries since the risk limit for the deposit have already been achieved (this is fixed in the module before the market entry algorithm).

Let's have a look at the code fragment:

```
 //MANAGE FINANCIAL PARAMETERS RELATED TO LIMITING THE TOTAL RISK FOR THE CLIENT'S DEPOSIT-------------------------------

  if(kol < 1) //no orders
  {
   if(AccountBalance() <=  NormalizeDouble( (Depo_first*((100 - Percent_risk_depo)/100)), 0))//if the risk limit for the entire deposit have been previously exceeded
    {
     Print("Entry disabled-risk limit reached earlier=",Percent_risk_depo, " % for the entire deposit=", Depo_first);
     Alert("Entry disabled-risk limit reached earlier=",Percent_risk_depo, " % for the entire deposit=", Depo_first);
     return;
    }


   if(AccountFreeMargin() < (1000*Lots)) //if margin funds allowed for opening orders on the current account are insufficient
    {
     Print("Insufficient margin funds. Free account margin = ",AccountFreeMargin());
     Alert("Insufficient margin funds. Free account margin = ",AccountFreeMargin());
     return; //...then exit
    }
  }

 //--------------
```

Fig. 9. Module before the entry algorithm: Reducing risks associated with exceeding a loss limit for a deposit

The first block manages the client deposit status (AccountBalance() variable). Customizable variables:

- Depo\_first — initial deposit in monetary terms;
- Percent\_risk\_depo — loss limit by deposit (in % of its initial value).

The algorithm is triggered if the losses on the deposit reach the value set in the Percent\_risk\_depo variable. At the same time, the market entry is disabled, and we exit the program.

The second block manages the account's margin, at which a market entry is allowed. The algorithm is triggered when the margin volume is below a certain number of lots. The market entry is disabled, and we exit the program.

#### Risks associated with unfavorable trading conditions

These may be: quality of quotes provided by a broker; spread expansion when opening and closing a position; price slippage when executing orders; order execution delays.

All this can strongly affect the results of trading (up to the total loss of the deposit), so you need to monitor the relevant parameters. In particular, you need to automatically track the values ​​of the price, spread, entry and exit algorithm trigger time, and then compare them with the values ​​at the actual opening and closing of the position.

You can develop such a module in your trading system on your own.

#### Risks determined by the quality of connection to a broker's trade server

The quality of connection to a trade server depends on two factors:

- trade server status on the broker's side;
- Internet connection on the client's side.

The second factor can be eliminated by placing the client terminal on an external server. The first factor is verified programmatically by adding a code to the TS algorithm.

Let's have a look at the code fragment:

```
 if(IsConnected() == false) //check the main connection of the client terminal to the broker's server
    {
     Print("Entry disabled-broker's server offline");
     Alert("Entry disabled-broker's server offline");
     return;
    }
```

Fig. 10. The module of reducing risks related to the quality of connection to a broker's trade server

Fig. 10 shows the code fragment allowing to monitor connection to the broker's server. It needs to be built into the EA before the entry algorithm. When these conditions occur, opening positions is disabled.

#### Risks associated with the permission for automated trading on the broker's and client's sides

Sometimes, a TS sends an order to the client terminal, but it is not executed. However, the TS itself may not restrict trading. Possible reasons:

- the automated trading is disabled in the client terminal settings on the client's side;
- trading on the current account is disabled by a broker.

To eliminate these risks, it is necessary to control whether automatic trading is allowed in the trading system program.

Let's have a look at the code fragment:

```
 if(IsTradeAllowed() == false) //check the ability to trade using EAs (broker's flow, trading permission)
    {

     Print("Entry disabled-broker's trading flow busy and/or the robot has no permission to trade :", IsTradeAllowed());
     Alert("Entry disabled-broker's trading flow busy and/or the robot has no permission to trade :", IsTradeAllowed());
     return;
    }


    if( !AccountInfoInteger(ACCOUNT_TRADE_EXPERT) ) //check trading account properties
    {
     Print("Automated trading disabled for the account :", AccountInfoInteger(ACCOUNT_LOGIN), "on trade server side");
     Alert("Automated trading disabled for the account :", AccountInfoInteger(ACCOUNT_LOGIN), "on trade server side");
     return;
    }


    if(IsExpertEnabled() == false) //check if launching EAs is allowed in the client terminal
    {

     Print("Entry disabled- robot's trading permission disabled in the terminal :", IsExpertEnabled());
     Alert("Entry disabled- robot's trading permission disabled in the terminal :", IsExpertEnabled());
     return;
    }


   if(IsStopped() == true) //check the arrival of the command to complete mql4 program execution
    {
     Print("Entry disabled-command to complete mql4 program execution triggered");
     Alert("Entry disabled-command to complete mql4 program execution triggered");
     return;
    }
```

Fig. 11. Module for reducing the risks associated with the permission for automated trading on the broker's and client's sides

The above code fragment controls the state of permissions for automated trading both on the broker's and client's sides. When you meet any of these conditions, the market entry is disabled.

#### Risks associated with changes in the financial markets legislation

These are the rules of relationships in the financial markets industry that directly or indirectly affect the client's risks. Various countries develop legislative acts regulating the relations in the stock and financial markets. In the future, many risks associated with various violations should decrease.

### A sample EA considering some of the mentioned risks

We have identified and examined the risks both related and not related to the market dynamics. Besides, we have defined specific solutions to reduce them.

The risks are reduced by inserting specific modules to the EA (before the entry algorithm, as well as in the entry and exit algorithms). Each module manages a certain type of risk.

The main analysis elements are candles of different timeframes. The dynamics is analyzed both inside candles (single structure analysis) and between them (structure groups analysis).

Below is a code of a simple EA featuring solutions for reducing some risks described above. Please note that it applies only Moving Averages out of all conventional indicators (considering the above mentioned limitations).

The EA code for 5-digit quotes (or 3-digit ones for USDJPY):

```
//+------------------------------------------------------------------+
//|                                                 Reduce_risks.mq4 |
//|                            Copyright 2017, Alexander Masterskikh |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright   "2017, Alexander Masterskikh"
#property link        "https://www.mql5.com/en/users/a.masterskikh"

input double TakeProfit     = 600;   //take profit
input double StopLoss       = 300;   //stop loss
input double Lots           = 1;     //number of lots
input int Depo_first        = 10000; //initial client deposit in monetary terms
input int Percent_risk_depo = 5;     //maximum allowed risk per deposit (in % to the client's initial deposit)
input bool test             = false; //set: true - for test (default is false for trading)

//-------------------------------------------------------------------------------------------------

void OnTick(void)
  {
   //---define variables---
   int f,numb,kol;
   double sl_buy, tp_buy, sl_sell, tp_sell;
   double L_prev_m15, L_2p_m15, L_3p_m15;
   double H_prev_m15, H_2p_m15, H_3p_m15;
   double O_cur_m15, O_prev_m15, O_2p_m15, C_prev_m15, C_2p_m15;
   double MA4_cur_m15, MA4_prev_m15, MA4_2p_m15, MA5_prev_m15, MA8_cur_m15, MA8_prev_m15, MA8_2p_m15;
   double MA8_cur, MA8_prev, MA8_2p, MA8_3p, MA5_cur, MA5_prev, MA5_2p, MA13_cur, MA13_prev, MA13_2p,
       MA60_cur, MA60_prev, MA60_2p, MA24_cur_h1;
   double C_prev_h1, O_prev_h1, H_prev_h1, L_prev_h1, H_2p_h1, L_2p_h1;
   datetime Time_cur, Time_cur_m15;
   double shift_buy, shift_sell;
   double Max_pos, Min_pos;
  //--------

   //MANAGING TECHNICAL PARAMETERS (STATES AND PERMISSIONS) - ON BROKER'S AND CLIENT TERMINAL'S SIDES-------------------------------

 if(test==false) //check: test or trade (test - true, trade - false)
 {//parameters that are not checked when testing on history data---

    if(IsConnected() == false) //check the status of the main connection of the client terminal with the broker server
    {
     Print("Entry disabled-broker server offline");
     Alert("Entry disabled-broker server offline");
     return;
    }


    if(IsTradeAllowed() == false) //check ability to trade using EAs (broker flow, permission to trade)
    {

     Print("Entry disabled-broker trade flow busy and/or EA trading permission disabled :", IsTradeAllowed());
     Alert("Entry disabled-broker trade flow busy and/or EA trading permission disabled :", IsTradeAllowed());
     return;
    }


    if( !AccountInfoInteger(ACCOUNT_TRADE_EXPERT) ) //check trade account properties
    {
     Print("Auto trading disabled for account :", AccountInfoInteger(ACCOUNT_LOGIN), "on trade server side");
     Alert("Auto trading disabled for account :", AccountInfoInteger(ACCOUNT_LOGIN), "on trade server side");
     return;
    }


    if(IsExpertEnabled() == false) //check if launching EAs is allowed in the client terminal
    {

     Print("Entry disabled- robot's trading permission disabled in the terminal :", IsExpertEnabled());
     Alert("Entry disabled- robot's trading permission disabled in the terminal :", IsExpertEnabled());
     return;
    }


   if(IsStopped() == true) //check the arrival of the command to complete mql4 program execution
    {
     Print("Entry disabled-command to complete mql4 program execution triggered");
     Alert("Entry disabled-command to complete mql4 program execution triggered");
     return;
    }

 } //parameters not checked when testing on history data end here


  //Manage the presence of a sufficient number of quotes in the terminal by a used symbol---

     if(Bars<100)
     {
      Print("insufficient number of bars on the current chart");
      return;
     }

  //Manage placing an EA on a necessary timeframe----

    if(Period() != PERIOD_M1)
    {
     Print("EA placed incorrectly, place EA on tf :", PERIOD_M1);
     Alert("EA placed incorrectly, place EA on tf :", PERIOD_M1);
     return;
    }

  //Manage placing the EA to necessary financial instruments----

   if(Symbol() != "EURUSD" && Symbol() != "USDCHF" && Symbol() != "USDJPY")
     {
     Print("EA placed incorrectly-invalid financial instrument");
     Alert("EA placed incorrectly-invalid financial instrument");
     return;
    }

    //----------


  //MANAGE FINANCIAL PARAMETERS RELATED TO LIMITING THE TOTAL RISK FOR THE CLIENT'S DEPOSIT-------------------------------

  if(kol < 1) //no orders
  {
   if(AccountBalance() <=  NormalizeDouble( (Depo_first*((100 - Percent_risk_depo)/100)), 0))//if the risk limit for the entire deposit have been previously exceeded
    {
     Print("Entry disabled-risk limit reached earlier=",Percent_risk_depo, " % for the entire deposit=", Depo_first);
     Alert("Entry disabled-risk limit reached earlier=",Percent_risk_depo, " % for the entire deposit=", Depo_first);
     return;
    }


   if(AccountFreeMargin() < (1000*Lots)) //if margin funds allowed for opening orders on the current account are insufficient
    {
     Print("Insufficient margin funds. Free account margin = ",AccountFreeMargin());
     Alert("Insufficient margin funds. Free account margin = ",AccountFreeMargin());
     return; //...then exit
    }
  }

 //--------------


  //Variable values:

    L_prev_m15 = iLow(NULL,PERIOD_M15,1);
    L_2p_m15 = iLow(NULL,PERIOD_M15,2);
    L_3p_m15 = iLow(NULL,PERIOD_M15,3);
    H_prev_m15 = iHigh(NULL,PERIOD_M15,1);
    H_2p_m15 = iHigh(NULL,PERIOD_M15,2);
    H_3p_m15 = iHigh(NULL,PERIOD_M15,3);

    O_cur_m15 = iOpen(NULL,PERIOD_M15,0);
    O_prev_m15 = iOpen(NULL,PERIOD_M15,1);
    O_2p_m15 = iOpen(NULL,PERIOD_M15,2);
    C_prev_m15 = iClose(NULL,PERIOD_M15,1);
    C_2p_m15 = iClose(NULL,PERIOD_M15,2);
    Time_cur_m15 = iTime(NULL,PERIOD_M15,0);

    C_prev_h1 = iClose(NULL,PERIOD_H1,1);
    O_prev_h1 = iOpen(NULL,PERIOD_H1,1);
    H_prev_h1 = iHigh(NULL,PERIOD_H1,1);
    L_prev_h1 = iLow(NULL,PERIOD_H1,1);
    H_2p_h1 = iHigh(NULL,PERIOD_H1,2);
    L_2p_h1 = iLow(NULL,PERIOD_H1,2);

    MA4_cur_m15 = iMA(NULL,PERIOD_M15,4,0,MODE_SMA,PRICE_TYPICAL,0);
    MA4_prev_m15 = iMA(NULL,PERIOD_M15,4,0,MODE_SMA,PRICE_TYPICAL,1);
    MA4_2p_m15 = iMA(NULL,PERIOD_M15,4,0,MODE_SMA,PRICE_TYPICAL,2);
    MA5_prev_m15 = iMA(NULL,PERIOD_M15,5,0,MODE_SMA,PRICE_TYPICAL,1);
    MA8_cur_m15 = iMA(NULL,PERIOD_M15,8,0,MODE_SMA,PRICE_TYPICAL,0);
    MA8_prev_m15 = iMA(NULL,PERIOD_M15,8,0,MODE_SMA,PRICE_TYPICAL,1);
    MA8_2p_m15 = iMA(NULL,PERIOD_M15,8,0,MODE_SMA,PRICE_TYPICAL,2);

    MA8_cur = iMA(NULL,PERIOD_M1,8,0,MODE_SMA,PRICE_TYPICAL,0);
    MA8_prev = iMA(NULL,PERIOD_M1,8,0,MODE_SMA,PRICE_TYPICAL,1);
    MA8_2p = iMA(NULL,PERIOD_M1,8,0,MODE_SMA,PRICE_TYPICAL,2);
    MA8_3p = iMA(NULL,PERIOD_M1,8,0,MODE_SMA,PRICE_TYPICAL,3);
    MA5_cur = iMA(NULL,PERIOD_M1,5,0,MODE_SMA,PRICE_TYPICAL,0);
    MA5_prev = iMA(NULL,PERIOD_M1,5,0,MODE_SMA,PRICE_TYPICAL,1);
    MA5_2p = iMA(NULL,PERIOD_M1,5,0,MODE_SMA,PRICE_TYPICAL,2);
    MA13_cur = iMA(NULL,PERIOD_M1,13,0,MODE_SMA,PRICE_TYPICAL,0);
    MA13_prev = iMA(NULL,PERIOD_M1,13,0,MODE_SMA,PRICE_TYPICAL,1);
    MA13_2p = iMA(NULL,PERIOD_M1,13,0,MODE_SMA,PRICE_TYPICAL,2);
    MA60_cur = iMA(NULL,PERIOD_M1,60,0,MODE_SMA,PRICE_TYPICAL,0);
    MA60_prev = iMA(NULL,PERIOD_M1,60,0,MODE_SMA,PRICE_TYPICAL,1);
    MA60_2p = iMA(NULL,PERIOD_M1,60,0,MODE_SMA,PRICE_TYPICAL,2);
    MA24_cur_h1 = iMA(NULL,PERIOD_H1,24,0,MODE_SMA,PRICE_TYPICAL,0);

    kol = OrdersTotal();
    Time_cur = TimeCurrent();

 if(kol < 1) //continue if there are no open orders
     {

 //---MARKET ENTRY ALGORITHM - BUY-------------------------------------------------------------------------------------------

  if(
   //----REDUCE RISKS RELATED TO THE PRESENCE OF A STRONG VOLATILITY AT THE TIME OF MARKET ENTRY ----

     //Simulate the absence of a strong volatility in the recent history:
     ( High[1] - Low[1]) <= 200*Point &&                       //limit the amplitude of a lower timeframe (М1)
     ( High[2] - Low[2]) <= 200*Point &&
     ( High[3] - Low[3]) <= 200*Point &&
     (H_prev_m15 - L_prev_m15) <= 300*Point &&                 //limit the amplitude of a higher timeframe (М15)
     (H_2p_m15 - L_2p_m15) <= 300*Point &&
     (H_3p_m15 - L_3p_m15) <= 300*Point &&
     (H_prev_m15 - L_3p_m15) <= 300*Point &&                   //limit the amplitude of the channel made of the higher timeframe candles (М15)
     (High[1] - Low[1]) >= (1.1*(High[2] - Low[2])) &&         //limit activity on the previous bar relative to the 2 nd bar in the quote history
     (High[1] - Low[1]) < (3.0*(High[2] - Low[2])) &&          //same


   //----REDUCE RISKS RELATED TO RESISTANCE LEVELS AT THE TIME OF MARKET ENTRY-----

     //Simulate the case when local resistance levels are broken by the current price:
       Bid > High[1] &&           //on М1
       Bid > H_prev_m15 &&        //on М15


   //---REDUCE RISKS RELATED TO ENTERING THE OVERBOUGHT AREA AT THE TIME OF MARKET ENTRY-----

    //Simulate binding to the start of the wave to decrease the entry probability in the overbought area:
     ((MA8_prev > Low[1] && MA8_prev < High[1]) || (MA8_2p > Low[2] && MA8_2p < High[2]) || //start of the wave - not farther than three bars in data history (М1)
     (MA8_3p > Low[3] && MA8_3p < High[3])) &&                                              //same
      MA5_prev_m15 > L_prev_m15 && MA5_prev_m15 < H_prev_m15 &&                             //start of the wave - on the previous bar of the higher timeframe (М15)


  //---REDUCE RISKS RELATED TO THE ABSENCE OF A CLEARLY DEFINED TREND AT THE TIME OF MARKET ENTRY-------

      //Simulate the candles direction on the lower timeframe:
      Close[2] > Open[2] &&      //upward candle direction on the 2 nd bar in history (М1)
      Close[1] > Open[1] &&      //previous candle's upward direction (М1)

      //Simulate direction of moving averages on the higher timeframe:
      MA5_cur > MA5_2p &&  MA60_cur > MA60_2p &&     //upward МАs: use moving averages with the periods of 5 and 60 (М1)

      //Simulate the hierarchy of moving averages on the lower timeframe:
      MA5_cur > MA8_cur && MA8_cur > MA13_cur &&     //form the "hierarchy" of three МАs on М1 (Fibo periods:5,8,13), this is the indirect sign of the upward movement

      //Simulate the location of the current price relative to the lower timeframes' moving averages:
      Bid > MA5_cur && Bid > MA8_cur && Bid > MA13_cur && Bid > MA60_cur && //current price exceeds МА (5,8,13,60) on М1, this is an indirect sign of the upward movement

      //Simulate the candle direction on the higher timeframe:
      C_prev_m15 > O_prev_m15 &&       //previous candle's upward direction (М15)

      //Simulate the MA direction on the higher timeframe:
      MA4_cur_m15 > MA4_2p_m15 &&     //upward МА with the period of 4 (М15)

      //Simulate the hierarchy of moving averages on the higher timeframe:
      MA4_prev_m15 > MA8_prev_m15 &&  //form the "hierarchy" of two МАs on М15 (periods 4 and 8), this is the indirect sign of the upward movement

      //Simulate the location of the current price relative to the higher timeframes' moving averages:
      Bid > MA4_cur_m15 &&            //current price exceeds МА4 (М15), this is an indirect sign of the upward movement
      Bid > MA24_cur_h1 &&            //current price exceeds МА24 (МН1), this is an indirect sign of the upward movement

      //Simulate a micro-trend inside the current candle of the lower timeframe, as well as the entry point:
      Bid > Open[0] &&               //presence of the upward movement inside the current candle (М1)

     //Simulate sufficient activity of the previous process at the higher timeframe:
     (C_prev_m15 - O_prev_m15) > (0.5*(H_prev_m15 - L_prev_m15)) &&  //share of the candle "body" exceeds 50% of the candle amplitude value (previous М15 candle)
     (H_prev_m15 - C_prev_m15) < (0.25*(H_prev_m15 - L_prev_m15)) && //correction depth limitation is less than 25% of the candle amplitude (previous М15 candle)
      H_prev_m15 > H_2p_m15 &&                                       //upward trend by local resistance levels (two М15 candles)
      O_prev_m15 < H_prev_m15 && O_prev_m15 > L_prev_m15 &&          //presence of a wick (previous М15 candle) relative to the current candle's Open price

     //Simulate sufficient activity of the previous process on the lower timeframe:
     (Close[1] - Open[1]) > (0.5*(High[1] - Low[1])) &&              //share of the candle "body" exceeds 50% of the candle amplitude value (previous М1 candle)
     (High[1] - Low[1]) > 70*Point &&                                //previous candle has an amplitude exceeding the threshold one (excluding an evident flat)
     (High[2] - Close[2]) < (0.25*(High[2] - Low[2])) &&             //correction depth limitation is less than 20% of the candle amplitude (the second candle in the М1 data history)
      High[1] > High[2] &&                                           //upward trend by local resistance levels (two М1 candles)
      Open[1] < High[1] && Open[1] > Low[1] )                        //presence of the wick (previous tfМ1 candle) relative to the current candle's Open price

        {
        //if the Buy entry algorithm conditions specified above are met, generate the Buy entry order:
        sl_buy = NormalizeDouble((Bid-StopLoss*Point),Digits);
        tp_buy = NormalizeDouble((Ask+TakeProfit*Point),Digits);

         numb = OrderSend(Symbol(),OP_BUY,Lots,Ask,3,sl_buy,tp_buy,"Reduce_risks",16384,0,Green);

         if(numb > 0)

           {
            if(OrderSelect(numb,SELECT_BY_TICKET,MODE_TRADES))
            {
               Print("Buy entry : ",OrderOpenPrice());
            }
           }
         else
            Print("Error when opening Buy order : ",GetLastError());
         return;
        }

 //--- MARKET ENTRY ALGORITHM SELL--------------------------------------------------------------------------------------------------

  if(
     //----REDUCE RISKS RELATED TO THE PRESENCE OF A STRONG VOLATILITY AT THE TIME OF MARKET ENTRY ----

     //Simulate the absence of a strong volatility in the recent history:
     ( High[1] - Low[1]) <= 200*Point &&                       ///limit the amplitude of a lower timeframe (М1)
     ( High[2] - Low[2]) <= 200*Point &&
     ( High[3] - Low[3]) <= 200*Point &&
     (H_prev_m15 - L_prev_m15) <= 300*Point &&                 //limit the amplitude of a higher timeframe (М15)
     (H_2p_m15 - L_2p_m15) <= 300*Point &&
     (H_3p_m15 - L_3p_m15) <= 300*Point &&
     (H_prev_m15 - L_3p_m15) <= 300*Point &&                   //limit the amplitude of the channel made of the higher timeframe candles (М15)
     (High[1] - Low[1]) >= (1.1*(High[2] - Low[2])) &&         //limit activity on the previous bar relative to the 2 nd bar in the quote history
     (High[1] - Low[1]) < (3.0*(High[2] - Low[2])) &&          //same


  //----REDUCE RISKS RELATED TO RESISTANCE LEVELS AT THE TIME OF MARKET ENTRY-----

     //Simulate the case when local resistance levels are broken by the current price:
       Bid < Low[1] &&           //on М1
       Bid < L_prev_m15 &&       //on М15


  //---REDUCE RISKS RELATED TO ENTERING IN THE OVERSOLD AREA AT THE TIME OF MARKET ENTRY-----

    //Simulate binding to the start of the wave to decrease the entry probability in the oversold area:
     ((MA8_prev > Low[1] && MA8_prev < High[1]) || (MA8_2p > Low[2] && MA8_2p < High[2]) || //start of the wave - not farther than three bars in data history (М1)
     (MA8_3p > Low[3] && MA8_3p < High[3])) &&                                              //same
      MA5_prev_m15 > L_prev_m15 && MA5_prev_m15 < H_prev_m15 &&                             //start of the wave - on the previous bar of the higher timeframe (М15)


  //---REDUCE RISKS RELATED TO THE ABSENCE OF A CLEARLY DEFINED TREND AT THE TIME OF MARKET ENTRY-------

      //Simulate the candles direction on the lower timeframe:
      Close[2] < Open[2] &&      //downward candle direction on the 2 nd bar in history (М1)
      Close[1] < Open[1] &&      //previous candle's downward direction (М1)

      //Simulate direction of moving averages on the lower timeframe:
      MA5_cur < MA5_2p &&  MA60_cur < MA60_2p &&     //downward МАs: use moving averages with the periods of 5 and 60 (М1)

      //Simulate the hierarchy of moving averages on the lower timeframe:
      MA5_cur < MA8_cur && MA8_cur < MA13_cur &&    //form the "hierarchy" of three МАs on М1 (Fibo periods:5,8,13), this is the indirect sign of the downward movement

      //Simulate the location of the current price relative to the lower timeframes' moving averages:
      Bid < MA5_cur && Bid < MA8_cur && Bid < MA13_cur && Bid < MA60_cur && //current price exceeds МА (5,8,13,60) on М1, this is an indirect sign of the downward movement

      //Simulate the candle direction on the higher timeframe:
      C_prev_m15 < O_prev_m15 &&      //previous candle's downward direction (М15)

      //Simulate the candle direction on the higher timeframe:
      MA4_cur_m15 < MA4_2p_m15 &&     //previous candle's downward direction 4 (М15)

      //Simulate the MA direction on the higher timeframe:
      MA4_prev_m15 < MA8_prev_m15 &&  //form the "hierarchy" of two МАs on М1 (periods 4 and 8), this is the indirect sign of the downward movement

      //Simulate the location of the current price relative to the higher timeframes' moving averages:
      Bid < MA4_cur_m15 &&            //current price is lower than МА4 (М15), this is an indirect sign of the downward movement
      Bid < MA24_cur_h1 &&            //current price is lower than МА24 (МН1), this is an indirect sign of the downward movement

      //Simulate a micro-trend inside the current candle of the lower timeframe, as well as the entry point:
      Bid < Open[0] &&                //presence of the downward movement inside the current candle (М1)

     //Simulate sufficient activity of the previous process at the higher timeframe:
     (O_prev_m15 - C_prev_m15) > (0.5*(H_prev_m15 - L_prev_m15)) &&  //share of the candle "body" exceeds 50% of the candle amplitude value (previous М15 candle)
     (C_prev_m15 - L_prev_m15) < (0.25*(H_prev_m15 - L_prev_m15)) && //correction depth limitation is less than 25% of the candle amplitude (previous М15 candle)
      L_prev_m15 < L_2p_m15 &&                                       //upward trend by local resistance levels (two М15 candles)
      O_prev_m15 < H_prev_m15 && O_prev_m15 > L_prev_m15 &&          //presence of a wick (previous М15 candle) relative to the current candle's Open price

     //Simulate sufficient activity of the previous process on the lower timeframe:
     (Open[1] - Close[1]) > (0.5*(High[1] - Low[1])) &&              //share of the candle "body" exceeds 50% of the candle amplitude value (previous М1 candle)
     (High[1] - Low[1]) > 70*Point &&                                //previous candle has an amplitude exceeding the threshold one (excluding an evident flat)
     (Close[2] - Low[2]) < (0.25*(High[2] - Low[2])) &&              //correction depth limitation is less than 20% of the candle amplitude (the second candle in the М1 data history)
      Low[1] < Low[2] &&                                             //downward trend by local resistance levels (two М1 candles)
      Open[1] < High[1] && Open[1] > Low[1] )                        //presence of the wick (previous М1 candle) relative to the current candle's Open price

        {
         //if the Sell entry algorithm conditions specified above are met, generate the Sell entry order:

         sl_sell = NormalizeDouble((Ask+StopLoss*Point),Digits);
         tp_sell = NormalizeDouble((Bid-TakeProfit*Point),Digits);

         numb = OrderSend(Symbol(),OP_SELL,Lots,Bid,3,sl_sell,tp_sell,"Reduce_risks",16384,0,Red);

         if(numb > 0)
           {

            if(OrderSelect(numb,SELECT_BY_TICKET,MODE_TRADES))
              {
               Print("Sell entry : ",OrderOpenPrice());
              }
           }
         else
            Print("Error when opening Sell order : ",GetLastError());
        }
      //--- the market entry algorithm (Buy, Sell) ends here
      return;
     }
//--- Check open orders and financial instrument symbol to prepare position closing:

   for(f=0; f < kol; f++)
     {
      if(!OrderSelect(f,SELECT_BY_POS,MODE_TRADES))
         continue;
      if(OrderType()<=OP_SELL &&   //check the order type
         OrderSymbol()==Symbol())  //check the symbol
        {

         if(OrderType()==OP_BUY) //if the order is "Buy", move to closing Buy position:
           {

   //------BUY POSITION CLOSING ALGORITHM-------------------------------------------------------------------------------------------------


     //The module to search for the price maximum inside the open position--------------

          //first, define the distance from the current point inside the open position up to the market entry point:

             shift_buy = 0;

           if(Time_cur > OrderOpenTime() && OrderOpenTime() > 0)                          //if the current time is farther than the entry point...
            { shift_buy = NormalizeDouble( ((Time_cur - OrderOpenTime() ) /60), 0 ); }    //define the distance in tfM1 bars up to the entry point

          //now, define the price maximum after the market entry:

            Max_pos = 0;

           if(Time_cur > OrderOpenTime() && shift_buy > 0)
            { Max_pos = NormalizeDouble((High[iHighest(NULL,PERIOD_M1, MODE_HIGH ,(shift_buy + 1), 0)]), Digits);}

          //the module to search for the price maximum inside the open position ends here--

          //Pass to closing the Buy position (OR logic options):

            if(
            //REDUCE RISKS RELATED TO PRICE COLLAPSES AFTER MARKET ENTRY----------------------

              (Bid < Open[0] && (Open[0] - Bid) >= 100*Point && (Time_cur - Time[0]) <= 20)           //entry conditions (in any area) in case of a price collapse (reference point - current M1 candle Open price)
                     ||
              (Bid < O_cur_m15 && (O_cur_m15 - Bid) >= 200*Point && (Time_cur - Time_cur_m15) <= 120) //exit conditions (in any area) in case of a price collapse (reference point - current M15 candle Open price)
                     ||
              ((Time_cur - OrderOpenTime()) > 60 && Close[1] < Open[1] &&
              (Open[1] - Close[1]) >= 200*Point)                                                      //exit conditions (in any area) in case of a price collapse (reference parameter - previous M1 candle amplitude)
                     ||

            //REDUCE RISKS RELATED TO UNCERTAIN PRICE MOVEMENT AMPLITUDE AFTER MARKET ENTRY--

              //Manage fixed profit (per position):
              (Bid > OrderOpenPrice() && (Bid - OrderOpenPrice()) >= 100*Point)                       //exit conditions in profit area (shadow take profit)
                     ||

             //Manage the maximum acceptable price deviation
             //from the current High after market entry:
             (shift_buy >= 1 &&                                                                          //shift no less than 1 bar from the entry point
             Time_cur > OrderOpenTime() && Max_pos > 0 && OrderOpenTime() > 0 && OrderOpenPrice() > 0 && //there is the current maximum after the entry
             Max_pos > OrderOpenPrice() &&                                                               //current maximum is in the profit area
             Bid < Max_pos &&                                                                            //there is the reversal price movement
             (Max_pos - Bid) >= 200*Point)                                                               //reverse deviation from the current maximum for market entry
                    ||

             //Manage pre-defined risk limit (per position):
              (Bid < OrderOpenPrice() && (OrderOpenPrice() - Bid) >= 200*Point)                          //exit conditions in the loss area (shadow stop loss)
                    ||

             //Manage pre-defined risk limit (for the entire deposit):
              (AccountBalance() <=  NormalizeDouble( (Depo_first*((100 - Percent_risk_depo)/100)), 0)) )  //if the risk limit is exceeded for the entire deposit during the current trading


              {

               if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,Violet))     //if the closing algorithm is processed, form the order for closing the Buy position
                  Print("Error closing Buy position ",GetLastError());    //otherwise print Buy position closing error
               return;
              }

           }
           else //otherwise, move to closing the Sell position:
           {

           //------SELL POSITION CLOSING ALGORITHM------------------------------------------------------------------------------------


            //The module to search for the price maximum inside the open position--------------

          //first, define the distance from the current point inside the open position up to the market entry point:

             shift_sell = 0;

           if(Time_cur > OrderOpenTime() && OrderOpenTime() > 0)                          //if the current time is farther than the entry point...
            { shift_sell = NormalizeDouble( ((Time_cur - OrderOpenTime() ) /60), 0 ); }   //define the distance in M1 bars up to the entry point

          //now, define the price minimum after entering the market:

            Min_pos = 0;

           if(Time_cur > OrderOpenTime() && shift_sell > 0)
           { Min_pos = NormalizeDouble( (Low[iLowest(NULL,PERIOD_M1, MODE_LOW ,(shift_sell + 1), 0)]), Digits); }

          //the module to search for the price maximum inside the open position ends here--


          //Pass to closing the open Sell position (OR logic options):

           if(
            //REDUCE RISKS RELATED TO PRICE COLLAPSES AFTER MARKET ENTRY-----------------

              (Bid > Open[0] && (Bid - Open[0]) >= 100*Point && (Time_cur - Time[0]) <= 20)          //exit conditions (in any area) during a price collapse (reference point - current M1 candle Open price)
                     ||
              (Bid > O_cur_m15 && (Bid - O_cur_m15) >= 200*Point && (Time_cur - Time_cur_m15) <= 120) //exit conditions (in any area) during a price collapse (reference point - current M15 candle Open price)
                     ||
              ((Time_cur - OrderOpenTime()) > 60 && Close[1] > Open[1] &&
              (Close[1] - Open[1]) >= 200*Point)                                                      //exit conditions in any zone during a price collapse (reference parameter - previous M1 candle amplitude)
                     ||

           //REDUCE RISKS RELATED TO UNCERTAIN PRICE MOVEMENT AMPLITUDE AFTER MARKET ENTRY--

              //Manage fixed profit (per position):
              (Bid < OrderOpenPrice() && (OrderOpenPrice()- Bid) >= 100*Point)                         //exit conditions in profit area (shadow take profit)
                     ||

             //Manage the maximum acceptable price deviation
             //from the current minimum after market entry:
             (shift_sell >= 1 &&                                                                         //shift no less than 1 bar from the entry point
             Time_cur > OrderOpenTime() && Min_pos > 0 && OrderOpenTime() > 0 && OrderOpenPrice() > 0 && //there is the current minimum after entry
             Min_pos < OrderOpenPrice() &&                                                               //current minimum is in the profit area
             Bid > Min_pos &&                                                                            //there is a reverse price movement
             (Bid - Min_pos) >= 200*Point)                                                               //reverse deviation from the current minimum to exit the market
                    ||

             //Manage pre-defined risk limit (per position):
             (Bid > OrderOpenPrice() && (Bid - OrderOpenPrice()) >= 200*Point)                            //exit conditions in the loss area (shadow stop loss)
                    ||

             //Manage pre-defined risk limit (for the entire deposit):
             (AccountBalance() <=  NormalizeDouble( (Depo_first*((100 - Percent_risk_depo)/100)), 0)) )   //if the risk limit exceeded for the entire deposit during the current trading

              {

               if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,Violet))     //if the close algorithm is executed, generate the Sell position close order
                  Print("Error closing Sell position ",GetLastError());   //otherwise, print the Sell position closing error
               return;
              }

           }
        }
     }
//---
  }
//-----------------------------------------------------------------------------------------------------------
```

Fig. 12. The sample EA considering some of the mentioned risks

The EA's code structure considers some of the risks described above. It also includes appropriate modules to reduce them.

The EA has been launched on М1 and tested on history data for the following symbols: EURUSD, USDCHF and USDJPY.

**EURUSD test results:**

![test_EURUSD](https://c.mql5.com/2/30/pic13_test_EURUSD__1.png)

Fig. 13. EURUSD test results

A positive result was obtained when testing for a year and a half.

**USDCHF test results:**

![test_USDCHF](https://c.mql5.com/2/30/pic14_test_USDCHF__1.png)

Fig. 14. USDCHF test results

A positive result was obtained when testing for a year and a half. However, the number of market entries is not sufficient.

**USDJPY test result:**

![test_USDJPY](https://c.mql5.com/2/30/pic_15_test_USDJPY__1.png)

Fig. 15. USDJPY test results

A positive result was obtained when testing for a year and a half.

Now, **evaluate the significance of each type of risk**. For this, we use the following procedure:

- the net profit gained by the EA is regarded the main parameter;
- define the significance of a certain risk type by disabling (separately) the appropriate module in the EA algorithm;
- the worse the financial result, the more important this type of risk.

Use EURUSD test results as a level, relative to which the impact of each module is defined. Have a look at the following graph:

![segnificance_types_risks](https://c.mql5.com/2/30/pic16_significance_types_risks.png)

Fig.16. Significance of risk types with the use of the previously described EA as an example

_Fig. 16 notations:_

- NetProfit is an axis where the EA's net profit is displayed;
- А \- net profit "in assembled form" (all risk reduction modules are included);
- 1 - the module for reducing risks related to high volatility at the time of a market entry is disabled;
- 2 - the module for reducing risks related to the presence of resistance levels at the time of a market entry is disabled;
- 3 - the module for reducing risks related to falling into overbought/oversold area at the time of a market entry is disabled;
- 4 - the module for reducing risks related to the absence of a clearly defined trend at the time of a market entry is disabled (the block simulating the sufficient activity of the previous process);
- 5 - the module for reducing risks related to the absence of a clearly defined trend at the time of a market entry is disabled (the block simulating the direction of candles and moving averages).

The figure shows that the maximum loss corresponds to point 4. The result here is several times worse than at other points on the graph. Hence, the most important TS component is the module, where the process activity is specified.

This means that the most significant risk (at least for the given EA) is the one associated with the absence of a clearly defined trend (insufficient process activity).

### Conclusion

We have classified the main types of risk — both related and not related to the market dynamics. We have also described the ways to reduce them and showed the algorithms of implementing some of them. Next, we have examined the code of the simple EA containing some of these algorithms. Besides, we have tested the EA on three financial instruments — EURUSD, USDCHF and USDJPY. Please note that the results have been obtained without the EA optimization. I have set the threshold price levels applied in the modules based on my personal experience.

Besides, a set of Moving Averages has been offered for usage along with the method of their inclusion into the set based on calendar cycles. I applied this method when developing the risk reduction modules and the EA as a whole. The method allowing users to somewhat decrease the probability of falling into an overbought/oversold area (based on simulating the wave-like movement) has also been offered.

The positive result has been obtained on three financial instruments, which in general confirms the correctness of the described approaches, both in the classification of risks and in the ways to reduce them, as well as in determining the structure of a trading system that takes these risks into account.

The conducted analysis has shown that the trend-following trading systems not featuring the process activity simulation are the ones subjected to the highest risk. It is this factor that, to a greater extent than the rest, is reflected in financial results.

The advantage of the above EA (and the described approach to TS construction) is the absence of traditional indicators in the input and output algorithm. Only the candle analysis and Moving Averages (with limitations) are applied. This significantly decreases the amount of optimization. Here, it is reduced to selecting threshold levels, at which the appropriate modules are triggered, as well as to selecting the MA periods from the available list of calculation periods (see the section ["Risks associated with the incorrect selection of the indicator calculation period").](https://www.mql5.com/en/articles/4233#n9)

The disadvantage of the described approach is the "over-filtering" of the entry algorithm, since there can be too many risks (and the appropriate entry filters). The result is a limited number of market entries. However, this is partly mitigated by the ability to trade simultaneously on several financial instruments (after preliminary testing).

The objective of this work has been to create a simple EA for reducing the main risks, while featuring the algorithm that can be easily grasped by novice traders. Of course, we can significantly improve the efficiency of the EA by including the algorithm for analyzing fractal levels. However, this involves searching for fractals (including in a loop, which would significantly complicate the code). I believe that the most important thing for novice analysts and traders is to grasp the structure of a trading system (in terms of market dynamics parameters) before further honing their programming skills.

Using this EA as an example, you will be able to perform experiments by disabling certain modules or changing their set depending on your logical decisions.

**NOTE! This EA is a simplified demo version not meant for real trading.**

The described approaches to reducing market risks can be useful in developing your own trading system.

**The MQL5 version of the EA from this article is available here - [Reduce\_risks](https://www.mql5.com/en/code/19726).**

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/4233](https://www.mql5.com/ru/articles/4233)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/4233.zip "Download all attachments in the single ZIP archive")

[Reduce\_risks.mq4](https://www.mql5.com/en/articles/download/4233/reduce_risks.mq4 "Download Reduce_risks.mq4")(60.14 KB)

[test\_EURUSD.png](https://www.mql5.com/en/articles/download/4233/test_eurusd.png "Download test_EURUSD.png")(147.15 KB)

[test\_USDCHF.png](https://www.mql5.com/en/articles/download/4233/test_usdchf.png "Download test_USDCHF.png")(148.42 KB)

[test\_USDJPY.png](https://www.mql5.com/en/articles/download/4233/test_usdjpy.png "Download test_USDJPY.png")(143.03 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [On Methods to Detect Overbought/Oversold Zones. Part I](https://www.mql5.com/en/articles/7782)
- [False trigger protection for Trading Robot](https://www.mql5.com/en/articles/2110)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/226246)**
(25)


![Brian Lillard](https://c.mql5.com/avatar/2023/3/640988fd-3ec4.png)

**[Brian Lillard](https://www.mql5.com/en/users/subgenius)**
\|
23 Dec 2018 at 09:53

lots of good pointers but not with regarding having a margin [check](https://www.mql5.com/en/docs/common/GetTickCount64 "MQL5 Documentation: GetTickCount64 function") to avoid a margin call.

so assuming all open positions have SL set the factors on validating margin would entail risk,

limiting risk factors of stops and their lots means cannot open a trade that causes margin call.

![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
23 Feb 2019 at 16:20

**ffoorr:**

Good ideas, but the EA cannot work because these condition will almost never be true at the same time :

These conditions are easily implemented for a trend strategy.

![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
23 Feb 2019 at 16:22

**Merrygoround:**

I tried the code snippet for "reducing risk related to high volatility at the moment of a market entry" in my own EA, and noticed that zero trades were being taken. Only then I remembered that my expert advisor was a break-out strategy, so this could never be a good match. In any case this article is full of innovative ideas and proves the author has a lot of hands-on trading experience. It made it to my all-time top ten list of Metatrader articles.

Thank you for the positive evaluation of my article.

![Aleksandr Masterskikh](https://c.mql5.com/avatar/2017/5/591837C6-87CC.jpg)

**[Aleksandr Masterskikh](https://www.mql5.com/en/users/a.masterskikh)**
\|
23 Feb 2019 at 16:25

**JuniorFurtado:**

Very good article. It is very useful for the development of my EAs.

Congratulations to the author.

Thanks for the positive evaluation of my article.

![ffoorr](https://c.mql5.com/avatar/avatar_na2.png)

**[ffoorr](https://www.mql5.com/en/users/ffoorr)**
\|
2 Dec 2020 at 19:45

**Aleksandr Masterskikh:**

These conditions are easily implemented for a trend strategy.

Yes you are right, i made an error asserting this.

But there is too many conditions : 39 conditions are linked with an "&&".

These conditions cannot be true at the same time  =>  the EA do not open any

order.

But I understand the idea which is good,

Any trader has to look for that kind of idea

![The Channel Breakout pattern](https://c.mql5.com/2/30/breakthow_channel.png)[The Channel Breakout pattern](https://www.mql5.com/en/articles/4267)

Price trends form price channels that can be observed on financial symbol charts. The breakout of the current channel is one of the strong trend reversal signals. In this article, I suggest a way to automate the process of finding such signals and see if the channel breakout pattern can be used for creating a trading strategy.

![Automatic Selection of Promising Signals](https://c.mql5.com/2/30/xf1zfo07t1b6ty_wozfke_cxp3ajzhsku9i_e6dfkszd.png)[Automatic Selection of Promising Signals](https://www.mql5.com/en/articles/3398)

The article is devoted to the analysis of trading signals for the MetaTrader 5 platform, which enable the automated execution of trading operations on subscribers' accounts. Also, the article considers the development of tools, which help search for potentially promising trading signals straight from the terminal.

![Testing patterns that arise when trading currency pair baskets. Part III](https://c.mql5.com/2/30/LOGO__2.png)[Testing patterns that arise when trading currency pair baskets. Part III](https://www.mql5.com/en/articles/4197)

In this article, we finish testing the patterns that can be detected when trading currency pair baskets. Here we present the results of testing the patterns tracking the movement of pair's currencies relative to each other.

![Risk Evaluation in the Sequence of Deals with One Asset. Continued](https://c.mql5.com/2/30/Risk_estimation.png)[Risk Evaluation in the Sequence of Deals with One Asset. Continued](https://www.mql5.com/en/articles/3973)

The article develops the ideas proposed in the previous part and considers them further. It describes the problems of yield distributions, plotting and studying statistical regularities.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/4233&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082982014346334868)

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