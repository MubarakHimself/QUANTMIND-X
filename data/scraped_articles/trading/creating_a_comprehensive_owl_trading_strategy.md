---
title: Creating a comprehensive Owl trading strategy
url: https://www.mql5.com/en/articles/12026
categories: Trading, Trading Systems, Indicators
relevance_score: 6
scraped_at: 2026-01-22T20:44:31.736145
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/12026&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5051649764065399805)

MetaTrader 5 / Trading


My strategy is based on the classic trading fundamentals and the refinement of indicators that are widely used in all types of markets. Handling trends and applying proven indicators that are part of the interface of popular trading terminals make it relevant and convenient for use in any markets and exchanges. The strategy is based on the simultaneous combined use of several indicators.

Some of them have been finalized, modified, and their further work has been tested in practice over a long time. This is a comprehensive trading system, or smart strategy with a very good risk/reward ratio. All the necessary parameters are combined into an indicator called Owl Smart Levels, which displays the resulting work of all parts of the trading system interacting with one another. I named my strategy Owl because the owl is associated with wisdom. I made an attempt to combine the well-known classical instruments so that they are used together as correctly as possible.

This is a ready-made tool allowing you to follow the proposed new profitable trading strategy.

### Contents

[Introduction](https://www.mql5.com/en/articles/12026#introduction)

> [1\. Profitable trading - trading in the direction of a trend](https://www.mql5.com/en/articles/12026#para1)
>
> [1.1. How to determine a trend correctly?](https://www.mql5.com/en/articles/12026#para1.1)
>
> [1.2. What is the difference between a global and a local trend?](https://www.mql5.com/en/articles/12026#para1.2)
>
> [1.3. Trend following trading](https://www.mql5.com/en/articles/12026#para1.3)
>
> [2\. Owl strategy tools and their construction](https://www.mql5.com/en/articles/12026#para2)
>
> [2.1. Fractals](https://www.mql5.com/en/articles/12026#para2.1)
>
> [2.2. Valable ZigZag](https://www.mql5.com/en/articles/12026#para2.2)
>
> [2.3. Fibo levels](https://www.mql5.com/en/articles/12026#para2.3)
>
> [3\. Trading strategy](https://www.mql5.com/en/articles/12026#para3)
>
> [3.1. Basic principles of the Owl strategy](https://www.mql5.com/en/articles/12026#para3.1)
>
> [3.2. Stop Loss and Take Profit levels](https://www.mql5.com/en/articles/12026#para3.2)
>
> [3.3. A. Elder's Triple Screen method](https://www.mql5.com/en/articles/12026#para3.3)
>
> [3.4. Dead zone](https://www.mql5.com/en/articles/12026#para3.4)
>
> [4\. Additional tools and entry points](https://www.mql5.com/en/articles/12026#para4)
>
> [4.1. Slope channel](https://www.mql5.com/en/articles/12026#para4.1)
>
> [4.2. Fibo fan](https://www.mql5.com/en/articles/12026#para4.2)
>
> [5\. Money management](https://www.mql5.com/en/articles/12026#para5)
>
> [6\. Indicator source code](https://www.mql5.com/en/articles/12026#para6)
>
> [6.1. Full Fractals indicator](https://www.mql5.com/en/articles/12026#para6.1)
>
> [6.2. Valable ZigZag indicator](https://www.mql5.com/en/articles/12026#para6.2)

[Conclusion](https://www.mql5.com/en/articles/12026#conclusion)

### Introduction

It is well known that trading requires psychological preparation, financial calculation and a trading strategy. Psychological preparation consists of discipline, stress resistance, sober self-esteem and emotional literacy. Financial calculation makes it possible to optimally allocate available resources and not only minimize risks, but also work with them correctly, including justified increase in exposure. A trading strategy is the trader's main tool. This complex concept includes many constants and variables. These are indicators, charts of different timeframes, predicted start and end times of a trade, hedging, changing trading lot volumes, and much more.

If we remove one of these three components, we will not succeed. The article considers a new trading strategy named Owl and includes the analysis of classic indicators, their modification, trading methods and, of course, illustrative images.

### 1\. Profitable trading means trading in the direction of a trend.

The fundamental basis of stock exchange trading is the ability to correctly determine the global trend - the direction of price movement and the movement of money in a particular asset for a maximum time distance. Trading on an established global trend is the main factor for successful profitable trading.

**What is a trend?**

All changes in the market are a consequence of the movement of money, which is fixed by a change in prices. Thus, markets are driven by money, and the more money flows into the market, the more demand grows and, accordingly, the price grows forming an uptrend. Conversely, if money starts to exit the asset, demand decreases and the price falls, which is displayed by a downtrend on a chart.

#### 1.1. How to determine a trend correctly?

The sign of a trend is actually simple and consists of a _constant updating of the maximum and minimum values of the wave-like movement of the price chart_ in ascending or descending direction. This can be seen in any market and on any asset chart.

Charts move in waves, entering overbought and oversold zones or from correction to trend continuation and vice versa. If there is a constant update of price highs in overbought zones and its support levels rise after a downward rollback during corrections, we are talking about an uptrend (Fig. 1.) and the demand for the asset remains high.

![Owl Smart Level - uptrend](https://c.mql5.com/2/52/01.png)

_Fig. 1. Uptrend_

If lows are updated in oversold zones and resistance levels decrease during reverse corrections, the downtrend persists and demand for the asset continues to decline steadily (Fig. 2.).

![Owl Smart Level - downtrend](https://c.mql5.com/2/52/02.png)

_Fig. 2. Downtrend_

The correct definition of a global trend allows finding the optimal entry points to the market for making profitable trades.

#### 1.2. What is the difference between a global and a local trend?

There are two most important factors that make it possible to distinguish between local and global trends. The first factor is a significant price change allowing trader to earn on buying and then selling. The second one is the duration of the price range change over time. The systematic growth of prices and the corrective trend are most clearly visible on a scale. A tendency (a local trend) is always much shorter than a trend both in the chart segment and, accordingly, in the time range (Fig. 3).

![Owl Smart Levels - Local trend](https://c.mql5.com/2/51/g5waxy3nk-3ge121gn8.png)

_Fig. 3. Downward local trends_

If we build a visual display of the boundaries of a channel or a trend, it will be clearly seen that a tendency is always within these limits and does not break the boundaries of the channel, while a trend destroys all boundaries when changing its direction.

A sign for distinguishing between a tendency and a trend is the scale: a tendency is always several times smaller than a trend in its scale.

#### 1.3.Trend following trading

Why is it advisable to follow a trend? If money takes a certain direction, it takes some time for it to redistribute in any market. A global trend does not reverse instantly.

The difficulty in finding a moment or entry point into the market is to reliably determine a trend reversal or a rebound from a support/resistance level after a corrective trend.

Why should we always follow a trend? First of all, because **a trend is continuous**, while a counter-trend tendency is small. If, when following a trend, an entry point to the market is chosen unsuccessfully and the market turns against a trader during the correction, a reverse rebound and continuation of the price movement along the trend will allow the trader to get out of the drawdown and receive profit with proper money management and hedging.

If a stop loss is triggered at the moment of a trend reversal and a trade closes in the red, a trader can open a new trade with a larger volume to compensate for the losses and get profit.

Understanding what is happening in the market provides additional benefits for making money and you can only build on the trend here.

There is also counter-trend trading, in which profit is earned during the correction. It involves more short-term trades compared to trend-following trading. But it still involves defining a trend. We cannot call it trading against the trend in the truest sense of the word since it is carried out on reverse correction sections.

Trading with a trend, as well as correct determination of market entry and exit points allows staying at a distance from the shocks associated with the market reversal and remaining in statistical profit.

Apart from up and downtrends, the market may demonstrate the horizontal movement, or "flat". The wave-like price movement here goes in a horizontal channel of minimum and maximum values.

Understanding the principle of trend-following trading makes it possible to minimize the risks of financial losses and is the basis for building real and effective trading strategies.

### 2\. Owl strategy tools and their construction

#### 2.1. Fractals

**Fractal indicator** was developed by the famous market trading practitioner and theorist [Bill M. Williams](https://en.wikipedia.org/wiki/Bill_Williams_(trader) "https://en.wikipedia.org/wiki/Bill_Williams_(trader)") (1932–2019). _A fractal is a graphical combination of five bars or candles that shows the strength of buyers or sellers at a certain point in time._ Some bars or candles can be at the same level, but to determine the fractal, as a rule, five adjacent multi-level candles are taken, among which there should be two candles to the left and to the right from the top one (high) on an uptrend or from the bottom one (low) during a downward.

In the interface of trading platforms, [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") and 4 Fractal indicator is pre-installed and can be easily configured on any timeframe. Some traders compare a fractal with a palm, where the upper or lower position (depending on its direction - up or downwards) can be indicated by the middle (longest) finger (Fig. 4).

![Classical B. Williams' fractals](https://c.mql5.com/2/52/04.png)

_Fig. 4. Classical B. Williams' fractals_

The fractal is marked on the chart with an arrow, and along the line of fractals, you can clearly see the direction of movement and characteristic changes in the price movement, as well as determine support and resistance levels.

![Fractals in the Owl strategy](https://c.mql5.com/2/52/05.png)

_Fig. 5. Fractals in the Owl strategy_

In the Owl strategy, the definition of a fractal differs from the conventional one. In [Full Fractals indicator](https://www.mql5.com/en/market/product/93115 "Full Fractals technical indicator for MetaTrader 5"), we take five last candles for indexing a fractal to the left of a high or low candle (compared to the conventional two last candles), while two candles are taken to the right (Fig. 5).

All fractals in the Owl strategy are marked with short horizontal colored lines.

#### 2.2. Valable ZigZag

The wave nature of the market was carefully developed by [Ralph Nelson Elliott](https://en.wikipedia.org/wiki/Ralph_Nelson_Elliott "https://en.wikipedia.org/wiki/Ralph_Nelson_Elliott") (1871—1948) — an American financier who created the theory of waves, which interprets the processes in the financial markets through a system of visual wave-like patterns on price charts.

ZigZag indicator helps in understanding the wave-like movement of the market. Let's consider it in more detail. The proposed version of ZigZag according to the Owl strategy contains fewer bends making the wave larger.

![Valable ZigZag](https://c.mql5.com/2/52/06.png)

_Fig. 6. Valable ZigZag_

**ZigZag defines the main trading direction.** The operation of the indicator and its correct definition on the chart are closely related to the previous indicator - the fractal. As the fractal highs or lows move in the direction of a trend, [Valable ZigZag](https://www.mql5.com/en/market/product/93144) line does not change direction, unlike the conventional one (Fig.6).

![Valable ZigZag direction change](https://c.mql5.com/2/52/07.png)

_Fig. 7. Valable ZigZag direction change_

Thus, [Valable ZigZag](https://www.mql5.com/en/market/product/93144 "Valable ZigZag technical indicator for MetaTrader 5") combines several movements within a trend into one and does not change as long as the highs/lows of the fractals move in the direction of ZigZag. In order for ZigZag to change its direction, a candle should break through the level of the previous fractal candle in the opposite direction and fix above or below this fractal (Fig. 7).

Since ZigZag sets the direction of trading, its upward direction will mean buying only, while its downward direction will mean selling only.

#### 2.3. Fibo levels

Almost every trader has Fibo levels, Fibo grid in their arsenal as it shows very good trading results be it stock market, Forex or cryptocurrencies.

The indicator is based on a sequence of numbers discovered by the Italian scholar [Leonardo of Pisa](https://en.wikipedia.org/wiki/Fibonacci "https://en.wikipedia.org/wiki/Fibonacci") (c. 1170 — c. 1250) - one of the first major mathematicians of medieval Europe. The basic principle of the sequence is that the first two numbers in it are equal to 0 and 1, and each subsequent number is equal to the sum of the previous two.

![Owl Smart Levels - Fibo Grid](https://c.mql5.com/2/52/08.png)

_Fig. 8. Setting up Fibo Grid for a downtrend. А – Valable ZigZag direction change. 2 – upper Owl fractal or level 0 of Fibo Grid. 1 – lower Owl fractal_

The Fibo Grid is stretching from fractals indicating the beginning of a corrective movement that can go further and change the trend. Numeric levels, such as 0; 23.6; 38.2; 50; 61.2; 161.8 correspond to the mathematical sequence discovered by the Italian mathematician. The number 161.8 indicates the level of the golden ratio and indicates the maximum movement in a certain direction and a possible imminent reverse correction. If the price reaches this level, then we can take profit and adjust the Fibo Grid again after a while.

The peculiarity of using the Fibo Grid in the Owl strategy is that the indicator is used in close connection with Valable ZigZag and Owl Fractal indicators.

It is necessary to set up the Fibo levels grid between two opposite fractals located to the right of the trend change point according to the Valable ZigZag indicator. If it is directed upwards (buy), then level 0 is set at the point of the extreme lower fractal and level 100 is set at the point of the extreme upper fractal. If ZigZag indicates the downward direction, Fibo Grid stretches from the extreme upper fractal with level 0 to the extreme lower fractal with level 100.

We can enter the market at levels 38.2; 50; 61.2 and if the trend change is correct, there will still be enough movement to the level of 161.8 on the chart to make a profit.

If the trend is not strongly-pronounced and tends to flat, we can enter the market at the level of 38.2; 50; realizing at the same time that the price may not reach the level of 161.8 and trying to take profits at other, earlier Fibo Grid levels. On a strongly-pronounced trend, the level of 61.8 will be the best entry point, while the market may simply not reach the level of 38.2, and a number of possible profitable trades will be missed.

### 3\. Trading strategy

#### 3.1. Basic principles of the Owl Strategy

The Owl strategy is based on the simultaneous use of several indicators: ZigZag, Fractal and Fibo Grid. At the same time, the Fractal indicator has been improved and contains eight candles instead of five, and ZigZag contains fewer bends, since it serves only for determining the direction of the trend and is called Valable ZigZag in the strategy.

A trading signal appears only when it is possible to construct Fibo levels at a certain location of neighboring fractals on the chart relative to Valable ZigZag.

![Owl Smart Levels - Opening an order](https://c.mql5.com/2/52/09.png)

_Fig. 9. Fibo Grid trading. А – Valable ZigZag direction change. 2 – upper Owl fractal or level 0 of Fibo Grid. 1 – lower Owl fractal_

_A market entry is possible from one of the levels 38.2, 50 or 61.8._ The entry point should be chosen relative to the dynamics of the market, which in any case should be determined before entering it. If a good market movement has begun, we need to enter from 61.2, because if we enter at 38.2, a number of profitable trades may be missed. If a trend movement is not strong, it is better to enter from 38.2. When trading commodities and stocks, 50% of the movement works very well (Fig. 9).

#### 3.2. Stop Loss and Take Profit levels

Stop Loss should be placed 2-5 points beyond the zero line of the Fibo Grid, and the Take Profit level is 161.8. You can close the order not completely, but at 50% of 161.8. In other words, close 50% of the position or half of the volume of orders, while the rest should be closed when the direction of the Valable ZigZag changes (Fig. 9).

At the level of 100% Fibo Grid, orders should be moved to a breakeven position: the Stop Loss level should be set to the opening level +1 point to cover commissions.

#### 3.3. A. Elder's Triple Screen method

The Triple Screen method of [Alexander Elder](https://www.mql5.com/go?link=https://www.investopedia.com/articles/trading/03/040903.asp "https://www.investopedia.com/articles/trading/03/040903.asp") (born in 1951) is based on the assumption that, in order to make trading decisions, charts on three timeframes are simultaneously studied: the main one trading takes place on, and two timeframes, each four times older than the previous one.

For example, if trading is carried out on M15 chart, then H1 and H4 timeframes are used to confirm the direction of the market movement.

The direction of the main trend determined by [Valable ZigZag](https://www.mql5.com/en/market/product/93144 "Valable ZigZag technical indicator for MetaTrader 5") is important. If the direction is the same on all timeframes, this means that it is possible to search for the main entry signal with the construction of a Fibo Grid.

![Owl Smart Levels - Elder's Triple Screen system](https://c.mql5.com/2/52/10.png)

_Fig. 10. Displaying the direction of price movement in three timeframes on the indicator_

If Valable ZigZag and M15 timeframe show upward movement, H1 shows the upward direction and H4 shows the same vector - this means that the probability of maintaining this movement for some time is maximum, the market is going up and you can take profit by following it. To do this, you need to find fractal 1, fractal 2, located to the right of the change in direction of movement, set the Fibo Grid and enter the market.

Elder's technique is described by the author himself in more detail, while Owl applies only its main principle.

#### 3.4. Dead zone

A dead zone is formed when the movement on H1 is different from that on H4. One timeframe may show upward movement, while another shows downward movement.

This is a market indecision zone. So it is rather risky to enter both for buying and selling at this time. It is better to wait until the market leaves this zone and shows a clear direction. After that, we can look for entry points into the market.

[Owl Smart Levels indicator](https://www.mql5.com/en/market/product/93396 "Owl Smart Levels technical indicator for MetaTrader 5") shows the dead zone in red, warning that it is undesirable to trade in it.

![Owl Smart Levels - Dead zone](https://c.mql5.com/2/52/11.png)

_Fig. 11. Dead zone_

### 4\. Additional tools and entry points

#### 4.1. Slope channel

The Owl strategy assumes the presence of such an additional tool for working with the price chart as a slope channel. Just like Valable ZigZag, it is built on fractals in the direction of a trend. To build the line of the upper border of the channel, it is necessary to take the points of two vertices of the upper fractals, and in order to lay a parallel line of the lower border, the third point of the lower fractal will be enough.

Thus, if there are two upper fractals, which are located to the right of the ZigZag trend change point, and there is a third, lower fractal, which is located between them, you can safely build a channel. This upward channel can be used to look for additional buy entry points (Fig. 12).

When the chart moves downwards with the corresponding direction of the Valable ZigZag indicator, it is necessary to define two lower fractals in order to draw the lower boundary line, and one upper fractal in order to draw the upper channel boundary line parallel to the lower one.

#### 4.2. Fibo Fan

After the channel is built, we can use another additional tool used along with the channel - Fibo Fan. Keep in mind that in the Owl strategy, if the channel is not used, Fibo Fan is also not used - these charting tools are built one relative to the other.

Fibo Fan consists of several rays, fan-shaped from the main ray to the right, and, like the Grid, have values corresponding to the sequence of Fibo numbers - 38.2; 50; 61.8.

Fibo Fan is, in a certain sense, a leading indicator, and gives an early reversal signal. Its rays represent additional support levels relative to the inclined channel. If the price "breaks through" the fan and moves further away from it, then you should not open deals for a reverse rollback from the channel line. In this case, we can already trade for a reversal from the main trend, since, as a rule, when the fan breaks, this finally confirms the breakout of the corresponding channel boundary.

![Owl Smart Levels - Slope channel and Fan tools](https://c.mql5.com/2/52/12.png)

_Fig. 12. Slope channel and Fibo Fan (yellow)_

If the price "bounces" from the fan, then it becomes possible to enter the market at a trend reversal or during a trend earlier than most traders using only a slope channel or even a Fibo Grid even before ZigZag reverses.

### 5\. Money management

The basis for entering any market is thoughtful and planned money management. This is literally more important than choosing a trading strategy.

It is necessary to determine the optimal size of the deposit and the volume of the trading lot in order to fulfill **the main task of the trader - preserving the deposit**. If the task of preserving the deposit is solved and a trader earns a small profit, this means that the trader is trading successfully. You should not strive to achieve an overestimated percentage of profitable trades, such as 70-90%, and increase the corresponding risks in your trading strategy. Many large traders consistently earn on the market with a percentage of profitable trades not much more than 50%. If the strategy is effective and competent, it will bring profit. You should not intervene and reduce the volume of transactions after losses, as well as try to catch up by unnecessarily increasing the volume of lots.

Initially, it is sufficient to lay down minimal risks for a series of 10 losing trades and for an amount not exceeding 15% of the deposit. The probability would seem small, but we should never forget about possible market crashes, sudden corrections or sharp gap-like growth. Many traders do not take into account the possibility of a long series of losing trades, while this is precisely one of the main reasons why they eventually lose their entire deposit. Therefore, the amount of the deposit should be sufficient to overcome the drawdown. This does not mean that the deposit should be huge. Instead, it means that there is a certain ratio of the size of the deposit and the volume of the trading lot.

To get out of the drawdown, you can try to gradually slightly increase the order sizes when the market movement becomes clear, because the probability that the next trade will be profitable increases with each losing trade.

![Owl Smart Levels - Money management](https://c.mql5.com/2/52/13.png)

_Fig. 13. Money management rules_

_The Owl strategy has a rate of return at least 2 times higher than the rate of objective losses._ Therefore, even with a quantitative ratio of profitable trades to unprofitable trades as 1:3, it allows the trader to remain profitable and preserve the deposit. At the same time, it is important to understand that the risk management system still remains primary for those who enter the market with the Owl strategy.

### 6\. Indicator source code

Traders like simple trading systems that are based on the fundamental principles of market movement. They work more or less reliably even today, and are best understood, since they were created before the advent of personal computers which makes them easy to calculate.

#### 6.1. Full Fractals indicator

Since the principles of constructing the Fractals indicator are not complicated, the Full Fractals indicator should not be very complicated. The indicator code is based on just one function, which looks something like this:

```
bool IsFractal(int _i, bool _type)
  {
   if(_type)
     {
      double low = iLow(_Symbol,PERIOD_CURRENT,_i);
      for(int j=1; j<=FrBarsLeft; j++)
         if(iLow(_Symbol,PERIOD_CURRENT,_i+j) < low)
            return false;
      for(int j=1; j<=FrBarsRight; j++)
         if(iLow(_Symbol,PERIOD_CURRENT,_i-j) < low)
            return false;
      return true;
     }
   else
     {
      double high = iHigh(_Symbol,PERIOD_CURRENT,_i);
      for(int j=1; j<=FrBarsLeft; j++)
         if(iHigh(_Symbol,PERIOD_CURRENT,_i+j) > high)
            return false;
      for(int j=1; j<=FrBarsRight; j++)
         if(iHigh(_Symbol,PERIOD_CURRENT,_i-j) > high)
            return false;
      return true;
     }
   return false;
  }
```

Only two parameters are passed to the function:

- bar index on the chart, on which we are looking for a fractal,
- fractal direction for the check – Up or Down.

If a fractal is found, we need to fill the array buffer with the High or Low value of a specific candle we are checking.

```
for(int i=start; i>FrBarsRight && !IsStopped(); i--) {
      if (IsFractal(i,false)) frUp[i] = iHigh(_Symbol,PERIOD_CURRENT,i);
      if (IsFractal(i,true)) frDown[i] = iLow(_Symbol,PERIOD_CURRENT,i);
   }
```

Find the full [source code](https://www.mql5.com/en/articles/download/12026/full_fractals_base.mq5) in the attachment below.

#### 6.2. Valable ZigZag indicator

Since the Valable ZigZag indicator is based on the Full Fractals indicator, its development was a kind of continuation of the previous indicator. The principle of the indicator remains just as simple and consists of only a couple of functions.

Let's consider the first of them named Logic.

```
void Logic(int i)
  {
   if(IsFractal(i,false))
     {
      frUp[i] = iHigh(_Symbol,PERIOD_CURRENT,i);
     }
   if(IsFractal(i,true))
     {
      frDown[i] = iLow(_Symbol,PERIOD_CURRENT,i);
     }
   if(direction == 0)
     {
      if(l_level_down > 0 && iClose(_Symbol,PERIOD_CURRENT,i) < l_level_down)
        {
         gzz[i] = iLow(_Symbol,PERIOD_CURRENT,i);
         l_zz_low = gzz[i];
         direction = 1;
        }
      if(frUp[i] == iHigh(_Symbol,PERIOD_CURRENT,i) && l_zz_high < frUp[i])
        {
         gzz[i] = iHigh(_Symbol,PERIOD_CURRENT,i);
         l_zz_high = gzz[i];
         ClearTheExtraValue(i,direction);
        }
     }
   else
     {
      if(l_level_up > 0 && iClose(_Symbol,PERIOD_CURRENT,i) > l_level_up)
        {
         gzz[i] = iHigh(_Symbol,PERIOD_CURRENT,i);
         l_zz_high = gzz[i];
         direction = 0;
        }
      if(frDown[i] == iLow(_Symbol,PERIOD_CURRENT,i) && l_zz_low > frDown[i])
        {
         gzz[i] = iLow(_Symbol,PERIOD_CURRENT,i);
         l_zz_low = gzz[i];
         ClearTheExtraValue(i,direction);
        }
     }
   if(frUp[i] == iHigh(_Symbol,PERIOD_CURRENT,i))
     {
      l_level_up = frUp[i];
     }
   if(frDown[i] == iLow(_Symbol,PERIOD_CURRENT,i))
     {
      l_level_down = frDown[i];
     }
  }
```

Only one parameter is passed to it - _i_. It is responsible for the number of the candle on the chart the calculation is being made for. The calculation should be carried out from right to left, that is, from the beginning of history to the current moment in time.

The first part of the code is known - this is the calculation of fractals. Their values are written to two separate arrays ( _frUp_ and _frDown_), which are not displayed on the chart but only participate in further calculations.

Next comes the code that does the basic calculations. It is divided into two parts: calculation when ZigZag is directed upwards ( _direction_ =0) and when ZigZag is directed down ( _direction_ =1).

These functions contain the entire construction logic, which, in turn, consists of two main checks:

1. Reversal check. An upward reversal occurs at the moment the candle closes above the last up fractal when the ZigZag is down, and a ZigZag downward reversal takes place if the candle was closed below the last down fractal.
2. Checking for the extension of the value of the ZigZag extreme point in the direction of a trend. Here it is worth using one more additional function ( _ClearTheExtraValue_), which removes unnecessary values from the indicator buffer in order to remove unnecessary bend points from ZigZag.

These are the main functions that form the basis of the Owl strategy, which is based on proven indicators.

### Conclusion

The Owl strategy development involved fragments of the trading system of the world-famous trader Bill Williams, elements of the theory of wave analysis by the major trading analyst Ralph Elliott. Besides, we used very popular and time-proven indicators: Grid and Fibo Fan, as well as ZigZag. The strategy also includes Alexander Elder's Triple Screen method.

The refinement of classic indicators carried out in Owl, does not change their essence, but dynamically adapts to changing market conditions at the present stage. Just as not all classic candlestick patterns work (at least not in all markets), some indicators can work with a lag, making their use in modern trading less meaningful.

The Owl strategy is based on the principle of simultaneous combined use of several indicators. Some of them have been finalized, modified, and their further work has been tested in practice over a long time. This is a comprehensive trading system, or smart strategy, with a very good risk/reward ratio.

All the necessary parameters are combined into an indicator called [Owl Smart Levels](https://www.mql5.com/en/market/product/93396), which displays the resulting work of all parts of the trading system interacting with one another. This is a ready-made tool allowing you to follow the proposed new profitable trading strategy.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12026](https://www.mql5.com/ru/articles/12026)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12026.zip "Download all attachments in the single ZIP archive")

[Full\_Fractals\_Base.mq5](https://www.mql5.com/en/articles/download/12026/full_fractals_base.mq5 "Download Full_Fractals_Base.mq5")(3.71 KB)

[Valable\_ZigZag\_Base.mq5](https://www.mql5.com/en/articles/download/12026/valable_zigzag_base.mq5 "Download Valable_ZigZag_Base.mq5")(10.08 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/444512)**
(8)


![Mikhail Dovbakh](https://c.mql5.com/avatar/2010/1/4B4A6B44-E70D.jpg)

**[Mikhail Dovbakh](https://www.mql5.com/en/users/avatara)**
\|
11 Mar 2023 at 01:52

![](https://c.mql5.com/3/402/GEndGpg2ep.png)

?

![Dmitrii Kim](https://c.mql5.com/avatar/avatar_na2.png)

**[Dmitrii Kim](https://www.mql5.com/en/users/kimdmitri)**
\|
22 Oct 2023 at 13:47

Can this strategy be applied on older timeframes? For example, the main H4, then the older ones should be taken H16 and H64 (can they be added to MT4 somehow?). Or can I use H4-D-W?


![Sergey Ermolov](https://c.mql5.com/avatar/2024/4/662acc47-70b1.jpg)

**[Sergey Ermolov](https://www.mql5.com/en/users/dj_ermoloff)**
\|
22 Oct 2023 at 18:39

**Dmitrii Kim [#](https://www.mql5.com/ru/forum/441498#comment_50075786):**

Can this strategy be applied on older timeframes? For example, the main H4, then the older ones should be taken H16 and H64 (can they be added to MT4 somehow?). Or can I use H4-D-W?

Yes, you can use H4-D1-W1

![Andrii Vashchyshyn](https://c.mql5.com/avatar/avatar_na2.png)

**[Andrii Vashchyshyn](https://www.mql5.com/en/users/andrew9301)**
\|
28 Jan 2024 at 20:31

**Mikhail Dovbakh [#](https://www.mql5.com/ru/forum/441498#comment_45528070):**

?

Hi. If the issue is that the top two peaks are connected, I have noticed the same on the standard Zigzag. Does anyone know how to fix it?

![guicai liu](https://c.mql5.com/avatar/2022/12/63AA85F8-269F.png)

**[guicai liu](https://www.mql5.com/en/users/gcliu14)**
\|
2 Mar 2024 at 13:45

Hi blogger, I am getting this when I call your Valable\_ZigZag\_Base indicator to create an EA, can you help me with this error? The zigzag indicator that comes with MT5 is working fine for me when I call it. [![](https://c.mql5.com/3/430/3152717023430__1.png)](https://c.mql5.com/3/430/3152717023430.png "https://c.mql5.com/3/430/3152717023430.png")

![Neural networks made easy (Part 35): Intrinsic Curiosity Module](https://c.mql5.com/2/50/Neural_Networks_Made_035_avatar.png)[Neural networks made easy (Part 35): Intrinsic Curiosity Module](https://www.mql5.com/en/articles/11833)

We continue to study reinforcement learning algorithms. All the algorithms we have considered so far required the creation of a reward policy to enable the agent to evaluate each of its actions at each transition from one system state to another. However, this approach is rather artificial. In practice, there is some time lag between an action and a reward. In this article, we will get acquainted with a model training algorithm which can work with various time delays from the action to the reward.

![Category Theory in MQL5 (Part 4): Spans, Experiments, and Compositions](https://c.mql5.com/2/52/Category-Theory-p4-avatar.png)[Category Theory in MQL5 (Part 4): Spans, Experiments, and Compositions](https://www.mql5.com/en/articles/12394)

Category Theory is a diverse and expanding branch of Mathematics which as of yet is relatively uncovered in the MQL5 community. These series of articles look to introduce and examine some of its concepts with the overall goal of establishing an open library that provides insight while hopefully furthering the use of this remarkable field in Traders' strategy development.

![Canvas based indicators: Filling channels with transparency](https://c.mql5.com/2/52/filling-channels-avatar.png)[Canvas based indicators: Filling channels with transparency](https://www.mql5.com/en/articles/12357)

In this article I'll introduce a method for creating custom indicators whose drawings are made using the class CCanvas from standard library and see charts properties for coordinates conversion. I'll approach specially indicators which need to fill the area between two lines using transparency.

![Creating an EA that works automatically (Part 08): OnTradeTransaction](https://c.mql5.com/2/50/aprendendo_construindo_008_avatar.png)[Creating an EA that works automatically (Part 08): OnTradeTransaction](https://www.mql5.com/en/articles/11248)

In this article, we will see how to use the event handling system to quickly and efficiently process issues related to the order system. With this system the EA will work faster, so that it will not have to constantly search for the required data.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/12026&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5051649764065399805)

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