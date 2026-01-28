---
title: Combine Fundamental And Technical Analysis Strategies in MQL5 For Beginners
url: https://www.mql5.com/en/articles/15293
categories: Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T21:24:14.205821
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/15293&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071806552391757516)

MetaTrader 5 / Examples


### Introduction

Fundamental analysis and trend-following strategies are often seen as opposing approaches. Many traders who favor fundamental analysis believe that technical analysis is a waste of time because all necessary information is already reflected in the price. Conversely, technical analysts often view fundamental analysis as flawed because identical patterns, like a head and shoulders, can lead to different outcomes in the same market.

For any new traders, this may easily feel overwhelming because you are faced with so many choices, but which one is the best one? Which strategy will consistently leave your trading account in profits while keeping you out of the market during unfavourable conditions?

As the writer, I believe the truth lies somewhere in the middle. The purpose of this article is to explore whether we can create a stable trading strategy that combines the best aspects of both fundamental and technical analysis, and to determine if such a strategy is worth the investment of time.

We will build our Expert Advisor from the ground up using native MQL5, allowing us the flexibility to test our strategy on any market. By the end of this article, you will understand:

1. How to build your own Expert Advisors in MQL5.
2. How to combine an ensemble of technical indicators.
3. A framework for conceptualizing fundamental and technical data.

### Overview of The Trading Strategy

Our trading strategy is made up of 2 components:

1. Fundamental analysis
2. Technical analysis


Let us consider each approach in turn to understand how they complement each other, rather than trying to determine which is superior. We will begin by understanding the fundamental principles motivating our strategy.

At first glance, financial charts can appear very random and unpredictable. Financial datasets are notoriously noisy and sometimes even unstable. However, analyzing these charts from a fundamental perspective can lead to entirely different conclusions about market behavior.

### Fundamental Analysis

Fundamental analysis is rooted in an understanding of how markets operate. In our discussion, we will focus on currency pairs and build our trading algorithm to leverage our understanding of the foreign exchange markets and its major participants.

![Financial data](https://c.mql5.com/2/84/Screenshot_from_2024-07-12_12-24-42.png)

Fig 1: An example of fundamental analysis on the AUDJPY pair.

Fundamental traders often discuss support and resistance, although there are no definitive definitions for these concepts. I would like to offer one possible interpretation from a fundamental perspective.

When we look at the exchange rate between two currencies, it's easy to forget the real-world implications. For example, if the USDJPY chart is rising, it means the Japanese Yen is depreciating against the Dollar. Since 90% of the world's commodities are priced in Dollars, a rising chart indicates that Japan's exports are fetching less money abroad.

If the exchange rate continues to rise without bounds, the Japanese government would face significant challenges. Their exports would be worth very little, leading to economic strain and a decline in living standards. Families might struggle to afford necessities, and the country could face hyperinflation and dire economic conditions.

To prevent such outcomes, it is crucial for Japan's well-being that the exchange rate between the Yen and the US Dollar remains within a tolerable range. When the exchange rate becomes too high, the Japanese government is incentivized to intervene in the foreign exchange markets to protect their economy. Conversely, when the exchange rate is too low, the US government might provide support to maintain balance. This is how support and resistance can be interpreted from a fundamental perspective.

For the most part, foreign exchange rates are set by the cumulative decisions made by large financial institutions, such as retail banks, investment banks, and hedge funds.  These financial powerhouses control large volumes of money, and their decisions as a group are what drive the markets.

So from a fundamental perspective, we would not want to trade against the large corporate players, rather we would prefer to find opportunities to trade in the same direction with the dominant market players.

By analyzing the changes in price on higher timeframes, such as weekly or monthly time frames, we may gain insight into what the large institutional players believe is a fair price level for the security under observation. Therefore, we will look for trading opportunities that align with the long run change in price.

Putting it all together, our fundamental strategy, involves several steps. First, we will analyze higher time frames to understand where institutional players are likely to take the price. Once confirmed, we will identify our support and resistance levels by examining the highest and lowest prices from the previous week. We aim to trade high-probability setups, so if the price breaks above the resistance level, we will enter a buy position. Conversely, if the price breaks below the support level, we will enter a sell position.

### Technical Analysis

Now, we will define the technical analysis involved in our trading strategy. Our technical analysis focuses on identifying trading setups where our indicators align with our fundamental analysis. This means that if the trend on higher time frames is bullish, we only seek setups where our technical indicators signal us to go long. Conversely, if the trend on higher time frames is bearish, we only look for setups where our indicators align for a short trade.

The first indicator in our ensemble is the Money Flow Index (MFI). The MFI serves as a volume indicator and plays a crucial role in our strategy. We only consider trades supported by significant volume. Trades with weak or opposing volume are not within our scope. In our strategy, we interpret the MFI differently from conventional methods. We center the MFI on 50: readings below 50 indicate bearish volume, while readings above 50 indicate bullish volume.

Next, we utilize the Moving Average Convergence Divergence (MACD) indicator. Similar to our approach with the MFI, we do not use the MACD in its traditional sense. Instead, we center our MACD around 0. A MACD signal line below 0 signals a bearish trend, whereas a signal line above 0 indicates a bullish trend.

In addition to the MFI and MACD, our strategy incorporates a classic trend-following approach using a moving average. Unlike our approach with other technical indicators, we interpret the moving average in a conventional manner: if the price is below the moving average, it signals a sell opportunity; if the price is above the moving average, it signals a buy opportunity.

Furthermore, we integrate the Stochastic oscillator into our strategy, adhering to its traditional interpretation. When the oscillator provides a reading above 20, we interpret this as a buy signal. Conversely, a reading below 80 from the oscillator indicates a sell signal.

Therefore, for us to buy a security:

1. Price must close above the moving average
2. The MACD signal should be above 0
3. The MFI reading should be greater than 50
4. The stochastic oscillator should be above 20
5. Price must have appreciated over the past 3 months
6. Price levels should be above the support level.

And conversely, for us to sell a security:

1. Price must close below the moving average
2. The MACD signal should be below 0
3. The MFI reading should be less than 50
4. The stochastic oscillator should be below 80
5. Price must have depreciated over the past 3 months
6. Price levels should be beneath the resistance level.

### Let's Get Started

First we import the libraries we need. In this case, we will import the trade library to execute trade orders.

```
//+------------------------------------------------------------------+
//|                           Price Action & Trend Following.mq5     |
//|                        Copyright 2024, MetaQuotes Ltd.           |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Gamuchirai Zororo Ndawana"
#property link      "https://www.mql5.com"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
/*
   This Expert Advisor will help us implement a combination of fundamental analysis
   and trend following principles to trade financial securities with precision
   thanks to the easy to learn MQL5 Language that gives us a blend of creative
   flexibility and technical control at the same time.

   Our Fundamental strategy follows the principles outlined below:
   1)Respect price movements on higher order time frames
   2)Enter at support levels and exit on resistance levels

   Our Trend following strategy follows principles that have been proven over time:
   1)Only enter trades backed by volume
   2)Do not trade against the dominant trend, rather wait for setups to go with the bigger trend.
   3)Use an ensemble of good confirmation indicators

   Gamuchirai Zororo Ndawana
   Selebi Phikwe
   Botswana
   11:06 Thursday 11 July 2024
*/

//+------------------------------------------------------------------+
//| Include necessary libraries                                      |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>
CTrade Trade;
```

Progressing on, we now need to define inputs for our program. These inputs will control the periods of our technical indicators, desired trading lot size and other variables of that nature.

```
//+------------------------------------------------------------------+
//| Input parameters for technical indicators                        |
//+------------------------------------------------------------------+
input int stoch_percent_k = 5; // Stochastic %K
input int stoch_percent_d = 3; // Stochastic %D
input int stoch_slowing   = 3; // Stochastic Slowing

input int macd_fast_ema = 12; // MACD Fast EMA
input int macd_slow_ema = 26; // MACD Slow EMA
input int macd_sma = 9;       // MACD SMA

input int ma_period = 60; // Moving Average Period
input int mfi_period = 14; // MFI Period

input int lot_multiple = 10; // Lot size multiplier
```

Now we shall create global variables that we will use throughout our application. These variables will store our support and resistance levels, technical indicator buffers, ask and bid prices and other information of that nature.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
double ask, bid;               // Ask and Bid prices
double min_distance = 0.2;     // Minimum distance for stoploss
double min_lot_size = 0;       // Minimum lot size
double position_size;          // Actual position size

double last_week_high = 0;     // High of the previous week
double last_week_low = 0;      // Low of the previous week

string last_week_high_name = "last week high"; // Name for high level object
string last_week_low_name = "last week low";   // Name for low level object

double higher_time_frame_change = 0.0; // Change on higher time frame
string zone_location = "";             // Current zone location

int zone = 0;                // Zone indicator
string higher_time_frame_trend = ""; // Higher time frame trend
int trend = 0;               // Trend indicator

int ma_handler, stoch_handler, macd_handler, mfi_handler; // Handlers for indicators
double ma_reading[], stoch_signal_reading[], macd_signal_reading[], mfi_reading[]; // Buffers for indicator readings
```

Our goal now is to define our OnInit() handler, in this function we will initialize our technical indicators and adjust our trading lot size appropriately.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Set up handlers for technical indicators
   ma_handler = iMA(_Symbol, PERIOD_CURRENT, ma_period, 0, MODE_EMA, PRICE_CLOSE);
   macd_handler = iMACD(_Symbol, PERIOD_CURRENT, macd_fast_ema, macd_slow_ema, macd_sma, PRICE_CLOSE);
   stoch_handler = iStochastic(_Symbol, PERIOD_CURRENT, stoch_percent_k, stoch_percent_d, stoch_slowing, MODE_EMA, STO_CLOSECLOSE);
   mfi_handler = iMFI(_Symbol, PERIOD_CURRENT, mfi_period, VOLUME_TICK);

//--- Adjust lot size
   min_lot_size = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   position_size = min_lot_size * lot_multiple;

//--- Initialization done
   return(INIT_SUCCEEDED);
  }
```

Now we shall define a function that will fetch the support and resistance levels that were defined by last week's trading history.

```
//+------------------------------------------------------------------+
//| Function to get the previous week's high and low prices          |
//+------------------------------------------------------------------+
bool get_last_week_high_low(void)
  {
//--- Reset values
   last_week_high = 0;
   last_week_low = 0;

//--- Remove old levels if any
   ObjectDelete(0, last_week_high_name);
   ObjectDelete(0, last_week_low_name);

//--- Update high and low values
   last_week_high = iHigh(_Symbol, PERIOD_W1, 1);
   last_week_low = iLow(_Symbol, PERIOD_W1, 1);

//--- Mark current levels of support and resistance
   ObjectCreate(0, last_week_high_name, OBJ_HLINE, 0, 0, last_week_high);
   ObjectCreate(0, last_week_low_name, OBJ_HLINE, 0, 0, last_week_low);

//--- Check for valid values
   return((last_week_high * last_week_low) != 0);
  }
```

Moving on, we need to understand the price action happening on higher time frame. Remember that we'll look back over the past business cycle, about 3 months, to infer what institutional players have been doing in the market.

```
//+------------------------------------------------------------------+
//| Function to determine higher time frame price movement           |
//+------------------------------------------------------------------+
bool get_higher_time_frame_move(void)
  {
//--- Analyze weekly time frame
   higher_time_frame_change = iClose(_Symbol, PERIOD_CURRENT, 1) - iClose(_Symbol, PERIOD_W1, 12);

//--- Check for valid values
   return((iClose(_Symbol, PERIOD_W1, 12) * iClose(_Symbol, PERIOD_W1, 1)) != 0);
  }
```

Our next agenda will be to interpret the price action signals we have collected. Particularly we need to understand whether we are above last week's high, which we indicate as zone 1, or if we are beneath last week's low we are in zone 3 and finally if we are in between then we are in zone 2. Then we will label the trend we identified on the higher time frame, if price was appreciating on the higher time frame we will label the trend 1 otherwise we will label the trend -1.

```
//+------------------------------------------------------------------+
//| Function to interpret price action data                          |
//+------------------------------------------------------------------+
void interpet_price_action(void)
  {
//--- Determine zone location based on last week's high and low
   if(iClose(_Symbol, PERIOD_CURRENT, 0) > last_week_high)
     {
      zone = 1;
      zone_location = "We are above last week's high";
     }
   else if(iClose(_Symbol, PERIOD_CURRENT, 0) < last_week_low)
     {
      zone = 3;
      zone_location = "We are below last week's low";
     }
   else
     {
      zone = 2;
      zone_location = "We are stuck inside last week's range";
     }

//--- Determine higher time frame trend
   if(higher_time_frame_change > 0)
     {
      higher_time_frame_trend = "Higher time frames are in an up trend";
      trend = 1;
     }
   else if(higher_time_frame_change < 0)
     {
      higher_time_frame_trend = "Higher time frames are in a down trend";
      trend = -1;
     }
  }
```

Now we will need a function that will update our technical indicator values and fetch the current market data.

```
//+------------------------------------------------------------------+
//| Function to update technical indicators and fetch market data    |
//+------------------------------------------------------------------+
void update_technical_indicators(void)
  {
//--- Update market prices
   ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

//--- Copy indicator buffers
   CopyBuffer(ma_handler, 0, 1, 1, ma_reading);
   CopyBuffer(stoch_handler, 1, 1, 1, stoch_signal_reading);
   CopyBuffer(macd_handler, 1, 1, 1, macd_signal_reading);
   CopyBuffer(mfi_handler, 0, 1, 1, mfi_reading);
  }
```

This function will execute our trade entries for us, if we have permission from our bearish sentiment function we will open a sell trade. Conversely, we can only open buy positions if we have permission from our bullish sentiment function.

```
//+------------------------------------------------------------------+
//| Function to find entry points for trades                         |
//+------------------------------------------------------------------+
void find_entry(void)
  {
//--- Check for bullish sentiment
   if(bullish_sentiment())
     {
      Trade.Buy(position_size, _Symbol, ask, (last_week_low - min_distance), (last_week_high + min_distance));
     }
//--- Check for bearish sentiment
   else if(bearish_sentiment())
     {
      Trade.Sell(position_size, _Symbol, bid, (last_week_high + min_distance), (last_week_low - min_distance));
     }
  }
```

Now let us carefully define what it means for our two trading systems to align for a buy setup. Recall the conditions we defined earlier in our discussion, we want to see price closing above the moving average, our MFI indicator should be above 50, our MACD reading should be above 0, the stochastic oscillator should be above 20, the trend on higher time frame should be bullish, and we should be above the support level.

```
//+------------------------------------------------------------------+
//| Function to analyze bullish signals                              |
//+------------------------------------------------------------------+
bool bullish_sentiment(void)
  {
//--- Analyze conditions for bullish sentiment
   return((mfi_reading[0] > 50) &&
          (iClose(_Symbol, PERIOD_CURRENT, 1) > ma_reading[0]) &&
          (macd_signal_reading[0] > 0) &&
          (stoch_signal_reading[0] > 20) &&
          (trend == 1) &&
          (zone < 3));
  }
```

And the converse holds true for our sell setups.

```
//+------------------------------------------------------------------+
//| Function to analyze bearish signals                              |
//+------------------------------------------------------------------+
bool bearish_sentiment(void)
  {
//--- Analyze conditions for bearish sentiment
   return((mfi_reading[0] < 50) &&
          (iClose(_Symbol, PERIOD_CURRENT, 1) < ma_reading[0]) &&
          (macd_signal_reading[0] > 0) &&
          (stoch_signal_reading[0] < 80) &&
          (trend == -1) &&
          (zone > 1));
  }
```

Finally we need an OnTick() event handler that will ensure the flow of events in our application runs as we intended. Notice that we start by checking the time stamp, this helps us make sure that our application will only check for support and resistance levels once a week. Otherwise if we do not have any new weekly candles, then there's no point checking for the same information over and over on every tick! If everything works fine, then our Expert Advisor will then proceed to interpret the price action, update our technical indicators and then look for a trade entry.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- Check for new candle on higher time frame
   static datetime time_stamp;
   datetime current_time = iTime(_Symbol, PERIOD_W1, 0);

   if(time_stamp != current_time)
     {
      time_stamp = current_time;

      if(!get_last_week_high_low())
        {
         Print("Failed to get historical performance of ", _Symbol);
         Print("[ERROR]: ", GetLastError());
        }
      else if(!get_higher_time_frame_move())
        {
         Print("Failed to analyze historical performance of ", _Symbol);
         Print("[ERROR]: ", GetLastError());
        }
     }
   else
     {
      interpet_price_action();
      update_technical_indicators();
      if(PositionsTotal() == 0)
        {
         find_entry();
        }
      Comment("Last week high: ", last_week_high, "\nLast week low: ", last_week_low, "\nZone: ", zone_location, "\nTrend: ", higher_time_frame_trend);
     }
  }
```

![Our system in action](https://c.mql5.com/2/84/Screenshot_from_2024-07-12_11-25-04.png)

Fig 2: Our Expert Advisor trading the AUDJPY on the H1 time frame.

![Our system in action](https://c.mql5.com/2/84/Screenshot_from_2024-07-12_11-32-15.png)

Fig 3: The results of back testing our trading algorithm on H1 Data from the AUDJPY symbol over 1 month.

### Conclusion

This article illustrates the integration of fundamental and technical analysis in trading strategies. Using MQL5, we've shown how to seamlessly combine insights from these two perspectives into actionable trading instructions. By presenting a framework where fundamental and technical data complement each other, rather than compete, we empower readers to leverage both approaches effectively.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15293.zip "Download all attachments in the single ZIP archive")

[Price\_Action\_x\_Trend\_Following.mq5](https://www.mql5.com/en/articles/download/15293/price_action_x_trend_following.mq5 "Download Price_Action_x_Trend_Following.mq5")(10.73 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Reimagining Classic Strategies (Part 21): Bollinger Bands And RSI Ensemble Strategy Discovery](https://www.mql5.com/en/articles/20933)
- [Reimagining Classic Strategies (Part 20): Modern Stochastic Oscillators](https://www.mql5.com/en/articles/20530)
- [Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)
- [Reimagining Classic Strategies (Part 19): Deep Dive Into Moving Average Crossovers](https://www.mql5.com/en/articles/20488)
- [Overcoming The Limitation of Machine Learning (Part 8): Nonparametric Strategy Selection](https://www.mql5.com/en/articles/20317)
- [Overcoming The Limitation of Machine Learning (Part 7): Automatic Strategy Selection](https://www.mql5.com/en/articles/20256)
- [Self Optimizing Expert Advisors in MQL5 (Part 17): Ensemble Intelligence](https://www.mql5.com/en/articles/20238)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/470282)**
(4)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
22 Jul 2024 at 20:14

Personally I do not see any [traces](https://www.mql5.com/en/docs/matrix/matrix_characteristics/matrix_trace " MQL5 Documentation: function Trace") of fundamental analysis in the presented strategy.


![200080101](https://c.mql5.com/avatar/avatar_na2.png)

**[200080101](https://www.mql5.com/en/users/200080101)**
\|
25 Jul 2024 at 21:07

Can I have the program file?


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
25 Jul 2024 at 22:25

**200080101 [#](https://www.mql5.com/en/forum/470282#comment_54102053):**

Can I have the program file?

There is no program file.

Click the link in the first post to read the article.

![Vladislav Kozlov](https://c.mql5.com/avatar/2024/9/66ef0439-cf3d.jpg)

**[Vladislav Kozlov](https://www.mql5.com/en/users/milenkii246)**
\|
3 Mar 2025 at 04:44

as far as I understand deals are very rare 4-5 per month, maybe this strategy will be boring, but I like the logic in the article


![Hybridization of population algorithms. Sequential and parallel structures](https://c.mql5.com/2/73/Hybridization_of_population_algorithms_Series_and_parallel_circuit___LOGO.png)[Hybridization of population algorithms. Sequential and parallel structures](https://www.mql5.com/en/articles/14389)

Here we will dive into the world of hybridization of optimization algorithms by looking at three key types: strategy mixing, sequential and parallel hybridization. We will conduct a series of experiments combining and testing relevant optimization algorithms.

![Data Science and ML (Part 27): Convolutional Neural Networks (CNNs) in MetaTrader 5 Trading Bots — Are They Worth It?](https://c.mql5.com/2/84/Data_Science_and_ML_Part_27.png)[Data Science and ML (Part 27): Convolutional Neural Networks (CNNs) in MetaTrader 5 Trading Bots — Are They Worth It?](https://www.mql5.com/en/articles/15259)

Convolutional Neural Networks (CNNs) are renowned for their prowess in detecting patterns in images and videos, with applications spanning diverse fields. In this article, we explore the potential of CNNs to identify valuable patterns in financial markets and generate effective trading signals for MetaTrader 5 trading bots. Let us discover how this deep machine learning technique can be leveraged for smarter trading decisions.

![Reimagining Classic Strategies (Part II): Bollinger Bands Breakouts](https://c.mql5.com/2/85/Reimagining_Classic_Strategies_Part_II__LOGO.png)[Reimagining Classic Strategies (Part II): Bollinger Bands Breakouts](https://www.mql5.com/en/articles/15336)

This article explores a trading strategy that integrates Linear Discriminant Analysis (LDA) with Bollinger Bands, leveraging categorical zone predictions for strategic market entry signals.

![MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://c.mql5.com/2/85/MQL5_Wizard_Techniques_you_should_know_Part_28____LOGO.png)[MQL5 Wizard Techniques you should know (Part 28): GANs Revisited with a Primer on Learning Rates](https://www.mql5.com/en/articles/15349)

The Learning Rate, is a step size towards a training target in many machine learning algorithms’ training processes. We examine the impact its many schedules and formats can have on the performance of a Generative Adversarial Network, a type of neural network that we had examined in an earlier article.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15293&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071806552391757516)

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