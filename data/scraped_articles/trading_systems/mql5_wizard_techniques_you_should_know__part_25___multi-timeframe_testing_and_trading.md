---
title: MQL5 Wizard Techniques you should know (Part 25): Multi-Timeframe Testing and Trading
url: https://www.mql5.com/en/articles/15185
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:13:38.551581
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/15185&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070114567205359665)

MetaTrader 5 / Trading systems


### Introduction

In our last article we looked at [Pythagorean Means](https://en.wikipedia.org/wiki/Pythagorean_means#:~:text=In%20mathematics%2C%20the%20three%20classical,importance%20in%20geometry%20and%20music. "https://en.wikipedia.org/wiki/Pythagorean_means#:~:text=In%20mathematics%2C%20the%20three%20classical,importance%20in%20geometry%20and%20music.")which are a group of moving averages of which some are quite novel and not common enough despite their potential in benefiting some traders as we hinted in the test reports. These Pythagorean Means were represented in a semicircle diagram that summarized what each mean value was when presented with two unequal values that added up to the diameter of the semicircle. Among the chord values in the semicircle that was not touched on in the article was the value indicated as Q that represented the [quadratic mean](https://en.wikipedia.org/wiki/Root_mean_square "https://en.wikipedia.org/wiki/Root_mean_square")of the two values a and b.

The quadratic mean (QM) is also commonly referred to as the root-mean-square and as a mean it tends to be weighted more towards the larger values in the set whose mean is sought, which is unlike the geometric and harmonic mean’s we looked at in the previous article. It is also like the geometric mean only returns positive values, so the sampled set whose mean is sought needs to have only positive values. The title of this article though is implementing multi-timeframe strategies in wizard assembled Expert Advisors, so QM is simply going to be the tool that we use to show how multiple timeframes can be tested on in a wizard-built Expert.

So why is testing on multiple timeframes tricky with wizard-built Experts? Well in my opinion this is because the customization of this should happen for each added signal in the wizard assembly space and this is often missed. The customization of the Symbol-names and Time-frames for a wizard assembled Expert Advisor can be done during the wizard signal selection steps, but often most people assume once you pick a signal then you get to pick the symbol and period, but this should not be the case with wizard assembly. In [this](https://www.mql5.com/en/articles/14806) prior article, I showed various ways of accommodating multiple symbol trading in wizard assembled Experts by modifying assembled source code. One obvious and primary approach I did not share, performing multiple attachments of signals, where each signal attachment would be for a specific symbol. These multiple attachments happen despite attaching the very same signal. The signal customization in assigning the symbol should be done in these steps indicated below:

![n1](https://c.mql5.com/2/82/name_1.png)

![n2](https://c.mql5.com/2/82/name_2.png)

Similarly, multiple timeframes can be customized for each Expert Advisor within a signal as an input parameter (sample code of this is attached). However, the preferred approach could be assigning each required timeframe to a signal and independently attaching this signal in the wizard assembly. Steps in customizing timeframes for each signal are very similar to what we've shared above with multi symbol trading.

![f1](https://c.mql5.com/2/82/frame_1.png)

![f2](https://c.mql5.com/2/82/frame_2.png)

Multiple time frame strategies can be a boon in instances such as when looking for divergence opportunities between the tracked time frames such as a short-term bearish trend in the faster (or shorter time frame) could be a major signal if this is taking place in the backdrop of a major bullish trend in the larger time frame. Alternatively, multiple time frame strategies could evolve around confirmation setups between the two or multiple tested time frames. These and other reasons are why we will delve into multi-time frame strategies implementation in wizard assembled Expert Advisors.

Likewise, the already looked at multi symbol trading in this [article](https://www.mql5.com/en/articles/14806), certainly extends a trader's possibilities. For this article though, we will consider 3 pairs of currencies that have the potential to cancel each other out in an arbitrage setup, but we'll not be looking to develop or test hedged arbitrage systems but rather we'll try to explore common moving average settings, if any, when all pairs are being traded by one Expert Advisor.

Multi timeframe trading and multi symbol trading are very interesting and yet because of how wizard assembled Experts are coded, the inherent design is for parallel or multi signals being added at wizard assembly with each added signal catering for a specific symbol or timeframe. The main reason for this in my opinion stems from the initialization phases of the wizard assembled Expert Advisors. There are 4 initialization phases of the ‘CExpertBase’ class, which serves as the anchor class not just for signals but also money management and trailing stop management. These phases are starting Initialization, setting parameter values, checking parameter values, and finally finishing Initialization.

Specifically, in the parameter setting phase is when an instance of the Expert signal class gets assigned its trade symbol and timeframe. According to the [documentation,](https://www.mql5.com/en/docs/standardlibrary/expertclasses/expertbaseclasses/cexpertbase/cexpertbaseinitphase)though, this can also happen in the parameter checking phase. I think the parameter setting phase is called the tuning phase, but I have failed to find documentation to that effect. So, because a trade symbol determines a lot of the parameters the signal class would rely on like what OHLC prices to load, the point size, etc. this is something that by design is set once on Initialization. Similarly, the timeframe used to retrieve the OHLC price buffers needs to be predetermined and this also happens during Initialization, and never after.

These design constraints therefore mean that the intended way of using multi symbols and timeframes is where an instance of a signal class gets pre-assigned during the wizard assembly such that these symbol and timeframe values act as constant parameters.

There is still a caveat though when it comes to trading multi symbols under this pre-assigning regime within the wizard. As we saw in the multi-currency [article](https://www.mql5.com/en/articles/14806), we still need to create an array instance of the ‘CExpert’ class and re-assign our intended trade symbols to each instance of the expert class in this array. It appears, what the wizard symbol assignment does is only determine what symbol will be used in fetching OHLC price data under the ‘USED\_SERIES’ enumerations. However, the actual symbol that gets traded after being initialized under the ‘m\_symbol’ Symbol-Info class, is the symbol that is assigned on the very first step of the wizard assembly, before any signal is selected.

![s1](https://c.mql5.com/2/82/step_1.png)

This constraint on multi-symbol trading is not reflected in multi-time frame trading, as the signal assignment of a time frame during wizard assembly is enough to ensure that each assigned time frame will be used in signal processing once the Expert Advisor is trading.

### Implications of Tuning Phase in Initialization.

Because we only get to set customizations for trade symbols and timeframes at initialization, this constraint means we are unable to explore or optimize for tradeable symbols or ideal timeframes once an Expert Advisor has been assembled. There are workarounds to this where the sought trade symbol or timeframe can serve as an input parameter to a custom signal class. Such input parameters would take the form of a string and the enumeration-for-time frames, respectively. They naturally would also have default values, and in general the assigning of custom timeframe values should be more straight forward since the options are from the already set ENUM\_TIMEFRAMES enumeration. However, when it comes to symbol names for trading then an extra step or function should be used to ensure that the typed symbol name is actually valid, is listed in the market watch, and is tradeable.

### Quadratic Moving Averages

Quadratic Averages (QA) which are also referred to as [root mean squares](https://en.m.wikipedia.org/wiki/Root_mean_square "https://en.m.wikipedia.org/wiki/Root_mean_square")are simply the square root of the average of all squared values in a set. Like most non-arithmetic mean averages, they do have a bias or preference weighting and this, unlike the Geometric mean and the Harmonic mean, is towards the larger values in the set. In the semicircle diagram that was shown in this previous [article](https://www.mql5.com/en/articles/15135), where the different types of means for two u equal values a and b were represented to scale within the semicircle, the quadratic mean was also indicated, and it was equivalent to the length of the chord marked Q.

![](https://c.mql5.com/2/82/4748754854977.png)

Where

- n is the number of values in the set
- x are the values in the set at their respective index

We use the QM to develop and test a custom signal that trades on multiple timeframes and another custom signal that also trades across multiple symbols. In the previous article because of the bias or preference weighting towards smaller values in a set for the geometric mean and harmonic mean, we developed mirror averages that had biases towards larger values. The implementation of QM in MQL5 is given by the code below:

```
//+------------------------------------------------------------------+
//| Quadratic Mean                                                   |
//+------------------------------------------------------------------+
double CSignalQM::QM(int Index, int Mask = 8)
{  vector _f;
   _f.CopyRates(m_symbol.Name(), m_period, Mask, Index, m_length);
   vector _p = _f*_f;
   double _m = _p.Mean();
   return(sqrt(_m));
}
```

Before we dig in and start to explore possible applications, it would be prudent to do something similar for the QM since it leans more towards the larger values by having a mirror mean labelled QM’ that is weighted more towards the smaller values. Once again, this dichotomy of smaller value weighting vs larger value weighting can allow us to derive low price buffers and high price buffers respectively.

In the prior article, we used these buffer pairs to generate envelope (Bollinger Bands) and divergence signals. Besides these 2 applications we can generate alternative applications if we were to for instance, derive OSMA signals, however since the main purpose of this article is not QM or Averages introduction per se but the trading of multiple symbols and the trading on multiple time frames, we will stick to the applications already introduced in the prior article. As always, the reader can independently modify the attached code to explore other implementation avenues of QM.

The QM mirror QM' can be given by a formula similar to what we did share in the last article:

```
//+------------------------------------------------------------------+
//| Inverse Quadratic Mean                                           |
//+------------------------------------------------------------------+
double CSignalQM::QM_(int Index, int Mask = 8)
{  double _am = AM(Index, Mask);
   double _qm = QM(Index, Mask);
   return(_am + (_am - _qm));
}
```

The central thesis of this formula is that the arithmetic mean provides the most impartial and fair mean, implying that any distance from this mean off an alternative mean can be ‘mirrored’ if we take this raw arithmetic mean as a mirror line. The MQL5 source code that implements this is therefore identical to what we already shared and will not be shared here. It is attached below.

Once again, our objective is showcasing multi-symbol and multi-timeframe trading. Two types of Expert Advisors. It follows therefore that we could have the QM as a Bollinger Bands signal for one and the other using the Divergence signal, with both implementations already introduced in the last article.

As we have highlighted above, the MQL5 Wizard assembled Expert Advisors, were designed to allow multi-timeframe and multi-symbol testing (and trading) on a per signal basis. Each used signal by the Expert Advisor gets its custom symbol and or time frame before it is added in the wizard. This implies we could have tested for multi-symbols and multi-time frames on any signal, however since we just looked at novel averages in the last article, we are continuing in that vein by covering quadratic means.

### Developing Custom Signal Classes with QMA

We will not perform tests for the multi-currency symbol Expert Advisor, since what is assembled in the wizard requires modifications that we have already addressed in this [article](https://www.mql5.com/en/articles/14806). As mentioned above, testing for multi-symbol and multi-timeframe Expert Advisors can be done with the same signal file, since these symbol and time frame values are assigned per signal. The processing of the long and short conditions will follow the overall approach we had with generating signals for the geometric mean in the previous article. We are making some changes though since in the previous article we were dealing with only one signal, we could afford to look for sharp entry points such as the upper bands price cross we sought when looking for a bearish signal or the lower bands cross that sought a bullish signal.

In this article since we are handling multiple signals in parallel, it is unlikely they all concurrently have a cross at the same time so instead of looking for such sharp entry points we use the bands as a probability gauge for bullishness and bearishness by measuring how far above (for bearishness) or below (for bullishness) the close price is relative to the two bands’ baseline. So, the further above the baseline the current close price is the higher the weight for bearishness, similarly the lower below the baseline the close price is the higher the bullish weight.

So, each of the used signals, which ran at different time frames, provide a condition for either bullishness or bearishness depending on where the positions of their respective close prices. These conditions are then combined using the individual optimized signal weights to arrive at a single condition, via weighted average. It is this average that determines whether current positions can be closed and whether to open any new positions. The code for our modified long and short conditions is shared below:

```
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalQM::LongCondition(void)
{  int result = 0;
   m_close.Refresh(-1);
   if(m_close.GetData(StartIndex()) > m_close.GetData(StartIndex() + 1) && m_close.GetData(StartIndex()) > BandsDn(StartIndex()) && m_close.GetData(StartIndex() + 1) < BandsDn(StartIndex() + 1))
   {  result = int(round(100.0 * ((m_close.GetData(StartIndex()) - m_close.GetData(StartIndex()+1))/(fabs(m_close.GetData(StartIndex()) - m_close.GetData(StartIndex()+1)) + fabs(BandsUp(StartIndex()) - BandsDn(StartIndex()))))));
   }
   return(result);
}
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalQM::ShortCondition(void)
{  int result = 0;
   m_close.Refresh(-1);
   if(m_close.GetData(StartIndex()) < m_close.GetData(StartIndex() + 1) && m_close.GetData(StartIndex()) < BandsUp(StartIndex()) && m_close.GetData(StartIndex() + 1) > BandsUp(StartIndex() + 1))
   {  result = int(round(100.0 * ((m_close.GetData(StartIndex()+1) - m_close.GetData(StartIndex()))/(fabs(m_close.GetData(StartIndex()) - m_close.GetData(StartIndex()+1)) + fabs(BandsUp(StartIndex()) - BandsDn(StartIndex()))))));
   }
   return(result);
}
```

### Practical Implementation and Testing

Once an Expert that uses our signal above is assembled in the wizard, if we are using multiple time frames, then the header part of the \*MQ5 file will look as follows:

```
//+------------------------------------------------------------------+
//|                                                           qm.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include                                                          |
//+------------------------------------------------------------------+
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\Signal\My\SignalWZ_25.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingNone.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedMargin.mqh>
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string Expert_Title           = "qm_frame"; // Document name
ulong        Expert_MagicNumber     = 2028; //
bool         Expert_EveryTick       = false; //
//--- inputs for main signal
input int    Signal_ThresholdOpen   = 10;   // Signal threshold value to open [0...100]
input int    Signal_ThresholdClose  = 10;   // Signal threshold value to close [0...100]
input double Signal_PriceLevel      = 0.0;  // Price level to execute a deal
input double Signal_StopLevel       = 50.0; // Stop Loss level (in points)
input double Signal_TakeLevel       = 50.0; // Take Profit level (in points)
input int    Signal_Expiration      = 4;    // Expiration of pending orders (in bars)
input int    Signal_0_QM_Length     = 50;   // QM(50) H1 Averaging Length
input double Signal_0_QM_Weight     = 1.0;  // QM(50) H1 Weight [0...1.0]
input int    Signal_1_QM_Length     = 50;   // QM(50) H4 Averaging Length
input double Signal_1_QM_Weight     = 1.0;  // QM(50) H4 Weight [0...1.0]
input int    Signal_2_QM_Length     = 50;   // QM(50) D1 Averaging Length
input double Signal_2_QM_Weight     = 1.0;  // QM(50) D1 Weight [0...1.0]
//--- inputs for money
input double Money_FixMargin_Percent = 10.0; // Percentage of margin
```

As can be seen, we assign a moving average length parameter and weighting parameter to each of the three chosen time frames, which in this case are PERIOD\_H1, PERIOD\_H4, and PERIOD\_D1. Since the generated signal for the Expert Advisor is a weighted mean of the distance between the close price and the Bollinger Band's baseline, this implies we always have a signal at any time, unlike the sharp entries we explored with the geometric mean in the previous article. This does mean it would probably be a good idea to perform tests without price targets for exits like stop loss or take profits. In the last article, the only price target we used was the take profit. Stop loss setting is always touted as a sound loss limiting strategy however the stop loss price is never guaranteed and also from extensive testing most accounts do run out of margin mostly because their position sizing was too optimistic as opposed to not having a stop loss. Obviously, the debate on the importance of a stop loss is something that am sure will carry on, but that is my take. For this article though we will stick to the take profit only target price and the reader is welcome to make changes in the input parameters when testing to suite his loss management approach.

### Strategy Tester Reports and Analysis

If we do test runs for the year 2023 on the one-hour time frame for the EURJPY pair, while seeking a balanced signal from the 1-hour, 4-hour, and Daily time frames, we do get the following as some of our fair results:

![r2](https://c.mql5.com/2/82/frame_r2.png)

![c2](https://c.mql5.com/2/82/frame_c2.png)

The results in a sense are a far cry from what we achieved when we were relying on sharp entries off the upper bands and lower bands price crosses. We therefore did perform tests with multiple timeframes while utilizing the sharp entry from the price crossing of the Bollinger Bands and surprisingly, the performance aside, the number of trades placed when waiting for these crossovers was more than in the weighted average approach we just adopted above. Below are the results:

![r1](https://c.mql5.com/2/82/frame_r1.png)

![c1](https://c.mql5.com/2/82/frame_c1.png)

### Conclusion

In conclusion, we have shown how multiple time frames can be used in wizard assembled Expert Advisors. The main point we covered in showing this is that the wizard assembly process allows each signal to not only have its own time frame, but also its own trade symbol. Customizing a signal by assigning it a specific time frame implies the OHLC buffers for that signal will be bound to that time frame, and this is a relatively straight forward process, unlike the customization of a signal to trade a particular symbol. This is because when signals are assigned to trade a specific symbol, extra changes need to be made to the ‘CExpert’ class instance where it should be in an array format to accommodate each symbol that is to be traded.In demonstrating multi-time frame trading, we used Quadratic Means as our signal where like in the previous article we derived a mirror version of it that is more weighted towards the smaller values in the averaged set given that the quadratic mean by default is weighted more towards the larger values. We tested this in two settings, where one was always assessing probabilities and another was looking for specific entry signals, and we got significantly different results with either approach.

Multi-time frame trading in general though is meant to target the sharp or ‘most correct’ entry points by concurrently monitoring a large time frame for the macro trend while picking the signal for actual entry on the shorter time frame. To this end it perhaps would have been more prudent to use different signals each with a different time frame where the larger time frame takes the probability approach we considered in our first case of testing above, and the smaller time frame considers the actual cross-over points as the trigger point for the signal.

This can be implemented and tested by the reader since both instances of these signal classes are attached at the bottom and in addition there are alternative multi-time frame implementations that can take an indicator time frame as an input where this time frame would be different from the time frame of the chart to which the Expert Advisor is attached. This approach provides a work around the problem of not being able to select the optimal time frame that is present when time frames are assigned to a signal during wizard assembly. By having it as a signal input parameter, it could be optimized to best fit the strategy. Interestingly enough, this approach can even be used to assigning a time frame to a custom OHLC series buffer. Its main shortcoming is in multi-symbol trading where even though price and indicator readings can be made off a symbol name that is a parameter, placing of trades can only be made if changes are made to the assembled Expert file as mentioned above.

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Codex Pipelines, from Python to MQL5, for Indicator Selection: A Multi-Quarter Analysis of the XLF ETF with Machine Learning](https://www.mql5.com/en/articles/20595)
- [Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)
- [Market Positioning Codex for VGT with Kendall's Tau and Distance Correlation](https://www.mql5.com/en/articles/20271)
- [Markets Positioning Codex in MQL5 (Part 2): Bitwise Learning, with Multi-Patterns for Nvidia](https://www.mql5.com/en/articles/20045)
- [Markets Positioning Codex in MQL5 (Part 1): Bitwise Learning for Nvidia](https://www.mql5.com/en/articles/20020)
- [MQL5 Wizard Techniques you should know (Part 85): Using Patterns of Stochastic-Oscillator and the FrAMA with Beta VAE Inference Learning](https://www.mql5.com/en/articles/19948)
- [MQL5 Wizard Techniques you should know (Part 84): Using Patterns of Stochastic Oscillator and the FrAMA - Conclusion](https://www.mql5.com/en/articles/19890)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469533)**
(2)


![Marcel Fitzner](https://c.mql5.com/avatar/2020/3/5E8026F2-4070.png)

**[Marcel Fitzner](https://www.mql5.com/en/users/creativewarlock)**
\|
5 Oct 2024 at 13:22

Besides showing how different timeframes are being added in the EA Wizard the article does not demonstrate how exactly the mult-timeframe testing is handled in the code - or am I missing something?


![Khaled Ali E Msmly](https://c.mql5.com/avatar/2020/12/5FE5FF28-4741.jpg)

**[Khaled Ali E Msmly](https://www.mql5.com/en/users/kamforex9496)**
\|
8 Oct 2024 at 19:44

Hi  [Stephen](https://www.mql5.com/en/users/ssn "ssn")

Your article is very wonderful, thank you for it. Can you attach the files so we can try it, or is the code in the article all the code used in the experiment?

![Developing a Replay System (Part 41): Starting the second phase (II)](https://c.mql5.com/2/65/Desenvolvendo_um_sistema_de_Replay_4Parte_41g____LOGO.png)[Developing a Replay System (Part 41): Starting the second phase (II)](https://www.mql5.com/en/articles/11607)

If everything seemed right to you up to this point, it means you're not really thinking about the long term, when you start developing applications. Over time you will no longer need to program new applications, you will just have to make them work together. So let's see how to finish assembling the mouse indicator.

![Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://c.mql5.com/2/82/Data_Science_and_ML_Part_25__LOGO.png)[Data Science and Machine Learning (Part 25): Forex Timeseries Forecasting Using a Recurrent Neural Network (RNN)](https://www.mql5.com/en/articles/15114)

Recurrent neural networks (RNNs) excel at leveraging past information to predict future events. Their remarkable predictive capabilities have been applied across various domains with great success. In this article, we will deploy RNN models to predict trends in the forex market, demonstrating their potential to enhance forecasting accuracy in forex trading.

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part III)](https://c.mql5.com/2/83/Building_A_Candlestick_Trend_Constraint_Model__Part_5___CONT___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part III)](https://www.mql5.com/en/articles/14969)

This part of the article series is dedicated to integrating WhatsApp with MetaTrader 5 for notifications. We have included a flow chart to simplify understanding and will discuss the importance of security measures in integration. The primary purpose of indicators is to simplify analysis through automation, and they should include notification methods for alerting users when specific conditions are met. Discover more in this article.

![Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://c.mql5.com/2/64/RestAPIs_em_MQL5_Logo.png)[Developing an MQL5 RL agent with RestAPI integration (Part 4): Organizing functions in classes in MQL5](https://www.mql5.com/en/articles/13863)

This article discusses the transition from procedural coding to object-oriented programming (OOP) in MQL5 with an emphasis on integration with the REST API. Today we will discuss how to organize HTTP request functions (GET and POST) into classes. We will take a closer look at code refactoring and show how to replace isolated functions with class methods. The article contains practical examples and tests.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15185&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070114567205359665)

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