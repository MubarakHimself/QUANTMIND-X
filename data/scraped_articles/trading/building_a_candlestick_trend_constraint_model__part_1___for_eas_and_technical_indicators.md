---
title: Building A Candlestick Trend Constraint Model (Part 1): For EAs And Technical Indicators
url: https://www.mql5.com/en/articles/14347
categories: Trading, Trading Systems, Indicators, Expert Advisors
relevance_score: 8
scraped_at: 2026-01-22T17:45:42.881888
---

[![](https://www.mql5.com/ff/sh/6zw0dkux8bqt7m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Install the app and receive market analytics and trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=iuciwacmrxvmiibwyujliagqikizpsoo&s=268cbb13914c54b6c5c875db99b154944f6e0122b3400b54c9ac0d4f69f0f0d6&uid=&ref=https://www.mql5.com/en/articles/14347&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049359060437936661)

MetaTrader 5 / Trading


### **Contents**

1. Introduction
2. Anatomy Of Higher Timeframe Candlesticks
3. Strategy development (Moving Average Crossover) plus the code
4. Justification of the constraint and its application plus the code
5. Benefits of Using the Code

6. Conclusion

### Introduction

As an alternative to using moving averages to define market trends, the bullish or bearish nature of higher timeframe candlesticks can provide valuable insights into market direction. For example, within a D1 or H4 candlestick, there is significant underlying activity occurring at M1 timeframes and even ticks that shape its formation. By capitalizing on buying opportunities presented by bullish D1 candles and selling during bearish phases, traders can gain an advantage. Combining this with native technical indicators at lower timeframes helps pinpoint entry points, offering a strategic edge to traders. When dealing with a bullish daily candle, traders should patiently wait for favorable market conditions to align before confidently riding the trend.

This article aims to effectively classify the current candle as bullish or bearish using MQL5 code, establishing a condition to sell only when it is bearish and buy when it is bullish.

This model aims to confine the signal generator to producing signals aligned with the current candle trend. Think of a fence that restricts certain creatures from entering your yard based on their body size while permitting others. We are applying a similar concept to filter out select signals and retain only the most optimal ones. The model accomplishes this by analyzing the higher timeframe candlestick and market trends, effectively creating a virtual barrier that allows only signals conforming to the prevailing trend to pass through. This selective filtration process enhances the accuracy and reliability of the generated signals, ensuring that only the most favorable trading opportunities are presented to the user.

At the end of this article, you must be able to:

1. Understand the price action with the entire D1 candlestick at microscopic timeframes available.
2. Create MA crossover indicator Buffer that include the Higher Timeframe Trend Constraint condition.
3. Understand the concept of  screen out best signals from a given strategy.

### Anatomy Of Higher Timeframe Candlesticks

![Boom 500 index, D1 candlestick anatomy M5,13.04.24](https://c.mql5.com/2/75/Boom_500_IndexM5_Candle_anatomy.png)

FIG 1.1: The anatomy of D1 as viewed at M5 timeframe for Boom 500 index synthetics.

The image above depicts a D1 candlestick red bar on the far left and an M5 price action on the right bound between daily period separators. A distinct downtrend is evident, supported by the bearish nature of the Daily candle, indicating a higher probability of selling. The setup emphasizes executing trades in alignment with the daily candlestick, reflecting the emergence of constraint idea from the development of a higher timeframe trend.

### Strategy development (Moving Average Crossover)

Developing a trading strategy requires a combination of analysis, testing, and ongoing refinement. A successful trading strategy should be based on a thorough understanding of the market, as well as a clear set of rules and guidelines to follow. It is important to constantly monitor and adjust the strategy as market conditions change, in order to adapt to new trends and opportunities. By continuously analyzing data, backtesting different approaches, and making adjustments as needed, traders can increase their chances of success in the market.

Before we proceed to our MA crossover strategy below is an outline summarizing key tips for strategy development:

1. Define Your Objectives and Risk Tolerance
2. Understand the Market
3. Choose Your Trading Style
4. Develop Entry and Exit Rules
5. Implement Risk Management Strategies
6. Backtest Your Strategy
7. Optimize and Refine
8. Paper Trade Before Going Live
9.  Monitor and Evaluate

Here we will develop a basic moving average crossover indicator that shows an arrow and triggers a notification as soon as the crossover occurs. The following are the steps for our strategy development criteria.

1. Set the conditions of the strategy ( In this case crossover of EMA7 above or below EMA21)
2. Set the display style for the indicator which can be an arrow or any geometric shape available in meta trader 5
3. (Optional) if the indicator is going to be customizable by user, set the inputs

I decided to include the final code here without much explanation to focus on the constraint algorithm, our main topic in this article. The following program is ready to compile and generate buy and sell signals. After the code, we will analyze the results on the mt5 chart and identify the issue that the constraint algorithm will address.

```
//Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.00"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"

//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots 2

#property indicator_type1 DRAW_ARROW
#property indicator_width1 5
#property indicator_color1 0xFFAA00
#property indicator_label1 "Buy"

#property indicator_type2 DRAW_ARROW
#property indicator_width2 5
#property indicator_color2 0x0000FF
#property indicator_label2 "Sell"

#define PLOT_MAXIMUM_BARS_BACK 5000
#define OMIT_OLDEST_BARS 50

//--- indicator buffers
double Buffer1[];
double Buffer2[];

double myPoint; //initialized in OnInit
int MA_handle;
double MA[];
int MA_handle2;
double MA2[];
double Low[];
double High[];

void myAlert(string type, string message)
  {
   if(type == "print")
      Print(message);
   else if(type == "error")
     {
      Print(type+" | Trend constraint @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
   else if(type == "order")
     {
     }
   else if(type == "modify")
     {
     }
  }

// Custom indicator initialization function
int OnInit()
  {
   SetIndexBuffer(0, Buffer1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(0, PLOT_ARROW, 241);
   SetIndexBuffer(1, Buffer2);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(1, PLOT_ARROW, 242);
   //initialize myPoint
   myPoint = Point();
   if(Digits() == 5 || Digits() == 3)
     {
      myPoint *= 10;
     }
   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle < 0)
     {
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle2 < 0)
     {
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//Custom indicator iteration function
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   int limit = rates_total - prev_calculated;
   //--- counting from 0 to rates_total
   ArraySetAsSeries(Buffer1, true);
   ArraySetAsSeries(Buffer2, true);
   //--- initial zero
   if(prev_calculated < 1)
     {
      ArrayInitialize(Buffer1, EMPTY_VALUE);
      ArrayInitialize(Buffer2, EMPTY_VALUE);
     }
   else
      limit++;

   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      //Indicator Buffer 1
      if(MA[i] > MA2[i]
      && MA[i+1] < MA2[i+1] //Moving Average crosses above Moving Average
      )
        {
         Buffer1[i] = Low[i]; //Set indicator value at Candlestick Low
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(MA[i] < MA2[i]
      && MA[i+1] > MA2[i+1] //Moving Average crosses below Moving Average
      )
        {
         Buffer2[i] = High[i]; //Set indicator value at Candlestick High
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
//copy the code to meta editor to compile it
```

On 12.04.24, the EURUSD test result chart indicates the initiation of a period observed at M1 timeframes. Crossovers are evident with arrows denoting their positions - red for sell signals and blue for buys. Despite this, a broader view reveals a distinct downtrend, indicating a bearish D1 candle. The crossover indicator is issuing both signals, disregarding the prevailing trend, posing a significant issue. The conflicting signals create a challenging scenario for traders attempting to navigate the market. While the M1 timeframes suggest potential short-term opportunities, the overarching downtrend on the D1 candle raises concerns about the sustainability of any upward movements. In this article, we will address this issue by limiting our signals to the D1 trend.

![Moving Average Crossover Signals](https://c.mql5.com/2/75/EURUSDM1Xover_.png)

FIG 1.2 : Test result of MA crossover indicator before the constraint

The D1 candle result is Bearish with both buy and sell arrows generated by the MA crossover condition. Many signals are deemed fake or off-trend, which can be rectified by incorporating the Higher Timeframe trend constraint. The resulting table below was created using chart information from the start of the day to closing.

| Signal Type | Units |
| --- | --- |
| Sell Signals | 29 |
| Buy Signals | 28 |
| Total | 57 |
| Fake and off Trend signals | 41 |
| Successful Signals | 25 |

### Justification of the constraint and its application

Imagine having a blend of maize grain and sorghum; one is coarser than the other. To segregate them, we employ a sieve. This action confines the maize grain, permitting only the sorghum to sift through, demonstrating a level of control. This analogy aligns with the concept of the Higher timeframe constraint under examination. It acts as a filter, sieving out certain signals and retaining only those in harmony with the prevailing trend. The Higher timeframe constraint, akin to the sieve, refines our focus, allowing us to discern the essential elements of the market trend. By sifting through the noise, we can better grasp the underlying dynamics at play, facilitating more informed decision-making. This strategic approach enhances our ability to navigate the complexities of the financial landscape, ensuring that we stay aligned with the overarching direction, much like the sorghum separated from the maize grain, revealing the true essence of the market movement.

Defining the nature of the D1 candle as a condition for the Trend Constraint we previously coded.

- I defined my current market sentiment as bullish or bearish by comparing the previous day's closing price which is similar to the opening price of the current day and the closing prices of the lower timeframe M1 candlesticks.

**For a BULL candlestick:**

close of the previous M1 candle  >=  close of previous D1 candle

For a BEAR candlestick:

close of the previous M1 candle  <=  close of previous D1 candle

The math suggests that with a bull D1 candle as a trend driver, we will only receive buy signals and the opposite is true for a bear D1 candle.

We opted for the lower timeframe close as a point of comparison to our D1 open rather than utilizing other aspects such as Bid and Ask price or the Current day Close, as the latter does not exhibit arrows on the chart as desired according to this strategy and indicator style. Therefore, by focusing on the lower timeframe close in relation to our D1 open, we can effectively align our strategy with the desired visual cues and indicators on the chart. This approach enhances clarity and precision in our trading decisions, ensuring a more streamlined and efficient analysis process.

- Let's examine the code. The code is well-structured and easy to follow.

```
 if(Close[1+barshift_M1[i]] >= Open[1+barshift_D1[i]] //Candlestick Close >= Candlestick Open
      )
        {
         Buffer1[i] = Low[i]; //Set indicator value at Candlestick Low
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
```

The code above represents a Bullish D1 candle condition. Remember the flip holds true for the Bearish.

- Finally, I present the final code where the constraint is seamlessly integrated with the moving average crossover indicator.

```
/Indicator Name: Trend Constraint
#property copyright "Clemence Benjamin"
#property link      "https://mql5.com"
#property version   "1.00"
#property description "A model that seek to produce sell signal when D1 candle is Bearish only and  buy signal when it is Bullish"

//--- indicator settings
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots 2

#property indicator_type1 DRAW_ARROW
#property indicator_width1 5/
#property indicator_color1 0xFFAA00
#property indicator_label1 "Buy"

#property indicator_type2 DRAW_ARROW
#property indicator_width2 5
#property indicator_color2 0x0000FF
#property indicator_label2 "Sell"

#define PLOT_MAXIMUM_BARS_BACK 5000
#define OMIT_OLDEST_BARS 50

//--- indicator buffers
double Buffer1[];
double Buffer2[];

double myPoint; //initialized in OnInit
int MA_handle;
double MA[];
int MA_handle2;
double MA2[];
double Close[];
double Close2[];
double Low[];
double High[];

void myAlert(string type, string message)
  {
   if(type == "print")
      Print(message);
   else if(type == "error")
     {
      Print(type+" | Trend constraint @ "+Symbol()+","+IntegerToString(Period())+" | "+message);
     }
   else if(type == "order")
     {
     }
   else if(type == "modify")
     {
     }
  }

// Custom indicator initialization function
int OnInit()
  {
   SetIndexBuffer(0, Buffer1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(0, PLOT_ARROW, 241);
   SetIndexBuffer(1, Buffer2);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(1, PLOT_DRAW_BEGIN, MathMax(Bars(Symbol(), PERIOD_CURRENT)-PLOT_MAXIMUM_BARS_BACK+1, OMIT_OLDEST_BARS+1));
   PlotIndexSetInteger(1, PLOT_ARROW, 242);
   //initialize myPoint
   myPoint = Point();
   if(Digits() == 5 || Digits() == 3)
     {
      myPoint *= 10;
     }
   MA_handle = iMA(NULL, PERIOD_CURRENT, 7, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle < 0)
     {
      Print("The creation of iMA has failed: MA_handle=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   MA_handle2 = iMA(NULL, PERIOD_CURRENT, 21, 0, MODE_EMA, PRICE_CLOSE);
   if(MA_handle2 < 0)
     {
      Print("The creation of iMA has failed: MA_handle2=", INVALID_HANDLE);
      Print("Runtime error = ", GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

// Custom indicator iteration function
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime& time[],
                const double& open[],
                const double& high[],
                const double& low[],
                const double& close[],
                const long& tick_volume[],
                const long& volume[],
                const int& spread[])
  {
   int limit = rates_total - prev_calculated;
   //--- counting from 0 to rates_total
   ArraySetAsSeries(Buffer1, true);
   ArraySetAsSeries(Buffer2, true);
   //--- initial zero
   if(prev_calculated < 1)
     {
      ArrayInitialize(Buffer1, EMPTY_VALUE);
      ArrayInitialize(Buffer2, EMPTY_VALUE);
     }
   else
      limit++;

   datetime TimeShift[];
   if(CopyTime(Symbol(), PERIOD_CURRENT, 0, rates_total, TimeShift) <= 0) return(rates_total);
   ArraySetAsSeries(TimeShift, true);
   int barshift_M1[];
   ArrayResize(barshift_M1, rates_total);
   int barshift_D1[];
   ArrayResize(barshift_D1, rates_total);
   for(int i = 0; i < rates_total; i++)
     {
      barshift_M1[i] = iBarShift(Symbol(), PERIOD_M1, TimeShift[i]);
      barshift_D1[i] = iBarShift(Symbol(), PERIOD_D1, TimeShift[i]);
   }
   if(BarsCalculated(MA_handle) <= 0)
      return(0);
   if(CopyBuffer(MA_handle, 0, 0, rates_total, MA) <= 0) return(rates_total);
   ArraySetAsSeries(MA, true);
   if(BarsCalculated(MA_handle2) <= 0)
      return(0);
   if(CopyBuffer(MA_handle2, 0, 0, rates_total, MA2) <= 0) return(rates_total);
   ArraySetAsSeries(MA2, true);
   if(CopyClose(Symbol(), PERIOD_M1, 0, rates_total, Close) <= 0) return(rates_total);
   ArraySetAsSeries(Close, true);
   if(CopyClose(Symbol(), PERIOD_D1, 0, rates_total, Close2) <= 0) return(rates_total);
   ArraySetAsSeries(Close2, true);
   if(CopyLow(Symbol(), PERIOD_CURRENT, 0, rates_total, Low) <= 0) return(rates_total);
   ArraySetAsSeries(Low, true);
   if(CopyHigh(Symbol(), PERIOD_CURRENT, 0, rates_total, High) <= 0) return(rates_total);
   ArraySetAsSeries(High, true);
   //--- main loop
   for(int i = limit-1; i >= 0; i--)
     {
      if (i >= MathMin(PLOT_MAXIMUM_BARS_BACK-1, rates_total-1-OMIT_OLDEST_BARS)) continue; //omit some old rates to prevent "Array out of range" or slow calculation

      if(barshift_M1[i] < 0 || barshift_M1[i] >= rates_total) continue;
      if(barshift_D1[i] < 0 || barshift_D1[i] >= rates_total) continue;

      //Indicator Buffer 1
      if(MA[i] > MA2[i]
      && MA[i+1] < MA2[i+1] //Moving Average crosses above Moving Average
      && Close[1+barshift_M1[i]] >= Close2[1+barshift_D1[i]] //Candlestick Close >= Candlestick Close
      )
        {
         Buffer1[i] = Low[i]; //Set indicator value at Candlestick Low
        }
      else
        {
         Buffer1[i] = EMPTY_VALUE;
        }
      //Indicator Buffer 2
      if(MA[i] < MA2[i]
      && MA[i+1] > MA2[i+1] //Moving Average crosses below Moving Average
      && Close[1+barshift_M1[i]] <= Close2[1+barshift_D1[i]] //Candlestick Close <= Candlestick Close
      )
        {
         Buffer2[i] = High[i]; //Set indicator value at Candlestick High
        }
      else
        {
         Buffer2[i] = EMPTY_VALUE;
        }
     }
   return(rates_total);
  }
```

The result of the code above is excellent. Please refer to the image below.

![Trend Constraint applied on Moving Average Crossover Indicator](https://c.mql5.com/2/75/EURUSDM1TCapplied.png)

FIG 1.4: Trend Constraint applied and the result is amazing

| Signal Type | Units |
| --- | --- |
| Sell Signal | 27 |
| Buy Signal | 1 |
| Total | 28 |
| Fake and off trend signals | 3 |
| Successful signals | 25 |

From the table above, we notice a positive impact of the Higher Timeframe Trend Constraint as a filter, resulting in more wins than failures, making it ideal for EAs. Comparing the previous and current results tables, we notice that the successful signals maintained their value while all false signals were reduced. This improvement in signal accuracy indicates a positive trend in our data analysis methodology. By refining our algorithms and criteria, we have effectively minimized the occurrence of false signals, enhancing the overall reliability of our results. Moving forward, this progress will undoubtedly contribute to more informed decision-making and improved outcomes in our strategies.

### The Benefits of using the code

The advantages of imposing constraints on higher timeframe trends enhance the clarity of market direction, reducing overtrading tendencies and fostering a more disciplined approach to trading decisions. This method can also provide a broader perspective, helping traders avoid getting caught up in short-term fluctuations and enabling them to align their strategies with the prevailing long-term trend. By focusing on the bigger picture, traders are better equipped to filter out noise and make more informed decisions based on the underlying trend. This approach encourages patience and a deeper understanding of market dynamics, ultimately leading to more consistent and profitable trading outcomes. Additionally, constraints on higher timeframe trends can serve as valuable tools for risk management, allowing traders to set clear levels for entry and exit points based on a strategic assessment of market conditions.

In summary,

- Improved accuracy of signal-generating indicators
- Better risk management
- Increased profitability
- Less work
- Fewer signals

### Conclusion

Higher timeframe candlesticks exert a significant influence on the lower timeframe trends, whether bullish or bearish, essentially steering the markets. Based on various studies, it is advisable to consider buying during bullish day candles at lower timeframes and selling during bearish candles. Personally, integrating the concept of higher timeframe trend restrictions, like this one, could be crucial in EA and indicator development for sustained positive outcomes. The query arises: Should we now discard moving averages as our trend-defining instruments? Perhaps the forthcoming article will shed light on this as we delve deeper into refining and evolving this concept. Attached are the source code file you can view them in Meta editor and Ex5 files in Meta trader 5 platform.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14347.zip "Download all attachments in the single ZIP archive")

[Moving\_Average\_Crossover.mq5](https://www.mql5.com/en/articles/download/14347/moving_average_crossover.mq5 "Download Moving_Average_Crossover.mq5")(5.08 KB)

[Moving\_Average\_Crossover.ex5](https://www.mql5.com/en/articles/download/14347/moving_average_crossover.ex5 "Download Moving_Average_Crossover.ex5")(8.29 KB)

[Trend\_constraint.mq5](https://www.mql5.com/en/articles/download/14347/trend_constraint.mq5 "Download Trend_constraint.mq5")(6.26 KB)

[Trend\_constraint.ex5](https://www.mql5.com/en/articles/download/14347/trend_constraint.ex5 "Download Trend_constraint.ex5")(9.89 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/466132)**

![Developing a Replay System (Part 35): Making Adjustments (I)](https://c.mql5.com/2/60/Desenvolvendo_um_sistema_de_Replay_dParte_35a_Logo.png)[Developing a Replay System (Part 35): Making Adjustments (I)](https://www.mql5.com/en/articles/11492)

Before we can move forward, we need to fix a few things. These are not actually the necessary fixes but rather improvements to the way the class is managed and used. The reason is that failures occurred due to some interaction within the system. Despite attempts to find out the cause of such failures in order to eliminate them, all these attempts were unsuccessful. Some of these cases make no sense, for example, when we use pointers or recursion in C/C++, the program crashes.

![Creating a market making algorithm in MQL5](https://c.mql5.com/2/64/Creating_a_market_making_algorithm_in_MQL5____LOGO____2.png)[Creating a market making algorithm in MQL5](https://www.mql5.com/en/articles/13897)

How do market makers work? Let's consider this issue and create a primitive market-making algorithm.

![Population optimization algorithms: Micro Artificial immune system (Micro-AIS)](https://c.mql5.com/2/64/Bacterial_Foraging_Optimization_-_Genetic_Algorithmi_BFO-GA____LOGO.png)[Population optimization algorithms: Micro Artificial immune system (Micro-AIS)](https://www.mql5.com/en/articles/13951)

The article considers an optimization method based on the principles of the body's immune system - Micro Artificial Immune System (Micro-AIS) - a modification of AIS. Micro-AIS uses a simpler model of the immune system and simple immune information processing operations. The article also discusses the advantages and disadvantages of Micro-AIS compared to conventional AIS.

![A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://c.mql5.com/2/76/A_Generic_Optimization_Formulation_2GOFt_to_Implement_Custom_Max_with_Constraints____LOGO.png)[A Generic Optimization Formulation (GOF) to Implement Custom Max with Constraints](https://www.mql5.com/en/articles/14365)

In this article we will present a way to implement optimization problems with multiple objectives and constraints when selecting "Custom Max" in the Setting tab of the MetaTrader 5 terminal. As an example, the optimization problem could be: Maximize Profit Factor, Net Profit, and Recovery Factor, such that the Draw Down is less than 10%, the number of consecutive losses is less than 5, and the number of trades per week is more than 5.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mzeqdvrzpokvswknbjvrbeklnnkhzumt&ssn=1769093142581938124&ssn_dr=0&ssn_sr=0&fv_date=1769093142&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14347&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20A%20Candlestick%20Trend%20Constraint%20Model%20(Part%201)%3A%20For%20EAs%20And%20Technical%20Indicators%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909314203957057&fz_uniq=5049359060437936661&sv=2552)

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