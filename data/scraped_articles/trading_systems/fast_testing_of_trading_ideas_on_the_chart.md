---
title: Fast Testing of Trading Ideas on the Chart
url: https://www.mql5.com/en/articles/505
categories: Trading Systems, Indicators
relevance_score: 0
scraped_at: 2026-01-24T13:54:58.357920
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/505&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6428866254849709824)

MetaTrader 5 / Trading systems


### Introduction

The sixth Automated Trading Championship has started at last. All initial excitement is over and we finally can relax a bit and examine submitted trading robots. I decided to do a little research to find out the most noticeable features of modern trading robots and define what we can expect from their trading activity.

That proved to be difficult enough. Therefore, my calculations cannot be called perfectly accurate or complete, as Expert Advisor descriptions and rare comments of the developers were the only things I had. However, we still can draw some conclusions and below are the results of my calculations: 451 Expert Advisors participate in the Championship but only 316 of them contain some meaningful descriptions. Developers of the remaining ones filled their descriptions with greetings to their friends and family, messages to extraterrestrial civilizations or self-applause.

Most popular strategies on ATC 2012:

- trading using various graphical constructions (important price levels, support-resistance levels, channels) – 55;
- price movement analysis (for various time frames) – 33;
- trend tracking systems (I guess, these big words hide some overoptimized combination of moving averages but I may be wrong) :) ) – 31;
- statistical price patterns – 10:
- arbitration, symbols correlation analysis – 8;
- analysis of volatility – 8;
- neural networks – 7;
- candlestick analysis – 5;
- averagers – 5;
- strategy bundles – 5;
- trading session time – 4;
- random-number generator – 4;
- trading the news – 3,
- Elliott Waves – 2.

Indicator strategies are traditionally the most popular ones, of course. It is difficult to define the role of each particular indicator in a particular Expert Advisor but it is possible to estimate the absolute number of their use:

- Moving Average – 75;
- MACD – 54;
- Stochastic Oscillator – 25;
- RSI – 23;
- Bollinger Bands – 19;
- Fractals – 8;
- CCI, ATR – 7 indicators each;
- Zigzag, Parabolic SAR – 6 indicators each;
- ADX – 5;
- Momentum – 4;
- custom indicators (how intriguing :) ) – 4;
- Ichimoku, AO – 3 indicators each;
- ROC, WPR, StdDev, Volumes – 2 indicators each.

The data suggests the following conclusions - most participants use trade following strategies with indicators. Perhaps, I have missed something when collecting the data, and we will see the advent of some outstanding personalities in the field of the automated trading but it seems unlikely for now. I think that the main problem is that newcomers attracted by the market in most cases receive rules instead of knowledge.

For example, here are the rules of using MACD, here are the signals - now optimize the parameters and make money. What about using brains a bit? Nonsense! The standards have already been developed! Why reinvent the wheel? However, we often forget that the indicators that are so popular now have also been invented by traders just like me and you. They also had their standards and authorities. Perhaps, a new indicator bearing your name will become a standard one in some ten years.

I would like to share my method of searching for trading ideas, as well as the method I use for fast testing of these ideas.

### Method Description

All technical analysis is based on one simple axiom - prices consider everything. But there is one issue - this statement lacks dynamics. We look at the chart and see a static image: the price has actually considered everything. However, we want to know what the price will consider in a certain period of time in the future and where it will go, so that we can make profit. The indicators derived of the price have been designed exactly to predict possible future movements.

As we know from physics, the first-order derivative of the magnitude is velocity. Therefore, the indicators calculate the current price change velocity. We also know that significant magnitudes have inertia preventing velocity from sharp changes of its value without the intervention of considerable external forces. That is how we gradually approach the concept of a trend - the price state when its first-order derivative (velocity) keeps its value during the period of time when external forces (news, central banks' policies, etc.) do not affect the market.

But let's go back to where we started from - prices consider everything. To develop new ideas, we should examine the behavior of the price and its derivatives at the same time interval. Only careful examination of price charts will raise your trading from blind faith up to the level of genuine understanding.

This may not lead to the immediate changes in trading results but the ability to answer numerous why-questions will play a positive role sooner or later. Besides, visual analysis of charts and indicators will let you find some brand new correlations between prices and indicators completely unforeseen by their developers.

Suppose that you have found a new correlation that seemingly works in your favor. What's next? The easiest way is to write an Expert Advisor and test it on historical data making sure that your assumption is correct. If that is not the case, we have to choose a common way of optimizing parameters. The worst thing about it is that we were not able to answer the why-question. Why has our Expert Advisor turned out to be loss-making/profitable? Why was there such a huge drawdown? Without the answers, you won't be able to implement your idea efficiently.

I perform the following actions to visualize the results of an obtained correlation right on the chart:

1. I create or change the necessary indicator, so that it generates a signal: -1 for sell and 1 for buy.
2. I connect the balance indicator displaying entry and exit points to the chart. The indicator also shows the changes of the balance and equity (in points) when processing the signal.
3. I analyze in what cases and circumstances my assumptions are correct.

The method has certain advantages.

- First, the balance indicator is entirely calculated using OnCalculate method providing maximum calculation speed and automatic availability of historical data in the input calculation arrays.

- Second, adding the signal to the existing indicator is an intermediate step between creating an Expert Advisor via Wizard and developing it on your own.

- Third, an idea and a final result can be seen on a single chart. Of course, the method has some limitations: a signal is tied to the bar's closing price, the balance is calculated for the constant lot, there are no options for trading using pending orders. However, all these limitations can be easily fixed/improved.

### Implementation

Let's develop a simple signal indicator to understand how it works and evaluate the method's convenience. I have long heard of the candlestick patterns. So, why not check out their work in practice? I have selected "hammer" and "shooting star" reverse patterns as buying and selling signals, respectively. The images below show their schematic look:

![Figure 1. "Hammer" and "shooting star" candlestick patterns](https://c.mql5.com/2/4/models__1.PNG)

Figure 1. "Hammer" and "shooting star" candlestick patterns

Now, let's define the market entry rules when the "hammer" pattern appears.

1. The candle's lowest value should be lower than the ones of **five** previous candles;
2. The candle's body should not exceed **50%** of its total height;
3. The candle's upper shadow should not exceed **0%** of its total height;
4. The candle's height should be not less than **100%** of the average height of **five** candles before it;
5. The pattern's close price should be lower than the **10**-period Moving Average.

If these conditions are met, we should open a long position. The rules are the same for the "shooting star" pattern. The only difference is that we should open a short position:

1. The candle's highest value should be higher than the ones of **five** previous candles;
2. The candle's body should not exceed **50%** of its total height;
3. The candle's lower shadow should not exceed **0%** of its total height;
4. The candle's height should be not less than **100%** of the average height of **five** candles before it;
5. The pattern's close price should be higher than the **10**-period Moving Average.

I used bold style for the parameters I used based on drawings that can be optimized in the future (if the pattern shows acceptable results). Limitations I want to implement allow us to clear the patterns from the ones having inappropriate appearance (pp. 1-3), as well as from the knowingly weak ones that cannot be accepted as signals.

Besides, we should determine the exit moments. Since the mentioned patterns appear as trend reversal signals, the trend exists at the moment the appropriate candle appears. Therefore, the moving average chasing the price will also be present. The exit signal is formed by crossing of the price and its **10**-period moving average.

Now, it is time to do some programming. Let's develop a new custom indicator in MQL5 Wizard, name it PivotCandles and describe its behavior. Let's define the returned values to connect the balance indicator:

- -1 – open a sell position;
- -2 – close a buy position;
- 0 – no signal;
- 1 – open buy position;
- 2 – close sell position.

As you know, genuine programmers do not look for easy ways. They look for the easiest ones. :) I am not an exception. While listening to music in my headphones and drinking aromatic coffee, I created the file with the class to be implemented in an indicator and in an Expert Advisor (in case I decide to develop it based on the indicator). Perhaps, it can even be modified for other candlestick patterns. The code does not contain anything brand new. I believe that implemented comments to the code cover any possible questions.

```
//+------------------------------------------------------------------+
//|                                            PivotCandlesClass.mqh |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int      iMaxBodySize            = 50;  // Maximum candle body, %
input int      iMaxShadowSize          = 0;   // Maximum allowed candle shadow, %
input int      iVolatilityCandlesCount = 5;   // Number of previous bars for calculation of an average volatility
input int      iPrevCandlesCount       = 5;   // Number of previous bars, for which the current bar should be an extremum
input int      iVolatilityPercent      = 100; // Correlation of a signal candle with a previous volatility, %
input int      iMAPeriod               = 10;  // Period of a simple signal moving average
//+------------------------------------------------------------------+
//| Class definition                                                 |
//+------------------------------------------------------------------+
class CPivotCandlesClass
  {
private:
   MqlRates          m_candles[];              // Array for storing the history necessary for calculations
   int               m_history_depth;          // Array length for storing the history
   int               m_handled_candles_count;  // Number of the already processed candles

   double            m_ma_value;               // Current calculated moving average value
   double            m_prev_ma_value;          // Previous calculated moving average value
   bool              m_is_highest;             // Check if the current candle is the highest one
   bool              m_is_lowest;              // Check if the current candle is the lowest one
   double            m_volatility;             // Average volatility
   int               m_candle_pattern;         // Current recognized pattern

   void              PrepareArrayForNewCandle();        // Prepare the array for accepting the new candle
   int               CheckCandleSize(MqlRates &candle); // Check the candle for conformity with patterns
   void              PrepareCalculation();
protected:
   int               DoAnalizeNewCandle();              // Calculation function
public:
   void              CPivotCandlesClass();

   void              CleanupHistory();                  // Clean up all calculation variables
   double            MAValue() {return m_ma_value;}     // Current value of the moving average
   int               AnalizeNewCandle(MqlRates& candle);
   int               AnalizeNewCandle( const datetime time,
                                       const double open,
                                       const double high,
                                       const double low,
                                       const double close,
                                       const long tick_volume,
                                       const long volume,
                                       const int spread );
  };
//+------------------------------------------------------------------+
//| CPivotCandlesClass                                               |
//+------------------------------------------------------------------+
//| Class initialization                                             |
//+------------------------------------------------------------------+
void CPivotCandlesClass::CPivotCandlesClass()
  {
   // History depth should be enough for all calculations
   m_history_depth = (int)MathMax(MathMax(
      iVolatilityCandlesCount + 1, iPrevCandlesCount + 1), iMAPeriod);
   m_handled_candles_count = 0;
   m_prev_ma_value = 0;
   m_ma_value = 0;

   ArrayResize(m_candles, m_history_depth);
  }
//+------------------------------------------------------------------+
//| CleanupHistory                                                   |
//+------------------------------------------------------------------+
//| Clean up the candle buffer for recalculation                     |
//+------------------------------------------------------------------+
void CPivotCandlesClass::CleanupHistory()
  {
   // Clean up the array
   ArrayFree(m_candles);
   ArrayResize(m_candles, m_history_depth);

   // Null calculation variables
   m_handled_candles_count = 0;
   m_prev_ma_value = 0;
   m_ma_value = 0;
  }
//+-------------------------------------------------------------------+
//| AnalizeNewCandle                                                  |
//+-------------------------------------------------------------------+
//| Preparations for analyzing the new candle and the analysis itself |
//| based on candle's separate parameter values                       |
//+-------------------------------------------------------------------+
int CPivotCandlesClass::AnalizeNewCandle( const datetime time,
                                          const double open,
                                          const double high,
                                          const double low,
                                          const double close,
                                          const long tick_volume,
                                          const long volume,
                                          const int spread )
  {
   // Prepare the array for the new candle
   PrepareArrayForNewCandle();

   // Fill out the current value of the candle
   m_candles[0].time          = time;
   m_candles[0].open          = open;
   m_candles[0].high          = high;
   m_candles[0].low           = low;
   m_candles[0].close         = close;
   m_candles[0].tick_volume   = tick_volume;
   m_candles[0].real_volume   = volume;
   m_candles[0].spread        = spread;

   // Check if there is enough data for calculation
   if (m_handled_candles_count < m_history_depth)
      return 0;
   else
      return DoAnalizeNewCandle();
  }
//+-------------------------------------------------------------------+
//| AnalizeNewCandle                                                  |
//+-------------------------------------------------------------------+
//| Preparations for analyzing the new candle and the analysis itself |
//| based on the received candle                                      |
//+-------------------------------------------------------------------+
int CPivotCandlesClass::AnalizeNewCandle(MqlRates& candle)
  {
   // Prepare the array for the new candle
   PrepareArrayForNewCandle();

   // Add the candle
   m_candles[0] = candle;

   // Check if there is enough data for calculation
   if (m_handled_candles_count < m_history_depth)
      return 0;
   else
      return DoAnalizeNewCandle();
  }
//+------------------------------------------------------------------+
//| PrepareArrayForNewCandle                                         |
//+------------------------------------------------------------------+
//| Prepare the array for the new candle                             |
//+------------------------------------------------------------------+
void CPivotCandlesClass::PrepareArrayForNewCandle()
  {
   // Shift the array by one position to write the new value there
   ArrayCopy(m_candles, m_candles, 1, 0, m_history_depth-1);

   // Increase the counter of added candles
   m_handled_candles_count++;
  }
//+------------------------------------------------------------------+
//| CalcMAValue                                                      |
//+------------------------------------------------------------------+
//| Calculate the current values of the Moving Average, volatility   |
//|   and the value extremality                                      |
//+------------------------------------------------------------------+
void CPivotCandlesClass::PrepareCalculation()
  {
   // Store the previous value
   m_prev_ma_value = m_ma_value;
   m_ma_value = 0;

   m_is_highest = true; 	// check if the current candle is the highest one
   m_is_lowest = true;  	// check if the current candle is the lowest one
   m_volatility = 0;  	// average volatility

   double price_sum = 0; // Variable for storing the sum
   for (int i=0; i<m_history_depth; i++)
     {
      if (i<iMAPeriod)
         price_sum += m_candles[i].close;
      if (i>0 && i<=iVolatilityCandlesCount)
         m_volatility += m_candles[i].high - m_candles[i].low;
      if (i>0 && i<=iPrevCandlesCount)
        {
         m_is_highest = m_is_highest && (m_candles[0].high > m_candles[i].high);
         m_is_lowest = m_is_lowest && (m_candles[0].low < m_candles[i].low);
        }
     }
   m_ma_value = price_sum / iMAPeriod;
   m_volatility /= iVolatilityCandlesCount;

   m_candle_pattern = CheckCandleSize(m_candles[0]);
  }
//+------------------------------------------------------------------+
//| CheckCandleSize                                                  |
//+------------------------------------------------------------------+
//| Check if the candle sizes comply with the patterns               |
//| The function returns:                                            |
//|   0 - if the candle does not comply with the patterns            |
//|   1 - if "hammer" pattern is detected                            |
//|   -1 - if "shooting star" pattern is detected                    |
//+------------------------------------------------------------------+
int CPivotCandlesClass::CheckCandleSize(MqlRates &candle)
  {
   double candle_height=candle.high-candle.low;          // candle's full height
   double candle_body=MathAbs(candle.close-candle.open); // candle's body height

   // Check if the candle has a small body
   if(candle_body/candle_height*100.0>iMaxBodySize)
      return 0;

   double candle_top_shadow=candle.high-MathMax(candle.open,candle.close);   // candle upper shadow height
   double candle_bottom_shadow=MathMin(candle.open,candle.close)-candle.low; // candle bottom shadow height

   // If the upper shadow is very small, that indicates the "hammer" pattern
   if(candle_top_shadow/candle_height*100.0<=iMaxShadowSize)
      return 1;
   // If the bottom shadow is very small, that indicates the "shooting star" pattern
   else if(candle_bottom_shadow/candle_height*100.0<=iMaxShadowSize)
      return -1;
   else
      return 0;
  }
//+------------------------------------------------------------------+
//| DoAnalizeNewCandle                                               |
//+------------------------------------------------------------------+
//| Real analysis of compliance with the patterns                    |
//+------------------------------------------------------------------+
int CPivotCandlesClass::DoAnalizeNewCandle()
  {
   // Prepare data for analyzing the current situation
   PrepareCalculation();

   // Process prepared data and set the exit signal
   int signal = 0;

   ///////////////////////////////////////////////////////////////////
   // EXIT SIGNALS                                                  //
   ///////////////////////////////////////////////////////////////////
   // If price crosses the moving average downwards, short position is closed
   if(m_candles[1].close > m_prev_ma_value && m_candles[0].close < m_ma_value)
      signal = 2;
   // If price crosses the moving average upwards, long position is closed
   else if (m_candles[1].close < m_prev_ma_value && m_candles[0].close > m_ma_value)
      signal = -2;

   ///////////////////////////////////////////////////////////////////
   // ENTRY SIGNALS                                                 //
   ///////////////////////////////////////////////////////////////////
   // Check if the minimum volatility condition is met
   if (m_candles[0].high - m_candles[0].low >= iVolatilityPercent / 100.0 * m_volatility)
     {
      // Checks for "shooting star" pattern
      if (m_candle_pattern < 0 && m_is_highest && m_candles[0].close > m_ma_value)
         signal = -1;
      // Checks for "hammer" pattern
      else if (m_candle_pattern > 0 && m_is_lowest && m_candles[0].close < m_ma_value)
         signal = 1;
     }

   return signal;
  }
//+------------------------------------------------------------------+
```

We can see that the entire calculation part is performed by **CPivotCandlesClass** class.It is considered to be a good programming to separate the calculation part from the visual one and I try to do my best to follow this recommendation. The benefits are not late in coming - below is the code of the indicator itself:

```
//+------------------------------------------------------------------+
//|                                                 PivotCandles.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window

// Use four buffers, while drawing two
#property indicator_buffers 4
#property indicator_plots   2
//--- plot SlowMA
#property indicator_label1  "SlowMA"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrAliceBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot ChartSignal
#property indicator_label2  "ChartSignal"
#property indicator_type2   DRAW_COLOR_ARROW
#property indicator_color2  clrLightSalmon,clrOrangeRed,clrBlack,clrSteelBlue,clrLightBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  3

#include <PivotCandlesClass.mqh>
//+------------------------------------------------------------------+
//| Common arrays and structures                                     |
//+------------------------------------------------------------------+
//--- Indicator buffers
double   SMA[];            // Values of the Moving Average
double   Signal[];         // Signal values
double   ChartSignal[];    // Location of signals on the chart
double   SignalColor[];    // Signal color array
//--- Calculation class
CPivotCandlesClass PivotCandlesClass;
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,SMA,INDICATOR_DATA);
   SetIndexBuffer(1,ChartSignal,INDICATOR_DATA);
   SetIndexBuffer(2,SignalColor,INDICATOR_COLOR_INDEX);
   SetIndexBuffer(3,Signal,INDICATOR_CALCULATIONS);

//--- set 0 as an empty value
   PlotIndexSetDouble(1,PLOT_EMPTY_VALUE,0);

   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   // If there have not been calculations yet or (!) the new history is uploaded, clean up the calculation object
   if (prev_calculated == 0)
      PivotCandlesClass.CleanupHistory();

   int end_calc_edge = rates_total-1;
   if (prev_calculated >= end_calc_edge)
      return end_calc_edge;

   for(int i=prev_calculated; i<end_calc_edge; i++)
     {
      int signal = PivotCandlesClass.AnalizeNewCandle(time[i],open[i],high[i],low[i],close[i],tick_volume[i],volume[i],spread[i]);
      Signal[i] = signal;
      SMA[i] = PivotCandlesClass.MAValue();

      // Signals are processed, display them on the chart
      // Set the location of our signals...
      if (signal < 0)
         ChartSignal[i]=high[i];
      else if (signal > 0)
         ChartSignal[i]=low[i];
      else
         ChartSignal[i]=0;
      // .. as well as their color
      // Signals have a range of [-2..2], while color indices - [0..4]. Align them
      SignalColor[i]=signal+2;
     }

   // Set the Moving Average value similar to the previous one to prevent it from sharp fall
   SMA[end_calc_edge] = SMA[end_calc_edge-1];

//--- return value of prev_calculated for next call
   return(end_calc_edge);
  }
//+------------------------------------------------------------------+
```

The indicator is ready. Now, let's test it on any of the charts. To do this, install the compiled indicator on the chart. After that, we will see something similar to what is shown in the image below.

![Figure 2. Indicator of "hammer" and "shooting star" candlestick patterns](https://c.mql5.com/2/4/EN_Figure2.png)

Figure 2. Indicator of "hammer" and "shooting star" candlestick patterns

Colored points indicate possible market entries and exits. The colors are selected as follows:

- dark red – sell;
- dark blue – buy;
- light red – closing long position;
- light red – closing short position.

Close signals are formed every time when the price reaches its moving average. The signal is ignored, if there were no positions at that moment.

Now, let's pass to the main topic of the article. We have the indicator with the signal buffer generating only some certain signals. Let's display in a separate window of the same chart how profitable/loss-making these signals can be if actually being followed. The indicator has been developed especially for that case. It can connect to another indicator and open/close virtual positions depending on the incoming signals.

Just like with the previous indicator, we should split the code into two parts - calculation and visual one. Below is the result of a sleepless night but I hope it is worth it. :)

```
//+------------------------------------------------------------------+
//|                                                 BalanceClass.mqh |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
//+------------------------------------------------------------------+
//| Common structures                                                |
//+------------------------------------------------------------------+
// Structure for returning calculation results
// using only return command;
struct BalanceResults
  {
   double balance;
   double equity;
  };
//+------------------------------------------------------------------+
//| Common function                                                  |
//+------------------------------------------------------------------+
//  Function for searching for the indicator handle by its name
int FindIndicatorHandle(string _name)
  {
   // Receive the number of open charts
   int windowsCount = (int)ChartGetInteger(0,CHART_WINDOWS_TOTAL);

   // Search all of them
   for(int w=windowsCount-1; w>=0; w--)
     {
      // How many indicators are attached to the current chart
      int indicatorsCount = ChartIndicatorsTotal(0,w);

      // Search by all chart indicators
      for(int i=0;i<indicatorsCount;i++)
        {
         string name = ChartIndicatorName(0,w,i);
         // If such an indicator is found, return its handle
         if (name == _name)
            return ChartIndicatorGet(0,w,name);
        }
     }

   // If there is no such an indicator, return the incorrect handle
   return -1;
  }
//+------------------------------------------------------------------+
//| Base calculation class                                           |
//+------------------------------------------------------------------+
class CBaseBalanceCalculator
  {
private:
   double            m_position_volume; // Current open position volume
   double            m_position_price;  // Position opening price
   double            m_symbol_points;   // Value of one point for the current symbol
   BalanceResults    m_results;         // Calculation results
public:
   void              CBaseBalanceCalculator(string symbol_name = "");
   void              Cleanup();
   BalanceResults    Calculate( const double _prev_balance,
                                const int    _signal,
                                const double _next_open,
                                const double _next_spread );
  };
//+------------------------------------------------------------------+
//| CBaseBalanceCalculator                                           |
//+------------------------------------------------------------------+
void CBaseBalanceCalculator::CBaseBalanceCalculator(string symbol_name = "")
  {
   // Clean up state variables
   Cleanup();

   // Define point size (because we will calculate the profit in points)
   if (symbol_name == "")
      m_symbol_points = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   else
      m_symbol_points = SymbolInfoDouble(symbol_name, SYMBOL_POINT);
  }
//+------------------------------------------------------------------+
//| Cleanup                                                          |
//+------------------------------------------------------------------+
//| Clean up data on positions and prices                            |
//+------------------------------------------------------------------+
void CBaseBalanceCalculator::Cleanup()
  {
   m_position_volume = 0;
   m_position_price = 0;
  }
//+------------------------------------------------------------------+
//| Calculate                                                        |
//+------------------------------------------------------------------+
//| Main calculation block                                           |
//+------------------------------------------------------------------+
BalanceResults CBaseBalanceCalculator::Calculate(
                                       const double _prev_balance,
                                       const int _signal,
                                       const double _next_open,
                                       const double _next_spread )
  {
   // Clean up the output structure from the previous values
   ZeroMemory(m_results);

   // Initialize additional variables
   double current_price = 0; // current price (bid or ask depending on position direction)
   double profit = 0;        // profit calculated value

   // If there was no signal, the balance remains the same
   if (_signal == 0)
      m_results.balance = _prev_balance;
   // the signal coincides with the direction or no positions are opened yet
   else if (_signal * m_position_volume >= 0)
     {
      // Position already exists, the signal is ignored
      if (m_position_volume != 0)
         // Balance is not changed
         m_results.balance = _prev_balance;
      // No positions yet, buy signal

      else if (_signal == 1)
        {
         // Calculate current ASK price, recalculate price, volume and balance
         current_price = _next_open + _next_spread * m_symbol_points;
         m_position_price = (m_position_volume * m_position_price + current_price) / (m_position_volume + 1);
         m_position_volume = m_position_volume + 1;
         m_results.balance = _prev_balance;
        }
      // No positions yet, sell signal
      else if (_signal == -1)
        {
         // Calculate current BID price, recalculate price, volume and balance
         current_price = _next_open;
         m_position_price = (-m_position_volume * m_position_price + current_price) / (-m_position_volume + 1);
         m_position_volume = m_position_volume - 1;
         m_results.balance = _prev_balance;
        }
      else
         m_results.balance = _prev_balance;
     }
   // Position is set already, the opposite direction signal is received
   else
     {
      // buy signal/close sell position
      if (_signal > 0)
        {
         // Close position by ASK price, recalculate profit and balance
         current_price = _next_open + _next_spread * m_symbol_points;
         profit = (current_price - m_position_price) / m_symbol_points * m_position_volume;
         m_results.balance = _prev_balance + profit;

         // If there is a signal for opening a new position, open it at once
         if (_signal == 1)
           {
            m_position_price = current_price;
            m_position_volume = 1;
           }
         else
            m_position_volume = 0;
        }
      // sell signal/close buy position
      else
        {
         // Close position by BID price, recalculate profit and balance
         current_price = _next_open;
         profit = (current_price - m_position_price) / m_symbol_points * m_position_volume;
         m_results.balance = _prev_balance + profit;

         // If there is a signal for opening a new position, open it at once
         if (_signal == -1)
           {
            m_position_price = current_price;
            m_position_volume = -1;
           }
         else
           m_position_volume = 0;
        }
     }

   // Calculate the current equity
   if (m_position_volume > 0)
     {
      current_price = _next_open;
      profit = (current_price - m_position_price) / m_symbol_points * m_position_volume;
      m_results.equity = m_results.balance + profit;
     }
   else if (m_position_volume < 0)
     {
      current_price = _next_open + _next_spread * m_symbol_points;
      profit = (current_price - m_position_price) / m_symbol_points * m_position_volume;
      m_results.equity = m_results.balance + profit;
     }
   else
      m_results.equity = m_results.balance;

   return m_results;
  }
//+------------------------------------------------------------------+
```

The calculation class is ready. Now, we should implement the indicator display to see how it works.

```
//+------------------------------------------------------------------+
//|                                                      Balance.mq5 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.mql5.com"
#property version   "1.00"
#property indicator_separate_window

#property indicator_buffers 4
#property indicator_plots   3
#property indicator_level1  0.0
#property indicator_levelcolor Silver
#property indicator_levelstyle STYLE_DOT
#property indicator_levelwidth 1
//--- plot Balance
#property indicator_label1  "Balance"
#property indicator_type1   DRAW_COLOR_HISTOGRAM
#property indicator_color1  clrBlue,clrRed
#property indicator_style1  STYLE_DOT
#property indicator_width1  1
//--- plot Equity
#property indicator_label2  "Equity"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrLime
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- plot Zero
#property indicator_label3  "Zero"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGray
#property indicator_style3  STYLE_DOT
#property indicator_width3  1

#include <BalanceClass.mqh>
//+------------------------------------------------------------------+
//| Input and global variables                                       |
//+------------------------------------------------------------------+
input string   iParentName        = "";             // Indicator name for balance calculation
input int      iSignalBufferIndex = -1;            // Signal buffer's index number
input datetime iStartTime         = D'01.01.2012';  // Calculation start date
input datetime iEndTime           = 0;             // Calculation end date
//--- Indicator buffers
double   Balance[];       // Balance values
double   BalanceColor[];  // Color index for drawing the balance
double   Equity[];        // Equity values
double   Zero[];          // Zero value for histogram's correct display
//--- Global variables
double   Signal[1];       // Array for receiving the current signal
int      parent_handle;   // Indicator handle, the signals of which are to be used

CBaseBalanceCalculator calculator; // Object for calculating balance and equity
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Binding indicator buffers
   SetIndexBuffer(0,Balance,INDICATOR_DATA);
   SetIndexBuffer(1,BalanceColor,INDICATOR_COLOR_INDEX);
   SetIndexBuffer(2,Equity,INDICATOR_DATA);
   SetIndexBuffer(3,Zero,INDICATOR_DATA);

   // Search for indicator handle by its name
   parent_handle = FindIndicatorHandle(iParentName);
   if (parent_handle < 0)
     {
      Print("Error! Parent indicator not found");
      return -1;
     }

   return(0);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   // Set the borders for calculating the indicator
   int start_index = prev_calculated;
   int end_index = rates_total-1;

   // Calculate balance and equity values
   for(int i=start_index; i<end_index; i++)
     {
      // Check if the balance calculation corresponds the interval
      if (time[i] < iStartTime)
        {
         Balance[i] = 0;
         Equity[i] = 0;
         continue;
        }
      if (time[i] > iEndTime && iEndTime != 0)
        {
         Equity[i] = (i==0) ? 0 : Equity[i-1];
         Balance[i] = Equity[i];
         continue;
        }

      // Request a signal from the parent indicator
      if(CopyBuffer(parent_handle,iSignalBufferIndex,time[i],1,Signal)==-1) // Copy the indicator main line data
        {
         Print("Data copy error: " + IntegerToString(GetLastError()));
         return(0);  // Finish the function operation and send indicator for the full recalculation
        }

      // Initialize balance and equity calculation
      // Since the signal is formed when the candle is closing, we will be able
      //   to perform any operation only at the next candle's opening price
      BalanceResults results = calculator.Calculate(i==0?0:Balance[i-1], (int)Signal[0], open[i+1], spread[1+1]);

      // Fill out all indicator buffers
      Balance[i] = results.balance;
      Equity[i] = results.equity;
      Zero[i] = 0;
      if (Balance[i] >= 0)
         BalanceColor[i] = 0;
      else
         BalanceColor[i] = 1;
     }

   // Fill out buffers for the last candle
   Balance[end_index] = Balance[end_index-1];
   Equity[end_index] = Equity[end_index-1];
   BalanceColor[end_index] = BalanceColor[end_index-1];
   Zero[end_index] = 0;

   return rates_total;
  }
//+------------------------------------------------------------------+
```

It is finally over! Let's compile it and examine the results.

### Instructions for Use

To evaluate the operation of our newly developed indicator, it should be attached to the chart containing at least one signal indicator. If you followed all steps, then we already have such an indicator – PivotCandles. So, we have to configure the input parameters. Let's see what we should specify:

- **Indicator name for balance calculation** (string) – we should keep in mind that binding of the balance indicator is performed by name. Therefore, this field is mandatory.
- **Signal buffer's index number** (integer) – another critical parameter. The signal indicator may generate several signals according to previously defined algorithm. Therefore, the balance indicator should have the data concerning the buffer's signal it should calculate.
- **Calculation start date** (date/time) – initial date of the balance calculation.
- **Calculation end date** (date/time) – end date of the balance calculation. If the date is not selected (equal to zero), calculation will be carried out till the last bar.

Figure 3 shows configuration of the first two parameters for attaching the balance indicator to the third buffer of PivotCandles indicator. The remaining two parameters can be set to your liking.

![Figure 3. Balance indicator parameters](https://c.mql5.com/2/4/EN_Figure3.png)

Figure 3. Balance indicator parameters

If all previous steps have been performed correctly, you should see an image that is very similar to the one shown below.

![Figure 4. Balance and equity curves generated using PivotCandles indicator's signals](https://c.mql5.com/2/4/EN_Figure4__1.png)

Figure 4. Balance and equity curves generated using PivotCandles indicator's signals

Now, we can try different time frames and symbols and find out the most profitable and loss-making market entries. It should be added that this approach helps to find the market correlations affecting your trading results.

Originally, I wanted to compare the time spent in testing the Expert Advisor based on the same signals with the time spent using the method described above. But then I abandoned the idea, as recalculation of the indicator takes about a second. Such a short time is certainly cannot be reached by the Expert Advisor with its history uploading and ticks generating algorithms yet.

### Conclusion

The method described above is very fast. Besides, it provides the clarity in testing the indicators that generate position open/close signals. It allows traders to analyze the signals and the deposit's responses to them in a single chart window. But it still has its limitations we should be aware of:

- the analyzed indicator's signal buffer should be preliminarily prepared;
- the signals are bound to the new bar's open time;
- no ММ when calculating the balance;

However, despite these shortcomings, I hope that the benefits will be more significant and this testing method will take its place among the other tools designed for analyzing the market behavior and processing the signals generated by the market.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/505](https://www.mql5.com/ru/articles/505)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/505.zip "Download all attachments in the single ZIP archive")

[balanceclass.mqh](https://www.mql5.com/en/articles/download/505/balanceclass.mqh "Download balanceclass.mqh")(8.07 KB)

[balance.mq5](https://www.mql5.com/en/articles/download/505/balance.mq5 "Download balance.mq5")(5.57 KB)

[pivotcandles.mq5](https://www.mql5.com/en/articles/download/505/pivotcandles.mq5 "Download pivotcandles.mq5")(4.14 KB)

[pivotcandlesclass.mqh](https://www.mql5.com/en/articles/download/505/pivotcandlesclass.mqh "Download pivotcandlesclass.mqh")(12.61 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/10569)**
(20)


![Sergey Petruk](https://c.mql5.com/avatar/2012/10/507BFB7A-2783.jpg)

**[Sergey Petruk](https://www.mql5.com/en/users/vspexp)**
\|
13 Nov 2012 at 21:36

**astrohelper:**

This is what happens when several platforms are on the same computer - first open a debugger from an open platform and find a file in it and compile it.


![astrohelper](https://c.mql5.com/avatar/avatar_na2.png)

**[astrohelper](https://www.mql5.com/en/users/astrohelper)**
\|
15 Nov 2012 at 00:54

Thank you, it worked.

![Shital Patil](https://c.mql5.com/avatar/avatar_na2.png)

**[Shital Patil](https://www.mql5.com/en/users/shantala)**
\|
11 Feb 2013 at 03:51

Thanks


![Dennis Ring](https://c.mql5.com/avatar/avatar_na2.png)

**[Dennis Ring](https://www.mql5.com/en/users/7007903)**
\|
13 Feb 2013 at 15:53

I find a slight similarity between 'shooting stars'/'hammer' and the Chakin Indicator

![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
5 Mar 2013 at 04:23

Thanks for this.

I have a problem in the PivotCandles.mq5 indicator's OnCalculate event..  The open,high,low,close array member elements are all returning the same value.  That is the same value in \[i\] for each OHLC index.  Also the time first element in the array is reporting 1971 as the date.  It appears as if the array passed into OnCalculate is not valid.

Any ideas?

![Change Expert Advisor Parameters From the User Panel "On the Fly"](https://c.mql5.com/2/0/avatar__24.png)[Change Expert Advisor Parameters From the User Panel "On the Fly"](https://www.mql5.com/en/articles/572)

This article provides a small example demonstrating the implementation of an Expert Advisor whose parameters can be controlled from the user panel. When changing the parameters "on the fly", the Expert Advisor writes the values obtained from the info panel to a file to further read them from the file and display accordingly on the panel. This article may be relevant to those who trade manually or in semi-automatic mode.

![MQL5 Market Turns One Year Old](https://c.mql5.com/2/0/mql5-market-1year-avatar.png)[MQL5 Market Turns One Year Old](https://www.mql5.com/en/articles/632)

One year has passed since the launch of sales in MQL5 Market. It was a year of hard work, which turned the new service into the largest store of trading robots and technical indicators for MetaTrader 5 platform.

![MetaTrader 4 Expert Advisor exchanges information with the outside world](https://c.mql5.com/2/13/1062_113.jpg)[MetaTrader 4 Expert Advisor exchanges information with the outside world](https://www.mql5.com/en/articles/1361)

A simple, universal and reliable solution of information exchange between МetaТrader 4 Expert Advisor and the outside world. Suppliers and consumers of the information can be located on different computers, the connection is performed through the global IP addresses.

![MetaTrader 4 and MetaTrader 5 Trading Signals Widgets](https://c.mql5.com/2/0/MetaTrader_trading_signal_widget_avatar__1.png)[MetaTrader 4 and MetaTrader 5 Trading Signals Widgets](https://www.mql5.com/en/articles/626)

Recently MetaTrader 4 and MetaTrader 5 user received an opportunity to become a Signals Provider and earn additional profit. Now, you can display your trading success on your web site, blog or social network page using the new widgets. The benefits of using widgets are obvious: they increase the Signals Providers' popularity, establish their reputation as successful traders, as well as attract new Subscribers. All traders placing widgets on other web sites can enjoy these benefits.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mwfsmsxovtflrrdbvovlfjhbkzciwhup&ssn=1769252096627429096&ssn_dr=1&ssn_sr=0&fv_date=1769252096&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F505&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Fast%20Testing%20of%20Trading%20Ideas%20on%20the%20Chart%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925209702522274&fz_uniq=6428866254849709824&sv=2552)

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