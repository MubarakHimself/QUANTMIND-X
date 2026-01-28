---
title: Building A Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (I)
url: https://www.mql5.com/en/articles/15509
categories: Trading Systems, Integration, Indicators, Machine Learning
relevance_score: 8
scraped_at: 2026-01-22T17:45:52.662065
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/15509&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049361109137336862)

MetaTrader 5 / Trading systems


### Core Content:

- [Introduction](https://www.mql5.com/en/articles/15509#para1)
- [The 7 select Famous Strategies Discussion](https://www.mql5.com/en/articles/15509#para2)
- [Incorporation of a magic number.](https://www.mql5.com/en/articles/15509#para3)
- [Implementation of the Trend Following Strategy](https://www.mql5.com/en/articles/15509#para4)
- [Test Results](https://www.mql5.com/en/articles/15509#para5)
- [Conclusion](https://www.mql5.com/en/articles/15509#para6)

### Introduction

From the beginning of this article series, our emphasis has been on aligning our experts with the prevailing sentiment of the daily (D1) candles. The shape of the daily candle has served as the primary guiding feature. However, we needed to scale down to lower timeframes to identify entry levels within the D1 market. For example, at the M1 timeframe, we wanted the market to reach extreme levels on the Relative Strength Index (RSI) to signal potential trades for the Expert Advisor. At this early stage, we did not introduce too many strategies to keep the content easy to understand for beginners.

However, there is a vast collection of strategies to study and integrate into the algorithm of our Trend Constraint Expert Advisor. Today, we will take a close look at some well-known strategies developed by influential figures in trading, whom I refer to as the "Market Fathers." This phase of our discussion will elevate our understanding while still maintaining the original theme of our title. We will also address the limitations of the strategies discussed in our previous development, which primarily focused on RSI and Constraining logic. Additionally, we will explore how to incorporate new strategies into the Expert Advisor.

Before we dive deeper into the seven top strategies, let’s rewind and recap the constraining logic in our code:

A Bullish candlestick condition in MQL5:

```
void OnTick()
{
    // Determine current daily trend (bullish )
    double daily_open = iOpen(_Symbol, PERIOD_D1, 0);
    double daily_close = iClose(_Symbol, PERIOD_D1, 0);

    bool is_bullish = daily_close > daily_open;
}
```

A Bearish candlestick condition in MQL5:

```
void OnTick()
{
    // Determine current daily trend (bearish)
    double daily_open = iOpen(_Symbol, PERIOD_D1, 0);
    double daily_close = iClose(_Symbol, PERIOD_D1, 0);


    bool is_bearish = daily_close < daily_open;
}
```

In the code above, the sentiment from the higher timeframe serves as our constraining factor. In this, case we opted D1 timeframe.

**Implementation of the constraint condition**:

Now that we have established the bearish and bullish sentiment of our daily candlestick, we can utilize '' **if()''** function alongside our entry condition logic.

For a Bullish Day:

```
 if (is_bullish)
    {
        // Logic for bullish trend
        Print("The daily trend is bullish.");

   // You can add your trading logic here, for example:
        // if (OrderSelect(...)) { /* Place buy order */ }
    }
```

For a Bearish Day:

```
 if (is_bearish)
    {
        // Logic for bearish trend
        Print("The daily trend is bearish.");
        // You can add your trading logic here, for example:
        // if (OrderSelect(...)) { /* Place sell order */ }
    }
```

By the end of this discussion you may be able to note:

1\. Other strategies and the man behind them.

2\. Incorporation of new strategy into modules of the existing Expert Advisor.

3\. Implementation of a magic number.

In summary, the three key points outlined above will provide traders with a better understanding of various strategies and the motivations behind them, as well as insights into implementing MQL5 to integrate new strategies into existing code.

Before we begin developing the main code, let’s focus on discussing these strategies based on my research in the next segment. You'll see the necessity to expand our Expert Advisor strategies, especially as the market is constantly evolving. These compound strategies are essential for adapting to any market scenario. When one strategy fails, there’s always another that can be employed. I have made every effort to uncover the mathematical aspects underpinning each strategy, as these mathematical functions serve as a foundational element for efficient algorithm development.

After reviewing the recent Expert Advisor I encountered some warning errors, which you may have noticed if you followed along. Please see the image below:

![Warning error](https://c.mql5.com/2/95/error.PNG)

A compilation warning

That error was on line 78 and column 28 see it highlighted below:

![warning](https://c.mql5.com/2/95/erroron_code.PNG)

Warning on line 78, column 28

This warning is one of the simplest to fix, as it is self-explanatory. Here’s a code snippet with the fix:

```
long position_type = PositionGetInteger(POSITION_TYPE);
```

The fix above involved replacing **int** with **long.**

Our latest Trend Constraint Expert Advisor source code is provided here, but you can also revisit the [previous article](https://www.mql5.com/en/articles/15322) for the source file. Be sure to practice debugging the code as we did above before implementing the new features.

```
//+------------------------------------------------------------------+
//|                                      Trend Constraint Expert.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>  // Include the trade library

// Input parameters
input int    RSI_Period = 14;            // RSI period
input double RSI_Overbought = 70.0;      // RSI overbought level
input double RSI_Oversold = 30.0;        // RSI oversold level
input double Lots = 0.1;                 // Lot size
input double StopLoss = 100;             // Stop Loss in points
input double TakeProfit = 200;           // Take Profit in points
input double TrailingStop = 50;          // Trailing Stop in points

// Global variables
double rsi_value;
int rsi_handle;
CTrade trade;  // Declare an instance of the CTrade class

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Create an RSI indicator handle
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   if (rsi_handle == INVALID_HANDLE)
     {
      Print("Failed to create RSI indicator handle");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Release the RSI indicator handle
   IndicatorRelease(rsi_handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Determine current daily trend (bullish or bearish)
   double daily_open = iOpen(_Symbol, PERIOD_D1, 0);
   double daily_close = iClose(_Symbol, PERIOD_D1, 0);

   bool is_bullish = daily_close > daily_open;
   bool is_bearish = daily_close < daily_open;

   // Get the RSI value for the current bar
   double rsi_values[];
   if (CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0)
     {
      Print("Failed to get RSI value");
      return;
     }
   rsi_value = rsi_values[0];

   // Close open positions if the trend changes
   for (int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
        {
         long position_type = PositionGetInteger(POSITION_TYPE);
         ulong ticket = PositionGetInteger(POSITION_TICKET);  // Get the position ticket

         if ((position_type == POSITION_TYPE_BUY && is_bearish) ||
             (position_type == POSITION_TYPE_SELL && is_bullish))
           {
            trade.PositionClose(ticket);  // Use the ulong variable directly
           }
        }
     }

   // Check for buy condition (bullish trend + RSI oversold)
   if (is_bullish && rsi_value < RSI_Oversold)
     {
      // No open positions? Place a buy order
      if (PositionsTotal() == 0)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = price - StopLoss * _Point;
         double tp = price + TakeProfit * _Point;

         // Open a buy order
         trade.Buy(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Buy");
        }
     }

   // Check for sell condition (bearish trend + RSI overbought)
   if (is_bearish && rsi_value > RSI_Overbought)
     {
      // No open positions? Place a sell order
      if (PositionsTotal() == 0)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + StopLoss * _Point;
         double tp = price - TakeProfit * _Point;

         // Open a sell order
         trade.Sell(Lots, _Symbol, price, sl, tp, "TrendConstraintExpert Sell");
        }
     }

   // Apply trailing stop
   for (int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if (PositionSelect(PositionGetSymbol(i)))  // Corrected usage
        {
         double price = PositionGetDouble(POSITION_PRICE_OPEN);
         double stopLoss = PositionGetDouble(POSITION_SL);
         double current_price;

         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
           {
            current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            if (current_price - price > TrailingStop * _Point)
              {
               if (stopLoss < current_price - TrailingStop * _Point)
                 {
                  trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price - TrailingStop * _Point, PositionGetDouble(POSITION_TP));
                 }
              }
           }
         else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
           {
            current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            if (price - current_price > TrailingStop * _Point)
              {
               if (stopLoss > current_price + TrailingStop * _Point || stopLoss == 0)
                 {
                  trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price + TrailingStop * _Point, PositionGetDouble(POSITION_TP));
                 }
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

### The 7 select Famous Strategies discussion

Trading in the Forex market requires a blend of skill and strategy. Over the years, various approaches have emerged that can help traders maximize their chances of success. Here, I will briefly delve into seven effective Forex trading strategies, each grounded in essential mathematical principles and theories developed by influential economists and traders. Once the strategy is absorbed into mind, it becomes easier to implement the coding skill and create an Algorithm for the strategy.

### (i) Trend Following Strategy

The trend following strategy is based on the principles of [Charles Dow,](https://en.wikipedia.org/wiki/Charles_Dow "https://en.wikipedia.org/wiki/Charles_Dow") who formulated Dow Theory in the late 19th century (1896). Dow believed that markets move in trends, and identifying these trends can provide substantial trading opportunities. The mathematics behind trend following often employs moving averages, where traders use the simple moving average (SMA) to smooth out price fluctuations.

The actual formula for the Simple Moving Average (SMA) is:

![SMA](https://c.mql5.com/2/95/SMA.PNG)

Where:

- **N** is the number of periods (or data points).
- **(P\_i)** is the price (or value) at the (i-th) period.

This formula calculates the average price over N periods. Supporting research, such as the works of [Richard Dennis and his "Turtles"](https://en.wikipedia.org/wiki/Richard_Dennis "https://en.wikipedia.org/wiki/Richard_Dennis") experiment in the 1980s, demonstrates that trend following can yield significant profits, particularly in strong trending markets.

**(ii) Range Trading**

Range trading is grounded in the concepts of support and resistance levels, crucial to technical analysis. These concepts were well-documented by [J. Welles Wilder](https://en.wikipedia.org/wiki/J._Welles_Wilder_Jr. "https://en.wikipedia.org/wiki/J._Welles_Wilder_Jr.") in his book "New Concepts in Technical Trading Systems" (1978). Traders identify support as a price level where buying interest prevents declines, while resistance hinders price increases. The mathematical basis for calculating pivot points, a common method for identifying these levels, is given by:

The actual formula for the Pivot Point (PP) is:

![pivot](https://c.mql5.com/2/95/Pivot.PNG)​

Where:

- **High** is the highest price of the previous trading period.
- **Low** is the lowest price of the previous trading period.
- **Close** is the closing price of the previous trading period.

This formula calculates the central pivot point used in technical analysis to determine potential support and resistance levels for the upcoming trading period. Researches, such as that by John Murphy in "Technical Analysis of the Financial Markets" (1986), emphasizes that support and resistance levels often serve as psychological indicators that can help traders make informed decisions.

(iii) Breakout Trading

Breakout trading capitalizes on the belief that substantial price movements occur when prices break through established support or resistance levels. [Richard Donchian](https://en.wikipedia.org/wiki/Richard_Donchian "https://en.wikipedia.org/wiki/Richard_Donchian") introduced this strategy in the 1950s, and his work with Donchian Channels emphasizes using volatility to identify breakout points. Traders calculate Donchian Channels using the highest high (HH) and the lowest low (LL) over a specified period:

(Upper Band = HH)

(Lower Band = LL)

![Donchian Channels](https://c.mql5.com/2/95/Donchian_Channel.PNG)

Donchian Channel

Research in behavioral finance supports the idea that traders often follow breakouts due to herd behavior, which reinforces momentum in the direction of the breakout. The work published by researchers such as [Daniel Kahneman and Amos Tversky](https://en.wikipedia.org/wiki/Amos_Tversky "https://en.wikipedia.org/wiki/Amos_Tversky") in their 1979 paper, "Prospect Theory: An Analysis of Decision under Risk," highlights how psychological factors influence trading behavior during breakouts.

### (iv) Carry Trade

The carry trade leverages the interest rate differentials between two currencies, based on theories of Interest Rate Parity (IRP). This principle is rooted in economic theories formulated in the early 20th century, notably by economists like [Paul Samuelson in works like "Foundations of Economic Analysis" (1947).](https://www.mql5.com/go?link=https://www.britannica.com/money/Paul-Samuelson "https://www.britannica.com/money/Paul-Samuelson")

The basic formula for calculating the expected return on a carry trade is given by:

![ER](https://c.mql5.com/2/95/Expected_Return.PNG)

where:

- **(i\_high)** and **(i\_low)** are the interest rates of the high-yield and low-yield currencies, respectively.
- **(Delta S)** represents the expected change in the exchange rate.

[Kenneth Rogoff](https://www.mql5.com/go?link=https://scholar.harvard.edu/rogoff/publications/blessing-or-curse-foreign-and-underground-demand-euro-notes "https://scholar.harvard.edu/rogoff/publications/blessing-or-curse-foreign-and-underground-demand-euro-notes") in his 1998 paper, highlights how deviations from interest rate parity can create profitable trading opportunities.

(v) Scalping

Scalping is a high-frequency trading strategy that exploits small price movements, underpinned by Market Microstructure Theory (MMT). This theory has evolved through the work of various researchers, including [Lawrence Harris](https://www.mql5.com/go?link=https://www.marshall.usc.edu/personnel/lawrence-eugene-harris "https://www.marshall.usc.edu/personnel/lawrence-eugene-harris") in "Market Making and the Financial Performance of Nasdaq Firms" (1998). Scalpers execute multiple trades within short timeframes, seeking to profit from minute price changes.

Important metrics for scalpers include bid-ask spreads and order flow analysis, often calculated through various metrics like:

![profit](https://c.mql5.com/2/95/Profit.PNG)

Where:

- **(Selling Price)** is the price at which the asset is sold.

- **(Buying Price)** is the price at which the asset was originally purchased.

- **(Transaction Costs)** are any fees or costs associated with the buying and selling process (e.g., broker fees, taxes).

Risk management becomes critical, as a few unfavorable trades can lead to significant losses.

(vi) Fundamental Analysis

Fundamental analysis involves examining economic indicators and geopolitical factors that impact currency valuations. Pioneered by [Benjamin Graham](https://en.wikipedia.org/wiki/Benjamin_Graham "https://en.wikipedia.org/wiki/Benjamin_Graham") and [David Dodd](https://en.wikipedia.org/wiki/David_Dodd "https://en.wikipedia.org/wiki/David_Dodd") in "Security Analysis" (1934), this approach emphasizes the importance of assessing the intrinsic value of currencies. Traders utilize various indicators, such as Gross Domestic Product (GDP) growth and unemployment rates, to make informed decisions. Mathematical techniques, such as the Price-to-Earnings (P/E) ratio for stocks, can also be adapted;

![P^E](https://c.mql5.com/2/95/PperE.PNG)

Where:

- **(Market Price per share)** is the current price of one share of the company's stock.

- **Earnings per Share (EPS)** is the company's profit allocated to each outstanding share of common stock.

Extensive research supports the efficacy of fundamental analysis, especially during significant economic reports or central bank announcements, which often lead to substantial market movement.

### (vii) Using Technical Indicators

Technical analysis, grounded in the ideas of [Charles Dow](https://en.wikipedia.org/wiki/Charles_Dow "https://en.wikipedia.org/wiki/Charles_Dow") and others, assumes that past price movements can provide insights into future behavior. Various technical indicators, like the Relative Strength Index (RSI), are employed to measure market momentum. The RSI is calculated using the formula:

![RSI](https://c.mql5.com/2/95/RSI.PNG)

where:

- **(RS)** represents the average gains and losses over a specified period.

![RS](https://c.mql5.com/2/95/RS.PNG)

Supporting research, such as that by [Thomas Bulkowski](https://www.mql5.com/go?link=https://thepatternsite.com/about.html "https://thepatternsite.com/about.html") in "Encyclopedia of Candlestick Charts" (2008), shows that certain patterns and indicators can enhance trading precision, allowing traders to anticipate market movements based on historical price action. Each of these strategies is underpinned by foundational theories, mathematical principles, and extensive research that inform traders' decisions in the Forex market. Understanding the intricacies of these strategies can significantly enhance a trader's ability to navigate the complexities of Forex trading, ultimately leading to more informed and profitable trading outcomes. By employing these strategies thoughtfully, traders can develop a robust trading approach tailored to their individual preferences and risk tolerances.

### Incorporation of a Magic Number

Magic numbers in MQL5 serve as a pivotal organizational tool, akin to a unique signature that an Expert Advisor (EA) uses to mark its trades. Think of it as each EA having its own personal identifier that it stamps on every trade it opens. This is particularly useful in accounts running numerous strategies concurrently, as it helps the EA keep its operations cleanly segregated from others.

By defining the magic number as an input parameter, you afford flexibility, allowing adjustments in EA settings without delving back into the source code. When the EA scans through open positions, it uses this number to quickly find which trades are it's.

The function **PositionGetInteger(POSITION\_MAGIC)** retrieves the magic number attached to each trade, enabling the EA to decide its next step: whether to amend, monitor, or close a position. For a trader or developer, this means less worry about accidental interference between different strategies and easier debugging, as each trade can be traced back to its source strategy. Thus, incorporating and managing magic numbers effectively ensures that your trading strategies remain precise and well-organized in financial markets.

Here is a code snippet showing the incorporation:

```
input int MagicNumber = 12345678; // Define a unique magic number for the EA

// Within the OnTick() or relevant function
for (int i = 0; i < PositionsTotal(); i++) {
    if (PositionGetSymbol(i) == _Symbol && PositionSelect(PositionGetSymbol(i))) {
        long currentMagicNumber = PositionGetInteger(POSITION_MAGIC);

        if (currentMagicNumber == MagicNumber) { // Check if the position belongs to this EA
            // Perform actions on this position, like closing or modifying
        }
    }
}

// When opening a new trade
double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
double sl = price - StopLoss * _Point;
double tp = price + TakeProfit * _Point;

if (trade.Buy(Lots, _Symbol, price, sl, tp, "TrendFollowing Buy")) {
    Print("Buy order placed with Magic Number: ", MagicNumber);
} else {
    Print("Error placing buy order: ", GetLastError());
```

### Implementation of Trend Following Strategy

From the above select strategies most of them cannot be implemented in our program as they lack some data required to use them at the current time for example, Carry Trade, is not feasible with only this code, as it involves interest rate differentials, and fundamental analysis requires external economic data feeds detailing GDP, employment rates, etc. To save for further exploration in the future, we will focus on the most feasible strategy.

Trend following is an approach that identifies and trades in the direction of a sustained market trend, using techniques like moving averages to confirm trend direction. Our expert code already uses RSI to gauge overbought or oversold conditions, but integrating moving averages can further refine the decision-making process. Here, we use a short-term moving average (50-period) and a long-term moving average (200-period) to determine the trend's direction. If the short-term average crosses above the long-term average, it's a signal for an uptrend; conversely, if it crosses below, it indicates a downtrend. The integration of this method helps confirm bullish or bearish trends, enhancing the accuracy of trade entries alongside RSI conditions.

Here is our code snippet for calculating moving averages before integration into the main program.

```
//Calculate moving averages
double ma_short = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE, 0);
double ma_long = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE, 0);

//Determine the trend  direction
bool is_uptrend = ma_short > ma_long;
bool is_downtrend = ma_short < ma_long;

if (is_bullish && is_uptrend && rsi_value < RSI_Oversold)
{
   // Implement buy order logic here
}

if (is_bearish && is_downtrend && rsi_value > RSI_Overbought)
{
   // Implement sell order logic here
}
```

To effectively incorporate a trend following strategy into an existing Expert Advisor (EA), we begin by enhancing the initialization and processing logic within its modules. Starting with the **OnInit()** function, we introduce the calculation of two critical Exponential Moving Averages (EMAs), typically a 50-period for short-term trends and a 200-period for long-term trends, using the iMA function.

As we progress to the **OnTick()** method, these EMAs are recalculated with each new market tick, enabling the EA to determine the current market trend direction; a crossover where the short-term EMA rises above the long-term signals an uptrend, whereas a crossover where it falls below indicates a downtrend.

Integrating these trend signals with the existing RSI analysis, we craft conditions where the algorithm opens buy orders during confirmed uptrend when RSI is oversold, and sell orders during downtrend when RSI is overbought.

Using the **CTrade** class for execution, we meticulously set parameters for stop loss, take profit, and a trailing stop, ensuring robust risk management. Additionally, within the same **OnTick()** loop, a system is incorporated to gracefully close any existing positions should the identified trend reverse, thus aligning every trade with the verified market direction. Through these strategic updates, the EA is adeptly tailored to employ a trend following strategy, seamlessly integrating it with its current functionalities and enhancing overall trading efficacy.

Here, is a fully integrated program:

```
//+------------------------------------------------------------------+
//|                                      Trend Constraint Expert.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>  // Include the trade library

// Input parameters
input int    RSI_Period = 14;            // RSI period
input double RSI_Overbought = 70.0;      // RSI overbought level
input double RSI_Oversold = 30.0;        // RSI oversold level
input double Lots = 0.1;                 // Lot size
input double StopLoss = 100;             // Stop Loss in points
input double TakeProfit = 200;           // Take Profit in points
input double TrailingStop = 50;          // Trailing Stop in points
input int    MagicNumber = 12345678;     // Magic number for this EA

// Global variables
double rsi_value;
int rsi_handle;
CTrade trade;  // Declare an instance of the CTrade class

// Variables for moving averages
double ma_short;
double ma_long;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Create an RSI indicator handle
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   if (rsi_handle == INVALID_HANDLE)
     {
      Print("Failed to create RSI indicator handle");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Release the RSI indicator handle
   IndicatorRelease(rsi_handle);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Calculate moving averages
   ma_short = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
   ma_long  = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);

   // Determine trend direction
   bool is_uptrend = ma_short > ma_long;
   bool is_downtrend = ma_short < ma_long;

   // Get the RSI value for the current bar
   double rsi_values[];
   if (CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0)
     {
      Print("Failed to get RSI value");
      return;
     }
   rsi_value = rsi_values[0];

   // Close open positions if the trend changes
   for (int i = 0; i < PositionsTotal(); i++) // Correct loop initialization
     {
      if (PositionGetSymbol(i) == _Symbol && PositionSelect(PositionGetSymbol(i)))  // Select position by symbol
        {
         long position_type = PositionGetInteger(POSITION_TYPE);
         long currentMagicNumber = PositionGetInteger(POSITION_MAGIC);
         ulong ticket = PositionGetInteger(POSITION_TICKET);

         if (currentMagicNumber == MagicNumber) // Ensure only this EA's orders are checked
           {
            if ((position_type == POSITION_TYPE_BUY && is_downtrend) ||
                (position_type == POSITION_TYPE_SELL && is_uptrend))
              {
               trade.PositionClose(ticket);
              }
           }
        }
     }

   // Check for buy condition (uptrend + RSI oversold)
   if (is_uptrend && rsi_value < RSI_Oversold)
     {
      // No open positions? Place a buy order
      bool open_position = false;

      for (int i = 0; i < PositionsTotal(); i++) // Correct loop initialization
        {
         if (PositionGetSymbol(i) == _Symbol && PositionSelect(PositionGetSymbol(i)))
           {
            long currentMagicNumber = PositionGetInteger(POSITION_MAGIC);

            if (currentMagicNumber == MagicNumber)
              {
               open_position = true;
               break;
              }
           }
        }

      if (!open_position)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = price - StopLoss * _Point;
         double tp = price + TakeProfit * _Point;

         // Open a buy order
         if (trade.Buy(Lots, _Symbol, price, sl, tp, "TrendFollowing Buy"))
            Print("Buy order placed with Magic Number: ", MagicNumber);
         else
            Print("Error placing buy order: ", GetLastError());
        }
     }

   // Check for sell condition (downtrend + RSI overbought)
   if (is_downtrend && rsi_value > RSI_Overbought)
     {
      // No open positions? Place a sell order
      bool open_position = false;

      for (int i = 0; i < PositionsTotal(); i++) // Correct loop initialization
        {
         if (PositionGetSymbol(i) == _Symbol && PositionSelect(PositionGetSymbol(i)))
           {
            long currentMagicNumber = PositionGetInteger(POSITION_MAGIC);

            if (currentMagicNumber == MagicNumber)
              {
               open_position = true;
               break;
              }
           }
        }

      if (!open_position)
        {
         double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = price + StopLoss * _Point;
         double tp = price - TakeProfit * _Point;

         // Open a sell order
         if (trade.Sell(Lots, _Symbol, price, sl, tp, "TrendFollowing Sell"))
            Print("Sell order placed with Magic Number: ", MagicNumber);
         else
            Print("Error placing sell order: ", GetLastError());
        }
     }

   // Apply trailing stop
   for (int i = 0; i < PositionsTotal(); i++) // Correct loop initialization
     {
      if (PositionGetSymbol(i) == _Symbol && PositionSelect(PositionGetSymbol(i))) // Select position by symbol
        {
         long currentMagicNumber = PositionGetInteger(POSITION_MAGIC);

         if (currentMagicNumber == MagicNumber) // Apply trailing stop only to this EA's positions
           {
            double price = PositionGetDouble(POSITION_PRICE_OPEN);
            double stopLoss = PositionGetDouble(POSITION_SL);
            double current_price;

            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
              {
               current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
               if (current_price - price > TrailingStop * _Point)
                 {
                  if (stopLoss < current_price - TrailingStop * _Point)
                    {
                     trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price - TrailingStop * _Point, PositionGetDouble(POSITION_TP));
                    }
                 }
              }
            else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
              {
               current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
               if (price - current_price > TrailingStop * _Point)
                 {
                  if (stopLoss > current_price + TrailingStop * _Point || stopLoss == 0)
                    {
                     trade.PositionModify(PositionGetInteger(POSITION_TICKET), current_price + TrailingStop * _Point, PositionGetDouble(POSITION_TP));
                    }
                 }
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
```

### Test Results

I tested the Expert Advisor on the B00M 500 Index and observed the orders. However, there's still a lot to be done to prevent multiple orders from being placed simultaneously. The performance on a $10,000 demo account shows positive outcomes, but the results may not be the same with real and smaller funded accounts. It's important to note that this system does not guarantee profits; it is intended for educational purposes, experiments, and research.

![Expert Launch](https://c.mql5.com/2/95/terminal64_UvtqZb9bv4.gif)

Expert Advisor Launch on Boom 500 index

![tester](https://c.mql5.com/2/95/Graph.PNG)

Boom 500 index tester result 2019

### Conclusion

In conclusion, our discussion on building a Multiple Strategies Expert Advisor using MQL5 focused on understanding the origins of the top existing strategies and integrating a trend-following strategy alongside key elements such as the Relative Strength Index (RSI). The trend-following component utilizes moving averages to determine the market trend direction, enabling traders to align their positions with longer-term movements that are often more stable and potentially more profitable. By employing these moving averages, our system can identify uptrend and downtrend facilitating informed trading decisions based on well-defined market directions.

The strategic incorporation of a trend-following methodology complements the use of RSI, which is essential for identifying overbought and oversold market conditions. Together, these components create a well-rounded approach that captures both trend momentum and aligns with the foundational concept of the series, Trend Constraint. Another critical enhancement made during our development was the integration of magic numbers—an initially missing feature that is crucial for precise trade management in environments with multiple strategies. Magic numbers serve as unique identifiers, helping to keep strategies separate and allowing traders to track and manage each one independently.

Of the seven strategies we've considered, we have only implemented one so far, leaving many potential developments for future exploration. Compile and test the attached source file to evaluate its performance and gather ideas for implementation in your own projects. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15509.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_Expert.mq5](https://www.mql5.com/en/articles/download/15509/trend_constraint_expert.mq5 "Download Trend_Constraint_Expert.mq5")(7.7 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/473806)**

![Creating an MQL5-Telegram Integrated Expert Advisor (Part 7): Command Analysis for Indicator Automation on Charts](https://c.mql5.com/2/95/Creating_an_MQL5-Telegram_Integrated_Expert_Advisor_Part_7__LOGO.png)[Creating an MQL5-Telegram Integrated Expert Advisor (Part 7): Command Analysis for Indicator Automation on Charts](https://www.mql5.com/en/articles/15962)

In this article, we explore how to integrate Telegram commands with MQL5 to automate the addition of indicators on trading charts. We cover the process of parsing user commands, executing them in MQL5, and testing the system to ensure smooth indicator-based trading

![Example of new Indicator and Conditional LSTM](https://c.mql5.com/2/95/Example_of_new_Indicator_and_Conditional_LSTM__LOGO.png)[Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)

This article explores the development of an Expert Advisor (EA) for automated trading that combines technical analysis with deep learning predictions.

![HTTP and Connexus (Part 2): Understanding HTTP Architecture and Library Design](https://c.mql5.com/2/99/http60x60__2.png)[HTTP and Connexus (Part 2): Understanding HTTP Architecture and Library Design](https://www.mql5.com/en/articles/15897)

This article explores the fundamentals of the HTTP protocol, covering the main methods (GET, POST, PUT, DELETE), status codes and the structure of URLs. In addition, it presents the beginning of the construction of the Conexus library with the CQueryParam and CURL classes, which facilitate the manipulation of URLs and query parameters in HTTP requests.

![How to develop any type of Trailing Stop and connect it to an EA](https://c.mql5.com/2/78/How_to_make_any_type_of_Trailing_Stop____LOGO.png)[How to develop any type of Trailing Stop and connect it to an EA](https://www.mql5.com/en/articles/14862)

In this article, we will look at classes for convenient creation of various trailings, as well as learn how to connect a trailing stop to any EA.

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/15509&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049361109137336862)

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