---
title: Building a Candlestick Trend Constraint Model (Part 9): Multiple Strategies Expert Advisor (III)
url: https://www.mql5.com/en/articles/16549
categories: Trading, Integration
relevance_score: 8
scraped_at: 2026-01-22T17:44:31.956353
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/16549&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049344689477364173)

MetaTrader 5 / Examples


### Introduction

In algorithmic trading, identifying optimal entry points within a prevailing trend remains a significant challenge, as many strategies struggle to capture the right moment or generate frequent false signals, resulting in suboptimal performance. This issue is particularly pronounced in daily trends, where minor fluctuations can disrupt execution accuracy.

Divergence offers a robust solution by acting as a filter to identify potential reversals or continuations through discrepancies between price movements and momentum indicators. By integrating divergence detection into the Trend Constraint Expert Advisor, traders can significantly enhance their precision in pinpointing entry levels.

This approach improves trade accuracy and also ensures consistent and efficient trading when combined with the advanced capabilities of MQL5. In this article, we will explore the fundamentals of divergence, the steps to integrate it into MQL5 Expert Advisors, enhancements to the Trend Constraint Expert Advisor with new trade execution conditions, and insights from back-testing to demonstrate its practical application.

Core Content:

1. [The Fundamentals of Divergence](https://www.mql5.com/en/articles/16549#para2)
2. [Steps to Integrate Divergence Detection](https://www.mql5.com/en/articles/16549#para3)
3. [Enhancements to the Trend Constraint Expert Advisor: Implementing new trade execution conditions to leverage divergence](https://www.mql5.com/en/articles/16549#para4).
4. [Back-testing Results and Practical Application](https://www.mql5.com/en/articles/16549#para5)
5. [Conclusion](https://www.mql5.com/en/articles/16549#para6)

### The Fundamentals of Divergence

Divergence is a key concept in technical analysis, providing traders with insights into potential price reversals or continuations by comparing price movements with an indicator's direction.

Significance of Divergence:

Divergence happens when the price of an asset moves in opposition to a technical indicator, often signaling a weakening trend or an impending reversal. This concept is particularly useful in identifying when a trend might be due for a correction or a complete turnabout.

Types of Divergence:

1. Bullish Divergence is noted when the price of the asset hits lower lows, but the indicator, like RSI, starts showing higher lows, suggesting that downward momentum is waning. This could mean that selling pressure is diminishing, potentially setting up for an upward price movement.
2. Bearish Divergence is observed when the price achieves higher highs, but the indicator shows lower highs, indicating that the upward momentum is decreasing. This might foreshadow a decline in price.

Background Check:

Divergence is a pivotal concept in technical analysis, influencing market behavior and trader strategies. Bart and Masse (1981), in " [Divergence of Opinion and Risk](https://www.mql5.com/go?link=https://ideas.repec.org/a/cup/jfinqa/v16y1981i01p23-34_00.html "https://ideas.repec.org/a/cup/jfinqa/v16y1981i01p23-34_00.html")," emphasize how discrepancies in market opinion can increase risk and price volatility, reflecting divergence's role in technical analysis.

Empirical evidence from [Tilehnouei and Shivaraj](https://www.mql5.com/go?link=https://www.researchgate.net/publication/274138586_A_Comparative_Study_of_Two_Technical_Analysis_Tools_Moving_Average_Convergence_and_Divergence_VS_Relative_Strength_Index_A_Case_Study_of_HDFC_Bank_ltd_listed_in_National_Stock_Exchange_of_India_NSE "https://www.researchgate.net/publication/274138586_A_Comparative_Study_of_Two_Technical_Analysis_Tools_Moving_Average_Convergence_and_Divergence_VS_Relative_Strength_Index_A_Case_Study_of_HDFC_Bank_ltd_listed_in_National_Stock_Exchange_of_India_NSE")(2013) suggests that tools like MACD may outperform RSI in certain contexts, offering valuable insights into market momentum through divergence signals. From this research, we can say the integration of divergence with other indicators, such as the interplay between RSI, MACD, and price action, reinforces its utility in a comprehensive trading approach, as supported by various industry sources.

In the next section, we will take a practical step to implement the idea of divergence in our EA development.

### Steps to Integrate Divergence Detection

To incorporate divergence detection into an MQL5 Expert Advisor (EA), we start by calculating Relative Strength Index (RSI) values using functions like _iRSI()_ and comparing them with price action. Significant price extremes are identified using _iHigh()_ and _iLow()_ over a set period. For, this project, I will classify divergence into two types: regular (reversal) and hidden.

Regular Divergence:

Regular divergence signals potential trend reversals, where a bullish setup occurs when price makes a lower low while the indicator forms a higher low, and a bearish setup arises when price achieves a higher high, but the indicator shows a lower high.

```
// Regular divergence conditions in code
bool CheckRegularBearishDivergence(int period = 14, ENUM_TIMEFRAMES timeframe = PERIOD_H1)
{
    double priceHigh1 = iHigh(_Symbol, timeframe, 2);
    double priceHigh2 = iHigh(_Symbol, timeframe, 8);
    double rsiHigh1 = iRSI(_Symbol, timeframe, period, PRICE_CLOSE, 2);
    double rsiHigh2 = iRSI(_Symbol, timeframe, period, PRICE_CLOSE, 8);

    if(priceHigh1 > priceHigh2 && rsiHigh1 < rsiHigh2) return true;
    return false;
}
```

To illustrate these regular divergences, here are two images depicting bullish divergence and bearish divergence, respectively:

![Boom 300 IndexH4_Bullish Divergence](https://c.mql5.com/2/134/Boom_300_IndexH4_Bullish_Divergence1__2.png)

Boom 300 IndexH4-Bullish Divergence: (A lower low price B against a higher low RSI value at D)

![Boom 300 IndexH4_Bearish_Divergence](https://c.mql5.com/2/134/Boom_300_IndexH4_Bearish_Divergencei__2.png)

Boom 300 IndexH4-Bearish Divergence: (A higher high price B against a Lower High RSI value at D)

Hidden Divergence:

Hidden divergence, on the other hand, suggests trend continuation. A hidden bullish divergence occurs in an uptrend when price makes a higher low while the indicator forms a lower low, while a hidden bearish divergence appears in a downtrend when price makes a lower high and the indicator forms a higher high.

```
//RSI and Price Levels declaration and hidden divergence condition
bool CheckHiddenBullishDivergence(int period = 14, ENUM_TIMEFRAMES timeframe = PERIOD_H1)
{
    double priceLow1 = iLow(_Symbol, timeframe, 2);
    double priceLow2 = iLow(_Symbol, timeframe, 8);
    double rsiLow1 = iRSI(_Symbol, timeframe, period, PRICE_CLOSE, 2);
    double rsiLow2 = iRSI(_Symbol, timeframe, period, PRICE_CLOSE, 8);

    if(priceLow1 > priceLow2 && rsiLow1 < rsiLow2) return true;
    return false;
}
```

Below are images illustrating bullish hidden divergence and bearish hidden divergence. Be sure to compare these with the descriptions provided earlier, and practice identifying similar patterns on your own charts.

![Boom 300 IndexH4_Bullish_Hidden_ Divergence](https://c.mql5.com/2/134/Boom_300_IndexH4_Bullish_Hidden__Divergence1__2.png)

Boom 300 IndexH4: Bullish Hidden Divergence

![Boom 300 IndexH4_Bearish_Hidden_Divergence](https://c.mql5.com/2/134/Boom_300_IndexH4_Bearish_Hidden_Divergencei__2.png)

Boom 300 IndexH4: Bearish Hidden Divergence

Integrating these divergence types into the EA requires coding conditions to detect them on each tick or bar close, using tools like _iRSI()_ or _iMACD()_ for indicator calculations. Once detected, divergence signals are aligned with trend constraints, identified using daily market sentiment.

### Enhancements to the Trend Constraint Expert Advisor: Implementing new trade execution conditions to leverage divergence

About the above section, when divergence is finally detected, we will need an additional indicator to confirm and signal order execution process. There are many options for this, but we will consider implementing MACD and RSI for that. Here is a list of other optional confirmation indicators:

1. Bollinger bands
2. Stochastic Oscillator
3. On Balance Volume (OBV) and Volume Weighted Average Price (VWAP)
4. Average Dynamic Index (ADX)

Moving Average Convergence Divergence (MACD):

Why Use It: MACD can confirm momentum changes. If there is a detected divergence with RSI, MACD can provide a secondary confirmation of trend strength or weakening.

How it works:

The EA will look for MACD line crossovers or histogram changes that align with divergence findings. For instance, a bearish divergence might be confirmed if the MACD line crosses below the signal line or the histogram starts decreasing.

Highlights of the MACD indicator:

To have a preview of the inbuilt indicator such as the MACD we can access within the Examples' folder under the Indicators' folder in the MetaEditor 5 software, check the explainer image below.

![Access MACD source in MetaEditor 5](https://c.mql5.com/2/134/ShareX_do2vX0uprO__2.gif)

MetaEditor 5: Accessing the MACD source file

The reason for accessing the code is for us to easily figure out how the buffers are designed so that we can easily adapt them in our Expert Advisor. Below is a code snippet for the buffered declaratiions within the indicator that we are interested in.

```
//--- indicator buffers
double ExtMacdBuffer[];
double ExtSignalBuffer[];
double ExtFastMaBuffer[];
double ExtSlowMaBuffer[];

int    ExtFastMaHandle;
int    ExtSlowMaHandle;
```

Now, with our buffers in mind, let's go through a step-by-step development process outlined below.

Divergence Strategy development:

Step 1: Declarations and Input Parameters

We start by declaring input parameters for the divergence strategy, MACD buffers, and initializing the CTrade class.

```
#include <Trade\Trade.mqh>
CTrade trade;

input bool UseDivergenceStrategy = true;        // Enable/Disable Divergence Strategy
input int DivergenceMACDPeriod = 12;            // MACD Fast EMA period
input int DivergenceSignalPeriod = 9;           // MACD Signal period
input double DivergenceLots = 1.0;              // Lot size for Divergence trades
input double DivergenceStopLoss = 300;          // Stop Loss in points for Divergence
input double DivergenceTakeProfit = 500;        // Take Profit in points for Divergence
input int DivergenceMagicNumber = 87654321;     // Magic number for Divergence Strategy
input int DivergenceLookBack = 8;               // Number of periods to look back for divergence

double ExtMacdBuffer[]; // MACD values
double ExtSignalBuffer[]; // Signal line values
int macd_handle; // MACD indicator handle
```

Step 2: Initialize MACD Indicator

Initialize the MACD handle and allocate buffer memory during OnInit().

```
int OnInit()
{
    macd_handle = iMACD(_Symbol, PERIOD_CURRENT, DivergenceMACDPeriod, 26, DivergenceSignalPeriod, PRICE_CLOSE);
    if (macd_handle == INVALID_HANDLE)
    {
        Print("Failed to initialize MACD. Error: ", GetLastError());
        return INIT_FAILED;
    }
    ArrayResize(ExtMacdBuffer, DivergenceLookBack);
    ArrayResize(ExtSignalBuffer, DivergenceLookBack);
    return INIT_SUCCEEDED;
}
```

Step 3:  Divergence Detection

Divergence occurs when the price action of an asset disagrees with an indicator, in this case, the MACD. This strategy detects four types of divergences: bullish regular, bullish hidden, bearish regular, and bearish hidden. Each type has specific conditions, comparing price highs or lows to corresponding MACD highs or lows to determine the presence of divergence.

```
bool CheckBullishRegularDivergence()
{
    double priceLow1 = iLow(_Symbol, PERIOD_CURRENT, 2);
    double priceLow2 = iLow(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdLow1 = ExtMacdBuffer[2];
    double macdLow2 = ExtMacdBuffer[DivergenceLookBack - 1];
    return (priceLow1 < priceLow2 && macdLow1 > macdLow2);
}

bool CheckBearishHiddenDivergence()
{
    double priceHigh1 = iHigh(_Symbol, PERIOD_CURRENT, 2);
    double priceHigh2 = iHigh(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdHigh1 = ExtMacdBuffer[2];
    double macdHigh2 = ExtMacdBuffer[DivergenceLookBack - 1];
    return (priceHigh1 < priceHigh2 && macdHigh1 > macdHigh2);
}
```

Step 4: Trading Logic

First, the strategy ensures divergence trading is enabled and no more than 3 divergence-based positions are open. It retrieves MACD buffer data, retries on failure, and only executes trades on complete bars. Additionally, it aligns trades with daily candlestick trends, ensuring only buys on bullish days and sells on bearish days.

```
void CheckDivergenceTrading()
{
    if (!UseDivergenceStrategy) return;

    int openDivergencePositions = CountOrdersByMagic(DivergenceMagicNumber);
    if (openDivergencePositions == 0 || openDivergencePositions < 3)
    {
        if (CopyBuffer(macd_handle, 0, 0, DivergenceLookBack, ExtMacdBuffer) > 0 &&
            CopyBuffer(macd_handle, 1, 0, DivergenceLookBack, ExtSignalBuffer) > 0)
        {
            double dailyClose = iClose(_Symbol, PERIOD_D1, 0);
            double dailyOpen = iOpen(_Symbol, PERIOD_D1, 0);
            bool isDailyBullish = dailyClose > dailyOpen;
            bool isDailyBearish = dailyClose < dailyOpen;

            if (isDailyBullish &&
                (CheckBullishRegularDivergence() && ExtMacdBuffer[0] > ExtSignalBuffer[0]) ||
                CheckBullishHiddenDivergence())
            {
                ExecuteDivergenceOrder(true);
            }

            if (isDailyBearish &&
                (CheckBearishRegularDivergence() && ExtMacdBuffer[0] < ExtSignalBuffer[0]) ||
                CheckBearishHiddenDivergence())
            {
                ExecuteDivergenceOrder(false);
            }
        }
    }
}
```

Step 5: Order Execution

Once a divergence is detected, the strategy executes trades with predefined parameters like lot size, stop loss, and take profit. The ExecuteDivergenceOrder function calculates appropriate levels based on the trade direction and uses the trade object to place buy or sell orders.

```
void ExecuteDivergenceOrder(bool isBuy)
{
    trade.SetExpertMagicNumber(DivergenceMagicNumber);

    double currentPrice = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double stopLossPrice = isBuy ? currentPrice - DivergenceStopLoss * _Point : currentPrice + DivergenceStopLoss * _Point;
    double takeProfitPrice = isBuy ? currentPrice + DivergenceTakeProfit * _Point : currentPrice - DivergenceTakeProfit * _Point;

    if (isBuy)
    {
        if (trade.Buy(DivergenceLots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Divergence Buy"))
            Print("Divergence Buy order placed.");
    }
    else
    {
        if (trade.Sell(DivergenceLots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Divergence Sell"))
            Print("Divergence Sell order placed.");
    }
}
```

Step 6: Order Management

To prevent overtrading, the strategy uses the CountOrdersByMagic utility function to count all open positions with the specified magic number. This ensures adherence to the maximum position limit set for divergence-based trades.

```
int CountOrdersByMagic(int magic)
{
    int count = 0;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            if (PositionGetInteger(POSITION_MAGIC) == magic)
            {
                count++;
            }
        }
    }
    return count;
}
```

Magic Number:

Another important aspect is to give our positions an identity. In this case, we assigned a unique magic number to the trades governed by the divergence strategy.

```
    // Ensure the magic number is set for the trade
    trade.SetExpertMagicNumber(DivergenceMagicNumber);
```

Step 7: Profit Protection

We introduced a Profit Protection Function that incorporates a Profit Locking Logic for our Trend Constraint Expert Advisor to dynamically secure profits from open trades. The _LockProfits_ function scans all active positions, identifying those with profits exceeding a threshold of 100 points. For each qualifying position, it calculates a new stop-loss level based on the _profitLockerPoints_ parameter (e.g., 20 points from the entry price).

This adjustment ensures the stop-loss moves closer to the current price, effectively locking in gains. For buy positions, the stop-loss is shifted above the entry price, while on Sell positions, it is shifted below. The function updates the stop-loss only if the new level offers better protection, ensuring optimal risk management. Successful modifications are logged with a message for tracking. This feature helps secure profits while allowing trades the potential for further growth.

Here, is the lock profits function:

```
//+------------------------------------------------------------------+
//| Profit Locking Logic                                             |
//+------------------------------------------------------------------+
void LockProfits()
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentProfit = PositionGetDouble(POSITION_PROFIT);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);

            // Convert profit to points
            double profitPoints = MathAbs(currentProfit / _Point);

            // Check if profit has exceeded 100 points
            if (profitPoints >= 100)
            {
                double newStopLoss;

                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    newStopLoss = entryPrice + profitLockerPoints * _Point; // 20 points above entry for buys
                }
                else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                {
                    newStopLoss = entryPrice - profitLockerPoints * _Point; // 20 points below entry for sells
                }
                else
                {
                    continue; // Skip if not a buy or sell position
                }

                // Modify stop loss only if the new stop loss is more protective
                double currentStopLoss = PositionGetDouble(POSITION_SL);
                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    if (currentStopLoss < newStopLoss || currentStopLoss == 0)
                    {
                        if (trade.PositionModify(ticket, newStopLoss, PositionGetDouble(POSITION_TP)))
                        {
                            Print("Profit locking for buy position: Stop Loss moved to ", newStopLoss);
                        }
                    }
                }
                else // POSITION_TYPE_SELL
                {
                    if (currentStopLoss > newStopLoss || currentStopLoss == 0)
                    {
                        if (trade.PositionModify(ticket, newStopLoss, PositionGetDouble(POSITION_TP)))
                        {
                            Print("Profit locking for sell position: Stop Loss moved to ", newStopLoss);
                        }
                    }
                }
            }
        }
    }
}
```

Step 8: Integrate into OnTick()

Call the divergence trading logic within the main EA loop.

```
void OnTick()
{
    CheckDivergenceTrading();
}
```

Step 9: ShutDown

```
void OnDeinit(const int reason)
{
    IndicatorRelease(macd_handle);
}
```

Integration of the strategy into the main Trend Constrant Expert.

From the previous article in this series we developed a Donchian Channel-based expert advisor, marking two strategies in one expert. Today we incorporate the 3rd strategy, and we use boolean data tpye for switching the strategies on and off.

```
// Input parameters for controlling strategies
input bool UseTrendFollowingStrategy = false;   // Enable/Disable Trend Constraint Strategy
input bool UseBreakoutStrategy = false;         // Enable/Disable Breakout Strategy
input bool UseDivergenceStrategy = true;        // Enable/Disable Divergence Strategy
```

We set the Divergence strategy to true so that we do not get confused during the Strategy Tester.

Finally, we carefully integrate our strategy into the main code:

```
//+------------------------------------------------------------------+
//|                                      Trend Constraint Expert.mq5 |
//|                                Copyright 2024, Clemence Benjamin |
//|             https://www.mql5.com/en/users/billionaire2024/seller |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Clemence Benjamin"
#property link      "https://www.mql5.com/en/users/billionaire2024/seller"
#property version   "1.02"

#include <Trade\Trade.mqh>
CTrade trade;

// Input parameters for controlling strategies
input bool UseTrendFollowingStrategy = false;   // Enable/Disable Trend Following Strategy
input bool UseBreakoutStrategy = false;         // Enable/Disable Breakout Strategy
input bool UseDivergenceStrategy = true;        // Enable/Disable Divergence Strategy

// Input parameters for Trend Constraint Strategy
input int    RSI_Period = 14;           // RSI period
input double RSI_Overbought = 70.0;     // RSI overbought level
input double RSI_Oversold = 30.0;       // RSI oversold level
input double Lots = 0.1;                // Lot size
input double StopLoss = 100;            // Stop Loss in points
input double TakeProfit = 200;          // Take Profit in points
input double TrailingStop = 50;         // Trailing Stop in points
input int    MagicNumber = 12345678;    // Magic number for the Trend Constraint EA
input int    OrderLifetime = 43200;     // Order lifetime in seconds (12 hours)

// Input parameters for Breakout Strategy
input int InpDonchianPeriod = 20;       // Period for Donchian Channel
input double RiskRewardRatio = 1.5;     // Risk-to-reward ratio
input double LotSize = 0.1;             // Default lot size for trading
input double pipsToStopLoss = 15;       // Stop loss in pips for Breakout
input double pipsToTakeProfit = 30;     // Take profit in pips for Breakout

// Input parameters for Divergence Strategy
input int DivergenceMACDPeriod = 12;    // MACD Fast EMA period
input int DivergenceSignalPeriod = 9;   // MACD Signal period
input double DivergenceLots = 1.0;      // Lot size for Divergence trades
input double DivergenceStopLoss = 300;   // Stop Loss in points for Divergence
input double DivergenceTakeProfit = 500; // Take Profit in points for Divergence
input int DivergenceMagicNumber = 87654321;     // Magic number for Divergence Strategy
input int DivergenceLookBack = 8;       // Number of periods to look back for divergence
input double profitLockerPoints  = 20;  // Number of profit points to lock

// Indicator handle storage
int rsi_handle;
int handle;                             // Handle for Donchian Channel
int macd_handle;

double ExtUpBuffer[];                   // Upper Donchian buffer
double ExtDnBuffer[];                   // Lower Donchian buffer
double ExtMacdBuffer[];                 // MACD buffer
double ExtSignalBuffer[];               // Signal buffer
int globalMagicNumber;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize RSI handle
    rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
    if (rsi_handle == INVALID_HANDLE)
    {
        Print("Failed to create RSI indicator handle. Error: ", GetLastError());
        return INIT_FAILED;
    }

    // Create a handle for the Donchian Channel
    handle = iCustom(_Symbol, PERIOD_CURRENT, "Free Indicators\\Donchian Channel", InpDonchianPeriod);
    if (handle == INVALID_HANDLE)
    {
        Print("Failed to load the Donchian Channel indicator. Error: ", GetLastError());
        return INIT_FAILED;
    }

    // Initialize MACD handle for divergence
    globalMagicNumber = DivergenceMagicNumber;
    macd_handle = iMACD(_Symbol, PERIOD_CURRENT, DivergenceMACDPeriod, 26, DivergenceSignalPeriod, PRICE_CLOSE);
    if (macd_handle == INVALID_HANDLE)
    {
        Print("Failed to create MACD indicator handle for divergence strategy. Error: ", GetLastError());
        return INIT_FAILED;
    }

    // Resize arrays for MACD buffers
    ArrayResize(ExtMacdBuffer, DivergenceLookBack);
    ArrayResize(ExtSignalBuffer, DivergenceLookBack);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(rsi_handle);
    IndicatorRelease(handle);
    IndicatorRelease(macd_handle);
}

//+------------------------------------------------------------------+
//| Check and execute Trend Following EA trading logic               |
//+------------------------------------------------------------------+
void CheckTrendFollowing()
{
    if (PositionsTotal() >= 2) return; // Ensure no more than 2 orders from this strategy

    double rsi_value;
    double rsi_values[];
    if (CopyBuffer(rsi_handle, 0, 0, 1, rsi_values) <= 0)
    {
        Print("Failed to get RSI value. Error: ", GetLastError());
        return;
    }
    rsi_value = rsi_values[0];

    double ma_short = iMA(_Symbol, PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
    double ma_long = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);

    bool is_uptrend = ma_short > ma_long;
    bool is_downtrend = ma_short < ma_long;

    if (is_uptrend && rsi_value < RSI_Oversold)
    {
        double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double stopLossPrice = currentPrice - StopLoss * _Point;
        double takeProfitPrice = currentPrice + TakeProfit * _Point;

        // Corrected Buy method call with 6 parameters
        if (trade.Buy(Lots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Trend Following Buy"))
        {
            Print("Trend Following Buy order placed.");
        }
    }
    else if (is_downtrend && rsi_value > RSI_Overbought)
    {
        double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double stopLossPrice = currentPrice + StopLoss * _Point;
        double takeProfitPrice = currentPrice - TakeProfit * _Point;

        // Corrected Sell method call with 6 parameters
        if (trade.Sell(Lots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Trend Following Sell"))
        {
            Print("Trend Following Sell order placed.");
        }
    }
}

//+------------------------------------------------------------------+
//| Check and execute Breakout EA trading logic                      |
//+------------------------------------------------------------------+
void CheckBreakoutTrading()
{
    if (PositionsTotal() >= 2) return; // Ensure no more than 2 orders from this strategy

    ArrayResize(ExtUpBuffer, 2);
    ArrayResize(ExtDnBuffer, 2);

    if (CopyBuffer(handle, 0, 0, 2, ExtUpBuffer) <= 0 || CopyBuffer(handle, 2, 0, 2, ExtDnBuffer) <= 0)
    {
        Print("Error reading Donchian Channel buffer. Error: ", GetLastError());
        return;
    }

    double closePrice = iClose(_Symbol, PERIOD_CURRENT, 0);
    double lastOpen = iOpen(_Symbol, PERIOD_D1, 1);
    double lastClose = iClose(_Symbol, PERIOD_D1, 1);

    bool isBullishDay = lastClose > lastOpen;
    bool isBearishDay = lastClose < lastOpen;

    if (isBullishDay && closePrice > ExtUpBuffer[1])
    {
        double stopLoss = closePrice - pipsToStopLoss * _Point;
        double takeProfit = closePrice + pipsToTakeProfit * _Point;
        if (trade.Buy(LotSize, _Symbol, 0, stopLoss, takeProfit, "Breakout Buy") > 0)
        {
            Print("Breakout Buy order placed.");
        }
    }
    else if (isBearishDay && closePrice < ExtDnBuffer[1])
    {
        double stopLoss = closePrice + pipsToStopLoss * _Point;
        double takeProfit = closePrice - pipsToTakeProfit * _Point;
        if (trade.Sell(LotSize, _Symbol, 0, stopLoss, takeProfit, "Breakout Sell") > 0)
        {
            Print("Breakout Sell order placed.");
        }
    }
}

//+------------------------------------------------------------------+
//| DIVERGENCE TRADING STRATEGY                                      |
//+------------------------------------------------------------------+

bool CheckBullishRegularDivergence()
{
    double priceLow1 = iLow(_Symbol, PERIOD_CURRENT, 2);
    double priceLow2 = iLow(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdLow1 = ExtMacdBuffer[2];
    double macdLow2 = ExtMacdBuffer[DivergenceLookBack - 1];

    return (priceLow1 < priceLow2 && macdLow1 > macdLow2);
}

bool CheckBullishHiddenDivergence()
{
    double priceLow1 = iLow(_Symbol, PERIOD_CURRENT, 2);
    double priceLow2 = iLow(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdLow1 = ExtMacdBuffer[2];
    double macdLow2 = ExtMacdBuffer[DivergenceLookBack - 1];

    return (priceLow1 > priceLow2 && macdLow1 < macdLow2);
}

bool CheckBearishRegularDivergence()
{
    double priceHigh1 = iHigh(_Symbol, PERIOD_CURRENT, 2);
    double priceHigh2 = iHigh(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdHigh1 = ExtMacdBuffer[2];
    double macdHigh2 = ExtMacdBuffer[DivergenceLookBack - 1];

    return (priceHigh1 > priceHigh2 && macdHigh1 < macdHigh2);
}

bool CheckBearishHiddenDivergence()
{
    double priceHigh1 = iHigh(_Symbol, PERIOD_CURRENT, 2);
    double priceHigh2 = iHigh(_Symbol, PERIOD_CURRENT, DivergenceLookBack);
    double macdHigh1 = ExtMacdBuffer[2];
    double macdHigh2 = ExtMacdBuffer[DivergenceLookBack - 1];

    return (priceHigh1 < priceHigh2 && macdHigh1 > macdHigh2);
}

void CheckDivergenceTrading()
{
    if (!UseDivergenceStrategy) return;

    // Check if no position is open or if less than 3 positions are open
    int openDivergencePositions = CountOrdersByMagic(DivergenceMagicNumber);
    if (openDivergencePositions == 0 || openDivergencePositions < 3)
    {
        int barsAvailable = Bars(_Symbol, PERIOD_CURRENT);
        if (barsAvailable < DivergenceLookBack * 2)
        {
            Print("Not enough data bars for MACD calculation.");
            return;
        }

        int attempt = 0;
        while(attempt < 6)
        {
            if (CopyBuffer(macd_handle, 0, 0, DivergenceLookBack, ExtMacdBuffer) > 0 &&
                CopyBuffer(macd_handle, 1, 0, DivergenceLookBack, ExtSignalBuffer) > 0)
                break;

            Print("Failed to copy MACD buffer, retrying...");
            Sleep(1000);
            attempt++;
        }
        if(attempt == 6)
        {
            Print("Failed to copy MACD buffers after ", attempt, " attempts.");
            return;
        }

        if(TimeCurrent() == iTime(_Symbol, PERIOD_CURRENT, 0))
        {
            Print("Skipping trade due to incomplete bar data.");
            return;
        }

        double currentClose = iClose(_Symbol, PERIOD_CURRENT, 0);
        double dailyClose = iClose(_Symbol, PERIOD_D1, 0);
        double dailyOpen = iOpen(_Symbol, PERIOD_D1, 0);
        bool isDailyBullish = dailyClose > dailyOpen;
        bool isDailyBearish = dailyClose < dailyOpen;

        // Only proceed with buy orders if D1 is bullish
        if (isDailyBullish)
        {
            if ((CheckBullishRegularDivergence() && ExtMacdBuffer[0] > ExtSignalBuffer[0]) ||
                CheckBullishHiddenDivergence())
            {
                ExecuteDivergenceOrder(true);
            }
        }

        // Only proceed with sell orders if D1 is bearish
        if (isDailyBearish)
        {
            if ((CheckBearishRegularDivergence() && ExtMacdBuffer[0] < ExtSignalBuffer[0]) ||
                CheckBearishHiddenDivergence())
            {
                ExecuteDivergenceOrder(false);
            }
        }
    }
    else
    {
        Print("Divergence strategy: Maximum number of positions reached.");
    }
}

void ExecuteDivergenceOrder(bool isBuy)
{
    // Ensure the magic number is set for the trade
    trade.SetExpertMagicNumber(DivergenceMagicNumber);

    double currentPrice = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double stopLossPrice = isBuy ? currentPrice - DivergenceStopLoss * _Point : currentPrice + DivergenceStopLoss * _Point;
    double takeProfitPrice = isBuy ? currentPrice + DivergenceTakeProfit * _Point : currentPrice - DivergenceTakeProfit * _Point;

    if (isBuy)
    {
        if (trade.Buy(DivergenceLots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Divergence Buy"))
        {
            Print("Divergence Buy order placed.");
        }
    }
    else
    {
        if (trade.Sell(DivergenceLots, _Symbol, 0, stopLossPrice, takeProfitPrice, "Divergence Sell"))
        {
            Print("Divergence Sell order placed.");
        }
    }
}

int CountOrdersByMagic(int magic)
{
    int count = 0;
    for (int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            if (PositionGetInteger(POSITION_MAGIC) == magic)
            {
                count++;
            }
        }
    }
    return count;
}
//+------------------------------------------------------------------+
//| Profit Locking Logic                                             |
//+------------------------------------------------------------------+
void LockProfits()
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)

    {
        ulong ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(ticket))
        {
            double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentProfit = PositionGetDouble(POSITION_PROFIT);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);

            // Convert profit to points
            double profitPoints = MathAbs(currentProfit / _Point);

            // Check if profit has exceeded 100 points
            if (profitPoints >= 100)
            {
                double newStopLoss;

                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    newStopLoss = entryPrice + profitLockerPoints * _Point; // 20 points above entry for buys
                }
                else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                {
                    newStopLoss = entryPrice - profitLockerPoints * _Point; // 20 points below entry for sells
                }
                else
                {
                    continue; // Skip if not a buy or sell position
                }

                // Modify stop loss only if the new stop loss is more protective
                double currentStopLoss = PositionGetDouble(POSITION_SL);
                if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {
                    if (currentStopLoss < newStopLoss || currentStopLoss == 0)
                    {
                        if (trade.PositionModify(ticket, newStopLoss, PositionGetDouble(POSITION_TP)))
                        {
                            Print("Profit locking for buy position: Stop Loss moved to ", newStopLoss);
                        }
                    }
                }
                else // POSITION_TYPE_SELL
                {
                    if (currentStopLoss > newStopLoss || currentStopLoss == 0)
                    {
                        if (trade.PositionModify(ticket, newStopLoss, PositionGetDouble(POSITION_TP)))
                        {
                            Print("Profit locking for sell position: Stop Loss moved to ", newStopLoss);
                        }
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if (UseTrendFollowingStrategy) CheckTrendFollowing();
    if (UseBreakoutStrategy) CheckBreakoutTrading();
    if (UseDivergenceStrategy) CheckDivergenceTrading();
    LockProfits(); // Call this function to check and lock profits
}
```

### Backtesting Results and Practical Application

To perform our test, we need to locate the Trend Constraint Expert under the Experts on the terminal. Make sure you are running on demo account. A Strategy Tester window opens, and input setting can be adjusted for different optimizations. See the images below with default settings:

![Input Settings](https://c.mql5.com/2/134/Input_Settings__2.PNG)

Trend Constraint Expert: Demo settings

We ran the Strategy Tester, and the trades executed successfully. The system limited the maximum number of positions to three per session, effectively preventing multiple uncontrolled order executions. Below is an image showcasing a segment of the back-testing process in action:

![EA on Tester](https://c.mql5.com/2/134/ShareX_Y7ogkAKthK__2.gif)

Trend Constraint Expert: Testing on EURUSD M15

The image below demonstrates that our position management functions are operating as expected. As per the implemented logic, a maximum of three orders are active, each labeled with the comment Divergence Sell, making them easily identifiable and aligned with the strategy.

![](https://c.mql5.com/2/134/Trade_comments__2.PNG)

Trend Constraint Expert: Maximum of 3 positions per

### Conclusion

We explored various types of divergences and implemented them in code as part of a confluence strategy that combines indicators like RSI and MACD to generate precise trading signals. These signals were further refined by incorporating daily candlestick trend constraints, ensuring they align with broader market trends for improved reliability. Our Trend Constraint Expert Advisor (EA) now features three distinct, configurable strategies, allowing users to tailor the EA to their trading preferences and adapt to various market conditions.

To enhance trade management, we introduced features like unique MAGIC numbers for each position, enabling precise control over open trades and limiting the number of positions per strategy. Additionally, a custom profit-locking function was developed to secure gains by dynamically adjusting stop-loss levels if the market reverses before reaching the Take Profit target. This functionality ensures both risk management and flexibility, making the EA robust and adaptable. Attached below are the source files for the EA program. We encourage you to explore them, test various configurations, and share your feedback in the comments section. Please note: These examples are for educational purposes only, and all tests should be conducted on demo accounts.

Happy developing, traders!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16549.zip "Download all attachments in the single ZIP archive")

[Trend\_Constraint\_Expert\_.mq5](https://www.mql5.com/en/articles/download/16549/trend_constraint_expert_.mq5 "Download Trend_Constraint_Expert_.mq5")(16.34 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/478339)**

![Across Neighbourhood Search (ANS)](https://c.mql5.com/2/82/Across_Neighbourhood_Search__LOGO__1.png)[Across Neighbourhood Search (ANS)](https://www.mql5.com/en/articles/15049)

The article reveals the potential of the ANS algorithm as an important step in the development of flexible and intelligent optimization methods that can take into account the specifics of the problem and the dynamics of the environment in the search space.

![Price Action Analysis Toolkit Development (Part 5): Volatility Navigator EA](https://c.mql5.com/2/105/Price_Action_Analysis_Toolkit_Development_Part_5___LOGO.png)[Price Action Analysis Toolkit Development (Part 5): Volatility Navigator EA](https://www.mql5.com/en/articles/16560)

Determining market direction can be straightforward, but knowing when to enter can be challenging. As part of the series titled "Price Action Analysis Toolkit Development", I am excited to introduce another tool that provides entry points, take profit levels, and stop loss placements. To achieve this, we have utilized the MQL5 programming language. Let’s delve into each step in this article.

![Build Self Optimizing Expert Advisors in MQL5 (Part 2): USDJPY Scalping Strategy](https://c.mql5.com/2/106/Build_Self_Optimizing_Expert_Advisors_in_MQL5_Part_2_Logo.png)[Build Self Optimizing Expert Advisors in MQL5 (Part 2): USDJPY Scalping Strategy](https://www.mql5.com/en/articles/16643)

Join us today as we challenge ourselves to build a trading strategy around the USDJPY pair. We will trade candlestick patterns that are formed on the daily time frame because they potentially have more strength behind them. Our initial strategy was profitable, which encouraged us to continue refining the strategy and adding extra layers of safety, to protect the capital gained.

![Ensemble methods to enhance numerical predictions in MQL5](https://c.mql5.com/2/105/logo-ensemble_methods_to_enhance_numerical_predictions-2.png)[Ensemble methods to enhance numerical predictions in MQL5](https://www.mql5.com/en/articles/16630)

In this article, we present the implementation of several ensemble learning methods in MQL5 and examine their effectiveness across different scenarios.

[![](https://www.mql5.com/ff/sh/0hvxp984jjj79943z2/6373d9e5710a718ffa6a7d50a5db9dd1.jpg)\\
Web terminal on your iPhone or Android\\
\\
Full-featured MetaTrader 5 platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=uyigsjnbfcdvysiynusmriwvhincciwd&s=c95531ae2fd8a81b0fac3def2e4cf820a67584bbf4b02f76ec75f808942dbbd2&uid=&ref=https://www.mql5.com/en/articles/16549&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049344689477364173)

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