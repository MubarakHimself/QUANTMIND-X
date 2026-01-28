---
title: Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor
url: https://www.mql5.com/en/articles/17781
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:26:21.687729
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/17781&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049133334136726923)

MetaTrader 5 / Trading


1. [Introduction](https://www.mql5.com/en/articles/17781#sec1)

2. [Building an Adaptive Expert Advisor](https://www.mql5.com/en/articles/17781#sec2)
3. [Practical Considerations and Optimization](https://www.mql5.com/en/articles/17781#sec3)
4. [Indicator: Multi Timeframe Regimes](https://www.mql5.com/en/articles/17781#sec4)
5. [Evaluating the Adaptive Expert Advisor: Backtesting Results](https://www.mql5.com/en/articles/17781#sec5)
6. [Conclusion](https://www.mql5.com/en/articles/17781#sec6)

### Introduction

In [Part 1](https://www.mql5.com/en/articles/17737) of this series, we laid the essential groundwork for tackling the challenge of ever-changing market dynamics. We established a robust statistical foundation, constructed a CMarketRegimeDetector class capable of objectively classifying market behavior, and developed a custom indicator ( MarketRegimeIndicator ) to visualize these regimes directly on our charts. We moved from recognizing the problem – the performance degradation of static strategies in dynamic markets – to developing a system that can _identify_ the prevailing market state: Trending (Up/Down), Ranging, or Volatile [(Link Part 1 for Reference)](https://www.mql5.com/en/articles/17737).

However, merely identifying the current regime is only half the battle. The true power lies in _adapting_ our trading approach based on this knowledge. A detector, no matter how sophisticated, remains an analytical tool until its insights are translated into actionable trading decisions. What if our trading system could automatically switch gears, deploying trend-following logic when trends are strong, shifting to mean-reversion tactics in sideways markets, and adjusting risk parameters when volatility spikes?

This is precisely the gap we will bridge in Part 2. Building upon the foundation laid previously, we will now focus on practical application and refinement. In this article, we will delve into:

- Building an Adaptive Expert Advisor (EA): We'll construct a complete MarketRegimeEA that integrates our CMarketRegimeDetector to automatically select and execute different trading strategies (trend-following, mean-reversion, breakout) tailored to the detected regime.
- Implementing Regime-Specific Risk Management: The EA will demonstrate how to adjust parameters like lot size, stop loss, and take profit based on the current market state.
- Practical Considerations: We'll explore crucial aspects of real-world implementation, including parameter optimization for different instruments and timeframes.
- Handling Regime Transitions: Strategies to manage the critical moments when the market shifts from one regime to another, ensuring smoother strategy adjustments.
- Integration Techniques: Discussing how to incorporate the Market Regime Detection System into your existing trading frameworks to enhance their adaptability.

By the end of this second part, you will not only understand how to detect market regimes but also how to build an automated trading system that intelligently adapts its behavior, aiming for more consistent performance across the diverse conditions financial markets present. Let's transform our detection system into a truly adaptive trading solution.

### **Building an Adaptive Expert Advisor**

In this section, we'll create an Expert Advisor that uses our Market Regime Detector to adapt its trading strategy based on the current market conditions. This demonstrates how regime detection can be integrated into a complete trading system.

**The MarketRegimeEA**

Our Expert Advisor will use different trading approaches for different market regimes:

- In trending markets, it will use trend-following strategies
- In ranging markets, it will use mean-reversion strategies
- In volatile markets, it will use breakout strategies with reduced position sizes

Here's the implementation:

```
// Include the Market Regime Detector
#include <MarketRegimeEnum.mqh>
#include <MarketRegimeDetector.mqh>

// EA input parameters
input int      LookbackPeriod = 100;       // Lookback period for calculations
input int      SmoothingPeriod = 10;       // Smoothing period for regime transitions
input double   TrendThreshold = 0.2;       // Threshold for trend detection (0.1-0.5)
input double   VolatilityThreshold = 1.5;  // Threshold for volatility detection (1.0-3.0)

// Trading parameters
input double   TrendingLotSize = 0.1;      // Lot size for trending regimes
input double   RangingLotSize = 0.05;      // Lot size for ranging regimes
input double   VolatileLotSize = 0.02;     // Lot size for volatile regimes
input int      TrendingStopLoss = 100;     // Stop loss in points for trending regimes
input int      RangingStopLoss = 50;       // Stop loss in points for ranging regimes
input int      VolatileStopLoss = 150;     // Stop loss in points for volatile regimes
input int      TrendingTakeProfit = 200;   // Take profit in points for trending regimes
input int      RangingTakeProfit = 80;     // Take profit in points for ranging regimes
input int      VolatileTakeProfit = 300;   // Take profit in points for volatile regimes

// Global variables
CMarketRegimeDetector *Detector = NULL;
int OnBarCount = 0;
datetime LastBarTime = 0;
```

The EA includes parameters for configuring both the regime detection and the trading strategy. Note how we use different lot sizes, stop losses, and take profits for different market regimes. This allows the EA to adapt its risk management approach to the current market conditions.

**EA Initialization**

The OnInit() function creates and configures the Market Regime Detector:

```
int OnInit()
{
    // Create and initialize the Market Regime Detector
    Detector = new CMarketRegimeDetector(LookbackPeriod, SmoothingPeriod);
    if(Detector == NULL)
    {
        Print("Failed to create Market Regime Detector");
        return INIT_FAILED;
    }

    // Configure the detector
    Detector.SetTrendThreshold(TrendThreshold);
    Detector.SetVolatilityThreshold(VolatilityThreshold);
    Detector.Initialize();

    // Initialize variables
    OnBarCount = 0;
    LastBarTime = 0;

    return INIT_SUCCEEDED;
}
```

This function creates and configures the Market Regime Detector with the user-specified parameters. It also initializes the bar counting variables that we'll use to track new bars.

**EA Tick Processing**

The OnTick() function processes new price data and executes the regime-based trading strategy:

```
void OnTick()
{
    // Check for new bar
    datetime currentBarTime = iTime(Symbol(), PERIOD_CURRENT, 0);
    if(currentBarTime == LastBarTime)
        return; // No new bar

    LastBarTime = currentBarTime;
    OnBarCount++;

    // Wait for enough bars to accumulate
    if(OnBarCount < LookbackPeriod)
    {
        Comment("Accumulating data: ", OnBarCount, " of ", LookbackPeriod, " bars");
        return;
    }

    // Get price data
    double close[];
    ArraySetAsSeries(close, true);
    int copied = CopyClose(Symbol(), PERIOD_CURRENT, 0, LookbackPeriod, close);

    if(copied != LookbackPeriod)
    {
        Print("Failed to copy price data: copied = ", copied, " of ", LookbackPeriod);
        return;
    }

    // Process data with the detector
    if(!Detector.ProcessData(close, LookbackPeriod))
    {
        Print("Failed to process data with Market Regime Detector");
        return;
    }

    // Get current market regime
    ENUM_MARKET_REGIME currentRegime = Detector.GetCurrentRegime();

    // Display current regime information
    string regimeText = "Current Market Regime: " + Detector.GetRegimeDescription();
    string trendText = "Trend Strength: " + DoubleToString(Detector.GetTrendStrength(), 4);
    string volatilityText = "Volatility: " + DoubleToString(Detector.GetVolatility(), 4);

    Comment(regimeText + "\n" + trendText + "\n" + volatilityText);

    // Execute trading strategy based on market regime
    ExecuteRegimeBasedStrategy(currentRegime);
}
```

This function:

1. Checks for a new bar to avoid redundant calculations
2. Waits until enough bars have accumulated for reliable regime detection
3. Retrieves the latest price data
4. Processes the data with the Market Regime Detector
5. Gets the current market regime
6. Displays the regime information
7. Executes the regime-based trading strategy

The use of ArraySetAsSeries(close, true) is important as it ensures that the price array is indexed in reverse order, with the most recent price at index 0. This is the standard indexing convention in MQL5 for working with time series data.

**Regime-Based Trading Strategy**

The ExecuteRegimeBasedStrategy() function implements different trading approaches for different market regimes:

```
void ExecuteRegimeBasedStrategy(ENUM_MARKET_REGIME regime)
{
    // Check if we already have open positions
    if(PositionsTotal() > 0)
        return; // Don't open new positions if we already have one

    // Get current market information
    double ask = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    double bid = SymbolInfoDouble(Symbol(), SYMBOL_BID);
    double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);

    // Determine trading parameters based on regime
    double lotSize = 0.0;
    int stopLoss = 0;
    int takeProfit = 0;
    ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY;

    switch(regime)
    {
        case REGIME_TRENDING_UP: {
            lotSize = TrendingLotSize;
            stopLoss = TrendingStopLoss;
            takeProfit = TrendingTakeProfit;
            orderType = ORDER_TYPE_BUY; // Buy in uptrend
            break;
        }
        case REGIME_TRENDING_DOWN: {
            lotSize = TrendingLotSize;
            stopLoss = TrendingStopLoss;
            takeProfit = TrendingTakeProfit;
            orderType = ORDER_TYPE_SELL; // Sell in downtrend
            break;
	}

        case REGIME_RANGING: {
            // In ranging markets, we can use mean-reversion strategies
            // For simplicity, we'll use RSI to determine overbought/oversold
            double rsi[];
            ArraySetAsSeries(rsi, true);
            int rsiCopied = CopyBuffer(iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE), 0, 0, 2, rsi);

            if(rsiCopied != 2)
                return;

            lotSize = RangingLotSize;
            stopLoss = RangingStopLoss;
            takeProfit = RangingTakeProfit;

            if(rsi[0] < 30) // Oversold
                orderType = ORDER_TYPE_BUY;
            else if(rsi[0] > 70) // Overbought
                orderType = ORDER_TYPE_SELL;
            else
                return; // No signal

            break;
        }

        case REGIME_VOLATILE: {
            // In volatile markets, we can use breakout strategies
            // For simplicity, we'll use Bollinger Bands
            double upper[], lower[];
            ArraySetAsSeries(upper, true);
            ArraySetAsSeries(lower, true);

            int bbCopied1 = CopyBuffer(iBands(Symbol(), PERIOD_CURRENT, 20, 2, 0, PRICE_CLOSE), 1, 0, 2, upper);
            int bbCopied2 = CopyBuffer(iBands(Symbol(), PERIOD_CURRENT, 20, 2, 0, PRICE_CLOSE), 2, 0, 2, lower);

            if(bbCopied1 != 2 || bbCopied2 != 2)
                return;

            lotSize = VolatileLotSize;
            stopLoss = VolatileStopLoss;
            takeProfit = VolatileTakeProfit;

            double close[];
            ArraySetAsSeries(close, true);
            int copied = CopyClose(Symbol(), PERIOD_CURRENT, 0, 2, close);

            if(copied != 2)
                return;

            if(close[1] < upper[1] && close[0] > upper[0]) // Breakout above upper band
                orderType = ORDER_TYPE_BUY;
            else if(close[1] > lower[1] && close[0] < lower[0]) // Breakout below lower band
                orderType = ORDER_TYPE_SELL;
            else
                return; // No signal

            break;
        }

        default:
            return; // No trading in undefined regime
    }

    // Calculate stop loss and take profit levels
    double slLevel = 0.0;
    double tpLevel = 0.0;

    if(orderType == ORDER_TYPE_BUY)
    {
        slLevel = ask - stopLoss * point;
        tpLevel = ask + takeProfit * point;
    }
    else if(orderType == ORDER_TYPE_SELL)
    {
        slLevel = bid + stopLoss * point;
        tpLevel = bid - takeProfit * point;
    }

    // Execute trade
    MqlTradeRequest request;
    MqlTradeResult result;

    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = Symbol();
    request.volume = lotSize;
    request.type = orderType;
    request.price = (orderType == ORDER_TYPE_BUY) ? ask : bid;
    request.sl = slLevel;
    request.tp = tpLevel;
    request.deviation = 10;
    request.magic = 123456; // Magic number for this EA
    request.comment = "Market Regime: " + Detector.GetRegimeDescription();
    request.type_filling = ORDER_FILLING_FOK;

    bool success = OrderSend(request, result);

    if(success)
    {
        Print("Trade executed successfully: ", result.retcode, " ", result.comment);
    }
    else
    {
        Print("Trade execution failed: ", result.retcode, " ", result.comment);
    }
}
```

This function implements a comprehensive regime-based trading strategy:

1. Trending Regimes: Uses trend-following strategies, buying in uptrends and selling in downtrends.
2. Ranging Regimes: Uses mean-reversion strategies based on RSI, buying when oversold and selling when overbought.
3. Volatile Regimes: Uses breakout strategies based on Bollinger Bands, with reduced position sizes to manage risk.

For each regime, the function sets appropriate lot sizes, stop losses, and take profits based on the user-specified parameters. It then calculates the specific stop loss and take profit levels and executes the trade using the OrderSend() function.

The use of different strategies for different regimes is the key innovation of this EA. By adapting its approach to the current market conditions, the EA can achieve more consistent performance across a wide range of market environments.

**EA Cleanup**

The OnDeinit() function ensures proper cleanup when the EA is removed:

```
void OnDeinit(const int reason)
{
    // Clean up
    if(Detector != NULL)
    {
        delete Detector;
        Detector = NULL;
    }

    // Clear the comment
    Comment("");
}
```

This function deletes the Market Regime Detector object to prevent memory leaks and clears any comments from the chart.

**Advantages of Regime-Based Trading**

This Expert Advisor demonstrates several key advantages of regime-based trading:

1. Adaptability: The EA automatically adapts its trading strategy to the current market conditions, using different approaches for different regimes.
2. Risk Management: The EA adjusts position sizes based on market volatility, using smaller positions in more volatile regimes to manage risk.
3. Strategy Selection: The EA selects the most appropriate strategy for each regime, using trend-following in trending markets and mean-reversion in ranging markets.
4. Transparency: The EA clearly displays the current market regime and its key characteristics, providing traders with valuable context for their trading decisions.

By incorporating market regime detection into your trading systems, you can achieve similar benefits, creating more robust and adaptive strategies that perform well across a wide range of market conditions.

In the next section, we'll discuss practical considerations for implementing and optimizing the Market Regime Detection System in real trading environments.

### Practical Considerations and Optimization

In this final section, we'll discuss practical considerations for implementing and optimizing the Market Regime Detection System in real trading environments. We'll cover parameter optimization, regime transition handling, and integration with existing trading systems.

**Parameter Optimization**

The effectiveness of our Market Regime Detection System depends significantly on the choice of parameters. Here are key parameters that traders should optimize for their specific trading instruments and timeframes:

1. Lookback Period:The lookback period determines how much historical data is used for regime detection. Longer periods provide more stable regime classifications but may be less responsive to recent market changes. Shorter periods are more responsive but may generate more false signals.



```
// Example of testing different lookback periods
for(int lookback = 50; lookback <= 200; lookback += 25)
{
       CMarketRegimeDetector detector(lookback, SmoothingPeriod);
       detector.SetTrendThreshold(TrendThreshold);
       detector.SetVolatilityThreshold(VolatilityThreshold);

       // Process historical data and evaluate performance
       // ...
}
```

Typically, lookback periods between 50 and 200 bars work well for most instruments and timeframes. The optimal value depends on the typical duration of market regimes for the specific instrument being traded.

2. Trend Threshold:The trend threshold determines how strong a trend must be to classify the market as trending. Higher thresholds result in fewer trending classifications but with higher confidence. Lower thresholds identify more trends but may include weaker ones.



```
// Example of testing different trend thresholds
for(double threshold = 0.1; threshold <= 0.5; threshold += 0.05)
{
       CMarketRegimeDetector detector(LookbackPeriod, SmoothingPeriod);
       detector.SetTrendThreshold(threshold);
       detector.SetVolatilityThreshold(VolatilityThreshold);

       // Process historical data and evaluate performance
       // ...
}
```

Trend thresholds between 0.1 and 0.3 are common starting points. The optimal value depends on the instrument's typical trending behavior.

3. Volatility Threshold: The volatility threshold determines how much volatility is required to classify the market as volatile. Higher thresholds result in fewer volatile classifications, while lower thresholds identify more volatile periods.



```
// Example of testing different volatility thresholds
for(double threshold = 1.0; threshold <= 3.0; threshold += 0.25)
{
       CMarketRegimeDetector detector(LookbackPeriod, SmoothingPeriod);
       detector.SetTrendThreshold(TrendThreshold);
       detector.SetVolatilityThreshold(threshold);

       // Process historical data and evaluate performance
       // ...
}
```

Volatility thresholds between 1.5 and 2.5 are common starting points. The optimal value depends on the instrument's typical volatility characteristics.


**Handling Regime Transitions**

Note: The following 5 code blocks are not the implementation of the idea rather just the ideas and pseudo code of how teh implementation might look like.

Regime transitions are critical moments that require special attention. Abrupt changes in trading strategy during transitions can lead to poor execution and increased slippage. Here are strategies for handling transitions more effectively

1. Smoothing Transitions: The smoothing period parameter helps reduce regime classification noise by requiring a regime to persist for a minimum number of bars before being recognized:



```
// Example of implementing smoothed regime transitions
ENUM_MARKET_REGIME SmoothRegimeTransition(ENUM_MARKET_REGIME newRegime)
{
       static ENUM_MARKET_REGIME regimeHistory[20];
       static int historyCount = 0;

       // Add new regime to history
       for(int i = 19; i > 0; i--)
           regimeHistory[i] = regimeHistory[i-1];

       regimeHistory[0] = newRegime;

       if(historyCount < 20)
           historyCount++;

       // Count occurrences of each regime
       int regimeCounts[5] = {0};

       for(int i = 0; i < historyCount; i++)
           regimeCounts[regimeHistory[i]]++;

       // Find most common regime
       int maxCount = 0;
       ENUM_MARKET_REGIME dominantRegime = REGIME_UNDEFINED;

       for(int i = 0; i < 5; i++)
       {
           if(regimeCounts[i] > maxCount)
           {
               maxCount = regimeCounts[i];
               dominantRegime = (ENUM_MARKET_REGIME)i;
           }
       }

       return dominantRegime;
}
```

This function maintains a history of recent regime classifications and returns the most common regime, reducing the impact of temporary fluctuations.

2. Gradual Position Sizing: During regime transitions, it's often prudent to adjust position sizes gradually rather than making abrupt changes:



```
// Example of gradual position sizing during transitions
double CalculateTransitionLotSize(ENUM_MARKET_REGIME previousRegime,
                                    ENUM_MARKET_REGIME currentRegime,
                                    int transitionBars,
                                    int maxTransitionBars)
{
       // Base lot sizes for each regime
       double regimeLotSizes[5] = {
           TrendingLotSize,    // REGIME_TRENDING_UP
           TrendingLotSize,    // REGIME_TRENDING_DOWN
           RangingLotSize,     // REGIME_RANGING
           VolatileLotSize,    // REGIME_VOLATILE
           0.0                 // REGIME_UNDEFINED
       };

       // If not in transition, use current regime's lot size
       if(previousRegime == currentRegime || transitionBars >= maxTransitionBars)
           return regimeLotSizes[currentRegime];

       // Calculate weighted average during transition
       double previousWeight = (double)(maxTransitionBars - transitionBars) / maxTransitionBars;
       double currentWeight = (double)transitionBars / maxTransitionBars;

       return regimeLotSizes[previousRegime] * previousWeight +
              regimeLotSizes[currentRegime] * currentWeight;
}
```

This function calculates a weighted average of position sizes during regime transitions, providing a smoother adjustment as the market changes.

**Integration with Existing Trading Systems**

The Market Regime Detection System can be integrated with existing trading systems to enhance their performance. Here are strategies for effective integration:

1. Strategy Selection: Use the detected regime to select the most appropriate trading strategy:



```
// Example of strategy selection based on market regime
bool ExecuteTradeSignal(ENUM_MARKET_REGIME regime, int strategySignal)
{
       // Strategy signal: 1 = buy, -1 = sell, 0 = no signal

       switch(regime)
       {
           case REGIME_TRENDING_UP:
           case REGIME_TRENDING_DOWN:
               // In trending regimes, only take signals in the direction of the trend
               if((regime == REGIME_TRENDING_UP && strategySignal == 1) ||
                  (regime == REGIME_TRENDING_DOWN && strategySignal == -1))
                   return true;
               break;

           case REGIME_RANGING:
               // In ranging regimes, take all signals
               if(strategySignal != 0)
                   return true;
               break;

           case REGIME_VOLATILE:
               // In volatile regimes, be more selective
               // Only take strong signals (implementation depends on strategy)
               if(IsStrongSignal(strategySignal))
                   return true;
               break;

           default:
               // In undefined regimes, don't trade
               break;
       }

       return false;
}
```

This function filters trading signals based on the current market regime, only executing trades that align with the regime's characteristics.

2. Parameter Adaptation: Adapt strategy parameters based on the detected regime:



```
// Example of parameter adaptation based on market regime
void AdaptStrategyParameters(ENUM_MARKET_REGIME regime)
{
       switch(regime)
       {
           case REGIME_TRENDING_UP:
           case REGIME_TRENDING_DOWN:
               // In trending regimes, use longer moving averages
               FastPeriod = 20;
               SlowPeriod = 50;
               // Use wider stop losses
               StopLoss = TrendingStopLoss;
               // Use larger take profits
               TakeProfit = TrendingTakeProfit;
               break;

           case REGIME_RANGING:
               // In ranging regimes, use shorter moving averages
               FastPeriod = 10;
               SlowPeriod = 25;
               // Use tighter stop losses
               StopLoss = RangingStopLoss;
               // Use smaller take profits
               TakeProfit = RangingTakeProfit;
               break;

           case REGIME_VOLATILE:
               // In volatile regimes, use very short moving averages
               FastPeriod = 5;
               SlowPeriod = 15;
               // Use wider stop losses
               StopLoss = VolatileStopLoss;
               // Use larger take profits
               TakeProfit = VolatileTakeProfit;
               break;

           default:
               // In undefined regimes, use default parameters
               FastPeriod = 14;
               SlowPeriod = 28;
               StopLoss = 100;
               TakeProfit = 200;
               break;
       }
}
```

This function adjusts strategy parameters based on the current market regime, optimizing the strategy for the specific market conditions.

**Performance Monitoring**

Regularly monitor the performance of your Market Regime Detection System to ensure it's accurately classifying market regimes:

```
// Example of performance monitoring for regime detection
void MonitorRegimeDetectionPerformance()
{
    static int regimeTransitions = 0;
    static int correctPredictions = 0;
    static ENUM_MARKET_REGIME lastRegime = REGIME_UNDEFINED;

    // Get current regime
    ENUM_MARKET_REGIME currentRegime = Detector.GetCurrentRegime();

    // If regime has changed, evaluate the previous regime's prediction
    if(currentRegime != lastRegime && lastRegime != REGIME_UNDEFINED)
    {
        regimeTransitions++;

        // Evaluate if the previous regime's prediction was correct
        // Implementation depends on your specific evaluation criteria
        if(EvaluateRegimePrediction(lastRegime))
            correctPredictions++;

        // Log performance metrics
        double accuracy = (double)correctPredictions / regimeTransitions * 100.0;
        Print("Regime Detection Accuracy: ", DoubleToString(accuracy, 2), "% (",
              correctPredictions, "/", regimeTransitions, ")");
    }

    lastRegime = currentRegime;
}
```

This function tracks regime transitions and evaluates the accuracy of regime predictions, providing valuable feedback for system optimization.

### Indicator: Multi Timeframe Regimes

Full Code:

```
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

// Include the Market Regime Detector
#include <MarketRegimeEnum.mqh>
#include <MarketRegimeDetector.mqh>

// Input parameters
input int      LookbackPeriod = 100;       // Lookback period for calculations
input double   TrendThreshold = 0.2;       // Threshold for trend detection (0.1-0.5)
input double   VolatilityThreshold = 1.5;  // Threshold for volatility detection (1.0-3.0)

// Timeframes to analyze
input bool     UseM1 = false;              // Use 1-minute timeframe
input bool     UseM5 = false;              // Use 5-minute timeframe
input bool     UseM15 = true;              // Use 15-minute timeframe
input bool     UseM30 = true;              // Use 30-minute timeframe
input bool     UseH1 = true;               // Use 1-hour timeframe
input bool     UseH4 = true;               // Use 4-hour timeframe
input bool     UseD1 = true;               // Use Daily timeframe
input bool     UseW1 = false;              // Use Weekly timeframe
input bool     UseMN1 = false;             // Use Monthly timeframe

// Global variables
CMarketRegimeDetector *Detectors[];
ENUM_TIMEFRAMES Timeframes[];
int TimeframeCount = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize timeframes array
    InitializeTimeframes();

    // Create detectors for each timeframe
    ArrayResize(Detectors, TimeframeCount);

    for(int i = 0; i < TimeframeCount; i++)
    {
        Detectors[i] = new CMarketRegimeDetector(LookbackPeriod);
        if(Detectors[i] == NULL)
        {
            Print("Failed to create Market Regime Detector for timeframe ", EnumToString(Timeframes[i]));
            return INIT_FAILED;
        }

        // Configure the detector
        Detectors[i].SetTrendThreshold(TrendThreshold);
        Detectors[i].SetVolatilityThreshold(VolatilityThreshold);
        Detectors[i].Initialize();
    }

    // Set indicator name
    IndicatorSetString(INDICATOR_SHORTNAME, "Multi-Timeframe Regime Analysis");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Initialize timeframes array based on user inputs                 |
//+------------------------------------------------------------------+
void InitializeTimeframes()
{
    // Count selected timeframes
    TimeframeCount = 0;
    if(UseM1)  TimeframeCount++;
    if(UseM5)  TimeframeCount++;
    if(UseM15) TimeframeCount++;
    if(UseM30) TimeframeCount++;
    if(UseH1)  TimeframeCount++;
    if(UseH4)  TimeframeCount++;
    if(UseD1)  TimeframeCount++;
    if(UseW1)  TimeframeCount++;
    if(UseMN1) TimeframeCount++;

    // Resize and fill timeframes array
    ArrayResize(Timeframes, TimeframeCount);

    int index = 0;
    if(UseM1)  Timeframes[index++] = PERIOD_M1;
    if(UseM5)  Timeframes[index++] = PERIOD_M5;
    if(UseM15) Timeframes[index++] = PERIOD_M15;
    if(UseM30) Timeframes[index++] = PERIOD_M30;
    if(UseH1)  Timeframes[index++] = PERIOD_H1;
    if(UseH4)  Timeframes[index++] = PERIOD_H4;
    if(UseD1)  Timeframes[index++] = PERIOD_D1;
    if(UseW1)  Timeframes[index++] = PERIOD_W1;
    if(UseMN1) Timeframes[index++] = PERIOD_MN1;
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
    // Check if there's enough data
    if(rates_total < LookbackPeriod)
        return 0;

    // Process data for each timeframe
    string commentText = "Multi-Timeframe Regime Analysis\n\n";

    for(int i = 0; i < TimeframeCount; i++)
    {
        // Get price data for this timeframe
        double tfClose[];
        ArraySetAsSeries(tfClose, true);
        int copied = CopyClose(Symbol(), Timeframes[i], 0, LookbackPeriod, tfClose);

        if(copied != LookbackPeriod)
        {
            Print("Failed to copy price data for timeframe ", EnumToString(Timeframes[i]));
            continue;
        }

        // Process data with the detector
        if(!Detectors[i].ProcessData(tfClose, LookbackPeriod))
        {
            Print("Failed to process data for timeframe ", EnumToString(Timeframes[i]));
            continue;
        }

        // Add timeframe information to comment
        commentText += TimeframeToString(Timeframes[i]) + ": ";
        commentText += Detectors[i].GetRegimeDescription();
        commentText += " (Trend: " + DoubleToString(Detectors[i].GetTrendStrength(), 2);
        commentText += ", Vol: " + DoubleToString(Detectors[i].GetVolatility(), 2) + ")";
        commentText += "\n";
    }

    // Display the multi-timeframe analysis
    Comment(commentText);

    // Return the number of calculated bars
    return rates_total;
}

//+------------------------------------------------------------------+
//| Convert timeframe enum to readable string                        |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES timeframe)
{
    switch(timeframe)
    {
        case PERIOD_M1:  return "M1";
        case PERIOD_M5:  return "M5";
        case PERIOD_M15: return "M15";
        case PERIOD_M30: return "M30";
        case PERIOD_H1:  return "H1";
        case PERIOD_H4:  return "H4";
        case PERIOD_D1:  return "D1";
        case PERIOD_W1:  return "W1";
        case PERIOD_MN1: return "MN1";
        default:         return "Unknown";
    }
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization function                       |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Clean up
    for(int i = 0; i < TimeframeCount; i++)
    {
        if(Detectors[i] != NULL)
        {
            delete Detectors[i];
            Detectors[i] = NULL;
        }
    }

    // Clear the comment
    Comment("");
}
```

This indicator doesn't plot lines on the chart. Instead, it analyzes market regimes (Trending Up/Down, Ranging, Volatile) across multiple user-selected timeframes using the CMarketRegimeDetector class (developed in Part 1) and displays the results as text in the top-left corner of the chart.

**Code Explanation:**

1. Properties & Includes:

   - #property indicator\_chart\_window : Makes the indicator run in the main chart window.
   - #property indicator\_buffers 0 , #property indicator\_plots 0 : Specifies that this indicator doesn't use any data buffers or plot lines/histograms itself.
   - #include <MarketRegimeEnum.mqh> , #include <MarketRegimeDetector.mqh> : Includes the necessary definitions and the core detector class from our previous work.
2. Input Parameters:

   - LookbackPeriod , TrendThreshold , VolatilityThreshold : These are the core settings for the regime detection logic, applied uniformly to all analyzed timeframes.
   - UseM1 to UseMN1 : A series of boolean inputs allowing the user to toggle ( true / false ) which standard timeframes (1-minute to Monthly) should be included in the analysis.
3. Global Variables:

   - Detectors\[\] : An array designed to hold separate instances of the CMarketRegimeDetector . Each selected timeframe will have its own dedicated detector object stored here.
   - Timeframes\[\] : An array to store the MQL5 timeframe identifiers (like PERIOD\_H1 , PERIOD\_D1 ) corresponding to the timeframes chosen by the user via the inputs.
   - TimeframeCount : An integer to keep track of how many timeframes were actually selected.
4. OnInit() (Initialization Function):

   - Runs once when the indicator starts.
   - Calls InitializeTimeframes() to figure out which timeframes the user selected and populate the Timeframes array.
   - Resizes the Detectors array to match the TimeframeCount .
   - Loops TimeframeCount times:
     - For each selected timeframe, it creates a _new_ CMarketRegimeDetector object using the shared LookbackPeriod .
     - It configures this specific detector instance with the shared TrendThreshold and VolatilityThreshold .
     - Crucially, each detector instance maintains its own internal state based on the data from its assigned timeframe.
   - Sets the indicator's display name.
5. InitializeTimeframes() (Helper Function):

   - Counts how many Use... inputs are set to true .
   - Resizes the global Timeframes array accordingly.
   - Populates the Timeframes array with the actual PERIOD\_... constants for the selected timeframes.
6. OnCalculate() (Main Calculation Function):

   - Runs on every new tick or bar.
   - Checks if enough historical bars ( rates\_total ) are available on the _chart's current timeframe_ to meet the LookbackPeriod requirement.
   - Initializes an empty string commentText .
   - Loops through each selected timeframe (tracked by i ):
     - Uses CopyClose() to fetch the last LookbackPeriod close prices specifically for the Symbol() and Timeframes\[i\] (e.g., fetches H4 close prices even if the indicator is on an M15 chart).
     - If data fetching is successful, it calls the ProcessData() method of the _corresponding detector object_ ( Detectors\[i\] ) with the fetched price data ( tfClose ).
     - Appends the results (Timeframe name, Regime Description, Trend Strength, Volatility) from Detectors\[i\] to the commentText string.
   - Finally, uses Comment(commentText) to display the aggregated analysis for all selected timeframes in the chart's corner.
7. TimeframeToString() (Helper Function):

   - A simple utility to convert MQL5 PERIOD\_... constants into readable strings like "M1", "H4", "D1".
8. OnDeinit() (Deinitialization Function):

   - Runs once when the indicator is removed from the chart or the terminal closes.
   - Loops through the Detectors array and uses delete to free the memory allocated for each CMarketRegimeDetector object created in OnInit() , preventing memory leaks.
   - Clears the text comment from the chart corner.

In essence, this code efficiently sets up independent regime detectors for multiple timeframes, fetches the necessary data for each, performs the analysis, and presents a consolidated multi-timeframe regime overview directly on the user's chart.

![](https://c.mql5.com/2/133/4918525112342.png)

Using this above multi-time frame indicator, another EA can be created that analyses multiple time frame before placing the trade. Although for the sake of brevity, we will not create another Expert Advisor rather we should test this one first.

### Evaluating the Adaptive Expert Advisor: Backtesting Results

Having constructed the MarketRegimeEA , the next logical step is to evaluate its performance through backtesting. This allows us to observe how the regime-adaptive logic functions on historical data and assess the impact of its parameter settings.

**Initial Test Configuration**

For this demonstration, we selected Gold (XAUUSD) as the testing instrument, utilizing data on the M1 timeframe. The initial parameters applied to the EA were arbitrarily chosen as:

- LookbackPeriod: 100
- SmoothingPeriod: 10
- TrendThreshold: 0.2
- VolatilityThreshold: 1.5
- Lot Sizes: Trending=0.1, Ranging=0.1, Volatile=0.1
- SL: Trending=1000, Ranging=1600, Volatile=2000
- TP: Trending=1400, Ranging=1000, Volatile=1800

![](https://c.mql5.com/2/133/6093239133219.png)

Running the EA with these default parameters yielded the following results:

![](https://c.mql5.com/2/133/792681443635.png)

As we observe from the equity curve and performance metrics, the initial backtest with these settings produced suboptimal results. While there were periods where the strategy generated profit (at one point reaching approximately 20% equity growth), the overall performance indicates a lack of consistent profitability and significant drawdowns. This outcome underscores the importance of parameter tuning for the specific instrument and timeframe being traded. These parameters serve as a starting point, but does not represent the optimal configuration.

To explore the potential for improved performance, we employed the parameter optimization capabilities within the MetaTrader 5 Strategy Tester, utilizing its genetic algorithm. The goal was to identify a set of parameters (within the tested range) that better align the regime detection and trading logic with the historical price behavior of Gold during the backtest period. The parameters targeted for optimization included the regime-specific Stop Loss and Take Profit values with other params as default.

Following the optimization process, the Strategy Tester identified the following parameter set as yielding significantly improved historical performance:

![](https://c.mql5.com/2/133/5753089943008.png)

Executing the backtest again using these optimized parameters resulted in a notably different performance profile:

![](https://c.mql5.com/2/133/385294018195.png)

The backtest utilizing the optimized parameters demonstrates a marked improvement, showing a positive net profit and a more favorable equity curve compared to the initial run.

![](https://c.mql5.com/2/133/3043417282445.png)

However,  It is crucial to interpret these results with caution. Parameter optimization, especially using genetic algorithms on historical data and running the backtest on the same period, inherently introduces a risk of **overfitting**. This means the parameters may be exceptionally well-suited to the _specific historical data used_ but may not generalize well to future, unseen market data.

Therefore, while the optimized results showcase the _potential_ of the adaptive strategy framework when parameters are tuned, they should **not** be taken as a definitive proof of future profitability. The primary purpose of this exercise is to demonstrate:

1. The functionality of the regime-adaptive EA.
2. The significant impact parameters have on performance.
3. That the underlying concept (adapting strategy to regime) can yield positive results on historical data _when appropriately configured_.

The fact that even the unoptimized version showed periods of profitability suggests the core logic isn't fundamentally flawed. However, developing this into a robust, production-ready strategy would require further steps beyond simple optimization, including:

- Rigorous out-of-sample testing to validate parameter robustness.
- Incorporating more sophisticated risk management.
- Potentially implementing techniques discussed earlier, such as Smoothing Transitions and Gradual Position Sizing , to handle regime changes more effectively.

### Conclusion

Over the course of this two-part series, we have journeyed from identifying a fundamental challenge in algorithmic trading – the detrimental effect of changing market conditions on static strategies – to architecting and implementing a complete, adaptive solution. In Part 1, we built the engine for understanding the market's state by developing a statistically grounded Market Regime Detection System and visualizing its output.

In this second and concluding part, we took the crucial step from detection to action. We demonstrated how to harness the power of our CMarketRegimeDetector by building the MarketRegimeEA, an Expert Advisor capable of automatically switching between trend-following, mean-reversion, and breakout strategies based on the identified market regime. We saw how adapting not just the entry logic but also risk parameters like lot size and stop levels can create a more resilient trading approach.

Furthermore, we addressed the practical realities of deploying such a system. We explored the importance of parameter optimization (Lookback Period, Trend Threshold, Volatility Threshold) to tune the detector to specific market characteristics and discussed techniques for handling regime transitions smoothly, minimizing potential whipsaws or abrupt strategy shifts. Finally, we touched upon how this regime detection framework can be integrated into existing trading systems, acting as an intelligent filter or parameter adjuster.

The goal was ambitious: to create trading systems that acknowledge and adapt to the market's inherent non-stationarity. By combining the detection capabilities developed in Part 1 with the adaptive execution framework and practical considerations discussed in Part 2, you now possess the blueprint for building strategies that are significantly more robust and attuned to the dynamic nature of financial markets. The journey from a static approach to a regime-adaptive one is complete, empowering you to navigate the market's complexities with greater intelligence and flexibility.

### File Overview

Here's a summary of all the files created in this article:

| File Name | Description |
| --- | --- |
| MarketRegimeEnum.mqh | Defines the market regime enumeration types used throughout the system |
| CStatistics.mqh | Statistical calculations class for market regime detection |
| MarketRegimeDetector.mqh | Core market regime detection implementation |
| MarketRegimeEA.mq5 | Expert Advisor that adapts to different market regimes |
| MultiTimeframeRegimes.mq5 | Example for analyzing regimes across multiple timeframes |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17781.zip "Download all attachments in the single ZIP archive")

[MarketRegimeEnum.mqh](https://www.mql5.com/en/articles/download/17781/marketregimeenum.mqh "Download MarketRegimeEnum.mqh")(0.79 KB)

[CStatistics.mqh](https://www.mql5.com/en/articles/download/17781/cstatistics.mqh "Download CStatistics.mqh")(9.28 KB)

[MarketRegimeDetector.mqh](https://www.mql5.com/en/articles/download/17781/marketregimedetector.mqh "Download MarketRegimeDetector.mqh")(16.5 KB)

[MarketRegimeEA.mq5](https://www.mql5.com/en/articles/download/17781/marketregimeea.mq5 "Download MarketRegimeEA.mq5")(9.21 KB)

[MultiTimeframeRegimes.mq5](https://www.mql5.com/en/articles/download/17781/multitimeframeregimes.mq5 "Download MultiTimeframeRegimes.mq5")(7.13 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Custom Debugging and Profiling Tools for MQL5 Development (Part I): Advanced Logging](https://www.mql5.com/en/articles/17933)
- [Building a Custom Market Regime Detection System in MQL5 (Part 1): Indicator](https://www.mql5.com/en/articles/17737)
- [Advanced Memory Management and Optimization Techniques in MQL5](https://www.mql5.com/en/articles/17693)
- [Mastering JSON: Create Your Own JSON Reader from Scratch in MQL5](https://www.mql5.com/en/articles/16791)
- [Mastering File Operations in MQL5: From Basic I/O to Building a Custom CSV Reader](https://www.mql5.com/en/articles/16614)
- [Modified Grid-Hedge EA in MQL5 (Part IV): Optimizing Simple Grid Strategy (I)](https://www.mql5.com/en/articles/14518)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/485549)**
(1)


![abhishek maderana](https://c.mql5.com/avatar/2025/6/6856A6FA-A383.jpg)

**[abhishek maderana](https://www.mql5.com/en/users/abhishekmaderan)**
\|
12 Jul 2025 at 21:53

i think you bymistake uploaded the same parameter image for the refined parameters, can you share what were the refined parameters after mt5's optimization?


![From Basic to Intermediate: FOR Statement](https://c.mql5.com/2/94/Do_b4sico_ao_intermediqrio_Comando_FOR___LOGO.png)[From Basic to Intermediate: FOR Statement](https://www.mql5.com/en/articles/15406)

In this article, we will look at the most basic concepts of the FOR statement. It is very important to understand everything that will be shown here. Unlike the other statements we've talked about so far, the FOR statement has some quirks that quickly make it very complex. So don't let stuff like this accumulate. Start studying and practicing as soon as possible.

![Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://c.mql5.com/2/137/websockets.png)[Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)

This article details the development of a custom dynamically linked library designed to facilitate asynchronous websocket client connections for MetaTrader programs.

![MQL5 Wizard Techniques you should know (Part 61): Using Patterns of ADX and CCI with Supervised Learning](https://c.mql5.com/2/138/article-17910-logo.png)[MQL5 Wizard Techniques you should know (Part 61): Using Patterns of ADX and CCI with Supervised Learning](https://www.mql5.com/en/articles/17910)

The ADX Oscillator and CCI oscillator are trend following and momentum indicators that can be paired when developing an Expert Advisor. We look at how this can be systemized by using all the 3 main training modes of Machine Learning. Wizard Assembled Expert Advisors allow us to evaluate the patterns presented by these two indicators, and we start by looking at how Supervised-Learning can be applied with these Patterns.

![Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://c.mql5.com/2/137/logo-17865.png)[Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://www.mql5.com/en/articles/17865)

In this article, we explore the automation of the Cypher harmonic pattern in MQL5, detailing its detection and visualization on MetaTrader 5 charts. We implement an Expert Advisor that identifies swing points, validates Fibonacci-based patterns, and executes trades with clear graphical annotations. The article concludes with guidance on backtesting and optimizing the program for effective trading.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=vafgfrolgefzeblnufxpimckriidecrv&ssn=1769091980107049155&ssn_dr=0&ssn_sr=0&fv_date=1769091980&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17781&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Building%20a%20Custom%20Market%20Regime%20Detection%20System%20in%20MQL5%20(Part%202)%3A%20Expert%20Advisor%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909198022579571&fz_uniq=5049133334136726923&sv=2552)

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