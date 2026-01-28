---
title: Automating The Market Sentiment Indicator
url: https://www.mql5.com/en/articles/19609
categories: Trading Systems, Indicators, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T18:31:16.375242
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/19609&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069510102098052673)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/19422), we developed a custom market sentiment indicator, which combines signals from multiple timeframes into a single, easy-to-read panel. We analyzed higher timeframe trends alongside lower timeframe structures—examining price action, swing highs/lows, moving averages, and volatility—to classify the market into five clear sentiments: bullish, bearish, risk-on, risk-off, or neutral.

Automating the custom indicator brings several key advantages in trading. It ensures continuous, objective monitoring of sentiment without emotional bias, enabling faster execution to market changes. We can maintain consistency in analysis across timeframes, catch early opportunities via systematic detection of breakout or trend-confirmation signals, and allocate risk more effectively. Ultimately, automation turns what would be a labor-intensive, error-prone process into an efficient EA that supports data-driven decision-making.

### Execution Logic

Bullish/Risk-On Sentiment:

Bullish sentiment reflects strong upward momentum, where higher timeframe trends and price structures confirm that buyers are in control, while risk-on sentiment indicates increased investor confidence and appetite for risk, often pushing prices higher. In both cases, the market environment supports upward movement, and our strategy is to execute a buy trade whenever the indicator signals either bullish or risk-on sentiment, since the price is more likely to continue rising.

![](https://c.mql5.com/2/170/BullishOrRiskOn__1.png)

Bearish/Risk-Off Sentiment:

Bearish sentiment reflects strong downward momentum, where higher timeframe trends and price structures confirm that sellers dominate the market, while risk-off sentiment signals a shift toward safety, with investors avoiding riskier assets and driving prices lower. In both cases, the market environment supports downward movement, and our strategy is to execute a sell trade whenever the indicator signals either bearish or risk-off sentiment, since the price is more likely to continue falling.

![](https://c.mql5.com/2/170/BearishOrRiskOff__1.png)

Neutral Sentiment:

Neutral or choppy sentiment occurs when the market lacks clear direction, with price action consolidating in a range or fluctuating unpredictably between levels. During these uncertain periods, the probability of false signals and whipsaws is higher, so we use the neutral reading as a filter to avoid entering trades. By staying out of the market in such conditions, we protect capital and wait for a clearer bullish, bearish, risk-on, or risk-off signal before taking action.

![](https://c.mql5.com/2/170/Choppy__1.png)

### Getting Started

```
//+------------------------------------------------------------------+
//|                                               Mark_Sent_Auto.mq5 |
//|                        GIT under Copyright 2025, MetaQuotes Ltd. |
//|                     https://www.mql5.com/en/users/johnhlomohang/ |
//+------------------------------------------------------------------+
#property copyright "GIT under Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/johnhlomohang/"
#property version   "1.00"
#include <Trade/Trade.mqh>

CTrade trade;
```

In this code, the line #include <Trade/Trade.mqh> brings in the MetaTrader 5 trading library, which provides the built-in CTrade class. This class contains all the essential functions needed to automate trade execution, such as opening, closing, modifying, or managing positions and orders. After including the library, we create an instance of the class with CTrade trade;. This object acts as our interface to the trading system—through it, we can call methods like trade.Buy(), trade.Sell(), or trade.PositionClose(). Since our goal is to automate trading decisions from the sentiment indicator, including this library and defining a CTrade object is necessary to actually send and manage trades programmatically.

```
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input group "Timeframe Settings"
input ENUM_TIMEFRAMES HigherTF = PERIOD_H4;     // Higher Timeframe
input ENUM_TIMEFRAMES LowerTF1 = PERIOD_H1;     // Lower Timeframe 1
input ENUM_TIMEFRAMES LowerTF2 = PERIOD_M30;    // Lower Timeframe 2

input group "Indicator Settings"
input int MAPeriod = 200;                       // EMA Period
input int SwingLookback = 5;                    // Swing Lookback Bars
input double ATRThreshold = 0.002;              // ATR Threshold for Neutral

input group "Trading Settings"
input double LotSize = 0.1;                     // Trade Lot Size
input int StopLossPips = 50;                    // Stop Loss in Pips
input int TakeProfitPips = 100;                 // Take Profit in Pips
input int MagicNumber = 12345;                  // Magic Number for trades
input int Slippage = 3;                         // Slippage in points

input group "Trailing Stop Parameters"
input bool UseTrailingStop = true;              // Enable trailing stops
input int BreakEvenAtPips = 500;                // Move to breakeven at this profit (pips)
input int TrailStartAtPips = 600;               // Start trailing at this profit (pips)
input int TrailStepPips = 100;                  // Trail by this many pips

input group "Risk Management"
input bool UseRiskManagement = true;            // Enable Risk Management
input double RiskPercent = 2.0;                 // Risk Percentage per Trade
input int MinBarsBetweenTrades = 3;             // Minimum bars between trades

input group "Visual Settings"
input bool ShowPanel = true;                    // Show Information Panel
input int PanelCorner = 1;                      // Panel Corner: 0=TopLeft,1=TopRight,2=BottomLeft,3=BottomRight
input int PanelX = 10;                          // Panel X Position
input int PanelY = 10;                          // Panel Y Position
input string FontFace = "Arial";                // Font Face
input int FontSize = 10;                        // Font Size
input color BullishColor = clrLimeGreen;        // Bullish Color
input color BearishColor = clrRed;              // Bearish Color
input color RiskOnColor = clrDodgerBlue;        // Risk-On Color
input color RiskOffColor = clrDarkGray;         // Risk-Off Color
input color NeutralColor = clrGold;             // Neutral Color
```

Here, the first section groups inputs related to timeframe configuration and indicator settings. By defining HigherTF, LowerTF1, and LowerTF2, the Expert Advisor can analyze market sentiment across multiple layers of price action, such as a higher timeframe for trend direction and lower timeframes for structure confirmation. Under indicator settings, parameters like MAPeriod (the period for the Exponential Moving Average), SwingLookback (the number of bars to check for swing highs and lows), and ATRThreshold (used to detect neutral or choppy markets) give traders flexibility to fine-tune how sentiment is detected.

The second section controls trading behavior and risk management. Inputs such as LotSize, StopLossPips, and TakeProfitPips define the core trade execution parameters, while MagicNumber helps the EA distinguish its trades from others, and Slippage manages order execution tolerance. Trailing stop parameters (UseTrailingStop, BreakEvenAtPips, TrailStartAtPips, TrailStepPips) allow dynamic protection of profits once a trade moves favorably. In addition, the risk management block (UseRiskManagement, RiskPercent, MinBarsBetweenTrades) ensures position sizing aligns with account equity, prevents overexposure, and enforces a minimum spacing between trades for discipline.

The final section focuses on visual presentation of the sentiment indicator. Options like ShowPanel, PanelCorner, PanelX, and PanelY allow the trader to control where the sentiment dashboard appears on the chart. Font customization (FontFace, FontSize) ensures readability, while color-coded inputs (BullishColor, BearishColor, RiskOnColor, RiskOffColor, NeutralColor) provide instant visual cues for each sentiment type. This customization enhances clarity and usability, enabling traders to quickly assess market conditions at a glance.

```
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
int higherTFHandle, lowerTF1Handle, lowerTF2Handle;
double higherTFMA[], lowerTF1MA[], lowerTF2MA[];
datetime lastUpdateTime = 0;
string indicatorName = "MarketSentEA";
int currentSentiment = 0;
int previousSentiment = 0;
datetime lastTradeTime = 0;
double global_PipValueInPoints; // Stores pip value in points
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    indicatorName = "MarketSentEA_" + IntegerToString(ChartID());

    // Create handles for MA indicators on different timeframes
    higherTFHandle = iMA(_Symbol, HigherTF, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);
    lowerTF1Handle = iMA(_Symbol, LowerTF1, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);
    lowerTF2Handle = iMA(_Symbol, LowerTF2, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);

    // Set array as series
    ArraySetAsSeries(higherTFMA, true);
    ArraySetAsSeries(lowerTF1MA, true);
    ArraySetAsSeries(lowerTF2MA, true);

     //global_PipValueInPoints = GetPipValueInPoints();

    // Create panel if enabled
    if(ShowPanel)
        CreatePanel();

    // Check for trading permission
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
        Print("Error: Trading is not allowed! Check AutoTrading permission.");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Delete all objects
    ObjectsDeleteAll(0, indicatorName);
    IndicatorRelease(higherTFHandle);
    IndicatorRelease(lowerTF1Handle);
    IndicatorRelease(lowerTF2Handle);
}
```

Here, we define global variables that the expert advisor will use throughout it's execution. Handles (higherTFHandle, lowerTF1Handle, and lowerTF2Handle) are created to store references to Moving Average (MA) indicators across different timeframes. Arrays (higherTFMA, lowerTF1MA, lowerTF2MA) hold the calculated values of these indicators for later analysis. Variables like lastUpdateTime, lastTradeTime, and currentSentiment/previousSentiment help track when calculations or trades were last performed and manage sentiment changes. The indicatorName string uniquely identifies objects drawn on the chart, while global\_PipValueInPoints will store pip-to-point conversions for risk and trade size calculations.

The OnInit() function initializes the Expert Advisor. It sets a unique indicator name (tied to the chart ID), creates MA handles for the chosen timeframes, and configures arrays for time-series data. It also includes logic to create a panel. The OnDeinit() function is responsible for cleanup when the EA is removed—deleting chart objects associated with the indicator and releasing the MA handles to free resources. Together, these functions ensure the EA starts cleanly and exits without leaving clutter or consuming memory unnecessarily.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Update only on new bar or every 10 seconds
    if(TimeCurrent() - lastUpdateTime < 10)
        return;

    lastUpdateTime = TimeCurrent();

    // Get MA values
    if(CopyBuffer(higherTFHandle, 0, 0, 3, higherTFMA) < 3 ||
       CopyBuffer(lowerTF1Handle, 0, 0, 3, lowerTF1MA) < 3 ||
       CopyBuffer(lowerTF2Handle, 0, 0, 3, lowerTF2MA) < 3)
    {
        Print("Error copying indicator buffers");
        return;
    }

    // Get current prices
    double higherTFPrice = iClose(_Symbol, HigherTF, 0);
    double lowerTF1Price = iClose(_Symbol, LowerTF1, 0);
    double lowerTF2Price = iClose(_Symbol, LowerTF2, 0);

    // Determine higher timeframe bias
    int higherTFBias = GetHigherTFBias(higherTFPrice, higherTFMA[0]);

    // Determine lower timeframe structures
    bool lowerTF1Bullish = IsBullishStructure(LowerTF1, SwingLookback);
    bool lowerTF1Bearish = IsBearishStructure(LowerTF1, SwingLookback);
    bool lowerTF2Bullish = IsBullishStructure(LowerTF2, SwingLookback);
    bool lowerTF2Bearish = IsBearishStructure(LowerTF2, SwingLookback);

    // Check for breakouts (BOS)
    bool lowerTF1Breakout = HasBreakout(LowerTF1, SwingLookback, higherTFBias);
    bool lowerTF2Breakout = HasBreakout(LowerTF2, SwingLookback, higherTFBias);

    // Determine final sentiment
    previousSentiment = currentSentiment;
    currentSentiment = DetermineSentiment(
        higherTFBias,
        lowerTF1Bullish, lowerTF1Bearish, lowerTF1Breakout,
        lowerTF2Bullish, lowerTF2Bearish, lowerTF2Breakout
    );

    // Update panel if enabled
    if(ShowPanel)
        UpdatePanel(higherTFBias, currentSentiment);

    // Check if we should trade
    CheckForTradingOpportunity();

    if(UseTrailingStop) ManageOpenTrades();
}
```

The OnTick() function is the core execution loop of the Expert Advisor, running every time the market generates a new tick. To optimize performance, it updates only once every 10 seconds or on a new bar, preventing unnecessary calculations. It first retrieves Moving Average values from the predefined timeframes using CopyBuffer and compares them with current prices obtained via iClose(). Based on this data, the higher timeframe bias is established, while bullish and bearish structures on the lower timeframes are identified, along with potential breakouts.

The function then determines the overall market sentiment by combining these conditions, updates the on-chart panel if enabled, and checks for possible trading opportunities. Finally, if trailing stops are active, it manages open trades dynamically to protect profits. This ensures the EA continually analyzes sentiment, updates visuals, and executes trades in a structured and automated manner.

```
//+------------------------------------------------------------------+
//| Determine higher timeframe bias                                  |
//+------------------------------------------------------------------+
int GetHigherTFBias(double price, double maValue)
{
    double deviation = MathAbs(price - maValue) / maValue;

    if(price > maValue && deviation > ATRThreshold)
        return 1; // Bullish
    else if(price < maValue && deviation > ATRThreshold)
        return -1; // Bearish
    else
        return 0; // Neutral
}

//+------------------------------------------------------------------+
//| Check for bullish structure (HH, HL)                             |
//+------------------------------------------------------------------+
bool IsBullishStructure(ENUM_TIMEFRAMES tf, int lookback)
{
    // Get swing highs and lows
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback * 2, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback * 2, 1);

    // Get previous swings for comparison
    int prevSwingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback, lookback + 1);
    int prevSwingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback, lookback + 1);

    if(swingHighIndex == -1 || swingLowIndex == -1 ||
       prevSwingHighIndex == -1 || prevSwingLowIndex == -1)
        return false;

    double swingHigh = iHigh(_Symbol, tf, swingHighIndex);
    double swingLow = iLow(_Symbol, tf, swingLowIndex);
    double prevSwingHigh = iHigh(_Symbol, tf, prevSwingHighIndex);
    double prevSwingLow = iLow(_Symbol, tf, prevSwingLowIndex);

    // Check for higher high and higher low
    return (swingHigh > prevSwingHigh && swingLow > prevSwingLow);
}

//+------------------------------------------------------------------+
//| Check for bearish structure (LH, LL)                             |
//+------------------------------------------------------------------+
bool IsBearishStructure(ENUM_TIMEFRAMES tf, int lookback)
{
    // Get swing highs and lows
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback * 2, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback * 2, 1);

    // Get previous swings for comparison
    int prevSwingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback, lookback + 1);
    int prevSwingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback, lookback + 1);

    if(swingHighIndex == -1 || swingLowIndex == -1 ||
       prevSwingHighIndex == -1 || prevSwingLowIndex == -1)
        return false;

    double swingHigh = iHigh(_Symbol, tf, swingHighIndex);
    double swingLow = iLow(_Symbol, tf, swingLowIndex);
    double prevSwingHigh = iHigh(_Symbol, tf, prevSwingHighIndex);
    double prevSwingLow = iLow(_Symbol, tf, prevSwingLowIndex);

    // Check for lower high and lower low
    return (swingHigh < prevSwingHigh && swingLow < prevSwingLow);
}

//+------------------------------------------------------------------+
//| Check for breakout of structure                                  |
//+------------------------------------------------------------------+
bool HasBreakout(ENUM_TIMEFRAMES tf, int lookback, int higherTFBias)
{
    // Get recent swing points
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback, 1);

    if(swingHighIndex == -1 || swingLowIndex == -1)
        return false;

    double swingHigh = iHigh(_Symbol, tf, swingHighIndex);
    double swingLow = iLow(_Symbol, tf, swingLowIndex);
    double currentPrice = iClose(_Symbol, tf, 0);

    // Check for breakout based on higher timeframe bias
    if(higherTFBias == 1) // Bullish bias - look for breakout above swing high
        return (currentPrice > swingHigh);
    else if(higherTFBias == -1) // Bearish bias - look for breakout below swing low
        return (currentPrice < swingLow);

    return false;
}
```

The GetHigherTFBias() function compares the current price against a moving average value on the higher timeframe to determine the dominant trend. By calculating the relative deviation between the two, it ensures that small fluctuations near the moving average are treated as neutral. If the price is meaningfully above the moving average, the function returns 1 for a bullish bias, while a price significantly below returns -1 for a bearish bias. If neither condition is met, the bias defaults to 0, representing a neutral state. This provides the EA with a clear directional context before analyzing lower timeframes.

The functions IsBullishStructure() and IsBearishStructure() examine swing highs and lows on the specified timeframe to confirm structural patterns. For bullish structure, the code checks whether the most recent swing high is higher than the previous swing high, and the most recent swing low is also higher than the previous swing low—indicating higher highs and higher lows (HH, HL). Conversely, the bearish check validates lower highs and lower lows (LH, LL). These structural validations add another layer of confirmation, ensuring trades are aligned not just with moving average bias but also with actual market structure.

The HasBreakout() function looks for price breaking out of recent swing levels in the direction of the higher timeframe bias. For example, with a bullish bias, it confirms whether the current price has broken above the most recent swing high, signaling continuation. In a bearish bias, it checks if the price has broken below the latest swing low. If no clear bias exists, no breakout is considered valid. This function prevents premature trades by ensuring entries only occur when price action aligns with both structure and higher timeframe momentum, strengthening the reliability of signals.

```
//+------------------------------------------------------------------+
//| Determine final sentiment                                        |
//+------------------------------------------------------------------+
int DetermineSentiment(int higherTFBias,
                      bool tf1Bullish, bool tf1Bearish, bool tf1Breakout,
                      bool tf2Bullish, bool tf2Bearish, bool tf2Breakout)
{
    // Bullish sentiment
    if(higherTFBias == 1 && tf1Bullish && tf2Bullish)
        return 1; // Bullish

    // Bearish sentiment
    if(higherTFBias == -1 && tf1Bearish && tf2Bearish)
        return -1; // Bearish

    // Risk-on sentiment
    if(higherTFBias == 1 && (tf1Breakout || tf2Breakout))
        return 2; // Risk-On

    // Risk-off sentiment
    if(higherTFBias == -1 && (tf1Breakout || tf2Breakout))
        return -2; // Risk-Off

    // Neutral/choppy sentiment
    return 0; // Neutral
}

//+------------------------------------------------------------------+
//| Check for trading opportunity                                    |
//+------------------------------------------------------------------+
void CheckForTradingOpportunity()
{
    // Check if enough time has passed since last trade
    if(TimeCurrent() - lastTradeTime < PeriodSeconds(_Period) * MinBarsBetweenTrades)
        return;

    // Check if sentiment has changed
    if(currentSentiment == previousSentiment)
        return;

    // Close existing positions if sentiment changed significantly
    if((previousSentiment > 0 && currentSentiment < 0) ||
       (previousSentiment < 0 && currentSentiment > 0))
    {
        CloseAllPositions();
    }

    // Execute new trades based on sentiment
    switch(currentSentiment)
    {
        case 1: // Bullish - Buy
        case 2: // Risk-On - Buy
            ExecuteTrade(ORDER_TYPE_BUY, _Symbol);
            lastTradeTime = TimeCurrent();
            break;

        case -1: // Bearish - Sell
        case -2: // Risk-Off - Sell
            ExecuteTrade(ORDER_TYPE_SELL, _Symbol);
            lastTradeTime = TimeCurrent();
            break;

        case 0: // Neutral - Close all positions
            CloseAllPositions();
            break;
    }
}

//+------------------------------------------------------------------+
//| Execute trade with risk parameters                               |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE tradeType, string symbol)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double price = (tradeType == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) :
                                                SymbolInfoDouble(symbol, SYMBOL_BID);

   // Convert StopLoss and TakeProfit from pips to actual price distances
   double sl_distance = StopLossPips * point;
   double tp_distance = TakeProfitPips * point;

   double sl = (tradeType == ORDER_TYPE_BUY) ? price - sl_distance :
                                             price + sl_distance;

   double tp = (tradeType == ORDER_TYPE_BUY) ? price + tp_distance :
                                             price - tp_distance;

   trade.PositionOpen(symbol, tradeType, LotSize, price, sl, tp, NULL);
}
```

The DetermineSentiment() function consolidates signals from higher and lower timeframes into a single sentiment classification. If the higher timeframe bias is bullish and both lower timeframes confirm bullish structures, the function returns a bullish sentiment. Similarly, if all conditions align on the bearish side, it signals bearish sentiment. When the higher timeframe is bullish but only breakouts occur on lower timeframes, the sentiment is marked as risk-on, while bearish bias with breakouts signals risk-off. If none of these conditions are met, the function defaults to neutral, representing a choppy or indecisive market. This structured hierarchy ensures sentiment is consistent, context-aware, and reflects both trend and breakout conditions.

The CheckForTradingOpportunity() function acts on the sentiment signals to automate trades. It enforces rules like spacing trades out over time (MinBarsBetweenTrades) and reacting only when sentiment changes from the previous state. If sentiment flips direction, existing positions are closed before new trades are placed, ensuring alignment with the latest market outlook. Depending on the sentiment, the EA executes buy orders for bullish or risk-on conditions, sell orders for bearish or risk-off conditions, or closes all trades during neutral phases. The ExecuteTrade() function manages order placement with risk controls by calculating stop-loss and take-profit levels in price terms, while the PipsToPrice() helper ensures compatibility with symbols that use different pip formats (e.g., 3/5-digit pricing). Together, these functions create a complete loop from sentiment detection to automated, risk-managed trade execution.

```
//+------------------------------------------------------------------+
//| Trailing stop function                                           |
//+------------------------------------------------------------------+
void ManageOpenTrades()
{
   if(!UseTrailingStop) return;

   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      // get ticket (PositionGetTicket returns ulong; it also selects the position)
      ulong ticket = PositionGetTicket(i);                  // correct usage. :contentReference[oaicite:3]{index=3}
      if(ticket == 0) continue;

      // ensure the position is selected (recommended)
      if(!PositionSelectByTicket(ticket)) continue;

      // Optional: only operate on same symbol or your EA's magic number
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      // if(PositionGetInteger(POSITION_MAGIC) != MyMagicNumber) continue;

      // read position properties AFTER selecting
      double open_price   = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_price= PositionGetDouble(POSITION_PRICE_CURRENT); // current price for the position. :contentReference[oaicite:4]{index=4}
      double current_sl   = PositionGetDouble(POSITION_SL);
      double current_tp   = PositionGetDouble(POSITION_TP);
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // pip size
      double pip_price = PipsToPrice(1);

      // profit in pips (use current_price returned above)
      double profit_price = (pos_type == POSITION_TYPE_BUY) ? (current_price - open_price)
                                                             : (open_price - current_price);
      double profit_pips = profit_price / pip_price;
      if(profit_pips <= 0) continue;

      // get broker min stop distance (in price units)
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double stop_level_points = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL); // integer property. :contentReference[oaicite:5]{index=5}
      double stopLevelPrice = stop_level_points * point;

      // get market Bid/Ask for stop-level checks
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      // -------------------------
      // 1) Move to breakeven
      // -------------------------
      if(profit_pips >= BreakEvenAtPips)
      {
         double breakeven = open_price;
         // small adjustment to help account for spread/commissions (optional)
         if(pos_type == POSITION_TYPE_BUY)  breakeven += point;
         else                                breakeven -= point;

         // Check stop-level rules: for BUY SL must be >= (bid - stopLevelPrice) distance below bid
         if(pos_type == POSITION_TYPE_BUY)
         {
            if((bid - breakeven) >= stopLevelPrice) // allowed by server
            {
               if(breakeven > current_sl) // only move SL up
               {
                  if(!trade.PositionModify(ticket, NormalizeDouble(breakeven, _Digits), current_tp))
                     PrintFormat("PositionModify failed (BE) ticket %I64u error %d", ticket, GetLastError());
               }
            }
         }
         else // SELL
         {
            if((breakeven - ask) >= stopLevelPrice)
            {
               if(current_sl == 0.0 || breakeven < current_sl) // move SL down
               {
                  if(!trade.PositionModify(ticket, NormalizeDouble(breakeven, _Digits), current_tp))
                     PrintFormat("PositionModify failed (BE) ticket %I64u error %d", ticket, GetLastError());
               }
            }
         }
      } // end breakeven

      // -------------------------
      // 2) Trailing in steps after TrailStartAtPips
      // -------------------------
      if(profit_pips >= TrailStartAtPips)
      {
         double extra_pips = profit_pips - TrailStartAtPips;
         int step_count = (int)(extra_pips / TrailStepPips);

         // compute desired SL relative to open_price (as per your original request)
         double desiredOffsetPips = (double)(TrailStartAtPips + step_count * TrailStepPips);
         double new_sl_price;

         if(pos_type == POSITION_TYPE_BUY)
         {
            new_sl_price = open_price + PipsToPrice((int)desiredOffsetPips);
            // ensure new SL respects server min distance from current Bid
            if((bid - new_sl_price) < stopLevelPrice)
               new_sl_price = bid - stopLevelPrice;

            if(new_sl_price > current_sl) // only move SL up
            {
               if(!trade.PositionModify(ticket, NormalizeDouble(new_sl_price, _Digits), current_tp))
                  PrintFormat("PositionModify failed (Trail Buy) ticket %I64u error %d", ticket, GetLastError());
            }
         }
         else // SELL
         {
            new_sl_price = open_price - PipsToPrice((int)desiredOffsetPips);
            // ensure new SL respects server min distance from current Ask
            if((new_sl_price - ask) < stopLevelPrice)
               new_sl_price = ask + stopLevelPrice;

            if(current_sl == 0.0 || new_sl_price < current_sl) // only move SL down (more profitable)
            {
               if(!trade.PositionModify(ticket, NormalizeDouble(new_sl_price, _Digits), current_tp))
                  PrintFormat("PositionModify failed (Trail Sell) ticket %I64u error %d", ticket, GetLastError());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                MqlTradeRequest request = {};
                MqlTradeResult result = {};

                request.action = TRADE_ACTION_DEAL;
                request.symbol = _Symbol;
                request.volume = PositionGetDouble(POSITION_VOLUME);
                request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ?
                              ORDER_TYPE_SELL : ORDER_TYPE_BUY;
                request.price = SymbolInfoDouble(_Symbol,
                    (request.type == ORDER_TYPE_BUY) ? SYMBOL_ASK : SYMBOL_BID);
                request.deviation = Slippage;
                request.magic = MagicNumber;
                request.comment = "MarketSentEA Close";

                if(OrderSend(request, result))
                {
                    if(result.retcode == TRADE_RETCODE_DONE)
                        Print("Position closed successfully. Ticket: ", ticket);
                    else
                        Print("Error closing position. Code: ", result.retcode);
                }
            }
        }
    }
}
```

Here, this section of the EA focuses on two major risk management mechanisms: trailing stop management and forced position closure. The ManageOpenTrades() function first ensures trailing stops are only applied when enabled, then loops through all open positions. It checks profitability in pips, applies a breakeven stop once the minimum profit is reached, and continues with step-based trailing as the position moves further into profit. Broker stop-level restrictions and spread considerations are integrated to ensure all stop modifications are valid and safe.

Together, these functions create a robust exit and protection system for trades, reducing risk while maximizing gains. By combining break-even logic, step-based trailing, and a reliable mechanism to close all positions when sentiment or strategy rules demand it, the EA can protect capital during adverse conditions and lock in profits as trades move favorably. This ultimately improves the strategy’s consistency, resilience, and long-term profitability.

```
//+------------------------------------------------------------------+
//| Calculate lot size with risk management                          |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLossPrice)
{
    if(!UseRiskManagement)
        return LotSize;

    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * RiskPercent / 100.0;

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);

    // Calculate stop loss in points
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double slPoints = MathAbs(currentPrice - stopLossPrice) / point;

    // Calculate lot size
    double lotSize = riskAmount / (slPoints * tickValue);

    // Normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    lotSize = MathRound(lotSize / lotStep) * lotStep;

    return lotSize;
}

//+------------------------------------------------------------------+
//| Helper: convert timeframe to string                              |
//+------------------------------------------------------------------+
string TFtoString(int tf)
{
   switch(tf)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN";
      default: return "TF?";
   }
}

//+------------------------------------------------------------------+
//| Create information panel                                         |
//+------------------------------------------------------------------+
void CreatePanel()
{
   //--- background panel
   string bg = indicatorName + "_BG";
   ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, PanelX);
   ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, PanelY);
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, 200);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, 120);
   ObjectSetInteger(0, bg, OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, clrBlack);
   ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bg, OBJPROP_BORDER_COLOR, clrWhite);
   ObjectSetInteger(0, bg, OBJPROP_BACK, true);
   ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, bg, OBJPROP_ZORDER, 0);

   //--- title
   string title = indicatorName + "_Title";
   ObjectCreate(0, title, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, title, OBJPROP_XDISTANCE, PanelX + 10);
   ObjectSetInteger(0, title, OBJPROP_YDISTANCE, PanelY + 10);
   ObjectSetInteger(0, title, OBJPROP_CORNER, PanelCorner);
   ObjectSetString (0, title, OBJPROP_TEXT, "Market Sentiment EA");
   ObjectSetInteger(0, title, OBJPROP_COLOR, clrWhite);
   ObjectSetString (0, title, OBJPROP_FONT, FontFace);
   ObjectSetInteger(0, title, OBJPROP_FONTSIZE, FontSize);
   ObjectSetInteger(0, title, OBJPROP_BACK, false);
   ObjectSetInteger(0, title, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, title, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, title, OBJPROP_ZORDER, 1);

   //--- timeframe labels + sentiment placeholders
   string timeframes[3] = { TFtoString(HigherTF), TFtoString(LowerTF1), TFtoString(LowerTF2) };
   for(int i=0; i<3; i++)
   {
      string tfLabel = indicatorName + "_TF" + IntegerToString(i);
      ObjectCreate(0, tfLabel, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, tfLabel, OBJPROP_XDISTANCE, PanelX + 10);
      ObjectSetInteger(0, tfLabel, OBJPROP_YDISTANCE, PanelY + 30 + i * 20);
      ObjectSetInteger(0, tfLabel, OBJPROP_CORNER, PanelCorner);
      ObjectSetString (0, tfLabel, OBJPROP_TEXT, timeframes[i] + ":");
      ObjectSetInteger(0, tfLabel, OBJPROP_COLOR, clrLightGray);
      ObjectSetString (0, tfLabel, OBJPROP_FONT, FontFace);
      ObjectSetInteger(0, tfLabel, OBJPROP_FONTSIZE, FontSize);
      ObjectSetInteger(0, tfLabel, OBJPROP_BACK, false);
      ObjectSetInteger(0, tfLabel, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, tfLabel, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, tfLabel, OBJPROP_ZORDER, 1);

      string sentLabel = indicatorName + "_Sentiment" + IntegerToString(i);
      ObjectCreate(0, sentLabel, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, sentLabel, OBJPROP_XDISTANCE, PanelX + 100);
      ObjectSetInteger(0, sentLabel, OBJPROP_YDISTANCE, PanelY + 30 + i * 20);
      ObjectSetInteger(0, sentLabel, OBJPROP_CORNER, PanelCorner);
      ObjectSetString (0, sentLabel, OBJPROP_TEXT, "N/A");
      ObjectSetInteger(0, sentLabel, OBJPROP_COLOR, NeutralColor);
      ObjectSetString (0, sentLabel, OBJPROP_FONT, FontFace);
      ObjectSetInteger(0, sentLabel, OBJPROP_FONTSIZE, FontSize);
      ObjectSetInteger(0, sentLabel, OBJPROP_BACK, false);
      ObjectSetInteger(0, sentLabel, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, sentLabel, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, sentLabel, OBJPROP_ZORDER, 1);
   }

   //--- final sentiment label
   string fnl = indicatorName + "_Final";
   ObjectCreate(0, fnl, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, fnl, OBJPROP_XDISTANCE, PanelX + 10);
   ObjectSetInteger(0, fnl, OBJPROP_YDISTANCE, PanelY + 100);
   ObjectSetInteger(0, fnl, OBJPROP_CORNER, PanelCorner);
   ObjectSetString (0, fnl, OBJPROP_TEXT, "Final: Neutral");
   ObjectSetInteger(0, fnl, OBJPROP_COLOR, NeutralColor);
   ObjectSetString (0, fnl, OBJPROP_FONT, FontFace);
   ObjectSetInteger(0, fnl, OBJPROP_FONTSIZE, FontSize + 2);
   ObjectSetInteger(0, fnl, OBJPROP_BACK, false);
   ObjectSetInteger(0, fnl, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, fnl, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, fnl, OBJPROP_ZORDER, 1);
}

//+------------------------------------------------------------------+
//| Update information panel                                         |
//+------------------------------------------------------------------+
void UpdatePanel(int higherTFBias, int sentiment)
{
    // Update higher timeframe sentiment
    string higherTFSentiment = "Neutral";
    color higherTFColor = NeutralColor;

    if(higherTFBias == 1)
    {
        higherTFSentiment = "Bullish";
        higherTFColor = BullishColor;
    }
    else if(higherTFBias == -1)
    {
        higherTFSentiment = "Bearish";
        higherTFColor = BearishColor;
    }

    ObjectSetString(0, indicatorName + "_Sentiment0", OBJPROP_TEXT, higherTFSentiment);
    ObjectSetInteger(0, indicatorName + "_Sentiment0", OBJPROP_COLOR, higherTFColor);

    // Update final sentiment
    string finalSentiment = "Neutral";
    color finalColor = NeutralColor;

    switch(sentiment)
    {
        case 1:
            finalSentiment = "Bullish";
            finalColor = BullishColor;
            break;
        case -1:
            finalSentiment = "Bearish";
            finalColor = BearishColor;
            break;
        case 2:
            finalSentiment = "Risk-On";
            finalColor = RiskOnColor;
            break;
        case -2:
            finalSentiment = "Risk-Off";
            finalColor = RiskOffColor;
            break;
        default:
            finalSentiment = "Neutral";
            finalColor = NeutralColor;
    }

    ObjectSetString(0, indicatorName + "_Final", OBJPROP_TEXT, "Final: " + finalSentiment);
    ObjectSetInteger(0, indicatorName + "_Final", OBJPROP_COLOR, finalColor);
}
```

The CalculateLotSize function provides dynamic position sizing based on account balance and risk percentage. Instead of using a fixed lot size, it calculates the appropriate volume by measuring the distance from entry to stop-loss, converting that into points, and dividing the risk capital by the potential loss per pip. This ensures each trade risks a consistent portion of equity regardless of volatility or instrument price, while also adjusting the final lot size to broker limits such as minimum, maximum, and step size.

The helper function TFtoString translates numeric timeframe constants into human-readable strings. This is essential for displaying meaningful information on panels, dashboards, or logs, allowing traders to quickly understand which timeframe bias or signals are being referenced. By converting these integer codes into clear text such as "M5" or "H1," the code becomes more user-friendly and avoids confusion when tracking multiple timeframes.

The CreatePanel and UpdatePanel functions handle the visual interface of the EA. CreatePanel initializes a clean dashboard with background, title, and sentiment placeholders for each monitored timeframe, plus a final sentiment status. UpdatePanel dynamically updates these labels with the current higher timeframe bias and final calculated sentiment, changing both the text and color to visually represent bullish, bearish, risk-on, risk-off, or neutral states. Together, they create a live and intuitive monitoring panel that allows traders to quickly gauge market conditions without scanning raw data.

### **Backtest Results**

The back-testing was evaluated on the 1H timeframe across roughly a 2-month testing window (17 February 2025 to 06 May 2025), with the following settings:

![](https://c.mql5.com/2/170/input__1.png)

![](https://c.mql5.com/2/170/Mark_curv__1.png)

![](https://c.mql5.com/2/170/Mark_BT__1.png)

### Conclusion

In summary, we designed and implemented a complete framework to automate the market sentiment indicator. This included building sentiment detection logic across multiple timeframes, defining clear categories such as bullish, bearish, risk-on, risk-off, and neutral, and coding the trade execution rules to act on those signals. We also integrated essential features like risk management through dynamic lot sizing, trailing stop functions for securing profits, and a real-time information panel to visually display sentiment conditions. Altogether, these components transformed the indicator into a fully automated trading system capable of both analysis and execution.

In conclusion, this automation helps traders by removing emotional decision-making and ensuring consistency in how trades are taken and managed. By combining sentiment analysis, structured trade rules, and built-in risk management, traders can navigate the market with a more disciplined approach. The system provides a clear visual interface for transparency while executing trades automatically in alignment with sentiment, allowing traders to save time, minimize errors, and operate with a professional-grade tool that adapts to changing market conditions.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19609.zip "Download all attachments in the single ZIP archive")

[Mark\_Sent\_Auto.mq5](https://www.mql5.com/en/articles/download/19609/Mark_Sent_Auto.mq5 "Download Mark_Sent_Auto.mq5")(62.08 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**[Go to discussion](https://www.mql5.com/en/forum/496034)**

![Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://c.mql5.com/2/171/19383-overcoming-the-limitation-of-logo__1.png)[Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://www.mql5.com/en/articles/19383)

Machine learning is often viewed through statistical or linear algebraic lenses, but this article emphasizes a geometric perspective of model predictions. It demonstrates that models do not truly approximate the target but rather map it onto a new coordinate system, creating an inherent misalignment that results in irreducible error. The article proposes that multi-step predictions, comparing the model’s forecasts across different horizons, offer a more effective approach than direct comparisons with the target. By applying this method to a trading model, the article demonstrates significant improvements in profitability and accuracy without changing the underlying model.

![Building AI-Powered Trading Systems in MQL5 (Part 2): Developing a ChatGPT-Integrated Program with User Interface](https://c.mql5.com/2/171/19567-building-ai-powered-trading-logo.png)[Building AI-Powered Trading Systems in MQL5 (Part 2): Developing a ChatGPT-Integrated Program with User Interface](https://www.mql5.com/en/articles/19567)

In this article, we develop a ChatGPT-integrated program in MQL5 with a user interface, leveraging the JSON parsing framework from Part 1 to send prompts to OpenAI’s API and display responses on a MetaTrader 5 chart. We implement a dashboard with an input field, submit button, and response display, handling API communication and text wrapping for user interaction.

![Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://c.mql5.com/2/170/19436-perehodim-na-mql5-algo-forge-logo.png)[Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)

Let's explore how you can start integrating external code from any repository in the MQL5 Algo Forge storage into your own project. In this article, we finally turn to this promising, yet more complex, task: how to practically connect and use libraries from third-party repositories within MQL5 Algo Forge.

![Developing Trading Strategies with the Parafrac and Parafrac V2 Oscillators: Single Entry Performance Insights](https://c.mql5.com/2/170/19439-developing-trading-strategies-logo.png)[Developing Trading Strategies with the Parafrac and Parafrac V2 Oscillators: Single Entry Performance Insights](https://www.mql5.com/en/articles/19439)

This article introduces the ParaFrac Oscillator and its V2 model as trading tools. It outlines three trading strategies developed using these indicators. Each strategy was tested and optimized to identify their strengths and weaknesses. Comparative analysis highlighted the performance differences between the original and V2 models.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=mnowktjefumdmwqmicnegmwltdyabjxc&ssn=1769182273052154056&ssn_dr=0&ssn_sr=0&fv_date=1769182273&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19609&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20The%20Market%20Sentiment%20Indicator%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918227388976786&fz_uniq=5069510102098052673&sv=2552)

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