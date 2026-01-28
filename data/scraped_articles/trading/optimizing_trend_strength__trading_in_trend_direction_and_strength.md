---
title: Optimizing Trend Strength: Trading in Trend Direction and Strength
url: https://www.mql5.com/en/articles/19755
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 1
scraped_at: 2026-01-23T21:32:07.544983
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/19755&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071905813380935845)

MetaTrader 5 / Trading


### Introduction

This article aims to address both traders who have a difficulty in choosing the correct trade and trend direction and the traders who are looking to forge discipline and avoid or reduce losses by only taking trades in the trend direction by not trading against the trend or in short-term swings. It will also be excellent for the traders who have execution paralysis or overanalyze when they find a trade entry and hence end up losing the whole trade by not pushing the buttons and taking it because they could not identify the ideal trend direction and bias. I believe another beneficiary of this article is recently new or developing traders who are still grappling with the idea and difficulty in choosing the ideal trend direction, whether long-term, intraday, or overall trend strength and direction, to determine their trading decisions.

I believe most, if not all, the time, trading with strength and in trend direction can be the game changer and difference maker between Trader A and Trader B with equal skill and experience. The need for a trader to make profitable and excellent decisions in trading, especially intraday and long-term trades, is largely dependent on how one positions oneself relative to the trend direction and strength. Statistically it is often proven time and again that trading against the trend often offers short-term price movements, which are mostly retracements just before a big move in the direction of the trend, and if one is not careful, he might find himself with very heavy losses. While executions relative to trend strength and direction often prove more forgiving even if one has poor trade entry positions and at times can result in extra hundreds of dollars, if not thousands, made in profits if the trader is patient enough to wait for the trend to continue.

As a trader myself, I know traders are faced with countless challenges, in-fact most traders face almost similar, if not identical, challenges at some point in their trading careers, but what separates all of them is that different traders handle, react to, and anticipate this challenges differently; hence the difference in trading results, which has proven to be colossal even when traders who are evenly matched and separated by fine margins in skill, experience, composure, decision-making, and technique are different.

One of the major if not most common challenge is determining trend direction and strenght which this article aims to address. This is a very major challenge for developing and new traders and even some far more experienced traders, as it once was for me as a developing trader. With this article and EA, it will help align the trades executed relative to the trend, hence avoiding short-term trades against the trend. This will help developing traders to be more confident and discerning when executing trades, as they will eventually learn and grasp how to determine trends and directions of trade movements. And once this challenge of determining trade direction is solved, all that is left is finding the correct setup, time, and position to execute trades.

Also, trades analyzed, executed, traded, and tracked regarding the trend direction have proven not only to be more accurate faster but also to have very little drawdown as price moves quickly towards its target points with more speed and less drawdown. Besides, price swings are more powerful, meaning even if one is in drawdown, price will soon trade according to the trend's direction and strength narrative in play and will come out of drawdown.

This article will explore in detail an EA that is specifically designed for traders who are patient enough to execute, withstand, and hold their positions only when trading relative to the trend direction and strength and avoid trading against the trend direction. They refrain from contrarian trades and hold positions without changing their bias frequently until take-profit targets are hit or the overall trend shifts. This EA is also critical for intraday traders and short-term traders since it also helps them trade according to the most possible price movement regarding the trend, hence increasing the success rate.

![Moving average depicting a strong bull trend ](https://c.mql5.com/2/189/GBPUSDnew10.png)

### Introspection of Trading with Trend Strength and Direction

The main agenda of this article is to emphasize how trading in only the trend direction and strength, which are often ignored by most traders, especially those with limited experience, and even those with so much skin in the game, may overlook such an important aspect that may ultimately turn out to be a game changer.

The truth is that price moves from one key level to another key level on the higher time frame, such as MN, W1, and D1, and the direction in which price is moving towards in this higher time frame is what is referred to as the trend; whatever happens in the lower time frame is just a consequential eventuality. This is actually opposite to what the majority believe, which is that the lower timeframe is the one that determines or begins high timeframe trends, price movements, and trades begin and originate from lower timeframes and in turn determine higher timeframe price and price action.

On that note, this simply means that we can easily determine the trend of a trading pair or asset by paying close attention to the higher time frame price action, moving averages, and setups. Before we do anything, our focus should be quickly taken to the monthly chart (MN), and succeeding, drop to lower successive timeframes after we make our observations and even markings of the key areas and interest points. The key areas and interest points are often referred to with different names and terminologies when used in different strategies. They should be regarded as very key and important since they help us understand and interpret the trend and direction of price better and sometimes even help determine the speed and strength of an asset pair.

Some traders view them as order blocks, others as support and resistance, others as supply and demand zones, others as liquidity raids, and so on, but the most important thing all this strategies and theories agree on is that this points are very key and useful levels that are often points of interest where one can anticipate strong price reactions and also where most clinically accurate and professional traders hunt for their ideal setups. And that is what this article aims to exploit: how to create and use an EA that effectively makes use of this phenomenon.

![Moving averages depicting a bullish trend on D1 time frame ](https://c.mql5.com/2/189/GBPUSDnew11.png)

### Trend detection, weekly, monthly, and daily trend, and anticipating trend directions and strength

The first thing after logging into an MN chart is marking the key points (reaction areas) discussed earlier, which are usually located at previous weeks lows and highs. This key reaction area should always be kept in mind, as price usually always has a reaction here, and it may also determine how price will move for that month or the rest of the month. Hence, once we confirm price is moving from one key area to another on this MN timeframe, this will mean we would have determined the monthly trend direction, and trades on that month will be drawn towards the key areas on the MN timeframe.

This is what we may call a low-hanging fruit, as it is the closest reaction area based on the (MN) chart timeframe. Another point of interest would be moving averages, as they may be used to gauge the current strength and speed of price towards achieving its target, and the last key area to note is the price action, as it is a very key determinant of how price will move and may also alert one in case of a sudden change or break in trend. Moving averages may take time to show clear signals, but price action is way more informative, especially if you're keen enough and understand how to read price action.

Traders who are keen enough will notice from price action that for some long-term trends to begin, the price may come to a key area and quickly purge a low/high and then form engulfing candles quickly or a market structure shift. This may happen in an even lower time frame and is not a must; it occurs in the same time frame to be successful and effective. When this occurs, for example, price purges the previous week's low, and a bullish engulfing candle or market structure shift occurs, price quickly expands in the opposite direction; we may be witnessing the beginning of a new trend.

The goal is to patiently wait for the engulfing candle/market structure shift to complete and close, and once we are sure it is successful, we have identified our new trends and may enter buy positions relative to the new trend's direction. All executed trades should be made according to the current trend, as trades in the direction of the trend are faster and have cleaner price action with very minimal drawdown. It is also ideal not to trade against the trend until it changes again.

There will be swings and short-term reversals, especially in the lower time frame, but they won't really affect the long-term trend outlook. The end of trends or reversals may occur on the opposite end of the month or in key areas explained earlier. The same formula or theory is also present on the weekly chart and is used to determine the weekly trend, outlook, and narrative.

The last scenario this great and distinct feature occurs in is the D1 time frame, which I believe is a critical and fundamental time frame because it acts as a middle ground or bridge between two extremes: the lower time frames and the higher time frames. It is even probably the best time frame to frame intraday and short-term trades that range a few days or hours by just using the D1 time frame.

It is also relatively easier to spot both short-term swings and long-term swings on this time frame and, lastly, spot the narrative and trend. Being a very key timeframe here, traders should focus on price action, trend, and moving averages, as all the data needed lies and is anticipated and traded here to determine the trends and their direction.

Some important factors to note are that resistance and support areas are also key areas to watch out for, as well as supply and demand and order blocks, since they all refer to the same pattern with just different names, and even the overall trend may sometimes change depending on the geopolitical climate or factors supporting the traded asset.

The final golden nugget that I have saved for last is that to achieve optimum results and success, a trader should prioritize executing trades when both or all 3 time frames align and have the same bias, narrative, and trend. When W1, D1, and MN all speak the same language, those are some of the easiest trading days ever—very fast with minimum drawdowns or retracements.

Below, in the next chapters, I will elaborate on how to design and configure an EA that is based on trend analysis and uses it to determine its trade convictions.

![Example of market structure shit for trade entries on bearish trend ](https://c.mql5.com/2/189/GBPUSDnew12.png)

### Automating trading decisions with Trend King expert advisor in MQL5

To automate, illustrate, and implement this strategy, I have created:

- A trend following expert advisor to analyze trends and execute possible trade entry points only in the direction of the trend, incorporating trailing stop loss and moving averages to aid in trend detection.

**Decision-making process**

This expert advisor’s decisions (trend detection, price action detection, and trade execution) are driven by the following logic:

Source code of the trend king expert advisor:

This expert advisor detects potential trade signals that align with the overall trend direction and strength and waits for trade setups only in the trend direction, avoiding positions opposite to the trend. It also utilizes  exponential and simple moving averages to get the general trend direction to make trade entries and applies a trailing stop-loss logic to determine stop-loss levels and protect capital when price moves in its favor. It includes risk management (1% risk per trade).

Input Parameters: Configurable Trading Settings

The input parameters allow users to tailor the EA’s behavior for GBPJPY, GBPUSD and EURUSD trading without modifying the core code. \`LotSize\` is set to 0.1, a conservative size suitable for GBP pairs to limit risk in their moderate volatility (100–300 pips daily). \`LookbackBars\` (3) defines the window for identifying recent highs/lows on H1/H4/D1, balancing responsiveness with reliability. \`StopLossPips\` (150) and \`TakeProfitPips\` (600) are adjusted for GBPJPY, GBPUSD/EURUSD’s typical ranges, targeting a 1:4 risk-to-reward ratio to capitalize on strong trends post-purge. \`TrailingStopPips\` (100) ensures profits are locked in during GBPJPY’s sharper moves. \`EngulfingMinBodyRatio\` (0.3) ensures the engulfing candle’s body is significant relative to the previous candle, filtering weak signals.

\`TradeTimeframe\` (H1) and \`ConfirmationTimeframe\` (M15) allow flexible purge detection and faster entry confirmation, respectively. \`MaxCandlesPostPurge\` (3) limits the window for engulfing signals post-purge, avoiding late entries. \`VolumeThreshold\` (1.0) requires a volume spike to confirm liquidity, critical for GBPUSD’s news-driven moves. UseTrendFilter\` (true) and \`SMAPeriod\` (50) enable a 50-period SMA to ensure trades align with the broader trend, reducing false breakouts in choppy GBP markets.

```
//+------------------------------------------------------------------+
//|                 TrendKingEA.mq5                                  |
//| Expert Advisor for MetaTrader 5                                  |
//| Description:Trades based on Trend strenght and direction, waits  |
//| for trade set ups on trend direction, with flexible execution.   |
//+------------------------------------------------------------------+
#property copyright "Eugene Mmene"
#property link "https://www.EMcapital.com"
#property version "1.06"
//--- Input parameters
input double LotSize = 0.1;                              // Lot size for trades
input int LookbackBars = 5;                              // Number of bars to check for highs/lows
input double StopLossPips = 30.0;                        // Stop loss in pips (adjusted for GOLD#)
input double TakeProfitPips = 90.0;                      // Take profit in pips (adjusted for GOLD#)
input double TrailingStopPips = 40.0;                    // Trailing stop in pips
input double BreakevenPips = 30.0;                       // Move to BE after 30 pips profit
input double EngulfingMinBodyRatio = 0.5;                // Min body ratio for engulfing candle
input ENUM_TIMEFRAMES TradeTimeframe = PERIOD_M15;       // Primary timeframe for trading
input ENUM_TIMEFRAMES ConfirmationTimeframe = PERIOD_M5; // Lower timeframe for engulfing
input ENUM_TIMEFRAMES TrendTimeframe = PERIOD_H1;        // Trend from H1 SMA
input int MaxCandlesPostPurge = 5;                       // Max H1 candles to wait for engulfing
input double VolumeThreshold = 1.2;                      // Volume multiplier for liquidity confirmation
input bool UseTrendFilter = true;                        // Use SMA trend filter
input int SMAPeriod = 50;                                // SMA period for trend filter
```

OnInit Function: Setup and Validation for Robust Trading

The \`OnInit\` function is executed once when the EA is loaded onto the chart, ensuring proper setup for trading GBPUSD, GBPJPY, and EURUSD. It validates the \`TradeTimeframe\` (H1, H4, or D1) and \`ConfirmationTimeframe\` (M15 or M30) to ensure compatibility with the strategy’s multi-timeframe approach, rejecting invalid inputs to prevent errors. The function retrieves the symbol’s contract size using \`SymbolInfoDouble\`, critical for accurate position sizing in GBP pairs, where pip values differ (e.g., GBPJPY’s pip value is smaller due to JPY’s scale).

Global variables tracking highs/lows (\`lastHighH1\`, \`lastLowH1\`, etc.), candle times, and trade status are initialized to zero or false, resetting the EA’s state. The contract size check ensures the EA can calculate correct lot sizes, especially important for GBPJPY’s higher pip value sensitivity. A successful initialization logs the timeframe and contract size, confirming readiness for trading. This setup ensures the EA is robust and tailored to the specific market dynamics of EURUSD/GBPUSD/GBPJPY, unlike gold’s broader price swings.

```
int OnInit()
{
   if(TradeTimeframe != PERIOD_M15 && TradeTimeframe != PERIOD_M30)
   {
      Print("Invalid trade timeframe. Use M15 or M30.");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(ConfirmationTimeframe != PERIOD_M5 && ConfirmationTimeframe != PERIOD_M15)
   {
      Print("Invalid confirmation timeframe. Use M5 or M15.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   contractSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   if(contractSize == 0)
   {
      Print("Failed to get contract size for ", _Symbol);
      return(INIT_PARAMETERS_INCORRECT);
   }

   lastHigh = 0; lastLow = 0;
   lastCandleTime = 0;
   purgeDetected = false;
   lastPurgeTime = 0;

   Print("EA Initialized on ", EnumToString(TradeTimeframe), ", Confirmation TF: ", EnumToString(ConfirmationTimeframe), ", Contract Size: ", contractSize);
   return(INIT_SUCCEEDED);
}
```

IsNewCandle Function: Trigger Logic on New Bar

The \`IsNewCandle\` function checks if a new candle has formed on the primary trade timeframe (default H1) by comparing the current candle’s timestamp (\`iTime\`) with the last recorded candle time (\`lastCandleTime\`). If a new candle is detected, \`lastCandleTime\` is updated, and the function returns \`true\`, triggering the main trading logic in \`OnTick\`. This ensures the EA evaluates trade conditions only once per H1 candle, reducing processing overhead and preventing redundant checks during volatile intrabar movements in GBPUSD/GBPJPY. By focusing on new bars, the EA aligns with the strategy’s reliance on complete candle data for purge detection and engulfing patterns, critical for capturing significant price moves in GBP pairs’ trend-driven markets. This function is lightweight yet essential for maintaining the EA’s efficiency and accuracy.

```
bool IsNewCandle()
{
   datetime currentCandleTime = iTime(_Symbol, TradeTimeframe, 0);
   if(currentCandleTime != lastCandleTime)
   {
      lastCandleTime = currentCandleTime;
      return true;
   }
   return false;
}
```

UpdateHighsLows Function: Monitor Liquidity Purge Levels

The \`UpdateHighsLows\` function tracks the highest highs and lowest lows over the past \`LookbackBars\` (default 3) on H1, H4, and D1 timeframes, storing them in global variables (\`lastHighH1\`, \`lastLowH1\`, etc.). It uses \`CopyRates\` to fetch historical price data, with \`ArraySetAsSeries\` ensuring the most recent candle is at index 0. For each timeframe, it initializes the high/low with the first candle’s values and updates them by iterating through the lookback period, capturing the extreme prices.

This function is critical for detecting liquidity purges—when the current H1 candle breaks above a previous high or below a low on any timeframe, signaling a potential trend reversal or continuation in EURUSD GBPUSD, or GBPJPY. The multi-timeframe approach ensures the EA captures significant liquidity sweeps, such as those triggered by stop hunts in GBPJPY’s volatile moves. Error handling logs failures to load rates, ensuring robustness. This function lays the foundation for the strategy’s purge-based entry logic, tailored to GBP pairs’ sensitivity to key support/resistance levels.

```
void UpdateHighsLows()
{
   // H1
   MqlRates ratesH1[];
   ArraySetAsSeries(ratesH1, true);
   if(CopyRates(_Symbol, PERIOD_H1, 1, LookbackBars, ratesH1) >= LookbackBars)
   {
      lastHighH1 = ratesH1[0].high;
      lastLowH1 = ratesH1[0].low;
      for(int i = 1; i < LookbackBars; i++)
      {
         if(ratesH1[i].high > lastHighH1) lastHighH1 = ratesH1[i].high;
         if(ratesH1[i].low < lastLowH1) lastLowH1 = ratesH1[i].low;
      }
   }
   else Print("Failed to load H1 rates");

   // H4
   MqlRates ratesH4[];
   ArraySetAsSeries(ratesH4, true);
   if(CopyRates(_Symbol, PERIOD_H4, 1, LookbackBars, ratesH4) >= LookbackBars)
   {
      lastHighH4 = ratesH4[0].high;
      lastLowH4 = ratesH4[0].low;
      for(int i = 1; i < LookbackBars; i++)
      {
         if(ratesH4[i].high > lastHighH4) lastHighH4 = ratesH4[i].high;
         if(ratesH4[i].low < lastLowH4) lastLowH4 = ratesH4[i].low;
      }
   }
   else Print("Failed to load H4 rates");

   // D1
   MqlRates ratesD1[];
   ArraySetAsSeries(ratesD1, true);
   if(CopyRates(_Symbol, PERIOD_D1, 1, LookbackBars, ratesD1) >= LookbackBars)
   {
      lastHighD1 = ratesD1[0].high;
      lastLowD1 = ratesD1[0].low;
      for(int i = 1; i < LookbackBars; i++)
      {
         if(ratesD1[i].high > lastHighD1) lastHighD1 = ratesD1[i].high;
         if(ratesD1[i].low < lastLowD1) lastLowD1 = ratesD1[i].low;
      }
   }
   else Print("Failed to load D1 rates");
}
```

IsVolumeSpike Function: Validate Trade Liquidity

The \`IsVolumeSpike\` function confirms trade setups by checking if the current candle’s tick volume exceeds the average volume over \`LookbackBars\` (3) by a factor of \`VolumeThreshold\` (1.0). It uses \`CopyRates\` to fetch recent candles on the trade timeframe (H1), calculating the average volume of valid bars (those with volume > 1 to avoid data issues). If no valid bars exist, it bypasses the check to prevent stalling, logging a warning for debugging. For EURUSD, GBPUSD, and GBPJPY, high volume often accompanies significant price moves, such as post-news breakouts or liquidity sweeps, making this filter crucial for validating purges. The function logs current and average volumes for transparency. By ensuring volume confirmation, the EA avoids low-liquidity setups that lead to false signals in GBP pairs, enhancing trade quality and profitability in trend-driven markets like GBPJPY.

```
bool IsVolumeSpike(long currentVolume)
{
   double avgVolume = 0;
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, TradeTimeframe, 1, LookbackBars, rates) < LookbackBars)
   {
      Print("Failed to load rates for volume check");
      return true;
   }

   int validBars = 0;
   for(int i = 0; i < LookbackBars; i++)
   {
      if(rates[i].tick_volume > 1)
      {
         avgVolume += rates[i].tick_volume;
         validBars++;
      }
   }
   if(validBars == 0)
   {
      Print("Warning: No valid volume data. Average volume is 0. Bypassing volume check.");
      return true;
   }
   avgVolume /= validBars;

   Print("Current volume: ", currentVolume, ", Average volume: ", avgVolume, ", Valid bars: ", validBars);
   return currentVolume >= VolumeThreshold * avgVolume;
}
```

IsBullishTrend Function: Ensure Trend Alignment

The \`IsBullishTrend\` function uses a 50-period Simple Moving Average (SMA) on the trade timeframe (H1) to determine the market’s trend direction. It creates an SMA handle with \`iMA\`, retrieves the latest SMA value via \`CopyBuffer\`, and compares it to the current closing price (\`iClose\`). If the price is above the SMA, the function returns \`true\`, indicating a bullish trend; otherwise, it implies a bearish or neutral trend.

When \`UseTrendFilter\` is enabled (default true), this function ensures buy trades are only taken in bullish conditions and sell trades in bearish conditions, aligning with EURUSD/GBPUSD/GBPJPY’s tendency to follow sustained trends during London/New York sessions. Error handling logs SMA data failures, ensuring reliability. This filter reduces false breakouts in choppy markets, a common issue in GBP pairs, by confirming the broader market context before entering trades post-purge.

```
bool IsBullishTrend()
{
   double sma[];
   ArraySetAsSeries(sma, true);
   int smaHandle = iMA(_Symbol, TradeTimeframe, SMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(CopyBuffer(smaHandle, 0, 0, 1, sma) < 1)
   {
      Print("Failed to load SMA data");
      return false;
   }
   double currentPrice = iClose(_Symbol, TradeTimeframe, 0);
   Print("Current price: ", currentPrice, ", SMA: ", sma[0]);
   return currentPrice > sma[0];
}
```

IsBullishEngulfing and IsBearishEngulfing: Confirm Reversal Signals

These functions detect bullish and bearish engulfing candles, key reversal patterns used to confirm entries after a purge. \`IsBullishEngulfing\` checks if the current candle is bullish (close > open) the previous candle is bearish (close < open) the current candle engulfs the previous one (open ≤ previous close, close ≥ previous open), and the current candle’s body is at least \`EngulfingMinBodyRatio\` (0.3) times the previous candle’s body. \`IsBearishEngulfing\` applies the inverse logic for bearish setups. For EURUSD, GBPUSD and GBPJPY, engulfing candles on M15 or H1 post-purge often signal strong reversals or continuations, especially after liquidity sweeps at key levels.

The body ratio check ensures the engulfing candle is significant, filtering out weak patterns in GBP pairs’ volatile sessions. These functions are applied to the last three candles on both H1 and M15, providing flexibility to catch signals on either timeframe, enhancing the EA’s responsiveness to GBP market dynamics.

```
bool IsBullishEngulfing(MqlRates &current, MqlRates &previous)
{
   double currentBody = MathAbs(current.close - current.open);
   double prevBody = MathAbs(previous.close - previous.open);

   if(current.close > current.open &&
      previous.close < previous.open &&
      current.open <= previous.close &&
      current.close >= previous.open &&
      currentBody >= EngulfingMinBodyRatio * prevBody)
   {
      return true;
   }
   return false;
}

bool IsBearishEngulfing(MqlRates &current, MqlRates &previous)
{
   double currentBody = MathAbs(current.close - current.open);
   double prevBody = MathAbs(previous.close - previous.open);

   if(current.close < current.open &&
      previous.close > previous.open &&
      current.open >= previous.close &&
      current.close <= previous.open &&
      currentBody >= EngulfingMinBodyRatio * prevBody)
   {
      return true;
   }
   return false;
}
```

PlaceTrade Function: Execute Market Orders with SL/TP

The \`PlaceTrade\` function executes buy or sell orders when a valid signal is confirmed (purge + engulfing + volume + trend). It constructs an \`MqlTradeRequest\` with the symbol, fixed \`LotSize\` (0.1), order type (buy/sell), and entry price. Stop-loss is set at 150 pips and take-profit at 600 pips, adjusted for EURUSD GBPUSD, and GBPJPY’s point value (\`\_Point \* 100\`) to account for their pip scaling. The \`ORDER\_FILLING\_IOC\` (Immediate or Cancel) ensures fast execution, critical in GBPJPY’s fast-moving market.

On success, it logs the trade details, sets \`tradePlaced\` to true to prevent multiple trades, and resets \`purgeDetected\` to avoid re-entering on the same signal. On failure, it logs the error code for debugging. This function ensures precise trade execution with predefined risk parameters, optimizing profitability by targeting high-probability setups in GBP pairs’ trend-following behavior.

```
void PlaceTrade(ENUM_ORDER_TYPE orderType, double price)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = orderType;
   request.price = price;
   request.sl = (orderType == ORDER_TYPE_BUY) ? price - StopLossPips * _Point * 100 : price + StopLossPips * _Point * 100;
   request.tp = (orderType == ORDER_TYPE_BUY) ? price + TakeProfitPips * _Point * 100 : price - TakeProfitPips * _Point * 100;
   request.type_filling = ORDER_FILLING_IOC;

   if(OrderSend(request, result))
   {
      Print("Trade placed successfully: ", orderType == ORDER_TYPE_BUY ? "BUY" : "SELL", " at ", price, " SL: ", request.sl, " TP: ", request.tp);
      tradePlaced = true;
      purgeDetected = false; // Reset purge after trade
   }
   else
   {
      Print("Trade failed: ", result.retcode);
   }
}
```

ManageTrailingStop Function: Protect Profits with Trailing Stops

The \`ManageTrailingStop\` function dynamically adjusts the stop-loss of open positions to lock in profits as the price moves favorably. It checks if a position exists for the symbol using \`PositionSelect\`. For buy positions, it calculates a new stop-loss at the current bid price minus \`TrailingStopPips\` (100 pips), adjusted for GBPUSD EURUSD, and GBPJPY’s point value. The new stop-loss is applied only if it’s higher than the current stop-loss and above the open price, ensuring breakeven or profit. For sell positions, it sets the stop-loss above the current ask price, applying it if lower than the current stop-loss (or if none exists) and below the open price.

The function uses \`MqlTradeRequest\` with \`TRADE\_ACTION\_SLTP\` to update the stop-loss, preserving the take-profit. This trailing mechanism is crucial for GBPJPY’s volatile swings, allowing the EA to capture large moves while protecting gains. Error handling logs failed updates, ensuring transparency. This function enhances profitability by reducing the risk of giving back profits in GBP pairs’ trend-driven markets.

```
void ManageTrailingStop()
{
   if(!PositionSelect(_Symbol)) return;

   double currentPrice = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ?
                         SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                         SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
   double currentSL = PositionGetDouble(POSITION_SL);

   double newSL = 0;
   if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
   {
      newSL = currentPrice - TrailingStopPips * _Point * 100;
      if(newSL > currentSL && newSL > openPrice)
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         request.action = TRADE_ACTION_SLTP;
         request.position = PositionGetInteger(POSITION_TICKET);
         request.symbol = _Symbol;
         request.sl = newSL;
         request.tp = PositionGetDouble(POSITION_TP);
         if(OrderSend(request, result))
            Print("Trailing stop updated for BUY: ", newSL);
         else
            Print("Failed to update trailing stop: ", result.retcode);
      }
   }
   else // POSITION_TYPE_SELL
   {
      newSL = currentPrice + TrailingStopPips * _Point * 100;
      if((newSL < currentSL || currentSL == 0) && newSL < openPrice)
      {
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         request.action = TRADE_ACTION_SLTP;
         request.position = PositionGetInteger(POSITION_TICKET);
         request.symbol = _Symbol;
         request.sl = newSL;
         request.tp = PositionGetDouble(POSITION_TP);
         if(OrderSend(request, result))
            Print("Trailing stop updated for SELL: ", newSL);
         else
            Print("Failed to update trailing stop: ", result.retcode);
      }
   }
}
```

OnTick Function: Core Trading Logic for Purge and Engulfing

The \`OnTick\` function is the EA’s main decision-making hub, executed on every price tick but processing only on new H1 candles via \`IsNewCandle\`. It resets \`tradePlaced\` if no positions are open, ensuring one trade per purge cycle. Trailing stops are managed for open positions using \`ManageTrailingStop\`. The function loads candle data for H1 and M15 using \`CopyRates\`, checking the current H1 candle against previous highs/lows (updated via \`UpdateHighsLows\`) to detect purges—price breaking above H1/H4/D1 highs or below lows, signaling liquidity sweeps common in EURUSD, GBPUSD/GBPJPY during key sessions. If a purge is detected, \`purgeDetected\` is set, and \`lastPurgeTime\` tracks the window (up to 3 H1 candles).

Volume is validated with \`IsVolumeSpike\`, and the trend is checked with \`IsBullishTrend\` (if enabled). Engulfing patterns are identified on both H1 and M15, allowing flexible confirmation. A buy trade is triggered if a high purge is followed by a bullish engulfing candle (H1 or M15), high volume, and a bullish trend; a sell trade requires a low purge, bearish engulfing, high volume, and a bearish trend. Extensive logging provides transparency for debugging. This logic ensures high-probability trades in GBP pairs by combining multi-timeframe purges with robust confirmation signals, optimizing for their trend-following nature.

```
void OnTick()
{
   // Check if a new candle has formed on the trade timeframe
   if(!IsNewCandle()) return;

   // Reset tradePlaced if no open positions
   if(!PositionSelect(_Symbol)) tradePlaced = false;

   // Manage trailing stop for open positions
   ManageTrailingStop();

   // Get current and previous candle data for trade timeframe
   MqlRates ratesH1[];
   ArraySetAsSeries(ratesH1, true);
   if(CopyRates(_Symbol, TradeTimeframe, 0, 4, ratesH1) < 4)
   {
      Print("Failed to load H1 rates data");
      return;
   }

   // Get candle data for confirmation timeframe (M15)
   MqlRates ratesM15[];
   ArraySetAsSeries(ratesM15, true);
   if(CopyRates(_Symbol, ConfirmationTimeframe, 0, 4, ratesM15) < 4)
   {
      Print("Failed to load M15 rates data");
      return;
   }

   // Current H1 candle (index 0)
   double currentOpenH1 = ratesH1[0].open;
   double currentCloseH1 = ratesH1[0].close;
   double currentHighH1 = ratesH1[0].high;
   double currentLowH1 = ratesH1[0].low;
   long currentVolumeH1 = ratesH1[0].tick_volume;
   datetime currentTimeH1 = ratesH1[0].time;

   // Update highs and lows for all timeframes
   UpdateHighsLows();

   // Check for purge (liquidity sweep) on any timeframe
   bool highPurgedH1 = (currentHighH1 > lastHighH1 && lastHighH1 > 0);
   bool highPurgedH4 = (currentHighH1 > lastHighH4 && lastHighH4 > 0);
   bool highPurgedD1 = (currentHighH1 > lastHighD1 && lastHighD1 > 0);
   bool lowPurgedH1 = (currentLowH1 < lastLowH1 && lastLowH1 > 0);
   bool lowPurgedH4 = (currentLowH1 < lastLowH4 && lastLowH4 > 0);
   bool lowPurgedD1 = (currentLowH1 < lastLowD1 && lastLowD1 > 0);
   bool highPurged = highPurgedH1 || highPurgedH4 || highPurgedD1;
   bool lowPurged = lowPurgedH1 || lowPurgedH4 || lowPurgedD1;

   // Update purge status
   if(highPurged || lowPurged)
   {
      purgeDetected = true;
      lastPurgeTime = currentTimeH1;
      highPurge = highPurged;
   }

   // Check if within the post-purge window
   bool withinPurgeWindow = false;
   if(purgeDetected)
   {
      int candlesSincePurge = iBarShift(_Symbol, TradeTimeframe, lastPurgeTime, true);
      withinPurgeWindow = candlesSincePurge <= MaxCandlesPostPurge;
      if(!withinPurgeWindow)
      {
         purgeDetected = false; // Reset if window expires
         Print("Purge window expired: ", candlesSincePurge, " candles since last purge");
      }
   }

   // Check volume for liquidity confirmation
   bool volumeConfirmed = IsVolumeSpike(currentVolumeH1);
   if(currentVolumeH1 <= 1)
   {
      Print("Warning: Tick volume is ", currentVolumeH1, ". Possible data issue. Bypassing volume check.");
      volumeConfirmed = true;
   }

   // Check trend with SMA
   bool isBullishTrend = UseTrendFilter ? IsBullishTrend() : true;

   // Check for engulfing candles on H1 (current + previous 2 candles)
   bool bullishEngulfingH1 = false, bearishEngulfingH1 = false;
   for(int i = 0; i < 3; i++)
   {
      if(IsBullishEngulfing(ratesH1[i], ratesH1[i+1]))
         bullishEngulfingH1 = true;
      if(IsBearishEngulfing(ratesH1[i], ratesH1[i+1]))
         bearishEngulfingH1 = true;
   }

   // Check for engulfing candles on M15 (current + previous 2 candles)
   bool bullishEngulfingM15 = false, bearishEngulfingM15 = false;
   for(int i = 0; i < 3; i++)
   {
      if(IsBullishEngulfing(ratesM15[i], ratesM15[i+1]))
         bullishEngulfingM15 = true;
      if(IsBearishEngulfing(ratesM15[i], ratesM15[i+1]))
         bearishEngulfingM15 = true;
   }

   // Trade logic
   if(purgeDetected && withinPurgeWindow && !tradePlaced)
   {
      if(highPurge && (bullishEngulfingH1 || bullishEngulfingM15) && volumeConfirmed && isBullishTrend)
      {
         Print("Buy signal: High purged, bullish engulfing on ", bullishEngulfingH1 ? "H1" : "M15", ", volume confirmed, bullish trend");
         PlaceTrade(ORDER_TYPE_BUY, currentCloseH1);
      }
      else if(!highPurge && (bearishEngulfingH1 || bearishEngulfingM15) && volumeConfirmed && !isBullishTrend)
      {
         Print("Sell signal: Low purged, bearish engulfing on ", bearishEngulfingH1 ? "H1" : "M15", ", volume confirmed, bearish trend");
         PlaceTrade(ORDER_TYPE_SELL, currentCloseH1);
      }
   }
}
```

Installation and backtesting: Compile on MetaEditor and attach to chart. Backtesting on GBPUSD, H1 (2025-2026) with 1% risk.

### Strategy testing

**Strategy testing on the Trending King EA**

The strategy works best on most, if not all, pairs due to its core logic or quick adaptability to trend trading. It uses a liquidity purge concept for trade setups and high volatility, which are beneficial and essential for most trading strategies. We will test this strategy by trading GBPUSD from January 1, 2025, to January 8, 2026, on the 60-minute (H1) timeframe. Here are the parameters I have chosen for this strategy.

GBPUSD

![Input settings ](https://c.mql5.com/2/189/GBPUSDnew.png)

![Input settings](https://c.mql5.com/2/189/GBPUSDnew2.png)

### **Strategy tester results**

Upon testing on the strategy tester, here are the results of how it works, analyzes, and Performs.

**Strategy tester results on Trending king EA**

Balance/Equity graph:

GBPUSD

![Test results graph](https://c.mql5.com/2/189/GBPUSDnewCapture.png)

Backtest results:

GBPUSD

![Test results](https://c.mql5.com/2/189/GBPUSDnew4.png)

### Summary

I wrote this article to try to explain a MetaTrader 5 Expert Advisor that is specifically tailored for trend following, taking advantage of its speed and power, and combines trade and risk management techniques to systematically reduce risk, exposure, and human errors while identifying and executing high-probability trading setups on GBPUSD and also possible exit points using the same trade and risk management protocol.

This Expert Advisor is one of the most simple and yet powerful trading Expert Advisors and trending-based concepts used to capture possible trade price entries and trend shifts. The robust and well-adaptive risk and trade management logic helps the Expert Advisor perform at an optimum level and minimize drawdown and trading losses.

I tested the Expert Advisor on GBPUSD, and it revealed its ability to detect possible trade entries efficiently and aptly on any time frame, but the trade entry point detection is only part of the equation because it has an optimum entry validation strategy built into the core logic that allows execution only if certain criteria are met. As soon as the trades are validated and executed, then the trade and risk management logic is quickly implemented to ensure proper execution until the trade is closed.

To implement this Expert Advisor strategy, configure the input parameters on the Expert Advisor as shown below to get desirable results. The Expert Advisor is designed to scan for possible trade entries on the set timeframe a trader selects to view, from M15 to D1, ensuring the possible trade entry points align with the trend and moving averages and the average true range for trailing stop-loss. Interested traders should back-test this Expert Advisor on their demo accounts with GBPUSD; it works optimally well and is designed for GBPUSD but may also be applied to GBPJPY, EURUSD, and GOLD. The main agenda and goal for this Expert Advisor were to optimize it for strictly trend-following trading with advanced trade logic and for high-probability setups that occur in any time frame, for depending on a trader's choice, and also incorporate risk management with the implemented trailing stops.

I would also advise traders to regularly review performance logs to refine settings and input parameters depending on one's goals, asset class or risk appetite. Disclaimer: Anybody using this Expert Advisor should first test and start trading on his demo account to master this trend-following and trading idea approach for consistent profits before risking live funds.

### Conclusion

The article highlights the main challenges traders face in narrative, bias, direction, and trend identification; risk management; trade management; and avoiding drawdowns—and explains how to design an Expert Advisor that simplifies this process and increases the chances of being profitable and executing only high-probability trades. The article further elaborates how trading only in the trend's direction and ignoring countertrend trade ideas and setups greatly aids in this.

Many traders lack a clear understanding of how trend-following trading can really have a major positive impact on their performance, especially when combined with proper risk and trade management; it becomes a very major game changer that separates the average from the very best. The proposed Expert Advisor helps enforce discipline and allows traders to validate their trade ideas, position sizing, and setups even if they don’t use its entries directly.

The automated MQL5 Expert Advisor provides:

- high probability trades in the same direction as trend of the asset being traded;
- protection from news volatility by blocking trades around news releases;
- trade entries only on confirmed signals with dynamic SL/TP;
- adaptive risk management (reducing lot size during losing streaks, increasing during winning streaks);
- logging of results for ongoing strategy optimization;
- strict adherence to following and trading only in trend direction;
- removal of emotional decision-making;
- automated trade management (SL, TP, partial closes).

Together, these features deliver consistent execution of high-probability trades and optimal risk management, increasing the chance of profitable trading and improving performance.

All code referenced in the article is attached below. The following table describes all the source code files that accompany the article.

| File Name | Description: |
| --- | --- |
| Trending King EA.mq5 | File containing the full source code for the Trend King EA |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19755.zip "Download all attachments in the single ZIP archive")

[Trending\_King.mq5](https://www.mql5.com/en/articles/download/19755/Trending_King.mq5 "Download Trending_King.mq5")(16.29 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Automated Risk Management for Passing Prop Firm Challenges](https://www.mql5.com/en/articles/19655)
- [Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://www.mql5.com/en/articles/19756)
- [Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)
- [Mastering Fair Value Gaps: Formation, Logic, and Automated Trading with Breakers and Market Structure Shifts](https://www.mql5.com/en/articles/18669)

**[Go to discussion](https://www.mql5.com/en/forum/503504)**

![Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://c.mql5.com/2/127/Analyzing_Synthetic_Currencies_Movements_and_Mean_Reversion___LOGO.png)[Forex arbitrage trading: Analyzing synthetic currencies movements and their mean reversion](https://www.mql5.com/en/articles/17512)

In this article, we will examine the movements of synthetic currencies using Python and MQL5 and explore how feasible Forex arbitrage is today. We will also consider ready-made Python code for analyzing synthetic currencies and share more details on what synthetic currencies are in Forex.

![Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://c.mql5.com/2/190/20851-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 54): Filtering Trends with EMA and Smoothed Price Action](https://www.mql5.com/en/articles/20851)

This article explores a method that combines Heikin‑Ashi smoothing with EMA20 High and Low boundaries and an EMA50 trend filter to improve trade clarity and timing. It demonstrates how these tools can help traders identify genuine momentum, filter out noise, and better navigate volatile or trending markets.

![MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://c.mql5.com/2/190/20945-mql5-trading-tools-part-11-logo__1.png)[MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)

In this article, we build a correlation matrix dashboard in MQL5 to compute asset relationships using Pearson, Spearman, and Kendall methods over a set timeframe and bars. The system offers standard mode with color thresholds and p-value stars, plus heatmap mode with gradient visuals for correlation strengths. It includes an interactive UI with timeframe selectors, mode toggles, and a dynamic legend for efficient analysis of symbol interdependencies.

![Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://c.mql5.com/2/190/20745-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 5): Automating the Volatility Breakout Strategy in MQL5](https://www.mql5.com/en/articles/20745)

This article demonstrates how to automate Larry Williams’ volatility breakout strategy in MQL5 using a practical, step-by-step approach. You will learn how to calculate daily range expansions, derive buy and sell levels, manage risk with range-based stops and reward-based targets, and structure a professional Expert Advisor for MetaTrader 5. Designed for traders and developers looking to transform Larry Williams’ market concepts into a fully testable and deployable automated trading system.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jrpquwndwgqswlmljwkentpeyynntwvj&ssn=1769193125884007045&ssn_dr=0&ssn_sr=0&fv_date=1769193125&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19755&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Optimizing%20Trend%20Strength%3A%20Trading%20in%20Trend%20Direction%20and%20Strength%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919312598695344&fz_uniq=5071905813380935845&sv=2552)

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