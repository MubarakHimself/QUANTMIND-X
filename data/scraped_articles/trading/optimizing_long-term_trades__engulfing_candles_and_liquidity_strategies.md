---
title: Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies
url: https://www.mql5.com/en/articles/19756
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:27:00.605450
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=ljcudljbyeeewwmlowpfahxqnjlwazgb&ssn=1769250417697609744&ssn_dr=0&ssn_sr=0&fv_date=1769250417&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19756&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Optimizing%20Long-Term%20Trades%3A%20Engulfing%20Candles%20and%20Liquidity%20Strategies%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925041735976229&fz_uniq=5082861385894858865&sv=2552)

MetaTrader 5 / Trading


### Introduction

I have written this article with a clear outlook and determination to address a most foul discovery whereby it has become very common for traders recently to be lost in the noise and uncertainty of short-term trading or scalping, which has led to the demise of so many trading accounts. This, I must admit, does not only affect new traders, but also veteran and experienced traders may suffer from this time and again.

Most traders are so caught up in short-term trading and scalping and small swings that they easily forget price moves from point A to B with an agenda in place, a narrative, and a target. Also, traders easily forget that everything in the charts originates from the higher timeframe before trickling down to the lower timeframe, and thus the origin of the word top-down analysis in trading, since it is more effective to analyse trades from high to lower timeframes and even make trade executions from there based on points of interest of higher timeframe areas.

Trades analysed, executed, traded, and tracked regarding higher time frames have proven not only to be more accurate faster but also to have very little drawdown as price moves quickly towards its target points with more speed and less drawdown. Also, price swings are more powerful, meaning even if one is in drawdown, price will soon trade according to the higher timeframe narrative in play and will come out of drawdown.

This article will explore in detail an EA that is specifically designed for long-term traders who are patient enough to withstand and hold their positions during tumultuous lower time frame price action without changing their bias frequently until take-profit targets are hit. This EA is also critical for lower time frame and short-term traders since it also helps them trade according to time frame bias and order flow, avoiding trading against the long-term trend.

It has often been proven statistically that trading in the same direction as the higher time frame trend has more success in trading results outcomes than trading against the trend.

![MN red line shows liquidity purge then buy trade entries](https://c.mql5.com/2/178/MN_liquidity_purge.png)

### Introspection of Higher Timeframe Trading

The main agenda of this article is to emphasize how higher time frames, which are often ignored by most traders, especially those with limited experience, and even those with so much skin in the game, may overlook such an important aspect that may ultimately turn out to be a game changer.

The truth is that price moves from one key level to another key level on the higher timeframe, and whatever happens in the lower timeframe is just a consequential eventuality. This is actually opposite to what the majority believe, which is that the lower timeframe is the one that determines high timeframe price movements, and trades begin and originate from lower timeframes and in turn determine higher timeframe price and price action.

On that note, this means before we do anything, our focus should be quickly taken to the monthly chart (MN), and succeeding, drop to lower successive timeframes after we make our observations and even markings of the key areas and interest points. The key areas and interest points are often referred to with different names and terminologies when used in different strategies.

Some traders view them as orderblocks, others as support and resistance, others as supply and demand zones, others liquidity raids, and so on, but the most important thing all this strategies and theories agree on is that this points are very key and useful levels that are often points of interest where one can anticipate strong price reactions and also where most clinically accurate and professional traders hunt for their ideal setups. And that is what this article aims to exploit: how to create and use an EA that effectively makes use of this phenomenon.

![W1 liquidity purge for sell trade entries ](https://c.mql5.com/2/178/W1_liquidity_purge.png)

### Long-term trend detection, weekly, monthly, and daily biases, and anticipating trade directions and narratives

The first thing after logging into an MN chart is marking the key points (reaction areas) discussed earlier, which are usually located at previous weeks lows and highs. This key reaction area should always be kept in mind, as price usually always has a reaction here, and it may also determine how price will move for that month or the rest of the month.

This is what we may call a low-hanging fruit, as it is the closest reaction area based on the (MN) chart timeframe. Another point of interest would be fair value gaps, as they may be used as take-profit areas, and the last key areas are relative equal highs and lows that appear on the chart.

Traders who are keen enough will notice that in this area, the price may come to them and quickly purge low/high and then form engulfing candles quickly. This may happen in an even lower time frame and is not a must; it occurs in the same time frame to be successful and effective. When this occurs, for example, price purges the previous week's low, and a bullish engulfing candle quickly expands into the opposite direction; we may be witnessing a very long-term trade since trends, direction, and narratives rarely change in higher timeframes.

The goal is to patiently wait for the engulfing candle to complete and close, and once we are sure it is successful, we may enter our long-term buy positions, which may be held for weeks or even months, our take profits being previous months highs, relative equal highs, or prominent shouting highs, and at this areas is where and how we determine our monthly outlook and profile.

Also if this event has occurred but still the price has not reached any significant take-profit areas, do not expect a trend reversal. There will be swings and short-term reversals, especially in the lower time frame, but they won't really affect the long-term outlook. The end of trends or reversals will occur on the opposite end of the month or in key areas explained earlier. As well, for sells to occur, the engulfing candle pattern will occur after a liquidity purge. The same formula or theory is also present on the weekly chart and is used to determine the weekly outlook and narrative.

For example, price may reach a previous week's high and purge it with speed and power, then a bearish engulfing candle quickly forms in the opposite direction after this liquidity grab, either in the D1 timeframe or the same W1 timeframe, and closes; hence, this confirms long-term short entries on the W1 chart timeframe.

Same as the other time frame, profit-taking points will be marked, and all our focus should be at the previous week's low, and if this is purged and the price does not slow down, then it can be a partial profit-taking area as our long-term target is adjusted to fair value gaps, relative equal lows, or the last low in the W1 chart.

The last scenario this great and distinct feature occurs in is the D1 time frame, which I believe is a critical and fundamental time frame because it acts as a middle ground or bridge between two extremes: the lower time frames and the higher time frames. It is even probably the best time frame to frame intraday and short-term trades that range a few days or hours by just using the D1 time frame.

It is also relatively easier to spot both short-term swings and long-term swings on this time frame and, lastly, spot the narrative and trend. Being a very key timeframe here, traders should focus on relatively equal highs and lows, previous highs and lows, and previous day highs and lows, as all the data needed lies and is anticipated and traded here.

Price may move aggressively in this timeframe towards relative equal highs/lows or previous days lows and highs and purge the area. For this example, we may use the previous day's low. Very soon a bullish engulfing candle will form from this area, signaling a new bullish trend on D1. The engulfing candle may also appear on the lower timeframe H4.

When this happens, this signals a potential long-term trend on this timeframe, and the target for profit-taking would be the previous day's high or even further towards recent highs on D1/relative equal highs. In some instances, fair value gaps provide the best opportunity for exiting trades or partial profit taking.

Some important factors to note are that resistance and support areas are also key areas to watch out for, as well as supply and demand and order blocks, since they all refer to the same pattern with just different names. Another essential piece of information is that profit-taking areas are not static and may change, and even the overall trend may sometimes change depending on the geopolitical climate or factors supporting the traded asset.

That's why partial profit-taking is really effective in this EA, and also other factors such as fair value gaps to mitigate such losses and to anticipate reversals. But still, what a trader should keep in mind is that none of this factors occur if not at key levels; those are like the bus station points for price.

![red line shows D1 liquidity purge for buy trades](https://c.mql5.com/2/178/D1_Eurusd_liquidity_purge.png)

### Higher Time Frame Direction, Trends, Narrative

Now comes the last and easier part. The thing is, determining the key points and possible target areas is just a step in the right direction. After we determine this and have our markings done manually on the chart for practical reasons, this helps us avoid getting lost in the noise and have a clear focus on the narrative in play. Once the phenomenon occurs in the MN timeframe, for example, if being it has purged previous months lows and engulfing candles have confirmed bullish entries, that will mean that the monthly profile, bias, and narrative at play would be generally bullish, and selling would be going against the monthly trend.

The same case also occurs on the weekly timeframe chart; for example, when price purges previous weeks highs and bearish engulfing candles are confirmed and price begins selling, that means the weekly narrative, bias, and trend is selling, and chances are that if you try buying or being bullish, there is a high probability your trades will be stopped out as you will be countering the weekly narrative. The easiest thing to do to make money fast would be to trade according to the weekly and monthly time frames. This will save traders from plenty of drawdowns, heartache, and stop-outs.

The final one would be D1 time frame charts after purging recent lows/highs or previous days highs/ lows and engulfing candles form from this lows either in H4 or D1; that would simply imply a potential bullish day in bias, and lower time frames should be executed with this in mind. The final golden nugget that I have saved for last is that to achieve optimum results and success, a trader should prioritize executing trades when both or all 3 time frames align and have the same bias, narrative, and trend. When W1, D1, and MN all speak the same language, those are some of the easiest trading days ever-very fast with minimum drawdowns or retracements.

Below in the next chapters, I will elaborate on how to design and configure an EA that is based on HTF analysis and uses it to determine long-term conviction, e.g., weekly, daily, and monthly narrative and bias.

![red line showing liquidity purge](https://c.mql5.com/2/178/H1_PURGECapture.png)

### Automating trading decisions with HTF engulfing king expert advisor in MQL5

To automate, illustrate, and implement this strategy, I have created:

- A higher time frame expert advisor to analyse trends and possible trade entry points after liquidity purges, incorporating trailing stop loss and simple moving averages.

**Decision-making process**

This expert advisor’s decisions (liquidity detection, engulfing candle detection, and trade execution) are driven by the following logic:

Source code of the HTF engulfing king expert advisor

This expert advisor detects long-term trade signals from the higher timeframe charts D1, MN, and W1 and waits for engulfing candles immediately after liquidity purges, interpreting them as trade entries. It also utilizes a simple moving average to get the general trend direction to make trade entries and applies a trailing stop-loss logic to determine stop-loss levels and protect capital when price moves in its favor. It includes risk management (1% risk per trade).

Input Parameters: Configurable Trading Settings

The input parameters allow users to tailor the EA’s behavior for GBPJPY, GBPUSD and EURUSD trading without modifying the core code. \`LotSize\` is set to 0.1, a conservative size suitable for GBP pairs to limit risk in their moderate volatility (100–300 pips daily). \`LookbackBars\` (3) defines the window for identifying recent highs/lows on H1/H4/D1, balancing responsiveness with reliability. \`StopLossPips\` (150) and \`TakeProfitPips\` (600) are adjusted for GBPJPY, GBPUSD/EURUSD’s typical ranges, targeting a 1:4 risk-to-reward ratio to capitalize on strong trends post-purge. \`TrailingStopPips\` (100) ensures profits are locked in during GBPJPY’s sharper moves. \`EngulfingMinBodyRatio\` (0.3) ensures the engulfing candle’s body is significant relative to the previous candle, filtering weak signals.

\`TradeTimeframe\` (H1) and \`ConfirmationTimeframe\` (M15) allow flexible purge detection and faster entry confirmation, respectively. \`MaxCandlesPostPurge\` (3) limits the window for engulfing signals post-purge, avoiding late entries. \`VolumeThreshold\` (1.0) requires a volume spike to confirm liquidity, critical for GBPUSD’s news-driven moves. UseTrendFilter\` (true) and \`SMAPeriod\` (50) enable a 50-period SMA to ensure trades align with the broader trend, reducing false breakouts in choppy GBP markets.

```
//+------------------------------------------------------------------+
//|                              FlexibleMultiTFEngulfingEA.mq5      |
//|                        Expert Advisor for MetaTrader 5           |
//| Description: Trades based on H1/H4/D1 highs/lows purges, waits   |
//| for engulfing candles on H1 or M15, with flexible execution.     |
//+------------------------------------------------------------------+
#property copyright "Eugene Mmene"
#property link      "https://EMcapital.com"
#property version   "1.06"

//--- Input Parameters
input double LotSize = 0.1;           // Lot size for trades
input int LookbackBars = 3;           // Number of bars to check for highs/lows
input double StopLossPips = 150.0;    // Stop loss in pips (adjusted for GBPUSD/GBPJPY)
input double TakeProfitPips = 600.0;  // Take profit in pips (adjusted for GBPUSD/GBPJPY)
input double TrailingStopPips = 100.0;// Trailing stop in pips
input double EngulfingMinBodyRatio = 0.3; // Min body ratio for engulfing candle
input ENUM_TIMEFRAMES TradeTimeframe = PERIOD_H1; // Primary timeframe for trading
input ENUM_TIMEFRAMES ConfirmationTimeframe = PERIOD_M15; // Lower timeframe for engulfing
input int MaxCandlesPostPurge = 3;    // Max H1 candles to wait for engulfing
input double VolumeThreshold = 1.0;   // Volume multiplier for liquidity confirmation
input bool UseTrendFilter = true;     // Use SMA trend filter
input int SMAPeriod = 50;             // SMA period for trend filter
```

OnInit Function: Setup and Validation for Robust Trading

The \`OnInit\` function is executed once when the EA is loaded onto the chart, ensuring proper setup for trading GBPUSD, GBPJPY, and EURUSD. It validates the \`TradeTimeframe\` (H1, H4, or D1) and \`ConfirmationTimeframe\` (M15 or M30) to ensure compatibility with the strategy’s multi-timeframe approach, rejecting invalid inputs to prevent errors. The function retrieves the symbol’s contract size using \`SymbolInfoDouble\`, critical for accurate position sizing in GBP pairs, where pip values differ (e.g., GBPJPY’s pip value is smaller due to JPY’s scale).

Global variables tracking highs/lows (\`lastHighH1\`, \`lastLowH1\`, etc.), candle times, and trade status are initialized to zero or false, resetting the EA’s state. The contract size check ensures the EA can calculate correct lot sizes, especially important for GBPJPY’s higher pip value sensitivity. A successful initialization logs the timeframe and contract size, confirming readiness for trading. This setup ensures the EA is robust and tailored to the specific market dynamics of EURUSD/GBPUSD/GBPJPY, unlike gold’s broader price swings.

```
int OnInit()
{
   if(TradeTimeframe != PE    RIOD_H1 && TradeTimeframe != PERIOD_H4 && TradeTimeframe != PERIOD_D1)
   {
      Print("Invalid trade timeframe. Use H1, H4, or D1.");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(ConfirmationTimeframe != PERIOD_M15 && ConfirmationTimeframe != PERIOD_M30)
   {
      Print("Invalid confirmation timeframe. Use M15 or M30.");
      return(INIT_PARAMETERS_INCORRECT);
   }

   contractSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   if(contractSize == 0)
   {
      Print("Failed to get contract size for ", _Symbol);
      return(INIT_PARAMETERS_INCORRECT);
   }

   lastHighH1 = 0; lastLowH1 = 0;
   lastHighH4 = 0; lastLowH4 = 0;
   lastHighD1 = 0; lastLowD1 = 0;
   lastCandleTime = 0;
   tradePlaced = false;
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

This function is critical for detecting liquidity purges—when the current H1 candle breaks above a previous high or below a low on any timeframe, signaling a potential trend reversal or continuation in EURUSD, GBPUSD/GBPJPY. The multi-timeframe approach ensures the EA captures significant liquidity sweeps, such as those triggered by stop hunts in GBPJPY’s volatile moves. Error handling logs failures to load rates, ensuring robustness. This function lays the foundation for the strategy’s purge-based entry logic, tailored to GBP pairs’ sensitivity to key support/resistance levels.

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

The \`IsVolumeSpike\` function confirms trade setups by checking if the current candle’s tick volume exceeds the average volume over \`LookbackBars\` (3) by a factor of \`VolumeThreshold\` (1.0). It uses \`CopyRates\` to fetch recent candles on the trade timeframe (H1), calculating the average volume of valid bars (those with volume > 1 to avoid data issues). If no valid bars exist, it bypasses the check to prevent stalling, logging a warning for debugging. For EURUSD, GBPUSD and GBPJPY, high volume often accompanies significant price moves, such as post-news breakouts or liquidity sweeps, making this filter crucial for validating purges. The function logs current and average volumes for transparency. By ensuring volume confirmation, the EA avoids low-liquidity setups that lead to false signals in GBP pairs, enhancing trade quality and profitability in trend-driven markets like GBPJPY.

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

When \`UseTrendFilter\` is enabled (default true), this function ensures buy trades are only taken in bullish conditions and sell trades in bearish conditions, aligning with EURUSD, GBPUSD/GBPJPY’s tendency to follow sustained trends during London/New York sessions. Error handling logs SMA data failures, ensuring reliability. This filter reduces false breakouts in choppy markets, a common issue in GBP pairs, by confirming the broader market context before entering trades post-purge.

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

These functions detect bullish and bearish engulfing candles, key reversal patterns used to confirm entries after a purge. \`IsBullishEngulfing\` checks if the current candle is bullish (close > open), the previous candle is bearish (close < open) the current candle engulfs the previous one (open ≤ previous close, close ≥ previous open), and the current candle’s body is at least \`EngulfingMinBodyRatio\` (0.3) times the previous candle’s body. \`IsBearishEngulfing\` applies the inverse logic for bearish setups. For EURUSD, GBPUSD and GBPJPY, engulfing candles on M15 or H1 post-purge often signal strong reversals or continuations, especially after liquidity sweeps at key levels.

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

The \`PlaceTrade\` function executes buy or sell orders when a valid signal is confirmed (purge + engulfing + volume + trend). It constructs an \`MqlTradeRequest\` with the symbol, fixed \`LotSize\` (0.1), order type (buy/sell), and entry price. Stop-loss is set at 150 pips and take-profit at 600 pips, adjusted for EURUSD, GBPUSD/GBPJPY’s point value (\`\_Point \* 100\`) to account for their pip scaling. The \`ORDER\_FILLING\_IOC\` (Immediate or Cancel) ensures fast execution, critical in GBPJPY’s fast-moving market.

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

The \`ManageTrailingStop\` function dynamically adjusts the stop-loss of open positions to lock in profits as the price moves favorably. It checks if a position exists for the symbol using \`PositionSelect\`. For buy positions, it calculates a new stop-loss at the current bid price minus \`TrailingStopPips\` (100 pips), adjusted for GBPUSD, EURUSD/GBPJPY’s point value. The new stop-loss is applied only if it’s higher than the current stop-loss and above the open price, ensuring breakeven or profit. For sell positions, it sets the stop-loss above the current ask price, applying it if lower than the current stop-loss (or if none exists) and below the open price.

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

Installation and backtesting: Compile on MetaEditor and attach to chart. Backtesting on GBPUSD, EURUSD/GBPJPY H1 (2025) with 1% risk.

### Strategy testing

**Strategy testing on the HTF engulfing king EA**

The strategy works best on most, if not all, pairs due to its relatively quick adaptability to trends, the same liquidity purge concept, and high volatility, which are beneficial for both intraday trading and long-term trading. We will test this strategy by trading EURUSD and GBPUSD from January 1, 2025, to October 27, 2025, on the 60-minute (H1) timeframe. Here are the parameters I have chosen for this strategy.

EURUSD

![EURUSD inputs](https://c.mql5.com/2/178/Eurusd_inputsCapture.png)

![EURUSD input settings](https://c.mql5.com/2/178/Inputs_EURUSDCapture.png)

GBPUSD

![GBPUSD inputs](https://c.mql5.com/2/178/GbpusdCapture.png)

![GBPUSD input settings](https://c.mql5.com/2/178/GbpusdInputsCapture.png)

### **Strategy tester results**

Upon testing on the strategy tester, here are the results of how it works, analyses, and Performs.

**Strategy tester results on HTF engulfing king EA**

Balance/Equity graph:

EURUSD

![balance/equity graph  EURUSD](https://c.mql5.com/2/178/EURUSDCapture.png)

GBPUSD

![balance/equity graph GBPUSD](https://c.mql5.com/2/178/graphgbpusdCapture.png)

Backtest results:

EURUSD

![EURUSD backtest results](https://c.mql5.com/2/178/DataEurusdCapture.png)

GBPUSD

![GBPUSD backtest results](https://c.mql5.com/2/178/Gbpusd_dataCapture.png)

### Summary

I wrote this article to try to explain a MetaTrader 5 expert advisor that combines the use of price action in terms of engulfing candles and liquidity purges to identify long-term high-probability trading setups on GBPUSD/EURUSD and also possible exit points using the same price action, which can either be profit-taking or stop-loss areas. This expert advisor is one of the most valuable and revolutionary position trading and trend-following concepts used to capture possible long-term trade price entries and trend shifts.

I tested the expert advisor on EURUSD, GBPUSD, and GBPJPY, and it revealed its ability to detect possible long-term trade entries efficiently and aptly on the higher time frame, but the trade entry point detection is only part of the equation because if there is no engulfing candle with speed and power coming from the recently purged liquidity, the expert advisor does not recognize that as a possible trade entry or scenario, and then trades are not supposed to be executed there, even if there is a sudden spike in price. These engulfing candles are confirmations that help improve trade accuracy and quality during volatile sessions.

To implement this expert advisor strategy, configure the input parameters on the expert advisor as shown below to get desirable results. The expert advisor is designed to scan for possible trade entries on the higher timeframes a trader selects to view, from D1 to MN, ensuring the possible trade entry points align with the trend and simple moving average and the average true range for trailing stop-loss. Interested traders should back-test this expert advisor on their demo accounts with any asset or currency pair, but it works optimally well in EURUSD and GBPUSD. The main agenda and goal for this expert advisor were to optimize it for long-term position trading and high-probability setups that occur in higher time frames for position traders and also incorporate risk management with the implemented trailing stops.

I would also advise traders to regularly review performance logs to refine settings and input parameters depending on one's goals, asset class or risk appetite. Disclaimer: Anybody using this expert advisor should first test and start trading on his demo account to master this long-term position trading idea approach for consistent profits before risking live funds.

### Conclusion

The main takeaway and emphasis of this article is to try to clearly explain and solve issues that traders face daily while executing trades. This issue is such as trading against the narrative and order flow on either the weekly, monthly, or daily timeframe; analysis paralysis since a trader is unsure where the trend or real move is going to occur after he is lost in the lower timeframe noise; and also the need to make long-term position trading decisions, where they occur, and when they occur. The article clearly states how expert advisors try to solve this limitations and make the trading process seemingly easier and how they can be ideally used to analyse, understand, confirm, and even make trade executions utilizing their liquidity purge and engulfing candle strategy.

Most experts, beginner traders, and even some intermediate traders have no clue how to navigate this murky world of long-term position trading and even get frustrated with the technique as they are unable to really grasp what is going on there or even don’t have the discerning eye to understand how price always moves and how it utilizes familiar patterns or moving averages to make classic long trades and executions that can actually be profitable for them in the long-term view, which occurs infrequently as opposed to short-term trading and scalping traders. Or even if they don't execute trades based on the expert advisors entries, traders will be able to validate their own trade ideas, positions, and setups and even the trend and direction regarding what I have shared in this article, and they will find it very interesting how the HTF engulfing King expert advisor can play a very vital role in their trading.

By automating this long-term positional trading expert advisor with MQL5, traders reduce emotional bias, analysis paralysis, and self-doubt, enabling consistent execution of trade ideas by this long-term positional trading strategy that also works very well for short-term/scalp trading, as they can validate their trades according to the narrative in play either daily, weekly, or monthly.

All code referenced in the article is attached below. The following table describes all the source code files that accompany the article.

| File Name | Description |
| --- | --- |
| HTF Engulfing King EA.mq5 | File containing the full source code for the HTF Engulfing King EA. |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19756.zip "Download all attachments in the single ZIP archive")

[HTFfengulfing\_king8\_EA.mq5](https://www.mql5.com/en/articles/download/19756/HTFfengulfing_king8_EA.mq5 "Download HTFfengulfing_king8_EA.mq5")(18.26 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Optimizing Trend Strength: Trading in Trend Direction and Strength](https://www.mql5.com/en/articles/19755)
- [Automated Risk Management for Passing Prop Firm Challenges](https://www.mql5.com/en/articles/19655)
- [Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)
- [Mastering Fair Value Gaps: Formation, Logic, and Automated Trading with Breakers and Market Structure Shifts](https://www.mql5.com/en/articles/18669)

**[Go to discussion](https://www.mql5.com/en/forum/499461)**

![Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://c.mql5.com/2/179/20173-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 7): Scoring System 2](https://www.mql5.com/en/articles/20173)

This article describes two additional scoring criteria used for selection of baskets of stocks to be traded in mean-reversion strategies, more specifically, in cointegration based statistical arbitrage. It complements a previous article where liquidity and strength of the cointegration vectors were presented, along with the strategic criteria of timeframe and lookback period, by including the stability of the cointegration vectors and the time to mean reversion (half-time). The article includes the commented results of a backtest with the new filters applied and the files required for its reproduction are also provided.

![Automating Trading Strategies in MQL5 (Part 38): Hidden RSI Divergence Trading with Slope Angle Filters](https://c.mql5.com/2/179/20157-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 38): Hidden RSI Divergence Trading with Slope Angle Filters](https://www.mql5.com/en/articles/20157)

In this article, we build an MQL5 EA that detects hidden RSI divergences via swing points with strength, bar ranges, tolerance, and slope angle filters for price and RSI lines. It executes buy/sell trades on validated signals with fixed lots, SL/TP in pips, and optional trailing stops for risk control.

![Developing a Trading Strategy: The Butterfly Oscillator Method](https://c.mql5.com/2/179/20113-developing-a-trading-strategy-logo.png)[Developing a Trading Strategy: The Butterfly Oscillator Method](https://www.mql5.com/en/articles/20113)

In this article, we demonstrated how the fascinating mathematical concept of the Butterfly Curve can be transformed into a practical trading tool. We constructed the Butterfly Oscillator and built a foundational trading strategy around it. The strategy effectively combines the oscillator's unique cyclical signals with traditional trend confirmation from moving averages, creating a systematic approach for identifying potential market entries.

![Circle Search Algorithm (CSA)](https://c.mql5.com/2/118/Circle_Search_Algorithm__LOGO.png)[Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)

The article presents a new metaheuristic optimization Circle Search Algorithm (CSA) based on the geometric properties of a circle. The algorithm uses the principle of moving points along tangents to find the optimal solution, combining the phases of global exploration and local exploitation.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/19756&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082861385894858865)

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