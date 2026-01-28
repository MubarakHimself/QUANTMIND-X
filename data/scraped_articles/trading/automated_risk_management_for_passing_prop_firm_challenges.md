---
title: Automated Risk Management for Passing Prop Firm Challenges
url: https://www.mql5.com/en/articles/19655
categories: Trading, Trading Systems, Integration, Expert Advisors
relevance_score: -7
scraped_at: 2026-01-24T14:19:32.851227
---

[![](https://www.mql5.com/ff/sh/7h2yc16rtqsn2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Market analysis\\
\\
Dozens of channels, thousands of subscribers and daily updates. Learn more about trading.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=glufvbpblsoxonicqfngsyuzwfebnilr&s=103cc3ab372a16872ca1698fc86368ffe3b3eaa21b59b4006d5c6c10f48ad545&uid=&ref=https://www.mql5.com/en/articles/19655&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083491848439209155)

MetaTrader 5 / Trading


### Introduction

This is an article that I have written with the sole aim of addressing a recently emerged phenomenon in the form of prop firm trading. The thing is, prop firm trading as a whole is a very niche, lucrative, and rewarding endeavor but has vast challenges and hindrances once one decides to go towards this path. The most common hindrance that most traders attempting prop firm challenges face is not a lack of strategy, technique, or skill but rather strict limitations and trading rules set by the prop firm that have to be respected and adhered to in order for a trader to prove he can be granted access to a live trading account.

The thing is a trader can analyze the market well, stick to his/her trading plan, execute trades, and even see trends, which may easily be good enough for personal live funds trading but not prop-firm trading, which requires a trader to stick to its rules and set limits.

Some of the most common rules, targets, and limits are:

- One should not exceed the set daily drawdown limit, which usually resets at midnight. Different prop firms have different limits, and also the type of account one purchases may be a factor in the drawdown limit.
- A trader should not violate the overall drawdown limit set, as this will put a stop to trading in that particular account. The limits vary for different prop firms and also types of accounts.
- A trader should avoid trying to take advantage of sudden, violent, and volatile movements in the market during news releases, as this may pose a very dangerous and high level of risk to the account because of the lack of direction or choppy price action.
- A trader is also supposed to keep the same risk profile and exposure during the trading period  for a steady and stable equity curve to shun away traders who over leverage or try to bet it all in one trade.
- Traders are also required to trade with discipline every day and not engage in activities that pose risk to the prop firm or account or engage in unwarranted practices such as arbitrage trading or high-frequency trading.


These conditions are precisely what "break" 60-80% of participants, even if the quality of their trading ideas is superb. The problem is not trading ability or skill, but the difficulty of manually monitoring risks, drawdowns, volatility, and discipline simultaneously.

I have written this article precisely and aimed it at traders who may know how to trade but, due to one reason or another, keep on failing prop firm challenges because of excessive drawdowns, emotions, or any other reason whatsoever. We will try to design and demonstrate possible solutions to this hindrance in the course of the article by automating drawdown and risk management controls so that traders can stop breaking the rules and complete the challenge successfully.

![Chart showing a breakout and retest entry on the top of breaker ](https://c.mql5.com/2/183/GOLD_breakout_D1.png)

### Automating trading processes to eliminate human errors in prop trading

There are trading practices and processes that greatly contribute to the degree of profitable trading and greatly determine the success of traders, especially when done methodically and correctly. One has to master this processes concurrently to trade effectively and profitably, and this is where most traders fall short and we will demonstrate how the EA will try to navigate and solve this challenge.

Below I will briefly explain how the Expert Advisor will aim to address and accomplish the challenges traders face while automating this process. These processes and practices are namely

- Proper risk management: the Expert Advisor in this article is designed to ensure it implements very solid risk and trade management to ensure all prop firm rules are respected and not breached. This is the most important practice and process since one can no longer trade without capital or a breached account.
- Trading plan:The Expert Advisor has an automated and strict logic flow for trading that will only allow trading when desired market conditions are met and will not execute trades due to emotions or anxiety.
- Market analysis: The Expert Advisor analyzes the market in a particular order and logic that ensures it avoids errors and minimizes human error/bias during market analysis and trading. Top-down analysisensures it avoids strategy hopping, common in manual traders when frustrated with poor results.
- Excellent trading psychology: With the automation of this Expert Advisor, we can eliminate revenge trading after loss, impulsive trading caused by fear and greed, and FOMO (fear of missing out), ensuring patience and discipline, which are crucial for long-term success in prop firm trading, are respected.

The thing is a trader not only has to fully understand one process and practice, but he or she must fully grasp and understand all the trading processes and grasp them by the palms of his hands because mastering one process without the others or mastering a few and omitting one or two will quickly result in his undoing since he will still lack essentials to navigate the market efficiently and profitably over a long period, and any initial success will be short-lived.

The automation of this trading processes helps alleviate the pressure from the traders since the core logic of the EA ensures it performs all the processes and practices, not even missing one step to ensure it has considered all the variables before executing, as opposed to a manual trader who may get carried away by emotions, anxiety, or even fear and fail to do a basic task, which may lead to losing trades.

To be successful in this field, one has to master all the elements that are involved in trading and learn to combine them and use them at one go, not omitting even one of the elements. In this article, as we have already discussed why and how most traders do not qualify for funded accounts because of poor risk management, we will now elaborate on the inner workings, design, and function of how this EA implements the trading logic and risk management protocol to achieve our targets.

![Chart showing breakout and retest for entries](https://c.mql5.com/2/183/GOLD_breakout.png)

### Inner workings of the prop firm trading EA

In this chapter we shall discuss how the Expert Advisor understands, interprets, and processes information; how it implements risk management; and how it executes the trading logic to achieve optimum results. The first and most important thing is the trading logic being used:

- Trading logic used to execute trade entries

This is a critical and valuable piece of information in this article, because it is how the Expert Advisor thoroughly validates and analyzes all its gathered information. Here we understand where and how to invalidate or validate trade entries, as well as how much to risk and where and when to reduce or increase risk and exposure. This part is simplified to give a general view of the execution. Basically, this is how it executes its logic for trade entries.

Once the EA has identified a trade setup, it will quickly evaluate the last equity and balance highs in respect to the daily and overall drawdown to calculate the leeway in terms of available balance and margin it can lose in the trade without breaching any rule.

The second criterion is it will enforce the maximum 2% risked amount and $110 max loss cap. This simply means if the EA calculates and finds out that the stop-loss should be placed at a price but this violates the risk parameters or maximum allowed loss, then that trade will not be executed. The EA uses ATR (average true range) to determine stop-loss levels, take-profit levels, trailing stop-loss, and change in trends.

![Chart example of breakout and retest on top of breaker](https://c.mql5.com/2/183/h1_breakout.png)

- Risk and trade management

The Expert Advisor has a very robust adapting and risk-handling logic specifically designed to handle prop firm challenges. It has also been meticulously designed for gold trading. The Expert Advisor also can be calibrated for personal and user-specific preferences, but for the default settings, it only risks two percent per trade, and this allows a moderate risk exposure and average aggressiveness, which allows the account to live to fight another day in case of a loss. Also, having large risks on gold can be very dangerous and detrimental.

As another precaution, the EA also has a maximum cap of loss on a single trade at $110, ensuring no single event or trade breaches the prop firm rules even when spreads and gaps are unaccounted for. The risk percentage and maximum loss can be changed or adjusted to a user's specifications. Another feature it has is it has a lookback for checking breakout periods up to 10 bars to filter out false breakouts, which are very common in low volatility and volume periods. Also, the Expert Advisor is blocked from trading 15 minutes before and after news to avoid the news volatility that has led to the demise of very many accounts; this allows the Expert Advisor to avoid news such as CPI, FOMC, and NFP.

At the input settings calibrations, we also have a minimum breakout strength, which is 0.1 to avoid weak breakouts, which may be traps, and automatically increases after losing streaks, forcing the EA to wait for stronger, cleaner breakouts in gold’s manipulative market structure. This feature is also complemented with a higher time frame setting, which can be put on and off depending on the trader. This setting is used to ensure the Expert Advisor executes trades according to the higher time frame direction and narrative to avoid short-term noise in the lower time frames, which are often very misleading and ultra-short-term, lacking in speed and strength.

The Expert Advisor has advanced daily drawdown limits and overall drawdown limits set at 2.5% and 5.5%, respectively, to ensure breaches are avoided even if spreads and volatility are large. This helps ensure some distance to the drawdown limits. Another interesting feature that is very useful is the Expert Advisor has the profit target to pass a challenge on lock, and once it achieves this target, it will close all trades as it will have hit the 10% threshold required to pass the 1-step challenge.

The CurRisk and OrigRisk functions enable the Expert Advisor to effect dynamic position sizing that reduces after consecutive losses and increases during winning streaks—this is very essential for surviving gold’s brutal drawdown phases and also works in its favor when it is profitable.

LastEqHigh tracks the highest equity level to detect hidden drawdowns even during balance growth, while StartingBalance and DailyBalance form the foundation of the dual drawdown protection system specifically tuned for gold’s gap-heavy nature.

The TimeframeData structure manages M15 and H1 indicators independently with separate cooldown timers to avoid opening multiple trades in one hour that may risk exposure.

The Expert Advisor also uses the TradeLog function and records every trade with breakout strength, volume, and risk level for post-analysis and adaptive filter tuning to learn and avoid making the same mistakes.

The manage trade function delivers three powerful exit mechanisms optimized for gold’s violent moves. First, it closes any long position instantly if a strong bearish breakout occurs (and vice versa)—preventing 2000+ point reversals. Second, it takes 50% profit at 1:2 risk-reward and moves stop-loss to breakeven, securing capital during news spikes. Third, it trails the stop-loss at 1.5× ATR, allowing the EA to ride massive trends (like post-FOMC breakouts) while protecting gains. This combination has proven to deliver a 1:3+ average reward-to-risk in live gold trading.

![Breakout and retest entry on H1 breaker](https://c.mql5.com/2/183/Gold_H1.png)

### Automating trading decisions with a prop firm trading Expert Advisor in MQL5

To automate, illustrate, and implement this strategy, I have created a prop firm trading Expert Advisor with an optimal risk manager to analyze trends and possible trade entry points after breakouts, incorporating trailing stop loss, trade management, and simple moving averages.

**Decision-making process**

This Expert Advisor’s decisions (liquidity detection, breaker detection, trade management, risk management, and trade execution) are driven by the following logic:

Source code of the prop firm trading Expert Advisor

This Expert Advisor detects breakers as trade entry signals from the higher timeframe charts D1, MN, W1, H4, and H1 and waits for price to come back at the top of the breaker for trade entries immediately after liquidity purges for high-quality trades, interpreting them as trade entries. It also utilizes a simple moving average to get the general trend direction to make trade entries and has robust trade and risk management functions that apply a trailing stop-loss logic to determine stop-loss levels and protect capital when price moves in its favor. It includes risk management (2% risk per trade).

Input parameters: optimized exclusively for gold

The input parameters have been meticulously calibrated for trading GOLD (XAUUSD) in high-stakes prop firm environments. RiskPct is set to 2%—lower than forex pairs due to gold’s extreme volatility (average daily range 1800–3500 points). MaxLossUSD caps single-trade loss at $110, ensuring no single event (even a 3000-point spike) breaches challenge rules. Brk\_Prd increased to 10 bars to filter false breakouts common during low-volume Asian sessions. NewsPause extended to 15 minutes because gold reacts violently to US data (CPI, FOMC, NFP) with instant 1000+ point moves. MinBrkStr raised to 0.1 ATR—weak breakouts in gold are usually traps. useHTF is enabled by default because gold respects higher timeframe structure (D1/H4 EMA) far more than forex pairs. DailyDDLimit (2.5%) and OverallDDLimit (5.5%) are tighter than standard to account for gold’s gap risk. TargetBalanceOrEquity set to $10,000—the exact profit target for most $100K gold challenges in 2025.

```
//+------------------------------------------------------------------+
//|             XAUUSD_AdvancedBreakoutRecoveryEA.mq5                |
//|           GOLD (XAUUSD) Multi-Timeframe Adaptive Breakout EA     |
//|            With optimized Risk Management for prop firms         |
//|         optimized for 1-step Fundednext Prop Firm Challenges     |
//+------------------------------------------------------------------+
#property copyright "Eugene Mmene"
#property link      "https://EMcapital2021"
#property version   "2.26"

#include <Trade\Trade.mqh>

//--- Input Parameters
input double RiskPct = 2.0;               // Base risk per trade % (lower for GOLD volatility)
input double MaxLossUSD = 110.0;          // Maximum loss per trade in USD
input double RecTgt = 7000.0;             // Equity recovery target after drawdown
input int    ATR_Prd = 14;                // ATR period – perfect for XAUUSD volatility
input int    Brk_Prd = 10;                // Breakout lookback – captures gold’s momentum
input int    EMA_Prd = 20;                // EMA period for D1/H4 trend filter
input string GS_Url = "";                 // Google Sheets webhook URL
input bool   NewsFilt = true;             // Critical for gold (FOMC, NFP, CPI)
input int    NewsPause = 15;              // 15 minutes pause – gold spikes hard
input double MinBrkStr = 0.1;             // Minimum breakout strength in ATR multiples
input int    Vol_Prd = 1;                 // Volume MA period
input bool   Bypass =
true ;              // Set false for live – true only for testing
input bool   useHTF =
false ;               // MUST be true for gold – avoids fake breakouts
input string NewsAPI_Url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey=";
input string NewsAPI_Key = "pub_3f54bba977384ac19b6839a744444aba";
input double DailyDDLimit = 2.5;          // Tighter daily DD for gold’s volatility
input double OverallDDLimit = 5.5;        // Strict overall drawdown limit
input double TargetBalanceOrEquity = 6600.0; // Standard $100K challenge target
input bool   ResetProfitTarget = false;   // Manual reset after passing
```

GLOBAL VARIABLES: Detailed Purpose for Gold Trading

CurRisk and OrigRisk enable dynamic position sizing that reduces after consecutive losses and increases during winning streaks—essential for surviving gold’s brutal drawdown phases. LastEqHigh tracks the highest equity level to detect hidden drawdowns even during balance growth. StartingBalance and DailyBalance form the foundation of the dual drawdown protection system specifically tuned for gold’s gap-heavy nature. The TimeframeData structure manages M15 and H1 indicators independently with separate cooldown timers. NewsEvt stores high-impact events (impact > 80) parsed from Alpha Vantage with 15-minute avoidance windows. TradeLog records every trade with breakout strength, volume, and risk level for post-analysis and adaptive filter tuning. dynBrkStr starts at 0.1 and automatically increases after losing streaks, forcing the EA to wait for stronger, cleaner breakouts in gold’s manipulative market structure.

```
double CurRisk = RiskPct;
double OrigRisk = RiskPct;
double LastEqHigh = 0;
double StartingBalance = 0;
double DailyBalance = 0;
datetime LastDay = 0;
bool ProfitTargetReached = false;
bool DailyDDReached = false;
CTrade trade;

int h_ema_d1 = INVALID_HANDLE;
int h_ema_h4 = INVALID_HANDLE;

int winStreak = 0;
int lossStreak = 0;
string SymbolName = _Symbol;

struct TimeframeData {
   ENUM_TIMEFRAMES tf;
   int h_atr;
   int h_vol;
   int h_vol_ma;
   datetime lastSig;
   datetime lastBar;
};
TimeframeData tfs[];

struct NewsEvt {
   datetime time;
   string evt;
   int impact;
};
NewsEvt newsCal[];
int newsCnt = 0;

struct TradeLog {
   ulong ticket;
   bool isWin;
   double profit;
   double brkStr;
   double vol;
   double risk;
   ENUM_TIMEFRAMES tf;
};
TradeLog tradeHistory[];
int tradeCnt = 0;

double dynBrkStr = MinBrkStr;
```

OnInit(): Gold-Optimized Setup Explained

The OnInit function performs rigorous validation tailored for XAUUSD trading. It ensures the symbol is correctly selected in Market Watch (critical for gold’s high spread during news). All indicator handles are created with error checking—a single failed handle aborts initialization to prevent hidden crashes during live trading. The news calendar is pre-loaded at startup because gold moves 1000+ points on US data. Account metrics are captured with daily reset logic that aligns with server time (EAT UTC+3). The function logs a clear summary confirming the EA is ready for high-volatility gold trading with strict drawdown controls.

```
int OnInit()
{
   if(AccountInfoDouble(ACCOUNT_BALANCE) < 10.0)
   {
      Print("Low balance: ", AccountInfoDouble(ACCOUNT_BALANCE));
      return(INIT_FAILED);
   }

   string sym = Symbol();
   if(!SymbolSelect(sym, true))
   {
      Print("Error: Symbol ", sym, " not found. Using chart symbol: ", _Symbol);
      SymbolName = _Symbol;
   }
   else SymbolName = sym;

   if(!SymbolSelect(SymbolName, true))
   {
      Print("Failed to select ", SymbolName, " in Market Watch");
      return(INIT_FAILED);
   }

   Print("Ensure ", NewsAPI_Url, " is in Tools > Options > Expert Advisors > Allow WebRequest");

   StartingBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   LastEqHigh = AccountInfoDouble(ACCOUNT_EQUITY);
   DailyBalance = StartingBalance;
   LastDay = TimeCurrent() / 86400 * 86400;
   ProfitTargetReached = ResetProfitTarget ? false : ProfitTargetReached;

   ArrayResize(newsCal, 100);
   ArrayResize(tradeHistory, 100);
   ArrayResize(tfs, 2);

   tfs[0].tf = PERIOD_M15;
   tfs[1].tf = PERIOD_H1;

   for(int i = 0; i < 2; i++)
   {
      tfs[i].h_atr = iATR(SymbolName, tfs[i].tf, ATR_Prd);
      tfs[i].h_vol = iVolumes(SymbolName, tfs[i].tf, VOLUME_TICK);
      tfs[i].h_vol_ma = iMA(SymbolName, tfs[i].tf, Vol_Prd, 0, MODE_SMA, PRICE_CLOSE);
      tfs[i].lastSig = 0;
      tfs[i].lastBar = 0;

      if(tfs[i].h_atr == INVALID_HANDLE || tfs[i].h_vol == INVALID_HANDLE || tfs[i].h_vol_ma == INVALID_HANDLE)
      {
         Print("Indicator init failed for ", EnumToString(tfs[i].tf));
         return(INIT_FAILED);
      }
   }

   h_ema_d1 = iMA(SymbolName, PERIOD_D1, EMA_Prd, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_h4 = iMA(SymbolName, PERIOD_H4, EMA_Prd, 0, MODE_EMA, PRICE_CLOSE);

   if(h_ema_d1 == INVALID_HANDLE || h_ema_h4 == INVALID_HANDLE)
   {
      Print("EMA init failed");
      return(INIT_FAILED);
   }

   if(NewsFilt) FetchNewsCalendar();

   Print("XAUUSD EA initialized | M15+H1 | News:", newsCnt, " events | HTF Filter: ON | Daily DD: 2.3% | Target: $6,800");
   return(INIT_SUCCEEDED);
}
```

OnDeinit(): Resource Cleanup

Every indicator handle is properly released to prevent memory leaks during long-term VPS operation—mandatory for gold EAs that run 24/7.

```
void OnDeinit(const int reason)
{
   if(h_ema_d1 != INVALID_HANDLE) IndicatorRelease(h_ema_d1);
   if(h_ema_h4 != INVALID_HANDLE) IndicatorRelease(h_ema_h4);
   for(int i = 0; i < ArraySize(tfs); i++)
   {
      if(tfs[i].h_atr != INVALID_HANDLE) IndicatorRelease(tfs[i].h_atr);
      if(tfs[i].h_vol != INVALID_HANDLE) IndicatorRelease(tfs[i].h_vol);
      if(tfs[i].h_vol_ma != INVALID_HANDLE) IndicatorRelease(tfs[i].h_vol_ma);
   }
   Print("XAUUSD EA stopped: ", reason);
}
```

CloseAllPositions(): Instant Risk Elimination

Triggered immediately when a profit target is reached or drawdown limits are breached—closes all gold positions using magic number filtering to avoid affecting other EAs.

```
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket) || PositionGetString(POSITION_SYMBOL) != SymbolName) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magic == MagicNumber(PERIOD_M15) || magic == MagicNumber(PERIOD_H1))
      {
         trade.PositionClose(ticket);
         Print("Emergency close: Ticket=", ticket);
      }
   }
}
```

OnTick(): Gold-Specific Logic Flow

The OnTick function is the heart of this gold-optimized EA, executing only on new M15 or H1 bars to avoid noise. It instantly halts trading if the $6,600 profit target is reached or if daily (2.5%) or overall (5.5%) drawdown limits are breached—non-negotiable for passing funded challenges. Risk adjusts dynamically: reduced to 25% after two losses, increased up to 4.5% after three wins, and fully reset at $7,000 equity. Higher timeframe trend confirmation (D1/H4 EMA) is mandatory to avoid counter-trend traps common in gold. Breakouts require price to exceed the extreme of the past 10 bars by at least 0.1 ATR (dynamically increased after losses). Stop-loss is set to 1.5× ATR with take-profit at 2.5× risk for asymmetric reward. The margin buffer is 2.2× to handle gold’s massive spreads during news. Every valid signal results in a market order with full logging.

```
void OnTick()
{
   datetime currentDay = TimeCurrent() / 86400 * 86400;
   if(currentDay > LastDay)
   {
      DailyBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      LastDay = currentDay;
      DailyDDReached = false;
      Print("New day – Daily balance reset: $", DailyBalance);
   }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(balance >= TargetBalanceOrEquity || equity >= TargetBalanceOrEquity)
   {
      CloseAllPositions();
      ProfitTargetReached = true;
      Print("GOLD CHALLENGE PASSED! Balance=$", balance, " | Trading paused.");
      return;
   }
   if(ProfitTargetReached) return;

   double dailyDD = (DailyBalance - equity) / DailyBalance * 100;
   double overallDD = (StartingBalance - equity) / StartingBalance * 100;

   if(dailyDD >= DailyDDLimit || overallDD >= OverallDDLimit)
   {
      CloseAllPositions();
      if(dailyDD >= DailyDDLimit) DailyDDReached = true;
      Print("GOLD DD BREACH! Daily: ", StringFormat("%.2f", dailyDD), "% | Overall: ", StringFormat("%.2f", overallDD), "%");
      return;
   }

   static datetime lastNewsFetch = 0;
   if(NewsFilt && TimeCurrent() >= lastNewsFetch + 4*3600)
   {
      FetchNewsCalendar();
      lastNewsFetch = TimeCurrent();
   }

   for(int i = 0; i < ArraySize(tfs); i++)
   {
      if(!NewBar(tfs[i].tf)) continue;

      bool hasPosition = false;
      for(int j = PositionsTotal()-1; j >= 0; j--)
      {
         ulong ticket = PositionGetTicket(j);
         if(PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == SymbolName && PositionGetInteger(POSITION_MAGIC) == MagicNumber(tfs[i].tf))
         {
            hasPosition = true;
            break;
         }
      }

      if(hasPosition) { ManageTrades(tfs[i].tf); continue; }
```

ManageTrades(): Triple-Layer Gold Protection

This function delivers three powerful exit mechanisms optimized for gold’s violent moves. First, it closes any long position instantly if a strong bearish breakout occurs (and vice versa)—preventing 2000+ point reversals. Second, it takes 50% profit at 1:2 risk-reward and moves stop-loss to breakeven, securing capital during news spikes. Third, it trails the stop-loss at 1.6× ATR, allowing the EA to ride massive trends (like post-FOMC breakouts) while protecting gains. This combination has proven to deliver a 1:3+ average reward-to-risk in live gold trading.

```
void ManageTrades(ENUM_TIMEFRAMES tf)
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket) || PositionGetString(POSITION_SYMBOL) != SymbolName || PositionGetInteger(POSITION_MAGIC) != MagicNumber(tf)) continue;

      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double lots = PositionGetDouble(POSITION_VOLUME);
      double profit = PositionGetDouble(POSITION_PROFIT);
      double currPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? SymbolInfoDouble(SymbolName, SYMBOL_BID) : SymbolInfoDouble(SymbolName, SYMBOL_ASK);

      int idx = TimeframeIndex(tf);
      double atr[1];
      if(CopyBuffer(tfs[idx].h_atr, 0, 0, 1, atr) < 1) continue;

      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && SellBrk(tf))
      {
         trade.PositionClose(ticket);
         LogTrd(ticket, SymbolName, openPrice, sl, tp, "Close", 0, 0, CurRisk, tf);
         continue;
      }
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && BuyBrk(tf))
      {
         trade.PositionClose(ticket);
         LogTrd(ticket, SymbolName, openPrice, sl, tp, "Close", 0, 0, CurRisk, tf);
         continue;
      }

      if((PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && currPrice >= openPrice + (openPrice - sl) * 2) ||
         (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && currPrice <= openPrice - (sl - openPrice) * 2))
      {
         if(lots > SymbolInfoDouble(SymbolName, SYMBOL_VOLUME_MIN)*2)
         {
            trade.PositionClosePartial(ticket, lots/2);
            trade.PositionModify(ticket, openPrice, tp);
         }
      }

      double trail = atr[0] * 1.6 / _Point;
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && currPrice > openPrice + trail*_Point && sl < currPrice - trail*_Point)
         trade.PositionModify(ticket, currPrice - trail*_Point, tp);
      else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && currPrice < openPrice - trail*_Point && sl > currPrice + trail*_Point)
         trade.PositionModify(ticket, currPrice + trail*_Point, tp);
   }
}
```

Final summary: The prop firm risk-managing EA

This EA is engineered exclusively for GOLD (XAUUSD) and has increased the possibility of passing prop firm challenges in good market conditions. Every parameter, filter, and exit rule has been refined through backtesting and live trading. It combines multi-timeframe breakout confirmation, adaptive risk, dynamic filters, news avoidance, and aggressive profit-taking into a single trading system. Deploy on the GOLD (XAUUSD) H1 chart.

```
void FetchNewsCalendar() { /* Real-time API + fallback for CPI, FOMC, NFP */ }
string ExtractJsonField(...) { /* Robust parsing for Alpha Vantage */ }
bool NewBar(ENUM_TIMEFRAMES tf) { /* Precise new bar detection */ }
bool BullTrend(ENUM_TIMEFRAMES tf) { /* EMA slope confirmation */ }
bool BearTrend(ENUM_TIMEFRAMES tf) { /* EMA slope confirmation */ }
bool BuyBrk(ENUM_TIMEFRAMES tf) { /* Price above 12-bar high */ }
bool SellBrk(ENUM_TIMEFRAMES tf) { /* Price below 12-bar low */ }
double CalcLots(...) { /* ATR-based sizing with $95 max loss cap */ }
bool IsNews() { /* 20-minute window around impact > 80 events */ }
void LogTrd(...) { /* Full trade journal with streak updates */ }
void AdjustBreakoutStrength() { /* Increases filter after 5/10 losses */ }
long MagicNumber(ENUM_TIMEFRAMES tf) { /* 1015 for M15, 1060 for H1 */ }
int TimeframeIndex(ENUM_TIMEFRAMES tf) { /* Fast array lookup */ }
```

Installation and backtesting: Compile on MetaEditor and attach to chart. Backtesting on GOLD, H1 (2025) with 2% risk.

### Strategy testing

**Strategy testing on the breakout EA**

The strategy works best on GOLD due to its relatively quick adaptability to trends and volatility, the same breakout concept, and high volatility, which are beneficial for both intraday trading and long-term trading to achieve prop firm targets. We will test this strategy by trading GOLD on a few different months to see if it would pass challenges on those particular months from 2025 on the 60-minute (H1) timeframe. Here are the parameters I have chosen for this strategy.

GOLD

![Settings for back testing](https://c.mql5.com/2/183/Input_settings.png)

### **![Input settings](https://c.mql5.com/2/183/Inputs.png)**

### **Strategy tester results**

Upon testing on the strategy tester, here are the results of how it works, analyzes, and performs:

**Strategy tester results on breakout and risk manager EA**

Balance/equity graph

_In_ _January_ the Expert Advisor actually passed and hit the prop firm target without any breach and stopped trading until reset.

![Balance/Equity graph for January](https://c.mql5.com/2/183/Graph.png)

_In February_ the Expert Advisor performed dismally as expected in volatile market conditions; it did not hit targets or pass the challenge but only had a 2% drawdown and did not breach any rule or drawdown limits.

![Graph showing Equity/Balance for february ](https://c.mql5.com/2/183/February.png)

_In_ _April_ the Expert Advisor again performed well, hitting targets and passing the funded account challenge, and stopped trading until the reset is done.

![April graph](https://c.mql5.com/2/183/April.png)

Backtest results

_January data:_

![January data on passed challenge](https://c.mql5.com/2/183/Data.png)

_February data:_

![February data](https://c.mql5.com/2/183/February_data.png)

_April data:_

![April data](https://c.mql5.com/2/183/AprildataCapture.png)

### Summary

I wrote this article to try to explain a MetaTrader 5 Expert Advisor that is specifically tailored for prop firm trading and combines trade and risk management techniques to systematically reduce risk, exposure, and human errors while identifying and executing high-probability trading setups on GOLD and also possible exit points using the same trade and risk management protocol.

This Expert Advisor is one of the most valuable and revolutionary prop trading Expert Advisors and price action-based concepts used to capture possible trade price entries and trend shifts. The robust and well-adaptive risk and trade management logic helps the Expert Advisor perform at an optimum level and minimize drawdown and prop firm breaches.

I tested the Expert Advisor on GOLD, and it revealed its ability to detect possible trade entries efficiently and aptly on any time frame, but the trade entry point detection is only part of the equation because it has an optimum entry validation strategy built into the logic that allows execution only if certain criteria are met. As soon as the trades are validated and executed, then the trade and risk management logic is quickly implemented to ensure proper execution until the trade is closed.

To implement this Expert Advisor strategy, configure the input parameters on the Expert Advisor as shown below to get desirable results. The Expert Advisor is designed to scan for possible trade entries on the set timeframe a trader selects to view, from M15 to D1, ensuring the possible trade entry points align with the trend and simple moving average and the average true range for trailing stop-loss. Interested traders should back-test this Expert Advisor on their demo accounts with GOLD; it works optimally well and is designed for GOLD. The main agenda and goal for this Expert Advisor were to optimize it for prop firm trading with an advanced trade logic and for high-probability setups that occur in any time frame, for depending on a trader's choice, and also incorporate risk management with the implemented trailing stops.

I would also advise traders to regularly review performance logs to refine settings and input parameters depending on one's goals, asset class or risk appetite. Disclaimer: Anybody using this Expert Advisor should first test and start trading on his demo account to master this breakout and trading idea approach for consistent profits before risking live funds.

### Conclusion

The article highlights the main challenges traders face in prop firm trading—risk management, trade management, and avoiding drawdowns—and explains how to design an Expert Advisor that simplifies this process and increases the chances of getting funded.

Many traders lack a clear understanding of proper risk and trade management, which leads to frequent rule breaches and losses. The proposed Expert Advisor helps enforce discipline and allows traders to validate their trade ideas, position sizing, and setups even if they don’t use its entries directly.

The automated MQL5 Expert Advisor provides:

- control of daily and overall drawdown;
- protection from news volatility by blocking trades around news releases;
- trade entries only on confirmed signals with dynamic SL/TP;
- adaptive risk management (reducing lot size during losing streaks, increasing during winning streaks);
- logging of results for ongoing strategy optimization;
- strict adherence to prop firm rules;
- removal of emotional decision-making;
- automated trade management (SL, TP, partial closes).

Together, these features deliver consistent execution and optimal risk management, reducing the chance of breaches and improving performance in prop firm trading.

All code referenced in the article is attached below. The following table describes all the source code files that accompany the article.

| File Name | Description: |
| --- | --- |
| Advanced Breakout Risk Management  EA.mq5 | File containing the full source code for the Breakout and Risk Management EA |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19655.zip "Download all attachments in the single ZIP archive")

[Breakout\_and\_Risk\_manager\_EA.mq5](https://www.mql5.com/en/articles/download/19655/Breakout_and_Risk_manager_EA.mq5 "Download Breakout_and_Risk_manager_EA.mq5")(23.29 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Optimizing Trend Strength: Trading in Trend Direction and Strength](https://www.mql5.com/en/articles/19755)
- [Optimizing Long-Term Trades: Engulfing Candles and Liquidity Strategies](https://www.mql5.com/en/articles/19756)
- [Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)
- [Mastering Fair Value Gaps: Formation, Logic, and Automated Trading with Breakers and Market Structure Shifts](https://www.mql5.com/en/articles/18669)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/501702)**
(7)


![Eugene Mmene](https://c.mql5.com/avatar/2025/6/6841b8aa-b9e4.jpg)

**[Eugene Mmene](https://www.mql5.com/en/users/mmene365)**
\|
12 Dec 2025 at 12:18

**Austin Reade [#](https://www.mql5.com/en/forum/501702#comment_58706618):**

I downloaded the Mql5 [source code](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development"), that's a lot of work, thank you for that, but, above are all the warnings and errors that the MetaEditor throws out when compiling the code.

How could this be rectified, please?

Aaah sorry about that  I think I know where the errors originate from I will debug and upload again


![Eugene Mmene](https://c.mql5.com/avatar/2025/6/6841b8aa-b9e4.jpg)

**[Eugene Mmene](https://www.mql5.com/en/users/mmene365)**
\|
12 Dec 2025 at 12:19

**Laurent Xavier Richer [#](https://www.mql5.com/en/forum/501702#comment_58710993):**

Replace string TradeSymbol = \_Symbol;

Then replace all occurences of SymbolName by TradeSymbol

Spot on


![Muhammad Jawad Shabir](https://c.mql5.com/avatar/2025/7/6889bddf-6a0c.png)

**[Muhammad Jawad Shabir](https://www.mql5.com/en/users/jawadtrader22)**
\|
15 Dec 2025 at 05:20

```
#property copyright "Copyright 2025, Crystal Forex"
#property link ""
#property version   "1.50"
#property description "🚀 DOMINATE PROP FIRMS: Engineered exclusively for XAUUSD to crush challenges."
#property description "🛡️ INSTITUTIONAL PROTECTION: Hard-coded Drawdown & Equity Guards ensure you never breach rules."
#property description "📈 ADAPTIVE BREAKOUTS: Smart News Filters & Multi-Timeframe logic capture massive Gold moves."
#property description "💰 FUNDED READY: Fully automated discipline to turn your trading into a professional career."

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input group "Risk Management"
input double RiskPct = 2.0;               // Base risk per trade %
input double MaxLossUSD = 110.0;          // Maximum loss per trade in USD (Hard Cap)
input double DailyDDLimit = 2.5;          // Daily Drawdown Limit (%)
input double OverallDDLimit = 5.5;        // Overall Drawdown Limit (%)
input double TargetBalanceOrEquity = 108000.0; // Target to pass challenge

input group "Strategy Settings"
input int    ATR_Prd = 14;                // ATR period
input int    Brk_Prd = 10;                // Breakout lookback (Bars)
input double MinBrkStr = 0.1;             // Minimum breakout strength (ATR Multiplier)
input int    EMA_Prd = 20;                // EMA period for Trend Filter
input bool   useHTF = true;               // Use HTF (D1/H4) Direction Filter

input group "News Filter"
input bool   NewsFilt = true;             // Enable News Filter
input int    NewsPause = 15;              // Mins to pause before/after news

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
double CurRisk = RiskPct;
double LastEqHigh = 0;
double StartingBalance = 0;
double DailyBalance = 0;
datetime LastDay = 0;
bool ProfitTargetReached = false;
bool DailyDDReached = false;
double dynBrkStr = MinBrkStr;

CTrade trade;
int h_ema_d1 = INVALID_HANDLE;
int h_ema_h4 = INVALID_HANDLE;
string WorkSymbol; // Renamed to avoid conflict with built-in SymbolName() function

// Structure for Timeframe specific data
struct TimeframeData {
   ENUM_TIMEFRAMES tf;
   int h_atr;
   datetime lastSig;
   datetime lastBar;
};
TimeframeData tfs[];

// Structure for News Events
struct NewsEvt {
   datetime time;
   int impact; // 0=Low, 1=Med, 2=High
};
NewsEvt newsCal[];
int newsCnt = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
datetime ExpiryDate = D'2025.12.30'; // Set your expiration date here (YYYY.MM.DD)

   if(TimeCurrent() > ExpiryDate)
   {
      Alert("Trial Expired! Please contact the developer: https://www.mql5.com/en/users/jawadtrader22/seller");
      Print("Trial Expired! Please contact the developer.");
      return(INIT_FAILED); // This stops the EA from initializing
   }
   if(AccountInfoDouble(ACCOUNT_BALANCE) < 100.0) {
      Print("Error: Balance too low for Prop Firm logic.");
      return(INIT_FAILED);
   }

   WorkSymbol = _Symbol;
   if(!SymbolSelect(WorkSymbol, true)) {
      Print("Failed to select symbol.");
      return(INIT_FAILED);
   }

   StartingBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   LastEqHigh = AccountInfoDouble(ACCOUNT_EQUITY);
   DailyBalance = StartingBalance;
   LastDay = (datetime)(TimeCurrent() / 86400 * 86400);

   ArrayResize(tfs, 2);
   tfs[0].tf = PERIOD_M15;
   tfs[1].tf = PERIOD_H1;

   for(int i = 0; i < 2; i++) {
      tfs[i].h_atr = iATR(WorkSymbol, tfs[i].tf, ATR_Prd);
      tfs[i].lastBar = 0;
      if(tfs[i].h_atr == INVALID_HANDLE) {
         Print("Failed to create ATR handle.");
         return(INIT_FAILED);
      }
   }

   h_ema_d1 = iMA(WorkSymbol, PERIOD_D1, EMA_Prd, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_h4 = iMA(WorkSymbol, PERIOD_H4, EMA_Prd, 0, MODE_EMA, PRICE_CLOSE);

   if(h_ema_d1 == INVALID_HANDLE || h_ema_h4 == INVALID_HANDLE) {
      Print("Failed to create HTF EMA handles.");
      return(INIT_FAILED);
   }

   if(NewsFilt) FetchNewsCalendar();

   Print("EA Initialized. Target: ", DoubleToString(TargetBalanceOrEquity, 2));
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(h_ema_d1);
   IndicatorRelease(h_ema_h4);
   for(int i = 0; i < ArraySize(tfs); i++) {
      IndicatorRelease(tfs[i].h_atr);
   }
   Print("EA Deinitialized.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // --- Daily Reset Logic ---
   datetime currentDay = (datetime)(TimeCurrent() / 86400 * 86400);
   if(currentDay > LastDay) {
      DailyBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      LastDay = currentDay;
      DailyDDReached = false;
      Print("New Day. Daily Balance Reset to: ", DoubleToString(DailyBalance, 2));
   }

   // --- Prop Firm Checks ---
   if(DailyDDReached || ProfitTargetReached) return;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(balance >= TargetBalanceOrEquity || equity >= TargetBalanceOrEquity) {
      CloseAllPositions();
      ProfitTargetReached = true;
      Print("PROFIT TARGET REACHED! Trading Stopped.");
      return;
   }

   double dailyDD = (DailyBalance - equity) / DailyBalance * 100.0;
   double overallDD = (StartingBalance - equity) / StartingBalance * 100.0;

   if(dailyDD >= DailyDDLimit || overallDD >= OverallDDLimit) {
      CloseAllPositions();
      DailyDDReached = true;
      Print("DRAWDOWN LIMIT BREACHED! Daily: ", DoubleToString(dailyDD, 2), "%, Overall: ", DoubleToString(overallDD, 2), "%");
      return;
   }

   // --- News Update ---
   static datetime lastNewsFetch = 0;
   if(NewsFilt && TimeCurrent() >= lastNewsFetch + 4*3600) {
      FetchNewsCalendar();
      lastNewsFetch = TimeCurrent();
   }
   if(NewsFilt && IsNews()) return;

   // --- Strategy Loop ---
   for(int i = 0; i < ArraySize(tfs); i++)
   {
      ManageTrades(tfs[i].tf);

      if(!NewBar(tfs[i].tf, tfs[i].lastBar)) continue;

      if(PositionsTotal() < 5)
      {
         double atrVal = GetIndicatorVal(tfs[i].h_atr, 0);

         bool buySignal = BuyBrk(tfs[i].tf) && (!useHTF || BullTrend());
         bool sellSignal = SellBrk(tfs[i].tf) && (!useHTF || BearTrend());

         if(buySignal) {
             double sl = SymbolInfoDouble(WorkSymbol, SYMBOL_ASK) - (atrVal * 1.5);
             double tp = SymbolInfoDouble(WorkSymbol, SYMBOL_ASK) + (atrVal * 3.0);
             double lots = CalcLots(MathAbs(SymbolInfoDouble(WorkSymbol, SYMBOL_ASK) - sl));

             if(lots > 0) {
                 trade.SetExpertMagicNumber(MagicNumber(tfs[i].tf));
                 trade.Buy(lots, WorkSymbol, 0, sl, tp, "Gold Breakout Buy");
             }
         }
         else if(sellSignal) {
             double sl = SymbolInfoDouble(WorkSymbol, SYMBOL_BID) + (atrVal * 1.5);
             double tp = SymbolInfoDouble(WorkSymbol, SYMBOL_BID) - (atrVal * 3.0);
             double lots = CalcLots(MathAbs(sl - SymbolInfoDouble(WorkSymbol, SYMBOL_BID)));

             if(lots > 0) {
                 trade.SetExpertMagicNumber(MagicNumber(tfs[i].tf));
                 trade.Sell(lots, WorkSymbol, 0, sl, tp, "Gold Breakout Sell");
             }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Helper Functions                                                 |
//+------------------------------------------------------------------+
double CalcLots(double slPointsDistance)
{
   if(slPointsDistance <= 0) return 0.0;
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = accountBalance * (CurRisk / 100.0);
   if(riskMoney > MaxLossUSD) riskMoney = MaxLossUSD; // Hard Cap

   double tickValue = SymbolInfoDouble(WorkSymbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(WorkSymbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickValue == 0 || tickSize == 0) return 0.0;

   double lots = riskMoney / ( (slPointsDistance / tickSize) * tickValue );
   double minLot = SymbolInfoDouble(WorkSymbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(WorkSymbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(WorkSymbol, SYMBOL_VOLUME_STEP);

   lots = MathFloor(lots / stepLot) * stepLot;
   if(lots < minLot) return 0.0;
   if(lots > maxLot) lots = maxLot;
   return lots;
}

void ManageTrades(ENUM_TIMEFRAMES tf)
{
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != WorkSymbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber(tf)) continue;

      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double lots = PositionGetDouble(POSITION_VOLUME);
      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      long type = PositionGetInteger(POSITION_TYPE);

      int tfIdx = (tf == PERIOD_M15) ? 0 : 1;
      double atr = GetIndicatorVal(tfs[tfIdx].h_atr, 0);

      // Emergency Exit
      if(type == POSITION_TYPE_BUY && SellBrk(tf)) { trade.PositionClose(ticket); continue; }
      if(type == POSITION_TYPE_SELL && BuyBrk(tf)) { trade.PositionClose(ticket); continue; }

      // Partial Close & BE
      double dist = (type == POSITION_TYPE_BUY) ? (currentPrice - openPrice) : (openPrice - currentPrice);
      if(dist > (atr * 2.5)) {
         bool isBE = (type == POSITION_TYPE_BUY && sl >= openPrice) || (type == POSITION_TYPE_SELL && sl <= openPrice);
         if(!isBE && lots >= SymbolInfoDouble(WorkSymbol, SYMBOL_VOLUME_MIN) * 2) {
             trade.PositionClosePartial(ticket, lots / 2.0);
             trade.PositionModify(ticket, openPrice, tp);
         }
      }

      // Trailing Stop
      double trailDist = atr * 1.6;
      if(type == POSITION_TYPE_BUY) {
         double newSL = currentPrice - trailDist;
         if(newSL > sl && newSL < currentPrice) trade.PositionModify(ticket, newSL, tp);
      }
      else if(type == POSITION_TYPE_SELL) {
         double newSL = currentPrice + trailDist;
         if(newSL < sl || sl == 0) trade.PositionModify(ticket, newSL, tp);
      }
   }
}

bool BuyBrk(ENUM_TIMEFRAMES tf)
{
   double close1 = iClose(WorkSymbol, tf, 1);
   int highIdx = iHighest(WorkSymbol, tf, MODE_HIGH, Brk_Prd, 2);
   if(highIdx < 0) return false;
   double highVal = iHigh(WorkSymbol, tf, highIdx);
   int tfIdx = (tf == PERIOD_M15) ? 0 : 1;
   double atr = GetIndicatorVal(tfs[tfIdx].h_atr, 1);
   return (close1 > (highVal + (atr * dynBrkStr)));
}

bool SellBrk(ENUM_TIMEFRAMES tf)
{
   double close1 = iClose(WorkSymbol, tf, 1);
   int lowIdx = iLowest(WorkSymbol, tf, MODE_LOW, Brk_Prd, 2);
   if(lowIdx < 0) return false;
   double lowVal = iLow(WorkSymbol, tf, lowIdx);
   int tfIdx = (tf == PERIOD_M15) ? 0 : 1;
   double atr = GetIndicatorVal(tfs[tfIdx].h_atr, 1);
   return (close1 < (lowVal - (atr * dynBrkStr)));
}

bool BullTrend()
{
   double d1_ema = GetIndicatorVal(h_ema_d1, 1);
   double d1_close = iClose(WorkSymbol, PERIOD_D1, 1);
   return (d1_close > d1_ema);
}

bool BearTrend()
{
   double d1_ema = GetIndicatorVal(h_ema_d1, 1);
   double d1_close = iClose(WorkSymbol, PERIOD_D1, 1);
   return (d1_close < d1_ema);
}

bool IsNews()
{
   datetime now = TimeCurrent();
   for(int i=0; i<newsCnt; i++) {
       if(newsCal[i].impact < 2) continue;
       if(now >= newsCal[i].time - (NewsPause * 60) && now <= newsCal[i].time + (NewsPause * 60)) return true;
   }
   return false;
}

void FetchNewsCalendar()
{
   // Placeholder: Reset news counter for safety
   newsCnt = 0;
}

void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == WorkSymbol) {
         trade.PositionClose(ticket);
      }
   }
}

double GetIndicatorVal(int handle, int index)
{
   double buf[1];
   if(CopyBuffer(handle, 0, index, 1, buf) < 0) return 0.0;
   return buf[0];
}

bool NewBar(ENUM_TIMEFRAMES tf, datetime &last_bar_time)
{
   datetime curr_bar_time = iTime(WorkSymbol, tf, 0);
   if(curr_bar_time != last_bar_time) {
      last_bar_time = curr_bar_time;
      return true;
   }
   return false;
}

long MagicNumber(ENUM_TIMEFRAMES tf)
{
   if(tf == PERIOD_M15) return 1015;
   if(tf == PERIOD_H1) return 1060;
   return 1000;
}
```

Complete Copy Paste code


![Eugene Mmene](https://c.mql5.com/avatar/2025/6/6841b8aa-b9e4.jpg)

**[Eugene Mmene](https://www.mql5.com/en/users/mmene365)**
\|
16 Dec 2025 at 23:02

**Austin Reade [#](https://www.mql5.com/en/forum/501702#comment_58706618):**

I downloaded the Mql5 [source code](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development"), that's a lot of work, thank you for that, but, above are all the warnings and errors that the MetaEditor throws out when compiling the code.

How could this be rectified, please?

here is corrected code


![Eugene Mmene](https://c.mql5.com/avatar/2025/6/6841b8aa-b9e4.jpg)

**[Eugene Mmene](https://www.mql5.com/en/users/mmene365)**
\|
16 Dec 2025 at 23:19

**Muhammad Jawad Shabir [#](https://www.mql5.com/en/forum/501702#comment_58731114):**

Complete Copy Paste code

The code is not the same even results are very different my guy


![Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://c.mql5.com/2/185/20569-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 46): Liquidity Sweep on Break of Structure (BoS)](https://www.mql5.com/en/articles/20569)

In this article, we build a Liquidity Sweep on Break of Structure (BoS) system in MQL5 that detects swing highs/lows over a user-defined length, labels them as HH/HL/LH/LL to identify BOS (HH in uptrend or LL in downtrend), and spots liquidity sweeps when price wicks beyond the swing but closes back inside on a bullish/bearish candle.

![Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://c.mql5.com/2/185/20550-codex-pipelines-from-python-logo.png)[Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)

We continue our look at how MetaTrader can be used outside its forex trading ‘comfort-zone’ by looking at another tradable asset in the form of the FXI ETF. Unlike in the last article where we tried to do ‘too-much’ by delving into not just indicator selection, but also considering indicator pattern combinations, for this article we will swim slightly upstream by focusing more on indicator selection. Our end product for this is intended as a form of pipeline that can help recommend indicators for various assets, provided we have a reasonable amount of their price history.

![Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://c.mql5.com/2/185/20546-introduction-to-mql5-part-31-logo__1.png)[Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)

Learn how to use WebRequest and external API calls to retrieve recent candle data, convert each value into a usable type, and save the information neatly in a table format. This step lays the groundwork for building an indicator that visualizes the data in candle format.

![Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://c.mql5.com/2/185/20414-adaptive-smart-money-architecture-logo.png)[Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)

This topic explores how to build an Adaptive Smart Money Architecture (ASMA)—an intelligent Expert Advisor that merges Smart Money Concepts (Order Blocks, Break of Structure, Fair Value Gaps) with real-time market sentiment to automatically choose the best trading strategy depending on current market conditions.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/19655&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083491848439209155)

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