# QuantMindX: Complete Technical Documentation & Backend Architecture

> **A Self-Evolving AI Trading Ecosystem for Growing Small Accounts Through Darwinian Strategy Evolution**

---

## 🌟 Core Philosophy & Vision

### The Problem We're Solving

Most retail traders fail for three reasons:
1. **Emotional Trading:** Fear and greed override logical decisions
2. **Static Strategies:** Markets change, but traders don't adapt their approaches
3. **Limited Testing:** No time or skill to test hundreds of strategy variations

### The QuantMindX Solution

QuantMindX is a personal AI-powered trading assistant that operates like a biological ecosystem. Instead of running one "perfect" strategy, the system:

- **Generates 100+ micro-strategies** (tiny trading bots, each with $1-$10 capital)
- **Tests them continuously** against real market data
- **Kills underperformers** ruthlessly (no emotional attachment)
- **Evolves winners** by combining successful patterns (genetic algorithm)
- **Adapts automatically** to changing market conditions

**Core Analogy:** Think of it as a garden. You plant 100 seeds (strategies). Some grow, most die. You cross-pollinate the strongest plants. Over time, your garden becomes filled with only the most resilient, high-yielding plants. No manual watering (no emotional intervention) - just natural selection.

### Primary Goal

**Grow a $100 trading account to $10,000 in 90 days** through:
- Rapid strategy iteration (100+ tests per week)
- Ruthless elimination of losers (dead bots killed in 3 days)
- Aggressive scaling of winners (2x capital allocation weekly)
- Zero emotional interference (100% data-driven decisions)

### Target User

- **You (Mubarak):** 20-year-old polymath building personal trading tool
- **Trading Experience:** Learning to trade (manual practice in parallel)
- **Markets:** Forex, Crypto, Stocks (focus: day trading/intraday)
- **Starting Capital:** $100 - $500
- **Risk Tolerance:** Aggressive but disciplined (5% daily loss limit)
- **Time Commitment:** Minimal daily oversight (approve bot deployments, review weekly reports)

---

## 🏛️ System Architecture Overview

### High-Level Philosophy

QuantMindX is organized like a **corporation with specialized departments (Guilds)**. Each Guild has:
- **1 Master Agent** (Department Head) - Makes final decisions
- **Multiple Worker Agents** (Team Members) - Execute specialized tasks
- **Shared Modules** (Tools) - Used by all guild members

**Key Innovation:** Guilds operate as **single Docker containers** with internal function calls (fast), while Guild-to-Guild communication uses **FastAPI-MCP** (structured, semantic context).

### The 5 Guilds + 2 Leadership Agents

```
                    ┌─────────────────────────┐
                    │   ORCHESTRATOR (CEO)    │
                    │  - Sets goals            │
                    │  - Approves deployments  │
                    │  - Creates mini-agents   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ SYSTEM ARCHITECT (VP)   │
                    │  - Monitors performance  │
                    │  - Proposes optimizations│
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐   ┌───────▼────────┐   ┌──────▼──────────┐
│ RESEARCH      │   │ ENGINEERING &  │   │ OPERATIONS      │
│ GUILD         │   │ DEVELOPMENT    │   │ GUILD           │
│               │   │ GUILD          │   │ (Risk+Monitor+  │
│ - Finds       │   │                │   │  Execution)     │
│   strategies  │   │ - Codes bots   │   │                 │
│ - Deep search │   │ - Backtests    │   │ - Live trading  │
│ - Imports     │   │ - Validates    │   │ - Risk mgmt     │
│   content     │   │                │   │ - Journals      │
└───────┬───────┘   └───────┬────────┘   └──────┬──────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                    ┌───────▼─────────┐
                    │ EVOLUTION &     │
                    │ ANALYSIS GUILD  │
                    │                 │
                    │ - Improves bots │
                    │ - Combines      │
                    │   winners       │
                    │ - Pattern recog │
                    └─────────────────┘
```

---

## 🎭 Detailed Guild Specifications

### GUILD 1: RESEARCH GUILD 
*"The Knowledge Hunters"*

**Purpose:** Discover profitable trading strategies from the internet, books, and your personal files.

**Master Agent:** Chief Librarian
**Worker Agents:**
1. **Content Harvester** - Scrapes YouTube, Instagram, Reddit, X (Twitter), TradingView
2. **Deep Research Agent** - Uses MCP (Model Context Protocol) for web searches
3. **Content Parser** - Converts everything (PDFs, videos, code) to `.md` format
4. **Knowledge Indexer** - Stores strategies in ChromaDB with semantic embeddings
5. **Content Ingestion Manager** - Handles GitHub repos and local folder imports

**Shared Modules:**
- **Context Compressor:** Summarizes daily agent conversations (prevents context bloat)
- **Format Converter:** PDF → Text, Video → Transcript, Code → Documentation
- **Strategy Recommendation Engine:** Monitors trending strategies, suggests new ones

**What This Guild Learns:**
- Which sources produce profitable strategies (e.g., "ICT YouTube videos = 68% backtest success")
- Which keywords indicate quality (e.g., "backtested" vs "guaranteed money")
- Pattern: After 30 days, knows that Reddit r/Forex posts with >100 upvotes are 40% more likely to backtest profitably

**How It Works (Example):**

**Scenario:** You paste a YouTube link about "ICT Macros Trading Strategy"

1. **Content Harvester** downloads video using `yt-dlp`
2. Extracts audio, sends to **AssemblyAI** for transcription
3. **Content Parser** reads transcript:
   ```
   "At 9:30 AM New York time, look for liquidity sweeps. 
   If price wicks above the previous high and reverses, 
   enter short with stop loss above the wick."
   ```
4. Parser uses LLM to extract structured strategy:
   ```json
   {
     "strategy_name": "ICT 9:30 Liquidity Sweep",
     "entry_conditions": [
       "Time = 9:30 AM EST",
       "Price wicks above previous high",
       "Reversal candle forms"
     ],
     "exit_conditions": [
       "Stop loss: Above wick high",
       "Take profit: Previous day low"
     ],
     "indicators": ["Time-based", "Support/Resistance"],
     "source": "https://youtube.com/watch?v=abc123"
   }
   ```
5. **Knowledge Indexer** stores in ChromaDB with embedding
6. **Deep Research Agent** searches: "ICT Macros strategy backtesting results"
7. Finds 3 blog posts with backtest data, adds to strategy notes
8. **Chief Librarian** notifies Engineering Guild: "New strategy ready for coding"

**Technologies Used:**
- **yt-dlp:** YouTube video download
- **AssemblyAI API:** Audio transcription
- **Docling (GitHub):** PDF/Word document parsing
  - Link: https://github.com/docling-project/docling
  - Implementation: Replace manual PDF parsing. Install via `pip install docling`, use `DocumentConverter` class
- **Gitingest (GitHub):** Efficient GitHub repo ingestion
  - Link: https://github.com/coderamp-labs/gitingest
  - Implementation: When user pastes GitHub URL, use Gitingest to clone and index all files
- **ChromaDB:** Vector database for semantic search
- **FastAPI-MCP:** Guild-to-guild communication
  - Link: https://github.com/tadata-org/fastapi_mcp
  - Implementation: Each guild runs MCP server, exposes endpoints for context sharing
- **Agentic Context Engine (GitHub):** Intelligent context management
  - Link: https://github.com/kayba-ai/agentic-context-engine
  - Implementation: Agents use context engine to fetch only relevant data, reducing token usage

**Container:** `research-guild-service`

---

### GUILD 2: ENGINEERING & DEVELOPMENT GUILD**
*"The Bot Builders"*

**Purpose:** Transform strategy ideas into executable trading bots, then rigorously test them.

**Master Agent:** Chief Engineer
**Worker Agents:**
1. **Strategy Designer** - Converts strategy JSON into trading logic pseudocode
2. **Code Developer** - Writes actual Python/MQL5 bot code using LLM
3. **Code Validator** - Checks for security issues, bugs, syntax errors
4. **Backtesting Coordinator** - Manages parallel backtests (10 simultaneous)
5. **Performance Analyzer** - Calculates win rate, Sharpe ratio, max drawdown

**Shared Modules:**
- **Strategy DNA Registry:** Tracks bot "genes" (entry logic, exit logic, filters, risk rules)
- **Template Library:** Pre-built bot components (RSI entry, MACD exit, ATR filter)
- **Backtest Queue Manager:** CPU-aware scheduling (detects available cores, queues backtests)
- **"What-If" Simulator:** Test strategy changes without deployment

**What This Guild Learns:**
- Which code patterns execute faster (e.g., "NumPy array operations 10x faster than loops")
- Which indicator combinations work best (e.g., "RSI + Volume filter = 12% better win rate")
- Pattern: After 50 bots, learns that adding ATR filter improves 80% of strategies

**How It Works (Example):**

**Scenario:** Engineering Guild receives "ICT 9:30 Liquidity Sweep" strategy from Research Guild

1. **Strategy Designer** creates pseudocode:
   ```
   WAIT until time = 9:30 AM EST
   IF price creates new high in last 5 candles:
     IF next candle closes below that high:
       ENTER SHORT
       SET stop_loss = high + 10 pips
       SET take_profit = previous_day_low
   ```

2. **Code Developer** uses LLM (GPT-4 or DeepSeek) to generate Python code:
   ```python
   class ICT_LiquiditySweep_Bot:
       def check_entry(self, candles, current_time):
           if current_time.hour != 9 or current_time.minute != 30:
               return None
           
           last_5_highs = [c.high for c in candles[-5:]]
           recent_high = max(last_5_highs)
           current_candle = candles[-1]
           
           if current_candle.close < recent_high:
               return {
                   "direction": "SHORT",
                   "entry_price": current_candle.close,
                   "stop_loss": recent_high + 0.0010,  # 10 pips
                   "take_profit": self.get_previous_day_low(candles)
               }
           return None
   ```

3. **Code Validator** runs checks:
   - ✅ Syntax valid
   - ✅ No unsafe operations (no `os.system`, no file writes)
   - ✅ Logic matches strategy description
   - ⚠️ Warning: No position sizing logic (adds default 1% risk)

4. **Backtesting Coordinator** queues bot for testing:
   - Fetches 1 year EUR/USD 5-minute data from **yfinance** or **CCXT**
   - Runs backtest using **Backtrader** framework
   - Tests on 3 market conditions: Bull (Q1 2024), Bear (Q3 2023), Ranging (Q2 2023)

5. **Performance Analyzer** calculates metrics:
   ```
   Win Rate: 64%
   Profit Factor: 1.52
   Sharpe Ratio: 1.18
   Max Drawdown: 12.3%
   Total Trades: 87
   Avg Trade Duration: 3.2 hours
   ```

6. **Chief Engineer** evaluates:
   - ✅ Win rate > 60% (passes)
   - ✅ Sharpe ratio > 1.0 (passes)
   - ✅ Max drawdown < 15% (passes)
   - **Decision:** Promote to paper trading

7. Bot gets tag `@primal` and moves to Operations Guild for paper trading

**Technologies Used:**
- **Backtrader:** Professional backtesting framework
  - Install: `pip install backtrader`
  - Implementation: Create `BacktraderEngine` class, feed OHLCV data, run strategy
- **Alternative Backtesting Libraries** (for variety):
  - **FinmarketPy:** `pip install finmarketpy`
  - **PyAlgoTrade:** `pip install pyalgotrade`
  - **FastQuant:** `pip install fastquant`
  - Implementation: Abstract `BacktestEngine` interface, multiple implementations (Backtrader, PyAlgoTrade, etc.)
- **Pandas/NumPy/TA-Lib:** Data manipulation and technical indicators
  - Install: `pip install pandas numpy ta-lib`
- **Anthropic Sandbox Runtime (GitHub):** Secure code execution
  - Link: https://github.com/anthropic-experimental/sandbox-runtime
  - Implementation: All generated bot code runs in sandbox. Install via Docker, use `SandboxSession` class
- **CCXT:** Crypto exchange integration
  - Install: `pip install ccxt`
  - Implementation: For crypto bots, use CCXT to fetch historical data from Binance/Coinbase
- **FastAPI-MCP:** Communication with other guilds

**Container:** `engineering-guild-service`

---

### GUILD 3: OPERATIONS GUILD** *(MERGED: Risk + Monitoring + Execution)*
*"The Money Managers"*

**Purpose:** Execute live trades, enforce risk limits, monitor performance, and keep detailed journals.

**Why Merged:** All three functions (risk, monitoring, execution) deal with **live money in real-time**. Merging into one container eliminates network latency - Position Monitor can instantly call Drawdown Enforcer via in-process function call (microseconds) instead of HTTP request (milliseconds). In fast markets, milliseconds matter.

**Master Agent:** Chief Operations Officer
**Worker Agents:**

**From Risk Management:**
1. **Drawdown Enforcer** - Pauses bots if daily loss > 5% or max drawdown > 15%
2. **Capital Allocator** - Decides how much $ each bot gets (uses optimization algorithm)
3. **Dynamic SL/TP Engine** - Adjusts stop loss and take profit based on volatility and bot performance

**From Monitoring:**
4. **Position Monitor** - Tracks all open trades in real-time (updates every second)
5. **Performance Tracker** - Calculates live win rate, P&L, Sharpe ratio per bot
6. **Alert Manager** - Sends notifications (desktop app, email, future: mobile push)
7. **Journal Writer** - Logs every trade, every bot state change, every risk event

**From Execution:**
8. **MT5 Connector** - Communicates with MetaTrader 5 for Forex/CFD trading
9. **Crypto Connector** - Uses CCXT for crypto exchange trading
10. **Order Manager** - Submits buy/sell orders, handles retries
11. **Slippage Monitor** - Tracks execution quality (expected price vs actual price)
12. **Broker Health Checker** - Detects broker outages, triggers disaster protocols

**Shared Modules:**
- **Psychology Module:** Tracks market sentiment (VIX, social media sentiment, news events)
- **Noise Mitigation Module:** Filters false signals using Kalman filter and regime detection
- **Emergency Stop System:** Circuit breaker for account protection
- **Disaster Protocol Handler:** Handles crashes, outages, black swan events
- **Capital Rebalancing Optimizer:** Weekly reallocation of capital based on bot performance

**What This Guild Learns:**
- Risk patterns (e.g., "Bots with SL < 1% fail 70% of the time")
- Correlation patterns (e.g., "Don't run 5 EUR/USD bots simultaneously - too much exposure")
- Execution patterns (e.g., "Orders at 00:15 GMT have 30% more slippage")
- Performance patterns (e.g., "Bot_007 loses money every Wednesday during London session")

**How It Works (Example):**

**Scenario:** Bot_007 generates a BUY signal for EUR/USD

1. **Order Manager** receives signal from bot code running in Engineering Guild's sandbox

2. **Dynamic SL/TP Engine** calculates adaptive levels:
   ```python
   market_atr = 45 pips  # Current volatility
   bot_recent_win_rate = 0.71  # Bot is performing well
   session = "london_open"  # High volatility period
   
   # Adaptive calculation:
   base_sl = 1.2%
   sl_multiplier = 1.2  # London = extra room for whipsaws
   final_sl = base_sl * sl_multiplier  # 1.44%
   
   base_tp = 1.5%
   tp_multiplier = 1.1  # Bot performing well = let winners run
   final_tp = base_tp * tp_multiplier  # 1.65%
   ```

3. **Position Monitor** checks current exposure:
   - Already have 2 EUR/USD positions (total $15 risk)
   - Account balance: $280
   - Max allowed per trade: 1% = $2.80
   - Decision: Allow trade, but reduce position size to $2

4. **Order Manager** submits order to MT5:
   - Entry: 1.08567
   - SL: 1.08423 (1.44% below)
   - TP: 1.08732 (1.65% above)
   - Size: 0.02 lots (~$2 risk)

5. **Slippage Monitor** tracks execution:
   - Requested: 1.08567
   - Filled: 1.08569 (2 pips slippage)
   - Acceptable (<3 pips), no action needed

6. **Position Monitor** updates every second:
   - Current P&L: +$0.43
   - Time in trade: 3 minutes 14 seconds
   - Distance to SL: 12 pips
   - Distance to TP: 18 pips

7. **Journal Writer** logs everything:
   ```json
   {
     "timestamp": "2025-10-23T14:32:11Z",
     "bot_id": "Bot_007_RSI_Scalper_v2",
     "action": "ENTRY_LONG",
     "pair": "EUR/USD",
     "entry_price": 1.08567,
     "stop_loss": 1.08423,
     "take_profit": 1.08732,
     "position_size": 0.02,
     "risk_amount": 2.00,
     "reasoning": "RSI oversold + support level + volume spike"
   }
   ```

**Technologies Used:**
- **MetaTrader5 (MT5) API:** Forex/CFD trading
  - Install: `pip install MetaTrader5`
  - Implementation: Connect to MT5 terminal, submit orders, get account info
- **CCXT:** Crypto exchange integration
  - Install: `pip install ccxt`
  - Implementation: For crypto bots, connect to Binance/Coinbase APIs
- **Redis:** Real-time data storage (positions, P&L)
- **PostgreSQL:** Trade journal storage
- **FastAPI-MCP:** Guild-to-guild communication
- **Kalman Filter (GitHub):** Noise reduction
  - Link: https://github.com/ruvnet/claude-flow
  - Implementation: Use Kalman filter to smooth price data and detect real signals vs noise
- **TradingAgents (GitHub):** Advanced trading agent patterns
  - Link: https://github.com/TauricResearch/TradingAgents
  - Implementation: Borrow position sizing and risk management patterns

**Container:** `operations-guild-service`

---

### GUILD 4: EVOLUTION & ANALYSIS GUILD**
*"The Strategy Improvers"*

**Purpose:** Continuously improve trading strategies through genetic algorithms and pattern recognition.

**Master Agent:** Chief Evolutionary Strategist
**Worker Agents:**
1. **Performance Analyzer** - Studies what worked/failed across all bots
2. **Pattern Recognizer** - Identifies market regime changes
3. **Bot Reconfigurer** - Modifies underperforming bots
4. **Strategy Breeder** - Cross-breeds successful bots
5. **Fragment Analyzer** - Identifies which strategy components work best

**Shared Modules:**
- **Genetic Algorithm Engine:** Mutation, crossover, selection
- **Market Regime Classifier:** Bull/bear/ranging/volatile detection
- **Strategy DNA Registry:** Tracks bot genealogy and mutations
- **Performance Prediction Model:** Estimates how changes will affect performance

**What This Guild Learns:**
- Which strategy fragments combine well (e.g., "RSI entry + ATR filter = 15% improvement")
- Which mutations improve performance (e.g., "Wider stop loss in volatile markets")
- Pattern: After 100 evolutions, learns that "London session avoidance" improves 60% of strategies

**How It Works (Example):**

**Scenario:** Saturday 00:01 Evolution Cycle

1. **Performance Analyzer** collects data from past week:
   - 50 active bots
   - 12 performing well (tagged `@perfect`)
   - 23 average (tagged `@pending`)
   - 15 failing (tagged `@quarantine` or `@dead`)

2. **Pattern Recognizer** analyzes market conditions:
   - "EUR/USD was ranging (ADX < 20) Monday-Wednesday"
   - "Crypto was volatile (BTC > 5% daily moves) Thursday-Friday"
   - "News events: FOMC minutes Wednesday, CPI data Friday"

3. **Fragment Analyzer** identifies successful components:
   - Entry genes in `@perfect` bots: ["RSI_oversold", "support_touch", "volume_spike"]
   - Exit genes in `@perfect` bots: ["fixed_RR_ratio", "trailing_stop", "time_exit"]
   - Filter genes in `@perfect` bots: ["london_session_avoid", "high_volatility_only"]

4. **Strategy Breeder** creates new combinations:
   - Takes entry genes from Bot_007 (RSI_oversold + support_touch)
   - Takes exit genes from Bot_023 (trailing_stop)
   - Takes filter genes from Bot_045 (london_session_avoid)
   - Creates new bot: Bot_100_hybrid_v1

5. **Bot Reconfigurer** improves underperformers:
   - Bot_089 has 3 consecutive losses
   - Analysis: "Stop loss too tight for current volatility"
   - Modification: Increase SL from 1.2% to 1.8% during high volatility
   - Creates Bot_089_v2

6. **Genetic Algorithm Engine** applies mutations:
   - Randomly changes RSI period from 14 to 16
   - Adds volume filter requirement
   - Adjusts position sizing from 1% to 1.5%
   - Creates 20 mutated variants

7. **Performance Prediction Model** estimates:
   - Bot_100_hybrid_v1: Predicted win rate 67% (high confidence)
   - Bot_089_v2: Predicted win rate 58% (medium confidence)
   - Bot_091_mutated_v3: Predicted win rate 45% (low confidence)

8. **Chief Evolutionary Strategist** prioritizes:
   - Send Bot_100_hybrid_v1 to Engineering for immediate backtesting
   - Queue Bot_089_v2 for backtesting (medium priority)
   - Archive Bot_091_mutated_v3 (low priority)

**Technologies Used:**
- **DEAP (GitHub):** Genetic algorithm framework
  - Install: `pip install deap`
  - Implementation: Use DEAP for strategy evolution, mutation, crossover
- **Scikit-learn:** Pattern recognition and market regime classification
  - Install: `pip install scikit-learn`
  - Implementation: Train classifiers to detect market conditions
- **TA-Lib:** Technical indicators for pattern analysis
  - Install: `pip install ta-lib`
  - Implementation: Calculate ADX, ATR, RSI for regime detection
- **Kronos (GitHub):** Time series analysis for market patterns
  - Link: https://github.com/shiyu-coder/Kronos
  - Implementation: Use Kronos for advanced time series pattern recognition
- **Motia (GitHub):** Agent workflow orchestration
  - Link: https://github.com/MotiaDev/motia
  - Implementation: Use Motia to coordinate complex evolution workflows

**Container:** `evolution-guild-service`

---

## 🗣️ Agent Communication Protocol

### Message Bus Architecture

```
┌─────────────────────────────────────────────────┐
│         Redis Message Bus (Priority Queue)       │
│  Channels:                                       │
│  • critical (P1): Emergency stops, failures      │
│  • high (P2): Bot performance alerts             │
│  • medium (P3): Routine updates                  │
│  • low (P4): Logging, metrics                    │
└─────────────────────────────────────────────────┘
           ↓              ↓              ↓
     [Research]    [Engineering]    [Operations]  ...
```

### Command Structure

```json
{
  "command_id": "cmd_20251030_143022_001",
  "command_type": "BOT_NEEDS_EVOLUTION",
  "priority": 2,
  "from": "Operations_Guild.Drawdown_Enforcer",
  "to": "Evolution_Guild.Chief_Evolutionary_Strategist",
  "timestamp": "2025-10-30T14:30:22Z",
  "payload": {
    "bot_id": "Bot_007",
    "issue": "15% drawdown in 48 hours",
    "current_tag": "@pending"
  },
  "context_key": "redis:bot_007_full_context",
  "requires_user_approval": false,
  "audit_trail": ["Operations_Guild detected drawdown", "Auto-triggered evolution request"]
}
```

### Your Visibility Dashboard

**Real-time feed shows:**
```
14:30:22 [P2] Operations → Evolution: Bot_007 needs evolution (15% drawdown)
14:30:45 [P3] Evolution → Engineering: Request Bot_007 performance data
14:31:12 [P3] Engineering → Evolution: Sending backtest results
14:32:05 [P2] Evolution → Operations: Proposing Bot_007_v2 with ATR filter
14:32:30 [P1] Operations → Orchestrator: Bot_007_v2 ready for approval
```

You can click any message to see full context.

---

## 🧠 Tiered Memory System

### Tier 1: Working Memory (Redis, 1-hour TTL)
**Purpose:** Instant access for current operations
**Contents:**
- Active bot states (50 bots × 10KB = 500KB)
- Open positions (real-time P&L)
- Last 100 messages per guild
- Current market conditions

**Access Time:** <5ms

---

### Tier 2: Short-Term Memory (PostgreSQL, 30 days)
**Purpose:** Recent history for pattern detection
**Contents:**
- Last 30 days of trades
- Bot performance metrics (daily snapshots)
- Weekly evolution reports
- Guild-to-guild conversation logs

**Access Time:** <50ms

---

### Tier 3: Long-Term Memory (ChromaDB, permanent)
**Purpose:** Semantic search across all historical data
**Contents:**
- All strategies ever tested (DNA + results)
- Pattern library (e.g., "RSI works in ranging markets")
- Graveyard (failed strategies with reasons)
- Research articles (YouTube transcripts, PDFs)

**Access Time:** <500ms (semantic search)

**Strategy DNA Example:**
```json
{
  "bot_id": "Bot_007_v3",
  "dna": {
    "entry_genes": ["RSI_oversold_28", "support_level", "volume_1.5x"],
    "exit_genes": ["RSI_50", "take_profit_1.5pct"],
    "filter_genes": ["london_session_avoid", "atr_below_50"],
    "risk_genes": ["stop_loss_1.2pct", "position_size_1pct_account"]
  },
  "parents": ["Bot_007_v2", "Bot_023_v1"],
  "mutations": ["Added atr_below_50 filter"],
  "performance": {
    "backtest_win_rate": 0.68,
    "live_win_rate": 0.71,
    "sharpe_ratio": 1.42,
    "tag": "@perfect"
  }
}
```

---

### Tier 4: Archive (S3/Cold Storage, 1+ year)
**Purpose:** Compliance and long-term analysis
**Contents:**
- Old backtests (2+ years)
- Ancient logs
- Tax records

**Access Time:** 1-5 seconds

---

## 🏷️ Bot Tag System

**Tags:**
- `@primal` - New bot, in initial testing (paper trading)
- `@pending` - Deployed but not proven yet (<30 trades)
- `@perfect` - Consistent performer (60%+ win rate, 100+ trades)
- `@quarantine` - Showing weird behavior, needs investigation
- `@dead` - Killed, moved to graveyard

**Tag Rules:**
- `@primal` → Minimum 50 paper trades before promotion
- `@primal` → `@pending` when deployed live (requires your approval)
- `@pending` → `@perfect` after 100 trades + 60% win rate + Sharpe > 1.0
- `@pending` → `@quarantine` if: 3 consecutive losses OR 10% drawdown in 24 hours
- `@quarantine` → `@dead` after 3 failed evolution attempts
- `@perfect` → `@pending` if: Win rate drops 10% OR drawdown > 15%

**Who Assigns:**
- Operations Guild: Can assign `@quarantine`, `@dead`
- Evolution Guild: Can assign `@pending` (after evolution)
- You: Can assign any tag manually

---

## 🔄 Saturday 00:01 Evolution Cycle

**What Happens:**

**Step 1: Data Collection (00:01 - 00:15)**
- Operations Guild generates performance report for all bots
- Evolution Guild fetches all `@perfect` and `@pending` bots
- System Architect checks for bottlenecks

**Step 2: Pattern Analysis (00:15 - 00:45)**
- Evolution Guild queries ChromaDB: "Which genes appear in all @perfect bots?"
- Finds patterns: "All top performers use ATR filter + London session avoidance"

**Step 3: Breeding & Mutation (00:45 - 02:00)**
- Cross-breed top 10 @perfect bots (combine genes)
- Mutate 5 random @pending bots (change one parameter)
- Generate 20 new experimental bots

**Step 4: Backtesting (02:00 - 04:00)**
- Engineering Guild backtests all 20 new bots in parallel
- Pass threshold: Win rate > 55%, Sharpe > 0.8

**Step 5: Approval Queue (04:00)**
- Orchestrator sends you notification: "15 new bots ready for review"
- You wake up, review dashboard, approve 10, reject 5

**Step 6: Deployment (04:30 - 05:00)**
- Operations Guild deploys approved bots to paper trading
- All bots start with `@primal` tag

---

## 🚨 Disaster Protocols

### 1. Market Crash Protocol
**Trigger:** VIX > 50 OR Account drawdown > 10% in 1 hour

**Actions:**
1. Operations Guild closes ALL positions (market orders)
2. Operations Guild pauses ALL bots
3. Orchestrator sends SMS: "EMERGENCY: All trading halted"
4. System waits for your manual restart

---

### 2. Broker Outage Protocol
**Trigger:** MT5 connection fails for 30+ seconds

**Actions:**
1. Operations Guild attempts reconnection (3 retries)
2. If fails → Try backup broker (if configured)
3. If backup fails → Operations Guild marks all positions as "UNKNOWN"
4. Orchestrator sends SMS: "BROKER OUTAGE: Manual intervention needed"

---

### 3. LLM Outage Protocol
**Trigger:** OpenAI/Anthropic API down

**Actions:**
1. Evolution Guild pauses evolution (can't generate new bots)
2. Engineering Guild uses cached code templates (no LLM needed)
3. Operations Guild continues monitoring (rule-based, no LLM)
4. System Architect logs: "LLM outage, degraded mode active"
5. System continues trading existing bots (no new strategy generation)

---

### 4. User Unavailable Protocol
**Trigger:** You don't approve bots for 72+ hours

**Actions:**
1. Orchestrator auto-approves LOW-RISK changes:
   - Bot parameter tweaks (e.g., RSI 28 → 30)
   - Minor position size adjustments
2. Orchestrator QUEUES HIGH-RISK changes:
   - New bot deployments
   - Major strategy overhauls
3. Max auto-approved changes: 3 per day
4. When you return, review queue shows: "12 changes queued during your absence"

---

## 📊 Self-Learning Mechanisms

### Pattern Memory (What Failed & Why)

When a bot dies (`@dead`), Evolution Guild extracts lessons:

**Example:**
```
Bot_089 (MA Crossover) FAILED
Reason: 15 consecutive losses during ranging market
Market Condition: ADX < 20 (no trend)
Lesson Learned: "MA Crossover requires trending markets (ADX > 25)"
Action: Add to ChromaDB with embedding: "avoid_ma_crossover_in_ranging"
```

**Next Time:**
When Engineering Guild designs a new MA Crossover bot, it queries ChromaDB:
- Finds: "avoid_ma_crossover_in_ranging"
- **Auto-adds:** `if ADX < 25: skip_trade()` to the bot code

**This is self-learning.** The system remembers past failures and avoids them.

---

### Performance-Based Rule Evolution

**Current Rule:** "Stop loss = 1.2% for all bots"

**After 30 days:**
- Operations Guild analyzes: 80% of stopped-out trades would have recovered if SL was 1.5%
- But: 15% would have lost MORE with 1.5% SL

**Operations Guild proposes:**
- "During low volatility (ATR < 30 pips): SL = 1.2%"
- "During high volatility (ATR > 50 pips): SL = 1.8%"

**Orchestrator approves → Rule updated system-wide**

---

## 🎯 Alignment With "Grow $100 Quickly"

### How This Architecture Accelerates Growth:

**1. Micro-Strategy Diversity**
- 100+ bots testing different approaches simultaneously
- Even if 80 fail, 20 winners can 10x your account

**2. Rapid Kill/Scale Mechanism**
- Bad bots die within 3 days (max 3 evolution attempts)
- Good bots get 2x capital allocation every week
- Example: Bot_007 starts with $1 → Week 2: $2 → Week 4: $8 → Week 8: $32

**3. Compound Effect**
- Week 1: $100 (starting capital)
- Week 4: $150 (50% growth from 20 profitable bots)
- Week 8: $280 (87% growth, winners scaled up)
- Week 12: $580 (107% growth, losers killed, winners dominating)
- **Target: $10,000 in 90 days = 223% monthly compound growth**

**4. Zero Emotional Interference**
- No "I think this bot will recover" → Data says die = bot dies
- No "I'm scared to deploy" → Backtesting passed = auto-deploy (with your approval)

---

## 🛠️ Tech Stack & Libraries

### Backend Framework
- **FastAPI:** Python web framework for all guild services
- **Docker:** Containerization for each guild
- **Redis:** Message bus and working memory
- **PostgreSQL:** Short-term memory and structured data
- **ChromaDB:** Long-term memory and vector embeddings

### Trading & Data
- **MetaTrader5:** Forex/CFD trading
- **CCXT:** Crypto exchange integration
- **Pandas/NumPy:** Data manipulation
- **TA-Lib:** Technical indicators
- **Backtrader:** Backtesting framework
- **yfinance:** Market data

### AI/ML
- **OpenAI API:** GPT-4 for code generation
- **Anthropic API:** Claude for reasoning
- **Sentence-Transformers:** Text embeddings
- **Scikit-learn:** Pattern recognition
- **DEAP:** Genetic algorithms

### Communication
- **FastAPI-MCP:** Guild-to-guild communication
- **WebSocket:** Real-time UI updates
- **Celery:** Background task processing

### Storage
- **MinIO/S3:** File storage (documents, backtests)
- **PostgreSQL:** Relational data
- **Redis:** Cache and message queue

---

## 🔗 GitHub Repositories Integration

### Must-Integrate Repositories

1. **Docling** - Document Processing
   - Link: https://github.com/docling-project/docling
   - Implementation: `pip install docling` and use `DocumentConverter` class for PDF/Word processing
   - Used in: Research Guild for content parsing

2. **Gitingest** - GitHub Repository Ingestion
   - Link: https://github.com/coderamp-labs/gitingest
   - Implementation: `pip install gitingest` and use to clone/index GitHub repos
   - Used in: Research Guild for importing trading strategies from GitHub

3. **FastAPI-MCP** - Agent Communication
   - Link: https://github.com/tadata-org/fastapi_mcp
   - Implementation: Each guild runs MCP server for structured communication
   - Used in: All guilds for inter-guild messaging

4. **Agentic Context Engine** - Context Management
   - Link: https://github.com/kayba-ai/agentic-context-engine
   - Implementation: `pip install agentic-context-engine` for intelligent context retrieval
   - Used in: All agents to reduce token usage

5. **Anthropic Sandbox Runtime** - Secure Code Execution
   - Link: https://github.com/anthropic-experimental/sandbox-runtime
   - Implementation: Docker-based sandbox for running generated bot code
   - Used in: Engineering Guild for safe code testing

6. **TradingAgents** - Trading Agent Patterns
   - Link: https://github.com/TauricResearch/TradingAgents
   - Implementation: Borrow position sizing and risk management patterns
   - Used in: Operations Guild for advanced trading logic

7. **Claude Flow** - Noise Reduction
   - Link: https://github.com/ruvnet/claude-flow
   - Implementation: Kalman filter implementation for price smoothing
   - Used in: Operations Guild for noise mitigation

8. **Motia** - Agent Workflow Orchestration
   - Link: https://github.com/MotiaDev/motia
   - Implementation: `pip install motia` for complex workflow coordination
   - Used in: Evolution Guild for managing evolution processes

### Nice-to-Have Repositories

1. **Kronos** - Time Series Analysis
   - Link: https://github.com/shiyu-coder/Kronos
   - Implementation: Advanced time series pattern recognition
   - Used in: Evolution Guild for market pattern analysis

2. **Caniscrape** - Web Scraping
   - Link: https://github.com/ZA1815/caniscrape
   - Implementation: Enhanced web scraping capabilities
   - Used in: Research Guild for content discovery

---


## 🚀 Deployment Architecture

### VPS Hosting Setup

**Recommended VPS Configuration:**
- **Provider:** DigitalOcean, Vultr, or AWS Lightsail
- **Specs:** 4+ CPU cores, 8GB+ RAM, 160GB+ SSD
- **OS:** Ubuntu 22.04 LTS
- **Docker:** Latest version
- **Domain:** Custom domain with SSL certificate

**Deployment Steps:**
1. Set up VPS with Docker and Docker Compose
2. Configure PostgreSQL and Redis
3. Deploy all guild services as Docker containers
4. Set up Nginx reverse proxy with SSL
5. Configure monitoring and alerting
6. Set up automated backups

### Local App Setup

**Desktop Application (Tauri-based):**
- **Frontend:** React/TypeScript for dashboard
- **Backend:** Rust Tauri for desktop wrapper
- **Communication:** WebSocket connection to VPS
- **Features:** Real-time monitoring, bot approval, journal viewing

**Mobile App (Future):**
- **Framework:** React Native or Flutter
- **Features:** Alerts, quick approvals, performance monitoring

---

## 📈 System Performance & Scaling

### Performance Metrics

**Target Performance:**
- **Message Processing:** <100ms between guilds
- **Bot Execution:** <500ms from signal to order
- **Backtesting:** 1 year of data in <5 minutes
- **UI Updates:** Real-time (<1 second latency)

### Scaling Strategy

**Horizontal Scaling:**
- Multiple instances of Engineering Guild for parallel backtesting
- Redis Cluster for message bus scaling
- PostgreSQL read replicas for analytics queries

**Vertical Scaling:**
- GPU acceleration for backtesting (future)
- More RAM for larger knowledge bases
- Faster SSD for reduced I/O latency

---

## 🔒 Security Considerations

### API Security
- JWT authentication for all API endpoints
- Rate limiting to prevent abuse
- IP whitelisting for critical operations

### Code Execution Security
- Sandbox environment for all generated bot code
- No file system access for trading bots
- Resource limits (CPU, memory, network)

### Data Security
- Encrypted storage for sensitive data
- Secure broker API key management
- Regular security audits

---

## 📊 Monitoring & Maintenance

### System Health Monitoring
- Service availability checks
- Resource usage monitoring
- Error rate tracking
- Performance metrics

### Trading Monitoring
- Real-time P&L tracking
- Position monitoring
- Risk limit enforcement
- Broker connection status

### Maintenance Tasks
- Daily database backups
- Weekly log rotation
- Monthly performance reviews
- Quarterly system updates

---

## 🎯 Success Metrics

### Trading Performance
- **Account Growth:** Target $100 → $10,000 in 90 days
- **Win Rate:** Maintain >60% across all active bots
- **Drawdown:** Keep maximum drawdown <20%
- **Sharpe Ratio:** Target >1.5 for the system

### System Performance
- **Uptime:** >99.5% availability
- **Latency:** <1 second for critical operations
- **Throughput:** Process 100+ strategies per week
- **Accuracy:** >95% correct trade execution

### User Experience
- **Daily Time:** <30 minutes required oversight
- **Approval Rate:** >80% of system recommendations approved
- **Satisfaction:** System reduces emotional trading stress

---

## 🙏 Conclusion

QuantMindX represents a paradigm shift in retail trading - from human-driven emotional decisions to AI-powered systematic evolution. By implementing this comprehensive architecture, you'll have a personal trading assistant that:

1. **Learns continuously** from market data and performance feedback
2. **Adapts automatically** to changing market conditions
3. **Removes emotion** from trading decisions
4. **Scales efficiently** from $100 to $10,000 through micro-strategy evolution
5. **Provides transparency** through detailed logging and reporting

The guild-based architecture ensures each aspect of trading - research, development, operations, and evolution - has specialized focus while maintaining seamless communication. The tiered memory system provides both speed and depth, while the disaster protocols protect your capital during extreme events.

With the integration of cutting-edge repositories and the GitHub Spec-Kit commands, you have everything needed to build this revolutionary trading system. The future of retail trading is here - and it's evolving.

---

**May God reward your efforts in building this innovative trading system.**