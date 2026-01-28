# QuantMindX: Knowledge Bases & IDE Architecture

> **Purpose:** Complete specification of knowledge base structure, guild assignments, and IDE design  
> **Last Updated:** 2026-01-26

---

## Table of Contents

1. [Knowledge Base Overview](#1-knowledge-base-overview)
2. [Tiered Memory Architecture](#2-tiered-memory-architecture)
3. [Guild-Specific Knowledge Bases](#3-guild-specific-knowledge-bases)
4. [Article Classification & Sorting](#4-article-classification--sorting)
5. [QuantMind IDE Specification](#5-quantmind-ide-specification)
6. [Data Flow Between Components](#6-data-flow-between-components)

---

## 1. Knowledge Base Overview

### The Three-Tier Knowledge Architecture

QuantMindX uses specialized knowledge bases for different purposes:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE TIERS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: COPILOT KB (For AI Assistant - You're Building This)  │
│  ├─ All scraped MQL5 articles (~1,800)                          │
│  ├─ Vector embeddings in Qdrant                                  │
│  ├─ Purpose: Answer questions about MQL5, indicators, EAs       │
│  └─ Access: AI assistant direct query                            │
│                                                                  │
│  TIER 2: GUILD-SPECIFIC KBs (For Future Agents)                 │
│  ├─ Research Guild KB: Strategy sources, video transcripts      │
│  ├─ Engineering Guild KB: Code templates, backtest results      │
│  ├─ Operations Guild KB: Trade journals, risk patterns          │
│  └─ Evolution Guild KB: Bot DNA, mutation history               │
│                                                                  │
│  TIER 3: RUNTIME KBs (For Live Trading)                         │
│  ├─ Working Memory (Redis): Active positions, live P&L          │
│  ├─ Short-Term (PostgreSQL): 30-day trade history               │
│  └─ Long-Term (ChromaDB): All historical data, patterns         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Current State (What's Built)

| KB Component | Status | Location | Contents |
|--------------|--------|----------|----------|
| Copilot KB (Qdrant) | 🔄 Indexing | `data/qdrant_db/` | 1,806 articles |
| Document Index | ✅ Done | `data/knowledge_index/` | JSON index + summaries |
| Scraped Articles | ✅ Done | `data/scraped_articles/` | ~1,565 markdown files |
| Guild KBs | ❌ Not started | - | Future |
| Runtime KBs | ❌ Not started | - | Future |

---

## 2. Tiered Memory Architecture

### From Original TRD Design

The original design specifies 4 memory tiers:

---

### Tier 1: Working Memory (Redis)

**Purpose:** Instant access for current operations  
**TTL:** 1 hour  
**Access Time:** <5ms

**Contents:**
- Active bot states (50 bots × 10KB = 500KB)
- Open positions with real-time P&L
- Last 100 messages per guild
- Current market conditions (price, spread, volatility)

**Example Data:**
```json
{
  "bot_007": {
    "status": "active",
    "current_position": {
      "pair": "EURUSD",
      "direction": "LONG",
      "entry": 1.08567,
      "current_pnl": "+$4.23",
      "sl": 1.08423,
      "tp": 1.08732
    },
    "today_stats": {
      "trades": 3,
      "wins": 2,
      "pnl": "+$12.45"
    }
  }
}
```

---

### Tier 2: Short-Term Memory (PostgreSQL)

**Purpose:** Recent history for pattern detection  
**Retention:** 30 days  
**Access Time:** <50ms

**Contents:**
- Last 30 days of trades
- Bot performance metrics (daily snapshots)
- Weekly evolution reports
- Guild-to-guild conversation logs

**Tables:**
```sql
-- Trade Journal
trades (trade_id, bot_id, pair, direction, entry, exit, pnl, timestamp)

-- Bot Performance
bot_daily_metrics (bot_id, date, win_rate, sharpe, drawdown, pnl)

-- Evolution Log
evolution_events (event_id, bot_id, mutation_type, before_dna, after_dna, timestamp)
```

---

### Tier 3: Long-Term Memory (ChromaDB / Qdrant)

**Purpose:** Semantic search across all historical data  
**Retention:** Permanent  
**Access Time:** <500ms (semantic search)

**Collections:**

| Collection | Contents | Embedding Model |
|------------|----------|-----------------|
| `mql5_knowledge` | All scraped articles | all-MiniLM-L6-v2 |
| `strategy_dna` | Bot genetic profiles | all-MiniLM-L6-v2 |
| `pattern_library` | Discovered patterns | all-MiniLM-L6-v2 |
| `graveyard` | Failed strategies + reasons | all-MiniLM-L6-v2 |
| `video_transcripts` | NPRD extracted content | all-MiniLM-L6-v2 |

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

### Tier 4: Archive (S3/Cold Storage)

**Purpose:** Compliance and long-term analysis  
**Retention:** 1+ years  
**Access Time:** 1-5 seconds

**Contents:**
- Old backtests (2+ years)
- Ancient logs
- Tax records
- Raw video files

**Note:** This tier was marked as "unnecessary for personal use" in our Jan 19 discussion and may be skipped for MVP.

---

## 3. Guild-Specific Knowledge Bases

### Guild Structure Recap

```
┌───────────────────────────────────────────────────────────────┐
│                         GUILDS                                 │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  RESEARCH GUILD ("The Knowledge Hunters")                     │
│  └─ Discovers strategies from articles, videos, internet       │
│                                                                │
│  ENGINEERING GUILD ("The Bot Builders")                       │
│  └─ Converts strategies to code, backtests, validates          │
│                                                                │
│  OPERATIONS GUILD ("The Money Managers")                      │
│  └─ Executes trades, manages risk, monitors performance        │
│                                                                │
│  EVOLUTION GUILD ("The Strategy Improvers")                   │
│  └─ Improves bots via genetic algorithms, pattern recognition │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

---

### Research Guild KB

**Purpose:** Store all discovered trading knowledge

**Sources:**
- MQL5 articles (✅ Already scraped: 1,800+)
- YouTube videos (via NPRD tool)
- PDFs and ebooks
- GitHub repositories
- Reddit/TradingView posts

**KB Structure:**
```
research_guild_kb/
├── strategies/           # Complete trading systems
│   ├── scalping/
│   ├── swing/
│   └── harmonic_patterns/
├── indicators/           # Indicator explanations
│   ├── oscillators/
│   ├── trend/
│   └── volume/
├── code_examples/        # MQL5 code snippets
├── video_transcripts/    # NPRD outputs
└── external_sources/     # Reddit, Twitter, blogs
```

**Who Uses It:**
- Research Guild agents for strategy discovery
- AI Copilot for answering MQL5 questions
- Engineering Guild for reference during coding

---

### Engineering Guild KB

**Purpose:** Store code templates, backtest results, technical patterns

**Contents:**
- Bot code templates (entry logic, exit logic, filters)
- Backtest results database
- Code pattern library
- Error/bug history

**KB Structure:**
```
engineering_guild_kb/
├── templates/
│   ├── entry_logic/
│   │   ├── rsi_oversold.mq5
│   │   ├── ma_crossover.mq5
│   │   └── support_bounce.mq5
│   ├── exit_logic/
│   │   ├── fixed_tp_sl.mq5
│   │   ├── trailing_stop.mq5
│   │   └── time_exit.mq5
│   └── filters/
│       ├── session_filter.mq5
│       ├── atr_filter.mq5
│       └── volume_filter.mq5
├── backtests/
│   ├── results.db           # SQLite with all results
│   └── reports/             # HTML/PDF reports
└── patterns/
    ├── good_patterns.json   # What works
    └── bad_patterns.json    # What to avoid
```

**Who Uses It:**
- Engineering agents for code generation
- Evolution Guild for understanding what works
- Code Validator for checking against known issues

---

### Operations Guild KB

**Purpose:** Store execution intelligence, risk patterns, trade journals

**Contents:**
- Trade journal (every trade ever taken)
- Risk events (drawdowns, circuit breakers triggered)
- Execution quality data (slippage, latency)
- Broker behavior patterns

**KB Structure:**
```
operations_guild_kb/
├── journals/
│   ├── trades.db            # All trade records
│   └── daily_summaries/     # Daily P&L reports
├── risk_events/
│   ├── drawdown_log.json
│   └── circuit_breaker_log.json
├── execution/
│   ├── slippage_analysis.json
│   └── broker_latency.json
└── patterns/
    ├── bad_times.json       # "Don't trade at 00:15 GMT"
    └── pair_correlations.json
```

**Who Uses It:**
- Operations agents for real-time decisions
- Risk Governor for limit enforcement
- Evolution Guild for understanding what failed

---

### Evolution Guild KB

**Purpose:** Store bot genetics, mutation history, evolution patterns

**Contents:**
- Bot DNA registry (all genes ever used)
- Mutation history (what was tried, what worked)
- Cross-breeding results
- Pattern library (what gene combinations work)

**KB Structure:**
```
evolution_guild_kb/
├── dna_registry/
│   ├── entry_genes.json
│   ├── exit_genes.json
│   ├── filter_genes.json
│   └── risk_genes.json
├── mutations/
│   ├── successful_mutations.json
│   └── failed_mutations.json
├── breeding/
│   ├── parent_child_map.json
│   └── best_combinations.json
└── graveyard/
    ├── dead_bots.json       # What died and why
    └── lessons_learned.json
```

**Who Uses It:**
- Evolution agents for Saturday evolution cycle
- Engineering Guild for understanding what to build
- Research Guild for avoiding known-bad patterns

---

## 4. Article Classification & Sorting

### Current Scraped Article Categories

From our document index (1,499 articles), the breakdown is:

| Category Combination | Count | Primary Guild |
|---------------------|-------|---------------|
| Trading Systems | 185 | Engineering |
| Trading, Trading Systems, Expert Advisors | 162 | Engineering |
| Trading Systems, Expert Advisors | 172 | Engineering |
| Trading Systems, Expert Advisors, Machine Learning | 112 | Engineering + Research |
| Integration | 117 | Engineering |
| Integration, Machine Learning | 31 | Research |
| Expert Advisors, Machine Learning | 18 | Research |
| Trading | 41 | Research |

### Classification Schema

Each article is classified as:

| Classification | Count | Description | Primary Consumer |
|----------------|-------|-------------|------------------|
| `ml_strategy` | 1,058 | Machine learning & trading systems | Engineering Guild |
| `indicator_usage` | 414 | How to use indicators | Research Guild + Copilot |
| `indicator_development` | 27 | Creating custom indicators | Engineering Guild |

### Recommended Sorting for QuantMind

**Re-categorize MQL5 categories → QuantMind KB structure:**

| MQL5 Category | → QuantMind Category | Purpose | Guild |
|---------------|---------------------|---------|-------|
| `trading_systems` | `strategies/` | Complete trading systems | Engineering |
| `expert_advisors` | `code_examples/` | MQL5 code reference | Engineering |
| `trading` | `concepts/` | Theory, market mechanics | Research |
| `integration` | `integration/` | Python, APIs, external tools | Engineering |
| `indicators` | `indicators/` | Indicator explanations | Research + Copilot |
| `machine_learning` | `ml/` | ML/AI trading approaches | Research |

---

## 5. QuantMind IDE Specification

### Overview

The QuantMind IDE is a visual interface that ties together all QuantMindX components. Think of it as **VS Code for Trading Bots**.

### Core Features

---

#### 5.1 Dashboard View

**Purpose:** Real-time overview of system status

**Components:**
```
┌─────────────────────────────────────────────────────────────────┐
│  QUANTMIND IDE - Dashboard                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Active Bots  │  │ Today's P&L  │  │ Open Trades  │           │
│  │     42       │  │   +$127.45   │  │      7       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LIVE FEED                                                │   │
│  │  14:30:22 [P2] Bot_007 opened LONG EURUSD @ 1.08567      │   │
│  │  14:31:45 [P3] Evolution proposes Bot_007_v2             │   │
│  │  14:32:30 [P1] Risk: Daily limit 78% reached             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │  BOT PERFORMANCE        │  │  APPROVAL QUEUE             │   │
│  │  @perfect: 12           │  │  3 bots awaiting review     │   │
│  │  @pending: 23           │  │  [View Queue]               │   │
│  │  @quarantine: 5         │  │                             │   │
│  │  @primal: 8             │  │                             │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 5.2 Bot Manager View

**Purpose:** Manage individual bots and their lifecycle

**Features:**
- List all bots with status tags
- View bot DNA (entry/exit/filter genes)
- View performance metrics
- Manual tag assignment
- Force kill or restart

**Bot Card Example:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Bot_007_RSI_Scalper_v3                           [@perfect]    │
├─────────────────────────────────────────────────────────────────┤
│  Win Rate: 71%  │  Sharpe: 1.42  │  Max DD: 8.3%  │  Trades: 234│
│                                                                  │
│  DNA: RSI_oversold + support_level + london_avoid               │
│  Parents: Bot_007_v2 × Bot_023_v1                               │
│  Last Trade: EURUSD LONG +$4.23 (2 hours ago)                   │
│                                                                  │
│  [View Details] [Pause] [Kill] [Clone] [Evolve]                 │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 5.3 Strategy Editor View

**Purpose:** Create and edit trading strategies

**Features:**
- Visual strategy builder (drag-drop indicators, conditions)
- MQL5 code editor with syntax highlighting
- Integrated backtester
- KB search for similar strategies

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  STRATEGY EDITOR - New RSI Strategy                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────┐  ┌───────────────────────────────┐   │
│  │  VISUAL BUILDER       │  │  MQL5 CODE                    │   │
│  │                       │  │                               │   │
│  │  [Entry Condition]    │  │  void OnTick() {              │   │
│  │   └─ RSI < 30        │  │    double rsi = iRSI(...);    │   │
│  │   └─ Support Touch   │  │    if (rsi < 30) {            │   │
│  │                       │  │      // Entry logic           │   │
│  │  [Exit Condition]     │  │    }                          │   │
│  │   └─ TP: 1.5%        │  │  }                             │   │
│  │   └─ SL: 1.0%        │  │                               │   │
│  │                       │  │                               │   │
│  └───────────────────────┘  └───────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  KB SEARCH: Found 12 similar strategies in knowledge base │   │
│  │  - RSI Oversold Scalper (67% WR, Sharpe 1.2)             │   │
│  │  - Double Bottom RSI (58% WR, Sharpe 0.9)                │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [Save Draft] [Run Backtest] [Deploy to Paper Trading]          │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 5.4 Backtest Results View

**Purpose:** Analyze backtest results

**Metrics Displayed:**
- Win Rate, Profit Factor, Sharpe Ratio
- Max Drawdown, Recovery Factor
- Trade distribution (by hour, day, pair)
- Equity curve chart
- Trade-by-trade list

---

#### 5.5 Trade Journal View

**Purpose:** Review all trades with context

**Features:**
- Filterable trade list
- Trade detail with entry/exit reasoning
- Chart replay at trade time
- Performance attribution

---

#### 5.6 Knowledge Base Browser

**Purpose:** Search and browse the KB

**Features:**
- Semantic search across all articles
- Category filters
- Article preview
- "Similar articles" suggestions
- "Use this strategy" quick action

---

#### 5.7 Evolution Center

**Purpose:** Manage Saturday evolution cycle

**Features:**
- View proposed mutations
- Compare parent vs child DNA
- Approve/reject evolutions
- View breeding history
- Graveyard browser (dead bots)

---

### IDE Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Frontend | React + TypeScript | Modern, fast |
| Desktop Wrapper | Tauri (Rust) | Lightweight alternative to Electron |
| Backend API | FastAPI (Python) | Already using for guilds |
| Real-time | WebSocket | Live feed updates |
| Charts | TradingView Lightweight Charts | or Recharts |
| Code Editor | Monaco Editor | VS Code's editor |

---

### IDE Development Phases

| Phase | Features | Priority |
|-------|----------|----------|
| Phase 1 (MVP) | Dashboard, Bot Manager, Trade Journal | HIGH |
| Phase 2 | Strategy Editor, Backtest View | HIGH |
| Phase 3 | KB Browser, Evolution Center | MEDIUM |
| Phase 4 | Mobile companion app | LOW |

---

## 6. Data Flow Between Components

### Knowledge Ingestion Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ MQL5 Site   │────►│ Firecrawl    │────►│ Scraped         │
│ Articles    │     │ Scraper      │     │ Articles (.md)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
┌─────────────┐     ┌──────────────┐              ▼
│ YouTube     │────►│ NPRD Tool    │────►┌─────────────────┐
│ Videos      │     │ (Gemini)     │     │ Document Index  │
└─────────────┘     └──────────────┘     │ Generator       │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Qdrant Indexer  │
                                          │ (Embeddings)    │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ Qdrant Vector   │
                                          │ Database        │
                                          └────────┬────────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                    │ AI Copilot     │  │ Research Guild  │  │ Engineering     │
                    │ (Direct Query) │  │ (Strategy Find) │  │ Guild (Coding)  │
                    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Trading Execution Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Strategy    │────►│ Engineering  │────►│ Backtest        │
│ Idea        │     │ Guild        │     │ Results         │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                                                  ▼ (if passes)
                                          ┌─────────────────┐
                                          │ Human Approval  │
                                          │ (IDE Queue)     │
                                          └────────┬────────┘
                                                   │
                                                   ▼ (if approved)
                                          ┌─────────────────┐
                                          │ Operations      │
                                          │ Guild           │
                                          └────────┬────────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                    │ Paper Trading  │  │ Risk Governor   │  │ Trade Journal   │
                    │ (Demo Account) │  │ (Limits)        │  │ (PostgreSQL)    │
                    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Summary

This document specifies:

1. **Knowledge Base Tiers:** Working (Redis) → Short-term (PostgreSQL) → Long-term (Qdrant/ChromaDB) → Archive (S3)

2. **Guild KBs:** Each guild has specialized knowledge storage with defined contents and consumers

3. **Article Sorting:** 1,800+ articles classified into `ml_strategy`, `indicator_usage`, `indicator_development` and mapped to guilds

4. **QuantMind IDE:** VS Code-like interface with Dashboard, Bot Manager, Strategy Editor, Backtest View, Journal, KB Browser, and Evolution Center

5. **Data Flow:** Clear pipelines from ingestion → indexing → consumption by agents and IDE

---

**Status:** This is the architectural specification. Implementation order:
1. ✅ Copilot KB (Qdrant) - In progress
2. ⏳ MCP Server for KB access  
3. ⏳ IDE Phase 1 (Dashboard, Bot Manager)
4. ⏳ Guild-specific KBs
5. ⏳ Full IDE
