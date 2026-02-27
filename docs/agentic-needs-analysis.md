# QuantMindX Agentic System Analysis

**Analysis Date:** 2025-02-25
**Analyst:** Claude Code Research Agent

---

## System Overview

QuantMindX is an autonomous trading intelligence system that combines multiple AI agents with a sophisticated strategy router to manage algorithmic trading operations. The system is built around:

1. **Multi-Agent Architecture** - Specialized agents for different tasks (Analyst, QuantCode, Copilot, Executor, Router)
2. **Strategy Router** - Central execution engine that routes trading signals to appropriate bots/accounts
3. **Regime Detection** - HMM and Ising models for market state detection
4. **Position Sizing** - Enhanced Kelly Criterion with multi-layer risk protection
5. **Paper Trading Pipeline** - Docker-based paper trading validation before live deployment
6. **Kill Switch & Circuit Breakers** - Multi-level safety mechanisms

The system follows a **Paper -> Live** promotion workflow where strategies must prove themselves in simulated trading before being deployed to live accounts.

---

## Current Agent Capabilities

### 1. Analyst Agent (`src/agents/analyst_v2.py`)

**Purpose:** Market research, NPRD parsing, and TRD generation

**Capabilities:**
- `research_market_data()` - Gather market data from multiple sources
- `extract_insights()` - Extract key insights from research data
- `parse_nprd()` - Parse Natural Language Product Requirements Documents
- `validate_nprd()` - Validate NPRD completeness and quality
- `generate_trd()` - Generate Technical Requirements Documents
- `analyze_backtest()` - Analyze backtest results and calculate metrics
- `compare_strategies()` - Compare multiple strategies
- `generate_optimization_report()` - Generate optimization recommendations

**MCP Tools Available:**
- Filesystem access for NPRD/TRD files
- GitHub integration for research
- Context7 for documentation lookup
- Brave Search for market research
- Memory for persistent findings
- Sequential Thinking for task decomposition
- PageIndex Articles for trading articles

---

### 2. QuantCode Agent (`src/agents/quantcode_v2.py`)

**Purpose:** Strategy development, MQL5 code generation, and backtesting

**Capabilities:**
- `create_strategy_plan()` - Create strategy development plan
- `generate_mql5_code()` - Generate MQL5 Expert Advisor code
- `generate_component()` - Generate specific MQL5 components (signal generator, order manager, risk manager)
- `validate_syntax()` - Validate MQL5 syntax
- `fix_syntax()` - Fix MQL5 syntax errors
- `compile_mql5()` - Compile MQL5 code
- `debug_code()` - Debug MQL5 code
- `optimize_code()` - Optimize code performance
- `run_backtest()` - Run strategy backtest
- `analyze_backtest_results()` - Analyze backtest results
- `run_monte_carlo()` - Run Monte Carlo simulation
- `deploy_paper_trading()` - Deploy to Docker-based paper trading
- `check_paper_trading_status()` - Check paper trading status
- `get_paper_trading_performance()` - Get performance metrics
- `promote_to_live_trading()` - Promote validated agent to live trading
- `generate_documentation()` - Generate code documentation

**MCP Tools Available:**
- Filesystem access for EA code
- GitHub integration for EA sync
- Context7 for MQL5 documentation
- MT5 Compiler for code compilation
- Backtest Server for strategy testing

---

### 3. Copilot Agent (`src/agents/copilot_v2.py`)

**Purpose:** Master orchestrator for agent handoffs and global task queuing

**Capabilities:**
- Agent coordination and task routing
- Deployment orchestration
- Agent handoff management
- Global task queue management

**MCP Tools Available:**
- Filesystem access for project management
- GitHub integration
- Context7 for documentation
- Sequential Thinking for task decomposition
- PageIndex Articles for trading knowledge

---

### 4. Executor Agent (`src/agents/executor.py`)

**Purpose:** Trade execution and position management

**Capabilities:**
- `validate_proposal()` - Validate trade proposals against risk parameters
- `execute_trade()` - Execute trades via broker connection
- `monitor_position()` - Monitor open positions
- `close_position()` - Close positions based on exit conditions

**MCP Tools Available:**
- Minimal (primarily bash tools for order execution)

---

### 5. Router Agent

**Purpose:** Strategy routing and bot management

**Capabilities:**
- Bot manifest management (BotRegistry)
- Circuit breaker enforcement
- Promotion/demotion workflow
- Multi-timeframe sentinel coordination

**MCP Tools Available:**
- Minimal (primarily bash tools for internal state management)

---

## Trading Operations Available

### API Endpoints (from `src/api/server.py`)

| Category | Endpoints | Purpose |
|----------|-----------|---------|
| **HMM** | `/api/hmm/*` | Regime detection, mode management, model sync |
| **Agent Queue** | `/api/agents/*` | Task queue management, status, cancellation |
| **Router** | `/api/router/*` | Bot management, routing, deployment |
| **Analytics** | `/api/analytics/*` | Performance analytics, metrics |
| **Kill Switch** | `/api/kill-switch/*` | Emergency trading halt |
| **Paper Trading** | `/api/paper-trading/*` | Paper trading deployment and monitoring |
| **GitHub EA** | `/api/eas/*` | EA sync with GitHub repositories |
| **Broker** | `/api/brokers/*` | Broker connection management |
| **Metrics** | `/api/metrics/*` | Prometheus metrics exposure |
| **Agent Tools** | `/api/agent-tools/*` | Tool discovery and execution |

### Strategy Router Operations (`src/router/`)

The Strategy Router (`src/router/engine.py`) provides:

1. **Bot Routing** - Routes signals to appropriate bots based on:
   - Strategy type (SCALPER, STRUCTURAL, SWING, HFT)
   - Account type (machine_gun, sniper, prop_firm)
   - Session preferences (LONDON, NEW_YORK, OVERLAP)
   - Time windows (ICT-style filters)

2. **Circuit Breaker** - Bot-level performance tracking:
   - 3 consecutive losses = quarantine
   - Daily trade limit (default 20)
   - Fee kill switch integration

3. **Kill Switch** - System-wide trading halt

4. **Multi-Timeframe Sentinel** - MTF signal confirmation

5. **HMM Deployment Manager** - Regime-based mode transitions:
   - ISING_ONLY (0% HMM)
   - HMM_SHADOW (shadow validation mode)
   - HMM_HYBRID_20/50/80 (mixed modes)
   - HMM_ONLY (100% HMM)

---

## Data Access

### Read Operations

1. **Bot Manifests** - BotRegistry (`data/bot_registry.json`)
   - Bot configuration, performance stats, promotion status

2. **HMM Models** - Regime detection models
   - Current regime, confidence, state transitions

3. **Backtest Results** - Strategy performance data
   - Trade history, metrics, optimization results

4. **Market Data** - Real-time and historical
   - Via MT5 integration or data providers

5. **Circuit Breaker State** - Database-backed
   - Quarantine status, consecutive losses, daily trade counts

6. **Agent Queue State** - Persistent queue storage
   - Task status, results, retry counts

### Write Operations

1. **Bot Deployment** - Deploy/stop bots
2. **Trade Execution** - Open/close positions
3. **Configuration** - Update bot manifests, settings
4. **HMM Sync** - Transfer models between servers
5. **Mode Changes** - Switch HMM deployment modes
6. **Quarantine Management** - Quarantine/reactivate bots

---

## MCP Integrations

### Common MCP Servers (shared across agents)

| MCP Server | Purpose | Used By |
|------------|---------|---------|
| `filesystem` | File system access | All agents |
| `github` | GitHub integration | All agents |
| `context7` | Documentation lookup | All agents |

### Agent-Specific MCP Servers

**Analyst:**
- `brave-search` - Web search for market research
- `memory` - Persistent research findings
- `sequential-thinking` - Task decomposition
- `pageindex-articles` - Trading articles search
- `backtest-server` - Strategy backtesting

**QuantCode:**
- `mt5-compiler` - MQL5 code compilation
- `backtest-server` - Strategy backtesting

**Copilot:**
- `sequential-thinking` - Task decomposition

**PineScript:**
- `context7` - Pine Script documentation
- `pageindex-books` - Trading books and patterns

---

## What an Agent SHOULD Be Able To Do

Based on the system's capabilities, a comprehensive agentic interface should provide:

### 1. Market Research & Analysis
- [ ] Fetch real-time market data for symbols
- [ ] Get current regime/market state from HMM
- [ ] Query historical price data and indicators
- [ ] Search trading articles and documentation
- [ ] Get broker conditions (spread, commission, swap)

### 2. Strategy Development
- [ ] Create/edit NPRD documents
- [ ] Generate TRD from NPRD
- [ ] Generate MQL5 code from TRD
- [ ] Compile and validate MQL5 code
- [ ] Run backtests with custom parameters
- [ ] Get backtest results and analysis

### 3. Bot Management
- [ ] List all bots and their status
- [ ] Get bot manifests and performance stats
- [ ] Create new bot manifests
- [ ] Update bot configuration
- [ ] Check promotion eligibility
- [ ] Promote/demote bots

### 4. Trading Operations
- [ ] Submit trade proposals
- [ ] Check if bot is allowed to trade (circuit breaker)
- [ ] Execute trades (via Executor)
- [ ] Monitor open positions
- [ ] Close positions
- [ ] Get current account balance and margin

### 5. Risk Management
- [ ] Calculate position size using Enhanced Kelly
- [ ] Check fee kill switch status
- [ ] Get broker fee conditions
- [ ] Calculate volatility-adjusted position size
- [ ] Get account tier and max bots

### 6. System Control
- [ ] Get system status (HMM, router, agents)
- [ ] Change HMM deployment mode
- [ ] Trigger model sync
- [ ] Activate kill switch
- [ ] Quarantine/reactivate bots
- [ ] Check agent queue status

---

## Gaps

### High Priority

1. **No Unified Agent Interface**
   - Each agent has separate MCP config
   - No common tool registry for agents to discover available tools
   - Agents cannot easily delegate tasks to each other

2. **Limited Real-Time Data Access**
   - No direct MCP for fetching current market data
   - Agent must go through API endpoints (not ideal for agentic workflows)

3. **No Agent-to-Agent Communication**
   - Agents are isolated
   - No direct messaging or handoff protocol
   - Copilot coordinates but other agents can't directly interact

4. **Missing Trading Operations for Agents**
   - No tool for "get current signal for symbol"
   - No tool for "check if bot should trade now"
   - No tool for "get current positions"

5. **No Agent Capability Discovery**
   - Agents don't know what other agents can do
   - No registry of agent skills/tools

### Medium Priority

1. **Limited State Querying**
   - No unified state query interface
   - Each component has separate API

2. **No Notification System**
   - Agents can't subscribe to events
   - Must poll for status changes

3. **Missing Documentation Tools**
   - No tool for generating strategy documentation
   - No tool for creating trading manuals

### Low Priority

1. **No Multi-Agent Collaboration**
   - Can't run parallel agents on same task
   - No consensus mechanisms

2. **Limited Logging/Debugging**
   - No agent activity log query tool
   - No debugging interface for agent workflows

---

## Recommendations

### 1. Create a Unified Agent MCP Server

Create `/config/mcp/unified-agent-mcp.json` with tools for:

```python
# Market Data
get_current_price(symbol)
get_market_regime(symbol, timeframe)
get_hmm_prediction(symbol)

# Bot Management
list_bots()
get_bot_status(bot_id)
create_bot_manifest(...)
update_bot_manifest(bot_id, ...)

# Trading Operations
calculate_position_size(...)
check_circuit_breaker(bot_id)
submit_trade_proposal(...)

# System Status
get_system_status()
get_hmm_status()
get_queue_status()
```

### 2. Implement Agent Handoff Protocol

Add tools for:
- `delegate_task(to_agent, task, context)`
- `get_agent_capabilities(agent_id)`
- `register_skill(skill_name, description)`

### 3. Add Real-Time Event Streaming

WebSocket-based event system where agents can:
- Subscribe to regime changes
- Subscribe to bot status changes
- Subscribe to trade execution events

### 4. Create Agent Discovery Service

```python
discover_agents() -> List[AgentInfo]
get_agent_tools(agent_id) -> List[ToolInfo]
call_agent_tool(agent_id, tool_name, args) -> Result
```

---

## Appendix: File Reference

| Component | Path |
|-----------|------|
| Agents | `/home/mubarkahimself/Desktop/QUANTMINDX/src/agents/` |
| Router | `/home/mubarkahimself/Desktop/QUANTMINDX/src/router/` |
| API | `/home/mubarkahimself/Desktop/QUANTMINDX/src/api/` |
| Position Sizing | `/home/mubarkahimself/Desktop/QUANTMINDX/src/position_sizing/` |
| MCP Configs | `/home/mubarkahimself/Desktop/QUANTMINDX/config/mcp/` |
