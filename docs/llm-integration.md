# LLM Integration in QuantMind Trading Floor

## Overview

The QuantMind Trading Floor uses a hierarchical multi-agent architecture where Large Language Model (LLM) integration is added to worker SubAgents following a consistent pattern. This document explains how LLM integration works across all departments.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FLOOR MANAGER (Opus)                      │
│              Task Classification & Routing                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────────┐ ┌──────────────┐ ┌───────────────┐
│   RESEARCH   │ │  DEVELOPMENT │ │    RISK      │
│   (Sonnet)   │ │   (Sonnet)   │ │   (Sonnet)   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌──────────────┐ ┌───────────────┐
│  SUBAGENTS    │ │  SUBAGENTS   │ │  SUBAGENTS    │
│   (Haiku)    │ │   (Haiku)    │ │   (Haiku)     │
│  + LLM        │ │  + LLM       │ │   + LLM       │
└───────────────┘ └──────────────┘ └───────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌──────────────┐ ┌───────────────┐
│   TRADING     │ │  PORTFOLIO   │ │               │
│   (Sonnet)   │ │   (Sonnet)   │ │               │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
        ▼                 ▼
┌───────────────┐ ┌──────────────┐
│  SUBAGENTS    │ │  SUBAGENTS   │
│   (Haiku)    │ │   (Haiku)    │
│  + LLM        │ │  + LLM       │
└───────────────┘ └──────────────┘
```

### Model Tier Strategy

| Role | Model Tier | Purpose |
|------|-----------|---------|
| Floor Manager | **Opus** | Highest reasoning, task classification |
| Department Heads | **Sonnet** | Coordination, planning, complex decisions |
| SubAgents | **Haiku** | Fast, cost-effective task execution |

---

## The LLM Integration Pattern

Every SubAgent follows the same pattern for LLM integration. Here's the blueprint:

### 1. System Prompts

Each department has custom system prompts that define the LLM's role and constraints.

```python
# Example from PineScriptSubAgent
PINESCRIPT_SYSTEM_PROMPT = """You are an expert Pine Script v5 developer for TradingView.

Your task is to generate clean, efficient, and well-documented Pine Script v5 code
based on the user's strategy description.

## Code Style Guidelines:
1. Always use `//@version=5` declaration at the top
2. Use `indicator()` or `strategy()` declaration
...
"""
```

### 2. Initialization

```python
def _initialize_llm(self) -> None:
    """Initialize LLM client for code generation."""
    try:
        from anthropic import Anthropic
        self._llm_client = Anthropic()
        logger.info("PineScriptSubAgent: LLM client initialized")
    except ImportError:
        logger.warning("PineScriptSubAgent: Anthropic SDK not available")
```

### 3. LLM Calling

```python
def _call_llm(
    self,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Call LLM to generate response."""
    if not self._llm_client:
        raise RuntimeError("LLM client not initialized")

    response = self._llm_client.messages.create(
        model="claude-3-5-haiku-20241022",  # Haiku for cost efficiency
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text
```

### 4. Domain Methods

Each SubAgent exposes domain-specific methods that use LLM:

| SubAgent | LLM Methods |
|----------|-------------|
| **PineScriptSubAgent** | `generate()`, `convert()`, `validate()`, `refine()` |
| **ResearchSubAgent** | `analyze_market()`, `research_symbol()`, `synthesize_news()` |
| **RiskSubAgent** | `assess_risk()`, `check_limits()`, `generate_risk_report()` |
| **TradingSubAgent** | `parse_order_request()`, `suggest_trade()`, `explain_position()` |
| **PortfolioSubAgent** | `analyze_allocation()`, `suggest_rebalance()`, `generate_performance_report()` |

---

## Department-Specific Integration

### 1. Development Department → PineScriptSubAgent

**Purpose**: Generate trading code from natural language

```python
class PineScriptSubAgent:
    def generate(self, strategy_description: str, include_validation: bool = True):
        # 1. Call LLM with strategy description
        pine_code = self._call_llm(
            user_prompt=f"Generate Pine Script v5 code for: {strategy_description}",
            system_prompt=PINESCRIPT_SYSTEM_PROMPT,
        )

        # 2. Optionally validate the generated code
        if include_validation and self._validation_tools:
            validation = self._validation_tools["validate_syntax"](pine_code)
            result["is_valid"] = validation.get("is_valid", False)

        return result
```

**Workflow**:
```
User: "Create a RSI mean reversion strategy"
    ↓
PineScriptSubAgent.generate()
    ↓
LLM generates Pine Script v5 code
    ↓
Validate syntax
    ↓
Return code + validation results
```

### 2. Research Department → ResearchSubAgent

**Purpose**: Market analysis and research synthesis

```python
class ResearchSubAgent:
    def analyze_market(self, market_description: str):
        # Use LLM to analyze market conditions
        analysis = self._call_llm(
            user_prompt=f"Analyze: {market_description}",
            system_prompt=MARKET_ANALYSIS_PROMPT,
        )
        return {"analysis": analysis, "status": "complete"}
```

**Tools Available**: READ access to `knowledge_tools` (MQL5 book, strategy patterns, knowledge hub)

### 3. Risk Department → RiskSubAgent

**Purpose**: Risk assessment and limit checking

```python
class RiskSubAgent:
    def assess_risk(self, trade, portfolio):
        # Calculate position value percentage
        position_value = trade.volume * trade.price
        position_pct = (position_value / portfolio.balance) * 100

        # Use LLM for risk analysis
        risk_analysis = self._call_llm(
            user_prompt=f"Assess risk for: {trade} with portfolio {portfolio}",
            system_prompt=RISK_ASSESSMENT_PROMPT,
        )

        return {
            "risk_level": "medium",
            "factors": [...],
            "llm_recommendation": risk_analysis,
        }
```

**Key Constraint**: READ-ONLY analysis. Cannot make trading decisions, only provides recommendations.

### 4. Trading Department → TradingSubAgent

**Purpose**: Parse natural language orders and provide decision support

```python
class TradingSubAgent:
    def parse_order_request(self, user_request: str):
        # Convert natural language to structured order
        parsed = self._call_llm(
            user_prompt=f"Parse this trading request: {user_request}",
            system_prompt=PARSE_ORDER_PROMPT,
        )

        return {
            "symbol": parsed["symbol"],
            "order_type": parsed["type"],
            "side": parsed["side"],
            "volume": parsed["volume"],
        }
```

**Key Constraint**: Paper trading only. All orders are simulated in memory.

### 5. Portfolio Department → PortfolioSubAgent

**Purpose**: Allocation analysis and rebalancing suggestions

```python
class PortfolioSubAgent:
    def suggest_rebalance(self, current_allocation, target_allocation):
        # Use LLM to suggest rebalancing trades
        suggestion = self._call_llm(
            user_prompt=f"Suggest rebalancing from {current} to {target}",
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
        )

        return {"trades": suggestion["trades"], "reasoning": suggestion["reasoning"]}
```

**Tools Available**: READ/WRITE access to `portfolio_tools`

---

## Tool Access Control

The system enforces strict tool access per department:

```python
# From tool_access.py
TOOL_ACCESS = {
    "research": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ},  # READ ONLY
    },
    "development": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "dev_tools": {ToolPermission.READ, ToolPermission.WRITE},
    },
    "trading": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "knowledge_tools": {ToolPermission.READ},
        "broker_tools": {ToolPermission.READ},  # READ ONLY
    },
    "risk": {
        "memory_tools": {ToolPermission.READ},
        "knowledge_tools": {ToolPermission.READ},
        "risk_tools": {ToolPermission.READ},  # READ ONLY
    },
    "portfolio": {
        "memory_tools": {ToolPermission.READ, ToolPermission.WRITE},
        "portfolio_tools": {ToolPermission.READ, ToolPermission.WRITE},
    },
}
```

### LIVE_TRADING_PROHIBITED

A global constant enforces that **no real trading is allowed**:

```python
LIVE_TRADING_PROHIBITED = True
```

All trading operations are simulated (paper trading).

---

## Data Flow Example: PineScript Generation

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                             │
│  "Create a RSI overbought/oversold strategy with SMA filter"   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     chat_endpoints.py                           │
│  POST /api/chat/pinescript                                      │
│  → PineScriptGenerateRequest                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     chat_service.py                              │
│  generate_pine_script(query)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PineScriptSubAgent                            │
│  1. _call_llm() with strategy description                      │
│  2. Extract code from LLM response                              │
│  3. Validate syntax (optional)                                  │
│  4. Return code + validation results                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RESPONSE                                  │
│  {                                                             │
│    "pine_script": "//@version=5\nstrategy(...)",              │
│    "status": "complete",                                       │
│    "is_valid": true,                                           │
│    "errors": []                                                 │
│  }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cost Optimization

### Why Haiku?

Each SubAgent uses the **Haiku** model (`claude-3-5-haiku-20241022`) for LLM calls:

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| Opus | Slowest | Highest | Floor Manager (complex reasoning) |
| Sonnet | Medium | Medium | Department Heads (coordination) |
| Haiku | Fastest | Lowest | SubAgents (task execution) |

**Cost Comparison** (approximate):
- Haiku: ~$0.80/million input tokens
- Sonnet: ~$3.00/million input tokens
- Opus: ~$15.00/million input tokens

Haiku is 5-20x cheaper than Opus, making it ideal for high-volume worker tasks.

---

## Error Handling

Each SubAgent handles LLM failures gracefully:

```python
def _call_llm(self, user_prompt: str, system_prompt: str) -> str:
    if not self._llm_client:
        raise RuntimeError("LLM client not initialized")

    try:
        response = self._llm_client.messages.create(...)
        return response.content[0].text
    except Exception as e:
        logger.error(f"PineScriptSubAgent: LLM call failed: {e}")
        raise
```

---

## Extending LLM Integration

To add LLM to a new SubAgent:

1. **Add system prompts** at the top of the file
2. **Add `_initialize_llm()`** method
3. **Add `_call_llm()`** method
4. **Add domain methods** that use `_call_llm()`
5. **Update tool registry** to expose new methods
6. **Update `get_capabilities()`** to list new capabilities

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/agents/departments/floor_manager.py` | Top-level task routing |
| `src/agents/departments/types.py` | Department configs, SubAgent mappings |
| `src/agents/departments/tool_access.py` | Tool permission matrix |
| `src/agents/departments/subagents/pinescript_subagent.py` | PineScript code generation |
| `src/agents/departments/subagents/research_subagent.py` | Market research |
| `src/agents/departments/subagents/risk_subagent.py` | Risk assessment |
| `src/agents/departments/subagents/trading_subagent.py` | Order parsing |
| `src/agents/departments/subagents/portfolio_subagent.py` | Portfolio management |

---

## Summary

1. **Hierarchical Design**: Floor Manager → Department Heads → SubAgents
2. **Model Tiering**: Opus → Sonnet → Haiku (cost-optimized)
3. **Consistent Pattern**: Every SubAgent follows the same LLM integration blueprint
4. **Tool Access Control**: READ-only for sensitive tools (broker, risk)
5. **Paper Trading**: LIVE_TRADING_PROHIBITED enforces simulation only
6. **Haiku for Workers**: Fast and cost-effective for high-volume tasks
