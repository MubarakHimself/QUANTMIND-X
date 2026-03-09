# Prop Firm Integration

This document covers the prop firm integration in QUANTMINDX, including risk management, research tools, and configuration.

## Overview

QUANTMINDX supports proprietary trading firm accounts through two main components:

1. **PropFirmRiskOverlay** - Risk management with firm-specific constraints
2. **PropFirmResearchTool** - Research and analysis of prop firm rules

## Supported Prop Firms

### Risk Overlay Firms

| Firm | Max Drawdown | Daily Loss | Profit Target |
|------|--------------|------------|---------------|
| FTMO | 3% | 4% | 10% |
| Topstep | 4% | 4% | 10% |
| FundedNext | 3% | 4% | 8% |
| FundingPips | 5% | 4% | 10% |

### Research Tool Firms

| Firm | Tiers Available |
|------|-----------------|
| FundedNext | starter, standard, pro, elite |
| TrueForex | starter, standard, pro |
| MyCryptoBuddy | starter, standard, pro |

## PropFirmRiskOverlay

The `PropFirmRiskOverlay` class applies prop firm-specific risk constraints to trading bots.

### Location

`/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/prop_firm_overlay.py`

### Key Features

- **Tighter drawdown limits** (3% vs 5% for personal accounts)
- **Daily loss limits** (4% for prop firms)
- **P_pass calculation** - Probability of passing the challenge
- **Recovery mode** - Automatic position reduction after drawdown breach

### Usage Example

```python
from src.risk.prop_firm_overlay import PropFirmRiskOverlay

# Initialize for FTMO account
overlay = PropFirmRiskOverlay(firm_name="FTMO", initial_balance=100000.0)

# Apply risk limits based on account book
limits = overlay.apply_risk_limits(
    account_book="prop_firm",
    max_drawdown=0.05,  # Personal limit
    daily_loss=0.03,
    profit_target=0.15
)
# Returns: {"effective_max_drawdown": 0.03, "effective_daily_loss": 0.04, ...}

# Calculate P_pass (probability of passing)
result = overlay.calculate_p_pass(
    win_rate=0.70,
    avg_win=100,
    avg_loss=50,
    account_book="prop_firm"
)
# Returns: {"p_pass": 0.85, "prop_score": 1.2, "book_type": "prop_firm", ...}

# Update balance and check recovery mode
overlay.update_balance(97000)
recovery_status = overlay.check_recovery_mode()
# Returns: {"in_recovery": False, "current_drawdown": 0.03, "drawdown_limit": 0.03}

# Get position size multiplier in recovery
multiplier = overlay.get_recovery_risk_multiplier()
```

### Recovery Mode

When drawdown exceeds the limit:
- Position sizes are reduced by 50%
- System monitors for recovery
- Exits recovery when drawdown drops below 2.5%

```python
# Check if can exit recovery
can_exit = overlay.can_exit_recovery()
# Returns: {"can_exit": True, "current_drawdown": 0.01, "required_drawdown": 0.025}

# Exit recovery mode
result = overlay.exit_recovery_mode()
```

## PropFirmResearch

The research tool analyzes prop firm rules and provides comparison capabilities.

### Location

`/home/mubarkahimself/Desktop/QUANTMINDX/src/agents/tools/prop_firm_research_tool.py`

### Usage Example

```python
from src.agents.tools.prop_firm_research_tool import PropFirmResearch

research = PropFirmResearch()

# Get list of supported firms
firms = research.get_firms()
# Returns: ["fundednext", "trueforex", "mycryptobuddy"]

# Get tiers for a firm
tiers = research.get_tiers("fundednext")
# Returns: ["starter", "standard", "pro", "elite"]

# Get rules for a specific firm/tier
rules = research.get_rules("fundednext", "standard")
# Returns: {
#     "firm": "fundednext",
#     "tier": "standard",
#     "initial_balance": 10000,
#     "max_drawdown_pct": 10.0,
#     "daily_drawdown_pct": 5.0,
#     "profit_target_pct": 8.0,
#     "ea_allowed": True,
#     "hedge_allowed": True,
#     ...
# }

# Analyze a firm (pros/cons/recommendations)
analysis = research.firm_analysis("fundednext", "standard")
# Returns: {
#     "firm": "fundednext",
#     "overall_score": 75,
#     "pros": ["Generous max drawdown", "EA allowed", ...],
#     "cons": [...],
#     "recommendations": [...],
#     "risk_assessment": "Medium Risk - Strict rules",
#     "suitability": "Best for beginners, diverse instruments"
# }

# Compare multiple firms
comparison = research.compare_firms(
    ["fundednext", "trueforex", "mycryptobuddy"],
    tier="standard"
)
# Returns sorted by score descending
```

### Tool Schemas

The research tool provides the following agent tool schemas:

```python
from src.agents.tools.prop_firm_research_tool import get_prop_firm_tool_schemas

schemas = get_prop_firm_tool_schemas()
# [
#     {"name": "get_firms", "description": "Get supported prop firms", ...},
#     {"name": "get_tiers", "description": "Get tiers for a firm", ...},
#     {"name": "get_rules", "description": "Get rules for firm/tier", ...},
#     {"name": "firm_analysis", "description": "Analyze firm rules", ...},
#     {"name": "compare_firms", "description": "Compare multiple firms", ...}
# ]
```

## Bot Configuration

### Account Book Type

Bots can be configured with an account book type in their manifest:

```python
from src.router.bot_manifest import AccountBook, BotManifest

manifest = BotManifest(
    bot_id="bot-001",
    name="My Strategy",
    # ...
    account_book_type=AccountBook.PROP_FIRM,  # or AccountBook.PERSONAL
    prop_firm_name="FTMO",  # Required if PROP_FIRM
    max_drawdown_pct=10.0   # Firm-specific drawdown limit
)
```

### Serialization

```python
# To dictionary
data = manifest.to_dict()
# Includes: "account_book_type", "prop_firm_name", "max_drawdown_pct"

# From dictionary
manifest = BotManifest.from_dict(data)
```

## Circuit Breaker Integration

The bot circuit breaker respects account book types:

```python
from src.router.bot_circuit_breaker import BotCircuitBreaker, AccountBook

# For prop firm accounts
breaker = BotCircuitBreaker(
    account_book=AccountBook.PROP_FIRM,
    max_consecutive_losses=3
)
```

## API Endpoints

### Paper Trading

Prop firm safety is checked in paper trading endpoints:

```python
# In src/api/paper_trading_endpoints.py
prop_firm_safe = request.strategy_type.upper() in ["STRUCTURAL", "SWING"]
```

## Testing

Run prop firm tests:

```bash
# Risk overlay tests
pytest tests/risk/test_prop_overlay.py -v

# Research tool tests
pytest tests/agents/tools/test_prop_firm_research_tool.py -v

# Bot tags tests
pytest tests/bots/test_prop_firm_tags.py -v
```

## Configuration Summary

| Component | File | Key Classes |
|-----------|------|-------------|
| Risk Overlay | `src/risk/prop_firm_overlay.py` | `PropFirmRiskOverlay`, `RiskLimits` |
| Research Tool | `src/agents/tools/prop_firm_research_tool.py` | `PropFirmResearch`, `PropFirmRules`, `FirmAnalysis` |
| Bot Manifest | `src/router/bot_manifest.py` | `AccountBook`, `BotManifest` |
| Circuit Breaker | `src/router/bot_circuit_breaker.py` | `BotCircuitBreaker`, `AccountBook` |
