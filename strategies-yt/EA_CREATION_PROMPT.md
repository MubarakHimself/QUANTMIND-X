# EA Creation Prompt Template

Use this prompt when creating Expert Advisors for the QuantMindX system. This ensures all required parameters and configurations are included for proper registration and runtime enforcement.

---

## Prompt for EA Creation Agent

```
You are creating an Expert Advisor (EA) for the QuantMindX automated trading system.
The EA must follow this structure to be properly registered and integrated with:
- Strategy Router (multi-symbol routing)
- Kelly Risk Management (position sizing)
- Governor (risk limits enforcement)
- Broker Registry (fee-aware calculations)

## Required Outputs

Create the following files for each EA:

### 1. MQL5 EA File: `EA_[strategy_name].mq5`

Include these standard input parameters:

```mql5
//--- Input Parameters (Required for QuantMindX Registration)
input string   EA_Name = "[Strategy Name]";           // EA Identifier
input int      MagicNumber = 100001;                  // Unique Magic Number
input double   BaseLotSize = 0.01;                    // Base lot size (Kelly adjusts)
input double   MaxLotSize = 0.5;                      // Maximum allowed lot
input double   StopLossPips = 50;                     // Stop loss in pips
input double   TakeProfitPips = 100;                  // Take profit in pips
input bool     UseTrailingStop = true;                // Enable trailing stop
input double   TrailingStopPips = 20;                 // Trailing stop distance
input double   BreakEvenPips = 30;                    // Break-even trigger
input string   PreferredSymbols = "EURUSD,GBPUSD";    // Comma-separated symbols
input ENUM_TIMEFRAMES PreferredTimeframe = PERIOD_H1; // Primary timeframe
input int      MaxSpreadPips = 3;                     // Maximum spread filter
input string   TradingHours = "08:00-17:00";          // Trading hours (UTC)
```

### 2. Config File: `[strategy_name]_config.json`

```json
{
  "ea_id": "[strategy_name]",
  "name": "[Strategy Name]",
  "version": "1.0.0",
  "description": "[Strategy description]",

  "strategy": {
    "type": "SCALPER|STRUCTURAL|SWING|HFT",
    "frequency": "HFT|HIGH|MEDIUM|LOW",
    "direction": "BOTH|LONG|SHORT"
  },

  "symbols": {
    "primary": ["EURUSD", "GBPUSD"],
    "timeframes": ["H1", "M15"],
    "symbol_groups": ["majors", "crosses"]
  },

  "trading_conditions": {
    "sessions": ["LONDON", "NEW_YORK", "OVERLAP"],
    "timezone": "UTC",
    "custom_windows": [
      {
        "name": "London Open",
        "start": "08:00",
        "end": "12:00",
        "days": ["MON", "TUE", "WED", "THU", "FRI"]
      }
    ],
    "volatility": {
      "min_atr": 0.0005,
      "max_atr": 0.003,
      "atr_period": 14
    },
    "preferred_regime": "TRENDING|RANGING|BREAKOUT",
    "min_spread_pips": 0,
    "max_spread_pips": 3
  },

  "risk_parameters": {
    "kelly_fraction": 0.25,
    "max_risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "max_drawdown": 0.15,
    "max_open_trades": 3,
    "correlation_limit": 0.7
  },

  "position_sizing": {
    "base_lot": 0.01,
    "max_lot": 0.5,
    "scale_with_account": true,
    "respect_prop_limits": true
  },

  "broker_preferences": {
    "preferred_type": "RAW_ECN",
    "min_leverage": 100,
    "allowed_brokers": [],
    "excluded_brokers": []
  },

  "prop_firm": {
    "compatible": true,
    "supported_firms": ["FTMO", "The5ers", "FundingPips"],
    "daily_loss_limit": 0.05,
    "max_trailing_drawdown": 0.10
  },

  "tags": ["@primal", "demo"],
  "author": "[Your Name]",
  "created_at": "[Date]"
}
```

### 3. Technical Requirements Document (TRD): `[strategy_name]_TRD.md`

Include:
- Strategy logic description
- Entry conditions (exact rules)
- Exit conditions (exact rules)
- Risk management rules
- Backtest expectations
- Known limitations

## Strategy Classification

Choose the appropriate classification:

| Type | Frequency | Description | Kelly Fraction |
|------|-----------|-------------|----------------|
| HFT | Tick-level | High-frequency, sub-minute | 0.10-0.15 |
| SCALPER | HIGH | Quick 5-15 min trades | 0.15-0.25 |
| STRUCTURAL | MEDIUM | ICT/SMC concepts | 0.20-0.30 |
| SWING | LOW | Multi-day positions | 0.25-0.40 |

## Trading Sessions

| Session | UTC Hours | Best For |
|---------|-----------|----------|
| ASIAN | 00:00-08:00 | Ranging strategies |
| LONDON | 08:00-13:00 | Breakouts, trends |
| OVERLAP | 13:00-17:00 | Highest volatility |
| NEW_YORK | 13:00-21:00 | News, trends |

## Registration Flow

1. Create EA in `strategies-yt/[strategy_name]/`
2. Include all 3 files (MQ5, JSON config, TRD)
3. Push to GitHub
4. QuantMindX imports and registers automatically
5. BotManifest created with all parameters
6. EA starts in PAPER mode for validation

## Validation Criteria (PAPER → LIVE)

- Minimum 30 days active
- Sharpe ratio > 1.5
- Win rate > 55%
- Maximum drawdown < 10%
- At least 50 trades executed
```

---

## Quick Reference: EA Input Parameters

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `EA_Name` | Identifier | "LondonBreakout" |
| `MagicNumber` | Unique ID | 100001-999999 |
| `BaseLotSize` | Starting lot | 0.01 |
| `MaxLotSize` | Lot cap | 0.5 |
| `StopLossPips` | SL distance | 50 |
| `TakeProfitPips` | TP distance | 100 |
| `UseTrailingStop` | Trail enable | true |
| `TrailingStopPips` | Trail distance | 20 |
| `PreferredSymbols` | Target pairs | "EURUSD,GBPUSD" |
| `PreferredTimeframe` | Chart TF | PERIOD_H1 |
| `MaxSpreadPips` | Spread filter | 3 |
| `TradingHours` | Time window | "08:00-17:00" |

---

## Quick Reference: Config JSON Fields

| Section | Key Fields |
|---------|-----------|
| `strategy` | type, frequency, direction |
| `symbols` | primary, timeframes, symbol_groups |
| `trading_conditions` | sessions, volatility, preferred_regime |
| `risk_parameters` | kelly_fraction, max_risk_per_trade, max_drawdown |
| `position_sizing` | base_lot, max_lot, scale_with_account |
| `broker_preferences` | preferred_type, allowed_brokers |
| `prop_firm` | compatible, supported_firms |

---

*Template Version: 1.0*
*Last Updated: 2026-02-22*
