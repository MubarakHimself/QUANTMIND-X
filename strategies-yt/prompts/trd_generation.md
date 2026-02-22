# TRD Generation Prompt Template

Use this prompt when generating Trading Requirements Documents (TRDs) from video-based Truth Objects. This ensures comprehensive strategy documentation with proper configuration for QuantMindX integration.

---

## Prompt for TRD Generation Agent

```
You are generating a Trading Requirements Document (TRD) from a Truth Object (video analysis output).
The Truth Object contains a timeline of clips extracted from trading educational videos.

Your task is to:
1. Analyze the Truth Object timeline for trading strategy elements
2. Extract and synthesize strategy logic from visual descriptions and transcripts
3. Generate a complete TRD.md file with strategy specifications
4. Generate a config JSON file with all required parameters

## Input: Truth Object Structure

The Truth Object is a JSON output from video analysis:

```json
{
  "video_id": "vid_12345",
  "meta": {
    "title": "Strategy Title",
    "total_duration": "45:00",
    "speakers": {
      "SPEAKER_01": "Host/Interviewer",
      "SPEAKER_02": "Expert/Guest"
    }
  },
  "timeline": [
    {
      "clip_id": 1,
      "timestamp_start": "04:15",
      "timestamp_end": "05:30",
      "speaker": "SPEAKER_02",
      "visual_description": "Screen shows chart with indicators...",
      "transcript": "So what we are looking for here is the strategy logic...",
      "ocr_content": ["Chart labels", "Rules"],
      "slide_detected": false
    }
  ]
}
```

## Required Extractions

### 1. Strategy Type
Determine from frequency and holding period descriptions:
- **HFT** (Tick-level): Sub-minute trades, Kelly fraction: 0.10-0.15
- **SCALPER** (HIGH): 5-15 minute trades, Kelly fraction: 0.15-0.25
- **STRUCTURAL** (MEDIUM): ICT/SMC concepts, 30min-4hr trades, Kelly fraction: 0.20-0.30
- **SWING** (LOW): Multi-day positions, Kelly fraction: 0.25-0.40

### 2. Entry-Exit Conditions
Extract exact rules from timeline clips:
- Entry signals (indicator crossovers, price patterns, time-based triggers)
- Exit signals (take profit rules, stop loss logic, time-based exits)
- Confirmation requirements (multi-timeframe, filter conditions)
- Order types (market, limit, stop, stop-limit)

### 3. Trading Sessions
Identify from time references in transcripts:
- **ASIAN**: 00:00-08:00 UTC (Ranging strategies)
- **LONDON**: 08:00-13:00 UTC (Breakouts, trends)
- **OVERLAP**: 13:00-17:00 UTC (Highest volatility)
- **NEW_YORK**: 13:00-21:00 UTC (News, trends)

### 4. Custom Variables
Extract strategy-specific parameters:
- Indicator periods (EMA length, RSI period, ATR multiplier)
- Price levels (support/resistance distances, pip targets)
- Time parameters (session windows, cooldown periods)
- Filter thresholds (spread limits, volatility ranges)

### 5. Condition Checks
Identify technical requirements:
- Indicator calculations (moving averages, oscillators, volume)
- Pattern recognition (candlestick patterns, chart patterns)
- Market structure (highs/lows, order blocks, liquidity zones)
- Risk conditions (max spread, slippage tolerance)

### 6. Kelly Fraction
Assign based on strategy type:
| Type | Kelly Fraction | Rationale |
|------|----------------|-----------|
| HFT | 0.10-0.15 | High frequency, small edge |
| SCALPER | 0.15-0.25 | Quick trades, moderate edge |
| STRUCTURAL | 0.20-0.30 | ICT concepts, defined edge |
| SWING | 0.25-0.40 | Longer hold, larger edge |

## Output 1: TRD.md File

Generate a markdown file with this structure:

```markdown
# [Strategy Name] TRD

## Version
1.0.0

## Description
[Comprehensive strategy description based on video content]

## Strategy Classification
- **Type**: SCALPER|STRUCTURAL|SWING|HFT
- **Frequency**: HFT|HIGH|MEDIUM|LOW
- **Direction**: LONG|SHORT|BOTH
- **Kelly Fraction**: [0.10-0.40 based on type]

## Trading Parameters
- **Timeframe**: [Primary timeframe]
- **Symbols**: [Trading instruments]
- **Sessions**: [LONDON|NEW_YORK|OVERLAP|ASIAN]
- **Time Window**: [UTC hours]

## Entry Rules

### Primary Entry
**Name**: [Entry rule name]
**Description**: [Detailed description]

**Conditions**:
1. [Exact condition 1 - from video extraction]
2. [Exact condition 2 - from video extraction]
3. [Exact condition 3 - from video extraction]

**Order Type**: market|limit|stop|stop_limit
**Confirmation**: [Multi-timeframe or filter requirements]

### Secondary Entry (if applicable)
[Additional entry setups from video]

## Exit Rules

### Take Profit
**Type**: fixed|risk_reward|fibonacci
**Value**: [Target value]
**Partial Exits**: [If mentioned in video]

### Stop Loss
**Type**: fixed|trailing|atr|swing_low_high
**Value**: [Stop distance]
**Trailing**: [If applicable]

### Time-Based Exit
[Any time-based exit rules mentioned]

## Risk Management
- **Max Risk Per Trade**: [0.01-0.05 based on video]
- **Max Daily Drawdown**: [Default 0.05 if not specified]
- **Max Open Positions**: [Based on video mentions]
- **Correlation Limit**: [Default 0.7]

## Indicators

### [Indicator Name 1]
- **Parameters**: [From extraction]
- **Purpose**: [How it's used in strategy]

### [Indicator Name 2]
- **Parameters**: [From extraction]
- **Purpose**: [How it's used in strategy]

## Additional Filters
- **Spread Filter**: [Max spread in pips]
- **Volatility Filter**: [ATR range if mentioned]
- **Session Filter**: [Trading hours]
- **News Filter**: [If mentioned]

## Custom Variables
[Strategy-specific variables extracted from video]

## Known Limitations
[Any limitations or caveats mentioned]

## Backtest Expectations
[Expected performance characteristics if mentioned]

---
Generated from: [Video ID]
Extracted: [Timestamp]
Author: TRD Generation Agent
```

## Output 2: Config JSON File

Generate a JSON configuration file:

```json
{
  "ea_id": "[strategy_name_snake_case]",
  "name": "[Strategy Name]",
  "version": "1.0.0",
  "description": "[Strategy description from video]",

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

  "custom_variables": {
    "[var_name_1]": [value],
    "[var_name_2]": [value],
    "indicator_periods": {
      "fast_ema": [value],
      "slow_ema": [value],
      "rsi_period": [value],
      "atr_period": [value]
    },
    "entry_thresholds": {
      "[threshold_name]": [value]
    },
    "exit_levels": {
      "[level_name]": [value]
    }
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

  "tags": ["@primal"],
  "source": {
    "video_id": "[from Truth Object]",
    "video_title": "[from Truth Object]",
    "extracted_at": "[ISO timestamp]"
  },
  "author": "TRD Generation Agent",
  "created_at": "[ISO date]"
}
```

## Custom Variables Examples

Based on common strategy types, include relevant variables:

### Scalper Example
```json
"custom_variables": {
  "entry_timeout_minutes": 5,
  "min_profit_pips": 5,
  "max_hold_minutes": 30,
  "quick_target_pips": 3,
  "breakout_pips": 2
}
```

### Structural (ICT) Example
```json
"custom_variables": {
  "order_block_lookback": 5,
  "fair_value_gap_pips": 5,
  "liquidity_sweep_pips": 3,
  "displacement_threshold": 1.5,
  "mitigation_confirmation": true
}
```

### Swing Example
```json
"custom_variables": {
  "swing_period": 20,
  "daily_high_low_lookback": 5,
  "weekly_target_pips": 100,
  "min_hold_hours": 24,
  "trailing_activation_pips": 50
}
```

## Broker Preferences Guide

Select based on strategy requirements:

| Strategy Type | Preferred Type | Min Leverage | Rationale |
|---------------|----------------|--------------|-----------|
| HFT | RAW_ECN | 500+ | Lowest latency, tightest spreads |
| SCALPER | RAW_ECN | 100+ | Tight spreads critical |
| STRUCTURAL | ECN | 100 | Standard execution sufficient |
| SWING | STANDARD | 50 | Spread less critical |

## Extraction Guidelines

### From Visual Descriptions
- Chart patterns highlighted
- Indicator settings shown
- Price level annotations
- Time windows displayed

### From Transcripts
- Rules explicitly stated
- Entry/exit logic explained
- Risk management discussed
- Time/session references

### From OCR Content
- Slide titles and bullet points
- Rule lists
- Parameter values
- Stop/take profit levels

### From Slide Detection
- Structured rule presentations
- Multi-step processes
- Checklists and conditions

## Validation Checklist

After generation, verify:
- [ ] Strategy type matches Kelly fraction range
- [ ] Entry conditions are specific and testable
- [ ] Exit conditions include both TP and SL
- [ ] Sessions align with strategy type
- [ ] Custom variables are named meaningfully
- [ ] Broker preferences match strategy needs
- [ ] @primal tag is included
- [ ] Source video metadata preserved
- [ ] All timeline clips were analyzed
- [ ] Contradictions in video content resolved

## Integration Flow

1. Truth Object generated from video analysis
2. TRD Generation Agent extracts strategy elements
3. TRD.md and config JSON created
4. Files placed in: `strategies-yt/[strategy_name]/`
5. QuantMindX imports and validates
6. BotManifest created with @primal tag
7. EA ready for PAPER trading validation

---

*Template Version: 1.2*
*Based on: EA_CREATION_PROMPT.md v1.2*
*Last Updated: 2026-02-22*
