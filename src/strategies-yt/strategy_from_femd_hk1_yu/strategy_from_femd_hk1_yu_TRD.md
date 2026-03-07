# Technical Requirements Document (TRD)

## Strategy from FEmD-hK1-yU

**Version**: 1.0.0
**Strategy Type**: STRUCTURAL
**Frequency**: MEDIUM
**Direction**: BOTH

---

## 1. Strategy Overview

Trading strategy extracted from video FEmD-hK1-yU

### Strategy Classification

| Parameter | Value |
|-----------|-------|
| Type | STRUCTURAL |
| Frequency | MEDIUM |
| Direction | BOTH |
| Kelly Fraction | 0.250 |

### Trading Instruments

**Symbols**: EURUSD
**Timeframes**: H1
**Sessions**: LONDON, NEW_YORK


---

## 2. Entry Conditions

1. Define entry conditions from video

---

## 3. Exit Conditions

1. Define exit conditions from video

---

## 4. Risk Management

### Stop Loss & Take Profit

- **Stop Loss**: 50.0 pips
- **Take Profit**: 100.0 pips
- **Risk/Reward Ratio**: 2.00

### Kelly Position Sizing

**Base Kelly Fraction**: 0.250

### Risk Limits

| Parameter | Value |
|-----------|-------|
| Max Risk Per Trade | 2% |
| Max Daily Loss | 5% |
| Max Drawdown | 15% |

---

## 5. Filters

No additional filters defined.

---

## 6. ZMQ Strategy Router Integration

The EA integrates with QuantMindX ZMQ Strategy Router for:

- **Real-time risk multiplier updates**
- **Circuit breaker notifications**
- **Regime change alerts**
- **Trade event logging**

### Connection Details

- **Endpoint**: tcp://localhost:5555
- **Heartbeat Interval**: 5000ms
- **Message Types**: TRADE_OPEN, TRADE_CLOSE, TRADE_MODIFY, HEARTBEAT, RISK_UPDATE

---

## 7. MQL5 Implementation Notes

### Required Inputs

```mql5
input string EA_Name = "Strategy from FEmD-hK1-yU";
input int MagicNumber = 100001;
input double BaseLotSize = 0.01;
input double MaxLotSize = 0.5;
input double StopLossPips = 50.0;
input double TakeProfitPips = 100.0;
input string PreferredSymbols = "EURUSD";
input ENUM_TIMEFRAMES PreferredTimeframe = PERIOD_H1;
```

---

## 8. Backtest Expectations




### Validation Criteria (PAPER -> LIVE)

- Minimum 30 days active
- Sharpe ratio > 1.5
- Win rate > 55%
- Maximum drawdown < 10%
- At least 50 trades executed

---

**Generated**: 2026-03-03T17:46:57.786173+00:00
**Author**: TRD Generation Agent

**Tags**: @primal, demo
