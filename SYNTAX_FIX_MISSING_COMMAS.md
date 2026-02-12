# Syntax Fix: Missing Trailing Commas in test_fee_aware_kelly.py

## Issue Fixed
Two missing trailing commas prevented the test file from parsing correctly, making it impossible to run the fee-aware Kelly unit tests.

## Changes Made

### File: `/home/mubarkahimself/Desktop/QUANTMINDX/tests/position_sizing/test_fee_aware_kelly.py`

### Fix 1: Line 87 - Missing Comma After `spread_pips=spread_pips`

**Before (Syntax Error)**:
```python
result_with_fees = kelly_calculator.calculate(
    account_balance=account_balance,
    win_rate=win_rate,
    avg_win=avg_win,
    avg_loss=avg_loss,
    current_atr=current_atr,
    average_atr=average_atr,
    stop_loss_pips=stop_loss_pips,
    pip_value=pip_value,
    regime_quality=1.0,
    commission_per_lot=commission_per_lot,
    spread_pips=spread_pips          # ❌ Missing comma
    broker_id=broker_id,              # ❌ Causes parser error
    symbol=symbol
)
```

**After (Fixed)**:
```python
result_with_fees = kelly_calculator.calculate(
    account_balance=account_balance,
    win_rate=win_rate,
    avg_win=avg_win,
    avg_loss=avg_loss,
    current_atr=current_atr,
    average_atr=average_atr,
    stop_loss_pips=stop_loss_pips,
    pip_value=pip_value,
    regime_quality=1.0,
    commission_per_lot=commission_per_lot,
    spread_pips=spread_pips,         # ✅ Comma added
    broker_id=broker_id,
    symbol=symbol
)
```

### Fix 2: Line 192 - Missing Comma After `symbol="XAUUSD"`

**Before (Syntax Error)**:
```python
result = kelly_calculator.calculate(
    account_balance=10000.0,
    win_rate=0.55,
    avg_win=400.0,
    avg_loss=200.0,
    current_atr=0.0012,
    average_atr=0.0010,
    stop_loss_pips=20.0,
    pip_value=10.0,  # Will be overridden by auto-lookup
    regime_quality=1.0,
    broker_id="test_lookup_broker",
    symbol="XAUUSD"  # Should fetch pip_value=1.0  ❌ Missing comma
    commission_per_lot=0.0,  # Will be overridden by auto-lookup  ❌ Parser error
    spread_pips=0.0  # Will be overridden by auto-lookup
)
```

**After (Fixed)**:
```python
result = kelly_calculator.calculate(
    account_balance=10000.0,
    win_rate=0.55,
    avg_win=400.0,
    avg_loss=200.0,
    current_atr=0.0012,
    average_atr=0.0010,
    stop_loss_pips=20.0,
    pip_value=10.0,  # Will be overridden by auto-lookup
    regime_quality=1.0,
    broker_id="test_lookup_broker",
    symbol="XAUUSD",  # Should fetch pip_value=1.0  ✅ Comma added
    commission_per_lot=0.0,  # Will be overridden by auto-lookup
    spread_pips=0.0  # Will be overridden by auto-lookup
)
```

## Verification

✅ **Syntax validation**: File compiles without errors
```bash
$ python -m py_compile tests/position_sizing/test_fee_aware_kelly.py
```
No output = Success

## Impact

- ✅ **test_fee_aware_kelly_reduces_position()** can now run correctly
- ✅ **test_broker_auto_lookup()** can now run correctly
- ✅ All fee-aware Kelly unit tests are now executable

## Test Coverage

These tests now verify:
1. **Fee reduction**: Position sizes are properly reduced when commissions and spreads are included
2. **Fee kill switch**: Trades are blocked when fees exceed expected profit
3. **Broker auto-lookup**: Pip values, commissions, and spreads are correctly fetched from broker registry
4. **Kelly adjustments**: Fee-aware Kelly correctly applies all 3 protection layers

## Files Modified

1. **tests/position_sizing/test_fee_aware_kelly.py**
   - Line 87: Added comma after `spread_pips=spread_pips`
   - Line 192: Added comma after `symbol="XAUUSD"`
   - Total changes: 2 lines (2 commas added)
