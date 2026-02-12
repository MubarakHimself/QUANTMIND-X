# Verification Comment Implementation: House-Money Multiplier Scaling

## Issue
**House-money multiplier not applied to position_size or risk_amount returned by EnhancedGovernor.**

The Kelly fraction was scaled by the house-money multiplier, but the resulting position_size and risk_amount were not scaled consistently, leading to inconsistent risk reporting.

## Solution Implemented

### File: [src/router/enhanced_governor.py](src/router/enhanced_governor.py#L153-L179)

### Changes Made

**Before (Lines 154-178)**:
```python
# Apply house money multiplier
kelly_f = kelly_result.kelly_f * self.house_money_multiplier

# Apply base Governor physics-based throttling
base_mandate = super().calculate_risk(regime_report, trade_proposal)

# Combine Kelly fraction with physics throttle
allocation_scalar = min(kelly_f, base_mandate.allocation_scalar)

# Build mandate with extended fields
mandate = RiskMandate(
    allocation_scalar=allocation_scalar,
    risk_mode=base_mandate.risk_mode,
    position_size=kelly_result.position_size,  # ❌ NOT scaled
    kelly_fraction=kelly_result.kelly_f,       # ❌ Using base kelly_f
    risk_amount=kelly_result.risk_amount,      # ❌ NOT scaled
    kelly_adjustments=kelly_result.adjustments_applied,
    notes=(
        f"Kelly: {kelly_result.kelly_f:.4f} × "
        f"House Money: {self.house_money_multiplier:.2f} = {kelly_f:.4f}. "
        f"{base_mandate.notes or ''}"
    )
)
```

**After (Lines 153-179)**:
```python
# Apply house money multiplier consistently to all risk metrics
kelly_f = kelly_result.kelly_f * self.house_money_multiplier
scaled_position_size = kelly_result.position_size * self.house_money_multiplier  # ✅ SCALED
scaled_risk_amount = kelly_result.risk_amount * self.house_money_multiplier      # ✅ SCALED

# Apply base Governor physics-based throttling
base_mandate = super().calculate_risk(regime_report, trade_proposal)

# Combine Kelly fraction with physics throttle
allocation_scalar = min(kelly_f, base_mandate.allocation_scalar)

# Build mandate with extended fields
# Note: position_size and risk_amount now include house_money_multiplier scaling
mandate = RiskMandate(
    allocation_scalar=allocation_scalar,
    risk_mode=base_mandate.risk_mode,
    position_size=scaled_position_size,  # ✅ Scaled by house_money_multiplier
    kelly_fraction=kelly_f,              # ✅ Scaled kelly fraction
    risk_amount=scaled_risk_amount,      # ✅ Scaled by house_money_multiplier
    kelly_adjustments=kelly_result.adjustments_applied,
    notes=(
        f"Kelly: {kelly_result.kelly_f:.4f} × "
        f"House Money: {self.house_money_multiplier:.2f} = {kelly_f:.4f}. "
        f"Scaled position: {scaled_position_size:.4f} lots, risk: ${scaled_risk_amount:.2f}. "
        f"{base_mandate.notes or ''}"
    )
)
```

## Impact

### Consistent Risk Reporting
- **Kelly fraction**: Scaled by house_money_multiplier ✅
- **Position size**: Scaled by house_money_multiplier ✅
- **Risk amount**: Scaled by house_money_multiplier ✅

### House-Money Multiplier Effects

| Scenario | Multiplier | Impact |
|----------|-----------|--------|
| Up > 5% | 1.5x | Position = 1.5x normal, Risk = 1.5x normal |
| Normal ±3% | 1.0x | Position = 1.0x normal, Risk = 1.0x normal |
| Down > 3% | 0.5x | Position = 0.5x normal, Risk = 0.5x normal |

### Example Behavior

**With Base Kelly Result**:
- position_size = 2.5 lots
- risk_amount = $250
- kelly_fraction = 0.02 (2%)

**After House-Money Scaling (multiplier = 1.5x)**:
- kelly_fraction = 0.02 × 1.5 = 0.03 (3%)
- position_size = 2.5 × 1.5 = 3.75 lots ✅ (was 2.5)
- risk_amount = $250 × 1.5 = $375 ✅ (was $250)

## Backward Compatibility

✅ **Fully backward compatible**:
- Default house_money_multiplier = 1.0 (no scaling)
- When multiplier = 1.0, position_size and risk_amount unchanged
- Existing code continues to work without modification
- All three metrics (kelly_f, position_size, risk_amount) now consistently scaled

## Testing Recommendations

### Unit Test: House-Money Scaling
```python
def test_house_money_multiplier_scaling():
    """Verify position_size and risk_amount scale with house_money_multiplier."""
    governor = EnhancedGovernor()
    
    # Set aggressive house money (up 5%)
    governor.house_money_multiplier = 1.5
    
    # Calculate risk with mock regime
    mandate = governor.calculate_risk(regime_report, trade_proposal)
    
    # Verify all metrics scaled consistently
    assert mandate.kelly_fraction == expected_kelly_f * 1.5
    assert mandate.position_size == expected_position_size * 1.5
    assert mandate.risk_amount == expected_risk_amount * 1.5
```

### Integration Test: Preservation Mode
```python
def test_preservation_mode_scaling():
    """Verify position_size reduced in preservation mode."""
    governor = EnhancedGovernor()
    
    # Set preservation mode (down 3%)
    governor.house_money_multiplier = 0.5
    
    # Calculate risk
    mandate = governor.calculate_risk(regime_report, trade_proposal)
    
    # Verify smaller positions in preservation mode
    assert mandate.position_size == base_position_size * 0.5
    assert mandate.risk_amount == base_risk_amount * 0.5
```

## Enhanced Logging

The mandate notes now explicitly show scaled values:
```
"Kelly: 0.0200 × House Money: 1.5000 = 0.0300. 
Scaled position: 3.7500 lots, risk: $375.00. [base governor notes]"
```

This provides transparency in how house-money adjustments affect actual position sizing.

## Files Modified

1. **[src/router/enhanced_governor.py](src/router/enhanced_governor.py)**
   - Lines 153-179: Apply house-money multiplier to position_size and risk_amount
   - Enhanced notes field to show scaled values
   - Added comment explaining scaling consistency
   - Total: ~7 lines modified/added

## Verification Checklist

✅ House-money multiplier applied to kelly_f
✅ House-money multiplier applied to position_size  
✅ House-money multiplier applied to risk_amount
✅ All three metrics scaled by same multiplier (consistency)
✅ Enhanced logging shows scaled values
✅ No syntax errors
✅ Backward compatible (multiplier=1.0 unchanged)
✅ RiskMandate fields reflect house-money adjustments

## Summary

The house-money multiplier is now **consistently applied** to all risk metrics in the RiskMandate:
- kelly_fraction: scaled ✅
- position_size: scaled ✅
- risk_amount: scaled ✅

This ensures that traders receive accurate position sizing recommendations that reflect both Kelly Criterion and psychological adjustments from running P&L.
