"""
Verification script for ENUM_RISK_TIER enumeration in KellySizer.mqh

This script verifies that:
1. The ENUM_RISK_TIER enumeration is properly defined
2. All three tier values are present (TIER_GROWTH, TIER_SCALING, TIER_GUARDIAN)
3. The enumeration is used in the DetermineRiskTier() method
4. The enumeration is used in the CalculateTieredPositionSize() method
"""

import re
from pathlib import Path


def verify_enum_risk_tier():
    """Verify ENUM_RISK_TIER enumeration in KellySizer.mqh"""
    
    kelly_sizer_path = Path("src/mql5/Include/QuantMind/Risk/KellySizer.mqh")
    
    if not kelly_sizer_path.exists():
        print(f"❌ ERROR: KellySizer.mqh not found at {kelly_sizer_path}")
        return False
    
    content = kelly_sizer_path.read_text()
    
    # Check 1: Enumeration is defined
    enum_pattern = r'enum\s+ENUM_RISK_TIER\s*\{'
    if not re.search(enum_pattern, content):
        print("❌ ERROR: ENUM_RISK_TIER enumeration not found")
        return False
    print("✅ ENUM_RISK_TIER enumeration is defined")
    
    # Check 2: All three tier values are present
    tiers = ['TIER_GROWTH', 'TIER_SCALING', 'TIER_GUARDIAN']
    for tier in tiers:
        if tier not in content:
            print(f"❌ ERROR: {tier} not found in enumeration")
            return False
        print(f"✅ {tier} is defined")
    
    # Check 3: Enumeration is used in DetermineRiskTier()
    determine_tier_pattern = r'ENUM_RISK_TIER\s+DetermineRiskTier\s*\('
    if not re.search(determine_tier_pattern, content):
        print("❌ ERROR: DetermineRiskTier() method not found or doesn't return ENUM_RISK_TIER")
        return False
    print("✅ DetermineRiskTier() method uses ENUM_RISK_TIER")
    
    # Check 4: Enumeration is used in CalculateTieredPositionSize()
    tiered_position_pattern = r'ENUM_RISK_TIER\s+tier\s*='
    if not re.search(tiered_position_pattern, content):
        print("❌ ERROR: ENUM_RISK_TIER not used in CalculateTieredPositionSize()")
        return False
    print("✅ ENUM_RISK_TIER is used in CalculateTieredPositionSize()")
    
    # Check 5: Switch statement uses the enumeration
    switch_pattern = r'switch\s*\(\s*tier\s*\)'
    if not re.search(switch_pattern, content):
        print("❌ ERROR: Switch statement not found for tier-based logic")
        return False
    print("✅ Switch statement uses tier enumeration")
    
    # Check 6: All tier cases are handled in switch
    for tier in tiers:
        case_pattern = rf'case\s+{tier}\s*:'
        if not re.search(case_pattern, content):
            print(f"❌ ERROR: Case for {tier} not found in switch statement")
            return False
        print(f"✅ Case for {tier} is handled in switch statement")
    
    # Check 7: Comments describe each tier
    tier_comments = {
        'TIER_GROWTH': r'TIER_GROWTH.*\$100-\$1K',
        'TIER_SCALING': r'TIER_SCALING.*\$1K-\$5K',
        'TIER_GUARDIAN': r'TIER_GUARDIAN.*\$5K\+'
    }
    
    for tier, comment_pattern in tier_comments.items():
        if not re.search(comment_pattern, content):
            print(f"⚠️  WARNING: Comment for {tier} may be missing or incomplete")
        else:
            print(f"✅ Comment for {tier} describes equity range")
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED: ENUM_RISK_TIER is properly implemented")
    print("="*60)
    return True


if __name__ == "__main__":
    success = verify_enum_risk_tier()
    exit(0 if success else 1)
