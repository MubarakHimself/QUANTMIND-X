#!/usr/bin/env python3
"""
Broker Registry Seeding Validation Script

Validates that the broker registry seeding was successful and that
all three brokers are properly populated with real fee data.

Usage:
    python scripts/validate_broker_registry.py

Exit Codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import sys
import logging
from typing import Dict, List, Optional, cast

# Add parent directory to path for imports
sys.path.insert(0, '/home/mubarkahimself/Desktop/QUANTMINDX')

from src.router.broker_registry import BrokerRegistryManager
from src.database.models import BrokerRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_broker_values(broker: BrokerRegistry) -> tuple[float, float, Optional[Dict[str, float]], Optional[List[str]]]:
    """Extract values from broker model with proper typing."""
    # Use getattr to avoid Pylance seeing these as Column types
    spread_avg = getattr(broker, 'spread_avg', 0.0)
    commission_per_lot = getattr(broker, 'commission_per_lot', 0.0)
    pip_values = getattr(broker, 'pip_values', None)
    preference_tags = getattr(broker, 'preference_tags', None)
    
    return (
        cast(float, spread_avg),
        cast(float, commission_per_lot),
        cast(Optional[Dict[str, float]], pip_values),
        cast(Optional[List[str]], preference_tags)
    )


def validate_broker_exists(mgr: BrokerRegistryManager, broker_id: str) -> bool:
    """Validate that broker exists in registry."""
    broker = mgr.get_broker(broker_id)
    if broker is None:
        logger.error(f"❌ Broker '{broker_id}' not found in registry")
        return False
    logger.info(f"✓ Broker '{broker_id}' exists")
    return True


def validate_icmarkets_raw(mgr: BrokerRegistryManager) -> bool:
    """Validate icmarkets_raw broker profile."""
    broker = mgr.get_broker("icmarkets_raw")
    if broker is None:
        return False
    
    checks = []
    
    # Extract values with proper typing
    spread_avg, commission_per_lot, pip_values, preference_tags = get_broker_values(broker)
    
    # Check spread
    if spread_avg != 0.1:
        logger.error(f"  ❌ Spread: expected 0.1, got {spread_avg}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Spread: {spread_avg} pips")
        checks.append(True)
    
    # Check commission
    if commission_per_lot != 7.0:
        logger.error(f"  ❌ Commission: expected 7.0, got {commission_per_lot}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Commission: ${commission_per_lot}/lot")
        checks.append(True)
    
    # Check pip values
    if pip_values is None:
        logger.error("  ❌ Pip values not set")
        checks.append(False)
    else:
        # Validate key symbols
        required_symbols = {
            "EURUSD": 10.0,
            "XAUUSD": 1.0,
            "XAGUSD": 50.0,
        }
        for symbol, expected_pip in required_symbols.items():
            actual_pip = pip_values.get(symbol)
            if actual_pip != expected_pip:
                logger.error(f"  ❌ {symbol} pip: expected {expected_pip}, got {actual_pip}")
                checks.append(False)
            else:
                logger.info(f"  ✓ {symbol} pip value: {actual_pip}")
                checks.append(True)
    
    # Check tags
    if preference_tags is None or "RAW_ECN" not in preference_tags:
        logger.error("  ❌ RAW_ECN tag not found")
        checks.append(False)
    else:
        tags_str = ', '.join(preference_tags)
        logger.info(f"  ✓ Tags: {tags_str}")
        checks.append(True)
    
    return all(checks)


def validate_roboforex_prime(mgr: BrokerRegistryManager) -> bool:
    """Validate roboforex_prime broker profile."""
    broker = mgr.get_broker("roboforex_prime")
    if broker is None:
        return False
    
    checks = []
    
    # Extract values with proper typing
    spread_avg, commission_per_lot, pip_values, _ = get_broker_values(broker)
    
    # Check spread
    if spread_avg != 0.2:
        logger.error(f"  ❌ Spread: expected 0.2, got {spread_avg}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Spread: {spread_avg} pips")
        checks.append(True)
    
    # Check commission (should be 0)
    if commission_per_lot != 0.0:
        logger.error(f"  ❌ Commission: expected 0.0, got {commission_per_lot}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Commission: no commission")
        checks.append(True)
    
    # Check pip values
    if pip_values is None:
        logger.error("  ❌ Pip values not set")
        checks.append(False)
    else:
        required_symbols = {
            "EURUSD": 10.0,
            "XAUUSD": 1.0,
        }
        for symbol, expected_pip in required_symbols.items():
            actual_pip = pip_values.get(symbol)
            if actual_pip != expected_pip:
                logger.error(f"  ❌ {symbol} pip: expected {expected_pip}, got {actual_pip}")
                checks.append(False)
            else:
                logger.info(f"  ✓ {symbol} pip value: {actual_pip}")
                checks.append(True)
    
    return all(checks)


def validate_mt5_default(mgr: BrokerRegistryManager) -> bool:
    """Validate mt5_default broker profile."""
    broker = mgr.get_broker("mt5_default")
    if broker is None:
        return False
    
    checks = []
    
    # Extract values with proper typing
    spread_avg, _, pip_values, _ = get_broker_values(broker)
    
    # Check spread
    if spread_avg != 0.5:
        logger.error(f"  ❌ Spread: expected 0.5, got {spread_avg}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Spread: {spread_avg} pips")
        checks.append(True)
    
    # Check pip values exist (at minimum)
    if pip_values is None:
        logger.error("  ❌ Pip values not set")
        checks.append(False)
    else:
        logger.info(f"  ✓ Pip values defined")
        checks.append(True)
    
    return all(checks)


def validate_pip_lookup(mgr: BrokerRegistryManager) -> bool:
    """Validate pip value lookup functionality."""
    checks = []
    
    # Test XAUUSD pip lookup (should be 1.0 for icmarkets, not default 10.0)
    xau_pip = mgr.get_pip_value("XAUUSD", "icmarkets_raw")
    if xau_pip != 1.0:
        logger.error(f"  ❌ XAUUSD pip: expected 1.0, got {xau_pip}")
        checks.append(False)
    else:
        logger.info(f"  ✓ XAUUSD pip lookup: {xau_pip}")
        checks.append(True)
    
    # Test spread lookup
    spread = mgr.get_spread("icmarkets_raw")
    if spread != 0.1:
        logger.error(f"  ❌ Spread lookup: expected 0.1, got {spread}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Spread lookup: {spread} pips")
        checks.append(True)
    
    # Test commission lookup
    commission = mgr.get_commission("icmarkets_raw")
    if commission != 7.0:
        logger.error(f"  ❌ Commission lookup: expected 7.0, got {commission}")
        checks.append(False)
    else:
        logger.info(f"  ✓ Commission lookup: ${commission}/lot")
        checks.append(True)
    
    return all(checks)


def main() -> int:
    """Main validation function."""
    print("=" * 70)
    print("BROKER REGISTRY VALIDATION")
    print("=" * 70)
    
    try:
        mgr = BrokerRegistryManager()
    except Exception as e:
        logger.error(f"❌ Failed to initialize BrokerRegistryManager: {e}")
        return 1
    
    all_passed = True
    
    # Test 1: Broker existence
    print("\n1. BROKER EXISTENCE CHECK")
    print("-" * 70)
    for broker_id in ["icmarkets_raw", "roboforex_prime", "mt5_default"]:
        if not validate_broker_exists(mgr, broker_id):
            all_passed = False
    
    # Test 2: ICMarkets Raw profile
    print("\n2. ICMARKETS RAW PROFILE")
    print("-" * 70)
    if not validate_icmarkets_raw(mgr):
        all_passed = False
    
    # Test 3: RoboForex Prime profile
    print("\n3. ROBOFOREX PRIME PROFILE")
    print("-" * 70)
    if not validate_roboforex_prime(mgr):
        all_passed = False
    
    # Test 4: MT5 Default profile
    print("\n4. MT5 DEFAULT PROFILE")
    print("-" * 70)
    if not validate_mt5_default(mgr):
        all_passed = False
    
    # Test 5: Pip value lookup functionality
    print("\n5. PIP VALUE LOOKUP FUNCTIONALITY")
    print("-" * 70)
    if not validate_pip_lookup(mgr):
        all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("=" * 70)
        print("\nBroker registry is properly seeded with real fee data.")
        print("Position sizing will now use accurate broker fees.")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("=" * 70)
        print("\nBroker registry may be incomplete or inconsistent.")
        print("Please run: python scripts/populate_broker_registry.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
