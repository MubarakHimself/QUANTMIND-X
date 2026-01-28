#!/usr/bin/env python3
"""
Test script to verify package imports for Enhanced Kelly Position Sizing
"""

import sys
import importlib

def test_imports():
    """Test that all required imports work correctly"""
    print("Testing package imports for Enhanced Kelly Position Sizing...")

    # Test risk module imports
    try:
        import risk
        print("✓ risk module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import risk module: {e}")
        return False

    # Test physics submodules
    try:
        from risk.physics import chaos_sensor, correlation_sensor, ising_sensor
        print("✓ risk.physics imports successful")
    except ImportError as e:
        print(f"✗ Failed to import risk.physics: {e}")
        return False

    # Test sizing submodules
    try:
        from risk.sizing import kelly_engine, monte_carlo_validator
        print("✓ risk.sizing imports successful")
    except ImportError as e:
        print(f"✗ Failed to import risk.sizing: {e}")
        return False

    # Test models
    try:
        from risk.models import (
            market_physics,
            position_sizing_result,
            sizing_recommendation,
            strategy_performance
        )
        print("✓ risk.models imports successful")
    except ImportError as e:
        print(f"✗ Failed to import risk.models: {e}")
        return False

    print("\nAll imports verified successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)