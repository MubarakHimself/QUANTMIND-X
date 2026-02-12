#!/usr/bin/env python3
"""
Broker Registry Population Script

Populates the broker_registry table with real broker profiles for
fee-aware position sizing.

Usage:
    python scripts/populate_broker_registry.py
"""

import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, '/home/mubarkahimself/Desktop/QUANTMINDX')

from src.router.broker_registry import BrokerRegistryManager
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


def populate_icmarkets_raw():
    """Create ICMarkets Raw ECN broker profile."""
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("icmarkets_raw")
    if existing is not None:
        logger.info("ICMarkets Raw broker already exists, updating...")
        broker_mgr.update_broker(
            "icmarkets_raw",
            spread_avg=0.1,  # 0.1 pips average spread
            commission_per_lot=7.0,  # $7 per standard lot
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 1.0,
                "XAGUSD": 50.0,
                "NAS100": 1.0,
                "SPX500": 1.0,
                "US30": 1.0,
                "GER40": 1.0,
            },
            preference_tags=["RAW_ECN", "SCALPER_FRIENDLY", "LOW_SPREAD"]
        )
    else:
        logger.info("Creating ICMarkets Raw broker profile...")
        broker_mgr.create_broker(
            broker_id="icmarkets_raw",
            broker_name="IC Markets Raw Spread",
            spread_avg=0.1,
            commission_per_lot=7.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 1.0,
                "XAGUSD": 50.0,
                "NAS100": 1.0,
                "SPX500": 1.0,
                "US30": 1.0,
                "GER40": 1.0,
            },
            preference_tags=["RAW_ECN", "SCALPER_FRIENDLY", "LOW_SPREAD"]
        )


def populate_roboforex_prime():
    """Create RoboForex Prime broker profile."""
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("roboforex_prime")
    if existing is not None:
        logger.info("RoboForex Prime broker already exists, updating...")
        broker_mgr.update_broker(
            "roboforex_prime",
            spread_avg=0.2,
            commission_per_lot=0.0,  # No commission
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 1.0,
                "XAGUSD": 50.0,
            },
            preference_tags=["STANDARD", "NO_COMMISSION"]
        )
    else:
        logger.info("Creating RoboForex Prime broker profile...")
        broker_mgr.create_broker(
            broker_id="roboforex_prime",
            broker_name="RoboForex Prime",
            spread_avg=0.2,
            commission_per_lot=0.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 1.0,
                "XAGUSD": 50.0,
            },
            preference_tags=["STANDARD", "NO_COMMISSION"]
        )


def populate_mt5_default():
    """Create MT5 Default broker profile for testing."""
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("mt5_default")
    if existing is not None:
        logger.info("MT5 Default broker already exists, updating...")
        broker_mgr.update_broker(
            "mt5_default",
            spread_avg=0.5,  # 0.5 pips average spread
            commission_per_lot=0.0,  # No commission
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 10.0,  # Default fallback value
                "XAGUSD": 10.0,  # Default fallback value
            },
            preference_tags=["STANDARD", "TESTING"]
        )
    else:
        logger.info("Creating MT5 Default broker profile...")
        broker_mgr.create_broker(
            broker_id="mt5_default",
            broker_name="MT5 Default (Testing)",
            spread_avg=0.5,
            commission_per_lot=0.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 10.0,  # Default fallback value
                "XAGUSD": 10.0,  # Default fallback value
            },
            preference_tags=["STANDARD", "TESTING"]
        )


def main():
    """Main function to populate all broker profiles."""
    logger.info("Starting broker registry population...")

    try:
        # Populate all broker profiles
        populate_icmarkets_raw()
        populate_roboforex_prime()
        populate_mt5_default()

        logger.info("Broker registry population complete!")
        print("Successfully populated broker registry with 3 broker profiles:")
        print("  - icmarkets_raw: IC Markets Raw Spread ($7/lot, 0.1 pip spread)")
        print("  - roboforex_prime: RoboForex Prime (no commission, 0.2 pip spread)")
        print("  - mt5_default: MT5 Default for testing (no commission, 0.5 pip spread)")
        return 0

    except Exception as e:
        logger.error(f"Failed to populate broker registry: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
