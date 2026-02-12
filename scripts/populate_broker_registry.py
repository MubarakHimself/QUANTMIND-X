#!/usr/bin/env python3
"""
Broker Registry Population Script

Populates the broker_registry table with real broker profiles for
fee-aware position sizing.

This script is IDEMPOTENT:
- Safe to run multiple times
- Checks for existing brokers before creating
- Updates existing brokers with new values
- No duplicate entries or data corruption

Usage:
    # Run normally (creates/updates brokers)
    python scripts/populate_broker_registry.py
    
    # Run in dry-run mode (show what would happen, no DB changes)
    python scripts/populate_broker_registry.py --dry-run
    
    # Run with verbose output
    python scripts/populate_broker_registry.py --verbose

Broker Profiles:
    - icmarkets_raw: Raw ECN with $7/lot commission, 0.1 pip spread
    - roboforex_prime: Standard spread, no commission, 0.2 pip spread
    - mt5_default: Testing broker, 0.5 pip spread

Impact:
    The EnhancedKellyCalculator uses these real broker values for
    accurate fee-aware position sizing instead of fallback defaults.
    Pip values enable dynamic position sizing for commodities/indices.
"""

import sys
import logging
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, '/home/mubarkahimself/Desktop/QUANTMINDX')

from src.router.broker_registry import BrokerRegistryManager
from src.database.manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def populate_icmarkets_raw(dry_run: bool = False):
    """
    Create or update ICMarkets Raw ECN broker profile.
    
    Args:
        dry_run: If True, show what would happen without changing DB
    """
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("icmarkets_raw")
    if existing is not None:
        action = "[DRY RUN] Would update" if dry_run else "Updating"
        logger.info(f"{action} ICMarkets Raw broker profile...")
        if not dry_run:
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
        action = "[DRY RUN] Would create" if dry_run else "Creating"
        logger.info(f"{action} ICMarkets Raw broker profile...")
        if not dry_run:
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


def populate_roboforex_prime(dry_run: bool = False):
    """
    Create or update RoboForex Prime broker profile.
    
    Args:
        dry_run: If True, show what would happen without changing DB
    """
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("roboforex_prime")
    if existing is not None:
        action = "[DRY RUN] Would update" if dry_run else "Updating"
        logger.info(f"{action} RoboForex Prime broker profile...")
        if not dry_run:
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
        action = "[DRY RUN] Would create" if dry_run else "Creating"
        logger.info(f"{action} RoboForex Prime broker profile...")
        if not dry_run:
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


def populate_mt5_default(dry_run: bool = False):
    """
    Create or update MT5 Default broker profile for testing.
    
    Args:
        dry_run: If True, show what would happen without changing DB
    """
    broker_mgr = BrokerRegistryManager()

    # Check if broker already exists
    existing = broker_mgr.get_broker("mt5_default")
    if existing is not None:
        action = "[DRY RUN] Would update" if dry_run else "Updating"
        logger.info(f"{action} MT5 Default broker profile...")
        if not dry_run:
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
        action = "[DRY RUN] Would create" if dry_run else "Creating"
        logger.info(f"{action} MT5 Default broker profile...")
        if not dry_run:
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


def main(dry_run: bool = False, verbose: bool = False):
    """
    Main function to populate all broker profiles.
    
    Args:
        dry_run: If True, show what would happen without changing DB
        verbose: If True, enable verbose logging
    
    Returns:
        0 on success, 1 on failure
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    mode = "[DRY RUN]" if dry_run else "[PRODUCTION]"
    logger.info(f"Starting broker registry population {mode}...")

    try:
        # Populate all broker profiles
        populate_icmarkets_raw(dry_run=dry_run)
        populate_roboforex_prime(dry_run=dry_run)
        populate_mt5_default(dry_run=dry_run)

        if dry_run:
            logger.info("[DRY RUN] Broker registry population would complete without errors")
            print("\n[DRY RUN MODE] Would create/update broker profiles:")
            print("  - icmarkets_raw: IC Markets Raw Spread ($7/lot, 0.1 pip spread)")
            print("  - roboforex_prime: RoboForex Prime (no commission, 0.2 pip spread)")
            print("  - mt5_default: MT5 Default for testing (no commission, 0.5 pip spread)")
            print("\nRun without --dry-run to actually populate the database.")
        else:
            logger.info("Broker registry population complete!")
            print("\n✅ Successfully populated broker registry with 3 broker profiles:")
            print("  ✓ icmarkets_raw: IC Markets Raw Spread ($7/lot, 0.1 pip spread)")
            print("  ✓ roboforex_prime: RoboForex Prime (no commission, 0.2 pip spread)")
            print("  ✓ mt5_default: MT5 Default for testing (no commission, 0.5 pip spread)")
            print("\nBroker auto-lookup will now return real fee data instead of fallback defaults.")
        
        return 0

    except Exception as e:
        logger.error(f"Failed to populate broker registry: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate broker registry with real broker profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run (create/update brokers in database)
  python scripts/populate_broker_registry.py
  
  # Dry-run mode (see what would happen, no changes)
  python scripts/populate_broker_registry.py --dry-run
  
  # Verbose output (debug logging)
  python scripts/populate_broker_registry.py --verbose
  
  # Dry-run with verbose
  python scripts/populate_broker_registry.py --dry-run --verbose
        """
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would happen without modifying the database'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )
    
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run, verbose=args.verbose))
