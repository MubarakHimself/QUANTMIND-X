# Broker Registry Seeding Documentation

## Overview

The **Broker Registry Seeding Script** (`scripts/populate_broker_registry.py`) populates the `broker_registry` database table with real broker profiles including pip values, average spreads, and commissions. This enables fee-aware position sizing and dynamic pip value calculation without relying on fallback default values.

---

## Purpose & Impact

### Problem Solved
Without populated broker data, the broker auto-lookup system (`BrokerRegistryManager`) returns fallback defaults:
- Default pip value: `10.0` (incorrect for many symbols)
- Default spread: `0.0` (inaccurate)
- Default commission: `0.0` (incomplete fee calculation)

This causes the **EnhancedKellyCalculator** to perform suboptimal position sizing because it cannot accurately account for:
- Real pip values per symbol (especially commodities like XAUUSD)
- Real trading commissions and spreads
- Broker-specific lot constraints

### Solution
The seeding script uses `BrokerRegistryManager` to upsert broker profiles with realistic data for three core brokers:

1. **icmarkets_raw** - Raw ECN broker with tight spreads and per-lot commissions
2. **roboforex_prime** - Standard spread broker with no commission
3. **mt5_default** - Generic MT5 testing broker

---

## How to Run

### Quick Start
```bash
# Navigate to workspace root
cd /home/mubarkahimself/Desktop/QUANTMINDX

# Run the seeding script
python scripts/populate_broker_registry.py
```

### Expected Output
```
Starting broker registry population...
Successfully populated broker registry with 3 broker profiles:
  - icmarkets_raw: IC Markets Raw Spread ($7/lot, 0.1 pip spread)
  - roboforex_prime: RoboForex Prime (no commission, 0.2 pip spread)
  - mt5_default: MT5 Default for testing (no commission, 0.5 pip spread)
```

### Logging
Enable debug logging to see detailed operation logs:
```bash
# Run with debug output
PYTHONPATH=/home/mubarkahimself/Desktop/QUANTMINDX python -u scripts/populate_broker_registry.py 2>&1 | grep -E "^(Starting|Creating|Updating|Successfully)"
```

---

## When to Run

### Initial Setup
Run this script **once** after:
1. Database initialization (`src/database/engine.py`)
2. BrokerRegistry table creation
3. First application deployment

### After Broker Changes
Re-run the script whenever you need to:
- Update broker commission or spread values
- Add new symbols to pip_values mapping
- Add new brokers to the system

### Idempotent Behavior
The script is **idempotent** — running it multiple times is safe:
- Checks if broker already exists before creating
- Updates existing brokers with new values if they already exist
- No duplicate entries or data corruption

---

## Script Details

### Broker Profiles

#### 1. **icmarkets_raw**
```python
broker_id = "icmarkets_raw"
broker_name = "IC Markets Raw Spread"
spread_avg = 0.1  # 0.1 pips average
commission_per_lot = 7.0  # $7 per standard lot
tags = ["RAW_ECN", "SCALPER_FRIENDLY", "LOW_SPREAD"]

pip_values = {
    "EURUSD": 10.0,   # EUR major
    "GBPUSD": 10.0,   # GBP major
    "USDJPY": 9.09,   # JPY pair (smaller pip)
    "XAUUSD": 1.0,    # Gold: 1 pip = 1 point
    "XAGUSD": 50.0,   # Silver: 1 pip = 0.01 (50x)
    "NAS100": 1.0,    # Nasdaq index
    "SPX500": 1.0,    # S&P 500 index
    "US30": 1.0,      # Dow Jones
    "GER40": 1.0,     # DAX
}
```

#### 2. **roboforex_prime**
```python
broker_id = "roboforex_prime"
broker_name = "RoboForex Prime"
spread_avg = 0.2  # 0.2 pips average
commission_per_lot = 0.0  # No per-lot commission
tags = ["STANDARD", "NO_COMMISSION"]

pip_values = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 9.09,
    "XAUUSD": 1.0,
    "XAGUSD": 50.0,
}
```

#### 3. **mt5_default**
```python
broker_id = "mt5_default"
broker_name = "MT5 Default (Testing)"
spread_avg = 0.5  # 0.5 pips average (wider for safety)
commission_per_lot = 0.0  # No commission
tags = ["STANDARD", "TESTING"]

pip_values = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 9.09,
    "XAUUSD": 10.0,  # Fallback value for testing
    "XAGUSD": 10.0,  # Fallback value for testing
}
```

---

## Integration with Position Sizing

### Before Seeding (Using Defaults)
```python
from src.router.broker_registry import BrokerRegistryManager

manager = BrokerRegistryManager()
pip_value = manager.get_pip_value("XAUUSD", "icmarkets_raw")
# ❌ Returns 10.0 (incorrect default)

spread = manager.get_spread("icmarkets_raw")
# ❌ Returns 0.0 (not found, wrong fee data)
```

### After Seeding (Using Real Data)
```python
from src.router.broker_registry import BrokerRegistryManager

manager = BrokerRegistryManager()
pip_value = manager.get_pip_value("XAUUSD", "icmarkets_raw")
# ✅ Returns 1.0 (correct real value)

spread = manager.get_spread("icmarkets_raw")
# ✅ Returns 0.1 (real market data)

commission = manager.get_commission("icmarkets_raw")
# ✅ Returns 7.0 (real per-lot fee)
```

### Enhanced Kelly Calculator
The **EnhancedKellyCalculator** uses these broker values:

```python
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

calc = EnhancedKellyCalculator(
    win_rate=0.55,
    avg_win=100.0,
    avg_loss=100.0,
    broker_id="icmarkets_raw"  # Uses real fee data from seeded registry
)

kelly_fraction = calc.calculate()
# Now accounts for $7/lot commission + 0.1 pip spread
# ✅ More accurate position sizing
```

---

## Database Schema

The script populates the `broker_registry` table:

```sql
CREATE TABLE broker_registry (
    id INTEGER PRIMARY KEY,
    broker_id VARCHAR(255) UNIQUE NOT NULL,
    broker_name VARCHAR(255) NOT NULL,
    spread_avg FLOAT DEFAULT 0.0,
    commission_per_lot FLOAT DEFAULT 0.0,
    lot_step FLOAT DEFAULT 0.01,
    min_lot FLOAT DEFAULT 0.01,
    max_lot FLOAT DEFAULT 100.0,
    pip_values JSON,           -- {"EURUSD": 10.0, "XAUUSD": 1.0, ...}
    preference_tags JSON,      -- ["RAW_ECN", "LOW_SPREAD", ...]
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## Adding New Brokers

To add a new broker to the seeding script:

### Step 1: Update the Script
```python
def populate_my_broker():
    """Create My Broker broker profile."""
    broker_mgr = BrokerRegistryManager()
    
    existing = broker_mgr.get_broker("my_broker_id")
    if existing is not None:
        logger.info("My Broker already exists, updating...")
        broker_mgr.update_broker(
            "my_broker_id",
            spread_avg=0.3,
            commission_per_lot=5.0,
            pip_values={
                "EURUSD": 10.0,
                "XAUUSD": 1.0,
                # Add all your symbols
            },
            preference_tags=["STANDARD"]
        )
    else:
        logger.info("Creating My Broker profile...")
        broker_mgr.create_broker(
            broker_id="my_broker_id",
            broker_name="My Broker Name",
            spread_avg=0.3,
            commission_per_lot=5.0,
            pip_values={"EURUSD": 10.0, "XAUUSD": 1.0},
            preference_tags=["STANDARD"]
        )
```

### Step 2: Call in main()
```python
def main():
    """Main function to populate all broker profiles."""
    try:
        populate_icmarkets_raw()
        populate_roboforex_prime()
        populate_mt5_default()
        populate_my_broker()  # ← Add here
        
        logger.info("Broker registry population complete!")
        return 0
    except Exception as e:
        logger.error(f"Failed to populate broker registry: {e}")
        return 1
```

### Step 3: Re-run
```bash
python scripts/populate_broker_registry.py
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:** Run from workspace root directory
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python scripts/populate_broker_registry.py
```

### Issue: Database connection error
**Solution:** Ensure database is initialized
```bash
# Check if database exists
ls -la data/db/

# If not, initialize via your database setup:
python -c "from src.database.engine import init_db; init_db()"
```

### Issue: Script runs but shows "already exists" for all brokers
**Solution:** This is normal! The script is idempotent
- First run: Creates all 3 brokers
- Subsequent runs: Updates existing brokers with same data
- This is by design — safe to re-run anytime

### Issue: BrokerRegistryManager returns defaults instead of seeded values
**Solution:** Verify script ran successfully
```bash
# Check database directly
python -c "
from src.router.broker_registry import BrokerRegistryManager
mgr = BrokerRegistryManager()
broker = mgr.get_broker('icmarkets_raw')
print(f'Broker found: {broker is not None}')
print(f'Spread: {broker.spread_avg if broker else None}')
print(f'Commission: {broker.commission_per_lot if broker else None}')
"
```

---

## Validation

### Verify Seeding Success
```bash
python -c "
from src.router.broker_registry import BrokerRegistryManager
from src.database.manager import DatabaseManager

mgr = BrokerRegistryManager()

# Check all three brokers exist
for broker_id in ['icmarkets_raw', 'roboforex_prime', 'mt5_default']:
    broker = mgr.get_broker(broker_id)
    assert broker is not None, f'{broker_id} not found'
    print(f'✓ {broker_id}: spread={broker.spread_avg}, commission={broker.commission_per_lot}')

print('✓ All brokers seeded successfully!')
"
```

### Integration Test
```bash
# Verify position sizing uses real broker data
python -c "
from src.router.broker_registry import BrokerRegistryManager

mgr = BrokerRegistryManager()

# Test pip value lookup
xau_pip = mgr.get_pip_value('XAUUSD', 'icmarkets_raw')
assert xau_pip == 1.0, f'Expected 1.0, got {xau_pip}'
print(f'✓ XAUUSD pip value: {xau_pip}')

# Test commission lookup
commission = mgr.get_commission('icmarkets_raw')
assert commission == 7.0, f'Expected 7.0, got {commission}'
print(f'✓ Commission: {commission}')

# Test spread lookup
spread = mgr.get_spread('icmarkets_raw')
assert spread == 0.1, f'Expected 0.1, got {spread}'
print(f'✓ Average spread: {spread}')
"
```

---

## Maintenance

### Regular Updates
Update broker data quarterly or when market conditions change:
```bash
# Update icmarkets_raw spread to 0.15
python -c "
from src.router.broker_registry import BrokerRegistryManager
mgr = BrokerRegistryManager()
mgr.update_broker('icmarkets_raw', spread_avg=0.15)
print('Updated icmarkets_raw spread to 0.15')
"
```

### Backup Before Updates
```bash
# Backup existing broker data
python -c "
from src.router.broker_registry import BrokerRegistryManager
import json
mgr = BrokerRegistryManager()
brokers = mgr.list_all_brokers()
with open('broker_registry_backup.json', 'w') as f:
    json.dump(brokers, f, indent=2)
print('Backed up broker registry')
"
```

---

## References

- **BrokerRegistryManager**: [src/router/broker_registry.py](../../src/router/broker_registry.py)
- **EnhancedKellyCalculator**: [src/position_sizing/enhanced_kelly.py](../../src/position_sizing/enhanced_kelly.py)
- **Enhanced Governor**: [src/router/enhanced_governor.py](../../src/router/enhanced_governor.py)
- **Database Models**: [src/database/models.py](../../src/database/models.py)

---

**Last Updated:** February 12, 2026
**Status:** ✅ Complete and Idempotent
