# Broker Registry Initialization Checklist

## Pre-Deployment Steps

Before deploying the trading system, ensure the broker registry is properly seeded with real fee data.

---

## Setup Checklist

- [ ] **1. Database Initialization**
  ```bash
  # Initialize database with all tables
  python -c "from src.database.engine import init_db; init_db()"
  ```
  - Verify: `data/db/` directory exists
  - Check: `broker_registry` table is created

---

- [ ] **2. Run Broker Registry Seeding (Dry Run)**
  ```bash
  # First, test without making changes
  python scripts/populate_broker_registry.py --dry-run --verbose
  ```
  - Expected output: Shows 3 brokers that would be created/updated
  - No database changes made

---

- [ ] **3. Run Broker Registry Seeding (Production)**
  ```bash
  # Actually populate the database
  python scripts/populate_broker_registry.py
  ```
  - Expected output: ✅ Successfully populated with 3 brokers
  - Database now has real fee data

---

- [ ] **4. Verify Broker Data**
  ```bash
  # Quick verification
  python -c "
from src.router.broker_registry import BrokerRegistryManager

mgr = BrokerRegistryManager()

# Verify icmarkets_raw
broker = mgr.get_broker('icmarkets_raw')
assert broker is not None, 'icmarkets_raw not found'
assert broker.spread_avg == 0.1, f'Wrong spread: {broker.spread_avg}'
assert broker.commission_per_lot == 7.0, f'Wrong commission: {broker.commission_per_lot}'
print('✓ icmarkets_raw: OK')

# Verify roboforex_prime
broker = mgr.get_broker('roboforex_prime')
assert broker is not None, 'roboforex_prime not found'
assert broker.spread_avg == 0.2, f'Wrong spread: {broker.spread_avg}'
print('✓ roboforex_prime: OK')

# Verify mt5_default
broker = mgr.get_broker('mt5_default')
assert broker is not None, 'mt5_default not found'
print('✓ mt5_default: OK')

# Verify pip values
pip = mgr.get_pip_value('XAUUSD', 'icmarkets_raw')
assert pip == 1.0, f'Wrong pip value: {pip}'
print('✓ XAUUSD pip value: OK')

print('\n✅ All broker data verified!')
  "
  ```

---

- [ ] **5. Start Position Sizing System**
  ```bash
  # Now the EnhancedKellyCalculator will use real broker fees
  python -m src.router.enhanced_governor --test
  ```
  - Verify logs show broker fee data being used
  - Check: No "using default pip value" warnings

---

## Integration with Deployment

### Option A: Manual Deployment
```bash
#!/bin/bash
set -e

echo "1. Initializing database..."
python -c "from src.database.engine import init_db; init_db()"

echo "2. Populating broker registry..."
python scripts/populate_broker_registry.py

echo "3. Starting system..."
python -m uvicorn src.api.trading_endpoints:create_fastapi_app --host 0.0.0.0 --port 8000

echo "✅ System ready with real broker fees!"
```

### Option B: Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Initialize database
RUN python -c "from src.database.engine import init_db; init_db()"

# Populate broker registry
RUN python scripts/populate_broker_registry.py

# Start system
CMD ["python", "-m", "uvicorn", "src.api.trading_endpoints:create_fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option C: Kubernetes Deployment
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: broker-registry-init
spec:
  template:
    spec:
      containers:
      - name: init
        image: quantmindx:latest
        command:
        - /bin/sh
        - -c
        - |
          python -c "from src.database.engine import init_db; init_db()"
          python scripts/populate_broker_registry.py
      restartPolicy: Never
```

---

## Troubleshooting

### Issue: Script says "already exists" when I want fresh data
**Solution:** This is expected! The script is idempotent
```bash
# Re-running updates to latest values
python scripts/populate_broker_registry.py

# To force-delete and recreate, manually delete from DB:
python -c "
from src.router.broker_registry import BrokerRegistryManager
mgr = BrokerRegistryManager()
mgr.delete_broker('icmarkets_raw')
mgr.delete_broker('roboforex_prime')
mgr.delete_broker('mt5_default')
print('Deleted all brokers')
"

# Now re-run
python scripts/populate_broker_registry.py
"
```

### Issue: `ImportError: No module named 'src'`
**Solution:** Run from workspace root
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python scripts/populate_broker_registry.py
```

### Issue: Database connection error
**Solution:** Ensure database is initialized
```bash
# Initialize database first
python -c "from src.database.engine import init_db; init_db()"

# Then run seeding
python scripts/populate_broker_registry.py
```

---

## Impact Verification

After seeding, verify position sizing is accurate:

```python
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

# Before seeding (uses defaults):
# - Pip value: 10.0 (wrong for XAUUSD)
# - Commission: $0 (missing fee)
# - Position size: Wrong

# After seeding (uses real data):
calc = EnhancedKellyCalculator(
    win_rate=0.55,
    avg_win=100.0,
    avg_loss=100.0,
    broker_id="icmarkets_raw"  # Uses real: $7/lot, 0.1 pip spread
)
kelly = calc.calculate()

# Result: More conservative position sizing
# Accounts for real trading costs
# ✅ Accurate fee-aware position sizing
```

---

## Maintenance

### Monthly Broker Fee Review
```bash
# Check if broker fees have changed in production
python -c "
from src.router.broker_registry import BrokerRegistryManager

mgr = BrokerRegistryManager()

for broker_id in ['icmarkets_raw', 'roboforex_prime', 'mt5_default']:
    broker = mgr.get_broker(broker_id)
    if broker:
        print(f'{broker_id}:')
        print(f'  Spread: {broker.spread_avg} pips')
        print(f'  Commission: \${broker.commission_per_lot}/lot')
"
```

### Updating Broker Fees
```bash
# If broker changes fees, update manually:
python -c "
from src.router.broker_registry import BrokerRegistryManager

mgr = BrokerRegistryManager()

# Update spread
mgr.update_broker('icmarkets_raw', spread_avg=0.12)

# Or re-run full seeding
# python scripts/populate_broker_registry.py
"
```

---

## References

- Script: [scripts/populate_broker_registry.py](../../scripts/populate_broker_registry.py)
- Documentation: [docs/implementation/BROKER_REGISTRY_SEEDING.md](./BROKER_REGISTRY_SEEDING.md)
- Manager: [src/router/broker_registry.py](../../src/router/broker_registry.py)
- Calculator: [src/position_sizing/enhanced_kelly.py](../../src/position_sizing/enhanced_kelly.py)

---

**Last Updated:** February 12, 2026
**Status:** ✅ Ready for Deployment
