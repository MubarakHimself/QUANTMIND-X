# Code Fixes Required - Architecture Alignment

This document outlines the code changes needed to align the QuantMindX codebase with the architecture decisions made on 2026-02-21.

---

## 1. Merge Paper and Demo Modes

**Decision:** Paper and Demo are the same mode. No separate PAPER → DEMO promotion.

### 1.1 Update TradingMode Enum

**File:** `src/router/bot_manifest.py`

```python
# BEFORE
class TradingMode(str, Enum):
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"

# AFTER
class TradingMode(str, Enum):
    PAPER = "paper"  # Includes demo - same thing
    LIVE = "live"
```

### 1.2 Update Promotion Logic

**File:** `src/router/promotion_manager.py`

```python
# BEFORE: PAPER → DEMO → LIVE (3 stages)
# AFTER:  PAPER → LIVE (2 stages)

# Find and remove DEMO-related promotion logic
# Update promotion criteria to go directly PAPER → LIVE

PROMOTION_CRITERIA = {
    "paper_to_live": {
        "min_trades": 30,
        "min_days": 30,
        "min_sharpe": 1.5,
        "min_win_rate": 0.55,
        "max_drawdown": 0.10
    }
}
```

### 1.3 Update BotManifest Defaults

**File:** `src/router/bot_manifest.py`

```python
# Ensure new bots default to PAPER mode
trading_mode: TradingMode = TradingMode.PAPER

# Remove any DEMO references in defaults
```

### 1.4 Update Tests

**Files:** `tests/router/test_*.py`

```python
# Remove tests for PAPER → DEMO promotion
# Update tests to expect PAPER → LIVE directly
```

---

## 2. Update Data Retention to 7 Days

**Decision:** Warm tier retention reduced from 30 days to 7 days.

### 2.1 Update Archive Script

**File:** `scripts/archive_warm_to_cold.py`

```python
# BEFORE
RETENTION_DAYS = 30

# AFTER
RETENTION_DAYS = 7

# Also update the cutoff calculation
cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
```

### 2.2 Update HMM Training Script

**File:** `scripts/schedule_hmm_training.py`

```python
# Update min samples calculation based on 7-day retention
# Ensure training doesn't fail with shorter retention
```

### 2.3 Update Contabo Cron Job

**File:** `scripts/setup_contabo_crons.sh` (already on server)

```bash
# The cron job is already daily, but add comment about 7-day retention
# Warm to Cold Archiving - Daily 03:00 CET (7-day retention)
0 3 * * * cd /opt/quantmindx && python scripts/archive_warm_to_cold.py >> /var/log/quantmindx/archive_warm_cold.log 2>&1
```

---

## 3. Configure Demo Tick Source for Live Trading

**Decision:** Use demo account tick data even when live trading (to prevent broker manipulation).

### 3.1 Add Environment Variable

**File:** `.env.example`

```bash
# Tick Data Configuration
USE_DEMO_TICKS=true
MT5_DEMO_LOGIN=
MT5_DEMO_PASSWORD=
MT5_DEMO_SERVER=
```

### 3.2 Update Tick Stream Handler

**File:** `src/api/tick_stream_handler.py`

```python
import os

# Add configuration
USE_DEMO_TICKS = os.getenv("USE_DEMO_TICKS", "true").lower() == "true"

# Modify tick source selection
def get_tick_source(trading_mode: TradingMode):
    """
    Get the appropriate tick data source.

    If USE_DEMO_TICKS is true, always use demo account ticks
    to prevent broker manipulation of live data.
    """
    if USE_DEMO_TICKS:
        return get_demo_tick_connection()
    else:
        if trading_mode == TradingMode.LIVE:
            return get_live_tick_connection()
        else:
            return get_demo_tick_connection()
```

### 3.3 Update MT5 Client

**File:** `src/risk/integrations/mt5_client.py`

```python
# Add method to maintain separate connections
class MT5Client:
    def __init__(self, use_demo_ticks: bool = True):
        self.use_demo_ticks = use_demo_ticks
        self.demo_connection = None
        self.live_connection = None

    def get_tick_data(self, symbol: str):
        """Always use demo connection if USE_DEMO_TICKS is true."""
        if self.use_demo_ticks and self.demo_connection:
            return self.demo_connection.get_tick(symbol)
        # Fallback to live connection
        return self.live_connection.get_tick(symbol)
```

---

## 4. Update Database Configuration

**Decision:** Hot and Warm tiers on Cloudzy, Cold on Contabo.

### 4.1 Update Environment Variables

**File:** `.env.example`

```bash
# Hot Tier (PostgreSQL) - Cloudzy
HOT_DB_URL=postgresql://postgres:password@localhost:5432/quantmind_hot
CLOUDZY_HOT_DB_URL=postgresql://postgres:password@cloudzy-ip:5432/quantmind_hot

# Warm Tier (DuckDB) - Cloudzy
WARM_DB_PATH=/data/market_data.duckdb

# Cold Tier (Parquet) - Contabo
COLD_STORAGE_PATH=/data/cold_storage
CONTABO_COLD_STORAGE=root@155.133.27.86:/data/cold_storage
```

### 4.2 Update Migration Script

**File:** `scripts/migrate_hot_to_warm.py`

```python
# Update to use CLOUDZY_HOT_DB_URL when available
def get_hot_connection():
    from sqlalchemy import create_engine

    hot_db_url = os.environ.get("CLOUDZY_HOT_DB_URL") or os.environ.get("HOT_DB_URL")
    if not hot_db_url:
        raise ValueError("HOT_DB_URL or CLOUDZY_HOT_DB_URL not set")

    return create_engine(hot_db_url)
```

### 4.3 Update Archive Script

**File:** `scripts/archive_warm_to_cold.py`

```python
# Update to sync to Contabo
CONTABO_COLD_STORAGE = os.getenv("CONTABO_COLD_STORAGE")

def sync_to_contabo(archive_path: str):
    """Sync archived data to Contabo cold storage."""
    if CONTABO_COLD_STORAGE:
        import subprocess
        subprocess.run([
            "rsync", "-avz", archive_path, CONTABO_COLD_STORAGE
        ])
```

---

## 5. Add UI Location Configuration

**Decision:** UI on local machine, CLI/TUI on Cloudzy. Need configuration for both.

### 5.1 Add API Endpoint Configuration

**File:** `quantmind-ide/src/lib/config/api.ts` (create if needed)

```typescript
// API endpoint configuration
export const API_CONFIG = {
  // Local development
  LOCAL_API_URL: 'http://localhost:8000',

  // Cloudzy production (for data access)
  CLOUDZY_API_URL: 'http://<cloudzy-ip>:8000',

  // Contabo HMM API
  CONTABO_HMM_API: 'http://155.133.27.86:8001',

  // Which API to use
  get API_URL() {
    return process.env.NODE_ENV === 'production'
      ? this.CLOUDZY_API_URL
      : this.LOCAL_API_URL;
  }
};
```

### 5.2 Update Agent Configuration

**File:** `quantmind-ide/src/lib/agents/claudeCodeAgent.ts`

```typescript
// Add configuration for remote vs local agent execution
const AGENT_CONFIG = {
  // Local development - agents run locally
  LOCAL_AGENT_URL: 'http://localhost:8000/api/v2',

  // Production - could point to Cloudzy or local
  get AGENT_URL() {
    return import.meta.env.VITE_AGENT_URL || this.LOCAL_AGENT_URL;
  }
};
```

---

## 6. Summary of Files to Modify

| File | Change |
|------|--------|
| `src/router/bot_manifest.py` | Merge Paper/Demo modes |
| `src/router/promotion_manager.py` | Remove DEMO promotion stage |
| `scripts/archive_warm_to_cold.py` | Change retention to 7 days |
| `src/api/tick_stream_handler.py` | Add USE_DEMO_TICKS config |
| `src/risk/integrations/mt5_client.py` | Separate demo/live connections |
| `.env.example` | Add new environment variables |
| `scripts/migrate_hot_to_warm.py` | Support CLOUDZY_HOT_DB_URL |
| `quantmind-ide/src/lib/config/api.ts` | API endpoint configuration |

---

## 7. Testing Checklist

After making changes:

- [ ] Paper mode creates bots correctly
- [ ] No DEMO mode references in UI or logs
- [ ] Retention is 7 days (check archive script)
- [ ] Demo ticks used when USE_DEMO_TICKS=true
- [ ] Migration scripts work with Cloudzy URLs
- [ ] UI connects to correct API (local vs production)

---

## 8. Quick Reference

### Environment Variables to Add

```bash
# .env
USE_DEMO_TICKS=true
CLOUDZY_HOT_DB_URL=postgresql://postgres:password@<cloudzy-ip>:5432/quantmind_hot
CONTABO_COLD_STORAGE=root@155.133.27.86:/data/cold_storage
RETENTION_DAYS=7
```

### Key Configuration Values

| Setting | Value |
|---------|-------|
| Trading modes | PAPER, LIVE (no DEMO) |
| Warm retention | 7 days |
| Tick source | Demo account (always) |
| Hot DB | PostgreSQL on Cloudzy |
| Warm DB | DuckDB on Cloudzy |
| Cold storage | Parquet on Contabo |

---

*Document created: 2026-02-21*
