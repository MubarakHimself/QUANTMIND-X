# Cloudzy VPS Setup Plan

## Overview

This document outlines the complete setup plan for the Cloudzy VPS, which will serve as the primary trading infrastructure for QuantMindX. The system uses a three-tier architecture with Cloudzy handling high-frequency trading operations and Contabo handling long-term storage and compute-heavy tasks.

---

## 1. Server Architecture

### Server Roles

| Server | Role | OS | Key Services |
|--------|------|----|--------------|
| **Cloudzy** | Trading Server | Windows | MT5, Strategy Router, Kelly, Hot/Warm data |
| **Contabo** | Storage + Compute | Linux | Cold storage, HMM training, Monitoring |
| **Local** | Development | Linux | IDE, strategy building, Claude Code |

### Cloudzy Specifications

**Recommended Plan: Advanced ($31.77/month)**
- 8 GB DDR5 Memory
- 4 vCPU (4.2+ GHz)
- 240 GB NVMe/SSD Storage
- 7 TB Transfer
- Windows Server (for MT5)

---

## 2. Data Pipeline Architecture

### Three-Tier Storage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUDZY VPS                                     │
│                                                                              │
│  MT5 Terminal ──tick data──▶ PostgreSQL (HOT) ──hourly──▶ DuckDB (WARM)    │
│                               1-hour           │           7-day            │
│                               retention        │           retention         │
│                                               │                             │
│                          Strategy Router ◀────┘                             │
│                          Kelly/Risk Mgmt                                     │
│                          (<5ms ZMQ latency)                                  │
│                                                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                              Daily sync (backup)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTABO VPS                                     │
│                                                                              │
│                          Parquet Files (COLD)                               │
│                          Indefinite retention                               │
│                          Historical data archive                            │
│                                                                              │
│                          + HMM Training (Saturdays 03:00 CET)               │
│                          + Grafana/Prometheus Monitoring                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Retention Policy (UPDATED)

| Tier | Database | Location | Retention | Purpose |
|------|----------|----------|-----------|---------|
| **Hot** | PostgreSQL | Cloudzy | 1 hour | Real-time tick cache |
| **Warm** | DuckDB | Cloudzy | **7 days** | Strategy Router access |
| **Cold** | Parquet | Contabo | Indefinite | Historical archive |

### Data Flow Scripts

| Script | Purpose | Schedule |
|--------|---------|----------|
| `scripts/migrate_hot_to_warm.py` | Hot → Warm migration | Hourly |
| `scripts/archive_warm_to_cold.py` | Warm → Cold archive | Daily |
| `scripts/sync_config.sh` | Config sync Cloudzy → Contabo | Every 15 min |

---

## 3. EA Lifecycle

### State Machine

```
@primal ──▶ @pending ──▶ @perfect ──▶ @live ──▶ @quarantine ──▶ @dead

Promotion Criteria:
  @primal → @pending:  20+ trades, 50% win rate, 7+ days
  @pending → @perfect: 50+ trades, 55% win rate, 30+ days, Sharpe > 1.5
  @perfect → @live:    100+ trades, 58% win rate, 60+ days, Sharpe > 2.0

Quarantine Triggers:
  - Win rate drops below 45%
  - Sharpe ratio < 0.5
  - Drawdown > 20%
  - 5 consecutive losing days
```

### Trading Modes (UPDATED)

```
PAPER = DEMO  (Same mode - not separate)
   │
   └──▶ LIVE

Criteria for PAPER/DEMO → LIVE:
  - 30+ days active
  - Sharpe > 1.5
  - Win Rate > 55%
  - Max DD < 10%
```

### EA Creation Flow

```
strategies-yt/          ──▶ GitHub ──▶ Cloudzy (sync) ──▶ BotManifest
(Local development)                              │
                                                 ▼
                                          Paper Trading
                                                 │
                                                 ▼
                                            Live Trading
```

### Key Files

| File | Purpose |
|------|---------|
| `src/router/bot_manifest.py` | EA state management, BotRegistry |
| `src/router/ea_registry.py` | EA configuration registry |
| `src/router/lifecycle_manager.py` | Tag-based progression |
| `src/router/promotion_manager.py` | Mode-based promotion |
| `src/integrations/github_ea_sync.py` | GitHub EA import |
| `src/agents/paper_trading_validator.py` | Paper trading validation |

---

## 4. Tick Data Configuration

### Data Source

**Requirement:** Use demo account tick data even for live trading (to prevent broker manipulation).

**Configuration needed:**
```bash
# .env on Cloudzy
USE_DEMO_TICKS=true
MT5_DEMO_LOGIN=12345678
MT5_DEMO_PASSWORD=your_password
MT5_DEMO_SERVER=Broker-Demo
```

### Tick Flow

```
MT5 Demo Account ──▶ Tick Stream Handler ──▶ Hot Tier (PostgreSQL)
                              │
                              ▼
                     Strategy Router (<5ms)
                              │
                              ▼
                     Kelly Position Sizing
```

---

## 5. Cloudzy Setup Checklist

### Phase 1: Server Preparation

- [ ] Purchase Cloudzy Advanced VPS ($31.77/mo)
- [ ] Note server IP and credentials
- [ ] Set up SSH access for Claude Code
- [ ] Configure Windows Firewall for required ports

### 2. Install Required Software

- [ ] Install PostgreSQL 15+ (Hot tier)
- [ ] Install Python 3.11+
- [ ] Install MetaTrader 5
- [ ] Install Docker Desktop (optional, for containerization)
- [ ] Install Git

### 3. Clone and Configure QuantMindX

- [ ] Clone repository to `C:\QuantMindX` or preferred location
- [ ] Create Python virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Copy `.env.example` to `.env`
- [ ] Configure all environment variables (see Section 6)

### 4. Set Up Databases

- [ ] Create PostgreSQL database: `quantmind_hot`
- [ ] Create DuckDB file: `/data/market_data.duckdb`
- [ ] Run database migrations
- [ ] Verify connections work

### 5. Configure Data Pipeline

- [ ] Update `scripts/migrate_hot_to_warm.py` for 1-week retention
- [ ] Update `scripts/archive_warm_to_cold.py` for daily sync to Contabo
- [ ] Set up Windows Task Scheduler for cron-like jobs:
  - Hot → Warm: Every hour
  - Warm → Cold: Daily at 03:00
  - Config sync: Every 15 minutes

### 6. Configure MT5 Integration

- [ ] Install MT5 terminal
- [ ] Log in to demo account
- [ ] Configure MT5 to allow DLL imports
- [ ] Test MT5 Python integration
- [ ] Verify tick data is flowing to Hot tier

### 7. Configure Strategy Router

- [ ] Verify ZMQ is installed (`pip install pyzmq`)
- [ ] Test sub-5ms latency requirements
- [ ] Configure routing matrix
- [ ] Test multi-symbol processing

### 8. Configure EA System

- [ ] Set up GitHub EA sync (configure repo URL, access token)
- [ ] Create `strategies-yt/` folder structure
- [ ] Test EA import flow
- [ ] Verify BotManifest creation works
- [ ] Configure Paper=Demo mode merge

### 9. Configure Monitoring

- [ ] Set up connection to Contabo Prometheus
- [ ] Configure log shipping to Contabo
- [ ] Test alerting

### 10. Test End-to-End

- [ ] Create test EA in `strategies-yt/`
- [ ] Push to GitHub
- [ ] Verify sync to Cloudzy
- [ ] Verify BotManifest creation
- [ ] Start paper trading
- [ ] Verify data flows through all tiers
- [ ] Check Contabo receives archived data

---

## 7. Configuration Changes Needed

### 7.1 Merge Paper/Demo Modes

**File:** `src/router/bot_manifest.py`

**Change:** Make PAPER and DEMO the same mode.

**Option A - Config flag:**
```python
# Add to config
PAPER_EQUALS_DEMO = True

# Modify promotion logic
if PAPER_EQUALS_DEMO:
    # Skip PAPER → DEMO promotion, go directly PAPER → LIVE
    pass
```

**Option B - Merge modes:**
```python
# Remove DEMO from TradingMode enum
class TradingMode(str, Enum):
    PAPER = "paper"  # Includes demo
    LIVE = "live"
```

### 7.2 Update Retention Period

**File:** `scripts/archive_warm_to_cold.py`

**Change:** Update retention from 30 days to 7 days.

```python
# Find and change
RETENTION_DAYS = 7  # Was 30
```

### 7.3 Configure Demo Tick Source

**File:** `src/api/tick_stream_handler.py` or config

**Change:** Add configuration to use demo account ticks for live trading.

```python
# Add configuration
USE_DEMO_TICKS = os.getenv("USE_DEMO_TICKS", "true").lower() == "true"

# Modify tick source selection
if USE_DEMO_TICKS and trading_mode == TradingMode.LIVE:
    tick_source = demo_account_ticks
```

### 7.4 Update Cron Jobs

**File:** `scripts/setup_contabo_crons.sh` (for Contabo reference)

**New Cloudzy scheduled tasks:**
```
# Hot → Warm: Every hour
0 * * * * cd /c/QuantMindX && python scripts/migrate_hot_to_warm.py

# Warm → Cold: Daily at 03:00
0 3 * * * cd /c/QuantMindX && python scripts/archive_warm_to_cold.py

# Config sync to Contabo: Every 15 minutes
*/15 * * * * cd /c/QuantMindX && bash scripts/sync_to_contabo.sh
```

---

## 8. UI/CLI Location Decision

### Options

| Option | UI Location | CLI/TUI Location | Pros | Cons |
|--------|-------------|------------------|------|------|
| **A** | Local machine | Cloudzy (SSH) | Fast development | Separate access points |
| **B** | Cloudzy | Cloudzy | Single access point | Uses server resources |
| **C** | Local + Remote access | Cloudzy | Flexibility | More complex |

### CLI/TUI Access

**File:** `src/router/cli.py` - Already exists

**Access methods:**
- SSH into Cloudzy
- Termux on phone (as mentioned)
- Windows Terminal

### Decision: PENDING

**Recommendation:** Start with Option A (UI local, CLI on Cloudzy). Migrate to Option B if needed.

---

## 9. Database Technology

### PostgreSQL vs DuckDB

| Database | Best For | Use Case |
|----------|----------|----------|
| **PostgreSQL** | Real-time writes, concurrent access, ACID | Hot tier (tick writes) |
| **DuckDB** | Analytics, aggregations, OLAP, fast queries | Warm tier (Strategy Router) |

### Recommendation: Use Both

- **Hot tier:** PostgreSQL (already configured)
- **Warm tier:** DuckDB (already configured)
- **Cold tier:** Parquet files (already configured)

---

## 10. Contabo Configuration Updates

### Update Cron Jobs

Current Contabo cron jobs need updating for new Cloudzy IP:

```bash
# SSH into Contabo
ssh root@155.133.27.86

# Update .env with Cloudzy details
nano /opt/quantmindx/.env

# Add:
CLOUDZY_HOST=<cloudzy-ip>
CLOUDZY_USER=<username>
CLOUDZY_HOT_DB_URL=postgresql://postgres:password@<cloudzy-ip>:5432/quantmind_hot
```

### Update Archive Script

Ensure `scripts/archive_warm_to_cold.py` on Contabo syncs FROM Cloudzy:

```python
# Configure to pull from Cloudzy Warm tier
WARM_DB_HOST = os.getenv("CLOUDZY_HOST", "localhost")
```

---

## 11. Security Considerations

### API Keys and Secrets

- [ ] Generate secure API keys for:
  - Contabo HMM API
  - Cloudzy Trading API
  - GitHub EA Sync
- [ ] Store in `.env` files (never commit)
- [ ] Use environment variables in production

### Network Security

- [ ] Configure firewall rules
- [ ] Restrict PostgreSQL access to localhost + Contabo IP
- [ ] Use SSL for database connections
- [ ] Set up VPN or private network if available

### Access Control

- [ ] Create non-root user for services
- [ ] Restrict SSH access
- [ ] Use key-based authentication

---

## 12. Monitoring and Alerting

### Metrics to Track

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| Tick latency | ZMQ | >5ms |
| Hot tier size | PostgreSQL | >1GB |
| Warm tier size | DuckDB | >10GB |
| EA performance | BotManifest | Sharpe <0.5 |
| Data pipeline health | Cron logs | Any failure |

### Grafana Dashboards (on Contabo)

- [ ] Create Cloudzy trading dashboard
- [ ] Add data pipeline metrics
- [ ] Configure alerts

---

## 13. Rollback Plan

If Cloudzy doesn't meet requirements:

1. **Data Backup:** Daily sync to Contabo ensures no data loss
2. **Service Migration:** All services can run on Contabo (with higher latency)
3. **EA Export:** GitHub sync allows easy EA recovery
4. **Configuration:** All configs in `.env` files, easily transferable

---

## 14. Task Checklist

### Pre-Purchase
- [x] Architecture designed
- [x] Requirements documented
- [ ] Server purchased

### Post-Purchase Setup
- [ ] Server access configured
- [ ] PostgreSQL installed
- [ ] DuckDB configured
- [ ] MT5 installed and configured
- [ ] QuantMindX cloned and configured
- [ ] Data pipeline tested
- [ ] EA registration tested
- [ ] Paper trading working
- [ ] Monitoring connected

### Configuration Changes
- [ ] Paper=Demo mode merged
- [ ] Retention updated to 7 days
- [ ] Demo tick source configured
- [ ] Cron jobs/scheduled tasks set up

### Testing
- [ ] End-to-end data flow
- [ ] Latency requirements met
- [ ] EA lifecycle working
- [ ] Paper trading functional
- [ ] Monitoring working

---

## 15. References

### Key Files

| File | Purpose |
|------|---------|
| `docs/Contabo_Deployment_Guide.md` | Contabo setup reference |
| `docs/Coding_Agent_Handoff_Plan.md` | UI integration plan |
| `docs/UI_Claude_Code_Native_Integration_Plan.md` | Claude Code migration |
| `scripts/setup_contabo_crons.sh` | Contabo cron setup |
| `.env.example` | Environment configuration template |

### Server Information

**Contabo VPS:**
- IP: `155.133.27.86`
- HMM API: `http://155.133.27.86:8001`
- Grafana: `http://155.133.27.86:3001`
- Prometheus: `http://155.133.27.86:9090`

**Cloudzy VPS:**
- IP: `<to be filled after purchase>`
- Trading API: Port 8000
- MT5 Bridge: Port 5005

---

## 16. Next Session

When continuing this work:

1. **If Cloudzy purchased:** Start with Section 5 (Setup Checklist)
2. **If not purchased:** Review architecture decisions in Section 4
3. **Configuration changes:** Follow Section 7
4. **Testing:** Follow Section 14 checklist

**Contact points:**
- Contabo SSH: `ssh root@155.133.27.86`
- Cloudzy SSH: `ssh <user>@<cloudzy-ip>` (after purchase)

---

*Document created: 2026-02-21*
*Last updated: 2026-02-21*
