# QuantMindX Production Deployment Guide

# QuantMindX Production Deployment Guide

> **Scope:** This guide covers the full deployment of QuantMindX across two VPS instances (Cloudzy Trading VPS + Contabo HMM/Storage VPS) and your local laptop. It is grounded in the actual scripts, Docker Compose files, and configuration files in the codebase.

---

## 📐 Infrastructure Overview

```mermaid
graph TD
    A[Local Laptop<br/>QuantMind IDE - Svelte UI<br/>Browser - Grafana Cloud<br/>SSH Client] -->|WebSocket + HTTP API| B
    A -->|SSH Tunnel| B
    A -->|SSH| C

    B[Cloudzy VPS - Trading Server<br/>12GB RAM · 4 vCPU 4.2GHz<br/>300GB NVMe · 40Gbps<br/>Windows RDP]
    C[Contabo VPS - HMM + Storage<br/>HMM Training + Inference<br/>Warm + Cold Data Tiers]

    B -->|HMM Regime API :8001| C
    B -->|Hot→Warm Migration| C
    C -->|Model Sync| B

    B -->|Prometheus remote_write| D
    C -->|Prometheus remote_write| D
    B -->|Promtail logs| D
    C -->|Promtail logs| D

    D[Grafana Cloud<br/>Prometheus · Loki · Grafana]
```

### VPS Roles

| Component | Cloudzy VPS | Contabo VPS |
|---|---|---|
| **Primary Role** | Low-latency trading execution | HMM training + cold storage |
| **Latency Target** | ≤5ms tick-to-decision | Not latency-sensitive |
| **Docker Compose** | `file:docker-compose.production.yml` | `file:docker-compose.contabo.yml` |
| **Crontab** | `file:config/crontab.cloudzy` | `file:config/crontab.contabo` |
| **Services** | API, MT5 Bridge, PageIndex (×3), Prometheus Agent, Promtail | HMM Inference API, HMM Scheduler, Prometheus Agent, Promtail |
| **Data Tier** | HOT (1-hour retention) | WARM (7–30 days) + COLD (30+ days) |
| **Access** | RDP + SSH | SSH only |

### Port Reference

| Port | Service | VPS |
|---|---|---|
| `8000` | QuantMind API | Cloudzy |
| `5005` | MT5 Bridge | Cloudzy |
| `3000` | PageIndex Articles | Cloudzy |
| `3001` | PageIndex Books | Cloudzy |
| `3002` | PageIndex Logs | Cloudzy |
| `9090` | API Prometheus metrics | Cloudzy |
| `9091` | MT5 Bridge Prometheus metrics | Cloudzy |
| `9099` | Prometheus Agent UI | Cloudzy |
| `9080` | Promtail health | Cloudzy |
| `8001` | HMM Inference API | Contabo |
| `9093` | HMM Scheduler metrics | Contabo |
| `9099` | Prometheus Agent UI | Contabo |

---

## 🔑 Prerequisites

Before starting, gather the following credentials and accounts:

### Accounts Required

| Service | Purpose | URL |
|---|---|---|
| **Grafana Cloud** | Centralized monitoring | https://grafana.com/auth/sign-up |
| **GitHub** | Code repository + EA sync | https://github.com |
| **OpenRouter** | LLM provider for agents | https://openrouter.ai |
| **Firecrawl** | Web scraping for knowledge base | https://firecrawl.dev |
| **MT5 Broker** | Demo account for paper trading | Your chosen broker |

### API Keys to Collect

| Key | Where to Find | `.env` Variable |
|---|---|---|
| Grafana Prometheus URL | Grafana Cloud → Stack → Configuration → Details | `GRAFANA_PROMETHEUS_URL` |
| Grafana Loki URL | Grafana Cloud → Stack → Configuration → Details | `GRAFANA_LOKI_URL` |
| Grafana Instance ID | Grafana Cloud → Stack → Configuration → Details | `GRAFANA_INSTANCE_ID` |
| Grafana API Key | Grafana Cloud → Configuration → API Keys → MetricsPublisher role | `GRAFANA_API_KEY` |
| Firecrawl API Key | https://firecrawl.dev/dashboard | `FIRECRAWL_API_KEY` |
| OpenRouter API Key | https://openrouter.ai/keys | `OPENROUTER_API_KEY` |
| MT5 Demo Login | Your broker's demo account | `MT5_ACCOUNT_ID` |
| Cross-VPS API Key | Generate a strong random string | `CONTABO_HMM_API_KEY` |

### SSH Key Setup (Between VPS Instances)

Generate a dedicated SSH key on the Cloudzy VPS for Contabo access:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/contabo_id_rsa -C "cloudzy-to-contabo"
ssh-copy-id -i ~/.ssh/contabo_id_rsa.pub root@<CONTABO_IP>
```

Set in `.env` on Cloudzy VPS:
```
CONTABO_SSH_KEY_PATH=/root/.ssh/contabo_id_rsa
```

---

## 🖥️ PART 1: Cloudzy VPS Setup (Trading Server)

### Step 1.1 — Clone the Repository

```bash
git clone https://github.com/MubarakHimself/QUANTMIND-X.git /opt/quantmindx
cd /opt/quantmindx
```

### Step 1.2 — Run the Automated Installer

The `file:scripts/install.sh` script handles everything in one go:

```bash
chmod +x scripts/install.sh
sudo bash scripts/install.sh
```

**What the installer does (11 steps):**

| Step | Action |
|---|---|
| 1 | Checks sudo privileges |
| 2 | Installs system dependencies (Python 3.11, Docker, Git, curl, jq) |
| 3 | Clones or detects existing repository |
| 4 | Creates Python virtual environment and installs `requirements.txt` |
| 5 | Copies `.env.example` → `.env` and prompts for Grafana Cloud + MT5 + Firecrawl keys |
| 6 | Runs `scripts/validate_env.sh` to confirm all variables are set |
| 7 | Creates `/opt/quantmindx/snapshots` directory for rollback |
| 8 | Creates `data/logs/` and `logs/` directories |
| 9 | Installs `quantmind` CLI to `/usr/local/bin/quantmind` |
| 10 | Pulls Docker images and starts all services via `docker-compose.production.yml` |
| 11 | Runs health checks on all services |

### Step 1.3 — Configure `.env` Manually (If Needed)

If the installer was skipped or you need to update values, edit `.env` directly. All required variables are documented in `file:.env.example`:

**Critical variables for Cloudzy VPS:**

```
# Grafana Cloud
GRAFANA_PROMETHEUS_URL=https://prometheus-prod-XX-prod-XX-XX.grafana.net
GRAFANA_LOKI_URL=https://logs-prod-XX.grafana.net
GRAFANA_INSTANCE_ID=your_instance_id
GRAFANA_API_KEY=your_api_key

# MT5 Bridge
MT5_VPS_HOST=localhost
MT5_VPS_PORT=5555
MT5_ACCOUNT_ID=your_demo_login

# Multi-VPS Sync
CONTABO_HOST=your_contabo_ip
CONTABO_USER=root
CONTABO_SSH_KEY_PATH=/root/.ssh/contabo_id_rsa
CONTABO_HMM_API_URL=http://your_contabo_ip:8001
CONTABO_HMM_API_KEY=your_cross_vps_api_key

# HOT Tier Database
HOT_DB_URL=postgresql://postgres:password@localhost:5432/quantmind_hot

# Paper Trading
PAPER_TRADING_REDIS_HOST=localhost
PAPER_TRADING_REDIS_PORT=6379
```

Validate after editing:
```bash
bash scripts/validate_env.sh
```

### Step 1.4 — Verify Services Are Running

```bash
quantmind status
```

Expected output:
```
✅ quantmind-api healthy
✅ mt5-bridge healthy
✅ prometheus-agent healthy
✅ promtail healthy
✅ pageindex-articles healthy
✅ pageindex-books healthy
✅ pageindex-logs healthy
All services healthy
```

### Step 1.5 — Install Crontab

```bash
crontab config/crontab.cloudzy
```

This installs one cron job:
- **Every 15 minutes:** `scripts/sync_config.sh` — pulls latest config from GitHub and restarts affected services

### Step 1.6 — Install Systemd Services (Optional)

For services that should survive Docker restarts:

```bash
sudo cp systemd/quantmind-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantmind-api
sudo systemctl start quantmind-api
```

### Step 1.7 — Install TUI

```bash
bash scripts/setup_tui_ssh.sh
source ~/.bashrc
```

Start the TUI:
```bash
quantmind-tui
```

Install as a background service:
```bash
sudo cp systemd/quantmind-tui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantmind-tui
sudo systemctl start quantmind-tui
```

To attach to the running TUI session via RDP or SSH:
```bash
tmux attach -t quantmind-tui
```

---

## 🖥️ PART 2: Contabo VPS Setup (HMM + Cold Storage)

### Step 2.1 — Clone the Repository

```bash
git clone https://github.com/MubarakHimself/QUANTMIND-X.git /opt/quantmindx
cd /opt/quantmindx
```

### Step 2.2 — Configure `.env`

Copy and edit the environment file:

```bash
cp .env.example .env
```

**Critical variables for Contabo VPS:**

```
# Grafana Cloud (same as Cloudzy)
GRAFANA_PROMETHEUS_URL=...
GRAFANA_LOKI_URL=...
GRAFANA_INSTANCE_ID=...
GRAFANA_API_KEY=...

# Cross-VPS API Key (must match Cloudzy VPS)
CONTABO_HMM_API_KEY=your_cross_vps_api_key

# Cloudzy VPS reference (for data migration)
CLOUDZY_HOST=your_cloudzy_ip
CLOUDZY_HOT_DB_URL=postgresql://postgres:password@your_cloudzy_ip:5432/quantmind_hot

# Storage Paths
WARM_DB_PATH=/data/market_data.duckdb
COLD_STORAGE_PATH=/data/cold_storage
```

### Step 2.3 — Create Storage Directories

```bash
sudo mkdir -p /data/hmm/models
sudo mkdir -p /data/hmm/metadata
sudo mkdir -p /data/cold_storage
sudo mkdir -p /var/log/quantmindx
```

### Step 2.4 — Start Contabo Services

```bash
docker compose -f docker-compose.contabo.yml up -d
```

**Services started:**

| Container | Port | Purpose |
|---|---|---|
| `quantmind-hmm-inference-api` | `8001` | Serves regime detection to Cloudzy VPS |
| `quantmind-hmm-scheduler` | `9093` | Runs weekly HMM training |
| `quantmind-prometheus-agent-contabo` | `9099` | Ships metrics to Grafana Cloud |
| `quantmind-promtail` | `9080` | Ships logs to Grafana Cloud |

### Step 2.5 — Install Crontab

```bash
crontab config/crontab.contabo
```

**Cron jobs installed:**

| Schedule | Job | Purpose |
|---|---|---|
| Every hour | `scripts/migrate_hot_to_warm.py` | Migrates tick data from Cloudzy HOT tier to Contabo WARM tier |
| Daily at 03:00 UTC | `scripts/archive_warm_to_cold.py` | Archives WARM tier data to COLD tier |
| Saturday at 02:00 UTC | `scripts/schedule_hmm_training.py --run-now` | Weekly HMM model retraining |
| Every 15 minutes | `scripts/sync_config.sh` | Config sync from GitHub |

### Step 2.6 — Verify HMM Inference API

From the Cloudzy VPS, test the connection:

```bash
curl -H "X-API-Key: your_cross_vps_api_key" http://your_contabo_ip:8001/health
curl -H "X-API-Key: your_cross_vps_api_key" http://your_contabo_ip:8001/api/hmm/regime
```

Expected response:
```json
{"regime": "TRENDING_BULL", "confidence": 0.85, "timestamp": "..."}
```

---

## 💻 PART 3: Local Laptop Setup

### Step 3.1 — Configure SSH Access

Add to `~/.ssh/config` on your laptop:

```
Host cloudzy
    HostName your_cloudzy_ip
    User root
    IdentityFile ~/.ssh/your_key

Host contabo
    HostName your_contabo_ip
    User root
    IdentityFile ~/.ssh/your_key
```

Test connections:
```bash
ssh cloudzy "quantmind status"
ssh contabo "docker ps"
```

### Step 3.2 — Set Up SSH Tunnel for TUI Access

To access the TUI running on Cloudzy VPS from your laptop:

```bash
ssh -t cloudzy "tmux attach -t quantmind-tui"
```

Or create a shell alias on your laptop:

```bash
echo "alias qm-tui='ssh -t cloudzy tmux attach -t quantmind-tui'" >> ~/.bashrc
echo "alias qm-status='ssh cloudzy quantmind status'" >> ~/.bashrc
source ~/.bashrc
```

### Step 3.3 — Configure QuantMind IDE

The Svelte UI (`file:quantmind-ide/`) connects to the Cloudzy VPS API. Update the API base URL:

```bash
cd quantmind-ide
cp .env.example .env
# Set VITE_API_URL=http://your_cloudzy_ip:8000
npm install
npm run dev
```

For production build:
```bash
npm run build
```

---

## 🔄 PART 4: Multi-VPS Sync Verification

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant MT5 as MT5 Terminal
    participant Cloudzy as Cloudzy VPS
    participant Contabo as Contabo VPS
    participant Grafana as Grafana Cloud

    MT5->>Cloudzy: Tick data (ZMQ, <5ms)
    Cloudzy->>Cloudzy: HOT tier (PostgreSQL, 1hr retention)
    Cloudzy->>Contabo: Hourly migration (hot→warm)
    Contabo->>Contabo: WARM tier (DuckDB, 7-30 days)
    Contabo->>Contabo: Daily archival (warm→cold)
    Contabo->>Contabo: COLD tier (files, 30+ days)
    Contabo->>Cloudzy: HMM regime (HTTP API :8001)
    Cloudzy->>Grafana: Metrics (Prometheus remote_write)
    Contabo->>Grafana: Metrics (Prometheus remote_write)
    Cloudzy->>Grafana: Logs (Promtail → Loki)
    Contabo->>Grafana: Logs (Promtail → Loki)
```

### Sync Verification Checklist

**HMM Sync (Contabo → Cloudzy):**
```bash
# On Cloudzy VPS
curl -H "X-API-Key: $CONTABO_HMM_API_KEY" $CONTABO_HMM_API_URL/api/hmm/regime
# Should return current regime JSON
```

**Data Migration (Cloudzy → Contabo):**
```bash
# On Contabo VPS — check migration logs
tail -f /var/log/quantmindx/migration_hot_warm.log
# Should show rows migrated each hour
```

**Config Sync (Both VPS):**
```bash
# On either VPS
tail -f /var/log/quantmindx/sync_config.log
# Should show "Config sync completed" every 15 minutes
```

---

## 🤖 PART 5: MT5 Demo Account Setup

### Step 5.1 — Install MetaTrader 5

On the Cloudzy VPS (Windows RDP):
1. Download MT5 from your broker's website
2. Install MT5 on the VPS
3. Log in with your **demo account** credentials

### Step 5.2 — Connect MT5 to QuantMindX

The MT5 Bridge (`file:mt5-bridge/`) handles the connection. Configure via the QuantMind IDE:

1. Open QuantMind IDE → Settings → MT5 Configuration
2. Enter demo account credentials:
   - **Login:** Your demo account number
   - **Password:** Your demo account password
   - **Server:** Your broker's demo server (e.g., `ICMarkets-Demo`)
3. Click **Connect**

### Step 5.3 — Verify Demo Account Data Streaming

```bash
# Check MT5 Bridge is receiving tick data
quantmind logs --service mt5-bridge

# Check tick data is flowing into HOT tier
curl http://localhost:8000/api/tick/status
```

Expected: Tick data streaming at <5ms latency for subscribed symbols.

### Step 5.4 — Broker Selection for Demo Account

Choose a broker with **identical demo/live price feeds**:

| Broker | Demo Data Quality | Notes |
|---|---|---|
| IC Markets | ✅ Excellent | Claims identical feeds |
| FP Markets | ✅ Excellent | Claims identical feeds |
| Pepperstone | ✅ Good | Reputable data quality |
| MetaQuotes Demo | ⚠️ Simulated | Use only for testing infrastructure |

> **Important:** Demo accounts may be deactivated after 30 days of inactivity. Log in periodically to keep the account active.

---

## 📦 PART 6: Paper Trading Setup (Week 1 Goal)

### Step 6.1 — Import EAs from GitHub

1. Open QuantMind IDE
2. Click the **GitHub icon** in the sidebar
3. Click **Sync EAs** — this triggers `file:src/integrations/github_ea_sync.py`
4. EAs are imported from your configured GitHub repository
5. Imported EAs appear in the **EA Registry** panel

### Step 6.2 — Deploy EA to Paper Trading

1. In the EA Registry, select an imported EA
2. Click **Deploy to Paper Trading**
3. Select format: **Expert Advisor (MT5)**
4. The EA is assigned tag `@primal` automatically
5. The system deploys it to the MT5 demo account

### Step 6.3 — Bot Lifecycle (Tag Progression)

```mermaid
stateDiagram
    [*] --> primal: EA Imported
    primal --> pending: Manual Review
    pending --> perfect: Performance Criteria Met
    perfect --> live: Human Approval
    live --> quarantine: Risk Violation
    quarantine --> pending: Investigation Complete
    live --> [*]: Decommissioned
```

| Tag | Meaning | Action Required |
|---|---|---|
| `@primal` | Just imported, not yet reviewed | Review EA parameters |
| `@pending` | Under paper trading evaluation | Monitor performance |
| `@perfect` | Meets performance criteria | Approve for live |
| `@live` | Running on live account | Monitor continuously |
| `@quarantine` | Risk violation detected | Investigate and fix |

### Step 6.4 — Monitor Paper Trading Performance

Access via QuantMind IDE → **Paper Trading** tab:
- Virtual balance tracking
- Trade history
- Win rate, Sharpe ratio, max drawdown
- Bot-by-bot performance breakdown

The QuantMind Copilot agent has access to paper trading results for analysis.

---

## 📊 PART 7: Grafana Cloud Monitoring Setup

> A detailed guide already exists at `file:docs/deployment/grafana-cloud-setup.md`. This section summarizes the key steps.

### Step 7.1 — Create Grafana Cloud Account

1. Sign up at https://grafana.com/auth/sign-up/create-user
2. Create a stack named `quantmindx-prod`
3. Select region closest to your Cloudzy VPS

### Step 7.2 — Get Credentials

From your Grafana Cloud stack → Configuration → Details:
- Copy **Prometheus URL**, **Loki URL**, **Instance ID**
- Generate API Key with **MetricsPublisher** role

### Step 7.3 — Import Dashboards

Four pre-built dashboards are in `file:monitoring/dashboards/`:

| File | Dashboard |
|---|---|
| `trading-vps-overview.json` | Cloudzy VPS — API, MT5, bots, latency |
| `contabo-vps-overview.json` | Contabo VPS — HMM, storage tiers |
| `paper-trading-monitor.json` | Paper trading performance |
| `data-flow.json` | End-to-end data flow and latency |

Import via: Grafana Cloud → Dashboards → Import → Upload JSON

### Step 7.4 — Import Alert Rules

```
Grafana Cloud → Alerting → Alert rules → Import → monitoring/alert-rules.yml
```

### Step 7.5 — Verify Metrics Flow

```bash
# Check Prometheus Agent is pushing metrics
docker logs quantmind-prometheus-agent --tail 20

# Check Promtail is shipping logs
docker logs quantmind-promtail --tail 20

# Verify metrics endpoint
curl http://localhost:9090/metrics | grep quantmind_
```

---

## 🛠️ PART 8: Day-to-Day Operations

### The `quantmind` CLI

The `file:scripts/quantmind_cli.sh` is installed system-wide as `quantmind`. All commands:

| Command | Description |
|---|---|
| `quantmind status` | Show system status and health of all services |
| `quantmind health` | Run health checks (exits 0 if healthy, 1 if not) |
| `quantmind update` | Pull latest code from GitHub and restart services |
| `quantmind update --version v1.2.0` | Update to a specific version |
| `quantmind rollback` | Roll back to the previous snapshot |
| `quantmind rollback --version v1.0.0` | Roll back to a specific version |
| `quantmind restart` | Staged restart of all services |
| `quantmind restart --service quantmind-api` | Restart a single service |
| `quantmind logs` | Follow logs from all services |
| `quantmind logs --service mt5-bridge` | Follow logs from a specific service |
| `quantmind version` | Show current version |
| `quantmind history` | Show deployment history |

### Update Workflow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant GitHub as GitHub
    participant Cloudzy as Cloudzy VPS
    participant Contabo as Contabo VPS

    Dev->>GitHub: git push (new code)
    Note over Cloudzy: Every 15 min cron
    Cloudzy->>GitHub: sync_config.sh (git pull)
    Cloudzy->>Cloudzy: Restart affected services
    Note over Contabo: Every 15 min cron
    Contabo->>GitHub: sync_config.sh (git pull)
    Contabo->>Contabo: Restart affected services

    Note over Cloudzy: Manual update
    Dev->>Cloudzy: quantmind update
    Cloudzy->>Cloudzy: Snapshot current version
    Cloudzy->>GitHub: git pull
    Cloudzy->>Cloudzy: pip install -r requirements.txt
    Cloudzy->>Cloudzy: Staged restart (health checks)
    Cloudzy->>Dev: Update complete
```

### Staged Restart Order

Services restart in this order to respect dependencies:

1. `pageindex-articles`
2. `pageindex-books`
3. `pageindex-logs`
4. `prometheus-agent`
5. `promtail`
6. `hmm-scheduler`
7. `mt5-bridge`
8. `quantmind-api` (last — depends on all others)

### Rollback Workflow

Snapshots are stored in `/opt/quantmindx/snapshots/` (last 5 kept):

```bash
# List available snapshots
ls /opt/quantmindx/snapshots/

# Roll back to previous version
quantmind rollback

# Roll back to specific version
quantmind rollback --version v1.0.0
```

### TUI Quick Reference

| Keyboard Shortcut | Action |
|---|---|
| `q` | Quit TUI |
| `r` | Refresh all panels |
| `b` | Switch to Bots view |
| `t` | Switch to Trades view |
| `s` | Switch to Sync status view |
| `h` | Switch to Health view |

CLI shortcuts (from laptop via SSH):
```bash
qm-tui        # Attach to TUI on Cloudzy VPS
qm-status     # Quick status check via SSH
```

---

## 🔒 PART 9: Security Hardening

### Firewall Configuration (Cloudzy VPS)

```bash
# Allow SSH
ufw allow 22/tcp

# Allow QuantMind API (restrict to your laptop IP if possible)
ufw allow 8000/tcp

# Allow MT5 Bridge (internal only)
ufw allow from 127.0.0.1 to any port 5005

# Allow PageIndex (internal only)
ufw allow from 127.0.0.1 to any port 3000:3002

# Allow Prometheus Agent UI (internal only)
ufw allow from 127.0.0.1 to any port 9099

# Enable firewall
ufw enable
```

### Firewall Configuration (Contabo VPS)

```bash
# Allow SSH
ufw allow 22/tcp

# Allow HMM Inference API (from Cloudzy VPS only)
ufw allow from your_cloudzy_ip to any port 8001

# Block everything else
ufw default deny incoming
ufw enable
```

### Secrets Management

- Never commit `.env` to Git (already in `.gitignore`)
- Rotate `CONTABO_HMM_API_KEY` every 90 days
- Rotate `GRAFANA_API_KEY` before expiry
- Use strong random strings for all API keys: `openssl rand -hex 32`

---

## 🚨 PART 10: Troubleshooting

### Service Won't Start

```bash
# Check service logs
quantmind logs --service quantmind-api

# Check Docker container status
docker ps -a

# Check container logs directly
docker logs quantmind-api --tail 50
```

### MT5 Bridge Not Connecting

```bash
# Check MT5 Bridge logs
quantmind logs --service mt5-bridge

# Verify MT5 is running on VPS
# (Check via RDP — MT5 terminal should be open and logged in)

# Test bridge health
curl http://localhost:5005/health
```

### HMM Sync Failing

```bash
# On Cloudzy VPS — test Contabo connectivity
curl -H "X-API-Key: $CONTABO_HMM_API_KEY" $CONTABO_HMM_API_URL/health

# Check SSH key access
ssh -i $CONTABO_SSH_KEY_PATH $CONTABO_USER@$CONTABO_HOST "echo connected"

# Check HMM Inference API on Contabo
ssh contabo "docker logs quantmind-hmm-inference-api --tail 20"
```

### Data Migration Not Running

```bash
# On Contabo VPS — check cron logs
tail -f /var/log/quantmindx/migration_hot_warm.log

# Run migration manually
python3 scripts/migrate_hot_to_warm.py

# Check DuckDB warm tier
python3 -c "import duckdb; con = duckdb.connect('/data/market_data.duckdb'); print(con.execute('SELECT COUNT(*) FROM tick_data').fetchone())"
```

### Metrics Not Appearing in Grafana Cloud

```bash
# Check Prometheus Agent
docker logs quantmind-prometheus-agent --tail 20

# Verify remote_write config
cat docker/prometheus/agent-prometheus.yml

# Test metrics endpoint
curl http://localhost:9090/metrics | head -20
```

### Logs Not Appearing in Loki

```bash
# Check Promtail
docker logs quantmind-promtail --tail 20

# Verify logs directory is populated
ls -la logs/

# Check Promtail config
cat docker/promtail/promtail-config.yml
```

### Environment Validation Fails

```bash
# Run validator to see which keys are missing
bash scripts/validate_env.sh

# Compare .env with .env.example
diff <(grep -E "^[A-Z_]+=" .env.example | sed 's/=.*//') \
     <(grep -E "^[A-Z_]+=" .env | sed 's/=.*//')
```

### Roll Back After Failed Update

```bash
# Immediate rollback to last known good version
quantmind rollback

# Check deployment history
quantmind history

# Roll back to specific version
quantmind rollback --version v1.0.0
```

---

## ✅ Deployment Checklist

### Cloudzy VPS

- [ ] Repository cloned to `/opt/quantmindx`
- [ ] `scripts/install.sh` completed successfully
- [ ] `.env` configured with all required variables
- [ ] `scripts/validate_env.sh` passes
- [ ] All 7 Docker services healthy (`quantmind status`)
- [ ] `config/crontab.cloudzy` installed (`crontab -l` shows sync job)
- [ ] `quantmind` CLI available system-wide
- [ ] TUI running in tmux session
- [ ] Firewall configured
- [ ] MT5 connected to demo account
- [ ] Tick data streaming verified (<5ms latency)

### Contabo VPS

- [ ] Repository cloned to `/opt/quantmindx`
- [ ] `.env` configured with Contabo-specific variables
- [ ] Storage directories created (`/data/hmm`, `/data/cold_storage`)
- [ ] All 4 Docker services healthy
- [ ] `config/crontab.contabo` installed (4 cron jobs)
- [ ] HMM Inference API accessible from Cloudzy VPS
- [ ] Firewall configured (only Cloudzy IP can reach port 8001)

### Grafana Cloud

- [ ] Account created and stack configured
- [ ] Prometheus URL, Loki URL, Instance ID, API Key collected
- [ ] Metrics appearing in Grafana Cloud (within 5 minutes of deployment)
- [ ] Logs appearing in Loki
- [ ] 4 dashboards imported from `monitoring/dashboards/`
- [ ] Alert rules imported from `monitoring/alert-rules.yml`
- [ ] Notification channels configured (email/Slack)

### Local Laptop

- [ ] SSH config set up for `cloudzy` and `contabo` hosts
- [ ] Shell aliases configured (`qm-tui`, `qm-status`)
- [ ] QuantMind IDE configured with Cloudzy VPS API URL
- [ ] Browser bookmarks for Grafana Cloud dashboards

### Paper Trading (Week 1)

- [ ] MT5 demo account connected and streaming data
- [ ] GitHub EA Sync button working in IDE
- [ ] At least one EA imported and deployed to paper trading
- [ ] EA assigned `@primal` tag and visible in bot lifecycle panel
- [ ] Paper trading stats visible in IDE
- [ ] QuantMind Copilot can access paper trading results

---

## 📚 Related Specs

| Spec | Reference |
|---|---|
| Multi-VPS Synchronization System | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/4c3aef8a-c398-4814-b56f-7ebcfd574f4a` |
| Self-Update & Deployment Automation | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/5f9b63a5-0d54-4b26-a635-3e731a924ab7` |
| Production Monitoring & Verification | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/18f1e056-f93d-46b0-a99b-26abe6e833da` |
| Terminal User Interface (TUI) | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/c10875a4-945d-440d-8062-02d2b32eea8e` |
| Enhanced Paper Trading System | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/422dcd66-7802-485b-9a7c-33f0e87bc257` |
| Middleware Stabilization & Critical Fixes | `spec:af37afc7-1666-472b-98f4-9ca0f6d5c861/1289f8aa-aed7-4a11-a3e0-576692087df5` |
| Grafana Cloud Setup (existing doc) | `file:docs/deployment/grafana-cloud-setup.md` |