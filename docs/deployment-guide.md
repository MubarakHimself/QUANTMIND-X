# QUANTMINDX — VPS Deployment Guide

**Generated:** 2026-03-11
**Updated:** 2026-03-22 (OAuth, k6, APM, ZAP added)

---

## Infrastructure Overview

QUANTMINDX uses a two-VPS architecture:

| VPS | Provider | Role |
|-----|---------|------|
| Contabo | Contabo | HMM model training + regime API |
| Cloudzy | Cloudzy | Live API server + trading execution |

MetaTrader 5 runs on a Windows machine or Wine (not VPS-compatible natively).

---

## Prerequisites

- Ubuntu 22.04+ on both VPS
- Python 3.12+
- Node.js 18+
- Docker + Docker Compose
- Git
- SSH access between Contabo and Cloudzy (for HMM model sync)

---

## 1. Initial VPS Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Git
sudo apt install -y git
```

---

## 2. Clone and Configure

```bash
git clone https://github.com/MubarakHimself/QUANTMIND-X.git
cd QUANTMIND-X

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install MT5 MCP server (editable)
pip install -e ./mcp-metatrader5-server

# Install frontend dependencies (optional — IDE not needed on VPS)
# cd quantmind-ide && npm install
```

---

## 3. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional AI providers
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///data/db/quantmind.db

# MT5 Bridge (Windows machine running mt5-bridge)
# Default bridge port is 5005; override with MT5_BRIDGE_PORT on the Windows host if needed.
MT5_BRIDGE_URL=http://YOUR_MT5_MACHINE_IP:5005
MT5_BRIDGE_TOKEN=your-secret-token

# HMM Contabo API URL (used by Cloudzy to fetch regime data)
CONTABO_HMM_API_URL=http://YOUR_CONTABO_IP:8000
CONTABO_HMM_API_KEY=your-api-key

# Prometheus
PROMETHEUS_PORT=9090
MT5_PROMETHEUS_PORT=9091

# GitHub EA repository
GITHUB_EA_REPO_URL=https://github.com/youruser/your-ea-repo

# Workspaces
WORKSPACES_DIR=/home/user/QUANTMINDX/workspaces
CONFIG_DIR=/home/user/QUANTMINDX/config

# ========== OAuth/OIDC (Auth0) ==========
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_AUDIENCE=https://api.quantmindx.com
AUTH0_CALLBACK_URL=https://your-domain.com/api/auth/callback
OAUTH_PKCE_CODE_VERIFIER_LENGTH=64

# ========== Grafana APM ==========
GRAFANA_TEMPO_OTLP_ENDPOINT=https://tempo.example.com:4317
GRAFANA_TEMPO_USER=your_tempo_user
GRAFANA_TEMPO_API_KEY=your_tempo_api_key
```

---

## 4. Database Initialization

```bash
# Run all pending migrations
python3 -m src.database.migrations.migration_runner

# Optionally seed broker registry
python3 scripts/populate_broker_registry.py
```

---

## 5. Systemd Services

Three systemd service files are in `systemd/`:

### Copy and Enable Services

```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Reload daemon
sudo systemctl daemon-reload

# Enable services on boot
sudo systemctl enable quantmind-api
sudo systemctl enable quantmind-tui
sudo systemctl enable hmm-training-scheduler

# Start services
sudo systemctl start quantmind-api
sudo systemctl start quantmind-tui
```

### Service Descriptions

| Service | Purpose | Port |
|---------|---------|------|
| `quantmind-api.service` | FastAPI backend (main API) | 8000 |
| `quantmind-tui.service` | Terminal UI (tmux session) | — |
| `hmm-training-scheduler.service` | HMM retraining cron (Contabo only) | — |

### Check Service Status

```bash
sudo systemctl status quantmind-api
sudo journalctl -u quantmind-api -f    # Follow logs
```

---

## 6. Docker Services

### Development (Local)

```bash
# Start PageIndex search services + monitoring
docker-compose up -d
```

Services started:
- `pageindex-articles` (port 3000) — MQL5 articles search
- `pageindex-books` (port 3001) — Knowledge books search
- `pageindex-logs` (port 3002) — Logs search

### Production

```bash
# Start production services (API + monitoring)
docker-compose -f docker-compose.production.yml up -d
```

Adds:
- Prometheus (port 9090) — metrics collection
- Grafana (port 3000) — dashboards
- Loki — log aggregation
- Promtail — log shipping agent

### Contabo-specific

```bash
# Start Contabo services
docker-compose -f docker-compose.contabo.yml up -d
```

---

## 7. Contabo VPS Setup (Training Server)

Contabo is dedicated to HMM model training and serving regime data.

```bash
# Additional setup for Contabo
bash scripts/setup_contabo_crons.sh

# Configure cron jobs for scheduled tasks
# (populates crontab with HMM training schedule)
```

### Contabo Cron Jobs

| Schedule | Script | Purpose |
|----------|--------|---------|
| Daily 2 AM | `scripts/train_hmm.py` | Retrain HMM model |
| Daily 3 AM | `scripts/schedule_lifecycle_check.py` | Bot lifecycle check |
| Hourly | `scripts/schedule_market_scanner.py` | Market scanning |

### HMM Training on Contabo

```bash
# Manual training run
source venv/bin/activate
python3 scripts/train_hmm.py --n-components 4 --symbols EURUSD,GBPUSD,XAUUSD

# Validate model
python3 scripts/validate_hmm.py

# Model is automatically versioned and available for sync to Cloudzy
```

---

## 8. Cloudzy VPS Setup (Trading Server)

Cloudzy runs the live StrategyRouter and executes trades via the MT5 Bridge.

### HMM Model Sync

```bash
# Trigger sync from Cloudzy API
curl -X POST http://localhost:8000/api/hmm/sync \
  -H "Content-Type: application/json" \
  -d '{"verify_checksum": true}'

# Check sync status
curl http://localhost:8000/api/hmm/status
```

Alternatively, sync via the IDE `TradingFloorPanel.svelte` → HMM controls.

### HMM Mode Progression

```bash
# 1. Start in shadow mode (no trading impact)
curl -X POST http://localhost:8000/api/hmm/mode \
  -d '{"mode": "hmm_shadow"}'

# 2. Generate approval token for hybrid mode
curl -X POST http://localhost:8000/api/hmm/approval-token \
  -d '{"target_mode": "hmm_hybrid_20", "requester": "admin"}'

# 3. Enable 20% HMM blending
curl -X POST http://localhost:8000/api/hmm/mode \
  -d '{"mode": "hmm_hybrid_20", "approval_token": "TOKEN_FROM_STEP_2"}'
```

---

## 9. Reverse Proxy (Nginx)

For production, serve the API behind Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }

    # SSE endpoint — disable buffering
    location /api/agents/stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
    }
}
```

---

## 10. Monitoring Setup

### Prometheus + Grafana Cloud

The API server starts a Prometheus metrics server on startup (port `PROMETHEUS_PORT`, default 9090).

For Grafana Cloud remote_write, configure in `docker-compose.production.yml`:

```yaml
prometheus:
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    # remote_write to Grafana Cloud configured in prometheus.yml
```

### Loki + Promtail (Log Shipping)

Log files at `logs/api.json`, `logs/router.json`, `logs/mt5-bridge.json` are in JSON format for Promtail.

Promtail config: `docker/promtail/config.yml`

```bash
# Start Promtail as Docker container
docker run -d \
  -v /path/to/logs:/var/log/quantmind \
  -v $(pwd)/docker/promtail/config.yml:/etc/promtail/config.yml \
  grafana/promtail:latest \
  -config.file=/etc/promtail/config.yml
```

### Pre-built Grafana Dashboards

Located at `monitoring/dashboards/`. Import into Grafana:
1. Grafana UI → Dashboards → Import
2. Upload JSON files from `monitoring/dashboards/`

---

## 11. MT5 Bridge Deployment (Windows)

The MT5 Bridge must run on a Windows machine (or Wine) with MetaTrader 5 installed.

```cmd
# Windows command prompt
cd mt5-bridge
pip install -r requirements.txt
python server.py
```

The bridge runs at `http://MT5_MACHINE_IP:5005` by default.
If you set `MT5_BRIDGE_PORT` on the Windows host, use that port in `MT5_BRIDGE_URL`.

Set `MT5_BRIDGE_URL` in Cloudzy's `.env` to point to this machine.

---

## 12. Config Sync

When config files change locally:

```bash
# Sync config to VPS
bash scripts/sync_config.sh

# Sync EA files from GitHub repo
curl -X POST http://localhost:8000/api/github/sync
```

---

## 13. Pre-Deployment Validation

### 13.1 Run k6 Load Tests

Before deploying to production, validate system capacity with k6 load tests:

```bash
# Install k6 (if not already installed)
sudo gpg -k
sudo curl -sL https://sdk.k6.io/gpg.asc | sudo gpg --dearmor -o /usr/share/keyrings/k6-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt update && sudo apt install -y k6

# Run smoke test (quick validation)
k6 run k6/scripts/smoke-test.js

# Run full load test (50-bot capacity)
k6 run k6/k6.conf.js

# Run specific profile (sustained load)
k6 run k6/k6.conf.js --profile sustained

# Check results
# Report generated at: _bmad-output/test-artifacts/k6-load-test-report.md
```

**Acceptance Criteria:**
- P95 latency < 500ms for all critical endpoints
- Error rate < 1%
- Circuit breaker triggers correctly under load

---

### 13.2 Run Security Scans

Run SAST/DAST security scans before production:

```bash
# Local ZAP scan
./scripts/zap_scan_wrapper.sh http://localhost:8000

# Or run via Docker
docker run -v $(pwd):/zap/wrk:rw \
  owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:8000 \
  -J zap-report.json

# npm audit (frontend dependencies)
cd quantmind-ide && npm audit --audit-level=critical

# pip-audit (Python dependencies)
pip-audit
```

**Acceptance Criteria:**
- ZAP: 0 Critical/High vulnerabilities
- npm audit: 0 Critical vulnerabilities
- pip-audit: 0 Critical vulnerabilities

---

### 13.3 OAuth Flow Validation

Test the OAuth authentication flow:

```bash
# 1. Verify Auth0 configuration
curl http://localhost:8000/api/auth/login
# Should redirect to Auth0

# 2. Test token exchange
# Complete OAuth flow in browser, then verify:
curl http://localhost:8000/api/auth/me
# Should return user info with valid session

# 3. Test token refresh
# Tokens should refresh automatically (15-min access token TTL)
```

---

## 14. Production Checklist

Before going live:

### Infrastructure & Auth
- [ ] `.env` configured with production values (not test keys)
- [ ] `MT5_BRIDGE_TOKEN` set to strong random value
- [ ] OAuth/OIDC (Auth0) configured with `AUTH0_*` env vars
- [ ] Database migrations run (`migration_runner.py`)
- [ ] Broker registry seeded (`populate_broker_registry.py`)

### Monitoring & Observability
- [ ] Prometheus + Grafana dashboards configured (import `docker/grafana/dashboards/apm-dashboard.json`)
- [ ] APM runbook reviewed (`_bmad-output/test-artifacts/apm-runbook.md`)
- [ ] Promtail log shipping configured
- [ ] OpenTelemetry collector running (for distributed tracing)

### Load & Security Testing ✅ VALIDATED
- [ ] **k6 load tests passed** (P95 < 500ms, error rate < 1%)
- [ ] **ZAP DAST scan passed** (0 Critical/High vulnerabilities)
- [ ] **OAuth flow validated** (login, token exchange, refresh working)
- [ ] npm audit passed (0 Critical vulnerabilities)
- [ ] pip-audit passed (0 Critical vulnerabilities)

### HMM & Trading
- [ ] HMM model trained and validated
- [ ] HMM deployed in `hmm_shadow` mode (not `hmm_only` immediately)
- [ ] Bot manifests reviewed — only `@primal`-tagged bots in auction
- [ ] Prop firm account drawdown limits set correctly
- [ ] Progressive kill switch tiers configured

### Deployment
- [ ] Systemd services enabled and running
- [ ] Nginx reverse proxy configured
- [ ] SSH key configured for Contabo ↔ Cloudzy model sync
- [ ] Kill switch tested (soft and hard)

---

## 14. Useful Operational Commands

```bash
# View API logs
sudo journalctl -u quantmind-api -f

# Restart API service
sudo systemctl restart quantmind-api

# Run database migration manually
source venv/bin/activate
python3 -m src.database.migrations.migration_runner

# Check router status
curl http://localhost:8000/api/router/status

# Soft kill switch
curl -X POST http://localhost:8000/api/kill-switch/soft

# Hard kill switch (emergency)
curl -X POST http://localhost:8000/api/kill-switch/hard

# Check HMM status
curl http://localhost:8000/api/hmm/status

# Run bot lifecycle check
curl -X POST http://localhost:8000/api/lifecycle-scanner/check

# Health check
curl http://localhost:8000/health
```

---

## 15. Rollback Procedures

### API Service Rollback

```bash
# Roll back to previous git commit
git log --oneline -5  # Find previous commit
git checkout <commit-hash>
sudo systemctl restart quantmind-api
```

### HMM Model Rollback

```bash
# List available model versions
curl http://localhost:8000/api/hmm/models

# Revert to ISING_ONLY mode immediately
curl -X POST http://localhost:8000/api/hmm/mode \
  -d '{"mode": "ising_only"}'
```

### Database Rollback

SQLite migrations are not automatically reversible. Keep backups:

```bash
# Backup database before major changes
cp data/db/quantmind.db data/db/quantmind.db.backup.$(date +%Y%m%d)
```
