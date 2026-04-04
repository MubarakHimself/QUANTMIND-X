# QUANTMINDX — Development Guide

**Generated:** 2026-03-11

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Frontend build |
| npm | 9+ | Frontend package manager |
| Rust | Latest stable | Tauri desktop build |
| Tauri CLI | 2.x | Desktop packaging |
| Docker | Latest | PageIndex + monitoring services |
| MetaTrader 5 | 5.x | Trading platform (Windows/Wine) |
| Git | Latest | Version control |

---

## Environment Setup

### 1. Clone and set up Python environment

```bash
git clone https://github.com/MubarakHimself/QUANTMIND-X.git
cd QUANTMIND-X

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install MCP MetaTrader5 server (editable)
pip install -e ./mcp-metatrader5-server
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Required
ANTHROPIC_API_KEY=your_key_here

# Optional AI providers
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///data/db/quantmind.db

# MT5 Bridge
MT5_BRIDGE_URL=http://localhost:5005
MT5_BRIDGE_TOKEN=secret-token-change-me

# Contabo HMM API (optional)
CONTABO_HMM_API_URL=http://localhost:8001
CONTABO_HMM_API_KEY=

# Prometheus
MT5_PROMETHEUS_PORT=9091

# Workspaces (agent workspace dir)
WORKSPACES_DIR=/home/user/QUANTMINDX/workspaces
CONFIG_DIR=/home/user/QUANTMINDX/config
```

### 3. Initialize database

```bash
# Run migrations
python3 -m src.database.migrations.migration_runner

# Optionally seed broker registry
python3 scripts/populate_broker_registry.py
```

### 4. Set up frontend (IDE)

```bash
cd quantmind-ide
npm install
```

### 5. Set up feature server

```bash
cd feature-server
npm install
```

---

## Running in Development

### Backend API

```bash
# From project root, with venv activated
python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### Frontend IDE (Web)

```bash
cd quantmind-ide
npm run dev
# → http://localhost:3001
```

### Frontend IDE (Tauri Desktop)

```bash
cd quantmind-ide
npm run tauri:dev
# Opens native desktop window
```

### MT5 Bridge (requires MetaTrader 5)

```bash
cd mt5-bridge
pip install -r requirements.txt
python3 server.py
# → http://localhost:5005
```

### Feature Server

```bash
cd feature-server
npm run dev
# → http://localhost:3002
```

### MCP MetaTrader5 Server

```bash
# Run as MCP server (stdio transport)
python3 -m mcp_mt5.main
```

### MCP Knowledge Base Server

```bash
cd mcp-servers/quantmindx-kb
./start.sh
# or: python3 server.py
```

### All Services

```bash
./scripts/start_all.sh
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test directory
pytest tests/risk/
pytest tests/api/
pytest tests/agents/

# Run with timeout (recommended for integration tests)
pytest --timeout=60

# Run faster (skip slow integration tests)
pytest -m "not integration"
```

### Test Structure

```
tests/
├── agents/         # Agent unit tests
├── api/            # API endpoint tests
├── backtesting/    # Backtesting engine tests
├── benchmarks/     # Performance benchmarks
├── bots/           # Bot logic tests
├── brokers/        # Broker integration tests
├── cache/          # Cache tests
├── crypto/         # Crypto module tests
├── data/           # Data pipeline tests
├── database/       # DB model tests
├── e2e/            # End-to-end tests
├── integration/    # Integration tests
├── load/           # Load tests
├── mcp_servers/    # MCP server tests
├── mcp_tools/      # MCP tool tests
├── memory/         # Memory system tests
├── mql5/           # MQL5 tests
├── nprd/           # NPRD tests
├── position_sizing/# Position sizing tests
├── properties/     # Property-based tests (Hypothesis)
├── queues/         # Queue tests
├── risk/           # Risk engine tests
├── router/         # Router tests
├── tui/            # TUI tests
└── unit/           # Misc unit tests
```

---

## Linting & Code Quality

```bash
# Python linting (ruff)
ruff check src/ tests/

# Python formatting
ruff format src/ tests/

# TypeScript (from quantmind-ide/)
cd quantmind-ide && npm run check
```

---

## Build

### Backend (no build step)
Python runs directly. No compilation needed.

### Frontend Web Build

```bash
cd quantmind-ide
npm run build
# Output: quantmind-ide/.svelte-kit/output/
```

### Tauri Desktop Build

```bash
cd quantmind-ide
npm run tauri:build
# Output: quantmind-ide/src-tauri/target/release/
```

---

## Docker Services (Knowledge Base + Monitoring)

```bash
# Start PageIndex search services
docker-compose up -d

# Start production services (API + monitoring)
docker-compose -f docker-compose.production.yml up -d

# Start Contabo-specific services
docker-compose -f docker-compose.contabo.yml up -d
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/start_all.sh` | Start all services |
| `scripts/install.sh` | Full installation script |
| `scripts/validate_env.sh` | Validate env vars before start |
| `scripts/train_hmm.py` | Train HMM model |
| `scripts/validate_hmm.py` | Validate trained HMM |
| `scripts/schedule_hmm_training.py` | Scheduled HMM training |
| `scripts/populate_broker_registry.py` | Seed broker data |
| `scripts/validate_broker_registry.py` | Validate broker data |
| `scripts/firecrawl_scraper.py` | Scrape MQL5 articles |
| `scripts/video_ingest_cli.py` | Video ingest CLI |
| `scripts/schedule_market_scanner.py` | Scheduled market scanner |
| `scripts/schedule_lifecycle_check.py` | Scheduled bot lifecycle check |
| `scripts/migrate_hot_to_warm.py` | Data tier migration |
| `scripts/archive_warm_to_cold.py` | Data archiving |
| `scripts/sync_config.sh` | Sync config to VPS |
| `scripts/setup_contabo_crons.sh` | Set up cron jobs on Contabo |
| `scripts/quantmind_cli.sh` | Main CLI wrapper |

---

## VPS Deployment (systemd)

Three systemd services in `systemd/`:

| Service | Purpose |
|---------|---------|
| `quantmind-api.service` | FastAPI backend (port 8000) |
| `quantmind-tui.service` | TUI server (tmux session) |
| `hmm-training-scheduler.service` | HMM training cron |

**Deploy:**
```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantmind-api
sudo systemctl start quantmind-api
```

See `docs/Contabo_Deployment_Guide.md` and `docs/QuantMindX_Production_Deployment_Guide.md` for full VPS setup.

---

## Common Development Tasks

### Add a new API endpoint

1. Create file in `src/api/new_endpoints.py`
2. Define FastAPI router:
   ```python
   from fastapi import APIRouter
   router = APIRouter(prefix="/api/new", tags=["new"])
   ```
3. Register in `src/api/server.py`:
   ```python
   from src.api.new_endpoints import router as new_router
   app.include_router(new_router)
   ```

### Add a new database model

1. Create in `src/database/models/new_model.py` (inherit from `Base`)
2. Import in `src/database/models/__init__.py`
3. Create migration in `src/database/migrations/add_new_table.py`
4. Migration runs automatically on next server start

### Add a new agent tool

1. Create in `src/agents/tools/new_tool.py` with `@tool` decorator
2. Register in appropriate department via `src/agents/departments/tool_registry.py`

### HMM Model Training

```bash
# 1. Ensure market data is in DuckDB
# 2. Train the model
python3 scripts/train_hmm.py --n-components 4 --symbols EURUSD,GBPUSD

# 3. Validate
python3 scripts/validate_hmm.py

# 4. Deploy to shadow mode
# (via /api/hmm/deploy endpoint)
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables (not in git) |
| `config/settings/` | Application settings |
| `config/mcp/*.json` | MCP server configs per agent |
| `config/agents/` | Agent configuration |
| `quantmind-ide/svelte.config.js` | SvelteKit config |
| `quantmind-ide/src-tauri/tauri.conf.json` | Tauri desktop config |
| `pytest.ini` | Test configuration |
| `pyproject.toml` | Python project config (if exists) |
| `requirements.txt` | Python dependencies |
