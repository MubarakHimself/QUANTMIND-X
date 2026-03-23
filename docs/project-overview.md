# QUANTMINDX — Project Overview

**Generated:** 2026-03-11 | **Scan Level:** Exhaustive | **Version:** Active Development

---

## What is QUANTMINDX?

QUANTMINDX is an **autonomous AI-powered algorithmic trading platform** that combines multi-agent AI orchestration, physics-based market regime detection, and MetaTrader 5 integration into a single monorepo. It is designed for active prop-firm trading (FTMO, The5ers, FundingPips) and private capital management.

---

## Executive Summary

| Attribute | Value |
|-----------|-------|
| **Repository Type** | Monorepo — 7 distinct parts |
| **Primary Languages** | Python 3.12, TypeScript |
| **Backend Framework** | FastAPI (Python) |
| **Frontend** | SvelteKit 4 + Tauri 2 (Desktop IDE) |
| **Primary Database** | SQLite (operational) + DuckDB (analytics) |
| **AI Stack** | Anthropic Claude API (claude-sonnet-4, claude-opus-4) |
| **Trading Integration** | MetaTrader 5 via bridge server + MCP |
| **Deployment** | VPS (Contabo/Cloudzy) via systemd + Docker |
| **Observability** | Prometheus + Grafana Cloud + Loki/Promtail |

---

## Architecture Type

**Multi-part Microservices + Desktop GUI**

The system runs as a set of cooperating services:

```
[quantmind-ide]  ←→  [FastAPI Backend (src/)]  ←→  [mt5-bridge]  ←→  MetaTrader 5
       ↕                      ↕
[feature-server]        [MCP Servers]
                               ↕
                    [mcp-metatrader5-server]
```

---

## Core Capabilities

### 1. AI Agent Trading Floor
A department-based multi-agent system orchestrated by a **Floor Manager** (Opus tier) with 5 department heads (Sonnet tier) and spawnable worker agents (Haiku tier):
- **Research** — strategy research, market analysis, backtesting
- **Development** — EA coding (Python, MQL5, PineScript)
- **Trading** — order execution via MT5 bridge (read-only access)
- **Risk** — position sizing, drawdown monitoring, VaR
- **Portfolio** — allocation, rebalancing, performance attribution

### 2. Physics-Aware Risk Engine
- **Ising Model** sensor: treats price correlations as spin magnetization
- **HMM (Hidden Markov Model)**: regime detection (TREND/RANGE/BREAKOUT/CHAOS)
- **Lyapunov Exponent** sensor: market instability detection
- **Physics-Aware Kelly Engine**: position sizing with econophysics multipliers

### 3. Strategy Router (The Sentient Loop)
- **Sentinel** → regime classification from raw tick data
- **Governor** → risk scalar calculation (STANDARD/CLAMPED/HALTED)
- **Commander** → strategy auction, selects bots per regime
- **Kill Switch** → progressive circuit breakers (soft/hard)
- **Routing Matrix** → assigns bots to broker accounts

### 4. Trading IDE
- SvelteKit 4 + Tauri 2 desktop application
- Monaco Editor for strategy code editing
- 60+ UI panels: backtesting, trading floor, agent chat, monitoring
- Real-time WebSocket feed from backend

### 5. Knowledge & Data Pipeline
- Firecrawl scraper for MQL5 articles
- Video ingest pipeline (YouTube → transcript → strategy extraction)
- PageIndex Docker services for full-text search
- ChromaDB + Qdrant for vector semantic search

---

## Tech Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| Backend API | Python 3.12, FastAPI, uvicorn | REST + WebSocket API |
| Agent Runtime | Anthropic Claude API, Agent SDK | AI agents |
| Databases | SQLite, DuckDB, Redis, ChromaDB, Qdrant | Data storage |
| Frontend | SvelteKit 4, TypeScript, Tailwind | Desktop IDE |
| Desktop Shell | Tauri 2 | Native desktop wrapper |
| Trading | MetaTrader 5, MQL5 | EA execution |
| Analysis | NumPy, Pandas, SciPy | Quant analysis |
| Risk Physics | HMM, Ising model, Lyapunov | Regime detection |
| Backtesting | DuckDB engine, Monte Carlo, Walk-Forward | Strategy validation |
| Observability | Prometheus, Grafana Cloud, Loki, Promtail | Monitoring |
| Containerization | Docker, docker-compose | Services orchestration |
| CI/CD | GitHub Actions (CodeQL security scan) | Security |

---

## Repository Structure

```
QUANTMINDX/
├── src/                     # Python backend (FastAPI + agents + risk + router)
│   ├── api/                 # 80+ REST/WebSocket endpoint modules
│   ├── agents/              # AI agent system
│   │   ├── departments/     # Floor Manager + 5 Department Heads (CANONICAL)
│   │   ├── subagent/        # Claude SDK-based worker spawner
│   │   ├── memory/          # Unified memory facade + vector/graph memory
│   │   ├── skills/          # Agent skills registry
│   │   ├── core/            # base_agent.py (LEGACY - LangGraph)
│   │   ├── registry.py      # DEPRECATED - use floor_manager
│   │   └── claude_config.py # Workspace-based agent config
│   ├── risk/                # Risk engine
│   │   ├── sizing/          # Physics-Aware Kelly Engine
│   │   └── physics/         # HMM, Ising, Lyapunov sensors
│   ├── router/              # Strategy Router (Sentient Loop)
│   │   ├── sentinel.py      # Market regime classifier
│   │   ├── governor.py      # Risk compliance layer
│   │   ├── commander.py     # Bot selection & execution
│   │   └── kill_switch.py   # Circuit breakers
│   ├── database/            # SQLAlchemy models + DuckDB + migrations
│   ├── backtesting/         # Backtesting engine (DuckDB, MC, Walk-Forward)
│   └── integrations/        # Crypto (Binance), PineScript, GitHub
├── quantmind-ide/           # SvelteKit + Tauri desktop IDE
├── feature-server/          # Lightweight SvelteKit server
├── mt5-bridge/              # FastAPI bridge to MetaTrader 5
├── mcp-metatrader5-server/  # MCP server for MT5 tools
├── mcp-servers/             # Backtest MCP + Knowledge Base MCP
├── extensions/              # MQL5 library + Gemini video extension
├── scripts/                 # Automation: scraper, HMM training, KB indexing
├── data/                    # Scraped articles, strategies, backtests, logs
├── docker/                  # Docker configs (PageIndex, Prometheus, Promtail)
├── monitoring/              # Grafana dashboards
├── systemd/                 # Service unit files for VPS deployment
├── tests/                   # Test suite (pytest, 20+ test directories)
└── docs/                    # Project documentation
```

---

## Links to Detailed Documentation

- [Architecture Overview](./architecture.md)
- [Agentic Setup & Migration Status](./agentic-setup.md)
- [Risk Management & Position Sizing](./risk-management.md)
- [Strategy Router (Sentient Loop)](./strategy-router.md)
- [HMM Regime Detection](./hmm-system.md)
- [Database Schema](./data-models.md)
- [API Contracts](./api-contracts.md)
- [UI Component Inventory](./component-inventory.md)
- [Data Pipeline](./data-pipeline.md)
- [Development Guide](./development-guide.md)
- [Deployment Guide](./deployment-guide.md)
- [Source Tree Analysis](./source-tree-analysis.md)
