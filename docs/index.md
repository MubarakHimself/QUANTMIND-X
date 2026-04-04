# QUANTMINDX — Documentation Index

**Generated:** 2026-03-11 | **Last Refreshed:** 2026-03-25 | **Project:** QUANTMINDX | **Scan Level:** Exhaustive

> **⚠️ Architecture Reference Updated (2026-03-25):** The authoritative architecture decision document is at `_bmad-output/planning-artifacts/architecture.md`. The planning artifacts (PRD, epics, sprint status) are the source of truth for all implemented features. See §Implementation Summary at the top of that file.

---

## About QUANTMINDX

QUANTMINDX is an AI-powered autonomous algorithmic trading platform. It combines a physics-inspired market analysis engine (Ising + HMM regime detection), a real-time strategy router (the "Sentient Loop"), an AI-agent department system, and a desktop IDE — all coordinated to trade Forex and crypto on prop firm accounts.

---

## Documentation Map

### 1. Getting Started

| Document | Description |
|----------|-------------|
| [Project Overview](./project-overview.md) | Executive summary, capabilities, tech stack, and repo structure |
| [Development Guide](./development-guide.md) | Prerequisites, environment setup, running services, testing, linting |
| [Source Tree Analysis](./source-tree-analysis.md) | Annotated directory tree — every key file explained |

---

### 2. Architecture

| Document | Description |
|----------|-------------|
| [System Architecture](./architecture.md) | Full multi-part system architecture: 8 parts, inter-service communication, startup sequence, key design decisions |
| [API Contracts](./api-contracts.md) | All 50+ API endpoint groups: REST, WebSocket, SSE — with request/response patterns |
| [Component Inventory](./component-inventory.md) | All 65+ Svelte UI components with purpose and category |

---

### 3. Core Systems

| Document | Description |
|----------|-------------|
| [Strategy Router](./strategy-router.md) | The Sentient Loop — tick → regime → risk → bot auction → MT5 execution |
| [Risk Management](./risk-management.md) | 8-layer risk system: Ising sensor, Sentinel, Governor, Kelly Engine, Kill Switches |
| [HMM System](./hmm-system.md) | HMM regime detection: 10-feature vector, deployment modes, version control, training pipeline |
| [Agentic Setup](./agentic-setup.md) | AI agent architecture: Department System, known routing bug, deprecated paradigms, memory systems |

---

### 4. Data

| Document | Description |
|----------|-------------|
| [Data Models](./data-models.md) | Database schema: SQLite (SQLAlchemy), DuckDB analytics, AgentDB, Department Mail |
| [Data Pipeline](./data-pipeline.md) | MQL5 scraper, Video Ingest, PageIndex Docker services, ChromaDB/Qdrant, HMM training, tiered storage |

---

### 5. Operations

| Document | Description |
|----------|-------------|
| [Deployment Guide](./deployment-guide.md) | VPS deployment: systemd services, Docker, Contabo/Cloudzy two-VPS setup, MT5 Bridge, monitoring |
| [Development Guide](./development-guide.md) | Local development: env setup, running services, test commands, scripts reference |

---

### 6. Plans & Decisions (Reference)

| Document | Description |
|----------|-------------|
| [Agent Architecture Diagnosis](./plans/2026-03-08-agent-architecture-diagnosis.md) | Critical analysis of 3 agent paradigms, routing conflicts, deprecated systems |
| [Agent Migration Map](./plans/2026-03-08-agent-architecture-migration-map.md) | 5-phase migration plan to consolidate agent systems |
| [Agent Migration Implementation](./plans/2026-03-08-agent-architecture-migration-implementation.md) | Step-by-step implementation tasks for migration |

---

### 7. Implemented Features (2026-03-25) — See Planning Artifacts for Specs

All features below are **fully implemented and tested**. Source of truth: `_bmad-output/implementation-artifacts/sprint-status.yaml`

| Epic | Feature | Spec Location |
|------|---------|--------------|
| E14 | **Three-Layer SL/TP** — Layer 1 EA Hard Safety, Layer 2 Tier-1 Position Monitor, Layer 3 CHAOS Kill Switch | `_bmad-output/planning-artifacts/epics.md` (F-01) |
| E15 | **SVSS** — Shared VWAP/RVOL/MFI/Volume Profile service, jittered TTL cache | `_bmad-output/planning-artifacts/epics.md` (F-03) |
| E16 | **Session Cycle** — Tilt mechanism (LOCK→SIGNAL→WAIT→RE-RANK→ACTIVATE), 10-window canonical cycle, Inter-Session Cooldown | `_bmad-output/planning-artifacts/epics.md` (F-09) |
| E17 | **DPR** — Composite 0–100 scoring (WR25%/PnL30%/consistency20%/EV25%), Queue Tier Remix | `_bmad-output/planning-artifacts/epics.md` (F-12) |
| E18 | **SSL** — Per-bot consecutive loss counter, paper rotation (TIER_1/2), DPR integration | `_bmad-output/planning-artifacts/epics.md` (F-13) |
| E4 (ext) | **Risk Extensions** — SQS, Correlation Sensor, RVOL Kelly, Session Kelly modifiers, Portfolio math | `_bmad-output/planning-artifacts/epics.md` (F-04 through F-07, F-14) |
| E8 (ext) | **Alpha Forge Extensions** — Regime pools, HMM WFA, Workflow 3 (Dead Zone), Workflow 4 (Weekend) | `_bmad-output/planning-artifacts/epics.md` (F-02, F-08, F-10, F-11) |
| E11 (ext) | **FlowForge-Prefect API** — Workflow endpoints, SSE events | `_bmad-output/implementation-artifacts/` story specs |

**Test suite:** 614+ tests across all implemented epics. Run with `pytest tests/` from project root.

---

## Quick Reference

### System Ports

| Service | Port | Notes |
|---------|------|-------|
| FastAPI Backend | 8000 | Main API |
| SvelteKit IDE | 3001 | Web dev mode |
| MT5 Bridge | 5005 | Windows/Wine only; override with `MT5_BRIDGE_PORT` if needed |
| Feature Server | 3002 | Experimental |
| Prometheus | 9090 | Metrics |
| MT5 Prometheus | 9091 | MT5 metrics |
| PageIndex Articles | 3000 | Docker |
| PageIndex Books | 3001 | Docker |

### Key File Paths

| Purpose | Path |
|---------|------|
| API Entry Point | `src/api/server.py` |
| Floor Manager (Canonical) | `src/agents/departments/floor_manager.py` |
| Kelly Engine | `src/risk/sizing/kelly_engine.py` |
| Ising Sensor | `src/risk/physics/ising_sensor.py` |
| HMM Feature Extractor | `src/risk/physics/hmm/models.py` |
| Strategy Router | `src/router/engine.py` |
| HMM Version Control | `src/router/hmm_version_control.py` |
| IDE Entry Point | `quantmind-ide/src/routes/+page.svelte` |
| Agent Floor Routing Bug | `quantmind-ide/src/lib/components/TradingFloorPanel.svelte` (CopilotPanel.svelte:123) |

### Department System

```
FloorManager (Opus)
  → Research | Development | Trading | Risk | Portfolio (Sonnet)
    → 15 SubAgent worker types (Haiku)
```

### HMM Deployment Modes

```
ISING_ONLY → HMM_SHADOW → HMM_HYBRID_20 → HMM_HYBRID_50 → HMM_HYBRID_80 → HMM_ONLY
```

### Kelly Formula

```
f* = (p×b − q) / b          (base Kelly)
f_base = f* × 0.5           (Half-Kelly)
P_λ = max(0, 1 − 2λ)        (Lyapunov multiplier)
P_χ = 0.5 if χ > 0.8 else 1.0  (susceptibility multiplier)
P_E = min(1, 1.5/λ_max) if λ_max > 1.5 (eigenvalue multiplier)
M_physics = min(P_λ, P_χ, P_E)  (weakest-link)
final_risk = f_base × M_physics  (capped at 10%)
```

### Known Issues / Active Bugs

1. **Floor Manager Routing Bug** — Workshop UI sends to `/api/chat/send` instead of `/api/floor-manager/chat`. Fix: `CopilotPanel.svelte:123`. See [Migration Map](./plans/2026-03-08-agent-architecture-migration-map.md).

2. **Three deprecated agent paradigms coexist** — `base_agent.py` (LangGraph), `registry.py` (factory), `claude_config.py` (workspace-based). Only `departments/` is canonical.

3. **Workspace directories deleted from git** — `workspaces/` referenced by `claude_config.py` no longer exists in repo. Pre/post hooks in that file are all stubs returning empty lists.

---

## Glossary

| Term | Meaning |
|------|---------|
| **TRD** | Trading Rule Document — structured JSON/Markdown strategy spec |
| **@primal tag** | Bot manifest tag required to participate in the strategy auction |
| **RegimeReport** | Output of Sentinel — contains regime, chaos_score, regime_quality, susceptibility, news_state |
| **RiskMandate** | Output of Governor — contains allocation_scalar, risk_mode, position_size |
| **ISING_ONLY** | Regime detection using only Ising physics model (HMM disabled) |
| **HMM Shadow** | HMM runs in parallel for observation; does not affect trade decisions |
| **Floor Manager** | Canonical AI orchestrator that classifies tasks and routes to departments |
| **Department Mail** | SQLite-backed async message bus between AI departments |
| **PageIndex** | Docker-based full-text search service |
| **Contabo** | Training VPS — runs HMM training and serves regime API |
| **Cloudzy** | Trading VPS — runs live StrategyRouter and executes trades |
| **Kelly Criterion** | Position sizing formula: `f* = (p×b−q)/b` |
| **Half-Kelly** | Conservative variant: `f_base = f* × 0.5` |
| **Prop Firm** | Proprietary trading firm providing funded accounts (FTMO, The5ers, FundingPips) |
| **EA** | Expert Advisor — MQL5/MT5 automated trading script |
| **SSE** | Server-Sent Events — used for agent event streaming |
